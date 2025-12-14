from typing import List
import os

import torch
import torch.distributed as dist
# from sklearn.metrics import roc_auc_score

def write_info_to_file(message, file_path, append=True):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    file_exists = os.path.exists(file_path)
    mode = 'a' if append or file_exists else 'w'
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(message + '\n')

def clip_prop(prop):
    index = prop > 1.0
    prop = torch.masked_fill(prop, index, 1.0)
    index = prop < 0.0
    prop = torch.masked_fill(prop, index, 0.0)
    return prop

def get_cosine(
    item: torch.Tensor,
    seq: torch.Tensor,
    steps: List[int],
    n_dim: int,
    indicator: torch.Tensor = None,
    sorted=False,
    int8=True,
) -> torch.Tensor:
    if int8:
        item = item.to(torch.float32)
        seq = seq.to(torch.float32)
    item = torch.nn.functional.normalize(item.view(-1, 1, n_dim), dim=2)
    seq = torch.nn.functional.normalize(seq.view(-1, steps, n_dim), dim=2)
    if indicator is not None:
        seq = torch.index_select(seq, index=indicator, dim=0)
    cosine = (item * seq).sum(dim=2)
    if sorted:
        cosine = torch.sort(cosine, dim=-1, descending=True)
    return cosine

@torch.no_grad()
def sim_mm_top_k(target_content_emb, uni_seq_content_emb, keep_top=50):
    q = torch.nn.functional.normalize(target_content_emb, dim=-1)
    k = torch.nn.functional.normalize(uni_seq_content_emb, dim=-1)
    qk = torch.bmm(q, k.transpose(-1, -2)).squeeze(1)

    # to avoid fetch padding
    zero_mask = (qk == 0.0)
    qk = qk.masked_fill(zero_mask, -1.1)

    top_k_indices = torch.topk(qk, k=keep_top, dim=1, 
                           sorted=True, largest=True).indices
    return top_k_indices

@torch.no_grad()
def sim_soft_top_k(target_emb, uni_seq_emb, W_a=None, W_b=None, keep_top=50):
    if W_a is not None and W_b is not None:
        target_emb = W_a(target_emb)
        uni_seq_emb = W_b(uni_seq_emb)

    q = torch.nn.functional.normalize(target_emb, dim=-1)
    k = torch.nn.functional.normalize(uni_seq_emb, dim=-1)
    qk = torch.bmm(q, k.transpose(-1, -2)).squeeze(1) # (B, 1000)

    # to avoid fetch padding
    zero_mask = (qk == 0.0)
    qk = qk.masked_fill(zero_mask, -1.1)

    top_k_indices = torch.topk(qk, k=keep_top, dim=1, 
                           sorted=True, largest=True).indices
    
    return top_k_indices


@torch.no_grad()
def sim_hard_top_k(seq_cate, target_cate, keep_top=50):
    top_k_mask = (seq_cate==target_cate)
    actual_one_num = top_k_mask.sum(dim=1)
    # newer is on the right side
    top_k_indices = torch.argsort(top_k_mask.float(), dim=1, descending=False, stable=True)[:,-keep_top:]
    # move newer to the left side (for padding)
    top_k_indices = torch.flip(top_k_indices, dims=[1])
    
    # get pad mask
    pos = torch.arange(keep_top, device=seq_cate.device).unsqueeze(0)
    neg_pad_mask = pos >= actual_one_num.unsqueeze(1)
    
    # an example to use neg_pad_mask and top_k_indices
    # batch_indices = torch.arange(seq_cate.shape[0], device=seq_cate.device).unsqueeze(1).expand(-1, keep_top)
    # for k in range(len(uni_seq_sim)):
    #     uni_seq_sim[k] = uni_seq_sim[k][batch_indices, top_k_indices]
    #     uni_seq_sim[k][neg_pad_mask] = 0.0

    # in returned mask, 1 means padded and thus masked
    return neg_pad_mask, top_k_indices

# @torch.no_grad()
# def calc_auc_cpu(label: torch.Tensor, prop: torch.Tensor):
#     '''
#     use the API to verify AUC calculations
#     '''
#     label_cpu = label.detach().cpu().numpy()
#     prop_cpu = prop.detach().cpu().numpy()
#     y_true = label_cpu[:, 1]
#     y_score = prop_cpu[:, 1]
#     auc_score = roc_auc_score(y_true, y_score)
#     return float(auc_score)

@torch.no_grad()
def calc_auc_gpu(label: torch.Tensor, prop: torch.Tensor):
    y_score = prop[:, 1]   
    y_true = label[:, 1]

    P = y_true.sum().item() 
    N = len(y_true) - P

    if P == 0 or N == 0:
        return 1.0, P, N, 0

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    comparison = (pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0))

    wins = comparison.sum().item()
    auc = comparison.float().mean().item()

    return auc, P, N, wins

@torch.no_grad()
def calc_gauc_gpu(label: torch.Tensor, prop: torch.Tensor, user_id: torch.Tensor):
    unique_users = user_id.unique()
    impression_counts = 0
    weighted_auc_sum = 0.0

    for uid in unique_users:
        mask = (user_id == uid)
        group_label = label[mask]
        group_prop = prop[mask]

        auc, p, n, wins = calc_auc_gpu(group_label, group_prop)

        if p==0 or n==0:
            continue

        # write_info_to_file(f"{uid},{auc},{p},{n}", "gauc.csv")

        weighted_auc_sum += auc * (p+n)
        impression_counts += (p+n)

    if impression_counts == 0:
        return 0.0, 0.0, 0

    gauc = weighted_auc_sum / impression_counts
    return gauc, weighted_auc_sum, impression_counts

@torch.no_grad()
def _confusion_matrix_at_thresholds(label: torch.Tensor, prop: torch.Tensor, thresholds: torch.Tensor):
    y_score = prop[:, 1]   
    y_true = label[:, 1]

    predictions_1d = y_score.view(-1)
    labels_1d = y_true.to(dtype=torch.bool).view(-1)
    thresholds = thresholds.to(predictions_1d.device)

    # Compute predictions > threshold for all thresholds
    pred_is_pos = predictions_1d.unsqueeze(-1) > thresholds

    # Transpose to get shape (num_thresholds, num_samples)
    pred_is_pos = pred_is_pos.t()
    pred_is_neg = torch.logical_not(pred_is_pos)
    label_is_pos = labels_1d.repeat(thresholds.shape[0], 1)
    label_is_neg = torch.logical_not(label_is_pos)

    # Compute confusion matrix components
    is_true_positive = torch.logical_and(label_is_pos, pred_is_pos)
    is_true_negative = torch.logical_and(label_is_neg, pred_is_neg)
    is_false_positive = torch.logical_and(label_is_neg, pred_is_pos)
    is_false_negative = torch.logical_and(label_is_pos, pred_is_neg)

    # Sum across samples for each threshold
    tp = is_true_positive.sum(1)
    fn = is_false_negative.sum(1)
    tn = is_true_negative.sum(1)
    fp = is_false_positive.sum(1)

    return tp, fp, tn, fn

@torch.no_grad()
def calc_auroc_gpu(tp, fp, tn, fn):
    epsilon = 1.0e-6
    num_thresholds = tp.shape[0]

    # Compute True Positive Rate (Recall/Sensitivity)
    rec = torch.div(tp + epsilon, tp + fn + epsilon)

    # Compute False Positive Rate (1 - Specificity)
    fp_rate = torch.div(fp, fp + tn + epsilon)

    x = fp_rate
    y = rec

    # Compute AUC using trapezoidal rule
    auc = torch.multiply(
        x[: num_thresholds - 1] - x[1:],
        (y[: num_thresholds - 1] + y[1:]) / 2.0,
    ).sum()

    return auc.item()
    