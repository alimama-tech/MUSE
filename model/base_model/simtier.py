import torch
from typing import List, Tuple
from utils.utils import get_cosine

def cosine_simtier_list(n_dim, steps, scope_list, eps_list, dim_list):
    return CosineSimTierList(n_dim, steps, scope_list, eps_list, dim_list)

class CosineSimTierList(torch.nn.Module):
    def __init__(self, n_dim, steps, scope_list, eps_list, dim_list):
        super().__init__()
        self.n_dim = n_dim
        self.steps = steps
        self.module_lists = []
        for scope, eps, dim in zip(scope_list, eps_list, dim_list):
            module = simtier_level(eps, dim)
            self.module_lists.append(module)
            self.add_module(f"{scope}_simtier", module)

    def reset_parameters(self):
        for module in self.module_lists:
            module.reset_parameters()

    def forward(
        self, item, cosine_list, indicator_list
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        res_cosine = []
        res_simtier = []
        for seq, step, indicator, module in zip(
            cosine_list, self.steps, indicator_list, self.module_lists
        ):
            cosine_ = get_cosine(
                item, seq, steps=step, indicator=indicator, n_dim=self.n_dim
            )
            set_ = module(cosine_)
            res_cosine.append(cosine_)
            res_simtier.append(set_)
        return res_cosine, res_simtier
    
def simtier_level(eps=0.1, n_dim=4):
    return SimTierLevel(eps, n_dim)

class SimTierLevel(torch.nn.Module):
    def __init__(self, eps=0.1, n_dim=4):
        super().__init__()
        self.eps = eps
        self.n_dim = n_dim
        self.bias = int(1 / eps)
        self.register_buffer(
            "equal_ranges",
            torch.reshape(
                torch.arange(0, 2 * self.bias + 2, 1), [1, 2 * self.bias + 2, 1]
            ),
            persistent=False,
        )
        self.register_buffer("bias_power", 1/torch.scalar_tensor(self.eps), persistent=False)
        self.emb = torch.nn.Parameter(torch.zeros(1, 2 * self.bias + 2, self.n_dim))

    def reset_parameters(self):
        torch.nn.init.uniform_(self.emb.data)

    def forward(self, cosine: torch.Tensor):
        cosine_ids = torch.ceil(cosine * self.bias_power).to(torch.int32) + self.bias
        #self.equal_ranges.to(cosine_ids.device)
        weight = (self.equal_ranges == torch.unsqueeze(cosine_ids, dim=1)).type(
            torch.float32
        )
        times = weight.sum(dim=2, keepdim=True)
        return torch.reshape(
            torch.log(times + 1) * self.emb, [-1, (2 * self.bias + 2) * self.n_dim]
        )
