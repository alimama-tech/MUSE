import os
import time
import random
import sys
import json
import logging
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.muse_dataset import create_dataloader
from model.muse import MUSE_DIN
from model.base_model.feature_embedding import FeatureEmbeddingDict
from trainer import Trainer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        force=True
    )

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_model(args):
    if args["method"] in ["muse", "din", "sim-soft", "sim-hard"]:
        return MUSE_DIN(args=args, D=args["embedding_dim"], UNI_STEPS=args["keep_top"])
    else:
        raise ValueError(f"Unknown method: {args['method']}")

def train_and_eval_ddp(args, use_ddp=True):
    if use_ddp:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # We don't use nccl for sparseAdam is not supported by nccl 
        # os.environ["NCCL_DEBUG"] = "NONE"
        # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        os.environ["GLOO_DEBUG"] = "NONE"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        
        torch.cuda.set_device(rank)
        # print(f"[Rank {rank}] Ready")
    else:
        rank = 0
        world_size = 1
    
    # Set seed for each rank
    setup_seed(args["seed"])

    if rank == 0:
        print("="*50)
        print("Configuration:")
        print("="*50)
        for k, v in args.items():
            print(f"  {k:20} : {v}")
        print("="*50)

    # Datset
    train_dataloader = create_dataloader(
        data_dir=args["train_data_path"],
        mode="train",
        batch_size=args["batch_size"],
        max_seq_len=1000,
        num_workers=2,
        shuffle=args["shuffle"],
        shuffle_buffer_size=args["shuffle_buffer_size"],
        rank=rank,
        world_size=world_size
    )
    test_dataloader = create_dataloader(
        data_dir=args["test_data_path"],
        mode="test",
        batch_size=args["batch_size"],
        max_seq_len=1000,
        num_workers=2,
        shuffle=False,
        rank=rank,
        world_size=world_size
    )

    # Model
    embedding_layer = FeatureEmbeddingDict(
        dim=args["embedding_dim"], 
        feature_map_dir=args["feature_map_path"],
        device=rank, 
        world_size=world_size, 
        item_id_p90=args["item_id_p90"],
        scl_emb_p90=args["scl_emb_p90"],
        feature_map_on_cuda=args["feature_map_on_cuda"],
        scl_emb_on_cuda=args["scl_emb_on_cuda"]
    )
    din_model = create_model(args)

    dense_params = (
        [p for p in embedding_layer.get_dense_parameters() if p.requires_grad] +
        [p for p in din_model.parameters() if p.requires_grad]
    )
    sparse_params = embedding_layer.get_sparse_parameters()

    # Optimizer
    dense_opt = optim.AdamW(dense_params, lr=args["dense_lr"], fused=True)
    sparse_opt = optim.SparseAdam(sparse_params, lr=args["sparse_lr"])

    if use_ddp:
        embedding_layer = embedding_layer.to(rank)
        din_model = din_model.to(rank)
        embedding_layer = DDP(embedding_layer, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        din_model = DDP(din_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    else:
        embedding_layer = embedding_layer.cuda()
        din_model = din_model.cuda()
    
    # Trainer
    trainer = Trainer(
        dense_model=din_model,
        sparse_model=embedding_layer,
        dense_opt=dense_opt,
        sparse_opt=sparse_opt,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        args=args,
        device=rank,
        world_size=world_size,
        keep_top=args["keep_top"],
        rank=rank
    )

    try:
        trainer.fit()
    except Exception as e:
        print(f"Rank {rank} caught exception:")
        raise e
    finally:
        dist.destroy_process_group()
    
    if rank == 0:
        logging.info("Training finished")

def eval_ddp(args, use_ddp=True):
    if use_ddp:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # We don't use nccl for sparseAdam is not supported by nccl 
        # os.environ["NCCL_DEBUG"] = "NONE"
        # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        os.environ["GLOO_DEBUG"] = "NONE"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        
        torch.cuda.set_device(rank)
        # print(f"[Rank {rank}] Ready")
    else:
        rank = 0
        world_size = 1
    
    # Set seed for each rank
    setup_seed(args["seed"])

    if rank == 0:
        print("="*50)
        print("Configuration:")
        print("="*50)
        for k, v in args.items():
            print(f"  {k:20} : {v}")
        print("="*50)

    # Datset
    train_dataloader = create_dataloader(
        data_dir=args["train_data_path"],
        mode="train",
        batch_size=args["batch_size"],
        max_seq_len=1000,
        num_workers=2,
        shuffle=args["shuffle"],
        shuffle_buffer_size=args["shuffle_buffer_size"],
        rank=rank,
        world_size=world_size
    )
    test_dataloader = create_dataloader(
        data_dir=args["test_data_path"],
        mode="test",
        batch_size=args["batch_size"],
        max_seq_len=1000,
        num_workers=2,
        shuffle=False,
        rank=rank,
        world_size=world_size
    )

    # Model
    embedding_layer = FeatureEmbeddingDict(
        dim=args["embedding_dim"], 
        feature_map_dir=args["feature_map_path"],
        device=rank, 
        world_size=world_size, 
        item_id_p90=args["item_id_p90"],
        scl_emb_p90=args["scl_emb_p90"],
        feature_map_on_cuda=args["feature_map_on_cuda"],
        scl_emb_on_cuda=args["scl_emb_on_cuda"]
    )
    din_model = create_model(args)

    dense_params = (
        [p for p in embedding_layer.get_dense_parameters() if p.requires_grad] +
        [p for p in din_model.parameters() if p.requires_grad]
    )
    sparse_params = embedding_layer.get_sparse_parameters()

    # Optimizer
    dense_opt = optim.AdamW(dense_params, lr=args["dense_lr"], fused=True)
    sparse_opt = optim.SparseAdam(sparse_params, lr=args["sparse_lr"])
    
    try:
        dense_ckpt_path = args["dense_ckpt_path"]
        sparse_ckpt_path = args["sparse_ckpt_path"]
    except:
        raise ValueError(f"ckpt path must be specified")

    if use_ddp:
        embedding_layer = embedding_layer.to(rank)
        din_model = din_model.to(rank)
        embedding_layer = DDP(embedding_layer, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        din_model = DDP(din_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        din_model.module.load_ckpt(dense_ckpt_path)
        embedding_layer.module.load_ckpt(sparse_ckpt_path)

        # embedding_layer.module.filter_low_freq_id()
    else:
        embedding_layer = embedding_layer.cuda()
        din_model = din_model.cuda()

        din_model.load_ckpt(dense_ckpt_path)
        embedding_layer.load_ckpt(sparse_ckpt_path)

    # Trainer
    trainer = Trainer(
        dense_model=din_model,
        sparse_model=embedding_layer,
        dense_opt=dense_opt,
        sparse_opt=sparse_opt,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        args=args,
        device=rank,
        world_size=world_size,
        keep_top=args["keep_top"],
        rank=rank
    )

    try:
        trainer.eval()
    except Exception as e:
        print(f"Rank {rank} caught exception:")
        raise e
    finally:
        dist.destroy_process_group()
    
    if rank == 0:
        logging.info("Evaluation finished")

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate model with config and CLI override.")

    parser.add_argument(
        '--config',
        type=str,
        nargs='+',
        help='One or more config files (JSON) to load. Later ones override earlier ones.'
    )

    # only part of configs are listed here
    parser.add_argument('--job_type', type=str, help='Job type: train or test')
    parser.add_argument('--train_data_path', type=str, help='Path to train data')
    parser.add_argument('--test_data_path', type=str, help='Path to test data')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--dense_lr', type=float, help='Learning rate for dense parameters')
    parser.add_argument('--sparse_lr', type=float, help='Learning rate for sparse parameters')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser.add_argument('--method', type=str, help='Method name: e.g., muse')
    parser.add_argument('--exp_name', type=str, help='Exp name')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle data')
    parser.add_argument('--shuffle_buffer_size', type=int, help='Shuffle buffer size')
    parser.add_argument('--use_ddp', action='store_true', help='Use DDP for distributed training')

    args, remaining = parser.parse_known_args()

    config = {}
    if args.config:
        for config_file in args.config:
            cfg = load_config(config_file)
            config.update(cfg)

    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    cli_args.pop('config', None)

    config.update(cli_args)
    
    init_logging()

    if config["job_type"] == "train":
        train_and_eval_ddp(args=config, use_ddp=config["use_ddp"])
    elif config["job_type"] == "eval":
        eval_ddp(args=config, use_ddp=config["use_ddp"])
    else:
        raise ValueError(f"Unknown job type: {config['job_type']}")

if __name__ == "__main__":
    main()