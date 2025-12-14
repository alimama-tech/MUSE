import json
import logging
import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import numpy as np
import random
from typing import Iterator, Dict, Any, Optional

class ParquetIterableDataset(IterableDataset):
    """
    Iterable dataset to stream training samples from sharded Parquet files.
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        batch_size: int = 1000,
        max_seq_len: int = 1000,
        pad_value: int = 0,
        shuffle_shards: bool = True,
        shuffle_buffer_size: int = 1000,
        seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            else:
                rank = 0
                world_size = 1
        
        self.rank = rank
        self.world_size = world_size

        # Read meta.json
        try:
            with open(self.data_dir / "metadata.json", "r") as f:
                self.metadata = json.load(f)
            total_steps = self.metadata["total_rows"] // (self.batch_size * self.world_size)
            if self.rank == 0:
                print(f"[{mode.upper()} Dataset] Total about {total_steps} steps")
        except FileNotFoundError:
            self.metadata = {}
            if self.rank == 0:
                print(f"[{mode.upper()} Dataset] No metadata.json found")

        # Discover all shard files
        self.shard_files = sorted(self.data_dir.glob(f"{mode}-shard-*.parquet"))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard files found in {data_dir}")
        
        shard_file_num = len(self.shard_files)
        shard_file_num -= shard_file_num % self.world_size

        if rank==0:
            logging.info(f"[INFO][{mode.upper()} Dataset] Found {len(self.shard_files)} shard files in {self.data_dir}, use {shard_file_num} shard files")

        self.shard_files = self.shard_files[:shard_file_num]

    def _process_row(self, row: pd.Series) -> Dict[str, Any]:
        """Process a single row into model-ready format."""
        # Extract label: list<int8> -> scalar
        label = row.get('label_0', [])

        # Helper: ensure field is list or empty
        def safe_list(x):
            return x if isinstance(x, (list, tuple)) else []

        hist_items = row.get('150_2_180', np.array([])).tolist()
        hist_cates = row.get('151_2_180', np.array([])).tolist()

        # Truncate to recent interactions (keep last max_seq_len)
        if(self.max_seq_len < len(hist_items)):
            hist_items = hist_items[-self.max_seq_len:]
            hist_cates = hist_cates[-self.max_seq_len:]

        # Pad on the left (older interactions on the left, recent on the right)
        pad_len = self.max_seq_len - len(hist_items)
        if pad_len > 0:
            hist_items = [self.pad_value] * pad_len + hist_items
            hist_cates = [self.pad_value] * pad_len + hist_cates

        return {
            'label': label,
            '129_1': row['129_1'],
            '205': row['205'],
            '130_1': row['130_1'],
            '130_2': row['130_2'],
            '130_3': row['130_3'],
            '130_4': row['130_4'],
            '130_5': row['130_5'],
            '150_2_180': hist_items,
            '151_2_180': hist_cates,
            '206': row['206'],
            '213': row['213'],
            '214': row['214'],
        }

    def _read_single_parquet_file(self, file_path: str) -> Iterator[dict]:
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            for _, row in df.iterrows():
                processed_row = self._process_row(row)
                yield processed_row
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # Step 1: determine which shards this DDP process (rank) is responsible for
        global_shard_indices = list(range(len(self.shard_files)))
        rank_shard_indices = [i for i in global_shard_indices if i % self.world_size == self.rank]
        rank_shard_files = [self.shard_files[i] for i in rank_shard_indices]

        # Step 2: within this rank, shards are assigned by each worker
        worker_shard_files = [
            shard for i, shard in enumerate(rank_shard_files)
            if i % num_workers == worker_id
        ]

        # Step 3: set local seed
        base_seed = self.seed
        epoch = self.epoch if hasattr(self, 'epoch') else 0
        local_seed = base_seed + epoch * 100 + self.rank * 10 + worker_id
        rng = random.Random(local_seed)

        # Step 4: shuffle (if needed)
        if self.shuffle_shards:
            rng.shuffle(worker_shard_files)

        if (not self.shuffle_shards) or self.shuffle_buffer_size <= 1:
            # no shuffle, only sequential read
            for shard_file in worker_shard_files:
                yield from self._read_single_parquet_file(shard_file)
        else:
            # enable shuffle buffer
            buffer = []
            for shard_file in worker_shard_files:
                for sample in self._read_single_parquet_file(shard_file):
                    if len(buffer) < self.shuffle_buffer_size:
                        buffer.append(sample)
                    else:
                        # pop a random sample for replacement
                        idx = rng.randint(0, len(buffer) - 1)
                        yield buffer[idx]
                        buffer[idx] = sample
            # yield remaining samples from the buffer
            rng.shuffle(buffer)
            for sample in buffer:
                yield sample

    def collate_fn(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Convert a list of samples into a batch of tensors.
        """
        keys = batch[0].keys()
        result = {}
        for key in keys:
            values = [item[key] for item in batch]
            if key in ['user_hist_items', 'user_hist_cates']:
                # List of lists -> tensor [B, T]
                result[key] = torch.tensor(np.array(values), dtype=torch.long)
            elif isinstance(values[0], (int, float)):
                dtype = torch.float32 if key == 'label' else torch.long
                result[key] = torch.tensor(np.array(values), dtype=dtype)
            else:
                result[key] = torch.tensor(np.array(values), dtype=torch.long)
        return result
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch


def create_dataloader(
    data_dir: str,
    mode: str = "train",
    batch_size: int = 1000,
    max_seq_len: int = 1000,
    num_workers: int = 4,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000,
    drop_last: bool = True,
    pin_memory: bool = True,
    seed: int = 42,
    rank: Optional[int] = None,
    world_size: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader for sharded Parquet training data.
    """
    seq_dataset = ParquetIterableDataset(
        data_dir=data_dir,
        mode=mode,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        shuffle_shards=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        rank=rank,
        world_size=world_size
    )

    return DataLoader(
        seq_dataset,
        batch_size=batch_size,
        collate_fn=seq_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
