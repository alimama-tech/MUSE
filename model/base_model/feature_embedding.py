import os
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

# these embeddings are trained from scratch
FEAT_NAME_WITH_UNIQUE_EMB = [
    "129_1", "130_1", "130_2", "130_3", "130_4", "130_5",
    "150_2_180", "151_2_180",
    "213", "214"
]

# ["205", "206", "150_1_180", "151_1_180"] are shared with some others
# ["205_c", "150_2_180_c", "150_1_180_c"] are obtained via scl embedding lookup

class FeatureEmbeddingDict(nn.Module):
    def __init__(self, 
        dim=32, 
        feature_map_dir="./taobao-mm/feature_map", 
        device=torch.device("cuda"),
        world_size=1,
        loaded_feature_map=None,
        loaded_scl_emb=None,
        item_id_p90=False,
        scl_emb_p90=True,
        feature_map_on_cuda=True,
        scl_emb_on_cuda=True
    ):
        super(FeatureEmbeddingDict, self).__init__()
        self.dim = dim
        self.embedding_columns = FEAT_NAME_WITH_UNIQUE_EMB
        self.sparse_columns = FEAT_NAME_WITH_UNIQUE_EMB 
        # self.sparse_columns = ["150_2_180", "129_1"]
        # self.sparse_columns = ["150_2_180"]
        self.device = device
        self.world_size = world_size
        self.item_id_p90 = item_id_p90
        
        # load feature map
        self.feature_maps = {}
        self.load_feature_map(feature_map_dir=feature_map_dir, loaded_feature_map=loaded_feature_map)

        # create embedding table
        self.embedding_layers = nn.ModuleDict()
        self.create_embedding_table()
        self.init_weights()

        # load scl embedding table
        self.scl_embedding_table = {}
        if scl_emb_p90:
            scl_emb_key_dir = os.path.join(feature_map_dir, "scl_emb_int8_p90_keys.npy")
            scl_emb_value_dir = os.path.join(feature_map_dir, "scl_emb_int8_p90_values.npy")
        else:
            # not supported now
            scl_emb_key_dir = os.path.join(feature_map_dir, "scl_emb_int8_p99_keys.npy")
            scl_emb_value_dir = os.path.join(feature_map_dir, "scl_emb_int8_p99_values.npy")
        self.create_scl_embedding_table(
            scl_emb_key_dir=scl_emb_key_dir,
            scl_emb_value_dir=scl_emb_value_dir,
            loaded_scl_emb=loaded_scl_emb
        )

        self.current_scl_cover_rate_target = 1.0
        self.current_scl_cover_rate_seq = 1.0

        self.feature_map_on_cuda = False
        self.scl_emb_on_cuda = False
        self.move_feature_map_to_cuda(feature_map_on_cuda, scl_emb_on_cuda)

        # TODO: add sharing rules instead of hardcoding them
        # self.shared_map = {
        #     "205": "150_2_180",
        #     "150_1_180": "150_2_180",
        #     "206": "151_2_180",
        #     "151_1_180": "151_2_180"
        # }

    def create_embedding_table(self):
        for column in self.embedding_columns:
            is_sparse = False
            if column in self.sparse_columns:
                is_sparse = True

            # some tricks on feature dim
            dim = self.dim
            if column in ["130_3", "130_4", "130_5"]:
                dim = self.dim // 2
            self.embedding_layers[column] = nn.Embedding(len(self.feature_maps[column])+1, dim, padding_idx=0, sparse=is_sparse, dtype=torch.float32)

        # self.embedding_layers["205"] = nn.Embedding(len(self.feature_maps["205"])+1, dim, padding_idx=0, sparse=is_sparse, dtype=torch.float32)
        # self.embedding_layers["206"] = nn.Embedding(len(self.feature_maps["206"])+1, dim, padding_idx=0, sparse=is_sparse, dtype=torch.float32)

        if self.device == 0 or dist.get_rank() == 0:
            for column in self.embedding_columns:
                print(f"[{column}] vocab size: {len(self.feature_maps[column])}")

    def get_dense_parameters(self):
        params = []
        for name, module in self.embedding_layers.items():
            if not module.sparse:
                params += list(module.parameters())
        return params

    def get_sparse_parameters(self):
        params = []
        for name, module in self.embedding_layers.items():
            if module.sparse:
                params += list(module.parameters())
        return params
    
    def init_weights(self):
        for name, layer in self.embedding_layers.items():
            # nn.init.normal_(layer.weight[1:, :], mean=0, std=0.001)
            nn.init.trunc_normal_(layer.weight[1:, :], mean=0, std=0.001, a=-2.0, b=2.0)

    def broadcast_np_load_ddp(self, file_path):
        objects_to_broadcast = [None]
        rank = dist.get_rank()
        if rank == 0:
            array_data = np.load(file_path)
            if isinstance(array_data, np.memmap):
                array_data = np.array(array_data)
            else:
                array_data = array_data.copy()
            objects_to_broadcast[0] = array_data
        else:
            objects_to_broadcast[0] = None
        dist.broadcast_object_list(objects_to_broadcast, src=0)
        return objects_to_broadcast[0]

    def create_scl_embedding_table(
        self, 
        scl_emb_key_dir="./taobao-mm/feature_map/scl_emb_int8_p90_keys.npz",
        scl_emb_value_dir="./taobao-mm/feature_map/scl_emb_int8_p90_values.npz", 
        loaded_scl_emb=None
    ):
        if self.device == 0 or dist.get_rank() == 0:
            logging.info(f"Loading scl embedding table from {scl_emb_key_dir} and {scl_emb_value_dir} started")
        if loaded_scl_emb is not None:
            scl_emb_key_data = loaded_scl_emb["keys"]
            scl_emb_value_data = loaded_scl_emb["values"]
        else:
            if self.world_size == 1:
                scl_emb_key_data = np.load(scl_emb_key_dir)
                scl_emb_value_data = np.load(scl_emb_value_dir)
            else:
                scl_emb_key_data = self.broadcast_np_load_ddp(scl_emb_key_dir)
                scl_emb_value_data = self.broadcast_np_load_ddp(scl_emb_value_dir)
        self.scl_embedding_table["keys"] = np.array(scl_emb_key_data)
        values = np.array(scl_emb_value_data)
        zero_row = np.zeros((1, values.shape[1]), dtype=values.dtype)
        self.scl_embedding_table["values"] = np.concatenate([zero_row, values], axis=0)
        if self.device == 0 or dist.get_rank() == 0:
            logging.info(f"Loading scl embedding table from {scl_emb_key_dir} and {scl_emb_value_dir} finished")

        del scl_emb_key_data, scl_emb_value_data

    def load_feature_map(self, 
        feature_map_dir="./taobao-mm/feature_map",
        loaded_feature_map=None
    ):
        if self.device == 0 or dist.get_rank() == 0:
            logging.info(f"Loading feature map from {feature_map_dir} started")
        for column in self.embedding_columns:
            if loaded_feature_map is not None:
                feature_map_data = loaded_feature_map[column]
            else:
                if self.item_id_p90 and column in ["150_2_180"]:
                    # only load id embeddings with high frequency (> 90%)
                    if self.world_size == 1:
                        feature_map_data = np.load(os.path.join(feature_map_dir, f"{column}_sorted_map_p90.npy"))
                    else:
                        feature_map_data = self.broadcast_np_load_ddp(os.path.join(feature_map_dir, f"{column}_sorted_map_p90.npy"))
                else:
                    if self.world_size == 1:
                        feature_map_data = np.load(os.path.join(feature_map_dir, f"{column}_sorted_map.npy"))
                    else:
                        feature_map_data = self.broadcast_np_load_ddp(os.path.join(feature_map_dir, f"{column}_sorted_map.npy"))

            feature_map_key = feature_map_data
            self.feature_maps[column] = np.array(feature_map_key)
        
        # load additional feature map for target
        # if self.world_size == 1:
        #     feature_map_data = np.load(os.path.join(feature_map_dir, "150_2_180_sorted_map_p90.npy"))
        # else:
        #     feature_map_data = self.broadcast_np_load_ddp(os.path.join(feature_map_dir, "150_2_180_sorted_map_p90.npy"))
        # feature_map_key = feature_map_data
        # self.feature_maps["205"] = np.array(feature_map_key)
        # if self.world_size == 1:
        #     feature_map_data = np.load(os.path.join(feature_map_dir, "151_2_180_sorted_map.npy"))
        # else:
        #     feature_map_data = self.broadcast_np_load_ddp(os.path.join(feature_map_dir, "151_2_180_sorted_map.npy"))
        # feature_map_key = feature_map_data
        # self.feature_maps["206"] = np.array(feature_map_key)

        if self.device == 0 or dist.get_rank() == 0:
            logging.info(f"Loading feature map from {feature_map_dir} finished")

        del feature_map_data

    def move_feature_map_to_cuda(self, feature_map_on_cuda, scl_emb_on_cuda):
        if feature_map_on_cuda:
            for column in self.embedding_columns:
                self.feature_maps[column] = torch.from_numpy(self.feature_maps[column]).to(self.device)
            self.scl_embedding_table["keys"] = torch.from_numpy(self.scl_embedding_table["keys"]).to(self.device)
            self.feature_map_on_cuda = True
        
        if scl_emb_on_cuda:
            self.scl_embedding_table["values"] = torch.from_numpy(self.scl_embedding_table["values"]).to(self.device)
            self.scl_emb_on_cuda = True
        else:
            self.scl_embedding_table["values"] = self.scl_embedding_table["values"]
        
    def lookup(self, column, raw_idx):
        device = raw_idx.device

        # sharing tables
        if column in ["205", "150_1_180"]:
            column = "150_2_180"
        if column in ["206", "151_1_180"]:
            column = "151_2_180"

        def vectorized_lookup_np(keys, x):
            original_shape = x.shape
            x_flat = x.ravel()
            indices_flat = np.searchsorted(keys, x_flat, side='left')
            result_flat = np.full_like(x_flat, -1, dtype=np.int32)
            
            valid_idx = indices_flat < len(keys)
            valid_mask = np.zeros_like(indices_flat, dtype=bool)
            valid_mask[valid_idx] = keys[indices_flat[valid_idx]] == x_flat[valid_idx]
            
            result_flat[valid_mask] = indices_flat[valid_mask]
            result_flat = result_flat + 1
            return result_flat.reshape(original_shape)
        
        def vectorized_lookup_tensor(keys: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            original_shape = x.shape
            x_flat = x.reshape(-1)
            indices_flat = torch.searchsorted(keys, x_flat, side='left', out_int32=True)
            result_flat = torch.full_like(x_flat, -1, dtype=torch.int32)

            valid_idx = indices_flat < len(keys)
            valid_mask = torch.zeros_like(x_flat, dtype=torch.bool)
            valid_mask[valid_idx] = keys[indices_flat[valid_idx]] == x_flat[valid_idx]

            result_flat[valid_mask] = indices_flat[valid_mask]
            result_flat = result_flat + 1
            return result_flat.reshape(original_shape)
        
        if not self.feature_map_on_cuda:
            raw_idx_np = raw_idx.cpu().numpy()
            if column in ["205_c", "150_2_180_c", "150_1_180_c"]:
                mapped_idx_np = vectorized_lookup_np(self.scl_embedding_table["keys"], raw_idx_np)
                return mapped_idx_np
            else:
                mapped_idx_np = vectorized_lookup_np(self.feature_maps[column], raw_idx_np)
                mapped_idx_tensor = torch.from_numpy(mapped_idx_np).to(device, dtype=torch.int32)
                return mapped_idx_tensor
        else:
            if column in ["205_c", "150_2_180_c", "150_1_180_c"]:
                mapped_idx_tensor = vectorized_lookup_tensor(self.scl_embedding_table["keys"], raw_idx)
                return mapped_idx_tensor
            else:
                mapped_idx_tensor = vectorized_lookup_tensor(self.feature_maps[column], raw_idx)
                return mapped_idx_tensor

    def forward(self, batch: dict):
        feature_emb_dict = {}
        for column in batch.keys():
            remapped_idx = self.lookup(column, batch[column])

            # calculate scl embedding cover rate
            if not self.feature_map_on_cuda:
                if column == "205_c":
                    self.current_scl_cover_rate_target = 1 - np.sum(remapped_idx == 0) / remapped_idx.size
                if column == "150_2_180_c":
                    self.current_scl_cover_rate_seq = 1 - np.sum(remapped_idx == 0) / remapped_idx.size
            else:
                if column == "205_c":
                    self.current_scl_cover_rate_target = (remapped_idx != 0).float().mean().item()
                if column == "150_2_180_c":
                    self.current_scl_cover_rate_seq = (remapped_idx != 0).float().mean().item()

            # shared embeddings: item id
            if column in ["205", "150_1_180", "150_2_180"]:
                feature_emb_dict[column] = self.embedding_layers["150_2_180"](remapped_idx.to(self.device))
            # shared embeddings: cate id
            elif column in ["206", "151_1_180", "151_2_180"]:
                feature_emb_dict[column] = self.embedding_layers["151_2_180"](remapped_idx.to(self.device))
            # scl embeddings
            elif column in ["205_c", "150_2_180_c", "150_1_180_c"]:
                if not self.feature_map_on_cuda:
                    if not self.scl_emb_on_cuda:
                        # remapped_idx -> np; scl_embedding_table -> np
                        scl_embeddings = torch.from_numpy(self.scl_embedding_table["values"][remapped_idx]).to(self.device)
                    else:
                        # remapped_idx -> np; scl_embedding_table -> tensor
                        scl_embeddings = self.scl_embedding_table["values"][torch.from_numpy(remapped_idx).to(self.device)]
                else:
                    if not self.scl_emb_on_cuda:
                        # remapped_idx -> tensor; scl_embedding_table -> np
                         scl_embeddings = torch.from_numpy(self.scl_embedding_table["values"][remapped_idx.cpu().numpy()]).to(self.device)
                    else:
                        # both on cuda
                        scl_embeddings = self.scl_embedding_table["values"][remapped_idx].to(self.device)
                # int8 -> float
                feature_emb_dict[column] = scl_embeddings.float()
            # other embeddings by default
            else:
                feature_emb_dict[column] = self.embedding_layers[column](remapped_idx.to(self.device))
        return feature_emb_dict
    
    def save_ckpt(self, ckpt_path: str, rank=None):
        if rank is not None and rank != 0:
            return

        state_dict = {'embedding_layers': self.embedding_layers.state_dict()}

        # os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(state_dict, ckpt_path)
        logging.info(f"Checkpoint saved to {ckpt_path}")
    
    def load_ckpt(self, ckpt_path: str, map_location=None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if map_location is None:
            map_location = lambda storage, loc: storage.cuda(self.device)

        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.embedding_layers.load_state_dict(ckpt['embedding_layers'])
        logging.info(f"[Rank {self.device}] Checkpoint loaded from {ckpt_path}")

        return self
    
    # not used now
    # def filter_low_freq_id(self, q=0.9, id_count_dict=None):
    #     if id_count_dict is None:
    #         id_counts_dict = dict()

    #         if dist.is_available() and dist.is_initialized():
    #             dist.barrier()

    #         id_counts_dict["150_2_180"] = torch.load("./ckpt/id_counts_uni_seq.pt").to(self.device)
    #         id_counts_dict["150_1_180"] = torch.load("./ckpt/id_counts_short_seq.pt").to(self.device)
    #         id_counts_dict["205"] = torch.load("./ckpt/id_counts_target.pt").to(self.device)

    #     # thresholds = dict()
    #     # for k in id_counts_dict:
    #     #     thresholds[k] = float(np.quantile(id_counts_dict[k].cpu().numpy(), q, method='linear'))

    #     thresholds = {
    #         "150_2_180": 10000,
    #         "150_1_180": 10000,
    #         "205": 1
    #     }

    #     mask = torch.zeros(self.embedding_layers["150_2_180"].weight.shape[0], dtype=torch.bool, device=self.device)
    #     for key in id_counts_dict:
    #         count = id_counts_dict[key]
    #         th = thresholds[key]
    #         if key in ["150_2_180", "150_1_180"]:
    #             mask |= (count < th)
    #         else:
    #             mask |= (count >= th)

    #     if self.device == 0 or dist.get_rank() == 0:
    #         logging.info(f"{mask.int().sum().item()} item id left after filtering.")

    #     with torch.no_grad():
    #         self.embedding_layers["150_2_180"].weight[~mask] = 0.0

