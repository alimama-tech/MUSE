import os
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

RAW_DATA_PATH = "taobao-mm/raw"
OUTPUT_DIR = "taobao-mm/"

user_fn = ["129_1", "130_1", "130_2", "130_3", "130_4", "130_5"]
ad_fn = ["205", "206", "213", "214"]
rt_seq_fn = ["150_1_180", "151_1_180"]
uni_seq_fn = ["150_2_180", "151_2_180"]
all_feat_fn = ["label_0"] + user_fn + ad_fn + rt_seq_fn + uni_seq_fn


def transform_feature(feat_fn, feat_in_text):
    '''
    from string to parquet type
    '''
    feature_dict = zip(feat_fn, feat_in_text)
    transformed_feat_dict = dict()
    for name, fn in feature_dict:
        if name == "label_0":
            transformed_fn = [int(f) for f in fn]
        elif name in rt_seq_fn + uni_seq_fn:
            transformed_fn = [int(f) for f in fn]
        elif name == "205_c":
            fn=fn[0].split("\x1d")
            # int or float
            transformed_fn = [float(f) for f in fn]
            assert len(transformed_fn) == 128
        else:
            transformed_fn = int(fn[0])
        transformed_feat_dict[name] = transformed_fn
    transformed_feat = [transformed_feat_dict[name] for name in feat_fn]
    return transformed_feat

def txt2parquet(input_file_name: str, feat_fn: list, feat_schema: list, fill_in_empty: bool = True):
    # ==============================
    # 配置参数
    # ==============================
    # feat_fn = ['label_0', '129_1', '205']

    output_name = input_file_name.split('.')[0]
    FINAL_OUTPUT = f"{output_name}.full.parquet"
    FINAL_OUTPUT = os.path.join(OUTPUT_DIR, FINAL_OUTPUT)
    SHARD_SIZE = 500_000
    COMPRESSION = "NONE"

    # ==============================
    # 创建输出目录
    # ==============================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shard_pattern = os.path.join(OUTPUT_DIR, f"{output_name}.part_*.parquet")
    for old_file in Path(OUTPUT_DIR).glob(f"{output_name}.part_*.parquet"):
        old_file.unlink()

    # ==============================
    # Step 1: 流式处理原始文件 → 写入分片
    # ==============================
    print(f"🚀 开始流式处理 {input_file_name}...")

    buffer = []
    shard_id = 0
    line_count = 0

    schema = pa.schema(feat_schema)
    # schema = pa.schema([
    #     ('label_0', pa.list_(pa.int8())),
    #     ('129_1', pa.int64()),
    #     ('205', pa.int64()),
    # ])

    input_file_path = os.path.join(RAW_DATA_PATH, input_file_name)

    all_processed_count = 0
    valid_count = 0

    with open(input_file_path, 'r') as file:
        for line in file:
            line_count += 1
            # if(line_count % 1000000 == 0):
            #     print(f"{line_count} lines processed")

            if not line.strip():
                continue
            all_processed_count += 1
            
            # 解析 tab 分隔字段
            text = line.strip().split('\t')
            feat_in_text = [fea.split(',') for fea in text]

            for i in range(len(feat_in_text)):
                if feat_in_text[i] == ['']:
                    feat_in_text[i] = ['0']

            # 补全缺失字段
            if len(feat_in_text) < len(feat_fn):
                if fill_in_empty:
                    # raise ValueError(f"Line {line_count} has {len(feat_in_text)} features, expected {len(feat_fn)}")
                    valid_count -= 1
                    while len(feat_in_text) < len(feat_fn):
                        feat_in_text.append(['0'])
                else:
                    continue

            valid_count += 1

            # 转换特征
            try:
                transformed_feat = transform_feature(feat_fn, feat_in_text)
                buffer.append(transformed_feat)
            except Exception as e:
                print(f"Error at line {line_count}: {e}")
                continue  # 跳过错误行

            # 写分片
            if len(buffer) >= SHARD_SIZE:
                df = pd.DataFrame(buffer)
                df.columns = feat_fn
                table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                shard_file = os.path.join(OUTPUT_DIR, f"{output_name}.part_{shard_id:04d}.parquet")
                pq.write_table(table, shard_file, compression=COMPRESSION)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 已写入分片 {shard_id}（累计 {line_count} 行）")
                buffer = []
                shard_id += 1

    # 写最后一批
    if buffer:
        df = pd.DataFrame(buffer)
        df.columns = feat_fn
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        shard_file = os.path.join(OUTPUT_DIR, f"{output_name}.part_{shard_id:04d}.parquet")
        pq.write_table(table, shard_file, compression=COMPRESSION)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 已写入最后一个分片 {shard_id}")

    num_shards = shard_id + 1 if buffer else shard_id
    print(f"📦 分片完成：共 {num_shards} 个分片，存储在 '{OUTPUT_DIR}'")

    # ==============================
    # Step 2: 合并所有分片为一个大文件
    # ==============================
    print("🔄 开始合并所有分片为单一 Parquet 文件...")

    shard_files = sorted(Path(OUTPUT_DIR).glob(f"{output_name}.part_*.parquet"))
    if not shard_files:
        raise FileNotFoundError("未找到任何分片文件！")

    tables = []
    total_rows = 0

    for shard_path in shard_files:
        table = pq.read_table(shard_path)
        tables.append(table)
        num_rows = table.num_rows
        total_rows += num_rows
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载分片: {shard_path.name} → {num_rows} 行")

    # 合并所有 Table
    merged_table = pa.concat_tables(tables)

    # 写入最终文件
    pq.write_table(merged_table, FINAL_OUTPUT, compression=COMPRESSION)

    # 获取文件大小
    final_size_gb = os.path.getsize(FINAL_OUTPUT) / (1024 ** 3)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 合并完成！")
    print(f"  - 输出文件: {FINAL_OUTPUT}")
    print(f"  - 总样本数: {total_rows:,}")
    print(f"  - 非空行数: {all_processed_count}")
    print(f"  - 有效行数: {valid_count}")
    print(f"  - 文件大小: {final_size_gb:.2f} GB")
    print(f"  - 压缩格式: {COMPRESSION}")

    # ==============================
    # Step 3: （可选）清理临时分片
    # ==============================
    print("🧹 正在清理临时分片...")
    for shard_file in shard_files:
        shard_file.unlink()
    print("✅ 临时文件已删除")

    # ==============================
    # Step 4: 验证输出（读取前几行）
    # ==============================
    # print("🔍 验证：读取输出文件前 5 行...")
    # test_df = pq.read_table(FINAL_OUTPUT, columns=None).to_pandas()
    # print(test_df.head())
    print("🎉 文件格式转换全流程完成！")

def count_frequency():
    '''
    count ID frequency
    '''
    file_name = "train_user_features.full.parquet"
    feat_fn = ["129_1", "130_1", "130_2", "130_3", "130_4", "130_5", "150_2_180", "151_2_180"]
    file_name_prefix = file_name.split(".")[0]

    PARQUET_FILE = os.path.join(OUTPUT_DIR, file_name)

    parquet_file = pq.ParquetFile(PARQUET_FILE)
    total_rows = 0

    freq_dicts = {name: defaultdict(int) for name in feat_fn}

    for i, batch in enumerate(parquet_file.iter_batches(batch_size=500_000)):
        df: pd.DataFrame = batch.to_pandas()
        total_rows += len(df)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理第 {i+1} 批... 当前累计 {total_rows:,} 行")

        for name in feat_fn:
            assert name in df.columns
            series = df[name]
            
            if name in ["150_2_180", "151_2_180"]:
                exploded = series.explode() # [ [a,b], [c], ... ] → [a, b, c, ...]
                for val in exploded.astype(int):
                    freq_dicts[name][val] += 1
            else:
                for val in series.astype(int):
                    freq_dicts[name][val] += 1

    # 统计一些值
    summary = {}
    for name in feat_fn:
        counter = freq_dicts[name]
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # 按频次降序

        # 保存为 JSON
        freq_file = f"{OUTPUT_DIR}/json/{file_name_prefix}_{name}_freq.json"
        with open(freq_file, "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in sorted_items}, f, indent=2, ensure_ascii=False)

        # 统计信息
        total_ids = len(sorted_items)
        top10_freq = sum(count for _, count in sorted_items[:10])
        min_freq = sorted_items[-1][1]
        max_freq = sorted_items[0][1]

        summary[name] = {
            "unique_count": total_ids,
            "total_occurrences": sum(count for _, count in sorted_items),
            "top1_freq": max_freq,
            "top1_value": sorted_items[0][0],
            "min_freq": min_freq,
            "top10_coverage": f"{top10_freq / sum(count for _, count in sorted_items):.2%}"
        }

        print(f"  '{name}' → 唯一值: {total_ids:,}, 最高频率: {max_freq}, 最低频率: {min_freq}")

def t_print(input_str: str):
    """
    print with timestamp
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + input_str)

def form_sample_table(mode: str):
    # 并不能保证gpu负载均衡，还需要对随后几个shard做一些调整

    if mode == "train":
        sample_file_name = "train_samples.full.parquet"
        user_file_name = "train_user_features.full.parquet"
    elif mode == "test":
        sample_file_name = "test_samples.full.parquet"
        user_file_name = "test_user_features.full.parquet"
    else:
        raise ValueError("mode must be 'train' or 'test'")
    
    item_file_name = "item_features.full.parquet"

    schema = [
        ('label_0', pa.list_(pa.int8())),
        ('129_1', pa.int64()), # udi
        ('130_1', pa.int64()), # u
        ('130_2', pa.int64()), # u
        ('130_3', pa.int64()), # u
        ('130_4', pa.int64()), # u
        ('130_5', pa.int64()), # u
        ('150_2_180', pa.list_(pa.int64())), # u
        ('151_2_180', pa.list_(pa.int64())), # u
        ('205', pa.int64()), # iid
        ('206', pa.int64()), # i
        ('213', pa.int64()), # i
        ('214', pa.int64()), # i
    ]
    schema = pa.schema(schema)

    t_print(f"🚀 开始构建 {mode} 样本表...")

    SAMPLE_FILE = os.path.join(OUTPUT_DIR, sample_file_name)
    ITEM_FILE = os.path.join(OUTPUT_DIR, item_file_name)
    USER_FILE = os.path.join(OUTPUT_DIR, user_file_name)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, mode)

    # ==============================
    # Step 1: 加载 item features 到内存（小文件）
    # ==============================
    t_print("📦 加载 item_features...")
    try:
        item_table = pq.read_table(ITEM_FILE)
        con = duckdb.connect()
        con.register("item_table", item_table)
        con.execute("CREATE OR REPLACE VIEW item_feat AS SELECT * FROM item_table")
    except Exception as e:
        t_print(f"❌ 加载 item features 失败: {e}")
        raise
    
    query = f"""
    SELECT 
        s."label_0",
        s."129_1",
        u."130_1",
        u."130_2",
        u."130_3",
        u."130_4",
        u."130_5",
        u."150_2_180",
        u."151_2_180",
        s."205",
        i."206",
        i."213",
        i."214"
    FROM '{SAMPLE_FILE}' AS s
    LEFT JOIN '{USER_FILE}' AS u ON s."129_1" = u."129_1"
    LEFT JOIN item_feat AS i ON s."205" = i."205"
    """
    
    # ==============================
    # Step 2: 使用 DuckDB 流式 join 并分片写入
    # ==============================
    t_print("🔍 执行高效 Join 并分片写入...")
    BATCH_SIZE = 100_000

    shard_id = 0
    total_rows = 0

    result = con.query(query).fetch_arrow_reader(batch_size=BATCH_SIZE)

    try:
        # 缓冲区变量
        batches_buffer = []           # 存放多个 batch 的 DataFrame
        rows_in_current_shard = 0     # 当前 shard 已累计行数
        TARGET_ROWS_PER_SHARD = 500_000

        for batch_idx, record_batch in enumerate(result):
            df = record_batch.to_pandas()

            # 类型转换
            for col in df.columns:
                expected_type = schema.field(col).type
                if pa.types.is_list(expected_type):
                    pass  # list 类型保持原样
                elif pa.types.is_int64(expected_type):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif pa.types.is_int8(expected_type):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int8')

            # 添加到缓冲区
            batches_buffer.append(df)
            rows_in_current_shard += len(df)

            total_rows += len(df)

            # 只有当累计足够多行时才写入一个 shard
            if rows_in_current_shard >= TARGET_ROWS_PER_SHARD:
                # 合并所有 batch
                shard_df = pd.concat(batches_buffer, ignore_index=True)

                # 转为 PyArrow Table
                table = pa.Table.from_pandas(shard_df, schema=schema, preserve_index=False)

                # 写入分片文件
                shard_file = os.path.join(OUTPUT_PATH, f"{mode}-shard-{shard_id:06d}.parquet")
                pq.write_table(
                    table,
                    shard_file,
                    row_group_size=10_000,
                    compression="ZSTD",
                    use_dictionary=False
                )

                # 日志
                t_print(f"  ✅ 写入 shard-{shard_id:06d}.parquet，包含 {len(shard_df):,} 行，共处理 {total_rows:,} 行")

                # 重置缓冲区
                batches_buffer = []
                rows_in_current_shard = 0
                shard_id += 1

        # ==============================
        # 处理最后剩余的数据（不足一个完整 shard）
        # ==============================
        if batches_buffer:
            shard_df = pd.concat(batches_buffer, ignore_index=True)
            table = pa.Table.from_pandas(shard_df, schema=schema, preserve_index=False)

            shard_file = os.path.join(OUTPUT_PATH, f"shard-{shard_id:06d}.parquet")
            pq.write_table(
                table,
                shard_file,
                row_group_size=10_000,
                compression="ZSTD",
                use_dictionary=False
            )
            t_print(f"  ✅ 写入最后一个 {mode}-shard-{shard_id:06d}.parquet，包含 {len(shard_df):,} 行")
            shard_id += 1

    except Exception as e:
        t_print(f"❌ 处理过程中出错: {e}")
        raise
    finally:
        result.close()

    
    t_print(f"🎉 {mode} 样本表构建完成！")
    t_print(f"   总样本数: {total_rows:,}")
    t_print(f"   分片数量: {shard_id}")
    t_print(f"   输出路径: {OUTPUT_PATH}")

    # ==============================
    # Step 3: 生成 metadata.json（可选）
    # ==============================
    meta = {
        "mode": mode,
        "total_rows": total_rows,
        "num_shards": shard_id,
        "schema": [str(field) for field in schema],
        "created_at": pd.Timestamp.now().isoformat()
    }
    with open(os.path.join(OUTPUT_PATH, "metadata.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)

    t_print(f"📌 元数据已保存至 {os.path.join(OUTPUT_PATH, 'metadata.json')}")

def find_feq_json(feat_name, freq_json_files):
    freq_json_file = [f for f in freq_json_files if feat_name in f]
    assert len(freq_json_file) == 1
    return freq_json_file[0]

def form_vocab_table(feat_name_list):
    JSON_FILE_PATH = os.path.join(OUTPUT_DIR, "json")
    # NPZ_FILE_PATH = os.path.join(OUTPUT_DIR, "npy")
    NPZ_FILE_PATH = "./npy"

    freq_json_files = os.listdir(JSON_FILE_PATH)

    for feat_name in feat_name_list:
        freq_json_file = find_feq_json(feat_name, freq_json_files)
        with open(os.path.join(JSON_FILE_PATH, freq_json_file), "r") as f:
            freq_dict = json.load(f)

        # share embeddings
        if feat_name in ["150_2_180", "151_2_180"]:
            if feat_name == "150_2_180":
                target_json_file = find_feq_json("205", freq_json_files)
            if feat_name == "151_2_180":
                target_json_file = find_feq_json("206", freq_json_files)
            with open(os.path.join(JSON_FILE_PATH, target_json_file), "r") as f:
                target_freq_dict = json.load(f)
            
            merged = {**target_freq_dict, **freq_dict}
            # sorted_merged = dict(sorted(merged.items(), key=lambda x: x[1], reverse=True))  
            freq_dict = merged

        if "0" in freq_dict:
            del freq_dict["0"]

        sorted_id_list = sorted([int(k) for k in freq_dict.keys()])
        keys_int = np.array(sorted_id_list, dtype=np.int64)

        print(f"{feat_name} vocab size: {len(keys_int)}")
        np.savez(os.path.join(NPZ_FILE_PATH, f"{feat_name}_sorted_map.npz"), keys=keys_int)

def convert_scl_int8():
    '''
    convert float to int8
    '''
    SCL_FILE_PATH = os.path.join(OUTPUT_DIR, "scl_embedding_p90.full.parquet")
    NPZ_FILE_PATH = "./"

    df = pd.read_parquet(SCL_FILE_PATH)
    ids = df['205'].values
    embeddings = np.stack(df['205_c'].values)
    embeddings_clipped = np.clip(embeddings, -1.0, 1.0)
    # int8_embeddings = np.round(embeddings_clipped * 127).astype(np.int8)
    int8_embeddings = np.trunc(embeddings_clipped * 127).astype(np.int8)
    sort_idx = np.argsort(ids)
    sorted_ids = ids[sort_idx]
    sorted_int8_embs = int8_embeddings[sort_idx]

    np.savez(
        os.path.join(NPZ_FILE_PATH, f"scl_emb_int8_p90.npz"),
        keys=sorted_ids,
        values=sorted_int8_embs,
        scale=np.array(1/127.0, dtype=np.float32)
    )

def parquet_to_npz():
    SCL_FILE_PATH = os.path.join(OUTPUT_DIR, "sscl_embedding_int8_p90.parquet")
    NPZ_FILE_PATH = "./"

    df = pd.read_parquet(SCL_FILE_PATH)
    ids = df['205'].values
    embeddings = np.stack(df['205_c'].values)

    sort_idx = np.argsort(ids)
    sorted_ids = ids[sort_idx]
    sorted_int8_embs = embeddings[sort_idx]

    np.savez(
        os.path.join(NPZ_FILE_PATH, f"scl_emb_int8_p90.npz"),
        keys=sorted_ids,
        values=sorted_int8_embs,
        scale=np.array(1/127.0, dtype=np.float32)
    )

def get_config_dict():
    config_dict = {
        "train_samples.txt": {
            "feat_fn": ['label_0', '129_1', '205'],
            "schema": [
                ('label_0', pa.list_(pa.int8())),
                ('129_1', pa.int64()),
                ('205', pa.int64()),
            ]
        },
        "test_samples.txt": {
            "feat_fn": ['label_0', '129_1', '205'],
            "schema": [
                ('label_0', pa.list_(pa.int8())),
                ('129_1', pa.int64()),
                ('205', pa.int64()),
            ]
        },
        "item_features.txt": {
            "feat_fn": ["205", "206", "213", "214"],
            "schema": [
                ('205', pa.int64()),
                ('206', pa.int64()),
                ('213', pa.int64()),
                ('214', pa.int64()),
            ]
        },
        "train_user_features.txt": {
            "feat_fn": user_fn+uni_seq_fn,
            "schema": [
                ('129_1', pa.int64()),
                ('130_1', pa.int64()),
                ('130_2', pa.int64()),
                ('130_3', pa.int64()),
                ('130_4', pa.int64()),
                ('130_5', pa.int64()),
                # ('150_1_180', pa.list_(pa.int64())),
                # ('151_1_180', pa.list_(pa.int64())),
                ('150_2_180', pa.list_(pa.int64())),
                ('151_2_180', pa.list_(pa.int64())),
            ]
        },
        "test_user_features.txt": {
            "feat_fn": user_fn+uni_seq_fn,
            "schema": [
                ('129_1', pa.int64()),
                ('130_1', pa.int64()),
                ('130_2', pa.int64()),
                ('130_3', pa.int64()),
                ('130_4', pa.int64()),
                ('130_5', pa.int64()),
                # ('150_1_180', pa.list_(pa.int64())),
                # ('151_1_180', pa.list_(pa.int64())),
                ('150_2_180', pa.list_(pa.int64())),
                ('151_2_180', pa.list_(pa.int64())),
            ]
        },
        "scl_embedding_p90.txt": {
            "feat_fn": ["205", "205_c"],
            "schema": [
                ('205', pa.int64()),
                ('205_c', pa.list_(pa.float32())),
            ]
        }
    }
    return config_dict

def reconstruct_shards():
    # ================= 配置参数 =================
    # mode = "train"      
    # start_idx = 152                               
    # num_tail_shards = 1

    mode = "test"      
    start_idx = 40                               
    num_tail_shards = 6

    num_gpus = 8
    new_shard_base_idx = start_idx

    data_dir = f"./taobao-mm/{mode}"         
    output_dir = "./parquet"  
    # ==============================================

    if start_idx % 8 != 0:
        raise ValueError("非法索引")
        
    tail_files = [
        os.path.join(data_dir, f"{mode}-shard-{i:06d}.parquet")
        for i in range(start_idx, start_idx + num_tail_shards)
    ]

    tables = []
    for file_path in tail_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        table = pq.read_table(file_path)
        tables.append(table)

    combined_table = pa.concat_tables(tables)
    total_rows = len(combined_table)
    print(f"最后{num_tail_shards}个 shard 共 {total_rows} 行数据。")

    # 3. 计算前8个 shard 每个的行数（尽量平均，第9个为尾部）
    K = total_rows // num_gpus      # 每个 GPU 分配的行数（前8个文件各 K 行）
    M = total_rows % num_gpus       # 余数，放入第9个文件

    print(f"前8个新 shard 每个包含 {K} 行, 第9个包含 {M} 行。")

    # 4. 分割并写入新文件
    os.makedirs(output_dir, exist_ok=True)

    next_idx = new_shard_base_idx
    for i in range(num_gpus):
        start = i * K
        end = start + K
        if K > 0:
            subset = combined_table.slice(start, K)
            out_path = os.path.join(output_dir, f"{mode}-shard-{next_idx:06d}.parquet")
            pq.write_table(subset, out_path, compression="ZSTD")
            print(f"写入: {out_path} ({K} 行)")
            next_idx += 1

    if M > 0:
        start = num_gpus * K
        subset = combined_table.slice(start, M)
        out_path = os.path.join(output_dir, f"{mode}-shard-{next_idx:06d}.parquet")
        pq.write_table(subset, out_path, compression="ZSTD")
        print(f"写入尾部: {out_path} ({M} 行)")
    else:
        print("无尾部数据，不生成多余文件。")

    print("✅ 重构完成。")


if __name__ == '__main__':

    # count_frequency()

    # form_sample_table("train")
    # form_sample_table("test")

    # 随后调整最后几个shard使之能被1、2、4、8 gpu均匀分配
    # reconstruct_shards()

    # feat_name_list = [
    #     "129_1", "130_1", "130_2", "130_3", "130_4", "130_5",
    #     "150_2_180", "151_2_180",
    #     "213", "214"
    # ]
    # form_vocab_table(feat_name_list)

    # convert_scl_int8()
    # or
    # parquet_to_npz()

    # transfer npz to npy for more efficient read

    pass