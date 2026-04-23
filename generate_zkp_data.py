#!/usr/bin/env python3
# coding=utf-8
"""
generate_zkp_data.py - 独立的零知识证明数据生成脚本

【使用方法】
在你原有的水印检测流程跑完之后，运行这个脚本来生成 ZKP 数据文件。

python generate_zkp_data.py \
    --model bigcode/starcoderbase \
    --generations_path outputs/generations.json \
    --bits "01010101" \
    --partition_seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23" \
    --gamma 0.25 \
    --entropy_threshold 1.2 \
    --tau_z 4.0 \
    --model_id 123456 \
    --e_max 1 \
    --output_dir outputs/zkp_data
"""

import argparse
import json
import os
import hashlib
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def prf(seed, secret_key: int) -> int:
    """伪随机函数，与 sweet_bimark.py 保持一致"""
    if isinstance(seed, tuple):
        seed_str = ''.join(map(str, seed)) + str(secret_key)
    else:
        seed_str = str(seed) + str(secret_key)
    hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
    return int(hash_digest, 16) % 2 ** 32


def calculate_entropy(model, tokenized_text, device):
    """计算每个位置的熵"""
    with torch.no_grad():
        output = model(torch.unsqueeze(tokenized_text, 0).to(device), return_dict=True)
        probs = torch.softmax(output.logits, dim=-1)
        entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
        return entropy[0].cpu().tolist()


def extract_zkp_data(
        input_ids: torch.Tensor,
        prefix_len: int,
        entropy: list,
        vocab_size: int,
        gamma: float,
        entropy_threshold: float,
        partition_seeds: list,
        c_key: int,
        bit_idx_key: int,
        window_size: int,
        bits: str,
        model_id: int,
        tau_z: float,
        e_max: int,
        use_hist: bool = False,
):
    """从检测数据中提取零知识证明所需的所有信息"""

    n = len(input_ids) - prefix_len
    l = len(partition_seeds)
    bits_len = len(bits)
    m = [int(b) for b in bits]

    # 构建词表划分掩码
    partition_masks = []
    for key in partition_seeds:
        num_V0 = int(vocab_size * gamma)
        rng = np.random.default_rng(key)
        mask = np.zeros(vocab_size, dtype=bool)
        mask[rng.choice(vocab_size, num_V0, replace=False)] = True
        partition_masks.append(mask)

    # 初始化数据结构
    h = []
    g = []
    r = []
    b = []
    bit_idx = []

    hist = set()

    for t_offset in range(n):
        t_abs = prefix_len + t_offset

        h_t = 0
        g_t = [0] * l
        r_t = [0] * l
        b_t = [0] * l
        bit_idx_t = 0

        skip = False

        # 熵门控检查
        if t_abs < len(entropy) and entropy[t_abs] <= entropy_threshold:
            skip = True

        # 窗口大小检查
        if t_abs < window_size:
            skip = True

        # 获取前缀
        if not skip:
            prefix_start = t_abs - window_size
            prefix = tuple(input_ids[prefix_start:t_abs].tolist())

            # hist 去重
            if use_hist:
                if prefix in hist:
                    skip = True
                else:
                    hist.add(prefix)

        if skip:
            h.append(0)
            g.append(g_t)
            r.append(r_t)
            b.append(b_t)
            bit_idx.append(0)
            continue

        # 触发位置
        h_t = 1
        curr_token = input_ids[t_abs].item()

        # 生成 c_list 和 bit_idx
        c_seed = prf(prefix, c_key)
        bit_idx_seed = prf(prefix, bit_idx_key)

        rng_c = np.random.default_rng(c_seed)
        c_list = rng_c.integers(0, 2, size=l)

        rng_bit_idx = np.random.default_rng(bit_idx_seed)
        bit_idx_t = int(rng_bit_idx.integers(0, bits_len))

        # 对每一层
        for i in range(l):
            in_V0 = bool(partition_masks[i][curr_token])
            c_i = int(c_list[i])

            g_t[i] = 1 if in_V0 else 0
            r_t[i] = 1 if not in_V0 else 0
            b_t[i] = c_i

        h.append(h_t)
        g.append(g_t)
        r.append(r_t)
        b.append(b_t)
        bit_idx.append(bit_idx_t)

    # 验证解码
    vote_0 = [0] * bits_len
    vote_1 = [0] * bits_len

    for t in range(n):
        if h[t] == 1:
            p = bit_idx[t]
            for i in range(l):
                m_hat = g[t][i] ^ b[t][i]
                if m_hat == 1:
                    vote_1[p] += 1
                else:
                    vote_0[p] += 1

    decoded = ''.join('1' if vote_1[p] > vote_0[p] else '0' for p in range(bits_len))
    ber = sum(1 for i in range(bits_len) if decoded[i] != bits[i])

    zkp_data = {
        "model_id_u64": model_id,
        "tau_times_1000": int(tau_z * 1000),
        "gamma_times_1000": int(gamma * 1000),
        "e_max": e_max,
        "n": n,
        "l": l,
        "bits_len": bits_len,
        "m": m,
        "h": h,
        "g": g,
        "r": r,
        "b": b,
        "bit_idx": bit_idx,
    }

    info = {
        "active_count": sum(h),
        "decoded_bits": decoded,
        "original_bits": bits,
        "decode_match": decoded == bits,
        "ber": ber,
    }

    return zkp_data, info


def main():
    parser = argparse.ArgumentParser(description="Generate ZKP data from watermarked generations")

    # 必需参数
    parser.add_argument("--model", type=str, required=True, help="Model name for tokenizer and entropy calculation")
    parser.add_argument("--generations_path", type=str, required=True, help="Path to generations.json")
    parser.add_argument("--bits", type=str, required=True, help="Message bits (e.g., '01010101')")

    # 水印参数（需要与生成时一致）
    parser.add_argument("--partition_seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated partition seeds")
    parser.add_argument("--gamma", type=float, default=0.5, help="Green list ratio")
    parser.add_argument("--entropy_threshold", type=float, default=0.5, help="Entropy threshold")
    parser.add_argument("--c_key", type=int, default=530773, help="Key for c_list generation")
    parser.add_argument("--bit_idx_key", type=int, default=283519, help="Key for bit_idx generation")
    parser.add_argument("--window_size", type=int, default=2, help="Window size for prefix")
    parser.add_argument("--use_hist", action="store_true", help="Enable hist deduplication")

    # ZKP 参数
    parser.add_argument("--tau_z", type=float, default=4.0, help="Z-score threshold for ZKP")
    parser.add_argument("--model_id", type=int, default=0, help="Model ID for ZKP")
    parser.add_argument("--e_max", type=int, default=1, help="Max allowed bit errors")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="zkp_data", help="Output directory for ZKP files")
    parser.add_argument("--task", type=str, default="humaneval", help="Task name for prompt extraction")
    parser.add_argument("--precision", type=str, default="bf16", help="Model precision")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to process")

    args = parser.parse_args()

    # 解析 partition_seeds
    partition_seeds = [int(x) for x in args.partition_seeds.split(',')]

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 generations
    print(f"Loading generations from {args.generations_path}")
    with open(args.generations_path, 'r') as f:
        generations = json.load(f)

    # 加载 tokenizer 和 model
    print(f"Loading model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    dict_precisions = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dict_precisions.get(args.precision, torch.bfloat16),
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # 加载 task 获取 prompts
    from lm_eval import tasks
    task = tasks.get_task(args.task)
    dataset = task.get_dataset()

    n_tasks = args.limit if args.limit else len(generations)

    # 处理每个生成
    zkp_summary = []
    total_match = 0
    total_count = 0

    print(f"\nProcessing {n_tasks} tasks...")
    for idx in tqdm(range(n_tasks)):
        if idx >= len(generations):
            break

        gens = generations[idx]
        prompt = task.get_prompt(dataset[idx])

        for idx2, gen in enumerate(gens):
            # Tokenize
            if not gen.startswith(prompt):
                gen = prompt + gen

            tokenized = tokenizer(gen, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = tokenized.input_ids.squeeze()

            tokenized_prefix = tokenizer(prompt, return_tensors="pt")
            prefix_len = tokenized_prefix.input_ids.shape[1]

            if len(input_ids) - prefix_len < 1:
                print(f"  Skipping task{idx}_sample{idx2}: empty generation")
                continue

            # 计算熵
            entropy = calculate_entropy(model, input_ids, device)
            entropy = [0] + entropy[:-1]  # 右移一位

            # 提取 ZKP 数据
            zkp_data, info = extract_zkp_data(
                input_ids=input_ids,
                prefix_len=prefix_len,
                entropy=entropy,
                vocab_size=vocab_size,
                gamma=args.gamma,
                entropy_threshold=args.entropy_threshold,
                partition_seeds=partition_seeds,
                c_key=args.c_key,
                bit_idx_key=args.bit_idx_key,
                window_size=args.window_size,
                bits=args.bits,
                model_id=args.model_id,
                tau_z=args.tau_z,
                e_max=args.e_max,
                use_hist=args.use_hist,
            )

            # 保存 ZKP 文件
            output_path = os.path.join(args.output_dir, f"zkp_task{idx}_sample{idx2}.json")
            with open(output_path, 'w') as f:
                json.dump(zkp_data, f, indent=2)

            # 记录结果
            zkp_summary.append({
                "task_idx": idx,
                "sample_idx": idx2,
                "zkp_file": output_path,
                "active_count": info["active_count"],
                "decoded_bits": info["decoded_bits"],
                "original_bits": info["original_bits"],
                "decode_match": info["decode_match"],
                "ber": info["ber"],
            })

            total_count += 1
            if info["decode_match"]:
                total_match += 1

            status = "✓" if info["decode_match"] else "✗"
            print(
                f"  task{idx}_sample{idx2}: {status} decoded={info['decoded_bits']} active={info['active_count']} ber={info['ber']}")

    # 保存汇总
    summary_path = os.path.join(args.output_dir, "zkp_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(zkp_summary, f, indent=2)

    print(f"\n" + "=" * 50)
    print(f"ZKP Data Generation Complete!")
    print(f"  Total: {total_count}")
    print(f"  Decode Match: {total_match}")
    print(f"  Accuracy: {total_match / total_count:.4f}" if total_count > 0 else "  Accuracy: N/A")
    print(f"  Output: {args.output_dir}")
    print(f"  Summary: {summary_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()