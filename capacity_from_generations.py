# coding=utf-8
import argparse
import csv
import json
import os
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lm_eval import tasks
from lm_eval.utils import calculate_entropy
from sweet_bimark import SweetBimarkDetector


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def norm_bits(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return "".join(str(v) for v in x if str(v) in ("0", "1", "x", "X")).replace("X", "x")
    s = str(x).replace("X", "x")
    return "".join(ch for ch in s if ch in ("0", "1", "x"))


def bit_metrics(decoded_bits: str, gt_bits: str):
    gt = norm_bits(gt_bits)
    dec = norm_bits(decoded_bits)

    if not gt:
        return 0.0, 0

    correct = 0
    for i in range(len(gt)):
        if i < len(dec) and dec[i] == gt[i]:
            correct += 1

    acc = 100.0 * correct / len(gt)
    exact = 1 if dec == gt else 0
    return acc, exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--generations", type=str, required=True)
    parser.add_argument("--bits", type=str, required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--entropy_threshold", type=float, default=1.2)
    parser.add_argument("--partition_seeds", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--c_key", type=int, default=530773)
    parser.add_argument("--bit_idx_key", type=int, default=283519)
    parser.add_argument("--z_threshold", type=float, default=4.0)
    parser.add_argument("--prefix_lengths", type=str, default="256,512,1024,1536,2048")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out_prefix", type=str, required=True)
    parser.add_argument("--precision", type=str, default="fp16")
    args = parser.parse_args()

    prefix_lengths = parse_int_list(args.prefix_lengths)
    partition_seeds = parse_int_list(args.partition_seeds)

    # 1) 载入 task，并确保 prompt 固定
    task = tasks.get_task(args.task)
    if hasattr(task, "few_shot"):
        task.few_shot = 0

    dataset = task.get_dataset()

    # 2) 载入 generations.json
    with open(args.generations, "r", encoding="utf-8") as f:
        generations = json.load(f)

    n = min(args.limit, len(generations), len(dataset))
    print(f"[INFO] use first {n} samples")

    # 3) 载入 tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if args.precision == "fp16" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # 4) 检测器
    detector = SweetBimarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        tokenizer=tokenizer,
        z_threshold=args.z_threshold,
        entropy_threshold=args.entropy_threshold,
        partition_seeds=partition_seeds,
        c_key=args.c_key,
        bit_idx_key=args.bit_idx_key,
        window_size=args.window_size,
        bits=args.bits,
        use_hist=False,
    )

    rows = []

    for i in range(n):
        doc = dataset[i]
        prompt = task.get_prompt(doc)

        gen_item = generations[i]
        if isinstance(gen_item, list):
            gen_text = gen_item[0]
        else:
            gen_text = gen_item

        # 有些保存结果里已经带 prompt，有些不带，这里两种都兼容
        if isinstance(gen_text, str) and gen_text.startswith(prompt):
            full_text = gen_text
        else:
            full_text = prompt + gen_text

        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

        if len(full_ids) <= len(prompt_ids):
            print(f"[WARN] sample {i}: full_ids <= prompt_ids, skip")
            continue

        completion_ids = full_ids[len(prompt_ids):]

        for L in prefix_lengths:
            if len(completion_ids) < L:
                continue

            prefix_completion_ids = completion_ids[:L]
            text_ids = torch.cat([prompt_ids, prefix_completion_ids], dim=0)

            # 熵必须对“prompt + 当前前缀”整体算
            entropy = calculate_entropy(model, text_ids.to(model.device))

            result = detector.detect(
                tokenized_text=text_ids.cpu(),
                tokenized_prefix=prompt_ids.cpu(),
                entropy=entropy,
                return_prediction=True,
                return_scores=True,
                decode=True,
            )

            decoded_bits = result.get("decoded_bits", "")
            bit_acc, exact = bit_metrics(decoded_bits, args.bits)

            rows.append({
                "task_id": i,
                "payload_bits": len(args.bits),
                "prefix_len": L,
                "num_tokens_generated_prefix": L,
                "num_tokens_scored": result.get("num_tokens_scored", 0),
                "z_score": result.get("z_score", 0.0),
                "prediction": int(bool(result.get("prediction", False))),
                "decoded_bits": decoded_bits,
                "original_bits": args.bits,
                "bit_accuracy": bit_acc,
                "exact_recovery": exact,
            })

    # 5) 保存明细
    detail_csv = args.out_prefix + "_detail.csv"
    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {detail_csv}")

    # 6) 按 prefix_len 聚合
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["prefix_len"]].append(r)

    summary_rows = []
    for L in sorted(grouped.keys()):
        g = grouped[L]
        n_samples = len(g)
        avg_scored = sum(x["num_tokens_scored"] for x in g) / n_samples
        avg_z = sum(x["z_score"] for x in g) / n_samples
        avg_acc = sum(x["bit_accuracy"] for x in g) / n_samples
        exact_rate = 100.0 * sum(x["exact_recovery"] for x in g) / n_samples
        det_rate = 100.0 * sum(x["prediction"] for x in g) / n_samples
        eff_per_bit = avg_scored / len(args.bits)

        summary_rows.append({
            "prefix_len": L,
            "n_samples": n_samples,
            "avg_num_tokens_scored": round(avg_scored, 4),
            "avg_z_score": round(avg_z, 4),
            "eff_per_bit": round(eff_per_bit, 4),
            "macro_bit_accuracy": round(avg_acc, 4),
            "exact_recovery_rate": round(exact_rate, 4),
            "detected_rate": round(det_rate, 4),
        })

    summary_csv = args.out_prefix + "_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[SAVED] {summary_csv}")

    print("\n===== SUMMARY =====")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()