# -*- coding: utf-8 -*-
"""
Compute bit recovery rate (bit accuracy) from evaluation_results.json
after attacks like variable renaming.

It searches for a list of per-sample dict records (raw_detection_results),
then extracts decoded bits from keys such as:
  decoded_bits / decoded_message / recovered_bits / message_bits / extracted_bits

It reports:
- completion-level mean bit accuracy
- completion-level exact match rate
- (optional) problem-level majority-vote bit accuracy if a problem id exists
- per-bit accuracy vector

Usage:
  python compute_bit_recovery_rate.py \
    --eval_json outputs/rename25_seed0/evaluation_results.json \
    --bits "01010101"

  python compute_bit_recovery_rate.py \
    --eval_dir outputs \
    --bits "01010101" \
    --out_csv bit_recovery_summary.csv
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


BIT_KEYS = [
    "decoded_bits",
    "decoded_message",
    "recovered_bits",
    "message_bits",
    "extracted_bits",
    "decoded",
    "message",
]

ID_KEYS = [
    "task_id",
    "doc_id",
    "problem_id",
    "prompt_id",
    "id",
    "idx",
    "index",
]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_bits(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = "".join(ch for ch in x if ch in "01")
        return s if s else None
    if isinstance(x, (list, tuple)):
        if not x:
            return None
        if all(isinstance(v, int) for v in x):
            return "".join("1" if int(v) else "0" for v in x)
        if all(isinstance(v, str) for v in x):
            s = "".join(ch for v in x for ch in v if ch in "01")
            return s if s else None
    return None


def pick_task_root(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("Root of eval json is not a dict.")
    for k in ["humaneval", "mbpp", "ds1000", "ds1000-all-completion", "ds-1000"]:
        if k in data and isinstance(data[k], dict):
            return k, data[k]
    dict_keys = [k for k, v in data.items() if isinstance(v, dict)]
    if len(dict_keys) == 1:
        k = dict_keys[0]
        return k, data[k]
    raise ValueError(f"Cannot determine task root. Top keys: {list(data.keys())[:30]}")


def find_records(task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # common SWEET layouts
    candidates = [
        ("watermark_detection", "raw_detection_results"),
        ("watermark_detection", "detection_results"),
        ("watermark_detection", "results"),
        ("raw_detection_results",),
        ("detection_results",),
    ]
    for path in candidates:
        cur: Any = task_data
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and isinstance(cur, list) and all(isinstance(x, dict) for x in cur):
            return cur

    # last resort scan
    def scan(obj: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            if any(any(k in d for k in BIT_KEYS) for d in obj):
                return obj
        if isinstance(obj, dict):
            for v in obj.values():
                r = scan(v)
                if r is not None:
                    return r
        if isinstance(obj, list):
            for v in obj:
                r = scan(v)
                if r is not None:
                    return r
        return None

    found = scan(task_data)
    if found is not None:
        return found
    raise ValueError("Cannot find per-sample detection records in eval json.")


def extract_decoded_bits(rec: Dict[str, Any]) -> Optional[str]:
    for k in BIT_KEYS:
        if k in rec:
            s = normalize_bits(rec[k])
            if s is not None:
                return s
    # sometimes nested
    for subk in ["watermark", "wm", "decode", "decoded", "message", "meta"]:
        v = rec.get(subk)
        if isinstance(v, dict):
            for k in BIT_KEYS:
                if k in v:
                    s = normalize_bits(v[k])
                    if s is not None:
                        return s
    return None


def extract_problem_id(rec: Dict[str, Any]) -> Optional[str]:
    for k in ID_KEYS:
        if k in rec:
            return str(rec[k])
    meta = rec.get("meta")
    if isinstance(meta, dict):
        for k in ID_KEYS:
            if k in meta:
                return str(meta[k])
    return None


def bit_acc(decoded: str, original: str) -> float:
    L = len(original)
    if L == 0:
        return 0.0
    correct = 0
    for i in range(L):
        if i < len(decoded) and decoded[i] == original[i]:
            correct += 1
    return correct / L


def majority_vote(bitstrings: List[str], L: int) -> str:
    out = []
    for i in range(L):
        ones = 0
        zeros = 0
        for s in bitstrings:
            if i >= len(s):
                continue
            if s[i] == "1":
                ones += 1
            elif s[i] == "0":
                zeros += 1
        out.append("1" if ones > zeros else "0")  # tie -> 0
    return "".join(out)


def infer_labels(path: str) -> Tuple[Optional[str], Optional[str]]:
    p = path.replace("\\", "/")
    m_ratio = re.search(r"rename(\d+)", p)
    m_seed = re.search(r"seed(\d+)", p)
    ratio = m_ratio.group(1) if m_ratio else None
    seed = m_seed.group(1) if m_seed else None
    return ratio, seed


def compute_one(eval_path: str, original_bits: str) -> Dict[str, Any]:
    data = load_json(eval_path)
    task, task_data = pick_task_root(data)
    records = find_records(task_data)

    decoded = []
    prob_ids = []
    missing = 0

    for rec in records:
        b = extract_decoded_bits(rec)
        if b is None:
            missing += 1
            continue
        decoded.append(b)
        prob_ids.append(extract_problem_id(rec))

    if not decoded:
        keys = list(records[0].keys()) if records else []
        raise ValueError(
            f"No decoded bits found in {eval_path}. "
            f"First record keys: {keys[:60]}."
        )

    L = len(original_bits)
    per_sample = [bit_acc(b, original_bits) for b in decoded]
    mean_acc = sum(per_sample) / len(per_sample)
    exact = sum(1 for b in decoded if b == original_bits) / len(decoded)

    # per-bit accuracy
    per_bit_correct = [0] * L
    per_bit_count = [0] * L
    for b in decoded:
        for i in range(L):
            per_bit_count[i] += 1
            if i < len(b) and b[i] == original_bits[i]:
                per_bit_correct[i] += 1
    per_bit = [per_bit_correct[i] / per_bit_count[i] for i in range(L)]

    # problem-level majority vote if ids exist
    problem_level = None
    if any(pid is not None for pid in prob_ids):
        by_id = defaultdict(list)
        for b, pid in zip(decoded, prob_ids):
            if pid is not None:
                by_id[pid].append(b)
        if by_id:
            voted = [majority_vote(arr, L) for arr in by_id.values()]
            voted_acc = sum(bit_acc(v, original_bits) for v in voted) / len(voted)
            voted_exact = sum(1 for v in voted if v == original_bits) / len(voted)
            problem_level = {
                "n_problems": len(by_id),
                "majority_vote_bit_accuracy_mean": voted_acc,
                "majority_vote_exact_match_rate": voted_exact,
            }

    ratio, seed = infer_labels(eval_path)

    return {
        "eval_path": eval_path,
        "task": task,
        "rename_ratio": ratio,
        "seed": seed,
        "n_samples": len(decoded),
        "n_missing_decodes": missing,
        "completion_bit_accuracy_mean": mean_acc,
        "completion_exact_match_rate": exact,
        "per_bit_accuracy": per_bit,
        "problem_level": problem_level,
    }


def scan_eval_files(root_dir: str, filename: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn == filename:
                out.append(os.path.join(r, fn))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", default=None,
                    help="Comma-separated paths to evaluation_results.json")
    ap.add_argument("--eval_dir", default=None,
                    help="Root dir to scan recursively for evaluation_results.json")
    ap.add_argument("--eval_name", default="evaluation_results.json",
                    help="Filename to scan under --eval_dir")
    ap.add_argument("--bits", required=True,
                    help='Original embedded bitstring like "01010101"')
    ap.add_argument("--out_csv", default=None,
                    help="Write per-file summary to CSV")
    args = ap.parse_args()

    original_bits = normalize_bits(args.bits)
    if not original_bits:
        raise ValueError("Invalid --bits, must contain 0/1.")

    files = []
    if args.eval_json:
        files += [p.strip() for p in args.eval_json.split(",") if p.strip()]
    if args.eval_dir:
        files += scan_eval_files(args.eval_dir, args.eval_name)
    files = sorted(set(files))

    if not files:
        raise ValueError("No eval files found. Use --eval_json or --eval_dir.")

    summaries = []
    for p in files:
        s = compute_one(p, original_bits)
        summaries.append(s)

        print("\n" + "=" * 90)
        print("file", p)
        print("task", s["task"])
        if s["rename_ratio"] or s["seed"]:
            print("label", "rename_ratio", s["rename_ratio"], "seed", s["seed"])
        print("samples", s["n_samples"], "missing_decodes", s["n_missing_decodes"])
        print("completion_bit_accuracy_mean", f"{s['completion_bit_accuracy_mean']:.4f}")
        print("completion_exact_match_rate", f"{s['completion_exact_match_rate']:.4f}")
        if s["problem_level"]:
            pl = s["problem_level"]
            print("problem_level_n", pl["n_problems"])
            print("majority_vote_bit_accuracy_mean", f"{pl['majority_vote_bit_accuracy_mean']:.4f}")
            print("majority_vote_exact_match_rate", f"{pl['majority_vote_exact_match_rate']:.4f}")
        print("per_bit_accuracy", " ".join(f"{x:.2f}" for x in s["per_bit_accuracy"]))

    if args.out_csv:
        cols = [
            "eval_path", "task", "rename_ratio", "seed",
            "n_samples", "n_missing_decodes",
            "completion_bit_accuracy_mean", "completion_exact_match_rate",
            "problem_level_n_problems",
            "problem_level_majority_vote_bit_accuracy_mean",
            "problem_level_majority_vote_exact_match_rate",
        ]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for s in summaries:
                pl = s["problem_level"] or {}
                w.writerow({
                    "eval_path": s["eval_path"],
                    "task": s["task"],
                    "rename_ratio": s["rename_ratio"],
                    "seed": s["seed"],
                    "n_samples": s["n_samples"],
                    "n_missing_decodes": s["n_missing_decodes"],
                    "completion_bit_accuracy_mean": f"{s['completion_bit_accuracy_mean']:.6f}",
                    "completion_exact_match_rate": f"{s['completion_exact_match_rate']:.6f}",
                    "problem_level_n_problems": pl.get("n_problems"),
                    "problem_level_majority_vote_bit_accuracy_mean": (
                        f"{pl.get('majority_vote_bit_accuracy_mean'):.6f}"
                        if "majority_vote_bit_accuracy_mean" in pl else ""
                    ),
                    "problem_level_majority_vote_exact_match_rate": (
                        f"{pl.get('majority_vote_exact_match_rate'):.6f}"
                        if "majority_vote_exact_match_rate" in pl else ""
                    ),
                })
        print("\nWrote", args.out_csv)


if __name__ == "__main__":
    main()
