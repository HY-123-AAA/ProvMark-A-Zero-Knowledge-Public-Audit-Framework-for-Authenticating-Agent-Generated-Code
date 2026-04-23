import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

ID_KEYS = [
    "task_idx", "task_id", "doc_id", "problem_id", "prompt_id",
    "id", "idx", "index"
]

Z_KEYS = ["z_score", "z", "old_z_score", "zscore", "wm_z"]

DECODE_KEYS = ["decoded_bits", "decoded", "decoded_message", "message", "bits"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_task_root(data: Dict[str, Any], task: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if task is not None:
        if task not in data:
            raise KeyError(f"task '{task}' not found at top-level keys: {list(data.keys())[:20]}")
        return task, data[task]

    if len(data) == 1 and isinstance(next(iter(data.values())), dict):
        k = next(iter(data.keys()))
        return k, data[k]

    for k in ["humaneval", "mbpp", "ds1000", "ds1000-all-completion"]:
        if k in data and isinstance(data[k], dict):
            return k, data[k]

    raise ValueError("Cannot infer task key. Please pass --task.")


def find_records(task_root: Dict[str, Any]) -> List[Dict[str, Any]]:
    wd = task_root.get("watermark_detection", {})
    recs = wd.get("raw_detection_results")
    if isinstance(recs, list) and recs and isinstance(recs[0], dict):
        return recs
    raise ValueError("Cannot find watermark_detection/raw_detection_results list in this file.")


def get_z_threshold(task_root: Dict[str, Any], override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    wd = task_root.get("watermark_detection", {})
    zt = wd.get("z_threshold")
    if isinstance(zt, (int, float)):
        return float(zt)
    return 4.0


def get_prediction(rec: Dict[str, Any]) -> Optional[bool]:
    v = rec.get("prediction")
    if isinstance(v, bool):
        return v
    return None


def get_z(rec: Dict[str, Any]) -> Optional[float]:
    for k in Z_KEYS:
        v = rec.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def normalize_decoded(s: str) -> str:
    out = []
    for ch in s:
        if ch in "01":
            out.append(ch)
        elif ch in "xX?*":
            out.append("x")
    return "".join(out)


def extract_decoded_bits(rec: Dict[str, Any]) -> Optional[str]:
    for container_key in ["decode_result", "confidences", "watermark", "wm", "decode", "decoded", "meta"]:
        v = rec.get(container_key)
        if isinstance(v, dict):
            for k in DECODE_KEYS:
                if k in v and isinstance(v[k], str):
                    s = normalize_decoded(v[k])
                    return s if s else None

    for k in DECODE_KEYS:
        v = rec.get(k)
        if isinstance(v, str):
            s = normalize_decoded(v)
            return s if s else None

    return None


def bimark_bit_accuracy(decoded: str, bits: str) -> Tuple[float, bool]:
    """
    BiMark 口径
    分母固定 len(bits)
    x 当错
    """
    L = len(bits)
    if len(decoded) < L:
        decoded = decoded + ("x" * (L - len(decoded)))
    if len(decoded) > L:
        decoded = decoded[:L]

    correct = 0
    for a, b in zip(decoded, bits):
        if a in "01" and a == b:
            correct += 1

    bit_acc = correct / L if L > 0 else 0.0
    exact_match = (decoded == bits)
    return bit_acc, exact_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True, help="path to evaluation_results.json")
    ap.add_argument("--task", default=None, help="task key, e.g. humaneval/mbpp")
    ap.add_argument("--bits", required=True, help="original bits, e.g. 01010101")
    ap.add_argument("--z_threshold", type=float, default=None, help="override z threshold; else read from file")
    ap.add_argument("--filter_mode", choices=["prediction", "zscore"], default="prediction",
                    help="prediction uses rec['prediction']==true; zscore uses z>=z_threshold")
    args = ap.parse_args()

    data = load_json(args.eval_json)
    task_name, task_root = pick_task_root(data, args.task)
    recs = find_records(task_root)
    zt = get_z_threshold(task_root, args.z_threshold)

    total = len(recs)

    selected = []
    for r in recs:
        if args.filter_mode == "prediction":
            pred = get_prediction(r)
            if pred is True:
                selected.append(r)
        else:
            z = get_z(r)
            if z is not None and z >= zt:
                selected.append(r)

    used = 0
    missing_decode = 0
    sum_bit_acc = 0.0
    exact_cnt = 0

    for r in selected:
        db = extract_decoded_bits(r)
        if db is None:
            missing_decode += 1
            continue

        used += 1
        bit_acc, exact = bimark_bit_accuracy(db, args.bits)
        sum_bit_acc += bit_acc
        exact_cnt += int(exact)

    def safe_div(a, b):
        return a / b if b else 0.0

    print("task", task_name)
    print("total_samples", total)
    print("filter_mode", args.filter_mode)
    if args.filter_mode == "zscore":
        print("z_threshold", zt)

    print("selected_samples", len(selected), "selected_rate", safe_div(len(selected), total))
    print("selected_with_decode", used, "missing_decode_in_selected", missing_decode)

    # 这两个就是对齐 BiMark 论文的主指标
    print("bimark_bit_accuracy_selected", safe_div(sum_bit_acc, used))
    print("bimark_exact_match_rate_selected", safe_div(exact_cnt, used))


if __name__ == "__main__":
    main()
