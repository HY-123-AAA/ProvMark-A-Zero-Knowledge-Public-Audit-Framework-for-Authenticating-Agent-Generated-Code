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
            raise KeyError(f"task '{task}' not found. top keys: {list(data.keys())[:30]}")
        return task, data[task]

    if len(data) == 1 and isinstance(next(iter(data.values())), dict):
        k = next(iter(data.keys()))
        return k, data[k]

    for k in ["humaneval", "mbpp", "ds1000", "ds1000-all-completion"]:
        if k in data and isinstance(data[k], dict):
            return k, data[k]

    raise ValueError("Cannot infer task key; please pass --task.")


def find_records(task_root: Dict[str, Any]) -> List[Dict[str, Any]]:
    wd = task_root.get("watermark_detection", {})
    recs = wd.get("raw_detection_results")
    if isinstance(recs, list) and (len(recs) == 0 or isinstance(recs[0], dict)):
        return recs
    raise ValueError("Cannot find watermark_detection/raw_detection_results list in this file.")


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


def get_problem_id(rec: Dict[str, Any]) -> Any:
    for k in ID_KEYS:
        if k in rec:
            return rec.get(k)
    for k in ["doc", "example_id", "qid", "question_id"]:
        if k in rec:
            return rec.get(k)
    return None


def get_sample_idx(rec: Dict[str, Any]) -> int:
    v = rec.get("sample_idx")
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return 0


def make_key(pid_val: Any, sample_idx: int) -> str:
    return f"{str(pid_val)}::{sample_idx}"


def normalize_decoded(s: str) -> str:
    # 保留 0/1/x；其它不确定符号也映射成 x
    out = []
    for ch in s:
        if ch in "01":
            out.append(ch)
        elif ch in "xX?*":
            out.append("x")
    return "".join(out)


def extract_decoded_bits(rec: Dict[str, Any]) -> Optional[str]:
    # 常见嵌套位置
    for container_key in ["decode_result", "confidences", "watermark", "wm", "decode", "decoded", "meta"]:
        v = rec.get(container_key)
        if isinstance(v, dict):
            for k in DECODE_KEYS:
                if k in v and isinstance(v[k], str):
                    s = normalize_decoded(v[k])
                    return s if s else None

    # 当前层
    for k in DECODE_KEYS:
        v = rec.get(k)
        if isinstance(v, str):
            s = normalize_decoded(v)
            return s if s else None

    return None


def bimark_bit_accuracy(decoded: str, bits: str) -> Tuple[float, bool]:
    """
    BiMark 口径：
    - 分母固定 L=len(bits)
    - 'x' 计错
    - exact_match: decoded 严格等于 bits（含长度一致）
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

    bit_acc = correct / L if L else 0.0
    exact = (decoded == bits)
    return bit_acc, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True, help="attacked or target evaluation_results.json")
    ap.add_argument("--selected_json", required=True, help="json output from export_selected_samples.py")
    ap.add_argument("--task", default=None, help="task key override (optional)")
    ap.add_argument("--bits", required=True, help="original bits, e.g. 01010101")
    ap.add_argument("--report_detection_in_subset", action="store_true",
                    help="also report how many in subset remain prediction=true / z>=threshold(if available)")
    args = ap.parse_args()

    sel = load_json(args.selected_json)
    selected_list = sel.get("selected", [])
    sel_keys = set()
    for it in selected_list:
        k = it.get("key")
        if isinstance(k, str):
            sel_keys.add(k)

    data = load_json(args.eval_json)
    task_name, task_root = pick_task_root(data, args.task or sel.get("task"))
    recs = find_records(task_root)

    # 建索引：key -> record
    idx = {}
    for r in recs:
        pid = get_problem_id(r)
        sidx = get_sample_idx(r)
        k = make_key(pid, sidx)
        idx[k] = r

    total_sel = len(sel_keys)
    found = 0
    missing = 0

    sum_bit_acc = 0.0
    exact_cnt = 0
    missing_decode = 0

    # 额外：subset 内检测情况
    pred_true_cnt = 0
    z_ge_cnt = 0
    zt = sel.get("z_threshold", None)

    for k in sel_keys:
        r = idx.get(k)
        if r is None:
            missing += 1
            # 缺记录：按失败计（可选；这里不计入均值，防止误伤）
            continue

        found += 1

        if args.report_detection_in_subset:
            if get_prediction(r) is True:
                pred_true_cnt += 1
            if isinstance(zt, (int, float)):
                z = get_z(r)
                if z is not None and z >= float(zt):
                    z_ge_cnt += 1

        db = extract_decoded_bits(r)
        if db is None:
            missing_decode += 1
            # 按失败计：bit_acc=0, exact=0
            continue

        bit_acc, exact = bimark_bit_accuracy(db, args.bits)
        sum_bit_acc += bit_acc
        exact_cnt += int(exact)

    def safe_div(a, b):
        return a / b if b else 0.0

    print("task", task_name)
    print("subset_size", total_sel)
    print("found_in_eval", found, "missing_in_eval", missing)
    print("missing_decode_in_found", missing_decode)

    # BiMark-style
    print("bimark_bit_accuracy_on_subset", safe_div(sum_bit_acc, found))
    print("bimark_exact_match_rate_on_subset", safe_div(exact_cnt, found))

    if args.report_detection_in_subset:
        print("subset_prediction_true_rate", safe_div(pred_true_cnt, found))
        if isinstance(zt, (int, float)):
            print("subset_z_ge_threshold_rate", safe_div(z_ge_cnt, found), "z_threshold", float(zt))


if __name__ == "__main__":
    main()
