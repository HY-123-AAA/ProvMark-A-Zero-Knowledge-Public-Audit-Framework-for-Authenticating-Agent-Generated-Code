import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

ID_KEYS = [
    "task_idx", "task_id", "doc_id", "problem_id", "prompt_id",
    "id", "idx", "index"
]

Z_KEYS = ["z_score", "z", "old_z_score", "zscore", "wm_z"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


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


def get_z_threshold(task_root: Dict[str, Any], override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    wd = task_root.get("watermark_detection", {})
    zt = wd.get("z_threshold")
    if isinstance(zt, (int, float)):
        return float(zt)
    return 4.0


def get_problem_id(rec: Dict[str, Any]) -> Tuple[str, Any]:
    for k in ID_KEYS:
        if k in rec:
            return k, rec.get(k)
    # 兜底：很多实现会有 "doc" 或 "example_id"
    for k in ["doc", "example_id", "qid", "question_id"]:
        if k in rec:
            return k, rec.get(k)
    return "unknown_id", None


def get_sample_idx(rec: Dict[str, Any]) -> int:
    v = rec.get("sample_idx")
    if isinstance(v, int):
        return v
    # 有些记录里 sample_idx 可能是字符串
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return 0


def make_key(pid_val: Any, sample_idx: int) -> str:
    return f"{str(pid_val)}::{sample_idx}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True, help="path to evaluation_results.json")
    ap.add_argument("--task", default=None, help="task key, e.g. humaneval/mbpp")
    ap.add_argument("--mode", choices=["prediction", "zscore"], default="prediction",
                    help="prediction: keep prediction==true; zscore: keep z>=threshold")
    ap.add_argument("--z_threshold", type=float, default=None,
                    help="override z threshold (only used in zscore mode); default read from file or 4.0")
    ap.add_argument("--out_json", required=True, help="output path, e.g. outputs/selected_ids.json")
    ap.add_argument("--save_debug_fields", action="store_true",
                    help="if set, save z/prediction alongside ids for debugging")
    args = ap.parse_args()

    data = load_json(args.eval_json)
    task_name, task_root = pick_task_root(data, args.task)
    recs = find_records(task_root)
    zt = get_z_threshold(task_root, args.z_threshold)

    selected = []
    for r in recs:
        if args.mode == "prediction":
            if get_prediction(r) is True:
                selected.append(r)
        else:
            z = get_z(r)
            if z is not None and z >= zt:
                selected.append(r)

    out_items = []
    for r in selected:
        pid_key, pid_val = get_problem_id(r)
        sidx = get_sample_idx(r)
        item = {
            "problem_id_key": pid_key,
            "problem_id": pid_val,
            "sample_idx": sidx,
            "key": make_key(pid_val, sidx),
        }
        if args.save_debug_fields:
            item["prediction"] = get_prediction(r)
            item["z_score"] = get_z(r)
        out_items.append(item)

    out_obj = {
        "task": task_name,
        "mode": args.mode,
        "z_threshold": zt if args.mode == "zscore" else None,
        "total_samples": len(recs),
        "selected_samples": len(out_items),
        "selected_rate": (len(out_items) / len(recs)) if len(recs) else 0.0,
        "selected": out_items,
    }

    ensure_parent_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("task", task_name)
    print("total_samples", len(recs))
    print("mode", args.mode)
    if args.mode == "zscore":
        print("z_threshold", zt)
    print("selected_samples", len(out_items))
    print("saved_to", args.out_json)


if __name__ == "__main__":
    main()
