# coding=utf-8
import argparse
import json
import os
from typing import Any, Dict, List, Set

def load_selected_ids(ids_json: str) -> Set[str]:
    obj = json.load(open(ids_json, "r", encoding="utf-8"))
    if isinstance(obj, dict) and "selected_ids" in obj:
        return set(map(str, obj["selected_ids"]))
    if isinstance(obj, list):
        return set(map(str, obj))
    raise ValueError("ids_json must be a list or a dict with key selected_ids")

def get_raw_list(eval_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 你这个项目就是这个路径
    return eval_obj["mbpp"]["watermark_detection"]["raw_detection_results"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_eval", required=True)
    ap.add_argument("--ids_json", required=True)
    ap.add_argument("--out_eval", required=True)

    ap.add_argument("--id_key", default="task_idx")
    ap.add_argument("--order_as_taskidx", action="store_true")
    ap.add_argument("--n_tasks", type=int, default=500)

    args = ap.parse_args()

    ids = load_selected_ids(args.ids_json)
    eval_obj = json.load(open(args.in_eval, "r", encoding="utf-8"))
    raw = get_raw_list(eval_obj)

    L = len(raw)
    per_task = None
    if args.order_as_taskidx:
        if args.n_tasks <= 0:
            raise ValueError("n_tasks must be positive")
        if L % args.n_tasks != 0:
            raise ValueError(f"len(raw)={L} not divisible by n_tasks={args.n_tasks}")
        per_task = L // args.n_tasks

    out_raw = []
    kept_tasks = set()

    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            continue

        if args.order_as_taskidx:
            tid = str(i // per_task)
        else:
            if args.id_key not in r:
                # 如果某条缺失就跳过
                continue
            tid = str(r[args.id_key])

        if tid in ids:
            out_raw.append(r)
            kept_tasks.add(tid)

    # 写回
    eval_obj["mbpp"]["watermark_detection"]["raw_detection_results"] = out_raw

    os.makedirs(os.path.dirname(args.out_eval), exist_ok=True)
    json.dump(eval_obj, open(args.out_eval, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"in_raw {L} out_raw {len(out_raw)}")
    print(f"kept_tasks {len(kept_tasks)}")
    print("write", args.out_eval)

if __name__ == "__main__":
    main()
