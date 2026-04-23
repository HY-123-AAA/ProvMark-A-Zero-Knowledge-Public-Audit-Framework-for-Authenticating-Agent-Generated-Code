import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

# 可能的 id 字段（按优先顺序尝试）
ID_KEYS = [
    "task_idx", "task_id", "doc_id", "problem_id", "prompt_id",
    "id", "idx", "index"
]
# sample 索引字段
SAMPLE_KEYS = ["sample_idx", "sample_id", "gen_idx", "generation_idx"]

# z-score 字段
Z_KEYS = ["z_score", "z", "old_z_score", "zscore", "wm_z"]

# decoded_bits 字段
DECODE_KEYS = ["decoded_bits", "decoded", "decoded_message", "message", "bits"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_task_root(data: Dict[str, Any], task: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if task is not None:
        if task not in data:
            raise KeyError(f"task '{task}' not found at top-level keys: {list(data.keys())[:20]}")
        return task, data[task]

    # 自动推断
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
    raise ValueError("Cannot find watermark_detection/raw_detection_results list.")


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
    # 只保留 0/1/x；其他不确定符号统一当 x
    out = []
    for ch in s:
        if ch in "01":
            out.append(ch)
        elif ch in "xX?*":
            out.append("x")
    return "".join(out)


def extract_decoded_bits(rec: Dict[str, Any]) -> Optional[str]:
    # 常见嵌套位置优先
    for container_key in ["decode_result", "decode", "decoded", "watermark", "wm", "meta", "confidences"]:
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


def make_sample_key(rec: Dict[str, Any]) -> str:
    """
    生成跨文件可对齐的唯一 key（尽量用 task_idx / doc_id + sample_idx）
    """
    rid = None
    for k in ID_KEYS:
        if k in rec:
            rid = rec.get(k)
            break

    sid = None
    for k in SAMPLE_KEYS:
        if k in rec:
            sid = rec.get(k)
            break

    # 兜底：如果缺 sample_idx，就用 rid；如果连 rid 都没有就用一个 JSON 字符串 hash 的简化形式
    if rid is None and sid is None:
        # 尽量稳定：挑一些可能存在的字段
        cand = {
            "prompt": rec.get("prompt"),
            "completion": rec.get("completion"),
            "text": rec.get("text"),
        }
        return "fallback:" + str(hash(json.dumps(cand, sort_keys=True, ensure_ascii=False)))
    if sid is None:
        return f"{rid}"
    return f"{rid}::{sid}"


def bimark_bit_accuracy(decoded: str, bits: str) -> Tuple[float, bool]:
    """
    BiMark 口径：
    - 分母固定 L=len(bits)
    - x 计错
    - exact_match 要求 decoded == bits（因此 x 直接失败）
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
    exact = (decoded == bits)
    return bit_acc, exact


def build_index(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for r in recs:
        k = make_sample_key(r)
        idx[k] = r
    return idx


def select_keys(clean_recs: List[Dict[str, Any]], mode: str, z_thres: float) -> List[str]:
    keys = []
    for r in clean_recs:
        ok = False
        if mode == "prediction":
            ok = (get_prediction(r) is True)
        else:
            z = get_z(r)
            ok = (z is not None and z >= z_thres)

        if ok:
            keys.append(make_sample_key(r))
    return keys


def eval_on_keyset(idx: Dict[str, Dict[str, Any]],
                   keys: List[str],
                   bits: str,
                   missing_as_fail: bool = True) -> Dict[str, Any]:
    used = 0
    miss = 0
    sum_bit = 0.0
    exact_cnt = 0
    pred_true_cnt = 0  # 统计这批 key 在该文件里 prediction=true 的数量

    for k in keys:
        r = idx.get(k)
        if r is None:
            miss += 1
            if missing_as_fail:
                used += 1
                # 视为全错
                sum_bit += 0.0
                exact_cnt += 0
            continue

        pred = get_prediction(r)
        if pred is True:
            pred_true_cnt += 1

        db = extract_decoded_bits(r)
        if db is None:
            if missing_as_fail:
                used += 1
                sum_bit += 0.0
                exact_cnt += 0
            else:
                miss += 1
            continue

        used += 1
        bit_acc, exact = bimark_bit_accuracy(db, bits)
        sum_bit += bit_acc
        exact_cnt += int(exact)

    def safe_div(a, b):
        return a / b if b else 0.0

    return {
        "keyset_size": len(keys),
        "used": used,
        "missing": miss,
        "bit_accuracy": safe_div(sum_bit, used),
        "exact_match": safe_div(exact_cnt, used),
        "pred_true_in_keyset": pred_true_cnt,
        "pred_true_rate_in_keyset": safe_div(pred_true_cnt, len(keys)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_eval", required=True, help="clean evaluation_results.json")
    ap.add_argument("--attack_eval", required=True, help="attacked evaluation_results.json (e.g., rename)")
    ap.add_argument("--task", default=None, help="task key, e.g. humaneval/mbpp")
    ap.add_argument("--bits", required=True, help="original bits, e.g. 01010101")
    ap.add_argument("--select_mode", choices=["prediction", "zscore"], default="zscore",
                    help="how to build fixed set S from clean")
    ap.add_argument("--z_threshold", type=float, default=4.0, help="used when select_mode=zscore")
    ap.add_argument("--save_keys", default=None, help="optional: save selected keys to json")
    ap.add_argument("--load_keys", default=None, help="optional: load selected keys from json (skip selecting)")
    ap.add_argument("--missing_as_fail", action="store_true",
                    help="if a key is missing in attack file, count it as failure (recommended)")
    args = ap.parse_args()

    clean = load_json(args.clean_eval)
    atk = load_json(args.attack_eval)

    t1, root1 = pick_task_root(clean, args.task)
    t2, root2 = pick_task_root(atk, args.task)
    if t1 != t2:
        print(f"[warn] task mismatch: clean={t1}, attack={t2} (still proceeding)")

    clean_recs = find_records(root1)
    atk_recs = find_records(root2)

    clean_idx = build_index(clean_recs)
    atk_idx = build_index(atk_recs)

    # fixed keyset S
    if args.load_keys:
        keys = load_json(args.load_keys)
        if isinstance(keys, dict) and "keys" in keys:
            keys = keys["keys"]
        if not isinstance(keys, list):
            raise ValueError("--load_keys must be a json list or {'keys': [...]} ")
    else:
        keys = select_keys(clean_recs, args.select_mode, args.z_threshold)

    if args.save_keys:
        with open(args.save_keys, "w", encoding="utf-8") as f:
            json.dump({"keys": keys}, f, ensure_ascii=False, indent=2)

    # eval
    clean_stats = eval_on_keyset(clean_idx, keys, args.bits, missing_as_fail=args.missing_as_fail)
    atk_stats = eval_on_keyset(atk_idx, keys, args.bits, missing_as_fail=args.missing_as_fail)

    print("task", args.task if args.task else t1)
    print("bits", args.bits)
    print("select_mode", args.select_mode, "z_threshold", args.z_threshold if args.select_mode == "zscore" else "-")
    print("fixed_set_size", len(keys))

    print("\n[CLEAN on fixed set]")
    for k, v in clean_stats.items():
        print(k, v)

    print("\n[ATTACK on fixed set]")
    for k, v in atk_stats.items():
        print(k, v)

    print("\n[DELTA attack - clean]")
    print("bit_accuracy_drop", atk_stats["bit_accuracy"] - clean_stats["bit_accuracy"])
    print("exact_match_drop", atk_stats["exact_match"] - clean_stats["exact_match"])
    print("pred_true_rate_drop", atk_stats["pred_true_rate_in_keyset"] - clean_stats["pred_true_rate_in_keyset"])


if __name__ == "__main__":
    main()
