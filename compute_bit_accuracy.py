# coding=utf-8
"""
Compute Bit Extraction Rate (Bit Decoding Accuracy) from evaluation_results.json

Typical usage (single file):
python compute_bit_accuracy.py \
  --eval_json outputs/attack_mbpp_rename25_seed0/evaluation_results.json \
  --bits "01010101"

Multiple files at once:
python compute_bit_accuracy.py \
  --eval_json \
    outputs/attack_mbpp_rename25_seed0/evaluation_results.json \
    outputs/attack_mbpp_rename50_seed0/evaluation_results.json \
    outputs/attack_mbpp_rename75_seed0/evaluation_results.json \
    outputs/attack_mbpp_rename100_seed0/evaluation_results.json \
  --bits "01010101"

Optional: only compute on "detected" subset (z >= threshold)
python compute_bit_accuracy.py \
  --eval_json outputs/attack_mbpp_rename25_seed0/evaluation_results.json \
  --bits "01010101" \
  --z_threshold 4.0 \
  --report_detected_subset
"""

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers: bit normalization
# -----------------------------
def to_bitstr(x: Any) -> str:
    """
    Normalize decoded/original bits into a compact string containing only '0','1','x'.

    Supports:
      - None -> ""
      - list[int]/list[str] -> "0101..."
      - str -> keep only 0/1/x characters (remove spaces, commas, etc.)
      - other types -> str(x) then filter
    """
    if x is None:
        return ""
    if isinstance(x, list):
        out = []
        for v in x:
            if v is None:
                continue
            s = str(v).strip()
            if s in ("0", "1", "x", "X"):
                out.append("x" if s in ("x", "X") else s)
            else:
                # if it's like 0/1 integers or strings
                try:
                    iv = int(float(s))
                    if iv in (0, 1):
                        out.append(str(iv))
                except Exception:
                    pass
        return "".join(out)

    s = str(x)
    s = s.replace("X", "x")
    # keep only 0/1/x
    return "".join(ch for ch in s if ch in ("0", "1", "x"))


def compute_bit_metrics(decoded_bits: str, original_bits: str, require_len_match: bool) -> Optional[Dict[str, Any]]:
    """
    Compute per-sample bit metrics.

    Rule:
      - 'x' in decoded is treated as incorrect (never equals '0'/'1')
      - If original_bits is empty -> return None (invalid sample)
      - If require_len_match and len(decoded)!=len(original) -> accuracy=0 and per_bit_correct all 0

    Returns dict with:
      bit_accuracy, exact_match, strict_exact_match, correct_bits, total_bits,
      per_bit_correct, length_match, decoded_len
    """
    gt = to_bitstr(original_bits)
    dec = to_bitstr(decoded_bits)

    if not gt:
        return None

    b = len(gt)
    length_match = (len(dec) == b)

    if require_len_match and not length_match:
        return {
            "bit_accuracy": 0.0,
            "exact_match": False,            # decoded may still share prefix but we treat mismatch as failure
            "strict_exact_match": False,     # strict requires length match and exact content match
            "correct_bits": 0,
            "total_bits": b,
            "per_bit_correct": [0] * b,
            "length_match": False,
            "decoded_len": len(dec),
            "decoded_bits_norm": dec,
            "original_bits_norm": gt,
        }

    per_bit_correct = []
    for i in range(b):
        if i < len(dec) and dec[i] == gt[i]:
            per_bit_correct.append(1)
        else:
            per_bit_correct.append(0)

    correct = sum(per_bit_correct)
    acc = 100.0 * correct / b if b > 0 else 0.0

    # exact_match: same bits string (may be different length, so only true if equal)
    exact_match = (dec == gt)
    strict_exact_match = (length_match and exact_match)

    return {
        "bit_accuracy": acc,
        "exact_match": exact_match,
        "strict_exact_match": strict_exact_match,
        "correct_bits": correct,
        "total_bits": b,
        "per_bit_correct": per_bit_correct,
        "length_match": length_match,
        "decoded_len": len(dec),
        "decoded_bits_norm": dec,
        "original_bits_norm": gt,
    }


# -----------------------------
# Helpers: JSON parsing
# -----------------------------
def find_raw_detection_results(data: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Try to find raw_detection_results in common SWEET/BiMark evaluation json structures.

    Supported patterns:
      - data["watermark_detection"]["raw_detection_results"]
      - data[task]["watermark_detection"]["raw_detection_results"]
      - data[task]["raw_detection_results"]

    Returns list or None.
    """
    if isinstance(data, dict):
        # pattern 1: direct nested under watermark_detection
        wd = data.get("watermark_detection")
        if isinstance(wd, dict) and isinstance(wd.get("raw_detection_results"), list):
            return wd["raw_detection_results"]

        # pattern 2 / 3: task-keyed
        for _, v in data.items():
            if isinstance(v, dict):
                # pattern 2: task -> watermark_detection -> raw_detection_results
                wd2 = v.get("watermark_detection")
                if isinstance(wd2, dict) and isinstance(wd2.get("raw_detection_results"), list):
                    return wd2["raw_detection_results"]

                # pattern 3: task -> raw_detection_results
                rdr = v.get("raw_detection_results")
                if isinstance(rdr, list):
                    return rdr

    return None


def extract_decoded_bits_one(result: Dict[str, Any]) -> str:
    """
    Extract decoded bits from one raw result record.
    Try multiple keys for compatibility.
    """
    decode_result = result.get("decode_result")
    candidates = []
    if isinstance(decode_result, dict):
        candidates.extend([
            decode_result.get("decoded_bits"),
            decode_result.get("decoded"),
            decode_result.get("decoded_new"),
            decode_result.get("decoded_message"),
        ])
    candidates.extend([
        result.get("decoded_bits"),
        result.get("decoded"),
        result.get("decoded_new"),
        result.get("decoded_message"),
    ])

    for c in candidates:
        s = to_bitstr(c)
        if s:
            return s
    return ""


def extract_original_bits_one(result: Dict[str, Any], fallback_bits: Optional[str]) -> str:
    """
    Extract original bits (ground truth) from record or fallback argument.
    """
    if fallback_bits:
        return to_bitstr(fallback_bits)

    decode_result = result.get("decode_result")
    candidates = []
    if isinstance(decode_result, dict):
        candidates.extend([
            decode_result.get("original_bits"),
            decode_result.get("bits"),
            decode_result.get("message_bits"),
            decode_result.get("gt_bits"),
        ])
    candidates.extend([
        result.get("original_bits"),
        result.get("bits"),
        result.get("message_bits"),
        result.get("gt_bits"),
    ])

    for c in candidates:
        s = to_bitstr(c)
        if s:
            return s
    return ""


def extract_z_score_one(result: Dict[str, Any]) -> float:
    z = result.get("z_score", None)
    if z is None and isinstance(result.get("watermark_detection"), dict):
        z = result["watermark_detection"].get("z_score", None)
    try:
        return float(z) if z is not None else 0.0
    except Exception:
        return 0.0


def extract_num_tokens_generated_one(
    result: Dict[str, Any],
    max_generation_cap: Optional[int] = None,
) -> Optional[int]:
    val = result.get("num_tokens_generated", None)
    if val is None and isinstance(result.get("watermark_detection"), dict):
        val = result["watermark_detection"].get("num_tokens_generated", None)
    try:
        val = int(val) if val is not None else None
        if val is not None and max_generation_cap is not None:
            val = min(val, int(max_generation_cap))
        return val
    except Exception:
        return None

def extract_num_tokens_scored_one(result: Dict[str, Any]) -> Optional[int]:
    val = result.get("num_tokens_scored", None)
    if val is None and isinstance(result.get("watermark_detection"), dict):
        val = result["watermark_detection"].get("num_tokens_scored", None)
    try:
        return int(val) if val is not None else None
    except Exception:
        return None


def parse_bins(spec: str) -> List[int]:
    if spec is None:
        return []
    parts = [x.strip() for x in str(spec).split(",") if x.strip()]
    bins = sorted(set(int(x) for x in parts))
    return bins


def bucket_label(value: Optional[float], bins: List[int]) -> str:
    if value is None:
        return "NA"
    if not bins:
        return "ALL"
    v = float(value)
    if v < bins[0]:
        return f"< {bins[0]}"
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= v < hi:
            return f"[{lo}, {hi})"
    return f">= {bins[-1]}"


def summarize_bucket_records(records: List[Dict[str, Any]], z_threshold: Optional[float]) -> Dict[str, Any]:
    if not records:
        return {}

    n = len(records)
    total_correct = sum(r["correct_bits"] for r in records)
    total_bits = sum(r["total_bits"] for r in records)
    macro_bit_acc = sum(r["bit_accuracy"] for r in records) / n
    micro_bit_acc = (100.0 * total_correct / total_bits) if total_bits > 0 else 0.0
    exact_match_rate = 100.0 * sum(1 for r in records if r["exact_match"]) / n

    lengths = [r["num_tokens_generated"] for r in records if r.get("num_tokens_generated") is not None]
    scoreds = [r["num_tokens_scored"] for r in records if r.get("num_tokens_scored") is not None]
    votes = [r["votes_per_bit"] for r in records if r.get("votes_per_bit") is not None]

    out = {
        "n_samples": n,
        "avg_num_tokens_generated": (sum(lengths) / len(lengths)) if lengths else None,
        "avg_num_tokens_scored": (sum(scoreds) / len(scoreds)) if scoreds else None,
        "avg_votes_per_bit": (sum(votes) / len(votes)) if votes else None,
        "macro_bit_accuracy": macro_bit_acc,
        "micro_bit_accuracy": micro_bit_acc,
        "exact_match_rate": exact_match_rate,
    }

    if z_threshold is not None:
        detected = [r for r in records if r.get("z_score", 0.0) >= z_threshold]
        out["n_detected"] = len(detected)
        out["detected_exact_match_rate"] = (100.0 * sum(1 for r in detected if r["exact_match"]) / len(detected)) if detected else None
        out["detected_macro_bit_accuracy"] = (sum(r["bit_accuracy"] for r in detected) / len(detected)) if detected else None

    return out


def build_bucket_tables(
    records: List[Dict[str, Any]],
    length_bins: List[int],
    scored_bins: List[int],
    z_threshold: Optional[float],
) -> Dict[str, List[Dict[str, Any]]]:
    length_groups: Dict[str, List[Dict[str, Any]]] = {}
    scored_groups: Dict[str, List[Dict[str, Any]]] = {}

    for r in records:
        lb = bucket_label(r.get("num_tokens_generated"), length_bins)
        sb = bucket_label(r.get("num_tokens_scored"), scored_bins)
        length_groups.setdefault(lb, []).append(r)
        scored_groups.setdefault(sb, []).append(r)

    def order_key(label: str):
        if label == "NA":
            return (2, float("inf"))
        if label == "ALL":
            return (0, -1)
        nums = re.findall(r"\d+", label)
        return (1, int(nums[0])) if nums else (1, float("inf"))

    length_table = []
    for label in sorted(length_groups.keys(), key=order_key):
        row = {"bucket": label}
        row.update(summarize_bucket_records(length_groups[label], z_threshold=z_threshold))
        length_table.append(row)

    scored_table = []
    for label in sorted(scored_groups.keys(), key=order_key):
        row = {"bucket": label}
        row.update(summarize_bucket_records(scored_groups[label], z_threshold=z_threshold))
        scored_table.append(row)

    return {
        "by_length_bucket": length_table,
        "by_scored_bucket": scored_table,
    }


def export_bucket_table_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_bucket_table(title: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print(f"\n{title}:")
    header = (
        f"{'bucket':<12} {'n':>6} {'avg_len':>10} {'avg_eff':>10} {'eff/bit':>10} "
        f"{'macro_acc':>10} {'exact%':>10}"
    )
    if "n_detected" in rows[0]:
        header += f" {'n_det':>8} {'det_exact%':>12}"
    print(header)
    print("-" * len(header))

    def fmt(x, nd=2):
        if x is None:
            return "NA"
        return f"{x:.{nd}f}"

    for r in rows:
        line = (
            f"{r['bucket']:<12} {r.get('n_samples', 0):>6} "
            f"{fmt(r.get('avg_num_tokens_generated')):>10} "
            f"{fmt(r.get('avg_num_tokens_scored')):>10} "
            f"{fmt(r.get('avg_votes_per_bit')):>10} "
            f"{fmt(r.get('macro_bit_accuracy')):>10} "
            f"{fmt(r.get('exact_match_rate')):>10}"
        )
        if "n_detected" in r:
            line += f" {r.get('n_detected', 0):>8} {fmt(r.get('detected_exact_match_rate')):>12}"
        print(line)


def infer_label_from_path(path: str) -> str:
    """
    Infer attack label from file path, mainly for renameXX.
    """
    p = path.replace("\\", "/").lower()

    m = re.search(r"rename(\d+)", p)
    if m:
        return f"rename{m.group(1)}"

    if "refactor" in p:
        return "refactor"

    return os.path.basename(os.path.dirname(path)) or "eval"


# -----------------------------
# Aggregation
# -----------------------------
def summarize_records(records: List[Dict[str, Any]], z_threshold: Optional[float]) -> Dict[str, Any]:
    """
    Summarize a list of per-sample computed metrics.
    Outputs both macro-average and micro-average.

    macro avg bit accuracy:
      mean of per-sample accuracies (each sample weight 1)

    micro avg bit accuracy:
      total_correct_bits / total_bits (bit-weighted)

    exact_match_rate:
      fraction of samples with decoded==gt (as normalized)

    strict_exact_match_rate:
      fraction with (len match and decoded==gt)
    """
    if not records:
        return {
            "n_used": 0,
            "macro_bit_accuracy": None,
            "micro_bit_accuracy": None,
            "exact_match_rate": None,
            "strict_exact_match_rate": None,
            "avg_z": None,
            "n_detected": 0,
        }

    n = len(records)
    macro = sum(r["bit_accuracy"] for r in records) / n

    total_correct = sum(r["correct_bits"] for r in records)
    total_bits = sum(r["total_bits"] for r in records)
    micro = (100.0 * total_correct / total_bits) if total_bits > 0 else 0.0

    exact = 100.0 * sum(1 for r in records if r["exact_match"]) / n
    strict_exact = 100.0 * sum(1 for r in records if r["strict_exact_match"]) / n
    avg_z = sum(r.get("z_score", 0.0) for r in records) / n

    out = {
        "n_used": n,
        "macro_bit_accuracy": macro,
        "micro_bit_accuracy": micro,
        "exact_match_rate": exact,
        "strict_exact_match_rate": strict_exact,
        "avg_z": avg_z,
    }

    if z_threshold is not None:
        detected = [r for r in records if r.get("z_score", 0.0) >= z_threshold]
        out["n_detected"] = len(detected)
        if detected:
            out["detected_macro_bit_accuracy"] = sum(r["bit_accuracy"] for r in detected) / len(detected)
            tc = sum(r["correct_bits"] for r in detected)
            tb = sum(r["total_bits"] for r in detected)
            out["detected_micro_bit_accuracy"] = (100.0 * tc / tb) if tb > 0 else 0.0
            out["detected_exact_match_rate"] = 100.0 * sum(1 for r in detected if r["exact_match"]) / len(detected)
            out["detected_strict_exact_match_rate"] = 100.0 * sum(1 for r in detected if r["strict_exact_match"]) / len(detected)
        else:
            out["detected_macro_bit_accuracy"] = None
            out["detected_micro_bit_accuracy"] = None
            out["detected_exact_match_rate"] = None
            out["detected_strict_exact_match_rate"] = None

    return out


def process_eval_file(
    eval_json_path: str,
    fallback_bits: Optional[str],
    require_len_match: bool,
    z_threshold: Optional[float],
    keep_details: bool,
    length_bins: List[int],
    scored_bins: List[int],
    max_generation_cap: Optional[int],
) -> Dict[str, Any]:
    """
    Load one evaluation_results.json and compute average bit decoding accuracy over the whole file (one attack setting).
    """
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = find_raw_detection_results(data)
    if not raw:
        return {
            "eval_json": eval_json_path,
            "label": infer_label_from_path(eval_json_path),
            "error": "raw_detection_results not found",
        }

    # Counters
    n_total = len(raw)
    n_missing_decoded = 0
    n_missing_gt = 0
    n_invalid = 0
    n_len_mismatch = 0

    computed: List[Dict[str, Any]] = []

    for idx, rec in enumerate(raw):
        if not isinstance(rec, dict):
            n_invalid += 1
            continue

        decoded = extract_decoded_bits_one(rec)
        if not decoded:
            n_missing_decoded += 1
            continue

        gt = extract_original_bits_one(rec, fallback_bits)
        if not gt:
            n_missing_gt += 1
            continue

        metrics = compute_bit_metrics(decoded, gt, require_len_match=require_len_match)
        if metrics is None:
            n_missing_gt += 1
            continue

        z = extract_z_score_one(rec)
        metrics["z_score"] = z

        num_tokens_generated = extract_num_tokens_generated_one(
            rec,
            max_generation_cap=max_generation_cap,
        )
        num_tokens_scored = extract_num_tokens_scored_one(rec)
        metrics["num_tokens_generated"] = num_tokens_generated
        metrics["num_tokens_scored"] = num_tokens_scored
        metrics["votes_per_bit"] = (
            float(num_tokens_scored) / metrics["total_bits"]
            if num_tokens_scored is not None and metrics["total_bits"] > 0
            else None
        )

        # length mismatch bookkeeping
        if not metrics.get("length_match", True):
            n_len_mismatch += 1

        if keep_details:
            # Attach minimal identifiers if present
            for k in ("task_id", "problem_id", "problem_idx", "id", "sample_id", "idx"):
                if k in rec:
                    metrics[k] = rec[k]
            metrics["_record_index"] = idx

        computed.append(metrics)

    summary = summarize_records(computed, z_threshold=z_threshold)
    bucket_tables = build_bucket_tables(
        computed,
        length_bins=length_bins,
        scored_bins=scored_bins,
        z_threshold=z_threshold,
    )

    return {
        "eval_json": eval_json_path,
        "label": infer_label_from_path(eval_json_path),
        "n_total_records": n_total,
        "n_used": summary.get("n_used", 0),
        "n_missing_decoded": n_missing_decoded,
        "n_missing_gt": n_missing_gt,
        "n_invalid_record": n_invalid,
        "n_len_mismatch": n_len_mismatch,
        "summary": summary,
        "bucket_tables": bucket_tables,
        "details": computed if keep_details else None,
    }


def print_file_report(report: Dict[str, Any], z_threshold: Optional[float], require_len_match: bool):
    label = report.get("label", "eval")
    path = report.get("eval_json", "")
    err = report.get("error")
    print("\n" + "=" * 90)
    print(f"[{label}] {path}")
    print("=" * 90)

    if err:
        print(f"ERROR: {err}")
        return

    print(f"require_len_match = {require_len_match}")
    print(f"Total raw records:         {report.get('n_total_records', 0)}")
    print(f"Used for bit metrics:      {report.get('n_used', 0)}")
    print(f"Missing decoded_bits:      {report.get('n_missing_decoded', 0)}")
    print(f"Missing original_bits(GT): {report.get('n_missing_gt', 0)}")
    print(f"Invalid records:           {report.get('n_invalid_record', 0)}")
    print(f"Len mismatch count:        {report.get('n_len_mismatch', 0)}")

    s = report.get("summary", {})
    if not s or s.get("n_used", 0) == 0:
        print("No usable samples.")
        return

    print("\nOverall (this attack setting):")
    print(f"  Macro bit accuracy (per-sample mean): {s['macro_bit_accuracy']:.4f}%")
    print(f"  Micro bit accuracy (bit-weighted):    {s['micro_bit_accuracy']:.4f}%")
    print(f"  Exact match rate (decoded==gt):       {s['exact_match_rate']:.4f}%")
    print(f"  Strict exact match (len&match):      {s['strict_exact_match_rate']:.4f}%")
    print(f"  Avg z-score:                          {s['avg_z']:.6f}")

    if z_threshold is not None:
        print(f"\nDetected subset (z >= {z_threshold}):")
        print(f"  N detected:                           {s.get('n_detected', 0)}")
        if s.get("n_detected", 0) > 0 and s.get("detected_macro_bit_accuracy") is not None:
            print(f"  Macro bit accuracy:                   {s['detected_macro_bit_accuracy']:.4f}%")
            print(f"  Micro bit accuracy:                   {s['detected_micro_bit_accuracy']:.4f}%")
            print(f"  Exact match rate:                     {s['detected_exact_match_rate']:.4f}%")
            print(f"  Strict exact match rate:              {s['detected_strict_exact_match_rate']:.4f}%")
        else:
            print("  No detected samples or missing z-scores.")

    bucket_tables = report.get("bucket_tables", {})
    print_bucket_table("By generated length bucket", bucket_tables.get("by_length_bucket", []))
    print_bucket_table("By effective-position bucket", bucket_tables.get("by_scored_bucket", []))


def main():
    parser = argparse.ArgumentParser(description="Compute average bit extraction accuracy for each attack setting file.")
    parser.add_argument(
        "--eval_json",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to evaluation_results.json. You can pass multiple files for different rename ratios.",
    )
    parser.add_argument(
        "--bits",
        type=str,
        default=None,
        help="Fallback ground-truth bits if JSON does not contain original_bits. Example: 01010101",
    )
    parser.add_argument(
        "--require_len_match",
        action="store_true",
        help="If set, decoded length mismatch is treated as a full failure (accuracy=0 for that sample).",
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=None,
        help="Optional z-score threshold. If provided, script also reports metrics on detected subset (z>=threshold).",
    )
    parser.add_argument(
        "--keep_details",
        action="store_true",
        help="If set, keep per-sample details in output JSON (may be large).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save a JSON report.",
    )
    parser.add_argument(
        "--length_bins",
        type=str,
        default="128,256,384,512",
        help="Comma-separated generated-length bucket boundaries, e.g. 128,256,384,512",
    )
    parser.add_argument(
        "--scored_bins",
        type=str,
        default="16,32,48,64,96",
        help="Comma-separated effective-position bucket boundaries, e.g. 16,32,48,64,96",
    )
    parser.add_argument(
        "--export_bucket_csv_prefix",
        type=str,
        default=None,
        help="Optional prefix for exporting bucket tables as CSV. Example: outputs/len_cap_mbpp",
    )
    parser.add_argument(
        "--max_generation_cap",
        type=int,
        default=None,
        help="Optional hard cap for num_tokens_generated. For apps-* runs, set this to your max_length_generation / max_new_tokens, e.g. 2048.",
    )

    args = parser.parse_args()

    length_bins = parse_bins(args.length_bins)
    scored_bins = parse_bins(args.scored_bins)

    reports: List[Dict[str, Any]] = []
    for p in args.eval_json:
        rep = process_eval_file(
            eval_json_path=p,
            fallback_bits=args.bits,
            require_len_match=args.require_len_match,
            z_threshold=args.z_threshold,
            keep_details=args.keep_details,
            length_bins=length_bins,
            scored_bins=scored_bins,
            max_generation_cap=args.max_generation_cap,
        )
        reports.append(rep)
        print_file_report(rep, z_threshold=args.z_threshold, require_len_match=args.require_len_match)

        if args.export_bucket_csv_prefix and not rep.get("error"):
            label = rep.get("label", "eval")
            bucket_tables = rep.get("bucket_tables", {})
            length_csv = f"{args.export_bucket_csv_prefix}_{label}_length.csv"
            scored_csv = f"{args.export_bucket_csv_prefix}_{label}_effective.csv"
            export_bucket_table_csv(bucket_tables.get("by_length_bucket", []), length_csv)
            export_bucket_table_csv(bucket_tables.get("by_scored_bucket", []), scored_csv)
            print(f"Saved bucket CSVs to: {length_csv} and {scored_csv}")

    if args.output:
        out = {
            "require_len_match": args.require_len_match,
            "z_threshold": args.z_threshold,
            "length_bins": length_bins,
            "scored_bins": scored_bins,
            "reports": reports,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report JSON to: {args.output}")


if __name__ == "__main__":
    main()