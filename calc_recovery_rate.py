# calc_recovery_rate.py
import argparse, json, math
from collections import defaultdict

def pick_bits(rec):
    # 兼容两种存储方式
    orig = rec.get("original_bits")
    dec  = rec.get("decoded_bits")
    if orig is None or dec is None:
        dr = rec.get("decode_result", {}) or {}
        orig = orig or dr.get("original_bits")
        dec  = dec  or dr.get("decoded_bits")
    return orig, dec

def bit_stats(orig, dec):
    if orig is None or dec is None:
        return None
    # 对齐长度
    if len(dec) < len(orig):
        dec = dec + ("x" * (len(orig) - len(dec)))
    if len(dec) > len(orig):
        dec = dec[:len(orig)]

    total = len(orig)
    known = sum(1 for d in dec if d in "01")
    correct = sum(1 for o, d in zip(orig, dec) if d in "01" and d == o)
    exact = 1 if dec == orig else 0
    return total, known, correct, exact

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True, help="evaluation_results.json path")
    ap.add_argument("--by_bits", action="store_true", help="group by bit-length")
    args = ap.parse_args()

    data = json.load(open(args.eval_json, "r", encoding="utf-8"))
    rdr = (
        data.get("watermark_detection", {})
            .get("raw_detection_results", [])
    )
    if not rdr:
        raise SystemExit("raw_detection_results 为空，确认你传的是 evaluation_results.json")

    # 汇总容器
    # key 可以按 bit 长度分组，也可以所有样本一起算
    agg = defaultdict(lambda: {"bits_total":0, "bits_known":0, "bits_correct":0, "msg_exact":0, "n":0})

    for rec in rdr:
        orig, dec = pick_bits(rec)
        st = bit_stats(orig, dec)
        if st is None:
            continue
        total, known, correct, exact = st

        key = len(orig) if args.by_bits else "ALL"
        agg[key]["bits_total"] += total
        agg[key]["bits_known"] += known
        agg[key]["bits_correct"] += correct
        agg[key]["msg_exact"] += exact
        agg[key]["n"] += 1

    # 打印结果
    print(f"\nLoaded: {args.eval_json}")
    for key in sorted(agg.keys(), key=lambda x: (x != "ALL", x)):
        a = agg[key]
        n = a["n"]
        if n == 0:
            continue
        bit_rate = a["bits_correct"] / a["bits_total"] if a["bits_total"] else 0.0
        coverage = a["bits_known"] / a["bits_total"] if a["bits_total"] else 0.0
        msg_acc = a["msg_exact"] / n if n else 0.0

        title = f"{key}-bit" if key != "ALL" else "ALL"
        print(f"\n[{title}] samples={n}")
        print(f"bit_recovery_rate = {bit_rate*100:.2f}%")
        print(f"bit_coverage       = {coverage*100:.2f}%   (decoded 位不是 x 的比例)")
        print(f"message_exact_acc  = {msg_acc*100:.2f}%    (整串完全一致的比例)")

if __name__ == "__main__":
    main()
