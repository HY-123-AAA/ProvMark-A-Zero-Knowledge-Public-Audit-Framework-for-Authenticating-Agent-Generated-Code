#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
from statistics import mean


def get_bits_len(sample):
    if "bits_len" in sample:
        return int(sample["bits_len"])
    if "m" in sample:
        return len(sample["m"])
    if "original_bits" in sample:
        return len(str(sample["original_bits"]))
    raise KeyError("Cannot infer bits_len from sample")


def get_num_layers(sample):
    if "L" in sample:
        return int(sample["L"])
    if "l" in sample:
        return int(sample["l"])
    if "g" in sample and len(sample["g"]) > 0:
        return len(sample["g"][0])
    if "b" in sample and len(sample["b"]) > 0:
        return len(sample["b"][0])
    raise KeyError("Cannot infer number of layers from sample")


def get_original_bits(sample):
    if "original_bits" in sample:
        return str(sample["original_bits"])
    if "m" in sample:
        return "".join(str(int(x)) for x in sample["m"])
    raise KeyError("Cannot infer original bits from sample")


def infer_tie_policy(data, user_choice):
    if user_choice != "auto":
        return user_choice

    # 你的两个导出链路 tie 处理不一样：
    # generate_zkp_data.py: 平票 -> '0'
    # zkp_extractor.py:    平票 -> 'x'
    #
    # 自动策略：
    # 1) 只要已有 decoded_bits 中出现过 x，就按 x
    # 2) 否则若样本里有 ber 字段，通常说明来自 generate_zkp_data.py，按 zero
    # 3) 否则默认按 x，更保守
    for s in data:
        if "decoded_bits" in s and "x" in str(s["decoded_bits"]):
            return "x"

    for s in data:
        if "ber" in s:
            return "zero"

    return "x"


def decode_with_first_k_active(sample, k_active, tie_policy="x"):
    h = sample["h"]
    g = sample["g"]
    b = sample["b"]
    bit_idx = sample["bit_idx"]

    bits_len = get_bits_len(sample)
    L = get_num_layers(sample)
    original_bits = get_original_bits(sample)

    active_positions = [t for t, v in enumerate(h) if int(v) == 1]
    if len(active_positions) < k_active:
        return None

    active_positions = active_positions[:k_active]

    vote_0 = [0] * bits_len
    vote_1 = [0] * bits_len

    # 严格复现 exporter 的逐层投票逻辑：
    # 对每个高熵位置 t、每一层 i，计算 m_hat = g[t][i] ^ b[t][i]
    # 然后累计到对应的 bit_idx[t]
    for t in active_positions:
        p = int(bit_idx[t])

        for i in range(L):
            m_hat = int(g[t][i]) ^ int(b[t][i])
            if m_hat == 1:
                vote_1[p] += 1
            else:
                vote_0[p] += 1

    decoded = []
    for p in range(bits_len):
        if vote_1[p] > vote_0[p]:
            decoded.append("1")
        elif vote_0[p] > vote_1[p]:
            decoded.append("0")
        else:
            if tie_policy == "zero":
                decoded.append("0")
            elif tie_policy == "x":
                decoded.append("x")
            else:
                raise ValueError(f"Unknown tie_policy: {tie_policy}")

    decoded = "".join(decoded)

    correct = sum(1 for a, bch in zip(decoded, original_bits) if a == bch)
    bit_acc = correct / bits_len
    exact = int(decoded == original_bits)

    return {
        "bit_acc": bit_acc,
        "exact": exact,
        "decoded": decoded,
        "original": original_bits,
        "vote_0": vote_0,
        "vote_1": vote_1,
        "active_used": k_active,
        "active_total": len([t for t, v in enumerate(h) if int(v) == 1]),
    }


def replay_full_active(sample, tie_policy="x"):
    active_total = sum(int(v) == 1 for v in sample["h"])
    if active_total == 0:
        return None
    return decode_with_first_k_active(sample, active_total, tie_policy=tie_policy)


def main():
    parser = argparse.ArgumentParser(
        description="Compute bit recovery from witness using the first K active high-entropy positions, strictly replaying exporter logic."
    )
    parser.add_argument(
        "--witness_path",
        type=str,
        required=True,
        help="Path to witness json"
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="50,100,200,300",
        help="Comma-separated K values over active positions"
    )
    parser.add_argument(
        "--tie_policy",
        type=str,
        default="auto",
        choices=["auto", "zero", "x"],
        help="Tie policy for bit decoding"
    )
    parser.add_argument(
        "--check_full_replay",
        action="store_true",
        help="Replay using all active positions and compare with stored decoded_bits if present"
    )

    args = parser.parse_args()

    with open(args.witness_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Empty witness file")

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    tie_policy = infer_tie_policy(data, args.tie_policy)
    bits_len = get_bits_len(data[0])

    if args.check_full_replay:
        total = 0
        matched = 0
        mismatched_examples = []

        for idx, sample in enumerate(data):
            rep = replay_full_active(sample, tie_policy=tie_policy)
            if rep is None:
                continue

            total += 1
            stored = sample.get("decoded_bits", None)
            if stored is None:
                continue

            if str(stored) == rep["decoded"]:
                matched += 1
            else:
                if len(mismatched_examples) < 5:
                    mismatched_examples.append({
                        "sample_idx": idx,
                        "stored": str(stored),
                        "replayed": rep["decoded"],
                        "original": rep["original"],
                    })

        print(f"[full replay check] tie_policy = {tie_policy}")
        print(f"[full replay check] checked samples = {total}")
        print(f"[full replay check] matched stored decoded_bits = {matched}")

        if mismatched_examples:
            print("[full replay check] first mismatches:")
            for ex in mismatched_examples:
                print(ex)

    print(f"| Bits | {' | '.join(str(k) for k in k_list)} |")
    print("|" + "---|" * (len(k_list) + 1))

    cells = []
    for K in k_list:
        accs = []
        exacts = []

        for sample in data:
            out = decode_with_first_k_active(sample, K, tie_policy=tie_policy)
            if out is None:
                continue
            accs.append(out["bit_acc"])
            exacts.append(out["exact"])

        if not accs:
            cells.append("N/A")
        else:
            cells.append(f"{mean(accs):.4f} (n={len(accs)})")

    print(f"| {bits_len} | " + " | ".join(cells) + " |")
    print(f"\n[tie_policy] {tie_policy}")


if __name__ == "__main__":
    main()