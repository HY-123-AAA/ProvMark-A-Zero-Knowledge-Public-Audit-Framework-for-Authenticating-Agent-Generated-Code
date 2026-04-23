import json
import os
from statistics import mean

K_LIST = [50, 100, 200, 300]

def decode_with_first_k_active(sample, K):
    h = sample["h"]
    g = sample["g"]
    b = sample["b"]
    bit_idx = sample["bit_idx"]
    bits_len = int(sample["bits_len"])
    L = int(sample["L"])
    original_bits = str(sample["original_bits"])

    active_positions = [t for t, v in enumerate(h) if v == 1]
    if len(active_positions) < K:
        return None

    active_positions = active_positions[:K]

    counts0 = [0] * bits_len
    counts1 = [0] * bits_len

    for t in active_positions:
        p = int(bit_idx[t])

        vote_for_0 = 0
        vote_for_1 = 0

        for i in range(L):
            c = int(b[t][i])
            in_V0 = (int(g[t][i]) == 1)

            if (c == 1 and (not in_V0)) or (c == 0 and in_V0):
                vote_for_1 += 1
            else:
                vote_for_0 += 1

        # 与 sweet_bimark.py 保持一致：平票归 0
        if vote_for_1 > vote_for_0:
            counts1[p] += 1
        else:
            counts0[p] += 1

    decoded = []
    for j in range(bits_len):
        if counts1[j] > counts0[j]:
            decoded.append('1')
        elif counts0[j] > counts1[j]:
            decoded.append('0')
        else:
            decoded.append('x')

    decoded = ''.join(decoded)

    correct = 0
    for a, bch in zip(decoded, original_bits):
        if a == bch:
            correct += 1
    bit_acc = correct / bits_len

    return {
        "decoded_bits": decoded,
        "original_bits": original_bits,
        "bit_acc": bit_acc,
        "exact_match": int(decoded == original_bits),
    }


def compute_one_witness_file(witness_path):
    with open(witness_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print(f"Empty witness file: {witness_path}")
        return None

    bits_len = int(data[0]["bits_len"])
    row = {"bits_len": bits_len}

    for K in K_LIST:
        acc_list = []
        exact_list = []

        for sample in data:
            out = decode_with_first_k_active(sample, K)
            if out is None:
                continue
            acc_list.append(out["bit_acc"])
            exact_list.append(out["exact_match"])

        if len(acc_list) == 0:
            row[K] = {"mean_acc": None, "n": 0, "exact": None}
        else:
            row[K] = {
                "mean_acc": mean(acc_list),
                "n": len(acc_list),
                "exact": mean(exact_list),
            }

    return row


def print_markdown_table(rows):
    print("| Bits | 50 | 100 | 200 | 300 |")
    print("|---|---|---|---|---|")
    for row in sorted(rows, key=lambda x: x["bits_len"]):
        vals = []
        for K in K_LIST:
            item = row[K]
            if item["n"] == 0:
                vals.append("N/A")
            else:
                vals.append(f'{item["mean_acc"]:.4f} (n={item["n"]})')
        print(f'| {row["bits_len"]} | ' + " | ".join(vals) + " |")


if __name__ == "__main__":
    witness_files = [
        "outputs/your_bits8_dir/zkp_witness_xxx.json",
        "outputs/your_bits16_dir/zkp_witness_xxx.json",
        "outputs/your_bits32_dir/zkp_witness_xxx.json",
    ]

    rows = []
    for fp in witness_files:
        if not os.path.exists(fp):
            print(f"Not found: {fp}")
            continue
        row = compute_one_witness_file(fp)
        if row is not None:
            rows.append(row)

    print_markdown_table(rows)