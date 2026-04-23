#!/usr/bin/env python3
"""
ZKP数据验证脚本

验证生成的ZKP输入文件是否满足所有约束条件。
"""

import json
import sys
from pathlib import Path


def validate_zkp_data(zkp_file):
    """验证ZKP数据文件"""

    print(f"📂 Loading: {zkp_file}")

    try:
        with open(zkp_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return False

    if not isinstance(data, list):
        print(f"❌ Data should be a list, got {type(data)}")
        return False

    print(f"✓ Found {len(data)} samples\n")

    all_valid = True

    for idx, sample in enumerate(data):
        print(f"{'=' * 60}")
        print(f"Sample {idx} (task={sample.get('task_idx')}, sample={sample.get('sample_idx')})")
        print(f"{'=' * 60}")

        valid = validate_sample(sample, idx)
        all_valid = all_valid and valid

        if valid:
            print(f"✅ Sample {idx}: VALID\n")
        else:
            print(f"❌ Sample {idx}: INVALID\n")

    print(f"\n{'=' * 60}")
    if all_valid:
        print(f"✅✅✅ ALL SAMPLES VALID ✅✅✅")
    else:
        print(f"❌❌❌ SOME SAMPLES INVALID ❌❌❌")
    print(f"{'=' * 60}\n")

    return all_valid


def validate_sample(sample, idx):
    """验证单个样本"""

    required_fields = [
        'model_id_u64', 'tau_times_1000', 'gamma_times_1000', 'e_max',
        'n', 'L', 'bits_len', 'm', 'h', 'g', 'r', 'b', 'bit_idx'
    ]

    # 1. 检查必需字段
    missing = [f for f in required_fields if f not in sample]
    if missing:
        print(f"  ❌ Missing fields: {missing}")
        return False
    print(f"  ✓ All required fields present")

    n = sample['n']
    L = sample['L']
    bits_len = sample['bits_len']

    # 2. 检查尺寸
    checks = [
        (len(sample['m']), bits_len, 'm'),
        (len(sample['h']), n, 'h'),
        (len(sample['g']), n, 'g'),
        (len(sample['r']), n, 'r'),
        (len(sample['b']), n, 'b'),
        (len(sample['bit_idx']), n, 'bit_idx'),
    ]

    for actual, expected, name in checks:
        if actual != expected:
            print(f"  ❌ {name} length mismatch: expected {expected}, got {actual}")
            return False
    print(f"  ✓ Array lengths correct (n={n}, L={L}, bits_len={bits_len})")

    # 3. 检查矩阵尺寸
    for t in range(n):
        if len(sample['g'][t]) != L:
            print(f"  ❌ g[{t}] length mismatch: expected {L}, got {len(sample['g'][t])}")
            return False
        if len(sample['r'][t]) != L:
            print(f"  ❌ r[{t}] length mismatch: expected {L}, got {len(sample['r'][t])}")
            return False
        if len(sample['b'][t]) != L:
            print(f"  ❌ b[{t}] length mismatch: expected {L}, got {len(sample['b'][t])}")
            return False
    print(f"  ✓ Matrix dimensions correct (n×L)")

    # 4. 检查布尔约束
    for t in range(n):
        # h[t] 应该是 0 或 1
        if sample['h'][t] not in [0, 1]:
            print(f"  ❌ h[{t}] = {sample['h'][t]} (should be 0 or 1)")
            return False

        for i in range(L):
            # g[t][i] 和 r[t][i] 应该是 0 或 1
            if sample['g'][t][i] not in [0, 1]:
                print(f"  ❌ g[{t}][{i}] = {sample['g'][t][i]} (should be 0 or 1)")
                return False
            if sample['r'][t][i] not in [0, 1]:
                print(f"  ❌ r[{t}][{i}] = {sample['r'][t][i]} (should be 0 or 1)")
                return False
            if sample['b'][t][i] not in [0, 1]:
                print(f"  ❌ b[{t}][{i}] = {sample['b'][t][i]} (should be 0 or 1)")
                return False
    print(f"  ✓ All values are binary (0 or 1)")

    # 5. 检查一致性约束：g[t][i] + r[t][i] = h[t]
    for t in range(n):
        h_t = sample['h'][t]
        for i in range(L):
            g_ti = sample['g'][t][i]
            r_ti = sample['r'][t][i]
            if g_ti + r_ti != h_t:
                print(f"  ❌ Consistency violation at t={t}, i={i}:")
                print(f"     g[{t}][{i}] + r[{t}][{i}] = {g_ti} + {r_ti} = {g_ti + r_ti}")
                print(f"     but h[{t}] = {h_t}")
                return False
    print(f"  ✓ Consistency constraint satisfied: g[t][i] + r[t][i] = h[t]")

    # 6. 检查bit_idx范围
    for t in range(n):
        if sample['h'][t] == 1:
            if not (0 <= sample['bit_idx'][t] < bits_len):
                print(f"  ❌ bit_idx[{t}] = {sample['bit_idx'][t]} out of range [0, {bits_len})")
                return False
    print(f"  ✓ bit_idx values in valid range [0, {bits_len})")

    # 7. 检查载荷m
    for j in range(bits_len):
        if sample['m'][j] not in [0, 1]:
            print(f"  ❌ m[{j}] = {sample['m'][j]} (should be 0 or 1)")
            return False
    print(f"  ✓ Payload m is binary")

    # 8. 统计信息
    active_count = sum(sample['h'])
    print(f"\n  📊 Statistics:")
    print(f"     - Total tokens: {n}")
    print(f"     - Active (h=1): {active_count} ({100 * active_count / n:.1f}%)")
    print(f"     - Layers: {L}")
    print(f"     - Payload length: {bits_len}")

    if 'original_bits' in sample and 'decoded_bits' in sample:
        print(f"     - Original:  {sample['original_bits']}")
        print(f"     - Decoded:   {sample['decoded_bits']}")
        print(f"     - Match: {sample['original_bits'] == sample['decoded_bits']}")
        print(f"     - Hamming distance: {sample.get('hamming_distance', '?')}")

    if 'z_score' in sample:
        print(f"     - Z-score: {sample['z_score']:.4f}")
    if 'p_value' in sample:
        print(f"     - P-value: {sample['p_value']:.6f}")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_zkp_data.py <zkp_file.json>")
        print("\nExample:")
        print("  python validate_zkp_data.py outputs/zkp_witness_humaneval.json")
        sys.exit(1)

    zkp_file = sys.argv[1]

    if not Path(zkp_file).exists():
        print(f"❌ File not found: {zkp_file}")
        sys.exit(1)

    valid = validate_zkp_data(zkp_file)

    if valid:
        print("\n🎉 Validation passed! You can use this file with your Rust ZKP code.")
        print("\nNext steps:")
        print(f"  cargo run --release -- setup {zkp_file} pk.bin vk.bin")
        print(f"  cargo run --release -- prove {zkp_file} pk.bin proof.bin")
        print(f"  cargo run --release -- verify {zkp_file} vk.bin proof.bin")
        sys.exit(0)
    else:
        print("\n❌ Validation failed. Please check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()