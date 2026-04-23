# coding=utf-8
"""
zkp_extractor.py - 零知识证明数据提取模块

在 SweetBimark 检测过程中，提取零知识证明所需的所有数据并生成JSON文件。

使用方法：
1. 在 evaluator.py 中导入本模块
2. 在检测时调用 extract_zkp_data() 函数
3. 生成的JSON文件可直接用于零知识证明

数据格式（符合论文 v0_3_2）：
- h[t]: 高熵门控标记
- g[t][i]: token 是否在 V0 中（固定词表划分）
- r[t][i]: token 是否在 V1 中（固定词表划分）
- b[t][i]: 掩码 c_list（由 PRF 生成）
- bit_idx[t]: 消息比特索引
- m: 原始消息比特

【重要】关于 g/r 的定义：
本代码中 g[t][i] = 1 表示 token 在第 i 层的 V0 中（固定定义）
这与代码中"动态绿表"的概念不同。在电路中会根据 b（掩码）来处理。
"""

import json
import os
import torch
import numpy as np
import hashlib
from typing import List, Dict, Optional, Tuple


def prf(seed, secret_key: int) -> int:
    """伪随机函数，与 sweet_bimark.py 保持一致"""
    if isinstance(seed, torch.Tensor):
        if seed.dim() == 1:
            seed_str = ''.join(map(str, seed.tolist())) + str(secret_key)
        else:
            seed_str = ''.join(map(str, seed[0].tolist())) + str(secret_key)
    elif isinstance(seed, tuple):
        seed_str = ''.join(map(str, seed)) + str(secret_key)
    else:
        seed_str = str(seed) + str(secret_key)

    hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
    return int(hash_digest, 16) % 2 ** 32


class ZKPDataExtractor:
    """
    零知识证明数据提取器

    在检测过程中收集所有需要的数据，生成符合论文格式的JSON文件。
    """

    def __init__(
            self,
            vocab_size: int,
            gamma: float = 0.5,
            entropy_threshold: float = 0.5,
            partition_seeds: List[int] = None,
            c_key: int = 530773,
            bit_idx_key: int = 283519,
            window_size: int = 2,
            bits: str = '0',
            use_hist: bool = False,
            model_id: int = 0,
            tau_z: float = 2.0,
            e_max: int = 1,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.bits = bits
        self.c_key = c_key
        self.bit_idx_key = bit_idx_key
        self.use_hist = use_hist
        self.model_id = model_id
        self.tau_z = tau_z
        self.e_max = e_max

        if partition_seeds is None:
            partition_seeds = list(range(10))
        self.partition_seeds = partition_seeds
        self.num_partitions = len(partition_seeds)

        # 构建词表划分掩码（与 sweet_bimark.py 一致）
        self.partition_masks = []
        for key in partition_seeds:
            num_V0 = int(self.vocab_size * self.gamma)
            rng = np.random.default_rng(key)
            mask = np.zeros(self.vocab_size, dtype=bool)
            mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
            self.partition_masks.append(mask)

        print(f"[ZKPDataExtractor] Initialized:")
        print(f"  vocab_size={self.vocab_size}")
        print(f"  gamma={self.gamma}")
        print(f"  num_partitions={self.num_partitions}")
        print(f"  bits={self.bits} (len={len(self.bits)})")

    def extract(
            self,
            input_ids: torch.Tensor,
            prefix_len: int,
            entropy: List[float],
    ) -> Dict:
        """
        从检测数据中提取零知识证明所需的所有信息

        Args:
            input_ids: 完整的 token 序列（包含 prompt）
            prefix_len: prompt 的长度
            entropy: 每个位置的熵值列表

        Returns:
            包含所有 ZKP 数据的字典
        """
        n = len(input_ids) - prefix_len  # 生成的 token 数量
        l = self.num_partitions  # 层数
        bits_len = len(self.bits)  # 消息比特数

        # 初始化数据结构
        h = []  # 高熵门控标记
        g = []  # g[t][i]: token 是否在 V0 中
        r = []  # r[t][i]: token 是否在 V1 中
        b = []  # b[t][i]: 掩码
        bit_idx = []  # bit_idx[t]: 消息比特索引
        m = [int(bit) for bit in self.bits]  # 原始消息

        # 用于投票计数（验证用）
        vote_counts = [[0, 0] for _ in range(bits_len)]

        hist = set()

        for t_offset in range(n):
            t_abs = prefix_len + t_offset  # 绝对位置

            # 初始化该位置的数据
            h_t = 0
            g_t = [0] * l
            r_t = [0] * l
            b_t = [0] * l
            bit_idx_t = 0

            # 检查是否跳过
            skip = False

            # 1. 熵门控检查
            if t_abs < len(entropy) and entropy[t_abs] <= self.entropy_threshold:
                skip = True

            # 2. 窗口大小检查
            if t_abs < self.window_size:
                skip = True

            # 3. 获取前缀和 token
            if not skip:
                prefix_start = t_abs - self.window_size
                prefix = tuple(input_ids[prefix_start:t_abs].tolist())

                # 4. hist 去重（如果启用）
                if self.use_hist:
                    if prefix in hist:
                        skip = True
                    else:
                        hist.add(prefix)

            if skip:
                # 非触发位置：所有数据为 0
                h.append(0)
                g.append(g_t)
                r.append(r_t)
                b.append(b_t)
                bit_idx.append(0)
                continue

            # 触发位置
            h_t = 1
            curr_token = input_ids[t_abs].item()

            # 生成 c_list（掩码）和 bit_idx
            c_seed = prf(prefix, self.c_key)
            bit_idx_seed = prf(prefix, self.bit_idx_key)

            rng_c = np.random.default_rng(c_seed)
            c_list = rng_c.integers(0, 2, size=l)

            rng_bit_idx = np.random.default_rng(bit_idx_seed)
            bit_idx_t = int(rng_bit_idx.integers(0, bits_len))

            # 获取该位置对应的消息比特
            bit = int(self.bits[bit_idx_t])

            # 对每一层
            for i in range(l):
                in_V0 = bool(self.partition_masks[i][curr_token])
                c_i = int(c_list[i])

                # g[t][i] = 1 if token in V0
                # r[t][i] = 1 if token in V1
                g_t[i] = 1 if in_V0 else 0
                r_t[i] = 1 if not in_V0 else 0

                # b[t][i] = c_list[i]（掩码）
                b_t[i] = c_i

            # 记录数据
            h.append(h_t)
            g.append(g_t)
            r.append(r_t)
            b.append(b_t)
            bit_idx.append(bit_idx_t)

            # 投票计数（用于验证）
            # 代码中的解码逻辑：m̂ = g ⊕ c
            for i in range(l):
                m_hat_i = g_t[i] ^ b_t[i]
                if m_hat_i == 1:
                    vote_counts[bit_idx_t][1] += 1
                else:
                    vote_counts[bit_idx_t][0] += 1

        # 验证解码结果
        decoded_bits = ''
        for p in range(bits_len):
            if vote_counts[p][1] > vote_counts[p][0]:
                decoded_bits += '1'
            elif vote_counts[p][0] > vote_counts[p][1]:
                decoded_bits += '0'
            else:
                decoded_bits += 'x'

        # 计算统计量
        active_count = sum(h)

        # 位置级多数投票统计（论文方案）
        x_hit = 0
        half_l = l / 2
        for t_offset in range(n):
            if h[t_offset] == 1:
                # N_t^g = Σ g[t][i]
                n_t_g = sum(g[t_offset])
                n_t_r = sum(r[t_offset])

                # g_t = 1[N_t^g > l/2]
                g_t_majority = 1 if n_t_g > half_l else 0
                r_t_majority = 1 if n_t_g < half_l else 0

                # e_t = 0 if N_t^g > N_t^r else 1
                e_t = 0 if n_t_g > n_t_r else 1

                # x_t = (1-e_t)*g_t + e_t*(1-r_t)
                x_t = (1 - e_t) * g_t_majority + e_t * (1 - r_t_majority)
                x_hit += x_t

        # 构建输出
        zkp_data = {
            # 公共输入
            "model_id_u64": self.model_id,
            "tau_times_1000": int(self.tau_z * 1000),
            "gamma_times_1000": int(self.gamma * 1000),
            "e_max": self.e_max,

            # 尺寸参数
            "n": n,
            "l": l,
            "bits_len": bits_len,

            # 私有见证
            "m": m,
            "h": h,
            "g": g,
            "r": r,
            "b": b,
            "bit_idx": bit_idx,
        }

        # 附加验证信息（不会传入电路，仅用于调试）
        verification_info = {
            "active_count": active_count,
            "x_hit": x_hit,
            "decoded_bits": decoded_bits,
            "original_bits": self.bits,
            "decode_match": decoded_bits == self.bits,
            "vote_counts": vote_counts,
        }

        return zkp_data, verification_info

    def save_zkp_json(
            self,
            zkp_data: Dict,
            output_path: str,
            verification_info: Dict = None,
    ):
        """
        保存 ZKP 数据为 JSON 文件
        """
        with open(output_path, 'w') as f:
            json.dump(zkp_data, f, indent=2)

        print(f"[ZKPDataExtractor] Saved ZKP data to: {output_path}")

        if verification_info:
            print(f"  n={zkp_data['n']}, l={zkp_data['l']}, bits_len={zkp_data['bits_len']}")
            print(f"  active_count={verification_info['active_count']}")
            print(f"  x_hit={verification_info['x_hit']}")
            print(f"  decoded={verification_info['decoded_bits']}, original={verification_info['original_bits']}")
            print(f"  match={verification_info['decode_match']}")


def extract_zkp_from_detection(
        detector,  # SweetBimarkDetector instance
        input_ids: torch.Tensor,
        prefix_len: int,
        entropy: List[float],
        model_id: int = 0,
        tau_z: float = 2.0,
        e_max: int = 1,
        output_path: str = None,
) -> Tuple[Dict, Dict]:
    """
    从 SweetBimarkDetector 的检测中提取 ZKP 数据

    这是一个便捷函数，可直接在 evaluator.py 中调用
    """
    extractor = ZKPDataExtractor(
        vocab_size=detector.vocab_size,
        gamma=detector.gamma,
        entropy_threshold=detector.entropy_threshold,
        partition_seeds=detector.partition_seeds,
        c_key=detector.c_key,
        bit_idx_key=detector.bit_idx_key,
        window_size=detector.window_size,
        bits=detector.bits,
        use_hist=detector.use_hist,
        model_id=model_id,
        tau_z=tau_z,
        e_max=e_max,
    )

    zkp_data, verification_info = extractor.extract(input_ids, prefix_len, entropy)

    if output_path:
        extractor.save_zkp_json(zkp_data, output_path, verification_info)

    return zkp_data, verification_info


# ============================================================
# 以下是与论文方案对齐的 ZKP 电路验证函数（用于本地测试）
# ============================================================

def verify_zkp_constraints(zkp_data: Dict) -> Dict:
    """
    验证 ZKP 数据是否满足电路约束（本地测试用）

    检查：
    1. 一致性约束：h=1 时，g[t][i] + r[t][i] = 1
    2. z 统计量约束
    3. BER 约束
    """
    n = zkp_data['n']
    l = zkp_data['l']
    bits_len = zkp_data['bits_len']
    h = zkp_data['h']
    g = zkp_data['g']
    r = zkp_data['r']
    b = zkp_data['b']
    bit_idx = zkp_data['bit_idx']
    m = zkp_data['m']
    gamma = zkp_data['gamma_times_1000'] / 1000
    tau_z = zkp_data['tau_times_1000'] / 1000
    e_max = zkp_data['e_max']

    errors = []

    # 1. 一致性约束检查
    for t in range(n):
        if h[t] == 1:
            for i in range(l):
                if g[t][i] + r[t][i] != 1:
                    errors.append(f"Consistency error at t={t}, i={i}: g+r={g[t][i] + r[t][i]}")
        else:
            for i in range(l):
                if g[t][i] != 0 or r[t][i] != 0:
                    errors.append(f"Non-active error at t={t}, i={i}: g={g[t][i]}, r={r[t][i]}")

    # 2. 位置级多数投票 z 统计量
    N = sum(h)  # 有效样本数
    X = 0  # 命中数
    half_l = l / 2

    for t in range(n):
        if h[t] == 1:
            n_t_g = sum(g[t])
            n_t_r = sum(r[t])

            g_t = 1 if n_t_g > half_l else 0
            r_t = 1 if n_t_g < half_l else 0
            e_t = 0 if n_t_g > n_t_r else 1
            x_t = (1 - e_t) * g_t + e_t * (1 - r_t)
            X += x_t

    # Δ = X - γN
    delta = X - gamma * N
    # V = γ(1-γ)N
    V = gamma * (1 - gamma) * N

    # z = Δ / sqrt(V)
    z_score = delta / (V ** 0.5) if V > 0 else 0

    # 检验：Δ² ≥ τ_z² * V
    z_pass = (delta ** 2) >= (tau_z ** 2 * V)

    # 3. 解码和 BER
    # 代码逻辑：m̂ = g ⊕ b
    vote_0 = [0] * bits_len
    vote_1 = [0] * bits_len

    for t in range(n):
        if h[t] == 1:
            p = bit_idx[t]
            for i in range(l):
                m_hat_i = g[t][i] ^ b[t][i]
                if m_hat_i == 1:
                    vote_1[p] += 1
                else:
                    vote_0[p] += 1

    m_decoded = []
    for p in range(bits_len):
        if vote_1[p] > vote_0[p]:
            m_decoded.append(1)
        else:
            m_decoded.append(0)

    # 汉明距离
    ber = sum(1 for i in range(bits_len) if m_decoded[i] != m[i])
    ber_pass = ber <= e_max

    return {
        "consistency_errors": errors[:5],  # 只返回前5个错误
        "N": N,
        "X": X,
        "delta": delta,
        "z_score": z_score,
        "z_pass": z_pass,
        "m_decoded": m_decoded,
        "m_original": m,
        "ber": ber,
        "ber_pass": ber_pass,
        "all_pass": len(errors) == 0 and z_pass and ber_pass,
    }


if __name__ == "__main__":
    # 测试代码
    print("ZKP Data Extractor - Test")

    # 模拟数据
    vocab_size = 50000
    extractor = ZKPDataExtractor(
        vocab_size=vocab_size,
        gamma=0.5,
        entropy_threshold=0.5,
        partition_seeds=list(range(18)),  # 18层
        bits="10110101",  # 8位消息
        model_id=123456,
        tau_z=2.0,
        e_max=1,
    )

    # 模拟输入
    import random

    random.seed(42)

    prefix_len = 50
    total_len = 150
    input_ids = torch.tensor([random.randint(0, vocab_size - 1) for _ in range(total_len)])
    entropy = [random.uniform(0, 2) for _ in range(total_len)]

    # 提取数据
    zkp_data, verification_info = extractor.extract(input_ids, prefix_len, entropy)

    # 验证约束
    result = verify_zkp_constraints(zkp_data)

    print(f"\nVerification Results:")
    print(f"  N={result['N']}, X={result['X']}")
    print(f"  z_score={result['z_score']:.4f}, z_pass={result['z_pass']}")
    print(f"  m_decoded={result['m_decoded']}, m_original={result['m_original']}")
    print(f"  BER={result['ber']}, ber_pass={result['ber_pass']}")
    print(f"  all_pass={result['all_pass']}")

    if result['consistency_errors']:
        print(f"  Consistency errors: {result['consistency_errors']}")

    # 保存测试
    extractor.save_zkp_json(zkp_data, "/tmp/test_zkp.json", verification_info)