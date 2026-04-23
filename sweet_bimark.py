# coding=utf-8
"""
SweetBiMark: SWEET + BiMark 融合方法
修复版：使用多层投票的z值计算方式

修复点：
1) 生成端：self.gamma 赋值笔误修复（self.gamma = gamma）
2) 检测端：self.gamma 初始化致命错误修复（self.gamma = gamma）
3) 检测端：hist 去重逻辑与生成端 use_hist 开关保持一致
4) 初始化信息中打印 gamma，便于确认参数生效
"""

from __future__ import annotations

from transformers import LogitsProcessor
import torch
import numpy as np
from math import sqrt
import scipy.stats
import hashlib


def prf(seed, secret_key: int):
    """伪随机函数（来自BiMark），基于SHA256"""
    if isinstance(seed, torch.Tensor):
        if seed.dim() == 1:
            seed_str = ''.join(map(str, seed.tolist())) + str(secret_key)
            hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
            return int(hash_digest, 16) % 2 ** 32
        else:
            result = []
            for row in seed:
                seed_str = ''.join(map(str, row.tolist())) + str(secret_key)
                hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
                result.append(int(hash_digest, 16) % 2 ** 32)
            return result
    elif isinstance(seed, tuple):
        seed_str = ''.join(map(str, seed)) + str(secret_key)
        hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
        return int(hash_digest, 16) % 2 ** 32
    else:
        seed_str = str(seed) + str(secret_key)
        hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
        return int(hash_digest, 16) % 2 ** 32


class SweetBimarkLogitsProcessor(LogitsProcessor):
    """
    SweetBiMark LogitsProcessor
    SWEET熵阈值 + BiMark无偏多层水印
    """

    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 1.0,
        entropy_threshold: float = 0.5,
        partition_seeds: list = None,
        c_key: int = 530773,
        bit_idx_key: int = 283519,
        window_size: int = 2,
        bits: str = '0',
        top_k: int = 50,
        use_hist: bool = False,
        alpha: float = 1.0,
    ):
        self.vocab = vocab
        self.vocab_size = len(vocab) if vocab else 50257

        # ✅ 修复：gamma 正确赋值
        self.gamma = gamma

        self.delta = delta
        self.alpha = alpha
        self.entropy_threshold = entropy_threshold

        self.window_size = window_size
        self.bits = bits
        self.c_key = c_key
        self.bit_idx_key = bit_idx_key
        self.top_k = top_k
        self.use_hist = use_hist

        if partition_seeds is None:
            partition_seeds = list(range(10))
        self.partition_seeds = partition_seeds
        self.num_partitions = len(partition_seeds)
        self.partition_masks = []

        # ✅ 生成端：V0大小按 gamma 构造
        for key in partition_seeds:
            num_V0 = int(self.vocab_size * self.gamma)
            rng = np.random.default_rng(key)
            mask = np.zeros(self.vocab_size, dtype=bool)
            mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
            self.partition_masks.append(torch.tensor(mask).to(torch.bool))

        self.hist = None
        self.initialized = False
        self.watermarked_count = 0
        self.total_count = 0

        print(f"[SweetBimark] Initialized:")
        print(f"  vocab_size={self.vocab_size}")
        print(f"  gamma={self.gamma} (V0/绿表大小比例)")
        print(f"  delta={self.delta} (水印强度)")
        print(f"  alpha={self.alpha}")
        print(f"  entropy_threshold={self.entropy_threshold}")
        print(f"  num_partitions={self.num_partitions}")
        print(f"  use_hist={self.use_hist}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        if not self.initialized or self.hist is None:
            self.hist = [set() for _ in range(batch_size)]
            self.partition_masks = [m.to(device) for m in self.partition_masks]
            self.initialized = True
            self.watermarked_count = 0
            self.total_count = 0

        self.total_count += batch_size

        if seq_len < self.window_size:
            return scores

        # 计算熵
        raw_probs = torch.softmax(scores, dim=-1)
        ent = -torch.where(
            raw_probs > 0,
            raw_probs * raw_probs.log(),
            raw_probs.new([0.0])
        ).sum(dim=-1)
        entropy_mask = (ent > self.entropy_threshold)

        if not entropy_mask.any():
            return scores

        # Top-k
        actual_top_k = min(self.top_k, scores.shape[-1])
        score_topk = torch.topk(scores, actual_top_k, dim=-1)
        prob_topk_values = torch.softmax(score_topk.values, dim=-1).to(torch.float32)
        prob_topk_indices = score_topk.indices

        prefix = input_ids[:, -self.window_size:]

        ops_stack = []
        skip_pos = []

        for i in range(batch_size):
            if not entropy_mask[i]:
                skip_pos.append(i)
                ops_stack.append([0] * len(self.partition_masks))
                continue

            prefix_tuple = tuple(prefix[i].tolist())
            if self.use_hist:
                if prefix_tuple in self.hist[i]:
                    skip_pos.append(i)
                    ops_stack.append([0] * len(self.partition_masks))
                    continue
                else:
                    self.hist[i].add(prefix_tuple)

            self.watermarked_count += 1

            c_seed = prf(prefix_tuple, self.c_key)
            bit_idx_seed = prf(prefix_tuple, self.bit_idx_key)

            rng_c = np.random.default_rng(c_seed)
            c_list = rng_c.integers(0, 2, size=len(self.partition_masks))

            rng_bit_idx = np.random.default_rng(bit_idx_seed)
            bit_idx = rng_bit_idx.integers(0, len(self.bits))
            bit = int(self.bits[bit_idx])

            ops_list = []
            for c in c_list:
                if (c == 1 and bit == 0) or (c == 0 and bit == 1):
                    ops_list.append(1)   # 增强V0
                else:
                    ops_list.append(-1)  # 增强V1
            ops_stack.append(ops_list)

        ops_stack = torch.tensor(ops_stack, device=device, dtype=torch.float32)

        # BiMark多层概率重加权
        prob_delta = torch.full((batch_size, 1), self.delta, dtype=torch.float32).to(device)
        alpha = torch.full((batch_size, 1), self.alpha, dtype=torch.float32).to(device)

        for i in range(len(self.partition_masks)):
            top_k_mask = self.partition_masks[i][prob_topk_indices]
            p0 = torch.sum(prob_topk_values * top_k_mask, -1, keepdim=True)
            mask_p0 = (p0 < 1e-30) + (1 - p0 < 1e-30)

            delta = torch.max(
                torch.min(alpha / p0, 1 + prob_delta),
                torch.ones(prob_delta.shape).to(device)
            ) - 1
            beta = torch.min(delta * p0 / (1 - p0), torch.ones(prob_delta.shape).to(device))

            delta[mask_p0 == 1] = 0
            beta[mask_p0 == 1] = 0

            for pos in skip_pos:
                delta[pos] = 0
                beta[pos] = 0

            direction = ops_stack[:, i:i + 1]
            delta = delta * direction
            beta = beta * direction

            delta = delta.expand(-1, prob_topk_values.shape[1])
            beta = beta.expand(-1, prob_topk_values.shape[1])

            prob_topk_values[top_k_mask == True] = prob_topk_values[top_k_mask == True] * (1 + delta)[top_k_mask == True]
            prob_topk_values[top_k_mask == False] = prob_topk_values[top_k_mask == False] * (1 - beta)[top_k_mask == False]
            prob_topk_values = torch.clamp(prob_topk_values, min=1e-10)

        prob = torch.zeros_like(scores, dtype=torch.float32)
        prob.scatter_(1, prob_topk_indices, prob_topk_values)
        new_scores = torch.log(prob + 1e-10)

        return new_scores.to(scores.dtype)

    def reset(self):
        self.hist = None
        self.initialized = False
        self.watermarked_count = 0
        self.total_count = 0


class SweetBimarkDetector:
    """
    SweetBiMark Detector - 修复版
    使用多层投票的z值计算方式
    """

    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        tokenizer=None,
        z_threshold: float = 4.0,
        entropy_threshold: float = 0.5,
        partition_seeds: list = None,
        c_key: int = 530773,
        bit_idx_key: int = 283519,
        window_size: int = 2,
        bits: str = '0',
        use_hist: bool = False,
    ):
        self.vocab = vocab
        self.vocab_size = len(vocab) if vocab else 50257

        # ✅ 修复：原来写成 self.gamma = self.gamma 会直接崩
        self.gamma = gamma

        self.tokenizer = tokenizer
        self.z_threshold = z_threshold
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.bits = bits
        self.c_key = c_key
        self.bit_idx_key = bit_idx_key
        self.use_hist = use_hist

        if partition_seeds is None:
            partition_seeds = list(range(10))
        self.partition_seeds = partition_seeds
        self.num_partitions = len(partition_seeds)

        self.partition_masks = []
        # ✅ 检测端：V0大小按 gamma 构造（需与生成一致）
        for key in partition_seeds:
            num_V0 = int(self.vocab_size * self.gamma)
            rng = np.random.default_rng(key)
            mask = np.zeros(self.vocab_size, dtype=bool)
            mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
            self.partition_masks.append(mask)

        print(f"[SweetBimarkDetector] Initialized:")
        print(f"  vocab_size={self.vocab_size}")
        print(f"  gamma={self.gamma} (V0/绿表大小比例)")
        print(f"  entropy_threshold={self.entropy_threshold}")
        print(f"  num_partitions={self.num_partitions}")
        print(f"  use_hist={self.use_hist}")
        print(f"  bits={self.bits}")

    def _compute_z_score(self, green_count, valid_count):
        """标准z-score计算（这里expected=0.5保持不变：因为“绿”的定义按(c,bit)在V0/V1间切换）"""
        if valid_count == 0:
            return 0.0
        expected = 0.5
        numer = green_count - expected * valid_count
        denom = sqrt(valid_count * expected * (1 - expected))
        return numer / denom if denom > 0 else 0.0

    def _compute_p_value(self, z):
        return scipy.stats.norm.sf(z)

    def _score_sequence_voting(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        entropy: list[float],
        bits: str = None,
    ):
        """
        【投票式打分】
        对每个高熵位置:
          1) 统计L层中落入“该层绿表”的次数 green_votes
          2) green_votes > L/2 => 该位置判为绿
        green_count = 判为绿的位置数
        valid_count = 参与统计的位置总数
        """
        if bits is None:
            bits = self.bits

        score_dict = dict()
        num_tokens_generated = len(input_ids) - prefix_len
        if num_tokens_generated < 1:
            score_dict["invalid"] = True
            return score_dict

        hist = set()
        num_tokens_scored = 0
        position_votes = []  # 每个高熵位置的 green_votes

        for idx in range(prefix_len, len(input_ids)):
            # SWEET: 只处理高熵位置
            if idx < len(entropy) and entropy[idx] <= self.entropy_threshold:
                continue
            if idx < self.window_size:
                continue

            prefix_start = idx - self.window_size
            prefix = tuple(input_ids[prefix_start:idx].tolist())

            # ✅ 修复：与生成端一致，仅在 use_hist=True 时去重
            if self.use_hist:
                if prefix in hist:
                    continue
                hist.add(prefix)

            num_tokens_scored += 1

            c_seed = prf(prefix, self.c_key)
            bit_idx_seed = prf(prefix, self.bit_idx_key)

            rng_c = np.random.default_rng(c_seed)
            c_list = rng_c.integers(0, 2, size=self.num_partitions)

            rng_bit_idx = np.random.default_rng(bit_idx_seed)
            bit_idx = rng_bit_idx.integers(0, len(bits))
            bit = int(bits[bit_idx])

            curr_token = input_ids[idx].item()

            green_votes = 0
            for i, mask in enumerate(self.partition_masks):
                c = c_list[i]
                in_V0 = mask[curr_token]

                # ops=1 => V0是绿；ops=-1 => V1是绿
                if (c == 1 and bit == 0) or (c == 0 and bit == 1):
                    # V0是绿表
                    if in_V0:
                        green_votes += 1
                else:
                    # V1是绿表
                    if not in_V0:
                        green_votes += 1

            position_votes.append(green_votes)

        half_layers = self.num_partitions / 2.0
        green_count = sum(1 for votes in position_votes if votes > half_layers)
        valid_count = len(position_votes)

        z_score = self._compute_z_score(green_count, valid_count)

        # 旧方法（跨层直接累计）用于对比
        total_green_across_layers = sum(position_votes)
        total_valid_across_layers = valid_count * self.num_partitions
        z_score_old = self._compute_z_score(total_green_across_layers, total_valid_across_layers)

        score_dict["num_tokens_generated"] = num_tokens_generated
        score_dict["num_tokens_scored"] = num_tokens_scored

        score_dict["green_count"] = green_count
        score_dict["valid_count"] = valid_count
        score_dict["green_fraction"] = green_count / valid_count if valid_count > 0 else 0.0
        score_dict["z_score"] = z_score
        score_dict["p_value"] = self._compute_p_value(z_score)

        score_dict["old_green_count"] = total_green_across_layers
        score_dict["old_valid_count"] = total_valid_across_layers
        score_dict["old_green_fraction"] = (
            total_green_across_layers / total_valid_across_layers if total_valid_across_layers > 0 else 0.0
        )
        score_dict["old_z_score"] = z_score_old

        score_dict["watermarking_fraction"] = (
            num_tokens_scored / num_tokens_generated if num_tokens_generated > 0 else 0.0
        )

        # 调试信息
        score_dict["position_votes"] = position_votes
        score_dict["half_layers"] = half_layers

        return score_dict

    def _decode_sequence_voting(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        entropy: list[float],
        bits_len: int = None,
    ):
        """解码多比特消息 - 使用投票机制"""
        if bits_len is None:
            bits_len = len(self.bits)

        COUNTS = [[0, 0] for _ in range(bits_len)]
        hist = set()
        processed_count = 0

        for idx in range(prefix_len, len(input_ids)):
            if idx < len(entropy) and entropy[idx] <= self.entropy_threshold:
                continue
            if idx < self.window_size:
                continue

            prefix_start = idx - self.window_size
            prefix = tuple(input_ids[prefix_start:idx].tolist())

            # ✅ 修复：与生成端一致，仅在 use_hist=True 时去重
            if self.use_hist:
                if prefix in hist:
                    continue
                hist.add(prefix)

            processed_count += 1

            c_seed = prf(prefix, self.c_key)
            bit_idx_seed = prf(prefix, self.bit_idx_key)

            rng_c = np.random.default_rng(c_seed)
            c_list = rng_c.integers(0, 2, size=self.num_partitions)

            rng_bit_idx = np.random.default_rng(bit_idx_seed)
            bit_idx = rng_bit_idx.integers(0, bits_len)

            curr_token = input_ids[idx].item()

            vote_for_0 = 0
            vote_for_1 = 0

            for i, mask in enumerate(self.partition_masks):
                c = c_list[i]
                in_V0 = mask[curr_token]

                # BiMark解码逻辑:
                # c=1且在V1 或 c=0且在V0 → 投票给bit=1
                # c=1且在V0 或 c=0且在V1 → 投票给bit=0
                if (c == 1 and not in_V0) or (c == 0 and in_V0):
                    vote_for_1 += 1
                else:
                    vote_for_0 += 1

            if vote_for_1 > vote_for_0:
                COUNTS[bit_idx][1] += 1
            else:
                COUNTS[bit_idx][0] += 1

        decoded_bits = ''
        confidences = []

        for j in range(bits_len):
            count_0, count_1 = COUNTS[j]
            total = count_0 + count_1

            if count_0 > count_1:
                decoded_bits += '0'
                conf = count_0 / total if total > 0 else 0.5
            elif count_1 > count_0:
                decoded_bits += '1'
                conf = count_1 / total if total > 0 else 0.5
            else:
                decoded_bits += 'x'
                conf = 0.5
            confidences.append(conf)

        green_count = sum(max(COUNTS[j]) for j in range(bits_len))
        valid_count = sum(sum(COUNTS[j]) for j in range(bits_len))
        z_score = self._compute_z_score(green_count, valid_count)

        return {
            'decoded_bits': decoded_bits,
            'original_bits': self.bits,
            'match': decoded_bits == self.bits,
            'confidences': confidences,
            'COUNTS': COUNTS,
            'z_score': z_score,
            'p_value': self._compute_p_value(z_score),
            'processed_tokens': processed_count,
        }

    def detect(
        self,
        tokenized_text: torch.Tensor = None,
        tokenized_prefix: torch.Tensor = None,
        entropy: list = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        decode: bool = True,
        **kwargs,
    ) -> dict:
        assert tokenized_text is not None
        assert entropy is not None, "SweetBimark需要entropy列表"

        prefix_len = len(tokenized_prefix) if tokenized_prefix is not None else 0
        output_dict = {}

        score_dict = self._score_sequence_voting(
            input_ids=tokenized_text,
            prefix_len=prefix_len,
            entropy=entropy,
            bits=self.bits,
        )

        if return_scores:
            output_dict.update(score_dict)

        if decode:
            decode_result = self._decode_sequence_voting(
                input_ids=tokenized_text,
                prefix_len=prefix_len,
                entropy=entropy,
                bits_len=len(self.bits),
            )
            output_dict['decode_result'] = decode_result
            output_dict['decoded_bits'] = decode_result['decoded_bits']
            output_dict['original_bits'] = decode_result['original_bits']

        if return_prediction:
            z_threshold = z_threshold if z_threshold is not None else self.z_threshold
            if score_dict.pop("invalid", False):
                output_dict["invalid"] = True
                return output_dict
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict
