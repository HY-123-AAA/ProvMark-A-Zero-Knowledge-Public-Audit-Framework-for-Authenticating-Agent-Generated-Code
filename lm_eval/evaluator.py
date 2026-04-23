# coding=utf-8
# 修改自SWEET的evaluator.py
# 添加SweetBimark检测支持

import json
import os
import warnings
import traceback

import torch
from tqdm import tqdm

from lm_eval import tasks
from lm_eval.generation import parallel_generations
from lm_eval.utils import calculate_entropy, calculate_entropy_safe_long
from watermark import WatermarkDetector
from sweet import SweetDetector
from sweet_bimark import SweetBimarkDetector  # 新增
# from exp import EXPDetector

import pdb
import hashlib

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


# ==================== ZKP辅助函数（新增）====================
def generate_zkp_witness_data(tokenized_text, prefix_len, entropy, watermark_detector, original_bits, decoded_bits,
                              args):
    """生成零知识证明数据（稳健版）"""
    from sweet_bimark import prf
    import numpy as np

    # -------- 参数兜底，避免因为 Namespace 缺字段直接炸掉 --------
    model_name = str(getattr(args, "model", "unknown_model"))
    entropy_threshold = float(getattr(args, "entropy_threshold", 0.5))
    window_size = int(getattr(args, "window_size", 2))
    c_key = int(getattr(args, "c_key", 530773))
    bit_idx_key = int(getattr(args, "bit_idx_key", 283519))
    gamma = float(getattr(args, "gamma", 0.5))
    detection_z_threshold = float(getattr(args, "detection_z_threshold", 4.0))

    # -------- 类型归一化，避免 torch / numpy 标量导致隐式报错 --------
    if isinstance(tokenized_text, torch.Tensor):
        tokenized_text = tokenized_text.detach().cpu()
    else:
        tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)

    entropy = [
        float(x.item()) if hasattr(x, "item") else float(x)
        for x in entropy
    ]

    original_bits = str(original_bits)
    decoded_bits = str(decoded_bits)

    n = int(len(tokenized_text) - prefix_len)
    L = int(watermark_detector.num_partitions)
    bits_len = len(original_bits)
    model_id_u64 = int(hashlib.sha256(model_name.encode()).hexdigest()[:16], 16)

    h = []
    g_matrix = []
    r_matrix = []
    b_matrix = []
    bit_idx_list = []
    partition_masks = watermark_detector.partition_masks

    for local_idx in range(n):
        absolute_idx = prefix_len + local_idx
        h_t = 1 if (absolute_idx < len(entropy) and entropy[absolute_idx] > entropy_threshold) else 0
        h.append(h_t)

        if h_t == 0 or absolute_idx < window_size:
            g_matrix.append([0] * L)
            r_matrix.append([0] * L)
            b_matrix.append([0] * L)
            bit_idx_list.append(0)
            continue

        prefix_tokens = tuple(int(x) for x in tokenized_text[absolute_idx - window_size:absolute_idx].tolist())

        c_seed = prf(prefix_tokens, c_key)
        bit_idx_seed = prf(prefix_tokens, bit_idx_key)

        c_list = np.random.default_rng(c_seed).integers(0, 2, size=L).tolist()
        bit_idx = int(np.random.default_rng(bit_idx_seed).integers(0, bits_len))
        bit_idx_list.append(bit_idx)

        curr_token = int(tokenized_text[absolute_idx].item())

        g_row = []
        r_row = []
        for i in range(L):
            in_V0 = bool(partition_masks[i][curr_token])
            g_row.append(1 if in_V0 else 0)
            r_row.append(0 if in_V0 else 1)

        g_matrix.append(g_row)
        r_matrix.append(r_row)
        b_matrix.append([int(x) for x in c_list])

    # Rust 电路中的 m 只接受 0/1，x 先按 0 落盘
    m = []
    for ch in decoded_bits[:bits_len]:
        if ch == '1':
            m.append(1)
        elif ch == '0':
            m.append(0)
        else:
            m.append(0)
    if len(m) < bits_len:
        m += [0] * (bits_len - len(m))

    decoded_bits_fixed = decoded_bits[:bits_len].ljust(bits_len, 'x')
    original_bits_fixed = original_bits[:bits_len].ljust(bits_len, '0')
    hamming_distance = sum(1 for a, b in zip(decoded_bits_fixed, original_bits_fixed) if a != b)

    return {
        "model_id_u64": model_id_u64,
        "tau_times_1000": int(detection_z_threshold * 1000),
        "gamma_times_1000": int(gamma * 1000),
        "e_max": max(1, bits_len // 5),
        "n": n,
        "L": L,
        "bits_len": bits_len,
        "m": m[:bits_len],
        "h": [int(x) for x in h],
        "g": [[int(v) for v in row] for row in g_matrix],
        "r": [[int(v) for v in row] for row in r_matrix],
        "b": [[int(v) for v in row] for row in b_matrix],
        "bit_idx": [int(x) for x in bit_idx_list],
        "original_bits": original_bits,
        "decoded_bits": decoded_bits,
        "hamming_distance": int(hamming_distance),
    }


def save_zkp_data(zkp_list, output_dir, task_name):
    """保存ZKP数据"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"zkp_witness_{task_name}.json")

    if not zkp_list:
        print(f"\n⚠️ No valid ZKP witness collected for task={task_name}, file not written.")
        return None

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(zkp_list, f, ensure_ascii=False, indent=2)

    print(f"\n✅ ZKP saved: {path} ({len(zkp_list)} samples)")
    return path


# ==================== ZKP辅助函数结束 ====================

def _normalize_detection_result(detection_result):
    """
    兼容 detection_result 为 dict 或 tuple 的情况，确保最终返回 dict。
    tuple 的常见形态：
      - (dict, extra)
      - (dict, extra_dict)
      - (dict, extra1, extra2, ...)
    """
    extra_items = None

    if isinstance(detection_result, tuple):
        if len(detection_result) == 0:
            detection_result = {}
        else:
            extra_items = detection_result[1:] if len(detection_result) > 1 else None
            detection_result = detection_result[0]

    if detection_result is None:
        detection_result = {}

    if not isinstance(detection_result, dict):
        detection_result = {"detection_result": detection_result}

    if extra_items:
        # 只有一个 extra
        if len(extra_items) == 1:
            extra0 = extra_items[0]
            if isinstance(extra0, dict):
                # 合并进主 dict（如有冲突，后者覆盖前者）
                detection_result.update(extra0)
            else:
                detection_result["detection_extra"] = extra0
        else:
            # 多个 extra，逐个挂载，尽量不撞 key
            for i, ex in enumerate(extra_items):
                if isinstance(ex, dict):
                    for k, v in ex.items():
                        if k in detection_result:
                            detection_result[f"extra{i}_{k}"] = v
                        else:
                            detection_result[k] = v
                else:
                    detection_result[f"detection_extra_{i}"] = ex

    return detection_result


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def watermark_detect(self, task_name, generations, watermark_detector):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(generations)

        def tokenize(example):
            inputs = self.tokenizer(
                example,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }

        # for results saving
        result = {"z_threshold": self.args.detection_z_threshold}
        zkp_data_list = []  # ZKP数据收集
        detect_list = []
        ent_list = []
        detection_results = []
        len_list = []

        prompt_contents = [task.get_prompt(dataset[sample]) for sample in range(n_tasks)]

        if self.accelerator.is_main_process:
            for idx, gens in tqdm(enumerate(generations), total=len(generations)):
                n_detection = 0

                # 对于SweetBimark，检测所有n_samples个生成
                # 对于其他方法，只检测n_detection个
                max_detection = len(gens) if self.args.sweet_bimark else self.args.n_detection

                for idx2, gen in enumerate(gens):

                    # we don't check all n_samples generations (除了SweetBimark)
                    if n_detection >= max_detection:
                        continue

                    prefix = prompt_contents[idx]
                    tokenized_prefix = tokenize(prefix)['input_ids'].squeeze()
                    prefix_len = len(tokenized_prefix)

                    # if the prompt is not part of generation
                    if not gen.startswith(prefix):
                        # 尝试拼接prompt和generation
                        gen_with_prefix = prefix + gen
                        print(
                            f"Warning: task {idx}, sample {idx2} - generation doesn't start with prompt, trying to concatenate")
                        gen = gen_with_prefix

                    tokenized_text = tokenize(gen)['input_ids'].squeeze()

                    # if tokenized are not same
                    if not torch.equal(tokenized_text[:prefix_len], tokenized_prefix):
                        print(f"Warning: task {idx}, sample {idx2} - tokenized mismatch")
                        if len(tokenized_text[:prefix_len]) != len(tokenized_prefix):
                            print(f"  Skipping this sample due to tokenization mismatch")
                            continue

                    # if len of generation is 0, check next generation
                    if len(tokenized_text) - prefix_len == 0:
                        continue
                    else:
                        if idx2 != 0 and not self.args.sweet_bimark:
                            print(idx2)
                        n_detection += 1

                    # entropy calculation (SWEET和SweetBimark都需要)
                    if self.args.sweet or self.args.sweet_bimark:
                        self.model = self.model.to(self.accelerator.device)
                        tokenized_text_on_device = tokenized_text.to(self.accelerator.device)

                        is_apps_task = task_name.startswith("apps-")

                        if is_apps_task:
                            entropy = calculate_entropy_safe_long(self.model, tokenized_text_on_device)
                        else:
                            entropy = calculate_entropy(self.model, tokenized_text_on_device)

                        # we need to shift entropy to the right, so the first item is dummy
                        entropy = [0] + entropy[:-1]

                        ent_list += entropy[prefix_len:]

                    # WLLM detector
                    if self.args.wllm:
                        detection_result = watermark_detector.detect(
                            tokenized_text=tokenized_text,
                            tokenized_prefix=tokenized_prefix,
                        )
                        if not detection_result.pop('invalid', False):
                            detect_list.append(1 if detection_result['prediction'] else 0)
                            detection_results.append(detection_result)

                    # SWEET detector
                    elif self.args.sweet:
                        detection_result = watermark_detector.detect(
                            tokenized_text=tokenized_text,
                            tokenized_prefix=tokenized_prefix,
                            entropy=entropy,
                        )
                        if not detection_result.pop('invalid', False):
                            detect_list.append(1 if detection_result['prediction'] else 0)
                            detection_results.append(detection_result)

                    # SweetBimark detector (检测所有n_samples)
                    elif self.args.sweet_bimark:
                        detection_result = watermark_detector.detect(
                            tokenized_text=tokenized_text,
                            tokenized_prefix=tokenized_prefix,
                            entropy=entropy,
                            decode=True,  # 启用解码功能
                        )
                        if not detection_result.pop('invalid', False):
                            detect_list.append(1 if detection_result['prediction'] else 0)
                            # 添加样本索引信息
                            detection_result['task_idx'] = idx
                            detection_result['sample_idx'] = idx2
                            detection_results.append(detection_result)

                            # ==================== ZKP数据收集（新增）====================
                            if detection_result.get('prediction') and detection_result.get('decode_result'):
                                try:
                                    dr = detection_result['decode_result']
                                    zkp = generate_zkp_witness_data(
                                        tokenized_text, prefix_len, entropy, watermark_detector,
                                        dr.get('original_bits', self.args.bits),
                                        dr.get('decoded_bits', ''),
                                        self.args
                                    )
                                    zkp.update({
                                        'task_idx': idx,
                                        'sample_idx': idx2,
                                        'z_score': detection_result.get('z_score', 0.0)
                                    })
                                    zkp_data_list.append(zkp)
                                    print(
                                        f"[ZKP OK] task={idx}, sample={idx2}, z_score={detection_result.get('z_score', 0.0)}")
                                except Exception as e:
                                    print(f"[ZKP ERROR] task={idx}, sample={idx2}, err={repr(e)}")
                                    traceback.print_exc()
                            # ==================== ZKP数据收集结束 ====================
                    # EXP-edit detector
                    # elif self.args.exp:
                    #     ...

                    # general info
                    len_list.append(len(tokenized_text) - prefix_len)

                # all generations' len are 0
                expected_detection = max_detection if self.args.sweet_bimark else self.args.n_detection
                if n_detection < expected_detection:
                    print(f"task {idx}: only {n_detection}/{expected_detection} valid generations (others are 0 len).")
                    if n_detection == 0:
                        len_list.append(0)

            if detect_list:
                print(f"total samples: {len(detect_list)}, positive samples: {sum(detect_list)}")
                result["total_samples"] = len(detect_list)
                result["positive_samples"] = sum(detect_list)
                result["detection_rate"] = sum(detect_list) / len(detect_list)

            if ent_list:
                print(f"average entropy value excluding prompt: {sum(ent_list) / len(ent_list)}")
                result["mean_entropy"] = sum(ent_list) / len(ent_list)

            if len_list:
                print(f"average len of generated code : {sum(len_list) / len(len_list)}")
                result["mean_len"] = sum(len_list) / len(len_list)

            if detection_results:
                # watermarking_fraction
                if "watermarking_fraction" in detection_results[0]:
                    wfs = [d['watermarking_fraction'] for d in detection_results]
                    result["watermarking_fraction"] = sum(wfs) / len(wfs)

                if "green_fraction" in detection_results[0]:
                    gfs = [d['green_fraction'] for d in detection_results]
                    result["green_fraction"] = sum(gfs) / len(gfs)

                # SweetBimark特有统计：z_score分布和解码结果
                if self.args.sweet_bimark:
                    z_scores = [d['z_score'] for d in detection_results if 'z_score' in d]
                    if z_scores:
                        result["z_score_mean"] = sum(z_scores) / len(z_scores)
                        result["z_score_min"] = min(z_scores)
                        result["z_score_max"] = max(z_scores)
                        print(
                            f"z_score: mean={result['z_score_mean']:.4f}, min={result['z_score_min']:.4f}, max={result['z_score_max']:.4f}")

                    # 解码统计
                    decode_results = [d.get('decode_result') for d in detection_results if d.get('decode_result')]
                    if decode_results:
                        match_count = sum(1 for dr in decode_results if dr.get('match', False))
                        match_count_new = sum(1 for dr in decode_results if dr.get('match_new', False))
                        result["decode_accuracy"] = match_count / len(decode_results)
                        result["decode_accuracy_new"] = match_count_new / len(decode_results)
                        result["decode_match_count"] = match_count
                        result["decode_match_count_new"] = match_count_new
                        result["decode_total_count"] = len(decode_results)
                        print(
                            f"Decode accuracy (original): {match_count}/{len(decode_results)} = {result['decode_accuracy']:.4f}")
                        print(
                            f"Decode accuracy (new):      {match_count_new}/{len(decode_results)} = {result['decode_accuracy_new']:.4f}")

                        # 打印前几个解码结果示例
                        print(f"\nFirst 5 decode results:")
                        for i, dr in enumerate(decode_results[:5]):
                            print(
                                f"  [{i}] decoded={dr.get('decoded_bits')}, decoded_new={dr.get('decoded_bits_new')}, original={dr.get('original_bits')}, match={dr.get('match')}, match_new={dr.get('match_new')}")

                    # 验证z值统计
                    verify_matches = [d.get('verify_match', True) for d in detection_results]
                    if verify_matches:
                        all_match = all(verify_matches)
                        result["verify_all_match"] = all_match
                        print(f"\nZ-score verification: {'ALL MATCH' if all_match else 'MISMATCH FOUND'}")

                # 清理大序列数据，避免JSON过大
                for d in detection_results:
                    d.pop('h_sequence', None)
                    d.pop('g_sequence', None)
                    d.pop('r_sequence', None)
                    d.pop('token_sequence', None)
                    if 'decode_result' in d:
                        d['decode_result'].pop('COUNTS', None)
                        d['decode_result'].pop('COUNTS_NEW', None)

                result["raw_detection_results"] = detection_results

            # ==================== ZKP数据保存（新增）====================
            if self.args.sweet_bimark:
                if zkp_data_list:
                    save_zkp_data(zkp_data_list, self.args.outputs_dir, task_name)
                else:
                    print(
                        f"[ZKP INFO] sweet_bimark is enabled, but no valid witness was collected for task={task_name}")
            # ==================== ZKP数据保存结束 ====================
        return result

    def evaluate(self, task_name):
        task = tasks.get_task(task_name)

        # -------- 参数兜底，避免 Namespace 缺字段直接报错 --------
        skip_evaluation = getattr(self.args, "skip_evaluation", False)
        detect_human_code = getattr(self.args, "detect_human_code", False)

        # 只有在需要评测时才检查代码执行权限
        if not skip_evaluation:
            if task.requires_execution and not self.allow_code_execution:
                raise ValueError(_WARNING)

        # detecting generated code
        if not detect_human_code:
            generations, references = self.generate_text(task_name)

            # ==================== 新增：在完整 evaluate 流程中也保存 generations ====================
            if self.accelerator.is_main_process and getattr(self.args, "save_generations", False):
                try:
                    with open(self.args.save_generations_path, "w", encoding="utf-8") as fp:
                        json.dump(generations, fp, ensure_ascii=False)
                    print(f"generations were saved at {self.args.save_generations_path}")

                    if getattr(self.args, "save_references", False):
                        references_path = os.path.join(self.args.outputs_dir, "references.json")
                        with open(references_path, "w", encoding="utf-8") as fp:
                            json.dump(references, fp, ensure_ascii=False)
                        print(f"references were saved at {references_path}")
                except Exception as e:
                    print(f"[Save Generations Error] {repr(e)}")
                    traceback.print_exc()
            # ==================== 新增结束 ====================


        else:

            dataset = task.get_dataset()

            n_tasks = self.args.limit if self.args.limit else len(dataset)

            generations = []

            for sample in range(n_tasks):

                if hasattr(task, "get_full_data"):

                    code_with_prompt = task.get_full_data(dataset[sample])

                else:

                    prompt = task.get_prompt(dataset[sample])

                    reference = task.get_reference(dataset[sample])

                    code_with_prompt = prompt + reference

                generations.append([code_with_prompt])

            references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

            if self.accelerator.is_main_process:
                print(f"Human-written code loaded for detection, {n_tasks} tasks")

        # Initialize detector
        if self.args.wllm:
            watermark_detector = WatermarkDetector(
                vocab=list(self.tokenizer.get_vocab().values()),
                gamma=self.args.gamma,
                seeding_scheme="simple_1",
                device=self.accelerator.device,
                tokenizer=self.tokenizer,
                z_threshold=self.args.detection_z_threshold,
                normalizers=[],
                ignore_repeated_bigrams=True,
            )
        elif self.args.sweet:
            watermark_detector = SweetDetector(
                vocab=list(self.tokenizer.get_vocab().values()),
                gamma=self.args.gamma,
                z_threshold=self.args.detection_z_threshold,
                entropy_threshold=self.args.entropy_threshold,
            )
        elif self.args.sweet_bimark:
            watermark_detector = SweetBimarkDetector(
                vocab=list(self.tokenizer.get_vocab().values()),
                gamma=self.args.gamma,
                z_threshold=self.args.detection_z_threshold,
                entropy_threshold=self.args.entropy_threshold,
                partition_seeds=self.args.partition_seeds,
                c_key=self.args.c_key,
                bit_idx_key=self.args.bit_idx_key,
                window_size=self.args.window_size,
                bits=self.args.bits,
                use_hist=self.args.use_hist,
            )

        # run watermark detection
        detection_result = self.watermark_detect(task_name, generations, watermark_detector)

        # 兼容 detection_result 是 tuple 的情况（保险）
        if isinstance(detection_result, tuple):
            detection_result = detection_result[0] if len(detection_result) > 0 else {}
        if detection_result is None:
            detection_result = {}
        if not isinstance(detection_result, dict):
            detection_result = {"detection_result": detection_result}

        # skip evaluation
        if skip_evaluation:
            return detection_result

        # run evaluation
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        results = task.process_results(generations, references)

        # ✅ 关键修复：兼容 process_results 返回 tuple 的情况
        if isinstance(results, tuple):
            base = results[0] if len(results) > 0 else {}
            extras = results[1:]
            if not isinstance(base, dict):
                base = {"task_results": base}
            if extras:
                base["task_results_extra"] = extras[0] if len(extras) == 1 else list(extras)
            results = base

        if results is None:
            results = {}
        if not isinstance(results, dict):
            results = {"task_results": results}

        # combine detection and evaluation results
        results.update(detection_result)
        return results
