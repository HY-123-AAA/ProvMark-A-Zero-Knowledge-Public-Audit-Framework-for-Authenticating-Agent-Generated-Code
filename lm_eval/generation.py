# coding=utf-8
# 修改自SWEET的generation.py
# 添加SweetBimark支持

import json
from math import ceil

from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessorList

from watermark import WatermarkLogitsProcessor
from sweet import SweetLogitsProcessor
from sweet_bimark import SweetBimarkLogitsProcessor  # 新增
# from exp import EXPLogitsProcessor  # 如果需要的话
from lm_eval.utils import TokenizedDataset, complete_code


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length:]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


def parallel_generations(task, dataset, accelerator, model, tokenizer, n_tasks, args):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    if args.task.startswith("apps-"):
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_length_generation,
        }
    else:
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_length": args.max_length_generation,
        }
    if task.stop_words:
        if tokenizer.eos_token:
            task.stop_words.append(tokenizer.eos_token)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
        )

    # 选择水印方法
    if args.wllm:
        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta
        )
        gen_kwargs["logits_processor"] = LogitsProcessorList([watermark_processor])

    elif args.sweet:
        sweet_processor = SweetLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            entropy_threshold=args.entropy_threshold
        )
        gen_kwargs["logits_processor"] = LogitsProcessorList([sweet_processor])

    # 新增：SweetBimark
    elif args.sweet_bimark:
        sweet_bimark_processor = SweetBimarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            entropy_threshold=args.entropy_threshold,
            # BiMark参数
            partition_seeds=args.partition_seeds,
            c_key=args.c_key,
            bit_idx_key=args.bit_idx_key,
            window_size=args.window_size,
            bits=args.bits,
            top_k=args.bimark_top_k,
            # alpha已删除，只用delta控制强度
        )
        gen_kwargs["logits_processor"] = LogitsProcessorList([sweet_bimark_processor])

    # elif args.exp:
    #     exp_processor = EXPLogitsProcessor(...)
    #     gen_kwargs["logits_processor"] = LogitsProcessorList([exp_processor])

    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = ceil(args.n_samples / args.batch_size)

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)
    model = model.to(accelerator.device)

    ds_loader = accelerator.prepare(ds_loader)

    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        prefix=args.prefix,
        postprocess=args.postprocess,
        **gen_kwargs,
    )
    return generations