# make_attacked_generations.py
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from lm_eval import tasks
from paraphrase_attacks import attack_completion_only


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def _parse_int_list(s: str) -> List[int]:
    if s is None or s.strip() == "":
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _format_out_path(path: str, seed: int) -> str:
    if "{seed}" in path:
        return path.format(seed=seed)
    base, ext = os.path.splitext(path)
    return f"{base}_seed{seed}{ext or '.json'}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, help="humaneval / mbpp / ...")
    p.add_argument("--in_json", type=str, required=True, help="path to generations.json (list[list[str]])")
    p.add_argument("--out_json", type=str, required=True, help="output path; supports {seed} placeholder")

    p.add_argument("--attack", type=str, choices=["rename", "refactor"], required=True)
    p.add_argument("--rename_ratio", type=float, default=0.25, help="used only when attack=rename")
    p.add_argument("--rename_min_len", type=int, default=2)
    p.add_argument("--rename_max_len", type=int, default=5)

    p.add_argument("--seed", type=int, default=0, help="single seed (used when --seeds empty)")
    p.add_argument("--seeds", type=str, default="", help="comma-separated list of seeds, e.g. 0,1,2,3,4")
    p.add_argument("--indices_json", type=str, default="", help="optional list of task indices to attack (e.g., 273 subset)")

    p.add_argument("--repair_prefix", action="store_true", help="heuristic: force prefix repair if generation not starting with prompt")
    p.add_argument("--strict_token_align", action="store_true", help="rename only: keep total tokenizer token count unchanged")

    # Refactor controls (local surrogate; paper uses external service)
    p.add_argument("--refactor_backend", type=str, default="ast+black", help="ast | ast+black | black | ws | cmd")
    p.add_argument("--refactor_black_line_length", type=int, default=88)
    p.add_argument("--refactor_ws_style", type=str, default="spaced", choices=["compact", "spaced"])
    p.add_argument("--refactor_cmd", type=str, default="", help="backend=cmd only; pass code via stdin, read stdout")
    p.add_argument("--refactor_cmd_timeout_s", type=int, default=30)

    # Optional HF tokenizer for strict_token_align
    p.add_argument("--hf_tokenizer_path", type=str, default="", help="HF tokenizer path or name (only needed for strict_token_align)")

    return p.parse_args()


def _load_indices(path: str) -> Optional[List[int]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(obj, list):
        return [int(x) for x in obj]
    raise ValueError("indices_json must be a JSON list of ints")


def _load_hf_tokenizer(tok_path: str):
    if not tok_path:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tok_path, use_fast=True)


def _run_one_seed(args, seed: int):
    gens = json.load(open(args.in_json, "r", encoding="utf-8"))
    task = tasks.get_task(args.task)
    dataset = task.get_dataset()

    indices = _load_indices(args.indices_json) if args.indices_json else None
    if indices is not None:
        gens = [gens[i] for i in indices]
        dataset = [dataset[i] for i in indices]

    hf_tokenizer = _load_hf_tokenizer(args.hf_tokenizer_path) if args.strict_token_align else None

    n = len(gens)
    skipped = 0
    out = []

    for i in range(n):
        prompt = task.get_prompt(dataset[i])
        new_list = []
        for gen in gens[i]:
            ar = attack_completion_only(
                full_generation=gen,
                prompt_prefix=prompt,
                attack_kind=args.attack,
                rename_ratio=args.rename_ratio,
                seed=seed,
                tokenizer=hf_tokenizer,
                repair_prefix=args.repair_prefix,
                rename_min_len=args.rename_min_len,
                rename_max_len=args.rename_max_len,
                strict_token_align=args.strict_token_align,
                refactor_backend=args.refactor_backend,
                refactor_black_line_length=args.refactor_black_line_length,
                refactor_ws_style=args.refactor_ws_style,
                refactor_cmd=(args.refactor_cmd or None),
                refactor_cmd_timeout_s=args.refactor_cmd_timeout_s,
            )
            if ar.meta.get("skipped"):
                skipped += 1
            new_list.append(ar.attacked_text)
        out.append(new_list)

    out_path = _format_out_path(args.out_json, seed)
    _ensure_dir(out_path)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"saved {out_path}")
    print(
        f"attack={args.attack} rename_ratio={args.rename_ratio} "
        f"seed={seed} n_tasks={n} skipped_items={skipped} "
        f"rename_len=[{args.rename_min_len},{args.rename_max_len}] "
        f"strict_align={args.strict_token_align} refactor_backend={args.refactor_backend}"
    )


def main():
    args = parse_args()
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        seeds = [args.seed]

    for s in seeds:
        _run_one_seed(args, s)


if __name__ == "__main__":
    main()
