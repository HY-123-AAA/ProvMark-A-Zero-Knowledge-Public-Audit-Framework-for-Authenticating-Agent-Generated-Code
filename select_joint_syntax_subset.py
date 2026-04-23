#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select a joint subset of task indices where BOTH methods produce syntax-valid code.

Paper-aligned intent:
- choose MBPP instances where the compared methods "succeed in generating with no syntax error"
  (SWEET paper uses 273 MBPP instances and then performs variable renaming with 2-5 char names and 5 seeds).
  See: "choose 273 source codes ... all three methods succeed ... no syntax error" and rename lengths 2-5, 5 seeds.  :contentReference[oaicite:1]{index=1}

This script:
- loads two generations.json files (list[list[str]]): ours and sweet
- for each task index i, chooses ONE sample per method (default: first sample)
- checks syntax validity using ast.parse with robust fenced-code handling
- outputs:
  - out_indices: JSON list[int] (selected original indices)
  - optional: out_you_gens / out_sweet_gens : filtered generations with ONE sample per task (list[list[str]] length = |S|)
  - optional: out_report: debug info (reasons for exclusion)

Typical usage:
python select_joint_syntax_subset.py \
  --task mbpp \
  --you_gens outputs/you/generations.json \
  --sweet_gens outputs/sweet/generations.json \
  --out_indices outputs/joint/mbpp_joint273_ids.json \
  --top_k 273 \
  --sample_policy first \
  --out_you_gens outputs/joint/you_subset_gens.json \
  --out_sweet_gens outputs/joint/sweet_subset_gens.json \
  --out_report outputs/joint/report.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lm_eval import tasks


# -----------------------------
# Helpers: fenced code handling
# -----------------------------

_FENCE_OPEN_RE = re.compile(r"```(?:python)?\s*\n", flags=re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\n```", flags=re.IGNORECASE)


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _split_fenced_python(text: str) -> Tuple[str, Optional[str], str]:
    """
    Split a text that may contain a fenced python block:

        ```python
        ...
        ```

    Return: (head_including_open_fence, fenced_code, tail_including_close_fence)
    If no fence: (text, None, "")
    """
    m1 = _FENCE_OPEN_RE.search(text)
    if not m1:
        return text, None, ""
    start = m1.end()
    m2 = _FENCE_CLOSE_RE.search(text[start:])
    if not m2:
        return text, None, ""
    end = start + m2.start()
    return text[:start], text[start:end], text[end:]


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _safe_json_dump(obj: Any, path: str):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Syntax check
# -----------------------------

@dataclass
class SyntaxCheckResult:
    ok: bool
    used: str  # "fence" or "full"
    err: str = ""


def _try_parse(code: str) -> Optional[str]:
    """
    Return empty string on success; error message on failure.
    """
    code = _normalize_newlines(code).strip("\n")
    if not code.strip():
        return "empty"
    try:
        ast.parse(code)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _code_for_syntax_check(gen: str, prompt: str) -> Tuple[str, Optional[str]]:
    """
    Build a parse candidate:
    - If gen already includes prompt prefix, treat as full code.
    - Otherwise, assume gen is completion, build full_code = prompt + gen.
    Then:
    - If fenced python exists, try parse fenced code first (if it parses).
    - Else parse full_code.
    Returns: (primary_candidate, fenced_candidate_or_None)
    """
    gen = _normalize_newlines(gen)
    prompt = _normalize_newlines(prompt)

    if gen.startswith(prompt):
        full = gen
    else:
        full = prompt + gen

    head, fenced, tail = _split_fenced_python(full)
    if fenced is not None:
        return full, fenced
    return full, None


def check_syntax(gen: str, prompt: str) -> SyntaxCheckResult:
    full, fenced = _code_for_syntax_check(gen, prompt)

    if fenced is not None:
        err_f = _try_parse(fenced)
        if err_f is None:
            return SyntaxCheckResult(ok=True, used="fence", err="")
        # fallback to full
        err_full = _try_parse(full)
        if err_full is None:
            return SyntaxCheckResult(ok=True, used="full", err="")
        return SyntaxCheckResult(ok=False, used="fence+full", err=f"fence={err_f} | full={err_full}")

    err_full = _try_parse(full)
    if err_full is None:
        return SyntaxCheckResult(ok=True, used="full", err="")
    return SyntaxCheckResult(ok=False, used="full", err=err_full)


# -----------------------------
# Sample selection per task
# -----------------------------

def pick_sample(samples: List[str], policy: str, rng: random.Random) -> Optional[str]:
    if not samples:
        return None
    if policy == "first":
        return samples[0]
    if policy == "random":
        return samples[rng.randrange(len(samples))]
    if policy == "shortest":
        return min(samples, key=lambda s: len(s))
    if policy == "longest":
        return max(samples, key=lambda s: len(s))
    raise ValueError(f"unknown sample_policy: {policy}")


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, help="mbpp / humaneval / ...")
    p.add_argument("--you_gens", type=str, required=True, help="your generations.json (list[list[str]])")
    p.add_argument("--sweet_gens", type=str, required=True, help="sweet generations.json (list[list[str]])")

    p.add_argument("--out_indices", type=str, required=True, help="output JSON list[int] of selected original indices")
    p.add_argument("--top_k", type=int, default=273, help="cap selected indices to top_k; set <=0 to keep all")
    p.add_argument("--sample_policy", type=str, default="first",
                   choices=["first", "random", "shortest", "longest"],
                   help="how to pick ONE sample per task for syntax check + export")

    p.add_argument("--seed", type=int, default=0, help="seed used when sample_policy=random")
    p.add_argument("--out_you_gens", type=str, default="", help="optional output filtered gens (list[list[str]])")
    p.add_argument("--out_sweet_gens", type=str, default="", help="optional output filtered gens (list[list[str]])")
    p.add_argument("--out_report", type=str, default="", help="optional debug report json")

    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(int(args.seed))

    you = json.load(open(args.you_gens, "r", encoding="utf-8"))
    sweet = json.load(open(args.sweet_gens, "r", encoding="utf-8"))

    if not isinstance(you, list) or not isinstance(sweet, list):
        raise ValueError("generations.json must be a list[list[str]]")

    n = min(len(you), len(sweet))

    task = tasks.get_task(args.task)
    dataset = task.get_dataset()

    if len(dataset) < n:
        # dataset shorter than generations; still proceed with min length
        n = len(dataset)

    selected_indices: List[int] = []
    you_out: List[List[str]] = []
    sweet_out: List[List[str]] = []

    report: Dict[str, Any] = {
        "task": args.task,
        "you_gens": args.you_gens,
        "sweet_gens": args.sweet_gens,
        "sample_policy": args.sample_policy,
        "seed": args.seed,
        "n_total_considered": n,
        "excluded": [],  # list of {idx, reason, you_err, sweet_err}
    }

    for i in range(n):
        prompt = task.get_prompt(dataset[i])

        you_pick = pick_sample(you[i], args.sample_policy, rng)
        sweet_pick = pick_sample(sweet[i], args.sample_policy, rng)

        if you_pick is None or sweet_pick is None:
            report["excluded"].append({
                "idx": i,
                "reason": "missing_samples",
                "you_empty": you_pick is None,
                "sweet_empty": sweet_pick is None,
            })
            continue

        y_chk = check_syntax(you_pick, prompt)
        s_chk = check_syntax(sweet_pick, prompt)

        if y_chk.ok and s_chk.ok:
            selected_indices.append(i)
            # export as one-sample-per-task list[list[str]] to avoid 273*20 mismatch
            you_out.append([you_pick])
            sweet_out.append([sweet_pick])
        else:
            report["excluded"].append({
                "idx": i,
                "reason": "syntax_error",
                "you_ok": y_chk.ok,
                "sweet_ok": s_chk.ok,
                "you_used": y_chk.used,
                "sweet_used": s_chk.used,
                "you_err": y_chk.err,
                "sweet_err": s_chk.err,
            })

    # cap to top_k deterministically by ascending index
    if args.top_k and args.top_k > 0 and len(selected_indices) > args.top_k:
        keep_set = set(selected_indices[:args.top_k])
        # re-filter outputs to match kept indices
        new_indices: List[int] = []
        new_you: List[List[str]] = []
        new_sweet: List[List[str]] = []
        for idx, yg, sg in zip(selected_indices, you_out, sweet_out):
            if idx in keep_set:
                new_indices.append(idx)
                new_you.append(yg)
                new_sweet.append(sg)
        selected_indices, you_out, sweet_out = new_indices, new_you, new_sweet

    report["n_selected"] = len(selected_indices)
    report["top_k"] = args.top_k

    _safe_json_dump(selected_indices, args.out_indices)

    if args.out_you_gens:
        _safe_json_dump(you_out, args.out_you_gens)
    if args.out_sweet_gens:
        _safe_json_dump(sweet_out, args.out_sweet_gens)
    if args.out_report:
        _safe_json_dump(report, args.out_report)

    print(f"selected={len(selected_indices)} saved_indices={args.out_indices}")
    if args.out_you_gens:
        print(f"saved_you_subset_gens={args.out_you_gens}")
    if args.out_sweet_gens:
        print(f"saved_sweet_subset_gens={args.out_sweet_gens}")
    if args.out_report:
        print(f"saved_report={args.out_report}")


if __name__ == "__main__":
    main()
