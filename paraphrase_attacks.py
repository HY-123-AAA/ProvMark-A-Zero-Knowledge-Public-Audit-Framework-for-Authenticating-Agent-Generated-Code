# paraphrase_attacks.py
from __future__ import annotations

import ast
import subprocess

import builtins
import io
import json
import keyword
import random
import re
import string
import tokenize
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------

def _split_fenced_python(text: str) -> Tuple[str, Optional[str], str]:
    """
    Split a text that may contain a fenced python block:

        ```python
        ...
        ```

    Return: (head_including_open_fence, fenced_code, tail_including_close_fence)
    If no fence: (text, None, "")
    """
    m1 = re.search(r"```(?:python)?\s*\n", text, flags=re.IGNORECASE)
    if not m1:
        return text, None, ""
    start = m1.end()
    m2 = re.search(r"\n```", text[start:])
    if not m2:
        return text, None, ""
    end = start + m2.start()
    return text[:start], text[start:end], text[end:]


def _tok_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _is_dunder(name: str) -> bool:
    return bool(re.fullmatch(r"__.*__", name))


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _collapse_blank_lines(lines: List[str], max_run: int = 2) -> List[str]:
    out: List[str] = []
    run = 0
    for ln in lines:
        if ln.strip() == "":
            run += 1
            if run <= max_run:
                out.append(ln)
        else:
            run = 0
            out.append(ln)
    return out


def _ws_refactor(code: str, style: str = "spaced") -> str:
    # Parser-free refactor surrogate, tries to keep syntax valid.
    # style: compact | spaced
    code = _normalize_newlines(code)
    lines = [ln.rstrip() for ln in code.split("\n")]
    lines = _collapse_blank_lines(lines, max_run=1 if style == "compact" else 2)
    out = "\n".join(lines).strip("\n")

    if style == "spaced":
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(out).readline))
            out = tokenize.untokenize(toks)
            out = _normalize_newlines(out)
        except Exception:
            pass

    if code.endswith("\n"):
        out += "\n"
    return out


def _try_black_format(code: str, line_length: int) -> Optional[str]:
    try:
        import black  # type: ignore
    except Exception:
        return None
    try:
        mode = black.Mode(line_length=int(line_length))
        return black.format_str(code, mode=mode)
    except Exception:
        return None


def _ast_roundtrip(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except Exception:
        return None


# -----------------------------
# Result
# -----------------------------

@dataclass
class AttackResult:
    attacked_text: str
    meta: Dict[str, Any]


# -----------------------------
# Rename attack
# -----------------------------

class VariableRenameAttack:
    """
    Variable rename attack (paper-aligned):
      - rename variables to random strings with length in [rename_min_len, rename_max_len]
      - deterministic given seed
      - optional strict_token_align keeps TOTAL tokenizer token count unchanged
    """

    def __init__(
        self,
        rename_ratio: float,
        seed: int,
        tokenizer=None,
        strict_token_align: bool = False,
        max_resample_maps: int = 50,
        max_name_tries: int = 20000,
        avoid_digits: bool = True,
        rename_min_len: int = 2,
        rename_max_len: int = 5,
        rename_alphabet: Optional[str] = None,
    ):
        self.rename_ratio = float(rename_ratio)
        self.rng = random.Random(int(seed))
        self.tokenizer = tokenizer
        self.strict_token_align = bool(strict_token_align and (tokenizer is not None))
        self.max_resample_maps = int(max_resample_maps)
        self.max_name_tries = int(max_name_tries)
        self.avoid_digits = bool(avoid_digits)
        self.rename_min_len = int(rename_min_len)
        self.rename_max_len = int(rename_max_len)
        self.rename_alphabet = str(rename_alphabet) if rename_alphabet else string.ascii_lowercase

        self._protected = set(keyword.kwlist) | set(dir(builtins)) | {
            "self", "cls", "True", "False", "None"
        }

    def _rand_name(self) -> str:
        L = self.rng.randint(self.rename_min_len, self.rename_max_len)
        cand = "".join(self.rng.choice(self.rename_alphabet) for _ in range(L))
        if self.avoid_digits and cand[0].isdigit():
            cand = self.rng.choice(string.ascii_lowercase) + cand[1:]
        return cand

    def _collect_identifiers(self, code: str) -> List[str]:
        code = _normalize_newlines(code)
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except Exception:
            return []

        names: List[str] = []
        prev = None
        prev_prev = None
        in_import = False
        after_from = False

        for tok in toks:
            ttype, tstr, _, _, _ = tok

            if ttype == tokenize.NAME:
                if tstr in self._protected or _is_dunder(tstr):
                    pass
                else:
                    if prev is not None and prev.type == tokenize.OP and prev.string == ".":
                        pass
                    elif in_import or after_from:
                        pass
                    elif prev_prev is not None and prev_prev.type == tokenize.NAME and prev_prev.string in ("def", "class"):
                        pass
                    else:
                        names.append(tstr)

            if ttype in (tokenize.NEWLINE, tokenize.NL):
                in_import = False
                after_from = False
            elif ttype == tokenize.NAME:
                if tstr == "import":
                    in_import = True
                elif tstr == "from":
                    after_from = True

            prev_prev = prev
            prev = tok

        seen = set()
        uniq: List[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq

    def _build_mapping(self, identifiers: List[str]) -> Dict[str, str]:
        if not identifiers:
            return {}

        k = max(1, int(round(len(identifiers) * self.rename_ratio)))
        chosen = identifiers[:]
        self.rng.shuffle(chosen)
        chosen = chosen[:k]

        mapping: Dict[str, str] = {}
        used = set(identifiers) | self._protected

        for old in chosen:
            for _ in range(self.max_name_tries):
                new = self._rand_name()
                if new not in used and not keyword.iskeyword(new):
                    mapping[old] = new
                    used.add(new)
                    break

        return mapping

    def _apply_mapping_tokenwise(self, code: str, mapping: Dict[str, str]) -> str:
        if not mapping:
            return code

        code = _normalize_newlines(code)
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
        out_toks: List[tokenize.TokenInfo] = []

        prev = None
        for tok in toks:
            ttype, tstr, start, end, line = tok
            if ttype == tokenize.NAME and tstr in mapping:
                if prev is not None and prev.type == tokenize.OP and prev.string == ".":
                    out_toks.append(tok)
                else:
                    out_toks.append(tokenize.TokenInfo(ttype, mapping[tstr], start, end, line))
            else:
                out_toks.append(tok)
            prev = tok

        out = tokenize.untokenize(out_toks)
        out = _normalize_newlines(out)

        if code.endswith("\n") and not out.endswith("\n"):
            out += "\n"
        return out

    def attack_code(self, code: str) -> AttackResult:
        identifiers = self._collect_identifiers(code)

        if not identifiers or self.rename_ratio <= 0.0:
            return AttackResult(code, {"skipped": True, "reason": "no_identifiers_or_zero_ratio"})

        if not self.strict_token_align:
            mapping = self._build_mapping(identifiers)
            attacked = self._apply_mapping_tokenwise(code, mapping)
            return AttackResult(attacked, {"rename_ratio": self.rename_ratio, "n_ids": len(identifiers), "n_renamed": len(mapping)})

        orig_tok = _tok_count(self.tokenizer, code)
        for attempt in range(self.max_resample_maps):
            mapping = self._build_mapping(identifiers)
            attacked = self._apply_mapping_tokenwise(code, mapping)
            if _tok_count(self.tokenizer, attacked) == orig_tok:
                return AttackResult(attacked, {"rename_ratio": self.rename_ratio, "strict_token_align": True, "attempt": attempt, "n_renamed": len(mapping)})

        return AttackResult(code, {"skipped": True, "reason": "strict_token_align_failed"})


# -----------------------------
# Refactor attack (improved)
# -----------------------------

class RefactoringAttack:
    # Local surrogate for the paper's external refactor service.
    # backends: ast | ast+black | black | ws | cmd

    def __init__(
        self,
        seed: int,
        backend: str = "ast+black",
        black_line_length: int = 88,
        ws_style: str = "spaced",
        cmd: Optional[str] = None,
        cmd_timeout_s: int = 30,
    ):
        self.rng = random.Random(int(seed))
        self.backend = str(backend).lower()
        self.black_line_length = int(black_line_length)
        self.ws_style = str(ws_style).lower()
        self.cmd = cmd
        self.cmd_timeout_s = int(cmd_timeout_s)

    def _run_cmd(self, code: str) -> Optional[str]:
        if not self.cmd:
            return None
        try:
            proc = subprocess.run(
                self.cmd.split(),
                input=code.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.cmd_timeout_s,
            )
            if proc.returncode != 0:
                return None
            return proc.stdout.decode("utf-8", errors="replace")
        except Exception:
            return None

    def attack_code(self, code: str) -> AttackResult:
        orig = code
        code = _normalize_newlines(code)

        meta: Dict[str, Any] = {"attack": "refactor", "backend": self.backend}
        tried: List[str] = []
        out: Optional[str] = None

        if self.backend == "cmd":
            tried.append("cmd")
            out = self._run_cmd(code)

        elif self.backend == "black":
            tried.append("black")
            out = _try_black_format(code, self.black_line_length)

        elif self.backend in ("ast", "ast+black"):
            tried.append("ast")
            ast_out = _ast_roundtrip(code)
            if ast_out is not None:
                if code.endswith("\n"):
                    ast_out = ast_out.rstrip("\n") + "\n"
                out = ast_out
                if self.backend == "ast+black":
                    tried.append("black")
                    blk = _try_black_format(out, self.black_line_length)
                    if blk is not None:
                        out = blk

        elif self.backend == "ws":
            tried.append("ws")
            out = _ws_refactor(code, style=self.ws_style)

        else:
            tried.extend(["ast", "black", "ws"])
            ast_out = _ast_roundtrip(code)
            if ast_out is not None:
                if code.endswith("\n"):
                    ast_out = ast_out.rstrip("\n") + "\n"
                out = ast_out
                blk = _try_black_format(out, self.black_line_length)
                if blk is not None:
                    out = blk
            if out is None:
                out = _ws_refactor(code, style=self.ws_style)

        if out is None:
            meta.update({"skipped": True, "reason": "refactor_failed", "tried": tried})
            return AttackResult(orig, meta)

        out = _normalize_newlines(out)
        if orig.endswith("\n") and not out.endswith("\n"):
            out += "\n"
        if (not orig.endswith("\n")) and out.endswith("\n"):
            out = out.rstrip("\n")

        if out == orig:
            meta.update({"skipped": True, "reason": "no_change", "tried": tried})
            return AttackResult(orig, meta)

        meta.update({"tried": tried, "skipped": False})
        return AttackResult(out, meta)


# -----------------------------
# Public API
# -----------------------------

def attack_completion_only(
    full_generation: str,
    prompt_prefix: str,
    attack_kind: str,
    rename_ratio: float,
    seed: int,
    tokenizer=None,
    repair_prefix: bool = False,
    rename_min_len: int = 2,
    rename_max_len: int = 5,
    strict_token_align: bool = False,
    # refactor params
    refactor_backend: str = "ast+black",
    refactor_black_line_length: int = 88,
    refactor_ws_style: str = "spaced",
    refactor_cmd: Optional[str] = None,
    refactor_cmd_timeout_s: int = 30,
) -> AttackResult:
    """
    Apply an attack to the completion part only.
    If the generation contains a fenced python block, attack only the fenced code.
    Otherwise, attack the full_generation string as code (best-effort).
    """
    repaired = False
    if full_generation.startswith(prompt_prefix):
        completion = full_generation[len(prompt_prefix):]
    else:
        completion = full_generation
        if repair_prefix:
            repaired = True

    before, fenced, after = _split_fenced_python(completion)
    code_to_attack = fenced if fenced is not None else completion

    if attack_kind == "rename":
        attacker = VariableRenameAttack(
            rename_ratio=rename_ratio,
            seed=seed,
            tokenizer=tokenizer,
            strict_token_align=strict_token_align,
            max_resample_maps=80,
            max_name_tries=30000,
            avoid_digits=True,
            rename_min_len=rename_min_len,
            rename_max_len=rename_max_len,
        )
        ar = attacker.attack_code(code_to_attack)

    elif attack_kind == "refactor":
        attacker = RefactoringAttack(
            seed=seed,
            backend=refactor_backend,
            black_line_length=refactor_black_line_length,
            ws_style=refactor_ws_style,
            cmd=refactor_cmd,
            cmd_timeout_s=refactor_cmd_timeout_s,
        )
        ar = attacker.attack_code(code_to_attack)

    else:
        raise ValueError("attack_kind must be 'rename' or 'refactor'")

    attacked_completion = (before + ar.attacked_text + after) if fenced is not None else ar.attacked_text

    if full_generation.startswith(prompt_prefix):
        attacked_full = prompt_prefix + attacked_completion
    else:
        attacked_full = (prompt_prefix + attacked_completion) if repair_prefix else attacked_completion

    meta = dict(ar.meta)
    meta["repaired_prefix"] = repaired
    meta["attack_kind"] = attack_kind
    meta["rename_min_len"] = rename_min_len
    meta["rename_max_len"] = rename_max_len
    meta["strict_token_align"] = bool(strict_token_align and (tokenizer is not None))
    return AttackResult(attacked_full, meta)
