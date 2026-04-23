#!/usr/bin/env python3
# tools/refactor_deepseek.py
# stdin: original code
# stdout: refactored code ONLY (no markdown, no explanation)

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error


def eprint(*args):
    print(*args, file=sys.stderr)


def strip_markdown_fences(text: str) -> str:
    """Strip ```python ... ``` if the model accidentally returns fenced code."""
    if not text:
        return text
    m = re.search(r"```[a-zA-Z0-9_-]*\s*\n([\s\S]*?)\n```", text)
    if m:
        return m.group(1)
    m2 = re.search(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)```", text)
    if m2:
        return m2.group(1).strip("\n")
    return text


def deepseek_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    # DeepSeek docs: base_url can be https://api.deepseek.com or https://api.deepseek.com/v1
    # Either way, chat endpoint is /chat/completions
    return base + "/chat/completions"


def request_with_retries(url: str, headers: dict, payload: dict, timeout_s: int, max_retries: int):
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return resp.status, body
        except urllib.error.HTTPError as ex:
            status = ex.code
            body = ex.read().decode("utf-8", errors="replace") if ex.fp else ""
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                sleep_s = min(2 ** attempt, 16) + 0.2 * attempt
                eprint(f"[deepseek] HTTP {status}, retry in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
            return status, body
        except Exception as ex:
            if attempt < max_retries:
                sleep_s = min(2 ** attempt, 16) + 0.2 * attempt
                eprint(f"[deepseek] {type(ex).__name__}: {ex}, retry in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
            raise


def main() -> int:
    code = sys.stdin.read()
    if not code.strip():
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        eprint("DEEPSEEK_API_KEY is not set")
        return 2

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
    # deepseek-reasoner 会输出 CoT（官方说可访问 CoT），做重构不推荐，用 deepseek-chat 更干净。:contentReference[oaicite:1]{index=1}
    temperature = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0"))
    top_p = float(os.environ.get("DEEPSEEK_TOP_P", "1"))
    max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096"))
    timeout_s = int(os.environ.get("DEEPSEEK_TIMEOUT_S", "60"))
    max_retries = int(os.environ.get("DEEPSEEK_MAX_RETRIES", "2"))

    url = deepseek_url(base_url)

    system_prompt = (
        "You are a code refactoring tool.\n"
        "Return ONLY the refactored code. No explanations. No markdown fences.\n"
        "Preserve exact behavior and I/O. Do NOT add prints/tests/examples.\n"
        "Keep required function/class names and signatures unchanged.\n"
    )

    user_prompt = (
        "Refactor the following Python code to be cleaner and more idiomatic while preserving semantics.\n"
        "Output ONLY the refactored code.\n"
        "<CODE>\n" + code + "\n</CODE>"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    status, body = request_with_retries(url, headers, payload, timeout_s=timeout_s, max_retries=max_retries)
    if status != 200:
        eprint(f"[deepseek] status={status}")
        eprint(body[:1000])
        return 1

    try:
        resp = json.loads(body)
        content = resp["choices"][0]["message"]["content"]
    except Exception as ex:
        eprint(f"[deepseek] parse failed: {type(ex).__name__}: {ex}")
        eprint(body[:1000])
        return 1

    out = strip_markdown_fences(content).rstrip("\n")
    if not out.strip():
        eprint("[deepseek] empty content")
        return 1

    # keep newline style similar to input
    if code.endswith("\n"):
        out += "\n"
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
