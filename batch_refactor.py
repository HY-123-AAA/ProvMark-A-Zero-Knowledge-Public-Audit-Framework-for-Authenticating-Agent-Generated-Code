#!/usr/bin/env python3
"""
批量重构 JSON 文件中的代码（修复版）
用法: python3 batch_refactor_fixed.py input.json output.json
"""

import json
import sys
import subprocess
import os
from pathlib import Path


def refactor_code(code: str, api_key: str) -> str:
    """
    调用 refactor_deepseek.py 重构单个代码片段
    """
    env = os.environ.copy()
    env['DEEPSEEK_API_KEY'] = api_key

    try:
        result = subprocess.run(
            ['python3', 'refactor_deepseek.py'],
            input=code,
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )

        if result.returncode == 0:
            refactored = result.stdout.strip()
            # 如果重构结果为空或和原代码一样，返回原代码
            if not refactored or refactored == code.strip():
                print("(未改变)", flush=True)
                return code
            return refactored
        else:
            print(f"✗ 失败", file=sys.stderr)
            print(f"stderr: {result.stderr[:200]}", file=sys.stderr)
            return code  # 失败则返回原代码
    except subprocess.TimeoutExpired:
        print(f"✗ 超时", file=sys.stderr)
        return code
    except Exception as e:
        print(f"✗ 出错: {e}", file=sys.stderr)
        return code


def main():
    if len(sys.argv) != 3:
        print("用法: python3 batch_refactor_fixed.py input.json output.json")
        return 1

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # 检查 API Key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("错误: 请设置 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        return 2

    # 读取输入 JSON
    print(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"找到 {total} 个任务\n")

    # 逐个重构
    success_count = 0
    for i, task in enumerate(data):
        print(f"[{i + 1}/{total}] ", end="", flush=True)

        # 检查是否是列表格式
        if isinstance(task, list) and len(task) > 0:
            original_code = task[0]

            print(f"重构中...", end=" ", flush=True)
            # 重构
            refactored_code = refactor_code(original_code, api_key)

            # 更新
            task[0] = refactored_code

            if refactored_code != original_code:
                success_count += 1
                print("✓ 已改进")
            else:
                print()
        else:
            print("⚠ 跳过（格式错误）")

    # 保存结果
    print(f"\n保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！成功重构 {success_count}/{total} 个样本")
    return 0


if __name__ == "__main__":
    sys.exit(main())