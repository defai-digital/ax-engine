#!/usr/bin/env python3
"""
Summarise thermal-fix benchmark run vs q-slice-fix and mlx_lm.

Usage:
  python3 benchmarks/results/mlx-inference/2026-05-09-thermal-fix/analyze.py
"""
import json
import glob
import os
import sys

Q_DIR = "benchmarks/results/mlx-inference/2026-05-09-q-slice-fix"
T_DIR = "benchmarks/results/mlx-inference/2026-05-09-thermal-fix"

MODEL_ORDER = [
    "gemma-4-e2b-it-4bit",
    "gemma-4-e2b-it-5bit",
    "gemma-4-e2b-it-6bit",
    "gemma-4-e2b-it-8bit",
    "gemma-4-e4b-it-4bit",
    "gemma-4-26b-a4b-it-4bit",
    "gemma-4-31b-it-4bit",
    "qwen3_5-9b-mlx-4bit",
    "qwen3_6-35b-a3b-ud-mlx-4bit",
    "qwen3_6-35b-a3b-5bit",
    "qwen3_6-35b-a3b-6bit",
    "qwen3_6-35b-a3b-8bit",
    "qwen3-coder-next-4bit",
    "glm-4.7-flash-4bit",
]


def ax_rows(path):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for r in d.get("results", []):
        if r.get("engine") in ("mlx_lm", "ax_mlx_lm"):
            continue
        pt = r.get("prompt_tokens", 0)
        ax_d = r.get("decode_tok_s", {})
        if isinstance(ax_d, dict):
            ax_d = ax_d.get("mean", 0)
        base = r.get("baseline", {})
        mlx_d = base.get("decode_tok_s", 0) if isinstance(base, dict) else 0
        rows.append((pt, r.get("ax_decode_policy", ""), ax_d, mlx_d))
    return rows


def pct(a, b):
    if b:
        return f"{(a / b - 1) * 100:+.1f}%"
    return "n/a"


print(f"{'Model':<28} {'p':<4} {'policy':<30} {'q-fix':>7} {'therm':>7} {'vs q-fix':>9} {'vs mlx':>8}")
print("-" * 100)

for slug in MODEL_ORDER:
    t_path = f"{T_DIR}/{slug}.json"
    q_path = f"{Q_DIR}/{slug}.json"
    if not os.path.exists(t_path):
        print(f"{slug:<28} (not yet complete)")
        continue
    try:
        q = ax_rows(q_path)
        t = ax_rows(t_path)
        for i, (pt, pol, tax, tmlx) in enumerate(t):
            qax = q[i][2] if i < len(q) else 0
            label = (pol or "reused")[:30]
            print(f"{slug:<28} {pt:<4} {label:<30} {qax:>7.1f} {tax:>7.1f} {pct(tax, qax):>9} {pct(tax, tmlx):>8}")
    except Exception as e:
        print(f"{slug}: {e}")
