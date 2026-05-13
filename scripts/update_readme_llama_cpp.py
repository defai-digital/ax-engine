#!/usr/bin/env python3
"""Render an 'External GGUF baseline (llama.cpp Metal)' section into README.md.

Consumes a sweep_results.json produced by bench_llama_cpp_metal_sweep.py.

The rendered section is inserted directly before '### Embedding throughput'
in README.md. If the section already exists it is replaced in place.

Best-practice framing:
  * Rows are shape-compatible external GGUF baselines, NOT prompt-hash parity
    evidence. The section header and intro paragraph make this explicit.
  * Numbers are reported as absolutes; no percentage deltas vs the MLX tables.
    Readers can eyeball against the existing tables above.
  * Rows resolved to a different architecture (or where the bench failed) are
    shown as 'n/a' with the reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Same ordering as the README MLX tables.
SLUG_ORDER = [
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

SECTION_HEADER = "### External GGUF baseline — llama.cpp Metal (shape-compatible, not prompt-hash parity)"
NEXT_SECTION = "### Embedding throughput"


def fmt_num(val: float) -> str:
    if val >= 1000:
        return f"{val:,.1f}"
    return f"{val:.1f}"


def extract_row_metrics(row: dict[str, Any]) -> dict[int, dict[str, float]] | None:
    """Return {prompt_tokens: {prefill, decode, ttft}} or None if no data."""
    doc = row.get("result_doc")
    if not doc:
        return None
    out: dict[int, dict[str, float]] = {}
    for cell in doc.get("results", []):
        if cell.get("engine") != "llama_cpp_metal":
            continue
        pt = int(cell["prompt_tokens"])
        prefill = cell["prefill_tok_s"].get("median")
        decode = cell["decode_tok_s"].get("median")
        ttft_raw = cell.get("ttft_ms")
        ttft = ttft_raw.get("median") if isinstance(ttft_raw, dict) else None
        out[pt] = {"prefill": prefill, "decode": decode, "ttft": ttft}
    return out or None


def render_row_cells(metrics: dict[int, dict[str, float]] | None, key: str, fmt) -> tuple[str, str]:
    if metrics is None:
        return ("n/a", "n/a")
    cell_128 = fmt(metrics[128][key]) if 128 in metrics and metrics[128].get(key) is not None else "n/a"
    cell_512 = fmt(metrics[512][key]) if 512 in metrics and metrics[512].get(key) is not None else "n/a"
    return (cell_128, cell_512)


def status_label(row: dict[str, Any]) -> str:
    status = row.get("status")
    if status == "ok":
        return ""
    if status == "unresolved":
        return " *(no GGUF found)*"
    if status == "bench_failed":
        return " *(llama-bench failed)*"
    if status == "bench_failed_no_output":
        return " *(llama-bench produced no output)*"
    if status == "download_failed":
        return " *(download failed)*"
    if status == "resolution_error":
        return " *(resolution error)*"
    if status == "mlx_model_dir_missing":
        return " *(local MLX dir missing — no prompt artifact)*"
    return f" *({status})*"


def render_section(sweep_doc: dict[str, Any]) -> str:
    rows_by_slug = {row["slug"]: row for row in sweep_doc["rows"]}

    lines: list[str] = [SECTION_HEADER, ""]
    lines.append(
        "External shape-compatible reference produced by `llama-bench` from the "
        "Metal-enabled `llama.cpp` build. **These rows are NOT prompt-hash parity "
        "with the MLX tables above** — `llama-bench` generates its own internal "
        "synthetic prompt tokens and does not consume the harness prompt JSON. The "
        "intent of this section is one of context (is the MLX engine in the same "
        "neighborhood as a well-known third-party Metal runtime?), not head-to-head "
        "comparison. MLX bit-widths are mapped to the nearest standard GGUF K-quant "
        "(4→Q4_K_M, 5→Q5_K_M, 6→Q6_K, 8→Q8_0; UD-MLX → unsloth UD-Q4_K_XL when "
        "available). Architectural bit-for-bit equivalence is not claimed."
    )
    lines.append("")
    manifest_path = sweep_doc.get("manifest_path", "benchmarks/manifests/llama_cpp_metal/inventory.json")
    # Render repo-relative when the recorded path lives under the repo root.
    try:
        manifest_path = str(Path(manifest_path).resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        pass
    lines.append(f"- Source: `{manifest_path}`")
    lines.append(
        f"- llama-bench: `{sweep_doc.get('llama_bench', '/opt/homebrew/bin/llama-bench')}`, "
        f"repetitions={sweep_doc.get('repetitions')}, "
        f"n-gpu-layers={sweep_doc.get('n_gpu_layers')}, "
        f"prompt-tokens={sweep_doc.get('prompt_tokens')}, "
        f"generation-tokens={sweep_doc.get('generation_tokens')}"
    )
    lines.append("")

    def render_table(title: str, key: str, fmt) -> list[str]:
        out = [f"#### {title}", ""]
        out.append("| Model | MLX quantization → GGUF quant | 128 tok | 512 tok | GGUF source |")
        out.append("|---|---|---:|---:|---|")
        for slug in SLUG_ORDER:
            row = rows_by_slug.get(slug)
            if row is None:
                continue
            metrics = extract_row_metrics(row)
            cell_128, cell_512 = render_row_cells(metrics, key, fmt)
            label = status_label(row)
            repo = row.get("resolved_repo") or "—"
            quant = row.get("gguf_quant_target") or "—"
            mlx_q = row.get("readme_quant") or "—"
            out.append(
                f"| {row['readme_model']}{label} | {mlx_q} → {quant} | {cell_128} | {cell_512} | `{repo}` |"
            )
        out.append("")
        return out

    lines.extend(render_table(
        "Prefill throughput (tok/s) — higher is better",
        "prefill",
        fmt_num,
    ))
    lines.extend(render_table(
        "Decode throughput (tok/s) — generation=128 tokens, higher is better",
        "decode",
        fmt_num,
    ))
    lines.extend(render_table(
        "Time to first token (ms) — derived from `prompt_tokens / prefill_tok_s × 1000`, lower is better",
        "ttft",
        fmt_num,
    ))

    return "\n".join(lines)


def splice_section(readme_text: str, section_md: str) -> str:
    lines = readme_text.splitlines()

    # Find existing section bounds (if any) so we can replace in place.
    start = -1
    end = -1
    for i, line in enumerate(lines):
        if line.startswith(SECTION_HEADER):
            start = i
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("### ") or lines[j].startswith("## "):
                    end = j
                    break
            if end < 0:
                end = len(lines)
            break

    new_lines = section_md.splitlines() + [""]

    if start >= 0:
        return "\n".join(lines[:start] + new_lines + lines[end:]) + "\n"

    # Otherwise, insert directly before the embedding section.
    for i, line in enumerate(lines):
        if line.startswith(NEXT_SECTION):
            return "\n".join(lines[:i] + new_lines + lines[i:]) + "\n"

    raise RuntimeError(
        f"Could not locate insertion anchor '{NEXT_SECTION}' in README — "
        f"refusing to append at EOF to avoid mis-placement."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True, help="Path to sweep_results.json")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep_doc = json.loads(args.sweep.read_text())
    section_md = render_section(sweep_doc)

    if args.dry_run:
        print(section_md)
        return

    original = args.readme.read_text()
    updated = splice_section(original, section_md)
    if updated == original:
        print("README.md unchanged.")
        return
    args.readme.write_text(updated)
    print(f"README.md updated ({len(updated) - len(original):+d} bytes).")


if __name__ == "__main__":
    main()
