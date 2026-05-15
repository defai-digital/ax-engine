#!/usr/bin/env python3
"""Render the AX MLX prefill-stage profile across one or more bench artifacts.

The bench harness emits `ax_mlx_prefill_profile_*` per-stage µs counters
under each `trials[].ax_mlx_prefill_profile` block when run with
`--ax-prefill-profile`. This renderer aggregates the median across
trials and produces a markdown table (one model per column or one row,
depending on `--layout`) with `%-of-forward` shares.

Usage:
    python3 scripts/render_mlx_prefill_profile_report.py \\
        --results-dir benchmarks/results/mlx-inference/2026-05-15-prefill-profile-baseline \\
        --output /tmp/prefill-profile-report.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


# (label, key, parent_key_or_None_for_top_level)
# Order matters — controls row order in the output table.
STAGES: list[tuple[str, str, str | None]] = [
    ("per-layer input",          "ax_mlx_prefill_profile_per_layer_input_wall_us",          None),
    ("pre-SDPA (umbrella)",      "ax_mlx_prefill_profile_pre_sdpa_wall_us",                  None),
    ("  QKV projection",         "ax_mlx_prefill_profile_pre_sdpa_qkv_proj_wall_us",         "ax_mlx_prefill_profile_pre_sdpa_wall_us"),
    ("  QK norm",                "ax_mlx_prefill_profile_pre_sdpa_qk_norm_wall_us",          "ax_mlx_prefill_profile_pre_sdpa_wall_us"),
    ("  RoPE + KV append",       "ax_mlx_prefill_profile_pre_sdpa_rope_kv_wall_us",          "ax_mlx_prefill_profile_pre_sdpa_wall_us"),
    ("SDPA",                     "ax_mlx_prefill_profile_sdpa_wall_us",                      None),
    ("post-attn (umbrella)",     "ax_mlx_prefill_profile_post_attn_wall_us",                 None),
    ("  output projection",      "ax_mlx_prefill_profile_post_attn_output_proj_wall_us",     "ax_mlx_prefill_profile_post_attn_wall_us"),
    ("  residual norm",          "ax_mlx_prefill_profile_post_attn_residual_norm_wall_us",   "ax_mlx_prefill_profile_post_attn_wall_us"),
    ("  residual gate",          "ax_mlx_prefill_profile_post_attn_residual_gate_wall_us",   "ax_mlx_prefill_profile_post_attn_wall_us"),
    ("  FFN (umbrella)",         "ax_mlx_prefill_profile_post_attn_ffn_wall_us",             "ax_mlx_prefill_profile_post_attn_wall_us"),
    ("    FFN gate+up",          "ax_mlx_prefill_profile_post_attn_ffn_gate_up_wall_us",     "ax_mlx_prefill_profile_post_attn_ffn_wall_us"),
    ("    FFN activation",       "ax_mlx_prefill_profile_post_attn_ffn_activation_wall_us",  "ax_mlx_prefill_profile_post_attn_ffn_wall_us"),
    ("    FFN down",             "ax_mlx_prefill_profile_post_attn_ffn_down_wall_us",        "ax_mlx_prefill_profile_post_attn_ffn_wall_us"),
    ("LM head",                  "ax_mlx_prefill_profile_lm_head_wall_us",                   None),
]


def median_stage_us(trials: list[dict[str, Any]], key: str) -> int:
    """Median across trials of a single profile field. Trials without the
    field contribute 0 (e.g., when the profile wasn't enabled for that
    trial — should not happen in a well-formed artifact, but we are
    defensive)."""
    vals = []
    for trial in trials:
        block = trial.get("ax_mlx_prefill_profile") or {}
        vals.append(int(block.get(key, 0)))
    if not vals:
        return 0
    return int(statistics.median(vals))


def load_model_profile(artifact_path: Path) -> dict[str, Any]:
    """Parse one bench artifact and return the AX direct prefill row's
    per-stage median µs values plus the forward total."""
    data = json.loads(artifact_path.read_text())
    for row in data.get("results", []):
        if row.get("engine") != "ax_engine_mlx":
            continue
        if int(row.get("prompt_tokens", 0)) != 4096:
            continue
        trials = row.get("trials", [])
        if not trials:
            continue
        forward_us = median_stage_us(trials, "ax_mlx_prefill_forward_wall_us") or (
            # Fall back to the umbrella field if present in ax_mlx_telemetry
            int(
                statistics.median(
                    [
                        int(t.get("ax_mlx_telemetry", {}).get("ax_mlx_prefill_forward_wall_us", 0))
                        for t in trials
                    ]
                )
            )
        )
        return {
            "model": data.get("model", artifact_path.stem),
            "model_dir": data.get("model_dir", ""),
            "prefill_tok_s_median": row.get("prefill_tok_s", {}).get("median"),
            "forward_wall_us": forward_us,
            "stages": {key: median_stage_us(trials, key) for _, key, _ in STAGES},
            "layers": median_stage_us(trials, "ax_mlx_prefill_profile_layers"),
            "tokens": median_stage_us(trials, "ax_mlx_prefill_profile_tokens"),
            "steps":  median_stage_us(trials, "ax_mlx_prefill_profile_prefill_steps"),
        }
    raise RuntimeError(f"no ax_engine_mlx pt=4096 row in {artifact_path}")


def fmt_us(x: int) -> str:
    if x == 0:
        return "—"
    return f"{x:,}"


def fmt_pct(part: int, whole: int) -> str:
    if whole <= 0 or part == 0:
        return "—"
    return f"{part / whole * 100:.1f}%"


def render_one_model(profile: dict[str, Any]) -> str:
    """Render a single-model report: stages × (µs, %-of-forward,
    %-of-parent)."""
    lines: list[str] = []
    lines.append(f"### {profile['model']} @ prompt=4096, generation=128")
    lines.append("")
    lines.append(
        f"Forward wall (median across trials): **{profile['forward_wall_us']:,} µs** — "
        f"{profile['steps']} prefill step(s), {profile['layers']} layer-passes, "
        f"{profile['tokens']} tokens, "
        f"median AX direct prefill = {profile['prefill_tok_s_median']:,.1f} tok/s."
    )
    lines.append("")
    lines.append("| Stage | µs (median) | %-of-forward | %-of-parent |")
    lines.append("|---|---:|---:|---:|")
    fw = profile["forward_wall_us"]
    for label, key, parent_key in STAGES:
        v = profile["stages"][key]
        pct_fw = fmt_pct(v, fw)
        if parent_key is None:
            pct_parent = "—"
        else:
            pv = profile["stages"].get(parent_key, 0)
            pct_parent = fmt_pct(v, pv)
        lines.append(f"| {label} | {fmt_us(v)} | {pct_fw} | {pct_parent} |")
    return "\n".join(lines) + "\n"


def render_all(profiles: list[dict[str, Any]]) -> str:
    out: list[str] = []
    out.append("# AX MLX prefill-stage profile baseline")
    out.append("")
    out.append(
        "Per-stage prefill wall-time breakdown for the 14 README models at "
        "prompt=4096 / generation=128. Captured with `--ax-prefill-profile` "
        "and `AX_MLX_PREFILL_PROFILE=1`. Stages match the "
        "`ax_mlx_prefill_profile_*` SSE telemetry keys; sub-stages are "
        "indented under their umbrella parent. `%-of-forward` is the share "
        "of `ax_mlx_prefill_forward_wall_us`. `%-of-parent` is the share "
        "of the immediate umbrella stage."
    )
    out.append("")
    out.append("Source: `benchmarks/results/mlx-inference/2026-05-15-prefill-profile-baseline/`.")
    out.append("")
    out.append("## Cross-model hot-stage summary")
    out.append("")
    out.append("| Model | Forward (µs) | FFN total % | SDPA % | pre-SDPA % | post-attn-output % | per-layer-input % |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for p in profiles:
        fw = p["forward_wall_us"]
        ffn = p["stages"]["ax_mlx_prefill_profile_post_attn_ffn_wall_us"]
        sdpa = p["stages"]["ax_mlx_prefill_profile_sdpa_wall_us"]
        pre = p["stages"]["ax_mlx_prefill_profile_pre_sdpa_wall_us"]
        oproj = p["stages"]["ax_mlx_prefill_profile_post_attn_output_proj_wall_us"]
        pli = p["stages"]["ax_mlx_prefill_profile_per_layer_input_wall_us"]
        out.append(
            f"| {p['model']} | {fmt_us(fw)} | {fmt_pct(ffn, fw)} | "
            f"{fmt_pct(sdpa, fw)} | {fmt_pct(pre, fw)} | "
            f"{fmt_pct(oproj, fw)} | {fmt_pct(pli, fw)} |"
        )
    out.append("")
    out.append("## Per-model breakdown")
    out.append("")
    for p in profiles:
        out.append(render_one_model(p))
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory of per-model bench artifact JSONs")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the report here. Default: print to stdout.")
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 2

    artifacts = sorted(args.results_dir.glob("*.json"))
    artifacts = [a for a in artifacts if "-prompts" not in str(a) and a.name != "logs"]

    profiles: list[dict[str, Any]] = []
    for a in artifacts:
        try:
            p = load_model_profile(a)
            # A profile is "populated" if at least one leaf stage has a
            # non-zero µs value. The umbrella stage (e.g. post_attn_wall_us)
            # may legitimately be 0 on architectures whose forward path
            # routes through stages the umbrella doesn't cover — most
            # visibly on GLM 4.7 Flash (pure MLA), where the bulk of
            # forward time lands in `post_attn_residual_norm_wall_us`
            # while the post_attn umbrella reports 0.
            total = sum(v for v in p["stages"].values())
            if total == 0:
                print(f"  [warn] {a.name}: prefill profile not populated, skipping",
                      file=sys.stderr)
                continue
            profiles.append(p)
        except Exception as e:
            print(f"  [warn] {a.name}: {e}", file=sys.stderr)

    if not profiles:
        print("no profiles loaded — was --ax-prefill-profile set?", file=sys.stderr)
        return 3

    report = render_all(profiles)
    if args.output:
        args.output.write_text(report)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
