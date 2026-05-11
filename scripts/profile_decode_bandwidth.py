#!/usr/bin/env python3
"""Profile decode-stage weight bandwidth utilization for an ax-engine model.

This script produces the W1 evidence artifact required by the Weight
Quantization & Speculation PRD (`.internal/planning/WEIGHT-QUANT-AND-SPECULATION-PRD.md`)
under ADR 0022. It answers one question per model: is decode binding on weight
bandwidth, on dispatch overhead, or mixed? That answer gates whether
rotation-based or sub-4-bit weight quantization is worth pursuing for the model.

The artifact is intentionally minimal — it does not replace the broader runtime
performance PRD's measurements. It records one long-context profile (default:
prompt=2048, decode=256) and classifies the model.

Artifact schema: ax.bw_profile.v1

    {
      "schema_version": "ax.bw_profile.v1",
      "generated_at_utc": "...",
      "model": {
        "model_id": str,
        "artifacts_dir": str,
        "weight_bytes_on_disk": int       # sum of *.safetensors / *.gguf etc.
      },
      "host": {
        "platform": str,
        "peak_bandwidth_gbps": float,
        "peak_bandwidth_source": str       # "user_provided"
      },
      "profile": {
        "prompt_tokens_requested": int,
        "decode_tokens_requested": int,
        "warmup_runs": int,
        "seed": int,
        "deterministic": bool
      },
      "measurements": {
        "decode_tokens_observed": int,
        "speculation_bonus_tokens": int,        # from ax_mlx_bonus_tokens route key
        "forward_pass_count": int,              # tokens_observed - bonus_tokens
        "decode_wall_time_s": float,
        "decode_tok_s": float,                  # tokens / wall (user-observed rate)
        "forward_pass_per_s": float,            # forward_passes / wall (kernel rate)
        "mean_forward_pass_us": float,          # wall / forward_pass_count
        "median_step_event_us": float,
        "p95_step_event_us": float,
        "mean_runner_us": float | null,
        "estimated_bytes_per_forward_pass": int,
        "estimated_effective_bandwidth_gbps": float,
        "bandwidth_utilization_ratio": float,
        "mlx_dispatch_per_step": null,          # not exposed to Python
        "mlx_dispatch_unavailable_reason": str
      },
      "classification": {
        "label": "weight-bandwidth-bound" | "dispatch-bound" | "mixed",
        "kernel_label": same enum | null,        # single-decode-only classification
        "rule_version": "v1",
        "thresholds": {"bandwidth_bound_min": 0.6, "dispatch_bound_max": 0.3},
        "regime_breakdown": {
          "single_decode": <regime>|null,
          "ngram_decode": <regime>|null
        }
      },
      # <regime> = { "steps": int, "wall_us": int, "us_per_step": float,
      #              "effective_bandwidth_gbps": float, "utilization_ratio": float }
      "route": { "selected_backend": str, "crossover_decisions": dict }
    }

Classification rule v1:
  - bandwidth_utilization_ratio >= 0.6  → "weight-bandwidth-bound"
  - bandwidth_utilization_ratio <= 0.3  → "dispatch-bound" (note: cannot
        distinguish dispatch vs other overhead without MLX-internal counters;
        recorded as a lower bound — promote with Instruments evidence)
  - otherwise                            → "mixed"

The estimated-bytes-per-step proxy is the on-disk weight footprint. At batch=1
decode each step touches ~all weights once; this is the standard memory-bound
analysis assumption. Promotion of any weight-quantization track must validate
this assumption with Instruments before basing decisions on a `mixed`
classification.

**Speculation isolation**: n-gram speculative decoding produces multiple
accepted tokens per forward pass, which inflates per-token throughput and
breaks the naive `tokens / time` bytes-per-token assumption. This script
recovers the raw forward-pass count from the engine's route decisions
(`ax_mlx_bonus_tokens`) and uses that — not the output-token count — to
compute kernel-level bandwidth utilization. Both rates are reported in the
artifact: `decode_tok_s` is the effective rate users observe;
`forward_pass_per_s` is what the bandwidth math is based on.

Usage:
    python scripts/profile_decode_bandwidth.py \\
        --model-id qwen3_dense \\
        --mlx-artifacts-dir /path/to/mlx-community/Qwen3-9B-4bit \\
        --peak-bandwidth-gbps 800 \\
        --output-root benchmarks/results/bw-profile

The script writes one artifact per invocation to
`{output_root}/{model_id_safe}-{YYYY-MM-DD}.json`.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.bw_profile.v1"
CLASSIFICATION_RULE_VERSION = "v1"
BANDWIDTH_BOUND_MIN = 0.6
DISPATCH_BOUND_MAX = 0.3
WEIGHT_FILE_SUFFIXES = (".safetensors", ".gguf", ".npz", ".bin")
DEFAULT_PROMPT_TOKENS = 2048
DEFAULT_DECODE_TOKENS = 256
DEFAULT_WARMUP_RUNS = 1
DEFAULT_SEED = 1234


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True, help="Model id passed to ax_engine.Session.")
    p.add_argument(
        "--mlx-artifacts-dir",
        required=True,
        type=Path,
        help="Directory containing the MLX-community model artifacts.",
    )
    p.add_argument(
        "--peak-bandwidth-gbps",
        required=True,
        type=float,
        help="Peak unified-memory bandwidth for the host SoC (GB/s). E.g. M5 Max ≈ 800.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/results/bw-profile"),
        help="Directory to write the artifact into.",
    )
    p.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    p.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    p.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--run-tag",
        default=None,
        help="Optional suffix appended to artifact filename (for variance runs).",
    )
    p.add_argument(
        "--prompt-text",
        default=None,
        help="Override the synthetic prompt text. Default repeats a fixed English paragraph.",
    )
    return p.parse_args()


def sum_weight_bytes(artifacts_dir: Path) -> int:
    if not artifacts_dir.is_dir():
        raise SystemExit(f"--mlx-artifacts-dir does not exist or is not a directory: {artifacts_dir}")
    total = 0
    for path in artifacts_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in WEIGHT_FILE_SUFFIXES:
            total += path.stat().st_size
    if total == 0:
        raise SystemExit(
            f"No weight files found under {artifacts_dir} "
            f"(looked for {WEIGHT_FILE_SUFFIXES})."
        )
    return total


# Tensor role substrings that designate stacked routed-expert weights in MLX
# model manifests. Routed experts fire on `experts_per_token` of the
# `expert_count` total per token; everything else is always active.
ROUTED_EXPERT_ROLE_TAGS = ("_exps", "routed_expert")


def is_routed_expert_role(role: str) -> bool:
    if "shared_expert" in role:
        return False
    return any(tag in role for tag in ROUTED_EXPERT_ROLE_TAGS)


def compute_weight_byte_breakdown(artifacts_dir: Path) -> dict:
    """Return weight byte accounting from `model-manifest.json`.

    Falls back to `total_disk_bytes` from filesystem scan when manifest is
    absent or has no MoE block. For MoE models, computes active bytes per
    forward pass by scaling routed-expert tensors by the
    `experts_per_token / expert_count` activation ratio.
    """
    disk_bytes = sum_weight_bytes(artifacts_dir)
    manifest_path = artifacts_dir / "model-manifest.json"
    if not manifest_path.is_file():
        return {
            "total_disk_bytes": disk_bytes,
            "active_bytes_per_forward_pass": disk_bytes,
            "is_moe": False,
            "moe_block": None,
            "routed_expert_bytes_total": 0,
            "moe_active_ratio": None,
            "breakdown_source": "filesystem_scan_no_manifest",
        }

    manifest = json.loads(manifest_path.read_text())
    moe_block = manifest.get("moe")
    tensors = manifest.get("tensors", [])
    routed_bytes = 0
    other_bytes = 0
    for t in tensors:
        role = t.get("role", "")
        nbytes = t.get("length_bytes", 0)
        if is_routed_expert_role(role):
            routed_bytes += nbytes
        else:
            other_bytes += nbytes

    manifest_total = routed_bytes + other_bytes
    if not moe_block or routed_bytes == 0:
        # Dense model: every byte is active per forward pass.
        return {
            "total_disk_bytes": disk_bytes,
            "manifest_weight_bytes": manifest_total,
            "active_bytes_per_forward_pass": manifest_total if manifest_total > 0 else disk_bytes,
            "is_moe": False,
            "moe_block": None,
            "routed_expert_bytes_total": 0,
            "moe_active_ratio": None,
            "breakdown_source": "manifest_dense",
        }

    expert_count = int(moe_block.get("expert_count", 0))
    experts_per_token = int(moe_block.get("experts_per_token", 0))
    if expert_count <= 0 or experts_per_token <= 0:
        # Manifest names an `moe` block but values aren't usable.
        return {
            "total_disk_bytes": disk_bytes,
            "manifest_weight_bytes": manifest_total,
            "active_bytes_per_forward_pass": manifest_total,
            "is_moe": True,
            "moe_block": moe_block,
            "routed_expert_bytes_total": routed_bytes,
            "moe_active_ratio": None,
            "breakdown_source": "manifest_moe_invalid_block",
        }

    active_ratio = experts_per_token / expert_count
    active_routed_bytes = int(routed_bytes * active_ratio)
    active_bytes = other_bytes + active_routed_bytes
    return {
        "total_disk_bytes": disk_bytes,
        "manifest_weight_bytes": manifest_total,
        "active_bytes_per_forward_pass": active_bytes,
        "is_moe": True,
        "moe_block": moe_block,
        "routed_expert_bytes_total": routed_bytes,
        "moe_active_ratio": round(active_ratio, 6),
        "breakdown_source": "manifest_moe_sparse",
    }


def synthesize_prompt(target_tokens: int) -> str:
    # Each repetition is ~80 chars ≈ 16 BPE tokens; oversample then let the
    # engine truncate during tokenization if needed.
    base = (
        "The quick brown fox jumps over the lazy dog while the engine "
        "warms its caches and the operator records throughput metrics. "
    )
    repeats = max(1, (target_tokens // 16) + 4)
    return base * repeats


def classify(utilization: float) -> str:
    if utilization >= BANDWIDTH_BOUND_MIN:
        return "weight-bandwidth-bound"
    if utilization <= DISPATCH_BOUND_MAX:
        return "dispatch-bound"
    return "mixed"


def load_tokenizer(artifacts_dir: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit(
            "tokenizers package not installed. Install with `pip install tokenizers`."
        ) from e
    tok_path = artifacts_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(
            f"Expected tokenizer.json under --mlx-artifacts-dir, not found: {tok_path}"
        )
    return Tokenizer.from_file(str(tok_path))


def tokenize_to_target(tokenizer, text: str, target_tokens: int) -> list[int]:
    ids = tokenizer.encode(text).ids
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
    return ids


def run_profile(args: argparse.Namespace) -> dict:
    try:
        from ax_engine import Session
    except ImportError as e:
        raise SystemExit(
            "ax_engine module not importable. Run `maturin develop` from the "
            "repo root to build and install the Python extension."
        ) from e

    weight_breakdown = compute_weight_byte_breakdown(args.mlx_artifacts_dir)
    weight_bytes = weight_breakdown["active_bytes_per_forward_pass"]
    prompt_text = args.prompt_text or synthesize_prompt(args.prompt_tokens)
    tokenizer = load_tokenizer(args.mlx_artifacts_dir)
    prompt_tokens = tokenize_to_target(tokenizer, prompt_text, args.prompt_tokens)

    session = Session(
        model_id=args.model_id,
        mlx=True,
        mlx_model_artifacts_dir=str(args.mlx_artifacts_dir),
        deterministic=True,
    )
    runtime = session.runtime()

    # Warmup: compile kernels, populate caches.
    for _ in range(max(0, args.warmup_runs)):
        _ = session.generate(
            input_tokens=prompt_tokens,
            max_output_tokens=8,
            temperature=0.0,
            seed=args.seed,
        )

    # Profile run: stream so we can separate prefill from decode.
    decode_started: float | None = None
    decode_ended: float | None = None
    step_wall_us: list[int] = []
    runner_us_samples: list[int] = []
    decode_tokens_observed = 0
    last_event_wall = time.monotonic()
    crossover_decisions: dict[str, int] = {}
    final_response = None

    for event in session.stream_generate(
        input_tokens=prompt_tokens,
        max_output_tokens=args.decode_tokens,
        temperature=0.0,
        seed=args.seed,
        deterministic=True,
    ):
        now = time.monotonic()
        if event.event == "step" and event.step is not None and decode_started is not None:
            step_wall_us.append(int((now - last_event_wall) * 1_000_000))
            if event.step.runner_time_us:
                runner_us_samples.append(event.step.runner_time_us)
        if event.delta_tokens:
            if decode_started is None:
                # First decode token marks end-of-prefill.
                decode_started = now
            decode_tokens_observed += len(event.delta_tokens)
            decode_ended = now
        if event.event == "response" and event.response is not None:
            final_response = event.response
            if event.response.route and event.response.route.crossover_decisions:
                crossover_decisions = dict(event.response.route.crossover_decisions)
        last_event_wall = now

    if decode_started is None or decode_ended is None or decode_tokens_observed == 0:
        raise SystemExit("No decode tokens observed. Aborting; cannot classify.")

    decode_wall_s = decode_ended - decode_started
    if decode_wall_s <= 0:
        raise SystemExit("Non-positive decode wall time. Aborting.")

    # Recover the true forward-pass count from the engine's route metadata.
    # Speculative decoding emits multiple accepted tokens per forward pass; the
    # `ax_mlx_bonus_tokens` counter records the speculation amplification.
    bonus_tokens = int(crossover_decisions.get("ax_mlx_bonus_tokens", 0))
    forward_pass_count = max(1, decode_tokens_observed - bonus_tokens)

    # Decompose the forward-pass population by regime. Single-decode steps run
    # one forward pass without speculation-verification overhead; n-gram steps
    # run one forward pass with N drafted tokens plus accept/reject logic.
    # Bytes-per-forward-pass is the same in both regimes (weights are touched
    # once per forward pass regardless of token count), but the wall time per
    # step differs sharply — n-gram steps carry verification overhead. The two
    # regimes therefore have very different bandwidth utilizations.
    single_decode_steps = int(crossover_decisions.get("ax_mlx_single_decode_steps", 0))
    single_decode_wall_us = int(crossover_decisions.get("ax_mlx_single_decode_wall_us", 0))
    ngram_decode_steps = int(crossover_decisions.get("ax_mlx_ngram_decode_steps", 0))
    ngram_decode_wall_us = int(crossover_decisions.get("ax_mlx_ngram_decode_wall_us", 0))

    def regime_stats(steps: int, wall_us: int) -> dict | None:
        if steps <= 0 or wall_us <= 0:
            return None
        us_per_step = wall_us / steps
        bw = weight_bytes / (us_per_step * 1e-6)
        return {
            "steps": steps,
            "wall_us": wall_us,
            "us_per_step": round(us_per_step, 1),
            "effective_bandwidth_gbps": round(bw / 1e9, 3),
            "utilization_ratio": round(
                bw / (args.peak_bandwidth_gbps * 1e9), 4
            ),
        }

    single_regime = regime_stats(single_decode_steps, single_decode_wall_us)
    ngram_regime = regime_stats(ngram_decode_steps, ngram_decode_wall_us)

    # A utilization ratio > 1 is physically impossible against advertised peak
    # bandwidth. It indicates the forward-pass benefited from on-chip caches
    # (consecutive steps reusing recently-touched weights) or that the user
    # supplied an under-stated --peak-bandwidth-gbps. Flag the anomaly rather
    # than classifying past it; the bytes-per-forward-pass model assumes a
    # cold weight fetch which is not always true for short bursts of single
    # decode steps following warm n-gram speculation.
    if single_regime and single_regime["utilization_ratio"] > 1.0:
        kernel_classification = "anomaly_above_peak_check_cache_or_peak_arg"
    elif single_regime:
        kernel_classification = classify(single_regime["utilization_ratio"])
    else:
        kernel_classification = None

    median_step_us = float(statistics.median(step_wall_us)) if step_wall_us else 0.0
    p95_step_us = (
        float(statistics.quantiles(step_wall_us, n=20)[-1]) if len(step_wall_us) >= 20 else 0.0
    )
    mean_runner_us = float(statistics.mean(runner_us_samples)) if runner_us_samples else None
    mean_forward_pass_us = decode_wall_s * 1_000_000 / forward_pass_count

    # Bytes-per-forward-pass proxy: full weight footprint, since batch=1 decode
    # touches the entire model once per forward pass (speculation runs the
    # drafted suffix through the same forward pass, not additional passes).
    bytes_per_step = weight_bytes
    eff_bandwidth_bytes_per_s = bytes_per_step / (decode_wall_s / forward_pass_count)
    eff_bandwidth_gbps = eff_bandwidth_bytes_per_s / 1e9
    peak_bytes_per_s = args.peak_bandwidth_gbps * 1e9
    utilization = eff_bandwidth_bytes_per_s / peak_bytes_per_s if peak_bytes_per_s > 0 else 0.0
    forward_pass_per_s = forward_pass_count / decode_wall_s

    label = classify(utilization)

    selected_backend = getattr(runtime, "selected_backend", "unknown")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "model_id": args.model_id,
            "artifacts_dir": str(args.mlx_artifacts_dir),
            "weight_bytes_on_disk": weight_breakdown["total_disk_bytes"],
            "active_weight_bytes_per_forward_pass": weight_breakdown[
                "active_bytes_per_forward_pass"
            ],
            "is_moe": weight_breakdown["is_moe"],
            "moe_active_ratio": weight_breakdown["moe_active_ratio"],
            "moe_block": weight_breakdown["moe_block"],
            "weight_breakdown_source": weight_breakdown["breakdown_source"],
        },
        "host": {
            "platform": platform.platform(),
            "peak_bandwidth_gbps": args.peak_bandwidth_gbps,
            "peak_bandwidth_source": "user_provided",
        },
        "profile": {
            "prompt_tokens_requested": args.prompt_tokens,
            "prompt_tokens_used": len(prompt_tokens),
            "decode_tokens_requested": args.decode_tokens,
            "warmup_runs": args.warmup_runs,
            "seed": args.seed,
            "deterministic": True,
        },
        "measurements": {
            "decode_tokens_observed": decode_tokens_observed,
            "speculation_bonus_tokens": bonus_tokens,
            "forward_pass_count": forward_pass_count,
            "decode_wall_time_s": round(decode_wall_s, 6),
            "decode_tok_s": round(decode_tokens_observed / decode_wall_s, 3),
            "forward_pass_per_s": round(forward_pass_per_s, 3),
            "mean_forward_pass_us": round(mean_forward_pass_us, 3),
            "median_step_event_us": round(median_step_us, 3),
            "p95_step_event_us": round(p95_step_us, 3),
            "mean_runner_us": round(mean_runner_us, 3) if mean_runner_us is not None else None,
            "estimated_bytes_per_forward_pass": bytes_per_step,
            "estimated_effective_bandwidth_gbps": round(eff_bandwidth_gbps, 3),
            "bandwidth_utilization_ratio": round(utilization, 4),
            "mlx_dispatch_per_step": None,
            "mlx_dispatch_unavailable_reason": (
                "MLX-internal dispatch count is not surfaced to the Python "
                "extension; the metal_dispatch field on StepReport tracks AX "
                "custom kernels only. Confirm dispatch-bound classifications "
                "with Instruments before acting."
            ),
        },
        "classification": {
            "label": label,
            "kernel_label": kernel_classification,
            "rule_version": CLASSIFICATION_RULE_VERSION,
            "thresholds": {
                "bandwidth_bound_min": BANDWIDTH_BOUND_MIN,
                "dispatch_bound_max": DISPATCH_BOUND_MAX,
            },
            "regime_breakdown": {
                "single_decode": single_regime,
                "ngram_decode": ngram_regime,
            },
            "note": (
                "label classifies the mixed pipeline (user-observed). "
                "kernel_label classifies single-decode steps only (raw "
                "forward-pass rate without speculation-verification overhead). "
                "The kernel rate is what weight-quantization changes would "
                "attack; the mixed rate is what user-facing latency sees."
            ),
        },
        "route": {
            "selected_backend": selected_backend,
            "crossover_decisions": crossover_decisions,
        },
    }


def write_artifact(args: argparse.Namespace, artifact: dict) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    safe_id = args.model_id.replace("/", "_").replace(" ", "_")
    date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = f"-{args.run_tag}" if args.run_tag else ""
    path = args.output_root / f"{safe_id}-{date_part}{suffix}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    args = parse_args()
    artifact = run_profile(args)
    out = write_artifact(args, artifact)
    label = artifact["classification"]["label"]
    util = artifact["measurements"]["bandwidth_utilization_ratio"]
    tok_s = artifact["measurements"]["decode_tok_s"]
    print(f"wrote {out}")
    print(f"  classification: {label}")
    print(f"  bandwidth_utilization_ratio: {util}")
    print(f"  decode_tok_s: {tok_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
