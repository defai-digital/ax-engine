#!/usr/bin/env python3
"""Offline rotation tool for P2a / P2b under WEIGHT-ROTATION-IMPLEMENTATION-PLAN.md.

Phase 2a: produce a "rotated checkpoint" where `gate_proj` and `up_proj`
of each dense FFN layer have their input side multiplied by R. Combined
with the P1 activation-side rotation (already shipped in
`crates/ax-engine-mlx/src/weight_rotation.rs`), the net effect on the
forward pass is identity: `(W @ R) @ (R @ x) = W @ x`.

Phase 2b extends this with 3-bit / 2-bit re-quantisation of the rotated
weights and is gated on P2a passing the equivalence harness.

**This first revision is dry-run only.** It walks the model manifest,
identifies rotation candidates, computes the rotation seed for each, and
writes a rotation-plan JSON. Actually applying the rotation requires (a)
dequantising 4-bit weights via MLX, (b) rotating in f32, (c) re-writing
the safetensors with rotated f32 (or rotated 4-bit for P2b). That work
is staged behind a `--apply` flag that currently fails closed.

Artifact schema: ax.rotated_checkpoint_plan.v1

    {
      "schema_version": "ax.rotated_checkpoint_plan.v1",
      "generated_at_utc": "...",
      "source_model": { "model_id": str, "artifacts_dir": str },
      "rotation": {
        "scheme": "symmetric_orthonormal_hadamard",
        "seed_strategy": "fixed_per_dim",
        "seed_constant": "0xA5A5_A5A5_A5A5_A5A5"
      },
      "candidates": [
        {
          "tensor_name": str,
          "role": str,
          "layer_index": int | null,
          "shape": [int, ...],
          "rotation_axis": int,        # axis to rotate (input side: 1 for [out, in])
          "rotation_dim": int,         # must be power of 2
          "logical_dim_check": "matches | mismatch_packed_quant",
          "action": "would_rotate" | "skip_not_power_of_two" | "skip_role_excluded"
        }
      ],
      "summary": {
        "would_rotate_count": int,
        "skipped_count": int,
        "rotation_dim_histogram": { "<dim>": int }
      }
    }

Usage:

    python scripts/quantize_rotated_weights.py \\
        --mlx-artifacts-dir .internal/models/Qwen3.5-9B-MLX-4bit \\
        --output benchmarks/results/rotated-checkpoint-plan/qwen3_5_9b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.rotated_checkpoint_plan.v1"
ROTATION_SEED_CONSTANT = "0xA5A5_A5A5_A5A5_A5A5"  # matches Rust-side default

# Roles whose INPUT side gets rotated in P2a. Restricted to dense-FFN entries
# that match P1's activation rotation insertion point (start of ffn_swiglu).
# Attention projections, MoE expert variants, packed forms, and linear-
# attention tensors are out of scope for P2a per the plan §5.
P2A_ROTATION_ROLES = {"ffn_gate", "ffn_up"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mlx-artifacts-dir", required=True, type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path for the rotation plan. Default: print to stdout.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply the rotation: dequantise candidate weights via mlx, "
        "multiply by R, write a sibling `model.rotated.safetensors` with the "
        "rotated f32 weights. Also writes `model.rotated.json` with metadata.",
    )
    p.add_argument(
        "--max-tensors",
        type=int,
        default=None,
        help="Limit the number of tensors processed when --apply is set. "
        "Useful for first-run validation before processing the full model.",
    )
    p.add_argument(
        "--apply-output",
        type=Path,
        default=None,
        help="Where to write the rotated safetensors. Default: "
        "<mlx-artifacts-dir>/model.rotated.safetensors",
    )
    p.add_argument(
        "--bits",
        type=int,
        default=0,
        help="If 0 (default), store rotated weights as f32 (P2a baseline). "
        "If >0, re-quantize the rotated weight via mx.quantize at this bit "
        "width and store packed weight + scales + biases (P2b). Bits 2..8 "
        "are accepted; only 3 and 4 are validated in the engine path.",
    )
    p.add_argument(
        "--smoothing",
        choices=["none", "weight_mag", "activation_max"],
        default="none",
        help="P2b §3a (`weight_mag`) or §3b (`activation_max`) AWQ-lite "
        "smoothing. `weight_mag` uses combined gate/up weight magnitude as a "
        "proxy — empirically unreliable. `activation_max` reads per-layer "
        "activation magnitudes from `--activation-stats` (captured by "
        "scripts/capture_ffn_activations.py) and is the AWQ paper's recipe.",
    )
    p.add_argument(
        "--activation-stats",
        type=Path,
        default=None,
        help="Required when `--smoothing activation_max`. Path to a "
        "safetensors file written by scripts/capture_ffn_activations.py "
        "containing `activation_max.layer.{i}` per-layer vectors.",
    )
    p.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.5,
        help="Exponent for the smoothing scale (default 0.5 = sqrt of mag). "
        "Higher alpha pushes more quantisation difficulty onto activations.",
    )
    return p.parse_args()


def logical_input_dim(role: str, shape: list[int]) -> tuple[int, int, str]:
    """Return (rotation_axis, logical_dim, axis_status) for an input-side rotation.

    Quantised 4-bit weights are stored as u32-packed shapes; the physical
    inner dim is `logical / 8`. For `gate_proj` / `up_proj` the manifest
    shape is [out, in_packed], so the rotation axis (input side) is axis 1
    and the logical input dim is `in_packed * 8` for u32-packed 4-bit.
    """
    if role in P2A_ROTATION_ROLES:
        # Input side is the last axis. For 4-bit u32-packed weights the
        # physical last-axis dim is logical/8. Detect this by checking if
        # the physical dim × 8 is power-of-2.
        physical = shape[-1]
        logical_candidate = physical * 8
        if logical_candidate >= 2 and (logical_candidate & (logical_candidate - 1)) == 0:
            return (len(shape) - 1, logical_candidate, "packed_4bit_logical_match")
        if physical >= 2 and (physical & (physical - 1)) == 0:
            return (len(shape) - 1, physical, "unpacked_logical_match")
        return (len(shape) - 1, physical, "mismatch_packed_quant")
    raise ValueError(f"unsupported role for P2a: {role}")


def build_plan(artifacts_dir: Path) -> dict:
    manifest_path = artifacts_dir / "model-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"model-manifest.json not found under {artifacts_dir}")
    manifest = json.loads(manifest_path.read_text())
    tensors = manifest.get("tensors", [])
    if not tensors:
        raise SystemExit("manifest has no tensors list")

    candidates = []
    rotation_dim_histogram: dict[int, int] = {}
    would_rotate = 0
    skipped = 0
    for t in tensors:
        role = t.get("role", "")
        if role not in P2A_ROTATION_ROLES:
            continue
        name = t["name"]
        shape = list(t.get("shape", []))
        layer_index = t.get("layer_index")
        try:
            axis, logical_dim, status = logical_input_dim(role, shape)
        except ValueError:
            continue
        is_pow2 = logical_dim >= 2 and (logical_dim & (logical_dim - 1)) == 0
        if is_pow2:
            action = "would_rotate"
            would_rotate += 1
            rotation_dim_histogram[logical_dim] = (
                rotation_dim_histogram.get(logical_dim, 0) + 1
            )
        else:
            action = "skip_not_power_of_two"
            skipped += 1
        candidates.append(
            {
                "tensor_name": name,
                "role": role,
                "layer_index": layer_index,
                "shape": shape,
                "rotation_axis": axis,
                "rotation_dim": logical_dim,
                "logical_dim_check": status,
                "action": action,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_model": {
            "model_id": manifest.get("model_family", "unknown"),
            "artifacts_dir": str(artifacts_dir),
        },
        "rotation": {
            "scheme": "symmetric_orthonormal_hadamard",
            "seed_strategy": "fixed_per_dim",
            "seed_constant": ROTATION_SEED_CONSTANT,
        },
        "candidates": candidates,
        "summary": {
            "would_rotate_count": would_rotate,
            "skipped_count": skipped,
            "rotation_dim_histogram": {str(k): v for k, v in sorted(rotation_dim_histogram.items())},
        },
    }


def build_orthonormal_hadamard(dim: int, seed: int) -> "np.ndarray":
    """Symmetric orthonormal randomised Hadamard matrix (R · R = I).

    Matches the Rust `build_rotation_matrix` in
    `crates/ax-engine-mlx/src/weight_rotation.rs` so offline-rotated weights
    cancel exactly against the runtime activation rotation when both are
    seeded with the same constant.
    """
    import numpy as np

    assert dim >= 2 and (dim & (dim - 1)) == 0, "dim must be power-of-2 >= 2"
    # Same xorshift64 PRNG as `generate_sign_flip` in the Rust module.
    state = seed if seed != 0 else 1
    sign = np.empty(dim, dtype=np.float32)
    for i in range(dim):
        state = (state ^ ((state << 13) & 0xFFFF_FFFF_FFFF_FFFF)) & 0xFFFF_FFFF_FFFF_FFFF
        state = (state ^ (state >> 7)) & 0xFFFF_FFFF_FFFF_FFFF
        state = (state ^ ((state << 17) & 0xFFFF_FFFF_FFFF_FFFF)) & 0xFFFF_FFFF_FFFF_FFFF
        sign[i] = 1.0 if (state & 1) == 0 else -1.0
    scale = 1.0 / np.sqrt(np.float32(dim))
    rows = np.arange(dim).reshape(-1, 1)
    cols = np.arange(dim).reshape(1, -1)
    popcount_parity = (np.bitwise_count(rows & cols) % 2).astype(np.float32)
    h = np.where(popcount_parity == 0, 1.0, -1.0).astype(np.float32)
    r = (sign.reshape(-1, 1) * h * sign.reshape(1, -1)) * scale
    return r


def apply_rotation(args: argparse.Namespace, plan: dict) -> dict:
    import numpy as np
    import mlx.core as mx

    candidates = [c for c in plan["candidates"] if c["action"] == "would_rotate"]
    if args.max_tensors is not None:
        candidates = candidates[: args.max_tensors]
    if not candidates:
        raise SystemExit("no rotation candidates to apply")

    # Locate per-tensor file path from the manifest (need original tensor file paths).
    manifest = json.loads((args.mlx_artifacts_dir / "model-manifest.json").read_text())
    name_to_file: dict[str, str] = {t["name"]: t["file"] for t in manifest["tensors"]}

    # Group by source file so each safetensors is opened at most once.
    by_file: dict[str, list[dict]] = {}
    for c in candidates:
        f = name_to_file[c["tensor_name"]]
        by_file.setdefault(f, []).append(c)

    rotation_cache: dict[int, "mx.array"] = {}
    rotated: dict[str, "mx.array"] = {}
    per_tensor_log: list[dict] = []
    seed_int = int(ROTATION_SEED_CONSTANT.replace("_", ""), 16)
    requantize_bits = args.bits if args.bits > 0 else None
    if requantize_bits is not None and not (2 <= requantize_bits <= 8):
        raise SystemExit(f"--bits must be in 2..8, got {requantize_bits}")
    smoothing_mode = args.smoothing
    smoothing_alpha = float(args.smoothing_alpha)

    # Load activation stats once if needed.
    activation_stats: dict[int, "mx.array"] = {}
    if smoothing_mode == "activation_max":
        if args.activation_stats is None:
            raise SystemExit("--smoothing activation_max requires --activation-stats")
        if not args.activation_stats.is_file():
            raise SystemExit(f"--activation-stats file not found: {args.activation_stats}")
        loaded_stats = mx.load(str(args.activation_stats))
        for key, val in loaded_stats.items():
            if key.startswith("activation_max.layer."):
                idx = int(key.split(".")[-1])
                activation_stats[idx] = val.astype(mx.float32)
        if not activation_stats:
            raise SystemExit(
                f"no `activation_max.layer.*` tensors in {args.activation_stats}"
            )
        print(f"loaded activation stats for {len(activation_stats)} layers")

    # Phase 1: rotate all candidate weights to f32 and keep them in memory
    # keyed by (layer_index, role). Phase 2 then performs per-layer smoothing
    # (which needs gate+up at the same time) and per-tensor quantisation.
    # Memory: 64 f32 tensors @ ~200MB each = ~13GB on Qwen3.5-9B. OK on the
    # 128 GB host.
    by_layer_role: dict[tuple[int, str], dict] = {}

    # PHASE 1: dequantise + rotate every candidate, store f32 result indexed
    # by (layer_index, role).
    for file_rel, cs in by_file.items():
        full = args.mlx_artifacts_dir / file_rel
        print(f"phase 1 opening {file_rel} for {len(cs)} candidates...")
        loaded = mx.load(str(full))
        file_keys = set(loaded.keys())
        for c in cs:
            name = c["tensor_name"]
            stem = name[: -len(".weight")] if name.endswith(".weight") else name
            scales_key = f"{stem}.scales"
            biases_key = f"{stem}.biases"
            w_packed = loaded[name]
            scales = loaded.get(scales_key)
            biases = loaded.get(biases_key)
            if scales is None:
                raise SystemExit(
                    f"missing scales for {name} (looked for {scales_key}); "
                    "cannot dequantise"
                )
            dim = c["rotation_dim"]
            w_de = mx.dequantize(
                w_packed,
                scales=scales,
                biases=biases,
                group_size=64,
                bits=4,
                mode="affine",
            )
            w_de_f32 = w_de.astype(mx.float32)
            R = rotation_cache.setdefault(
                dim,
                mx.array(build_orthonormal_hadamard(dim, seed_int)),
            )
            w_rotated_f32 = mx.matmul(w_de_f32, R)
            mx.eval(w_rotated_f32)
            shape = list(w_rotated_f32.shape)
            if shape[-1] != dim:
                raise SystemExit(
                    f"{name}: expected last dim {dim}, got rotated shape {shape}"
                )
            key = (int(c["layer_index"]), c["role"])
            if key in by_layer_role:
                raise SystemExit(
                    f"duplicate (layer, role) {key} during phase 1 — manifest invariant violated"
                )
            by_layer_role[key] = {
                "name": name,
                "w_rotated_f32": w_rotated_f32,
                "dim": dim,
                "shape": shape,
            }

    # PHASE 2: per-layer smoothing (if requested) + per-tensor quantisation.
    layer_indices = sorted({k[0] for k in by_layer_role})
    print(
        f"phase 2: processing {len(layer_indices)} layers, smoothing={smoothing_mode}, "
        f"alpha={smoothing_alpha if smoothing_mode != 'none' else 'n/a'}"
    )
    smoothing_log: list[dict] = []
    for layer_idx in layer_indices:
        gate_entry = by_layer_role.get((layer_idx, "ffn_gate"))
        up_entry = by_layer_role.get((layer_idx, "ffn_up"))
        s_inverse = None
        s_value = None
        s_source: str | None = None
        s_basis: "mx.array" | None = None
        if smoothing_mode == "weight_mag" and gate_entry and up_entry:
            gate_w = gate_entry["w_rotated_f32"]
            up_w = up_entry["w_rotated_f32"]
            gate_mag = mx.sqrt(mx.mean(gate_w * gate_w, axis=0) + 1e-12)
            up_mag = mx.sqrt(mx.mean(up_w * up_w, axis=0) + 1e-12)
            s_basis = mx.sqrt(gate_mag * up_mag + 1e-12)
            s_source = "weight_mag"
        elif smoothing_mode == "activation_max" and gate_entry and up_entry:
            # AWQ paper recipe: per-input-channel max(|x|) from a calibration
            # corpus. Pass `--rotation-dim` to capture_ffn_activations.py so
            # the stat is captured AFTER rotation (matches the runtime's
            # Apply-mode forward). Naively transforming the un-rotated stat
            # by |R| collapses to a constant (orthonormal Hadamard has
            # uniform |entries|), which is why we capture rotated stats
            # rather than transform them here.
            act = activation_stats.get(layer_idx)
            if act is None:
                raise SystemExit(
                    f"missing activation stats for layer {layer_idx}; "
                    "check that the capture covered every dense FFN layer."
                )
            s_basis = act + 1e-12
            s_source = "activation_max_captured"
        if s_basis is not None:
            # Canonical AWQ direction: W_new = W / s, x_new = x * s.
            # The Rust field `rotation_smoothing_inverse` is misnamed for
            # historical reasons — it's actually the runtime multiplier
            # applied to the activation, so we store `s` (the activation
            # amplifier) here, NOT `1/s`. W is divided by s offline so
            # outlier-activation columns of W become smaller and quantise
            # cleaner.
            s = mx.power(s_basis, smoothing_alpha)
            s_mean = mx.mean(s)
            s = s / mx.maximum(s_mean, mx.array(1e-12))
            mx.eval(s)
            # Divide weights by s (outlier channels of W shrink → better quant).
            gate_entry["w_rotated_f32"] = gate_entry["w_rotated_f32"] / s
            up_entry["w_rotated_f32"] = up_entry["w_rotated_f32"] / s
            mx.eval(gate_entry["w_rotated_f32"], up_entry["w_rotated_f32"])
            # The runtime multiplier (stored under `rotation_smoothing_inverse`
            # in safetensors for naming continuity) IS s, not 1/s.
            s_inverse = s
            s_value = s
            s_value = s
            smoothing_log.append(
                {
                    "layer_index": layer_idx,
                    "s_source": s_source,
                    "s_mean": float(mx.mean(s).item()),
                    "s_max": float(mx.max(s).item()),
                    "s_min": float(mx.min(s).item()),
                    "s_inverse_mean": float(mx.mean(s_inverse).item()),
                }
            )

        for role_name in ("ffn_gate", "ffn_up"):
            entry = by_layer_role.get((layer_idx, role_name))
            if entry is None:
                continue
            w = entry["w_rotated_f32"]
            name = entry["name"]
            log_entry: dict[str, object] = {
                "tensor_name": name,
                "layer_index": layer_idx,
                "role": role_name,
                "dequantised_shape": entry["shape"],
                "rotation_dim": entry["dim"],
                "smoothing": smoothing_mode,
            }
            if requantize_bits is None:
                rotated[name] = w.astype(mx.float32)
                log_entry["storage"] = "f32"
            else:
                wq, sq, bq = mx.quantize(
                    w,
                    group_size=64,
                    bits=requantize_bits,
                    mode="affine",
                )
                stem_q = name[: -len(".weight")] if name.endswith(".weight") else name
                rotated[name] = wq
                rotated[f"{stem_q}.scales"] = sq
                rotated[f"{stem_q}.biases"] = bq
                log_entry["storage"] = f"quantized_bits{requantize_bits}_g64_affine"
                log_entry["packed_shape"] = list(wq.shape)
            per_tensor_log.append(log_entry)
            print(f"  layer {layer_idx} {role_name}: stored ({log_entry['storage']})")

        if s_inverse is not None:
            # Store as bf16 to match the model's working dtype at runtime (the
            # Rust loader doesn't need to cast).
            rotated[f"ax_smoothing.layers.{layer_idx}"] = s_inverse.astype(mx.bfloat16)

    out_path = args.apply_output or (args.mlx_artifacts_dir / "model.rotated.safetensors")
    metadata = {
        "schema_version": "ax.rotated_checkpoint.v1",
        "rotation_scheme": "symmetric_orthonormal_hadamard",
        "rotation_seed": ROTATION_SEED_CONSTANT,
        "rotation_axis_convention": "input_side_last_axis",
        "source_manifest_layer_count": str(manifest.get("layer_count", "")),
        "rotated_storage": "f32" if requantize_bits is None else f"quantized_bits{requantize_bits}_g64_affine",
        "rotated_bits": str(requantize_bits) if requantize_bits is not None else "0",
        "rotated_group_size": "64",
        "smoothing": smoothing_mode,
        "smoothing_alpha": str(smoothing_alpha) if smoothing_mode != "none" else "",
    }
    mx.save_safetensors(str(out_path), rotated, metadata=metadata)
    print(f"\nwrote {out_path} ({len(rotated)} tensors)")
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "ax.rotated_checkpoint.v1",
                "generated_at_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "source_model": {
                    "model_id": manifest.get("model_family", "unknown"),
                    "artifacts_dir": str(args.mlx_artifacts_dir),
                },
                "rotation_metadata": metadata,
                "tensors": per_tensor_log,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {sidecar}")
    return {"rotated_count": len(rotated), "sidecar": str(sidecar)}


def main() -> int:
    args = parse_args()
    plan = build_plan(args.mlx_artifacts_dir)
    output_json = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
        print(f"wrote {args.output}")
    s = plan["summary"]
    print(
        f"  would_rotate={s['would_rotate_count']} skipped={s['skipped_count']} "
        f"dims={s['rotation_dim_histogram']}"
    )
    if args.apply:
        print("\napplying rotation...")
        apply_rotation(args, plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
