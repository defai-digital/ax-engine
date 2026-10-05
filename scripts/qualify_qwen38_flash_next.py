#!/usr/bin/env python3
"""Qwen 3.8 Flash Next Candidate qualification contract.

`--dry-run` prints the SKU and admission rule (CI-safe, no weights).
A live `--model-dir` run checks declared manifest fields, expert quantization
metadata, and recorded readiness without loading weights.
This metadata preflight does not validate files, tensor geometry, exporter identity,
or release qualification; the native loader remains authoritative.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import struct
import math
from pathlib import Path
from typing import Any, Sequence

FAMILY = "qwen4_exp"
HOST_CLASS = "Mac Studio, Ultra-class Apple Silicon (M2 Ultra or newer), 192 GB+"
PRIMARY_ALIAS = "qwen3.8-flash-next:mxfp4"
PRIMARY_REPO = "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP"
PACK_REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
SOURCE_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
STATUS = (
    "Second SKU. MXFP4 MTP target; native functional controls verified; checkpoint qualification pending. MTP Tier 2 pending. "
    "AX certification record: Candidate (gates open)."
)
EXPERIMENTAL_EXPERT_LAYOUTS = (
    {"mode": "mxfp4", "bits": 4, "group_size": 32},
)
EXPERT_ROLES = {
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_up_exps_packed",
}


def contract() -> dict[str, Any]:
    return {
        "family": FAMILY,
        "alias": PRIMARY_ALIAS,
        "aliases": [
            "qwen4_exp",
            "qwen3.8-flash-next",
            "qwen38_flash_next",
            "qwen3.8-flash-next:axq",
            PRIMARY_ALIAS,
        ],
        "repo_id": PRIMARY_REPO,
        "pack_revision": PACK_REVISION,
        "target_quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
        "mixed_tensor_overrides": (
            "affine8/group32 embedding/router overrides and affine8/group64 output head; "
            "preserve checkpoint metadata"
        ),
        "sixbit_in_target_scope": False,
        "source_revision": SOURCE_REVISION,
        "host_class": HOST_CLASS,
        "status": STATUS,
        "ci": "dry-run only; do not mount Flash Next weights on CI",
        "fail_closed": True,
        "ready": False,
        "readiness_scope": "MLX AXQ MXFP4/MXFP8 MTP only; standalone affine packs retired",
        "format_scope": ["mxfp4", "mxfp8"],
        "mtp_sidecar_required": True,
        "mxfp8_status": "in scope; audited pack pin and dedicated native-path validation pending",
        "qualification_storage": "record the actual storage medium and connection; no cross-storage inference",
        "release_ready": False,
        "qualification": False,
        "validation_scope": "manifest and sidecar header metadata only; native loader validation required",
        "convert": "metadata mapping; retired affine readiness is not MXFP4 readiness",
        "load_blocker": "none: MXFP4/group32 loads without an env var; target qualification pending",
        "unknown_layout_blocker": "qwen4_exp_weight_layout_unknown",
        "experimental_opt_in": "none for MXFP4; Candidate status and MTP-S/P/D gates remain open",
        "experimental_scope": "audited MXFP4 MTP only; standalone affine packs rejected even with opt-ins",
        "experimental_expert_layouts": list(EXPERIMENTAL_EXPERT_LAYOUTS),
        "mtp": (
            "sidecar attaches when mtp.safetensors is present; "
            "certified_default_on remains false; MTP-S, MTP-P and MTP-D require separate evidence"
        ),
        "mtp_gates": {
            "MTP-S": (
                "Shipping safety: every accepted draft equals the same-state verifier greedy "
                "decision, with zero invalid acceptances; not assessed by cross-route parity"
            ),
            "MTP-P": (
                "Scoped acceleration claims: weighted >= 1.20x and prompt-median >= 1.10x "
                "on two named authorizing workloads, a negative control, full source/build/"
                "pack/SKU/runtime evidence and divergence indices/logit-margin disclosures; "
                "passing does not change defaults"
            ),
            "MTP-D": (
                "Separate promotion decision and release tag after MTP-S/P; default-product-"
                "path greedy parity with shared deterministic tie-breaking, quality, endurance "
                "and long-context decode-at-depth; diagnostic arithmetic profiles do not qualify"
            ),
        },
        "mtp_certification": {gate: "not_assessed" for gate in ("MTP-S", "MTP-P", "MTP-D")},
        "diagnostic_only": [
            "Independent direct/MTP token differences must be disclosed; they alone neither "
            "fail nor establish MTP-S. A near-tie explanation requires measured logits.",
        ],
        "not": [
            "qwen3.8-27b:axq",
            "qwen3_5",
            "Super-class Qwen 3.8 2.4T",
        ],
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _print_contract(as_json: bool) -> None:
    payload = contract()
    if as_json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    print(payload["status"])
    print(f"family: {payload['family']}")
    print(f"alias:  {payload['alias'] or 'not registered for the MXFP4 target'}")
    print(f"repo:   {payload['repo_id']}")
    print(f"host:   {payload['host_class']}")
    print("target admission: MXFP4/group32 loads without an env var; certification stays Candidate (gates open)")
    print("standalone affine packs retired; MXFP8 pack audit and admission pending")
    print("fail-closed: convert must not remap onto qwen3_5")
    print("MTP gates (metadata preflight assesses none):")
    for gate, requirement in payload["mtp_gates"].items():
        print(f"  {gate} [{payload['mtp_certification'][gate]}]: {requirement}")


def _expert_layouts(manifest: dict[str, Any]) -> list[tuple[str, int, int]]:
    layouts: list[tuple[str, int, int]] = []
    for tensor in manifest.get("tensors") or []:
        if not isinstance(tensor, dict):
            continue
        if tensor.get("role") not in EXPERT_ROLES:
            continue
        quant = tensor.get("quantization")
        if not isinstance(quant, dict):
            raise SystemExit(f"expert tensor {tensor.get('name')} has no quantization metadata")
        if quant.get("mode") not in ("affine", "mxfp4"):
            raise SystemExit(
                f"expert tensor {tensor.get('name')} uses {quant.get('mode')!r}; "
                "only affine and MXFP4 expert formats are recognized"
            )
        bits = quant.get("bits")
        group = quant.get("group_size")
        if type(bits) is not int or type(group) is not int:
            raise SystemExit(
                f"expert tensor {tensor.get('name')} has invalid quantization metadata"
            )
        layout = (quant["mode"], bits, group)
        if layout not in layouts:
            layouts.append(layout)
    return layouts


def _mtp_metadata(model_dir: Path) -> dict[str, Any]:
    """Inspect bounded sidecar metadata without loading its 5 GB payload."""
    runtime = model_dir / "mtplx_runtime.json"
    sidecar = model_dir / "mtp.safetensors"
    if not runtime.is_file() or not sidecar.is_file():
        raise SystemExit("MXFP4 MTP requires mtp.safetensors and mtplx_runtime.json")
    try:
        declaration = json.loads(runtime.read_text(encoding="utf-8"))
        if not isinstance(declaration, dict) or declaration.get("mtp_norm_layout") not in (
            "raw_hf_delta", "mlx_multiplier"
        ):
            raise ValueError("mtp_norm_layout must be raw_hf_delta or mlx_multiplier")
        with sidecar.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("truncated sidecar header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 0 < header_length <= 64 * 1024 * 1024:
                raise ValueError("sidecar header exceeds the bounded metadata contract")
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                raise ValueError("truncated sidecar header")
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError("sidecar header must be an object")
        tensors = {name: info for name, info in header.items() if name != "__metadata__"}
        if len(tensors) != 31:
            raise ValueError("Flash Next MTP requires 31 sidecar tensors")
        payload_bytes = sidecar.stat().st_size - 8 - header_length
        spans = []
        for name, info in tensors.items():
            if not name.startswith("mtp.") or not isinstance(info, dict):
                raise ValueError(f"invalid sidecar tensor {name}")
            width = {"BF16": 2, "F16": 2, "F32": 4}.get(info.get("dtype"))
            shape, offsets = info.get("shape"), info.get("data_offsets")
            if (width is None or not isinstance(shape, list) or not shape
                    or any(type(dim) is not int or dim <= 0 for dim in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)):
                raise ValueError(f"unsupported sidecar tensor metadata: {name}")
            start, end = offsets
            if not 0 <= start < end <= payload_bytes or end - start != math.prod(shape) * width:
                raise ValueError(f"invalid sidecar payload range: {name}")
            spans.append((start, end))
        position = 0
        for start, end in sorted(spans):
            if start != position:
                raise ValueError("sidecar payload ranges overlap or have gaps")
            position = end
        if position != payload_bytes:
            raise ValueError("sidecar payload has unbound trailing bytes")
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"invalid Flash Next MTP metadata: {error}") from error
    return {"tensor_count": len(tensors), "mtp_norm_layout": declaration["mtp_norm_layout"],
            "payload_bytes": payload_bytes, "scope": "header metadata; native names/geometry checks required"}


def _live_preflight(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(f"model dir is not a directory: {model_dir}")
    config = model_dir / "config.json"
    if not config.is_file():
        raise SystemExit(f"missing config.json under {model_dir}")
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing model-manifest.json under {model_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid model-manifest.json: {error}") from error
    if not isinstance(manifest, dict):
        raise SystemExit("model-manifest.json must be an object")
    if manifest.get("model_family") != FAMILY:
        raise SystemExit(f"model_family must be {FAMILY}, got {manifest.get('model_family')!r}")
    status = manifest.get("runtime_status") or {}
    if not isinstance(status, dict):
        raise SystemExit("runtime_status must be an object")
    blockers = status.get("blockers", [])
    if not isinstance(blockers, list):
        raise SystemExit("runtime_status.blockers must be a list")
    if blockers:
        raise SystemExit(f"manifest is blocked: {blockers!r}")
    if status.get("ready") is not True:
        raise SystemExit(
            f"native model manifest is not runtime ready: ready={status.get('ready')} "
            f"blockers={blockers!r}"
        )
    tensors = manifest.get("tensors") or []
    if not isinstance(tensors, list) or not tensors:
        raise SystemExit("manifest tensors must be a nonempty list")
    layer_count = manifest.get("layer_count")
    hidden_size = manifest.get("hidden_size")
    if type(layer_count) is not int or layer_count <= 0:
        raise SystemExit("layer_count must be > 0")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise SystemExit("hidden_size must be > 0")
    layouts = _expert_layouts(manifest)
    if len(layouts) > 1:
        raise SystemExit(f"mixed expert layouts are rejected: {layouts}")
    allowed = {
        (item["mode"], item["bits"], item["group_size"]) for item in EXPERIMENTAL_EXPERT_LAYOUTS
    }
    if not layouts:
        raise SystemExit("manifest has no quantized expert tensors")
    if layouts[0] not in allowed:
        raise SystemExit(f"unsupported expert layout {layouts[0]}; standalone affine packs retired, MXFP8 admission pending")
    sidecar = _mtp_metadata(model_dir)
    print(f"MTP sidecar metadata: {sidecar['tensor_count']} tensors; {sidecar['mtp_norm_layout']}")
    print("MXFP4/group32 preflight: no experimental opt-in required")
    print(f"live preflight ok: {model_dir}")
    print(f"family qwen4_exp; declared ready and {layouts[0]} layout; metadata preflight only")
    print("this metadata preflight does not qualify the MXFP4 MTP target")
    print("next: ax-engine doctor, then QA surface direct+mtp on this snapshot")
    print("this script does not start the Flash Next server (operator-owned live run)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.model_dir is not None and not args.dry_run:
        # With --json, stdout must stay a single JSON document; the human
        # preflight lines go to stderr.
        with contextlib.redirect_stdout(sys.stderr if args.json else sys.stdout):
            _live_preflight(args.model_dir)
        if args.json:
            _print_contract(True)
        return 0
    _print_contract(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
