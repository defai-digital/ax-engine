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
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

FAMILY = "qwen4_exp"
HOST_CLASS = "MacBook Pro M5 Max, 128 GB"
PRIMARY_ALIAS = None
AFFINE_ALIAS = "qwen3.8-flash-next:axq"
AFFINE_REPO = "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP"
PRIMARY_REPO = "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP"
PACK_REVISION = "0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35"
SIXBIT_ALIAS = "qwen3.8-flash-next:axq-6bit"
SIXBIT_REPO = "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP"
SOURCE_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
STATUS = (
    "Second SKU. MXFP4 MTP target; native support and checkpoint qualification pending. MTP Tier 2 pending. "
    "AX certification record: Candidate (gates open)."
)
EXISTING_AFFINE_EXPERT_LAYOUTS = (
    {"bits": 4, "group_size": 64},
    {"bits": 6, "group_size": 64},
)
EXPERIMENTAL_EXPERT_LAYOUTS = (
    {"mode": "affine", "bits": 2, "group_size": 32},
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
            AFFINE_ALIAS,
            SIXBIT_ALIAS,
        ],
        "repo_id": PRIMARY_REPO,
        "pack_revision": PACK_REVISION,
        "target_quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
        "mixed_tensor_overrides": (
            "affine8/group32 embedding/router overrides and affine8/group64 output head; "
            "preserve checkpoint metadata"
        ),
        "sixbit_in_target_scope": False,
        "existing_affine_alias": AFFINE_ALIAS,
        "existing_affine_repo_id": AFFINE_REPO,
        "sixbit_alias": SIXBIT_ALIAS,
        "sixbit_repo_id": SIXBIT_REPO,
        "source_revision": SOURCE_REVISION,
        "host_class": HOST_CLASS,
        "status": STATUS,
        "ci": "dry-run only; do not mount Flash Next weights on CI",
        "fail_closed": True,
        "ready": False,
        "readiness_scope": "MXFP4 target only; existing affine loading is unchanged",
        "qualification_storage": "NAS-backed Hugging Face cache; record SMB conditions",
        "release_ready": False,
        "qualification": False,
        "validation_scope": "manifest metadata only; native loader validation required",
        "convert": "metadata mapping; existing audited affine readiness is not MXFP4 readiness",
        "load_blocker": "MXFP4 requires AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1; target qualification pending",
        "unknown_layout_blocker": "qwen4_exp_weight_layout_unknown",
        "existing_affine_expert_layouts": list(EXISTING_AFFINE_EXPERT_LAYOUTS),
        "experimental_opt_in": "AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1",
        "experimental_scope": (
            "MXFP4/group32 diagnostic paging; audited affine 2-bit/group32 also needs "
            "AX_ENGINE_2BIT_EXPERIMENTAL=1. Neither has target qualification."
        ),
        "experimental_expert_layouts": list(EXPERIMENTAL_EXPERT_LAYOUTS),
        "experimental_2bit_opt_in": "AX_ENGINE_2BIT_EXPERIMENTAL=1",
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
        "mtp_certification": {
            gate: "not_assessed" for gate in ("MTP-S", "MTP-P", "MTP-D")
        },
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
    print("target admission: MXFP4/group32 requires AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1")
    print("existing affine admission: 4-bit/group64 and 6-bit/group64; not target qualification")
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
            raise SystemExit(f"expert tensor {tensor.get('name')} has invalid quantization metadata")
        layout = (quant["mode"], bits, group)
        if layout not in layouts:
            layouts.append(layout)
    return layouts


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
        raise SystemExit(
            f"model_family must be {FAMILY}, got {manifest.get('model_family')!r}"
        )
    status = manifest.get("runtime_status") or {}
    if not isinstance(status, dict):
        raise SystemExit("runtime_status must be an object")
    blockers = status.get("blockers") or []
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
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise SystemExit("layer_count must be > 0")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise SystemExit("hidden_size must be > 0")
    layouts = _expert_layouts(manifest)
    if len(layouts) > 1:
        raise SystemExit(f"mixed expert layouts are rejected: {layouts}")
    allowed = {("affine", item["bits"], item["group_size"]) for item in EXISTING_AFFINE_EXPERT_LAYOUTS}
    allowed.update(
        (item["mode"], item["bits"], item["group_size"]) for item in EXPERIMENTAL_EXPERT_LAYOUTS
    )
    if not layouts:
        raise SystemExit("manifest has no quantized expert tensors")
    if layouts[0] == ("affine", 2, 32) and not all(
        os.environ.get(name) == "1"
        for name in ("AX_ENGINE_FLASH_NEXT_EXPERIMENTAL", "AX_ENGINE_2BIT_EXPERIMENTAL")
    ):
        raise SystemExit("2-bit requires both experimental opt-ins")
    if layouts[0][0] == "mxfp4" and os.environ.get("AX_ENGINE_FLASH_NEXT_EXPERIMENTAL") != "1":
        raise SystemExit("MXFP4 requires AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1")
    if layouts[0] not in allowed:
        raise SystemExit(f"unsupported expert layout {layouts[0]}")
    print(f"live preflight ok: {model_dir}")
    print(f"family qwen4_exp; declared ready and {layouts[0]} layout; metadata preflight only")
    print("this metadata preflight does not qualify the MXFP4 MTP target")
    print("next: ax-engine doctor, then QA surface direct+mtp on this snapshot")
    print("this script does not start the Flash Next server (operator-owned live run)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.model_dir is not None and not args.dry_run:
        _live_preflight(args.model_dir)
        if args.json:
            _print_contract(True)
        return 0
    _print_contract(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
