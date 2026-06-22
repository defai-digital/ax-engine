#!/usr/bin/env python3
"""Probe whether a downloaded MLX model is ready for repo-owned AX support.

This is a support-contract probe, not an inference benchmark. It reads the
model's config and safetensors index, then cross-checks local reference
implementations so unsupported architectures fail closed with named blockers.

Native-family policy is governed by ADR 0023
(`.internal/adr/0023-deepseek-v4-delegated-ds4-route.md`). The Qwen / Gemma
`REPO_OWNED_TYPES` allowlist below plus the dedicated `probe_glm4_moe_lite`
path implement D1 of that ADR; `deepseek_v4` is rejected per D4. Models
outside the native list are expected to route through `mlx_lm_delegated`
(ADR 0014, ADR 0023 D2) at the SDK layer, not promoted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ax.mlx_model_support_probe.v1"

REPO_OWNED_TYPES = {
    "qwen3",
    "qwen3_5",
    "qwen3.5",
    "qwen3_5_moe",
    "qwen3_5_text",
    "qwen3_next",
    "qwen3_6",
    "qwen3.6",
    "qwen3_moe",
    "gemma4",
    "gemma4_unified",
    "gemma4_unified_text",
    "diffusion_gemma",
    "gpt_oss",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _weight_keys(model_dir: Path) -> list[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return []
    return sorted(str(key) for key in weight_map)


def _has_any(keys: list[str], needle: str) -> bool:
    return any(needle in key for key in keys)


def _file_probe(path: Path, markers: list[str] | None = None) -> dict[str, Any]:
    exists = path.exists()
    result: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT) if path.is_absolute() else path),
        "exists": exists,
    }
    if exists and markers:
        text = path.read_text(errors="replace")
        result["markers"] = {marker: marker in text for marker in markers}
    return result


def probe_glm4_moe_lite(model_dir: Path, keys: list[str]) -> dict[str, Any]:
    manifest_path = model_dir / "model-manifest.json"
    manifest: dict[str, Any] = {}
    manifest_ready: bool | None = None
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            runtime_status = manifest.get("runtime_status")
            if isinstance(runtime_status, dict):
                manifest_ready = bool(runtime_status.get("ready"))
            else:
                manifest_ready = True
        except Exception:
            manifest_ready = False

    mlx_lm_ref = REPO_ROOT / ".internal/reference/mlx-lm/mlx_lm/models/glm4_moe_lite.py"
    swift_ref = (
        REPO_ROOT
        / ".internal/reference/mlx-swift-lm/Libraries/MLXLLM/Models/GLM4MOELite.swift"
    )
    reference_files = [
        _file_probe(
            mlx_lm_ref,
            ["q_a_proj", "kv_a_proj_with_mqa", "e_score_correction_bias"],
        ),
        _file_probe(
            swift_ref,
            ["GLM4MoELiteAttention", "GLM4MoELiteGate", "eScoreCorrectionBias"],
        ),
    ]
    features = {
        "mla_q_a_projection": _has_any(keys, ".self_attn.q_a_proj.weight"),
        "mla_q_b_projection": _has_any(keys, ".self_attn.q_b_proj.weight"),
        "mla_kv_a_projection": _has_any(keys, ".self_attn.kv_a_proj_with_mqa.weight"),
        "mla_latent_k_embedding": _has_any(keys, ".self_attn.embed_q.weight"),
        "mla_latent_v_unembedding": _has_any(keys, ".self_attn.unembed_out.weight"),
        "sigmoid_router_correction_bias": _has_any(keys, ".mlp.gate.e_score_correction_bias"),
        "routed_expert_stack": _has_any(keys, ".mlp.experts.gate_up_proj.weight")
        or _has_any(keys, ".mlp.switch_mlp.gate_proj.weight"),
    }
    reference_ready = all(item["exists"] for item in reference_files) and all(features.values())

    blockers = []
    if manifest_ready is None:
        blockers.append("AX model-manifest.json is absent for this artifact")
    elif not manifest_ready:
        blockers.append("AX model-manifest.json is not runtime-ready")
    if not reference_ready:
        blockers.insert(0, "local GLM reference files or checkpoint feature probes are incomplete")

    support_decision = (
        "repo_owned_runtime_ready"
        if reference_ready and manifest_ready is True
        else "implementation_candidate"
    )

    return {
        "support_decision": support_decision,
        "can_implement_repo_owned_runtime": reference_ready,
        "reference_support": "complete_enough_for_ax_port" if reference_ready else "incomplete",
        "reference_files": reference_files,
        "checkpoint_features": features,
        "draft_manifest_features": {
            "mla_attention_configured": bool((manifest.get("mla_attention") or {})),
            "glm_router_configured": bool((manifest.get("glm_router") or {})),
            "runtime_ready": manifest_ready,
        },
        "blockers": blockers,
        "next_steps": [
            "generate and inspect the GLM model-manifest.json"
            if manifest_ready is None
            else "refresh the GLM benchmark artifact after runtime changes",
            "compare AX decode telemetry against mlx_lm before optimizing performance",
        ],
    }


def probe_deepseek_v4(model_dir: Path, keys: list[str]) -> dict[str, Any]:
    apple_mlx_lm_ref = REPO_ROOT / ".internal/reference/mlx-lm/mlx_lm/models/deepseek_v4.py"
    swiftlm_ref = (
        REPO_ROOT
        / ".internal/reference/SwiftLM/mlx-swift-lm/Libraries/MLXLLM/Models/DeepseekV4.swift"
    )
    swiftlm_tests = (
        REPO_ROOT
        / ".internal/reference/SwiftLM/mlx-swift-lm/Tests/MLXLMTests/DeepseekV4Tests.swift"
    )
    reference_files = [
        _file_probe(apple_mlx_lm_ref, ["DeepseekV4"]),
        _file_probe(
            swiftlm_ref,
            ["DeepseekV4Attention", "compressor/indexer", "tid2eid"],
        ),
        _file_probe(swiftlm_tests, ["Compressor/indexer sub-modules are not yet implemented"]),
    ]
    features = {
        "attention_sinks": _has_any(keys, ".attn.attn_sink"),
        "grouped_output_projection": _has_any(keys, ".attn.wo_a.weight")
        and _has_any(keys, ".attn.wo_b.weight"),
        "attention_compressor": _has_any(keys, ".attn.compressor."),
        "attention_indexer": _has_any(keys, ".attn.indexer."),
        "hash_router_tid2eid": _has_any(keys, ".ffn.gate.tid2eid"),
        "router_correction_bias": _has_any(keys, ".ffn.gate.e_score_correction_bias"),
    }

    partial_swift = reference_files[1]["exists"] and (
        reference_files[1].get("markers", {}).get("compressor/indexer") is True
        or reference_files[1].get("markers", {}).get("tid2eid") is True
    )
    blockers = [
        "upstream Apple mlx-lm checkout has no deepseek_v4 architecture module",
        "available SwiftLM DeepSeek V4 port is partial and drops compressor/indexer weights",
        "available SwiftLM DeepSeek V4 port drops ffn.gate.tid2eid hash-routing weights",
        "downloaded checkpoint contains compressor/indexer/hash-router tensors that affect the forward contract",
        "AX does not yet have DeepSeek V4 hyper-connection, attention-sink, grouped-output, compressor/indexer, or hash-routing graph contracts",
    ]

    return {
        "support_decision": "fail_closed_partial_reference",
        "can_implement_repo_owned_runtime": False,
        "reference_support": "partial_only" if partial_swift else "missing",
        "reference_files": reference_files,
        "checkpoint_features": features,
        "blockers": blockers,
        "next_steps": [
            "do not promote DeepSeek V4 as repo-owned AX support from the current SwiftLM port",
            "identify one complete reference for compressor/indexer and hash-routing semantics",
            "write an AX architecture contract before adding tensor roles or runtime code",
            "only then add shape tests, tokenizer smoke, server smoke, and benchmark rows",
        ],
    }


def probe_gemma4_unified(model_dir: Path, keys: list[str]) -> dict[str, Any]:
    manifest_exists = (model_dir / "model-manifest.json").exists()
    vllm_unified_ref = (
        REPO_ROOT / ".internal/reference/vllm/vllm/model_executor/models/gemma4_unified.py"
    )
    vllm_mm_ref = REPO_ROOT / ".internal/reference/vllm/vllm/model_executor/models/gemma4_mm.py"
    llama_mtmd_ref = REPO_ROOT / ".internal/reference/llama.cpp/tools/mtmd/mtmd.cpp"
    mlx_vlm_unified_ref = (
        REPO_ROOT
        / ".internal/reference/mlx-vlm/mlx_vlm/models/gemma4_unified/gemma4_unified.py"
    )
    mlx_vlm_processor_ref = (
        REPO_ROOT
        / ".internal/reference/mlx-vlm/mlx_vlm/models/gemma4_unified/processing_gemma4_unified.py"
    )
    reference_files = [
        _file_probe(
            vllm_unified_ref,
            ["Gemma4UnifiedVisionEmbedder", "vision_embedder", "embed_audio"],
        ),
        _file_probe(
            vllm_mm_ref,
            ["get_image_repl", "use_bidirectional_attention", "mm_prefix_range"],
        ),
        _file_probe(llama_mtmd_ref, ["PROJECTOR_TYPE_GEMMA4V", "mtmd_decode_use_non_causal"]),
        _file_probe(
            mlx_vlm_unified_ref,
            [
                "Encoder-free Gemma 4 unified vision embedder",
                "get_input_embeddings",
                "mm_token_type_ids",
                "draft_kind == \"mtp\"",
            ],
        ),
        _file_probe(
            mlx_vlm_processor_ref,
            [
                "Gemma4UnifiedProcessor",
                "Gemma4UnifiedImageProcessor",
                "Gemma4UnifiedAudioFeatureExtractor",
                "Gemma4UnifiedVideoProcessor",
            ],
        ),
    ]
    features = {
        "language_model_prefix": any(key.startswith("language_model.model.") for key in keys),
        "vision_embedder": _has_any(keys, "vision_embedder.patch_dense.weight")
        and _has_any(keys, "vision_embedder.pos_embedding"),
        "vision_connector": _has_any(keys, "embed_vision.embedding_projection.weight"),
        "audio_connector": _has_any(keys, "embed_audio.embedding_projection.weight"),
        "vision_tower_absent": not any(key.startswith("vision_tower.") for key in keys),
        "audio_tower_absent": not any(key.startswith("audio_tower.") for key in keys),
    }
    reference_ready = all(item["exists"] for item in reference_files) and all(
        all(markers.values()) for markers in (item.get("markers") or {} for item in reference_files)
    )
    text_candidate = features["language_model_prefix"] and reference_ready
    runtime_source_ready = (
        (REPO_ROOT / "crates/ax-engine-mlx/src/gemma4_unified.rs").exists()
        and (REPO_ROOT / "crates/ax-engine-core/src/gemma4_unified.rs").exists()
    )
    multimodal_candidate = (
        features["vision_embedder"]
        and features["vision_connector"]
        and features["vision_tower_absent"]
    )
    multimodal_tensor_ready = manifest_exists and text_candidate and multimodal_candidate and runtime_source_ready
    audio_tensor_ready = multimodal_tensor_ready and features["audio_connector"]

    blockers: list[str] = []
    if not reference_ready:
        blockers.append("local Gemma4 unified vLLM/llama.cpp/mlx-vlm reference probes are incomplete")
    if not features["language_model_prefix"]:
        blockers.append("checkpoint does not expose language_model.model.* text weights")
    if not manifest_exists:
        blockers.append("AX model-manifest.json is absent for this artifact")
    if manifest_exists and text_candidate and multimodal_candidate and not runtime_source_ready:
        blockers.append("AX Gemma4 unified runtime tensor path is absent")

    if multimodal_tensor_ready:
        support_decision = "text_and_multimodal_tensor_runtime_ready"
    elif manifest_exists and text_candidate:
        support_decision = "text_manifest_ready"
    elif text_candidate:
        support_decision = "implementation_candidate"
    else:
        support_decision = "fail_closed_incomplete_reference"

    return {
        "support_decision": support_decision,
        "can_implement_repo_owned_runtime": text_candidate,
        "reference_support": "complete_enough_for_ax_port" if reference_ready else "incomplete",
        "reference_files": reference_files,
        "checkpoint_features": features,
        "modalities": {
            "text": "manifest_ready" if manifest_exists and text_candidate else "candidate",
            "image": "runtime_tensor_ready" if multimodal_tensor_ready else ("candidate" if multimodal_candidate else "unsupported"),
            "audio": "runtime_tensor_ready" if audio_tensor_ready else ("candidate" if features["audio_connector"] else "unsupported"),
            "video": "runtime_tensor_ready" if multimodal_tensor_ready else ("candidate" if multimodal_candidate else "unsupported"),
        },
        "integration_gaps": []
        if not multimodal_tensor_ready
        else [
            "OpenAI-compatible media URLs/files are not preprocessed by ax-engine-server; callers must provide Gemma4UnifiedRuntimeInputs tensors",
        ],
        "blockers": blockers,
        "next_steps": []
        if manifest_exists and text_candidate
        else [
            "generate and validate model-manifest.json before runtime claims",
            "keep image/audio/video unsupported until projector, prompt replacement, and attention-mask paths are implemented",
        ],
    }


def _expected_ax_family(model_type: str) -> str:
    if model_type in ("qwen3_5", "qwen3.5", "qwen3_5_moe", "qwen3_5_text"):
        return "qwen3_5"
    if model_type in ("qwen3_next", "qwen3_6", "qwen3.6"):
        return "qwen3_next"
    if model_type == "qwen3_moe":
        return "qwen3"
    return model_type


def _validate_manifest(
    model_dir: Path, expected_family: str
) -> tuple[bool | None, list[str], dict[str, Any]]:
    """Validate a model-manifest.json file.

    Returns (ready, blockers, manifest_dict).
    ready is None if manifest doesn't exist, False if invalid, True if valid.
    """
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.exists():
        return None, ["AX model-manifest.json is absent for this artifact"], {}

    try:
        manifest = _read_json(manifest_path)
    except Exception as e:
        return False, [f"AX model-manifest.json failed to parse: {e}"], {}

    blockers: list[str] = []

    manifest_family = manifest.get("model_family")
    if manifest_family and manifest_family != expected_family:
        blockers.append(
            f"AX model-manifest.json model_family {manifest_family!r} "
            f"does not match expected AX family {expected_family!r}"
        )

    tensors = manifest.get("tensors")
    if not tensors or not isinstance(tensors, list) or len(tensors) == 0:
        blockers.append(
            "AX model-manifest.json has empty or missing tensors array"
        )

    runtime_status = manifest.get("runtime_status")
    if isinstance(runtime_status, dict) and not runtime_status.get("ready"):
        blockers.append("AX model-manifest.json is not runtime-ready")

    ready = len(blockers) == 0
    return ready, blockers, manifest


def generic_probe(model_dir: Path, model_type: str, keys: list[str]) -> dict[str, Any]:
    expected_family = _expected_ax_family(model_type)
    manifest_ready, manifest_blockers, manifest = _validate_manifest(
        model_dir, expected_family
    )
    manifest_exists = manifest_ready is not None
    tensors = manifest.get("tensors") if isinstance(manifest, dict) else None
    runtime_status = manifest.get("runtime_status") if isinstance(manifest, dict) else None
    manifest_runtime_ready = (
        bool(runtime_status.get("ready")) if isinstance(runtime_status, dict) else None
    )
    manifest_features = {
        "expected_model_family": expected_family,
        "manifest_model_family": manifest.get("model_family"),
        "manifest_tensor_count": len(tensors) if isinstance(tensors, list) else 0,
        "manifest_runtime_ready": manifest_runtime_ready,
    }

    if model_type in REPO_OWNED_TYPES:
        if manifest_ready is True:
            return {
                "support_decision": "repo_owned_runtime_ready",
                "can_implement_repo_owned_runtime": True,
                "reference_support": "existing_ax_family",
                "reference_files": [],
                "checkpoint_features": manifest_features,
                "blockers": [],
                "next_steps": [],
            }
        blockers = manifest_blockers if manifest_exists else [
            "AX model-manifest.json is absent for this artifact"
        ]
        return {
            "support_decision": "known_family_manifest_invalid"
            if manifest_exists
            else "known_family_manifest_missing",
            "can_implement_repo_owned_runtime": False,
            "reference_support": "existing_ax_family",
            "reference_files": [],
            "checkpoint_features": manifest_features if manifest_exists else {},
            "blockers": blockers,
            "next_steps": [
                "regenerate and validate model-manifest.json before runtime claims"
            ],
        }
    return {
        "support_decision": "unknown_model_type",
        "can_implement_repo_owned_runtime": False,
        "reference_support": "not_checked",
        "reference_files": [],
        "checkpoint_features": {"safetensor_key_count": len(keys)},
        "blockers": [f"model_type {model_type!r} has no AX support contract"],
        "next_steps": ["add a reference-grounded support probe before implementation"],
    }


def probe_model(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    config = _read_json(config_path)
    model_type = str(config.get("model_type", ""))
    keys = _weight_keys(model_dir)

    if model_type == "glm4_moe_lite":
        architecture = probe_glm4_moe_lite(model_dir, keys)
    elif model_type in ("gemma4_unified", "gemma4_unified_text"):
        architecture = probe_gemma4_unified(model_dir, keys)
    elif model_type == "deepseek_v4":
        architecture = probe_deepseek_v4(model_dir, keys)
    else:
        architecture = generic_probe(model_dir, model_type, keys)

    return {
        "schema_version": SCHEMA_VERSION,
        "model_dir": str(model_dir),
        "model_type": model_type,
        "has_ax_manifest": (model_dir / "model-manifest.json").exists(),
        "safetensor_key_count": len(keys),
        **architecture,
    }


def format_text(report: dict[str, Any]) -> str:
    lines = [
        f"model_type: {report['model_type']}",
        f"decision: {report['support_decision']}",
        f"can_implement_repo_owned_runtime: {str(report['can_implement_repo_owned_runtime']).lower()}",
        f"reference_support: {report['reference_support']}",
        f"has_ax_manifest: {str(report['has_ax_manifest']).lower()}",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.append("blockers:")
        lines.extend(f"- {blocker}" for blocker in blockers)
    next_steps = report.get("next_steps") or []
    if next_steps:
        lines.append("next_steps:")
        lines.extend(f"- {step}" for step in next_steps)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args()

    report = probe_model(args.model_dir)
    if args.format == "text":
        print(format_text(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
