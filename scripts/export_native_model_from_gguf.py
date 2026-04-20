#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY_DIR = REPO_ROOT / ".internal" / "reference" / "llama.cpp" / "gguf-py"


try:
    sys.path.insert(0, str(GGUF_PY_DIR))
    import numpy as np
    from gguf.constants import GGMLQuantizationType
    from gguf.gguf_reader import GGUFReader, ReaderTensor
    from gguf.quants import dequantize
except ModuleNotFoundError as exc:
    missing = exc.name or "dependency"
    print(
        json.dumps(
            {
                "status": "error",
                "reason": "missing_python_dependency",
                "missing": missing,
                "hint": (
                    "Use a Python environment with numpy and PyYAML available. "
                    "Example: python3 -m venv /tmp/ax-gguf-venv && "
                    "/tmp/ax-gguf-venv/bin/pip install numpy PyYAML"
                ),
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


LAYER_TENSOR_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


@dataclass
class ExportSpec:
    role: str
    output_name: str
    layer_index: int | None
    tensor: ReaderTensor
    runtime_shape: list[int]
    dtype: str


@dataclass
class SupportReport:
    architecture: str
    layer_count: int
    hidden_size: int | None
    vocab_size: int | None
    fixed_attention_dims: bool
    has_moe: bool
    expert_count: int | None
    experts_per_token: int | None
    expert_intermediate_size: int | None
    supports_export: bool
    runtime_ready: bool
    blockers: list[str]
    runtime_blockers: list[str]
    notes: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "fixed_attention_dims": self.fixed_attention_dims,
            "has_moe": self.has_moe,
            "expert_count": self.expert_count,
            "experts_per_token": self.experts_per_token,
            "expert_intermediate_size": self.expert_intermediate_size,
            "supports_export": self.supports_export,
            "blockers": self.blockers,
            "runtime_ready": self.runtime_ready,
            "runtime_blockers": self.runtime_blockers,
            "notes": self.notes,
        }


def attention_layer_overrides(
    architecture: str, reader: GGUFReader, layer_count: int, layer_tensors: dict[int, dict[str, ReaderTensor]]
) -> tuple[list[int], list[int]]:
    value_from_key_layers: list[int] = []
    v_norm_no_scale_layers: list[int] = []
    architecture = architecture.lower()
    if architecture != "gemma4":
        return value_from_key_layers, v_norm_no_scale_layers

    sliding_pattern = field_contents(reader, "gemma4.attention.sliding_window_pattern") or []
    shared_kv_layers = int(field_contents(reader, "gemma4.attention.shared_kv_layers") or 0)
    first_shared_layer = max(layer_count - shared_kv_layers, 0)

    for layer_index in sorted(layer_tensors):
        is_shared = layer_index >= first_shared_layer if shared_kv_layers > 0 else False
        if not is_shared:
            v_norm_no_scale_layers.append(layer_index)
        is_sliding = True
        if layer_index < len(sliding_pattern):
            is_sliding = bool(sliding_pattern[layer_index])
        if not is_shared and not is_sliding and "attn_v.weight" not in layer_tensors[layer_index]:
            value_from_key_layers.append(layer_index)

    return value_from_key_layers, v_norm_no_scale_layers


def field_contents(reader: GGUFReader, key: str) -> Any | None:
    field = reader.get_field(key)
    return None if field is None else field.contents()


def tensor_shape(tensor: ReaderTensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape.tolist())


def runtime_tensor_shape(tensor: ReaderTensor) -> list[int]:
    shape = tensor_shape(tensor)
    if len(shape) <= 1:
        return list(shape)
    # GGUF stores tensors in ggml order; AX native expects the logical row-major
    # axis order, which corresponds to reversing the ggml dimensions.
    return [int(dim) for dim in reversed(shape)]


def runtime_matrix_shape(tensor: ReaderTensor) -> list[int]:
    shape = runtime_tensor_shape(tensor)
    if len(shape) != 2:
        raise ValueError(f"expected matrix tensor, got shape {shape}")
    return shape


def field_int(reader: GGUFReader, *keys: str) -> int | None:
    for key in keys:
        value = field_contents(reader, key)
        if isinstance(value, (int, np.integer)):
            return int(value)
    return None


def guess_native_family(architecture: str) -> str:
    architecture = architecture.lower()
    if architecture.startswith("gemma"):
        return f"{architecture}_dense"
    if architecture.startswith("qwen"):
        return f"{architecture}_dense"
    if architecture.startswith("llama"):
        return f"{architecture}_dense"
    return f"{architecture}_native"


def ffn_norm_name(layer_tensors: dict[str, ReaderTensor]) -> str | None:
    for candidate in (
        "ffn_norm.weight",
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
    ):
        if candidate in layer_tensors:
            return candidate
    return None


def optional_tensor_name(
    layer_tensors: dict[str, ReaderTensor], *candidates: str
) -> str | None:
    for candidate in candidates:
        if candidate in layer_tensors:
            return candidate
    return None


def analyze_support(reader: GGUFReader) -> SupportReport:
    architecture = str(field_contents(reader, "general.architecture") or "unknown").lower()
    layer_tensors: dict[int, dict[str, ReaderTensor]] = {}
    global_tensors: dict[str, ReaderTensor] = {}
    blockers: list[str] = []
    runtime_blockers: list[str] = []
    notes: list[str] = []

    for tensor in reader.tensors:
        match = LAYER_TENSOR_RE.match(tensor.name)
        if match is None:
            global_tensors[tensor.name] = tensor
            continue
        layer_index = int(match.group(1))
        suffix = match.group(2)
        layer_tensors.setdefault(layer_index, {})[suffix] = tensor

    layer_count = len(layer_tensors)
    if layer_count == 0:
        blockers.append("no_transformer_layers_detected")
    value_from_key_layers, v_norm_no_scale_layers = attention_layer_overrides(
        architecture, reader, layer_count, layer_tensors
    )

    hidden_size = None
    vocab_size = None
    if "token_embd.weight" in global_tensors:
        shape = runtime_matrix_shape(global_tensors["token_embd.weight"])
        vocab_size, hidden_size = shape[0], shape[1]

    if any(".ssm_" in tensor.name or ".attn_gate." in tensor.name for tensor in reader.tensors):
        blockers.append("hybrid_ssm_or_attention_gate_tensors_present")

    expert_count = field_int(reader, f"{architecture}.expert_count")
    experts_per_token = field_int(
        reader,
        f"{architecture}.expert_used_count",
        f"{architecture}.top_k_experts",
    )
    expert_intermediate_size = field_int(
        reader,
        f"{architecture}.expert_feed_forward_length",
        f"{architecture}.moe_intermediate_size",
    )

    required_globals = {"token_embd.weight", "output_norm.weight"}
    if not required_globals.issubset(global_tensors):
        blockers.append("missing_required_global_tensors")
    if "output.weight" not in global_tensors:
        notes.append("output.weight_missing_assuming_tied_embeddings")

    attention_dim_signatures: set[tuple[int, int, int, int]] = set()
    moe_expert_counts: set[int] = set()
    moe_expert_intermediate_sizes: set[int] = set()
    has_moe = False
    has_packed_qkv = False

    for layer_index, tensors in sorted(layer_tensors.items()):
        required = {
            "attn_norm.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        }
        layer_ffn_norm = ffn_norm_name(tensors)
        if layer_ffn_norm is None:
            blockers.append(f"layer_{layer_index}_missing_ffn_norm")
            continue
        if not required.issubset(tensors):
            blockers.append(f"layer_{layer_index}_missing_dense_core_tensors")
            continue

        if "attn_qkv.weight" in tensors:
            has_packed_qkv = True
            continue

        for split_name in ("attn_q.weight", "attn_k.weight"):
            if split_name not in tensors:
                blockers.append(f"layer_{layer_index}_missing_{split_name}")
                break
        else:
            if "attn_v.weight" not in tensors and layer_index not in value_from_key_layers:
                blockers.append(f"layer_{layer_index}_missing_attn_v.weight")
                continue
            if "attn_q_norm.weight" not in tensors and "attn_k_norm.weight" not in tensors:
                blockers.append(f"layer_{layer_index}_missing_qk_norm_for_split_attention")
                continue

            q_rows, q_cols = runtime_matrix_shape(tensors["attn_q.weight"])
            k_rows, k_cols = runtime_matrix_shape(tensors["attn_k.weight"])
            if "attn_v.weight" in tensors:
                v_rows, v_cols = runtime_matrix_shape(tensors["attn_v.weight"])
            else:
                v_rows, v_cols = k_rows, k_cols
            o_rows, o_cols = runtime_matrix_shape(tensors["attn_output.weight"])
            head_dim = None
            if "attn_q_norm.weight" in tensors:
                q_norm_shape = tensor_shape(tensors["attn_q_norm.weight"])
                if len(q_norm_shape) != 1:
                    blockers.append(f"layer_{layer_index}_invalid_attn_q_norm_rank")
                    continue
                head_dim = q_norm_shape[0]
            if "attn_k_norm.weight" in tensors:
                k_norm_shape = tensor_shape(tensors["attn_k_norm.weight"])
                if len(k_norm_shape) != 1:
                    blockers.append(f"layer_{layer_index}_invalid_attn_k_norm_rank")
                    continue
                if head_dim is None:
                    head_dim = k_norm_shape[0]
                elif head_dim != k_norm_shape[0]:
                    blockers.append(f"layer_{layer_index}_qk_norm_head_dim_mismatch")
                    continue
            if head_dim in (None, 0):
                blockers.append(f"layer_{layer_index}_missing_head_dim")
                continue
            if q_rows % head_dim != 0 or k_rows % head_dim != 0 or v_rows % head_dim != 0:
                blockers.append(f"layer_{layer_index}_attention_rows_not_divisible_by_head_dim")
                continue
            if k_rows != v_rows:
                blockers.append(f"layer_{layer_index}_kv_row_mismatch")
                continue
            if o_rows != (hidden_size or o_rows) or o_cols != q_rows:
                blockers.append(f"layer_{layer_index}_attention_output_shape_mismatch")
                continue
            if q_cols != (hidden_size or q_cols) or k_cols != (hidden_size or k_cols):
                blockers.append(f"layer_{layer_index}_attention_input_width_mismatch")
                continue

            attention_dim_signatures.add((q_rows, k_rows, v_rows, head_dim))

        has_moe_router = any(
            name in tensors
            for name in (
                "ffn_gate_inp.weight",
                "ffn_gate_inp.scale",
            )
        )
        has_packed_moe = "ffn_gate_up_exps.weight" in tensors
        has_split_moe = any(
            name in tensors
            for name in (
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
            )
        )
        has_moe_down = any(
            name in tensors
            for name in (
                "ffn_down_exps.weight",
                "ffn_down_exps.scale",
            )
        )
        layer_has_moe = has_moe_router or has_packed_moe or has_split_moe or has_moe_down
        has_moe = has_moe or layer_has_moe
        if not layer_has_moe:
            continue

        if "ffn_gate_inp.weight" not in tensors:
            blockers.append(f"layer_{layer_index}_missing_ffn_gate_inp.weight")
        if "ffn_down_exps.weight" not in tensors:
            blockers.append(f"layer_{layer_index}_missing_ffn_down_exps.weight")
        if has_packed_moe and has_split_moe:
            blockers.append(f"layer_{layer_index}_mixed_packed_and_split_moe_expert_tensors")
        if not has_packed_moe and not has_split_moe:
            blockers.append(f"layer_{layer_index}_missing_moe_expert_projection_weights")

        gate_inp = tensors.get("ffn_gate_inp.weight")
        down_exps = tensors.get("ffn_down_exps.weight")
        if gate_inp is not None:
            gate_shape = runtime_tensor_shape(gate_inp)
            if len(gate_shape) != 2:
                blockers.append(f"layer_{layer_index}_invalid_ffn_gate_inp_rank")
            else:
                layer_expert_count = gate_shape[0]
                layer_hidden_size = gate_shape[1]
                moe_expert_counts.add(layer_expert_count)
                if hidden_size is not None and layer_hidden_size != hidden_size:
                    blockers.append(f"layer_{layer_index}_ffn_gate_inp_hidden_size_mismatch")
                gate_scale = tensors.get("ffn_gate_inp.scale")
                if gate_scale is not None:
                    gate_scale_shape = runtime_tensor_shape(gate_scale)
                    if len(gate_scale_shape) != 1 or gate_scale_shape[0] != layer_hidden_size:
                        blockers.append(f"layer_{layer_index}_ffn_gate_inp_scale_shape_mismatch")
        else:
            layer_expert_count = None
            layer_hidden_size = None

        layer_expert_intermediate_size = None
        if down_exps is not None:
            down_shape = runtime_tensor_shape(down_exps)
            if len(down_shape) != 3:
                blockers.append(f"layer_{layer_index}_invalid_ffn_down_exps_rank")
            else:
                down_expert_count, down_hidden_size, down_intermediate_size = down_shape
                layer_expert_intermediate_size = down_intermediate_size
                moe_expert_counts.add(down_expert_count)
                moe_expert_intermediate_sizes.add(down_intermediate_size)
                if layer_expert_count is not None and down_expert_count != layer_expert_count:
                    blockers.append(f"layer_{layer_index}_moe_expert_count_mismatch")
                if layer_hidden_size is not None and down_hidden_size != layer_hidden_size:
                    blockers.append(f"layer_{layer_index}_moe_hidden_size_mismatch")
                down_scale = tensors.get("ffn_down_exps.scale")
                if down_scale is not None:
                    down_scale_shape = runtime_tensor_shape(down_scale)
                    if len(down_scale_shape) != 1 or down_scale_shape[0] != down_expert_count:
                        blockers.append(f"layer_{layer_index}_ffn_down_exps_scale_shape_mismatch")

        if has_packed_moe:
            packed_shape = runtime_tensor_shape(tensors["ffn_gate_up_exps.weight"])
            if len(packed_shape) != 3:
                blockers.append(f"layer_{layer_index}_invalid_ffn_gate_up_exps_rank")
            elif layer_expert_count is not None and layer_hidden_size is not None:
                packed_expert_count, packed_rows, packed_hidden_size = packed_shape
                if packed_expert_count != layer_expert_count:
                    blockers.append(f"layer_{layer_index}_ffn_gate_up_exps_expert_count_mismatch")
                if packed_hidden_size != layer_hidden_size:
                    blockers.append(f"layer_{layer_index}_ffn_gate_up_exps_hidden_size_mismatch")
                if (
                    layer_expert_intermediate_size is not None
                    and packed_rows != layer_expert_intermediate_size * 2
                ):
                    blockers.append(f"layer_{layer_index}_ffn_gate_up_exps_shape_mismatch")

        if has_split_moe:
            gate_exps = tensors.get("ffn_gate_exps.weight")
            up_exps = tensors.get("ffn_up_exps.weight")
            if gate_exps is None:
                blockers.append(f"layer_{layer_index}_missing_ffn_gate_exps.weight")
            if up_exps is None:
                blockers.append(f"layer_{layer_index}_missing_ffn_up_exps.weight")
            for tensor_name, tensor in (
                ("ffn_gate_exps.weight", gate_exps),
                ("ffn_up_exps.weight", up_exps),
            ):
                if tensor is None:
                    continue
                expert_shape = runtime_tensor_shape(tensor)
                if len(expert_shape) != 3:
                    blockers.append(f"layer_{layer_index}_invalid_{tensor_name}_rank")
                    continue
                expert_count_shape, intermediate_shape, hidden_shape = expert_shape
                moe_expert_counts.add(expert_count_shape)
                moe_expert_intermediate_sizes.add(intermediate_shape)
                if layer_expert_count is not None and expert_count_shape != layer_expert_count:
                    blockers.append(f"layer_{layer_index}_{tensor_name}_expert_count_mismatch")
                if layer_hidden_size is not None and hidden_shape != layer_hidden_size:
                    blockers.append(f"layer_{layer_index}_{tensor_name}_hidden_size_mismatch")
                if (
                    layer_expert_intermediate_size is not None
                    and intermediate_shape != layer_expert_intermediate_size
                ):
                    blockers.append(f"layer_{layer_index}_{tensor_name}_intermediate_size_mismatch")

    fixed_attention_dims = len(attention_dim_signatures) <= 1
    if has_packed_qkv:
        blockers.append("packed_qkv_export_not_implemented")
    if has_moe:
        if len(moe_expert_counts) > 1:
            blockers.append("moe_expert_count_varies_across_layers")
        if len(moe_expert_intermediate_sizes) > 1:
            blockers.append("moe_expert_intermediate_size_varies_across_layers")
        inferred_expert_count = next(iter(moe_expert_counts), None)
        inferred_expert_intermediate_size = next(iter(moe_expert_intermediate_sizes), None)
        if expert_count is None:
            expert_count = inferred_expert_count
        elif inferred_expert_count is not None and expert_count != inferred_expert_count:
            blockers.append("metadata_expert_count_mismatch")
        if expert_intermediate_size is None:
            expert_intermediate_size = inferred_expert_intermediate_size
        elif (
            inferred_expert_intermediate_size is not None
            and expert_intermediate_size != inferred_expert_intermediate_size
        ):
            blockers.append("metadata_expert_intermediate_size_mismatch")
        if expert_count is None:
            blockers.append("missing_moe_expert_count")
        if experts_per_token is None:
            blockers.append("missing_moe_experts_per_token")
        if expert_intermediate_size is None:
            blockers.append("missing_moe_expert_intermediate_size")
        if (
            expert_count is not None
            and experts_per_token is not None
            and experts_per_token > expert_count
        ):
            blockers.append("moe_experts_per_token_exceeds_expert_count")
        runtime_blockers.append("moe_execution_not_implemented")
    if not fixed_attention_dims:
        notes.append("variable_per_layer_attention_dims_detected")
    if value_from_key_layers:
        notes.append(
            "attention_value_from_key_layers=" + ",".join(str(index) for index in value_from_key_layers)
        )
    if v_norm_no_scale_layers:
        notes.append(
            "attention_v_norm_no_scale_layers="
            + ",".join(str(index) for index in v_norm_no_scale_layers)
        )
    if has_moe:
        notes.append("moe_manifest_export_supported_runtime_pending")

    supports_export = not blockers
    runtime_ready = supports_export and not runtime_blockers
    return SupportReport(
        architecture=architecture,
        layer_count=layer_count,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        fixed_attention_dims=fixed_attention_dims,
        has_moe=has_moe,
        expert_count=expert_count,
        experts_per_token=experts_per_token,
        expert_intermediate_size=expert_intermediate_size,
        supports_export=supports_export,
        runtime_ready=runtime_ready,
        blockers=sorted(set(blockers)),
        runtime_blockers=sorted(set(runtime_blockers)),
        notes=notes,
    )


def convert_array(tensor: ReaderTensor, output_dtype: str) -> np.ndarray:
    data = tensor.data
    if tensor.tensor_type == GGMLQuantizationType.F32:
        array = data.astype(np.float32, copy=False)
    elif tensor.tensor_type == GGMLQuantizationType.F16:
        array = data.view(np.float16).astype(np.float32)
    elif tensor.tensor_type == GGMLQuantizationType.BF16:
        array = dequantize(data, tensor.tensor_type)
    else:
        array = dequantize(data, tensor.tensor_type)
    if array.ndim > 1:
        array = np.ascontiguousarray(
            array.transpose(tuple(range(array.ndim - 1, -1, -1)))
        )
    if output_dtype == "f16":
        return np.ascontiguousarray(array.astype(np.float16))
    if output_dtype == "f32":
        return np.ascontiguousarray(array.astype(np.float32))
    raise ValueError(f"unsupported output dtype {output_dtype}")


def native_dtype_name(output_dtype: str) -> str:
    if output_dtype == "f16":
        return "f16"
    if output_dtype == "f32":
        return "f32"
    raise ValueError(f"unsupported output dtype {output_dtype}")


def build_export_specs(
    reader: GGUFReader, report: SupportReport, output_dtype: str, family_override: str | None
) -> tuple[dict[str, Any], list[ExportSpec]]:
    if not report.supports_export:
        raise ValueError("cannot export unsupported GGUF")

    architecture = report.architecture
    family = family_override or guess_native_family(architecture)
    hidden_size = int(report.hidden_size or 0)
    vocab_size = int(report.vocab_size or 0)
    layer_count = report.layer_count

    layer_tensors: dict[int, dict[str, ReaderTensor]] = {}
    global_tensors: dict[str, ReaderTensor] = {}
    for tensor in reader.tensors:
        match = LAYER_TENSOR_RE.match(tensor.name)
        if match is None:
            global_tensors[tensor.name] = tensor
            continue
        layer_index = int(match.group(1))
        suffix = match.group(2)
        layer_tensors.setdefault(layer_index, {})[suffix] = tensor

    value_from_key_layers, v_norm_no_scale_layers = attention_layer_overrides(
        architecture, reader, layer_count, layer_tensors
    )
    first_layer = layer_tensors[min(layer_tensors)]
    q_rows = runtime_matrix_shape(first_layer["attn_q.weight"])[0]
    kv_rows = runtime_matrix_shape(first_layer["attn_k.weight"])[0]
    head_dim_tensor = first_layer.get("attn_q_norm.weight") or first_layer.get("attn_k_norm.weight")
    if head_dim_tensor is None:
        raise ValueError("first layer must provide attn_q_norm.weight or attn_k_norm.weight")
    head_dim = tensor_shape(head_dim_tensor)[0]
    q_head_count = q_rows // head_dim
    kv_head_count = kv_rows // head_dim

    manifest: dict[str, Any] = {
        "schema_version": "ax.native_model.v1",
        "model_family": family,
        "tensor_format": "safetensors",
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "attention_head_count": q_head_count,
        "attention_head_dim": head_dim,
        "kv_head_count": kv_head_count,
        "vocab_size": vocab_size,
        "tie_word_embeddings": "output.weight" not in global_tensors,
        "tensors": [],
    }
    if report.has_moe:
        manifest["moe"] = {
            "expert_count": report.expert_count,
            "experts_per_token": report.experts_per_token,
            "expert_intermediate_size": report.expert_intermediate_size,
        }
    if value_from_key_layers:
        manifest["attention_value_from_key_layers"] = value_from_key_layers
    if v_norm_no_scale_layers:
        manifest["attention_v_norm_no_scale_layers"] = v_norm_no_scale_layers

    rope_theta = field_contents(reader, f"{architecture}.rope.freq_base")
    if isinstance(rope_theta, (int, float)) and rope_theta > 0:
        manifest["rope_theta"] = int(rope_theta)

    specs: list[ExportSpec] = []

    def add_global(name: str, role: str, output_name: str) -> None:
        tensor = global_tensors[name]
        specs.append(
            ExportSpec(
                role=role,
                output_name=output_name,
                layer_index=None,
                tensor=tensor,
                runtime_shape=runtime_tensor_shape(tensor),
                dtype=native_dtype_name(output_dtype),
            )
        )

    add_global("token_embd.weight", "token_embedding", "token_embd.bin")
    add_global("output_norm.weight", "final_norm", "output_norm.bin")
    if "output.weight" in global_tensors:
        add_global("output.weight", "lm_head", "output.bin")

    for layer_index in range(layer_count):
        tensors = layer_tensors[layer_index]
        ffn_norm = ffn_norm_name(tensors)
        assert ffn_norm is not None
        layer_specs = [
            ("attn_norm.weight", "attention_norm", f"blk.{layer_index}.attn_norm.bin"),
            ("attn_q_norm.weight", "attention_q_norm", f"blk.{layer_index}.attn_q_norm.bin"),
            ("attn_k_norm.weight", "attention_k_norm", f"blk.{layer_index}.attn_k_norm.bin"),
            ("attn_q.weight", "attention_q", f"blk.{layer_index}.attn_q.bin"),
            ("attn_k.weight", "attention_k", f"blk.{layer_index}.attn_k.bin"),
            ("attn_output.weight", "attention_o", f"blk.{layer_index}.attn_o.bin"),
            (ffn_norm, "ffn_norm", f"blk.{layer_index}.ffn_norm.bin"),
            ("ffn_gate.weight", "ffn_gate", f"blk.{layer_index}.ffn_gate.bin"),
            ("ffn_up.weight", "ffn_up", f"blk.{layer_index}.ffn_up.bin"),
            ("ffn_down.weight", "ffn_down", f"blk.{layer_index}.ffn_down.bin"),
        ]
        attention_post_norm = optional_tensor_name(
            tensors,
            "post_attention_norm.weight",
            "post_attention_layernorm.weight",
        )
        if attention_post_norm is not None:
            layer_specs.append(
                (
                    attention_post_norm,
                    "attention_post_norm",
                    f"blk.{layer_index}.attn_post_norm.bin",
                )
            )
        ffn_post_norm = optional_tensor_name(
            tensors,
            "post_ffw_norm.weight",
            "post_feedforward_layernorm.weight",
        )
        if ffn_post_norm is not None:
            layer_specs.append(
                (
                    ffn_post_norm,
                    "ffn_post_norm",
                    f"blk.{layer_index}.ffn_post_norm.bin",
                )
            )
        ffn_norm_2 = optional_tensor_name(
            tensors,
            "pre_ffw_norm_2.weight",
            "pre_feedforward_layernorm_2.weight",
        )
        if ffn_norm_2 is not None:
            layer_specs.append(
                (
                    ffn_norm_2,
                    "ffn_norm_2",
                    f"blk.{layer_index}.ffn_norm_2.bin",
                )
            )
        ffn_post_norm_1 = optional_tensor_name(
            tensors,
            "post_ffw_norm_1.weight",
            "post_feedforward_layernorm_1.weight",
        )
        if ffn_post_norm_1 is not None:
            layer_specs.append(
                (
                    ffn_post_norm_1,
                    "ffn_post_norm_1",
                    f"blk.{layer_index}.ffn_post_norm_1.bin",
                )
            )
        ffn_post_norm_2 = optional_tensor_name(
            tensors,
            "post_ffw_norm_2.weight",
            "post_feedforward_layernorm_2.weight",
        )
        if ffn_post_norm_2 is not None:
            layer_specs.append(
                (
                    ffn_post_norm_2,
                    "ffn_post_norm_2",
                    f"blk.{layer_index}.ffn_post_norm_2.bin",
                )
            )
        if "ffn_gate_inp.weight" in tensors:
            layer_specs.extend(
                [
                    ("ffn_gate_inp.weight", "ffn_gate_inp", f"blk.{layer_index}.ffn_gate_inp.bin"),
                ]
            )
        if "ffn_gate_inp.scale" in tensors:
            layer_specs.extend(
                [
                    (
                        "ffn_gate_inp.scale",
                        "ffn_gate_inp_scale",
                        f"blk.{layer_index}.ffn_gate_inp_scale.bin",
                    ),
                ]
            )
        if "ffn_gate_up_exps.weight" in tensors:
            layer_specs.extend(
                [
                    (
                        "ffn_gate_up_exps.weight",
                        "ffn_gate_up_exps_packed",
                        f"blk.{layer_index}.ffn_gate_up_exps.bin",
                    ),
                ]
            )
        if "ffn_gate_exps.weight" in tensors:
            layer_specs.extend(
                [
                    (
                        "ffn_gate_exps.weight",
                        "ffn_gate_exps",
                        f"blk.{layer_index}.ffn_gate_exps.bin",
                    ),
                ]
            )
        if "ffn_up_exps.weight" in tensors:
            layer_specs.extend(
                [
                    (
                        "ffn_up_exps.weight",
                        "ffn_up_exps",
                        f"blk.{layer_index}.ffn_up_exps.bin",
                    ),
                ]
            )
        if "ffn_down_exps.weight" in tensors:
            layer_specs.extend(
                [
                    (
                        "ffn_down_exps.weight",
                        "ffn_down_exps",
                        f"blk.{layer_index}.ffn_down_exps.bin",
                    ),
                ]
            )
        if "ffn_down_exps.scale" in tensors:
            layer_specs.extend(
                [
                    (
                        "ffn_down_exps.scale",
                        "ffn_down_exps_scale",
                        f"blk.{layer_index}.ffn_down_exps_scale.bin",
                    ),
                ]
            )
        if "attn_v.weight" in tensors:
            layer_specs.insert(5, ("attn_v.weight", "attention_v", f"blk.{layer_index}.attn_v.bin"))
        for tensor_name, role, output_name in layer_specs:
            tensor = tensors[tensor_name]
            specs.append(
                ExportSpec(
                    role=role,
                    output_name=output_name,
                    layer_index=layer_index,
                    tensor=tensor,
                    runtime_shape=runtime_tensor_shape(tensor),
                    dtype=native_dtype_name(output_dtype),
                )
            )

    return manifest, specs


def export_model(
    gguf_path: Path, output_dir: Path, output_dtype: str, family_override: str | None
) -> dict[str, Any]:
    reader = GGUFReader(str(gguf_path))
    report = analyze_support(reader)
    if not report.supports_export:
        return {
            "status": "unsupported",
            "gguf_path": str(gguf_path),
            "report": report.to_json(),
        }

    manifest, specs = build_export_specs(reader, report, output_dtype, family_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_tensors = []

    for spec in specs:
        array = convert_array(spec.tensor, output_dtype)
        output_path = output_dir / spec.output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(array.tobytes(order="C"))
        manifest_tensors.append(
            {
                "name": spec.tensor.name,
                "role": spec.role,
                "layer_index": spec.layer_index,
                "dtype": spec.dtype,
                "shape": spec.runtime_shape,
                "file": spec.output_name,
                "offset_bytes": 0,
                "length_bytes": output_path.stat().st_size,
            }
        )

    manifest["tensors"] = manifest_tensors
    manifest_path = output_dir / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "gguf_path": str(gguf_path),
        "output_dir": str(output_dir),
        "report": report.to_json(),
        "manifest_path": str(manifest_path),
        "tensor_count": len(manifest_tensors),
        "model_family": manifest["model_family"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or export an AX native model artifact directory from a GGUF model. "
            "The exporter currently supports only dense split-QKV transformer models "
            "with fixed attention dimensions across layers."
        )
    )
    parser.add_argument("--gguf-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    parser.add_argument("--family")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reader = GGUFReader(str(args.gguf_path))
    report = analyze_support(reader)

    if args.dry_run or args.output_dir is None:
        print(
            json.dumps(
                {
                    "status": "ok" if report.supports_export else "unsupported",
                    "gguf_path": str(args.gguf_path),
                    "report": report.to_json(),
                },
                indent=2,
            )
        )
        return 0 if report.supports_export else 1

    result = export_model(args.gguf_path, args.output_dir, args.dtype, args.family)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
