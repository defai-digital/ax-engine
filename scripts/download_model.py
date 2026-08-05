#!/usr/bin/env python3
"""Download an MLX model through Hugging Face Hub for use with ax-engine.

Downloads model weights and automatically generates the ax-engine manifest
(model-manifest.json). Prefers the ax-engine-bench bundled in the installed wheel,
then an ax-engine-bench on PATH, then a source-checkout build / cargo (dev).

Usage:
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit --dest /path/to/dest
  python scripts/download_model.py mlx-community/Qwen3-4B-4bit --force

For raw HuggingFace checkpoints (not from mlx-community), convert first:
  pip install mlx-lm
  mlx_lm.convert --hf-path <org/model> --mlx-path <dest> -q --q-bits 4
  ax-engine-bench generate-manifest <dest>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_MANIFEST_FILE = "model-manifest.json"
DOWNLOAD_PROVENANCE_FILE = ".ax-engine-download.json"
DOWNLOAD_PROVENANCE_SCHEMA_VERSION = "ax.download_provenance.v1"
NATIVE_MANIFEST_SCHEMA_VERSION = "ax.native_model.v1"
READY_STATUS = "ready"
MANIFEST_MISSING_STATUS = "manifest_missing"
INVALID_STATUS = "invalid"
DOWNLOAD_FAILED_STATUS = "download_failed"
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
HF_HOSTS = ("huggingface.co", "hf.co")
MAX_REPO_ID_BYTES = 96
INVALID_REVISION_CHARS = frozenset("~^:?*[\\")
HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _standalone_is_control(character: str) -> bool:
    return unicodedata.category(character) == "Cc"


def _trim_standalone_reference(value: str) -> str:
    """Match Rust ``str::trim`` rather than Python's four extra C0 separators."""
    start = 0
    end = len(value)
    while start < end and value[start].isspace() and value[start] not in "\x1c\x1d\x1e\x1f":
        start += 1
    while end > start and value[end - 1].isspace() and value[end - 1] not in "\x1c\x1d\x1e\x1f":
        end -= 1
    return value[start:end]


_REPO_REF_MODULE: ModuleType | None | bool = False


def _load_repo_ref_module() -> ModuleType | None:
    """Load the packaged parser without importing the native ``ax_engine`` package."""
    global _REPO_REF_MODULE
    if _REPO_REF_MODULE is not False:
        return _REPO_REF_MODULE
    candidates = (
        REPO_ROOT / "python" / "ax_engine" / "_repo_ref.py",
        REPO_ROOT / "ax_engine" / "_repo_ref.py",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_ax_engine_repo_ref", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REPO_REF_MODULE = module
        return module
    _REPO_REF_MODULE = None
    return None


def _valid_repo_segment(segment: str) -> bool:
    return (
        bool(segment)
        and segment[0] not in "-."
        and segment[-1] not in "-."
        and "--" not in segment
        and ".." not in segment
        and not segment.endswith(".git")
        and all(
            (character.isalnum() and character.isascii()) or character in "-_."
            for character in segment
        )
    )


def _validate_standalone_revision(revision: str, reference: str) -> str:
    for index, character in enumerate(revision):
        if character == "%" and (
            index + 2 >= len(revision)
            or revision[index + 1] not in HEX_DIGITS
            or revision[index + 2] not in HEX_DIGITS
        ):
            raise ValueError(f"invalid percent escape in revision {revision!r}")
    try:
        revision = unquote(revision, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid revision {revision!r} in {reference!r}") from error
    invalid_character = any(
        character.isspace()
        or _standalone_is_control(character)
        or character in INVALID_REVISION_CHARS
        for character in revision
    )
    invalid_component = any(
        not component or component.startswith(".") or component.lower().endswith(".lock")
        for component in revision.split("/")
    )
    if (
        not revision
        or revision == "@"
        or revision.startswith("/")
        or revision.endswith(("/", "."))
        or ".." in revision
        or "@{" in revision
        or invalid_character
        or invalid_component
    ):
        raise ValueError(
            f"invalid revision {revision!r} in {reference!r}; expected a safe Git branch, "
            "tag, or commit"
        )
    return revision


def _standalone_parse_repo_ref(value: str) -> tuple[str, str | None]:
    """Parser fallback for standalone release helpers without the Python package."""
    text = _trim_standalone_reference(value)
    if not text:
        raise ValueError("empty model reference; pass `owner/repo` or a Hugging Face URL")
    lower = text.lower()
    has_scheme = "://" in text
    schemeless_url = any(
        lower.startswith(f"{host}/") or lower.startswith(f"www.{host}/") for host in HF_HOSTS
    )
    if has_scheme or schemeless_url:
        structural_part = text
        for separator in ("?", "#"):
            structural_part = structural_part.split(separator, 1)[0]
        if any(
            character.isspace() or _standalone_is_control(character)
            for character in structural_part
        ):
            raise ValueError(
                f"unsupported model URL {value!r}; only huggingface.co links are supported"
            )
        try:
            parsed = urlsplit(text if has_scheme else f"https://{text}")
            host = (parsed.hostname or "").lower().removeprefix("www.")
            port = parsed.port
        except ValueError as error:
            raise ValueError(
                f"unsupported model URL {value!r}; only huggingface.co links are supported"
            ) from error
        explicit_empty_port = parsed.netloc.endswith(":") and port is None
        if (
            parsed.scheme.lower() not in ("http", "https")
            or host not in HF_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or explicit_empty_port
        ):
            raise ValueError(
                f"unsupported model URL {value!r}; only huggingface.co links are supported"
            )
        text = parsed.path.removeprefix("/")
    if text.endswith("/"):
        text = text[:-1]
    segments = text.split("/")
    if len(segments) < 2 or any(not segment for segment in segments):
        raise ValueError(f"invalid Hugging Face repo reference {value!r}")

    revision: str | None = None
    if "@" in segments[1]:
        repo, revision_head = segments[1].split("@", 1)
        revision = "/".join([revision_head, *segments[2:]])
        segments = [segments[0], repo]
    elif len(segments) > 2:
        if segments[2] == "tree":
            revision = "/".join(segments[3:])
            segments = segments[:2]
        elif segments[2] in ("blob", "resolve"):
            raise ValueError(f"{value!r} links to a file, not a model repository")
        else:
            raise ValueError(f"invalid Hugging Face repo reference {value!r}")
    if len(segments) == 2 and segments[1].endswith(".git"):
        segments[1] = segments[1][:-4]

    repo_id = "/".join(segments)
    if (
        len(segments) != 2
        or len(repo_id) > MAX_REPO_ID_BYTES
        or not all(_valid_repo_segment(segment) for segment in segments)
    ):
        raise ValueError(f"invalid Hugging Face repo reference {value!r}")
    if revision is not None:
        revision = _validate_standalone_revision(revision, value)
    return repo_id, revision


def _parse_repo_ref(value: str) -> tuple[str, str | None]:
    module = _load_repo_ref_module()
    if module is None:
        return _standalone_parse_repo_ref(value)
    parser = getattr(module, "parse_repo_ref", None)
    if not callable(parser):
        raise RuntimeError("ax_engine/_repo_ref.py does not provide parse_repo_ref()")
    parsed = parser(value)
    if (
        not isinstance(parsed, tuple)
        or len(parsed) != 2
        or not isinstance(parsed[0], str)
        or (parsed[1] is not None and not isinstance(parsed[1], str))
    ):
        raise RuntimeError("ax_engine/_repo_ref.py returned an invalid repo reference")
    return parsed


def _weight_tensor_names(model_dir: Path) -> set[str]:
    """Read tensor names without loading model payloads."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_bytes())
            weight_map = payload.get("weight_map", {})
            if isinstance(weight_map, dict):
                return {name for name in weight_map if isinstance(name, str)}
        except (OSError, ValueError, TypeError):
            return set()

    names: set[str] = set()
    for path in model_dir.glob("*.safetensors"):
        try:
            with path.open("rb") as handle:
                header_size_bytes = handle.read(8)
                if len(header_size_bytes) != 8:
                    continue
                header_size = int.from_bytes(header_size_bytes, "little")
                if not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                    continue
                header = json.loads(handle.read(header_size))
                if isinstance(header, dict):
                    names.update(
                        name for name in header if isinstance(name, str) and name != "__metadata__"
                    )
        except (OSError, ValueError, TypeError):
            continue
    return names


_SAFETENSORS_DTYPE_BYTES = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "I8": 1,
    "U8": 1,
    "U32": 4,
}
_SAFETENSORS_DTYPE_TO_MANIFEST = {dtype: dtype.lower() for dtype in _SAFETENSORS_DTYPE_BYTES}


def _checked_safetensors_length(
    shape: list[int],
    element_bytes: int,
    payload_size: int,
) -> int | None:
    """Return the expected byte length, bounded by the actual shard payload."""
    if any(dimension == 0 for dimension in shape):
        return 0
    expected = element_bytes
    for dimension in shape:
        if expected > payload_size // dimension:
            return None
        expected *= dimension
    return expected


def _safetensors_ranges_overlap(ranges: list[tuple[int, int]]) -> bool:
    max_end = 0
    found_range = False
    for start, end in sorted(data_range for data_range in ranges if data_range[0] < data_range[1]):
        if found_range and start < max_end:
            return True
        found_range = True
        max_end = max(max_end, end)
    return False


def _safetensors_file_error(path: Path) -> str | None:
    """Return a structural error without loading tensor payloads into memory."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header_size_bytes = handle.read(8)
            if len(header_size_bytes) != 8:
                return f"truncated safetensors header in {path}"
            header_size = int.from_bytes(header_size_bytes, "little")
            if not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                return f"invalid safetensors header size {header_size} in {path}"
            if 8 + header_size > file_size:
                return f"truncated safetensors metadata in {path}"
            try:
                header = json.loads(handle.read(header_size))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                return f"invalid safetensors metadata in {path}: {error}"
    except OSError as error:
        return f"unable to read safetensors shard {path}: {error}"

    if not isinstance(header, dict):
        return f"invalid safetensors metadata in {path}: root must be an object"
    payload_size = file_size - 8 - header_size
    tensor_count = 0
    data_ranges: list[tuple[int, int]] = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        if not isinstance(tensor, dict):
            return f"invalid safetensors tensor entry {name!r} in {path}"
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or any(
                not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
                for dimension in shape
            )
        ):
            return f"invalid dtype or shape for tensor {name!r} in {path}"
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
        ):
            return f"invalid data_offsets for tensor {name!r} in {path}"
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            return (
                f"tensor {name!r} in {path} declares data offsets "
                f"{offsets!r} outside the {payload_size}-byte payload"
            )
        data_ranges.append((start, end))
        element_bytes = _SAFETENSORS_DTYPE_BYTES.get(dtype)
        if element_bytes is not None:
            expected_length = _checked_safetensors_length(
                shape,
                element_bytes,
                payload_size,
            )
            if expected_length is None:
                return (
                    f"tensor {name!r} in {path} has dtype {dtype} and shape {shape!r}, "
                    f"which exceed the {payload_size}-byte payload"
                )
            actual_length = end - start
            if actual_length != expected_length:
                return (
                    f"tensor {name!r} in {path} has {actual_length} data bytes; "
                    f"dtype {dtype} and shape {shape!r} require {expected_length}"
                )
    if tensor_count == 0:
        return f"no tensors declared in safetensors shard {path}"
    if _safetensors_ranges_overlap(data_ranges):
        return f"overlapping tensor data ranges in {path}"
    return None


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _root_safetensors_header_metadata(
    model_dir: Path,
) -> (
    dict[
        tuple[str, str],
        tuple[str, str, str, tuple[int, ...], int, int],
    ]
    | None
):
    """Read bindable tensor metadata from root safetensors files, failing closed."""
    try:
        paths = sorted(model_dir.glob("*.safetensors"))
    except OSError:
        return None
    if not paths:
        return None

    metadata: dict[
        tuple[str, str],
        tuple[str, str, str, tuple[int, ...], int, int],
    ] = {}
    for path in paths:
        try:
            with path.open("rb") as handle:
                file_size = os.fstat(handle.fileno()).st_size
                header_size_bytes = handle.read(8)
                if len(header_size_bytes) != 8:
                    return None
                header_size = int.from_bytes(header_size_bytes, "little")
                if (
                    not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES
                    or 8 + header_size > file_size
                ):
                    return None
                header_bytes = handle.read(header_size)
                if len(header_bytes) != header_size:
                    return None
                header = json.loads(
                    header_bytes,
                    object_pairs_hook=_unique_json_object,
                )
        except (OSError, UnicodeDecodeError, ValueError, TypeError):
            return None

        if not isinstance(header, dict):
            return None
        data_base_offset = 8 + header_size
        payload_size = file_size - data_base_offset
        tensor_count = 0
        data_ranges: list[tuple[int, int]] = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            tensor_count += 1
            if not name or not isinstance(tensor, dict):
                return None
            dtype = tensor.get("dtype")
            shape = tensor.get("shape")
            offsets = tensor.get("data_offsets")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or any(
                    not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
                    for dimension in shape
                )
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets
                )
            ):
                return None
            start, end = offsets
            if start < 0 or end < start or end > payload_size:
                return None
            data_ranges.append((start, end))

            manifest_dtype = _SAFETENSORS_DTYPE_TO_MANIFEST.get(dtype)
            if manifest_dtype is None:
                continue
            expected_length = _checked_safetensors_length(
                shape,
                _SAFETENSORS_DTYPE_BYTES[dtype],
                payload_size,
            )
            if expected_length is None or end - start != expected_length:
                return None
            source_key = (path.name, name)
            if source_key in metadata:
                return None
            metadata[source_key] = (
                name,
                path.name,
                manifest_dtype,
                tuple(shape),
                data_base_offset + start,
                end - start,
            )
        if tensor_count == 0:
            return None
        if _safetensors_ranges_overlap(data_ranges):
            return None
    return metadata


def _manifest_missing_required_roles(manifest: dict) -> str | None:
    """Return a reason when the manifest lacks roles the native loader requires.

    Mirrors the essential checks in ``NativeModelArtifacts::from_dir`` /
    ``validate_manifest_roles`` so download tooling does not report AX-ready
    for a structurally intact but semantically incomplete manifest (e.g. only
    ``token_embedding``). Full shape / quantization validation still belongs
    to the native path; this is a fail-closed readiness gate.
    """
    model_family = manifest.get("model_family")
    if not isinstance(model_family, str) or not model_family:
        return "missing model_family"

    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        return "missing tensors"

    # Whisper preserves checkpoint names as role=other; skip language-model
    # role requirements here (native validates Whisper against config.json).
    if model_family == "whisper":
        if any(
            not isinstance(tensor, dict) or tensor.get("role") != "other"
            for tensor in tensors
        ):
            return "whisper tensors must use role=other"
        return None

    layer_count = manifest.get("layer_count")
    if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
        return "invalid layer_count"

    global_roles: set[str] = set()
    layer_roles: dict[int, set[str]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict):
            return "tensor entry is not an object"
        role = tensor.get("role")
        if not isinstance(role, str) or not role:
            return "tensor missing role"
        layer_index = tensor.get("layer_index")
        if layer_index is None:
            global_roles.add(role)
            continue
        if not isinstance(layer_index, int) or isinstance(layer_index, bool) or layer_index < 0:
            return f"invalid layer_index for role {role}"
        if layer_index >= layer_count:
            return f"layer_index {layer_index} exceeds layer_count {layer_count}"
        layer_roles.setdefault(layer_index, set()).add(role)

    if "token_embedding" not in global_roles:
        return "missing required tensor role token_embedding"
    if "final_norm" not in global_roles:
        return "missing required tensor role final_norm"

    if model_family == "embeddinggemma":
        if "embedding_dense0" not in global_roles:
            return "missing required tensor role embedding_dense0"
        if "embedding_dense1" not in global_roles:
            return "missing required tensor role embedding_dense1"
    elif not manifest.get("tie_word_embeddings", False) and "lm_head" not in global_roles:
        return "missing required tensor role lm_head"

    if model_family == "gemma4_assistant":
        if "assistant_pre_projection" not in global_roles:
            return "missing required tensor role assistant_pre_projection"
        if "assistant_post_projection" not in global_roles:
            return "missing required tensor role assistant_post_projection"

    is_nemotron_h = model_family == "nemotron_h"
    for layer_index in range(layer_count):
        roles = layer_roles.get(layer_index)
        if not roles:
            return f"missing tensors for layer {layer_index}"
        if "attention_norm" not in roles:
            return f"layer {layer_index} is missing required tensor role attention_norm"
        if is_nemotron_h:
            continue
        if "ffn_norm" not in roles and "attention_post_norm" not in roles:
            return (
                f"layer {layer_index} is missing required tensor role "
                "ffn_norm or attention_post_norm"
            )

        has_packed_gate_up = "ffn_gate_up_packed" in roles
        has_split_gate_up = "ffn_gate" in roles and "ffn_up" in roles
        has_dense_ffn = "ffn_down" in roles and (has_packed_gate_up or has_split_gate_up)
        has_shared_expert_ffn = (
            "ffn_shared_expert_gate_inp" in roles
            and "ffn_shared_expert_gate" in roles
            and "ffn_shared_expert_up" in roles
            and "ffn_shared_expert_down" in roles
        )
        has_mla_shared_expert_ffn = model_family in {
            "glm4_moe_lite",
            "deepseek_v3",
            "deepseek_v32",
            "unlimited_ocr",
        } and (
            "ffn_shared_expert_gate" in roles
            and "ffn_shared_expert_up" in roles
            and "ffn_shared_expert_down" in roles
        )
        has_gpt_oss_mxfp4_moe = model_family == "gpt_oss" and (
            "ffn_gate_up_exps_mxfp4_blocks" in roles
            and "ffn_gate_up_exps_mxfp4_scales" in roles
            and "ffn_down_exps_mxfp4_blocks" in roles
            and "ffn_down_exps_mxfp4_scales" in roles
        )
        has_moe_expert_ffn = "ffn_gate_inp" in roles and (
            has_gpt_oss_mxfp4_moe
            or (
                "ffn_down_exps" in roles
                and (
                    "ffn_gate_up_exps_packed" in roles
                    or "ffn_gate_exps" in roles
                    or "ffn_up_exps" in roles
                )
            )
        )
        if not (
            has_dense_ffn
            or has_shared_expert_ffn
            or has_mla_shared_expert_ffn
            or has_moe_expert_ffn
        ):
            return f"layer {layer_index} must provide dense FFN tensors or MoE expert tensors"

        has_any_attention = any(
            role in roles
            for role in (
                "attention_o",
                "attention_q",
                "attention_k",
                "attention_v",
                "attention_qkv_packed",
                "attention_qa",
                "attention_qb",
                "attention_kv_a",
                "attention_kv_b",
                "attention_embed_q",
                "attention_unembed_out",
            )
        )
        has_any_linear_attention = any(
            role in roles
            for role in (
                "linear_attention_in_proj_qkv",
                "linear_attention_in_proj_qkvz",
                "linear_attention_in_proj_z",
                "linear_attention_in_proj_a",
                "linear_attention_in_proj_b",
                "linear_attention_in_proj_ba",
                "linear_attention_conv1d",
                "linear_attention_dt_bias",
                "linear_attention_a_log",
                "linear_attention_norm",
                "linear_attention_out_proj",
            )
        )
        if has_any_attention:
            if "attention_o" not in roles:
                return f"layer {layer_index} is missing required tensor role attention_o"
            has_packed_qkv = "attention_qkv_packed" in roles
            has_split_qkv = (
                "attention_q" in roles
                and "attention_k" in roles
                and "attention_v" in roles
            )
            has_mla = any(
                role in roles
                for role in (
                    "attention_qa",
                    "attention_qb",
                    "attention_kv_a",
                    "attention_kv_b",
                    "attention_embed_q",
                    "attention_unembed_out",
                )
            )
            if not (has_packed_qkv or has_split_qkv or has_mla):
                return (
                    f"layer {layer_index} must provide attention_qkv_packed or "
                    "attention_q/attention_k/attention_v"
                )
        elif not has_any_linear_attention and not has_moe_expert_ffn:
            return (
                f"layer {layer_index} must provide attention, linear attention, "
                "or MoE expert tensors"
            )

    return None


def _manifest_needs_rebuild(model_dir: Path) -> bool:
    manifest_path = model_dir / MODEL_MANIFEST_FILE
    if not manifest_path.is_file():
        return True
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, ValueError, TypeError):
        return True
    if not isinstance(manifest, dict):
        return True
    if manifest.get("schema_version") != NATIVE_MANIFEST_SCHEMA_VERSION:
        return True
    if not isinstance(manifest.get("model_family"), str) or not manifest["model_family"]:
        return True
    if manifest.get("tensor_format") != "safetensors":
        return True
    for field in (
        "layer_count",
        "hidden_size",
        "attention_head_count",
        "attention_head_dim",
        "kv_head_count",
        "vocab_size",
    ):
        value = manifest.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return True

    runtime_status = manifest.get("runtime_status")
    if runtime_status is not None:
        if not isinstance(runtime_status, dict):
            return True
        ready = runtime_status.get("ready", True)
        blockers = runtime_status.get("blockers", [])
        if ready is not True or not isinstance(blockers, list) or blockers:
            return True

    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        return True
    if _manifest_missing_required_roles(manifest) is not None:
        return True
    source_metadata = _root_safetensors_header_metadata(model_dir)
    if source_metadata is None:
        return True
    declared_source_keys: set[tuple[str, str]] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict):
            return True
        for field in ("name", "role", "dtype", "file"):
            value = tensor.get(field)
            if not isinstance(value, str) or not value:
                return True
        shape = tensor.get("shape")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
                for dimension in shape
            )
            or (not shape and tensor["role"] != "other")
        ):
            return True
        for field in ("offset_bytes", "length_bytes"):
            value = tensor.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (1 if field == "length_bytes" else 0)
            ):
                return True
        source_key = (tensor["file"], tensor["name"])
        if source_key in declared_source_keys:
            return True
        declared_source_keys.add(source_key)
        declared_metadata = (
            tensor["name"],
            tensor["file"],
            tensor["dtype"],
            tuple(shape),
            tensor["offset_bytes"],
            tensor["length_bytes"],
        )
        if source_metadata.get(source_key) != declared_metadata:
            return True
    return False


def manifest_needs_media_rebuild(model_dir: Path) -> bool:
    """Detect published manifests that silently omitted declared media towers."""
    try:
        config = json.loads((model_dir / "config.json").read_bytes())
    except (OSError, ValueError, TypeError):
        return False
    try:
        manifest = json.loads((model_dir / MODEL_MANIFEST_FILE).read_bytes())
    except (OSError, ValueError, TypeError):
        return True
    if not isinstance(manifest, dict):
        return True
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not isinstance(config.get("vision_config"), dict):
        return False

    if model_type in {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3-vl",
        "qwen3-vl-moe",
    }:
        required_prefix_groups = (("vision_tower.", "visual.", "model.visual."),)
    elif model_type in {"gemma4", "gemma4_vl", "gemma4-vl"}:
        required_prefix_groups = (
            ("vision_tower.", "model.vision_tower."),
            ("embed_vision.", "model.embed_vision."),
        )
    else:
        return False

    source_names = _weight_tensor_names(model_dir)
    if not source_names:
        return False
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        manifest_names: set[str] = set()
    else:
        manifest_names = {
            name
            for tensor in tensors
            if isinstance(tensor, dict) and isinstance((name := tensor.get("name")), str)
        }
    return any(
        any(name.startswith(prefixes) for name in source_names)
        and not any(name.startswith(prefixes) for name in manifest_names)
        for prefixes in required_prefix_groups
    )


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def default_mlx_lm_cache_root() -> Path:
    """Return the shared Hugging Face Hub cache root for model snapshots."""
    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        return Path(hf_hub_cache).expanduser()
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache_home / "huggingface" / "hub"


def default_mlx_lm_repo_cache_dir(repo_id: str) -> Path:
    """Return the repository cache directory that contains snapshot revisions."""
    return default_mlx_lm_cache_root() / f"models--{_slug(repo_id)}"


def _path_exists(path: Path) -> bool:
    """Like ``lexists``: broken symlinks still occupy a destination name."""
    return os.path.lexists(path)


def _remove_path(path: Path) -> None:
    """Remove exactly one known path without following a symlink."""
    if not _path_exists(path):
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _download_provenance(repo_id: str, revision: str | None) -> dict:
    return {
        "schema_version": DOWNLOAD_PROVENANCE_SCHEMA_VERSION,
        "repo_id": repo_id,
        "revision": revision,
    }


def _write_download_provenance(dest: Path, repo_id: str, revision: str | None) -> None:
    (dest / DOWNLOAD_PROVENANCE_FILE).write_text(
        json.dumps(_download_provenance(repo_id, revision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_download_provenance(dest: Path) -> object | None:
    try:
        return json.loads((dest / DOWNLOAD_PROVENANCE_FILE).read_bytes())
    except (OSError, ValueError, TypeError):
        return None


def _destination_matches(dest: Path, repo_id: str, revision: str | None) -> bool:
    return _read_download_provenance(dest) == _download_provenance(repo_id, revision)


def _destination_has_download_provenance(dest: Path) -> bool:
    payload = _read_download_provenance(dest)
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == DOWNLOAD_PROVENANCE_SCHEMA_VERSION
        and isinstance(payload.get("repo_id"), str)
        and bool(payload["repo_id"])
        and (payload.get("revision") is None or isinstance(payload.get("revision"), str))
    )


def _destination_has_model_markers(dest: Path) -> bool:
    try:
        if (dest / MODEL_MANIFEST_FILE).is_file():
            return True
        return any(path.is_file() for path in dest.glob("*.safetensors"))
    except OSError:
        return False


def _is_unrelated_nonempty_dir(dest: Path) -> bool:
    """A non-empty real directory that shows no sign of being a model download."""
    return (
        dest.is_dir()
        and not dest.is_symlink()
        and _destination_is_nonempty(dest)
        and not _destination_has_download_provenance(dest)
        and not _destination_has_model_markers(dest)
    )


def _destination_is_nonempty(dest: Path) -> bool:
    if not _path_exists(dest):
        return False
    if not dest.is_dir() or dest.is_symlink():
        return True
    try:
        next(dest.iterdir())
    except StopIteration:
        return False
    except OSError:
        return True
    return True


def _validate_destination_before_activation(
    dest: Path,
    *,
    repo_id: str,
    revision: str | None,
    force: bool,
) -> None:
    """Revalidate the destination after staging to close download-time races."""
    if not _path_exists(dest):
        return
    if force:
        if _is_unrelated_nonempty_dir(dest):
            raise RuntimeError(
                f"refusing to replace unrelated non-empty directory {dest}; "
                "it appeared or changed while the model was being prepared"
            )
        return

    if dest.is_symlink() or not dest.is_dir():
        raise RuntimeError(
            f"refusing to replace destination {dest}: it appeared or changed "
            "while the model was being prepared"
        )
    if not _destination_is_nonempty(dest):
        return
    if not _destination_matches(dest, repo_id, revision):
        raise RuntimeError(
            f"refusing to replace destination {dest}: it no longer matches "
            f"{repo_id} at revision {revision!r}"
        )
    if (
        not _validation_errors(dest)
        and not _manifest_needs_rebuild(dest)
        and not manifest_needs_media_rebuild(dest)
    ):
        raise RuntimeError(
            f"refusing to replace destination {dest}: another process made it ready "
            "while this model was being prepared"
        )


def _validate_explicit_destination(dest: Path) -> Path:
    """Reject broad destinations before any download or staging work begins."""
    dest = Path(dest).expanduser()
    try:
        cwd = Path.cwd().resolve(strict=False)
        home = Path.home().resolve(strict=False)
        resolved = (
            dest.resolve(strict=False) if dest.is_absolute() else (cwd / dest).resolve(strict=False)
        )
    except OSError as error:
        raise RuntimeError(f"cannot resolve destination {dest}: {error}") from error

    if not dest.name or resolved.parent == resolved:
        raise RuntimeError(
            f"refusing unsafe destination {dest}: choose a dedicated model directory"
        )
    if resolved == cwd or resolved in cwd.parents:
        raise RuntimeError(
            f"refusing unsafe destination {dest}: the current working directory "
            "or one of its ancestors cannot be replaced"
        )
    if resolved == home or resolved in home.parents:
        raise RuntimeError(
            f"refusing unsafe destination {dest}: the home directory or one of its "
            "ancestors cannot be replaced"
        )
    if dest.is_symlink() and dest.is_dir():
        # A symlinked directory (e.g. a link onto an external volume) is a
        # valid destination: operate on its target so the emptiness checks,
        # staging, disk-space preflight, and atomic swap all happen on the real
        # directory instead of replacing the link itself.
        return resolved
    return dest


def _validate_snapshot_destination(snapshot: Path, dest: Path) -> None:
    """Prevent an atomic destination swap from consuming its own source."""
    try:
        resolved_snapshot = snapshot.resolve(strict=True)
        resolved_dest = dest.resolve(strict=False)
    except OSError as error:
        raise RuntimeError(
            f"cannot validate snapshot {snapshot} against destination {dest}: {error}"
        ) from error
    if (
        resolved_snapshot == resolved_dest
        or resolved_snapshot in resolved_dest.parents
        or resolved_dest in resolved_snapshot.parents
    ):
        raise RuntimeError(f"refusing destination {dest}: it overlaps source snapshot {snapshot}")


def _contained_path(base: Path, *parts: str) -> Path | None:
    """Join untrusted cache-ref components and fail closed on path escape."""
    candidate = base.joinpath(*parts)
    try:
        base_resolved = base.resolve()
        candidate.resolve().relative_to(base_resolved)
    except (OSError, ValueError):
        return None
    return candidate


def _latest_mlx_lm_snapshot(repo_id: str, revision: str | None = None) -> Path | None:
    repo_cache = default_mlx_lm_repo_cache_dir(repo_id)
    snapshots_root = repo_cache / "snapshots"
    if snapshots_root.is_symlink():
        return None
    if revision:
        # A branch/tag ref resolves through refs/<name>; a commit sha is the
        # snapshot directory itself.
        ref_file = _contained_path(repo_cache / "refs", revision)
        if ref_file is None:
            return None
        if ref_file.is_file():
            try:
                resolved = ref_file.read_text().strip()
            except (OSError, UnicodeError):
                resolved = ""
            if resolved:
                snapshot = _contained_path(snapshots_root, resolved)
                if snapshot is None:
                    return None
                if snapshot.is_dir() and not snapshot.is_symlink():
                    return snapshot
        snapshot = _contained_path(snapshots_root, revision)
        if snapshot is None:
            return None
        return snapshot if snapshot.is_dir() and not snapshot.is_symlink() else None
    refs_main = repo_cache / "refs" / "main"
    if refs_main.is_file():
        try:
            revision = refs_main.read_text().strip()
        except (OSError, UnicodeError):
            revision = ""
        if revision:
            snapshot = _contained_path(snapshots_root, revision)
            if snapshot is None:
                return None
            if snapshot.is_dir() and not snapshot.is_symlink():
                return snapshot
    if not snapshots_root.is_dir():
        return None
    candidates = [
        path for path in snapshots_root.iterdir() if path.is_dir() and not path.is_symlink()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "estimating"
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _snapshot_weight_progress(snapshot: Path) -> tuple[int, int] | None:
    index_path = snapshot / "model.safetensors.index.json"
    total = 0
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            total = int(index.get("metadata", {}).get("total_size") or 0)
        except (OSError, ValueError, TypeError):
            total = 0
    downloaded = 0
    for path in snapshot.glob("*.safetensors"):
        with suppress(OSError):
            downloaded += path.stat().st_size
    if total <= 0 and downloaded > 0:
        total = downloaded
    if total <= 0:
        return None
    return min(downloaded, total), total


def _download_progress_message(
    repo_id: str,
    started_at: float,
    revision: str | None = None,
) -> tuple[int, str]:
    elapsed = time.monotonic() - started_at
    snapshot = _latest_mlx_lm_snapshot(repo_id, revision)
    if snapshot is not None and (progress := _snapshot_weight_progress(snapshot)) is not None:
        downloaded, total = progress
        ratio = 0.0 if total == 0 else downloaded / total
        eta = elapsed * (1.0 - ratio) / ratio if ratio > 0 else None
        gib = 1024**3
        return (
            5 + int(min(ratio, 1.0) * 80),
            "Downloading weights "
            f"({downloaded / gib:.1f}/{total / gib:.1f} GiB, "
            f"elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)})",
        )
    synthetic = min(25, 5 + int(elapsed // 20))
    return (
        synthetic,
        f"Downloading with Hugging Face Hub (elapsed {_format_duration(elapsed)})",
    )


def _format_bytes(num: float | None) -> str:
    if num is None:
        return "?"
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            with suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total


def _cache_storage_bytes(path: Path) -> int:
    """Return physical cache file bytes without following snapshot symlinks."""
    total = 0
    seen_files: set[tuple[int, int]] = set()
    if path.is_symlink() or not path.exists():
        return 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not os.path.islink(os.path.join(root, name))]
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.islink(file_path):
                continue
            try:
                metadata = os.stat(file_path, follow_symlinks=False)
            except OSError:
                continue
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            total += metadata.st_size
    return total


def _total_repo_bytes(repo_id: str, revision: str | None = None) -> int | None:
    """Best-effort total download size from the Hub, summed across all repo files."""
    # huggingface_hub reads the xet preference while importing constants.
    _prefer_classic_hf_transfer()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    try:
        kwargs = {"repo_id": repo_id, "files_metadata": True}
        if revision is not None:
            kwargs["revision"] = revision
        info = HfApi().repo_info(**kwargs)
    except Exception:
        return None
    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        size = getattr(sibling, "size", None)
        if isinstance(size, int):
            total += size
    return total or None


def _render_progress_bar(
    downloaded: int,
    total: int | None,
    speed: float | None,
    eta: float | None,
    width: int = 24,
) -> str:
    speed_text = f"{_format_bytes(speed)}/s" if speed else "-- B/s"
    if total and total > 0:
        ratio = min(downloaded / total, 1.0)
        filled = int(ratio * width)
        bar = "#" * filled + "-" * (width - filled)
        return (
            f"[{bar}] {ratio * 100:4.0f}%  "
            f"{_format_bytes(downloaded)}/{_format_bytes(total)}  "
            f"{speed_text}  ETA {_format_duration(eta)}"
        )
    return f"{_format_bytes(downloaded)} downloaded  {speed_text}"


class _ProgressBarReporter:
    """Poll the Hugging Face cache directory and render a live progress bar to a stream.

    Disk polling is deliberately independent of huggingface_hub internals so it keeps
    working across hub versions. It owns no download state; it only observes bytes on disk.
    """

    def __init__(self, repo_id: str, total: int | None, stream, *, interval: float = 0.4) -> None:
        self._repo_dir = default_mlx_lm_repo_cache_dir(repo_id)
        self._total = total
        self._stream = stream
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_time: float | None = None
        self._last_bytes = 0
        self._speed_ema: float | None = None
        self._baseline_bytes = 0

    def __enter__(self) -> _ProgressBarReporter:
        self._baseline_bytes = _cache_storage_bytes(self._repo_dir)
        self._stream.write(f"Downloading {os.path.basename(str(self._repo_dir))}\n")
        self._stream.flush()
        self._thread = threading.Thread(target=self._run, name="ax-download-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._render(final=True, complete=exc_type is None)

    def _measure(self) -> tuple[int, float | None, float | None]:
        downloaded = max(_cache_storage_bytes(self._repo_dir) - self._baseline_bytes, 0)
        now = time.monotonic()
        speed = None
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                inst = max(downloaded - self._last_bytes, 0) / dt
                self._speed_ema = (
                    inst if self._speed_ema is None else 0.6 * self._speed_ema + 0.4 * inst
                )
                speed = self._speed_ema
        self._last_time = now
        self._last_bytes = downloaded
        eta = None
        if self._total and speed and speed > 0:
            eta = max(self._total - downloaded, 0) / speed
        return downloaded, speed, eta

    def _render(self, *, final: bool = False, complete: bool = False) -> None:
        downloaded, speed, eta = self._measure()
        if complete and self._total:
            downloaded = max(downloaded, self._total)
            eta = 0.0
        line = _render_progress_bar(downloaded, self._total, speed, eta)
        self._stream.write("\r\033[K" + line)
        if final:
            self._stream.write("\n")
        self._stream.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._render()
            self._stop.wait(self._interval)


def _emit_progress(done: int, total: int, file: str) -> None:
    print(json.dumps({"event": "progress", "done": done, "total": total, "file": file}), flush=True)


def _prefer_classic_hf_transfer() -> None:
    """Avoid hf_xet log-init races that surface as `File exists (os error 17)`.

    Recent `huggingface_hub` + `hf_xet` stacks on macOS can fail mid-download
    when concurrent processes both try to create the same xet log file. Classic
    Hub transfer is slightly slower but reliable for CLI/TUI downloads. Users
    who want xet can set ``HF_HUB_DISABLE_XET=0`` before launching.
    """
    if "HF_HUB_DISABLE_XET" not in os.environ:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


def _run_hf_snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
    quiet: bool = False,
    progress_json: bool = False,
    progress_bar: bool = False,
    total_bytes: int | None = None,
) -> Path:
    # Must run before importing huggingface_hub (constants read env at import).
    _prefer_classic_hf_transfer()
    show_bar = progress_bar and sys.stderr.isatty()
    suppress_hub_bars = quiet or show_bar
    previous_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    if suppress_hub_bars:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        if suppress_hub_bars:
            if previous_progress is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_progress
        raise RuntimeError(
            "huggingface_hub is required for model downloads. Install it with:\n"
            "  pip install huggingface_hub\n"
            "or:\n"
            "  pip install 'ax-engine[download]'"
        ) from error

    started_at = time.monotonic()
    try:
        if progress_json:
            done, message = _download_progress_message(repo_id, started_at, revision)
            _emit_progress(done, 100, message)
        kwargs = {"repo_id": repo_id}
        if revision:
            kwargs["revision"] = revision
        if force_download:
            # Ask the Hub client to refresh the requested snapshot in place.
            # Removing the whole repository cache would also erase unrelated
            # revisions and any blobs still used by them.
            kwargs["force_download"] = True
        if max_workers := os.environ.get("AX_ENGINE_HF_MAX_WORKERS"):
            kwargs["max_workers"] = int(max_workers)
        if show_bar:
            # The caller's disk preflight already sized the repo; reuse it
            # instead of a second repo_info round trip.
            with _ProgressBarReporter(repo_id, total_bytes, sys.stderr):
                snapshot = Path(snapshot_download(**kwargs))
        else:
            snapshot = Path(snapshot_download(**kwargs))
        if progress_json:
            _emit_progress(85, 100, "Downloaded Hugging Face Hub snapshot")
        return snapshot
    except Exception as error:
        raise RuntimeError(f"Hugging Face Hub download failed for {repo_id}: {error}") from error
    finally:
        if suppress_hub_bars:
            if previous_progress is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_progress


def _copy_snapshot_to_dest(
    snapshot: Path,
    dest: Path,
    *,
    repo_id: str,
    revision: str | None,
    force: bool = False,
    prepare_destination: Callable[[Path], None] | None = None,
) -> None:
    """Copy the snapshot into `dest` atomically.

    Builds a sibling temp directory first and swaps it into place, so an
    interrupted copy never leaves a partial `dest` that the idempotence fast
    path could mistake for a complete model. A pre-existing `dest` (e.g. a
    forced refresh) is moved to a unique backup only after the temp build
    succeeds, and is restored if activation fails.
    """
    dest = Path(dest)
    _validate_snapshot_copy_links(snapshot)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(
        tempfile.mkdtemp(
            prefix=f".{dest.name}.download-tmp-",
            dir=dest.parent,
        )
    )
    # mkdtemp creates mode 0700; restore the umask-derived mode a plain mkdir
    # would have used so the activated destination stays readable by other
    # users and services.
    current_umask = os.umask(0)
    os.umask(current_umask)
    os.chmod(tmp, 0o777 & ~current_umask)
    backup: Path | None = None
    activated = False
    try:
        for path in snapshot.iterdir():
            target = tmp / path.name
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if resolved.is_dir():
                    shutil.copytree(resolved, target)
                else:
                    shutil.copy2(resolved, target)
            elif path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)
        _write_download_provenance(tmp, repo_id, revision)
        if prepare_destination is not None:
            # Keep the previous destination untouched until the staged model
            # is fully AX-ready, including manifest generation.
            prepare_destination(tmp)

        _validate_destination_before_activation(
            dest,
            repo_id=repo_id,
            revision=revision,
            force=force,
        )
        if _path_exists(dest):
            # Reserve a collision-free name created by this invocation; remove
            # only the empty placeholder before moving the old destination.
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{dest.name}.previous-",
                    dir=dest.parent,
                )
            )
            backup.rmdir()
            dest.rename(backup)
        try:
            tmp.rename(dest)
            activated = True
        except BaseException as activate_error:
            if backup is not None and not _path_exists(dest):
                try:
                    backup.rename(dest)
                    backup = None
                except OSError as restore_error:
                    raise RuntimeError(
                        f"failed to activate {dest}; the previous destination is preserved "
                        f"at {backup}, but automatic rollback failed: {restore_error}"
                    ) from activate_error
            if backup is not None:
                # An unexpected destination appeared before rollback; tell the
                # user where their previous contents went instead of stranding
                # them in a hidden temp-named sibling.
                raise RuntimeError(
                    f"failed to activate {dest}; the previous destination is "
                    f"preserved at {backup}: {activate_error}"
                ) from activate_error
            raise
        if backup is not None:
            try:
                _remove_path(backup)
            except OSError as cleanup_error:
                # The new destination is fully activated; a leftover backup is
                # not a download failure. Report it and succeed.
                print(
                    f"warning: model installed at {dest}, but the previous "
                    f"destination could not be removed and remains at "
                    f"{backup}: {cleanup_error}",
                    file=sys.stderr,
                )
            backup = None
    except BaseException:
        if not activated:
            shutil.rmtree(tmp, ignore_errors=True)
        if backup is not None and not _path_exists(dest):
            try:
                backup.rename(dest)
                backup = None
            except OSError:
                # Keep the uniquely named backup: never delete the only copy.
                pass
        raise


def _validate_snapshot_copy_links(snapshot: Path) -> None:
    """Allow Hub blob links, but reject directory or out-of-cache symlinks."""
    if snapshot.is_symlink():
        raise RuntimeError(f"refusing symlinked snapshot directory {snapshot}")
    try:
        resolved_snapshot = snapshot.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot validate snapshot directory {snapshot}: {error}") from error
    allowed_roots = [resolved_snapshot]
    if resolved_snapshot.parent.name == "snapshots":
        blobs = resolved_snapshot.parent.parent / "blobs"
        if blobs.is_dir() and not blobs.is_symlink():
            try:
                allowed_roots.append(blobs.resolve(strict=True))
            except OSError as error:
                raise RuntimeError(
                    f"cannot validate Hub blob directory {blobs}: {error}"
                ) from error

    for current, directories, filenames in os.walk(snapshot, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"refusing symlinked directory in snapshot: {path}")
        for name in filenames:
            path = current_path / name
            if not path.is_symlink():
                try:
                    mode = path.stat(follow_symlinks=False).st_mode
                except OSError as error:
                    raise RuntimeError(f"cannot inspect snapshot entry {path}: {error}") from error
                if not stat.S_ISREG(mode):
                    raise RuntimeError(f"refusing special file in snapshot: {path}")
                continue
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(
                    f"refusing snapshot symlink outside allowed roots: {path}"
                ) from error
            contained = any(resolved == root or root in resolved.parents for root in allowed_roots)
            if not contained:
                raise RuntimeError(f"refusing snapshot symlink outside allowed roots: {path}")
            if not resolved.is_file():
                raise RuntimeError(f"refusing non-file snapshot symlink target: {path}")


def _nearest_existing_dir(path: Path) -> Path:
    candidate = path
    while not candidate.is_dir():
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return candidate


def _preflight_disk_space(
    repo_id: str,
    dest: Path | None,
    *,
    revision: str | None = None,
    needs_cache_download: bool = True,
    known_snapshot_bytes: int | None = None,
) -> int | None:
    """Fail fast when the download plainly does not fit.

    Returns the repo's total bytes when known (also used by callers for
    progress display). Best-effort: offline/sizeless repos skip the check.
    """
    total = known_snapshot_bytes or _total_repo_bytes(repo_id, revision)
    if not total:
        return None

    requirements: list[tuple[Path, int]] = []
    if needs_cache_download:
        requirements.append((default_mlx_lm_cache_root(), total))
    if dest is not None:
        requirements.append((dest.parent, total))

    # Cache and explicit destination commonly share a volume. Aggregate their
    # requirements; checking each against the same free-space value would let
    # a 2x operation pass with only ~1x space available.
    by_device: dict[int, tuple[Path, int]] = {}
    for target, required in requirements:
        probe = _nearest_existing_dir(target)
        try:
            device = probe.stat().st_dev
        except OSError:
            continue
        existing = by_device.get(device)
        if existing is None:
            by_device[device] = (probe, required)
        else:
            by_device[device] = (existing[0], existing[1] + required)

    for probe, required in by_device.values():
        try:
            free = shutil.disk_usage(probe).free
        except OSError:
            continue
        required_with_headroom = int(required * 1.05)
        if free < required_with_headroom:
            raise RuntimeError(
                f"insufficient disk space for {repo_id}: the download is "
                f"~{_format_bytes(total)} and this operation needs "
                f"~{_format_bytes(required_with_headroom)} on the volume, but only "
                f"{_format_bytes(free)} is free "
                f"on the volume holding {probe}. Free space or pass --dest to a "
                "larger volume."
            )
    return total


def download(
    repo_id: str,
    dest: Path | None,
    force: bool = False,
    *,
    revision: str | None = None,
    quiet: bool = False,
    progress_json: bool = False,
    progress_bar: bool = False,
    prepare_destination: Callable[[Path], None] | None = None,
) -> Path:
    parsed_repo_id, embedded_revision = _parse_repo_ref(repo_id)
    repo_id = parsed_repo_id
    if revision is not None:
        # Reuse the shared parser for explicit revisions too. This blocks
        # absolute/traversal refs before they are joined into local cache paths.
        _, revision = _parse_repo_ref(f"{repo_id}@{revision}")
    else:
        # Already normalized by the reference parse above; running it through
        # the parser again would percent-decode a second time.
        revision = embedded_revision

    if dest is not None:
        dest = _validate_explicit_destination(Path(dest))

    if dest is not None and _path_exists(dest) and not dest.is_dir() and not force:
        raise RuntimeError(
            f"destination {dest} exists and is not a directory; pass --force to replace it"
        )

    if dest is not None and force and _is_unrelated_nonempty_dir(dest):
        raise RuntimeError(
            f"refusing to replace unrelated non-empty directory {dest}; choose a dedicated "
            "model destination or remove its contents explicitly"
        )

    if dest is not None and _destination_is_nonempty(dest) and not force:
        if not _destination_matches(dest, repo_id, revision):
            raise RuntimeError(
                f"destination {dest} is non-empty and is not a matching "
                f"{repo_id} download"
                + (f" at revision {revision}" if revision else "")
                + "; choose an empty destination or pass --force to replace it"
            )
        safetensors = list(dest.glob("*.safetensors"))
        # Only trust a destination whose contents actually validate; a partial
        # or corrupted copy (interrupted older-version download) is recopied.
        validation_ok = bool(safetensors) and not _validation_errors(dest)
        if validation_ok and (dest / MODEL_MANIFEST_FILE).exists():
            if not quiet:
                print(f"  already present with manifest: {dest}")
            if progress_json:
                _emit_progress(100, 100, "Ready")
            return dest
        if validation_ok:
            if not quiet:
                print(f"  weights present but manifest missing: {dest}")
            if progress_json:
                _emit_progress(85, 100, "Weights already present")
            return dest
        if safetensors and not quiet:
            print(f"  ignoring incomplete destination contents: {dest}")

    snapshot = None if force else _latest_mlx_lm_snapshot(repo_id, revision)
    if snapshot is not None:
        _validate_snapshot_copy_links(snapshot)
    if snapshot is not None and not _validation_errors(snapshot):
        if progress_json:
            _emit_progress(85, 100, "Using existing Hugging Face Hub cache snapshot")
        if dest is None:
            if not quiet:
                print(f"  already present in Hugging Face Hub cache: {snapshot}")
            return snapshot
        _validate_snapshot_destination(snapshot, dest)
        _preflight_disk_space(
            repo_id,
            dest,
            revision=revision,
            needs_cache_download=False,
            known_snapshot_bytes=_dir_size_bytes(snapshot),
        )
        _copy_snapshot_to_dest(
            snapshot,
            dest,
            repo_id=repo_id,
            revision=revision,
            force=force,
            prepare_destination=prepare_destination,
        )
        return dest

    if not quiet:
        destination = (
            "Hugging Face Hub cache" if dest is None else f"{dest} via Hugging Face Hub cache"
        )
        revision_note = f" @ {revision}" if revision else ""
        print(f"  downloading {repo_id}{revision_note} -> {destination}")
    # Set transfer/progress preferences before preflight imports
    # huggingface_hub to query repository metadata.
    _prefer_classic_hf_transfer()
    suppress_hub_bars = quiet or (progress_bar and sys.stderr.isatty())
    previous_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    if suppress_hub_bars:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        total_bytes = _preflight_disk_space(
            repo_id,
            dest,
            revision=revision,
        )
        if total_bytes and not quiet:
            print(f"  download size: ~{_format_bytes(total_bytes)}")
        if total_bytes and progress_json:
            _emit_progress(5, 100, f"Download size ~{_format_bytes(total_bytes)}")
        snapshot = _run_hf_snapshot_download(
            repo_id,
            revision=revision,
            force_download=force,
            quiet=quiet,
            progress_json=progress_json,
            progress_bar=progress_bar,
            total_bytes=total_bytes,
        )
    finally:
        if suppress_hub_bars:
            if previous_progress is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_progress

    _validate_snapshot_copy_links(snapshot)
    snapshot_errors = _validation_errors(snapshot)
    if snapshot_errors:
        raise RuntimeError(
            f"downloaded snapshot for {repo_id} is invalid: " + "; ".join(snapshot_errors)
        )

    if dest is None:
        return snapshot

    _validate_snapshot_destination(snapshot, dest)
    _copy_snapshot_to_dest(
        snapshot,
        dest,
        repo_id=repo_id,
        revision=revision,
        force=force,
        prepare_destination=prepare_destination,
    )
    return dest


def _run_manifest_command(
    command: list[str],
    *,
    quiet: bool = False,
    cwd: Path | None = None,
    label: str,
) -> bool:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        if not quiet:
            print(f"  {label} could not be launched: {error}", file=sys.stderr)
        return False
    if result.returncode == 0:
        out = result.stdout.strip()
        if out and not quiet:
            print(f"  {out}")
        return True
    if not quiet:
        print(f"  {label} failed: {result.stderr.strip()}", file=sys.stderr)
    return False


def _bundled_bench_bin() -> str | None:
    """Return the ax-engine-bench bundled in the installed ax_engine wheel, if present.

    When this script runs from a pip-installed wheel it lives at
    ``site-packages/scripts/download_model.py`` and the binary is staged alongside the
    package at ``site-packages/ax_engine/_bin/ax-engine-bench``. Preferring it over a
    bare PATH lookup avoids picking up a stale ax-engine-bench from an unrelated install
    (e.g. an old cargo-installed binary that cannot handle newer model types).
    """
    candidate = Path(__file__).resolve().parent.parent / "ax_engine" / "_bin" / "ax-engine-bench"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _try_generate_manifest(dest: Path, *, quiet: bool = False, force: bool = False) -> bool:
    """Try bundled, installed, and source-checkout manifest generators. Returns True on success."""
    manifest_path = dest / MODEL_MANIFEST_FILE
    if manifest_path.is_symlink():
        # HF snapshot entries commonly point into the shared content-addressed
        # blob store. A generator opening the symlink for writing would corrupt
        # every snapshot that references that blob, so break the link — but
        # keep the current content as a regular file so a run where every
        # generator is unavailable does not destroy the only manifest.
        try:
            preserved = manifest_path.read_bytes()
        except OSError:
            preserved = None
        manifest_path.unlink()
        if preserved is not None:
            manifest_path.write_bytes(preserved)
    # Keep relative destinations beginning with "-" from being interpreted as
    # generator options, without resolving any directory symlinks.
    manifest_dest = os.path.abspath(dest)
    # Always re-read through NativeModelArtifacts::from_dir after generate so
    # incomplete / family-mismatched manifests cannot be reported as ready.
    force_args = ["--force"] if force else []
    validate_args = ["--validate"]
    if (bundled := _bundled_bench_bin()) is not None:
        command = [bundled, "generate-manifest", *force_args, *validate_args, manifest_dest]
        if _run_manifest_command(
            command,
            quiet=quiet,
            label="bundled ax-engine-bench generate-manifest",
        ):
            return True

    if shutil.which("ax-engine-bench"):
        command = [
            "ax-engine-bench",
            "generate-manifest",
            *force_args,
            *validate_args,
            manifest_dest,
        ]
        if _run_manifest_command(command, quiet=quiet, label="ax-engine-bench generate-manifest"):
            return True

    for local_bin in (
        REPO_ROOT / "target" / "release" / "generate-manifest",
        REPO_ROOT / "target" / "debug" / "generate-manifest",
    ):
        if local_bin.is_file() and _run_manifest_command(
            [str(local_bin), *force_args, *validate_args, manifest_dest],
            quiet=quiet,
            label=str(local_bin),
        ):
            return True

    if shutil.which("cargo"):
        return _run_manifest_command(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "ax-engine-core",
                "--bin",
                "generate-manifest",
                "--",
                *force_args,
                *validate_args,
                manifest_dest,
            ],
            quiet=quiet,
            cwd=REPO_ROOT,
            label="cargo run generate-manifest",
        )

    return False


def _print_manifest_hint(dest: Path) -> None:
    print(
        "\nManifest generation not available automatically. Run manually:\n"
        f"  ax-engine-bench generate-manifest {dest}\n"
        "or (from source):\n"
        f"  cargo run -p ax-engine-core --bin generate-manifest -- {dest}\n"
        "\nThen start the server:\n"
        f"  ax-engine-server --mlx --mlx-model-artifacts-dir {dest} --port 31418"
    )


def _validation_errors(dest: Path) -> list[str]:
    errors = []
    safetensors = list(dest.glob("*.safetensors"))
    if not safetensors:
        errors.append(f"no .safetensors files found in {dest}")
    else:
        errors.extend(
            error for path in safetensors if (error := _safetensors_file_error(path)) is not None
        )
    index_path = dest / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"unable to read safetensors index {index_path}: {error}")
        else:
            if not isinstance(index_payload, dict):
                errors.append(f"invalid safetensors index {index_path}: root must be an object")
            else:
                weight_map = index_payload.get("weight_map", {})
                if not isinstance(weight_map, dict):
                    errors.append(f"invalid weight_map in safetensors index {index_path}")
                else:
                    expected = {
                        filename for filename in weight_map.values() if isinstance(filename, str)
                    }
                    present = {path.name for path in safetensors if path.is_file()}
                    # Some MLX community conversions retain the source checkpoint's
                    # stale index while publishing a differently sharded artifact.
                    # Enforce an index only when it refers to at least one local shard.
                    if expected & present:
                        missing = sorted(expected - present)
                        errors.extend(
                            f"missing safetensors shard {filename} in {dest}"
                            for filename in missing
                        )
    config_path = dest / "config.json"
    if not config_path.exists():
        errors.append(f"config.json missing in {dest}")
    else:
        try:
            config = json.loads(config_path.read_bytes())
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"unable to read config.json in {dest}: {error}")
        else:
            if not isinstance(config, dict):
                errors.append(f"invalid config.json in {dest}: root must be an object")
    return errors


def _manifest_rebuild_plan(dest: Path) -> tuple[bool, bool]:
    """Return (rebuild_needed, force) for regenerating ``dest``'s manifest.

    ``force`` is set when an existing manifest must be replaced rather than
    created, so the generator overwrites it.
    """
    manifest_path = dest / MODEL_MANIFEST_FILE
    rebuild_manifest = _manifest_needs_rebuild(dest)
    replace_invalid_manifest = manifest_path.exists() and rebuild_manifest
    rebuild_media_manifest = (
        not rebuild_manifest and manifest_path.exists() and manifest_needs_media_rebuild(dest)
    )
    return (
        rebuild_manifest or rebuild_media_manifest,
        replace_invalid_manifest or rebuild_media_manifest,
    )


def _prepare_staged_destination(
    dest: Path,
    *,
    quiet: bool,
    progress_json: bool,
) -> None:
    """Make a staged explicit copy AX-ready before it replaces an old one."""
    errors = _validation_errors(dest)
    if errors:
        raise RuntimeError("staged model is invalid: " + "; ".join(errors))

    rebuild_needed, force_rebuild = _manifest_rebuild_plan(dest)
    if not rebuild_needed:
        return

    if not quiet:
        print("  generating manifest in staged destination...")
    if progress_json:
        _emit_progress(90, 100, "Generating manifest")
    if not _try_generate_manifest(
        dest,
        quiet=quiet,
        force=force_rebuild,
    ):
        raise RuntimeError(
            "model manifest is missing or invalid and regeneration failed; "
            "the previous destination was preserved"
        )
    if _manifest_needs_rebuild(dest) or manifest_needs_media_rebuild(dest):
        raise RuntimeError(
            "manifest generator reported success but the staged manifest is still invalid; "
            "the previous destination was preserved"
        )


def _server_command(dest: Path) -> list[str]:
    return [
        "ax-engine-server",
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(dest),
        "--port",
        "31418",
    ]


def _summary(
    repo_id: str,
    dest: Path,
    *,
    revision: str | None,
    status: str,
    errors: list[str] | None = None,
) -> dict:
    manifest_path = dest / MODEL_MANIFEST_FILE
    try:
        manifest_present = manifest_path.exists()
    except OSError:
        manifest_present = False
    try:
        safetensors_count = sum(1 for _ in dest.glob("*.safetensors"))
    except OSError:
        safetensors_count = 0
    try:
        config_present = (dest / "config.json").exists()
    except OSError:
        config_present = False
    return {
        "schema_version": "ax.download_model.v1",
        "repo_id": repo_id,
        "revision": revision,
        "dest": str(dest),
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_present,
        "safetensors_count": safetensors_count,
        "config_present": config_present,
        "status": status,
        "errors": errors or [],
        "server_command": _server_command(dest),
    }


def _print_json(summary: dict) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def _print_json_line(summary: dict) -> None:
    print(json.dumps(summary, sort_keys=True))


def _download_argument_error_summary() -> dict:
    return {
        "schema_version": "ax.download_model.v1",
        "repo_id": None,
        "revision": None,
        "dest": None,
        "manifest_path": None,
        "manifest_present": False,
        "safetensors_count": 0,
        "config_present": False,
        "status": DOWNLOAD_FAILED_STATUS,
        "errors": ["invalid command-line arguments; see stderr for details"],
        "server_command": None,
    }


def main() -> int:
    progress_requested = "--progress-json" in sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Download an MLX model through Hugging Face Hub for ax-engine"
    )
    parser.add_argument("repo_id", help="MLX LLM repo id, e.g. mlx-community/Qwen3-4B-4bit")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: Hugging Face Hub cache snapshot)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch, tag, or commit sha to download (default: the repo's main branch)",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable run summary")
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit newline-delimited progress JSON before the final summary",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Render a live progress bar (bytes, speed, ETA) to stderr while downloading",
    )
    try:
        args = parser.parse_args()
    except SystemExit as error:
        if not progress_requested or error.code in (None, 0):
            raise
        _print_json_line(_download_argument_error_summary())
        return int(error.code)
    machine_json = args.json or args.progress_json
    # --progress-json takes precedence over --json: one NDJSON line, never a
    # second pretty document on the same stream.
    emit_summary = _print_json_line if args.progress_json else _print_json

    dest = args.dest
    repo_id = args.repo_id
    revision = args.revision
    summary_dest = dest or default_mlx_lm_repo_cache_dir(repo_id)

    def prepare_explicit_destination(candidate: Path) -> None:
        _prepare_staged_destination(
            candidate,
            quiet=machine_json,
            progress_json=args.progress_json,
        )

    try:
        repo_id, parsed_revision = _parse_repo_ref(args.repo_id)
        if args.revision is not None:
            _, revision = _parse_repo_ref(f"{repo_id}@{args.revision}")
        else:
            revision = parsed_revision
        summary_dest = dest or default_mlx_lm_repo_cache_dir(repo_id)
        if not machine_json:
            revision_note = f" @ {revision}" if revision else ""
            print(f"\n[{repo_id}{revision_note}]")
        if args.progress_json:
            _emit_progress(0, 100, "Starting Hugging Face Hub download")
        # Pass the raw reference and revision through: download() applies the
        # same single normalization pass as above, so its effective revision
        # matches the one reported in the summary.
        dest = download(
            args.repo_id,
            dest,
            force=args.force,
            revision=args.revision,
            quiet=machine_json,
            progress_json=args.progress_json,
            progress_bar=args.progress_bar,
            prepare_destination=prepare_explicit_destination if dest is not None else None,
        )
    except (RuntimeError, ValueError, OSError, shutil.Error) as error:
        if machine_json:
            summary = _summary(
                repo_id,
                summary_dest,
                revision=revision,
                status=DOWNLOAD_FAILED_STATUS,
                errors=[str(error)],
            )
            emit_summary(summary)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 1

    errors = _validation_errors(dest)
    if errors:
        if machine_json:
            summary = _summary(
                repo_id,
                dest,
                revision=revision,
                status=INVALID_STATUS,
                errors=errors,
            )
            emit_summary(summary)
        else:
            for error in errors:
                print(f"warning: {error}", file=sys.stderr)
        return 1
    if not machine_json:
        print(f"  safetensors shards: {len(list(dest.glob('*.safetensors')))}")

    rebuild_needed, force_rebuild = _manifest_rebuild_plan(dest)
    if rebuild_needed:
        if not machine_json:
            print("  generating manifest...")
        if args.progress_json:
            _emit_progress(90, 100, "Generating manifest")
        if not _try_generate_manifest(
            dest,
            quiet=machine_json,
            force=force_rebuild,
        ):
            if machine_json:
                summary = _summary(
                    repo_id,
                    dest,
                    revision=revision,
                    status=MANIFEST_MISSING_STATUS,
                    errors=["model manifest is missing or invalid and regeneration failed"],
                )
                emit_summary(summary)
            else:
                _print_manifest_hint(dest)
            # The weights downloaded but the model is not AX-ready without a manifest.
            # Return non-zero so automation/CI does not treat this as success.
            return 1

    if _manifest_needs_rebuild(dest) or manifest_needs_media_rebuild(dest):
        error = "manifest generator reported success but the manifest is still invalid"
        if machine_json:
            summary = _summary(
                repo_id,
                dest,
                revision=revision,
                status=MANIFEST_MISSING_STATUS,
                errors=[error],
            )
            emit_summary(summary)
        else:
            print(f"error: {error}: {dest / MODEL_MANIFEST_FILE}", file=sys.stderr)
        return 1

    if machine_json:
        summary = _summary(
            repo_id,
            dest,
            revision=revision,
            status=READY_STATUS,
        )
        if args.progress_json:
            _emit_progress(100, 100, "Ready")
        emit_summary(summary)
    else:
        print(f"\nReady - model artifacts at: {dest}")
        print(f"  ax-engine-server --mlx --mlx-model-artifacts-dir {dest} --port 31418")
    return 0


if __name__ == "__main__":
    sys.exit(main())
