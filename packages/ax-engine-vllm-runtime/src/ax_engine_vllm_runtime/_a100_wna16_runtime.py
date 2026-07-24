"""Exact, fail-closed vLLM override for the validated A100 WNA16 profile.

This private module deliberately imports vLLM lazily.  The public plugin first
attests the host and package environment; this module then attests the precise
upstream implementation before replacing one module-local predicate alias.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ax_engine_vllm_runtime.a100_wna16_compat import (
    A100_ATTESTATION_PREFIX,
    A100_COMPRESSED_TENSORS_MOE_MODULE,
    A100_COMPRESSED_TENSORS_MOE_SHA256,
    A100_KERNEL_CONFIG,
    A100_MARLIN_PREDICATE_NAME,
    A100_MARLIN_PREDICATE_SOURCE_SHA256,
    A100_MARLIN_UTILS_MODULE,
    A100_MARLIN_UTILS_SHA256,
    A100_PATCHED_ALIAS,
    A100_PROFILE_ENV,
    A100_PROFILE_VALUE,
)

_VLLM_CONFIG_MODULE = "vllm.config"

_TARGET_MODEL_TYPE = "unlimited-ocr"
_TARGET_ARCHITECTURE = "UnlimitedOCRForCausalLM"
_TARGET_LAYER_NAMES = frozenset(
    f"language_model.model.layers.{index}.mlp.experts" for index in range(1, 12)
)
logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL: Callable[..., Any] | None = None
_WRAPPER: Callable[..., Any] | None = None

_MISSING = object()


def _sha256_regular_file(path: Path) -> str:
    """Hash one non-symlink regular file without following its final path."""
    if not path.is_absolute():
        raise RuntimeError(f"Attested vLLM source path is not absolute: {path}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("This platform cannot attest vLLM source symlinks")
    flags = os.O_RDONLY | nofollow
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"Could not open attested vLLM source: {path}") from exc
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"Attested vLLM source is not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"Could not read attested vLLM source: {path}") from exc
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _module_source_path(module: Any, expected_name: str) -> Path:
    if getattr(module, "__name__", None) != expected_name:
        raise RuntimeError(f"Unexpected vLLM module identity for {expected_name}")
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"vLLM module {expected_name} has no source file")
    path = Path(value)
    if path.suffix != ".py":
        raise RuntimeError(f"vLLM module {expected_name} did not load from Python source")
    return path


def _callable_source_sha256(function: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError("Could not attest the upstream Marlin predicate source") from exc
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _validate_predicate(function: Any) -> Callable[..., Any]:
    if not callable(function):
        raise RuntimeError("The upstream Marlin support predicate is not callable")
    if (
        getattr(function, "__module__", None) != A100_MARLIN_UTILS_MODULE
        or getattr(function, "__name__", None) != A100_MARLIN_PREDICATE_NAME
    ):
        raise RuntimeError("The upstream Marlin support predicate identity changed")
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Could not inspect the upstream Marlin predicate") from exc
    parameters = list(signature.parameters.values())
    expected = (
        ("layer", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        (
            "group_size",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
        ),
        ("allow_tile_padding", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    )
    actual = tuple((item.name, item.kind, item.default) for item in parameters)
    if actual != expected:
        raise RuntimeError("The upstream Marlin support predicate signature changed")
    if _callable_source_sha256(function) != A100_MARLIN_PREDICATE_SOURCE_SHA256:
        raise RuntimeError("The upstream Marlin support predicate source changed")
    return function


def _validate_upstream_sources(compressed_tensors_moe: Any, marlin_utils: Any) -> None:
    sources = (
        (
            compressed_tensors_moe,
            A100_COMPRESSED_TENSORS_MOE_MODULE,
            A100_COMPRESSED_TENSORS_MOE_SHA256,
        ),
        (marlin_utils, A100_MARLIN_UTILS_MODULE, A100_MARLIN_UTILS_SHA256),
    )
    for module, name, expected_digest in sources:
        path = _module_source_path(module, name)
        if _sha256_regular_file(path) != expected_digest:
            raise RuntimeError(f"Attested vLLM source hash changed for {name}")


def _require_profile() -> None:
    if os.environ.get(A100_PROFILE_ENV) != A100_PROFILE_VALUE:
        raise RuntimeError(f"A100 runtime requires {A100_PROFILE_ENV}={A100_PROFILE_VALUE!r}")


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        result = value.get(name, _MISSING)
    else:
        result = getattr(value, name, _MISSING)
    if result is _MISSING:
        if default is not _MISSING:
            return default
        raise RuntimeError(f"A100 runtime configuration is missing {name}")
    return result


def _require_equal(value: Any, name: str, expected: Any) -> None:
    actual = _field(value, name)
    if isinstance(expected, bool):
        matches = actual is expected
    elif isinstance(expected, int):
        matches = isinstance(actual, int) and not isinstance(actual, bool)
        matches = matches and actual == expected
    else:
        matches = actual == expected
    if not matches:
        raise RuntimeError(f"A100 runtime requires {name}={expected!r}; found {actual!r}")


def _require_bfloat16(value: Any, name: str) -> None:
    normalized = str(value).strip().lower()
    if normalized not in {"bfloat16", "torch.bfloat16"}:
        raise RuntimeError(f"A100 runtime requires {name}=bfloat16; found {value!r}")


def _model_identity(hf_config: Any) -> tuple[Any, tuple[Any, ...]]:
    model_type = _field(hf_config, "model_type", None)
    architectures = _field(hf_config, "architectures", ())
    if not isinstance(architectures, (list, tuple)):
        architectures = ()
    return model_type, tuple(architectures)


def _is_target_model(hf_config: Any) -> bool:
    model_type, architectures = _model_identity(hf_config)
    return model_type == _TARGET_MODEL_TYPE or _TARGET_ARCHITECTURE in architectures


def _validate_target_model(vllm_config: Any, layer: Any) -> str:
    model_config = _field(vllm_config, "model_config")
    hf_config = _field(model_config, "hf_config")
    model_type, architectures = _model_identity(hf_config)
    if model_type != _TARGET_MODEL_TYPE:
        raise RuntimeError("A100 runtime target model_type changed")
    if architectures != (_TARGET_ARCHITECTURE,):
        raise RuntimeError("A100 runtime target architectures changed")

    text_config = _field(hf_config, "text_config")
    for name, expected in (
        ("hidden_size", 1280),
        ("intermediate_size", 6848),
        ("moe_intermediate_size", 896),
        ("n_routed_experts", 64),
        ("n_shared_experts", 2),
        ("num_experts_per_tok", 6),
        ("num_hidden_layers", 12),
        ("first_k_dense_replace", 1),
        ("max_position_embeddings", 32768),
    ):
        _require_equal(text_config, name, expected)
    _require_equal(hf_config, "moe_layer_freq", 1)

    _require_equal(model_config, "max_model_len", 32768)
    _require_equal(model_config, "quantization", "compressed_tensors")
    _require_bfloat16(_field(model_config, "dtype"), "model dtype")
    if _field(vllm_config, "lora_config", None) is not None:
        raise RuntimeError("A100 runtime requires LoRA to be disabled")

    parallel_config = _field(vllm_config, "parallel_config")
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
    ):
        _require_equal(parallel_config, name, 1)
    _require_equal(parallel_config, "enable_expert_parallel", False)

    kernel_config = _field(vllm_config, "kernel_config")
    for name, expected in A100_KERNEL_CONFIG.items():
        _require_equal(kernel_config, name, expected)

    layer_name = _field(layer, "layer_name")
    if layer_name not in _TARGET_LAYER_NAMES:
        raise RuntimeError(f"A100 runtime received unexpected target layer {layer_name!r}")
    for name, expected in (
        ("hidden_size", 1280),
        ("global_num_experts", 64),
        ("local_num_experts", 64),
        ("top_k", 6),
    ):
        _require_equal(layer, name, expected)
    _require_equal(layer, "apply_router_weight_on_input", False)
    _require_bfloat16(_field(layer, "params_dtype"), "layer params_dtype")
    moe_config = _field(layer, "moe_config")
    _require_equal(moe_config, "intermediate_size_per_partition_unpadded", 896)
    return layer_name


def _emit_event(event: str, **values: Any) -> None:
    payload = {
        "event": event,
        "pid": os.getpid(),
        "profile": A100_PROFILE_VALUE,
        **values,
    }
    logger.warning(
        "%s%s",
        A100_ATTESTATION_PREFIX,
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
    )


def _build_wrapper(
    original: Callable[..., Any], get_current_vllm_config: Callable[[], Any]
) -> Callable[..., Any]:
    @functools.wraps(original)
    def prefer_generic_triton_moe(
        layer: Any,
        group_size: int,
        allow_tile_padding: bool = False,
    ) -> Any:
        _require_profile()
        vllm_config = get_current_vllm_config()
        model_config = _field(vllm_config, "model_config")
        hf_config = _field(model_config, "hf_config")
        if not _is_target_model(hf_config):
            return original(layer, group_size, allow_tile_padding)

        layer_name = _validate_target_model(vllm_config, layer)
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise RuntimeError("A100 runtime requires WNA16 group_size=128")
        if group_size != 128:
            raise RuntimeError("A100 runtime requires WNA16 group_size=128")
        if allow_tile_padding is not True:
            raise RuntimeError("A100 runtime requires allow_tile_padding=True")

        supported = original(layer, group_size, allow_tile_padding)
        if supported is not True:
            raise RuntimeError("A100 runtime expected the pinned Marlin predicate to return True")
        _emit_event(
            "moe_backend_override",
            allow_tile_padding=True,
            forced_backend="generic_triton",
            group_size=group_size,
            layer=layer_name,
            original_backend="marlin",
            original_supported=True,
        )
        return False

    return prefer_generic_triton_moe


def install() -> None:
    """Install the exact module-local WNA16 predicate override once."""
    global _INSTALLED, _ORIGINAL, _WRAPPER

    _require_profile()
    compressed_tensors_moe = importlib.import_module(A100_COMPRESSED_TENSORS_MOE_MODULE)
    marlin_utils = importlib.import_module(A100_MARLIN_UTILS_MODULE)
    vllm_config_module = importlib.import_module(_VLLM_CONFIG_MODULE)

    if _INSTALLED:
        if (
            _WRAPPER is None
            or _ORIGINAL is None
            or getattr(compressed_tensors_moe, A100_MARLIN_PREDICATE_NAME, None) is not _WRAPPER
            or getattr(marlin_utils, A100_MARLIN_PREDICATE_NAME, None) is not _ORIGINAL
        ):
            raise RuntimeError("A100 runtime predicate alias changed after installation")
        return

    _validate_upstream_sources(compressed_tensors_moe, marlin_utils)
    original = _validate_predicate(getattr(marlin_utils, A100_MARLIN_PREDICATE_NAME, None))
    if getattr(compressed_tensors_moe, A100_MARLIN_PREDICATE_NAME, None) is not original:
        raise RuntimeError("The compressed-tensors MoE predicate alias was pre-patched")
    get_current_vllm_config = getattr(vllm_config_module, "get_current_vllm_config", None)
    if not callable(get_current_vllm_config):
        raise RuntimeError("vLLM get_current_vllm_config identity changed")

    wrapper = _build_wrapper(original, get_current_vllm_config)
    setattr(compressed_tensors_moe, A100_MARLIN_PREDICATE_NAME, wrapper)
    _ORIGINAL = original
    _WRAPPER = wrapper
    _INSTALLED = True
    _emit_event(
        "installed",
        compressed_tensors_moe_sha256=A100_COMPRESSED_TENSORS_MOE_SHA256,
        marlin_utils_sha256=A100_MARLIN_UTILS_SHA256,
        patched_alias=A100_PATCHED_ALIAS,
        predicate_source_sha256=A100_MARLIN_PREDICATE_SOURCE_SHA256,
    )
