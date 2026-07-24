"""Fail-closed vLLM compatibility registration for the validated A100 path.

The plugin is inert unless an explicit profile is requested.  Once requested,
it accepts only the exact native A100-SXM4-80GB and CUDA/Python package stack
used by the release suite before loading the narrowly scoped runtime patch.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

A100_PLUGIN_NAME = "ax_engine_vllm_a100_wna16_compat"
A100_PLUGIN_VALUE = "ax_engine_vllm_runtime.a100_wna16_compat:register"
A100_PROFILE_ENV = "AX_ENGINE_VLLM_A100_WNA16_PROFILE"
A100_PROFILE_VALUE = "marlin-linear-generic-triton-moe-v1"
A100_KERNEL_CONFIG = {
    "linear_backend": "auto",
    "moe_backend": "auto",
}
A100_ATTESTATION_PREFIX = "AX_ENGINE_VLLM_A100_ATTESTATION "

A100_CUDA_CAPABILITY = (8, 0)
A100_GPU_NAME = "NVIDIA A100-SXM4-80GB"
A100_GPU_MEMORY_BYTES = 85_119_205_376
A100_TORCH_CUDA_VERSION = "13.0"
A100_COMPRESSED_TENSORS_MOE_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
A100_MARLIN_UTILS_MODULE = "vllm.model_executor.layers.quantization.utils.marlin_utils"
A100_MARLIN_PREDICATE_NAME = "check_moe_marlin_supports_layer"
A100_PATCHED_ALIAS = f"{A100_COMPRESSED_TENSORS_MOE_MODULE}.{A100_MARLIN_PREDICATE_NAME}"
A100_COMPRESSED_TENSORS_MOE_SHA256 = (
    "36d9f20271434f85f5d2adc8339655511db968d10685e8ab5c78c1f049923837"
)
A100_MARLIN_UTILS_SHA256 = "e09f6cd5869e5d14121f296eafc9bb0fc7b6768c8c6470d1348ffcc84bf2bfd7"
A100_MARLIN_PREDICATE_SOURCE_SHA256 = (
    "dcf54bb8c06545657211a6d79579b549e57937a7eceeaf7b4750b4d7aba0e180"
)

_RUNTIME_DISTRIBUTION_VERSIONS = MappingProxyType(
    {
        "compressed-tensors": "0.17.0",
        "torch": "2.11.0+cu130",
        "triton": "3.6.0",
        "vllm": "0.25.1",
    }
)

_REQUIRED_ENVIRONMENT = MappingProxyType(
    {
        "PYTHONNOUSERSITE": "1",
        "VLLM_PLUGINS": A100_PLUGIN_NAME,
    }
)

_FORBIDDEN_ENVIRONMENT = (
    "CUDA_HOME",
    "CUDA_INCLUDE_PATH",
    "CUDA_NVCC_PATH",
    "CUDA_PATH",
    "GLIBC_TUNABLES",
    "LD_AUDIT",
    "LD_DEBUG",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
)


def _requested_profile() -> bool:
    profile = os.environ.get(A100_PROFILE_ENV)
    if profile in {None, ""}:
        return False
    if profile != A100_PROFILE_VALUE:
        raise RuntimeError(
            f"Unsupported {A100_PROFILE_ENV} value {profile!r}; expected {A100_PROFILE_VALUE!r}"
        )
    return True


def _cuda_runtime() -> Any | None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    return getattr(torch, "cuda", None)


def is_native_linux_x86_64_a100_sm80() -> bool:
    """Return whether this process has the single validated A100 device."""
    if platform.system() != "Linux" or platform.machine().strip().lower() != "x86_64":
        return False
    cuda = _cuda_runtime()
    if cuda is None:
        return False
    try:
        count = cuda.device_count()
        if not isinstance(count, int) or isinstance(count, bool) or count != 1:
            return False
        if not cuda.is_available():
            return False
        name = cuda.get_device_name(0)
        capability = tuple(int(value) for value in cuda.get_device_capability(0))
        properties = cuda.get_device_properties(0)
        memory = getattr(properties, "total_memory", None)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    return (
        name == A100_GPU_NAME
        and capability == A100_CUDA_CAPABILITY
        and isinstance(memory, int)
        and not isinstance(memory, bool)
        and memory == A100_GPU_MEMORY_BYTES
    )


def _validate_runtime_versions() -> None:
    if (
        sys.implementation.name != "cpython"
        or sys.version_info.major != 3
        or sys.version_info.minor != 12
    ):
        raise RuntimeError("A100 WNA16 compatibility requires CPython 3.12")
    for distribution, expected in _RUNTIME_DISTRIBUTION_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"A100 WNA16 compatibility requires {distribution}=={expected}"
            ) from exc
        if actual != expected:
            raise RuntimeError(
                "A100 WNA16 compatibility requires exact runtime distribution "
                f"{distribution}=={expected}; found {actual}"
            )
    torch = importlib.import_module("torch")
    torch_version = getattr(getattr(torch, "version", None), "cuda", None)
    if torch_version != A100_TORCH_CUDA_VERSION:
        raise RuntimeError(
            "A100 WNA16 compatibility requires PyTorch CUDA "
            f"{A100_TORCH_CUDA_VERSION}; found {torch_version!r}"
        )


def _validate_activation_environment() -> None:
    for name, expected in _REQUIRED_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"A100 WNA16 compatibility requires {name}={expected!r}")
    inherited = [name for name in _FORBIDDEN_ENVIRONMENT if os.environ.get(name)]
    if inherited:
        raise RuntimeError(
            "A100 WNA16 compatibility forbids inherited environment: " + ", ".join(inherited)
        )
    library_path = os.environ.get("LD_LIBRARY_PATH")
    if not isinstance(library_path, str) or not library_path:
        raise RuntimeError("A100 WNA16 compatibility requires LD_LIBRARY_PATH")
    library_entries = library_path.split(":")
    if (
        len(library_entries) != 1
        or not Path(library_entries[0]).is_absolute()
        or Path(library_entries[0]).parts[-3:] != ("nvidia", "cu13", "lib")
    ):
        raise RuntimeError(
            "A100 WNA16 compatibility requires the single pinned nvidia/cu13/lib "
            "directory in LD_LIBRARY_PATH"
        )


def _validate_system_preload() -> None:
    preload = Path("/etc/ld.so.preload")
    try:
        value = preload.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("Could not validate /etc/ld.so.preload") from exc
    if value.strip():
        raise RuntimeError("A100 WNA16 compatibility forbids /etc/ld.so.preload")


def register() -> None:
    """Install the exact A100 WNA16 preference override when requested."""
    if not _requested_profile():
        return
    _validate_activation_environment()
    _validate_system_preload()
    if not is_native_linux_x86_64_a100_sm80():
        raise RuntimeError(
            f"A100 WNA16 compatibility requires one native {A100_GPU_NAME} CUDA SM80 device"
        )
    _validate_runtime_versions()
    runtime = importlib.import_module("ax_engine_vllm_runtime._a100_wna16_runtime")
    runtime.install()
