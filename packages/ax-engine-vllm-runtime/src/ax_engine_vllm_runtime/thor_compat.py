"""Version-pinned vLLM compatibility registration for NVIDIA Thor.

The public entry point is loaded by vLLM in every process.  It is intentionally
a no-op everywhere except native Linux/aarch64 CUDA capability 11.0.  On Thor,
the implementation fails closed unless the exact validated vLLM release is
installed, then installs the narrow SM110 runtime fallbacks.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform

from packaging.version import InvalidVersion, Version

THOR_PLUGIN_NAME = "ax_engine_vllm_thor_compat"
THOR_SUPPORTED_VLLM_VERSION = "0.25.1"
THOR_CUDA_CAPABILITY = (11, 0)


def is_supported_vllm_version(value: object) -> bool:
    """Accept only vLLM 0.25.1, with an optional local CUDA build tag."""
    if not isinstance(value, str):
        return False
    try:
        return Version(value).public == THOR_SUPPORTED_VLLM_VERSION
    except InvalidVersion:
        return False


def _native_linux_aarch64() -> bool:
    return platform.system() == "Linux" and platform.machine().strip().lower() in {
        "aarch64",
        "arm64",
    }


def _cuda_device_capability() -> tuple[int, int] | None:
    """Return the active CUDA capability without importing vLLM."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return int(major), int(minor)


def is_native_linux_aarch64_sm110() -> bool:
    """Return whether the active process is on the validated Thor target."""
    return _native_linux_aarch64() and _cuda_device_capability() == THOR_CUDA_CAPABILITY


def register() -> None:
    """Install the exact SM110 fallback when called by vLLM's plugin loader."""
    if not is_native_linux_aarch64_sm110():
        return

    try:
        installed_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("NVIDIA Thor compatibility requires vLLM") from exc
    try:
        installed_public_version = Version(installed_version).public
    except InvalidVersion as exc:
        raise RuntimeError(
            f"NVIDIA Thor compatibility found invalid vLLM version {installed_version!r}"
        ) from exc
    if installed_public_version != THOR_SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            "NVIDIA Thor compatibility was validated only with vLLM "
            f"public version {THOR_SUPPORTED_VLLM_VERSION}; found "
            f"{installed_version}"
        )

    runtime = importlib.import_module("ax_engine_vllm_runtime._thor_runtime")
    runtime.install()
