"""Versioned CUDA runtime profiles.

Platform differences live here and in narrowly guarded compatibility plugins;
they never fork the AX Engine provider or its OpenAI wire DTO.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

PROFILE_SCHEMA_VERSION = 1
PINNED_VLLM_VERSION = "0.25.1"
PINNED_PYTHON = (3, 12)
PINNED_TORCH_VERSION = "2.11.0"
PINNED_TORCH_CUDA = "13.0"
UNLIMITED_OCR_SOURCE_REVISION = "ee63731b6461c8afcdcc7b15352e7d2ffecc2ead"


@dataclass(frozen=True)
class RuntimeProfile:
    schema_version: int
    profile_id: str
    architecture: str
    gpu_names: tuple[str, ...]
    compute_capabilities: tuple[tuple[int, int], ...]
    vllm_version: str
    python_version: tuple[int, int]
    torch_version: str
    torch_cuda_version: str
    runtime_lock: str
    runtime_lock_sha256: str
    model_family: str
    model_revision: str | None
    plugin: str | None
    gpu_memory_utilization: float
    enforce_eager: bool
    attention_config: dict[str, Any] | None = None
    mm_encoder_attn_backend: str | None = None
    moe_backend: str | None = None
    compilation_config: dict[str, Any] | None = None
    kernel_config: dict[str, Any] | None = None
    status: str = "candidate"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["python_version"] = ".".join(str(part) for part in self.python_version)
        value["compute_capabilities"] = [
            ".".join(str(part) for part in capability) for capability in self.compute_capabilities
        ]
        return value


_PROFILES = MappingProxyType(
    {
        "cuda-linux-x86_64-a6000-sm86": RuntimeProfile(
            schema_version=PROFILE_SCHEMA_VERSION,
            profile_id="cuda-linux-x86_64-a6000-sm86",
            architecture="x86_64",
            gpu_names=("NVIDIA RTX A6000",),
            compute_capabilities=((8, 6),),
            vllm_version=PINNED_VLLM_VERSION,
            python_version=PINNED_PYTHON,
            torch_version=PINNED_TORCH_VERSION,
            torch_cuda_version=PINNED_TORCH_CUDA,
            runtime_lock="locks/requirements-runtime-amd64.lock",
            runtime_lock_sha256="e23696449a3bf6a78f58873298b9b32b2f3238da3ec4c31e9ebf07f373174b46",
            model_family="unlimited_ocr",
            model_revision=UNLIMITED_OCR_SOURCE_REVISION,
            plugin=None,
            gpu_memory_utilization=0.90,
            enforce_eager=False,
        ),
        "cuda-linux-aarch64-thor-sm110": RuntimeProfile(
            schema_version=PROFILE_SCHEMA_VERSION,
            profile_id="cuda-linux-aarch64-thor-sm110",
            architecture="aarch64",
            gpu_names=("NVIDIA Thor", "NVIDIA GB10"),
            compute_capabilities=((11, 0),),
            vllm_version=PINNED_VLLM_VERSION,
            python_version=PINNED_PYTHON,
            torch_version=PINNED_TORCH_VERSION,
            torch_cuda_version=PINNED_TORCH_CUDA,
            runtime_lock="locks/requirements-runtime-arm64.lock",
            runtime_lock_sha256="a80ae87d569c795785f315584a7a803ea302faef0b99187a45831ebcb199ce17",
            model_family="unlimited_ocr",
            model_revision=UNLIMITED_OCR_SOURCE_REVISION,
            plugin="ax_engine_vllm_thor_compat",
            gpu_memory_utilization=0.25,
            enforce_eager=True,
            attention_config={"backend": "TRITON_ATTN", "flash_attn_version": 2},
            mm_encoder_attn_backend="TORCH_SDPA",
            moe_backend="emulation",
            compilation_config={"custom_ops": ["none"]},
            kernel_config={
                "ir_op_priority": {
                    "rms_norm": ["native"],
                    "fused_add_rms_norm": ["native"],
                },
                "linear_backend": "emulation",
            },
        ),
        "cuda-linux-x86_64-a100-sm80-wna16": RuntimeProfile(
            schema_version=PROFILE_SCHEMA_VERSION,
            profile_id="cuda-linux-x86_64-a100-sm80-wna16",
            architecture="x86_64",
            gpu_names=("NVIDIA A100-SXM4-80GB",),
            compute_capabilities=((8, 0),),
            vllm_version=PINNED_VLLM_VERSION,
            python_version=PINNED_PYTHON,
            torch_version=PINNED_TORCH_VERSION,
            torch_cuda_version=PINNED_TORCH_CUDA,
            runtime_lock="locks/requirements-runtime-amd64.lock",
            runtime_lock_sha256="e23696449a3bf6a78f58873298b9b32b2f3238da3ec4c31e9ebf07f373174b46",
            model_family="unlimited_ocr",
            model_revision=None,
            plugin="ax_engine_vllm_a100_wna16_compat",
            gpu_memory_utilization=0.90,
            enforce_eager=False,
            kernel_config={"linear_backend": "auto", "moe_backend": "auto"},
        ),
    }
)


def profiles() -> tuple[RuntimeProfile, ...]:
    return tuple(_PROFILES.values())


def get_profile(profile_id: str) -> RuntimeProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        supported = ", ".join(_PROFILES)
        raise ValueError(
            f"unknown AX Engine vLLM runtime profile {profile_id!r}; supported: {supported}"
        ) from exc


def runtime_lock_sha256(profile: RuntimeProfile) -> str:
    """Hash the packaged lock, with a source-tree fallback for editable tests."""

    packaged = resources.files("ax_engine_vllm_runtime").joinpath(profile.runtime_lock)
    if packaged.is_file():
        stream = packaged.open("rb")
    else:
        source_lock = Path(__file__).resolve().parents[2] / profile.runtime_lock
        stream = source_lock.open("rb")
    digest = hashlib.sha256()
    with stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
