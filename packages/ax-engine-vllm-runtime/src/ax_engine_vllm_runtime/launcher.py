"""Shell-free, profile-driven vLLM launcher."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sysconfig
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

from .a100_wna16_compat import A100_PROFILE_ENV, A100_PROFILE_VALUE
from .profiles import RuntimeProfile

UNLIMITED_OCR_LOGITS_PROCESSOR = (
    "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor"
)

FORMAL_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_MODULES_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_MODULE_LOADING",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "XDG_CACHE_HOME",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
)


@dataclass(frozen=True)
class ServeConfig:
    model_path: str = "baidu/Unlimited-OCR"
    served_model_name: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = None
    max_model_len: int = 32768
    max_num_seqs: int = 4
    max_images_per_prompt: int = 40
    dtype: str = "bfloat16"
    revision: str | None = None
    trust_remote_code: bool = False
    enforce_eager: bool | None = None
    api_key: str | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        if self.served_model_name is not None and not self.served_model_name.strip():
            raise ValueError("served_model_name must be a non-empty string or None")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a non-empty string")
        for name, value, maximum in (
            ("port", self.port, 65535),
            ("tensor_parallel_size", self.tensor_parallel_size, None),
            ("max_model_len", self.max_model_len, None),
            ("max_num_seqs", self.max_num_seqs, None),
            ("max_images_per_prompt", self.max_images_per_prompt, None),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
            if maximum is not None and value > maximum:
                raise ValueError(f"{name} must be <= {maximum}")
        if self.gpu_memory_utilization is not None and (
            not isinstance(self.gpu_memory_utilization, Real)
            or isinstance(self.gpu_memory_utilization, bool)
            or not math.isfinite(float(self.gpu_memory_utilization))
            or not 0 < float(self.gpu_memory_utilization) <= 1
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.dtype not in {"auto", "bfloat16", "float16"}:
            raise ValueError("dtype must be one of: auto, bfloat16, float16")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision must be a non-empty string or None")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("api_key must be a non-empty string or None")
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("trust_remote_code must be a boolean")
        if self.enforce_eager is not None and not isinstance(self.enforce_eager, bool):
            raise TypeError("enforce_eager must be a boolean or None")


def _json_argument(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_vllm_command(config: ServeConfig, profile: RuntimeProfile) -> list[str]:
    config.validate()
    gpu_memory_utilization = (
        profile.gpu_memory_utilization
        if config.gpu_memory_utilization is None
        else float(config.gpu_memory_utilization)
    )
    command = [
        "vllm",
        "serve",
        config.model_path,
        "--served-model-name",
        config.served_model_name or config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--limit-mm-per-prompt",
        json.dumps({"image": config.max_images_per_prompt}, separators=(",", ":")),
        "--dtype",
        config.dtype,
        "--model-impl",
        "vllm",
        "--logits-processors",
        UNLIMITED_OCR_LOGITS_PROCESSOR,
        "--generation-config",
        "vllm",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
    ]
    if config.trust_remote_code:
        command.append("--trust-remote-code")
    if config.enforce_eager is True or (config.enforce_eager is None and profile.enforce_eager):
        command.append("--enforce-eager")
    for name, value in (
        ("--attention-config", profile.attention_config),
        ("--compilation-config", profile.compilation_config),
        ("--kernel-config", profile.kernel_config),
    ):
        if value is not None:
            command.extend([name, _json_argument(value)])
    if profile.mm_encoder_attn_backend is not None:
        command.extend(["--mm-encoder-attn-backend", profile.mm_encoder_attn_backend])
    if profile.moe_backend is not None:
        command.extend(["--moe-backend", profile.moe_backend])
    revision = config.revision
    if (
        revision is None
        and config.model_path == "baidu/Unlimited-OCR"
        and profile.model_revision is not None
    ):
        revision = profile.model_revision
    if revision:
        command.extend(["--revision", revision])
    return command


def _cuda_package_paths() -> tuple[Path, Path]:
    purelib = Path(sysconfig.get_paths()["purelib"])
    cuda_package = purelib / "nvidia" / "cu13"
    return cuda_package / "bin" / "ptxas", cuda_package / "lib"


def build_vllm_environment(
    config: ServeConfig,
    profile: RuntimeProfile,
    *,
    base_environment: Mapping[str, str] | None = None,
    formal_strict: bool = False,
) -> dict[str, str]:
    config.validate()
    source = dict(os.environ if base_environment is None else base_environment)
    if formal_strict or profile.profile_id.endswith("a100-sm80-wna16"):
        environment = {
            name: value for name, value in source.items() if name in FORMAL_ENVIRONMENT_ALLOWLIST
        }
    else:
        environment = source
    environment.setdefault("PATH", os.defpath)
    environment["VLLM_NO_USAGE_STATS"] = "1"
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    environment["VLLM_PLUGINS"] = profile.plugin or ""
    if config.api_key is not None:
        environment["VLLM_API_KEY"] = config.api_key
    else:
        environment.pop("VLLM_API_KEY", None)

    if profile.profile_id.endswith("a100-sm80-wna16"):
        environment[A100_PROFILE_ENV] = A100_PROFILE_VALUE
        environment["PYTHONNOUSERSITE"] = "1"
    else:
        environment.pop(A100_PROFILE_ENV, None)

    if profile.architecture == "aarch64":
        ptxas, cuda_lib = _cuda_package_paths()
        if not ptxas.is_file() or not os.access(ptxas, os.X_OK):
            raise RuntimeError(f"Thor profile requires executable CUDA 13 ptxas at {ptxas}")
        if not cuda_lib.is_dir():
            raise RuntimeError(f"Thor profile requires CUDA 13 library directory at {cuda_lib}")
        environment["TRITON_PTXAS_BLACKWELL_PATH"] = str(ptxas)
        current = environment.get("LD_LIBRARY_PATH", "")
        entries = [str(cuda_lib), *(part for part in current.split(":") if part)]
        environment["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(entries))
    elif profile.profile_id.endswith("a100-sm80-wna16"):
        _, cuda_lib = _cuda_package_paths()
        if not cuda_lib.is_dir():
            raise RuntimeError(f"A100 profile requires CUDA 13 library directory at {cuda_lib}")
        environment["LD_LIBRARY_PATH"] = str(cuda_lib)
        environment.pop("TRITON_PTXAS_BLACKWELL_PATH", None)
    return environment


def run_vllm_server(
    config: ServeConfig,
    profile: RuntimeProfile,
    *,
    runner: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
) -> int:
    if shutil.which("vllm") is None:
        raise RuntimeError(
            "vLLM executable not found; install ax-engine-vllm-runtime "
            "on a supported Linux CUDA host"
        )
    completed = runner(
        build_vllm_command(config, profile),
        check=False,
        env=build_vllm_environment(config, profile),
    )
    return int(completed.returncode)
