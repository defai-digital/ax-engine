"""Fail-closed runtime attestation for a selected profile."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

from packaging.version import InvalidVersion, Version

from .profiles import RuntimeProfile, runtime_lock_sha256


@dataclass(frozen=True)
class RuntimeFacts:
    system: str
    architecture: str
    python_implementation: str
    python_version: tuple[int, int]
    vllm_version: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    gpu_count: int
    gpu_name: str | None
    compute_capability: tuple[int, int] | None


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    expected: str
    actual: str


@dataclass(frozen=True)
class PreflightReport:
    profile_id: str
    ready: bool
    facts: RuntimeFacts
    checks: tuple[PreflightCheck, ...]

    def as_dict(self) -> dict[str, Any]:
        facts = asdict(self.facts)
        facts["python_version"] = ".".join(str(part) for part in self.facts.python_version)
        facts["compute_capability"] = (
            None
            if self.facts.compute_capability is None
            else ".".join(str(part) for part in self.facts.compute_capability)
        )
        return {
            "profile_id": self.profile_id,
            "ready": self.ready,
            "facts": facts,
            "checks": [asdict(check) for check in self.checks],
        }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_facts() -> RuntimeFacts:
    torch_version = None
    torch_cuda_version = None
    gpu_count = 0
    gpu_name = None
    compute_capability = None
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        torch = None
    if torch is not None:
        torch_version = str(getattr(torch, "__version__", "")) or None
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            gpu_count = int(cuda.device_count())
            if gpu_count == 1:
                gpu_name = str(cuda.get_device_name(0))
                major, minor = cuda.get_device_capability(0)
                compute_capability = int(major), int(minor)
    return RuntimeFacts(
        system=platform.system(),
        architecture=platform.machine().strip().lower(),
        python_implementation=sys.implementation.name,
        python_version=(sys.version_info.major, sys.version_info.minor),
        vllm_version=_distribution_version("vllm"),
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
    )


def _public_version(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return Version(value).public
    except InvalidVersion:
        return None


def validate_runtime_facts(
    profile: RuntimeProfile,
    facts: RuntimeFacts,
) -> PreflightReport:
    normalized_arch = "aarch64" if facts.architecture == "arm64" else facts.architecture
    try:
        lock_sha256 = runtime_lock_sha256(profile)
    except OSError as exc:
        lock_sha256 = f"unavailable:{type(exc).__name__}"
    checks = (
        PreflightCheck("system", facts.system == "Linux", "Linux", facts.system),
        PreflightCheck(
            "architecture",
            normalized_arch == profile.architecture,
            profile.architecture,
            normalized_arch,
        ),
        PreflightCheck(
            "python_implementation",
            facts.python_implementation == "cpython",
            "cpython",
            facts.python_implementation,
        ),
        PreflightCheck(
            "python_version",
            facts.python_version == profile.python_version,
            ".".join(str(part) for part in profile.python_version),
            ".".join(str(part) for part in facts.python_version),
        ),
        PreflightCheck(
            "vllm_version",
            _public_version(facts.vllm_version) == profile.vllm_version,
            profile.vllm_version,
            facts.vllm_version or "missing",
        ),
        PreflightCheck(
            "torch_version",
            _public_version(facts.torch_version) == profile.torch_version,
            profile.torch_version,
            facts.torch_version or "missing",
        ),
        PreflightCheck(
            "torch_cuda_version",
            facts.torch_cuda_version == profile.torch_cuda_version,
            profile.torch_cuda_version,
            facts.torch_cuda_version or "missing",
        ),
        PreflightCheck(
            "runtime_lock_sha256",
            lock_sha256 == profile.runtime_lock_sha256,
            profile.runtime_lock_sha256,
            lock_sha256,
        ),
        PreflightCheck("gpu_count", facts.gpu_count == 1, "1", str(facts.gpu_count)),
        PreflightCheck(
            "gpu_name",
            facts.gpu_name in profile.gpu_names,
            " or ".join(profile.gpu_names),
            facts.gpu_name or "missing",
        ),
        PreflightCheck(
            "compute_capability",
            facts.compute_capability in profile.compute_capabilities,
            " or ".join(
                ".".join(str(part) for part in capability)
                for capability in profile.compute_capabilities
            ),
            (
                "missing"
                if facts.compute_capability is None
                else ".".join(str(part) for part in facts.compute_capability)
            ),
        ),
    )
    return PreflightReport(
        profile_id=profile.profile_id,
        ready=all(check.passed for check in checks),
        facts=facts,
        checks=checks,
    )


def run_preflight(
    profile: RuntimeProfile,
    *,
    facts: RuntimeFacts | None = None,
) -> PreflightReport:
    return validate_runtime_facts(profile, facts or collect_runtime_facts())
