from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

_CUSPARSELT_DISTRIBUTION = "nvidia-cusparselt-cu13"
_CUSPARSELT_VERSION = "0.8.0"
_CUSPARSELT_WHEEL_TAG = "Tag: py3-none-manylinux2014_sbsa"
_CUSPARSELT_LIBRARY = Path("nvidia/cusparselt/lib/libcusparseLt.so.0")
_CUSPARSELT_PLATFORM_EXCEPTION = (
    "The package `nvidia-cusparselt-cu13` was built for a different platform"
)
_ELF_MACHINE_AARCH64 = 183


def _uv_incompatibilities(output: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in output.splitlines() if line.startswith("The package `"))


def _requires_verified_cusparselt_exception(return_code: int, output: str) -> bool:
    if return_code == 0:
        if _uv_incompatibilities(output):
            raise RuntimeError("uv reported incompatibilities with a successful exit status")
        return False
    if return_code != 1:
        raise RuntimeError(f"uv pip check exited with unexpected status {return_code}")

    incompatibilities = _uv_incompatibilities(output)
    if incompatibilities != (_CUSPARSELT_PLATFORM_EXCEPTION,):
        rendered = "\n".join(incompatibilities) or "<no parseable incompatibility>"
        raise RuntimeError(f"uv pip check found an unapproved incompatibility:\n{rendered}")
    return True


def _elf_machine(path: Path) -> int:
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise RuntimeError(f"{path} is not an ELF binary")
    if header[4] != 2 or header[5] != 1:
        raise RuntimeError(f"{path} is not a little-endian ELF64 binary")
    return int.from_bytes(header[18:20], byteorder="little")


def _validate_cusparselt_exception(
    *,
    architecture: str,
    version: str,
    wheel_metadata: str,
    elf_machine: int,
) -> None:
    if architecture not in {"aarch64", "arm64"}:
        raise RuntimeError("cuSPARSELt SBSA exception is only valid on Linux arm64")
    if version != _CUSPARSELT_VERSION:
        raise RuntimeError(f"cuSPARSELt SBSA exception is not approved for version {version!r}")
    tags = tuple(line.strip() for line in wheel_metadata.splitlines() if line.startswith("Tag:"))
    if tags != (_CUSPARSELT_WHEEL_TAG,):
        raise RuntimeError(f"unexpected cuSPARSELt wheel tags: {tags!r}")
    if elf_machine != _ELF_MACHINE_AARCH64:
        raise RuntimeError(f"cuSPARSELt binary has ELF machine {elf_machine}, expected AArch64")


def _verified_cusparselt_evidence() -> dict[str, Any]:
    distribution = importlib.metadata.distribution(_CUSPARSELT_DISTRIBUTION)
    wheel_metadata = distribution.read_text("WHEEL")
    if wheel_metadata is None:
        raise RuntimeError("cuSPARSELt distribution has no WHEEL metadata")
    library = Path(distribution.locate_file(_CUSPARSELT_LIBRARY))
    machine = _elf_machine(library)
    architecture = platform.machine()
    _validate_cusparselt_exception(
        architecture=architecture,
        version=distribution.version,
        wheel_metadata=wheel_metadata,
        elf_machine=machine,
    )
    return {
        "distribution": _CUSPARSELT_DISTRIBUTION,
        "version": distribution.version,
        "architecture": architecture,
        "wheel_tag": _CUSPARSELT_WHEEL_TAG.removeprefix("Tag: "),
        "elf_machine": machine,
        "library": str(_CUSPARSELT_LIBRARY),
        "reason": (
            "NVIDIA's aarch64 wheel filename is standards-compatible, but its internal "
            "WHEEL tag uses the vendor SBSA spelling; the installed shared object was "
            "independently verified as AArch64."
        ),
    }


def run_dependency_check(uv: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(uv), "pip", "check", "--python", sys.executable],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not _requires_verified_cusparselt_exception(result.returncode, result.stdout):
        return {"status": "pass", "checker": "uv pip check", "exceptions": []}
    return {
        "status": "pass_with_verified_exception",
        "checker": "uv pip check",
        "exceptions": [_verified_cusparselt_evidence()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed dependency closure check for the AX Engine vLLM image"
    )
    parser.add_argument("--uv", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_dependency_check(args.uv)
    except Exception as error:
        raise SystemExit(f"dependency closure check failed: {error}") from error
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
