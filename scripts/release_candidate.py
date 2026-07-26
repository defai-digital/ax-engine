#!/usr/bin/env python3
"""Create and verify immutable AX Engine release-candidate manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.release_candidate.v2"
BINARY_NAMES = ("ax-engine", "ax-engine-server", "ax-engine-bench")
MLX_RUNTIME_NAMES = ("libmlx.dylib", "libjaccl.dylib", "mlx.metallib")
MLX_LICENSE_NAME = "MLX-LICENSE.txt"
TAG_PATTERN = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:[-.][0-9A-Za-z][0-9A-Za-z.-]*)?)$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class CandidateError(ValueError):
    """Report an invalid or incomplete release candidate."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_identity(tag: str, commit: str) -> str:
    tag_match = TAG_PATTERN.fullmatch(tag)
    if tag_match is None:
        raise CandidateError(f"invalid release tag: {tag}")
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise CandidateError(f"invalid git commit: {commit}")
    return tag_match.group("version")


def asset_record(root: Path, path: Path) -> dict[str, object]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise CandidateError(f"asset is outside candidate root: {path}") from exc
    if not resolved_path.is_file():
        raise CandidateError(f"candidate asset is missing: {path}")
    return {
        "path": relative_path.as_posix(),
        "sha256": sha256(resolved_path),
        "size": resolved_path.stat().st_size,
    }


def validate_mlx_version(version: str) -> str:
    if VERSION_PATTERN.fullmatch(version) is None:
        raise CandidateError(f"invalid MLX version: {version}")
    return version


def create_manifest(
    root: Path,
    tag: str,
    commit: str,
    mlx_version: str,
) -> dict[str, object]:
    version = validate_identity(tag, commit)
    mlx_version = validate_mlx_version(mlx_version)
    root = root.resolve()
    binaries = {name: asset_record(root, root / "bin" / name) for name in BINARY_NAMES}
    mlx_runtime = {name: asset_record(root, root / "runtime" / name) for name in MLX_RUNTIME_NAMES}
    mlx_license = asset_record(root, root / "runtime" / MLX_LICENSE_NAME)
    wheels = sorted((root / "wheel").glob("ax_engine-*.whl"))
    if len(wheels) != 1:
        raise CandidateError(f"expected exactly one wheel, found {len(wheels)}")

    return {
        "schema_version": SCHEMA_VERSION,
        "project": "ax-engine",
        "tag": tag,
        "version": version,
        "git_commit": commit,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": "macos-arm64",
        "binaries": binaries,
        "mlx_runtime": {
            "version": mlx_version,
            "assets": mlx_runtime,
            "license": mlx_license,
        },
        "wheel": {
            "name": wheels[0].name,
            **asset_record(root, wheels[0]),
        },
    }


def resolve_manifest_asset(root: Path, record: object) -> Path:
    if not isinstance(record, dict):
        raise CandidateError("candidate asset record must be an object")
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise CandidateError("candidate asset path is missing")
    resolved_root = root.resolve()
    resolved_path = (resolved_root / relative).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise CandidateError(f"candidate asset escapes root: {relative}") from exc
    return resolved_path


def verify_asset(
    root: Path,
    label: str,
    record: object,
    expected_path: str | None = None,
) -> Path:
    path = resolve_manifest_asset(root, record)
    if not path.is_file():
        raise CandidateError(f"candidate {label} is missing: {path}")
    if not isinstance(record, dict):
        raise CandidateError(f"candidate {label} record must be an object")
    if expected_path is not None and record.get("path") != expected_path:
        raise CandidateError(
            f"candidate {label} path mismatch: expected {expected_path!r}, "
            f"got {record.get('path')!r}"
        )
    expected_size = record.get("size")
    expected_sha256 = record.get("sha256")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise CandidateError(f"candidate {label} has an invalid size")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise CandidateError(f"candidate {label} has an invalid sha256")
    if path.stat().st_size != expected_size:
        raise CandidateError(f"candidate {label} size mismatch")
    if sha256(path) != expected_sha256:
        raise CandidateError(f"candidate {label} sha256 mismatch")
    return path


def verify_manifest(
    root: Path,
    manifest: dict[str, object],
    tag: str,
    commit: str,
    mlx_version: str,
    group: str,
) -> list[Path]:
    version = validate_identity(tag, commit)
    mlx_version = validate_mlx_version(mlx_version)
    expected_fields = {
        "schema_version": SCHEMA_VERSION,
        "project": "ax-engine",
        "tag": tag,
        "version": version,
        "git_commit": commit,
        "platform": "macos-arm64",
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise CandidateError(
                f"candidate manifest {field} mismatch: expected {expected!r}, "
                f"got {manifest.get(field)!r}"
            )

    verified: list[Path] = []
    if group in {"all", "binaries", "standalone"}:
        binaries = manifest.get("binaries")
        if not isinstance(binaries, dict):
            raise CandidateError("candidate binaries record is missing")
        if set(binaries) != set(BINARY_NAMES):
            raise CandidateError("candidate binary set does not match the release contract")
        for name in BINARY_NAMES:
            verified.append(verify_asset(root, name, binaries[name], f"bin/{name}"))

    runtime = manifest.get("mlx_runtime")
    if not isinstance(runtime, dict) or runtime.get("version") != mlx_version:
        raise CandidateError(f"candidate manifest MLX version mismatch: expected {mlx_version!r}")

    if group in {"all", "runtime", "standalone"}:
        assets = runtime.get("assets")
        if not isinstance(assets, dict):
            raise CandidateError("candidate MLX runtime assets record is missing")
        if set(assets) != set(MLX_RUNTIME_NAMES):
            raise CandidateError("candidate MLX runtime set does not match the release contract")
        for name in MLX_RUNTIME_NAMES:
            verified.append(verify_asset(root, name, assets[name], f"runtime/{name}"))
        license_path = verify_asset(
            root,
            MLX_LICENSE_NAME,
            runtime.get("license"),
            f"runtime/{MLX_LICENSE_NAME}",
        )
        if license_path.name != MLX_LICENSE_NAME:
            raise CandidateError("candidate MLX license has an unexpected filename")
        verified.append(license_path)

    if group in {"all", "wheel"}:
        wheel = manifest.get("wheel")
        if not isinstance(wheel, dict):
            raise CandidateError("candidate wheel record is missing")
        wheel_name = wheel.get("name")
        if (
            not isinstance(wheel_name, str)
            or Path(wheel_name).name != wheel_name
            or not wheel_name.startswith("ax_engine-")
            or not wheel_name.endswith(".whl")
        ):
            raise CandidateError("candidate wheel has an unexpected filename")
        path = verify_asset(root, "wheel", wheel, f"wheel/{wheel_name}")
        if wheel_name != path.name:
            raise CandidateError("candidate wheel name does not match its path")
        verified.append(path)

    return verified


def load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"could not read candidate manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateError("candidate manifest root must be an object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a candidate manifest")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--tag", required=True)
    create.add_argument("--commit", required=True)
    create.add_argument("--mlx-version", required=True)
    create.add_argument("--output", type=Path)

    verify = subparsers.add_parser("verify", help="verify candidate assets")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)
    verify.add_argument("--tag", required=True)
    verify.add_argument("--commit", required=True)
    verify.add_argument("--mlx-version", required=True)
    verify.add_argument(
        "--group",
        choices=("all", "binaries", "runtime", "standalone", "wheel"),
        default="all",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "create":
            output = args.output or args.root / "candidate-manifest.json"
            manifest = create_manifest(
                args.root,
                args.tag,
                args.commit,
                args.mlx_version,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Created release candidate manifest: {output}")
            return 0

        manifest_path = args.manifest or args.root / "candidate-manifest.json"
        verified = verify_manifest(
            args.root,
            load_manifest(manifest_path),
            args.tag,
            args.commit,
            args.mlx_version,
            args.group,
        )
        for path in verified:
            print(f"Verified release candidate asset: {path}")
        return 0
    except CandidateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
