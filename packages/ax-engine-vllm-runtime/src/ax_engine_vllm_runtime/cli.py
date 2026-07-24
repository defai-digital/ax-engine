"""Operator CLI for the independent AX Engine vLLM worker."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import stat
import sys
from pathlib import Path

from .launcher import (
    ServeConfig,
    build_vllm_command,
    build_vllm_environment,
    run_vllm_server,
)
from .preflight import run_preflight
from .profiles import get_profile, profiles

DEFAULT_API_KEY_ENV = "AX_VLLM_API_KEY"
DEFAULT_API_KEY_FILE_ENV = "AX_VLLM_API_KEY_FILE"
MAX_SECRET_BYTES = 16 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ax-engine-vllm-runtime")
    parser.add_argument("--profile")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-images-per-prompt", type=int, default=40)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", default=None)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--allow-public-bind", action="store_true")
    return parser


def _read_secret_file(path: Path) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"API key path {path} must not be a symbolic link")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"API key path {path} is not a regular file")
    if metadata.st_size > MAX_SECRET_BYTES:
        raise ValueError(f"API key file {path} exceeds {MAX_SECRET_BYTES} bytes")
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("API key file must contain one non-empty line")
    return value


def _resolve_api_key(
    env_name: str,
    explicit_file: Path | None,
) -> str | None:
    env_name = env_name.strip()
    if not env_name or not env_name.replace("_", "a").isalnum() or env_name[0].isdigit():
        raise ValueError("--api-key-env must be a valid environment variable name")
    env_secret = os.environ.get(env_name, "").strip() or None
    file_from_env = os.environ.get(DEFAULT_API_KEY_FILE_ENV, "").strip() or None
    if explicit_file is not None and file_from_env is not None:
        raise ValueError(f"--api-key-file conflicts with populated {DEFAULT_API_KEY_FILE_ENV}")
    path = explicit_file or (Path(file_from_env) if file_from_env else None)
    if env_secret is not None and path is not None:
        raise ValueError("environment and file API key sources are mutually exclusive")
    return env_secret if env_secret is not None else (_read_secret_file(path) if path else None)


def _is_loopback(host: str) -> bool:
    value = host.strip().strip("[]")
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _emit(payload: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_profiles:
        _emit([profile.as_dict() for profile in profiles()], args.json)
        return 0
    if not args.profile:
        _parser().error("--profile is required unless --list-profiles is used")
    try:
        profile = get_profile(args.profile)
        api_key = _resolve_api_key(args.api_key_env, args.api_key_file)
        if not _is_loopback(args.host) and (not args.allow_public_bind or api_key is None):
            raise ValueError(
                "non-loopback bind requires --allow-public-bind and an API key env/file source"
            )
        config = ServeConfig(
            model_path=args.model,
            served_model_name=args.served_model_name,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_images_per_prompt=args.max_images_per_prompt,
            dtype=args.dtype,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            api_key=api_key,
        )
        report = run_preflight(profile)
        if args.check_only:
            _emit(report.as_dict(), args.json)
            return 0 if report.ready else 2
        if args.dry_run:
            environment = build_vllm_environment(config, profile)
            _emit(
                {
                    "profile_id": profile.profile_id,
                    "preflight_ready": report.ready,
                    "command": build_vllm_command(config, profile),
                    "credential_configured": api_key is not None,
                    "environment": {
                        "VLLM_NO_USAGE_STATS": environment["VLLM_NO_USAGE_STATS"],
                        "VLLM_PLUGINS": environment["VLLM_PLUGINS"],
                        "VLLM_USE_FLASHINFER_SAMPLER": environment["VLLM_USE_FLASHINFER_SAMPLER"],
                    },
                },
                args.json,
            )
            return 0
        if not report.ready:
            _emit(report.as_dict(), args.json)
            return 2
        return run_vllm_server(config, profile)
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        print(f"ax-engine-vllm-runtime: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
