from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Sequence


SERVER_PRESET_ALIASES = {
    "gemma4-e2b": "gemma4-e2b",
    "gemma-4-e2b": "gemma4-e2b",
    "gemma-4-e2b-it": "gemma4-e2b",
    "gemma4-31b": "gemma4-31b",
    "gemma-4-31b": "gemma4-31b",
    "gemma-4-31b-it": "gemma4-31b",
    "glm4.7-flash-4bit": "glm4.7-flash-4bit",
    "glm47-flash-4bit": "glm4.7-flash-4bit",
    "glm4-moe-lite": "glm4.7-flash-4bit",
    "glm4_moe_lite": "glm4.7-flash-4bit",
    "qwen3.6-35b": "qwen3.6-35b",
    "qwen36-35b": "qwen3.6-35b",
    "qwen3-6-35b": "qwen3.6-35b",
    "qwen3.6-35b-a3b": "qwen3.6-35b",
    "qwen36-35b-a3b": "qwen3.6-35b",
}


def _server_bin() -> pathlib.Path | str:
    bundled = pathlib.Path(__file__).parent / "_bin" / "ax-engine-server"
    if bundled.exists():
        return bundled
    return shutil.which("ax-engine-server") or "ax-engine-server"


def server() -> None:
    bin_path = _server_bin()
    os.execvp(str(bin_path), [str(bin_path)] + sys.argv[1:])


def _json_dump(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _normalize_alias(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _find_repo_script(name: str) -> pathlib.Path | None:
    explicit_root = os.environ.get("AX_ENGINE_REPO_ROOT")
    roots: list[pathlib.Path] = []
    if explicit_root:
        roots.append(pathlib.Path(explicit_root))

    cwd = pathlib.Path.cwd()
    roots.extend([cwd, *cwd.parents])

    file_path = pathlib.Path(__file__).resolve()
    roots.extend(file_path.parents)

    for root in roots:
        for candidate in (root / "scripts" / name, root / name):
            if candidate.exists():
                return candidate
    return None


def _strip_remainder_separator(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _serve_argv(args: argparse.Namespace) -> tuple[list[str], dict]:
    server_bin = str(_server_bin())
    target = args.model
    target_path = pathlib.Path(target).expanduser()
    argv = [server_bin, "--host", args.host, "--port", str(args.port)]

    if target_path.exists():
        resolved = {
            "kind": "local_dir",
            "model": str(target_path.resolve()),
        }
        argv.extend(["--mlx", "--mlx-model-artifacts-dir", resolved["model"]])
    else:
        preset = SERVER_PRESET_ALIASES.get(_normalize_alias(target))
        if preset is None:
            raise SystemExit(
                "unknown model alias or missing local directory: "
                f"{target!r}; pass a model directory or one of "
                f"{', '.join(sorted(set(SERVER_PRESET_ALIASES.values())))}"
            )
        resolved = {
            "kind": "preset",
            "model": target,
            "preset": preset,
            "resolution": "hf-cache",
        }
        argv.extend(["--mlx", "--preset", preset, "--resolve-model-artifacts", "hf-cache"])
        if args.hf_cache_root:
            argv.extend(["--hf-cache-root", args.hf_cache_root])

    argv.extend(_strip_remainder_separator(args.extra_server_args))
    return argv, resolved


def _cmd_serve(args: argparse.Namespace) -> int:
    argv, resolved = _serve_argv(args)
    url = f"http://{args.host}:{args.port}"
    plan = {
        "schema_version": "ax.local_serve_plan.v1",
        "command": "serve",
        "input": args.model,
        "resolved": resolved,
        "server": {
            "url": url,
            "argv": argv,
        },
    }

    if args.json:
        _json_dump(plan)
    else:
        print(f"AX Engine server: {url}")
        print("Command:")
        print("  " + " ".join(argv))

    if args.dry_run:
        return 0

    os.execvp(argv[0], argv)
    return 0


def _parse_output_dir(stdout: str, explicit_output: str | None) -> str | None:
    if explicit_output:
        return str(pathlib.Path(explicit_output).expanduser().resolve())
    match = re.search(r"^Sidecar ready at:\s*\n\s*(.+?)\s*$", stdout, re.MULTILINE)
    if match:
        return match.group(1)
    match = re.search(r"^Output dir:\s*(.+?)\s*$", stdout, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def _cmd_convert_mtplx(args: argparse.Namespace) -> int:
    prepare_script = _find_repo_script("prepare_mtp_sidecar.py")
    check_script = _find_repo_script("check_mtp_sidecar_provenance.py")
    if prepare_script is None:
        raise SystemExit("cannot locate scripts/prepare_mtp_sidecar.py")
    if check_script is None:
        raise SystemExit("cannot locate scripts/check_mtp_sidecar_provenance.py")

    prepare_cmd = [
        sys.executable,
        str(prepare_script),
        "--hf-repo",
        args.mtp_source,
        "--base",
        args.base_model,
        "--mtp-depth-max",
        str(args.mtp_depth_max),
        "--group-size",
        str(args.group_size),
    ]
    if args.output:
        prepare_cmd.extend(["--output", args.output])
    if args.quantize is not None:
        prepare_cmd.extend(["--quantize", str(args.quantize)])

    prepare = _run_capture(prepare_cmd)
    if not args.json:
        sys.stdout.write(prepare.stdout)
        sys.stderr.write(prepare.stderr)
    if prepare.returncode != 0:
        if args.json:
            sys.stderr.write(prepare.stderr)
        return prepare.returncode

    output_dir = _parse_output_dir(prepare.stdout, args.output)
    if output_dir is None:
        raise SystemExit("prepare_mtp_sidecar.py succeeded but output dir could not be determined")

    check_cmd = [sys.executable, str(check_script), output_dir, "--json"]
    if args.fair_base_only:
        check_cmd.append("--fair-base-only")
    provenance = _run_capture(check_cmd)
    if not args.json:
        sys.stdout.write(provenance.stdout)
        sys.stderr.write(provenance.stderr)
    if provenance.returncode != 0:
        if args.json:
            sys.stderr.write(provenance.stderr)
        return provenance.returncode

    try:
        provenance_summary = json.loads(provenance.stdout)
    except json.JSONDecodeError:
        provenance_summary = {"raw": provenance.stdout}

    if args.json:
        _json_dump(
            {
                "schema_version": "ax.convert_mtplx.v1",
                "command": "convert-mtplx",
                "base_model": args.base_model,
                "mtp_source": args.mtp_source,
                "output_dir": output_dir,
                "prepare_command": prepare_cmd,
                "provenance_command": check_cmd,
                "provenance": provenance_summary,
            }
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ax-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Launch ax-engine-server for a model")
    serve_parser.add_argument("model", help="Server preset alias or local model artifact directory")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--hf-cache-root", default=None)
    serve_parser.add_argument("--dry-run", action="store_true")
    serve_parser.add_argument("--json", action="store_true")
    serve_parser.set_defaults(func=_cmd_serve)

    convert_parser = subparsers.add_parser(
        "convert-mtplx",
        help="Package a base MLX model with standard HF MTP tensors",
    )
    convert_parser.add_argument("base_model", help="Base MLX model dir or repo id")
    convert_parser.add_argument("--mtp-source", required=True, help="HF repo that ships mtp.* tensors")
    convert_parser.add_argument("--output", default=None)
    convert_parser.add_argument("--quantize", type=int, choices=[4, 8], default=None)
    convert_parser.add_argument("--mtp-depth-max", type=int, default=1)
    convert_parser.add_argument("--group-size", type=int, default=64)
    convert_parser.add_argument("--fair-base-only", action="store_true")
    convert_parser.add_argument("--json", action="store_true")
    convert_parser.set_defaults(func=_cmd_convert_mtplx)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra_server_args: list[str] = []
    if argv and argv[0] == "serve" and "--" in argv:
        separator = argv.index("--")
        extra_server_args = argv[separator + 1 :]
        argv = argv[:separator]

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        args.extra_server_args = extra_server_args
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
