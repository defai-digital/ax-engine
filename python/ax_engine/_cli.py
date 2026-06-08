from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class ModelProfile:
    label: str
    preset: str | None
    repo_id: str
    aliases: tuple[str, ...]
    downloadable: bool = True


MODEL_PROFILES = (
    ModelProfile(
        label="gemma4-e2b",
        preset="gemma4-e2b",
        repo_id="mlx-community/gemma-4-e2b-it-4bit",
        aliases=("gemma4-e2b", "gemma-4-e2b", "gemma-4-e2b-it", "gemma4-e2b-4bit"),
    ),
    ModelProfile(
        label="gemma4-e2b-5bit",
        preset=None,
        repo_id="mlx-community/gemma-4-e2b-it-5bit",
        aliases=("gemma4-e2b-5bit", "gemma-4-e2b-5bit", "gemma-4-e2b-it-5bit"),
    ),
    ModelProfile(
        label="gemma4-e2b-6bit",
        preset=None,
        repo_id="mlx-community/gemma-4-e2b-it-6bit",
        aliases=("gemma4-e2b-6bit", "gemma-4-e2b-6bit", "gemma-4-e2b-it-6bit"),
    ),
    ModelProfile(
        label="gemma4-e2b-8bit",
        preset=None,
        repo_id="mlx-community/gemma-4-e2b-it-8bit",
        aliases=("gemma4-e2b-8bit", "gemma-4-e2b-8bit", "gemma-4-e2b-it-8bit"),
    ),
    ModelProfile(
        label="gemma4-12b",
        preset="gemma4-12b",
        repo_id="mlx-community/gemma-4-12B-it-4bit",
        aliases=("gemma4-12b", "gemma-4-12b", "gemma-4-12b-it", "gemma4-12b-4bit"),
    ),
    ModelProfile(
        label="gemma4-12b-6bit",
        preset=None,
        repo_id="mlx-community/gemma-4-12B-it-6bit",
        aliases=("gemma4-12b-6bit", "gemma-4-12b-6bit", "gemma-4-12b-it-6bit"),
    ),
    ModelProfile(
        label="gemma4-31b",
        preset="gemma4-31b",
        repo_id="mlx-community/gemma-4-31b-it-4bit",
        aliases=("gemma4-31b", "gemma-4-31b", "gemma-4-31b-it", "gemma4-31b-4bit"),
    ),
    ModelProfile(
        label="glm4.7-flash-4bit",
        preset="glm4.7-flash-4bit",
        repo_id="mlx-community/GLM-4.7-Flash-4bit",
        aliases=(
            "glm4.7-flash-4bit",
            "glm47-flash-4bit",
            "glm4-moe-lite",
            "glm4_moe_lite",
            "glm-4.7-flash-4bit",
            "glm-4-7-flash-4bit",
        ),
        downloadable=False,
    ),
    ModelProfile(
        label="qwen3.6-27b",
        preset=None,
        repo_id="mlx-community/Qwen3.6-27B-4bit",
        aliases=(
            "qwen3.6-27b",
            "qwen36-27b",
            "qwen3-6-27b",
            "qwen3.6-27b-4bit",
            "qwen36-27b-4bit",
        ),
    ),
    ModelProfile(
        label="qwen3.6-27b-5bit",
        preset=None,
        repo_id="mlx-community/Qwen3.6-27B-5bit",
        aliases=(
            "qwen3.6-27b-5bit",
            "qwen36-27b-5bit",
            "qwen3-6-27b-5bit",
        ),
    ),
    ModelProfile(
        label="qwen3.6-27b-6bit",
        preset=None,
        repo_id="mlx-community/Qwen3.6-27B-6bit",
        aliases=(
            "qwen3.6-27b-6bit",
            "qwen36-27b-6bit",
            "qwen3-6-27b-6bit",
        ),
    ),
    ModelProfile(
        label="qwen3.6-27b-8bit",
        preset=None,
        repo_id="mlx-community/Qwen3.6-27B-8bit",
        aliases=(
            "qwen3.6-27b-8bit",
            "qwen36-27b-8bit",
            "qwen3-6-27b-8bit",
        ),
    ),
    ModelProfile(
        label="qwen3.6-35b",
        preset="qwen3.6-35b",
        repo_id="mlx-community/Qwen3.6-35B-A3B-4bit",
        aliases=(
            "qwen3.6-35b",
            "qwen36-35b",
            "qwen3-6-35b",
            "qwen3.6-35b-a3b",
            "qwen36-35b-a3b",
        ),
    ),
)


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


def _profile_for_model(value: str) -> ModelProfile | None:
    normalized = _normalize_alias(value)
    for profile in MODEL_PROFILES:
        if normalized in {_normalize_alias(alias) for alias in profile.aliases}:
            return profile
    return None


def _downloadable_profiles() -> list[ModelProfile]:
    return [profile for profile in MODEL_PROFILES if profile.downloadable]


def _download_options_payload() -> dict:
    return {
        "schema_version": "ax.download_options.v1",
        "default_destination": {
            "kind": "huggingface_hub_cache",
            "env": ["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"],
            "dest_semantics": "--dest copies the resolved snapshot to an explicit directory",
        },
        "targets": [
            {
                "alias": profile.label,
                "repo_id": profile.repo_id,
                "preset": profile.preset,
                "aliases": list(profile.aliases),
            }
            for profile in _downloadable_profiles()
        ],
        "examples": [
            "ax-engine download qwen36-35b",
            "ax-engine download qwen36-27b-8bit",
            "ax-engine download gemma4-12b",
            "ax-engine download gemma4-e2b-6bit",
            "ax-engine download mlx-community/Qwen3.6-35B-A3B-4bit --json",
        ],
    }


def _format_download_options() -> str:
    lines = [
        "Available Qwen3.6 and Gemma 4 MLX download targets:",
    ]
    for profile in _downloadable_profiles():
        lines.append(f"  {profile.label:<20} {profile.repo_id}")
    lines.extend(
        [
            "",
            "Examples:",
            "  ax-engine download qwen36-35b",
            "  ax-engine download qwen36-27b-8bit",
            "  ax-engine download gemma4-12b",
            "  ax-engine download gemma4-e2b-6bit",
            "  ax-engine download mlx-community/Qwen3.6-35B-A3B-4bit --json",
            "",
            "Destination:",
            "  Default: Hugging Face Hub cache shared by mlx-lm and huggingface_hub.",
            "  Override cache location with HF_HUB_CACHE, HF_HOME, or XDG_CACHE_HOME.",
            "  Use --dest only when you want an explicit copy outside the shared cache.",
        ]
    )
    return "\n".join(lines)


SERVER_PRESET_ALIASES = {
    _normalize_alias(alias): profile.preset
    for profile in MODEL_PROFILES
    for alias in profile.aliases
    if profile.preset is not None
}


def _download_repo_id(value: str) -> tuple[str, ModelProfile | None]:
    profile = _profile_for_model(value)
    if profile is not None:
        if not profile.downloadable:
            raise SystemExit(
                f"{profile.label} is not managed by ax-engine download; "
                "use an explicit repo id or one of these targets:\n"
                f"{_format_download_options()}"
            )
        return profile.repo_id, profile
    if "/" in value:
        return value, None
    raise SystemExit(
        f"unknown model alias or repo id: {value!r}; pass a Hugging Face repo id "
        "or one of these targets:\n"
        f"{_format_download_options()}"
    )


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


def _parse_download_summary(stdout: str) -> dict | None:
    if not stdout.strip():
        return None
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("schema_version") == "ax.download_model.v1":
            return parsed
    return None


def _download_summary(
    model: str,
    *,
    dest: str | None = None,
    force: bool = False,
) -> tuple[int, dict | None, str]:
    repo_id, profile = _download_repo_id(model)
    download_script = _find_repo_script("download_model.py")
    if download_script is None:
        raise SystemExit(
            "cannot locate scripts/download_model.py. Reinstall ax-engine or run from a source checkout."
        )

    command = [sys.executable, str(download_script), repo_id, "--json"]
    if dest:
        command.extend(["--dest", dest])
    if force:
        command.append("--force")

    result = _run_capture(command)
    summary = _parse_download_summary(result.stdout)
    if summary is not None:
        summary["input"] = model
        if profile is not None:
            summary["alias"] = profile.label
            if profile.preset is not None:
                summary["preset"] = profile.preset
    return result.returncode, summary, result.stderr


def _print_download_summary(summary: dict) -> None:
    status = summary.get("status", "unknown")
    repo_id = summary.get("repo_id", "unknown")
    dest = summary.get("dest", "")
    print(f"AX Engine model: {repo_id}")
    print(f"Status: {status}")
    if dest:
        print(f"Path: {dest}")
    errors = summary.get("errors") or []
    for error in errors:
        print(f"Error: {error}", file=sys.stderr)
    if status == "ready" and dest:
        print("Next:")
        print(f"  ax-engine serve {dest}")
    elif dest:
        print("Next:")
        print(f"  ax-engine-bench generate-manifest {dest}")


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
        profile = _profile_for_model(target)
        preset = profile.preset if profile is not None else None
        if args.download and not args.dry_run:
            code, summary, stderr = _download_summary(target)
            if code != 0 or summary is None or summary.get("status") != "ready":
                if stderr:
                    sys.stderr.write(stderr)
                if summary is not None:
                    _print_download_summary(summary)
                raise SystemExit(
                    "model download did not produce ready AX artifacts; "
                    f"run: ax-engine download {target}"
                )
            model_dir = str(pathlib.Path(str(summary["dest"])).expanduser().resolve())
            resolved = {
                "kind": "downloaded",
                "model": target,
                "repo_id": summary.get("repo_id"),
                "path": model_dir,
                "download": {
                    "status": summary.get("status"),
                    "manifest_present": summary.get("manifest_present"),
                },
            }
            argv.append("--mlx")
            if preset is not None:
                resolved["preset"] = preset
                argv.extend(["--preset", preset])
            argv.extend(["--mlx-model-artifacts-dir", model_dir])
        elif preset is None:
            download_hint = (
                f" or run: ax-engine serve {target} --download"
                if "/" in target
                else ""
            )
            raise SystemExit(
                "unknown model alias or missing local directory: "
                f"{target!r}; pass a model directory or one of "
                f"{', '.join(sorted(set(SERVER_PRESET_ALIASES.values())))}"
                f"{download_hint}"
            )
        else:
            resolved = {
                "kind": "preset",
                "model": target,
                "preset": preset,
                "resolution": "hf-cache",
            }
            if args.download:
                resolved["download"] = {
                    "enabled": True,
                    "repo_id": profile.repo_id if profile is not None else None,
                    "dry_run": True,
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


def _cmd_download(args: argparse.Namespace) -> int:
    if args.list:
        if args.json:
            _json_dump(_download_options_payload())
        else:
            print(_format_download_options())
        return 0

    if not args.model:
        if args.json:
            _json_dump(_download_options_payload())
        else:
            print("missing model alias or repo id\n")
            print(_format_download_options())
        return 2

    code, summary, stderr = _download_summary(
        args.model,
        dest=args.dest,
        force=args.force,
    )
    if args.json:
        if summary is not None:
            _json_dump(summary)
        if stderr:
            sys.stderr.write(stderr)
        return code

    if stderr:
        sys.stderr.write(stderr)
    if summary is None:
        raise SystemExit("download helper did not emit an ax.download_model.v1 summary")
    _print_download_summary(summary)
    return code


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


def _default_mtp_depth_max(base_model: str, mtp_source: str) -> int:
    label = f"{base_model} {mtp_source}".lower()
    if "qwen3.6-27b" in label or "qwen3-6-27b" in label:
        return 3
    if "qwen3.6-35b" in label or "qwen3-6-35b" in label or "35b-a3b" in label:
        return 1
    return 1


def _cmd_convert_mtplx(args: argparse.Namespace) -> int:
    prepare_script = _find_repo_script("prepare_mtp_sidecar.py")
    check_script = _find_repo_script("check_mtp_sidecar_provenance.py")
    if prepare_script is None:
        raise SystemExit("cannot locate scripts/prepare_mtp_sidecar.py")
    if check_script is None:
        raise SystemExit("cannot locate scripts/check_mtp_sidecar_provenance.py")

    mtp_depth_max = args.mtp_depth_max
    if mtp_depth_max is None:
        mtp_depth_max = _default_mtp_depth_max(args.base_model, args.mtp_source)

    prepare_cmd = [
        sys.executable,
        str(prepare_script),
        "--hf-repo",
        args.mtp_source,
        "--base",
        args.base_model,
        "--mtp-depth-max",
        str(mtp_depth_max),
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
                "mtp_depth_max": mtp_depth_max,
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
    serve_parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing alias/repo artifacts first",
    )
    serve_parser.add_argument("--dry-run", action="store_true")
    serve_parser.add_argument("--json", action="store_true")
    serve_parser.set_defaults(func=_cmd_serve)

    download_parser = subparsers.add_parser(
        "download",
        help="Download an MLX model and generate its AX manifest",
    )
    download_parser.add_argument("model", nargs="?", help="Server preset alias or Hugging Face repo id")
    download_parser.add_argument(
        "--dest",
        default=None,
        help="Copy the resolved HF cache snapshot to this directory; default uses the shared Hugging Face Hub cache",
    )
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument("--list", action="store_true", help="Show supported download targets")
    download_parser.add_argument("--json", action="store_true")
    download_parser.set_defaults(func=_cmd_download)

    convert_parser = subparsers.add_parser(
        "convert-mtplx",
        help="Package a base MLX model with standard HF MTP tensors",
    )
    convert_parser.add_argument("base_model", help="Base MLX model dir or repo id")
    convert_parser.add_argument("--mtp-source", required=True, help="HF repo that ships mtp.* tensors")
    convert_parser.add_argument("--output", default=None)
    convert_parser.add_argument("--quantize", type=int, choices=[4, 8], default=None)
    convert_parser.add_argument(
        "--mtp-depth-max",
        type=int,
        default=None,
        help="Max MTP draft depth. Defaults by model: Qwen3.6 27B -> 3, Qwen3.6 35B-A3B -> 1.",
    )
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
