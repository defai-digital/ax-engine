from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import Sequence

from . import _bundled_binary


@dataclass(frozen=True)
class ModelProfile:
    label: str
    preset: str | None
    repo_id: str
    aliases: tuple[str, ...]
    downloadable: bool = True
    # When set, the model family has an MTP acceleration package reachable via
    # `ax-engine download-mtp <mtp_target>`. The interactive picker offers a
    # Direct-vs-MTP choice for these; --list flags them.
    mtp_target: str | None = None


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
        mtp_target="gemma-4-12b-4bit",
    ),
    ModelProfile(
        label="gemma4-12b-6bit",
        preset=None,
        repo_id="mlx-community/gemma-4-12B-it-6bit",
        aliases=("gemma4-12b-6bit", "gemma-4-12b-6bit", "gemma-4-12b-it-6bit"),
        mtp_target="gemma-4-12b",
    ),
    ModelProfile(
        label="gemma4-26b",
        preset="gemma4-26b",
        repo_id="mlx-community/gemma-4-26b-a4b-it-4bit",
        aliases=(
            "gemma4-26b",
            "gemma-4-26b",
            "gemma-4-26b-a4b-it",
            "gemma4-26b-4bit",
        ),
        mtp_target="gemma-4-26b",
    ),
    ModelProfile(
        label="gemma4-31b",
        preset="gemma4-31b",
        repo_id="mlx-community/gemma-4-31b-it-4bit",
        aliases=("gemma4-31b", "gemma-4-31b", "gemma-4-31b-it", "gemma4-31b-4bit"),
        mtp_target="gemma-4-31b",
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
        mtp_target="glm-4.7-flash",
    ),
    ModelProfile(
        label="qwen3.5-9b",
        preset="qwen3.5-9b",
        repo_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        aliases=(
            "qwen3.5-9b",
            "qwen35-9b",
            "qwen3-5-9b",
            "qwen3.5-9b-4bit",
            "qwen3-5-9b-mlx-4bit",
        ),
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
        mtp_target="qwen3.6-27b-6bit",
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
        mtp_target="qwen3.6-27b-6bit",
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
        mtp_target="qwen3.6-35b-a3b",
    ),
)


def _server_bin() -> pathlib.Path | str:
    bundled = _bundled_binary("ax-engine-server")
    if bundled is not None:
        return bundled
    return shutil.which("ax-engine-server") or "ax-engine-server"


def _bench_bin() -> pathlib.Path | str:
    bundled = _bundled_binary("ax-engine-bench")
    if bundled is not None:
        return bundled
    return shutil.which("ax-engine-bench") or "ax-engine-bench"


def _native_bin() -> str | None:
    """Locate the native Rust ``ax-engine`` binary (it hosts the ``tui`` subcommand).

    Resolution order: ``AX_ENGINE_NATIVE_BIN`` override, the binary bundled in the
    installed wheel, then a source-checkout ``target/{release,debug}`` build.  It
    deliberately never falls back to a bare ``ax-engine`` on ``PATH`` — that name
    resolves to this very Python console script, which would recurse.
    """
    override = os.environ.get("AX_ENGINE_NATIVE_BIN")
    if override and pathlib.Path(override).is_file():
        return override
    bundled = _bundled_binary("ax-engine")
    if bundled is not None:
        return str(bundled)
    roots: list[pathlib.Path] = []
    repo_env = os.environ.get("AX_ENGINE_REPO_ROOT")
    if repo_env:
        roots.append(pathlib.Path(repo_env))
    roots.append(pathlib.Path(__file__).resolve().parents[2])
    for root in roots:
        for profile in ("release", "debug"):
            candidate = root / "target" / profile / "ax-engine"
            if candidate.is_file():
                return str(candidate)
    return None


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


# ---------------------------------------------------------------------------
# Grouped catalog view (model family -> precision variants)
#
# MODEL_PROFILES is a flat list where each precision is its own row.  The
# interactive picker presents a model-first wizard, so we derive a grouped view
# from the flat list rather than duplicating the catalog.  These helpers are
# pure functions of MODEL_PROFILES and do not change the download/JSON contract.
# ---------------------------------------------------------------------------

_QUANT_RE = re.compile(r"(\d+)bit", re.IGNORECASE)
_FAMILY_SUFFIX_RE = re.compile(r"-\d+bit$", re.IGNORECASE)


def _profile_quant_bits(profile: ModelProfile) -> int | None:
    """Quantization bit-width parsed from a profile's repo_id (e.g. 4, 8)."""
    match = _QUANT_RE.search(profile.repo_id)
    return int(match.group(1)) if match else None


def _profile_family_key(profile: ModelProfile) -> str:
    """Family identifier: the label with any trailing ``-Nbit`` suffix removed."""
    return _FAMILY_SUFFIX_RE.sub("", profile.label)


@dataclass(frozen=True)
class ModelFamily:
    """A model and the set of precision variants it is published in."""

    key: str
    variants: tuple[ModelProfile, ...]  # ascending by bit-width

    @property
    def label(self) -> str:
        return self.key

    @property
    def has_mtp(self) -> bool:
        return any(variant.mtp_target for variant in self.variants)

    @property
    def quant_summary(self) -> str:
        bits = [b for b in (_profile_quant_bits(v) for v in self.variants) if b is not None]
        return ", ".join(f"{b}-bit" for b in bits) if bits else "--"


def _model_families() -> list[ModelFamily]:
    """Group downloadable profiles by family, variants sorted by bit-width."""
    groups: dict[str, list[ModelProfile]] = {}
    order: list[str] = []
    for profile in _downloadable_profiles():
        key = _profile_family_key(profile)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(profile)
    families: list[ModelFamily] = []
    for key in order:
        variants = sorted(groups[key], key=lambda p: (_profile_quant_bits(p) or 99))
        families.append(ModelFamily(key=key, variants=tuple(variants)))
    return families


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
                "mtp_target": profile.mtp_target,
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
        "Available Qwen3.5/3.6 and Gemma 4 MLX download targets:",
    ]
    for profile in _downloadable_profiles():
        mtp = f"  [MTP: download-mtp {profile.mtp_target}]" if profile.mtp_target else ""
        lines.append(f"  {profile.label:<20} {profile.repo_id}{mtp}")
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
    progress: bool = False,
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

    if progress:
        # Let the helper render its live progress bar straight to our stderr while we
        # still capture the stdout JSON summary.
        command.append("--progress-bar")
        result = _run_capture_stdout(command)
    else:
        result = _run_capture(command)
    summary = _parse_download_summary(result.stdout)
    if summary is not None:
        summary["input"] = model
        if profile is not None:
            summary["alias"] = profile.label
            if profile.preset is not None:
                summary["preset"] = profile.preset
    return result.returncode, summary, result.stderr or ""


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


def _supports_interactive() -> bool:
    """Interactive prompts are only safe when both stdin and stdout are a TTY."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, OSError):
        return False


def _default_download_root() -> pathlib.Path:
    """Mirror download_model.default_mlx_lm_cache_root for display in the wizard."""
    if hub := os.environ.get("HF_HUB_CACHE"):
        return pathlib.Path(hub).expanduser()
    if home := os.environ.get("HF_HOME"):
        return pathlib.Path(home).expanduser() / "hub"
    base = os.environ.get("XDG_CACHE_HOME") or (pathlib.Path.home() / ".cache")
    return pathlib.Path(base).expanduser() / "huggingface" / "hub"


def _wizard_input(prompt: str) -> str:
    return input(prompt)


def _select_profile_interactive() -> ModelProfile | None:
    profiles = _downloadable_profiles()
    print("AX Engine — download a model\n")
    print(f"  {'#':>2}  {'Model':<22} {'MTP':<5} Repo")
    for index, profile in enumerate(profiles, start=1):
        mtp = "yes" if profile.mtp_target else "—"
        print(f"  {index:>2}  {profile.label:<22} {mtp:<5} {profile.repo_id}")
    print()
    while True:
        raw = _wizard_input(f"Select a model [1-{len(profiles)}] (q to cancel): ").strip().lower()
        if raw in {"", "q", "quit", "exit"}:
            return None
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(profiles):
                return profiles[choice - 1]
        print("  invalid selection; enter a number from the list or q to cancel")


def _select_dest_interactive() -> str | None:
    default_root = _default_download_root()
    print(f"\nDefault download location: {default_root}")
    print("  (shared Hugging Face Hub cache; reused by mlx-lm and huggingface_hub)")
    raw = _wizard_input("Download path (Enter to accept default): ").strip()
    if not raw:
        return None
    return str(pathlib.Path(raw).expanduser())


def _validate_dest_writable(dest: str) -> None:
    probe = pathlib.Path(dest)
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    if not os.access(probe, os.W_OK):
        raise SystemExit(f"destination is not writable: {dest}")


def _confirm_interactive(prompt: str) -> bool:
    raw = _wizard_input(f"{prompt} [Y/n]: ").strip().lower()
    return raw in {"", "y", "yes"}


def _select_variant_interactive(profile: ModelProfile) -> str | None:
    """For an MTP-capable model, choose 'direct' or 'mtp'. None means cancel."""
    print(f"\n{profile.label} has an MTP acceleration package.")
    print("Download which variant?")
    print(f"  1  Direct download   {profile.repo_id}")
    print(f"  2  MTP package       ax-engine download-mtp {profile.mtp_target}")
    while True:
        raw = _wizard_input("Select [1-2] (q to cancel): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None
        if raw == "1":
            return "direct"
        if raw == "2":
            return "mtp"
        print("  invalid selection; enter 1, 2, or q to cancel")


def _run_interactive_direct_download(profile: ModelProfile, force: bool) -> int:
    dest = _select_dest_interactive()
    if dest is not None:
        _validate_dest_writable(dest)

    target = dest if dest is not None else f"{_default_download_root()} (shared cache)"
    if not _confirm_interactive(f"\nDownload {profile.label} ({profile.repo_id}) to {target}?"):
        print("Cancelled.")
        return 130

    print()
    code, summary, stderr = _download_summary(
        profile.label, dest=dest, force=force, progress=True
    )
    if stderr:
        sys.stderr.write(stderr)
    if summary is None:
        raise SystemExit("download helper did not emit an ax.download_model.v1 summary")
    _print_download_summary(summary)
    return code


def _run_interactive_mtp_download(profile: ModelProfile, force: bool) -> int:
    if not _confirm_interactive(
        f"\nPrepare MTP package for {profile.label} "
        f"(ax-engine download-mtp {profile.mtp_target})?"
    ):
        print("Cancelled.")
        return 130

    bench_bin = str(_bench_bin())
    argv = [bench_bin, "download-mtp", profile.mtp_target]
    if force:
        argv.append("--force")
    env = os.environ.copy()
    env.update(_download_mtp_helper_env())
    print(f"\nPreparing MTP package: ax-engine download-mtp {profile.mtp_target}\n")
    # Inherit stdio so the bench binary's own progress/output is shown live.
    return subprocess.run(argv, env=env).returncode


def _run_interactive_download(force: bool) -> int:
    profile = _select_profile_interactive()
    if profile is None:
        print("Cancelled.")
        return 130

    if profile.mtp_target:
        variant = _select_variant_interactive(profile)
        if variant is None:
            print("Cancelled.")
            return 130
        if variant == "mtp":
            return _run_interactive_mtp_download(profile, force)

    return _run_interactive_direct_download(profile, force)


def _cmd_ui_downloader(args: argparse.Namespace) -> int:
    if not _supports_interactive():
        raise SystemExit(
            "ax-engine ui-downloader needs an interactive terminal. "
            "Use: ax-engine download <model>"
        )
    return _run_interactive_download(args.force)


def _cmd_tui(args: argparse.Namespace) -> int:
    native = _native_bin()
    if native is None:
        raise SystemExit(
            "ax-engine tui requires the native ax-engine binary, which was not found.\n"
            "Reinstall ax-engine, or build it from a source checkout:\n"
            "  cargo build --release -p ax-engine-bench --bin ax-engine"
        )
    argv = [native, "tui", *args.tui_args]
    os.execvp(argv[0], argv)
    return 0


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

    interactive = args.interactive or (
        not args.model
        and not args.no_interactive
        and not args.json
        and _supports_interactive()
    )
    if interactive:
        return _run_interactive_download(args.force)

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


def _run_capture_stdout(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Capture stdout but let stderr pass through to the terminal (live progress)."""
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=None, text=True)


def _value_at(value: dict, path: tuple[str, ...]) -> object | None:
    current: object = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _value_str(value: dict, path: tuple[str, ...], default: str = "unknown") -> str:
    current = _value_at(value, path)
    return current if isinstance(current, str) else default


def _value_bool(value: dict, path: tuple[str, ...], default: bool = False) -> bool:
    current = _value_at(value, path)
    return current if isinstance(current, bool) else default


def _value_list(value: dict, path: tuple[str, ...]) -> list:
    current = _value_at(value, path)
    return current if isinstance(current, list) else []


def _package_version() -> str:
    try:
        return importlib.metadata.version("ax-engine")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _command_stdout(command: list[str]) -> str | None:
    try:
        result = _run_capture(command)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _sysctl_u64(name: str) -> int | None:
    value = _command_stdout(["sysctl", "-n", name])
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _host_os_version() -> str | None:
    if sys.platform == "darwin":
        return _command_stdout(["sw_vers", "-productVersion"])
    return platform.release() or None


def _host_os_build() -> str | None:
    if sys.platform == "darwin":
        return _command_stdout(["sw_vers", "-buildVersion"])
    return None


def _host_hardware_profile() -> str | None:
    if sys.platform != "darwin":
        return None
    return _command_stdout(["system_profiler", "SPHardwareDataType"])


def _parse_memory_bytes(output: str | None) -> int | None:
    if output is None:
        return None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Memory:"):
            continue
        parts = stripped.split(":", 1)[1].strip().split()
        if len(parts) < 2:
            return None
        try:
            amount = int(parts[0])
        except ValueError:
            return None
        unit = parts[1].lower()
        if unit in {"gb", "gib"}:
            return amount * 1024 * 1024 * 1024
        if unit in {"mb", "mib"}:
            return amount * 1024 * 1024
    return None


def _parse_cpu_core_summary(output: str | None) -> str | None:
    if output is None:
        return None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Total Number of Cores:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _parse_physical_cpu_cores(output: str | None) -> int | None:
    summary = _parse_cpu_core_summary(output)
    if summary is None:
        return None
    try:
        return int(summary.split()[0])
    except (IndexError, ValueError):
        return None


def _parse_cpu_core_types(summary: str | None) -> dict[str, int]:
    if summary is None or "(" not in summary or ")" not in summary:
        return {}
    inside = summary.split("(", 1)[1].split(")", 1)[0]
    types: dict[str, int] = {}
    for part in inside.split(" and "):
        words = part.split()
        if len(words) < 2:
            continue
        try:
            cores = int(words[0])
        except ValueError:
            continue
        label = "_".join(word.lower() for word in words[1:])
        types[label] = cores
    return types


def _host_ram_bytes(hardware_profile: str | None) -> int | None:
    if sys.platform == "darwin":
        return _sysctl_u64("hw.memsize") or _parse_memory_bytes(hardware_profile)
    return None


def _host_cpu_cores(hardware_profile: str | None) -> dict:
    performance: int | None = None
    efficiency: int | None = None
    types: dict[str, int] = {}
    if sys.platform == "darwin":
        for level in range(4):
            name = _command_stdout(["sysctl", "-n", f"hw.perflevel{level}.name"])
            cores = _sysctl_u64(f"hw.perflevel{level}.physicalcpu")
            normalized = (name or "").lower()
            if "performance" in normalized:
                performance = cores
            elif "efficiency" in normalized:
                efficiency = cores
            if name and cores is not None:
                types[normalized.replace(" ", "_")] = cores
    summary = _parse_cpu_core_summary(hardware_profile)
    if not types:
        types = _parse_cpu_core_types(summary)
        performance = performance or types.get("performance")
        efficiency = efficiency or types.get("efficiency")
    return {
        "physical": (
            _sysctl_u64("hw.physicalcpu") or _parse_physical_cpu_cores(hardware_profile)
            if sys.platform == "darwin"
            else os.cpu_count()
        ),
        "logical": os.cpu_count(),
        "performance": performance,
        "efficiency": efficiency,
        "summary": summary,
        "types": types,
    }


def _host_gpu_cores() -> int | None:
    if sys.platform != "darwin":
        return None
    output = _command_stdout(["system_profiler", "SPDisplaysDataType"])
    if output is None:
        return None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Total Number of Cores:"):
            try:
                return int(stripped.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _host_system_summary() -> dict:
    hardware_profile = _host_hardware_profile()
    ram_bytes = _host_ram_bytes(hardware_profile)
    return {
        "os": sys.platform,
        "arch": platform.machine() or "unknown",
        "os_version": _host_os_version(),
        "os_build": _host_os_build(),
        "ram_bytes": ram_bytes,
        "ram_gib": ram_bytes // (1024 * 1024 * 1024) if ram_bytes is not None else None,
        "cpu_cores": _host_cpu_cores(hardware_profile),
        "gpu_cores": _host_gpu_cores(),
    }


def _format_cpu_cores(cpu_cores: dict | None) -> str:
    if not isinstance(cpu_cores, dict):
        return "unknown"
    summary = cpu_cores.get("summary")
    if isinstance(summary, str) and summary:
        return summary
    physical = cpu_cores.get("physical")
    logical = cpu_cores.get("logical")
    performance = cpu_cores.get("performance")
    efficiency = cpu_cores.get("efficiency")
    if all(isinstance(value, int) for value in [physical, logical, performance, efficiency]):
        return f"{physical} physical / {logical} logical ({performance}P+{efficiency}E)"
    if isinstance(physical, int) and isinstance(logical, int):
        return f"{physical} physical / {logical} logical"
    if isinstance(physical, int):
        return f"{physical} physical"
    return "unknown"


def _format_ram_gib(value: object) -> str:
    return f"{value} GiB" if isinstance(value, int) else "unknown"


def _probe_binary(label: str, bin_path: pathlib.Path | str) -> dict:
    path = str(bin_path)
    try:
        result = _run_capture([path, "--help"])
    except OSError as exc:
        return {"id": label, "status": "fail", "detail": f"{path}: {exc}"}
    if result.returncode == 0:
        return {"id": label, "status": "pass", "detail": f"{path} ok"}
    return {
        "id": label,
        "status": "fail",
        "detail": f"{path} exited with status {result.returncode}",
    }


def _doctor_check(check_id: str, passed: bool, detail: str) -> dict:
    return {"id": check_id, "status": "pass" if passed else "fail", "detail": detail}


def _doctor_ready_for(result: str, model_status: str) -> list[str]:
    if result == "not_ready":
        return []
    if model_status == "ready":
        return ["serve", "python_sdk", "model_checks"]
    return ["serve", "python_sdk"]


def _format_doctor_text(report: dict) -> str:
    lines = [
        "AX Engine doctor",
        "",
        f"Result: {str(report.get('result', 'unknown')).replace('_', ' ')}",
        "",
        "Install:",
        f"  version: {report.get('install', {}).get('version', 'unknown')}",
        f"  mode: {report.get('install', {}).get('mode', 'unknown')}",
        (
            f"  host: {report.get('host', {}).get('os', 'unknown')} "
            f"{report.get('host', {}).get('os_version') or 'unknown'} "
            f"({report.get('host', {}).get('arch', 'unknown')})"
        ),
        f"  RAM: {_format_ram_gib(report.get('host', {}).get('ram_gib'))}",
        f"  CPU cores: {_format_cpu_cores(report.get('host', {}).get('cpu_cores'))}",
        f"  GPU cores: {report.get('host', {}).get('gpu_cores') or 'unknown'}",
        "",
        "Checks:",
    ]
    for check in report.get("checks", []):
        check_id = check.get("id", "unknown")
        status = check.get("status", "unknown")
        detail = check.get("detail")
        if isinstance(detail, str):
            lines.append(f"  {check_id}: {status} - {detail}")
        else:
            selected = check.get("selected", False)
            path = check.get("path") or "none"
            lines.append(f"  {check_id}: {status} (selected: {selected}, path: {path})")

    def append_section(title: str, values: list) -> None:
        lines.extend(["", f"{title}:"])
        if not values:
            lines.append("  none")
            return
        for value in values:
            if isinstance(value, str):
                lines.append(f"  {value}")

    append_section("Issues", report.get("issues", []))
    append_section("Model issues", report.get("model_issues", []))
    append_section("Next", report.get("next_actions", []))
    lines.extend(["", f"More details: {report.get('details_command', 'ax-engine-bench doctor')}"])
    return "\n".join(lines)


def _user_doctor_report(bench_report: dict) -> dict:
    server_check = _probe_binary("server_binary", _server_bin())
    bench_check = _probe_binary("bench_binary", _bench_bin())
    bench_status = _value_str(bench_report, ("status",))
    mlx_ready = _value_bool(bench_report, ("mlx_runtime_ready",))
    model_status = _value_str(bench_report, ("model_artifacts", "status"))
    model_selected = _value_bool(bench_report, ("model_artifacts", "selected"))
    model_path = _value_at(bench_report, ("model_artifacts", "path"))
    model_path = model_path if isinstance(model_path, str) else None

    if server_check["status"] != "pass" or bench_check["status"] != "pass" or bench_status == "not_ready":
        result = "not_ready"
    elif bench_status == "bringup_only":
        result = "degraded"
    else:
        result = "ready"

    next_actions: list[str] = []
    if server_check["status"] != "pass":
        next_actions.append("Reinstall ax-engine so ax-engine-server is on PATH.")
    elif bench_check["status"] != "pass":
        next_actions.append("Reinstall ax-engine so ax-engine-bench is on PATH.")
    elif not mlx_ready:
        next_actions.append("Fix the host or Metal runtime issues listed below.")
    elif model_status == "not_ready":
        if model_path:
            next_actions.append(f"ax-engine-bench generate-manifest {model_path} --json")
            next_actions.append(f"ax-engine doctor --mlx-model-artifacts-dir {model_path}")
        else:
            next_actions.append("Pass --mlx-model-artifacts-dir <model-dir> to inspect a model.")
    elif model_selected:
        next_actions.append(f"ax-engine serve {model_path or '<model-dir>'} --port 8080")
    else:
        next_actions.append("ax-engine serve qwen36-35b --download --port 8080")
        next_actions.append("ax-engine models list")

    host_detail = (
        f"{_value_str(bench_report, ('host', 'detected_soc'), 'unknown Apple Silicon')} "
        f"({_value_str(bench_report, ('host', 'os'))}/{_value_str(bench_report, ('host', 'arch'))})"
    )
    metal_detail = (
        "Metal compiler and metallib available"
        if _value_bool(bench_report, ("metal_toolchain", "fully_available"))
        else "Metal compiler or metallib missing"
    )
    return {
        "schema_version": "ax.engine.doctor.v1",
        "result": result,
        "ready_for": _doctor_ready_for(result, model_status),
        "install": {
            "version": _package_version(),
            "mode": _value_str(bench_report, ("workflow", "mode")),
            "cwd": _value_str(bench_report, ("workflow", "cwd")),
        },
        "host": _host_system_summary(),
        "checks": [
            server_check,
            bench_check,
            _doctor_check(
                "host",
                _value_bool(bench_report, ("host", "supported_mlx_runtime")),
                host_detail,
            ),
            _doctor_check(
                "metal_toolchain",
                _value_bool(bench_report, ("metal_toolchain", "fully_available")),
                metal_detail,
            ),
            _doctor_check("mlx_runtime", mlx_ready, bench_status),
            {
                "id": "model",
                "status": model_status,
                "selected": model_selected,
                "path": model_path,
            },
        ],
        "issues": _value_list(bench_report, ("issues",)),
        "model_issues": _value_list(bench_report, ("model_artifacts", "issues")),
        "next_actions": next_actions,
        "details_command": "ax-engine-bench doctor",
        "source": {
            "schema_version": _value_str(bench_report, ("schema_version",)),
            "status": bench_status,
            "details_command": "ax-engine-bench doctor --json",
        },
    }


def _default_mtp_depth_max(base_model: str, mtp_source: str) -> int:
    label = f"{base_model} {mtp_source}".lower()
    if "glm-4.7-flash" in label or "glm4.7-flash" in label or "glm47" in label:
        return 1
    if "qwen3.6-27b" in label or "qwen3-6-27b" in label:
        return 3
    if "qwen3.6-35b" in label or "qwen3-6-35b" in label or "35b-a3b" in label:
        return 1
    if "qwen3-coder-next" in label or "qwen3-next-80b" in label or "qwen3-next-80b-a3b" in label:
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


def _download_mtp_helper_env() -> dict[str, str]:
    helpers = {
        "AX_ENGINE_DOWNLOAD_HELPER": "download_model.py",
        "AX_ENGINE_PREPARE_MTP_SIDECAR_HELPER": "prepare_mtp_sidecar.py",
        "AX_ENGINE_PREPARE_GEMMA4_ASSISTANT_MTP_HELPER": "prepare_gemma4_assistant_mtp.py",
        "AX_ENGINE_PREPARE_GLM_MTP_SIDECAR_HELPER": "prepare_glm_mtp_sidecar.py",
        "AX_ENGINE_CHECK_MTP_SIDECAR_HELPER": "check_mtp_sidecar_provenance.py",
    }
    env: dict[str, str] = {}
    for env_name, script_name in helpers.items():
        script = _find_repo_script(script_name)
        if script is not None:
            env[env_name] = str(script)
    return env


def _cmd_download_mtp(args: argparse.Namespace) -> int:
    bench_bin = str(_bench_bin())
    argv = [bench_bin, "download-mtp", args.model]
    if args.output:
        argv.extend(["--output", args.output])
    if args.force:
        argv.append("--force")
    if args.quantize:
        argv.extend(["--quantize", args.quantize])
    if args.mtp_depth_max:
        argv.extend(["--mtp-depth-max", args.mtp_depth_max])
    if args.group_size:
        argv.extend(["--group-size", args.group_size])
    if args.fair_base_only:
        argv.append("--fair-base-only")
    if args.json:
        argv.append("--json")

    env = os.environ.copy()
    env.update(_download_mtp_helper_env())
    os.execvpe(argv[0], argv, env)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    bench_bin = str(_bench_bin())
    argv = [bench_bin, "doctor"]
    if args.verbose and args.json:
        argv.append("--json")
    if args.mlx_model_artifacts_dir:
        argv.extend(["--mlx-model-artifacts-dir", args.mlx_model_artifacts_dir])
    if args.verbose:
        os.execvp(argv[0], argv)
        return 0

    argv.append("--json")
    result = _run_capture(argv)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        return result.returncode
    try:
        bench_report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ax-engine-bench doctor did not emit valid JSON: {exc}") from exc

    report = _user_doctor_report(bench_report)
    if args.json:
        _json_dump(report)
    else:
        print(_format_doctor_text(report))
    return 1 if report.get("result") == "not_ready" else 0


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
    download_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pick a model and destination interactively with a live progress bar",
    )
    download_parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Never prompt; require a model argument",
    )
    download_parser.add_argument("--json", action="store_true")
    download_parser.set_defaults(func=_cmd_download)

    tui_parser = subparsers.add_parser(
        "tui",
        help="Launch the terminal UI for model download and serving",
    )
    tui_parser.add_argument(
        "tui_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the native ax-engine tui (e.g. --help)",
    )
    tui_parser.set_defaults(func=_cmd_tui)

    ui_downloader_parser = subparsers.add_parser(
        "ui-downloader",
        help="Deprecated: use 'ax-engine tui' instead",
    )
    ui_downloader_parser.add_argument("--force", action="store_true")
    ui_downloader_parser.set_defaults(func=_cmd_ui_downloader)

    download_mtp_parser = subparsers.add_parser(
        "download-mtp",
        help="Download a supported 6-bit target and prepare AX MTP artifacts",
    )
    download_mtp_parser.add_argument("model", help="Supported MTP target alias")
    download_mtp_parser.add_argument("--output", default=None)
    download_mtp_parser.add_argument("--force", action="store_true")
    download_mtp_parser.add_argument("--quantize", choices=("4", "8"), default=None)
    download_mtp_parser.add_argument("--mtp-depth-max", default=None)
    download_mtp_parser.add_argument("--group-size", default=None)
    download_mtp_parser.add_argument("--fair-base-only", action="store_true")
    download_mtp_parser.add_argument("--json", action="store_true")
    download_mtp_parser.set_defaults(func=_cmd_download_mtp)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check AX Engine system readiness (host, Metal toolchain, model artifacts)",
    )
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the detailed ax-engine-bench doctor report",
    )
    doctor_parser.add_argument(
        "--mlx-model-artifacts-dir",
        default=None,
        dest="mlx_model_artifacts_dir",
        help="Check a specific model artifact directory for AX readiness",
    )
    doctor_parser.set_defaults(func=_cmd_doctor)

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
