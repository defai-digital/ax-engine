#!/usr/bin/env python3
"""Verify immutable fresh-cache delivery of the pinned Flash Next MXFP4 MTP pack.

Downloads the pinned revision with the product downloader (`ax-engine download
qwen3.8-flash-next:mxfp4`) into an empty cache root, binds every published
member by size and SHA-256 against the staging inventory, requires the generated
native manifest to be runtime ready, and then runs the functional controls
(MTP disabled, required and the product default) on the delivered copy. The
first server start after delivery records its own readiness time; the page
cache is not controlled.

Dry-run is weight-free. Live execution needs network access and about 125 GiB
of internal-SSD space. This is delivery evidence for one revision/SKU, not
quality, performance, MTP-S/P/D or release qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

try:
    import check_flash_next_native_support as native
except ImportError:
    from scripts import check_flash_next_native_support as native

REPO_DIR = "models--" + native.PRIMARY_REPO.replace("/", "--")
PUBLISHED_MEMBERS = 49
PUBLISHED_BYTES = 132_261_877_478
ALIAS = "qwen3.8-flash-next:mxfp4"
CONTROL_MODES = ("disabled", "required", "default")
SPACE_MARGIN = 1.1
SCRIPTS = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scrub(value: Any) -> Any:
    """Remove the user's home path and loopback addresses from recorded text."""
    if isinstance(value, str):
        return value.replace(str(Path.home()), "[home]").replace("127.0.0.1", "[loopback]")
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    return value


def check_fresh(cache_root: Path, resume: bool) -> bool:
    """True when the cache starts empty; a non-empty cache is refused unless resuming."""
    if cache_root.exists() and not cache_root.is_dir():
        raise ValueError("cache root exists and is not a directory")
    empty = not cache_root.exists() or not any(cache_root.iterdir())
    if not empty and not resume:
        raise ValueError("fresh delivery requires an empty cache root (use --previous-attempt to resume)")
    return empty


def load_previous_attempts(directory: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Summaries of the failed attempts a resumed run continues, and whether the first started empty.

    A resumed delivery is still a fresh-cache delivery only if its first attempt
    started from an empty cache and nothing but the downloader wrote to it, so the
    chain is carried forward and every failed attempt stays in the record.
    """
    previous = json.loads((directory / "result.json").read_text())
    if previous.get("completed") or (previous.get("download") or {}).get("exit_code") in (0, None):
        raise ValueError("the previous attempt did not fail; there is nothing to resume")
    if previous.get("schema") != "ax-engine.flash-next.fresh-delivery.v2":
        raise ValueError("the previous attempt is not a v2 delivery record")
    stderr = directory / "download.stderr.log"
    tail = stderr.read_text(errors="replace").splitlines()[-12:] if stderr.is_file() else []
    summary = {"exit_code": previous["download"]["exit_code"],
               "elapsed_seconds": previous["download"]["elapsed_seconds"],
               "stderr_tail": scrub(tail)}
    return bool(previous.get("fresh_cache")), [*previous.get("previous_attempts", []), summary]


def snapshot_dir(cache_root: Path) -> Path:
    """The single pinned snapshot directory; any other revision is a failure."""
    snapshots = cache_root / REPO_DIR / "snapshots"
    found = sorted(path.name for path in snapshots.iterdir()) if snapshots.is_dir() else []
    if found != [native.PACK_REVISION]:
        raise ValueError(f"cache must hold exactly the pinned revision, found {found}")
    return snapshots / native.PACK_REVISION


TRANSPORT_OVERRIDES = ("AX_ENGINE_HF_MAX_WORKERS", "HF_HUB_DOWNLOAD_TIMEOUT", "HF_HUB_ETAG_TIMEOUT")


def parse_transport(items: list[str]) -> dict[str, str]:
    """Documented downloader transport knobs only; anything else is refused."""
    parsed: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or name not in TRANSPORT_OVERRIDES or not value.isdigit():
            raise ValueError(f"transport override must be NAME=<integer> with NAME in {TRANSPORT_OVERRIDES}")
        parsed[name] = value
    return parsed


def clean_env(cache_root: Path, python: Path, helper: Path | None = None,
              bench: Path | None = None, transport: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the downloader: inherited overrides dropped, the stated inputs set.

    A source build finds neither the bundled helper nor `ax-engine-bench`, so the
    helper path and the build's `ax-engine-bench` directory are passed explicitly.
    """
    kept = {name: value for name, value in os.environ.items()
            if not name.startswith(("AX_", "HF_", "MLX_", "XDG_CACHE"))}
    env = {**kept, "HF_HUB_CACHE": str(cache_root), "AX_ENGINE_PYTHON": str(python)}
    if helper is not None:
        env["AX_ENGINE_DOWNLOAD_HELPER"] = str(helper)
    if bench is not None:
        env["PATH"] = f"{bench.parent}{os.pathsep}{env.get('PATH', '')}"
    env.update(transport or {})
    return env


def terminal_record(stdout: str) -> dict[str, Any]:
    """The last JSON line the downloader prints is its terminal summary."""
    for line in reversed([item for item in stdout.splitlines() if item.strip()]):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            return record
    raise ValueError("downloader printed no JSON terminal record")


def verify_delivery(snapshot: Path, cache_root: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    derived = native.validate_inventory(snapshot, inventory, cache_root)
    members = inventory["members"]
    total = sum(member["size"] for member in members)
    if len(members) != PUBLISHED_MEMBERS or total != PUBLISHED_BYTES:
        raise ValueError(f"inventory is not the published pack: {len(members)} members, {total} bytes")
    manifest = snapshot / "model-manifest.json"
    if not manifest.is_file():
        raise ValueError("delivery produced no native model manifest")
    status = json.loads(manifest.read_text()).get("runtime_status") or {}
    if status.get("ready") is not True or status.get("blockers"):
        raise ValueError(f"delivered native manifest is not runtime ready: {status}")
    return {"members_verified": len(members), "bytes_verified": total,
            "all_sizes_and_sha256_match": True, "native_manifest_runtime_ready": True,
            "native_manifest_sha256": derived.get("model-manifest.json")}


def run_control(args: argparse.Namespace, snapshot: Path, output: Path) -> dict[str, Any]:
    command = [sys.executable, str(SCRIPTS / "check_flash_next_native_support.py"),
               "--model-dir", str(snapshot), "--link-root", str(args.cache_root),
               "--inventory", str(args.inventory), "--server-bin", str(args.server_bin),
               "--output", str(output), "--modes", *CONTROL_MODES]
    done = subprocess.run(command, capture_output=True, text=True)
    (output.parent / "native-support.log").write_text(scrub(done.stdout + done.stderr))
    if done.returncode != 0:
        raise RuntimeError(f"functional controls on the delivered copy failed ({done.returncode})")
    result = json.loads((output / "result.json").read_text())
    return {"passed": result["functional_controls_passed"], "modes": result["modes"],
            "readiness_seconds": result["readiness_seconds"], "page_cache": result["page_cache"],
            "result_sha256": sha256_file(output / "result.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ax-engine", type=Path, help="product CLI used to download")
    parser.add_argument("--python", type=Path, help="interpreter with huggingface_hub for the downloader")
    parser.add_argument("--helper", type=Path, help="download_model.py of the same source checkout")
    parser.add_argument("--bench-bin", type=Path, help="ax-engine-bench of the same build (manifest generation)")
    parser.add_argument("--cache-root", type=Path, help="empty Hugging Face cache root on the internal SSD")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--server-bin", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transport-env", action="append", default=[], metavar="NAME=VALUE",
                        help="downloader transport override (AX_ENGINE_HF_MAX_WORKERS, "
                             "HF_HUB_DOWNLOAD_TIMEOUT, HF_HUB_ETAG_TIMEOUT); recorded in the result")
    parser.add_argument("--previous-attempt", type=Path,
                        help="output directory of a failed attempt on this same cache root; "
                             "this run resumes it and records the failed attempt(s)")
    args = parser.parse_args()
    contract = {"schema": "ax-engine.flash-next.fresh-delivery.v2", "repo_id": native.PRIMARY_REPO,
                "revision": native.PACK_REVISION, "alias": ALIAS, "published_members": PUBLISHED_MEMBERS,
                "published_bytes": PUBLISHED_BYTES, "control_modes": list(CONTROL_MODES),
                "downloader": "ax-engine download <alias> --progress-json (product CLI and helper)",
                "page_cache": "not controlled", "qualification": False, "release_ready": False,
                "scope": "delivery integrity and functional controls on the delivered copy; "
                         "quality, performance and MTP-S/P/D not assessed"}
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return 0
    required = (args.ax_engine, args.python, args.cache_root, args.inventory, args.server_bin, args.output)
    if not all(required):
        parser.error("live run requires ax-engine, python, cache-root, inventory, server-bin and output")
    if platform.system() != "Darwin":
        parser.error("live run requires macOS on the Ultra-class 192 GiB+ target")
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if "Ultra" not in chip or memory < 192 * 1024**3:
        parser.error("live run requires the Ultra-class 192 GiB+ Flash Next target")
    transport = parse_transport(args.transport_env)
    resuming = args.previous_attempt is not None
    started_empty = check_fresh(args.cache_root, resuming)
    fresh, attempts = started_empty, []
    if resuming:
        fresh, attempts = load_previous_attempts(args.previous_attempt)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    storage = native.storage_info(args.cache_root)
    free = shutil.disk_usage(args.cache_root).free
    if free < PUBLISHED_BYTES * SPACE_MARGIN:
        raise ValueError(f"cache volume has {free} bytes free; need {int(PUBLISHED_BYTES * SPACE_MARGIN)}")
    hub = subprocess.run([str(args.python), "-c",
                          "import huggingface_hub, sys; print(huggingface_hub.__version__)"],
                         capture_output=True, text=True)
    if hub.returncode != 0:
        raise ValueError("the downloader interpreter cannot import huggingface_hub")
    inventory = json.loads(args.inventory.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (args.output / "download.ndjson").open("wb") as out, \
            (args.output / "download.stderr.log").open("wb") as err:
        done = subprocess.run([str(args.ax_engine), "download", ALIAS, "--progress-json"],
                              stdout=out, stderr=err, env=clean_env(args.cache_root, args.python, args.helper, args.bench_bin,
                                            transport))
    elapsed = round(time.monotonic() - started, 3)
    for name in ("download.ndjson", "download.stderr.log"):
        path = args.output / name
        path.write_text(scrub(path.read_text(errors="replace")))
    record: dict[str, Any] = {
        **contract, "completed": False, "fresh_cache": fresh, "previous_attempts": attempts,
        "resumed": resuming, "transport_override": transport,
        "download": {"exit_code": done.returncode, "elapsed_seconds": elapsed,
                     "hub_client_version": hub.stdout.strip(),
                     "ax_engine_sha256": sha256_file(args.ax_engine),
                     "helper_sha256": sha256_file(args.helper) if args.helper else None,
                     "bench_sha256": sha256_file(args.bench_bin) if args.bench_bin else None},
        "hardware": {"form_factor": native.form_factor(), "soc": chip, "memory_bytes": memory,
                     "storage": {"declared": inventory["storage"], **storage},
                     "os": platform.mac_ver()[0]},
        "server_sha256": sha256_file(args.server_bin), "inventory_sha256": sha256_file(args.inventory)}
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if done.returncode != 0:
        raise RuntimeError(f"downloader exited with {done.returncode}; see download.stderr.log")
    record["download"]["terminal"] = scrub(terminal_record(
        (args.output / "download.ndjson").read_text()))
    snapshot = snapshot_dir(args.cache_root)
    record["verification"] = verify_delivery(snapshot, args.cache_root, inventory)
    preflight = subprocess.run([sys.executable, str(SCRIPTS / "qualify_qwen38_flash_next.py"),
                                "--model-dir", str(snapshot)], capture_output=True, text=True)
    if preflight.returncode != 0:
        raise RuntimeError(f"metadata preflight failed: {scrub(preflight.stderr)}")
    record["metadata_preflight_passed"] = True
    record["native_support"] = run_control(args, snapshot, args.output / "native-support")
    if record["server_sha256"] != sha256_file(args.server_bin):
        raise ValueError("server binary changed during delivery controls")
    native.validate_inventory(snapshot, inventory, args.cache_root)
    record["completed"] = True
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Fresh delivery verified; qualification and MTP-S/P/D remain unassessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
