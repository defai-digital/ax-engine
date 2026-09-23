#!/usr/bin/env python3
"""Validate the omlx / MTPLX / ds4 peer-benchmark contract for Qwen 3.8 Flash Next.

The contract (`scripts/flash_next_peer_bench_plan.json`) declares, without
recording any numbers, what a real peer comparison needs: the ADR-037
Ultra-class target host, the MXFP4 pack revision, the three peers, and the
`/metrics` series the AX arm must expose. It is checked in, so the comparison
is reproducible instead of being a paragraph in a report.

This checker is fail-closed:

  * default                  - the contract is well-formed, every declared
                               `/metrics` series really is published by the
                               server crate, and the precondition gate is
                               proven to reject a host that has nothing. This host's real preconditions are
                               reported but are not fatal unless --require-preconditions is passed. Exit 0.
  * --dry-run                - identical to the default; accepted so a caller
                               can state the intent explicitly. Exit 0.
  * --require-preconditions  - the same, but exit nonzero when a precondition is
                               absent. This is the path a real run takes.

No weights, no target host and no network are required, so the default and
--dry-run paths pass on the authoring machine. Nothing here closes MTP-S,
MTP-P, MTP-D, default admission or release.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "scripts/flash_next_peer_bench_plan.json"
SERVER_METRICS = REPO_ROOT / "crates/ax-engine-server/src/metrics.rs"
FALLBACK_KEYS = REPO_ROOT / "crates/ax-engine-server/src/flash_next_fallback_keys.rs"

REQUIRED_PEERS = ("omlx", "mtplx", "ds4")
REQUIRED_GATES = ("MTP-S", "MTP-P", "MTP-D")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
FALLBACK_METRIC_RE = re.compile(r"^ax_engine_flash_next_mtp_direct_fallback_([a-z0-9_]+)_total$")


class Probe:
    """Live host facts. `EmptyProbe` reports a host that has nothing."""

    def is_apple_silicon(self) -> bool:
        return sys.platform == "darwin" and platform.machine() == "arm64"

    def memory_gib(self) -> int:
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return int(out.stdout.strip()) // (1024**3)
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0

    def pack_dir(self) -> str | None:
        return os.environ.get("AX_ENGINE_FLASH_NEXT_PACK_DIR")

    def tool(self, name: str) -> str | None:
        return shutil.which(name)


class EmptyProbe(Probe):
    """A host with no Apple Silicon, no memory, no pack and no peer tools."""

    def is_apple_silicon(self) -> bool:
        return False

    def memory_gib(self) -> int:
        return 0

    def pack_dir(self) -> str | None:
        return None

    def tool(self, name: str) -> str | None:
        return None


def server_publishes(metric: str, metrics_text: str, keys_text: str) -> bool:
    """True when the server crate really publishes this series name."""
    if f'"{metric}"' in metrics_text:
        return True
    match = FALLBACK_METRIC_RE.match(metric)
    if match:
        return f'"ax_mlx_flash_next_mtp_direct_fallback_{match.group(1)}"' in keys_text
    return False


def contract_problems(contract: dict, metrics_text: str, keys_text: str) -> list[str]:
    problems: list[str] = []

    peers = contract.get("peers")
    if not isinstance(peers, list) or not peers:
        return ["contract declares no peers"]
    peer_ids = {str(peer.get("id", "")).lower() for peer in peers}
    for required in REQUIRED_PEERS:
        if required.lower() not in peer_ids:
            problems.append(f"contract does not name the {required} peer")
    for peer in peers:
        for field in ("id", "tool", "role", "command", "records"):
            if not peer.get(field):
                problems.append(f"peer {peer.get('id', '?')!r} is missing {field}")

    host = contract.get("target_host") or {}
    if "Ultra-class" not in str(host.get("description", "")):
        problems.append("target_host does not declare the Ultra-class Apple Silicon host")
    if not host.get("requires_apple_silicon"):
        problems.append("target_host does not require Apple Silicon")
    if int(host.get("min_memory_gib", 0)) < 192:
        problems.append("target_host min_memory_gib is below 192")

    pack = contract.get("pack") or {}
    if not str(pack.get("repo", "")).strip():
        problems.append("pack does not declare a repo")
    if not REVISION_RE.match(str(pack.get("revision", ""))):
        problems.append("pack revision is not a 40-hex revision")
    if int(pack.get("published_bytes", 0)) <= 0:
        problems.append("pack does not declare published_bytes")

    ax_arm = contract.get("ax_arm") or {}
    if not ax_arm.get("records"):
        problems.append("ax_arm declares no recorded metrics")
    required_metrics = ax_arm.get("required_metrics") or []
    if not required_metrics:
        problems.append("ax_arm declares no required /metrics series")
    for metric in required_metrics:
        if not server_publishes(metric, metrics_text, keys_text):
            problems.append(f"ax_arm requires {metric}, which the server does not publish")

    gates = " ".join(str(gate) for gate in contract.get("open_gates_untouched", []))
    for gate in REQUIRED_GATES:
        if gate not in gates:
            problems.append(f"open_gates_untouched omits {gate}")
    if "admission" not in gates.lower():
        problems.append("open_gates_untouched omits default admission")
    if "release" not in gates.lower():
        problems.append("open_gates_untouched omits release")

    return problems


def missing_preconditions(contract: dict, probe: Probe) -> list[str]:
    missing: list[str] = []
    host = contract["target_host"]
    if host.get("requires_apple_silicon") and not probe.is_apple_silicon():
        missing.append("target host is not Apple Silicon")
    required_gib = int(host.get("min_memory_gib", 0))
    if probe.memory_gib() < required_gib:
        missing.append(f"host memory {probe.memory_gib()} GiB < required {required_gib} GiB")

    pack = contract["pack"]
    override = str(pack.get("env_override", "AX_ENGINE_FLASH_NEXT_PACK_DIR"))
    pack_dir = probe.pack_dir()
    if not pack_dir:
        missing.append(f"{override} is not set")
    elif not Path(pack_dir).is_dir():
        missing.append(f"{override}={pack_dir} is not a directory")

    for peer in contract["peers"]:
        if not probe.tool(str(peer["tool"])):
            missing.append(f"peer tool not on PATH: {peer['tool']}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the contract and report this host's preconditions without failing",
    )
    parser.add_argument(
        "--require-preconditions",
        action="store_true",
        help="exit nonzero when a declared precondition is absent on this host",
    )
    args = parser.parse_args()

    if not CONTRACT_PATH.is_file():
        print(f"FAIL: contract not found at {CONTRACT_PATH.relative_to(REPO_ROOT)}")
        return 1
    if not SERVER_METRICS.is_file() or not FALLBACK_KEYS.is_file():
        print("FAIL: server metric sources not found; run from the repository root")
        return 1
    try:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL: contract is not valid JSON: {exc}")
        return 1

    problems = contract_problems(
        contract,
        SERVER_METRICS.read_text(encoding="utf-8"),
        FALLBACK_KEYS.read_text(encoding="utf-8"),
    )
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1

    # Prove the gate is not vacuous: a host with nothing must be rejected.
    if not missing_preconditions(contract, EmptyProbe()):
        print("FAIL: precondition gate accepted a host with nothing present")
        return 1

    missing = missing_preconditions(contract, Probe())
    peers = ", ".join(REQUIRED_PEERS)
    if missing:
        detail = "; ".join(missing)
        if args.require_preconditions:
            print(f"FAIL: preconditions absent: {detail}")
            return 1
        mode = "--dry-run" if args.dry_run else "default"
        print(
            f"OK ({mode}): contract valid, peers {peers}; {len(missing)} "
            f"precondition(s) absent here: {detail}. "
            "Pass --require-preconditions to fail closed on these."
        )
        return 0

    print(f"OK: contract valid, all preconditions present, peers {peers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
