#!/usr/bin/env python3
"""Wave-1 AX vs mlxcel fleet bench for PRD-M5-FLEET-AX-VS-MLXCEL.

Runs shipped `bench_mlx_inference_stack.py --ax-direct --skip-mlx-lm` and
mlxcel `mlxcel-bench-decode` on the frozen Wave-1 checkpoints.

Example:
  python3 scripts/bench_ax_vs_mlxcel_fleet.py \\
    --out-dir .internal/qwen36-27b-m5-runs/fleet-baseline \\
    --models qwen3.5-9b,qwen3.6-27b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STACK = REPO / "scripts" / "bench_mlx_inference_stack.py"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

# Priority-ordered hub-layout roots searched before HF_HUB. `--hub-root`
# prepends more. On `df-macbookpro-m5` the default HF cache symlinks to a
# NAS share; `~/models` holds local-disk mirrors that load much faster.
EXTRA_HUB_ROOTS: list[Path] = []

# Factory / host-local AXQ pack roots (complete weights, not the stub Hub cache).
AXQ_LOCAL_ROOTS = (
    Path("/Volumes/Ext4T/axquant/axq-canonical-v2"),
    Path("/Volumes/Ext4T/axquant/axq-publish"),
    Path("/Volumes/Ext4T/axquant/axq-assistant-composites-v1"),
)

WAVE1 = {
    "qwen3.5-9b": {
        "repo": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "family": "qwen3_5",
        "mtp": False,
    },
    "qwen3.6-27b": {
        "repo": "mlx-community/Qwen3.6-27B-4bit",
        "family": "qwen3_5",
        "mtp": True,
    },
    "qwen36-35b": {
        "repo": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "family": "qwen3_5",
        "mtp": True,
    },
    "gemma4-12b": {
        "repo": "AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-it",
        "family": "gemma4",
        "mtp": True,
    },
    "gemma4-26b": {
        "repo": "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit",
        "family": "gemma4",
        "mtp": True,
    },
    "gemma4-31b": {
        "repo": "AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit",
        "family": "gemma4",
        "mtp": True,
    },
    "glm4.7-flash-4bit": {
        "repo": "mlx-community/GLM-4.7-Flash-4bit",
        "family": "glm4_moe_lite",
        "mtp": False,
    },
    "gpt-oss-20b": {
        "repo": "mlx-community/gpt-oss-20b-MXFP4-Q4",
        "family": "gpt_oss",
        "mtp": False,
    },
    "qwen3-coder-next": {
        "repo": "mlx-community/Qwen3-Coder-Next-4bit",
        "family": "qwen3_next",
        "mtp": False,
    },
    "muse-glimmer-30b": {
        "repo": "mlx-community/Muse-Glimmer-30B-4bit",
        "family": "muse_glimmer",
        "mtp": False,
    },
    # Wave-2 AXQ lanes (PRD §4 Wave 2): catalog-pinned AutomatosX packs.
    # Revisions mirror docs/SUPPORTED-MODELS.md; do not re-pin here.
    "qwen3.6-27b-axq": {
        "repo": "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        "revision": "8c37715c7b5f5ebca00eda6f73be47116a3e4ebc",
        "family": "qwen3_5",
        "mtp": True,
    },
    "qwen36-35b-axq": {
        "repo": "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP",
        "revision": "6a4c220734f81112555ee8783d91e0065c54301c",
        "family": "qwen3_5",
        "mtp": True,
    },
    "muse-glimmer-30b-axq": {
        "repo": "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit",
        "revision": "bcfb0b748fc44487c1657fb6ae190592d515398b",
        "family": "muse_glimmer",
        "mtp": False,
    },
    "holo3-35b-axq": {
        "repo": "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit",
        "revision": "e6cc340b04bfcec57544e462ec756e48dd248cf9",
        "family": "qwen3_5",
        "mtp": False,
    },
    "ornith-35b-axq": {
        "repo": "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit",
        "revision": "41015da430ae62802d9357b0ef31bf46c2b13b58",
        "family": "qwen3_5",
        "mtp": False,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pack_dir_looks_complete(path: Path) -> bool:
    """True when a local pack has config.json and at least one large weight file."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(
        child.is_file() and child.stat().st_size > 1_000_000
        for child in path.glob("*.safetensors")
    )


def hub_snapshot(repo_id: str, revision: str | None = None) -> Path | None:
    dirname = "models--" + repo_id.replace("/", "--")
    for root in [*EXTRA_HUB_ROOTS, HF_HUB]:
        snaps = root / dirname / "snapshots"
        if not snaps.is_dir():
            continue
        if revision:
            candidate = snaps / revision
            if pack_dir_looks_complete(candidate):
                return candidate
            continue
        refs = snaps.parent / "refs" / "main"
        if refs.is_file():
            rev = refs.read_text().strip()
            candidate = snaps / rev
            if pack_dir_looks_complete(candidate):
                return candidate
        kids = sorted(p for p in snaps.iterdir() if p.is_dir())
        for candidate in reversed(kids):
            if pack_dir_looks_complete(candidate):
                return candidate
    return None


def resolve_snapshot(
    spec: dict[str, object], extra_roots: list[Path] | None = None
) -> Path | None:
    """Resolve a Wave-1 row to a complete local pack.

    Order: explicit ``local_dir``, then ``--local-root`` / factory AXQ
    roots (repo basename), then a complete Hugging Face hub snapshot.
    Stub Hub trees (config-only, no weight blobs) are not accepted.
    """
    local = spec.get("local_dir")
    if isinstance(local, str) and local:
        candidate = Path(local).expanduser()
        if pack_dir_looks_complete(candidate):
            return candidate
    name = str(spec.get("repo") or "").rsplit("/", 1)[-1]
    roots: list[Path] = []
    if extra_roots:
        roots.extend(extra_roots)
    roots.extend(AXQ_LOCAL_ROOTS)
    if name:
        for root in roots:
            candidate = root / name
            if pack_dir_looks_complete(candidate):
                return candidate
    repo = spec.get("repo")
    if isinstance(repo, str) and repo:
        revision = spec.get("revision")
        return hub_snapshot(repo, revision if isinstance(revision, str) else None)
    return None


def run(cmd: list[str], log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{utc_now()}] $ {' '.join(cmd)}", flush=True)
    with log_path.open("w") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        started = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        log.write(f"\n[exit {result.returncode} after {elapsed:.1f}s]\n")
    print(
        f"[{utc_now()}] exit={result.returncode} after {elapsed:.1f}s log={log_path}",
        flush=True,
    )
    return result.returncode


_TOK_S_PAREN = re.compile(
    r"(prefill|decode)\s*:\s*[\d.]+\s*ms\s*\(([\d.]+)\s*tok/s\)",
    re.IGNORECASE,
)
_TOK_S_COMMA = re.compile(
    r"(prefill|decode)\s*:\s*[\d.]+\s*ms[,\s]+([\d.]+)\s*tok/s",
    re.IGNORECASE,
)


def parse_mlxcel_log(log_path: Path) -> dict[str, float]:
    """Read mlxcel-bench-decode Profile Results tok/s, not the millisecond field.

    Shipped format (2026-08 mlxcel-bench-decode):
      Prefill:          695.32 ms (2945.42 tok/s)
    Older / test fixture format:
      Prefill: 12.3 ms, 441.2 tok/s
    """
    pre = None
    dec = None
    text = log_path.read_text(errors="replace")
    for pattern in (_TOK_S_PAREN, _TOK_S_COMMA):
        for match in pattern.finditer(text):
            kind = match.group(1).lower()
            val = float(match.group(2))
            if kind == "prefill":
                pre = val
            else:
                dec = val
        if pre is not None and dec is not None:
            break
    return {"prefill_tok_s": pre or 0.0, "decode_tok_s": dec or 0.0}


def ax_medians(json_path: Path) -> dict[int, tuple[float, float]]:
    if not json_path.is_file():
        return {}
    data = json.loads(json_path.read_text())
    out: dict[int, tuple[float, float]] = {}

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            eng = obj.get("engine")
            pt = obj.get("prompt_tokens")
            pre = obj.get("prefill_tok_s")
            dec = obj.get("decode_tok_s")
            if (
                eng == "ax_engine_mlx"
                and pt in (128, 512, 2048)
                and isinstance(pre, dict)
                and "median" in pre
            ):
                out[int(pt)] = (
                    float(pre["median"]),
                    float(dec["median"] if isinstance(dec, dict) else dec),
                )
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(data)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        default=",".join(WAVE1),
        help="Comma-separated Wave-1 ids",
    )
    parser.add_argument(
        "--mlxcel-bin",
        type=Path,
        default=REPO / ".internal/reference/mlxcel/target/release/mlxcel-bench-decode",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--local-root",
        action="append",
        default=[],
        type=Path,
        help="Extra directory to search for AXQ pack basenames (repeatable)",
    )
    parser.add_argument(
        "--hub-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Extra hub-layout cache root (models--Org--Name/snapshots/rev) "
            "searched before the default HF cache (repeatable)"
        ),
    )
    parser.add_argument("--skip-ax", action="store_true")
    parser.add_argument("--skip-mlxcel", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    EXTRA_HUB_ROOTS.extend(Path(p).expanduser() for p in args.hub_root)
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [item for item in selected if item not in WAVE1]
    if unknown:
        print(f"unknown models: {unknown}", file=sys.stderr)
        return 2

    # Standing 2-bit path: keep closed 27B experiment flags off.
    os.environ.setdefault("AX_MLX_QWEN_COMPILED_GATED_DELTA_PREFILL", "0")
    os.environ.setdefault("AX_MLX_QWEN_PREFILL_CHUNK_1536", "0")
    os.environ.setdefault("AX_MLX_QWEN_PREFILL_DOWN_COMPILE", "0")
    os.environ.setdefault("AX_MLX_QWEN_LA_FUSED_QKVZ_BA_QMM", "0")

    host = subprocess.check_output(["hostname"], text=True).strip()
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    meta = {
        "prd": "PRD-M5-FLEET-AX-VS-MLXCEL",
        "hostname": host,
        "commit": commit,
        "started_at": utc_now(),
        "models": selected,
    }
    (out_dir / "host.json").write_text(json.dumps(meta, indent=2) + "\n")

    scoreboard: list[dict[str, object]] = []
    for model_id in selected:
        spec = WAVE1[model_id]
        snap = resolve_snapshot(spec, extra_roots=list(args.local_root))
        model_out = out_dir / model_id
        model_out.mkdir(parents=True, exist_ok=True)
        row: dict[str, object] = {
            "id": model_id,
            "repo": spec["repo"],
            "family": spec["family"],
            "snapshot": str(snap) if snap else None,
        }
        if snap is None:
            row["status"] = "missing_snapshot"
            scoreboard.append(row)
            continue

        if not args.skip_ax:
            ax_json = model_out / "ax-direct.json"
            ax_log = model_out / "ax-direct.log"
            cmd = [
                str(args.python),
                str(STACK),
                "--model",
                spec["repo"],
                "--model-repo-id",
                spec["repo"],
                "--model-dir",
                str(snap),
                "--prompt-tokens",
                "128,512,2048",
                "--generation-tokens",
                "128",
                "--repetitions",
                "5",
                "--warmup-repetitions",
                "2",
                "--cooldown",
                "15",
                "--no-build-ax-engine",
                "--skip-mlx-lm",
                "--ax-direct",
                "--output",
                str(ax_json),
            ]
            row["ax_exit"] = run(cmd, ax_log)
            row["ax"] = {
                str(pt): {"prefill_tok_s": pre, "decode_tok_s": dec}
                for pt, (pre, dec) in ax_medians(ax_json).items()
            }

        if not args.skip_mlxcel and args.mlxcel_bin.is_file():
            mlxcel: dict[str, dict[str, float]] = {}
            for pt in (128, 512, 2048):
                log = model_out / f"mlxcel-p{pt}.log"
                cmd = [
                    str(args.mlxcel_bin),
                    "-m",
                    str(snap),
                    "-p",
                    "Hello, how are you today?",
                    "-n",
                    "128",
                    "--warmup-tokens",
                    "32",
                    "--prompt-tokens",
                    str(pt),
                ]
                run(cmd, log)
                mlxcel[str(pt)] = parse_mlxcel_log(log)
            row["mlxcel"] = mlxcel

        row["status"] = "ok"
        scoreboard.append(row)
        (model_out / "row.json").write_text(json.dumps(row, indent=2) + "\n")

    meta["finished_at"] = utc_now()
    meta["scoreboard"] = scoreboard
    (out_dir / "scoreboard.json").write_text(json.dumps(meta, indent=2) + "\n")

    lines = [
        "# Fleet AX vs mlxcel scoreboard",
        "",
        f"Host: `{host}`",
        f"Commit: `{commit}`",
        "Bar: decode ≥ 0.97× mlxcel, prefill ≥ 0.90× mlxcel (unrounded).",
        "",
        "| model | p | AX pre | AX dec | mlxcel pre | mlxcel dec | pre ratio | dec ratio | 6a |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in scoreboard:
        ax = row.get("ax") or {}
        peer = row.get("mlxcel") or {}
        if not isinstance(ax, dict):
            lines.append(f"| {row['id']} |  |  |  |  |  |  |  | {row.get('status')} |")
            continue
        for pt in ("128", "512", "2048"):
            a = ax.get(pt) or {}
            m = peer.get(pt) or {}
            ap = float(a.get("prefill_tok_s") or 0)
            ad = float(a.get("decode_tok_s") or 0)
            mp = float(m.get("prefill_tok_s") or 0)
            md = float(m.get("decode_tok_s") or 0)
            prer = (ap / mp) if mp > 0 else 0.0
            decr = (ad / md) if md > 0 else 0.0
            gate = "PASS" if prer >= 0.90 and decr >= 0.97 else "FAIL"
            if mp <= 0 or md <= 0:
                gate = "NO_PEER"
            lines.append(
                f"| {row['id']} | {pt} | {ap:.4f} | {ad:.4f} | {mp:.4f} | {md:.4f} "
                f"| {prer:.6f} | {decr:.6f} | {gate} |"
            )
    (out_dir / "scoreboard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
