#!/usr/bin/env python3
"""Unified sequential QA matrix runner (direct / ngram / MTP / embed / multimodal).

Single orchestration entrypoint for local multi-model verification.
``qa/run_full_qa.sh`` is a thin wrapper that materializes an inventory and
invokes this script.

Inventory lines::

    OK|direct|model_id|/path/to/artifacts
    OK|ngram|model_id|/path/to/artifacts
    OK|mtp|model_id|/path/to/artifacts
    OK|embed|model_id|/path/to/artifacts
    OK|multimodal|model_id|/path/to/artifacts

Exit codes:
  0 — no engine failures (model_quality partials allowed by default)
  1 — one or more engine_fail cells
  2 — bad configuration (missing matrix / server binary)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCRATCH = REPO / "qa" / "reports" / "matrix"
HOST = "127.0.0.1"


@dataclass
class Cell:
    mode: str  # direct | ngram | mtp | embed | multimodal
    model_id: str
    artifacts: Path
    status: str = "pending"  # ok|skip|engine_fail|model_quality|error
    pass_line: str = ""
    log_path: Path | None = None
    note: str = ""
    surface_line: str = ""


def load_matrix(path: Path) -> list[Cell]:
    cells: list[Cell] = []
    for line in path.read_text().splitlines():
        if not line.startswith("OK|"):
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        _, mode, mid, art = parts
        mode = mode.strip().lower()
        if mode not in ("direct", "ngram", "mtp", "embed", "multimodal"):
            continue
        cells.append(Cell(mode=mode, model_id=mid.strip(), artifacts=Path(art.strip())))
    return cells


def write_inventory(path: Path, cells: list[tuple[str, str, str]]) -> None:
    """Write OK|mode|model_id|artifacts lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"OK|{mode}|{mid}|{art}" for mode, mid, art in cells]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def wait_ready(host: str, port: int, timeout: int) -> bool:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"], text=True)
    except subprocess.CalledProcessError:
        return
    for pid in out.split():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(1)
    try:
        out = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"], text=True)
    except subprocess.CalledProcessError:
        return
    for pid in out.split():
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass


def classify_engine_fail(log_text: str, qa_text: str) -> str | None:
    needles = [
        "panic",
        "InvalidManifest",
        "is not supported by the MLX runner",
        "not implemented for",
        'content": null',
        "AttributeError",
        "server_died",
        "failed to load",
        "Load error",
        "WorkerPanicked",
        "Engine(MetalRuntime",
        "HTTP Error 5",
        "Connection refused",
        "mtp_path_not_exercised",
        "surface_hard_fail",
        "multimodal_hard_fail",
        "not_multimodal_package",
    ]
    blob = (log_text + "\n" + qa_text).lower()
    for n in needles:
        if n.lower() in blob:
            return n
    if "results:" not in qa_text.lower() and "error" in qa_text.lower():
        return "qa_suite_incomplete"
    return None


def package_looks_like_mtp(artifacts: Path) -> bool:
    """True when package ships MTP weights (fused sidecar or Gemma assistant)."""
    if (artifacts / "mtp.safetensors").is_file():
        return True
    if (artifacts / "glm_mtp.safetensors").is_file():
        return True
    if (artifacts / "ax_gemma4_assistant_mtp.json").is_file():
        return True
    if (artifacts / "assistant").is_dir() and any(
        (artifacts / "assistant").rglob("*.safetensors")
    ):
        return True
    return False


def package_looks_like_multimodal(artifacts: Path) -> bool:
    """True when package config looks like Gemma 4 unified multimodal."""
    cfg_path = artifacts / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(cfg, dict):
        return False
    if "vision_config" in cfg and (
        "image_token_id" in cfg or "boi_token_id" in cfg
    ):
        return True
    if (artifacts / "preprocessor_config.json").is_file() and "image_token_id" in cfg:
        return True
    return False


def build_server_cmd(
    cell: Cell,
    model_id: str,
    *,
    server_bin: Path,
    host: str,
    port: int,
) -> list[str]:
    """Build ax-engine-server argv for direct / ngram / MTP / embed / multimodal.

    Critical: MTP packages must NOT pass ``--disable-ngram-acceleration``.
    That flag sets ``mtp_requested=false`` and forces DirectFallback.
    """
    cmd = [
        str(server_bin),
        "--host",
        host,
        "--port",
        str(port),
        "--model-id",
        model_id,
        "--support-tier",
        "mlx-preview",
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(cell.artifacts),
    ]
    if cell.mode in ("direct", "embed", "multimodal"):
        # Multimodal / embed / direct: pure path without n-gram stacking noise.
        cmd.append("--disable-ngram-acceleration")
    elif cell.mode == "ngram":
        # Default server path: n-gram acceleration eligible.
        pass
    elif cell.mode == "mtp":
        cmd.append("--mlx-mtp-disable-ngram-stacking")
    else:
        raise ValueError(f"unknown mode {cell.mode}")
    return cmd


def mtp_telemetry_active(crossover: dict) -> bool:
    if not crossover:
        return False
    if int(crossover.get("ax_mtp_source_mtp_proposer_wall_us") or 0) > 0:
        return True
    if int(crossover.get("ax_mtp_source_assistant_proposer_wall_us") or 0) > 0:
        return True
    if int(crossover.get("ax_mtp_verify_tokens") or 0) > 0:
        return True
    if int(crossover.get("ax_mtp_draft_tokens") or 0) > 0:
        return True
    if int(crossover.get("ax_mtp_decode_steps") or 0) > 0:
        return True
    if int(crossover.get("ax_mlx_gemma4_assistant_mtp_enabled") or 0) == 1 and (
        int(crossover.get("ax_mlx_gemma4_assistant_mtp_draft_tokens") or 0) > 0
        or int(crossover.get("ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us") or 0)
        > 0
    ):
        return True
    return False


def probe_mtp_route(
    artifacts: Path,
    out_json: Path,
    *,
    repo: Path,
    timeout: int,
) -> tuple[bool, dict, str]:
    bench = repo / "target/debug/ax-engine-bench"
    if not bench.is_file():
        bench = repo / "target/release/ax-engine-bench"
    if not bench.is_file():
        return False, {}, f"missing bench binary under {repo / 'target'}"
    tokens = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
    cmd = [
        str(bench),
        "generate",
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(artifacts),
        "--tokens",
        tokens,
        "--max-output-tokens",
        "32",
        "--json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=max(timeout * 4, 300),
            env={**os.environ, "AX_NO_SPEC": "0"},
        )
    except Exception as exc:
        return False, {}, f"bench generate failed: {exc}"
    raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out_json.write_text(proc.stdout or "")
    if proc.returncode != 0:
        return False, {}, f"bench_rc={proc.returncode}\n{raw[-2000:]}"
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return False, {}, f"invalid generate json: {exc}\n{raw[-1500:]}"
    decisions = (
        (data.get("route") or {}).get("crossover_decisions")
        if isinstance(data, dict)
        else None
    )
    if not isinstance(decisions, dict):
        return False, {}, "missing route.crossover_decisions"
    return mtp_telemetry_active(decisions), decisions, ""


def run_cell(
    cell: Cell,
    *,
    repo: Path,
    scratch: Path,
    server_bin: Path,
    host: str,
    port: int,
    seed: int,
    sample: int,
    timeout: int,
    ready_max: int,
    run_surface: bool,
    streams: str,
    embed_tier: str = "standard",
    multimodal_tier: str = "smoke",
) -> Cell:
    safe = cell.model_id.replace("/", "_").replace(" ", "_")
    log_path = scratch / f"qa-{cell.mode}-{safe}.log"
    server_log = scratch / f"server-{cell.mode}-{safe}.log"
    cell.log_path = log_path

    if not (cell.artifacts / "config.json").is_file():
        cell.status = "skip"
        cell.note = "missing config.json"
        log_path.write_text(f"SKIP {cell.note}\n")
        return cell

    model_id = cell.model_id if cell.mode != "mtp" else f"mtp-{safe}"
    cmd = build_server_cmd(
        cell, model_id, server_bin=server_bin, host=host, port=port
    )
    if cell.mode == "mtp" and "--disable-ngram-acceleration" in cmd:
        cell.status = "engine_fail"
        cell.note = "mtp_cmd_has_disable_ngram"
        log_path.write_text(f"ENGINE_FAIL bad cmd: {cmd}\n")
        return cell

    # Embedding packages must ship a tokenizer for client-side ID encoding.
    if cell.mode == "embed":
        tok = cell.artifacts / "tokenizer.json"
        if not tok.is_file():
            cell.status = "skip"
            cell.note = "missing tokenizer.json"
            log_path.write_text(f"SKIP {cell.note}\n")
            return cell

    mtp_probe_note = ""
    mtp_decisions: dict = {}
    if cell.mode == "mtp":
        if not package_looks_like_mtp(cell.artifacts):
            cell.status = "skip"
            cell.note = "not_mtp_package"
            log_path.write_text(
                "SKIP not_mtp_package\n"
                f"artifacts={cell.artifacts}\n"
                f"server_cmd={' '.join(cmd)}\n"
            )
            return cell
        probe_json = scratch / f"mtp-route-{safe}.json"
        active, mtp_decisions, mtp_err = probe_mtp_route(
            cell.artifacts, probe_json, repo=repo, timeout=timeout
        )
        if not active:
            cell.status = "engine_fail"
            cell.note = "mtp_path_not_exercised"
            log_path.write_text(
                "ENGINE_FAIL mtp_path_not_exercised\n"
                f"server_cmd={' '.join(cmd)}\n"
                f"err={mtp_err}\n"
                f"decisions_sample={json.dumps({k: mtp_decisions.get(k) for k in sorted(mtp_decisions) if 'mtp' in k.lower()}, indent=2)[:6000]}\n"
            )
            return cell
        mtp_probe_note = (
            f"mtp_active proposer_us={mtp_decisions.get('ax_mtp_source_mtp_proposer_wall_us')} "
            f"verify_tokens={mtp_decisions.get('ax_mtp_verify_tokens')}"
        )

    if cell.mode == "multimodal" and not package_looks_like_multimodal(cell.artifacts):
        cell.status = "skip"
        cell.note = "not_multimodal_package"
        log_path.write_text(
            "SKIP not_multimodal_package\n"
            f"artifacts={cell.artifacts}\n"
            f"server_cmd={' '.join(cmd)}\n"
        )
        return cell

    kill_port(port)
    with server_log.open("w") as slog:
        proc = subprocess.Popen(
            cmd,
            stdout=slog,
            stderr=subprocess.STDOUT,
            cwd=str(repo),
            env={**os.environ, "AX_ALLOW_UNSUPPORTED_HOST": os.environ.get("AX_ALLOW_UNSUPPORTED_HOST", "1")},
        )

    try:
        if not wait_ready(host, port, ready_max):
            slog_text = server_log.read_text(errors="replace")
            cell.status = "engine_fail"
            cell.note = classify_engine_fail(slog_text, "") or "server_not_ready"
            log_path.write_text(
                f"ENGINE_FAIL start\nnote={cell.note}\ncmd={' '.join(cmd)}\n"
                f"mtp_probe={mtp_probe_note}\n\n--- server log ---\n{slog_text[-8000:]}\n"
            )
            return cell

        surface_blob = ""
        base = f"http://{host}:{port}"

        # ---- Embedding cells: embedding probes only (no chat bank) ----
        if cell.mode == "embed":
            sys.path.insert(0, str(repo / "qa"))
            from embedding_probes import run_embedding_probes  # noqa: WPS433

            tok_path = str(cell.artifacts / "tokenizer.json")
            embed_report = run_embedding_probes(
                base,
                model_id,
                tok_path,
                artifacts_hint=str(cell.artifacts),
                timeout=float(timeout),
                tier=embed_tier,
            )
            embed_json = scratch / f"embed-{safe}.json"
            embed_json.write_text(json.dumps(embed_report.as_dict(), indent=2))
            cell.pass_line = embed_report.summary_line
            cell.surface_line = embed_report.summary_line
            log_path.write_text(
                f"mode=embed model={cell.model_id}\n"
                f"artifacts={cell.artifacts}\n"
                f"server_cmd={' '.join(cmd)}\n"
                f"{embed_report.summary_line}\n\n"
                f"{json.dumps(embed_report.as_dict(), indent=2)}\n\n"
                f"--- server tail ---\n"
                + server_log.read_text(errors="replace")[-6000:]
            )
            if embed_report.hard_passed:
                cell.status = "ok"
                cell.note = "embed_all_pass"
            else:
                failed = [
                    r.name
                    for r in embed_report.results
                    if r.hard and not r.passed and not r.skipped
                ]
                # Shape/L2/batch/empty are engine; semantic can be model_quality
                engine_names = {
                    "api_shape",
                    "l2_normalized",
                    "batch_vs_single",
                    "determinism",
                    "empty_rejected",
                }
                # STS / pair / retrieval failures are model_quality by default
                if any(n in engine_names for n in failed):
                    cell.status = "engine_fail"
                    cell.note = "embed_engine_fail:" + ",".join(failed)
                else:
                    cell.status = "model_quality"
                    cell.note = "embed_partial:" + ",".join(failed)
            return cell

        # ---- Multimodal cells: policy + image path (+ content at standard) ----
        if cell.mode == "multimodal":
            sys.path.insert(0, str(repo / "qa"))
            from multimodal_probes import run_multimodal_probes  # noqa: WPS433

            mm_report = run_multimodal_probes(
                base,
                model_id,
                timeout=float(timeout),
                tier=multimodal_tier,
                artifacts=cell.artifacts,
                require_image=True,
            )
            mm_json = scratch / f"multimodal-{safe}.json"
            mm_json.write_text(json.dumps(mm_report.as_dict(), indent=2))
            cell.pass_line = mm_report.summary_line
            cell.surface_line = mm_report.summary_line
            log_path.write_text(
                f"mode=multimodal model={cell.model_id}\n"
                f"artifacts={cell.artifacts}\n"
                f"server_cmd={' '.join(cmd)}\n"
                f"tier={multimodal_tier}\n"
                f"{mm_report.summary_line}\n\n"
                f"{json.dumps(mm_report.as_dict(), indent=2)}\n\n"
                f"--- server tail ---\n"
                + server_log.read_text(errors="replace")[-6000:]
            )
            if mm_report.hard_passed:
                cell.status = "ok"
                cell.note = "multimodal_all_pass"
            else:
                failed = [
                    r.name
                    for r in mm_report.results
                    if r.hard and not r.passed and not r.skipped
                ]
                # Policy / empty / 5xx = engine; color content may be model_quality
                engine_names = {
                    "remote_media_rejected",
                    "video_rejected",
                    "multimodal_image",
                    "image_describe_smoke",
                }
                quality_names = {"image_color_content"}
                if any(n in engine_names for n in failed):
                    cell.status = "engine_fail"
                    cell.note = "multimodal_hard_fail:" + ",".join(failed)
                elif any(n in quality_names for n in failed):
                    cell.status = "model_quality"
                    cell.note = "multimodal_partial:" + ",".join(failed)
                else:
                    cell.status = "engine_fail"
                    cell.note = "multimodal_hard_fail:" + ",".join(failed)
            return cell

        # ---- Chat cells: null-content smoke + optional surface + bank ----
        try:
            payload = json.dumps(
                {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Reply with the word ok."}],
                    "max_tokens": 16,
                    "temperature": 0,
                }
            ).encode()
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=payload,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())
            content = (body.get("choices") or [{}])[0].get("message", {}).get("content")
            if content is None:
                cell.status = "engine_fail"
                cell.note = "content_null"
                log_path.write_text(
                    "ENGINE_FAIL content null\n"
                    f"cmd={' '.join(cmd)}\n"
                    + json.dumps(body, indent=2)[:4000]
                    + "\n\n"
                    + server_log.read_text(errors="replace")[-4000:]
                )
                return cell
        except Exception as exc:
            cell.status = "engine_fail"
            cell.note = f"smoke_chat_error:{exc}"
            log_path.write_text(
                f"ENGINE_FAIL smoke\ncmd={' '.join(cmd)}\n{exc}\n\n"
                + server_log.read_text(errors="replace")[-6000:]
            )
            return cell

        if run_surface:
            sys.path.insert(0, str(repo / "qa"))
            from surface_probes import run_surface_probes  # noqa: WPS433

            surface = run_surface_probes(
                base,
                model_id,
                timeout=float(timeout),
            )
            surface_json = scratch / f"surface-{cell.mode}-{safe}.json"
            surface_json.write_text(json.dumps(surface.as_dict(), indent=2))
            cell.surface_line = surface.summary_line
            surface_blob = json.dumps(surface.as_dict(), indent=2)
            if not surface.hard_passed:
                cell.status = "engine_fail"
                cell.note = "surface_hard_fail"
                log_path.write_text(
                    f"ENGINE_FAIL surface\n{surface.summary_line}\n\n{surface_blob}\n\n"
                    + server_log.read_text(errors="replace")[-4000:]
                )
                return cell

        report_html = scratch / f"report-{cell.mode}-{safe}.html"
        report_json = scratch / f"report-{cell.mode}-{safe}.json"
        mode_label = cell.mode if cell.mode != "mtp" else "mtp"
        qa_cmd = [
            sys.executable,
            str(repo / "qa/run_qa.py"),
            "--base-url",
            base,
            "--model",
            model_id,
            "--mode",
            mode_label,
            "--streams",
            streams,
            "--max-tokens",
            "256",
            "--temperature",
            "0.0",
            "--timeout",
            str(timeout),
            "--sample",
            str(sample),
            "--seed",
            str(seed),
            "--allow-partial",
            "--output",
            str(report_html),
            "--json-output",
            str(report_json),
        ]

        qa_proc = subprocess.run(
            qa_cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout * (sample + 4) * (2 if streams == "both" else 1),
        )
        qa_out = (qa_proc.stdout or "") + "\n" + (qa_proc.stderr or "")
        server_tail = server_log.read_text(errors="replace")[-6000:]
        engine = classify_engine_fail(server_tail, qa_out)
        pass_line = ""
        for line in qa_out.splitlines():
            if line.strip().startswith("Results:"):
                pass_line = line.strip()

        x = y = None
        if report_json.is_file():
            try:
                payload = json.loads(report_json.read_text())
                totals = payload.get("totals") or {}
                x = int(totals.get("hard_passed", 0))
                y = int(totals.get("items", 0))
                if y > 0:
                    pass_line = (
                        f"Results: {x}/{y} passed ({x / y * 100:.1f}%) [hard checks]"
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if x is None or y is None:
            m = re.search(r"Results:\s*(\d+)/(\d+)\s*passed", pass_line)
            if m:
                x, y = int(m.group(1)), int(m.group(2))

        cell.pass_line = pass_line
        log_path.write_text(
            f"mode={cell.mode} model={cell.model_id}\n"
            f"artifacts={cell.artifacts}\n"
            f"server_cmd={' '.join(cmd)}\n"
            f"mtp_probe={mtp_probe_note}\n"
            f"surface={cell.surface_line}\n"
            f"qa_rc={qa_proc.returncode}\n"
            f"json={report_json}\n"
            f"{pass_line}\n"
            f"engine_needle={engine}\n\n"
            f"--- surface ---\n{surface_blob[:4000]}\n\n"
            f"--- qa stdout/err ---\n{qa_out}\n\n"
            f"--- server tail ---\n{server_tail}\n"
        )

        if engine:
            cell.status = "engine_fail"
            cell.note = engine
        elif x is None or y is None or y == 0:
            cell.status = "error"
            cell.note = "no Results / empty JSON"
        elif x == y:
            cell.status = "ok"
            cell.note = "all_pass"
        else:
            cell.status = "model_quality"
            cell.note = f"partial {x}/{y}"
        if mtp_probe_note and cell.status in ("ok", "model_quality"):
            cell.note = f"{cell.note}; {mtp_probe_note}"
        if cell.surface_line and cell.status in ("ok", "model_quality"):
            cell.note = f"{cell.note}; {cell.surface_line}"
        return cell
    finally:
        try:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:
            pass
        kill_port(port)
        time.sleep(1)


def write_summary(
    path: Path,
    results: list[Cell],
    *,
    seed: int,
    sample: int,
    server_bin: Path,
) -> list[Cell]:
    lines = [
        "# QA matrix summary",
        "",
        f"- seed: `{seed}`",
        f"- sample: `{sample}`",
        f"- server: `{server_bin}`",
        "",
        "| mode | model_id | status | results | note | log |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in results:
        logn = c.log_path.name if c.log_path else ""
        lines.append(
            f"| {c.mode} | `{c.model_id}` | {c.status} | {c.pass_line or '-'} | {c.note} | `{logn}` |"
        )
    engine_fails = [c for c in results if c.status == "engine_fail"]
    lines += ["", "## Engine failures", ""]
    if not engine_fails:
        lines.append("_None_")
    else:
        for c in engine_fails:
            lines.append(f"- **{c.mode}/{c.model_id}**: `{c.note}` — see `{c.log_path}`")
    lines += ["", "## Model-quality / partial", ""]
    mq = [c for c in results if c.status == "model_quality"]
    if not mq:
        lines.append("_None_")
    else:
        for c in mq:
            lines.append(f"- **{c.mode}/{c.model_id}**: {c.pass_line} ({c.note})")
    path.write_text("\n".join(lines) + "\n")
    return engine_fails


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified AX Engine QA matrix runner")
    p.add_argument(
        "--matrix",
        default=None,
        help="Inventory file (OK|mode|model_id|artifacts). Default: $QA_SCRATCH/qa-matrix.txt",
    )
    p.add_argument(
        "--scratch",
        default=None,
        help="Output/scratch directory (default: qa/reports/matrix or $QA_SCRATCH)",
    )
    p.add_argument(
        "--server-bin",
        default=None,
        help="ax-engine-server binary path",
    )
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--ready-max", type=int, default=None)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["direct", "ngram", "mtp", "embed", "multimodal"],
        default=None,
        help=(
            "Filter inventory to these modes "
            "(embed = embedding QA; multimodal = vision policy+path QA)"
        ),
    )
    p.add_argument(
        "--surface",
        action="store_true",
        help=(
            "Run product-surface probes "
            "(concurrency/stream/cancel/tools/media-policy/multimodal)"
        ),
    )
    p.add_argument(
        "--embed-tier",
        default=os.environ.get("QA_EMBED_TIER", "standard"),
        choices=["smoke", "standard"],
        help="Embedding QA tier: smoke (engine) or standard (+ pair/retrieval)",
    )
    p.add_argument(
        "--multimodal-tier",
        default=os.environ.get("QA_MULTIMODAL_TIER", "smoke"),
        choices=["smoke", "standard"],
        help=(
            "Multimodal QA tier for multimodal cells: "
            "smoke (policy+path) or standard (+ color/describe content)"
        ),
    )
    p.add_argument(
        "--streams",
        default="false",
        choices=["true", "false", "both"],
        help="Passed to run_qa.py (default: false)",
    )
    p.add_argument(
        "--fail-on-model-quality",
        action="store_true",
        help="Exit non-zero on model_quality partials as well as engine_fail",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scratch = Path(
        args.scratch
        or os.environ.get("QA_SCRATCH", str(DEFAULT_SCRATCH))
    )
    scratch.mkdir(parents=True, exist_ok=True)

    server_bin = Path(
        args.server_bin
        or os.environ.get(
            "QA_SERVER_BIN",
            str(
                REPO / "target/debug/ax-engine-server"
                if (REPO / "target/debug/ax-engine-server").is_file()
                else REPO / "target/release/ax-engine-server"
            ),
        )
    )
    port = int(args.port or os.environ.get("QA_PORT", "18440"))
    seed = int(args.seed or os.environ.get("QA_SEED", "20260716"))
    sample = int(args.sample or os.environ.get("QA_SAMPLE", "8"))
    timeout = int(args.timeout or os.environ.get("QA_TIMEOUT", "180"))
    ready_max = int(args.ready_max or os.environ.get("QA_READY_MAX", "420"))
    run_surface = bool(args.surface or os.environ.get("QA_SURFACE", "0") == "1")

    matrix = Path(args.matrix) if args.matrix else scratch / "qa-matrix.txt"
    if not matrix.is_file():
        catalog = REPO / "qa" / "matrix-catalog.txt"
        if catalog.is_file():
            matrix = catalog
        else:
            print("missing matrix inventory:", matrix, file=sys.stderr)
            print(
                "Write OK|mode|model_id|artifacts lines, or use qa/run_full_qa.sh",
                file=sys.stderr,
            )
            return 2
    if not server_bin.is_file():
        print("missing server binary:", server_bin, file=sys.stderr)
        return 2

    cells = load_matrix(matrix)
    if args.modes:
        wanted = set(args.modes)
        cells = [c for c in cells if c.mode in wanted]
    if not cells:
        print("no cells after filters", file=sys.stderr)
        return 2

    print(
        f"Running {len(cells)} cells sample={sample} seed={seed} "
        f"surface={run_surface} server={server_bin}"
    )
    results: list[Cell] = []
    for i, cell in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {cell.mode} {cell.model_id}", flush=True)
        try:
            run_cell(
                cell,
                repo=REPO,
                scratch=scratch,
                server_bin=server_bin,
                host=HOST,
                port=port,
                seed=seed,
                sample=sample,
                timeout=timeout,
                ready_max=ready_max,
                run_surface=run_surface,
                streams=args.streams,
                embed_tier=args.embed_tier,
                multimodal_tier=args.multimodal_tier,
            )
        except Exception as exc:
            cell.status = "error"
            cell.note = str(exc)
            if cell.log_path:
                cell.log_path.write_text(f"EXCEPTION {exc}\n")
        print(f"  -> {cell.status} {cell.pass_line} {cell.note}", flush=True)
        results.append(cell)

    summary = scratch / "qa-summary.md"
    engine_fails = write_summary(
        summary, results, seed=seed, sample=sample, server_bin=server_bin
    )
    print("\nWrote", summary)
    print("engine_fails", len(engine_fails))
    if engine_fails:
        return 1
    if args.fail_on_model_quality and any(c.status == "model_quality" for c in results):
        return 1
    if any(c.status == "error" for c in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
