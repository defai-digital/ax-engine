#!/usr/bin/env python3
"""Sequential direct/MTP QA matrix runner for AX Engine goal verification."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("QA_SCRATCH", str(REPO / "qa/reports/matrix")))
SERVER_BIN = Path(
    os.environ.get(
        "QA_SERVER_BIN",
        str(REPO / "target/debug/ax-engine-server"),
    )
)
PORT = int(os.environ.get("QA_PORT", "18440"))
SEED = int(os.environ.get("QA_SEED", "20260716"))
SAMPLE = int(os.environ.get("QA_SAMPLE", "8"))
TIMEOUT = int(os.environ.get("QA_TIMEOUT", "180"))
READY_MAX = int(os.environ.get("QA_READY_MAX", "420"))
HOST = "127.0.0.1"


@dataclass
class Cell:
    mode: str  # direct | mtp
    model_id: str
    artifacts: Path
    status: str = "pending"  # ok|skip|engine_fail|model_quality|error
    pass_line: str = ""
    log_path: Path | None = None
    note: str = ""


def load_matrix(path: Path) -> list[Cell]:
    cells: list[Cell] = []
    for line in path.read_text().splitlines():
        if not line.startswith("OK|"):
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        _, mode, mid, art = parts
        cells.append(Cell(mode=mode, model_id=mid, artifacts=Path(art)))
    return cells


def wait_ready(timeout: int) -> bool:
    url = f"http://{HOST}:{PORT}/v1/models"
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


def kill_port() -> None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f"tcp:{PORT}"], text=True)
    except subprocess.CalledProcessError:
        return
    for pid in out.split():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(1)
    try:
        out = subprocess.check_output(["lsof", "-ti", f"tcp:{PORT}"], text=True)
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
        "content\": null",
        "AttributeError",
        "server_died",
        "failed to load",
        "Load error",
        "WorkerPanicked",
        "Engine(MetalRuntime",
        "HTTP Error 5",
        "Connection refused",
    ]
    blob = (log_text + "\n" + qa_text).lower()
    for n in needles:
        if n.lower() in blob:
            return n
    if "results:" not in qa_text.lower() and "error" in qa_text.lower():
        return "qa_suite_incomplete"
    return None


def run_cell(cell: Cell) -> Cell:
    safe = cell.model_id.replace("/", "_").replace(" ", "_")
    log_path = SCRATCH / f"qa-{cell.mode}-{safe}.log"
    server_log = SCRATCH / f"server-{cell.mode}-{safe}.log"
    cell.log_path = log_path

    if not (cell.artifacts / "config.json").is_file():
        cell.status = "skip"
        cell.note = "missing config.json"
        log_path.write_text(f"SKIP {cell.note}\n")
        return cell

    kill_port()
    cmd = [
        str(SERVER_BIN),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--model-id",
        cell.model_id if cell.mode == "direct" else f"mtp-{safe}",
        "--support-tier",
        "mlx-preview",
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(cell.artifacts),
        "--disable-ngram-acceleration",
    ]
    with server_log.open("w") as slog:
        proc = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT, cwd=str(REPO))

    try:
        if not wait_ready(READY_MAX):
            slog_text = server_log.read_text(errors="replace")
            cell.status = "engine_fail"
            cell.note = classify_engine_fail(slog_text, "") or "server_not_ready"
            log_path.write_text(
                f"ENGINE_FAIL start\nnote={cell.note}\n\n--- server log ---\n{slog_text[-8000:]}\n"
            )
            return cell

        # Quick null-content smoke before full suite
        try:
            payload = json.dumps(
                {
                    "model": cmd[cmd.index("--model-id") + 1],
                    "messages": [{"role": "user", "content": "Reply with the word ok."}],
                    "max_tokens": 16,
                    "temperature": 0,
                }
            ).encode()
            req = urllib.request.Request(
                f"http://{HOST}:{PORT}/v1/chat/completions",
                data=payload,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = json.loads(resp.read().decode())
            content = (body.get("choices") or [{}])[0].get("message", {}).get("content")
            if content is None:
                cell.status = "engine_fail"
                cell.note = "content_null"
                log_path.write_text(
                    "ENGINE_FAIL content null\n"
                    + json.dumps(body, indent=2)[:4000]
                    + "\n\n"
                    + server_log.read_text(errors="replace")[-4000:]
                )
                return cell
        except Exception as exc:
            cell.status = "engine_fail"
            cell.note = f"smoke_chat_error:{exc}"
            log_path.write_text(
                f"ENGINE_FAIL smoke\n{exc}\n\n"
                + server_log.read_text(errors="replace")[-6000:]
            )
            return cell

        qa_cmd = [
            sys.executable,
            str(REPO / "qa/run_qa.py"),
            "--base-url",
            f"http://{HOST}:{PORT}",
            "--model",
            cmd[cmd.index("--model-id") + 1],
            "--modes",
            cell.mode if cell.mode in ("direct", "mtp") else "direct",
            "--streams",
            "false",
            "--max-tokens",
            "256",
            "--temperature",
            "0.0",
            "--timeout",
            str(TIMEOUT),
            "--sample",
            str(SAMPLE),
            "--seed",
            str(SEED),
            "--output",
            str(SCRATCH / f"report-{cell.mode}-{safe}.html"),
        ]
        # run_qa modes label only; keep direct label for both (product path is same server)
        qa_cmd[qa_cmd.index("--modes") + 1] = "direct"

        qa_proc = subprocess.run(
            qa_cmd,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=TIMEOUT * (SAMPLE + 4),
        )
        qa_out = (qa_proc.stdout or "") + "\n" + (qa_proc.stderr or "")
        server_tail = server_log.read_text(errors="replace")[-6000:]
        engine = classify_engine_fail(server_tail, qa_out)
        pass_line = ""
        for line in qa_out.splitlines():
            if line.strip().startswith("Results:"):
                pass_line = line.strip()
        cell.pass_line = pass_line

        log_path.write_text(
            f"mode={cell.mode} model={cell.model_id}\n"
            f"artifacts={cell.artifacts}\n"
            f"qa_rc={qa_proc.returncode}\n"
            f"{pass_line}\n"
            f"engine_needle={engine}\n\n"
            f"--- qa stdout/err ---\n{qa_out}\n\n"
            f"--- server tail ---\n{server_tail}\n"
        )

        if engine:
            cell.status = "engine_fail"
            cell.note = engine
        elif not pass_line:
            cell.status = "error"
            cell.note = "no Results line"
        else:
            # Extract numbers
            # Results: X/Y passed
            import re

            m = re.search(r"Results:\s*(\d+)/(\d+)\s*passed", pass_line)
            if not m:
                cell.status = "error"
                cell.note = "unparseable Results"
            else:
                x, y = int(m.group(1)), int(m.group(2))
                if x == y:
                    cell.status = "ok"
                    cell.note = "all_pass"
                else:
                    cell.status = "model_quality"
                    cell.note = f"partial {x}/{y}"
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
        kill_port()
        time.sleep(2)


def ensure_scratch() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_scratch()
    matrix = SCRATCH / "qa-matrix.txt"
    if not matrix.is_file():
        catalog = REPO / "qa" / "matrix-catalog.txt"
        if catalog.is_file():
            matrix = catalog
        else:
            print("missing matrix", matrix)
            return 2
    if not SERVER_BIN.is_file():
        print("missing server", SERVER_BIN)
        return 2

    cells = load_matrix(matrix)
    # Prefer chat LLMs: skip pure assistant-only weights that aren't packages?
    # Keep all OK MTP cells from matrix.
    print(f"Running {len(cells)} cells, sample={SAMPLE} seed={SEED}")
    results: list[Cell] = []
    for i, cell in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {cell.mode} {cell.model_id}", flush=True)
        # Skip enormous models that are optional if we want speed? Plan says attempt all.
        # gpt-oss-120b and llama3.3-70b may take long / OOM - still attempt.
        try:
            run_cell(cell)
        except Exception as exc:
            cell.status = "error"
            cell.note = str(exc)
            if cell.log_path:
                cell.log_path.write_text(f"EXCEPTION {exc}\n")
        print(f"  -> {cell.status} {cell.pass_line} {cell.note}", flush=True)
        results.append(cell)

    summary = SCRATCH / "qa-summary.md"
    lines = [
        "# QA matrix summary",
        "",
        f"- seed: `{SEED}`",
        f"- sample: `{SAMPLE}`",
        f"- server: `{SERVER_BIN}`",
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
    lines += [
        "",
        "## Engine failures",
        "",
    ]
    if not engine_fails:
        lines.append("_None_")
    else:
        for c in engine_fails:
            lines.append(f"- **{c.mode}/{c.model_id}**: `{c.note}` — see `{c.log_path}`")
    lines += [
        "",
        "## Model-quality / partial",
        "",
    ]
    mq = [c for c in results if c.status == "model_quality"]
    if not mq:
        lines.append("_None_")
    else:
        for c in mq:
            lines.append(f"- **{c.mode}/{c.model_id}**: {c.pass_line} ({c.note})")
    summary.write_text("\n".join(lines) + "\n")
    print("\nWrote", summary)
    print("engine_fails", len(engine_fails))
    return 1 if engine_fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
