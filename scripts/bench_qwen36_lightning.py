#!/usr/bin/env python3
"""Qwen3.6-27B and 35B MTP benchmark following lightning-mlx methodology.

Mirrors the lightning-mlx raw-decode benchmark approach:
  - Single sequence (batch=1, no prefix cache)
  - 3 prompt categories: high_repeat, med_repeat, low_repeat
  - temperature=0.6, top_p=0.95, top_k=20 (real-world sampling)
  - 2 warmup + 5 measured runs, median tok/s
  - Reports: baseline AR, MTP D1/D2/D3, speedup, accept rate by depth

Compare with lightning-mlx published numbers:
  - Qwen3.6-27B: 70.35 tok/s MTP-D3 (M5 Max, Burst, optimistic, no-thinking)
  - Qwen3.6-35B: 226.01 tok/s MTP-D3 (M5 Max, Burst, optimistic, no-thinking)
  NOTE: those numbers use --mtp-optimistic (greedy argmax), not temp=0.6
  sampling. Our benchmark uses correct rejection sampling — expect lower tok/s
  but honest numbers. See BENCH-DESIGN.md §acceptance-rate for discussion.

Prerequisites:
  1. python3 scripts/prepare_qwen36_mtp_sidecar.py --model 27b
  2. python3 scripts/prepare_qwen36_mtp_sidecar.py --model 35b
  3. cargo build -p ax-engine-server --release

Usage:
  python3 scripts/bench_qwen36_lightning.py \\
      --models 27b,35b \\
      --mtp-depths 1,2,3 \\
      --output-dir benchmarks/results/qwen36-lightning/$(date +%F)

  # 27B only, quick smoke (1 warmup + 2 measured)
  python3 scripts/bench_qwen36_lightning.py \\
      --models 27b \\
      --repetitions 2 --warmup 1 \\
      --output-dir benchmarks/results/qwen36-lightning/$(date +%F)-smoke
"""
from __future__ import annotations

import argparse
import gc
import http.client
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AX_ENGINE_SERVER = REPO_ROOT / "target" / "release" / "ax-engine-server"
HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)

# ---------------------------------------------------------------------------
# Model registry — points at sidecar dirs created by prepare_qwen36_mtp_sidecar.py
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "27b": {
        "label": "Qwen3.6-27B-4bit+MTP",
        "slug": "models--ax-local--Qwen3.6-27B-MTP",
        "snapshot": "v1",
        "thinking": False,
        "description": "Qwen3.6-27B dense, 4-bit base + BF16 MTP sidecar (depth-3)",
    },
    "35b": {
        "label": "Qwen3.6-35B-A3B-4bit+MTP",
        "slug": "models--ax-local--Qwen3.6-35B-MTP",
        "snapshot": "v1",
        "thinking": False,
        "description": "Qwen3.6-35B-A3B MoE, 4-bit base + BF16 MTP sidecar (depth-1)",
    },
}

# ---------------------------------------------------------------------------
# Prompts — matches lightning-mlx benchmark categories
# ---------------------------------------------------------------------------

PROMPTS: list[tuple[str, str, str]] = [
    # (id, category, prompt)
    # HIGH repeat — MTP should shine
    (
        "getter_setter",
        "high_repeat",
        "Generate a Python class called UserProfile with 10 getter/setter method pairs "
        "for fields: name, age, email, phone, address, city, state, zip_code, country, timezone. "
        "Each getter returns self._field, each setter validates and assigns. Include type hints. "
        "Output ONLY Python code.",
    ),
    (
        "json_array",
        "high_repeat",
        "Generate a JSON array with 20 user objects. Each object has: "
        "id (int), name (string), email (string), role (admin/user/editor/viewer), "
        "active (boolean), created_at (ISO date string). Output ONLY valid JSON.",
    ),
    (
        "sql_inserts",
        "high_repeat",
        "Generate 20 SQL INSERT INTO products (id, name, price, category, stock, sku) VALUES "
        "statements with realistic product data. Output ONLY SQL.",
    ),
    # MEDIUM repeat
    (
        "markdown_table",
        "med_repeat",
        "Write a markdown table of 15 programming languages with columns: "
        "Name, Year Created, Paradigm, Typing, Primary Use, Speed Rating. "
        "Output ONLY the markdown table.",
    ),
    (
        "csv_data",
        "med_repeat",
        "Generate CSV with header and 20 rows. Columns: employee_id, name, department, "
        "salary, hire_date, manager, location. Use realistic data. Output ONLY CSV.",
    ),
    # LOW repeat — MTP should not help
    (
        "creative_story",
        "low_repeat",
        "Write a short creative story (about 200 words) about a robot discovering "
        "music for the first time. Be vivid and original.",
    ),
    (
        "explain_tcp",
        "low_repeat",
        "Explain how TCP/IP congestion control works in 200 words. "
        "Be technical but clear. Cover slow start, AIMD, and fast recovery.",
    ),
]

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
MAX_TOKENS = 512
DEFAULT_WARMUP = 2
DEFAULT_REPS = 5
DEFAULT_COOLDOWN = 10.0
SERVER_PORT_BASE = 58100


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_model_dir(key: str) -> Path | None:
    cfg = MODEL_REGISTRY[key]
    snap_dir = HF_CACHE / cfg["slug"] / "snapshots" / cfg["snapshot"]
    if snap_dir.exists() and (snap_dir / "mtplx_runtime.json").exists():
        return snap_dir
    return None


def _wait_for_server(port: int, timeout: int = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            if resp.status == 200:
                return True
        except OSError:
            pass
        time.sleep(1)
    return False


def _free_port(base: int) -> int:
    for port in range(base, base + 100):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        try:
            s.connect(("127.0.0.1", port))
        except OSError:
            return port
        finally:
            s.close()
    raise RuntimeError("No free port found")


def _chat_request(port: int, prompt: str, max_tokens: int, thinking: bool) -> tuple[float, float, int]:
    """Returns (ttft_s, decode_s, completion_tokens)."""
    messages = [{"role": "user", "content": prompt}]
    if not thinking:
        messages.insert(0, {"role": "system", "content": "/no_think"})
    body = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": SAMPLING["temperature"],
        "top_p": SAMPLING["top_p"],
        "stream": False,
    }).encode()

    t0 = time.perf_counter()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request("POST", "/v1/chat/completions",
                 body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    payload = json.loads(resp.read())
    t1 = time.perf_counter()

    usage = payload.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)

    # ax-engine SSE headers carry decode timing; fall back to wall time.
    timing = payload.get("timing", {})
    ttft = timing.get("ttft_s", 0.0)
    decode_s = timing.get("decode_s", t1 - t0)

    return ttft, decode_s, completion_tokens


def _ngram_telemetry(port: int) -> dict:
    """Fetch last request's ngram/MTP telemetry from ax-engine debug endpoint."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/debug/last_request_telemetry")
        resp = conn.getresponse()
        if resp.status == 200:
            return json.loads(resp.read())
    except OSError:
        pass
    return {}


def run_prompt(port: int, prompt: str, thinking: bool, max_tokens: int, reps: int, warmup: int, cooldown: float) -> dict:
    """Run one prompt N times, return stats dict."""
    all_runs = warmup + reps
    measured_tps: list[float] = []
    measured_ttft: list[float] = []
    mtp_drafted_total = 0
    mtp_accepted_total = 0
    accepted_by_depth: list[int] = []

    for rep in range(all_runs):
        ttft, decode_s, n_tok = _chat_request(port, prompt, max_tokens, thinking)
        telem = _ngram_telemetry(port)

        if rep >= warmup:
            tps = n_tok / decode_s if decode_s > 0 else 0.0
            measured_tps.append(tps)
            measured_ttft.append(ttft)
            mtp_drafted_total += telem.get("ax_mtp_draft_tokens", 0)
            mtp_accepted_total += telem.get("ax_mtp_accepted_tokens", 0)
            for d in range(8):
                val = telem.get(f"ax_ngram_accept_at_depth_{d}", 0)
                if len(accepted_by_depth) <= d:
                    accepted_by_depth.append(0)
                accepted_by_depth[d] += val

        if rep < all_runs - 1 and cooldown > 0:
            time.sleep(cooldown)

    accept_rate = mtp_accepted_total / mtp_drafted_total if mtp_drafted_total > 0 else None
    return {
        "tok_s_median": statistics.median(measured_tps) if measured_tps else 0.0,
        "tok_s_values": measured_tps,
        "ttft_median": statistics.median(measured_ttft) if measured_ttft else 0.0,
        "mtp_accept_rate": accept_rate,
        "accepted_by_depth": accepted_by_depth,
        "mtp_drafted": mtp_drafted_total,
        "mtp_accepted": mtp_accepted_total,
    }


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _start_server(model_dir: Path, port: int, mtp_depth: int | None, env_extra: dict) -> subprocess.Popen:
    env = {**os.environ, **env_extra}
    if mtp_depth is not None:
        env["AX_MLX_MTP_MAX_DEPTH"] = str(mtp_depth)
    else:
        env["AX_MLX_MTP_MAX_DEPTH"] = "0"  # disable MTP for baseline

    cmd = [
        str(AX_ENGINE_SERVER),
        "--model-dir", str(model_dir),
        "--port", str(port),
        "--max-seqs", "1",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        proc.kill()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_model(model_key: str, model_dir: Path, mtp_depths: list[int],
                reps: int, warmup: int, cooldown: float) -> dict:
    """Run full benchmark for one model: baseline + each MTP depth."""
    cfg = MODEL_REGISTRY[model_key]
    results: list[dict] = []
    port = _free_port(SERVER_PORT_BASE)

    variants: list[tuple[str, int | None]] = [("baseline_ar", None)]
    for d in mtp_depths:
        variants.append((f"mtp_d{d}", d))

    for variant_label, mtp_depth in variants:
        print(f"\n  [{cfg['label']}] {variant_label} (port={port})", flush=True)
        proc = _start_server(model_dir, port, mtp_depth, env_extra={})
        try:
            if not _wait_for_server(port, timeout=180):
                print(f"    ERROR: server did not start", flush=True)
                continue

            prompt_results = []
            for pid, category, prompt in PROMPTS:
                print(f"    {pid} [{category}] ...", end="", flush=True)
                stats = run_prompt(port, prompt, cfg["thinking"], MAX_TOKENS, reps, warmup, cooldown)
                stats.update({"prompt_id": pid, "category": category, "variant": variant_label, "mtp_depth": mtp_depth})
                prompt_results.append(stats)
                accept_str = f" accept={stats['mtp_accept_rate']:.1%}" if stats["mtp_accept_rate"] is not None else ""
                print(f" {stats['tok_s_median']:.1f} tok/s{accept_str}", flush=True)

            results.append({"variant": variant_label, "mtp_depth": mtp_depth, "prompts": prompt_results})
        finally:
            _stop_server(proc)
            time.sleep(3)

    return {"model_key": model_key, "model_label": cfg["label"], "model_dir": str(model_dir), "variants": results}


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def _category_median(prompt_results: list[dict], category: str) -> float | None:
    vals = [r["tok_s_median"] for r in prompt_results if r["category"] == category]
    return statistics.median(vals) if vals else None


def _overall_median(prompt_results: list[dict]) -> float | None:
    vals = [r["tok_s_median"] for r in prompt_results]
    return statistics.median(vals) if vals else None


def _mean_accept(prompt_results: list[dict]) -> float | None:
    rates = [r["mtp_accept_rate"] for r in prompt_results if r["mtp_accept_rate"] is not None]
    return sum(rates) / len(rates) if rates else None


def write_summary(model_results: list[dict], output_dir: Path) -> None:
    lines = []
    lines.append("# Qwen3.6 MTP Benchmark — lightning-mlx methodology")
    lines.append("")
    lines.append(f"Date: {date.today().isoformat()}  ")
    lines.append(f"Sampling: temperature={SAMPLING['temperature']}, top_p={SAMPLING['top_p']}, top_k={SAMPLING['top_k']} (rejection sampling)  ")
    lines.append(f"Max tokens: {MAX_TOKENS}  ")
    lines.append(f"Prefix cache: disabled  ")
    lines.append(f"Prompts: {len(PROMPTS)} ({sum(1 for _,c,_ in PROMPTS if c=='high_repeat')} high, {sum(1 for _,c,_ in PROMPTS if c=='med_repeat')} med, {sum(1 for _,c,_ in PROMPTS if c=='low_repeat')} low repeat)")
    lines.append("")
    lines.append("> **Note on comparison with lightning-mlx published numbers:**")
    lines.append("> lightning-mlx uses `--mtp-optimistic` (greedy argmax acceptance) for their headline")
    lines.append("> numbers. This benchmark uses correct probability-ratio rejection sampling at temp=0.6,")
    lines.append("> which is the real-world sampling mode. Expect lower tok/s than their published figures.")
    lines.append("> The speedup ratio (MTP/AR) is the meaningful cross-methodology comparison point.")
    lines.append("")

    for mr in model_results:
        lines.append(f"## {mr['model_label']}")
        lines.append("")
        lines.append(f"Model dir: `{mr['model_dir']}`  ")
        lines.append("")

        # Find baseline
        baseline = next((v for v in mr["variants"] if v["variant"] == "baseline_ar"), None)
        base_overall = _overall_median(baseline["prompts"]) if baseline else None

        lines.append("| Variant | Overall (tok/s) | High-repeat | Med-repeat | Low-repeat | MTP accept | Speedup vs AR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")

        for v in mr["variants"]:
            label = v["variant"]
            prompts = v["prompts"]
            overall = _overall_median(prompts)
            high = _category_median(prompts, "high_repeat")
            med = _category_median(prompts, "med_repeat")
            low = _category_median(prompts, "low_repeat")
            accept = _mean_accept(prompts)
            speedup = (overall / base_overall) if (overall and base_overall) else None

            def fmt(v): return f"{v:.1f}" if v is not None else "—"
            accept_str = f"{accept:.1%}" if accept is not None else "—"
            speedup_str = f"{speedup:.2f}×" if speedup is not None else "—"
            lines.append(f"| {label} | {fmt(overall)} | {fmt(high)} | {fmt(med)} | {fmt(low)} | {accept_str} | {speedup_str} |")

        lines.append("")

    lines.append("## Prompt details")
    lines.append("")
    for pid, cat, _ in PROMPTS:
        lines.append(f"- `{pid}` ({cat})")
    lines.append("")

    out = output_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {out}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="27b,35b",
                        help="Comma-separated model keys to benchmark (27b, 35b). Default: 27b,35b")
    parser.add_argument("--mtp-depths", default="1,2,3",
                        help="Comma-separated MTP depths to test. Default: 1,2,3")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPS,
                        help=f"Measured repetitions per prompt (default: {DEFAULT_REPS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Warmup repetitions (default: {DEFAULT_WARMUP})")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                        help=f"Cooldown seconds between reps (default: {DEFAULT_COOLDOWN})")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for JSON artifact and summary.md")
    parser.add_argument("--model-dir-27b", type=Path, default=None,
                        help="Override 27B model dir (default: auto from HF cache)")
    parser.add_argument("--model-dir-35b", type=Path, default=None,
                        help="Override 35B model dir (default: auto from HF cache)")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    mtp_depths = [int(d) for d in args.mtp_depths.split(",") if d.strip()]

    if not AX_ENGINE_SERVER.exists():
        sys.exit(f"ERROR: ax-engine-server not found at {AX_ENGINE_SERVER}\nRun: cargo build -p ax-engine-server --release")

    # Resolve model dirs
    model_dirs: dict[str, Path] = {}
    overrides = {"27b": args.model_dir_27b, "35b": args.model_dir_35b}
    for key in model_keys:
        d = overrides.get(key) or _find_model_dir(key)
        if d is None:
            sys.exit(
                f"ERROR: Model '{key}' not found in HF cache.\n"
                f"Run: python3 scripts/prepare_qwen36_mtp_sidecar.py --model {key}"
            )
        model_dirs[key] = d
        print(f"[{key}] model dir: {d}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for key in model_keys:
        print(f"\n{'='*70}", flush=True)
        print(f"Benchmarking {MODEL_REGISTRY[key]['label']}", flush=True)
        print(f"{'='*70}", flush=True)
        result = bench_model(
            key, model_dirs[key], mtp_depths,
            reps=args.repetitions, warmup=args.warmup, cooldown=args.cooldown,
        )
        all_results.append(result)

    # Save artifact
    artifact = {
        "schema": "ax.bench.qwen36_lightning.v1",
        "date": date.today().isoformat(),
        "sampling": SAMPLING,
        "max_tokens": MAX_TOKENS,
        "mtp_depths": mtp_depths,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "prompts": [{"id": p, "category": c} for p, c, _ in PROMPTS],
        "results": all_results,
    }
    artifact_path = args.output_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"Artifact: {artifact_path}", flush=True)

    write_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
