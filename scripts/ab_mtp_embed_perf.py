#!/usr/bin/env python3
"""A/B microbench: MTP adaptive gate and embed max_len buckets.

Not a publication matrix. Reports wall-clock decode tok/s (chat) and embed
batch throughput with/without experimental flags. Uses ax-engine-server.

Example:
  python3 scripts/ab_mtp_embed_perf.py \\
    --server-bin target/release/ax-engine-server \\
    --mtp-model /path/to/Qwen3.6-27B-6bit-MTP \\
    --embed-model /path/to/Qwen3-Embedding-0.6B-8bit \\
    --output-dir /tmp/ab-results
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from e


def wait_ready(base: str, timeout: float = 900.0) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            http_json("GET", f"{base}/health")
            return
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(1.0)
    raise RuntimeError(f"server not ready: {last}")


def start_server(
    *,
    server_bin: Path,
    model_dir: Path,
    model_id: str,
    port: int,
    env_extra: dict[str, str],
    log_path: Path,
    extra_args: list[str],
) -> subprocess.Popen:
    env = os.environ.copy()
    # Clear experimental flags unless explicitly set in env_extra
    for k in (
        "AX_MLX_MTP_ADAPTIVE_GATE",
        "AX_MLX_MTP_ADAPTIVE_GATE_RESIDUAL",
        "AX_EMBED_MAX_LEN_BUCKETS",
        "AX_EMBED_LENGTH_SPLIT",
    ):
        env.pop(k, None)
    env.update(env_extra)
    cmd = [
        str(server_bin),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        model_id,
        "--support-tier",
        "mlx-preview",
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        *extra_args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    proc._logf = logf  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    logf = getattr(proc, "_logf", None)
    if logf:
        logf.close()


def chat_once(
    base: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    prompt: str,
) -> dict:
    t0 = time.perf_counter()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    resp = http_json("POST", f"{base}/v1/chat/completions", body, timeout=600.0)
    wall = time.perf_counter() - t0
    usage = resp.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    prompt_tok = int(usage.get("prompt_tokens") or 0)
    text = ""
    try:
        text = resp["choices"][0]["message"]["content"] or ""
    except Exception:  # noqa: BLE001
        pass
    return {
        "wall_s": wall,
        "completion_tokens": completion,
        "prompt_tokens": prompt_tok,
        "decode_tok_s": (completion / wall) if wall > 0 and completion > 0 else None,
        "text_preview": text[:120].replace("\n", " "),
    }


def run_mtp_ab(
    *,
    server_bin: Path,
    model_dir: Path,
    port: int,
    out: Path,
    max_tokens: int,
    warmup: int,
    reps: int,
    temperature: float,
) -> dict:
    prompts = [
        # Easy / repetitive-ish
        "Write a short flappy bird game loop description in 5 bullet points.",
        # Harder / code-like
        "Implement a pure-Python function that parses nested JSON with schema validation and unit tests. Keep under 80 lines.",
        "Explain how to debug a race condition in a multi-threaded queue with lock ordering; give a concrete example.",
    ]
    variants = [
        ("adaptive_off", {}, ["--mlx-mtp-disable-ngram-stacking"]),
        (
            "adaptive_on",
            {"AX_MLX_MTP_ADAPTIVE_GATE": "1"},
            ["--mlx-mtp-disable-ngram-stacking"],
        ),
    ]
    results: dict = {"variants": {}, "config": {
        "max_tokens": max_tokens,
        "warmup": warmup,
        "reps": reps,
        "temperature": temperature,
        "model": str(model_dir),
    }}
    for name, envx, extra in variants:
        log = out / f"server-mtp-{name}.log"
        print(f"\n=== MTP {name} ===", flush=True)
        proc = start_server(
            server_bin=server_bin,
            model_dir=model_dir,
            model_id="ab-mtp",
            port=port,
            env_extra=envx,
            log_path=log,
            extra_args=extra,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            wait_ready(base, timeout=900)
            # Warmup
            for i in range(warmup):
                chat_once(
                    base,
                    model="ab-mtp",
                    max_tokens=min(64, max_tokens),
                    temperature=temperature,
                    prompt=prompts[i % len(prompts)],
                )
            rows = []
            for r in range(reps):
                p = prompts[r % len(prompts)]
                row = chat_once(
                    base,
                    model="ab-mtp",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    prompt=p,
                )
                row["prompt_id"] = r % len(prompts)
                rows.append(row)
                print(
                    f"  rep{r}: tok/s={row['decode_tok_s'] and round(row['decode_tok_s'], 2)} "
                    f"tokens={row['completion_tokens']} wall={row['wall_s']:.2f}s",
                    flush=True,
                )
            speeds = [x["decode_tok_s"] for x in rows if x["decode_tok_s"]]
            results["variants"][name] = {
                "env": envx,
                "rows": rows,
                "decode_tok_s_median": statistics.median(speeds) if speeds else None,
                "decode_tok_s_mean": statistics.mean(speeds) if speeds else None,
                "decode_tok_s_min": min(speeds) if speeds else None,
                "decode_tok_s_max": max(speeds) if speeds else None,
            }
        finally:
            stop_server(proc)
            time.sleep(2)
    # Ratio
    a = results["variants"].get("adaptive_off", {}).get("decode_tok_s_median")
    b = results["variants"].get("adaptive_on", {}).get("decode_tok_s_median")
    results["adaptive_on_over_off_median"] = (b / a) if a and b else None
    return results


def load_tokenizer(model_dir: Path):
    from tokenizers import Tokenizer

    tok_path = model_dir / "tokenizer.json"
    return Tokenizer.from_file(str(tok_path))


def encode_batch(tok, texts: list[str], *, add_eos: bool, eos_id: int | None) -> list[list[int]]:
    rows = []
    for t in texts:
        ids = tok.encode(t).ids
        if add_eos and eos_id is not None and (not ids or ids[-1] != eos_id):
            ids = ids + [eos_id]
        rows.append(ids)
    return rows


def embed_once(
    base: str, model: str, token_rows: list[list[int]], pooling: str
) -> dict:
    t0 = time.perf_counter()
    body = {
        "model": model,
        "input": token_rows,
        "encoding_format": "float",
        "pooling": pooling,
    }
    resp = http_json("POST", f"{base}/v1/embeddings", body, timeout=300.0)
    wall = time.perf_counter() - t0
    data = resp.get("data") or []
    n_tok = sum(len(r) for r in token_rows)
    return {
        "wall_s": wall,
        "n_rows": len(token_rows),
        "n_tokens": n_tok,
        "max_len": max((len(r) for r in token_rows), default=0),
        "tok_s": (n_tok / wall) if wall > 0 else None,
        "items_s": (len(token_rows) / wall) if wall > 0 else None,
        "dim": len(data[0]["embedding"]) if data else 0,
    }


def run_embed_ab(
    *,
    server_bin: Path,
    model_dir: Path,
    port: int,
    out: Path,
    pooling: str,
    add_eos: bool,
    warmup: int,
    reps: int,
) -> dict:
    # Mixed lengths to stress pad + compile keys
    texts_short = [
        "hello",
        "query: what is rust?",
        "embedding test sentence number three",
    ]
    texts_mid = [
        " ".join(["mid length document about retrieval systems"] * 8),
        " ".join(["another paragraph with slightly different tokens"] * 10),
    ]
    texts_long = [
        " ".join(["long document chunk used for ingest style workload"] * 20),
    ]
    # Batches with heterogeneous lengths
    batches = [
        texts_short,  # short only
        texts_short + texts_mid,  # mixed short/mid
        texts_mid + texts_long,  # mixed mid/long
        texts_short + texts_mid + texts_long,  # full mix
    ]

    variants = [
        # Pre-sprint: no buckets, no length split
        (
            "legacy_off",
            {
                "AX_EMBED_MAX_LEN_BUCKETS": "off",
                "AX_EMBED_LENGTH_SPLIT": "off",
            },
        ),
        # New defaults: calibrated buckets + length-affinity split
        ("default_on", {}),
    ]
    results: dict = {
        "variants": {},
        "config": {
            "pooling": pooling,
            "add_eos": add_eos,
            "warmup": warmup,
            "reps": reps,
            "model": str(model_dir),
            "n_batch_shapes": len(batches),
        },
    }

    tok = load_tokenizer(model_dir)
    # Best-effort EOS for Qwen last-pool; Gemma mean ignores last-token EOS issues.
    eos_id = None
    cfg_path = model_dir / "tokenizer_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
            eos = cfg.get("eos_token")
            if isinstance(eos, dict):
                eos = eos.get("content")
            if isinstance(eos, str) and eos:
                tid = tok.token_to_id(eos)
                if tid is not None:
                    eos_id = int(tid)
        except Exception:  # noqa: BLE001
            pass
    if eos_id is None:
        for cand in ("<|im_end|>", "<|endoftext|>", "</s>", "<eos>"):
            tid = tok.token_to_id(cand)
            if tid is not None:
                eos_id = int(tid)
                break

    for name, envx in variants:
        log = out / f"server-embed-{name}.log"
        print(f"\n=== Embed {name} ===", flush=True)
        proc = start_server(
            server_bin=server_bin,
            model_dir=model_dir,
            model_id="ab-embed",
            port=port,
            env_extra=envx,
            log_path=log,
            extra_args=["--disable-ngram-acceleration"],
        )
        base = f"http://127.0.0.1:{port}"
        try:
            wait_ready(base, timeout=600)
            # Warm compile shapes
            for i in range(warmup):
                rows = encode_batch(
                    tok, batches[i % len(batches)], add_eos=add_eos, eos_id=eos_id
                )
                embed_once(base, "ab-embed", rows, pooling)
            rows_out = []
            for r in range(reps):
                batch = batches[r % len(batches)]
                token_rows = encode_batch(tok, batch, add_eos=add_eos, eos_id=eos_id)
                row = embed_once(base, "ab-embed", token_rows, pooling)
                row["batch_id"] = r % len(batches)
                rows_out.append(row)
                print(
                    f"  rep{r}: tok/s={row['tok_s'] and round(row['tok_s'], 1)} "
                    f"items/s={row['items_s'] and round(row['items_s'], 2)} "
                    f"max_len={row['max_len']} wall={row['wall_s']*1000:.1f}ms",
                    flush=True,
                )
            speeds = [x["tok_s"] for x in rows_out if x["tok_s"]]
            items = [x["items_s"] for x in rows_out if x["items_s"]]
            walls = [x["wall_s"] for x in rows_out]
            results["variants"][name] = {
                "env": envx,
                "rows": rows_out,
                "tok_s_median": statistics.median(speeds) if speeds else None,
                "tok_s_mean": statistics.mean(speeds) if speeds else None,
                "items_s_median": statistics.median(items) if items else None,
                "wall_ms_median": statistics.median(walls) * 1000 if walls else None,
            }
        finally:
            stop_server(proc)
            time.sleep(1.5)

    a = results["variants"].get("legacy_off", {}).get("tok_s_median")
    b = results["variants"].get("default_on", {}).get("tok_s_median")
    results["default_on_over_legacy_off_tok_s_median"] = (b / a) if a and b else None
    # Back-compat keys for older report readers
    results["buckets_on_over_off_tok_s_median"] = results[
        "default_on_over_legacy_off_tok_s_median"
    ]
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-bin", type=Path, required=True)
    ap.add_argument("--mtp-model", type=Path, default=None)
    ap.add_argument("--embed-model", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mtp-port", type=int, default=18881)
    ap.add_argument("--embed-port", type=int, default=18882)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--pooling", type=str, default="auto")
    ap.add_argument("--skip-mtp", action="store_true")
    ap.add_argument("--skip-embed", action="store_true")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    if not args.skip_mtp:
        if not args.mtp_model or not args.mtp_model.is_dir():
            print("ERROR: --mtp-model required unless --skip-mtp", file=sys.stderr)
            return 2
        report["mtp"] = run_mtp_ab(
            server_bin=args.server_bin,
            model_dir=args.mtp_model,
            port=args.mtp_port,
            out=out,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            reps=args.reps,
            temperature=args.temperature,
        )

    if not args.skip_embed:
        if not args.embed_model or not args.embed_model.is_dir():
            print("ERROR: --embed-model required unless --skip-embed", file=sys.stderr)
            return 2
        pooling = args.pooling
        if pooling == "auto":
            mid = args.embed_model.name.lower()
            pooling = "mean" if "embeddinggemma" in mid or "gemma-embedding" in mid else "last"
        report["embed"] = run_embed_ab(
            server_bin=args.server_bin,
            model_dir=args.embed_model,
            port=args.embed_port,
            out=out,
            pooling=pooling,
            add_eos=pooling == "last",
            warmup=args.warmup,
            reps=max(args.reps, 6),
        )

    (out / "ab_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    # Markdown report
    lines = ["# A/B microbench: MTP adaptive gate + embed max_len buckets", ""]
    lines.append("Not a publication matrix. Wall-clock HTTP client measurement.")
    lines.append("")
    if "mtp" in report:
        m = report["mtp"]
        lines.append("## MTP adaptive gate")
        lines.append("")
        lines.append(f"- model: `{m['config']['model']}`")
        lines.append(
            f"- max_tokens={m['config']['max_tokens']} warmup={m['config']['warmup']} "
            f"reps={m['config']['reps']} temperature={m['config']['temperature']}"
        )
        lines.append("")
        lines.append("| variant | median tok/s | mean | min | max |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, v in m["variants"].items():
            lines.append(
                f"| {name} | {v.get('decode_tok_s_median')} | {v.get('decode_tok_s_mean')} | "
                f"{v.get('decode_tok_s_min')} | {v.get('decode_tok_s_max')} |"
            )
        ratio = m.get("adaptive_on_over_off_median")
        lines.append("")
        lines.append(
            f"**adaptive_on / adaptive_off median decode tok/s:** "
            f"{ratio:.4f}" if ratio else "**ratio:** n/a"
        )
        lines.append("")
    if "embed" in report:
        e = report["embed"]
        lines.append("## Embed max_len buckets")
        lines.append("")
        lines.append(f"- model: `{e['config']['model']}`")
        lines.append(
            f"- pooling={e['config']['pooling']} add_eos={e['config']['add_eos']} "
            f"warmup={e['config']['warmup']} reps={e['config']['reps']}"
        )
        lines.append("")
        lines.append("| variant | median tok/s | mean tok/s | median items/s | median wall ms |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, v in e["variants"].items():
            lines.append(
                f"| {name} | {v.get('tok_s_median')} | {v.get('tok_s_mean')} | "
                f"{v.get('items_s_median')} | {v.get('wall_ms_median')} |"
            )
        ratio = e.get("default_on_over_legacy_off_tok_s_median") or e.get(
            "buckets_on_over_off_tok_s_median"
        )
        lines.append("")
        lines.append(
            f"**default_on / legacy_off median tok/s:** "
            f"{ratio:.4f}" if ratio else "**ratio:** n/a"
        )
        lines.append("")
    (out / "ab_summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out / 'ab_summary.json'} and {out / 'ab_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
