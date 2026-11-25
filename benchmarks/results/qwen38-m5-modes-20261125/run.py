#!/usr/bin/env python3
"""Measure greedy HTTP decode and effective prompt prefill in separate probes."""

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import time

from tokenizers import Tokenizer


WARM_CASE = ("short-answer",)
MODES = {
    "dense": (
        ("ax-engine", "direct"), ("ax-engine", "dflash2"),
        ("splash", "dflash2"), ("omlx", "direct"), ("omlx", "dflash2"),
        ("mlx-lm", "direct"), ("mtplx", "direct"), ("tensorfold", "direct"),
        ("tensorfold", "dflash2"), ("ds4", "direct"),
    ),
    "flash": (
        ("ax-engine", "direct"), ("omlx", "direct"), ("mlx-lm", "direct"),
        ("mtplx", "direct"), ("tensorfold", "direct"), ("ds4", "direct"),
        ("splash", "dflash2"),
    ),
}


def stream(base, port, model, text, tokenizer, cap):
    body = json.dumps({
        "model": model, "prompt": text, "max_tokens": cap,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    })
    fragments = []
    first = last = None
    usage = finish = None
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=base.REQUEST_TIMEOUT)
    start = time.monotonic()
    try:
        conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        with conn.getresponse() as response:
            if response.status != 200:
                raise base.Failure("http_error", response.read(3000).decode(errors="replace"))
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise base.Failure("stream_error", "Expected an SSE response.")
            for line in response:
                if time.monotonic() - start > base.REQUEST_TIMEOUT:
                    raise base.Failure("request_timeout", "Stream exceeded its deadline.")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise base.Failure("stream_error", str(event["error"]))
                usage = (event.get("usage") or {}).get("completion_tokens", usage)
                for choice in event.get("choices", []):
                    fragment = choice.get("text")
                    if fragment:
                        last = time.monotonic()
                        first = last if first is None else first
                        fragments.append(fragment)
                    finish = choice.get("finish_reason") or finish
    finally:
        conn.close()
    wall_seconds = time.monotonic() - start
    if first is None or finish not in {"length", "stop"}:
        raise base.Failure("stream_error", "The response did not finish with visible text.")
    output = "".join(fragments)
    encode = tokenizer.encode
    return {
        "text_tokens": len(encode(output, add_special_tokens=False).ids),
        "first_chunk_tokens": len(encode(fragments[0], add_special_tokens=False).ids),
        "decode_seconds": round(last - first, 6),
        "ttft_seconds": round(first - start, 6),
        "wall_seconds": round(wall_seconds, 6),
        "reported_tokens": usage,
        "finish_reason": finish,
        "text_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def prefill_pair(base, port, model, tokenizer, repeat):
    def request(kind, count):
        marker = (("Canary", "Plum"), ("Heron", "Olive"))[repeat][kind]
        text = (
            f"Test record {marker}\n<|im_start|>user\n"
            + "A garden has a low stone wall and a narrow wooden gate. " * count
            + "\nReply with the word Yes.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        size = len(tokenizer.encode(text, add_special_tokens=False).ids)
        row = stream(base, port, model, text, tokenizer, 1)
        return {"input_tokens": size, "first_text_seconds": row["ttft_seconds"]}

    if repeat == 0:
        short, long = request(0, 8), request(1, 113)
    else:
        long, short = request(1, 113), request(0, 8)
    elapsed = long["first_text_seconds"] - short["first_text_seconds"]
    if elapsed <= 0:
        raise base.Failure("prefill_measurement_failed", "Long prompt did not raise first-text time.")
    return {"short": short, "long": long}


def arm(base, stage, model_key, engine, mode, prompts, draft):
    base_command, base_complete = base.command, base.complete
    pack = Path((stage / f"{model_key}.path").read_text().strip())
    arm_stage = stage / f"{model_key}-{engine}-{mode}"
    arm_stage.mkdir()
    for name in ("ax-engine-server", "ds4-server", "tensorfold", "splash-drafts"):
        source = stage / name
        if source.exists():
            (arm_stage / name).symlink_to(source)
    tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
    prefill = []

    def command(stage, lane, pack, port, scratch):
        args = base_command(stage, lane, pack, port, scratch)
        if (engine, mode) == ("ax-engine", "dflash2"):
            args.remove("--disable-ngram-acceleration")
            args[args.index("--mlx-mtp-policy") + 1] = "auto"
            args = [
                Path("/usr/bin/env"), f"DYLD_LIBRARY_PATH={os.environ['AX_BENCH_MLX_LIB']}",
                "AX_MLX_DFLASH2=1", f"AX_MLX_DFLASH2_DIR={draft}",
                "AX_MLX_DFLASH2_ROUND_TRACE=1", *args,
            ]
        if (engine, mode) == ("omlx", "dflash2"):
            settings = {
                "version": 2,
                "models": {"peer-27b": {"dflash_enabled": True, "dflash_draft_model": str(draft)}},
            }
            omlx_home = scratch / "omlx"
            omlx_home.mkdir()
            (omlx_home / "model_settings.json").write_text(json.dumps(settings))
        if (engine, mode) == ("tensorfold", "dflash2"):
            args.remove("--no-drafts")
            args.extend(("--drafter", str(draft)))
        if (engine, model_key) == ("splash", "dense"):
            args[args.index("--draft-model") + 1] = draft
        return args

    def complete(port, served_model, prompt, tokenizer):
        row = stream(base, port, served_model, prompt["input_text"], tokenizer, base.CAP)
        if prompt["id"] == base.PROMPT_IDS[-1]:
            prefill.append(prefill_pair(base, port, served_model, tokenizer, len(prefill)))
        return {"prompt": prompt["id"], **row}

    base.command, base.complete = command, complete
    try:
        row = base.trial(arm_stage, engine, pack, tokenizer, prompts)
    finally:
        base.command, base.complete = base_command, base_complete
    server_log = arm_stage / "logs" / f"{model_key}-{engine}" / "server.log"
    log = server_log.read_text(errors="replace") if server_log.is_file() else ""
    if row["status"] == "passed" and mode == "dflash2":
        if engine == "ax-engine":
            route = sum(line.startswith("dflash2_round ") for line in log.splitlines())
        elif engine == "omlx":
            route = len(re.findall(r"DFlash drafter loaded:.*kind=dflash2", log, re.IGNORECASE))
        else:
            route = len(re.findall(r"Decode cycles verify|DFlash2 draft", log, re.IGNORECASE))
        row["dflash2_route_events"] = route
        if route == 0:
            row.update(status="route_unverified", detail="No DFlash2 activation was observed.")
    row.update(mode=mode, prefill=prefill, decode_tok_s=0, prefill_tok_s=0)
    if row["status"] == "passed":
        emitted = [r["text_tokens"] - r["first_chunk_tokens"] for r in row["rows"]]
        decode_seconds = sum(r["decode_seconds"] for r in row["rows"])
        if not all(r["text_tokens"] == base.CAP for r in row["rows"]) or decode_seconds <= 0:
            row.update(status="output_budget_mismatch", detail="The lane did not emit the fixed text budget.")
        else:
            rates = [
                (pair["long"]["input_tokens"] - pair["short"]["input_tokens"])
                / (pair["long"]["first_text_seconds"] - pair["short"]["first_text_seconds"])
                for pair in prefill
            ]
            row.update(decode_tok_s=round(sum(emitted) / decode_seconds, 4),
                       prefill_tok_s=round(statistics.median(rates), 4))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--baseline-runner", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--model", choices=MODES, required=True)
    parser.add_argument("--arm", help="Select one engine and mode as engine:mode")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("base", args.baseline_runner)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    stage = args.stage.resolve()
    corpus = stage / "qwen38_dflash2_paired.jsonl"
    prompts = {row["id"]: row for row in map(json.loads, corpus.read_text().splitlines())}
    arms = [pair for pair in MODES[args.model] if not args.arm or ":".join(pair) == args.arm]
    if not arms:
        parser.error("No arm matches the requested model and engine")
    for engine, mode in arms:
        row = arm(base, stage, args.model, engine, mode, prompts, args.draft.resolve())
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
