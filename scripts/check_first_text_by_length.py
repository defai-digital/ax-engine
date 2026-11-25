#!/usr/bin/env python3
"""Measure cold first-text latency by prompt length, and gate it against a baseline.

`measure` sends streaming requests with one output token to an OpenAI-compatible
server that is already running, for a list of prompt lengths. Every prompt starts
with a marker that is unique to the run, the length and the repeat, so no request
can reuse another one's prefix: each latency is a cold prefill plus one token.
The result is one `ax.first_text_by_length.v1` JSON document.

`compare` reads two such documents and fails when the candidate is slower than
the baseline at any length by more than both a relative and an absolute margin.
A length present in the baseline and missing from the candidate also fails.

This is a measurement tool for a host with a loaded model. It starts no server,
downloads nothing and is not part of CI; its parsing and comparison logic are
unit tested without a model.

    python scripts/check_first_text_by_length.py measure \\
        --base-url http://127.0.0.1:8080 --model my-model \\
        --sentences 1,8,16,30,60,113,170,226,340 --output candidate.json
    python scripts/check_first_text_by_length.py compare \\
        --baseline baseline.json --candidate candidate.json

Prompt lengths are given in sentences. The default sentence is 13 tokens with
the Qwen tokenizers; the document records the server's own prompt-token count
for each length when the server reports usage.
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import time
import urllib.parse
from pathlib import Path

SCHEMA = "ax.first_text_by_length.v1"
SENTENCE = "A garden has a low stone wall and a narrow wooden gate. "
QUESTION = "Reply with the word Yes."
# ChatML wrapping for the raw completions endpoint, with thinking closed so
# the first text chunk is the answer.
CHATML_PREFIX = "<|im_start|>user\n"
CHATML_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
WARMUP_SENTENCES = 8


class GateError(Exception):
    """A document or a server reply this tool cannot use."""


def build_prompt(marker: str, sentences: int) -> str:
    """The user text of one probe: a unique first line, then the filler."""
    if sentences < 1:
        raise GateError(f"prompt length must be at least one sentence, got {sentences}")
    return f"Test record {marker}\n" + SENTENCE * sentences + "\n" + QUESTION


def request_body(api: str, model: str, marker: str, sentences: int) -> tuple[str, dict]:
    """Endpoint path and JSON body of one probe."""
    text = build_prompt(marker, sentences)
    common = {
        "model": model,
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if api == "completions":
        first_line, rest = text.split("\n", 1)
        # The marker stays the very first tokens of the prompt.
        prompt = f"{first_line}\n{CHATML_PREFIX}{rest}{CHATML_SUFFIX}"
        return "/v1/completions", {**common, "prompt": prompt}
    if api == "chat":
        return "/v1/chat/completions", {**common, "messages": [{"role": "user", "content": text}]}
    raise GateError(f"unknown api {api!r}")


def chunk_text(event: dict) -> str:
    """Visible text carried by one stream event of either endpoint."""
    parts = []
    for choice in event.get("choices") or []:
        if isinstance(choice.get("text"), str):
            parts.append(choice["text"])
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            parts.append(delta["content"])
    return "".join(parts)


def read_stream(lines, clock, started: float) -> tuple[float, int | None]:
    """First-text seconds and the reported prompt-token count of one SSE reply."""
    first = None
    prompt_tokens = None
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            raise GateError(f"stream event is not JSON: {payload[:80]!r}") from error
        if first is None and chunk_text(event):
            first = clock() - started
        usage = event.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
            prompt_tokens = usage["prompt_tokens"]
    if first is None:
        raise GateError("the reply carried no text")
    return first, prompt_tokens


def probe(base_url: str, api: str, model: str, marker: str, sentences: int, timeout: float):
    """One request: first-text seconds and prompt tokens (or None)."""
    url = urllib.parse.urlsplit(base_url)
    if url.scheme != "http" or not url.hostname:
        raise GateError(f"--base-url must be http://host:port, got {base_url!r}")
    path, body = request_body(api, model, marker, sentences)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    try:
        started = time.monotonic()
        connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(300).decode("utf-8", "replace")
            raise GateError(f"{path} returned HTTP {response.status}: {detail}")
        return read_stream(response, time.monotonic, started)
    finally:
        connection.close()


def measure(args) -> dict:
    lengths = parse_lengths(args.sentences)
    for index in range(args.warmups):
        probe(args.base_url, args.api, args.model, f"{args.run_id}-warm{index}", WARMUP_SENTENCES, args.timeout)
    rows = []
    for sentences in lengths:
        seconds, tokens = [], None
        for repeat in range(args.repeats):
            marker = f"{args.run_id}-n{sentences}-r{repeat}"
            first, reported = probe(args.base_url, args.api, args.model, marker, sentences, args.timeout)
            seconds.append(round(first, 6))
            tokens = reported if reported is not None else tokens
        rows.append(
            {
                "sentences": sentences,
                "input_tokens": tokens,
                "first_text_seconds": seconds,
                "median_seconds": round(statistics.median(seconds), 6),
            }
        )
        print(
            f"{sentences:>6} sentences  {tokens if tokens is not None else '?':>7} tokens  "
            f"{rows[-1]['median_seconds']:.3f} s",
            file=sys.stderr,
        )
    return {
        "schema": SCHEMA,
        "label": args.label,
        "model": args.model,
        "api": args.api,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "run_id": args.run_id,
        "rows": rows,
    }


def parse_lengths(text: str) -> list[int]:
    try:
        lengths = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise GateError(f"--sentences must be comma-separated integers, got {text!r}") from error
    if not lengths or any(value < 1 for value in lengths) or len(set(lengths)) != len(lengths):
        raise GateError(f"--sentences needs distinct positive integers, got {text!r}")
    return lengths


def load_document(path: Path) -> dict:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"{path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise GateError(f"{path}: not an {SCHEMA} document")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise GateError(f"{path}: no rows")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sentences"), int):
            raise GateError(f"{path}: a row has no sentence count")
        median = row.get("median_seconds")
        if not isinstance(median, (int, float)) or isinstance(median, bool) or median <= 0:
            raise GateError(f"{path}: row {row.get('sentences')} has no positive median")
    return document


def compare(baseline: dict, candidate: dict, relative: float, absolute_seconds: float) -> list[dict]:
    """One verdict per baseline length. `regressed` needs both margins exceeded."""
    if baseline.get("api") != candidate.get("api"):
        raise GateError(
            f"the documents used different endpoints: {baseline.get('api')!r} and {candidate.get('api')!r}"
        )
    measured = {row["sentences"]: row for row in candidate["rows"]}
    verdicts = []
    for row in baseline["rows"]:
        sentences, base = row["sentences"], row["median_seconds"]
        other = measured.get(sentences)
        if other is None:
            verdicts.append({"sentences": sentences, "baseline_seconds": base, "status": "missing"})
            continue
        now = other["median_seconds"]
        slower = now - base
        regressed = slower > absolute_seconds and now > base * (1.0 + relative)
        verdicts.append(
            {
                "sentences": sentences,
                "input_tokens": other.get("input_tokens") or row.get("input_tokens"),
                "baseline_seconds": base,
                "candidate_seconds": now,
                "change": round(now / base - 1.0, 4),
                "status": "regressed" if regressed else "ok",
            }
        )
    return verdicts


def render(verdicts: list[dict]) -> str:
    lines = [f"{'sentences':>9} {'tokens':>7} {'baseline':>9} {'candidate':>9} {'change':>8}  status"]
    for verdict in verdicts:
        tokens = verdict.get("input_tokens")
        if verdict["status"] == "missing":
            lines.append(
                f"{verdict['sentences']:>9} {'?':>7} {verdict['baseline_seconds']:>9.3f} {'-':>9} {'-':>8}  missing"
            )
            continue
        lines.append(
            f"{verdict['sentences']:>9} {tokens if tokens is not None else '?':>7} "
            f"{verdict['baseline_seconds']:>9.3f} {verdict['candidate_seconds']:>9.3f} "
            f"{verdict['change']:>+8.1%}  {verdict['status']}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure", help="measure a running server")
    run.add_argument("--base-url", required=True, help="http://host:port of the server")
    run.add_argument("--model", required=True, help="model id the server serves")
    run.add_argument("--sentences", default="1,8,16,30,60,113", help="prompt lengths in sentences")
    run.add_argument("--api", choices=("completions", "chat"), default="completions")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmups", type=int, default=3)
    run.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    run.add_argument("--label", default="", help="free text recorded in the document")
    run.add_argument("--run-id", default=None, help="prefix of every prompt marker; default: current time")
    run.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("compare", help="gate a candidate against a baseline")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--relative", type=float, default=0.10, help="allowed slowdown as a fraction")
    check.add_argument("--absolute-ms", type=float, default=20.0, help="allowed slowdown in milliseconds")
    check.add_argument("--json", action="store_true", help="print the verdicts as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "measure":
            if args.repeats < 1 or args.warmups < 0:
                raise GateError("--repeats must be at least 1 and --warmups at least 0")
            if args.run_id is None:
                args.run_id = f"t{time.time_ns()}"
            document = measure(args)
            args.output.write_text(json.dumps(document, indent=2) + "\n")
            return 0
        if args.relative < 0 or args.absolute_ms < 0:
            raise GateError("--relative and --absolute-ms must not be negative")
        verdicts = compare(
            load_document(args.baseline),
            load_document(args.candidate),
            args.relative,
            args.absolute_ms / 1000.0,
        )
    except GateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(verdicts, indent=2) if args.json else render(verdicts))
    failed = [verdict for verdict in verdicts if verdict["status"] != "ok"]
    if failed:
        print(f"FAIL: {len(failed)} of {len(verdicts)} lengths regressed or are missing", file=sys.stderr)
        return 1
    print(f"OK: {len(verdicts)} lengths within {args.relative:.0%} or {args.absolute_ms:g} ms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
