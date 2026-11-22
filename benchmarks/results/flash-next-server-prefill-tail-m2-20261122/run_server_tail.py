"""Compare server-side Flash Next decode-tail scheduling at identical HTTP arrivals."""

from argparse import ArgumentParser
import json
from math import isclose, isfinite
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import run_late
import run_late_deferral as client

mixed = client.mixed
layer = client.layer
require = client.require
ROOT = Path(__file__).resolve().parent
TRIGGER = client.TRIGGER_TOKEN
FLAG = "ax_scheduler_prefill_tail_deferred"
ARMS = (("physical-1", False), ("held-1", True), ("held-2", True), ("physical-2", False))
trace_stream = client.capture.trace_stream


def capture_tail(events):
    trace = trace_stream(events)
    held = []
    for kind, payload, _ in events:
        if kind != "step":
            continue
        step = payload["step"]
        decisions = (step.get("route") or {}).get("crossover_decisions") or {}
        if FLAG in decisions:
            require(decisions[FLAG] == 1 and step["scheduled_requests"] == 1
                    and client.bench.delta_token_count(kind, payload) == 1,
                    "The scheduler marker lacks a singleton decode")
            held.append(step["step_id"])
    return trace | {"held": held}


def verify(raw, enabled):
    short, long = raw["traces"]
    short_row, long_row = raw["observations"]
    require(raw["trigger_token"] == TRIGGER and len(raw["trigger"]) == 1,
            "The arrival trigger changed")
    trigger = raw["trigger"][0]
    after = short["decode_step_ids"][TRIGGER:]
    require(len(after) == 12 and len(long["decode_step_ids"]) == 96
            and short["decode_step_ids"][TRIGGER - 1] == trigger["step_id"]
            and [x["request_id"] for x in raw["traces"]] == [x["request_id"] for x in raw["observations"]]
            and (short["initial_cursor"], long["initial_cursor"]) == (0, 0)
            and (short["cursor"], long["cursor"]) == (512, 8192)
            and not short["partial"] and not long["partial"] and not long["held"],
            "The short, long or held stream geometry changed")
    offered, sent = trigger["seconds"], long_row["started_at_s"]
    short_end = short_row["started_at_s"] + short_row["e2e_latency_ms"] / 1000
    delay = (sent - offered) * 1000
    require(offered <= sent < short_end and 0 <= delay <= 10
            and isclose(offered, long_row["scheduled_at_s"], abs_tol=1e-9)
            and isclose(delay, long_row["queue_delay_ms"], abs_tol=0.001),
            "The long request was not dispatched immediately in both arms")
    marked = short["held"]
    if enabled:
        require(marked and len(marked) <= len(after), "The treatment produced no bounded scheduler hold")
        first = after.index(marked[0])
        require(marked == after[first:first+len(marked)], "Hold was repeated or extended beyond the decode tail")
    else:
        require(not marked, "The control had an active scheduler hold")
    overlap = any(row["step_id"] in after for row in long["prefill"])
    if overlap:
        run_late.verify_late_trace(raw, False)
    else:
        require(enabled and len(long["prefill"]) == 4
                and [r["cursor"] for r in long["prefill"]] == [2048, 4096, 6144, 8192]
                and all(r["width"] == 2048 and r["scheduled"] == 1 and r["split_end"] is None
                        and r["step_id"] > after[-1] for r in long["prefill"]),
                "An uncontended long prefill changed its logical cuts")
    require(not set(marked).intersection(r["step_id"] for r in long["prefill"]),
            "Long prefill ran during a held decode step")
    tail = short_row["stream_step_interval_ms"][TRIGGER - 1:]
    require(len(tail) == 12 and all(isfinite(v) and v >= 0 for v in tail), "Invalid short stream timing")
    burst_seconds = max(r["started_at_s"] + r["e2e_latency_ms"] / 1000 for r in raw["observations"]) - short_row["started_at_s"]
    return {"held_decode_steps": len(marked), "prefill_overlapped": overlap,
            "short_tail_max_gap_ms": max(tail), "long_trigger_ttft_ms": delay + long_row["ttft_ms"],
            "client_dispatch_delay_ms": delay, "burst_output_tokens_per_second": 192 / burst_seconds}


def burst(base, cases, expected, enabled, raw_path):
    with (patch.object(client.capture, "trace_stream", capture_tail),
          patch.object(client, "verify_deferral", lambda raw, _: verify(raw, enabled))):
        return client.burst(base, cases, expected, False, raw_path)


def observe_solo(args, cases, options):
    filename = args.out / "solo.json"
    env = {"AX_MLX_FLASH_NEXT_DIRECT_COALESCING": "0", "AX_MLX_FLASH_NEXT_PREFILL_SPLIT": "0",
           "AX_MLX_FLASH_NEXT_PREFILL_TAIL_DEFERRAL": "0"}
    with mixed.server(args, "solo", mixed.MODEL, args.pack, 1, env=env, options=options,
                      stream_experts="auto", guard=True) as (base, _):
        require("decision=resident" in (args.out / "solo.log").read_text(), "Resident experts required")
        rows = []
        begin = perf_counter()
        for case in cases:
            payload = {"model": mixed.MODEL, "input_tokens": case.input_tokens,
                       "max_output_tokens": case.max_output_tokens,
                       "sampling": {"temperature": 0, "ignore_eos": True}}
            started = perf_counter()
            events = list(client.bench.http_sse_events(f"{base}/v1/generate/stream", payload, 120, started=started))
            relative = started - begin
            rows.append(client.bench.observe_stream(
                events, prompt=case, scheduled_at_s=relative, started_at_s=relative,
                completed_at_s=perf_counter() - begin, record_token_ids=True,
            ))
        require(mixed.wait_drained(base) == dict.fromkeys(mixed.COUNTERS, 0),
                "The single-slot oracle did not drain")
        require(json.loads(mixed.native.request(base, "/health"))["status"] == "ok", "Oracle health declined")
    samples = [json.loads(line) for line in (args.out / "solo-samples.jsonl").read_text().splitlines()]
    require(samples and all(row["pressure"] == 1 for row in samples), "Oracle memory pressure exceeded normal")
    filename.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    require(all(row["ok"] and row["prompt_token_ids"] == case.input_tokens
                and len(row["output_token_ids"]) == case.max_output_tokens
                for row, case in zip(rows, cases, strict=True)),
            "The same-binary single-slot oracle did not complete both prompts")
    return {case.id: row["output_token_ids"] for row, case in zip(rows, cases, strict=True)}, {
        "data_sha256": mixed.digest(filename), "sample_count": len(samples),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("binary", "pack", "link-root", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--functional", action="store_true")
    args = parser.parse_args()
    _, cases, expected = layer.files()["ending-decoder"]
    require(cases[0].max_output_tokens == cases[1].max_output_tokens == 96, "Unexpected token budget")
    if args.check_inputs:
        print("Verified the late-arrival prompts and all 192 solo output IDs.")
        return
    build = json.loads((ROOT / "build.json").read_bytes())
    inventory = json.loads((ROOT / "mxfp4-inventory.json").read_bytes())
    require(mixed.digest(args.binary) == build["server_sha256"], "Binary differs from build receipt")
    manifest = mixed.native.validate_inventory(args.pack, inventory, args.link_root)
    require(manifest.get("model-manifest.json") ==
            "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef",
            "The delivered manifest changed")
    args.out.mkdir()
    record = {"scope": "M2 functional" if args.functional else "M5 ordered serving comparison",
              "qualification": False, "build": build, "pack_revision": inventory["revision"], "manifest": manifest,
              "inventory_sha256": mixed.digest(ROOT / "mxfp4-inventory.json"), "arms": {}}

    def save(status):
        record["status"] = status
        temporary = args.out / "receipt.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "receipt.json")
        print(status, flush=True)

    options = ("--disable-ngram-acceleration", "--total-blocks", "2048", "--max-batch-tokens", "2049",
               *mixed.RESERVE_OPTIONS)
    try:
        if args.functional:
            save("solo-starting")
            expected, record["solo"] = observe_solo(args, cases, options)
            save("solo-complete")
        for name, enabled in ARMS[:2] if args.functional else ARMS:
            save(f"{name}-starting")
            idle_before = None if args.functional else mixed.native.wait_for_quiet()["cpu_idle_percent"]
            env = {"AX_MLX_FLASH_NEXT_DIRECT_COALESCING": "0", "AX_MLX_FLASH_NEXT_PREFILL_SPLIT": "1",
                   "AX_MLX_FLASH_NEXT_PREFILL_TAIL_DEFERRAL": "1" if enabled else "0"}
            with mixed.server(args, name, mixed.MODEL, args.pack, 2, env=env, options=options,
                              stream_experts="auto", guard=True) as (base, _):
                require("decision=resident" in (args.out / f"{name}.log").read_text(), "Resident experts required")
                data = burst(base, cases, expected, enabled, args.out / f"{name}.partial.json")
                require(json.loads(mixed.native.request(base, "/health"))["status"] == "ok", "Health declined")
            path = args.out / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
            samples = [json.loads(line) for line in (args.out / f"{name}-samples.jsonl").read_text().splitlines()]
            require(samples and all(row["pressure"] == 1 for row in samples), "Memory pressure exceeded normal")
            record["arms"][name] = {
                "scheduler_tail_deferral": enabled,
                "cpu_idle_before_percent": idle_before,
                "cpu_idle_after_percent": None if args.functional else mixed.native.cpu_idle_percent(),
                "sample_count": len(samples), "data_sha256": mixed.digest(path),
            }
            save(f"{name}-complete")
        require(mixed.native.validate_inventory(args.pack, inventory, args.link_root) == manifest, "Pack changed")
        save("complete")
    except Exception:
        save("failed")
        raise


if __name__ == "__main__":
    main()
