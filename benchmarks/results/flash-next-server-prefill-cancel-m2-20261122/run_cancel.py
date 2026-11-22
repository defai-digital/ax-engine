"""Close a real long prefill stream and verify the surviving and next requests."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
from threading import Event
from time import perf_counter

from run_early import ROOT, capture_tail, client, inputs, mixed, require

TRIGGER = 48
LABEL = "cancel-prefill"


def events(base, case, started):
    payload = {
        "model": mixed.MODEL,
        "input_tokens": case.input_tokens,
        "max_output_tokens": case.max_output_tokens,
        "sampling": {"temperature": 0, "ignore_eos": True},
    }
    return client.bench.http_sse_events(f"{base}/v1/generate/stream", payload, 90, started=started)


def observe(frames, case, started, begin):
    return client.bench.observe_stream(
        frames, prompt=case, scheduled_at_s=started - begin, started_at_s=started - begin,
        completed_at_s=perf_counter() - begin, record_token_ids=True,
    )


def run_burst(base, short, long):
    ready = Event()
    trigger = []
    begin = perf_counter()

    def survive():
        started = perf_counter()
        frames = []
        count = 0
        for frame in events(base, short, started):
            frames.append(frame)
            kind, payload, elapsed = frame
            count += client.bench.delta_token_count(kind, payload)
            if count >= TRIGGER and not ready.is_set():
                require(count == TRIGGER, "Short stream skipped the offered token")
                trigger.append({"step_id": payload["step"]["step_id"], "seconds": started + elapsed - begin})
                ready.set()
        return {"result": observe(frames, short, started, begin), "trace": capture_tail(frames),
                "ended_seconds": perf_counter() - begin}

    def disconnect():
        require(ready.wait(20), "The surviving stream did not emit the trigger token")
        started = perf_counter()
        frames = []
        with closing(events(base, long, started)) as stream:
            for frame in stream:
                frames.append(frame)
                kind, payload, _ = frame
                if kind == "step" and payload["request"]["processed_prompt_tokens"] > 0:
                    break
            else:
                raise ValueError("Long stream closed before its first prefill step")
        return {"events": frames, "closed_seconds": perf_counter() - begin,
                "offered_seconds": trigger[0]["seconds"], "started_seconds": started - begin}

    with ThreadPoolExecutor(max_workers=2) as workers:
        short_work = workers.submit(survive)
        long_work = workers.submit(disconnect)
        cancelled, survivor = long_work.result(), short_work.result()
    drained_after_cancel = mixed.wait_drained(base)
    started = perf_counter()
    frames = list(events(base, long, started))
    replacement = {"result": observe(frames, long, started, begin), "trace": capture_tail(frames),
                   "started_seconds": started - begin}
    return {"trigger": trigger, "cancelled": cancelled, "survivor": survivor, "replacement": replacement,
            "drained_after_cancel": drained_after_cancel, "drained_after_replacement": mixed.wait_drained(base)}


def verify(raw, oracle):
    cancel, short, new = (raw[name] for name in ("cancelled", "survivor", "replacement"))
    for result, solo in zip((short["result"], new["result"]), oracle, strict=True):
        require(result["ok"] and all(result[key] == solo[key] for key in
                ("status", "prompt_id", "prompt_token_ids", "output_token_ids")),
                "The survivor or replacement changed exact output")
    trigger, = raw["trigger"]
    trace = short["trace"]
    require(len(trace["decode_step_ids"]) == 96 and trace["decode_step_ids"][TRIGGER - 1] == trigger["step_id"]
            and not trace["held"] and not new["trace"]["held"], "Invalid surviving decode schedule")
    frames = cancel["events"]
    status = [payload["status"] for kind, payload, _ in frames if kind == "__http_status__"]
    request, = [payload["request"] for kind, payload, _ in frames if kind == "request"]
    progressed, = [payload for kind, payload, _ in frames
                   if kind == "step" and payload["request"]["processed_prompt_tokens"]]
    step = progressed["step"]
    ids = {request["request_id"], trace["request_id"], new["trace"]["request_id"]}
    require(status == [200] and ids == {
        progressed["request"]["request_id"], short["result"]["request_id"], new["result"]["request_id"]
    } and len(ids) == 3 and request["processed_prompt_tokens"] == 0,
        "The three HTTP request identities do not bind to their traces")
    require(progressed["request"]["processed_prompt_tokens"] == 512
            and step["scheduled_requests"] == 2
            and step["step_id"] in trace["decode_step_ids"][TRIGGER:83]
            and not any(kind == "response" or client.bench.delta_token_count(kind, payload)
                        for kind, payload, _ in frames), "The disconnect did not occur in mixed physical prefill")
    require(trigger["seconds"] == cancel["offered_seconds"] <= cancel["started_seconds"]
            < cancel["closed_seconds"] < short["ended_seconds"] <= new["started_seconds"],
            "The disconnect was not observed before the sibling finished")
    drained = dict.fromkeys(mixed.COUNTERS, 0)
    require(raw["drained_after_cancel"] == raw["drained_after_replacement"] == drained,
            "A canceled or replacement request leaked lifecycle counters")
    return {"cancelled_prompt_cursor": 512, "matching_generated_ids": 192,
            "short_tokens_after_cancelled_step": 95 - trace["decode_step_ids"].index(step["step_id"])}


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("binary", "pack", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    cases, build, inventory, receipt, oracle, _ = inputs()
    if args.check_inputs:
        print("Verified both prompts and the sealed single-slot oracle.")
        return
    require(mixed.digest(args.binary) == build["server_sha256"], "Native binary digest mismatch")
    manifest = mixed.native.validate_inventory(args.pack, inventory, None)
    require(manifest == receipt["manifest"], "The delivered inventory changed")
    args.out.mkdir()
    options = ("--disable-ngram-acceleration", "--total-blocks", "2048", "--max-batch-tokens", "2049",
               *mixed.RESERVE_OPTIONS)
    env = {"AX_MLX_FLASH_NEXT_DIRECT_COALESCING": "0", "AX_MLX_FLASH_NEXT_PREFILL_SPLIT": "1",
           "AX_MLX_FLASH_NEXT_PREFILL_TAIL_DEFERRAL": "1"}
    with mixed.server(args, LABEL, mixed.MODEL, args.pack, 2, env=env, options=options,
                      stream_experts="auto", guard=True) as (base, _):
        require("decision=resident" in (args.out / f"{LABEL}.log").read_text(), "Resident experts required")
        raw = run_burst(base, *cases)
        path = args.out / "raw.json"
        path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
        comparison = verify(raw, oracle)
        require(json.loads(mixed.native.request(base, "/health"))["status"] == "ok", "Health declined")
    pressure = args.out / f"{LABEL}-samples.jsonl"
    samples = [json.loads(line) for line in pressure.read_text().splitlines()]
    require(samples and all(row["pressure"] == 1 for row in samples), "Memory pressure exceeded normal")
    require(mixed.native.validate_inventory(args.pack, inventory, None) == manifest, "Pack changed")
    result = {
        "qualification": False, "scope": "M2 Ultra 192 GB HTTP disconnect during prefill",
        "build": build, "pack_revision": inventory["revision"], "manifest": manifest,
        "inventory_sha256": receipt["inventory_sha256"], "oracle_sha256": receipt["solo"]["data_sha256"],
        "raw_sha256": mixed.digest(path), "pressure_sha256": mixed.digest(pressure),
        "pressure_samples": len(samples), "comparison": comparison,
    }
    (args.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Complete.", flush=True)


if __name__ == "__main__":
    main()
