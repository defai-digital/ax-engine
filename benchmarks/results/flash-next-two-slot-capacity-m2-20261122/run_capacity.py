"""Check Flash Next two-slot rejection and early slot reuse over native HTTP."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event
from time import perf_counter, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from run_server_tail import ROOT, capture_tail, client, layer, mixed, require

SLOT_COUNTS = ("ax_engine_jobs_in_flight", "ax_engine_generation_active_streams")
TRIGGER_TOKEN = 48


def payload(case):
    return {"model": mixed.MODEL, "input_tokens": case.input_tokens,
            "max_output_tokens": case.max_output_tokens,
            "sampling": {"temperature": 0, "ignore_eos": True}}


def observe(base, case, begin, each_frame=None):
    started = perf_counter()
    entries = []
    for kind, event, elapsed in client.bench.http_sse_events(
            f"{base}/v1/generate/stream", payload(case), 120, started=started):
        entries.append((kind, event, elapsed))
        if each_frame is not None:
            each_frame(kind, event, started + elapsed - begin)
    ended = perf_counter()
    result = client.bench.observe_stream(
        entries, prompt=case, scheduled_at_s=started - begin, started_at_s=started - begin,
        completed_at_s=ended - begin, record_token_ids=True,
    )
    return {"request": result, "trace": capture_tail(entries),
            "started_seconds": started - begin, "ended_seconds": ended - begin}


def slots(base):
    metrics = mixed.native.metrics(mixed.native.request(base, "/metrics").decode())
    return {name: metrics[name] for name in SLOT_COUNTS}


def reject_third(base, case, begin):
    count = slots(base)
    started = perf_counter()
    request = Request(f"{base}/v1/generate", method="POST", data=json.dumps(payload(case)).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=20) as reply:
            return {"before": count, "started_seconds": started - begin,
                    "ended_seconds": perf_counter() - begin, "status": reply.status}
    except HTTPError as error:
        with error:
            response = json.loads(error.read())
        return {"before": count, "started_seconds": started - begin, "ended_seconds": perf_counter() - begin,
                "status": error.code, "code": response.get("error", {}).get("code")}


def run_burst(base, cases):
    short, long = cases
    offered_long, received_overload, short_done, long_done = (Event() for _ in range(4))
    begin = perf_counter()
    trigger, reject = {}, {}

    def original_short():
        tokens = 0

        def dispatch(kind, event, elapsed):
            nonlocal tokens
            tokens += client.bench.delta_token_count(kind, event)
            if tokens == TRIGGER_TOKEN and not offered_long.is_set():
                trigger.update(step_id=event["step"]["step_id"], seconds=elapsed)
                offered_long.set()

        observation = observe(base, short, begin, dispatch)
        short_done.set()
        return observation

    def original_long():
        require(offered_long.wait(25), "The short request never reached the long arrival token")

        def dispatch(kind, event, elapsed):
            if (kind == "step" and event["request"]["processed_prompt_tokens"] > 0
                    and not received_overload.is_set()):
                reject.update(reject_third(base, short, begin), prefill_elapsed=elapsed,
                              overlap_step=event["step"]["step_id"],
                              prompt_cursor=event["request"]["processed_prompt_tokens"])
                received_overload.set()

        observation = observe(base, long, begin, dispatch)
        long_done.set()
        return observation

    def replacement():
        require(received_overload.wait(60) and short_done.wait(90), "The overload or short terminal event was absent")
        deadline = perf_counter() + 5
        while perf_counter() < deadline and not long_done.is_set():
            current = slots(base)
            if current == dict.fromkeys(SLOT_COUNTS, 1.0):
                return {"before": current, **observe(base, short, begin)}
            sleep(0.02)
        raise ValueError("A released slot did not become available while the long stream was live")

    with ThreadPoolExecutor(max_workers=3) as workers:
        tasks = [workers.submit(job) for job in (original_short, original_long, replacement)]
        observations = [task.result() for task in tasks]
    return {"trigger": trigger, "overload": reject, "observations": observations,
            "drained": mixed.wait_drained(base)}


def verify(raw, cases, baseline):
    trigger, reject = raw["trigger"], raw["overload"]
    short, long, replacement = raw["observations"]
    for row, case, oracle in zip(raw["observations"], [*cases, cases[0]],
                                 [*baseline, baseline[0]], strict=True):
        request, trace = row["request"], row["trace"]
        solo = oracle["request"]
        require(request["ok"] and solo["ok"]
                and request["request_id"] == trace["request_id"]
                and request["prompt_token_ids"] == solo["prompt_token_ids"] == case.input_tokens
                and request["output_token_ids"] == solo["output_token_ids"]
                and len(request["output_token_ids"]) == case.max_output_tokens == 96
                and trace["initial_cursor"] == 0 and trace["cursor"] == len(case.input_tokens)
                and len(trace["decode_step_ids"]) == 96 and not trace["partial"] and not trace["held"],
                "Prompt, generated token, cursor, or route identity changed")
    require(len({row["trace"]["request_id"] for row in raw["observations"]}) == 3,
            "The replacement reused an existing request ID")
    short_steps = short["trace"]["decode_step_ids"]
    first_long = long["trace"]["prefill"][0]
    require(short_steps[TRIGGER_TOKEN - 1] == trigger["step_id"]
            and first_long["step_id"] == reject["overlap_step"]
            and reject["overlap_step"] in short_steps[TRIGGER_TOKEN:]
            and first_long["cursor"] == reject["prompt_cursor"] == 512 and first_long["scheduled"] == 2,
            "The third client did not arrive during confirmed two-request execution")
    require(reject["before"] == dict.fromkeys(SLOT_COUNTS, 2.0)
            and reject["status"] == 429 and reject["code"] == "concurrency_limit"
            and trigger["seconds"] <= long["started_seconds"] < reject["prefill_elapsed"]
            <= reject["started_seconds"] < reject["ended_seconds"] < short["ended_seconds"],
            "Native admission did not reject the third client with both slots occupied")
    require(replacement["before"] == dict.fromkeys(SLOT_COUNTS, 1.0)
            and short["ended_seconds"] <= replacement["started_seconds"] < long["ended_seconds"]
            and raw["drained"] == dict.fromkeys(mixed.COUNTERS, 0),
            "The freed slot was not reused before the second request completed")
    return {"overload_status": reject["status"], "overload_code": reject["code"],
            "max_live_streams": 2, "matching_generated_ids": 288,
            "replacement_admitted_while_long_live": True}


def main():
    parser = ArgumentParser(description=__doc__)
    for field in ("binary", "pack", "out"):
        parser.add_argument(f"--{field}", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    _, cases, _ = layer.files()["ending-decoder"]
    if args.check_inputs:
        require([len(case.input_tokens) for case in cases] == [512, 8192], "Frozen prompts changed")
        return
    inventory_path = ROOT / "mxfp4-inventory.json"
    inventory = json.loads(inventory_path.read_bytes())
    build = json.loads((ROOT / "build.json").read_bytes())
    require(mixed.digest(args.binary) == build["server_sha256"]
            and mixed.digest(inventory_path) == build["inventory_sha256"], "Source build or inventory changed")
    manifest = mixed.native.validate_inventory(args.pack, inventory, None)
    require(manifest.get("model-manifest.json") ==
            "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef",
            "The local pack does not preserve the delivered manifest")
    args.out.mkdir()
    flags = ("--disable-ngram-acceleration", "--total-blocks", "2048", "--max-batch-tokens", "2049",
             *mixed.RESERVE_OPTIONS)
    env = {"AX_MLX_FLASH_NEXT_DIRECT_COALESCING": "0", "AX_MLX_FLASH_NEXT_PREFILL_SPLIT": "1"}
    with mixed.server(args, "capacity", mixed.MODEL, args.pack, 2, env=env, options=flags,
                      stream_experts="auto", guard=True) as (base, _):
        require("decision=resident" in (args.out / "capacity.log").read_text(), "Resident experts required")
        baseline = []
        for case in cases:
            baseline.append(observe(base, case, perf_counter()))
            mixed.wait_drained(base)
        standalone = args.out / "solo.json"
        standalone.write_text(json.dumps(baseline, indent=2, allow_nan=False) + "\n")
        arms = []
        for index in range(1, 3):
            raw = run_burst(base, cases)
            path = args.out / f"burst-{index}.json"
            path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
            result = verify(raw, cases, baseline)
            require(json.loads(mixed.native.request(base, "/health"))["status"] == "ok", "Health declined")
            arms.append({"name": path.name, "sha256": mixed.digest(path), **result})
            print(f"Burst {index} verified two active streams, rejection, and live slot reuse.", flush=True)
    pressure = args.out / "capacity-samples.jsonl"
    records = [json.loads(line) for line in pressure.read_text().splitlines()]
    require(records and all(row["pressure"] == 1 for row in records), "Memory pressure exceeded normal")
    require(mixed.native.validate_inventory(args.pack, inventory, None) == manifest, "Publisher data changed")
    receipt = {
        "scope": "M2 Ultra 192 GB functional native HTTP two-slot capacity",
        "qualification": False, "build": build, "pack_revision": inventory["revision"], "manifest": manifest,
        "baseline_sha256": mixed.digest(standalone), "arms": arms,
        "normal_pressure_samples": len(records), "pressure_sha256": mixed.digest(pressure),
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
