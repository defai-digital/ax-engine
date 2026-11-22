"""Cancel an unstarted deferred Flash Next request through the stepwise HTTP route."""

from argparse import ArgumentParser
import json
from pathlib import Path
from urllib.error import HTTPError

from run_early import ROOT, inputs, mixed, require
from run_server_tail import FLAG, TRIGGER

LABEL = "cancel-held-prefill"
TERMINAL = {"finished", "cancelled", "failed"}


def post(base, path, payload):
    return json.loads(mixed.native.request(base, path, payload))


def submit(base, case):
    return post(base, "/v1/requests", {
        "model": mixed.MODEL, "input_tokens": case.input_tokens,
        "max_output_tokens": case.max_output_tokens, "sampling": {"temperature": 0, "ignore_eos": True},
    })


def snapshot(base, request_id):
    return json.loads(mixed.native.request(base, f"/v1/requests/{request_id}"))


def advance(base):
    step = post(base, "/v1/step", {})
    return {
        "step_id": step.get("step_id"), "scheduled_requests": step["scheduled_requests"],
        "scheduled_tokens": step["scheduled_tokens"],
        "hold": (step.get("route") or {}).get("crossover_decisions", {}).get(FLAG, 0),
    }


def run_control(base, short, long, trace):
    short_submit = submit(base, short)
    short_id = short_submit["request_id"]
    for _ in range(128):
        trigger = advance(base)
        short_at_trigger = snapshot(base, short_id)
        if short_at_trigger["output_len"] >= TRIGGER:
            break
    else:
        raise ValueError("The short decode never reached the trigger")
    require(short_at_trigger["output_len"] == TRIGGER and short_at_trigger["state"] not in TERMINAL,
            "The short decode skipped or exhausted the trigger")

    long_submit = submit(base, long)
    long_id = long_submit["request_id"]
    try:
        submit(base, long)
    except HTTPError as error:
        with error:
            rejection = {"status": error.code, "error": json.load(error)["error"]["code"]}
    else:
        raise ValueError("The occupied two-slot profile accepted a third request")
    require(rejection == {"status": 429, "error": "concurrency_limit"}, "Two-slot admission changed")

    path_to_hold = []
    initial_state = {
        "short_submit": short_submit, "short_at_trigger": short_at_trigger,
        "long_submit": long_submit, "capacity_rejection": rejection, "trigger": trigger,
    }
    for _ in range(12):
        step = advance(base)
        short_live, long_live = (snapshot(base, rid) for rid in (short_id, long_id))
        path_to_hold.append({"step": step, "short_output_len": short_live["output_len"],
                             "long_cursor": long_live["processed_prompt_tokens"]})
        trace.write_text(json.dumps(initial_state | {
            "path_to_hold": path_to_hold, "short_live": short_live, "long_live": long_live,
        }, indent=2, allow_nan=False) + "\n")
        require(long_live["processed_prompt_tokens"] == 0 and long_live["output_len"] == 0
                and short_live["state"] not in TERMINAL,
                "The intended deferred request started before cancellation could be tested")
        if step["hold"]:
            require(step["hold"] == 1 and step["scheduled_requests"] == step["scheduled_tokens"] == 1,
                    "The reported hold did not belong to a singleton decode")
            break
    else:
        raise ValueError("The scheduler never published a tail hold")

    cancel = post(base, f"/v1/requests/{long_id}/cancel", {})
    immediate = snapshot(base, long_id)
    short_after_cancel = snapshot(base, short_id)
    require(short_after_cancel["state"] not in TERMINAL, "The short peer finished before slot reuse")
    replacement = submit(base, long)
    replacement_id = replacement["request_id"]
    short_at_reuse = snapshot(base, short_id)
    for _ in range(160):
        advance(base)
        short_done, replacement_done = (snapshot(base, rid) for rid in (short_id, replacement_id))
        if short_done["state"] in TERMINAL and replacement_done["state"] in TERMINAL:
            break
    else:
        raise ValueError("Both surviving requests failed to drain within the step budget")

    return {
        "short_submit": short_submit, "trigger": trigger, "short_at_trigger": short_at_trigger,
        "long_submit": long_submit, "capacity_rejection": rejection, "path_to_hold": path_to_hold,
        "long_at_hold": long_live, "short_at_hold": short_live, "cancel": cancel, "cancelled_snapshot": immediate,
        "short_after_cancel": short_after_cancel, "replacement": replacement, "short_at_reuse": short_at_reuse,
        "short_done": short_done, "replacement_done": replacement_done, "drained": mixed.wait_drained(base),
    }


def verify(raw, cases, oracle):
    short, long, replacement = (raw[name] for name in ("short_submit", "long_submit", "replacement"))
    short_id, long_id, replacement_id = (row["request_id"] for row in (short, long, replacement))
    require(len({short_id, long_id, replacement_id}) == 3
            and raw["capacity_rejection"] == {"status": 429, "error": "concurrency_limit"}
            and all(row["processed_prompt_tokens"] == row["output_len"] == 0 and row["state"] not in TERMINAL
                    for row in (short, long, replacement))
            and all(row["prompt_tokens"] == case.input_tokens and row["max_output_tokens"] == 96
                    for row, case in zip((short, long, replacement), (cases[0], cases[1], cases[1]), strict=True)),
            "Submission, rejection or slot reuse did not bind the correct request")
    trigger, at_trigger, at_hold, short_live = (raw[name] for name in
        ("trigger", "short_at_trigger", "long_at_hold", "short_at_hold"))
    path = raw["path_to_hold"]
    require(at_trigger["request_id"] == short_live["request_id"] == short_id
            and at_trigger["output_len"] == TRIGGER and TRIGGER < short_live["output_len"] < 96
            and short_live["state"] not in TERMINAL
            and at_hold["request_id"] == long_id and at_hold["state"] not in TERMINAL
            and at_hold["processed_prompt_tokens"] == at_hold["output_len"] == 0
            and path and all(row["long_cursor"] == 0 and row["step"]["hold"] == 0 for row in path[:-1])
            and path[-1]["long_cursor"] == 0 and path[-1]["short_output_len"] == short_live["output_len"]
            and path[-1]["step"]["hold"] == 1
            and path[-1]["step"]["scheduled_requests"] == path[-1]["step"]["scheduled_tokens"] == 1
            and all(a < b for a, b in zip(
                [trigger["step_id"], *(row["step"]["step_id"] for row in path[:-1])],
                [row["step"]["step_id"] for row in path], strict=True)),
            "The observed singleton hold did not protect a zero-cursor long prefill")
    for label in ("cancel", "cancelled_snapshot"):
        cancelled = raw[label]
        require(cancelled["request_id"] == long_id and cancelled["state"] == "cancelled"
                and cancelled["cancel_requested"] is True and cancelled["processed_prompt_tokens"] == 0
                and cancelled["output_len"] == 0 and cancelled["output_tokens"] == [],
                "Cancellation did not terminate only the held request before prefill")
    after = raw["short_after_cancel"]
    for report in (after, raw["short_at_reuse"]):
        require(report["request_id"] == short_id and report["state"] not in TERMINAL
                and report["output_len"] == short_live["output_len"] and report["cancel_requested"] is False,
                "The short request did not remain active across cancellation and slot reuse")
    for name, rid, case, expected in zip(
            ("short_done", "replacement_done"), (short_id, replacement_id), cases, oracle, strict=True):
        result = raw[name]
        require(result["request_id"] == rid and result["state"] == "finished"
                and result["prompt_tokens"] == case.input_tokens
                and result["processed_prompt_tokens"] == len(case.input_tokens)
                and result["output_tokens"] == expected["output_token_ids"] and result["output_len"] == 96
                and result["cancel_requested"] is False,
                "The surviving or replacement request changed its exact solo trajectory")
    require(raw["drained"] == dict.fromkeys(mixed.COUNTERS, 0), "Cancellation or slot reuse leaked work")
    return {
        "held_step_id": path[-1]["step"]["step_id"],
        "cancelled_prompt_cursor": 0, "cancelled_output_tokens": 0,
        "short_tokens_at_cancellation": after["output_len"], "matching_generated_ids": 192,
        "third_request_http_status": 429, "slot_reused_while_short_active": True,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("binary", "pack", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    cases, baseline, inventory, oracle_receipt, oracle, _ = inputs()
    if args.check_inputs:
        print("Verified the native candidate and both fixed 96-token oracle trajectories.")
        return
    build = json.loads((ROOT / "candidate-build.json").read_bytes())
    require(build["runtime_commit"] == "14235e84b01456cfcb40ba6fcb717ae4798fa8f7"
            and build["runtime_parent"] == baseline["runtime_commit"]
            and all(build[key] == baseline[key] for key in ("profile", "rustc", "cargo", "mlx", "mlx_metal", "libmlx_sha256"))
            and build["codesign_verified"] is True
            and mixed.digest(args.binary) == build["server_sha256"], "Native candidate or toolchain digest changed")
    manifest = mixed.native.validate_inventory(args.pack, inventory, None)
    require(manifest == oracle_receipt["manifest"], "The delivered pack changed")
    args.out.mkdir()
    options = ("--disable-ngram-acceleration", "--total-blocks", "2048", "--max-batch-tokens", "2049",
               *mixed.RESERVE_OPTIONS)
    env = {"AX_MLX_FLASH_NEXT_DIRECT_COALESCING": "0", "AX_MLX_FLASH_NEXT_PREFILL_SPLIT": "1",
           "AX_MLX_FLASH_NEXT_PREFILL_TAIL_DEFERRAL": "1",
           "RUST_LOG": "info,ax_engine_core::engine=trace"}
    with mixed.server(args, LABEL, mixed.MODEL, args.pack, 2, env=env, options=options,
                      stream_experts="auto", guard=True) as (base, _):
        require("decision=resident" in (args.out / f"{LABEL}.log").read_text(), "Resident expert policy changed")
        raw = run_control(base, *cases, args.out / "pre-cancel-trace.json")
        record = args.out / "raw.json"
        record.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
        comparison = verify(raw, cases, oracle)
        require(json.loads(mixed.native.request(base, "/health"))["status"] == "ok", "Health declined")
    pressure = args.out / f"{LABEL}-samples.jsonl"
    samples = [json.loads(line) for line in pressure.read_text().splitlines()]
    require(samples and all(row["pressure"] == 1 for row in samples), "Memory pressure exceeded normal")
    require(mixed.native.validate_inventory(args.pack, inventory, None) == manifest, "Publisher input changed")
    result = {
        "qualification": False, "scope": "M2 Ultra 192 GB stepwise HTTP held prefill cancellation",
        "build": build, "pack_revision": inventory["revision"], "manifest": manifest,
        "inventory_sha256": oracle_receipt["inventory_sha256"], "oracle_sha256": oracle_receipt["solo"]["data_sha256"],
        "driver_sha256": mixed.digest(ROOT / "run_held_cancel.py"), "raw_sha256": mixed.digest(record),
        "pressure_sha256": mixed.digest(pressure), "pressure_samples": len(samples), "comparison": comparison,
    }
    (args.out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Complete.", flush=True)


if __name__ == "__main__":
    main()
