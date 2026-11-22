"""Exercise an early Flash Next long prompt with decode-tail deferral enabled."""

from argparse import ArgumentParser
import json
from math import isclose
from pathlib import Path
from unittest.mock import patch

from run_server_tail import ROOT, TRIGGER, capture_tail, client, layer, mixed, require

EARLY_TRIGGERS = (48, 65)
SERVER = "early-arrivals"


def verify(raw, at, cases, oracle):
    short, long = raw["traces"]
    short_row, long_row = raw["observations"]
    short_decode = short["decode_step_ids"]
    require(at in EARLY_TRIGGERS and raw["trigger_token"] == at and len(raw["trigger"]) == 1,
            "Unexpected arrival trigger")
    trigger = raw["trigger"][0]
    require(len(short_decode) == len(long["decode_step_ids"]) == 96
            and short_decode[at - 1] == trigger["step_id"]
            and short["request_id"] == short_row["request_id"]
            and long["request_id"] == long_row["request_id"]
            and short["request_id"] != long["request_id"]
            and (short["initial_cursor"], long["initial_cursor"]) == (0, 0)
            and (short["cursor"], long["cursor"]) == (512, 8192)
            and not short["partial"] and not long["partial"]
            and not short["held"] and not long["held"],
            "The early run changed cursor/stream identity or entered a hold")
    offered, sent = trigger["seconds"], long_row["started_at_s"]
    short_end = short_row["started_at_s"] + short_row["e2e_latency_ms"] / 1000
    require(offered <= sent < short_end and (sent - offered) * 1000 <= 10
            and isclose(offered, long_row["scheduled_at_s"], abs_tol=1e-9)
            and isclose((sent - offered) * 1000, long_row["queue_delay_ms"], abs_tol=0.001),
            "The early long request was not sent at the trigger")
    for row, case, solo in zip(raw["observations"], cases, oracle, strict=True):
        require(row["ok"] and row["status"] == solo["status"]
                and row["prompt_id"] == solo["prompt_id"] == case.id
                and row["prompt_token_ids"] == solo["prompt_token_ids"] == case.input_tokens
                and row["output_token_ids"] == solo["output_token_ids"]
                and len(row["output_token_ids"]) == 96,
                "The two-slot output changed from the same-binary solo oracle")

    peer_steps = set(short_decode[at:])
    threshold = short_decode[TRIGGER - 1]
    prefill = long["prefill"]
    require(prefill and trigger["step_id"] < prefill[0]["step_id"] < threshold
            and prefill[0]["step_id"] in peer_steps,
            "Long prefill did not enter before the short decode tail")
    cursor = 0
    previous_step = trigger["step_id"]
    groups = {}
    for row in prefill:
        end = (cursor // client.LOGICAL + 1) * client.LOGICAL
        step = row["step_id"]
        require(step > previous_step and row["width"] > 0
                and cursor + row["width"] == row["cursor"] <= end
                and row["scheduled"] == (2 if step in peer_steps else 1),
                "Physical prefill skipped a cursor, logical boundary, or peer")
        groups.setdefault(end, []).append(row)
        cursor = row["cursor"]
        previous_step = step

    split = []
    for end, rows in groups.items():
        markers = {row["split_end"] for row in rows}
        widths = [row["width"] for row in rows]
        first = rows[0]["step_id"]
        if markers == {end}:
            require(widths == client.SPLIT_WIDTHS and first in peer_steps,
                    "A split group changed its widths or began without the peer")
            split.append(end)
        else:
            require(markers == {None} and widths == [client.LOGICAL] and first > short_decode[-1],
                    "A solo group changed its boundary or overlapped the decoder")
    require(cursor == 8192 and len(groups) == 4 and split, "Incomplete mixed prefill geometry")
    return {
        "first_prefill_step": prefill[0]["step_id"],
        "short_token_84_step": threshold,
        "last_prefill_step": prefill[-1]["step_id"],
        "prefill_active_at_tail": prefill[0]["step_id"] < threshold <= prefill[-1]["step_id"],
        "split_logical_ends": split,
        "split_groups_crossing_tail": [
            end for end in split
            if groups[end][0]["step_id"] < threshold <= groups[end][-1]["step_id"]
        ],
    }


def inputs():
    _, cases, _ = layer.files()["ending-decoder"]
    build = json.loads((ROOT / "build.json").read_bytes())
    inventory_file = ROOT / "mxfp4-inventory.json"
    inventory = json.loads(inventory_file.read_bytes())
    oracle_file = ROOT / "oracle.json"
    oracle = json.loads(oracle_file.read_bytes())
    receipt = json.loads((ROOT / "oracle-receipt.json").read_bytes())
    require(receipt["build"] == build and receipt["scope"] == "M2 functional"
            and receipt["status"] == "complete"
            and receipt["pack_revision"] == inventory["revision"]
            and receipt["inventory_sha256"] == mixed.digest(inventory_file)
            and receipt["solo"]["data_sha256"] == mixed.digest(oracle_file),
            "The previous single-slot receipt does not bind this run")
    expected = {}
    for case, row in zip(cases, oracle, strict=True):
        require(row["ok"] and row["prompt_id"] == case.id
                and row["prompt_token_ids"] == case.input_tokens
                and len(row["output_token_ids"]) == case.max_output_tokens == 96,
                "The sealed solo oracle uses different prompts or token budgets")
        expected[case.id] = row["output_token_ids"]
    return cases, build, inventory, receipt, oracle, expected


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("binary", "pack", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    cases, build, inventory, receipt, oracle, expected = inputs()
    if args.check_inputs:
        print("Verified both prompts, 192 solo token IDs, and the M2 build binding.")
        return
    require(mixed.digest(args.binary) == build["server_sha256"], "Native binary digest mismatch")
    manifest = mixed.native.validate_inventory(args.pack, inventory, None)
    require(manifest == receipt["manifest"], "The delivered inventory changed")
    args.out.mkdir()
    options = ("--disable-ngram-acceleration", "--total-blocks", "2048", "--max-batch-tokens", "2049",
               *mixed.RESERVE_OPTIONS)
    env = {"AX_MLX_FLASH_NEXT_DIRECT_COALESCING": "0", "AX_MLX_FLASH_NEXT_PREFILL_SPLIT": "1",
           "AX_MLX_FLASH_NEXT_PREFILL_TAIL_DEFERRAL": "1"}
    trials = {}
    with mixed.server(args, SERVER, mixed.MODEL, args.pack, 2, env=env, options=options,
                      stream_experts="auto", guard=True) as (base, _):
        require("decision=resident" in (args.out / f"{SERVER}.log").read_text(), "Resident experts required")
        require(json.loads(mixed.native.request(base, "/health"))["status"] == "ok", "Readiness failed")
        for at in EARLY_TRIGGERS:
            print(f"Running with the long request sent at short token {at}.", flush=True)
            with (patch.object(client, "TRIGGER_TOKEN", at),
                  patch.object(client.capture, "trace_stream", capture_tail),
                  patch.object(client, "verify_deferral", lambda raw, _: verify(raw, at, cases, oracle))):
                data = client.burst(base, cases, expected, False, args.out / f"trigger-{at}.partial.json")
            require(data["drained"] == dict.fromkeys(mixed.COUNTERS, 0), "Requests did not drain")
            require(json.loads(mixed.native.request(base, "/health"))["status"] == "ok", "Health declined")
            path = args.out / f"trigger-{at}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
            trials[str(at)] = mixed.digest(path)
    pressure = args.out / f"{SERVER}-samples.jsonl"
    samples = [json.loads(row) for row in pressure.read_text().splitlines()]
    require(samples and all(row["pressure"] == 1 for row in samples), "Memory pressure exceeded normal")
    require(mixed.native.validate_inventory(args.pack, inventory, None) == manifest, "Pack changed")
    result = {
        "scope": "M2 Ultra 192 GB native HTTP functional test",
        "qualification": False,
        "build": build,
        "pack_revision": inventory["revision"],
        "manifest": manifest,
        "inventory_sha256": receipt["inventory_sha256"],
        "oracle_sha256": receipt["solo"]["data_sha256"],
        "pressure_sha256": mixed.digest(pressure),
        "pressure_samples": len(samples),
        "trials": trials,
    }
    (args.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Complete.", flush=True)


if __name__ == "__main__":
    main()
