"""Check exact Flash Next L1 reuse and strict-extension fallback through HTTP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from threading import Event
import time
from urllib.request import Request, urlopen

import check_flash_next_native_support as native
from run_flash_next_lifecycle import wait_drained
from run_flash_next_prefill_split_probe import MODEL, RESERVE_OPTIONS, read_stream
from run_qwen38_reference_serving import server

PACK_REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
SAVED_PREFIX = 2048
REUSED = "ax_mlx_prefix_cache_reused_tokens"
SOURCE = "ax_mlx_prefix_cache_restore_source"
STORES = "ax_mlx_prefix_cache_stores"
EVICTIONS = "ax_mlx_prefix_cache_evictions"
L1 = 2


def build_prompts(corpus):
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    seeds = [row["input_tokens"] for row in rows if len(row["input_tokens"]) == SAVED_PREFIX][:2]
    short = [row["input_tokens"] for row in rows if len(row["input_tokens"]) == 512]
    assert len({tuple(row[:64]) for row in short[:3]}) == 3
    assert len({seeds[0][0], seeds[1][0], short[3][0], short[4][0]}) == 4
    prompts = {"seed-a": seeds[0], "seed-b": seeds[1], "exact-replay": seeds[0],
               "target": seeds[0] + short[0][:64],
               "cancelled": seeds[1] + short[1][:64], "replacement": seeds[1] + short[2][:64],
               "survivor-a": short[3], "survivor-b": short[4]}
    return {name: {"id":name, "input_tokens":tokens,
                   "max_output_tokens":1 if name.startswith("seed-") else 192 if name.startswith("survivor") else 96}
            for name, tokens in prompts.items()}


def capture_cache_events(lines, rows):
    event = None
    for line in lines:
        if not line.strip():
            event = None
        elif line.startswith(b"event:"):
            event = line.partition(b":")[2].strip()
        elif line.startswith(b"data:") and event in (b"step", b"response"):
            data = json.loads(line.partition(b":")[2].strip())
            report = data["request"] if event == b"step" else data["response"]
            decisions = report.get("route", {}).get("crossover_decisions", {})
            relevant = {key: decisions[key] for key in (REUSED, SOURCE, STORES, EVICTIONS) if decisions.get(key, 0)}
            if relevant:
                rows.append({"event": event.decode(), "step_id": data.get("step", {}).get("step_id"),
                             "request_id": report["request_id"], "decisions": relevant})
        yield line


def generate(base, prompt, *, first=lambda: None, disconnect=False):
    call = Request(base + "/v1/generate/stream", data=json.dumps({
        "model": MODEL, "input_tokens": prompt["input_tokens"], "max_output_tokens": prompt["max_output_tokens"],
        "sampling": {"temperature": 0, "ignore_eos": True},
    }).encode(), headers={"Content-Type": "application/json"})
    cache = []
    started = time.monotonic()
    with urlopen(call, timeout=30) as response:
        result = read_stream(capture_cache_events(response, cache), prompt, started, first,
                             "output" if disconnect else None)
    result["cache_events"] = cache
    return result


def overlap(base, peer, target, *, disconnect=False):
    decoding = Event()

    def arrival():
        if not decoding.wait(30):
            raise TimeoutError("The independent peer did not reach the first token")
        return generate(base, target, disconnect=disconnect)

    with ThreadPoolExecutor(max_workers=2) as pool:
        peer_reply = pool.submit(generate, base, peer, first=decoding.set)
        target_reply = pool.submit(arrival)
        survivor, resumed = peer_reply.result(timeout=120), target_reply.result(timeout=120)
    return survivor, resumed


def restored_l1(reply):
    return any(row["decisions"].get(SOURCE) == L1 and row["decisions"].get(REUSED) == SAVED_PREFIX
               for row in reply["cache_events"])


def summarize_resume(survivor, target, cold_survivor, cold_target, *, disconnected=False):
    own = target["first_output_tokens"] if disconnected else target["output_tokens"]
    expected = cold_target["output_tokens"][:len(own)] if disconnected else cold_target["output_tokens"]
    decoder = [row["step_id"] for row in survivor["scheduled_work"] if row["delta_token_count"] > 0]
    ongoing = set(decoder[1:-1])
    shared_steps = [row["step_id"] for row in target["prefill_progress"]
                    if row["scheduled_requests"] == 2 and row["step_id"] in ongoing]
    return {"target_tokens_match": own == expected, "peer_tokens_match": survivor["output_tokens"] == cold_survivor["output_tokens"],
            "peer_no_restored_tokens": all(not row["decisions"].get(REUSED, 0) for row in survivor["cache_events"]),
            "l1_restore_2048_proven": restored_l1(target), "shared_prefill_steps_with_ongoing_decode": shared_steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("binary", "pack", "inventory", "build-receipt", "corpus", "out"):
        parser.add_argument("--" + field, type=Path, required=field == "corpus")
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--check-workload", action="store_true")
    args = parser.parse_args()
    assert native.sha256(args.corpus) == args.corpus_sha256
    prompts = build_prompts(args.corpus)
    if args.check_workload:
        print(json.dumps({key: len(row["input_tokens"]) for key, row in prompts.items()}))
        return
    if any(getattr(args, name) is None for name in ("binary", "pack", "inventory", "build_receipt", "out")):
        parser.error("measurement requires the binary, pack, inventory, build receipt and output")
    build = json.loads(args.build_receipt.read_bytes())
    assert native.sha256(args.binary) == build["server_sha256"]
    assert build["commit"] == "cdc7ce2e9e5bb7a3fb7c9dd26d62f6e4a07b8a51"
    inventory = json.loads(args.inventory.read_bytes())
    assert inventory["revision"] == PACK_REVISION and len(inventory["members"]) == 49
    assert not any(Path(command.strip()).name.startswith("ax-engine-server") for command in
                   subprocess.check_output(["/bin/ps", "-axo", "comm="], text=True).splitlines())
    args.out.mkdir()
    record = {"status":"verifying", "pack_revision":PACK_REVISION, "binary_sha256":build["server_sha256"],
              "inventory_sha256":native.sha256(args.inventory), "driver_sha256":native.sha256(Path(__file__)),
              "prompts":prompts, "arms":{}}

    def save(status):
        record["status"] = status
        tmp = args.out / "receipt.tmp"
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.out / "receipt.json")
        print(status, flush=True)

    try:
        save("verifying")
        native.validate_inventory(args.pack, inventory)
        manifest = args.pack / "model-manifest.json"
        record["manifest_sha256"] = native.sha256(manifest)
        standard = ("--disable-ngram-acceleration", "--total-blocks", "2048")
        off = {"AX_MLX_FLASH_NEXT_PREFILL_SPLIT":"0", "AX_MLX_FLASH_NEXT_DIRECT_COALESCING":"0"}
        cold = record["arms"]["cold"] = {"quiet":native.wait_for_quiet()}
        with server(args, "cold", MODEL, args.pack, 1, env=off, guard=True,
                    options=(*standard, "--max-batch-tokens", "2048")) as (base, _):
            cold["outputs"] = {key: generate(base, row) for key, row in prompts.items() if not key.startswith("seed-")}
            cold["drained"] = wait_drained(base)
        if any(event["decisions"].get(REUSED, 0) or event["decisions"].get(STORES, 0)
               for reply in cold["outputs"].values() for event in reply["cache_events"]):
            raise RuntimeError("The cold baseline returned prefix cache activity")
        save("cold-complete")
        warm_env = off | {"AX_ENGINE_PREFIX_REUSE_DISABLED":"0", "AX_MLX_PREFIX_CACHE_MAX_BYTES":"2147483648",
                          "AX_MLX_PREFIX_CACHE_MAX_ENTRIES":"8"}
        warm = record["arms"]["l1"] = {"quiet":native.wait_for_quiet()}
        with server(args, "l1", MODEL, args.pack, 2, env=warm_env, guard=True,
                    options=(*standard, "--max-batch-tokens", "2049", *RESERVE_OPTIONS)) as (base, _):
            warm["seed-a"] = generate(base, prompts["seed-a"])
            warm["after_seed_a"] = wait_drained(base)
            warm["exact-replay"] = generate(base, prompts["exact-replay"])
            warm["exact_replay_drained"] = wait_drained(base)
            peer, target = overlap(base, prompts["survivor-a"], prompts["target"])
            warm["paired"] = {"peer":peer, "target":target, "drained":wait_drained(base)}
            warm["paired"]["checks"] = summarize_resume(peer, target, cold["outputs"]["survivor-a"], cold["outputs"]["target"])
            warm["seed-b"] = generate(base, prompts["seed-b"])
            warm["after_seed_b"] = wait_drained(base)
            peer, target = overlap(base, prompts["survivor-b"], prompts["cancelled"], disconnect=True)
            warm["disconnect"] = {"peer":peer, "target":target, "drained":wait_drained(base)}
            warm["disconnect"]["checks"] = summarize_resume(
                peer, target, cold["outputs"]["survivor-b"], cold["outputs"]["cancelled"], disconnected=True
            )
            warm["replacement"] = generate(base, prompts["replacement"])
            warm["replacement_tokens_match"] = warm["replacement"]["output_tokens"] == cold["outputs"]["replacement"]["output_tokens"]
            warm["replacement_l1_restore_2048_proven"] = restored_l1(warm["replacement"])
            warm["replacement_drained"] = wait_drained(base)
            warm["health_after"] = json.loads(native.request(base, "/health"))["status"]
        for arm in ("cold", "l1"):
            rows = [json.loads(line) for line in (args.out / f"{arm}-samples.jsonl").read_text().splitlines()]
            assert rows and all(row["pressure"] == 1 for row in rows)
            record["arms"][arm]["peak_footprint_bytes"] = max(row["footprint_bytes"] for row in rows)
        native.validate_inventory(args.pack, inventory)
        assert native.sha256(manifest) == record["manifest_sha256"]
        warm["both_seed_stores_reported"] = all(
            any(row["decisions"].get(STORES, 0) for row in warm[seed]["cache_events"])
            for seed in ("seed-a", "seed-b")
        )
        record["warm_overlap_proven"] = warm["both_seed_stores_reported"] and all(
            all(warm[name]["checks"].values()) for name in ("paired", "disconnect")
        ) and warm["replacement_tokens_match"] and warm["replacement_l1_restore_2048_proven"]             and warm["health_after"] == "ok"
        record["exact_prompt_reuses_and_matches"] = (
            restored_l1(warm["exact-replay"])
            and warm["exact-replay"]["output_tokens"] == cold["outputs"]["exact-replay"]["output_tokens"]
            and warm["both_seed_stores_reported"]
        )
        extension_results = (warm["paired"]["target"], warm["disconnect"]["target"], warm["replacement"])
        record["strict_extensions_cold_prefill_and_match"] = (
            not any(event["decisions"].get(REUSED, 0) for result in extension_results
                    for event in result["cache_events"])
            and all(warm[name]["checks"]["target_tokens_match"] and warm[name]["checks"]["peer_tokens_match"]
                    for name in ("paired", "disconnect"))
            and warm["replacement_tokens_match"] and warm["health_after"] == "ok"
        )
        if not record["exact_prompt_reuses_and_matches"] or not record["strict_extensions_cold_prefill_and_match"]:
            raise RuntimeError("Exact reuse or strict-extension cold control did not match the shipping route")
        save("complete")
    except Exception:
        save("failed")
        raise


if __name__ == "__main__":
    main()
