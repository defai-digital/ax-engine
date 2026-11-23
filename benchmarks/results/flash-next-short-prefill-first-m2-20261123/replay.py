"""Reconcile bounded short-first activation, continuation, and token identity."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent
CORPUS_ARCHIVE = ROOT.parent / "flash-next-mxfp4-mixed-serving-m5-20261121" / "evidence.tar.xz"
CORPUS_HASH = "2c0c205d803207df6316dbd0076eef97451a55c80c0413fdc6b3bee5ffef7e9a"
MARKER = "ax_scheduler_resident_short_prefill_first"
COUNTERS = {
    "ax_engine_jobs_in_flight", "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued", "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
}


def load_file(name):
    return gzip.decompress((ROOT / (name + ".gz")).read_bytes())


def progress(step, long, short, enabled, prompts):
    assert step["scheduled_requests"] == 1
    for row, prompt in ((long, prompts[0]), (short, prompts[1])):
        assert row["state"] == "runnable"
        assert row["prompt_tokens"] == prompt and row["max_output_tokens"] == 96
    expected_long = 2048 if enabled else 4096
    expected_short = 512 if enabled else 0
    assert step["scheduled_tokens"] == (512 if enabled else 2048)
    assert long["processed_prompt_tokens"] == expected_long and long["output_len"] == 0
    assert short["processed_prompt_tokens"] == expected_short and short["output_len"] == int(enabled)
    marker = (short.get("route") or {}).get("crossover_decisions", {}).get(MARKER, 0)
    assert bool(marker) is enabled


def completions(rows, prompts, ids):
    assert len(rows) == len(ids) == len(prompts)
    for row, prompt, output in zip(rows, prompts, ids, strict=True):
        assert row["prompt_tokens"] == prompt and row["processed_prompt_tokens"] == len(prompt)
        assert row["state"] == "finished" and row["max_output_tokens"] == row["output_len"] == 96
        assert row["output_tokens"] == output
    return sum(map(len, ids))


def replay():
    assert (ROOT / "status.txt").read_text().strip() == "0"
    build = json.loads((ROOT / "build.json").read_bytes())
    inventory = json.loads((ROOT / "mxfp4-inventory.json").read_bytes())
    assert len(inventory["members"]) == 49 and build["codesign_verified"]
    assert build["patch_sha256"] == sha256(load_file("source.patch")).hexdigest()
    inputs = {name: digest for digest, name in map(str.split, (ROOT / "inputs.sha256").read_text().splitlines())}
    assert inputs["source.patch"] == build["patch_sha256"]
    assert inputs["ax-engine-server"] == build["server_sha256"]
    assert inputs["mxfp4-inventory.json"] == sha256((ROOT / "mxfp4-inventory.json").read_bytes()).hexdigest()
    assert inputs["check_short_prefill_first.py"] == sha256(load_file("check_short_prefill_first.py")).hexdigest()
    assert inputs["mixed-corpus.jsonl"] == CORPUS_HASH
    data = json.loads(load_file("receipt.json"))
    assert data["qualification"] is False
    assert data["revision"] == inventory["revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert data["build"] == build
    assert data["manifest"]["model-manifest.json"] == "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
    with tarfile.open(CORPUS_ARCHIVE) as archive:
        corpus = archive.extractfile("mixed-corpus.jsonl").read()
    assert sha256(corpus).hexdigest() == CORPUS_HASH
    by_id = {row["id"]: row["input_tokens"] for row in map(json.loads, corpus.splitlines())}
    ids = data["prompt_ids"]
    prompts = [by_id[name] for name in ids]
    assert list(map(len, prompts)) == [8192, 512, 512] and prompts[1] != prompts[2]
    assert set(data["solo"]) == set(ids)
    oracles = [data["solo"][name] for name in ids]
    assert all(len(row) == 96 for row in oracles)

    assert [arm["enabled"] for arm in data["arms"]] == [False, True]
    matched, samples = 0, 0
    peaks = []
    for arm in data["arms"]:
        enabled = arm["enabled"]
        assert arm["flags"]["AX_MLX_FLASH_NEXT_SHORT_PREFILL_FIRST"] == str(int(enabled))
        assert arm["flags"]["AX_MLX_FLASH_NEXT_PREFILL_SPLIT"] == "1"
        label = "short-first" if enabled else "fifo"
        pressure = [json.loads(line) for line in load_file(f"{label}-samples.jsonl").splitlines()]
        assert len(pressure) == arm["normal_pressure_samples"] > 0
        assert all(row["pressure"] == 1 for row in pressure)
        peak = max(row["footprint_bytes"] for row in pressure)
        assert peak == arm["peak_physical_footprint_bytes"] <= 96 << 30
        samples += len(pressure)
        peaks.append(peak)
        assert len(arm["drains"]) == 1 + int(enabled)
        assert all(set(drain) == COUNTERS and all(value == 0 for value in drain.values()) for drain in arm["drains"])
        first = arm["first"]
        progress(first["step"], first["long"], first["short"], enabled, prompts[:2])
        matched += completions(arm["completed"], prompts[:2], oracles[:2])
        cancel = arm["cancel_and_replace"]
        if not enabled:
            assert cancel is None
            continue
        first = cancel["first"]
        progress(first["step"], first["long"], first["short"], True, prompts[:2])
        ended = cancel["cancelled"]
        assert ended["state"] == "cancelled" and ended["request_id"] == first["short"]["request_id"]
        assert ended["output_tokens"] == first["short"]["output_tokens"] == oracles[1][:1]
        after_long, replacement = cancel["after_replacement"]
        assert after_long["request_id"] == first["long"]["request_id"]
        assert replacement["request_id"] not in (ended["request_id"], after_long["request_id"])
        progress(cancel["next_step"], after_long, replacement, False, (prompts[0], prompts[2]))
        matched += completions(cancel["completed"], (prompts[0], prompts[2]), (oracles[0], oracles[2]))
    return {
        "matched_full_output_ids": matched,
        "matched_cancelled_prefix_ids": 1,
        "normal_pressure_samples": samples,
        "peak_footprint_bytes": max(peaks),
        "server_sha256": build["server_sha256"],
        "short_precedes_long_second_slice": True,
        "replacement_preserves_long_turn": True,
        "qualification": False,
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))
