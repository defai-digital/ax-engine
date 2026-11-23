"""Measure native mixed-step execution for archived Flash Next prompts."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import check_flash_next_native_support as native
from run_qwen38_reference_serving import server

ROOT = Path(__file__).resolve().parent
MODEL = "qwen3.8-flash-next:axq"
OPTIONS = (
    "--disable-ngram-acceleration", "--total-blocks", "2048",
    "--max-batch-tokens", "2049", "--multi-prefill-fair",
    "--max-prefill-tokens-per-request-per-step", "2048",
    "--max-inflight-prefill-requests", "1",
)
FLAGS = {
    "AX_MLX_FLASH_NEXT_ALIGNED_BUFFERED": "0",
    "AX_MLX_FLASH_NEXT_DIRECT_COALESCING": "0",
    "AX_MLX_FLASH_NEXT_PREFILL_SPLIT": "1",
    "AX_MLX_FLASH_NEXT_SHORT_PREFILL_FIRST": "1",
    "AX_MLX_FLASH_NEXT_DRAIN_PRIORITY_SHORT": "0",
    "RUST_LOG": "info,ax_engine_mlx::flash_next_startup=debug",
}
TIMINGS = (
    "ax_mlx_prefill_wall_us", "ax_mlx_prefill_forward_wall_us",
    "ax_mlx_prefill_prefix_cache_wall_us", "ax_mlx_prefill_generation_state_wall_us",
    "ax_mlx_flash_next_mtp_verified_steps",
)
DRAINED = (
    "ax_engine_jobs_in_flight", "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued", "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
)


def read(path):
    return json.loads(path.read_bytes())


def call(base, path, body=None):
    return json.loads(native.request(base, path, body))


def submit(base, ids):
    result = call(base, "/v1/requests", {
        "model": MODEL, "input_tokens": ids, "max_output_tokens": 96,
        "sampling": {"temperature": 0, "top_p": 1.0, "top_k": 0,
                     "seed": 0, "ignore_eos": True},
    })
    assert result["state"] == "waiting"
    assert result["prompt_tokens"] == ids and result["processed_prompt_tokens"] == 0
    return result["request_id"]


def snapshot(base, request_id):
    result = call(base, f"/v1/requests/{request_id}")
    assert result["request_id"] == request_id
    assert result["output_len"] == len(result["output_tokens"])
    return result


def advance(base):
    started = time.perf_counter_ns()
    step = call(base, "/v1/step", {})
    client_us = (time.perf_counter_ns() - started) // 1000
    decisions = (step.get("route") or {}).get("crossover_decisions", {})
    runner_us = step["runner_time_us"]
    core_us = step["cpu_time_us"]
    assert step["prefix_hits"] == 0
    assert type(step["step_id"]) is int and 0 < runner_us <= core_us <= client_us
    return {
        "step_id": step["step_id"], "scheduled_requests": step["scheduled_requests"],
        "scheduled_tokens": step["scheduled_tokens"], "runner_us": runner_us,
        "core_us": core_us, "step_http_us": client_us,
        "decisions": {name: value for name, value in decisions.items()
                      if name in TIMINGS or name.startswith("ax_scheduler_")},
    }


def cancel(base, request_id):
    result = call(base, f"/v1/requests/{request_id}/cancel", {})
    assert result["state"] == "cancelled" and result["request_id"] == request_id


def drained(base):
    metrics = native.metrics(native.request(base, "/metrics").decode())
    counts = {name: metrics[name] for name in DRAINED}
    assert not any(counts.values()), counts
    return counts


def observed_mixed(base, case):
    long_id = submit(base, case["long_prompt_ids"])
    first = advance(base)
    long_state = snapshot(base, long_id)
    cursor = case["original_short_priority_long_cursor"]
    assert long_state["processed_prompt_tokens"] == cursor and not long_state["output_tokens"]
    assert first["scheduled_requests"] == 1 and first["scheduled_tokens"] == cursor
    short_id = submit(base, case["short_prompt_ids"])
    widths = [cursor]
    output = []
    history = []
    target_cursor = case["long_prefill_cursor_before_step"]
    target_depth = len(case["short_emitted_ids_before_step"])
    for _ in range(32):
        step = advance(base)
        new_long = snapshot(base, long_id)["processed_prompt_tokens"]
        new_output = snapshot(base, short_id)["output_tokens"]
        assert new_long >= cursor and new_output[:len(output)] == output
        assert len(new_output) - len(output) in (0, 1)
        history.append({
            **step, "long_before": cursor, "long_after": new_long,
            "short_before": len(output), "short_after": len(new_output),
        })
        width = new_long - cursor
        if cursor == target_cursor and len(output) == target_depth:
            assert widths == case["long_completed_widths"]
            assert width == case["observed_mixed_prefill_width"] and len(new_output) == target_depth + 1
            assert step["scheduled_requests"] == 2 and step["scheduled_tokens"] == width + 1
            assert step["decisions"].get(f"ax_scheduler_prefill_split_end.{long_id}") == case["observed_logical_end"]
            assert step["decisions"]["ax_mlx_prefill_forward_wall_us"] > 0
            cancel(base, short_id)
            cancel(base, long_id)
            return {"short_request_id": short_id, "long_request_id": long_id,
                    "history": history, "output": new_output, "drained": drained(base)}
        assert new_long <= target_cursor and len(new_output) <= target_depth
        if width:
            widths.append(width)
        cursor, output = new_long, new_output
    raise AssertionError("The mixed schedule did not reach the archived work geometry")


def observed_solo(base, case):
    request_id = submit(base, case["short_prompt_ids"])
    depth = len(case["short_emitted_ids_before_step"]) + 1
    for _ in range(depth):
        step = advance(base)
        state = snapshot(base, request_id)
        assert step["scheduled_requests"] == 1
    assert step["scheduled_tokens"] == 1 and len(state["output_tokens"]) == depth
    cancel(base, request_id)
    return {"step": step, "output": state["output_tokens"], "drained": drained(base)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "pack", "out", "link-root"):
        parser.add_argument(f"--{name}", type=Path, required=name != "link-root")
    parser.add_argument("--require-archive-ids", action="store_true")
    args = parser.parse_args()
    archive = read(ROOT / "fixtures.json")
    build = read(ROOT / "build.json")
    inventory = read(ROOT / "mxfp4-inventory.json")
    assert native.sha256(args.binary) == build["server_sha256"] and build["codesign_verified"]
    assert native.sha256(ROOT / "source.patch") == build["patch_sha256"]
    assert len(inventory["members"]) == 49
    before = native.validate_inventory(args.pack, inventory, args.link_root)
    assert before["model-manifest.json"] == native.sha256(ROOT / "model-manifest.json")
    args.out.mkdir()
    idle = [native.wait_for_quiet()["cpu_idle_percent"]]
    rows = []
    with server(args, "native", MODEL, args.pack, 2, env=FLAGS,
                options=OPTIONS, stream_experts="auto", guard=True) as (base, _):
        for case in archive["cases"]:
            mixed = observed_mixed(base, case)
            solo = observed_solo(base, case)
            expected = [*case["short_emitted_ids_before_step"], case["short_token_at_shared_step"]]
            row = {"short_id": case["short_id"], "mixed": mixed, "solo": solo,
                   "archived_ids_match": mixed["output"] == expected,
                   "same_host_solo_ids_match": mixed["output"] == solo["output"]}
            rows.append(row)
            (args.out / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
            assert row["same_host_solo_ids_match"]
            assert not args.require_archive_ids or row["archived_ids_match"]
            print(case["short_id"], mixed["history"][-1], flush=True)
    idle.append(native.cpu_idle_percent())
    samples = [json.loads(line) for line in (args.out / "native-samples.jsonl").read_text().splitlines()]
    assert samples and all(row["pressure"] == 1 for row in samples)
    log = (args.out / "native.log").read_text()
    assert "decision=resident" in log and "reader=Mapped" in log
    assert native.validate_inventory(args.pack, inventory, args.link_root) == before
    receipt = {
        "scope": "Native stepwise attribution, experimental two-slot disabled-MTP profile",
        "qualification": False, "build": build, "manifest": before, "flags": FLAGS,
        "options": OPTIONS, "cpu_idle_percent": idle, "storage": native.storage_info(args.pack),
        "soc": call_os("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(call_os("sysctl", "-n", "hw.memsize")),
        "macos": call_os("sw_vers", "-productVersion"),
        "fixtures_sha256": native.sha256(ROOT / "fixtures.json"),
        "probe_sha256": native.sha256(Path(__file__).resolve()),
        "pressure_samples": len(samples), "rows": rows,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def call_os(*arguments):
    return subprocess.check_output(arguments, text=True).strip()


if __name__ == "__main__":
    main()
