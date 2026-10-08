#!/usr/bin/env python3
"""Same-pack Qwen 3.8 Flash Next peer benchmark: AX Engine, MTPLX, and OMLX.

Mirrors the Qwen 3.8 27B peer contract (flappy suite, greedy, fixed token
budget, warmup plus measured repetitions) against one MXFP4 MTP directory.
mlx-lm has no ``qwen4_exp`` loader, so that lane is recorded as unsupported
without loading weights. A lane that fails to load is unsupported. This
does not qualify MTP-S, MTP-P, MTP-D, or a product default.

The preparation record in ``scripts/flash_next_peer_bench_plan.json`` stays
non-executable. This harness is the measured comparison, and it records its
own artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_REPO = "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP"
PACK_REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
PROMPTS = REPO_ROOT / "benchmarks/prompts/mtp-suites/flappy.jsonl"
MTPLX_SOURCE = REPO_ROOT / ".internal/reference/MTPLX"
OMLX_SOURCE = REPO_ROOT / ".internal/reference/omlx"
HARDWARE_LABEL = (
    "Mac Studio, Apple M2 Ultra, 192 GiB unified memory, internal NVMe SSD"
)
HOSTNAME_KEYS = {"hostname", "host_id", "machine_hostname", "serial_number", "hardware_uuid"}


def model_type_of(model_dir: Path) -> str | None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type
    text = config.get("text_config")
    if isinstance(text, dict) and isinstance(text.get("model_type"), str):
        return str(text["model_type"])
    return None


def mlx_lm_supports(model_type: str | None) -> bool:
    """True when mlx-lm ships a loader module for this model type."""
    if not model_type or model_type.startswith("qwen4_exp"):
        return False
    module = f"mlx_lm.models.{model_type}"
    try:
        __import__(module)
    except ModuleNotFoundError:
        return False
    return True


def _pythonpath(source: Path, base: str | None) -> str:
    parts = [str(source)]
    if base:
        parts.append(base)
    return os.pathsep.join(parts)


def lane_commands(
    *,
    python: str,
    omlx_python: str,
    pack: Path,
    prompts: Path,
    output_dir: Path,
    repetitions: int,
    warmup_repetitions: int,
    max_tokens: int,
    cooldown: float,
    depth: int,
    mtplx_source: Path,
    omlx_pack: Path | None = None,
) -> dict[str, list[str]]:
    """Argv for each lane. The mlx-lm lane is omitted when the loader is absent."""
    ax = [
        python,
        str(REPO_ROOT / "scripts/bench_mlx_inference_stack.py"),
        "--model-repo-id",
        PACK_REPO,
        "--model-dir",
        str(pack),
        "--prompt-source",
        "real",
        "--real-prompt-suite",
        str(prompts),
        "--no-thinking",
        "--generation-tokens",
        str(max_tokens),
        "--repetitions",
        str(repetitions),
        "--warmup-repetitions",
        str(warmup_repetitions),
        "--cooldown",
        str(cooldown),
        "--prefill-step-size",
        "2048",
        "--skip-mlx-lm",
        "--no-build-ax-engine",
        "--ax-ngram-accel",
        "--ax-mtp-policy",
        "required",
        "--ax-mtp-disable-ngram-stacking",
        "--ax-qwen-linear-mtp-exact",
        "--output",
        str(output_dir / "ax_engine.json"),
    ]
    mtplx = [
        python,
        str(REPO_ROOT / "scripts/bench_mtplx_prompt_suites.py"),
        "--model",
        str(pack),
        "--source-dir",
        str(mtplx_source),
        "--suite",
        "flappy",
        "--prompts",
        str(prompts),
        "--output",
        str(output_dir / "mtplx.json"),
        "--profile",
        "sustained",
        "--depth",
        str(depth),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--top-k",
        "0",
        "--seed",
        "0",
        "--max-tokens",
        str(max_tokens),
        "--repetitions",
        str(repetitions),
        "--warmup-repetitions",
        str(warmup_repetitions),
        "--cooldown",
        str(cooldown),
        "--ignore-eos",
        "--disable-thinking",
        "--allow-unverified-model",
    ]
    omlx = [
        omlx_python,
        str(REPO_ROOT / "scripts/bench_omlx_prompt_suites.py"),
        "--model",
        str(omlx_pack or pack),
        "--model-repo-id",
        PACK_REPO,
        "--suite",
        "flappy",
        "--prompts",
        str(prompts),
        "--output",
        str(output_dir / "omlx.json"),
        "--host-label",
        HARDWARE_LABEL,
        "--max-tokens",
        str(max_tokens),
        "--repetitions",
        str(repetitions),
        "--warmup-repetitions",
        str(warmup_repetitions),
        "--cooldown",
        str(cooldown),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--top-k",
        "0",
        "--seed",
        "0",
        "--engine",
        "vlm",
        "--mtp-draft-tokens",
        str(depth),
    ]
    mlx_lm = [
        python,
        str(REPO_ROOT / "scripts/bench_mlx_lm_prompt_suites.py"),
        "--model",
        str(pack),
        "--prompts",
        str(prompts),
        "--output",
        str(output_dir / "mlx_lm.json"),
        "--host-label",
        HARDWARE_LABEL,
        "--max-tokens",
        str(max_tokens),
        "--repetitions",
        str(repetitions),
        "--warmup-repetitions",
        str(warmup_repetitions),
        "--cooldown",
        str(cooldown),
    ]
    return {"ax": ax, "mtplx": mtplx, "omlx": omlx, "mlx_lm": mlx_lm}


def lane_env(lane: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """Drop inherited AX and MTPLX experiment flags, then point peers at source."""
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith("AX_") or key.startswith("MTPLX_"):
            del env[key]
    if lane == "mtplx":
        env["PYTHONPATH"] = _pythonpath(MTPLX_SOURCE, env.get("PYTHONPATH"))
    elif lane == "omlx":
        env["PYTHONPATH"] = _pythonpath(OMLX_SOURCE, env.get("PYTHONPATH"))
    return env


def scrub_host_identity(value: Any, hostname: str, hardware_label: str = HARDWARE_LABEL) -> Any:
    """Replace machine hostnames. Rust target triples and chip names stay."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in HOSTNAME_KEYS or (key == "host" and isinstance(item, str)):
                cleaned[key] = hardware_label
            else:
                cleaned[key] = scrub_host_identity(item, hostname, hardware_label)
        return cleaned
    if isinstance(value, list):
        return [scrub_host_identity(item, hostname, hardware_label) for item in value]
    if isinstance(value, str) and hostname and hostname in value:
        return value.replace(hostname, hardware_label)
    return value


def _trial_numbers(trials: list[Any], key: str) -> list[float]:
    numbers: list[float] = []
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        if trial.get("measured") is False:
            continue
        raw = trial.get(key)
        if isinstance(raw, (int, float)):
            numbers.append(float(raw))
    return numbers


def measured_medians(payload: dict[str, Any]) -> dict[str, Any]:
    """Median of measured repetitions, matching the 27B 20-run peer table."""
    decode: list[float] = []
    prefill: list[float] = []
    results = payload.get("results")
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, dict):
                continue
            values = row.get("decode_tok_s_values")
            if isinstance(values, list):
                decode.extend(float(item) for item in values if isinstance(item, (int, float)))
            trials = row.get("trials")
            if isinstance(trials, list) and trials:
                decode.extend(_trial_numbers(trials, "decode_tok_s"))
                prefill.extend(_trial_numbers(trials, "prefill_tok_s"))
                continue
            runs = row.get("runs")
            if isinstance(runs, list):
                decode.extend(_trial_numbers(runs, "decode_tok_s"))
                for run in runs:
                    if not isinstance(run, dict) or run.get("measured") is False:
                        continue
                    prompt_tokens = row.get("prompt_tokens")
                    prompt_eval = run.get("prompt_eval_time_s")
                    if (
                        isinstance(prompt_tokens, int)
                        and isinstance(prompt_eval, (int, float))
                        and float(prompt_eval) > 0
                    ):
                        prefill.append(float(prompt_tokens) / float(prompt_eval))
    summary = payload.get("summary")
    if not decode and isinstance(summary, dict):
        nested = summary.get("decode_tok_s")
        if isinstance(nested, dict) and isinstance(nested.get("median"), (int, float)):
            decode_median = float(nested["median"])
        else:
            decode_median = None
    else:
        decode_median = statistics.median(decode) if decode else None
    if isinstance(payload.get("decode_tok_s_median_20"), (int, float)) and not decode:
        decode_median = float(payload["decode_tok_s_median_20"])
    return {
        "decode_tok_s_median": decode_median,
        "prefill_tok_s_median": statistics.median(prefill) if prefill else None,
        "measured_decode_samples": len(decode),
        "measured_prefill_samples": len(prefill),
    }


def merge_sidecar_into_index(
    index: dict[str, Any],
    keys: list[str],
    filename: str,
) -> dict[str, Any]:
    """Return an index that also names an existing sidecar's tensors."""
    merged = json.loads(json.dumps(index))
    weight_map = merged.setdefault("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model index weight_map is not an object")
    for key in keys:
        weight_map.setdefault(key, filename)
    return merged


def omlx_index_view(pack: Path, dest: Path) -> Path:
    """Symlink the pack and list mtp.safetensors in the OMLX index.

    OMLX discovers embedded Qwen4 MTP only through
    model.safetensors.index.json. The pinned pack keeps that sidecar out of
    the language index. This view does not copy weight bytes and does not
    modify the pack.
    """
    sidecar = pack / "mtp.safetensors"
    index_path = pack / "model.safetensors.index.json"
    if not sidecar.is_file() or not index_path.is_file():
        return pack
    dest.mkdir(parents=True, exist_ok=True)
    for entry in pack.iterdir():
        if entry.name == "model.safetensors.index.json":
            continue
        link = dest / entry.name
        if not link.exists():
            link.symlink_to(entry)
    from safetensors import safe_open

    with safe_open(sidecar, framework="numpy") as handle:
        keys = list(handle.keys())
    index = json.loads(index_path.read_text(encoding="utf-8"))
    merged = merge_sidecar_into_index(index, keys, sidecar.name)
    (dest / "model.safetensors.index.json").write_text(
        json.dumps(merged, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest


def classify_load_failure(log_text: str) -> str | None:
    """Turn a peer load rejection into an unsupported record, not a tok/s."""
    if "parameters not in model" in log_text:
        return (
            "Peer load rejected checkpoint tensors that are not parameters "
            "of its model module. No tokens were served."
        )
    if "Model type qwen4_exp not supported" in log_text:
        return "Peer has no loader for model_type qwen4_exp. Weights were not served."
    return None


def unsupported_record(lane: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "ax.flash_next_peer_lane.v1",
        "lane": lane,
        "status": "unsupported",
        "reason": reason,
        "hardware": HARDWARE_LABEL,
    }


def write_json(path: Path, payload: Any, hostname: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = scrub_host_identity(payload, hostname)
    path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")


def scrub_file(path: Path, hostname: str) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    write_json(path, payload, hostname)


def run_command(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--omlx-python", default=sys.executable)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="MTP draft depth for MTPLX and OMLX. AX uses its required MTP policy.",
    )
    parser.add_argument(
        "--lanes",
        nargs="+",
        choices=("ax", "mtplx", "omlx", "mlx_lm"),
        default=["ax", "mtplx", "omlx", "mlx_lm"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One flappy case, 32 tokens, one warmup and one measured repetition.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write plan.json and do not launch a runtime.",
    )
    return parser.parse_args(argv)


def resolve_prompts(args: argparse.Namespace, output_dir: Path) -> Path:
    if not args.smoke:
        return PROMPTS
    cases = [
        json.loads(line)
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    smoke_path = output_dir / "smoke-flappy.jsonl"
    smoke_path.write_text(json.dumps(cases[0]) + "\n", encoding="utf-8")
    return smoke_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pack = args.model_dir.expanduser().resolve()
    if not (pack / "config.json").is_file():
        print(f"model dir has no config.json: {pack}", file=sys.stderr)
        return 2
    model_type = model_type_of(pack)
    if model_type != "qwen4_exp":
        print(f"expected model_type qwen4_exp, found {model_type!r}", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y-%m-%d")
    output_dir = args.output_dir or (
        REPO_ROOT / "benchmarks/results/mtp-axq-peer" / f"{stamp}-apple-m2-ultra-192gb-flash-next"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    repetitions = 1 if args.smoke else args.repetitions
    warmup = 1 if args.smoke else args.warmup_repetitions
    max_tokens = 32 if args.smoke else args.max_tokens
    prompts = resolve_prompts(args, output_dir)
    omlx_pack = pack
    if "omlx" in args.lanes:
        omlx_pack = omlx_index_view(pack, Path("/tmp/ax-flash-next-omlx-view"))
    commands = lane_commands(
        python=args.python,
        omlx_python=args.omlx_python,
        pack=pack,
        prompts=prompts,
        output_dir=output_dir,
        repetitions=repetitions,
        warmup_repetitions=warmup,
        max_tokens=max_tokens,
        cooldown=0.0 if args.smoke else args.cooldown,
        depth=args.depth,
        mtplx_source=MTPLX_SOURCE,
        omlx_pack=omlx_pack,
    )
    hostname = socket.gethostname()
    plan = {
        "schema_version": "ax.flash_next_peer_plan.v1",
        "hardware": HARDWARE_LABEL,
        "platform": platform.platform(),
        "pack_repo": PACK_REPO,
        "pack_revision": PACK_REVISION,
        "model_dir": str(pack),
        "omlx_model_dir": str(omlx_pack),
        "omlx_index_note": (
            "OMLX reads a same-byte symlink view whose index also names the "
            "existing mtp.safetensors sidecar. The pack directory is unchanged."
            if omlx_pack != pack
            else "OMLX uses the pack directory."
        ),
        "model_type": model_type,
        "prompts": str(prompts),
        "repetitions": repetitions,
        "warmup_repetitions": warmup,
        "max_tokens": max_tokens,
        "depth": args.depth,
        "smoke": bool(args.smoke),
        "lanes": {
            lane: {"command": commands[lane], "env_pythonpath": lane_env(lane).get("PYTHONPATH")}
            for lane in args.lanes
        },
        "qualification": False,
        "open_gates": ["MTP-S", "MTP-P", "MTP-D", "default admission", "release"],
    }
    write_json(output_dir / "plan.json", plan, hostname)
    if args.dry_run:
        print(f"Wrote {output_dir / 'plan.json'}")
        return 0

    summaries: dict[str, Any] = {}
    for lane in args.lanes:
        if lane == "mlx_lm" and not mlx_lm_supports(model_type):
            reason = (
                f"mlx-lm has no loader for model_type {model_type!r}; "
                "weights were not loaded"
            )
            record = unsupported_record(lane, reason)
            write_json(output_dir / "mlx_lm.json", record, hostname)
            summaries[lane] = {"status": "unsupported", "reason": reason}
            print(f"{lane}: unsupported ({reason})", flush=True)
            continue
        log_path = output_dir / f"{lane}.log"
        started = time.time()
        print(f"{lane}: starting", flush=True)
        code = run_command(commands[lane], lane_env(lane), log_path)
        elapsed = time.time() - started
        artifact_name = {
            "ax": "ax_engine.json",
            "mtplx": "mtplx.json",
            "omlx": "omlx.json",
            "mlx_lm": "mlx_lm.json",
        }[lane]
        artifact = output_dir / artifact_name
        scrub_file(artifact, hostname)
        load_failure = None
        if code != 0 and log_path.is_file():
            load_failure = classify_load_failure(log_path.read_text(encoding="utf-8", errors="replace"))
        reason = None
        if load_failure is not None and not artifact.is_file():
            write_json(
                output_dir / artifact_name,
                unsupported_record(lane, load_failure),
                hostname,
            )
            artifact = output_dir / artifact_name
            status = "unsupported"
            reason = load_failure
        else:
            status = "measured" if code == 0 and artifact.is_file() else "failed"
        medians: dict[str, Any] = {}
        if status == "measured":
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            if payload.get("status") == "unsupported":
                status = "unsupported"
                reason = payload.get("reason")
            else:
                medians = measured_medians(payload)
        summaries[lane] = {
            "status": status,
            "exit_code": code,
            "elapsed_s": round(elapsed, 3),
            "artifact": artifact_name,
            "log": log_path.name,
            **({"reason": reason} if reason else {}),
            **medians,
        }
        print(f"{lane}: {status} exit={code} elapsed_s={elapsed:.1f}", flush=True)
    summary = {
        "schema_version": "ax.flash_next_peer_summary.v1",
        "hardware": HARDWARE_LABEL,
        "pack_repo": PACK_REPO,
        "pack_revision": PACK_REVISION,
        "model_type": model_type,
        "contract": {
            "suite": "flappy" if not args.smoke else "flappy-smoke",
            "sampling": "temperature 0, top_p 1, top_k 0, seed 0",
            "max_tokens": max_tokens,
            "repetitions": repetitions,
            "warmup_repetitions": warmup,
            "peer_depth": args.depth,
            "ax_mtp_policy": "required",
            "stream_experts": "auto",
        },
        "qualification": False,
        "lanes": summaries,
    }
    write_json(output_dir / "summary.json", summary, hostname)
    print(f"Wrote {output_dir / 'summary.json'}")
    return 0 if all(item["status"] in {"measured", "unsupported"} for item in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
