#!/usr/bin/env python3
"""Run Rapid-MLX MTP on repo prompt suites through its OpenAI server path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
DEFAULT_RAPID_SOURCE = Path(".internal/reference/Rapid-MLX")
DEFAULT_LIGHTNING_SOURCE = Path(".internal/reference/lightning-mlx")
DEFAULT_RAPID_PYTHON = Path("/opt/homebrew/var/mtplx/venv-0.3.7/bin/python")
MIN_DECODE_TIME = 0.5
TPS_CEILING = 500.0


@dataclass(frozen=True)
class PromptCase:
    id: str
    category: str
    prompt: str
    max_tokens: int


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    base_url: str
    model: str
    log_path: Path
    command: list[str]
    compat_patch: dict[str, Any]

    def stop(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=20)


def prepare_rapid_mtp_compat_site(
    *,
    output_dir: Path,
    lightning_source: Path,
    mode: str,
) -> dict[str, Any]:
    if mode == "none":
        return {"mode": "none"}
    if mode != "lightning":
        raise ValueError(f"unsupported Rapid-MLX MTP compatibility patch: {mode}")

    patch_path = lightning_source / "vllm_mlx" / "patches" / "qwen3_next_mtp.py"
    if not patch_path.is_file():
        raise FileNotFoundError(f"lightning-mlx MTP patch not found: {patch_path}")

    site_dir = output_dir / "rapid-compat-site"
    site_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = site_dir / "sitecustomize.py"
    sitecustomize.write_text(
        """\
import importlib.util
import json
import os
import sys
from pathlib import Path

PATCH_PATH = os.environ.get('AX_RAPID_MLX_QWEN3_NEXT_MTP_PATCH')
MODULE_NAME = 'vllm_mlx.patches.qwen3_next_mtp'
if PATCH_PATH:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, PATCH_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load Rapid-MLX MTP patch: {PATCH_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)


def _ax_patch_tokenizer_module(mod):
    # Patch _try_inject_mtp_post_load to also accept mtp.safetensors.
    # The sidecar uses mtp.safetensors (not model-mtp.safetensors) to avoid
    # mlx_lm's model*.safetensors glob picking it up during base model loading.
    # Rapid-MLX's post-load gate only checks model-mtp.safetensors, so
    # inject_mtp_support is never called without this patch.
    if getattr(mod._try_inject_mtp_post_load, '_ax_mtp_patched', False):
        return

    def _patched(model, model_name):
        from mlx_lm.utils import _download
        model_path = Path(_download(model_name))
        config_path = model_path / 'config.json'
        if not config_path.exists():
            return
        with open(config_path) as f:
            config = json.load(f)
        num_mtp = config.get('num_nextn_predict_layers', 0)
        if num_mtp == 0:
            num_mtp = config.get('text_config', {}).get('num_nextn_predict_layers', 0)
        if num_mtp > 0 and getattr(model, 'mtp', None) is None:
            mtp_file = model_path / 'model-mtp.safetensors'
            if not mtp_file.exists():
                mtp_file = model_path / 'mtp.safetensors'
            if mtp_file.exists():
                if config.get('num_nextn_predict_layers', 0) == 0:
                    config = {**config, 'num_nextn_predict_layers': num_mtp}
                mod._try_inject_mtp(model, model_path, config)

    _patched._ax_mtp_patched = True
    mod._try_inject_mtp_post_load = _patched


class _AXTokenizerPatcher:
    _TARGET = 'vllm_mlx.utils.tokenizer'

    def find_spec(self, fullname, path, target=None):
        if fullname != self._TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            fn = getattr(finder, 'find_spec', None)
            if fn is None:
                continue
            spec = fn(fullname, path, target)
            if spec is not None:
                spec.loader = _AXPatchingLoader(spec.loader)
                return spec
        return None


class _AXPatchingLoader:
    def __init__(self, real_loader):
        self._real = real_loader

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        _ax_patch_tokenizer_module(module)


sys.meta_path.insert(0, _AXTokenizerPatcher())
"""
    )
    return {
        "mode": mode,
        "site_dir": str(site_dir),
        "sitecustomize": str(sitecustomize),
        "patch_path": str(patch_path),
    }


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def load_prompt_suite(path: Path) -> list[PromptCase]:
    cases: list[PromptCase] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{path}:{line_number}: prompt must be a non-empty string")
        cases.append(
            PromptCase(
                id=str(raw.get("id") or f"case_{line_number}"),
                category=str(raw.get("category") or "unknown"),
                prompt=prompt,
                max_tokens=int(raw.get("max_tokens") or 1000),
            )
        )
    return cases


def wait_until_ready(base_url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    ready_url = base_url.rsplit("/v1", 1)[0] + "/health/ready"
    last_error: str | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Rapid-MLX server exited early with code {proc.returncode}"
            )
        try:
            with urllib.request.urlopen(ready_url, timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(
        f"Rapid-MLX server did not become ready: {last_error or ready_url}"
    )


def start_server(
    *,
    model: str,
    rapid_python: Path,
    rapid_source: Path,
    lightning_source: Path,
    rapid_mtp_patch: str,
    port: int,
    depth: int,
    startup_timeout: float,
    output_dir: Path,
) -> ServerHandle:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"rapid-mlx-server-{port}.log"
    compat_patch = prepare_rapid_mtp_compat_site(
        output_dir=output_dir,
        lightning_source=lightning_source,
        mode=rapid_mtp_patch,
    )
    cmd = [
        str(rapid_python),
        "-m",
        "vllm_mlx.cli",
        "--no-telemetry",
        "serve",
        model,
        "--port",
        str(port),
        "--disable-prefix-cache",
        "--no-memory-aware-cache",
        "--enable-mtp",
        "--mtp-num-draft-tokens",
        str(depth),
        "--force-spec-decode",
        "--no-thinking",
        "--no-mllm",
        "--log-level",
        "WARNING",
    ]
    env = os.environ.copy()
    python_path_parts = []
    if compat_patch.get("site_dir"):
        python_path_parts.append(str(Path(str(compat_patch["site_dir"])).resolve()))
        env["AX_RAPID_MLX_QWEN3_NEXT_MTP_PATCH"] = str(
            Path(str(compat_patch["patch_path"])).resolve()
        )
    python_path_parts.append(str(rapid_source.resolve()))
    if env.get("PYTHONPATH"):
        python_path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    log_file = log_path.open("w")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(rapid_source),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.close()
        handle = ServerHandle(
            proc=proc,
            base_url=f"http://127.0.0.1:{port}/v1",
            model=model,
            log_path=log_path,
            command=cmd,
            compat_patch=compat_patch,
        )
        wait_until_ready(handle.base_url, proc, startup_timeout)
        return handle
    except BaseException:
        log_file.close()
        raise


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def classify_run(completion_tokens: int, decode_time: float) -> dict[str, Any]:
    if completion_tokens <= 0:
        return {"decode_tok_s": None, "rejected_reason": "zero_completion_tokens"}
    if decode_time < MIN_DECODE_TIME:
        return {
            "decode_tok_s": None,
            "rejected_reason": f"decode_time<{MIN_DECODE_TIME}s",
        }
    tok_s = completion_tokens / decode_time
    if tok_s > TPS_CEILING:
        return {"decode_tok_s": None, "rejected_reason": f"tps>{TPS_CEILING}_ceiling"}
    return {"decode_tok_s": tok_s, "rejected_reason": None}


def run_case(
    *,
    handle: ServerHandle,
    case: PromptCase,
    max_tokens: int,
    sampling: dict[str, int | float],
    seed: int,
    measured: bool,
    repetition: int,
) -> dict[str, Any]:
    payload = {
        "model": handle.model,
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": min(max_tokens, case.max_tokens),
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "seed": seed,
        "enable_thinking": False,
    }
    started = time.time()
    t0 = time.perf_counter()
    ttft: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    text_parts: list[str] = []
    with httpx.stream(
        "POST",
        f"{handle.base_url}/chat/completions",
        json=payload,
        timeout=300.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            blob = line[len("data: ") :]
            if blob.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(blob)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    text_parts.append(str(content))
                if ttft is None and (content or delta.get("tool_calls")):
                    ttft = time.perf_counter() - t0
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                completion_tokens = int(
                    usage.get("completion_tokens") or completion_tokens
                )
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)

    total = time.perf_counter() - t0
    decode_time = max(total - (ttft or 0.0), 0.0)
    gate = classify_run(completion_tokens, decode_time)
    decode_tok_s = gate["decode_tok_s"]
    return {
        "measured": measured,
        "repetition": repetition,
        "seed": seed,
        "started_at": started,
        "ended_at": time.time(),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": completion_tokens,
        "elapsed_s": total,
        "decode_elapsed_s": decode_time,
        "ttft_s": ttft,
        "decode_tok_s": decode_tok_s,
        "end_to_end_tok_s": completion_tokens / total
        if total > 0 and completion_tokens
        else None,
        "rejected_reason": gate["rejected_reason"],
        "text": "".join(text_parts),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    valid_decode = [
        float(run["decode_tok_s"])
        for run in runs
        if run.get("decode_tok_s") is not None and run.get("rejected_reason") is None
    ]
    end_to_end = [
        float(run["end_to_end_tok_s"])
        for run in runs
        if run.get("end_to_end_tok_s") is not None
        and run.get("rejected_reason") is None
    ]
    return {
        "runs": len(runs),
        "valid_runs": len(valid_decode),
        "generated_tokens": sum(int(run.get("generated_tokens") or 0) for run in runs),
        "decode_tok_s": {
            "median": median(valid_decode),
            "p10": percentile(valid_decode, 0.10),
            "p90": percentile(valid_decode, 0.90),
            "values": valid_decode,
        },
        "end_to_end_tok_s": {
            "median": median(end_to_end),
            "p10": percentile(end_to_end, 0.10),
            "p90": percentile(end_to_end, 0.90),
            "values": end_to_end,
        },
        "accepted_drafts": None,
        "drafted_tokens": None,
        "accept_rate": None,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    prompt_suite = args.prompts.resolve()
    cases = load_prompt_suite(prompt_suite)
    if args.limit is not None:
        cases = cases[: args.limit]
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    server_dir = args.output.parent / "rapid-server"
    handle = start_server(
        model=args.model,
        rapid_python=args.rapid_python,
        rapid_source=args.rapid_source,
        lightning_source=args.lightning_source,
        rapid_mtp_patch=args.rapid_mtp_patch,
        port=args.port,
        depth=args.depth,
        startup_timeout=args.startup_timeout,
        output_dir=server_dir,
    )
    try:
        if cases:
            run_case(
                handle=handle,
                case=cases[0],
                max_tokens=32,
                sampling=sampling,
                seed=args.seed - 1,
                measured=False,
                repetition=-1,
            )

        case_results = []
        for case_index, case in enumerate(cases):
            runs: list[dict[str, Any]] = []
            all_runs = args.warmup_repetitions + args.repetitions
            for rep in range(all_runs):
                measured = rep >= args.warmup_repetitions
                run = run_case(
                    handle=handle,
                    case=case,
                    max_tokens=args.max_tokens,
                    sampling=sampling,
                    seed=args.seed + case_index * 1000 + rep,
                    measured=measured,
                    repetition=rep,
                )
                print(
                    f"{case.id} rep {rep + 1}/{all_runs}: "
                    f"decode={run['decode_tok_s'] if run['decode_tok_s'] is not None else 'NA'} "
                    f"out={run['generated_tokens']} reject={run['rejected_reason']}",
                    file=sys.stderr,
                    flush=True,
                )
                if measured:
                    runs.append(run)
                if args.cooldown > 0 and rep < all_runs - 1:
                    time.sleep(args.cooldown)
            case_results.append(
                {
                    "prompt_id": case.id,
                    "category": case.category,
                    "prompt_sha256": prompt_sha256(case.prompt),
                    "max_tokens": min(args.max_tokens, case.max_tokens),
                    "runs": runs,
                    "summary": summarize_runs(runs),
                }
            )
    finally:
        handle.stop()

    measured_runs = [run for case in case_results for run in case["runs"]]
    return {
        "schema": "ax.rapid_mlx.prompt_suite_mtp.v1",
        "engine": "rapid_mlx",
        "model": args.model,
        "rapid_source": str(args.rapid_source),
        "rapid_python": str(args.rapid_python),
        "rapid_mtp_patch": handle.compat_patch,
        "server_command": handle.command,
        "server_log": str(handle.log_path),
        "prompt_suite": str(prompt_suite),
        "prompt_suite_sha256": file_sha256(prompt_suite),
        "suite": args.suite,
        "depth_requested": args.depth,
        "depth_effective_note": "Rapid-MLX reference MTP path is single-draft; requested depth may be capped by runtime.",
        "sampling": sampling,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "warmup_repetitions": args.warmup_repetitions,
        "cooldown_s": args.cooldown,
        "disable_thinking": True,
        "measurement_gates": {
            "min_decode_time_s": MIN_DECODE_TIME,
            "tps_ceiling": TPS_CEILING,
        },
        "build": {
            "host": platform.node(),
            "platform": platform.platform(),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_tracked_dirty": bool(git_value(["status", "--porcelain"])),
        },
        "results": case_results,
        "summary": {
            **summarize_runs(measured_runs),
            "case_count": len(case_results),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rapid-source", type=Path, default=DEFAULT_RAPID_SOURCE)
    parser.add_argument(
        "--lightning-source", type=Path, default=DEFAULT_LIGHTNING_SOURCE
    )
    parser.add_argument(
        "--rapid-mtp-patch",
        choices=["lightning", "none"],
        default="lightning",
        help="Compatibility patch used for Rapid-MLX Qwen3.6 MTP loading.",
    )
    parser.add_argument(
        "--rapid-python",
        type=Path,
        default=DEFAULT_RAPID_PYTHON
        if DEFAULT_RAPID_PYTHON.exists()
        else Path(sys.executable),
    )
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_SAMPLING["temperature"]
    )
    parser.add_argument("--top-p", type=float, default=DEFAULT_SAMPLING["top_p"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_SAMPLING["top_k"])
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    result = run_suite(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved to {args.output}", file=sys.stderr)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
