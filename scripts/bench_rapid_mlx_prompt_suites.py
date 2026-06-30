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
    model_layout: dict[str, Any]
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


def prepare_model_layout(model: str, output_dir: Path) -> tuple[str, dict[str, Any]]:
    source = Path(model)
    if not source.is_dir():
        return model, {"mode": "unchanged", "model": model}
    source = source.resolve()
    nested_mtp = source / "mtp" / "weights.safetensors"
    if (
        (source / "model-mtp.safetensors").exists()
        or (source / "mtp.safetensors").exists()
        or not nested_mtp.exists()
    ):
        return str(source), {"mode": "unchanged", "model": str(source)}

    layout_dir = (output_dir / "normalized-model-layout").resolve()
    layout_dir.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        link = layout_dir / child.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(child, target_is_directory=child.is_dir())
    mtp_link = layout_dir / "mtp.safetensors"
    if not mtp_link.exists() and not mtp_link.is_symlink():
        mtp_link.symlink_to(nested_mtp)
    return str(layout_dir), {
        "mode": "symlink_view",
        "source": str(source),
        "runtime_model": str(layout_dir),
        "added_sidecar": "mtp.safetensors",
        "sidecar_source": str(nested_mtp.resolve()),
    }


def prepare_rapid_mtp_compat_site(
    *,
    output_dir: Path,
    lightning_source: Path,
    mode: str,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    if mode == "none":
        return {"mode": "none", "ignore_eos": bool(ignore_eos)}
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
import types
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
    _orig_infer_mtp_group_size = getattr(module, '_infer_quantized_mtp_group_size', None)

    def _ax_infer_quantized_mtp_group_size(raw, bits, fallback):
        if int(bits or 0) == 6:
            for key, weight in raw.items():
                if not key.endswith('.weight') or not key.startswith('mtp.'):
                    continue
                scales = raw.get(key.removesuffix('.weight') + '.scales')
                if scales is None or len(weight.shape) != 2 or len(scales.shape) != 2:
                    continue
                packed_input_dim = int(weight.shape[1])
                scale_groups = int(scales.shape[1])
                if packed_input_dim <= 0 or scale_groups <= 0:
                    continue
                input_dim = packed_input_dim * 8
                if input_dim % scale_groups == 0:
                    return input_dim // scale_groups
        if _orig_infer_mtp_group_size is not None:
            return _orig_infer_mtp_group_size(raw, bits, fallback)
        return fallback

    module._infer_quantized_mtp_group_size = _ax_infer_quantized_mtp_group_size

share_pkg = types.ModuleType('vllm_mlx.share')
share_pkg.__path__ = []
share_cli = types.ModuleType('vllm_mlx.share.cli')

def _ax_noop_share_register(_subparsers):
    return None

share_cli.register = _ax_noop_share_register
sys.modules.setdefault('vllm_mlx.share', share_pkg)
sys.modules.setdefault('vllm_mlx.share.cli', share_cli)


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
        text_config = config.get('text_config', {})
        num_mtp = config.get('num_nextn_predict_layers', 0)
        if num_mtp == 0:
            num_mtp = config.get('mtp_num_hidden_layers', 0)
        if num_mtp == 0:
            num_mtp = text_config.get('num_nextn_predict_layers', 0)
        if num_mtp == 0:
            num_mtp = text_config.get('mtp_num_hidden_layers', 0)
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


class _AXSchedulerPatcher:
    _TARGET = 'vllm_mlx.scheduler'

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
                spec.loader = _AXSchedulerLoader(spec.loader)
                return spec
        return None


class _AXSchedulerLoader:
    def __init__(self, real_loader):
        self._real = real_loader

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        if os.environ.get('AX_RAPID_MLX_IGNORE_EOS') not in {'1', 'true', 'yes'}:
            return
        scheduler = getattr(module, 'Scheduler', None)
        if scheduler is None or getattr(scheduler, '_ax_ignore_eos_patched', False):
            return

        def _ax_empty_stop_tokens(self):
            return set()

        scheduler._get_stop_tokens = _ax_empty_stop_tokens
        scheduler._ax_ignore_eos_patched = True


sys.meta_path.insert(0, _AXSchedulerPatcher())
"""
    )
    return {
        "mode": mode,
        "ignore_eos": bool(ignore_eos),
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


def wait_until_ready(
    base_url: str,
    proc: subprocess.Popen,
    timeout_s: float,
    lightning_mode: bool = False,
) -> None:
    deadline = time.time() + timeout_s
    base = base_url.rsplit("/v1", 1)[0]
    # Lightning-mlx uses /health (model_loaded=true) instead of /health/ready
    ready_url = f"{base}/health" if lightning_mode else f"{base}/health/ready"
    last_error: str | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Rapid-MLX server exited early with code {proc.returncode}"
            )
        try:
            with urllib.request.urlopen(ready_url, timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    if lightning_mode:
                        data = json.loads(response.read())
                        if data.get("model_loaded"):
                            return
                    else:
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
    lightning_mode: bool = False,
    enable_ngram: bool = False,
    mtp_optimistic: bool = False,
    mtp_draft_temperature: float | None = None,
    disable_prefix_cache: bool = False,
    prefill_step_size: int | None = None,
    enable_thinking: bool = False,
    ignore_eos: bool = False,
    ngram_auto_disable_mtp_threshold: float = 0.85,
    ngram_auto_disable_min_ngram: float = 0.50,
    log_tag: str | None = None,
) -> ServerHandle:
    output_dir.mkdir(parents=True, exist_ok=True)
    # log_tag identifies the run variant (e.g. "lightning_mlx", "lightning_mtp_ngram")
    # so that multiple invocations sharing the same output_dir and port do not
    # clobber each other's server log. Without this the second run overwrites
    # the first, and post-hoc audits can't tell which preset the first run used.
    log_suffix = f"-{log_tag}" if log_tag else ""
    log_path = output_dir / f"rapid-mlx-server{log_suffix}-{port}.log"
    model_arg, model_layout = prepare_model_layout(model, output_dir)
    compat_patch = prepare_rapid_mtp_compat_site(
        output_dir=output_dir,
        lightning_source=lightning_source,
        mode=rapid_mtp_patch,
        ignore_eos=ignore_eos,
    )
    if lightning_mode:
        cmd = [
            str(rapid_python),
            "-m",
            "vllm_mlx.cli",
            "serve",
            model_arg,
            "--served-model-name",
            "local",
            "--port",
            str(port),
            "--max-num-seqs",
            "1",
            "--prefill-batch-size",
            "1",
            "--completion-batch-size",
            "1",
            "--default-temperature",
            "0.6",
            "--default-top-p",
            "0.95",
            "--stream-interval",
            "1",
            "--enable-mtp",
            "--mtp-num-draft-tokens",
            str(depth),
            "--log-level",
            "WARNING",
        ]
        if not enable_thinking:
            cmd.append("--no-thinking")
        if disable_prefix_cache:
            cmd.append("--disable-prefix-cache")
        if prefill_step_size is not None:
            cmd += ["--prefill-step-size", str(prefill_step_size)]
        if mtp_draft_temperature is not None:
            cmd += ["--mtp-draft-temperature", str(mtp_draft_temperature)]
        if mtp_optimistic:
            cmd.append("--mtp-optimistic")
        if enable_ngram:
            # Layer n-gram prompt-lookup before MTP.  These params mirror the
            # lightning-mlx production preset's "winning combination" for
            # MTP+ngram as documented in cli.py:
            #   K=6 wide, min_occ=2, greedy accept, hybrid MTP tail,
            #   draft everywhere (not just in <think>) since --no-thinking
            #   suppresses <think> blocks entirely, skip tool calls,
            #   self-tune on, auto-disable at the preset's MTP/ngram thresholds.
            cmd += [
                "--enable-ngram",
                "--ngram-num-draft-tokens", "6",
                "--ngram-min-occurrences", "2",
                "--ngram-acceptance-mode", "greedy",
                "--ngram-hybrid-verify",
                "--ngram-everywhere",
                "--ngram-skip-tool-calls",
                "--ngram-self-tune",
                "--ngram-self-tune-disable-threshold", "0.30",
                "--ngram-auto-disable-mtp-threshold",
                str(ngram_auto_disable_mtp_threshold),
                "--ngram-auto-disable-min-ngram",
                str(ngram_auto_disable_min_ngram),
            ]
    else:
        cmd = [
            str(rapid_python),
            "-m",
            "vllm_mlx.cli",
            "--no-telemetry",
            "serve",
            model_arg,
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
    if ignore_eos:
        env["AX_RAPID_MLX_IGNORE_EOS"] = "1"
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
            model="local" if lightning_mode else model_arg,
            model_layout=model_layout,
            log_path=log_path,
            command=cmd,
            compat_patch=compat_patch,
        )
        wait_until_ready(handle.base_url, proc, startup_timeout, lightning_mode=lightning_mode)
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
    enable_thinking: bool = False,
    require_full_output_tokens: bool = False,
) -> dict[str, Any]:
    requested_tokens = min(max_tokens, case.max_tokens)
    payload = {
        "model": handle.model,
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": requested_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "seed": seed,
        "enable_thinking": enable_thinking,
    }
    started = time.time()
    t0 = time.perf_counter()
    ttft_content: float | None = None
    ttft_reasoning: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    chunks_total = 0
    chunks_with_content = 0
    chunks_with_reasoning = 0
    chunks_with_tool_calls = 0
    chunks_empty_delta = 0
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
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                tool_calls = delta.get("tool_calls")
                chunks_total += 1
                if content:
                    content_parts.append(str(content))
                    chunks_with_content += 1
                    if ttft_content is None:
                        ttft_content = time.perf_counter() - t0
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                    chunks_with_reasoning += 1
                    if ttft_reasoning is None:
                        ttft_reasoning = time.perf_counter() - t0
                if tool_calls:
                    chunks_with_tool_calls += 1
                if not content and not reasoning and not tool_calls:
                    chunks_empty_delta += 1
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                completion_tokens = int(
                    usage.get("completion_tokens") or completion_tokens
                )
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)

    total = time.perf_counter() - t0
    # TTFT is the earliest of either content or reasoning_content arriving.
    # This is the time-to-first-visible-byte. If only reasoning arrived
    # (thinking streamed) ttft tracks that; if only content arrived
    # (thinking buffered or off) ttft tracks content.
    ttft_candidates = [t for t in (ttft_content, ttft_reasoning) if t is not None]
    ttft: float | None = min(ttft_candidates) if ttft_candidates else None
    decode_time = max(total - (ttft or 0.0), 0.0)
    gate = classify_run(completion_tokens, decode_time)
    fixed_token_complete = completion_tokens == requested_tokens
    if require_full_output_tokens and not fixed_token_complete:
        gate = {
            "decode_tok_s": None,
            "rejected_reason": "generated_tokens_lt_requested_tokens",
        }
    decode_tok_s = gate["decode_tok_s"]
    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)
    # Heuristic for the silent-thinking pathology: server reports many
    # completion_tokens but client received almost no visible content.
    # When suspected, downstream analysis can flag the run as "thinking
    # consumed the token budget even though --no-thinking was set".
    visible_chars = len(full_content) + len(full_reasoning)
    silent_thinking_suspected = (
        completion_tokens >= 200
        and visible_chars < completion_tokens  # <1 char/token average
        and chunks_with_content == 0
        and chunks_with_reasoning == 0
    )
    return {
        "measured": measured,
        "repetition": repetition,
        "seed": seed,
        "started_at": started,
        "ended_at": time.time(),
        "prompt_tokens": prompt_tokens,
        "requested_tokens": requested_tokens,
        "fixed_token_complete": fixed_token_complete,
        "generated_tokens": completion_tokens,
        "elapsed_s": total,
        "decode_elapsed_s": decode_time,
        "ttft_s": ttft,
        "ttft_content_s": ttft_content,
        "ttft_reasoning_s": ttft_reasoning,
        "decode_tok_s": decode_tok_s,
        "end_to_end_tok_s": completion_tokens / total
        if total > 0 and completion_tokens
        else None,
        "rejected_reason": gate["rejected_reason"],
        "stream_chunk_stats": {
            "total": chunks_total,
            "with_content": chunks_with_content,
            "with_reasoning_content": chunks_with_reasoning,
            "with_tool_calls": chunks_with_tool_calls,
            "empty_delta": chunks_empty_delta,
        },
        "visible_text_chars": len(full_content),
        "reasoning_text_chars": len(full_reasoning),
        "silent_thinking_suspected": silent_thinking_suspected,
        "content_head": full_content[:500],
        "content_tail": full_content[-500:] if len(full_content) > 500 else "",
        "reasoning_head": full_reasoning[:500],
        "reasoning_tail": full_reasoning[-500:] if len(full_reasoning) > 500 else "",
        # Backward-compat: prior schema used a single "text" field that was
        # content || reasoning_content. Keep it as the visible content stream
        # so older analysis scripts continue to work.
        "text": full_content,
    }


def fetch_last_mtp_accept_ratio(base_url: str) -> float | None:
    try:
        with urllib.request.urlopen(f"{base_url}/requests?limit=1", timeout=3) as resp:  # noqa: S310
            data = json.loads(resp.read())
            entries = data.get("entries") or []
            if entries:
                val = entries[-1].get("mtp_acceptance_ratio")
                return float(val) if val is not None else None
    except Exception:
        pass
    return None


def fetch_last_ngram_accept_ratio(base_url: str) -> float | None:
    """Fetch the n-gram acceptance ratio for the most recent request.

    Reads from the same /v1/requests?limit=1 endpoint as MTP accept ratio.
    The ngram_acceptance_ratio field is populated by lightning-mlx's
    request_metrics recorder when --enable-ngram is active.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/requests?limit=1", timeout=3) as resp:  # noqa: S310
            data = json.loads(resp.read())
            entries = data.get("entries") or []
            if entries:
                val = entries[-1].get("ngram_acceptance_ratio")
                return float(val) if val is not None else None
    except Exception:
        pass
    return None


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
    accept_values = [
        float(run["mtp_acceptance_ratio"])
        for run in runs
        if run.get("mtp_acceptance_ratio") is not None
    ]
    ngram_accept_values = [
        float(run["ngram_acceptance_ratio"])
        for run in runs
        if run.get("ngram_acceptance_ratio") is not None
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
        "accept_rate": median(accept_values) if accept_values else None,
        "ngram_accept_rate": median(ngram_accept_values) if ngram_accept_values else None,
    }


def capture_lightning_source_identity(source: Path) -> dict[str, Any]:
    """Record the exact state of the lightning-mlx source tree.

    Captures git commit, tag, dirty flag, list of modified files, and a
    SHA-256 of the working-tree diff. Lets the artifact reproduce or audit
    any row without having to keep the source dir untouched.
    """

    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(source),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    commit = _git("rev-parse", "HEAD")
    describe = _git("describe", "--tags", "--always")
    status = _git("status", "--porcelain")
    diff = _git("diff", "HEAD")
    pyproject_version: str | None = None
    pyproject = source / "pyproject.toml"
    if pyproject.is_file():
        for line in pyproject.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("version") and "=" in stripped:
                # naive parse: version = "0.6.10"
                parts = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                pyproject_version = parts
                break
    modified_files: list[str] = []
    if status:
        for line in status.splitlines():
            # git status --porcelain: " M path/to/file" or "?? untracked"
            if len(line) >= 3:
                modified_files.append(line[3:].strip())
    diff_sha256 = (
        hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
    )
    return {
        "source_path": str(source),
        "git_commit": commit,
        "git_describe": describe,
        "is_dirty": bool(status),
        "modified_files": modified_files,
        "diff_sha256": diff_sha256,
        "pyproject_version": pyproject_version,
    }


def read_log_tail(
    path: Path, *, max_lines: int = 200, max_bytes: int = 32 * 1024
) -> list[str]:
    """Best-effort read of a log file's last lines for inclusion in the artifact.

    Bounded so a runaway log doesn't blow up the JSON output.
    """
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-max_lines:]


def parse_server_header(lines: list[str]) -> dict[str, Any]:
    """Extract structured facts from the lightning-mlx server's banner.

    The banner is fairly stable across versions: ``Alias: ...``, ``Model: ...``,
    ``MTP: enabled, ...``, ``N-gram: enabled, ...``. We pull these out so the
    artifact records what the server actually loaded, independent of what the
    benchmark thought it requested.
    """
    info: dict[str, Any] = {
        "alias_line": None,
        "model_line": None,
        "mtp_line": None,
        "ngram_line": None,
        "features_line": None,
        "mtp_depth_cap_warnings": 0,
        "effective_mtp_depth": None,
    }
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Alias:") and info["alias_line"] is None:
            info["alias_line"] = stripped
        elif stripped.startswith("Model:") and info["model_line"] is None:
            info["model_line"] = stripped
        elif stripped.startswith("MTP:") and info["mtp_line"] is None:
            info["mtp_line"] = stripped
        elif stripped.startswith("N-gram:") and info["ngram_line"] is None:
            info["ngram_line"] = stripped
        elif stripped.startswith("Features:") and info["features_line"] is None:
            info["features_line"] = stripped
        if "depth will be capped to" in line:
            info["mtp_depth_cap_warnings"] += 1
            # Parse "depth will be capped to N per verify cycle."
            try:
                tail = line.split("capped to", 1)[1].strip()
                effective = int(tail.split()[0])
                # Take the smallest (most restrictive) cap seen.
                prev = info["effective_mtp_depth"]
                info["effective_mtp_depth"] = (
                    effective if prev is None else min(prev, effective)
                )
            except (IndexError, ValueError):
                pass
    return info


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
    lightning_mode = bool(getattr(args, "lightning_mode", False))
    enable_ngram = bool(getattr(args, "enable_ngram", False))
    mtp_optimistic = bool(getattr(args, "mtp_optimistic", False))
    enable_thinking = bool(getattr(args, "enable_thinking", False)) if lightning_mode else False
    # Tag the server log file with the output stem so concurrent or
    # sequential runs that share server_dir + port don't clobber each other.
    log_tag = args.output.stem
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
        lightning_mode=lightning_mode,
        enable_ngram=enable_ngram,
        mtp_optimistic=mtp_optimistic,
        mtp_draft_temperature=getattr(args, "mtp_draft_temperature", None),
        disable_prefix_cache=bool(getattr(args, "disable_prefix_cache", False)),
        prefill_step_size=getattr(args, "prefill_step_size", None),
        enable_thinking=enable_thinking,
        ignore_eos=bool(args.ignore_eos),
        ngram_auto_disable_mtp_threshold=args.ngram_auto_disable_mtp_threshold,
        ngram_auto_disable_min_ngram=args.ngram_auto_disable_min_ngram,
        log_tag=log_tag,
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
                enable_thinking=enable_thinking,
                require_full_output_tokens=False,
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
                    enable_thinking=enable_thinking,
                    require_full_output_tokens=bool(args.require_full_output_tokens),
                )
                print(
                    f"{case.id} rep {rep + 1}/{all_runs}: "
                    f"decode={run['decode_tok_s'] if run['decode_tok_s'] is not None else 'NA'} "
                    f"out={run['generated_tokens']}/{run['requested_tokens']} "
                    f"reject={run['rejected_reason']}",
                    file=sys.stderr,
                    flush=True,
                )
                if measured:
                    accept_ratio = fetch_last_mtp_accept_ratio(handle.base_url)
                    if accept_ratio is not None:
                        run["mtp_acceptance_ratio"] = accept_ratio
                    if enable_ngram:
                        ngram_ratio = fetch_last_ngram_accept_ratio(handle.base_url)
                        if ngram_ratio is not None:
                            run["ngram_acceptance_ratio"] = ngram_ratio
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
            if args.inter_case_cooldown > 0 and case_index < len(cases) - 1:
                print(
                    f"  [rapid-mlx] inter-case cooldown {args.inter_case_cooldown:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(args.inter_case_cooldown)
    finally:
        handle.stop()

    # Snapshot the server log after shutdown. The banner (alias/model/MTP/N-gram
    # lines) tells us what the server actually loaded — independent of what the
    # benchmark CLI intended. The tail captures the depth-cap warnings and any
    # late errors. Both go into the artifact so the log file can be deleted
    # later without losing audit data.
    server_log_lines = read_log_tail(handle.log_path, max_lines=400, max_bytes=64 * 1024)
    server_header_info = parse_server_header(server_log_lines)
    server_log_head: list[str] = []
    try:
        with handle.log_path.open("r", errors="replace") as f:
            for _ in range(80):
                line = f.readline()
                if not line:
                    break
                server_log_head.append(line.rstrip("\n"))
    except OSError:
        pass

    lightning_identity = (
        capture_lightning_source_identity(args.lightning_source)
        if lightning_mode
        else None
    )

    measured_runs = [run for case in case_results for run in case["runs"]]
    return {
        "schema": "ax.rapid_mlx.prompt_suite_mtp.v2",
        "engine": "rapid_mlx",
        "model": args.model,
        "runtime_model_layout": handle.model_layout,
        "rapid_source": str(args.rapid_source),
        "rapid_python": str(args.rapid_python),
        "rapid_mtp_patch": handle.compat_patch,
        "server_command": handle.command,
        "server_log": str(handle.log_path),
        "server_log_head": server_log_head,
        "server_log_tail": server_log_lines,
        "server_header_info": server_header_info,
        "lightning_source_identity": lightning_identity,
        "server_profile": "lightning_serve_preset" if lightning_mode else "rapid_mlx",
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
        "ignore_eos": bool(args.ignore_eos),
        "require_full_output_tokens": bool(args.require_full_output_tokens),
        "disable_thinking": not enable_thinking,
        "lightning_settings": {
            "source_profile": "lightning-mlx serve qwen3.6 MTPLX preset",
            "prefix_cache": "disabled" if args.disable_prefix_cache else "enabled",
            "max_num_seqs": 1,
            "prefill_batch_size": 1,
            "completion_batch_size": 1,
            "prefill_step_size": args.prefill_step_size,
            "stream_interval": 1,
            "mtp_optimistic": mtp_optimistic,
            "mtp_draft_temperature": args.mtp_draft_temperature,
            "enable_thinking": enable_thinking,
            "ngram_enabled": enable_ngram,
            "ngram_num_draft_tokens": 6 if enable_ngram else None,
            "ngram_min_occurrences": 2 if enable_ngram else None,
            "ngram_acceptance_mode": "greedy" if enable_ngram else None,
            "ngram_hybrid_verify": True if enable_ngram else None,
            "ngram_only_in_think": False if enable_ngram else None,
            "ngram_skip_tool_calls": True if enable_ngram else None,
            "ngram_self_tune": True if enable_ngram else None,
            "ngram_auto_disable_mtp_threshold": args.ngram_auto_disable_mtp_threshold,
            "ngram_auto_disable_min_ngram": args.ngram_auto_disable_min_ngram,
        },
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
    parser.add_argument(
        "--inter-case-cooldown",
        type=float,
        default=0.0,
        help="Extra sleep between prompt cases (seconds). Prevents GPU thermal throttling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Patch the benchmark server to ignore tokenizer EOS for fixed-token throughput runs.",
    )
    parser.add_argument(
        "--require-full-output-tokens",
        action="store_true",
        help="Reject measured runs that return fewer than the requested output tokens.",
    )
    parser.add_argument(
        "--lightning-mode",
        action="store_true",
        help=(
            "Run lightning-mlx directly via its vllm_mlx.cli (source at --rapid-source). "
            "Skips Rapid-MLX-only flags (--no-telemetry, --force-spec-decode, --no-mllm) "
            "and fetches per-run MTP accept rate from /v1/requests."
        ),
    )
    parser.add_argument(
        "--enable-ngram",
        action="store_true",
        default=False,
        help=(
            "Layer n-gram (prompt-lookup) speculative decoding before MTP in lightning mode. "
            "Uses the documented production preset: K=6, min_occ=2, greedy acceptance, "
            "hybrid MTP tail (--ngram-hybrid-verify), everywhere (not only in <think>), "
            "skip tool calls, self-tune on, and preset auto-disable thresholds. "
            "Only effective with --lightning-mode; ignored for Rapid-MLX."
        ),
    )
    parser.add_argument(
        "--mtp-optimistic",
        action="store_true",
        default=False,
        help=(
            "Pass --mtp-optimistic to lightning-mlx serve in --lightning-mode. "
            "Only effective with --lightning-mode; ignored for Rapid-MLX."
        ),
    )
    parser.add_argument(
        "--mtp-draft-temperature",
        type=float,
        default=None,
        help=(
            "Pass --mtp-draft-temperature to lightning-mlx serve in --lightning-mode. "
            "Only effective with --lightning-mode; ignored for Rapid-MLX."
        ),
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        default=False,
        help=(
            "Pass --disable-prefix-cache to lightning-mlx serve in --lightning-mode. "
            "Only effective with --lightning-mode; ignored for Rapid-MLX."
        ),
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help=(
            "Pass --prefill-step-size to lightning-mlx serve in --lightning-mode. "
            "Only effective with --lightning-mode; ignored for Rapid-MLX."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=False,
        help=(
            "Keep thinking enabled for lightning-mlx serve and send "
            "enable_thinking=true in chat requests. Benchmark default is disabled "
            "to match AX/MTPLX fair rows."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Pass --no-thinking to lightning-mlx serve and send enable_thinking=false.",
    )
    parser.add_argument(
        "--ngram-auto-disable-mtp-threshold",
        type=float,
        default=0.85,
        help="Pass --ngram-auto-disable-mtp-threshold in lightning n-gram mode.",
    )
    parser.add_argument(
        "--ngram-auto-disable-min-ngram",
        type=float,
        default=0.50,
        help="Pass --ngram-auto-disable-min-ngram in lightning n-gram mode.",
    )
    args = parser.parse_args()

    result = run_suite(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved to {args.output}", file=sys.stderr)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
