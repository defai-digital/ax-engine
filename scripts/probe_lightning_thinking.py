#!/usr/bin/env python3
"""Probe whether ``enable_thinking=False`` actually disables thinking in the
Qwen3.6 chat template, and run a single live request through lightning-mlx to
observe what the server emits.

Two probes:

1. **Template-level**: Apply the sidecar's chat template with both
   ``enable_thinking=True`` and ``enable_thinking=False``. Diff the rendered
   prompts. If they're identical, the template ignores the parameter and the
   benchmark's ``--no-thinking`` knob has zero effect at the model input.

2. **Server-level** (optional): Reuse ``bench_rapid_mlx_prompt_suites.py`` with
   ``--limit 1 --repetitions 1 --warmup-repetitions 0 --max-tokens 200
   --disable-thinking`` to issue one request, then read the diagnostic fields
   we added (``silent_thinking_suspected``, ``reasoning_text_chars``,
   ``stream_chunk_stats``, ``server_header_info``) to render a verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIDECAR = (
    Path.home()
    / ".cache/huggingface/hub/models--ax-local--Qwen3.6-27B-MTP/snapshots/v1"
)
DEFAULT_LIGHTNING_SOURCE = REPO_ROOT / ".internal/reference/lightning-mlx"
DEFAULT_RAPID_PYTHON = Path("/opt/homebrew/var/mtplx/venv-0.3.7/bin/python")


def probe_template(sidecar: Path, prompt: str) -> dict:
    """Render the chat template with enable_thinking True vs False.

    Returns the diff verdict. Run inside the venv that has transformers
    installed (the rapid python).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(sidecar), trust_remote_code=True)
    msgs = [{"role": "user", "content": prompt}]
    try:
        with_think = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError as exc:
        return {
            "verdict": "TEMPLATE_REJECTS_ENABLE_THINKING_KWARG",
            "detail": str(exc),
            "with_think": None,
            "without_think": None,
            "diff_bytes": None,
        }
    try:
        without_think = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        return {
            "verdict": "TEMPLATE_REJECTS_ENABLE_THINKING_KWARG",
            "detail": str(exc),
            "with_think": with_think,
            "without_think": None,
            "diff_bytes": None,
        }
    diff = len(with_think) - len(without_think)
    if with_think == without_think:
        verdict = "TEMPLATE_IGNORES_ENABLE_THINKING (no diff)"
    elif "/no_think" in without_think:
        verdict = "TEMPLATE_INJECTS_/no_think_HINT"
    elif "<think>" in with_think and "<think>" not in without_think:
        verdict = "TEMPLATE_OMITS_THINK_BLOCK_WHEN_DISABLED"
    else:
        verdict = "TEMPLATE_DIFFERS (rendered prompts not identical)"
    return {
        "verdict": verdict,
        "diff_bytes": diff,
        "with_think_tail": with_think[-400:],
        "without_think_tail": without_think[-400:],
    }


def probe_server(
    *,
    model: str,
    rapid_python: Path,
    rapid_source: Path,
    lightning_source: Path,
    prompt: str,
    max_tokens: int,
    port: int,
) -> dict:
    """Send one live request through lightning-mlx with --no-thinking and
    return the diagnostic fields from the artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        suite_path = tmp_path / "probe.jsonl"
        suite_path.write_text(
            json.dumps(
                {
                    "id": "thinking_probe",
                    "category": "probe",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                }
            )
            + "\n"
        )
        output_path = tmp_path / "probe.json"
        cmd = [
            str(rapid_python),
            str(REPO_ROOT / "scripts" / "bench_rapid_mlx_prompt_suites.py"),
            "--model",
            model,
            "--suite",
            "probe",
            "--prompts",
            str(suite_path),
            "--output",
            str(output_path),
            "--rapid-source",
            str(rapid_source),
            "--lightning-source",
            str(lightning_source),
            "--rapid-mtp-patch",
            "none",
            "--lightning-mode",
            "--depth",
            "1",
            "--max-tokens",
            str(max_tokens),
            "--repetitions",
            "1",
            "--warmup-repetitions",
            "0",
            "--cooldown",
            "0",
            "--inter-case-cooldown",
            "0",
            "--port",
            str(port),
            "--mtp-draft-temperature",
            "0.5",
            "--disable-thinking",
        ]
        print(
            "[probe] starting Lightning server (this takes ~30-60s)...",
            file=sys.stderr,
            flush=True,
        )
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        artifact = json.loads(output_path.read_text())
        case = artifact["results"][0]
        run = case["runs"][0]
        return {
            "artifact_path": str(output_path),
            "server_command": artifact["server_command"],
            "server_header_info": artifact.get("server_header_info"),
            "server_log_head": artifact.get("server_log_head", [])[:20],
            "ttft_s": run.get("ttft_s"),
            "ttft_content_s": run.get("ttft_content_s"),
            "ttft_reasoning_s": run.get("ttft_reasoning_s"),
            "generated_tokens": run.get("generated_tokens"),
            "visible_text_chars": run.get("visible_text_chars"),
            "reasoning_text_chars": run.get("reasoning_text_chars"),
            "silent_thinking_suspected": run.get("silent_thinking_suspected"),
            "stream_chunk_stats": run.get("stream_chunk_stats"),
            "content_head": run.get("content_head"),
            "reasoning_head": run.get("reasoning_head"),
        }


def render_verdict(template_result: dict, server_result: dict | None) -> str:
    lines = ["=" * 78, "THINKING PROBE VERDICT", "=" * 78, ""]
    lines.append("[1/2] Chat-template level (cheap, deterministic)")
    lines.append(f"  verdict: {template_result['verdict']}")
    if template_result.get("diff_bytes") is not None:
        lines.append(f"  rendered prompt size diff (bytes): {template_result['diff_bytes']}")
    if template_result.get("with_think_tail"):
        lines.append("  --- with_think tail (last 400 chars) ---")
        lines.append("  " + template_result["with_think_tail"].replace("\n", "\n  "))
        lines.append("  --- without_think tail (last 400 chars) ---")
        lines.append("  " + (template_result.get("without_think_tail") or "").replace("\n", "\n  "))

    lines.append("")
    if server_result is None:
        lines.append("[2/2] Server-level probe SKIPPED (use --run-server-probe)")
        return "\n".join(lines)

    lines.append("[2/2] Server-level live probe")
    lines.append(f"  ttft (any) s: {server_result['ttft_s']}")
    lines.append(f"  ttft_content s: {server_result['ttft_content_s']}")
    lines.append(f"  ttft_reasoning s: {server_result['ttft_reasoning_s']}")
    lines.append(f"  generated_tokens: {server_result['generated_tokens']}")
    lines.append(f"  visible_text_chars: {server_result['visible_text_chars']}")
    lines.append(f"  reasoning_text_chars: {server_result['reasoning_text_chars']}")
    lines.append(f"  silent_thinking_suspected: {server_result['silent_thinking_suspected']}")
    lines.append(f"  stream_chunk_stats: {server_result['stream_chunk_stats']}")
    lines.append("  --- content_head ---")
    lines.append(f"  {server_result.get('content_head', '')!r}")
    lines.append("  --- reasoning_head ---")
    lines.append(f"  {server_result.get('reasoning_head', '')!r}")
    lines.append("  --- server banner (first 20 lines) ---")
    for line in server_result.get("server_log_head", []):
        lines.append(f"  {line}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=DEFAULT_SIDECAR,
        help="Sidecar dir to load tokenizer from (template probe).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path for server probe. Defaults to --sidecar.",
    )
    parser.add_argument(
        "--rapid-python",
        type=Path,
        default=DEFAULT_RAPID_PYTHON,
        help="Python interpreter for the rapid bench subprocess.",
    )
    parser.add_argument(
        "--lightning-source",
        type=Path,
        default=DEFAULT_LIGHTNING_SOURCE,
    )
    parser.add_argument(
        "--rapid-source",
        type=Path,
        default=DEFAULT_LIGHTNING_SOURCE,
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python flappy bird game with pygame. Include the pipe spawning logic.",
        help="Single user prompt to issue.",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--port", type=int, default=18999)
    parser.add_argument(
        "--run-server-probe",
        action="store_true",
        help="Spawn a real lightning-mlx server and send one request.",
    )
    args = parser.parse_args()

    template_result = probe_template(args.sidecar, args.prompt)

    server_result = None
    if args.run_server_probe:
        server_result = probe_server(
            model=str(args.model or args.sidecar),
            rapid_python=args.rapid_python,
            rapid_source=args.rapid_source,
            lightning_source=args.lightning_source,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            port=args.port,
        )

    print(render_verdict(template_result, server_result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
