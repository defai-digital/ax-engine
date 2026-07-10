#!/usr/bin/env python3
"""Run AX-only MTP-vs-direct benchmarks for supported 6-bit MTP packages."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
DEFAULT_OUTPUT_BASE = (
    REPO_ROOT / "benchmarks" / "results" / "speculative" / "mtp-6bit"
)
DEFAULT_SUITES_DIR = REPO_ROOT / "benchmarks" / "prompts" / "mtp-suites"
README_PATH = REPO_ROOT / "README.md"

GENERATED_TOKENS = 1000
REPETITIONS = 5
WARMUP_REPETITIONS = 2
COOLDOWN_S = 15.0
INTER_CASE_COOLDOWN_S = 10.0
MTP_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
DEFAULT_SUITES = ("flappy", "long_code", "python_modules_long")
NGRAM_ZERO_KEYS = (
    "ax_ngram_accepted_tokens",
    "ax_ngram_draft_tokens",
    "ax_ngram_rejected_tokens",
    "ax_mtp_ngram_accepted_tokens",
    "ax_mtp_ngram_proposed_tokens",
    "ax_mtp_ngram_submitted_tokens",
    "ax_mtp_ngram_submitted_accepted_tokens",
    "ax_mtp_ngram_hit_steps",
    "ax_mtp_ngram_attempt_steps",
)


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    mode: str
    model_dir: Path
    mtp_depth: int
    assistant_mtp: bool = False


SUPPORTED_TARGETS = (
    Target(
        key="qwen3.6-27b-6bit",
        label="Qwen3.6 27B",
        mode="Qwen fused sidecar",
        model_dir=Path(
            "/Volumes/Ext4T/models/hub/models--ax-local--mlx-community--Qwen3.6-27B-6bit-MTP/snapshots/v1"
        ),
        mtp_depth=3,
    ),
    Target(
        key="qwen3.6-35b-a3b",
        label="Qwen3.6 35B-A3B",
        mode="Qwen fused sidecar",
        model_dir=Path(
            "/Volumes/Ext4T/models/hub/models--ax-local--mlx-community--Qwen3.6-35B-A3B-6bit-MTP/snapshots/v1"
        ),
        mtp_depth=1,
    ),
    Target(
        key="gemma-4-12b",
        label="Gemma 4 12B",
        mode="Gemma assistant-MTP",
        model_dir=Path(
            "/Volumes/Ext4T/models/hub/models--ax-local--gemma-4-12b-it-assistant-mtp/snapshots/v1"
        ),
        mtp_depth=2,
        assistant_mtp=True,
    ),
    Target(
        key="gemma-4-26b",
        label="Gemma 4 26B",
        mode="Gemma assistant-MTP",
        model_dir=Path(
            "/Volumes/Ext4T/models/hub/models--ax-local--gemma-4-26b-a4b-it-assistant-mtp/snapshots/v1"
        ),
        mtp_depth=2,
        assistant_mtp=True,
    ),
    Target(
        key="gemma-4-31b",
        label="Gemma 4 31B",
        mode="Gemma assistant-MTP",
        model_dir=Path(
            "/Volumes/Ext4T/models/hub/models--ax-local--gemma-4-31b-it-assistant-mtp/snapshots/v1"
        ),
        mtp_depth=2,
        assistant_mtp=True,
    ),
    Target(
        key="glm-4.7-flash",
        label="GLM-4.7 Flash",
        mode="GLM built-in MTP sidecar",
        model_dir=Path(
            "/Volumes/Ext4T/models/hub/models--ax-local--mlx-community--GLM-4.7-Flash-6bit-GlmMTP/snapshots/v1"
        ),
        mtp_depth=1,
    ),
)
TARGETS_BY_KEY = {target.key: target for target in SUPPORTED_TARGETS}


def existing_artifact_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        artifact = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(artifact.get("results"))


def validate_model_dir(path: Path) -> None:
    missing = []
    for name in ("config.json", "model-manifest.json"):
        if not (path / name).is_file():
            missing.append(name)
    if not any(path.glob("*.safetensors")):
        missing.append("*.safetensors")
    if missing:
        raise FileNotFoundError(f"{path} is missing {', '.join(missing)}")


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        started = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        log.write(f"\n[exit {result.returncode} after {elapsed:.1f}s]\n")
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"command failed; see {log_path}\n{tail}")


def build_server() -> None:
    cmd = ["cargo", "build", "-p", "ax-engine-server", "--release"]
    print("[build] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def bench_cmd(
    *,
    target: Target,
    suite: str,
    mode: str,
    output_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--model-dir",
        str(target.model_dir),
        "--prompt-source",
        "real",
        "--real-prompt-suite",
        str(args.suites_dir / f"{suite}.jsonl"),
        "--generation-tokens",
        str(args.generated_tokens),
        "--repetitions",
        str(args.repetitions),
        "--warmup-repetitions",
        str(args.warmup_repetitions),
        "--cooldown",
        str(args.cooldown),
        "--inter-case-cooldown",
        str(args.inter_case_cooldown),
        "--ax-sampling",
        json.dumps(MTP_SAMPLING, separators=(",", ":")),
        "--skip-mlx-lm",
        "--no-thinking",
        "--capture-output-token-ids",
        "--no-build-ax-engine",
        "--output",
        str(output_path),
    ]
    if mode == "direct":
        cmd.append("--ax-direct")
    elif target.assistant_mtp:
        cmd.extend(
            [
                "--ax-gemma4-assistant-mtp",
                "--ax-mtp-disable-ngram-stacking",
                "--ax-mtp-max-depth",
                str(target.mtp_depth),
            ]
        )
    else:
        cmd.extend(
            [
                "--ax-ngram-accel",
                "--ax-mtp-disable-ngram-stacking",
                "--ax-mtp-max-depth",
                str(target.mtp_depth),
            ]
        )
    if getattr(args, "approximate_speed_ceiling", False):
        cmd.append("--ax-mtp-approximate-optimistic")
    return cmd


def maybe_run_case(
    *,
    target: Target,
    suite: str,
    mode: str,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    if args.skip_existing and existing_artifact_ok(output_path):
        print(f"[skip] {target.key} {suite} {mode}: {output_path}", flush=True)
        return
    cmd = bench_cmd(
        target=target,
        suite=suite,
        mode=mode,
        output_path=output_path,
        args=args,
    )
    log_path = output_path.with_suffix(".log")
    print(f"[run] {target.key} {suite} {mode}", flush=True)
    print(f"      artifact: {output_path}", flush=True)
    print(f"      log:      {log_path}", flush=True)
    run_logged(cmd, log_path)


def metric_median(artifact: dict[str, Any], metric: str) -> float:
    values = [
        float(row[metric]["median"])
        for row in artifact.get("results", [])
        if row.get("prompt_case_id") is not None
        and row.get(metric, {}).get("median") is not None
    ]
    if not values:
        raise ValueError(f"artifact has no {metric} prompt-case medians")
    return float(statistics.median(values))


def telemetry_sum(artifact: dict[str, Any], key: str) -> int:
    total = 0
    for row in artifact.get("results", []):
        if row.get("prompt_case_id") is None:
            continue
        telemetry = row.get("ngram_acceleration_telemetry") or {}
        total += int(telemetry.get(key, 0) or 0)
    return total


def accept_rate_pct(artifact: dict[str, Any]) -> float:
    accepted = telemetry_sum(artifact, "ax_mtp_accepted_tokens")
    drafted = telemetry_sum(artifact, "ax_mtp_draft_tokens")
    if drafted <= 0:
        accepted = telemetry_sum(artifact, "ax_mlx_gemma4_assistant_mtp_accepted_tokens")
        drafted = telemetry_sum(artifact, "ax_mlx_gemma4_assistant_mtp_draft_tokens")
    if drafted <= 0:
        raise ValueError("MTP artifact has no draft-token telemetry")
    return accepted / drafted * 100.0


def aggregate_ngram_telemetry(artifact: dict[str, Any]) -> dict[str, int]:
    return {key: telemetry_sum(artifact, key) for key in NGRAM_ZERO_KEYS}


def load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_exact_mtp_artifact(path: Path, artifact: dict[str, Any]) -> None:
    correctness = artifact.get("mtp_correctness_summary") or {}
    if correctness.get("publication_candidate") is not True:
        raise ValueError(
            f"{path} is not an exact MTP publication candidate: "
            f"{correctness.get('ineligible_rows') or 'missing correctness summary'}"
        )


def validate_approximate_mtp_artifact(path: Path, artifact: dict[str, Any]) -> None:
    rows = [
        row
        for row in artifact.get("results", [])
        if row.get("prompt_case_id") is not None
    ]
    if not rows or any(
        (row.get("ax_mtp_correctness") or {}).get("effective_mode")
        != "approximate_optimistic"
        for row in rows
    ):
        raise ValueError(f"{path} is not an effective approximate MTP speed ceiling")
    if any(row.get("publication_candidate") is True for row in rows):
        raise ValueError(f"{path} incorrectly marks an approximate row publishable")


def validate_exact_token_equivalence(
    direct_path: Path,
    direct: dict[str, Any],
    mtp_path: Path,
    mtp: dict[str, Any],
) -> None:
    direct_rows = {
        str(row.get("prompt_case_id")): row
        for row in direct.get("results", [])
        if row.get("prompt_case_id") is not None
    }
    mtp_rows = {
        str(row.get("prompt_case_id")): row
        for row in mtp.get("results", [])
        if row.get("prompt_case_id") is not None
    }
    if not direct_rows or direct_rows.keys() != mtp_rows.keys():
        raise ValueError(
            f"direct/MTP prompt cases differ: {direct_path} vs {mtp_path}"
        )
    for case_id, direct_row in direct_rows.items():
        direct_tokens = [
            trial.get("output_token_ids") for trial in direct_row.get("trials", [])
        ]
        mtp_tokens = [
            trial.get("output_token_ids") for trial in mtp_rows[case_id].get("trials", [])
        ]
        if (
            not direct_tokens
            or not mtp_tokens
            or any(tokens != direct_tokens[0] for tokens in direct_tokens)
            or any(tokens != mtp_tokens[0] for tokens in mtp_tokens)
            or direct_tokens[0] != mtp_tokens[0]
        ):
            raise ValueError(
                f"exact MTP token-equivalence oracle failed for {case_id}: "
                f"{direct_path} vs {mtp_path}"
            )


def build_summary(
    output_dir: Path,
    args: argparse.Namespace,
    targets: tuple[Target, ...],
    suites: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for suite in suites:
            direct_path = output_dir / target.key / suite / "ax_direct.json"
            mtp_path = output_dir / target.key / suite / "ax_mtp.json"
            direct = load_artifact(direct_path)
            mtp = load_artifact(mtp_path)
            if args.approximate_speed_ceiling:
                validate_approximate_mtp_artifact(mtp_path, mtp)
            else:
                validate_exact_mtp_artifact(mtp_path, mtp)
                validate_exact_token_equivalence(
                    direct_path, direct, mtp_path, mtp
                )
            direct_decode = metric_median(direct, "decode_tok_s")
            mtp_decode = metric_median(mtp, "decode_tok_s")
            row = {
                "model_id": target.key,
                "model": target.label,
                "suite_id": suite,
                "suite": suite,
                "mode": target.mode,
                "depth": target.mtp_depth,
                "ax_direct_decode_tok_s": direct_decode,
                "ax_mtp_decode_tok_s": mtp_decode,
                "ax_mtp_speedup_x": mtp_decode / direct_decode,
                "ax_mtp_prefill_tok_s": metric_median(mtp, "prefill_tok_s"),
                "ax_mtp_ttft_ms": metric_median(mtp, "ttft_ms"),
                "ax_mtp_accept_rate_pct": accept_rate_pct(mtp),
                "ax_mtp_ngram_telemetry": aggregate_ngram_telemetry(mtp),
                "artifact": str(mtp_path.relative_to(REPO_ROOT)),
                "mtplx": "N/A",
                "lightning_mlx": "N/A",
            }
            rows.append(row)
    return {
        "schema": (
            "ax.mtp_6bit_approximate_speed_ceiling_summary.v1"
            if args.approximate_speed_ceiling
            else "ax.mtp_6bit_ax_acceleration_summary.v3"
        ),
        "run_dir": str(output_dir.relative_to(REPO_ROOT)),
        "methodology": {
            "targets": [target.key for target in targets],
            "suites": list(suites),
            "generated_tokens": args.generated_tokens,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "cooldown_s": args.cooldown,
            "inter_case_cooldown_s": args.inter_case_cooldown,
            "sampling": MTP_SAMPLING,
            "correctness_contract": (
                "explicit approximate optimistic speed ceiling; not exact and not publication eligible"
                if args.approximate_speed_ceiling
                else "exact MTP requires token-equivalence oracle; currently disabled"
            ),
            "comparison": "AX MTP decode median divided by AX direct decode median for the same model package and prompt suite.",
            "mtp_ngram": "disabled; no MTP+n-gram rows are run or promoted",
        },
        "peer_support": {
            "mtplx": {
                "value": None,
                "label": "N/A",
                "reason": "Not run: this artifact is AX Engine only and compares each prepared 6-bit download-mtp package against the same package with MTP disabled.",
            },
            "lightning_mlx": {
                "value": None,
                "label": "N/A",
                "reason": "Not run: this artifact is AX Engine only and compares each prepared 6-bit download-mtp package against the same package with MTP disabled.",
            },
        },
        "rows": rows,
    }


def fmt_tok(value: float) -> str:
    return f"{value:.1f} tok/s"


def fmt_ms(value: float) -> str:
    return f"{value:.0f} ms"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{model_id}` | `{suite_id}` | {direct} | {mtp} | {speedup:.2f}x | {prefill} | {ttft} | {accept} |".format(
                model_id=row["model_id"],
                suite_id=row["suite_id"],
                direct=fmt_tok(float(row["ax_direct_decode_tok_s"])),
                mtp=fmt_tok(float(row["ax_mtp_decode_tok_s"])),
                speedup=float(row["ax_mtp_speedup_x"]),
                prefill=fmt_tok(float(row["ax_mtp_prefill_tok_s"])),
                ttft=fmt_ms(float(row["ax_mtp_ttft_ms"])),
                accept=fmt_pct(float(row["ax_mtp_accept_rate_pct"])),
            )
        )
    return lines


def write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# 6-bit MTP AX acceleration summary",
        "",
        "This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.",
        "",
        *table_lines(summary["rows"]),
        "",
        "This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.",
        "",
        "Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))


def update_readme(readme: Path, summary: dict[str, Any]) -> None:
    text = readme.read_text()
    table = "\n".join(table_lines(summary["rows"]))
    table_start = text.index("| Target | Suite | AX direct decode | AX MTP decode |")
    table_end = text.index("\n\nMethodology:", table_start)
    text = text[:table_start] + table + text[table_end:]

    old_link_start = text.index("Pure-MTP verification:")
    old_link_end = text.index(
        "Detailed MTP notes, including the GLM-4.7 Flash smoke validation session",
        old_link_start,
    )
    run_dir = summary["run_dir"]
    new_block = (
        "Pure-MTP verification: all listed AX MTP artifacts record zero n-gram accepted,\n"
        "proposed, submitted, and hit-step telemetry. Summary artifacts:\n"
        f"[`summary.md`]({run_dir}/summary.md)\n"
        "and\n"
        f"[`summary.json`]({run_dir}/summary.json).\n"
    )
    text = text[:old_link_start] + new_block + text[old_link_end:]
    readme.write_text(text)


def parse_csv(value: str) -> tuple[str, ...]:
    entries = tuple(entry.strip() for entry in value.split(",") if entry.strip())
    if not entries:
        raise ValueError("comma-separated argument must not be empty")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE
        / f"{date.today().isoformat()}-supported-mtp-ax-only-refresh",
    )
    parser.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)
    parser.add_argument(
        "--targets",
        default=",".join(target.key for target in SUPPORTED_TARGETS),
        help="Comma-separated target keys to run.",
    )
    parser.add_argument(
        "--suites",
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated prompt-suite ids to run.",
    )
    parser.add_argument("--generated-tokens", type=int, default=GENERATED_TOKENS)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument(
        "--warmup-repetitions", type=int, default=WARMUP_REPETITIONS
    )
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S)
    parser.add_argument(
        "--inter-case-cooldown", type=float, default=INTER_CASE_COOLDOWN_S
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--approximate-speed-ceiling",
        action="store_true",
        help=(
            "Run explicit optimistic MTP as a non-publishable approximate speed ceiling. "
            "Exact MTP refreshes remain disabled until token equivalence passes."
        ),
    )
    parser.add_argument("--no-build-ax-engine", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0 or args.warmup_repetitions < 0:
        raise ValueError("repetitions must be positive and warmups must be non-negative")
    if not args.approximate_speed_ceiling:
        raise ValueError(
            "exact MTP refresh is disabled because the current verifier failed the "
            "direct token-equivalence oracle; use --approximate-speed-ceiling only "
            "for explicitly non-publishable diagnostics"
        )
    if args.update_readme:
        raise ValueError("approximate MTP speed ceilings cannot update README claims")
    args.output_dir = args.output_dir.resolve()
    target_keys = parse_csv(args.targets)
    unknown_targets = [key for key in target_keys if key not in TARGETS_BY_KEY]
    if unknown_targets:
        known = ", ".join(TARGETS_BY_KEY)
        raise ValueError(f"unknown target(s): {', '.join(unknown_targets)}; known: {known}")
    targets = tuple(TARGETS_BY_KEY[key] for key in target_keys)
    suites = parse_csv(args.suites)
    for target in targets:
        validate_model_dir(target.model_dir)
    for suite in suites:
        path = args.suites_dir / f"{suite}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.no_build_ax_engine:
        build_server()
    for target in targets:
        for suite in suites:
            suite_dir = args.output_dir / target.key / suite
            maybe_run_case(
                target=target,
                suite=suite,
                mode="direct",
                output_path=suite_dir / "ax_direct.json",
                args=args,
            )
            maybe_run_case(
                target=target,
                suite=suite,
                mode="mtp",
                output_path=suite_dir / "ax_mtp.json",
                args=args,
            )
    summary = build_summary(args.output_dir, args, targets, suites)
    write_summary_files(args.output_dir, summary)
    print(f"[summary] {args.output_dir / 'summary.json'}", flush=True)
    print(f"[summary] {args.output_dir / 'summary.md'}", flush=True)
    if args.update_readme:
        update_readme(README_PATH, summary)
        print(f"[readme] updated {README_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
