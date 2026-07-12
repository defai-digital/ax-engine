#!/usr/bin/env python3
"""Fair in-process embedding benchmark for mlx-lm and ax-engine.

This script measures the model/runtime path only. It intentionally excludes
HTTP, server micro-batching, cold start, and Swift adapters. Both backends
materialize the same output contract: one contiguous CPU float32 matrix shaped
`[batch_size, hidden_size]`.

Use this when publishing README-style embedding throughput claims. Keep
`bench_embedding_models.py` for legacy smoke coverage of the older API mix.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

SHORT_QUERY_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "AX Engine achieves 318 tokens per second on Apple M4",
    "Apple Silicon on-chip memory bandwidth enables low-latency inference",
    "What is the capital of France?",
    "Hello world",
    "Machine learning models require significant computational resources",
    "Natural language processing enables computers to understand human text",
    "The transformer architecture revolutionized deep learning",
    "Embeddings capture semantic relationships between words and phrases",
    "On-device inference preserves user privacy and reduces latency",
]

OUTPUT_CONTRACT = "contiguous_cpu_f32_batch_hidden"
SCHEMA_VERSION = "ax.embedding_fair.v2"
CLAIM_GATE = {
    "schema_version": "ax.embedding_claim_gate.v1",
    "scope": "embedding_in_process_publication",
    "requires_runtime_identity": True,
    "requires_host_metadata": True,
    "requires_build_commit": True,
    "paired_delta_requires_ax_only_false": True,
    "ax_only_allowed_claims": ["ax_absolute_trend"],
    "short_query_primary_metric": "median_ms_per_item",
    "fixed_length_primary_metric": "median_tokens_per_sec",
    "scale_primary_metrics": [
        "median_tokens_per_sec",
        "median_chunks_per_sec",
        "median_batch_p95_ms",
    ],
    "minimum_warmup_repetitions": 2,
    "minimum_measurement_repetitions": 5,
    "publication_cooldown_s": 15.0,
    "note": (
        "Publish reference deltas only from same-session paired artifacts "
        "(ax_only=false). AX-only runs are absolute trend evidence only. "
        "Short-query rows headline latency (ms/item), not tok/s."
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class Workload:
    name: str
    input_kind: str
    batch_size: int
    token_ids: list[list[int]]

    @property
    def token_counts(self) -> list[int]:
        return [len(ids) for ids in self.token_ids]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)

    @property
    def max_tokens(self) -> int:
        return max(self.token_counts) if self.token_ids else 0


@dataclass(frozen=True)
class EngineRunner:
    key: str
    label: str
    step_fn: Any


def parse_csv_ints(value: str, *, name: str) -> list[int]:
    if not value.strip():
        return []
    out = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"{name} values must be positive: {parsed}")
        out.append(parsed)
    if not out:
        raise ValueError(f"{name} must contain at least one positive integer")
    return out


def parse_model_spec(value: str) -> ModelSpec:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    elif ":" in value:
        label, raw_path = value.split(":", 1)
    else:
        path = Path(value).expanduser()
        label = path.name
        raw_path = value
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label:
        raise ValueError(f"model label is empty in {value!r}")
    return ModelSpec(label=label, path=path)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def model_vocab_size(model_dir: Path) -> int:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return 151_936
    config = load_json_file(config_path)
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get("vocab_size"), int):
        return int(text_config["vocab_size"])
    if isinstance(config.get("vocab_size"), int):
        return int(config["vocab_size"])
    return 151_936


def tokenize_short_queries(model_dir: Path) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    eos_id = tokenizer.eos_token_id
    rows = []
    for sentence in SHORT_QUERY_SENTENCES:
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        if eos_id is not None:
            token_ids = token_ids + [int(eos_id)]
        rows.append(token_ids)
    return rows


def cycle_batch(rows: list[list[int]], batch_size: int) -> list[list[int]]:
    if not rows:
        raise ValueError("cannot build a batch from an empty row set")
    return [list(rows[i % len(rows)]) for i in range(batch_size)]


def synthetic_batch(length: int, batch_size: int, vocab_size: int) -> list[list[int]]:
    upper = max(vocab_size - 1, 2)
    rows = []
    for row_idx in range(batch_size):
        start = 1 + row_idx * 997
        rows.append([1 + ((start + pos) % upper) for pos in range(length)])
    return rows


def build_workloads(
    model_dir: Path,
    batch_sizes: list[int],
    fixed_lengths: list[int],
    include_short_query: bool,
) -> list[Workload]:
    workloads: list[Workload] = []
    if include_short_query:
        short_rows = tokenize_short_queries(model_dir)
        for batch_size in batch_sizes:
            workloads.append(
                Workload(
                    name=f"short_query_b{batch_size}",
                    input_kind="short_query_text",
                    batch_size=batch_size,
                    token_ids=cycle_batch(short_rows, batch_size),
                )
            )
    vocab_size = model_vocab_size(model_dir)
    for length in fixed_lengths:
        for batch_size in batch_sizes:
            workloads.append(
                Workload(
                    name=f"fixed_{length}_b{batch_size}",
                    input_kind="synthetic_token_ids",
                    batch_size=batch_size,
                    token_ids=synthetic_batch(length, batch_size, vocab_size),
                )
            )
    return workloads


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stddev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def trial_stats(trials: list[dict[str, float]], engine: str) -> dict[str, Any]:
    ms = [row["ms_per_batch"] for row in trials]
    ms_item = [row["ms_per_item"] for row in trials]
    tokens = [row["tokens_per_sec"] for row in trials]
    items = [row["items_per_sec"] for row in trials]
    return {
        "engine": engine,
        "median_ms_per_batch": median(ms),
        "mean_ms_per_batch": mean(ms),
        "stddev_ms_per_batch": stddev(ms),
        "median_ms_per_item": median(ms_item),
        "mean_ms_per_item": mean(ms_item),
        "stddev_ms_per_item": stddev(ms_item),
        "median_tokens_per_sec": median(tokens),
        "mean_tokens_per_sec": mean(tokens),
        "median_items_per_sec": median(items),
        "mean_items_per_sec": mean(items),
        "trials": trials,
    }


def benchmark_step(fn, workload: Workload) -> dict[str, float]:
    started = time.perf_counter()
    output_bytes, batch_size, hidden_size = fn(workload.token_ids)
    elapsed = time.perf_counter() - started
    expected_bytes = batch_size * hidden_size * 4
    if len(output_bytes) != expected_bytes:
        raise RuntimeError(
            f"output byte length mismatch: got {len(output_bytes)}, expected {expected_bytes}"
        )
    return {
        "ms_per_batch": elapsed * 1000.0,
        "ms_per_item": elapsed * 1000.0 / workload.batch_size,
        "tokens_per_sec": workload.total_tokens / elapsed if elapsed > 0 else 0.0,
        "items_per_sec": workload.batch_size / elapsed if elapsed > 0 else 0.0,
        "hidden_size": float(hidden_size),
        "output_bytes": float(len(output_bytes)),
    }


def run_trials(
    engine: str,
    workload: Workload,
    step_fn,
    warmup: int,
    trials: int,
    cooldown: float,
) -> dict[str, Any]:
    print(f"    [{engine}] warmup x {warmup}", file=sys.stderr)
    for _ in range(warmup):
        benchmark_step(step_fn, workload)
    rows = []
    for idx in range(1, trials + 1):
        if cooldown > 0:
            time.sleep(cooldown)
        row = benchmark_step(step_fn, workload)
        rows.append(row)
        print(
            f"    [{engine}] trial {idx}: "
            f"{row['ms_per_item']:.2f} ms/item  {row['tokens_per_sec']:.1f} tok/s",
            file=sys.stderr,
        )
    return trial_stats(rows, engine)


def run_trials_interleaved(
    engines: list[EngineRunner],
    workload: Workload,
    warmup: int,
    trials: int,
    cooldown: float,
) -> dict[str, dict[str, Any]]:
    for runner in engines:
        print(f"    [{runner.label}] warmup x {warmup}", file=sys.stderr)
        for _ in range(warmup):
            benchmark_step(runner.step_fn, workload)

    rows_by_engine: dict[str, list[dict[str, float]]] = {
        runner.key: [] for runner in engines
    }
    for idx in range(1, trials + 1):
        trial_engines = (
            engines if idx % 2 == 1 or len(engines) == 1 else list(reversed(engines))
        )
        for runner in trial_engines:
            if cooldown > 0:
                time.sleep(cooldown)
            row = benchmark_step(runner.step_fn, workload)
            rows_by_engine[runner.key].append(row)
            print(
                f"    [{runner.label}] trial {idx}: "
                f"{row['ms_per_item']:.2f} ms/item  {row['tokens_per_sec']:.1f} tok/s",
                file=sys.stderr,
            )
    return {
        runner.key: trial_stats(rows_by_engine[runner.key], runner.label)
        for runner in engines
    }


def make_mlx_lm_step(model_dir: Path):
    print(f"  [mlx-lm] loading {model_dir}", file=sys.stderr)
    from mlx_lm import load
    import mlx.core as mx
    import numpy as np
    from transformers import AutoTokenizer

    model, _ = load(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def step(batch: list[list[int]]) -> tuple[bytes, int, int]:
        max_len = max(len(ids) for ids in batch)
        padded = [ids + [int(pad_id)] * (max_len - len(ids)) for ids in batch]
        last_positions = [len(ids) - 1 for ids in batch]
        x = mx.array(padded)
        hidden = model.model(x)
        rows = []
        for row_idx, position in enumerate(last_positions):
            last = hidden[row_idx, position, :].astype(mx.float32)
            norm = mx.sqrt(mx.sum(last * last))
            rows.append(last / (norm + 1e-12))
        matrix = mx.stack(rows, 0)
        array = np.array(matrix, dtype=np.float32, copy=True)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array.tobytes(), int(array.shape[0]), int(array.shape[1])

    return step


def make_mlx_embeddings_step(model_dir: Path):
    """Reference step for EmbeddingGemma-style sentence-transformers models.

    Uses `mlx-embeddings`, whose forward already applies mean pooling, the Dense
    projection head, and L2 normalization (the full sentence-transformers
    pipeline). mlx-lm has no EmbeddingGemma embedding path, so this is the
    apples-to-apples reference for that family.
    """
    print(f"  [mlx-embeddings] loading {model_dir}", file=sys.stderr)
    import mlx.core as mx
    import numpy as np
    import mlx_embeddings

    model, _ = mlx_embeddings.load(str(model_dir))

    def step(batch: list[list[int]]) -> tuple[bytes, int, int]:
        max_len = max(len(ids) for ids in batch)
        padded = [ids + [0] * (max_len - len(ids)) for ids in batch]
        attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch]
        out = model(mx.array(padded), attention_mask=mx.array(attention_mask))
        array = np.array(out.text_embeds, dtype=np.float32, copy=True)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array.tobytes(), int(array.shape[0]), int(array.shape[1])

    return step


def make_ax_engine_step(model_dir: Path, pooling: str = "last", model_id: str = "qwen3"):
    print(f"  [ax-engine-py] loading {model_dir}", file=sys.stderr)
    sys.path.insert(0, str(REPO_ROOT / "python"))
    import ax_engine

    session = ax_engine.Session(
        model_id=model_id,
        mlx=True,
        support_tier="mlx_preview",
        mlx_model_artifacts_dir=str(model_dir),
    )

    def step(batch: list[list[int]]) -> tuple[bytes, int, int]:
        return session.embed_batch_flat_bytes(batch, pooling=pooling, normalize=True)

    return step, session


def compare_results(results: dict[str, Any], reference_key: str = "mlx_lm") -> dict[str, float]:
    ref = results.get(reference_key)
    ax = results.get("ax_engine_py")
    if not ref or not ax:
        return {}
    ref_tps = float(ref["median_tokens_per_sec"])
    ax_tps = float(ax["median_tokens_per_sec"])
    ref_items = float(ref["median_items_per_sec"])
    ax_items = float(ax["median_items_per_sec"])
    ref_ms_item = float(ref.get("median_ms_per_item") or 0.0)
    ax_ms_item = float(ax.get("median_ms_per_item") or 0.0)
    ref_ms_batch = float(ref.get("median_ms_per_batch") or 0.0)
    ax_ms_batch = float(ax.get("median_ms_per_batch") or 0.0)
    comparison = {
        "ax_vs_reference_tokens_pct": ((ax_tps - ref_tps) / ref_tps * 100.0)
        if ref_tps
        else 0.0,
        "ax_vs_reference_items_pct": ((ax_items - ref_items) / ref_items * 100.0)
        if ref_items
        else 0.0,
    }
    # Latency deltas: lower is better (negative = AX faster).
    if ref_ms_item:
        comparison["ax_vs_reference_ms_per_item_pct"] = (
            (ax_ms_item - ref_ms_item) / ref_ms_item * 100.0
        )
    if ref_ms_batch:
        comparison["ax_vs_reference_ms_per_batch_pct"] = (
            (ax_ms_batch - ref_ms_batch) / ref_ms_batch * 100.0
        )
    return comparison


def run_model(
    spec: ModelSpec,
    batch_sizes: list[int],
    fixed_lengths: list[int],
    include_short_query: bool,
    warmup: int,
    trials: int,
    cooldown: float,
    reference: str = "mlx_lm",
    pooling: str = "last",
    ax_only: bool = False,
) -> dict[str, Any]:
    model_dir = spec.path.resolve()
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"model-manifest.json not found: {manifest_path}")
    print(f"[model] {spec.label}: {model_dir}", file=sys.stderr)
    workloads = build_workloads(model_dir, batch_sizes, fixed_lengths, include_short_query)
    if not workloads:
        raise ValueError("benchmark matrix has no workloads")

    if reference == "mlx_embeddings":
        ref_step = None if ax_only else make_mlx_embeddings_step(model_dir)
        ref_label, ax_model_id = "mlx-embeddings", "embeddinggemma"
    else:
        ref_step = None if ax_only else make_mlx_lm_step(model_dir)
        ref_label, ax_model_id = "mlx-lm", "qwen3"
    ax_step, ax_session = make_ax_engine_step(model_dir, pooling=pooling, model_id=ax_model_id)
    rows = []
    try:
        for workload in workloads:
            print(
                f"  [workload] {workload.name} tokens={workload.token_counts}",
                file=sys.stderr,
            )
            engines = []
            if ref_step is not None:
                engines.append(EngineRunner(reference, ref_label, ref_step))
            engines.append(EngineRunner("ax_engine_py", "ax-engine-py", ax_step))
            results = run_trials_interleaved(
                engines, workload, warmup, trials, cooldown
            )
            rows.append(
                {
                    "workload": workload.name,
                    "input_kind": workload.input_kind,
                    "batch_size": workload.batch_size,
                    "token_counts": workload.token_counts,
                    "total_tokens": workload.total_tokens,
                    "max_tokens": workload.max_tokens,
                    "primary_metric": primary_metric_for_workload(workload.name),
                    "results": results,
                    "comparison": compare_results(results, reference),
                }
            )
    finally:
        close = getattr(ax_session, "close", None)
        if close is not None:
            close()
    gc.collect()
    return {
        "model_label": spec.label,
        "model_dir": str(model_dir),
        "rows": rows,
    }


def fmt(value: float, digits: int = 1) -> str:
    if value == 0 or math.isfinite(value):
        return f"{value:,.{digits}f}"
    return "nan"


def _command_output_lines(cmd: list[str]) -> list[str]:
    try:
        output = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _sysctl(key: str) -> str:
    lines = _command_output_lines(["sysctl", "-n", key])
    return lines[0] if lines else "unknown"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_build_metadata() -> dict[str, Any]:
    """Git commit + dirty flag so artifacts identify the measured binary tree."""
    commit = git_commit() or "unknown"
    tracked_status: list[str] = []
    try:
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        tracked_status = [line for line in status.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "commit": commit,
        "git_tracked_dirty": bool(tracked_status),
        "git_tracked_status": tracked_status[:50],
    }


def collect_host_metadata() -> dict[str, Any]:
    chip = _sysctl("machdep.cpu.brand_string")
    if chip == "unknown":
        chip = _sysctl("hw.model")
    memory_bytes_raw = _sysctl("hw.memsize")
    memory_gb: float | None
    try:
        memory_gb = round(int(memory_bytes_raw) / (1024**3), 1)
    except (TypeError, ValueError):
        memory_gb = None
    return {
        "chip": chip,
        "memory_gb": memory_gb,
        "platform": platform.system().lower(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "os_version": platform.mac_ver()[0] or platform.version(),
    }


def embed_env_flags() -> dict[str, str]:
    """Capture embedding-path and linkage-relevant env flags.

    Recorded so a reader can tell which knobs were set: an ax_only artifact
    with `AX_MLX_EMBED_FFN_COMPILE=1` is not the shipped default path, and a
    delta computed from it would not describe defaults. Also tracks MLX/dylib
    override knobs that change which libmlx a process loads.
    """
    tracked = (
        "AX_MLX_DENSE_FFN_COMPILE",
        "MLX_METALLIB",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    )
    keys = sorted(
        {
            key
            for key in os.environ
            if key.startswith("AX_MLX_EMBED")
            or key.startswith("AX_MLX_")
            or key in tracked
        }
    )
    return {key: os.environ[key] for key in keys}


def discover_ax_engine_native_path() -> Path | None:
    """Locate the ax-engine-py cdylib used by the current interpreter."""
    candidates = [
        REPO_ROOT / "python" / "ax_engine" / "_ax_engine.abi3.so",
    ]
    try:
        import ax_engine  # type: ignore

        package_dir = Path(ax_engine.__file__).resolve().parent
        candidates.insert(0, package_dir / "_ax_engine.abi3.so")
        candidates.extend(sorted(package_dir.glob("_ax_engine*.so")))
        candidates.extend(sorted(package_dir.glob("_ax_engine*.dylib")))
    except Exception:
        pass
    for path in candidates:
        if path.is_file():
            return path
    return None


def discover_python_module_native_path(module_name: str) -> Path | None:
    """Best-effort path of a native extension for a Python module (e.g. mlx.core)."""
    try:
        module = __import__(module_name, fromlist=["*"])
    except Exception:
        return None
    origin = getattr(module, "__file__", None)
    if not origin:
        return None
    path = Path(origin)
    return path if path.is_file() else None


def otool_rpaths(binary: Path) -> list[str]:
    """Return LC_RPATH entries from `otool -l` for @rpath resolution."""
    lines = _command_output_lines(["otool", "-l", str(binary)])
    rpaths: list[str] = []
    for idx, line in enumerate(lines):
        if "LC_RPATH" not in line:
            continue
        for follow in lines[idx : idx + 6]:
            stripped = follow.strip()
            if not stripped.startswith("path "):
                continue
            # `path /some/dir (offset 12)`
            remainder = stripped[len("path ") :]
            path = remainder.split(" (", 1)[0].strip()
            if path:
                rpaths.append(path)
            break
    return rpaths


def expand_loader_path(path: str, binary: Path) -> str:
    """Expand `@loader_path` relative to the Mach-O binary being inspected."""
    if path.startswith("@loader_path/"):
        return str((binary.parent / path[len("@loader_path/") :]).resolve())
    if path == "@loader_path":
        return str(binary.parent.resolve())
    return path


def resolve_macho_path(install_name: str, binary: Path, rpaths: list[str]) -> Path | None:
    """Resolve install names that may use @rpath / @loader_path."""
    if install_name.startswith("@rpath/"):
        rel = install_name[len("@rpath/") :]
        for rpath in rpaths:
            expanded = expand_loader_path(rpath, binary)
            candidate = Path(expanded) / rel
            if candidate.is_file():
                return candidate
        return None
    if install_name.startswith("@loader_path/"):
        candidate = Path(expand_loader_path(install_name, binary))
        return candidate if candidate.is_file() else None
    if install_name.startswith("@executable_path/"):
        return None
    path = Path(install_name)
    return path if path.is_file() else None


def otool_mlx_libraries(binary: Path) -> list[dict[str, Any]]:
    """Parse otool -L for libmlx / libmlxc entries and fingerprint each dylib."""
    lines = _command_output_lines(["otool", "-L", str(binary)])
    rpaths = otool_rpaths(binary)
    records: list[dict[str, Any]] = []
    for line in lines:
        if not any(token in line for token in ("libmlx", "libmlxc")):
            continue
        # otool format: "<path> (compatibility version ..., current version ...)"
        path_token = line.split("(", 1)[0].strip()
        if not path_token or path_token.endswith(":"):
            continue
        resolved = resolve_macho_path(path_token, binary, rpaths)
        classify_src = str(resolved) if resolved is not None else path_token
        record: dict[str, Any] = {
            "install_name": path_token,
            "resolved_path": str(resolved) if resolved is not None else None,
            "sha256": sha256_file(resolved) if resolved is not None else None,
            "source_class": classify_mlx_path(classify_src),
            "rpaths": rpaths,
        }
        records.append(record)
    return records


def classify_mlx_path(path: str) -> str:
    lowered = path.lower()
    if "/site-packages/" in lowered or "/.venv/" in lowered or "/venv/" in lowered:
        return "pip_or_venv"
    if "/opt/homebrew/" in lowered or "/usr/local/opt/" in lowered or "/cellar/" in lowered:
        return "homebrew"
    if "/System/" in path or path.startswith("/usr/lib"):
        return "system"
    return "other"


def collect_runtime_identity(*, ax_only: bool, reference: str) -> dict[str, Any]:
    """Record which MLX dylibs AX and the reference backend are linked against.

    A Homebrew bottle vs a pip-wheel libmlx can differ by ~3× on the same host;
    without this fingerprint, paired deltas are not trustworthy for publication.
    """
    ax_native = discover_ax_engine_native_path()
    identity: dict[str, Any] = {
        "benchmark_surface": "embedding_in_process",
        "selected_backend": "mlx",
        "route_identity": "repo_owned_mlx",
        "resolution_policy": "mlx_only",
        "ax_only": ax_only,
        "reference": None if ax_only else reference,
        "ax_engine_native": {
            "path": str(ax_native) if ax_native else None,
            "sha256": sha256_file(ax_native) if ax_native else None,
            "linked_mlx": otool_mlx_libraries(ax_native) if ax_native else [],
        },
    }
    if not ax_only:
        if reference == "mlx_embeddings":
            ref_module = "mlx_embeddings"
            ref_core = discover_python_module_native_path("mlx.core")
        else:
            ref_module = "mlx_lm"
            ref_core = discover_python_module_native_path("mlx.core")
        identity["reference_runtime"] = {
            "module": ref_module,
            "mlx_core_path": str(ref_core) if ref_core else None,
            "mlx_core_sha256": sha256_file(ref_core) if ref_core else None,
            "linked_mlx": otool_mlx_libraries(ref_core) if ref_core else [],
        }
        ax_sources = {
            str(entry.get("source_class"))
            for entry in identity["ax_engine_native"].get("linked_mlx") or []
            if isinstance(entry, dict)
        }
        ref_sources = {
            str(entry.get("source_class"))
            for entry in identity["reference_runtime"].get("linked_mlx") or []
            if isinstance(entry, dict)
        }
        if "homebrew" in ax_sources and "pip_or_venv" in ref_sources:
            print(
                "warning: AX is linked to Homebrew libmlx while the reference "
                "uses pip/venv MLX — paired deltas are not publication-safe "
                "(run check_embedding_publish_gate.py). Prefer the venv wheel.",
                file=sys.stderr,
            )
        elif "pip_or_venv" in ax_sources and "homebrew" in ref_sources:
            print(
                "warning: reference MLX is Homebrew while AX uses pip/venv "
                "libmlx — run the harness under the same .venv interpreter as "
                "mlx-lm for fair paired deltas.",
                file=sys.stderr,
            )
    return identity


def primary_metric_for_workload(workload_name: str) -> str:
    if workload_name.startswith("short_query"):
        return "median_ms_per_item"
    return "median_tokens_per_sec"


def _is_short_query(workload_name: str) -> bool:
    return str(workload_name).startswith("short_query")


def render_summary(artifact: dict[str, Any]) -> str:
    reference = artifact.get("reference", "mlx_lm")
    ref_label = "mlx-embeddings" if reference == "mlx_embeddings" else "mlx-lm"
    if artifact.get("ax_only"):
        lines = [
            "# AX-Only Embedding Benchmark",
            "",
            f"Output contract: `{artifact['output_contract']}`. "
            f"Engine: `ax-engine-py`, pooling: `{artifact.get('pooling', 'last')}`.",
            "",
            "Short-query rows headline **ms/item** (lower is better). "
            "Fixed-length rows headline tok/s.",
            "",
            "| Model | Workload | Batch | Max tokens | Primary | AX value | AX tok/s | AX items/s |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
        for model in artifact["models"]:
            for row in model["rows"]:
                ax = row["results"]["ax_engine_py"]
                if _is_short_query(row["workload"]):
                    primary = "ms/item"
                    value = fmt(float(ax.get("median_ms_per_item") or 0.0), digits=2)
                else:
                    primary = "tok/s"
                    value = fmt(float(ax["median_tokens_per_sec"]))
                lines.append(
                    f"| {model['model_label']} | {row['workload']} | {row['batch_size']} | "
                    f"{row['max_tokens']} | {primary} | {value} | "
                    f"{fmt(ax['median_tokens_per_sec'])} | "
                    f"{fmt(ax['median_items_per_sec'])} |"
                )
        lines.append("")
        return "\n".join(lines)

    lines = [
        "# Fair Embedding Benchmark",
        "",
        f"Output contract: `{artifact['output_contract']}`. "
        f"Reference: `{ref_label}`, pooling: `{artifact.get('pooling', 'last')}`.",
        "",
        "Short-query rows headline **ms/item** (lower is better; negative % = AX faster). "
        "Fixed-length rows headline tok/s (higher is better).",
        "",
        f"| Model | Workload | Batch | Max tokens | Primary | {ref_label} | AX | AX vs {ref_label} |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for model in artifact["models"]:
        for row in model["rows"]:
            ref_row = row["results"][reference]
            ax_row = row["results"]["ax_engine_py"]
            comparison = row.get("comparison") or {}
            if _is_short_query(row["workload"]):
                primary = "ms/item"
                ref_value = fmt(float(ref_row.get("median_ms_per_item") or 0.0), digits=2)
                ax_value = fmt(float(ax_row.get("median_ms_per_item") or 0.0), digits=2)
                delta = float(comparison.get("ax_vs_reference_ms_per_item_pct") or 0.0)
            else:
                primary = "tok/s"
                ref_value = fmt(float(ref_row["median_tokens_per_sec"]))
                ax_value = fmt(float(ax_row["median_tokens_per_sec"]))
                delta = float(comparison.get("ax_vs_reference_tokens_pct") or 0.0)
            lines.append(
                f"| {model['model_label']} | {row['workload']} | {row['batch_size']} | "
                f"{row['max_tokens']} | {primary} | {ref_value} | {ax_value} | {delta:+.1f}% |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model as label=/hf/snapshot/path, label:/hf/snapshot/path, or /path.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,8,32",
        help="Comma-separated batch sizes. Default: 1,8,32.",
    )
    parser.add_argument(
        "--lengths",
        default="16,64,256",
        help="Comma-separated synthetic token lengths. Use '' for none.",
    )
    parser.add_argument("--skip-short-query", action="store_true")
    parser.add_argument(
        "--reference",
        choices=["mlx_lm", "mlx_embeddings"],
        default="mlx_lm",
        help="Reference engine. mlx_lm (decoder, last-token) or mlx_embeddings "
        "(EmbeddingGemma sentence-transformers: mean pool + Dense + L2).",
    )
    parser.add_argument(
        "--pooling",
        choices=["last", "cls", "mean"],
        default="last",
        help="ax-engine pooling mode. Use 'mean' for EmbeddingGemma.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument(
        "--ax-only",
        action="store_true",
        help="Benchmark only ax-engine-py and skip the reference engine.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "benchmarks"
        / "results"
        / "embedding"
        / "embedding-fair",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        model_specs = [parse_model_spec(raw) for raw in args.model]
        batch_sizes = parse_csv_ints(args.batch_sizes, name="batch-sizes")
        fixed_lengths = parse_csv_ints(args.lengths, name="lengths") if args.lengths else []
    except ValueError as error:
        parser.error(str(error))

    run_dir = args.output_dir / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    build = collect_build_metadata()
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(),
        "git_commit": build["commit"] if build["commit"] != "unknown" else git_commit(),
        "build": build,
        "host": collect_host_metadata(),
        "runtime_identity": collect_runtime_identity(
            ax_only=args.ax_only, reference=args.reference
        ),
        "claim_gate": dict(CLAIM_GATE),
        "publication_claim": (
            "ax_absolute_trend" if args.ax_only else "paired_delta"
        ),
        "embed_env_flags": embed_env_flags(),
        "output_contract": OUTPUT_CONTRACT,
        "trial_order": "interleaved_alternating",
        "warmup": args.warmup,
        "trials": args.trials,
        "cooldown_s": args.cooldown,
        "batch_sizes": batch_sizes,
        "synthetic_lengths": fixed_lengths,
        "include_short_query": not args.skip_short_query,
        "reference": args.reference,
        "pooling": args.pooling,
        "ax_only": args.ax_only,
        "models": [],
    }

    for spec in model_specs:
        artifact["models"].append(
            run_model(
                spec,
                batch_sizes,
                fixed_lengths,
                not args.skip_short_query,
                args.warmup,
                args.trials,
                args.cooldown,
                reference=args.reference,
                pooling=args.pooling,
                ax_only=args.ax_only,
            )
        )

    artifact_path = run_dir / "embedding_fair.json"
    summary_path = run_dir / "summary.md"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    summary_path.write_text(render_summary(artifact))
    print(f"Wrote {artifact_path}", file=sys.stderr)
    print(f"Wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
