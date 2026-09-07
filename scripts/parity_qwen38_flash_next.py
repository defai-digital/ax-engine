#!/usr/bin/env python3
"""Real-pack parity harness for the repo-owned qwen4_exp runtime (Qwen3.8-Flash-Next).

Validates, on the pinned AXQ packs:

1. Greedy token-id parity: ax_engine (Python ``Session`` API) vs the mlx-vlm
   reference implementation, fed the same pre-tokenized prompt ids.
2. MTP draft-head losslessness: engine MTP-on output ids == engine MTP-off
   output ids.
3. MTP engagement telemetry from ``route.crossover_decisions`` (summed
   ``ax_mtp_draft_tokens`` must be > 0, otherwise the losslessness pass would
   be vacuous).

Produces a deterministic evidence JSON under ``benchmarks/results/``.

Process discipline:

- The orchestrator (this process) never imports ``mlx``, ``mlx_vlm``, or
  ``ax_engine``. Every GPU phase runs as a short-lived subprocess of this same
  script via the hidden ``--internal-mlx-worker`` / ``--internal-engine-worker``
  flags; worker process exit is the GPU-release guarantee.
- MTP-off engine workers run with ``AX_NO_SPEC=1`` and without the
  certification-candidate env. MTP-on engine workers run with
  ``AX_MLX_QWEN4_EXP_MTP_CERTIFICATION_CANDIDATE=1`` and without ``AX_NO_SPEC``.
  Every engine worker also receives the prefix-cache kill set. These variables
  travel through the subprocess environment only; the parent never sets them in
  its own ``os.environ``.
- The Python ``Session`` latches the stream-experts mode ``Auto`` at
  construction, so ``--stream-experts`` accepts only ``auto``; anything else is
  a usage error instead of a silent no-op.

Exit codes: 0 PASS / 1 USAGE / 2 SKIP / 3 INFRA / 4 DIVERGENCE.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "ax.qwen38_flash_next_parity.v1"
WORKER_JOB_SCHEMA = "ax.qwen38_flash_next_parity.worker_job.v1"
MLX_DUMP_SCHEMA = "ax.qwen38_flash_next_parity.mlx_dump.v1"
ENGINE_DUMP_SCHEMA = "ax.qwen38_flash_next_parity.engine_dump.v1"

DEFAULT_PACK_4BIT = Path.home() / ".cache/huggingface/packs/qwen38-flash-next-4bit"
DEFAULT_PACK_6BIT = Path.home() / ".cache/huggingface/packs/qwen38-flash-next-6bit"
PINNED_REVISIONS = {
    "4bit": "680573112360bfd3f71556082f875c907c21a6e7",
    "6bit": "d514dcebf3086068ed7968caf395083c95ebcfca",
}
PACK_REPO_IDS = {
    "4bit": "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP",
    "6bit": "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP",
}

DEFAULT_MAX_TOKENS = 32
DEFAULT_LOAD_TIMEOUT_S = 3600.0
DEFAULT_REQUEST_TIMEOUT_S = 1800.0
DEFAULT_SETTLE_S = 10.0
DEFAULT_STOP_TIMEOUT_S = 120.0
DIVERGENCE_WINDOW = 8

EXIT_PASS = 0
EXIT_USAGE = 1
EXIT_SKIP = 2
EXIT_INFRA = 3
EXIT_DIVERGENCE = 4

PINNED_MLX_VLM_COMMIT = "64f44c00b9233ad1d9039987656f509f6a5708d9"

CERT_ENV = "AX_MLX_QWEN4_EXP_MTP_CERTIFICATION_CANDIDATE"
NO_SPEC_ENV = "AX_NO_SPEC"
MTP_FORCE_ENV = "AX_MLX_MTP_FORCE_REQUESTED"
SKIP_TELEMETRY_ENV = "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY"
PREFIX_CACHE_KILL_ENV = {
    "AX_ENGINE_PREFIX_REUSE_DISABLED": "1",
    "AX_MLX_PREFIX_CACHE_MAX_BYTES": "0",
    "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "0",
    "AX_MLX_PREFIX_CACHE_DISK_DISABLED": "1",
}

EXPECTED_MODEL_TYPE = "qwen4_exp"
EXPECTED_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"

CONTRACT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "ax_expert_stream.json",
    "mtplx_runtime.json",
    "axquant_mtp_sidecar_manifest.json",
    "sha256-manifest.json",
)

MTP_DECISION_KEYS = (
    "ax_mtp_draft_tokens",
    "ax_mtp_accepted_tokens",
    "ax_mtp_decode_steps",
    "ax_mtp_direct_fallback_steps",
    "ax_mtp_requested",
    "ax_mlx_mtp_model_policy",
    "ax_mlx_qwen4_exp_mtp_certification_candidate",
    "ax_mlx_qwen4_exp_mtp_direct_fallback",
    "ax_ngram_draft_tokens",
    "ax_ngram_accepted_tokens",
)

CHAT_TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")

ALLOWED_FINISH_REASONS = ("stop", "length", "max_output_tokens")
LENGTH_FINISH_REASONS = ("length", "max_output_tokens")

# Checks that must be "PASS" (or legitimately "n/a") before the run may report
# PASS. The mlx MTP accept-rate crosscheck is informational and excluded.
GATED_CHECK_NAMES = (
    "token_parity_mlx_vs_engine_off",
    "mtp_losslessness_engine_on_vs_off",
    "mtp_engagement",
    "prompt_tokens_gate",
)

# Telemetry keys every MTP-on generation must carry for engagement to count.
REQUIRED_MTP_ON_KEYS = (
    "ax_mtp_draft_tokens",
    "ax_mtp_requested",
    "ax_mlx_mtp_model_policy",
)

# Any of these present on an MTP-off record proves the route reported policy
# state (an entirely empty decision map is indistinguishable from stripped
# telemetry and must be INFRA, not a clean baseline).
OFF_TELEMETRY_SIGNAL_KEYS = (
    "ax_mtp_draft_tokens",
    "ax_mtp_requested",
    "ax_mlx_mtp_model_policy",
    "ax_mtp_direct_fallback_steps",
    "ax_mlx_qwen4_exp_mtp_direct_fallback",
)

MODES = ("full", "engine", "mlx-dump", "engine-dump")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HarnessError(Exception):
    """Carries an exit code, a machine class name, and evidence details."""

    exit_code = EXIT_INFRA
    class_name = "INFRA"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class UsageError(HarnessError):
    exit_code = EXIT_USAGE
    class_name = "USAGE"


class SkipError(HarnessError):
    exit_code = EXIT_SKIP
    class_name = "SKIP"


class InfraError(HarnessError):
    exit_code = EXIT_INFRA
    class_name = "INFRA"


class DivergenceError(HarnessError):
    exit_code = EXIT_DIVERGENCE
    class_name = "DIVERGENCE"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackPin:
    tag: str  # "4bit" | "6bit"
    repo_id: str
    revision: str
    default_dir: Path


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    category: str
    text: str


@dataclass(frozen=True)
class DivergenceWindow:
    start: int  # inclusive, max(0, index - window)
    end: int  # exclusive, min(max_len, index + window + 1)
    left_ids: list[int]
    right_ids: list[int]
    left_text: str | None
    right_text: str | None


@dataclass(frozen=True)
class CompareResult:
    equal: bool
    left_len: int
    right_len: int
    first_divergence_index: int | None
    window: DivergenceWindow | None


@dataclass
class GenerationRecord:
    prompt_id: str
    prompt_token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    token_ids: list[int]
    finish_reason: str | None
    status: str = "finished"
    step_count: int = 0
    ttft_step: int | None = None
    wall_s: float = 0.0
    mtp: dict[str, Any] = field(default_factory=dict)
    accept_proxy: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationRecord:
        """Strict parse: missing/malformed token arrays are INFRA, never []."""
        if not isinstance(data, dict):
            raise InfraError(
                f"generation record is not an object: {type(data).__name__}",
                details={"subtype": "worker_dump"},
            )
        prompt_id = data.get("prompt_id")
        for field_name in ("prompt_id", "token_ids", "prompt_token_ids"):
            if field_name not in data:
                raise InfraError(
                    f"generation record missing {field_name!r}: {prompt_id!r}",
                    details={"subtype": "worker_dump", "prompt_id": prompt_id},
                )
        try:
            token_ids = [int(t) for t in data["token_ids"]]
            prompt_token_ids = [int(t) for t in data["prompt_token_ids"]]
            mtp_raw = data.get("mtp") or {}
            mtp = {str(k): _telemetry_number(v, key=str(k), prompt_id=str(prompt_id))
                   for k, v in mtp_raw.items()} if isinstance(mtp_raw, dict) else {}
        except (TypeError, ValueError) as exc:
            raise InfraError(
                f"malformed generation record for {prompt_id!r}: {exc}",
                details={"subtype": "worker_dump", "prompt_id": prompt_id},
            ) from exc
        try:
            return cls(
                prompt_id=str(prompt_id),
                prompt_token_ids=prompt_token_ids,
                prompt_tokens=int(data.get("prompt_tokens", 0)),
                completion_tokens=int(data.get("completion_tokens", 0)),
                token_ids=token_ids,
                finish_reason=data.get("finish_reason"),
                status=data.get("status", "finished"),
                step_count=int(data.get("step_count", 0)),
                ttft_step=data.get("ttft_step"),
                wall_s=float(data.get("wall_s", 0.0)),
                mtp=mtp,
                accept_proxy=dict(data.get("accept_proxy") or {}),
                extras=dict(data.get("extras") or {}),
            )
        except (TypeError, ValueError) as exc:
            raise InfraError(
                f"malformed generation record for {prompt_id!r}: {exc}",
                details={"subtype": "worker_dump", "prompt_id": prompt_id},
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _log_tail(log_path: Path, max_lines: int = 100) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def _telemetry_number(value: Any, *, key: str, prompt_id: str) -> int | float:
    """Numeric telemetry value; anything else is INFRA, never a silent 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InfraError(
            f"telemetry {key} on {prompt_id} is not numeric: {value!r}",
            details={"subtype": "mtp_telemetry", "prompt_id": prompt_id},
        )
    return value


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def truncate_lookahead_tokens(ids: Sequence[int], max_tokens: int) -> list[int]:
    """mlx-vlm stream_generate can yield max_tokens+1 tokens; keep the first N."""
    return list(ids)[:max_tokens]


# ---------------------------------------------------------------------------
# Pack pins and snapshot resolution
# ---------------------------------------------------------------------------


def pack_pins() -> dict[str, PackPin]:
    """Return the two pinned packs keyed by tag (``4bit``, ``6bit``)."""
    return {
        "4bit": PackPin(
            tag="4bit",
            repo_id=PACK_REPO_IDS["4bit"],
            revision=PINNED_REVISIONS["4bit"],
            default_dir=DEFAULT_PACK_4BIT,
        ),
        "6bit": PackPin(
            tag="6bit",
            repo_id=PACK_REPO_IDS["6bit"],
            revision=PINNED_REVISIONS["6bit"],
            default_dir=DEFAULT_PACK_6BIT,
        ),
    }


def default_pack_dir(tag: str) -> Path:
    return pack_pins()[tag].default_dir


def default_tag_for_mode(mode: str) -> str:
    """full/mlx-dump default to the 4-bit parity pack; engine modes to 6-bit."""
    return "6bit" if mode in ("engine", "engine-dump") else "4bit"


def infer_pack_tag(pack_dir: Path) -> str | None:
    name = Path(pack_dir).name.lower()
    if "6bit" in name:
        return "6bit"
    if "4bit" in name:
        return "4bit"
    return None


def resolve_pack_dir(args: argparse.Namespace) -> tuple[Path, str]:
    """Resolve the pack directory and tag from CLI args (no filesystem checks)."""
    if args.pack_dir is not None:
        pack_dir = Path(args.pack_dir).expanduser()
        tag = args.pack_tag or infer_pack_tag(pack_dir) or default_tag_for_mode(args.mode)
        return pack_dir, tag
    tag = args.pack_tag or default_tag_for_mode(args.mode)
    if tag == "6bit" and args.pack_dir_6bit is not None:
        return Path(args.pack_dir_6bit).expanduser(), tag
    return default_pack_dir(tag), tag


def classify_snapshot(pack_dir: Path) -> dict[str, Any]:
    """Classify a local pack directory; return the parsed config.json.

    SKIP when the directory is missing/incomplete (config, tokenizer, weights,
    or leftover ``*.aria2`` download markers). INFRA when the snapshot exists
    but is not a qwen4_exp pack.
    """
    pack_dir = Path(pack_dir)
    if not pack_dir.is_dir():
        raise SkipError(
            f"pack snapshot not found: {pack_dir}",
            details={"subtype": "pack_missing", "path": str(pack_dir)},
        )
    aria2_leftovers = sorted(pack_dir.glob("*.aria2"))
    if aria2_leftovers:
        raise SkipError(
            f"pack download incomplete: {len(aria2_leftovers)} *.aria2 leftovers in {pack_dir}",
            details={
                "subtype": "pack_incomplete",
                "path": str(pack_dir),
                "aria2": [entry.name for entry in aria2_leftovers],
            },
        )
    config_path = pack_dir / "config.json"
    if not config_path.is_file():
        raise SkipError(
            f"pack snapshot incomplete: missing config.json in {pack_dir}",
            details={"subtype": "pack_incomplete", "path": str(pack_dir)},
        )
    if not (pack_dir / "tokenizer.json").is_file():
        raise SkipError(
            f"pack snapshot incomplete: missing tokenizer.json in {pack_dir}",
            details={"subtype": "pack_incomplete", "path": str(pack_dir)},
        )
    has_weights = any(pack_dir.glob("*.safetensors"))
    has_index = (pack_dir / "model.safetensors.index.json").is_file()
    if not has_weights and not has_index:
        raise SkipError(
            f"pack snapshot incomplete: no weights in {pack_dir}",
            details={"subtype": "pack_incomplete", "path": str(pack_dir)},
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InfraError(
            f"pack config.json does not parse: {exc}",
            details={"subtype": "pack_config", "path": str(config_path)},
        ) from exc
    architectures = config.get("architectures") or []
    if (
        config.get("model_type") != EXPECTED_MODEL_TYPE
        or EXPECTED_ARCHITECTURE not in architectures
    ):
        raise InfraError(
            f"pack is not {EXPECTED_MODEL_TYPE}: model_type={config.get('model_type')!r} "
            f"architectures={architectures!r}",
            details={
                "subtype": "pack_model_type",
                "path": str(pack_dir),
                "model_type": config.get("model_type"),
                "architectures": architectures,
            },
        )
    return config


def verify_sha256_manifest(
    pack_dir: Path,
    tag: str,
    *,
    hash_weights: bool = False,
) -> dict[str, Any]:
    """Verify ``sha256-manifest.json`` against the pin and on-disk files.

    A missing manifest is INFRA: the directory exists but its provenance
    cannot be verified, and evidence must never read as if a pinned revision
    was checked. A revision that contradicts the pack pin is INFRA. Entries
    with a null sha256 are size-checked; entries with a real sha256 are
    hashed, except ``*.safetensors`` weight shards which are size-checked
    unless ``hash_weights`` is set. Mismatches are INFRA.
    """
    pack_dir = Path(pack_dir)
    manifest_path = pack_dir / "sha256-manifest.json"
    if not manifest_path.is_file():
        raise InfraError(
            f"sha256-manifest.json missing in {pack_dir}: pack provenance cannot "
            "be verified",
            details={"subtype": "provenance", "path": str(manifest_path)},
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InfraError(
            f"sha256-manifest.json does not parse: {exc}",
            details={"subtype": "provenance", "path": str(manifest_path)},
        ) from exc
    revision = manifest.get("revision")
    pinned = PINNED_REVISIONS.get(tag)
    if pinned is not None and revision != pinned:
        raise InfraError(
            f"manifest revision {revision!r} does not match the pinned {tag} revision {pinned!r}",
            details={
                "subtype": "provenance",
                "tag": tag,
                "manifest_revision": revision,
                "pinned_revision": pinned,
            },
        )
    entries = manifest.get("files") or []
    complete = True
    sha256_checked = 0
    size_checked = 0
    for entry in entries:
        name = entry.get("file")
        if not name:
            raise InfraError(
                "sha256-manifest.json entry without a file name",
                details={"subtype": "provenance", "entry": entry},
            )
        target = pack_dir / name
        if not target.is_file():
            raise InfraError(
                f"manifest-listed file is missing: {name}",
                details={"subtype": "provenance", "file": name},
            )
        expected_size = entry.get("size")
        expected_sha = entry.get("sha256")
        is_weight_shard = name.endswith(".safetensors")
        if expected_sha is not None and (hash_weights or not is_weight_shard):
            actual_sha = _sha256_file(target)
            if actual_sha != expected_sha:
                raise InfraError(
                    f"sha256 mismatch on {name}",
                    details={"subtype": "provenance", "file": name},
                )
            sha256_checked += 1
        else:
            if expected_sha is None:
                complete = False
            if expected_size is not None and target.stat().st_size != expected_size:
                raise InfraError(
                    f"size mismatch on {name}: manifest {expected_size}, "
                    f"on-disk {target.stat().st_size}",
                    details={"subtype": "provenance", "file": name},
                )
            size_checked += 1
    return {
        "present": True,
        "complete": complete,
        "repo": manifest.get("repo"),
        "revision": revision,
        "files": len(entries),
        "sha256_checked": sha256_checked,
        "size_checked": size_checked,
    }


def hash_contract_files(pack_dir: Path, *, hash_weights: bool = False) -> dict[str, Any]:
    """sha256 the small contract JSONs; inventory ``*.safetensors`` by name/size."""
    pack_dir = Path(pack_dir)
    contract: dict[str, str] = {}
    for name in CONTRACT_FILES:
        candidate = pack_dir / name
        if candidate.is_file():
            contract[name] = _sha256_file(candidate)
    inventory: list[dict[str, Any]] = []
    for shard in sorted(pack_dir.glob("*.safetensors")):
        item: dict[str, Any] = {"name": shard.name, "size_bytes": shard.stat().st_size}
        if hash_weights:
            item["sha256"] = _sha256_file(shard)
        inventory.append(item)
    return {
        "contract_sha256": contract,
        "safetensors_inventory": inventory,
        "weights_hashed": bool(hash_weights),
    }


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def _sysctl_value(key: str) -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def collect_hardware() -> dict[str, Any]:
    mac_version = platform.mac_ver()[0]
    return {
        "hw.model": _sysctl_value("hw.model"),
        "hw.memsize": _sysctl_value("hw.memsize"),
        "machdep.cpu.brand_string": _sysctl_value("machdep.cpu.brand_string"),
        "platform": f"macOS-{mac_version}-{platform.machine()}" if mac_version else "unknown",
    }


def collect_git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else "unknown"


def collect_engine_version(repo_root: Path) -> str:
    cargo_toml = repo_root / "Cargo.toml"
    try:
        text = cargo_toml.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    match = re.search(r"\[workspace\.package\][^\[]*?^version = \"([^\"]+)\"", text, re.MULTILINE)
    return match.group(1) if match else "unknown"


# ---------------------------------------------------------------------------
# Embedded prompt set
# ---------------------------------------------------------------------------


def default_prompts() -> tuple[PromptSpec, ...]:
    """Return the embedded 7-prompt set. Tests assert identity and coverage."""
    return PROMPTS


# p04 drops the final sentence of the design-doc text so the character length
# lands inside the required [400, 800] band (the full text is 815 chars).
PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec(
        "p01_short_factual_en",
        "short_factual_en",
        "What is the capital of France? Answer with the city name only.",
    ),
    PromptSpec(
        "p02_code_generation",
        "code_generation",
        "Write a Python function named fibonacci that returns the n-th "
        "Fibonacci number using iteration, not recursion. Return only the "
        "function.",
    ),
    PromptSpec(
        "p03_cjk",
        "cjk",
        "用繁體中文解釋什麼是注意力機制，限制在三句以內。",
    ),
    PromptSpec(
        "p04_long_instruction",
        "long_instruction",
        "Follow these instructions exactly. You are preparing a short operator "
        "note for a local Apple Silicon inference runtime. Write three labeled "
        "sections in this order: Scope, Constraints, Checklist. Scope must "
        "state that the note covers text-only greedy decoding, not vision, not "
        "chat templates, and not multi-client serving. Constraints must include "
        "temperature 0, a 32-token generation budget, one request at a time, "
        "and no n-gram speculation. Checklist must contain exactly four numbered "
        "items: confirm the pack revision, confirm tokenizer.json exists, "
        "confirm GET /health returns status ok, and confirm the previous GPU "
        "process has exited before starting the next one. Use plain ASCII "
        "punctuation only. Do not invent extra sections. Do not wrap the answer "
        "in markdown fences.",
    ),
    PromptSpec(
        "p05_punct_numbers",
        "punct_numbers",
        "Compute 17*23+45-8/2. Then list the first eight primes as "
        "2, 3, 5, 7, 11, 13, 17, 19. End with the ISO date 2026-09-07 and "
        "the ratio 22/7 ≈ 3.142857.",
    ),
    PromptSpec(
        "p06_mixed_en_cjk",
        "mixed_en_cjk",
        "Translate the following to Traditional Chinese and keep the product "
        "name in English: AX Engine is a Mac-first LLM inference runtime for "
        "Apple Silicon.",
    ),
    PromptSpec(
        "p07_json_adjacent",
        "json_adjacent",
        'Return a JSON object with keys "status", "count", and "items". '
        'status must be "ok", count must be 2, and items must be '
        '["alpha", "beta"]. No markdown fences.',
    ),
)


def validate_prompt_specs(prompts: Sequence[PromptSpec]) -> None:
    """Reject empty/whitespace-padded texts, chat-template markers, duplicate ids."""
    seen: set[str] = set()
    for spec in prompts:
        if not spec.prompt_id or spec.prompt_id in seen:
            raise UsageError(
                f"duplicate or empty prompt id: {spec.prompt_id!r}",
                details={"subtype": "prompt_set"},
            )
        seen.add(spec.prompt_id)
        if not spec.category:
            raise UsageError(
                f"empty category for {spec.prompt_id}",
                details={"subtype": "prompt_set"},
            )
        if not spec.text or spec.text != spec.text.strip():
            raise UsageError(
                f"prompt {spec.prompt_id} text is empty or has surrounding whitespace",
                details={"subtype": "prompt_set"},
            )
        for marker in CHAT_TEMPLATE_MARKERS:
            if marker in spec.text:
                raise UsageError(
                    f"prompt {spec.prompt_id} contains chat-template marker {marker!r}",
                    details={"subtype": "prompt_set", "prompt_id": spec.prompt_id},
                )


def load_prompts(args: argparse.Namespace) -> tuple[PromptSpec, ...]:
    """Embedded set plus optional --prompts-json overrides, filtered by --prompt-ids."""
    prompts: dict[str, PromptSpec] = {spec.prompt_id: spec for spec in default_prompts()}
    order: list[str] = [spec.prompt_id for spec in default_prompts()]
    if args.prompts_json is not None:
        try:
            raw = _read_json(Path(args.prompts_json))
        except (OSError, json.JSONDecodeError) as exc:
            raise UsageError(f"--prompts-json does not parse: {exc}") from exc
        if not isinstance(raw, list):
            raise UsageError("--prompts-json must be a JSON array")
        for item in raw:
            if not isinstance(item, dict):
                raise UsageError("--prompts-json entries must be objects")
            prompt_id = item.get("prompt_id") or item.get("id")
            category = item.get("category")
            text = item.get("text")
            if not prompt_id or not category or not isinstance(text, str):
                raise UsageError(
                    "--prompts-json entries need id/prompt_id, category, and text",
                    details={"subtype": "prompt_set", "entry": item},
                )
            spec = PromptSpec(str(prompt_id), str(category), text)
            validate_prompt_specs([spec])
            if spec.prompt_id not in prompts:
                order.append(spec.prompt_id)
            prompts[spec.prompt_id] = spec
    selected = [prompts[prompt_id] for prompt_id in order]
    if args.prompt_ids:
        wanted = [piece.strip() for piece in args.prompt_ids.split(",") if piece.strip()]
        unknown = [prompt_id for prompt_id in wanted if prompt_id not in prompts]
        if unknown:
            raise UsageError(
                f"unknown --prompt-ids: {', '.join(unknown)} "
                f"(valid: {', '.join(order)})",
                details={"subtype": "prompt_ids", "unknown": unknown},
            )
        selected = [prompts[prompt_id] for prompt_id in wanted]
    validate_prompt_specs(selected)
    if not selected:
        raise UsageError("empty prompt selection", details={"subtype": "prompt_ids"})
    return tuple(selected)


# ---------------------------------------------------------------------------
# Tokenizer (lazy transformers import; orchestrator stays CPU-only)
# ---------------------------------------------------------------------------


def load_pack_tokenizer(pack_dir: Path) -> Any:
    """Load ``transformers.AutoTokenizer`` from the pack directory (lazy import)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise InfraError(
            f"transformers import failed after pack {pack_dir} was classified present: {exc}",
            details={"subtype": "tokenizer_import"},
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(str(pack_dir), local_files_only=True)
    except Exception as exc:  # tokenizer load failures are infra, not divergence
        raise InfraError(
            f"AutoTokenizer load failed for {pack_dir}: {type(exc).__name__}: {exc}",
            details={"subtype": "tokenizer_load"},
        ) from exc


def encode_prompts(tokenizer: Any, prompts: Sequence[PromptSpec]) -> list[dict[str, Any]]:
    """Encode each prompt once with add_special_tokens=False; empty ids are INFRA."""
    encoded: list[dict[str, Any]] = []
    for spec in prompts:
        ids = [int(token) for token in tokenizer.encode(spec.text, add_special_tokens=False)]
        if not ids:
            raise InfraError(
                f"empty encoding for {spec.prompt_id}",
                details={"subtype": "tokenizer", "prompt_id": spec.prompt_id},
            )
        encoded.append(
            {
                "prompt_id": spec.prompt_id,
                "category": spec.category,
                "text": spec.text,
                "input_ids": ids,
                "n_chars": len(spec.text),
                "n_prompt_tokens": len(ids),
                "text_sha256": hashlib.sha256(spec.text.encode("utf-8")).hexdigest(),
            }
        )
    return encoded


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_token_ids(
    left: Sequence[int],
    right: Sequence[int],
    *,
    window: int = DIVERGENCE_WINDOW,
    decode_fn: Callable[[list[int]], str] | None = None,
) -> CompareResult:
    """Equal iff sequences are identical (same length AND every index equal).

    A length mismatch is a divergence at index min(len(left), len(right)).
    The window covers [max(0, index - window), min(max_len, index + window + 1))
    with missing positions omitted, not padded. PASS/FAIL is sequence equality
    only; finish_reason is compared by the caller as a warning.
    """
    left_ids = list(left)
    right_ids = list(right)
    max_len = max(len(left_ids), len(right_ids))
    index: int | None = None
    for i in range(max_len):
        if i >= len(left_ids) or i >= len(right_ids) or left_ids[i] != right_ids[i]:
            index = i
            break
    if index is None:
        return CompareResult(
            equal=True,
            left_len=len(left_ids),
            right_len=len(right_ids),
            first_divergence_index=None,
            window=None,
        )
    start = max(0, index - window)
    end = min(max_len, index + window + 1)
    left_slice = left_ids[start:end]
    right_slice = right_ids[start:end]
    divergence_window = DivergenceWindow(
        start=start,
        end=end,
        left_ids=left_slice,
        right_ids=right_slice,
        left_text=decode_fn(left_slice) if decode_fn is not None else None,
        right_text=decode_fn(right_slice) if decode_fn is not None else None,
    )
    return CompareResult(
        equal=False,
        left_len=len(left_ids),
        right_len=len(right_ids),
        first_divergence_index=index,
        window=divergence_window,
    )


def normalize_finish_reason_bucket(reason: str | None) -> str | None:
    """Bucket engine ``max_output_tokens`` and mlx-vlm ``length`` together."""
    if reason == "stop":
        return "stop"
    if reason in LENGTH_FINISH_REASONS:
        return "length"
    return reason


def classify_finish_reason(reason: str | None, *, context: str = "") -> str:
    """Allow stop/length/max_output_tokens; anything else is INFRA."""
    if reason not in ALLOWED_FINISH_REASONS:
        raise InfraError(
            f"unexpected finish_reason {reason!r}{context}",
            details={"subtype": "finish_reason", "finish_reason": reason},
        )
    return reason


def compare_generations(
    left: Sequence[GenerationRecord],
    right: Sequence[GenerationRecord],
    *,
    compare: str,
    left_name: str,
    right_name: str,
    decode_fn: Callable[[list[int]], str] | None = None,
) -> dict[str, Any]:
    """Compare two generation sets prompt by prompt in harness (left) order.

    Token-id equality is mandatory; a finish_reason bucket mismatch with equal
    ids is a warning, never a failure. All prompts are compared before raising
    so the evidence carries per-prompt results for the full set.
    """
    right_by_id = {record.prompt_id: record for record in right}
    if not left:
        raise InfraError(
            f"{left_name} produced no generations; refusing to compare an empty set",
            details={"subtype": "compare_set", "left": left_name, "right": right_name},
        )
    if len(right_by_id) != len(left):
        raise InfraError(
            f"prompt set mismatch: {left_name} has {len(left)} generations, "
            f"{right_name} has {len(right_by_id)}",
            details={"subtype": "compare_set", "left": left_name, "right": right_name},
        )
    per_prompt: list[dict[str, Any]] = []
    failures: list[tuple[str, CompareResult]] = []
    warnings: list[str] = []
    for left_record in left:
        right_record = right_by_id.get(left_record.prompt_id)
        if right_record is None:
            raise InfraError(
                f"{right_name} is missing prompt {left_record.prompt_id}",
                details={"subtype": "compare_set", "prompt_id": left_record.prompt_id},
            )
        result = compare_token_ids(
            left_record.token_ids,
            right_record.token_ids,
            decode_fn=decode_fn,
        )
        finish_warning = bool(
            result.equal
            and normalize_finish_reason_bucket(left_record.finish_reason)
            != normalize_finish_reason_bucket(right_record.finish_reason)
        )
        if finish_warning:
            warnings.append(
                f"{left_record.prompt_id}: finish_reason differs with equal ids "
                f"({left_name}={left_record.finish_reason}, "
                f"{right_name}={right_record.finish_reason})"
            )
        per_prompt.append(
            {
                "prompt_id": left_record.prompt_id,
                "equal": result.equal,
                "left_len": result.left_len,
                "right_len": result.right_len,
                "first_divergence_index": result.first_divergence_index,
                "window": dataclasses.asdict(result.window) if result.window else None,
                "finish_reason_left": left_record.finish_reason,
                "finish_reason_right": right_record.finish_reason,
                "finish_reason_warning": finish_warning,
            }
        )
        if not result.equal:
            failures.append((left_record.prompt_id, result))
    report = {
        "equal": not failures,
        "left": left_name,
        "right": right_name,
        "per_prompt": per_prompt,
    }
    if warnings:
        report["warnings"] = warnings
    if failures:
        prompt_id, result = failures[0]
        raise DivergenceError(
            f"{left_name} vs {right_name} diverged at {prompt_id} "
            f"index {result.first_divergence_index}",
            details={
                "subtype": "token_ids",
                "compare": compare,
                "prompt_id": prompt_id,
                "first_divergence_index": result.first_divergence_index,
                "window": dataclasses.asdict(result.window) if result.window else None,
                "report": report,
            },
        )
    return report


# ---------------------------------------------------------------------------
# Gates and telemetry
# ---------------------------------------------------------------------------


def assert_prompt_tokens_identity(
    prompt_token_ids: Sequence[int],
    shared_ids: Sequence[int],
    *,
    prompt_id: str,
) -> None:
    """First checkpoint: engine prompt ids must equal the shared ids exactly."""
    if list(prompt_token_ids) != list(shared_ids):
        raise InfraError(
            f"prompt_tokens mismatch on {prompt_id}: reported sequence is not the "
            f"shared ids (reported[:8]={list(prompt_token_ids)[:8]}, "
            f"shared[:8]={list(shared_ids)[:8]}, len {len(prompt_token_ids)} vs "
            f"{len(shared_ids)})",
            details={
                "subtype": "tokenizer",
                "prompt_id": prompt_id,
                "reported_len": len(prompt_token_ids),
                "shared_len": len(shared_ids),
                "reported_prefix": list(prompt_token_ids)[:8],
                "shared_prefix": list(shared_ids)[:8],
            },
        )


def verify_prompt_token_gate(
    records: Sequence[GenerationRecord],
    encoded_prompts: Sequence[dict[str, Any]],
) -> None:
    """Parent-side re-verification of the first checkpoint.

    The engine worker already asserts the echo equals the shared ids; the
    parent re-checks from the dump so a truncated or hand-edited dump cannot
    leave a hard-coded PASS. The reported prompt_tokens count must match too.
    """
    shared_by_id = {prompt["prompt_id"]: list(prompt["input_ids"]) for prompt in encoded_prompts}
    for record in records:
        shared = shared_by_id.get(record.prompt_id)
        if shared is None:
            raise InfraError(
                f"worker dump contains unknown prompt_id {record.prompt_id!r}",
                details={"subtype": "worker_dump", "prompt_id": record.prompt_id},
            )
        assert_prompt_tokens_identity(
            record.prompt_token_ids,
            shared,
            prompt_id=record.prompt_id,
        )
        if record.prompt_tokens != len(shared):
            raise InfraError(
                f"prompt_tokens count mismatch on {record.prompt_id}: reported "
                f"{record.prompt_tokens}, expected {len(shared)}",
                details={
                    "subtype": "tokenizer",
                    "prompt_id": record.prompt_id,
                    "reported_len": record.prompt_tokens,
                    "shared_len": len(shared),
                },
            )


def compute_accept_proxy(step_count: int, output_len: int) -> dict[str, Any]:
    """Informational decode-step proxy from step_count (one prefill assumed)."""
    decode_steps_proxy = max(int(step_count) - 1, 0)
    tokens_per_decode_step = (
        round(output_len / decode_steps_proxy, 4) if decode_steps_proxy > 0 else None
    )
    return {
        "decode_steps_proxy": decode_steps_proxy,
        "tokens_per_decode_step": tokens_per_decode_step,
    }


def extract_mtp_decisions(crossover_decisions: dict[str, Any] | None) -> dict[str, Any]:
    """Keep the known MTP/n-gram decision keys from route.crossover_decisions."""
    decisions = crossover_decisions or {}
    return {key: decisions[key] for key in MTP_DECISION_KEYS if key in decisions}


def _telemetry_int(value: Any, *, key: str, prompt_id: str) -> int:
    """Optional telemetry counter as int; non-numeric is INFRA."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    raise InfraError(
        f"telemetry {key} on {prompt_id} is not numeric: {value!r}",
        details={"subtype": "mtp_telemetry", "prompt_id": prompt_id},
    )


def require_mtp_engaged(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    """Hard gate: summed ax_mtp_draft_tokens must be > 0, else INFRA.

    Every MTP-on generation must carry the required telemetry keys; a single
    generation missing them means telemetry was stripped or the policy did not
    engage for that prompt, and a losslessness pass would be vacuous.
    """
    if not records:
        raise InfraError(
            "mtp engagement gate called with no generations",
            details={"subtype": "mtp_engagement"},
        )
    draft = 0
    accepted = 0
    decode_steps = 0
    fallback_steps = 0
    requested: Any = None
    policy: Any = None
    for record in records:
        mtp = record.mtp or {}
        missing = [key for key in REQUIRED_MTP_ON_KEYS if key not in mtp]
        if missing:
            raise InfraError(
                f"mtp telemetry incomplete on {record.prompt_id}: missing "
                f"{', '.join(missing)}",
                details={
                    "subtype": "mtp_telemetry",
                    "prompt_id": record.prompt_id,
                    "missing": missing,
                },
            )
        draft += _telemetry_int(
            mtp.get("ax_mtp_draft_tokens"), key="ax_mtp_draft_tokens", prompt_id=record.prompt_id
        )
        accepted += _telemetry_int(
            mtp.get("ax_mtp_accepted_tokens"),
            key="ax_mtp_accepted_tokens",
            prompt_id=record.prompt_id,
        )
        decode_steps += _telemetry_int(
            mtp.get("ax_mtp_decode_steps"), key="ax_mtp_decode_steps", prompt_id=record.prompt_id
        )
        fallback_steps += _telemetry_int(
            mtp.get("ax_mtp_direct_fallback_steps"),
            key="ax_mtp_direct_fallback_steps",
            prompt_id=record.prompt_id,
        )
        if requested is None:
            requested = mtp["ax_mtp_requested"]
        if policy is None:
            policy = mtp["ax_mlx_mtp_model_policy"]
    if draft <= 0:
        raise InfraError(
            "mtp policy not engaged: summed ax_mtp_draft_tokens is 0",
            details={"subtype": "mtp_engagement", "ax_mtp_draft_tokens": draft},
        )
    return {
        "ax_mtp_draft_tokens": draft,
        "ax_mtp_accepted_tokens": accepted,
        "ax_mtp_decode_steps": decode_steps,
        "ax_mtp_direct_fallback_steps": fallback_steps,
        "accept_rate": round(accepted / draft, 4) if draft > 0 else None,
        "ax_mtp_requested": requested,
        "ax_mlx_mtp_model_policy": policy,
    }


def require_mtp_off_baseline_clean(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    """Hard gate on the MTP-off baseline: zero drafts, telemetry present.

    The off lifetime is the reference for both mlx parity and MTP
    losslessness. If it secretly drafted (env leak, policy change), the on/off
    comparison is MTP vs MTP and a lossy MTP reads as lossless. Any MTP or
    n-gram draft token, or a truthy ax_mtp_requested, is INFRA. A run with no
    MTP route keys at all is indistinguishable from stripped telemetry and is
    also INFRA.
    """
    if not records:
        raise InfraError(
            "mtp-off baseline gate called with no generations",
            details={"subtype": "mtp_off_contaminated"},
        )
    mtp_draft = 0
    ngram_draft = 0
    signal_seen = False
    for record in records:
        mtp = record.mtp or {}
        if any(key in mtp for key in OFF_TELEMETRY_SIGNAL_KEYS):
            signal_seen = True
        mtp_draft += _telemetry_int(
            mtp.get("ax_mtp_draft_tokens"), key="ax_mtp_draft_tokens", prompt_id=record.prompt_id
        )
        ngram_draft += _telemetry_int(
            mtp.get("ax_ngram_draft_tokens"),
            key="ax_ngram_draft_tokens",
            prompt_id=record.prompt_id,
        )
        requested = _telemetry_int(
            mtp.get("ax_mtp_requested"), key="ax_mtp_requested", prompt_id=record.prompt_id
        )
        if requested:
            raise InfraError(
                f"mtp-off baseline contaminated: ax_mtp_requested is truthy on "
                f"{record.prompt_id}",
                details={"subtype": "mtp_off_contaminated", "prompt_id": record.prompt_id},
            )
    if not signal_seen:
        raise InfraError(
            "mtp-off baseline produced no MTP route telemetry; cannot prove a "
            "clean baseline",
            details={"subtype": "mtp_telemetry"},
        )
    if mtp_draft > 0 or ngram_draft > 0:
        raise InfraError(
            f"mtp-off baseline contaminated: ax_mtp_draft_tokens sum "
            f"{mtp_draft}, ax_ngram_draft_tokens sum {ngram_draft}",
            details={
                "subtype": "mtp_off_contaminated",
                "ax_mtp_draft_tokens": mtp_draft,
                "ax_ngram_draft_tokens": ngram_draft,
            },
        )
    return {
        "ax_mtp_draft_tokens": mtp_draft,
        "ax_ngram_draft_tokens": ngram_draft,
        "baseline_clean": True,
    }


def sum_ngram_accepted(records: Sequence[GenerationRecord]) -> int:
    total = 0
    for record in records:
        total += int((record.mtp or {}).get("ax_ngram_accepted_tokens", 0) or 0)
    return total


# ---------------------------------------------------------------------------
# Engine worker environment and Session/generate kwargs
# ---------------------------------------------------------------------------


def build_engine_worker_env(
    mtp_on: bool,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Subprocess env for one engine worker polarity.

    MTP-off: ``AX_NO_SPEC=1``, no certification-candidate env. MTP-on:
    certification-candidate env set, ``AX_NO_SPEC`` removed (it would flip
    ``set_mtp_requested`` to false). Both polarities get the prefix-cache kill
    set, drop the wrong-family ``AX_MLX_MTP_FORCE_REQUESTED`` bypass, and pin
    ``AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY=0`` so a contaminated parent env
    cannot strip the route telemetry this harness gates on. The parent's own
    environment is never mutated.
    """
    env = dict(os.environ if base_env is None else base_env)
    env.update(PREFIX_CACHE_KILL_ENV)
    env.pop(MTP_FORCE_ENV, None)
    env[SKIP_TELEMETRY_ENV] = "0"
    if mtp_on:
        env.pop(NO_SPEC_ENV, None)
        env[CERT_ENV] = "1"
    else:
        env.pop(CERT_ENV, None)
        env[NO_SPEC_ENV] = "1"
    return env


def build_mlx_worker_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Subprocess env for the mlx-vlm worker.

    mlx-vlm does not read the engine spec vars, but keep the same discipline:
    drop all three and pin decode route telemetry on.
    """
    env = dict(os.environ if base_env is None else base_env)
    env.pop(CERT_ENV, None)
    env.pop(NO_SPEC_ENV, None)
    env.pop(MTP_FORCE_ENV, None)
    env[SKIP_TELEMETRY_ENV] = "0"
    return env


def build_session_kwargs(pack_dir: Path) -> dict[str, Any]:
    """Session constructor kwargs; n-gram is controlled via the worker env."""
    return {
        "mlx": True,
        "mlx_model_artifacts_dir": str(pack_dir),
        "model_id": EXPECTED_MODEL_TYPE,
        "deterministic": True,
    }


def build_generate_kwargs(shared_ids: Sequence[int], max_tokens: int) -> dict[str, Any]:
    """Greedy deterministic generate kwargs. Never carries input_text."""
    return {
        "input_tokens": list(shared_ids),
        "max_output_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "seed": 0,
        "deterministic": True,
    }


# ---------------------------------------------------------------------------
# Worker job/dump IO and subprocess machinery
# ---------------------------------------------------------------------------


def write_worker_job(path: Path, job: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")


def build_worker_job(
    *,
    kind: str,
    pack_dir: Path,
    encoded_prompts: Sequence[dict[str, Any]],
    max_tokens: int,
    request_timeout_s: float,
    mtp_on: bool | None = None,
    probe_mtp: bool = False,
) -> dict[str, Any]:
    job: dict[str, Any] = {
        "schema": WORKER_JOB_SCHEMA,
        "kind": kind,
        "pack_dir": str(pack_dir),
        "max_tokens": int(max_tokens),
        "request_timeout_s": float(request_timeout_s),
        "prompts": [
            {
                "prompt_id": prompt["prompt_id"],
                "category": prompt["category"],
                "input_ids": list(prompt["input_ids"]),
            }
            for prompt in encoded_prompts
        ],
    }
    if kind == "mlx":
        job["probe_mtp"] = bool(probe_mtp)
    elif kind == "engine":
        generate_kwargs = build_generate_kwargs([], max_tokens)
        generate_kwargs.pop("input_tokens")
        job["mtp_on"] = bool(mtp_on)
        job["session_kwargs"] = build_session_kwargs(pack_dir)
        job["generate_kwargs"] = generate_kwargs
    else:
        raise UsageError(f"unknown worker kind: {kind}")
    return job


def stop_process(process: subprocess.Popen[bytes], timeout_s: float) -> tuple[int, bool]:
    """SIGTERM the process group, wait, then SIGKILL. Returns (code, forced)."""
    if process.poll() is not None:
        return int(process.returncode), False
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    try:
        return int(process.wait(timeout=timeout_s)), False
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    return int(process.wait(timeout=30)), True


def validate_worker_dump(
    dump: dict[str, Any],
    *,
    kind: str,
    expected_prompt_ids: Sequence[str],
    expected_mtp_on: bool | None = None,
) -> None:
    """Fail closed on an incomplete or malformed worker dump.

    A dump that is truncated, hand-edited, or missing generations must never
    reach the comparison stage: an empty ``[]`` vs ``[]`` would otherwise read
    as token parity. Every generation must carry both token arrays.
    """
    expected_schema = MLX_DUMP_SCHEMA if kind == "mlx" else ENGINE_DUMP_SCHEMA
    if dump.get("schema") != expected_schema:
        raise InfraError(
            f"{kind} worker dump schema {dump.get('schema')!r} != {expected_schema!r}",
            details={"subtype": "worker_dump", "kind": kind},
        )
    if dump.get("kind") != kind:
        raise InfraError(
            f"{kind} worker dump kind {dump.get('kind')!r} mismatch",
            details={"subtype": "worker_dump", "kind": kind},
        )
    if dump.get("ok") is not True:
        raise InfraError(
            f"{kind} worker dump ok flag is not true: {dump.get('ok')!r}",
            details={"subtype": "worker_dump", "kind": kind},
        )
    if expected_mtp_on is not None and dump.get("mtp_on") is not expected_mtp_on:
        raise InfraError(
            f"engine worker dump mtp_on {dump.get('mtp_on')!r} != {expected_mtp_on!r}",
            details={"subtype": "worker_dump", "kind": kind},
        )
    generations = dump.get("generations")
    if not isinstance(generations, list):
        raise InfraError(
            f"{kind} worker dump generations is not a list",
            details={"subtype": "worker_dump", "kind": kind},
        )
    actual_ids: list[str] = []
    for generation in generations:
        if not isinstance(generation, dict):
            raise InfraError(
                f"{kind} worker dump generation is not an object",
                details={"subtype": "worker_dump", "kind": kind},
            )
        prompt_id = generation.get("prompt_id")
        for field_name in ("prompt_id", "token_ids", "prompt_token_ids"):
            if field_name not in generation:
                raise InfraError(
                    f"{kind} worker dump generation {prompt_id!r} missing "
                    f"{field_name!r}",
                    details={"subtype": "worker_dump", "prompt_id": prompt_id},
                )
        for field_name in ("token_ids", "prompt_token_ids"):
            if not _is_int_list(generation[field_name]):
                raise InfraError(
                    f"{kind} worker dump generation {prompt_id!r} has a "
                    f"non-int {field_name}",
                    details={"subtype": "worker_dump", "prompt_id": prompt_id},
                )
        actual_ids.append(str(prompt_id))
    if actual_ids != [str(prompt_id) for prompt_id in expected_prompt_ids]:
        raise InfraError(
            f"{kind} worker dump generations {actual_ids} do not match the "
            f"selected prompts {list(expected_prompt_ids)}",
            details={
                "subtype": "worker_dump",
                "kind": kind,
                "actual": actual_ids,
                "expected": list(expected_prompt_ids),
            },
        )


_ACTIVE_WORKER_PIDS: set[int] = set()


def assert_no_sibling_gpu_process() -> None:
    """Refuse to start a GPU phase while a sibling worker pid is still alive."""
    alive: list[int] = []
    for pid in list(_ACTIVE_WORKER_PIDS):
        try:
            os.kill(pid, 0)
        except OSError:
            _ACTIVE_WORKER_PIDS.discard(pid)
        else:
            alive.append(pid)
    if alive:
        raise InfraError(
            f"refusing to start a GPU phase while worker pids are alive: {alive}",
            details={"subtype": "sibling_gpu", "pids": alive},
        )


def run_worker_subprocess(
    *,
    kind: str,
    python_bin: Path,
    script_path: Path,
    job_path: Path,
    dump_path: Path,
    log_path: Path,
    env: dict[str, str],
    timeout_s: float,
    stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
    expected_prompt_ids: Sequence[str] = (),
    expected_mtp_on: bool | None = None,
) -> dict[str, Any]:
    """Spawn this script as a hidden worker, wait, and return the parsed dump.

    A non-zero worker exit, a missing/invalid dump, a dump with ok not true,
    or a dump that fails validation is INFRA (the worker always writes a dump,
    so its error payload is preserved). Any exception while waiting kills and
    reaps the worker before propagating so no 126 GiB process is left on the
    GPU for the next phase.
    """
    flag = "--internal-mlx-worker" if kind == "mlx" else "--internal-engine-worker"
    cmd = [
        str(python_bin),
        str(script_path),
        flag,
        "--worker-job-json",
        str(job_path),
        "--worker-dump-json",
        str(dump_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_sibling_gpu_process()
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        _ACTIVE_WORKER_PIDS.add(process.pid)
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stop_process(process, stop_timeout_s)
            raise InfraError(
                f"{kind} worker exceeded the {timeout_s}s load/generate budget",
                details={"subtype": "worker_timeout", "log_tail": _log_tail(log_path)},
            ) from None
        except BaseException:
            # KeyboardInterrupt, OSError, anything: reap the worker before the
            # next GPU phase can start.
            stop_process(process, stop_timeout_s)
            raise
        finally:
            _ACTIVE_WORKER_PIDS.discard(process.pid)
    dump: dict[str, Any] | None = None
    if dump_path.is_file():
        try:
            parsed = json.loads(dump_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                dump = parsed
        except (OSError, json.JSONDecodeError):
            dump = None
    if dump is None:
        raise InfraError(
            f"{kind} worker exited {code} without a valid dump JSON",
            details={
                "subtype": "worker_dump",
                "worker_exit": code,
                "log_tail": _log_tail(log_path),
            },
        )
    if code != 0 or dump.get("ok") is not True:
        error = dump.get("error") or {}
        raise InfraError(
            f"{kind} worker failed: {error.get('message', 'unknown error')}",
            details={
                "subtype": error.get("subtype", "worker"),
                "worker_exit": code,
                "error": error,
                "log_tail": _log_tail(log_path),
            },
        )
    validate_worker_dump(
        dump,
        kind=kind,
        expected_prompt_ids=expected_prompt_ids,
        expected_mtp_on=expected_mtp_on,
    )
    return dump


@contextlib.contextmanager
def _prompt_deadline(timeout_s: float | None):
    """Per-prompt SIGALRM deadline inside workers; no-op without SIGALRM."""
    if not timeout_s or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"prompt generation exceeded the {timeout_s}s request budget")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# Worker mains (heavy imports live here only)
# ---------------------------------------------------------------------------


def _dist_version(name: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return None


def _dist_commit(name: str) -> str | None:
    try:
        import importlib.metadata

        dist = importlib.metadata.distribution(name)
        text = dist.read_text("direct_url.json")
        if not text:
            return None
        info = json.loads(text)
        return (info.get("vcs_info") or {}).get("commit_id")
    except Exception:
        return None


def _worker_error_dump(
    *,
    schema: str,
    kind: str,
    exc: BaseException,
    started_mono: float,
) -> dict[str, Any]:
    if isinstance(exc, HarnessError):
        error = {
            "class": exc.class_name,
            "subtype": exc.details.get("subtype"),
            "message": str(exc),
            "details": exc.details,
        }
    else:
        error = {
            "class": "INFRA",
            "subtype": "worker_exception",
            "message": f"{type(exc).__name__}: {exc}",
            "details": {},
        }
    return {
        "schema": schema,
        "kind": kind,
        "ok": False,
        "error": error,
        "wall_s": round(time.monotonic() - started_mono, 3),
    }


def run_mlx_worker_main(job_path: Path, dump_path: Path) -> int:
    """Child entry for the mlx-vlm reference side. Imports mlx here, never earlier."""
    started_mono = time.monotonic()
    try:
        dump = _mlx_worker_execute(_read_json(Path(job_path)), started_mono)
        code = EXIT_PASS
    except BaseException as exc:  # always write a dump so the parent can classify
        dump = _worker_error_dump(
            schema=MLX_DUMP_SCHEMA,
            kind="mlx",
            exc=exc,
            started_mono=started_mono,
        )
        code = EXIT_INFRA
    try:
        Path(dump_path).write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"mlx worker could not write its dump: {exc}", file=sys.stderr)
        return EXIT_INFRA
    return code


def _mlx_worker_execute(job: dict[str, Any], started_mono: float) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm import load, stream_generate

    pack_dir = job["pack_dir"]
    max_tokens = int(job["max_tokens"])
    load_started = time.monotonic()
    model, processor = load(pack_dir)
    load_wall_s = round(time.monotonic() - load_started, 3)
    tokenizer = getattr(processor, "tokenizer", processor)

    def _consume(stream: Any) -> tuple[list[int], int | None, str | None]:
        out_ids: list[int] = []
        prompt_tokens: int | None = None
        finish_reason: str | None = None
        for res in stream:
            if getattr(res, "prompt_tokens", None) is not None:
                prompt_tokens = int(res.prompt_tokens)
            if res.token is not None:
                out_ids.append(int(res.token))
            if getattr(res, "finish_reason", None):
                finish_reason = res.finish_reason
        return out_ids, prompt_tokens, finish_reason

    def _generate(ids: list[int], prompt_id: str) -> tuple[list[int], int | None, str | None, str]:
        """Primary input_ids path; string fallback carries the re-encode gate."""
        with _prompt_deadline(job.get("request_timeout_s")):
            try:
                # input_ids must be a 2-D mx.array; never a raw prompt string.
                out, count, reason = _consume(
                    stream_generate(
                        model,
                        processor,
                        None,
                        input_ids=mx.array([ids]),
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                )
                return out, count, reason, "input_ids"
            except InfraError:
                raise
            except Exception as primary_exc:
                # Delta-doc string fallback: detokenize the shared ids, pass
                # the canonical string with specials off, then prove the
                # detokenize->encode roundtrip lands back on the shared ids.
                canonical_text = tokenizer.decode(ids)
                try:
                    out, count, reason = _consume(
                        stream_generate(
                            model,
                            processor,
                            canonical_text,
                            max_tokens=max_tokens,
                            temperature=0.0,
                            add_special_tokens=False,
                        )
                    )
                except Exception as fallback_exc:
                    raise InfraError(
                        f"mlx generation failed on {prompt_id}: input_ids path "
                        f"({type(primary_exc).__name__}: {primary_exc}); string "
                        f"fallback ({type(fallback_exc).__name__}: {fallback_exc})",
                        details={"subtype": "mlx_generate", "prompt_id": prompt_id},
                    ) from fallback_exc
                reencoded = [int(t) for t in tokenizer.encode(
                    canonical_text, add_special_tokens=False
                )]
                if reencoded != ids:
                    raise InfraError(
                        f"mlx string fallback re-encode mismatch on {prompt_id}: "
                        "the pack tokenizer does not round-trip the shared ids",
                        details={
                            "subtype": "tokenizer",
                            "prompt_id": prompt_id,
                            "reencoded_prefix": reencoded[:8],
                            "shared_prefix": ids[:8],
                        },
                    )
                return out, count, reason, "string_fallback"

    generations: list[dict[str, Any]] = []
    generate_started = time.monotonic()
    for prompt in job["prompts"]:
        ids = [int(token) for token in prompt["input_ids"]]
        prompt_id = prompt["prompt_id"]
        prompt_started = time.monotonic()
        out_ids, prompt_tokens, finish_reason, path = _generate(ids, prompt_id)
        out_ids = truncate_lookahead_tokens(out_ids, max_tokens)
        # An unreported finish_reason is INFRA; never inferred as stop/length.
        classify_finish_reason(finish_reason, context=f" (mlx, {prompt_id})")
        if prompt_tokens is None:
            raise InfraError(
                f"mlx did not report prompt_tokens on {prompt_id}",
                details={"subtype": "tokenizer", "prompt_id": prompt_id},
            )
        if prompt_tokens != len(ids):
            raise InfraError(
                f"mlx prompt_tokens {prompt_tokens} != shared ids length "
                f"{len(ids)} on {prompt_id} (special-token injection?)",
                details={"subtype": "tokenizer", "prompt_id": prompt_id},
            )
        record = GenerationRecord(
            prompt_id=prompt_id,
            prompt_token_ids=ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(out_ids),
            token_ids=out_ids,
            finish_reason=finish_reason,
            wall_s=round(time.monotonic() - prompt_started, 3),
            extras={"prompt_path": path, "prompt_ids_source": "job"},
        )
        generations.append(record.to_dict())
    generate_wall_s = round(time.monotonic() - generate_started, 3)

    mtp_probe = None
    if job.get("probe_mtp"):
        mtp_probe = _mlx_mtp_probe(pack_dir)

    return {
        "schema": MLX_DUMP_SCHEMA,
        "kind": "mlx",
        "ok": True,
        "load_wall_s": load_wall_s,
        "generate_wall_s": generate_wall_s,
        "wall_s": round(time.monotonic() - started_mono, 3),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "mlx_version": getattr(mx, "__version__", None),
        "mlx_vlm_version": _dist_version("mlx-vlm"),
        "mlx_vlm_commit": _dist_commit("mlx-vlm"),
        "pinned_mlx_vlm_commit": PINNED_MLX_VLM_COMMIT,
        "mtp_probe": mtp_probe,
        "generations": generations,
    }


def _mlx_mtp_probe(pack_dir: str) -> dict[str, Any]:
    """Optional mlx-vlm split_mtp probe; always degrades to a skipped status."""
    with tempfile.TemporaryDirectory(prefix="ax-parity-mtp-probe-") as tmp_dir:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlx_vlm.split_mtp",
                    "--model",
                    pack_dir,
                    "--output",
                    tmp_dir,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"status": "skipped", "reason": f"split_mtp failed to run: {exc}"}
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-2000:]
            return {"status": "skipped", "reason": f"split_mtp exited {result.returncode}: {tail}"}
    return {
        "status": "ok",
        "accept_rate": None,
        "note": "split_mtp succeeded against the AXQ sidecar layout; no "
        "drafter accept rate was measured",
    }


def run_engine_worker_main(job_path: Path, dump_path: Path) -> int:
    """Child entry for one engine Session lifetime (MTP-off or MTP-on)."""
    started_mono = time.monotonic()
    try:
        dump = _engine_worker_execute(_read_json(Path(job_path)), started_mono)
        code = EXIT_PASS
    except BaseException as exc:  # always write a dump so the parent can classify
        dump = _worker_error_dump(
            schema=ENGINE_DUMP_SCHEMA,
            kind="engine",
            exc=exc,
            started_mono=started_mono,
        )
        code = EXIT_INFRA
    try:
        Path(dump_path).write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"engine worker could not write its dump: {exc}", file=sys.stderr)
        return EXIT_INFRA
    return code


def _engine_worker_execute(job: dict[str, Any], started_mono: float) -> dict[str, Any]:
    import ax_engine

    session_kwargs = dict(job["session_kwargs"])
    load_started = time.monotonic()
    session = ax_engine.Session(**session_kwargs)
    load_wall_s = round(time.monotonic() - load_started, 3)

    generations: list[dict[str, Any]] = []
    generate_started = time.monotonic()
    try:
        for prompt in job["prompts"]:
            ids = [int(token) for token in prompt["input_ids"]]
            generate_kwargs = dict(job["generate_kwargs"])
            generate_kwargs["input_tokens"] = list(ids)
            prompt_started = time.monotonic()
            with _prompt_deadline(job.get("request_timeout_s")):
                result = session.generate(**generate_kwargs)
            prompt_id = prompt["prompt_id"]
            assert_prompt_tokens_identity(result.prompt_tokens, ids, prompt_id=prompt_id)
            if result.status != "finished":
                raise InfraError(
                    f"generate status {result.status!r} on {prompt_id}",
                    details={"subtype": "status", "prompt_id": prompt_id},
                )
            classify_finish_reason(result.finish_reason, context=f" (engine, {prompt_id})")
            route = getattr(result, "route", None)
            decisions = getattr(route, "crossover_decisions", None) if route else None
            record = GenerationRecord(
                prompt_id=prompt_id,
                prompt_token_ids=[int(token) for token in result.prompt_tokens],
                prompt_tokens=len(result.prompt_tokens),
                completion_tokens=len(result.output_tokens),
                token_ids=[int(token) for token in result.output_tokens],
                finish_reason=result.finish_reason,
                status=result.status,
                step_count=int(result.step_count),
                ttft_step=result.ttft_step,
                wall_s=round(time.monotonic() - prompt_started, 3),
                mtp=extract_mtp_decisions(decisions),
                accept_proxy=compute_accept_proxy(
                    int(result.step_count), len(result.output_tokens)
                ),
                extras={"prompt_ids_source": "engine_echo"},
            )
            generations.append(record.to_dict())
    finally:
        # GPU release is guaranteed by worker process exit, not by close().
        with contextlib.suppress(Exception):
            session.close()
    generate_wall_s = round(time.monotonic() - generate_started, 3)

    return {
        "schema": ENGINE_DUMP_SCHEMA,
        "kind": "engine",
        "ok": True,
        "mtp_on": bool(job.get("mtp_on")),
        "load_wall_s": load_wall_s,
        "generate_wall_s": generate_wall_s,
        "wall_s": round(time.monotonic() - started_mono, 3),
        "ax_engine_version": getattr(ax_engine, "__version__", None) or _dist_version("ax-engine"),
        "generations": generations,
    }


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


REQUIRED_EVIDENCE_KEYS = (
    "schema_version",
    "status",
    "exit_code",
    "mode",
    "started_at",
    "finished_at",
    "wall_s",
    "git_sha",
    "engine_version",
    "hardware",
    "pack",
    "software",
    "run_config",
    "prompts",
    "checks",
    "mlx",
    "engine_mtp_off",
    "engine_mtp_on",
    "comparisons",
    "warnings",
    "failure",
)

STATUS_EXIT_CODES = {"PASS": EXIT_PASS, "SKIP": EXIT_SKIP, "INFRA": EXIT_INFRA, "DIVERGENCE": 4}
CHECK_VALUES = ("PASS", "FAIL", "SKIPPED", "n/a")


def evidence_filename(*, pack_tag: str, mode: str, git_sha: str) -> Path:
    """Deterministic evidence path under benchmarks/results/."""
    sha8 = git_sha[:8] if re.fullmatch(r"[0-9a-f]{8,40}", git_sha or "") else "unknown"
    name = f"qwen38-flash-next-parity-{pack_tag}-{mode}-{sha8}.json"
    return REPO_ROOT / "benchmarks" / "results" / name


def dump_filename(*, pack_tag: str, mode: str, git_sha: str) -> Path:
    evidence = evidence_filename(pack_tag=pack_tag, mode=mode, git_sha=git_sha)
    return evidence.with_name(evidence.name[: -len(".json")] + "-dump.json")


def validate_evidence(doc: dict[str, Any]) -> list[str]:
    """Return a list of schema problems (empty list means the doc is valid)."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["evidence is not a JSON object"]
    for key in REQUIRED_EVIDENCE_KEYS:
        if key not in doc:
            problems.append(f"missing key: {key}")
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    status = doc.get("status")
    if status not in STATUS_EXIT_CODES:
        problems.append(f"status {status!r} not in {sorted(STATUS_EXIT_CODES)}")
    elif doc.get("exit_code") != STATUS_EXIT_CODES[status]:
        problems.append(
            f"exit_code {doc.get('exit_code')!r} does not match status {status!r}"
        )
    checks = doc.get("checks")
    if not isinstance(checks, dict):
        problems.append("checks is not an object")
    else:
        for name, value in checks.items():
            if value not in CHECK_VALUES:
                problems.append(f"checks.{name} value {value!r} not in {list(CHECK_VALUES)}")
        if status == "PASS":
            for name in GATED_CHECK_NAMES:
                value = checks.get(name)
                if value not in ("PASS", "n/a"):
                    problems.append(
                        f"checks.{name} is {value!r}; a PASS evidence requires "
                        "PASS or n/a on gated checks"
                    )
    return problems


def write_evidence(path: Path, doc: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _base_evidence(
    args: argparse.Namespace,
    *,
    tag: str,
    pack_dir: Path,
    git_sha: str,
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": None,
        "exit_code": None,
        "mode": args.mode,
        "started_at": started_at,
        "finished_at": None,
        "wall_s": None,
        "git_sha": git_sha,
        "engine_version": collect_engine_version(REPO_ROOT),
        "hardware": collect_hardware(),
        "pack": None,
        "software": {
            "mlx": None,
            "mlx_vlm_version": None,
            "mlx_vlm_commit": None,
            "python": platform.python_version(),
            "ax_engine": None,
        },
        "run_config": {
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream_experts": args.stream_experts,
            "stream_experts_applied": "auto",
            "disable_ngram_mtp_off": f"{NO_SPEC_ENV}=1",
            "mtp": effective_mtp(args),
            "session": {
                "mlx": True,
                "deterministic": True,
                "model_id": EXPECTED_MODEL_TYPE,
                "generate": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repetition_penalty": 1.0,
                    "no_repeat_ngram_size": 0,
                    "seed": 0,
                },
            },
            "cert_env_mtp_on": f"{CERT_ENV}=1",
            "extension_build": "maturin develop --profile release-pyext",
            "load_timeout_s": args.load_timeout_s,
            "request_timeout_s": args.request_timeout_s,
            "settle_s": args.settle_s,
            "stop_timeout_s": args.stop_timeout_s,
            "require_mlx_mtp": bool(args.require_mlx_mtp),
            "hash_weights": bool(args.hash_weights),
        },
        "prompts": [],
        "checks": _initial_checks(args),
        "mlx": None,
        "engine_mtp_off": None,
        "engine_mtp_on": None,
        "comparisons": {},
        "warnings": [],
        "failure": None,
    }


def _initial_checks(args: argparse.Namespace) -> dict[str, str]:
    """Pending ("SKIPPED") vs not-applicable ("n/a") checks for this mode/mtp."""
    mtp = effective_mtp(args)
    runs_mlx = args.mode in ("full", "mlx-dump")
    runs_engine = args.mode in ("full", "engine", "engine-dump")
    runs_mtp_on = args.mode in ("full", "engine") and mtp in ("on", "both")
    return {
        "token_parity_mlx_vs_engine_off": "SKIPPED" if args.mode == "full" else "n/a",
        "mtp_losslessness_engine_on_vs_off": "SKIPPED" if runs_mtp_on else "n/a",
        "mtp_engagement": "SKIPPED" if runs_mtp_on else "n/a",
        "prompt_tokens_gate": "SKIPPED" if runs_engine else "n/a",
        "mlx_mtp_accept_rate_crosscheck": "SKIPPED" if runs_mlx else "n/a",
    }


def _fill_pack_section(
    doc: dict[str, Any],
    *,
    tag: str,
    pack_dir: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    contracts: dict[str, Any],
) -> None:
    # repo/revision come only from the verified sha256-manifest.json; a pack
    # without a manifest never reaches this point (INFRA at prepare time).
    doc["pack"] = {
        "repo_id": manifest.get("repo"),
        "revision": manifest.get("revision"),
        "tag": tag,
        "path": str(pack_dir),
        "layout": "plain_dir",
        "sha256_manifest": "sha256-manifest.json" if manifest.get("present") else None,
        "sha256_manifest_complete": manifest.get("complete"),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "contract_sha256": contracts["contract_sha256"],
        "safetensors_inventory": contracts["safetensors_inventory"],
        "weights_hashed": contracts["weights_hashed"],
    }


def _mlx_section(dump: dict[str, Any], log_path: Path) -> dict[str, Any]:
    return {
        "status": "ok",
        "load_wall_s": dump.get("load_wall_s"),
        "generate_wall_s": dump.get("generate_wall_s"),
        "peak_memory_bytes": dump.get("peak_memory_bytes"),
        "worker_log": str(log_path),
        "mtp_probe": dump.get("mtp_probe"),
        "generations": dump.get("generations", []),
    }


def _engine_section(dump: dict[str, Any], log_path: Path) -> dict[str, Any]:
    return {
        "load_wall_s": dump.get("load_wall_s"),
        "generate_wall_s": dump.get("generate_wall_s"),
        "forced_kill": False,
        "worker_log": str(log_path),
        "mtp_lifetime": None,
        "generations": dump.get("generations", []),
    }


def _finalize(
    doc: dict[str, Any],
    output_path: Path,
    *,
    status: str,
    started_mono: float,
    failure: dict[str, Any] | None = None,
) -> int:
    """Write the evidence JSON and return the exit code actually used.

    A PASS that has not earned every gated check (or that fails schema
    validation) is rewritten as INFRA: schema problems are never a warning on
    a PASS.
    """
    doc["status"] = status
    doc["exit_code"] = STATUS_EXIT_CODES[status]
    doc["finished_at"] = _utc_now()
    doc["wall_s"] = round(time.monotonic() - started_mono, 3)
    doc["failure"] = failure
    if status == "PASS":
        unearned = {
            name: doc["checks"].get(name)
            for name in GATED_CHECK_NAMES
            if doc["checks"].get(name) not in ("PASS", "n/a")
        }
        problems = validate_evidence(doc)
        if unearned or problems:
            doc["status"] = "INFRA"
            doc["exit_code"] = EXIT_INFRA
            doc["failure"] = {
                "class": "INFRA",
                "subtype": "checks_incomplete" if unearned else "evidence_schema",
                "message": (
                    "refusing to report PASS: "
                    + (
                        f"gated checks not earned {unearned}"
                        if unearned
                        else f"evidence schema problems {problems}"
                    )
                ),
                "details": {"unearned_checks": unearned, "schema_problems": problems},
            }
    else:
        problems = validate_evidence(doc)
        if problems:
            print(f"WARNING: evidence schema problems: {problems}", file=sys.stderr)
    write_evidence(output_path, doc)
    return int(doc["exit_code"])


def _failure_payload(exc: HarnessError) -> dict[str, Any]:
    return {
        "class": exc.class_name,
        "subtype": exc.details.get("subtype"),
        "message": str(exc),
        "details": exc.details,
    }


# ---------------------------------------------------------------------------
# Run phases
# ---------------------------------------------------------------------------


def _prepare_run(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve, classify, verify the manifest, and encode prompts once."""
    pack_dir, tag = resolve_pack_dir(args)
    config = classify_snapshot(pack_dir)
    manifest = verify_sha256_manifest(pack_dir, tag, hash_weights=args.hash_weights)
    contracts = hash_contract_files(pack_dir, hash_weights=args.hash_weights)
    tokenizer = load_pack_tokenizer(pack_dir)
    prompts = load_prompts(args)
    encoded = encode_prompts(tokenizer, prompts)
    return {
        "pack_dir": pack_dir,
        "tag": tag,
        "config": config,
        "manifest": manifest,
        "contracts": contracts,
        "tokenizer": tokenizer,
        "prompts": prompts,
        "encoded": encoded,
    }


def _run_mlx_phase(
    args: argparse.Namespace,
    prep: dict[str, Any],
    work_dir: Path,
) -> tuple[dict, Path]:
    job = build_worker_job(
        kind="mlx",
        pack_dir=prep["pack_dir"],
        encoded_prompts=prep["encoded"],
        max_tokens=args.max_tokens,
        request_timeout_s=args.request_timeout_s,
        probe_mtp=not args.skip_mlx_mtp_probe,
    )
    job_path = work_dir / "mlx-worker-job.json"
    dump_path = work_dir / "mlx-worker-dump.json"
    log_path = work_dir / "mlx-worker.log"
    write_worker_job(job_path, job)
    dump = run_worker_subprocess(
        kind="mlx",
        python_bin=Path(args.python_bin),
        script_path=SCRIPT_PATH,
        job_path=job_path,
        dump_path=dump_path,
        log_path=log_path,
        env=build_mlx_worker_env(),
        timeout_s=args.load_timeout_s,
        stop_timeout_s=args.stop_timeout_s,
        expected_prompt_ids=[prompt["prompt_id"] for prompt in prep["encoded"]],
    )
    time.sleep(args.settle_s)
    return dump, log_path


def _run_engine_phase(
    args: argparse.Namespace,
    prep: dict[str, Any],
    work_dir: Path,
    *,
    mtp_on: bool,
) -> tuple[dict, Path]:
    label = "engine-on" if mtp_on else "engine-off"
    job = build_worker_job(
        kind="engine",
        pack_dir=prep["pack_dir"],
        encoded_prompts=prep["encoded"],
        max_tokens=args.max_tokens,
        request_timeout_s=args.request_timeout_s,
        mtp_on=mtp_on,
    )
    job_path = work_dir / f"{label}-worker-job.json"
    dump_path = work_dir / f"{label}-worker-dump.json"
    log_path = work_dir / f"{label}-worker.log"
    write_worker_job(job_path, job)
    dump = run_worker_subprocess(
        kind="engine",
        python_bin=Path(args.python_bin),
        script_path=SCRIPT_PATH,
        job_path=job_path,
        dump_path=dump_path,
        log_path=log_path,
        env=build_engine_worker_env(mtp_on),
        timeout_s=args.load_timeout_s,
        stop_timeout_s=args.stop_timeout_s,
        expected_prompt_ids=[prompt["prompt_id"] for prompt in prep["encoded"]],
        expected_mtp_on=mtp_on,
    )
    time.sleep(args.settle_s)
    return dump, log_path


def _records_from_dump(dump: dict[str, Any]) -> list[GenerationRecord]:
    return [GenerationRecord.from_dict(item) for item in dump.get("generations", [])]


def _apply_mlx_software(doc: dict[str, Any], dump: dict[str, Any]) -> None:
    doc["software"]["mlx"] = dump.get("mlx_version")
    doc["software"]["mlx_vlm_version"] = dump.get("mlx_vlm_version")
    doc["software"]["mlx_vlm_commit"] = dump.get("mlx_vlm_commit")


def _apply_engine_software(doc: dict[str, Any], dump: dict[str, Any]) -> None:
    if doc["software"].get("ax_engine") is None:
        doc["software"]["ax_engine"] = dump.get("ax_engine_version")


def _check_mlx_mtp_requirement(args: argparse.Namespace, dump: dict[str, Any]) -> str:
    """Informational mlx MTP cross-check; --require-mlx-mtp upgrades to INFRA.

    The check only reads "PASS" when an accept rate was actually measured. A
    probe that merely ran the splitter is "SKIPPED", never "PASS".
    """
    probe = dump.get("mtp_probe")
    if probe is None:
        if args.require_mlx_mtp:
            raise InfraError(
                "--require-mlx-mtp set but the mlx MTP probe did not run",
                details={"subtype": "mlx_mtp_probe"},
            )
        return "SKIPPED"
    if probe.get("status") != "ok":
        if args.require_mlx_mtp:
            raise InfraError(
                f"--require-mlx-mtp set but the mlx MTP probe was skipped: "
                f"{probe.get('reason', 'no reason recorded')}",
                details={"subtype": "mlx_mtp_probe", "probe": probe},
            )
        return "SKIPPED"
    if probe.get("accept_rate") is None:
        return "SKIPPED"
    return "PASS"


def _prompt_entries(encoded: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": prompt["prompt_id"],
            "category": prompt["category"],
            "n_chars": prompt["n_chars"],
            "n_prompt_tokens": prompt["n_prompt_tokens"],
            "text_sha256": prompt["text_sha256"],
        }
        for prompt in encoded
    ]


def _mark_failure_check(doc: dict[str, Any], exc: HarnessError) -> None:
    """Flip the check that failed to FAIL so evidence does not read as a pass."""
    compare = exc.details.get("compare")
    subtype = exc.details.get("subtype")
    if compare == "mlx_vs_engine_off":
        doc["checks"]["token_parity_mlx_vs_engine_off"] = "FAIL"
    elif compare == "engine_on_vs_engine_off":
        doc["checks"]["mtp_losslessness_engine_on_vs_off"] = "FAIL"
    elif subtype in ("mtp_engagement", "mtp_telemetry", "mtp_off_contaminated"):
        doc["checks"]["mtp_engagement"] = "FAIL"
    elif subtype == "tokenizer":
        doc["checks"]["prompt_tokens_gate"] = "FAIL"


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def _run_with_classification(
    args: argparse.Namespace,
    body: Callable[
        [argparse.Namespace, dict[str, Any], dict[str, Any], Path, Path | None],
        None,
    ],
) -> int:
    """Shared scaffolding: resolve, prepare, run the mode body, write evidence."""
    started_mono = time.monotonic()
    started_at = _utc_now()
    pack_dir, tag = resolve_pack_dir(args)
    git_sha = collect_git_sha(REPO_ROOT)
    evidence_path = evidence_filename(pack_tag=tag, mode=args.mode, git_sha=git_sha)
    if args.mode in ("mlx-dump", "engine-dump"):
        dump_output_path = (
            Path(args.output)
            if args.output
            else dump_filename(pack_tag=tag, mode=args.mode, git_sha=git_sha)
        )
        finalize_path = evidence_path
    else:
        dump_output_path = None
        finalize_path = Path(args.output) if args.output else evidence_path
    doc = _base_evidence(args, tag=tag, pack_dir=pack_dir, git_sha=git_sha, started_at=started_at)
    try:
        prep = _prepare_run(args)
    except SkipError as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        if args.output:
            _finalize(
                doc,
                Path(args.output),
                status="SKIP",
                started_mono=started_mono,
                failure=_failure_payload(exc),
            )
        return EXIT_SKIP
    except UsageError:
        raise
    except HarnessError as exc:
        _mark_failure_check(doc, exc)
        code = _finalize(
            doc,
            finalize_path,
            status=exc.class_name,
            started_mono=started_mono,
            failure=_failure_payload(exc),
        )
        print(f"{exc.class_name}: {exc}", file=sys.stderr)
        return code
    except Exception as exc:  # never let a non-HarnessError escape without evidence
        failure = {
            "class": "INFRA",
            "subtype": "unexpected",
            "message": f"{type(exc).__name__}: {exc}",
            "details": {},
        }
        code = _finalize(
            doc,
            finalize_path,
            status="INFRA",
            started_mono=started_mono,
            failure=failure,
        )
        print(f"INFRA: unexpected error during prepare: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return code
    doc["prompts"] = _prompt_entries(prep["encoded"])
    _fill_pack_section(
        doc,
        tag=tag,
        pack_dir=prep["pack_dir"],
        config=prep["config"],
        manifest=prep["manifest"],
        contracts=prep["contracts"],
    )
    if args.mode == "full" and tag == "6bit":
        warning = (
            "mode=full on the 6-bit pack: mlx-vlm will likely exceed the wired "
            "cap; --mode engine is the supported 6-bit path"
        )
        doc["warnings"].append(warning)
        print(f"WARNING: {warning}", file=sys.stderr)
    work_dir = Path(tempfile.mkdtemp(prefix="ax-parity-qwen38-"))
    doc["run_config"]["work_dir"] = str(work_dir)
    try:
        body(args, prep, doc, work_dir, dump_output_path)
    except UsageError:
        raise
    except HarnessError as exc:
        _mark_failure_check(doc, exc)
        if isinstance(exc, DivergenceError) and exc.details.get("report"):
            doc["comparisons"][exc.details["compare"]] = exc.details["report"]
        code = _finalize(
            doc,
            finalize_path,
            status=exc.class_name,
            started_mono=started_mono,
            failure=_failure_payload(exc),
        )
        print(f"{exc.class_name}: {exc}", file=sys.stderr)
        return code
    except Exception as exc:  # never let a non-HarnessError escape without evidence
        failure = {
            "class": "INFRA",
            "subtype": "unexpected",
            "message": f"{type(exc).__name__}: {exc}",
            "details": {},
        }
        code = _finalize(
            doc,
            finalize_path,
            status="INFRA",
            started_mono=started_mono,
            failure=failure,
        )
        print(f"INFRA: unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return code
    code = _finalize(doc, finalize_path, status="PASS", started_mono=started_mono)
    if code == EXIT_PASS:
        print(f"PASS: evidence written to {finalize_path}")
    else:
        print(f"INFRA: PASS was not earned; evidence written to {finalize_path}",
              file=sys.stderr)
    return code


def _full_body(
    args: argparse.Namespace,
    prep: dict[str, Any],
    doc: dict[str, Any],
    work_dir: Path,
    dump_output_path: Path | None,
) -> None:
    """mlx greedy -> engine MTP-off -> compare A -> engine MTP-on -> compare B."""
    decode_fn = lambda ids: prep["tokenizer"].decode(ids)  # noqa: E731

    mlx_dump, mlx_log = _run_mlx_phase(args, prep, work_dir)
    _apply_mlx_software(doc, mlx_dump)
    doc["mlx"] = _mlx_section(mlx_dump, mlx_log)
    mlx_records = _records_from_dump(mlx_dump)
    verify_prompt_token_gate(mlx_records, prep["encoded"])

    off_dump, off_log = _run_engine_phase(args, prep, work_dir, mtp_on=False)
    _apply_engine_software(doc, off_dump)
    doc["engine_mtp_off"] = _engine_section(off_dump, off_log)
    off_records = _records_from_dump(off_dump)
    verify_prompt_token_gate(off_records, prep["encoded"])
    doc["checks"]["prompt_tokens_gate"] = "PASS"
    doc["engine_mtp_off"]["mtp_lifetime"] = require_mtp_off_baseline_clean(off_records)

    report_a = compare_generations(
        mlx_records,
        off_records,
        compare="mlx_vs_engine_off",
        left_name="mlx",
        right_name="engine_mtp_off",
        decode_fn=decode_fn,
    )
    doc["comparisons"]["mlx_vs_engine_off"] = report_a
    doc["checks"]["token_parity_mlx_vs_engine_off"] = "PASS"
    doc["warnings"].extend(report_a.get("warnings", []))

    if effective_mtp(args) in ("on", "both"):
        on_dump, on_log = _run_engine_phase(args, prep, work_dir, mtp_on=True)
        _apply_engine_software(doc, on_dump)
        doc["engine_mtp_on"] = _engine_section(on_dump, on_log)
        on_records = _records_from_dump(on_dump)
        verify_prompt_token_gate(on_records, prep["encoded"])
        lifetime = require_mtp_engaged(on_records)
        doc["engine_mtp_on"]["mtp_lifetime"] = lifetime
        doc["checks"]["mtp_engagement"] = "PASS"
        ngram_accepted = sum_ngram_accepted(on_records)
        if ngram_accepted > 0:
            doc["warnings"].append(
                f"mtp-on lifetime recorded {ngram_accepted} n-gram accepted tokens; "
                "losslessness is still enforced against the mtp-off baseline"
            )
        report_b = compare_generations(
            on_records,
            off_records,
            compare="engine_on_vs_engine_off",
            left_name="engine_mtp_on",
            right_name="engine_mtp_off",
            decode_fn=decode_fn,
        )
        doc["comparisons"]["engine_on_vs_engine_off"] = report_b
        doc["checks"]["mtp_losslessness_engine_on_vs_off"] = "PASS"
        doc["warnings"].extend(report_b.get("warnings", []))

    doc["checks"]["mlx_mtp_accept_rate_crosscheck"] = _check_mlx_mtp_requirement(args, mlx_dump)


def _engine_body(
    args: argparse.Namespace,
    prep: dict[str, Any],
    doc: dict[str, Any],
    work_dir: Path,
    dump_output_path: Path | None,
) -> None:
    """Engine-only: MTP-off greedy sanity -> MTP-on -> engagement -> compare."""
    decode_fn = lambda ids: prep["tokenizer"].decode(ids)  # noqa: E731

    off_dump, off_log = _run_engine_phase(args, prep, work_dir, mtp_on=False)
    _apply_engine_software(doc, off_dump)
    doc["engine_mtp_off"] = _engine_section(off_dump, off_log)
    off_records = _records_from_dump(off_dump)
    verify_prompt_token_gate(off_records, prep["encoded"])
    doc["checks"]["prompt_tokens_gate"] = "PASS"
    doc["engine_mtp_off"]["mtp_lifetime"] = require_mtp_off_baseline_clean(off_records)
    for record in off_records:
        if record.completion_tokens == 0 and record.finish_reason == "stop":
            doc["warnings"].append(
                f"{record.prompt_id}: empty completion with finish_reason stop"
            )

    if effective_mtp(args) in ("on", "both"):
        on_dump, on_log = _run_engine_phase(args, prep, work_dir, mtp_on=True)
        _apply_engine_software(doc, on_dump)
        doc["engine_mtp_on"] = _engine_section(on_dump, on_log)
        on_records = _records_from_dump(on_dump)
        verify_prompt_token_gate(on_records, prep["encoded"])
        lifetime = require_mtp_engaged(on_records)
        doc["engine_mtp_on"]["mtp_lifetime"] = lifetime
        doc["checks"]["mtp_engagement"] = "PASS"
        ngram_accepted = sum_ngram_accepted(on_records)
        if ngram_accepted > 0:
            doc["warnings"].append(
                f"mtp-on lifetime recorded {ngram_accepted} n-gram accepted tokens; "
                "losslessness is still enforced against the mtp-off baseline"
            )
        report_b = compare_generations(
            on_records,
            off_records,
            compare="engine_on_vs_engine_off",
            left_name="engine_mtp_on",
            right_name="engine_mtp_off",
            decode_fn=decode_fn,
        )
        doc["comparisons"]["engine_on_vs_engine_off"] = report_b
        doc["checks"]["mtp_losslessness_engine_on_vs_off"] = "PASS"
        doc["warnings"].extend(report_b.get("warnings", []))


def _mlx_dump_body(
    args: argparse.Namespace,
    prep: dict[str, Any],
    doc: dict[str, Any],
    work_dir: Path,
    dump_output_path: Path | None,
) -> None:
    mlx_dump, mlx_log = _run_mlx_phase(args, prep, work_dir)
    _apply_mlx_software(doc, mlx_dump)
    doc["mlx"] = _mlx_section(mlx_dump, mlx_log)
    write_evidence(Path(dump_output_path), mlx_dump)
    doc["run_config"]["dump_path"] = str(dump_output_path)
    doc["checks"]["mlx_mtp_accept_rate_crosscheck"] = _check_mlx_mtp_requirement(args, mlx_dump)


def _engine_dump_body(
    args: argparse.Namespace,
    prep: dict[str, Any],
    doc: dict[str, Any],
    work_dir: Path,
    dump_output_path: Path | None,
) -> None:
    mtp_on = effective_mtp(args) == "on"
    dump, log_path = _run_engine_phase(args, prep, work_dir, mtp_on=mtp_on)
    _apply_engine_software(doc, dump)
    section = _engine_section(dump, log_path)
    records = _records_from_dump(dump)
    verify_prompt_token_gate(records, prep["encoded"])
    if mtp_on:
        doc["engine_mtp_on"] = section
    else:
        doc["engine_mtp_off"] = section
    doc["checks"]["prompt_tokens_gate"] = "PASS"
    write_evidence(Path(dump_output_path), dump)
    doc["run_config"]["dump_path"] = str(dump_output_path)


def run_full(args: argparse.Namespace) -> int:
    return _run_with_classification(args, _full_body)


def run_engine_only(args: argparse.Namespace) -> int:
    return _run_with_classification(args, _engine_body)


def run_mlx_dump(args: argparse.Namespace) -> int:
    return _run_with_classification(args, _mlx_dump_body)


def run_engine_dump(args: argparse.Namespace) -> int:
    return _run_with_classification(args, _engine_dump_body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def effective_mtp(args: argparse.Namespace) -> str:
    if args.mtp:
        return args.mtp
    return "both" if args.mode in ("full", "engine") else "off"


def _validate_mode_args(args: argparse.Namespace) -> None:
    mtp = effective_mtp(args)
    if args.mode in ("full", "engine") and mtp == "on":
        raise UsageError(
            "--mtp on alone cannot establish the MTP-off baseline; use --mtp both",
            details={"subtype": "cli"},
        )
    if args.mode == "engine-dump" and mtp == "both":
        raise UsageError(
            "--mtp both is not valid for engine-dump; choose off or on",
            details={"subtype": "cli"},
        )
    if args.mode == "mlx-dump" and args.mtp is not None:
        raise UsageError(
            "--mtp has no effect in mlx-dump; do not pass it",
            details={"subtype": "cli"},
        )
    if args.max_tokens <= 0:
        raise UsageError("--max-tokens must be positive", details={"subtype": "cli"})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parity_qwen38_flash_next.py",
        description=(
            "Real-pack parity harness for qwen4_exp (Qwen3.8-Flash-Next): "
            "greedy token-id parity vs mlx-vlm, MTP losslessness, MTP "
            "engagement telemetry. Exit codes: 0 PASS / 1 USAGE / 2 SKIP / "
            "3 INFRA / 4 DIVERGENCE."
        ),
    )
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--pack-dir-6bit", type=Path, default=None)
    parser.add_argument("--pack-tag", choices=("4bit", "6bit"), default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--prompt-ids", default=None)
    parser.add_argument("--prompts-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--stream-experts",
        choices=("auto",),
        default="auto",
        help="only 'auto' is accepted; the Python Session latches Auto at "
        "construction, so any other value would be a silent no-op",
    )
    parser.add_argument("--mtp", choices=("off", "on", "both"), default=None)
    parser.add_argument("--load-timeout-s", type=float, default=DEFAULT_LOAD_TIMEOUT_S)
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument("--stop-timeout-s", type=float, default=DEFAULT_STOP_TIMEOUT_S)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--require-mlx-mtp", action="store_true")
    parser.add_argument("--skip-mlx-mtp-probe", action="store_true")
    parser.add_argument("--internal-mlx-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-engine-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-job-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-dump-json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_USAGE
        return EXIT_PASS if code == 0 else EXIT_USAGE
    if args.internal_mlx_worker or args.internal_engine_worker:
        if args.worker_job_json is None or args.worker_dump_json is None:
            print(
                "USAGE: --internal-mlx-worker / --internal-engine-worker require "
                "--worker-job-json and --worker-dump-json",
                file=sys.stderr,
            )
            return EXIT_USAGE
        if args.internal_mlx_worker:
            return run_mlx_worker_main(args.worker_job_json, args.worker_dump_json)
        return run_engine_worker_main(args.worker_job_json, args.worker_dump_json)
    try:
        _validate_mode_args(args)
        if args.mode == "full":
            return run_full(args)
        if args.mode == "engine":
            return run_engine_only(args)
        if args.mode == "mlx-dump":
            return run_mlx_dump(args)
        return run_engine_dump(args)
    except HarnessError as exc:
        print(f"{exc.class_name}: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # unexpected errors are INFRA, never divergence
        print(f"INFRA: unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INFRA


if __name__ == "__main__":
    raise SystemExit(main())
