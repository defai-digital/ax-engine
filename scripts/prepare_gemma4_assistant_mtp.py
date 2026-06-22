#!/usr/bin/env python3
"""Package a Gemma 4 target + assistant pair for ax-engine assistant MTP.

Gemma 4's multi-token prediction is NOT a fused mtp.* sidecar (that is the Qwen
path — see prepare_mtp_sidecar.py). It is a separate small "assistant" drafter
model that shares the target's tokenizer/embedding table and drafts tokens off
the target's last-layer activations. The runtime loads it via an
``ax_gemma4_assistant_mtp.json`` contract placed in the target model root
(crates/ax-engine-mlx/src/gemma4_assistant_mtp.rs).

This tool assembles a self-contained, ready-to-serve model directory:

    <output>/
      config.json, *.safetensors, tokenizer.json   (target, hardlinked)
      model-manifest.json                           (target AX manifest)
      ax_gemma4_assistant_mtp.json                  (the contract)
      assistant/
        config.json, *.safetensors, tokenizer.json  (assistant)
        model-manifest.json                         (assistant AX manifest)

It also patches the ax-engine packaging markers the runtime requires on the
assistant config (``model_type: gemma4_assistant`` and ``backbone_hidden_size``),
copies the target tokenizer into the assistant dir so the byte-identity check
passes, and PRE-VALIDATES every check the runtime performs so failures surface
now rather than at server start.

Usage:
  # From local model dirs (or cached repo ids) — assistant downloaded separately.
  python3 scripts/prepare_gemma4_assistant_mtp.py \\
      --target mlx-community/gemma-4-e2b-it-4bit \\
      --assistant google/gemma-4-e2b-it-assistant

  # Canonical pair ids are derived (quant suffixes stripped); override if needed:
  python3 scripts/prepare_gemma4_assistant_mtp.py \\
      --target /path/to/gemma-4-e2b-it-4bit \\
      --assistant /path/to/assistant \\
      --target-model-id gemma-4-e2b-it \\
      --assistant-model-id gemma-4-e2b-it-assistant

``--target`` / ``--assistant`` accept a local dir or an ``org/name`` repo id
(looked up in the HF cache; download first via scripts/download_model.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)

CONTRACT_FILE = "ax_gemma4_assistant_mtp.json"
SCHEMA_VERSION = "ax.gemma4_assistant_mtp.v1"
ASSISTANT_MODEL_TYPE = "gemma4_assistant"

# Mirrors is_known_gemma4_assistant_pair() in gemma4_assistant_mtp.rs.
KNOWN_TARGETS = (
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
    "gemma-4-12b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
)
# Quant/format suffixes stripped when deriving the canonical pair id.
_QUANT_SUFFIXES = (
    "-4bit-ffn4",
    "-4bit",
    "-6bit",
    "-8bit",
    "-bf16",
    "-fp16",
    "-mlx",
    "-mlx-4bit",
)


# --------------------------------------------------------------------------- #
# HF cache / model-dir resolution (consistent with prepare_mtp_sidecar.py)
# --------------------------------------------------------------------------- #
def _repo_slug(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _latest_snapshot(model_dir: Path) -> Path | None:
    snaps_dir = model_dir / "snapshots"
    if not snaps_dir.exists():
        return None
    snapshots = sorted(snaps_dir.iterdir())
    return snapshots[-1] if snapshots else None


def _resolve_model_dir(ref: str, *, what: str) -> Path:
    candidate = Path(ref).expanduser()
    if candidate.exists() and (candidate / "config.json").exists():
        return candidate.resolve()
    if candidate.exists() and (candidate / "snapshots").exists():
        snap = _latest_snapshot(candidate)
        if snap:
            return snap.resolve()
    snap = _latest_snapshot(HF_CACHE / _repo_slug(ref))
    if snap is not None:
        return snap.resolve()
    sys.exit(
        f"ERROR: {what} model '{ref}' not found.\n"
        f"  Looked for a local dir and for {_repo_slug(ref)} in {HF_CACHE}.\n"
        f"  Download it first:  python3 scripts/download_model.py {ref}"
    )


# --------------------------------------------------------------------------- #
# Config helpers (handle Gemma's optional text_config nesting, like the loader)
# --------------------------------------------------------------------------- #
def _cfg_get(config: dict[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    text = config.get("text_config")
    if isinstance(text, dict) and key in text:
        return text[key]
    return None


def _model_id_leaf(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower()


def _has_safetensors(model_dir: Path) -> bool:
    return any(model_dir.glob("*.safetensors"))


def _looks_like_gemma(config: dict[str, Any]) -> bool:
    """True unless the config declares a clearly non-Gemma model_type.

    The runtime requires the target's model_family to be ``gemma4``; we guard
    against pointing --target at a non-Gemma checkpoint. We accept a missing
    model_type (return True) to avoid false rejects on unusual configs.
    """
    model_type = _cfg_get(config, "model_type")
    if not isinstance(model_type, str):
        return True
    return "gemma" in model_type.lower()


def _derive_canonical_target_id(ref: str) -> str:
    leaf = _model_id_leaf(ref)
    for suffix in sorted(_QUANT_SUFFIXES, key=len, reverse=True):
        if leaf.endswith(suffix):
            return leaf[: -len(suffix)]
    return leaf


def _derive_output_target_id(ref: str, canonical_target_id: str) -> str:
    leaf = _model_id_leaf(ref)
    if leaf == canonical_target_id or leaf.startswith(f"{canonical_target_id}-"):
        return leaf
    return canonical_target_id


def is_known_pair(assistant_model_id: str, target_model_id: str) -> bool:
    """Mirror is_known_gemma4_assistant_pair() in the runtime."""
    target = _model_id_leaf(target_model_id)
    assistant = _model_id_leaf(assistant_model_id)
    if target not in KNOWN_TARGETS:
        return False
    prefix = assistant[: -len("-assistant")] if assistant.endswith("-assistant") else None
    return prefix == target


def validate_assistant_arch(
    assistant_cfg: dict[str, Any], target_cfg: dict[str, Any]
) -> list[str]:
    """Pre-check every architectural rule parse_and_validate_contract() enforces.

    Returns a list of human-readable problems (empty list == valid). Each entry
    names the runtime disable reason it would otherwise trigger.
    """
    problems: list[str] = []

    a_vocab = _cfg_get(assistant_cfg, "vocab_size")
    t_vocab = _cfg_get(target_cfg, "vocab_size")
    if a_vocab is None or t_vocab is None:
        problems.append("InvalidConfig: vocab_size missing on target or assistant")
    elif a_vocab != t_vocab:
        problems.append(f"VocabMismatch: assistant {a_vocab} != target {t_vocab}")

    a_hidden = _cfg_get(assistant_cfg, "hidden_size")
    if a_hidden is None:
        problems.append("InvalidConfig: assistant hidden_size missing")

    backbone = assistant_cfg.get("backbone_hidden_size")
    t_hidden = _cfg_get(target_cfg, "hidden_size")
    if backbone is None or t_hidden is None:
        problems.append("InvalidConfig: backbone_hidden_size / target hidden_size missing")
    elif backbone != t_hidden:
        problems.append(
            f"UnsupportedAssistantConfig: backbone_hidden_size {backbone} != target hidden_size {t_hidden}"
        )

    n_layers = _cfg_get(assistant_cfg, "num_hidden_layers")
    n_kv_shared = _cfg_get(assistant_cfg, "num_kv_shared_layers")
    if n_layers is None or n_kv_shared is None:
        problems.append("UnsupportedAssistantConfig: num_hidden_layers / num_kv_shared_layers missing")
    elif n_kv_shared != n_layers:
        problems.append(
            f"UnsupportedAssistantConfig: num_kv_shared_layers {n_kv_shared} != num_hidden_layers {n_layers}"
        )

    if (_cfg_get(assistant_cfg, "hidden_size_per_layer_input") or 0) != 0:
        problems.append("UnsupportedAssistantConfig: hidden_size_per_layer_input must be 0")
    if (_cfg_get(assistant_cfg, "vocab_size_per_layer_input") or 0) != 0:
        problems.append("UnsupportedAssistantConfig: vocab_size_per_layer_input must be 0")
    if bool(_cfg_get(assistant_cfg, "enable_moe_block") or False):
        problems.append("UnsupportedAssistantConfig: enable_moe_block must be false")
    if bool(_cfg_get(assistant_cfg, "use_double_wide_mlp") or False):
        problems.append("UnsupportedAssistantConfig: use_double_wide_mlp must be false")

    return problems


def build_contract(
    *, target_model_id: str, assistant_model_id: str, assistant_rel_path: str, max_depth: int
) -> dict[str, Any]:
    """Build the ax.gemma4_assistant_mtp.v1 contract the runtime validates."""
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "gemma4_assistant",
        "target_model_id": target_model_id,
        "assistant_model_id": assistant_model_id,
        "assistant_path": assistant_rel_path,
        "max_depth": max_depth,
        "pairing": "exact",
    }


# --------------------------------------------------------------------------- #
# Filesystem packaging
# --------------------------------------------------------------------------- #
def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink a file (resolving symlinks to the cache blob), else copy."""
    if dst.exists():
        return
    try:
        os.link(os.path.realpath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _assemble_model_tree(src_dir: Path, out_dir: Path, *, copy_config: bool) -> None:
    """Materialize a model snapshot into out_dir (hardlinks; real config copy)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        dest = out_dir / item.name
        if item.is_dir():
            if not dest.exists():
                shutil.copytree(item, dest, copy_function=_safe_copy)
        elif item.name == "config.json" and copy_config:
            if not dest.exists():
                shutil.copy2(item, dest)  # real file so we can patch it safely
        else:
            _link_or_copy(item, dest)


def _reset_output_dir(out_dir: Path, *, target_dir: Path, assistant_dir: Path) -> None:
    """Remove a previous generated package so stale shards cannot mix in."""
    if out_dir == target_dir or out_dir == assistant_dir:
        sys.exit(f"ERROR: refusing to use source model dir as output: {out_dir}")
    if target_dir in out_dir.parents or assistant_dir in out_dir.parents:
        sys.exit(f"ERROR: refusing to write output inside a source model dir: {out_dir}")
    if out_dir == Path("/") or out_dir == Path.home() or out_dir == HF_CACHE:
        sys.exit(f"ERROR: refusing unsafe output dir: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _safe_copy(src: str, dst: str) -> None:
    try:
        os.link(os.path.realpath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _generate_manifest(model_dir: Path) -> None:
    """Generate and validate an AX model-manifest.json for a packaged subtree."""
    manifest_path = model_dir / "model-manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    root = _repo_root()
    commands = [
        ([str(root / "target" / "release" / "generate-manifest"), "--validate", str(model_dir)], root),
        ([str(root / "target" / "debug" / "generate-manifest"), "--validate", str(model_dir)], root),
        (["ax-engine-bench", "generate-manifest", str(model_dir), "--validate"], root),
        (
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "ax-engine-core",
                "--bin",
                "generate-manifest",
                "--",
                "--validate",
                str(model_dir),
            ],
            root,
        ),
    ]

    failures: list[str] = []
    for command, cwd in commands:
        executable = Path(command[0])
        if executable.is_absolute() and not executable.exists():
            continue
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError:
            continue
        if result.returncode == 0 and manifest_path.exists():
            return
        failures.append(
            "$ "
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout.strip()
            + "\nstderr:\n"
            + result.stderr.strip()
        )

    details = "\n\n".join(failures) if failures else "no manifest generator command was available"
    sys.exit(
        f"ERROR: failed to generate validated model-manifest.json for {model_dir}.\n{details}"
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def prepare(
    *,
    target: str,
    assistant: str,
    target_model_id: str | None,
    assistant_model_id: str | None,
    output: Path | None,
    max_depth: int,
) -> Path:
    target_dir = _resolve_model_dir(target, what="target")
    assistant_dir = _resolve_model_dir(assistant, what="assistant")
    print(f"Target model:    {target_dir}", flush=True)
    print(f"Assistant model: {assistant_dir}", flush=True)

    target_id = target_model_id or _derive_canonical_target_id(target)
    assistant_id = assistant_model_id or f"{target_id}-assistant"

    if not is_known_pair(assistant_id, target_id):
        sys.exit(
            f"ERROR: '{assistant_id}' / '{target_id}' is not a known Gemma 4 "
            f"assistant pair (PairMismatch).\n"
            f"  Known targets: {', '.join(KNOWN_TARGETS)}; assistant must be <target>-assistant.\n"
            f"  Override with --target-model-id / --assistant-model-id if your "
            f"local dirs carry quant suffixes."
        )

    if max_depth < 1:
        sys.exit(f"ERROR: --max-depth must be >= 1 (got {max_depth}).")

    target_cfg = json.loads((target_dir / "config.json").read_text())
    assistant_cfg = json.loads((assistant_dir / "config.json").read_text())

    # Fail-closed pre-checks that mirror runtime load requirements.
    if not _looks_like_gemma(target_cfg):
        sys.exit(
            f"ERROR: target model_type {_cfg_get(target_cfg, 'model_type')!r} is not "
            f"Gemma — the runtime requires a gemma4 target (NotGemma4Target)."
        )
    if not (target_dir / "tokenizer.json").exists():
        sys.exit(
            "ERROR: target tokenizer.json is missing — the runtime requires it for "
            "the assistant byte-identity check (TokenizerMismatch)."
        )
    if not _has_safetensors(target_dir):
        sys.exit(f"ERROR: no *.safetensors found in target dir {target_dir}.")
    if not _has_safetensors(assistant_dir):
        sys.exit(f"ERROR: no *.safetensors found in assistant dir {assistant_dir} (WeightLoadFailed).")

    # Output layout: a self-contained, ax-engine-bench --model-dir-ready entry.
    if output is not None:
        out_dir = output.expanduser().resolve()
    else:
        output_target_id = _derive_output_target_id(target, target_id)
        out_dir = HF_CACHE / f"models--ax-local--{output_target_id}-assistant-mtp" / "snapshots" / "v1"
    _reset_output_dir(out_dir, target_dir=target_dir, assistant_dir=assistant_dir)
    print(f"Output dir:      {out_dir}", flush=True)

    print("\nAssembling target tree...", flush=True)
    _assemble_model_tree(target_dir, out_dir, copy_config=False)

    print("Assembling assistant subtree...", flush=True)
    assistant_out = out_dir / "assistant"
    _assemble_model_tree(assistant_dir, assistant_out, copy_config=True)

    # Patch ax-engine packaging markers onto the assistant config. Only FILL
    # absent fields — never silently override a value the checkpoint declares,
    # since a conflict signals a genuine incompatibility (let validation flag it).
    patched = dict(assistant_cfg)
    patched["model_type"] = ASSISTANT_MODEL_TYPE
    backbone = _cfg_get(target_cfg, "hidden_size")
    if patched.get("backbone_hidden_size") is None and backbone is not None:
        patched["backbone_hidden_size"] = backbone
        print(f"  note: set backbone_hidden_size={backbone} (was absent)", flush=True)
    if _cfg_get(patched, "num_kv_shared_layers") is None:
        n_layers = _cfg_get(patched, "num_hidden_layers")
        if n_layers is not None:
            patched["num_kv_shared_layers"] = n_layers
            print(f"  note: set num_kv_shared_layers={n_layers} (was absent)", flush=True)
    (assistant_out / "config.json").write_text(json.dumps(patched, indent=2))
    print("  patched assistant config.json (model_type, backbone_hidden_size)", flush=True)

    # Guarantee the byte-identical tokenizer check: assistant shares the target
    # tokenizer by design, so copy the target's tokenizer into the assistant dir.
    # (Target tokenizer presence was verified above.)
    target_tok = out_dir / "tokenizer.json"
    dst_tok = assistant_out / "tokenizer.json"
    if dst_tok.exists():
        dst_tok.unlink()
    shutil.copy2(target_tok, dst_tok)
    print("  copied target tokenizer.json into assistant/ (byte-identity)", flush=True)

    # Pre-validate everything the runtime checks.
    print("\nValidating pair against runtime contract rules...", flush=True)
    problems = validate_assistant_arch(patched, target_cfg)
    if problems:
        for p in problems:
            print(f"  FAIL: {p}", flush=True)
        sys.exit("ERROR: assistant is not compatible; see failures above.")
    print("  OK: vocab, hidden, KV-sharing, no per-layer/MoE/double-wide.", flush=True)

    # Write the contract into the target root.
    contract = build_contract(
        target_model_id=target_id,
        assistant_model_id=assistant_id,
        assistant_rel_path="assistant",
        max_depth=max_depth,
    )
    contract_path = out_dir / CONTRACT_FILE
    contract_path.write_text(json.dumps(contract, indent=2))
    print(f"\nWrote {CONTRACT_FILE} -> {contract_path}", flush=True)

    print("\nGenerating AX manifests...", flush=True)
    _generate_manifest(out_dir)
    print(f"  OK: {out_dir / 'model-manifest.json'}", flush=True)
    _generate_manifest(assistant_out)
    print(f"  OK: {assistant_out / 'model-manifest.json'}", flush=True)

    print(f"\nGemma 4 assistant MTP package ready at:\n  {out_dir}", flush=True)
    print(
        "Serve with the gemma4 assistant MTP enabled "
        "(AX_MLX_GEMMA4_ASSISTANT_MTP=1 is the default).",
        flush=True,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target Gemma 4 model: local dir or org/name repo id (cached).",
    )
    parser.add_argument(
        "--assistant",
        required=True,
        help="Assistant/drafter model: local dir or org/name repo id (cached).",
    )
    parser.add_argument(
        "--target-model-id",
        default=None,
        help="Canonical target id for the pair check (default: derived, quant "
        "suffixes stripped). Must be one of the known Gemma 4 targets.",
    )
    parser.add_argument(
        "--assistant-model-id",
        default=None,
        help="Canonical assistant id (default: <target-model-id>-assistant).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir. Defaults to a synthetic HF cache entry "
        "models--ax-local--<target>-assistant-mtp/snapshots/v1/.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Draft depth written to the contract; runtime may cap or override it. "
        "Defaults to 2 to match the engine's shipped Gemma4 assistant-MTP depth "
        "(DEFAULT_GEMMA4_ASSISTANT_MTP_MAX_DEPTH); the runtime still clamps to "
        "min(contract, AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH).",
    )
    args = parser.parse_args()

    prepare(
        target=args.target,
        assistant=args.assistant,
        target_model_id=args.target_model_id,
        assistant_model_id=args.assistant_model_id,
        output=args.output,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
