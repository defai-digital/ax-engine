#!/usr/bin/env python3
"""Unit tests for scripts/parity_qwen38_flash_next.py.

Deterministic and offline: no network, no GPU, no mlx / mlx_vlm / ax_engine
imports, and no Session spawning. The harness is loaded via importlib from its
file path, same pattern as scripts/test_bench_single_client_mlx_serving.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("parity_qwen38_flash_next.py")
MODULE_SPEC = importlib.util.spec_from_file_location("parity_qwen38_flash_next", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
harness = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = harness
MODULE_SPEC.loader.exec_module(harness)

PIN_4BIT = harness.PINNED_REVISIONS["4bit"]


def _record(
    prompt_id: str = "p01",
    ids: list[int] | None = None,
    finish_reason: str | None = "stop",
    mtp: dict | None = None,
    step_count: int = 4,
) -> harness.GenerationRecord:
    token_ids = list(ids) if ids is not None else [1, 2, 3]
    return harness.GenerationRecord(
        prompt_id=prompt_id,
        prompt_token_ids=[10, 11],
        prompt_tokens=2,
        completion_tokens=len(token_ids),
        token_ids=token_ids,
        finish_reason=finish_reason,
        step_count=step_count,
        mtp=mtp or {},
    )


def _make_pack(
    root: Path,
    *,
    model_type: str = "qwen4_exp",
    architectures: list[str] | None = None,
    with_config: bool = True,
    with_tokenizer: bool = True,
    with_weights: bool = True,
    manifest: dict | None = None,
) -> Path:
    pack = root / "pack"
    pack.mkdir(exist_ok=True)
    if with_config:
        config = {
            "model_type": model_type,
            "architectures": (
                architectures
                if architectures is not None
                else ["Qwen4ExpForConditionalGeneration"]
            ),
        }
        (pack / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if with_tokenizer:
        (pack / "tokenizer.json").write_text("{}", encoding="utf-8")
    if with_weights:
        (pack / "model.safetensors").write_bytes(b"weights")
    if manifest is not None:
        (pack / "sha256-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack


def _minimal_evidence(status: str = "PASS", exit_code: int = 0) -> dict:
    doc = {key: None for key in harness.REQUIRED_EVIDENCE_KEYS}
    doc.update(
        {
            "schema_version": harness.SCHEMA_VERSION,
            "status": status,
            "exit_code": exit_code,
            "checks": {
                "token_parity_mlx_vs_engine_off": "PASS",
                "mtp_losslessness_engine_on_vs_off": "n/a",
                "mtp_engagement": "n/a",
                "prompt_tokens_gate": "PASS",
                "mlx_mtp_accept_rate_crosscheck": "SKIPPED",
            },
        }
    )
    return doc


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


def test_importing_harness_does_not_import_mlx_mlx_vlm_or_ax_engine() -> None:
    # Fresh-load the harness and diff sys.modules, so a host that already has
    # these modules loaded does not false-fail the isolation check.
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("parity_harness_fresh_load", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    gained = set(sys.modules) - before
    for name in ("mlx", "mlx.core", "mlx_vlm", "ax_engine", "transformers"):
        assert name not in gained, f"harness import pulled in {name}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = harness.parse_args([])
    assert args.mode == "full"
    assert args.max_tokens == 32
    assert args.stream_experts == "auto"
    assert harness.effective_mtp(args) == "both"
    assert args.pack_dir is None
    assert args.load_timeout_s == 3600.0
    assert args.request_timeout_s == 1800.0
    assert args.settle_s == 10.0
    assert args.stop_timeout_s == 120.0
    assert args.hash_weights is False


def test_parse_args_modes() -> None:
    for mode in ("engine", "mlx-dump", "engine-dump"):
        assert harness.parse_args(["--mode", mode]).mode == mode


def test_parse_args_rejects_non_auto_stream_experts(capsys) -> None:
    with pytest.raises(SystemExit):
        harness.parse_args(["--stream-experts", "on"])
    assert harness.main(["--stream-experts", "on"]) == 1
    assert harness.main(["--stream-experts", "off"]) == 1
    capsys.readouterr()


def test_parse_args_require_mlx_mtp_and_skip_probe() -> None:
    args = harness.parse_args(["--require-mlx-mtp", "--skip-mlx-mtp-probe"])
    assert args.require_mlx_mtp is True
    assert args.skip_mlx_mtp_probe is True


def test_hidden_mlx_worker_flag_requires_paths(capsys) -> None:
    assert harness.main(["--internal-mlx-worker"]) == 1
    assert "worker-job-json" in capsys.readouterr().err


def test_hidden_engine_worker_flag_requires_paths(capsys) -> None:
    assert harness.main(["--internal-engine-worker"]) == 1
    assert "worker-job-json" in capsys.readouterr().err


def test_main_help_exits_zero(capsys) -> None:
    assert harness.main(["--help"]) == 0
    capsys.readouterr()


def test_validate_mode_args_rejects_mtp_on_in_full(capsys) -> None:
    args = harness.parse_args(["--mode", "full", "--mtp", "on"])
    with pytest.raises(harness.UsageError):
        harness._validate_mode_args(args)
    assert harness.main(["--mode", "full", "--mtp", "on"]) == 1
    capsys.readouterr()


def test_validate_mode_args_rejects_mtp_both_in_engine_dump() -> None:
    args = harness.parse_args(["--mode", "engine-dump", "--mtp", "both"])
    with pytest.raises(harness.UsageError):
        harness._validate_mode_args(args)


# ---------------------------------------------------------------------------
# Prompt set integrity
# ---------------------------------------------------------------------------


def test_prompt_set_count_and_unique_ids() -> None:
    prompts = harness.default_prompts()
    assert len(prompts) == 7
    ids = [spec.prompt_id for spec in prompts]
    assert len(set(ids)) == 7
    for index, spec in enumerate(prompts, start=1):
        assert spec.prompt_id.startswith(f"p0{index}_")


def test_prompt_set_covers_required_categories() -> None:
    categories = {spec.category for spec in harness.default_prompts()}
    assert categories == {
        "short_factual_en",
        "code_generation",
        "cjk",
        "long_instruction",
        "punct_numbers",
        "mixed_en_cjk",
        "json_adjacent",
    }


def test_prompt_set_rejects_chat_template_markers() -> None:
    for spec in harness.default_prompts():
        for marker in harness.CHAT_TEMPLATE_MARKERS:
            assert marker not in spec.text


def test_long_instruction_is_400_to_800_chars() -> None:
    spec = next(s for s in harness.default_prompts() if s.prompt_id == "p04_long_instruction")
    assert 400 <= len(spec.text) <= 800


def test_prompts_have_no_surrounding_whitespace() -> None:
    for spec in harness.default_prompts():
        assert spec.text
        assert spec.text == spec.text.strip()


def test_validate_prompt_specs_rejects_marker_in_custom_prompt() -> None:
    bad = harness.PromptSpec("x1_custom", "custom", "hello <|im_start|> world")
    with pytest.raises(harness.UsageError):
        harness.validate_prompt_specs([bad])


def test_validate_prompt_specs_rejects_whitespace_padded_text() -> None:
    bad = harness.PromptSpec("x2_custom", "custom", " padded ")
    with pytest.raises(harness.UsageError):
        harness.validate_prompt_specs([bad])


# ---------------------------------------------------------------------------
# Comparison (locks the seven worked examples from the design doc)
# ---------------------------------------------------------------------------


def test_compare_equal_sequences() -> None:
    result = harness.compare_token_ids([1, 2, 3], [1, 2, 3])
    assert result.equal is True
    assert result.first_divergence_index is None
    assert result.window is None


def test_compare_value_mismatch_reports_index() -> None:
    result = harness.compare_token_ids([1, 2, 3], [1, 9, 3])
    assert result.equal is False
    assert result.first_divergence_index == 1


def test_compare_length_mismatch_left_longer_is_divergence_at_min_len() -> None:
    result = harness.compare_token_ids([1, 2, 3], [1, 2])
    assert result.equal is False
    assert result.first_divergence_index == 2


def test_compare_length_mismatch_right_longer_is_divergence_at_min_len() -> None:
    result = harness.compare_token_ids([1, 2], [1, 2, 3])
    assert result.equal is False
    assert result.first_divergence_index == 2


def test_compare_empty_vs_nonempty() -> None:
    result = harness.compare_token_ids([], [1])
    assert result.equal is False
    assert result.first_divergence_index == 0


def test_compare_empty_vs_empty() -> None:
    result = harness.compare_token_ids([], [])
    assert result.equal is True
    assert result.first_divergence_index is None


def test_compare_window_at_start_is_clamped() -> None:
    left = [0] * 20
    right = [99] + [0] * 19
    result = harness.compare_token_ids(left, right, window=8)
    assert result.first_divergence_index == 0
    assert result.window is not None
    assert result.window.start == 0
    assert result.window.end == 9
    assert result.window.left_ids == [0] * 9
    assert result.window.right_ids == [99] + [0] * 8


def test_compare_window_at_end_is_clamped() -> None:
    left = list(range(20))
    right = list(range(19)) + [99]
    result = harness.compare_token_ids(left, right, window=8)
    assert result.first_divergence_index == 19
    assert result.window is not None
    assert result.window.start == 11
    assert result.window.end == 20


def test_compare_decode_fn_used_for_window_text() -> None:
    result = harness.compare_token_ids(
        [1, 2, 3],
        [1, 9, 3],
        decode_fn=lambda ids: ",".join(map(str, ids)),
    )
    assert result.window is not None
    assert result.window.left_text == "1,2,3"
    assert result.window.right_text == "1,9,3"
    no_decode = harness.compare_token_ids([1, 2, 3], [1, 9, 3])
    assert no_decode.window is not None
    assert no_decode.window.left_text is None


def test_compare_generations_finish_reason_mismatch_is_warning_not_raise() -> None:
    left = [_record(ids=[5, 6], finish_reason="stop")]
    right = [_record(ids=[5, 6], finish_reason="length")]
    report = harness.compare_generations(
        left,
        right,
        compare="mlx_vs_engine_off",
        left_name="mlx",
        right_name="engine_mtp_off",
    )
    assert report["equal"] is True
    assert report["per_prompt"][0]["finish_reason_warning"] is True
    assert report["warnings"]


def test_compare_generations_length_vs_max_output_tokens_is_same_bucket() -> None:
    left = [_record(ids=[5, 6], finish_reason="length")]
    right = [_record(ids=[5, 6], finish_reason="max_output_tokens")]
    report = harness.compare_generations(
        left,
        right,
        compare="mlx_vs_engine_off",
        left_name="mlx",
        right_name="engine_mtp_off",
    )
    assert report["equal"] is True
    assert report["per_prompt"][0]["finish_reason_warning"] is False
    assert "warnings" not in report


def test_compare_generations_raises_divergence_after_all_prompts() -> None:
    left = [_record("p01", [1, 2]), _record("p02", [3, 4])]
    right = [_record("p01", [1, 9]), _record("p02", [3, 8])]
    with pytest.raises(harness.DivergenceError) as excinfo:
        harness.compare_generations(
            left,
            right,
            compare="engine_on_vs_engine_off",
            left_name="engine_mtp_on",
            right_name="engine_mtp_off",
        )
    details = excinfo.value.details
    assert details["compare"] == "engine_on_vs_engine_off"
    assert details["prompt_id"] == "p01"
    assert details["first_divergence_index"] == 1
    assert details["window"] is not None
    assert len(details["report"]["per_prompt"]) == 2
    assert all(not entry["equal"] for entry in details["report"]["per_prompt"])


# ---------------------------------------------------------------------------
# Gates and telemetry
# ---------------------------------------------------------------------------


def test_assert_prompt_tokens_identity_passes_on_exact() -> None:
    harness.assert_prompt_tokens_identity([11, 22, 33], [11, 22, 33], prompt_id="p01")


def test_assert_prompt_tokens_identity_mismatch_is_infra() -> None:
    with pytest.raises(harness.InfraError) as excinfo:
        harness.assert_prompt_tokens_identity([11, 22], [11, 22, 33], prompt_id="p01")
    assert excinfo.value.details["subtype"] == "tokenizer"


def test_assert_prompt_tokens_identity_same_length_different_ids_is_infra() -> None:
    with pytest.raises(harness.InfraError):
        harness.assert_prompt_tokens_identity([11, 22, 44], [11, 22, 33], prompt_id="p01")


def test_require_mtp_engaged_zero_is_infra() -> None:
    records = [
        _record(
            mtp={
                "ax_mtp_draft_tokens": 0,
                "ax_mtp_requested": 1,
                "ax_mlx_mtp_model_policy": 10,
            }
        )
    ]
    with pytest.raises(harness.InfraError):
        harness.require_mtp_engaged(records)


def test_require_mtp_engaged_partial_missing_key_is_infra() -> None:
    records = [
        _record(
            "p01",
            mtp={
                "ax_mtp_draft_tokens": 4,
                "ax_mtp_requested": 1,
                "ax_mlx_mtp_model_policy": 10,
            },
        ),
        _record("p02", mtp={"ax_mtp_draft_tokens": 3}),  # missing requested/policy
    ]
    with pytest.raises(harness.InfraError) as excinfo:
        harness.require_mtp_engaged(records)
    assert excinfo.value.details["subtype"] == "mtp_telemetry"


def test_require_mtp_engaged_non_numeric_telemetry_is_infra() -> None:
    records = [
        _record(
            mtp={
                "ax_mtp_draft_tokens": "four",
                "ax_mtp_requested": 1,
                "ax_mlx_mtp_model_policy": 10,
            }
        )
    ]
    with pytest.raises(harness.InfraError):
        harness.require_mtp_engaged(records)


def test_require_mtp_engaged_missing_key_is_infra() -> None:
    records = [_record(mtp={}), _record(mtp={})]
    with pytest.raises(harness.InfraError):
        harness.require_mtp_engaged(records)


def test_require_mtp_engaged_sums_crossover_draft_tokens() -> None:
    records = [
        _record(
            "p01",
            mtp={
                "ax_mtp_draft_tokens": 4,
                "ax_mtp_accepted_tokens": 3,
                "ax_mtp_decode_steps": 3,
                "ax_mtp_direct_fallback_steps": 1,
                "ax_mtp_requested": 1,
                "ax_mlx_mtp_model_policy": 10,
            },
        ),
        _record(
            "p02",
            mtp={
                "ax_mtp_draft_tokens": 6,
                "ax_mtp_accepted_tokens": 3,
                "ax_mtp_requested": 1,
                "ax_mlx_mtp_model_policy": 10,
            },
        ),
    ]
    summary = harness.require_mtp_engaged(records)
    assert summary["ax_mtp_draft_tokens"] == 10
    assert summary["ax_mtp_accepted_tokens"] == 6
    assert summary["ax_mtp_decode_steps"] == 3
    assert summary["ax_mtp_direct_fallback_steps"] == 1
    assert summary["accept_rate"] == pytest.approx(0.6)
    assert summary["ax_mtp_requested"] == 1
    assert summary["ax_mlx_mtp_model_policy"] == 10


def test_classify_finish_reason_cancel_is_infra() -> None:
    for reason in ("cancel", "cancelled", "content_filter", "error", None):
        with pytest.raises(harness.InfraError):
            harness.classify_finish_reason(reason)


def test_finish_reason_max_output_tokens_allowed() -> None:
    assert harness.classify_finish_reason("stop") == "stop"
    assert harness.classify_finish_reason("length") == "length"
    assert harness.classify_finish_reason("max_output_tokens") == "max_output_tokens"


def test_normalize_finish_reason_buckets_length_variants() -> None:
    assert harness.normalize_finish_reason_bucket("stop") == "stop"
    assert harness.normalize_finish_reason_bucket("length") == "length"
    assert harness.normalize_finish_reason_bucket("max_output_tokens") == "length"


def test_accept_proxy_from_step_count() -> None:
    assert harness.compute_accept_proxy(4, 6) == {
        "decode_steps_proxy": 3,
        "tokens_per_decode_step": 2.0,
    }
    assert harness.compute_accept_proxy(1, 6)["tokens_per_decode_step"] is None
    assert harness.compute_accept_proxy(0, 6)["decode_steps_proxy"] == 0


def test_extract_mtp_decisions_filters_known_keys() -> None:
    decisions = {
        "ax_mtp_draft_tokens": 4,
        "ax_ngram_accepted_tokens": 1,
        "unrelated_key": 9,
    }
    assert harness.extract_mtp_decisions(decisions) == {
        "ax_mtp_draft_tokens": 4,
        "ax_ngram_accepted_tokens": 1,
    }
    assert harness.extract_mtp_decisions(None) == {}


# ---------------------------------------------------------------------------
# Snapshot classify / resolve / manifest
# ---------------------------------------------------------------------------


def test_classify_snapshot_missing_dir_is_skip(tmp_path) -> None:
    with pytest.raises(harness.SkipError):
        harness.classify_snapshot(tmp_path / "missing")


def test_classify_snapshot_missing_config_is_skip(tmp_path) -> None:
    pack = _make_pack(tmp_path, with_config=False)
    with pytest.raises(harness.SkipError):
        harness.classify_snapshot(pack)


def test_classify_snapshot_missing_tokenizer_is_skip(tmp_path) -> None:
    pack = _make_pack(tmp_path, with_tokenizer=False)
    with pytest.raises(harness.SkipError):
        harness.classify_snapshot(pack)


def test_classify_snapshot_missing_weights_is_skip(tmp_path) -> None:
    pack = _make_pack(tmp_path, with_weights=False)
    with pytest.raises(harness.SkipError):
        harness.classify_snapshot(pack)


def test_classify_snapshot_aria2_is_skip(tmp_path) -> None:
    pack = _make_pack(tmp_path)
    (pack / "model-00002-of-00033.safetensors.aria2").write_bytes(b"")
    with pytest.raises(harness.SkipError) as excinfo:
        harness.classify_snapshot(pack)
    assert "aria2" in str(excinfo.value)


def test_classify_snapshot_wrong_model_type_is_infra(tmp_path) -> None:
    pack = _make_pack(tmp_path, model_type="qwen3_dense", architectures=["Qwen3ForCausalLM"])
    with pytest.raises(harness.InfraError):
        harness.classify_snapshot(pack)


def test_classify_snapshot_returns_config(tmp_path) -> None:
    pack = _make_pack(tmp_path)
    config = harness.classify_snapshot(pack)
    assert config["model_type"] == "qwen4_exp"


def test_resolve_pack_dir_missing_is_skip(tmp_path) -> None:
    args = harness.parse_args(["--mode", "full", "--pack-dir", str(tmp_path / "nope")])
    pack_dir, tag = harness.resolve_pack_dir(args)
    assert tag == "4bit"
    with pytest.raises(harness.SkipError):
        harness.classify_snapshot(pack_dir)


def test_resolve_pack_dir_defaults_by_mode() -> None:
    pack_dir, tag = harness.resolve_pack_dir(harness.parse_args(["--mode", "full"]))
    assert (pack_dir, tag) == (harness.DEFAULT_PACK_4BIT, "4bit")
    pack_dir, tag = harness.resolve_pack_dir(harness.parse_args(["--mode", "engine"]))
    assert (pack_dir, tag) == (harness.DEFAULT_PACK_6BIT, "6bit")
    pack_dir, tag = harness.resolve_pack_dir(
        harness.parse_args(["--mode", "engine", "--pack-dir-6bit", "/tmp/x6bit"])
    )
    assert (pack_dir, tag) == (Path("/tmp/x6bit"), "6bit")
    pack_dir, tag = harness.resolve_pack_dir(
        harness.parse_args(["--mode", "engine", "--pack-dir", "/tmp/qwen38-flash-next-6bit"])
    )
    assert tag == "6bit"


def test_sha256_manifest_revision_mismatch_is_infra(tmp_path) -> None:
    manifest = {"repo": "AutomatosX/x", "revision": "0" * 40, "files": []}
    pack = _make_pack(tmp_path, manifest=manifest)
    with pytest.raises(harness.InfraError) as excinfo:
        harness.verify_sha256_manifest(pack, "4bit")
    assert excinfo.value.details["subtype"] == "provenance"


def test_sha256_manifest_size_mismatch_is_infra(tmp_path) -> None:
    manifest = {
        "repo": "AutomatosX/x",
        "revision": PIN_4BIT,
        "files": [{"file": "model.safetensors", "size": 999, "sha256": None}],
    }
    pack = _make_pack(tmp_path, manifest=manifest)
    with pytest.raises(harness.InfraError):
        harness.verify_sha256_manifest(pack, "4bit")


def test_sha256_manifest_missing_listed_file_is_infra(tmp_path) -> None:
    manifest = {
        "repo": "AutomatosX/x",
        "revision": PIN_4BIT,
        "files": [{"file": "missing.json", "size": 1, "sha256": None}],
    }
    pack = _make_pack(tmp_path, manifest=manifest)
    with pytest.raises(harness.InfraError):
        harness.verify_sha256_manifest(pack, "4bit")


def test_sha256_manifest_absent_is_infra(tmp_path) -> None:
    pack = _make_pack(tmp_path)
    with pytest.raises(harness.InfraError) as excinfo:
        harness.verify_sha256_manifest(pack, "4bit")
    assert excinfo.value.details["subtype"] == "provenance"


def test_sha256_manifest_verifies_hashes_and_null_sha_sizes(tmp_path) -> None:
    pack = _make_pack(tmp_path)
    config_sha = hashlib.sha256((pack / "config.json").read_bytes()).hexdigest()
    manifest = {
        "repo": "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP",
        "revision": PIN_4BIT,
        "files": [
            {"file": "config.json", "size": 0, "sha256": config_sha},
            {"file": "model.safetensors", "size": len(b"weights"), "sha256": None},
        ],
    }
    (pack / "sha256-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    info = harness.verify_sha256_manifest(pack, "4bit")
    assert info["present"] is True
    assert info["complete"] is False  # the null sha256 on the weight shard
    assert info["sha256_checked"] == 1
    assert info["size_checked"] == 1


def test_main_skip_exit_code_2(tmp_path, capsys) -> None:
    code = harness.main(["--mode", "full", "--pack-dir", str(tmp_path)])
    assert code == 2
    assert "SKIP" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Evidence schema / filename
# ---------------------------------------------------------------------------


def test_evidence_filename_shape() -> None:
    path = harness.evidence_filename(pack_tag="4bit", mode="full", git_sha="a" * 40)
    assert path.name == f"qwen38-flash-next-parity-4bit-full-{'a' * 8}.json"
    assert path.parent.name == "results"
    assert path.parent.parent.name == "benchmarks"
    unknown = harness.evidence_filename(pack_tag="6bit", mode="engine", git_sha="unknown")
    assert unknown.name == "qwen38-flash-next-parity-6bit-engine-unknown.json"
    non_hex = harness.evidence_filename(pack_tag="4bit", mode="full", git_sha="not-a-sha")
    assert "unknown" in non_hex.name


def test_validate_evidence_accepts_minimal_pass_doc() -> None:
    assert harness.validate_evidence(_minimal_evidence()) == []


def test_validate_evidence_rejects_wrong_schema_version() -> None:
    doc = _minimal_evidence()
    doc["schema_version"] = "ax.other.v9"
    assert harness.validate_evidence(doc)


def test_validate_evidence_rejects_status_exit_mismatch() -> None:
    doc = _minimal_evidence(status="PASS", exit_code=3)
    assert harness.validate_evidence(doc)


def test_validate_evidence_rejects_bad_check_value() -> None:
    doc = _minimal_evidence()
    doc["checks"]["mtp_engagement"] = "MAYBE"
    assert harness.validate_evidence(doc)


def test_hash_contract_files_hashes_small_files_not_missing_optional(tmp_path) -> None:
    pack = _make_pack(tmp_path)
    result = harness.hash_contract_files(pack)
    assert set(result["contract_sha256"]) == {"config.json", "tokenizer.json"}
    assert "axquant_mtp_sidecar_manifest.json" not in result["contract_sha256"]
    assert result["safetensors_inventory"] == [
        {"name": "model.safetensors", "size_bytes": len(b"weights")}
    ]
    assert result["weights_hashed"] is False
    hashed = harness.hash_contract_files(pack, hash_weights=True)
    expected = hashlib.sha256(b"weights").hexdigest()
    assert hashed["safetensors_inventory"][0]["sha256"] == expected
    assert hashed["weights_hashed"] is True


def test_write_evidence_roundtrip_cjk_preserved(tmp_path) -> None:
    doc = _minimal_evidence()
    doc["failure"] = {"message": "注意力機制"}
    out = tmp_path / "nested" / "evidence.json"
    harness.write_evidence(out, doc)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["failure"]["message"] == "注意力機制"
    assert harness.validate_evidence(loaded) == []


# ---------------------------------------------------------------------------
# Engine worker env / kwargs (no spawning)
# ---------------------------------------------------------------------------


def test_build_engine_worker_env_mtp_off_sets_ax_no_spec_not_cert() -> None:
    env = harness.build_engine_worker_env(False, base_env={})
    assert env["AX_NO_SPEC"] == "1"
    assert harness.CERT_ENV not in env


def test_build_engine_worker_env_mtp_on_sets_cert_not_ax_no_spec() -> None:
    env = harness.build_engine_worker_env(True, base_env={})
    assert env[harness.CERT_ENV] == "1"
    assert "AX_NO_SPEC" not in env


def test_build_engine_worker_env_strips_forbidden_vars_from_base() -> None:
    base = {"AX_NO_SPEC": "1", harness.CERT_ENV: "1", harness.MTP_FORCE_ENV: "1"}
    off_env = harness.build_engine_worker_env(False, base_env=base)
    assert harness.CERT_ENV not in off_env
    assert harness.MTP_FORCE_ENV not in off_env
    on_env = harness.build_engine_worker_env(True, base_env=base)
    assert "AX_NO_SPEC" not in on_env
    assert harness.MTP_FORCE_ENV not in on_env


def test_prefix_cache_kill_env_always_set() -> None:
    for mtp_on in (False, True):
        env = harness.build_engine_worker_env(mtp_on, base_env={})
        for key, value in harness.PREFIX_CACHE_KILL_ENV.items():
            assert env[key] == value


def test_build_engine_worker_env_does_not_mutate_parent_environ(monkeypatch) -> None:
    monkeypatch.delenv(harness.CERT_ENV, raising=False)
    monkeypatch.delenv("AX_NO_SPEC", raising=False)
    harness.build_engine_worker_env(True)
    harness.build_engine_worker_env(False)
    assert harness.CERT_ENV not in os.environ
    assert "AX_NO_SPEC" not in os.environ


def test_build_generate_kwargs_has_no_input_text_and_temp_zero() -> None:
    kwargs = harness.build_generate_kwargs([11, 22], 32)
    assert kwargs["input_tokens"] == [11, 22]
    assert kwargs["max_output_tokens"] == 32
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 0
    assert kwargs["repetition_penalty"] == 1.0
    assert kwargs["no_repeat_ngram_size"] == 0
    assert kwargs["seed"] == 0
    assert kwargs["deterministic"] is True
    assert "input_text" not in kwargs
    assert "stop_sequences" not in kwargs


def test_build_session_kwargs_targets_qwen4_exp_mlx() -> None:
    kwargs = harness.build_session_kwargs(Path("/pack"))
    assert kwargs == {
        "mlx": True,
        "mlx_model_artifacts_dir": "/pack",
        "model_id": "qwen4_exp",
        "deterministic": True,
    }


# ---------------------------------------------------------------------------
# Worker dump validation (fail closed; empty-vs-empty must never reach compare)
# ---------------------------------------------------------------------------


def _dump(
    prompt_ids=("p01", "p02"),
    *,
    kind="mlx",
    schema=None,
    ok=True,
    mtp_on=None,
) -> dict:
    default_schema = harness.MLX_DUMP_SCHEMA if kind == "mlx" else harness.ENGINE_DUMP_SCHEMA
    dump = {
        "schema": schema if schema is not None else default_schema,
        "kind": kind,
        "ok": ok,
        "generations": [
            {"prompt_id": pid, "token_ids": [1, 2], "prompt_token_ids": [7, 8]}
            for pid in prompt_ids
        ],
    }
    if mtp_on is not None:
        dump["mtp_on"] = mtp_on
    return dump


def test_validate_worker_dump_accepts_complete_dump() -> None:
    harness.validate_worker_dump(_dump(), kind="mlx", expected_prompt_ids=["p01", "p02"])


def test_validate_worker_dump_rejects_wrong_schema() -> None:
    with pytest.raises(harness.InfraError):
        harness.validate_worker_dump(
            _dump(schema="ax.other.v1"), kind="mlx", expected_prompt_ids=["p01", "p02"]
        )


def test_validate_worker_dump_rejects_ok_not_strict_true() -> None:
    with pytest.raises(harness.InfraError):
        harness.validate_worker_dump(
            _dump(ok=1), kind="mlx", expected_prompt_ids=["p01", "p02"]
        )


def test_validate_worker_dump_rejects_missing_generation() -> None:
    with pytest.raises(harness.InfraError):
        harness.validate_worker_dump(
            _dump(("p01",)), kind="mlx", expected_prompt_ids=["p01", "p02"]
        )


def test_validate_worker_dump_rejects_extra_generation() -> None:
    with pytest.raises(harness.InfraError):
        harness.validate_worker_dump(
            _dump(("p01", "p02", "p03")), kind="mlx", expected_prompt_ids=["p01", "p02"]
        )


def test_validate_worker_dump_rejects_reordered_generations() -> None:
    with pytest.raises(harness.InfraError):
        harness.validate_worker_dump(
            _dump(("p02", "p01")), kind="mlx", expected_prompt_ids=["p01", "p02"]
        )


def test_validate_worker_dump_rejects_missing_token_ids() -> None:
    dump = _dump()
    del dump["generations"][0]["token_ids"]
    with pytest.raises(harness.InfraError):
        harness.validate_worker_dump(dump, kind="mlx", expected_prompt_ids=["p01", "p02"])


def test_validate_worker_dump_rejects_non_int_token_ids() -> None:
    dump = _dump()
    dump["generations"][1]["token_ids"] = [1, "two"]
    with pytest.raises(harness.InfraError):
        harness.validate_worker_dump(dump, kind="mlx", expected_prompt_ids=["p01", "p02"])


def test_validate_worker_dump_rejects_mtp_on_flag_mismatch() -> None:
    dump = _dump(kind="engine", mtp_on=False)
    with pytest.raises(harness.InfraError):
        harness.validate_worker_dump(
            dump,
            kind="engine",
            expected_prompt_ids=["p01", "p02"],
            expected_mtp_on=True,
        )


def test_from_dict_missing_token_ids_is_infra() -> None:
    with pytest.raises(harness.InfraError):
        harness.GenerationRecord.from_dict({"prompt_id": "p01", "prompt_token_ids": [1]})


def test_from_dict_non_int_token_ids_is_infra() -> None:
    with pytest.raises(harness.InfraError):
        harness.GenerationRecord.from_dict(
            {"prompt_id": "p01", "token_ids": ["a"], "prompt_token_ids": [1]}
        )


def test_from_dict_roundtrip_complete_record() -> None:
    record = _record("p01", [1, 2, 3], mtp={"ax_mtp_draft_tokens": 2})
    parsed = harness.GenerationRecord.from_dict(record.to_dict())
    assert parsed.token_ids == [1, 2, 3]
    assert parsed.prompt_token_ids == [10, 11]
    assert parsed.mtp["ax_mtp_draft_tokens"] == 2


# ---------------------------------------------------------------------------
# MTP-off baseline gate
# ---------------------------------------------------------------------------


def _off_record(prompt_id: str = "p01", mtp: dict | None = None) -> harness.GenerationRecord:
    return _record(
        prompt_id,
        mtp=mtp
        if mtp is not None
        else {"ax_mtp_draft_tokens": 0, "ax_mtp_requested": 0, "ax_mlx_mtp_model_policy": 11},
    )


def test_require_mtp_off_baseline_clean_passes_on_zero() -> None:
    summary = harness.require_mtp_off_baseline_clean([_off_record("p01"), _off_record("p02")])
    assert summary["baseline_clean"] is True
    assert summary["ax_mtp_draft_tokens"] == 0
    assert summary["ax_ngram_draft_tokens"] == 0


def test_require_mtp_off_baseline_clean_draft_positive_is_infra() -> None:
    records = [_off_record("p01", mtp={"ax_mtp_draft_tokens": 3, "ax_mtp_requested": 1})]
    with pytest.raises(harness.InfraError) as excinfo:
        harness.require_mtp_off_baseline_clean(records)
    assert excinfo.value.details["subtype"] == "mtp_off_contaminated"


def test_require_mtp_off_baseline_clean_ngram_positive_is_infra() -> None:
    records = [_off_record("p01", mtp={"ax_mtp_requested": 0, "ax_ngram_draft_tokens": 2})]
    with pytest.raises(harness.InfraError):
        harness.require_mtp_off_baseline_clean(records)


def test_require_mtp_off_baseline_clean_no_telemetry_is_infra() -> None:
    records = [_record("p01", mtp={}), _record("p02", mtp={})]
    with pytest.raises(harness.InfraError) as excinfo:
        harness.require_mtp_off_baseline_clean(records)
    assert excinfo.value.details["subtype"] == "mtp_telemetry"


def test_require_mtp_off_baseline_clean_requested_truthy_is_infra() -> None:
    records = [_off_record("p01", mtp={"ax_mtp_requested": 1, "ax_mtp_draft_tokens": 0})]
    with pytest.raises(harness.InfraError):
        harness.require_mtp_off_baseline_clean(records)


# ---------------------------------------------------------------------------
# Env scrubbing (skip-telemetry must never leak into workers)
# ---------------------------------------------------------------------------


def test_engine_worker_env_pins_skip_telemetry_off() -> None:
    for mtp_on in (False, True):
        env = harness.build_engine_worker_env(
            mtp_on, base_env={harness.SKIP_TELEMETRY_ENV: "1"}
        )
        assert env[harness.SKIP_TELEMETRY_ENV] == "0"


def test_mlx_worker_env_scrubs_spec_vars_and_pins_telemetry() -> None:
    base = {
        "AX_NO_SPEC": "1",
        harness.CERT_ENV: "1",
        harness.MTP_FORCE_ENV: "1",
        harness.SKIP_TELEMETRY_ENV: "1",
    }
    env = harness.build_mlx_worker_env(base_env=base)
    assert "AX_NO_SPEC" not in env
    assert harness.CERT_ENV not in env
    assert harness.MTP_FORCE_ENV not in env
    assert env[harness.SKIP_TELEMETRY_ENV] == "0"


# ---------------------------------------------------------------------------
# mlx lookahead truncation helper (pure function; no mlx import needed)
# ---------------------------------------------------------------------------


def test_truncate_lookahead_tokens_drops_trailing_token() -> None:
    assert harness.truncate_lookahead_tokens([1, 2, 3, 4, 5], 4) == [1, 2, 3, 4]
    assert harness.truncate_lookahead_tokens([1, 2], 4) == [1, 2]
    assert harness.truncate_lookahead_tokens([], 4) == []


# ---------------------------------------------------------------------------
# compare_generations set guards
# ---------------------------------------------------------------------------


def test_compare_generations_empty_left_is_infra() -> None:
    with pytest.raises(harness.InfraError):
        harness.compare_generations(
            [], [_record()], compare="c", left_name="l", right_name="r"
        )


def test_compare_generations_extra_right_prompt_is_infra() -> None:
    with pytest.raises(harness.InfraError):
        harness.compare_generations(
            [_record("p01")],
            [_record("p01"), _record("p02")],
            compare="c",
            left_name="l",
            right_name="r",
        )


# ---------------------------------------------------------------------------
# Parent-side prompt-token gate re-verification
# ---------------------------------------------------------------------------


def _encoded(prompt_id: str = "p01", ids=(11, 22, 33)) -> list[dict]:
    return [
        {
            "prompt_id": prompt_id,
            "input_ids": list(ids),
            "category": "c",
            "text": "t",
            "n_chars": 1,
            "n_prompt_tokens": len(ids),
            "text_sha256": "x",
        }
    ]


def _gate_record(
    prompt_id: str = "p01",
    ids=(11, 22, 33),
    count: int | None = None,
) -> harness.GenerationRecord:
    token_ids = list(ids)
    return harness.GenerationRecord(
        prompt_id=prompt_id,
        prompt_token_ids=token_ids,
        prompt_tokens=len(token_ids) if count is None else count,
        completion_tokens=1,
        token_ids=[5],
        finish_reason="stop",
    )


def test_verify_prompt_token_gate_passes_on_exact_echo() -> None:
    harness.verify_prompt_token_gate([_gate_record()], _encoded())


def test_verify_prompt_token_gate_id_mismatch_is_infra() -> None:
    with pytest.raises(harness.InfraError) as excinfo:
        harness.verify_prompt_token_gate([_gate_record(ids=[11, 22, 44])], _encoded())
    assert excinfo.value.details["subtype"] == "tokenizer"


def test_verify_prompt_token_gate_count_mismatch_is_infra() -> None:
    with pytest.raises(harness.InfraError):
        harness.verify_prompt_token_gate([_gate_record(count=2)], _encoded())


def test_verify_prompt_token_gate_unknown_prompt_is_infra() -> None:
    with pytest.raises(harness.InfraError):
        harness.verify_prompt_token_gate([_gate_record(prompt_id="p99")], _encoded())


# ---------------------------------------------------------------------------
# PASS must be earned: _finalize and validate_evidence enforcement
# ---------------------------------------------------------------------------


def test_finalize_pass_with_unearned_checks_becomes_infra(tmp_path) -> None:
    doc = _minimal_evidence()
    doc["checks"]["mtp_engagement"] = "SKIPPED"
    out = tmp_path / "evidence.json"
    code = harness._finalize(doc, out, status="PASS", started_mono=time.monotonic())
    assert code == 3
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["status"] == "INFRA"
    assert written["exit_code"] == 3
    assert written["failure"]["subtype"] == "checks_incomplete"


def test_finalize_pass_with_schema_problems_becomes_infra(tmp_path) -> None:
    doc = _minimal_evidence()
    del doc["hardware"]
    out = tmp_path / "evidence.json"
    code = harness._finalize(doc, out, status="PASS", started_mono=time.monotonic())
    assert code == 3
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["status"] == "INFRA"
    assert written["failure"]["subtype"] == "evidence_schema"


def test_finalize_pass_with_earned_checks_stays_pass(tmp_path) -> None:
    doc = _minimal_evidence()
    out = tmp_path / "evidence.json"
    code = harness._finalize(doc, out, status="PASS", started_mono=time.monotonic())
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["status"] == "PASS"
    assert harness.validate_evidence(written) == []


def test_validate_evidence_rejects_pass_with_skipped_gated_check() -> None:
    doc = _minimal_evidence()
    doc["checks"]["prompt_tokens_gate"] = "SKIPPED"
    assert harness.validate_evidence(doc)


def test_validate_evidence_rejects_pass_with_missing_gated_check() -> None:
    doc = _minimal_evidence()
    del doc["checks"]["mtp_engagement"]
    assert harness.validate_evidence(doc)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
