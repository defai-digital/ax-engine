from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from ax_engine_vllm_runtime.cli import _read_secret_file
from ax_engine_vllm_runtime.launcher import (
    ServeConfig,
    build_vllm_command,
    build_vllm_environment,
)
from ax_engine_vllm_runtime.preflight import RuntimeFacts, validate_runtime_facts
from ax_engine_vllm_runtime.profiles import (
    PROFILE_SCHEMA_VERSION,
    get_profile,
    profiles,
    runtime_lock_sha256,
)
from ax_engine_vllm_runtime.thor_compat import register as register_thor_compat


def test_profiles_are_versioned_and_platform_specific() -> None:
    values = profiles()
    assert len(values) == 3
    assert len({profile.profile_id for profile in values}) == len(values)
    assert all(profile.schema_version == PROFILE_SCHEMA_VERSION for profile in values)
    assert get_profile("cuda-linux-x86_64-a6000-sm86").plugin is None
    assert get_profile("cuda-linux-aarch64-thor-sm110").plugin == "ax_engine_vllm_thor_compat"


def test_profile_lock_digests_match_release_files() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for profile in profiles():
        lock = project_root / profile.runtime_lock
        assert sha256(lock.read_bytes()).hexdigest() == profile.runtime_lock_sha256
        assert runtime_lock_sha256(profile) == profile.runtime_lock_sha256


def test_a6000_preflight_is_exact_and_fail_closed() -> None:
    profile = get_profile("cuda-linux-x86_64-a6000-sm86")
    facts = RuntimeFacts(
        system="Linux",
        architecture="x86_64",
        python_implementation="cpython",
        python_version=(3, 12),
        vllm_version="0.25.1+cu130",
        torch_version="2.11.0+cu130",
        torch_cuda_version="13.0",
        gpu_count=1,
        gpu_name="NVIDIA RTX A6000",
        compute_capability=(8, 6),
    )
    assert validate_runtime_facts(profile, facts).ready
    wrong_gpu = RuntimeFacts(**{**facts.__dict__, "compute_capability": (8, 9)})
    report = validate_runtime_facts(profile, wrong_gpu)
    assert not report.ready
    assert (
        next(check for check in report.checks if check.name == "compute_capability").passed is False
    )


def test_command_keeps_wire_recipe_constant_and_profile_tuning_separate() -> None:
    config = ServeConfig(
        model_path="baidu/Unlimited-OCR",
        served_model_name="candidate",
    )
    a6000 = build_vllm_command(
        config,
        get_profile("cuda-linux-x86_64-a6000-sm86"),
    )
    thor = build_vllm_command(
        config,
        get_profile("cuda-linux-aarch64-thor-sm110"),
    )
    for command in (a6000, thor):
        assert command[:3] == ["vllm", "serve", "baidu/Unlimited-OCR"]
        assert command[command.index("--served-model-name") + 1] == "candidate"
        assert command[command.index("--max-model-len") + 1] == "32768"
        assert "--no-enable-prefix-caching" in command
        assert "--logits-processors" in command
        assert command[command.index("--revision") + 1] == (
            "ee63731b6461c8afcdcc7b15352e7d2ffecc2ead"
        )
    assert "--enforce-eager" not in a6000
    assert "--enforce-eager" in thor
    assert "--attention-config" not in a6000
    assert "--attention-config" in thor


def test_environment_selects_only_the_profile_plugin_and_secret_stays_out_of_repr() -> None:
    config = ServeConfig(api_key="top-secret")
    assert "top-secret" not in repr(config)
    environment = build_vllm_environment(
        config,
        get_profile("cuda-linux-x86_64-a6000-sm86"),
        base_environment={"PATH": "/usr/bin"},
    )
    assert environment["VLLM_PLUGINS"] == ""
    assert environment["VLLM_API_KEY"] == "top-secret"


def test_secret_reader_rejects_symbolic_links(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_text("top-secret\n", encoding="utf-8")
    link = tmp_path / "secret-link"
    os.symlink(secret, link)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        _read_secret_file(link)


def test_thor_plugin_is_inert_off_target() -> None:
    with (
        patch("ax_engine_vllm_runtime.thor_compat.platform.system", return_value="Darwin"),
        patch("ax_engine_vllm_runtime.thor_compat._cuda_device_capability") as capability,
        patch("ax_engine_vllm_runtime.thor_compat.importlib.import_module") as importer,
    ):
        register_thor_compat()
    capability.assert_not_called()
    importer.assert_not_called()
