"""Mutation checks for the default-server acceptance artifact."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("default_verify", ROOT / "verify.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def rows():
    return json.loads((ROOT / "trials.json").read_text())


def test_truncated_generation_fails(rows):
    rows[0]["trials"][0]["usage"]["completion_tokens"] -= 1
    with pytest.raises(AssertionError):
        module.verify(rows)


def test_output_change_fails(rows):
    rows[0]["trials"][1]["text"] += "changed"
    with pytest.raises(AssertionError):
        module.verify(rows)


def test_pressure_guard_fails(rows):
    rows[0]["samples"][0]["pressure"] = "2"
    with pytest.raises(AssertionError):
        module.verify(rows)


def test_missing_arm_fails(rows):
    with pytest.raises(AssertionError):
        module.verify(rows[:-1])


def test_ungraceful_exit_fails(rows):
    rows[0]["exit_code"] = -15
    with pytest.raises(AssertionError):
        module.verify(rows)


def test_auto_fallback_is_not_reported_as_resident(rows):
    row = next(row for row in rows if row["arm"] == "auto")
    row["route_events"] = [line.replace("permitted=true", "permitted=false") for line in row["route_events"]]
    with pytest.raises(AssertionError):
        module.verify(rows)
