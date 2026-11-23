"""Replay stock Required on the guarded candidate against the earlier signed controls."""

from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-probe-followup-fresh-m5-20261123"
AUTO = ROOT.parent / "flash-next-automatic-buffered-reader-m5-20261123"
MEMBERS = {"status.txt", "cpu-before.json", "receipt.json", "source.json",
           "native-support/result.json",
           *(f"native-support/server-{mode}.log" for mode in ("disabled", "required", "required-kill-switch"))}
MODES = {"disabled": 32768, "required": 16384}
VERIFIED = "ax_engine_flash_next_mtp_verified_steps_total"


def member(archive, name):
    info = archive.getmember(name)
    assert info.isfile()
    with archive.extractfile(info) as stream:
        return stream.read()


def read(data):
    return json.loads(data)


def replay():
    prior_archive = PRIOR / "evidence.tar.xz"
    pinned = {name: checksum for checksum, name in
              (line.split() for line in (PRIOR / "SHA256SUMS").read_text().splitlines())}
    assert sha256(prior_archive.read_bytes()).hexdigest() == pinned["evidence.tar.xz"]
    with (tarfile.open(ROOT / "evidence.tar.xz") as archive, tarfile.open(prior_archive) as older):
        names = archive.getnames()
        assert set(names) == MEMBERS and len(names) == len(MEMBERS)
        data = {name: member(archive, name) for name in MEMBERS}
        stock = read(member(older, "raw/delivery/native-support/result.json"))
    auto = read((AUTO / "comparison.json").read_bytes())
    source = read(data["source.json"])
    observed, summary = (read(data[name]) for name in ("native-support/result.json", "receipt.json"))
    before = read(data["cpu-before.json"])
    assert data["status.txt"] == b"0\n"
    assert not any(key in data[name] for name in MEMBERS for key in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    assert source["candidate_commit"] == auto["candidate_commit"]
    assert source["server_sha256"] == observed["server_sha256"] == auto["candidate_server_sha256"]
    assert source["inventory_sha256"] == observed["inventory_sha256"] == stock["inventory_sha256"]
    assert summary["server_sha256"] == observed["server_sha256"] and summary["functional_controls_passed"]
    assert observed["functional_controls_passed"] and observed["modes"] == list(MODES)
    assert observed["qualification"] is False and observed["release_ready"] is False
    assert observed["readiness_seconds"] == summary["ready_seconds"]
    assert summary["observed_context_tokens"] == MODES
    assert summary["required_warm_delivery_limit_seconds"] == 120
    assert summary["required_within_limit"] and 0 < summary["ready_seconds"]["required"] <= 120
    assert summary["cpu_idle_percent"][0] == before["cpu_idle_percent"]
    assert summary["timing_eligible"] and all(90 <= rate <= 100 for rate in summary["cpu_idle_percent"])
    assert observed["hardware"]["form_factor"] == "MacBook Pro" and observed["hardware"]["soc"] == "Apple M5 Max"
    assert observed["hardware"]["memory_bytes"] == 128 * (1 << 30) and observed["hardware"]["os"] == "27.2.1"
    assert observed["negative_control_result"]["activation_error_observed"] is True
    verified, matched = {}, 0
    for mode, context in MODES.items():
        log = data[f"native-support/server-{mode}.log"].decode()
        assert "decision=resident" in log
        assert re.search(r"resolved the Flash Next M5 Max 128 GB serving profile[^\n]*context_tokens="
                         + str(context) + r"\b[^\n]*effective_sequence_cap=Some\(1\)", log)
        rows = observed["results"][mode]
        assert len(rows) == 7
        for row, prior in zip(rows, stock["results"][mode], strict=True):
            assert row["name"] == prior["name"] and row["request"] == prior["request"]
            assert row["response"]["choices"] == prior["response"]["choices"]
            assert row["activation_delta"] == prior["activation_delta"]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                assert row["response"]["usage"][key] == prior["response"]["usage"][key]
            matched += 1
        verified[mode] = int(sum(row["activation_delta"][VERIFIED] for row in rows))
    assert verified == {"disabled": 0, "required": 107}
    return {**summary, "candidate_commit": source["candidate_commit"],
            "stable_prior_controls_matched": matched, "verified_steps": verified, "qualification": False}


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
