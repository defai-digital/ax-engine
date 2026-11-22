"""Verify the M5 reference bootstrap and recovery-only capacity decision."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tarfile import open as open_tar
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
SIGNED = ROOT.parent / "qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json"
SIGNED_SHA = "2eeeb8ab3b5f329726550b675ec61d57edeab6558a582582d75287983d4a19b7"


def member(archive, name):
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Missing archived source: {name}")
    with handle:
        return handle.read()


def digest(data):
    return sha256(data).hexdigest()


def require(valid, message):
    if not valid:
        raise ValueError(message)


def replay():
    signed_raw = SIGNED.read_bytes()
    signed = json.loads(signed_raw)
    boot = json.loads((ROOT / "boot-preflight.json").read_bytes())
    other = json.loads((ROOT / "virtualization-rss.json").read_bytes())
    require(digest(signed_raw) == boot["signed_reference_sha256"] == SIGNED_SHA,
            "The signed ordinary result changed")
    require(boot["versions"] == signed["versions"]
            and boot["language_sha256"] == signed["language_sha256"],
            "The pinned ordinary runtime changed")
    with open_tar(ROOT / "launcher-inputs.tar.xz") as archive:
        inputs = dict(line.split(maxsplit=1)[::-1]
                      for line in member(archive, "inputs.sha256").decode().splitlines())
        for name, sha in inputs.items():
            raw = signed_raw if name == "signed-reference.json" else member(archive, name)
            require(digest(raw) == sha, f"Launch input changed: {name}")
        frozen = member(archive, "frozen/source.json")
        require(digest(frozen) == boot["source_sha256"] and len(json.loads(frozen)) == 13,
                "The ordinary reference source manifest changed")
        for name, sha in json.loads(frozen).items():
            require(digest(member(archive, f"frozen/{name}")) == sha, f"Frozen reference input changed: {name}")
        module_source = member(archive, "capacity.py")

    detail = boot["capacity"]
    physical = signed["hardware"]["memory_bytes"]
    peak = signed["peak_mlx_memory_bytes"]
    margin = physical - peak
    with TemporaryDirectory() as directory:
        source = Path(directory) / "capacity.py"
        source.write_bytes(module_source)
        spec = importlib.util.spec_from_file_location("audited_capacity", source)
        if spec is None or spec.loader is None:
            raise ValueError("Missing capacity module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classify = module.recovery_capacity
        reproduced = classify(physical, peak, detail["pressure"], detail["largest_other_process_rss_bytes"])
        require(all(detail[key] == value for key, value in reproduced.items()),
                "The bootstrap capacity decision does not replay")
        for pressure, rss, outcome in (
            (1, margin - 1, "can-attempt"), (1, margin, "background-rss-exceeds-signed-margin"),
            (1, None, "background-rss-unavailable"), (2, 0, "host-pressure"),
            (4, 0, "host-pressure"), (True, 0, "host-pressure"),
        ):
            require(classify(physical, peak, pressure, rss)["reason"] == outcome,
                    "The recovery decision treated ineligible samples as usable")
        require(0 < other["largest_rss_bytes"] <= other["aggregate_rss_bytes"]
                and type(other["process_count"]) is int and other["process_count"] >= 1,
                "Invalid background diagnostic sample")
        virtualization = classify(physical, peak, 1, other["largest_rss_bytes"])
    return {"scope": "ordinary reference launcher bootstrap only",
            "qualification": False, "ordinary_reference_measured": False,
            "physical_bytes": physical, "observed_signed_peak_mlx_bytes": peak,
            "observed_signed_margin_bytes": margin,
            "observed_bootstrap_reason": detail["reason"],
            "virtualization_snapshot": other,
            "virtualization_snapshot_recovery_reason": virtualization["reason"]}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
