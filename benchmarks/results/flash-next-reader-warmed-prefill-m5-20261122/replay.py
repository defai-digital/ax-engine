"""Verify the two-order M5 model-only prefill attribution."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / "flash-next-reader-warmed-prefill-m2-20261122"
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
from run_flash_next_generate_matrix import workload_ids  # noqa: E402

ARMS = ("mmap-short-first", "pread-short-first", "pread-long-first", "mmap-long-first")
RECEIPT_SHA256 = "c283c80f745453815dfaaee409db4e7bb0e8426672a1d30e3e34a2090a31230c"
RUNNER_SHA256 = "91659f25a1b92b6369ac26c9483cbfa9ba1232fd3808253c872503834430d7f6"


def replay(receipt):
    shared = runpy.run_path(str(M2 / "replay.py"))
    require = shared["require"]
    proof = json.loads((M2 / "m2-verified.json").read_bytes())
    plan = json.loads((M2 / "run.json").read_bytes())
    prompts = [workload_ids(248320, width, trial) for width, trial in plan["prompt_cases"]]
    prompt_bytes = (json.dumps(prompts, separators=(",", ":")) + "\n").encode()
    require(receipt["probe_sha256"] == proof["inputs_sha256"]["reader-probe"]
            and receipt["prompt_json_sha256"] == proof["inputs_sha256"]["prompts.json"]
            == sha256(prompt_bytes).hexdigest()
            and proof["state_and_token_identity"], "Probe identity or prompts changed")
    require(receipt["status"] == "complete" and list(receipt["arms"]) == list(ARMS),
            "Missing final four-arm receipt")
    require(receipt["pack_revision"] == plan["pack_revision"]
            and receipt["inventory_sha256"] == "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
            and receipt["native_manifest"] == {
                "model-manifest.json": "06cc592252dfe740b078519091995e706f2a3ae4b2e8b863720bea179d7d0e76"},
            "Publisher inventory or manifest changed")
    profiles = []
    for label, arm in receipt["arms"].items():
        profile = arm["profile"]
        require(f'{profile["loader"]}-{profile["order"]}' == label, "Reader order changed")
        require(set(arm["pressure_samples"]) == {1} and arm["pressure_after"] == 1
                and all(90 <= value <= 100 for value in (
                    arm["host_quiet"]["cpu_idle_percent"], arm["host_idle_after"])),
                "Host observations disagree with the quiet pressure envelope")
        profiles.append(profile)
    return shared["compare"](profiles)


if __name__ == "__main__":
    raw = (ROOT / "receipt.json").read_bytes()
    if (sha256(raw).hexdigest() != RECEIPT_SHA256
            or sha256((ROOT / "run_m5_reader_profile.py").read_bytes()).hexdigest() != RUNNER_SHA256):
        raise ValueError("M5 reader receipt or runner digest changed")
    print(json.dumps(replay(json.loads(raw)), indent=2, sort_keys=True, allow_nan=False))
