"""Validate the exact Flash Next prefill layer interleave at fixed token widths."""

import gzip
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from string import hexdigits
from statistics import median


ROOT = Path(__file__).resolve().parent
PARTITION = ROOT.parent / "flash-next-mxfp4-prefill-partition-m5-20261121"
INVENTORY = ROOT.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
WIDTHS = [512, 512, 512, 511, 1] * 4
CUTS = [(offset, width, layers) for offset, width in ((0, 512), (7680, 511))
        for layers in range(8, 48, 8)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(blob):
    return sha256(blob).hexdigest()


def seconds(number):
    return type(number) in (int, float) and isfinite(number) and number > 0


def main():
    receipt = json.loads((ROOT / "receipt.json").read_bytes())
    result_data = (ROOT / "result.json").read_bytes()
    result = json.loads(result_data)
    prompt_archive = gzip.decompress((PARTITION / "holdout-8192-prompts.jsonl.gz").read_bytes())
    prompts = {row["id"]: row["input_tokens"] for row in map(json.loads, prompt_archive.splitlines())}
    prompt = prompts["prompt-1-2048-holdout-8192"]
    prompt_json = (json.dumps(prompt, separators=(",", ":")) + "\n").encode()
    build = receipt["build"]
    require(len(prompt) == 8192 and receipt["prompt_sha256"] == digest(prompt_json)
            and receipt["inventory_sha256"] == digest(INVENTORY.read_bytes())
            and receipt["result_sha256"] == digest(result_data), "Unbound pack, prompt or result")
    require(receipt["native_manifest"] == {
                "model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
            } and receipt["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3",
            "Unexpected delivered pack")
    require(receipt["status"] == "complete" and receipt["test_exit"] == 0
            and (ROOT / "status.txt").read_bytes() == b"0\n"
            and receipt["qualification"] is result["qualification"] is False
            and receipt["hardware"]["soc"] == "Apple M5 Max"
            and receipt["hardware"]["memory_bytes"] == 128 << 30
            and build["mlx"] == build["mlx_metal"] == "0.32.4"
            and build["profile"] == "release-server"
            and receipt["pressure_samples"] and all(level == 1 for level in receipt["pressure_samples"]),
            "Incomplete primary hardware run")
    require(set(build["changed_source_sha256"]) == {
        "crates/ax-engine-mlx/src/model/qwen4_exp.rs",
        "crates/ax-engine-mlx/src/model/qwen4_exp/profiling.rs",
    }, "Wrong probe source inventory")
    digests = [build["executable_sha256"], *build["changed_source_sha256"].values(),
               result["peer_state_sha256"], result["long_state_sha256"]]
    require(all(len(checksum) == 64 and set(checksum) <= set(hexdigits) for checksum in digests),
            "Missing binary, source or state digest")
    require(result["schema"] == "ax-engine.flash-next.prefill-layer-peer.v1"
            and result["both_states_and_logits_exact"] is True
            and result["peer_prompt_tokens"] == 69 and result["long_widths"] == WIDTHS
            and len(result["peer_outputs"]) == 11
            and all(type(token) is int and 0 <= token < 248320 for token in result["peer_outputs"]),
            "Incomplete peer parity evidence")
    cuts = result["cuts"]
    require([(row["offset"], row["width"], row["layers"]) for row in cuts] == CUTS
            and all(seconds(row["eval_seconds"]) and seconds(row["peer_seconds"]) for row in cuts),
            "Layer cuts changed or skipped actual work")
    geometry = [(sum(WIDTHS[:i]), width) for i, width in enumerate(WIDTHS)]
    for name in ("solo_chunks", "interleaved_chunks"):
        require([(offset, width) for offset, width, _ in result[name]] == geometry
                and all(seconds(duration) for _, _, duration in result[name]), "Physical geometry changed")
    print(json.dumps({"qualification": False, "peer_outputs": len(result["peer_outputs"]),
                      "peer_decodes_between_layers": len(cuts), "long_widths": WIDTHS,
                      "both_states_and_logits_exact": True,
                      "layer_barrier_eval_median_ms": median(row["eval_seconds"] for row in cuts) * 1000,
                      "measured_peer_median_ms": median(row["peer_seconds"] for row in cuts) * 1000,
                      "before_idle_pct": receipt["before_idle_percent"],
                      "after_idle_pct": receipt["after_idle_percent"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
