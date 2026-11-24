#!/usr/bin/env python3
"""Replay the Flash Next prefill-geometry report from raw decision rows."""

from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import tarfile

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "flash-next-qsa-grouped-http-m5-20261123" / "evidence.tar.xz"


def replay(archive, prior):
    def get(name):
        file = archive.extractfile(name)
        assert file is not None
        return file.read()

    def read(name):
        return json.loads(get(name))

    def digest(name):
        return sha256(get(name)).hexdigest()

    assert get("status.txt").strip() == b"0"
    assert sha256(prior.read_bytes()).hexdigest() == get("prior-evidence.sha256").decode().strip()
    prompt = read("prompt.json")
    expected = read("prior-ids.json")
    with tarfile.open(prior) as older:
        assert json.load(older.extractfile("prompts.json"))["long"] == prompt
        for width, arm in (("512", "stock-fallback"), ("2048", "stock-first")):
            rows = older.extractfile(f"observed/{arm}/requests.jsonl").read().splitlines()
            assert next(r for r in map(json.loads, rows)
                        if r["action"] == "long")["output_ids"] == expected[width]
    assert len(prompt) == 8192
    assert len(expected["512"]) == len(expected["2048"]) == 64
    assert expected["512"][0] == expected["2048"][0]
    assert expected["512"][1] != expected["2048"][1]

    soc, memory, macos, internal, ssd = get("host.txt").decode().splitlines()
    assert soc == "Apple M5 Max" and int(memory) == 128 * 2**30
    assert internal == ssd == "true"
    build = read("build.json")
    assert build["profile"] == "release-server" and build["codesign_verified"]
    assert "1.99.1" in build["rustc"] and "1.99.1" in build["cargo"]
    assert build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert digest("source.patch") == build["patch_sha256"]
    sealed = dict(line.split(maxsplit=1)[::-1] for line in get("inputs.sha256").decode().splitlines())
    for name, expected_digest in sealed.items():
        if name != "ax-flash-prefill-geometry-test":
            assert digest(name) == expected_digest
    assert sealed["ax-flash-prefill-geometry-test"] == build["binary_sha256"]
    assert len(read("mxfp4-inventory.json")["members"]) == 49

    receipt = read("observed/receipt.json")
    assert receipt["build"] == build and receipt["qualification"] is False
    assert receipt["manifest"]["model-manifest.json"] == build["manifest_sha256"]
    assert receipt["prompt_sha256"] == digest("prompt.json")
    assert receipt["prior_ids_sha256"] == digest("prior-ids.json")

    arms, decisions, observed_samples = {}, {}, 0
    for width in (512, 2048):
        directory = f"observed/{width}"
        arm = read(f"{directory}/receipt.json")
        proof = read(f"{directory}/log-proof.json")
        assert proof["original_sha256"] == arm["original_test_log_sha256"]
        assert proof["test_result"] == "1 passed; 0 failed"
        assert arm["flags"]["AX_MLX_LONG_PROMPT_PREFILL_CHUNK"] == str(width)
        assert arm["flags"]["AX_MLX_FLASH_NEXT_QSA_GROUPED_EVAL"] == "0"
        assert arm["flags"]["AX_MLX_FLASH_NEXT_QSA_PAIRED_GATHER"] == "0"
        assert arm["flags"]["AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL"] == "0"
        assert arm["result_sha256"] == digest(f"{directory}/result/result.json")
        assert arm["samples_sha256"] == digest(f"{directory}/native-samples.jsonl")
        samples = list(map(json.loads, get(f"{directory}/native-samples.jsonl").splitlines()))
        assert len(samples) == arm["sample_count"] and samples
        assert all(s["pressure"] == 1 and s["footprint_bytes"] <= 96 * 2**30 for s in samples)
        assert max(s["footprint_bytes"] for s in samples) == arm["sampled_footprint_peak_bytes"]
        observed_samples += len(samples)
        result = read(f"{directory}/result/result.json")
        assert result["output_ids"] == expected[str(width)] and result["width"] == width
        assert result["qualification"] is False
        cuts = result["cuts"]
        assert cuts[0][0] == 0 and cuts[-1] == [8191, 8192]
        assert all(left[1] == right[0] for left, right in zip(cuts, cuts[1:]))
        assert width - 1 <= max(end - start for start, end in cuts) <= width
        positions = [witness["position"] for witness in result["witnesses"]]
        assert positions in ([2048, 8192, 8193], [2047, 2048, 8192, 8193])
        for witness, expected_id in zip(result["witnesses"][-2:], expected[str(width)][:2], strict=True):
            row = witness["logits"]
            assert row["token"] == expected_id
            blob = row["logits_f32_le"]
            data = get(f"{directory}/result/{blob['file']}")
            assert len(data) == blob["bytes"] and sha256(data).hexdigest() == blob["sha256"]
            count = row["shape"][1]
            assert row["shape"] == [1, count]
            values = struct.unpack(f"<{count}f", data)
            assert all(map(math.isfinite, values))
            assert max(range(count), key=values.__getitem__) == expected_id
            decisions[(width, witness["position"])] = values
        arms[width] = result

    shared_cuts = sorted(set(end for _, end in arms[512]["cuts"]) &
                         set(end for _, end in arms[2048]["cuts"]))
    assert shared_cuts[0] in (2047, 2048)
    states = []
    logit_rows = []
    paired = [{row["position"]: row for row in arms[width]["witnesses"]} for width in (512, 2048)]
    for position in sorted(paired[0].keys() & paired[1].keys()):
        narrow, wide = (arm[position] for arm in paired)
        changed = [layer for layer, (a, b) in enumerate(
            zip(narrow["layer_hashes"], wide["layer_hashes"], strict=True)) if a != b]
        states.append({"position": position, "layers": len(narrow["layer_hashes"]),
                       "different_layers": changed})
        if (512, position) in decisions:
            a, b = decisions[512, position], decisions[2048, position]
            ids = [narrow["logits"]["token"], wide["logits"]["token"]]
            logit_rows.append({"position": position, "token_ids": ids,
                               "stock_512_winner_margin": winner_margin(a, ids[0]),
                               "stock_2048_winner_margin": winner_margin(b, ids[1]),
                               "stock_512_candidate_gap": a[ids[0]] - a[ids[1]],
                               "stock_2048_candidate_gap": b[ids[1]] - b[ids[0]],
                               "max_absolute_delta": max(abs(x - y) for x, y in zip(a, b, strict=True)),
                               "different_values": sum(x != y for x, y in zip(a, b, strict=True))})
    return {"qualification": False, "macos": macos, "source_base": build["source_base"],
            "earliest_shared_cut": shared_cuts[0], "state_witnesses": states, "logits": logit_rows,
            "normal_pressure_samples": observed_samples,
            "matched_http_token_ids": len(expected["512"]) + len(expected["2048"])}


def winner_margin(row, token):
    return row[token] - max(value for index, value in enumerate(row) if index != token)


if __name__ == "__main__":
    with tarfile.open(HERE / "evidence.tar.xz") as archive:
        print(json.dumps(replay(archive, PRIOR), indent=2, sort_keys=True))
