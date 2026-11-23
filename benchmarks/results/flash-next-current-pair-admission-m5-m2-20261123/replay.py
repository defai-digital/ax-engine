"""Replay the native Flash Next pair-admission controls against signed inputs."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
FRESH_SHA = "9e5eaff70d2d365c3f49ba7c1809b121aec5d447ccda2bfed14f826e4ed8230b"
HOSTS = {"m2": ("Mac Studio", "Apple M2 Ultra", 192), "m5": ("MacBook Pro", "Apple M5 Max", 128)}
RAW = ("status.txt", "receipt.json", "server.log", "samples.jsonl", "inputs.sha256")
COMMON = {"software.json", "source.patch", "old-control.json", "old-partial-wait.json", "new-build.json"}


def get(archive, name):
    member = archive.getmember(name)
    assert member.isfile(), name
    with archive.extractfile(member) as data:
        return data.read()


def replay(repo, archive, hosts):
    prior = repo / "benchmarks/results/flash-next-guarded-reader-fresh-m5-20261123/evidence.tar.xz"
    assert sha256(prior.read_bytes()).hexdigest() == FRESH_SHA
    expected = COMMON | {f"{host}/{name}" for host in hosts for name in RAW}
    with tarfile.open(archive) as bundle, tarfile.open(prior) as old:
        names = bundle.getnames()
        assert set(names) == expected and len(names) == len(expected)
        data = {name: get(bundle, name) for name in names}
        prior_build = json.loads(get(old, "build.json"))
    assert all(token not in value for value in data.values() for token in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    software, build, old, negative = map(lambda name: json.loads(data[name]),
                                         ("software.json", "new-build.json", "old-control.json", "old-partial-wait.json"))
    assert build["codesign_verified"] is True and build["server_sha256"] != prior_build["server_sha256"]
    assert old["server_sha256"] == prior_build["server_sha256"]
    assert old["raw_sha256"] == sha256(data["old-partial-wait.json"]).hexdigest()
    assert build["patch_sha256"] == software["signed_runtime_patch_sha256"] == sha256(data["source.patch"]).hexdigest()
    assert software["branch_patch_sha256"] == build["patch_sha256"]
    assert software["workspace_rust_tests"] == 4811 and software["python_tests"] == 345
    assert software["branch_core_tests"] == 813 and software["branch_clippy_passed"]
    assert build["mlx"] == build["mlx_metal"] == prior_build["mlx"] == "0.32.4"
    assert len(negative) == 1 and negative[0]["active"]["processed_prompt_tokens"] == 2048
    assert negative[0]["waiting"]["prompt_len"] == 8192 and negative[0]["waiting"]["state"] == "runnable"
    assert negative[0]["waiting"]["processed_prompt_tokens"] == 0
    hosts_record = {}
    for host in hosts:
        assert data[f"{host}/status.txt"] == b"0\n"
        inputs = {name: checksum for checksum, name in map(str.split, data[f"{host}/inputs.sha256"].decode().splitlines())}
        assert inputs["ax-engine-server"] == build["server_sha256"] and inputs["source.patch"] == build["patch_sha256"]
        assert inputs["build.json"] == sha256(data["new-build.json"]).hexdigest()
        assert inputs["candidate-software-receipt.json"] == sha256(data["software.json"]).hexdigest()
        result = json.loads(data[f"{host}/receipt.json"])
        assert result["build"] == build and result["candidate"] == software and result["qualification"] is False
        assert result["revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
        assert result["flags"]["AX_MLX_FLASH_NEXT_PREFILL_SPLIT"] == "1"
        assert result["flags"]["AX_MLX_FLASH_NEXT_DIRECT_COALESCING"] == "0"
        assert result["manifest"]["model-manifest.json"] == "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
        hardware = result["hardware"]
        form, soc, gib = HOSTS[host]
        assert (hardware["form_factor"], hardware["soc"], hardware["memory_bytes"]) == (form, soc, gib << 30)
        log = re.sub(r"\x1b\[[0-9;]*m", "", data[f"{host}/server.log"].decode())
        assert "decision=resident" in log
        assert re.findall(r"Flash Next resident reader[^\n]* reader=(\w+)", log) == ["Mapped"]
        samples = [json.loads(line) for line in data[f"{host}/samples.jsonl"].splitlines()]
        assert result["normal_pressure_samples"] == len(samples) > 0
        assert all(sample["pressure"] == 1 and 0 < sample["footprint_bytes"] <= 96 << 30 for sample in samples)
        queue, shared, solo = (result[key] for key in ("queue", "shared", "solo"))
        assert queue["rejection"] == {"status": 429, "code": "concurrency_limit"}
        assert queue["cancelled"]["state"] == "cancelled" and queue["promoted"]["processed_prompt_tokens"] == 2048
        assert len(queue["frames"]) == 2 and [row["active"]["processed_prompt_tokens"] for row in queue["frames"]] == [2048, 4096]
        for row in queue["frames"]:
            assert row["step"]["scheduled_requests"] == 1 and row["step"]["scheduled_tokens"] == 2048
            assert row["waiting"]["state"] == "waiting" and row["waiting"]["prompt_len"] == 8192
            assert row["waiting"]["processed_prompt_tokens"] == row["waiting"]["output_len"] == 0
            assert row["waiting"]["request_id"] == queue["promoted"]["request_id"] == queue["completion"]["request_id"]
        assert queue["completion"]["state"] == "finished"
        assert shared["shared"]["scheduled_requests"] == 2 and shared["shared"]["scheduled_tokens"] == 513
        assert shared["long_live"]["processed_prompt_tokens"] == 512
        assert shared["short_live"]["prompt_len"] == 512 and shared["long_live"]["prompt_len"] == 8192
        _, second, short = result["case_ids"]
        assert queue["completion"]["output_tokens"] == solo[second]
        assert [row["output_tokens"] for row in shared["completion"]] == [solo[short], solo[second]]
        assert all(len(ids) == 96 for ids in solo.values()) and result["matching_output_ids"] == 288
        assert all(len(values) == 5 and not any(values.values()) for values in (queue["drained"], shared["drained"]))
        hosts_record[host] = {"hardware": hardware, "matching_output_ids": 288, "reader": "Mapped",
                              "cpu_idle_percent": result["cpu_idle_percent"], "normal_pressure_samples": len(samples),
                              "peak_footprint_bytes": max(row["footprint_bytes"] for row in samples),
                              "queued_steps": 2, "shared_prefill_tokens": 512}
    return {"scope": "Explicit resident two-slot physical-prefill admission", "qualification": False,
            "candidate_commit": software["candidate_commit"], "server_sha256": build["server_sha256"],
            "old_control_server_sha256": old["server_sha256"], "hosts": hosts_record}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = cli.parse_args()
    print(json.dumps(replay(args.repo.resolve(), args.archive, tuple(HOSTS)),
                     sort_keys=True, indent=2, allow_nan=False))
