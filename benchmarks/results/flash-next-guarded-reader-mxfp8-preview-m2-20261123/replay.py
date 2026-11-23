"""Replay functional MXFP8 reader isolation and canonical MTP on M2 Ultra."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
PACK = "48b9aa5f002f29a9f8ab7cfe5c7185e342595b1b"
ORACLE_SHA = "84f2b0d7df140f8ba2e9b26be2b323d30aafe734990c53d9c66b2a9baa17839e"
INVENTORY_SHA = "7e2c19296d630d4e4495d022d080e761bd592a7265065fbc7ac369f9d2afb26d"
FRESH_SHA = "9e5eaff70d2d365c3f49ba7c1809b121aec5d447ccda2bfed14f826e4ed8230b"
MEMBERS = {"build.json", "inputs.sha256", "status.txt", "receipt.json", "completed-http-controls.json", "server.log"}


def member(archive, name):
    row = archive.getmember(name)
    assert row.isfile()
    with archive.extractfile(row) as stream:
        return stream.read()


def replay(repo, path):
    bench = repo / "benchmarks/results"
    fresh = bench / "flash-next-guarded-reader-fresh-m5-20261123/evidence.tar.xz"
    oracle_raw = (bench / "flash-next-mxfp8-resident-loader-m2-20261121/result.json").read_bytes()
    inventory_raw = (bench / "qualification/2026-11-18-flash-next-mxfp8-preview/pack-inventory.json").read_bytes()
    assert sha256(oracle_raw).hexdigest() == ORACLE_SHA and sha256(inventory_raw).hexdigest() == INVENTORY_SHA
    assert sha256(fresh.read_bytes()).hexdigest() == FRESH_SHA
    oracle, inventory = map(json.loads, (oracle_raw, inventory_raw))
    assert oracle["revision"] == inventory["revision"] == PACK and len(inventory["files"]) == 61
    with tarfile.open(path) as archive, tarfile.open(fresh) as mx4:
        names = archive.getnames()
        assert set(names) == MEMBERS and len(names) == len(MEMBERS)
        data = {name: member(archive, name) for name in names}
        fresh_build = json.loads(member(mx4, "build.json"))
    assert all(token not in value for value in data.values()
               for token in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    assert data["status.txt"] == b"0\n"
    hashes = {name: checksum for checksum, name in map(str.split, data["inputs.sha256"].decode().splitlines())}
    assert sha256(data["build.json"]).hexdigest() == hashes["build.json"]
    assert hashes["pack-inventory.json"] == INVENTORY_SHA and hashes["oracle.json"] == ORACLE_SHA
    sys.path.insert(0, str(repo / "scripts"))
    import run_flash_next_generate_matrix as matrix

    assert sha256(Path(matrix.__file__).read_bytes()).hexdigest() == hashes["run_flash_next_generate_matrix.py"]
    receipt, build, controls = (json.loads(data[name]) for name in ("receipt.json", "build.json", "completed-http-controls.json"))
    assert receipt["build"] == build
    assert build["server_sha256"] == hashes["ax-engine-server"] == fresh_build["server_sha256"]
    assert build["patch_sha256"] == hashes["source.patch"] == fresh_build["patch_sha256"]
    assert build["codesign_verified"] is True and build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert receipt["revision"] == PACK and receipt["qualification"] is False
    assert receipt["startup_timeout_seconds"] == 1800
    pack = receipt["pack"]
    assert pack["publisher_members"] == 61 and pack["publisher_bytes"] == 189240515155
    assert re.fullmatch(r"[0-9a-f]{64}", pack["derived_manifest_sha256"])
    hardware = receipt["hardware"]
    assert hardware["form_factor"] == "Mac Studio" and hardware["soc"] == "Apple M2 Ultra"
    assert hardware["memory_bytes"] == 192 * (1 << 30)
    observed = receipt["observed"]
    assert observed["reader"] == "Mapped" and observed["requested_buffered_override"] == "1"
    assert observed["health"] == "ok" and observed["context"] == 32768
    assert all(observed[name] == controls[name] for name in ("rows", "pressure_samples"))
    assert observed["pressure_samples"] and set(observed["pressure_samples"]) == {1}
    trace = data["server.log"].decode()
    assert re.findall(r"Flash Next resident reader[^\n]* reader=(\w+)", trace) == ["Mapped"]
    entries = re.findall(r'Flash Next resident shard shard=Some\("(model-\d{5}-of-00036\.safetensors)"\) '
                         r'tensor_count=(\d+) read_us=\d+ eval_us=\d+(?:\n|$)', trace)
    counts = {name: int(count) for name, count in entries}
    assert len(counts) == len(entries) == observed["resident_shards"] == 27
    assert counts == observed["resident_tensor_counts"] and sum(counts.values()) == observed["resident_tensors"] == 1842
    ids, steps = 0, 0
    for row, prompt, old in zip(observed["rows"], oracle["prompts"], oracle["arms"]["required"]["rows"], strict=True):
        assert row["name"] == prompt["name"] and row["route_errors"] == 0 and row["verified_steps"] > 0
        assert all(row[key] == old[key] for key in ("output_tokens", "prompt_tokens", "finish_reason", "status"))
        matrix.check_verifier_schedule(row, "required", "canonical")
        ids += len(row["output_tokens"])
        steps += row["verified_steps"]
    assert ids == 64
    return {"scope": receipt["scope"], "qualification": False, "revision": PACK, "pack": pack,
            "server_sha256": build["server_sha256"], "reader": observed["reader"], "resident_shards": len(counts),
            "resident_tensors": sum(counts.values()), "matched_output_ids": ids, "verified_steps": int(steps),
            "normal_pressure_samples": len(observed["pressure_samples"])}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = cli.parse_args()
    print(json.dumps(replay(args.repo.resolve(), args.archive), sort_keys=True, indent=2, allow_nan=False))
