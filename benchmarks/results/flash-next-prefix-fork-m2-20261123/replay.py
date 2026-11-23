"""Replay the signed Flash Next strict-prefix state and pressure observations."""

import json
from hashlib import sha256
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
MEMBERS = {
    "build.json", "fork-cases.json", "inputs.sha256", "launcher-failure.txt", "native-source.patch",
    "pressure.jsonl", "recovered-result.json", "recovery-status.txt", "status.txt", "test.log",
}


def require(valid, message):
    if not valid:
        raise ValueError(message)


def read(archive, name):
    member = archive.getmember(name)
    require(member.isfile(), f"Expected a regular file: {name}")
    with archive.extractfile(member) as stream:
        return stream.read()


def replay(path):
    with tarfile.open(path) as archive:
        names = archive.getnames()
        require(set(names) == MEMBERS and len(names) == len(MEMBERS), "The recovered evidence roster changed")
        build = json.loads(read(archive, "build.json"))
        run = json.loads(read(archive, "recovered-result.json"))
        manifest = {name: digest for digest, name in map(str.split, read(archive, "inputs.sha256").decode().splitlines())}
        fixture = read(archive, "fork-cases.json")
        require(run["build"] == build and build["codesign_verified"] is True
                and build["binary_sha256"] == manifest["ax-flash-prefix-fork-test"]
                and build["inventory_sha256"] == manifest["mxfp4-inventory.json"]
                and sha256(fixture).hexdigest() == build["fixture_sha256"] == manifest["fork-cases.json"]
                and sha256(read(archive,"native-source.patch")).hexdigest() == build["source_patch_sha256"],
                "The signed image, source patch, inventory or fixture changed")
        seed, first, second, peer = json.loads(fixture)
        require([len(seed), len(first), len(second), len(peer)] == [2048, 64, 64, 512]
                and first != second and seed[0] != peer[0], "The fixture no longer distinguishes the requests")
        require(read(archive, "status.txt") == b"1\n" and read(archive, "recovery-status.txt") == b"0\n"
                and read(archive, "launcher-failure.txt") == b"ValueError: not enough values to unpack (expected 1, got 0)\n"
                and run["original_launcher_exit"] == 1 and run["recovered_post_inventory"] is True,
                "The original launcher failure or post-run recovery changed")
        log = read(archive, "test.log").decode()
        raw, = re.findall(r"FLASH_NEXT_PREFIX_FORK (\{[^\n]+\})", log)
        count, = re.findall(r"(?m)^FLASH_NEXT_PREFIX_FORK_NATIVE_GDN_DISPATCHES ([0-9]+)$", log)
        probe = json.loads(raw)
        require("test result: ok. 1 passed; 0 failed; 0 ignored" in log and int(count) == run["native_gdn_dispatches"] > 0
                and probe == run["probe"] and probe["qualification"] is False and run["qualification"] is False,
                "The native state comparison did not complete with native GDN")
        require([probe[key] for key in ("prefix_tokens", "suffix_tokens", "physical_width", "suffix_width")]
                == [2048, 64, 512, 1024] and probe["suffix_state_sha256"][0] != probe["suffix_state_sha256"][1]
                and [len(probe[key]) for key in ("abandoned_outputs", "replacement_outputs", "peer_outputs")] == [4, 16, 21],
                "The fork, distinct suffix states or live peer changed")
        pressure = list(map(json.loads, read(archive, "pressure.jsonl").splitlines()))
        require(len(pressure) == run["normal_pressure_samples"] > 0
                and all(row["pressure"] == 1 for row in pressure), "The original pressure trace changed")
        return run


if __name__ == "__main__":
    print(json.dumps(replay(ROOT / "evidence.tar.xz"), indent=2, sort_keys=True, allow_nan=False))
