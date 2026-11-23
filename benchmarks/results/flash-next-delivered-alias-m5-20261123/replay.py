"""Replay the ordinary offline alias launch against signed input witnesses."""

from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
FILES = {"status.txt", "inputs.sha256", "resolver/resolver.json", "observed/alias.json", "observed/server.log",
         "driver/run_delivered_alias.py", "driver/signed-case.json"}


def raw(archive, name):
    entry = archive.getmember(name)
    assert entry.isfile()
    with archive.extractfile(entry) as stream:
        return stream.read()


def sealed(directory, digest):
    path = ROOT.parent / directory / "evidence.tar.xz"
    assert sha256(path.read_bytes()).hexdigest() == digest
    return tarfile.open(path)


def replay():
    with (tarfile.open(ROOT / "evidence.tar.xz") as archive,
          sealed("flash-next-automatic-buffered-reader-m5-20261123",
                 "386fa1213f1fa3721591e9cdf11ba3224b25dc58dc771b8c68c001522b3d455b") as prior,
          sealed("flash-next-guarded-reader-fresh-m5-20261123",
                 "9e5eaff70d2d365c3f49ba7c1809b121aec5d447ccda2bfed14f826e4ed8230b") as fresh):
        names = archive.getnames()
        assert len(names) == len(FILES) and set(names) == FILES
        data = {name: raw(archive, name) for name in names}
        stock = json.loads(raw(prior, "arms/buffered-first/receipt.json"))
        delivery = json.loads(raw(fresh, "delivery/result.json"))
    assert all(token not in value for value in data.values() for token in (b"/Users/", b"/private/tmp/"))
    assert data["status.txt"] == b"0\n"
    hashes = {name: checksum for checksum, name in map(str.split, data["inputs.sha256"].decode().splitlines())}
    for path in ("run_delivered_alias.py", "signed-case.json"):
        assert sha256(data[f"driver/{path}"]).hexdigest() == hashes[path]
    launch, resolver, witness = (json.loads(data[name]) for name in (
        "observed/alias.json", "resolver/resolver.json", "driver/signed-case.json",
    ))
    assert launch["server_sha256"] == resolver["server_sha256"] == witness["server_sha256"]
    assert launch["server_sha256"] == stock["build"]["server_sha256"] == delivery["server_sha256"]
    assert launch["cli_sha256"] == resolver["cli_sha256"] == delivery["download"]["ax_engine_sha256"]
    assert launch["revision"] == resolver["revision"] == delivery["revision"] == witness["ref"]
    assert launch["alias"] == resolver["alias"] == "qwen3.8-flash-next:axq"
    assert launch["offline_cache_hit"] is True and resolver["offline_cache_hit"] is True
    assert resolver["runtime_launched"] is False
    case = stock["arms"]["buffered-first"]["cases"][0]
    assert sha256(json.dumps(case, sort_keys=True).encode()).hexdigest() == witness["case_sha256"]
    assert case["prompt_hash"] == witness["prompt_sha256"] and case["prompt_tokens"] == witness["prompt_tokens"] == 512
    assert case["output_tokens"] == witness["output_tokens"] and len(witness["output_tokens"]) == launch["matched_output_ids"] == 32
    log = data["observed/server.log"].decode()
    beginning = re.search(r"(?m)^\{", log)
    assert beginning is not None
    plan, _ = json.JSONDecoder().raw_decode(log[beginning.start():])
    assert plan["schema_version"] == "ax.local_serve_plan.v1" and plan["command"] == "serve"
    resolved = plan["resolved"]
    assert plan["input"] == resolved["model"] == launch["alias"] and resolved["revision"] == launch["revision"]
    assert resolved["kind"] == "resolved_snapshot" and resolved["preset"] is None
    assert resolved["download"]["offline"] is True and resolved["download"]["status"] == "ready"
    exe, host, loopback, port, port_value, mlx, model_dir, installed = plan["server"]["argv"]
    assert exe == "[delivery]/ax-engine-server" and loopback == "[loopback]" and port_value.isdigit()
    assert [host, port, mlx, model_dir] == resolver["native_flags"] == ["--host", "--port", "--mlx", "--mlx-model-artifacts-dir"]
    assert installed == resolved["path"] and installed.endswith("/snapshots/" + launch["revision"])
    assert re.findall(r"Flash Next resident reader[^\n]* reader=(\w+)", log) == [launch["reader"]] == ["AlignedBuffered"]
    assert launch["context_tokens"] == 32768 and launch["max_live_requests"] == 1
    assert "context_tokens=32768 effective_sequence_cap=Some(1)" in log
    assert 0 < launch["http_ready_seconds_including_offline_resolution"] <= 120
    assert launch["health"] == "ok" and launch["qualification"] is False and launch["verified_mtp_steps"] == 0
    assert len(launch["drained_counters"]) == 5 and all(value == 0 for value in launch["drained_counters"].values())
    return launch


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
