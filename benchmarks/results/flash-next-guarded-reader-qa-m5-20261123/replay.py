"""Regrade guarded-reader M5 QA and compare its stable responses with the signed prior cohort."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
FRESH = "flash-next-guarded-reader-fresh-m5-20261123"
FRESH_SHA = "9e5eaff70d2d365c3f49ba7c1809b121aec5d447ccda2bfed14f826e4ed8230b"
PRIOR = "flash-next-probe-followup-qa-m5-20261122"
PRIOR_SHA = "e920cdd114b4cda88f106169a5035de0df1b81a962e4e4dfa18c9f7b95241007"


def member(archive, path):
    entry = archive.getmember(path)
    assert entry.isfile(), path
    with archive.extractfile(entry) as stream:
        return stream.read()


def sealed(repo, label, checksum):
    path = repo / "benchmarks/results" / label / "evidence.tar.xz"
    assert sha256(path.read_bytes()).hexdigest() == checksum, label
    return tarfile.open(path)


def stable(row):
    return (row["id"], row["text"], row["finish_reason"],
            *(row["usage"][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")))


def replay(repo, capture):
    sys.path.insert(0, str(repo / "scripts"))
    import check_flash_next_release_gates as gates
    import run_flash_next_qa_cohort as qa

    items = qa.load_items(gates.QA_ITEMS)
    entry_names = {mode: [f'{mode}/{index:03d}-{item["id"]}.json' for index, item in enumerate(items)]
                   for mode in gates.AX_ARMS}
    expected = {"result.json", "contract.json", *(name for rows in entry_names.values() for name in rows)}
    with (tarfile.open(capture) as archive, sealed(repo, FRESH, FRESH_SHA) as fresh,
          sealed(repo, PRIOR, PRIOR_SHA) as prior):
        names = archive.getnames()
        assert set(names) == expected and len(names) == len(expected)
        raw_result = member(archive, "result.json")
        result = json.loads(raw_result)
        contract = json.loads(member(archive, "contract.json"))
        build = json.loads(member(fresh, "build.json"))
        delivered = json.loads(member(fresh, "delivery/result.json"))
        manifest = {name: digest for digest, name in map(str.split, member(fresh, "inputs.sha256").decode().splitlines())}
        responses = {mode: [json.loads(member(archive, name), object_pairs_hook=gates.unique_json_members)
                            for name in paths] for mode, paths in entry_names.items()}
        previous = {mode: [json.loads(member(prior, name)) for name in paths] for mode, paths in entry_names.items()}
        rows_sha = {mode: sha256(b"".join(sha256(member(archive, name)).digest() for name in paths)).hexdigest()
                    for mode, paths in entry_names.items()}
    inputs = (Path(gates.__file__), gates.THRESHOLDS, Path(qa.__file__), gates.QA_ITEMS,
              *(qa.FROZEN / name for name in ("checkers.py", "closed_checks.py", "prompt_def.py", "qa_checks.py")))
    for path in inputs:
        assert sha256(path.read_bytes()).hexdigest() == manifest[path.relative_to(repo).as_posix()], path.name
    assert all(result[name] == value for name, value in contract.items())
    assert all(result[name] == delivered[name] for name in ("hardware", "revision", "inventory_sha256", "server_sha256"))
    assert result["server_sha256"] == build["server_sha256"] == manifest["ax-engine-server"]
    assert result["ordinary_launch"] is True and result["total_blocks"] * result["block_size_tokens"] == 32768
    assert result["qualification"] is False and result["release_ready"] is False
    assert result["modes"] == list(gates.AX_ARMS) and not gates.native_qa_problems(result, items, responses)
    runs = {}
    for mode, rows in responses.items():
        metadata = result["summary"]["modes"][mode]
        advertised = {"id": qa.ORDINARY_MODEL_ID if mode == "default" else qa.MODEL_ID, "context_length": 32768}
        assert metadata["advertised_model"] == advertised
        runs[mode] = {"rows": rows, **{field: metadata[field] for field in
                                      ("peak_rss_kib", "ready_seconds", "advertised_model")}}
    summary = qa.summarize(items, runs, result["grades"])
    assert summary == result["summary"] and summary["requests"] == 210
    assert all(row["verified_steps"] == row["route_errors"] == 0 for row in summary["modes"].values())
    differences = {mode: [row["id"] for row, old in zip(rows, previous[mode], strict=True) if stable(row) != stable(old)]
                   for mode, rows in responses.items()}
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())
    readiness = [(mode, row["ready_seconds"]) for mode, row in summary["modes"].items()]
    idle = [result["host_quiet"]["cpu_idle_percent"], result["host_idle_after"]]
    floor = thresholds["timing_host"]["min_cpu_idle_percent"]
    return {"scope": "Guarded-reader native M5 QA", "release_ready": False,
            "server_sha256": result["server_sha256"], "qa_receipt_sha256": sha256(raw_result).hexdigest(),
            "raw_qa_rows_sha256": rows_sha, "responses": summary["requests"], "modes": summary["modes"],
            "prior_response_differences": differences,
            "timing_cpu_idle_percent": idle, "timing_eligible": all(floor <= value <= 100 for value in idle),
            "memory": gates.gate_memory(result, thresholds)[0], "long_lookup": gates.gate_long_lookup(result, thresholds)[0],
            "readiness": gates.gate_readiness(delivered, readiness, thresholds)[0]}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = cli.parse_args()
    print(json.dumps(replay(args.repo.resolve(), args.archive), sort_keys=True, indent=2, allow_nan=False))
