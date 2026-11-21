"""Verify the isolated M5 sidecar experiment from its raw receipts."""

from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
READERS = {
    "mmap-1": "name-order",
    "descriptor-1": "descriptor-lazy",
    "descriptor-2": "descriptor-lazy",
    "mmap-2": "name-order",
}
LENGTHS = (512, 2048, 8192, 32736)


def require(check, message):
    if not check:
        raise RuntimeError(message)


def compare(ordinary, prior, fault, samples):
    require(ordinary["status"] == fault["status"] == "complete", "incomplete receipt")
    require(
        ordinary["pack_revision"] == prior["pack_revision"] == fault["pack_revision"]
        and ordinary["build"]["server_sha256"] == fault["server_sha256"],
        "mixed source or pack",
    )
    require(tuple(ordinary["arms"]) == tuple(READERS), "wrong launch order")
    reference = ordinary["arms"]["mmap-1"]["cases"]
    result = {}
    for name, reader in READERS.items():
        arm = ordinary["arms"][name]
        require(
            arm["profile"] == {"effective_sequence_cap": 1, "expert_residency": "resident"}
            and arm["model"]["context_length"] == 32768
            and arm["context_overflow"] == {
                "http_status": 400, "error_code": "context_length_exceeded"
            },
            f"{name} changed the resident 32K profile",
        )
        require(arm["health_ready"] == arm["health_after_requests"] == "ok", f"{name} failed health")
        require(
            arm["sidecar"]["loader_mode"] == arm["loader_mode"] == reader
            and arm["sidecar"]["tensor_count"] == 31
            and arm["sidecar_initial_page_state"]["resident_pages"] == 1,
            f"{name} changed the reader or initial page state",
        )
        require(tuple(case["prompt_tokens"] for case in arm["cases"]) == LENGTHS, f"{name} incomplete")
        for case, baseline in zip(arm["cases"], reference, strict=True):
            length = case["prompt_tokens"]
            require(
                case["prompt_hash"] == baseline["prompt_hash"]
                and case["output_tokens"] == baseline["output_tokens"]
                and len(case["output_tokens"]) == 32,
                f"{name} changed prompt or output at {length}",
            )
            require(
                case["status"] == "finished" and case["finish_reason"] == "max_output_tokens"
                and case["mtp"]["available"] and not case["mtp"]["active"]
                and case["verified_steps"] == case["route_errors"] == 0,
                f"{name} changed direct generation at {length}",
            )
            if str(length) in prior["cases"]:
                expected = prior["cases"][str(length)]
                require(
                    all(case[field] == expected[field] for field in ("prompt_hash", "output_tokens")),
                    f"{name} changed prior output at {length}",
                )
        rows = samples[name]
        require(bool(rows) and all(row["pressure"] == 1 for row in rows), f"{name} pressure changed")
        require(
            max(row["reference_rss_bytes"] for row in rows) == arm["peak_sampled_rss_bytes"]
            and max(
                row["kernel_lifetime_peak_footprint_bytes"]
                for row in (*rows, arm["process_before_stop"])
            )
            == arm["max_observed_kernel_lifetime_peak_footprint_bytes"],
            f"{name} changed resource summary",
        )
        result[name] = {
            "ready_seconds": arm["ready_seconds"],
            "trunk_seconds": arm["trunk"]["elapsed_seconds"],
            "sidecar_seconds": arm["sidecar"]["elapsed_seconds"],
            "ttft_seconds": {case["prompt_tokens"]: case["timing"]["ttft_seconds"] for case in arm["cases"]},
            "memory_samples": len(rows),
        }

    for boundary in ("artifact-integrity", "head-attachment"):
        for policy in ("required", "disabled"):
            row = fault["arms"][boundary][policy]
            serves = boundary == "head-attachment" and policy == "disabled"
            require(
                row["listener_opened"] is serves and row["rebound_port"]
                and (row["returncode"] == 0) == serves
                and row["direct_tokens_match"] is (True if serves else None),
                f"wrong failure outcome at {boundary}/{policy}",
            )
            is_manifest = boundary == "artifact-integrity"
            require(
                row["failure_boundary"] == ("manifest" if is_manifest else "head attachment")
                and row["attachment_failures_logged"] == (0 if is_manifest else 1),
                f"wrong failure boundary at {boundary}/{policy}",
            )
    return {"matched_output_ids_per_arm": len(LENGTHS) * 32, "fault_controls": 4, "arms": result}


def replay():
    source = json.loads((HERE / "sources.json").read_bytes())
    data = (HERE / "evidence.tar.xz").read_bytes()
    require(sha256(data).hexdigest() == source["archive_sha256"], "changed evidence archive")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
        require(archive.getnames() == list(source["members_sha256"]), "changed archive members")
        contents = {name: archive.extractfile(name).read() for name in archive.getnames()}
    hashes = {name: sha256(contents[name]).hexdigest() for name in contents}
    require(hashes == source["members_sha256"], "changed evidence member")
    ordinary = json.loads(contents["ordinary.json"])
    fault = json.loads(contents["fault.json"])
    binds = {
        "run_m5_native_sidecar.py": ordinary["runner_sha256"],
        "check_fault_policy.py": fault["fault_runner_sha256"],
        "m5-sidecar.patch": ordinary["build"]["source_patch_sha256"],
        "loader_footprint.c": ordinary["footprint_probe_source_sha256"],
        "model-manifest.json": ordinary["model_manifest_sha256"],
        "prior-native-prefixes.json": ordinary["prior_native_prefixes_sha256"],
        "publisher-inventory.json": ordinary["inventory_sha256"],
        "sidecar-selection.json": ordinary["sidecar_selection_sha256"],
    }
    require(all(hashes[name] == digest for name, digest in binds.items()), "changed experiment input")
    require(json.loads(contents["build.json"]) == ordinary["build"], "different build receipt")
    samples = {
        name: [json.loads(line) for line in contents[f"{name}-samples.jsonl"].splitlines()]
        for name in READERS
    }
    return compare(ordinary, json.loads(contents["prior-native-prefixes.json"]), fault, samples)


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
