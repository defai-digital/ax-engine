"""Recompute the Flash Next Required MTP reader comparison from captured evidence."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
VOCAB_SIZE = 248_320


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def replay(root):
    with TemporaryDirectory() as temp:
        files = Path(temp)
        with tarfile.open(root / "evidence.tar.xz") as archive:
            for member in archive:
                path = Path(member.name)
                require(member.isfile() and not path.is_absolute() and ".." not in path.parts,
                        "Unexpected evidence path")
                target = files / path
                require(not target.exists(), "Duplicate evidence file")
                target.parent.mkdir(exist_ok=True, parents=True)
                target.write_bytes(archive.extractfile(member).read())

        input_digests = {name: checksum for checksum, name in
                         (line.split() for line in (files / "inputs.sha256").read_text().splitlines())}
        elided = {"ax-engine-server", "loader_footprint.dylib"}
        for name, checksum in input_digests.items():
            if name not in elided:
                require(digest(files / name) == checksum, f"Input digest changed for {name}")

        sys.path.insert(0, str(files))
        from run_required_sidecar import ARMS, attach_loader_trace, compare

        observed = files / "observed"
        receipt = json.loads((observed / "receipt.json").read_bytes())
        build = json.loads((files / "build.json").read_bytes())
        inventory = json.loads((files / "mxfp4-inventory.json").read_bytes())
        require((observed / "status.txt").read_text() == "0\n" and receipt["status"] == "complete",
                "Campaign did not finish")
        require(receipt["build"] == build and build["server_sha256"] == input_digests["ax-engine-server"]
                and build["libmlx_sha256"] == receipt["libmlx_sha256"],
                "Unexpected server build")
        for receipt_field, name in (
            ("runner_sha256", "run_required_sidecar.py"),
            ("model_manifest_sha256", "model-manifest.json"),
            ("inventory_sha256", "mxfp4-inventory.json"),
            ("resident_page_selection_sha256", "resident-page-selection.json"),
            ("footprint_probe_source_sha256", "loader_footprint.c"),
            ("footprint_probe_sha256", "loader_footprint.dylib"),
        ):
            require(receipt[receipt_field] == input_digests[name], f"Receipt changed for {name}")
        require(len(inventory["members"]) == 49 and receipt["pack_revision"] == inventory["revision"]
                and receipt["launch"] == "required-resident-single-request-8224-context",
                "The measured pack or profile changed")

        for label, mode, _ in ARMS:
            arm = receipt["arms"][label]
            trace = {}
            attach_loader_trace((observed / f"{label}.log").read_text(), mode, trace)
            require(all(arm[key] == value for key, value in trace.items()),
                    f"Loader trace differs for {label}")
            samples = [json.loads(line) for line in
                       (observed / f"{label}-samples.jsonl").read_text().splitlines()]
            require(sorted({s["pressure"] for s in samples}) == arm["pressure_levels"] == [1],
                    f"Unexpected resource conditions for {label}")
            require(sum(row["verified_steps"] for row in arm["cases"]) == 279,
                    f"Unexpected verifier total for {label}")

        comparison = compare(receipt["arms"], VOCAB_SIZE)
        # JSON converts numeric prompt-length dictionary keys to strings.
        comparison = json.loads(json.dumps(comparison, allow_nan=False))
        require(comparison == receipt["diagnostic_comparison"]
                == json.loads((root / "comparison.json").read_bytes()),
                "Reported comparison differs from measured cases")
        return comparison


if __name__ == "__main__":
    comparison = replay(ROOT)
    print(f"{len(comparison['arms'])} arms, {comparison['matched_output_ids_per_arm']} matching IDs each")
    print(f"Warmed timing band met: {comparison['warmed_timing_band_met']}")
