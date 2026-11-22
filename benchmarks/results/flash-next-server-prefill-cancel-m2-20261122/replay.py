"""Replay the Flash Next HTTP disconnect and sibling recovery."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EARLY = ROOT.parent / "flash-next-server-prefill-early-m2-20261122"


def checksums(root):
    return dict(line.split()[::-1] for line in (root / "stage.sha256").read_text().splitlines())


def replay():
    spec = importlib.util.spec_from_file_location("early_arrival_verifier", EARLY / "replay.py")
    if spec is None or spec.loader is None:
        raise ValueError("Missing early-arrival verifier")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    require = frozen.require

    hashes = checksums(ROOT)
    prior = checksums(EARLY)
    prior.pop("run_m2.sh")
    sources = ("run_cancel.py", "run_m2.sh")
    for name in sources:
        require(frozen.digest(ROOT / name) == hashes.pop(name), f"Cancellation stage changed: {name}")
    require(hashes == prior, "The cancellation stage changed the sealed binary, helpers, prompts or oracle")

    with frozen.inputs(EARLY) as (early, folder):
        (folder / "run_cancel.py").write_bytes((ROOT / "run_cancel.py").read_bytes())
        import run_cancel as driver

        _, build, inventory, oracle_receipt, oracle, _ = early.inputs()
        samples_name = f"{driver.LABEL}-samples.jsonl"
        names = ("receipt.json", "status.txt", "raw.json", samples_name)
        frozen.unpack(ROOT / "evidence.tar.xz", folder, {name: name for name in names})
        receipt = frozen.read(folder / "receipt.json")
        raw_path = folder / "raw.json"
        samples_path = folder / samples_name
        require((folder / "status.txt").read_text() == "0\n"
                and receipt["scope"] == "M2 Ultra 192 GB HTTP disconnect during prefill"
                and receipt["qualification"] is False and receipt["build"] == build
                and receipt["pack_revision"] == inventory["revision"]
                and receipt["manifest"] == oracle_receipt["manifest"]
                and receipt["inventory_sha256"] == oracle_receipt["inventory_sha256"]
                and receipt["oracle_sha256"] == oracle_receipt["solo"]["data_sha256"]
                and receipt["raw_sha256"] == frozen.digest(raw_path)
                and receipt["pressure_sha256"] == frozen.digest(samples_path),
                "The cancellation receipt or sealed observations changed")
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
        require(samples and len(samples) == receipt["pressure_samples"]
                and all(row["pressure"] == 1 for row in samples), "The pressure samples changed")
        raw = frozen.read(raw_path)
        result = driver.verify(raw, oracle)
        require(result == receipt["comparison"], "The recorded summary does not replay")
        for role, cursor, widths in (("survivor", 512, [512]), ("replacement", 8192, [2048] * 4)):
            row = raw[role]
            trace = row["trace"]
            require(trace["request_id"] == row["result"]["request_id"]
                    and trace["initial_cursor"] == 0 and trace["cursor"] == cursor
                    and [step["width"] for step in trace["prefill"]] == widths
                    and all(step["scheduled"] == 1 and step["split_end"] is None for step in trace["prefill"])
                    and not trace["partial"],
                    f"The {role} trace reused or changed an unrelated request's prefill")
        return {"scope": receipt["scope"], "qualification": False,
                "runtime_commit": build["runtime_commit"], "pack_revision": inventory["revision"],
                "normal_pressure_samples": len(samples), **result}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
