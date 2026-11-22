"""Check Required MTP with one and three Flash Next worker warmup passes."""

from argparse import ArgumentParser
import json
from pathlib import Path

from run_warmup import INVENTORY, MANIFEST, REVISION, digest, measure, native, need, report_rows

MODE = "required"
ARMS = (("required-1", False), ("required-2", True))


def identity(raw):
    stable, _ = report_rows(raw, MODE)
    rows = raw["trajectories"] + raw["cells"]
    routing = []
    for row in rows:
        mtp = row["mtp"]
        need(mtp["available"] and mtp["requested"] and mtp["active"]
             and row["verified_steps"] > 0 and row["route_errors"] == 0
             and 0 <= mtp["accepted_tokens"] <= mtp["draft_tokens"],
             "A measured Required request did not exercise its verifier")
        routing.append({
            "verified_steps": row["verified_steps"], "schedule": row["verifier_schedule"],
            "mtp": mtp,
        })
    return stable | {"routing": routing}


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("server", "build", "pack", "inventory", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.functional = True
    build = json.loads(args.build.read_bytes())
    inventory = json.loads(args.inventory.read_bytes())
    need(digest(args.server) == build["server_sha256"] and digest(args.inventory) == INVENTORY
         and inventory["revision"] == REVISION, "Build or inventory changed")
    need(native.validate_inventory(args.pack, inventory) == MANIFEST, "The delivered manifest changed")
    args.output.mkdir()
    evidence = {
        "scope": "M2 Ultra 192 GB functional Required MTP worker warmup",
        "qualification": False,
        "build": build,
        "publisher_revision": REVISION,
        "inventory_sha256": INVENTORY,
        "manifest": MANIFEST,
        "arms": [],
    }
    oracle = None
    for name, single in ARMS:
        print(f"{name} starting.", flush=True)
        _, stats = measure(args, MODE, name, single)
        raw = json.loads((args.output / f"{name}.json").read_bytes())
        stable = identity(raw)
        if oracle is None:
            oracle = stable
        need(stable == oracle, "Required request outputs, stops or verifier routing changed")
        evidence["arms"].append(stats)
        (args.output / "progress.json").write_text(json.dumps(evidence, indent=2) + "\n")
    need(native.validate_inventory(args.pack, inventory) == MANIFEST, "Publisher bytes changed during the run")
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
