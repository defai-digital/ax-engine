"""Measure startup warmup before native serving on a local APFS view of the publisher pack."""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import check_flash_next_native_support as native
from run_load_order import fresh_clone
from run_warmup import MANIFEST

ROOT = Path(__file__).resolve().parent
parser = ArgumentParser(description=__doc__)
parser.add_argument("cache", type=Path)
args = parser.parse_args()
cache = args.cache.resolve(strict=True)
members = json.loads((ROOT / "inventory.json").read_bytes())
source = cache / ("models--" + members["repo_id"].replace("/", "--")) / "snapshots" / members["revision"]
manifest = source / "model-manifest.json"
if native.sha256(manifest) != MANIFEST["model-manifest.json"]:
    raise ValueError("The delivered manifest changed")
with TemporaryDirectory(prefix="startup-view-", dir=ROOT) as directory:
    pack = Path(directory)
    for member in members["members"]:
        path = (source / member["name"]).resolve(strict=True)
        if not path.is_relative_to(cache) or path.stat().st_size != member["size"]:
            raise ValueError("A publisher member escaped the cache or changed in size")
        destination = pack / member["name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        fresh_clone(path, destination)
    (pack / "model-manifest.json").write_bytes(manifest.read_bytes())
    subprocess.run(
        [sys.executable, "-B", "-u", ROOT / "run_warmup.py",
         "--server", ROOT / "ax-engine-server-startup", "--inventory", ROOT / "inventory.json",
         "--build", ROOT / "build.json", "--pack", pack, "--output", ROOT / "observed"],
        check=True,
    )
