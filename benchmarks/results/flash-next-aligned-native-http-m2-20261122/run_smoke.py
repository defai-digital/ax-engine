"""Compare the actual Flash Next resident reader through the native HTTP entry point."""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
from unittest.mock import patch

import check_flash_next_native_support as native

ROOT = Path(__file__).resolve().parent
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
READERS = (("default", False), ("default", True), ("required", True), ("required", False))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def pressure():
    return subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip()


def stable(row):
    response = row["response"]
    usage = response["usage"]
    return (row["name"], row["request"], response["choices"],
            tuple(usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")),
            row["activation_delta"])


def main():
    parser = ArgumentParser(description=__doc__)
    for field in ("server", "pack", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    build = json.loads((ROOT / "build.json").read_bytes())
    inv_file = ROOT / "inventory.json"
    if build["server_sha256"] != digest(args.server) or build["inventory_sha256"] != digest(inv_file):
        raise ValueError("Staged server or inventory changed")
    inventory = json.loads(inv_file.read_bytes())
    manifest = native.validate_inventory(args.pack, inventory)
    if manifest != {"model-manifest.json": MANIFEST}:
        raise ValueError("The ordinary MXFP4 manifest does not match frozen delivery")
    args.output.mkdir()
    original_env = native.server_env
    result = {}
    baseline = {}
    for mode, aligned in READERS:
        key = f'{mode}-{"aligned" if aligned else "mapped"}'
        output = args.output / key
        output.mkdir()
        before = pressure()
        if before != "1":
            raise ValueError("Wait for normal host pressure")
        options = {"RUST_LOG": "info,ax_engine_mlx::flash_next_startup=debug",
                   "AX_MLX_FLASH_NEXT_ALIGNED_NOCACHE": "1" if aligned else "0"}
        readiness = {}
        with patch.object(native, "server_env", lambda: original_env(options)):
            rows = native.run_mode(args.server.resolve(), args.pack.resolve(), output, mode, readiness)
        after = pressure()
        if after != "1":
            raise ValueError("Observed memory pressure exceeded normal")
        log = (output / f"server-{mode}.log").read_text(errors="replace")
        expected = str(aligned).lower()
        routes = re.findall(r"Flash Next resident reader aligned_nocache=(true|false)", log)
        if routes != [expected] or "decision=resident" not in log:
            raise ValueError(f"{key}: the path did not select the expected owned reader and residency")
        shard_tensors = [int(n) for n in re.findall(r"Flash Next resident shard[^\n]*tensor_count=(\d+)", log)]
        if len(shard_tensors) != 15 or sum(shard_tensors) != 1942:
            raise ValueError(f"{key}: resident coverage changed")
        observed = list(map(stable, rows))
        if mode in baseline and observed != baseline[mode]:
            raise ValueError(f"{key}: response, usage, or verification steps changed")
        baseline[mode] = observed
        result[key] = {"mode": mode, "aligned_nocache": aligned, "matches": len(rows),
                       "ready_seconds": readiness[mode], "native_shards": len(shard_tensors),
                       "native_tensors": sum(shard_tensors),
                       "verified_steps": sum(row["activation_delta"][native.VERIFIED] for row in rows),
                       "pressure": [before, after], "rows": rows}
        print(f"{key}: all {len(rows)} controls passed", flush=True)
    if native.validate_inventory(args.pack, inventory) != manifest:
        raise ValueError("Publisher files changed")
    (args.output / "result.json").write_text(json.dumps({"completed": True, "scope": "M2 Ultra functional native HTTP",
        "qualification": False, "build": build, "pack_revision": inventory["revision"],
        "manifest": manifest, "runs": result}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
