"""Exercise the real Flash Next required-MTP preflight with AX_NO_SPEC."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
INVENTORY = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
ACTIVATION_ERROR = ("no usable drafter was activated", "MlxMtpRequiredButUnavailable")
PAYLOAD_MARKERS = ("Flash Next resident shard", 'phase="resident_tensors"', 'phase="mtp_sidecar"',
                   "MTP admission at runner construction", "wired GPU residency for this session")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def require(valid, message):
    if not valid:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "server", "pack", "inventory", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.repo / "scripts"))
    import check_flash_next_native_support as native

    manifest = args.pack / "model-manifest.json"
    require(digest(manifest) == MANIFEST, "The qualified manifest changed")
    require(digest(args.inventory) == INVENTORY, "The publisher inventory changed")
    inventory = json.loads(args.inventory.read_bytes())
    require(inventory["revision"] == REVISION and len(inventory["members"]) == 49, "Wrong publisher revision")
    for member in inventory["members"]:
        path = args.pack / member["name"]
        require(path.stat().st_size == member["size"], f"Publisher size drift for {member['name']}")
        if member["name"] in ("config.json", "model.safetensors.index.json", "mtplx_runtime.json"):
            require(digest(path) == member["sha256"], f"Publisher metadata drift for {member['name']}")

    env = native.server_env({"AX_NO_SPEC": "1", "RUST_LOG": "info,ax_engine_mlx::flash_next_startup=debug"})
    command = native.server_command(args.server, args.pack, "required", 0)
    start = time.monotonic()
    outcome = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=60, text=True, start_new_session=True)
    elapsed = time.monotonic() - start
    log = outcome.stdout.replace(str(args.pack), "[model]").replace(str(Path.home()), "[home]")
    log = log.replace("127.0.0.1", "[loopback]")
    (args.output / "server.log").write_text(log)
    error = next((pattern for pattern in ACTIVATION_ERROR if pattern in outcome.stdout), None)
    markers = {marker: outcome.stdout.count(marker) for marker in PAYLOAD_MARKERS}
    require(outcome.returncode > 0 and error is not None, "The published MTP activation error did not surface")
    require(not any(markers.values()), "The process initialized MLX or loaded payloads before rejecting MTP")
    require(digest(manifest) == MANIFEST, "The qualified manifest changed during preflight")
    record = {
        "qualification": False,
        "scope": "Flash Next native CLI policy preflight",
        "hardware": {"soc": subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                                      text=True).strip(),
                     "form_factor": "Mac Studio",
                     "memory_bytes": int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True)),
                     "os": platform.mac_ver()[0], "storage": "external APFS SSD"},
        "server_sha256": digest(args.server),
        "pack_revision": REVISION,
        "manifest_sha256": MANIFEST,
        "inventory_sha256": INVENTORY,
        "inventory_verified_members": 49,
        "inventory_scope": "member sizes and runtime metadata hashes",
        "policy": "required",
        "environment": {"AX_NO_SPEC": "1"},
        "process_elapsed_seconds": elapsed,
        "exit_code": outcome.returncode,
        "activation_error": error,
        "payload_startup_markers": markers,
    }
    (args.output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"Required MTP rejected in {elapsed:.3f} s before loading Flash Next weights.")


if __name__ == "__main__":
    main()
