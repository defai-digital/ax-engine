"""Serial fresh-clone profile of the two resident reader routes."""
import ctypes
from hashlib import sha256
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep


ARMS = (("mmap", "short-first"), ("pread", "short-first"),
        ("pread", "long-first"), ("mmap", "long-first"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    production, pack, mlx_library = map(Path, sys.argv[1:])
    sys.path.insert(0, str(production / "scripts"))
    import check_flash_next_native_support as native

    root = Path.cwd()
    clonefile = ctypes.CDLL(None, use_errno=True).clonefile
    clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    clonefile.restype = ctypes.c_int
    pack_inventory = Path("benchmarks/results/qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json")
    inventory_path = production / pack_inventory
    matrix = json.loads((production / "evidence/generate-matrix/result.json").read_bytes())
    inventory = json.loads(inventory_path.read_bytes())
    if digest(inventory_path) != matrix["inventory_sha256"]:
        raise ValueError("Current native matrix and publisher inventory differ")
    if len(inventory["members"]) != 49:
        raise ValueError("Unexpected publisher member count")
    manifest = native.validate_inventory(pack, inventory)
    if not manifest:
        raise ValueError("The resident pack is missing its native manifest")
    filenames = [row["name"] for row in inventory["members"]] + list(manifest)
    report = {"status": "running", "hardware": "MacBook Pro M5 Max 128 GB, internal NVMe SSD",
              "pack_revision": inventory["revision"], "inventory_sha256": digest(inventory_path),
              "native_manifest": manifest, "probe_sha256": digest(root / "reader-probe"),
              "prompt_json_sha256": digest(root / "prompts.json"), "arms": {}}

    def save():
        tmp = root / "result.tmp"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(root / "result.json")

    save()
    for mode, order in ARMS:
        names = subprocess.check_output(["/bin/ps", "-axo", "comm="], text=True).splitlines()
        if any(Path(name.strip()).name.startswith(("ax-engine-server", "profile-probe", "reader-probe"))
               for name in names):
            raise RuntimeError("Another model process is active")
        label = f"{mode}-{order}"
        with TemporaryDirectory(dir=root) as temp:
            clone = Path(temp)
            for name in filenames:
                out = clone / name
                out.parent.mkdir(parents=True, exist_ok=True)
                if clonefile(os.fsencode(pack / name), os.fsencode(out), 0):
                    raise OSError(ctypes.get_errno(), f"APFS clone failed for {name}")
            if report["arms"]:
                sleep(20)
            quiet = native.wait_for_quiet()
            if subprocess.check_output(["/usr/sbin/sysctl", "-n",
                                        "kern.memorystatus_vm_pressure_level"]).strip() != b"1":
                raise RuntimeError("Preflight memory pressure is not normal")
            env = {name: value for name, value in os.environ.items()
                   if not name.startswith(("AX_", "MLX_", "DYLD_"))}
            env.update(AX_FLASH_NEXT_RESIDENT_POSITIONAL="1" if mode == "pread" else "",
                       AX_FLASH_NEXT_READER_PROBE_ORDER=order, AX_FLASH_NEXT_PROFILE_PACK=str(clone),
                       AX_FLASH_NEXT_READER_PROBE_PROMPTS=str(root / "prompts.json"),
                       AX_FLASH_NEXT_PROFILE_OUTPUT=str(root / f"{label}.json"))
            command = ["/usr/bin/caffeinate", "-i", "/usr/bin/time", "-l",
                       "/usr/bin/env", f"DYLD_LIBRARY_PATH={mlx_library}",
                       str(root / "reader-probe"), "--ignored", "--nocapture",
                       "flash_next_resident_reader_warmed_prefill_profile"]
            pressures = []
            began = monotonic()
            with (root / f"{label}.log").open("x") as log, subprocess.Popen(
                command, env=env, stdout=log, stderr=log, start_new_session=True
            ) as child:
                try:
                    while child.poll() is None:
                        pressure = int(subprocess.check_output([
                            "/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"
                        ]))
                        pressures.append(pressure)
                        if pressure >= 4 or monotonic() - began > 1200:
                            raise RuntimeError(f"Pressure or deadline stopped the owned {label} arm")
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                    if child.returncode:
                        raise subprocess.CalledProcessError(child.returncode, command)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=10)
            data = json.loads((root / f"{label}.json").read_bytes())
            if (data["loader"], data["order"]) != (mode, order):
                raise ValueError("Reader mode or warmup order did not match")
            report["arms"][label] = {"host_quiet": quiet, "host_idle_after": native.cpu_idle_percent(),
                                     "pressure_samples": pressures,
                                     "pressure_after": int(subprocess.check_output([
                                         "/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])),
                                     "profile": data}
            save()
    if native.validate_inventory(pack, inventory) != manifest:
        raise ValueError("Publisher files changed during the profile")
    profiles = [sorted((row["prompt_tokens"], row["first_token"], row["serialized_state_sha256"],
                        row["forwards"]) for row in arm["profile"]["profiles"])
                for arm in report["arms"].values()]
    if any(profile != profiles[0] for profile in profiles[1:]):
        raise ValueError("Reader comparison changed the serialized state, token or forward schedule")
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
