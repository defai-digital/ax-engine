"""Seal the exact delivery/control bytes and remove the ephemeral cache path."""
from argparse import ArgumentParser
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import tarfile

MODES = ("disabled", "required", "default")
PHASES = {
    "memory_plan": "memory plan",
    "runner_admission": "MTP admission at runner construction",
    "http_listening": "preview listening bind_address=",
}
UTC = re.compile(r"^(\d{4}-\d\d-\d\dT[0-9:.]+Z)\s")


def read(archive, name):
    item = archive.getmember(name)
    if not item.isfile():
        raise ValueError(f"Not a regular evidence file: {name}")
    with archive.extractfile(item) as stream:
        return stream.read()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def events(raw):
    times = {}
    for line in raw.decode(errors="replace").splitlines():
        for label, marker in PHASES.items():
            if marker in line:
                stamp = UTC.match(line)
                if label in times or stamp is None:
                    raise ValueError(f"Ambiguous startup marker {label}")
                times[label] = stamp.group(1)
    if list(times) != list(PHASES):
        raise ValueError("Missing or out-of-order startup log markers")
    return {"raw_log_sha256": sha256(raw).hexdigest(), "utc": times}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tarfile.open(args.source) as original, tarfile.open(args.logs) as logs:
        result = read(original, "delivery/result.json")
        support = read(original, "delivery/native-support/result.json")
        download = read(original, "delivery/download.ndjson")
        record = json.loads(result)
        terminal = record["download"]["terminal"]
        endline = json.loads(download.splitlines()[-1])
        if terminal != endline or record["native_support"]["result_sha256"] != sha256(support).hexdigest():
            raise ValueError("Raw delivery and control receipts disagree")
        snapshot = terminal["dest"]
        portable = "[cache]/snapshots/" + record["revision"]
        terminal["dest"] = portable
        terminal["manifest_path"] = terminal["manifest_path"].replace(snapshot, portable)
        terminal["server_command"] = [part.replace(snapshot, portable) for part in terminal["server_command"]]
        # Leave prior progress records byte-identical and keep the original digests.
        portable_dl = download.rstrip(b"\n").rsplit(b"\n", 1)[0] + b"\n" + json.dumps(terminal, allow_nan=False).encode() + b"\n"
        output = {
            "build.json": read(original, "build.json"),
            "inputs.sha256": read(original, "inputs.sha256"),
            "delivery.json": json_bytes(record),
            "native-support.json": support,
            "download.ndjson": portable_dl,
            "host-samples.jsonl": read(original, "watch-startup.jsonl"),
            "startup-events.json": json_bytes({mode: events(read(logs, f"server-{mode}.log")) for mode in MODES}),
        }
        originals = {"delivery.json": result, "native-support.json": support, "download.ndjson": download}
        output["source-sha256.json"] = json_bytes({
            name: {"original": sha256(raw).hexdigest(), "portable": sha256(output[name]).hexdigest()}
            for name, raw in originals.items()
        })
    forbidden = re.compile(rb"/Users/|/Volumes/|/private/|/var/folders/|\b127\.0\.0\.1\b|df-macbook|um-macstudio|tn-macstudio")
    for name, data in output.items():
        if forbidden.search(data):
            raise ValueError(f"Unexpected local identifier in {name}")
    with tarfile.open(args.output, "w:xz", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in output.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))


if __name__ == "__main__":
    main()
