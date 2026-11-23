"""Replay the reverse Flash Next canonical comparison."""
import json
from pathlib import Path
from runpy import run_path
import sys
import tarfile

root = Path(__file__).resolve().parent
forward = root.parent / "flash-next-canonical-cost-m5-20261123" / "replay.py"


def replies(archive, mode):
    with archive.extractfile(f"raw/rows-{mode}.jsonl") as stream:
        rows = [json.loads(line) for line in stream]
    return {row["id"]: (row["prompt_sha256"], row["output_tokens"], row["finish_reason"]) for row in rows}


def main():
    replay = run_path(str(forward))["replay"]
    first = forward.with_name("evidence.tar.xz")
    second = root / "evidence.tar.xz"
    if replay(first) != forward.with_name("comparison.json").read_bytes():
        raise ValueError("The direct-first comparison changed")
    reverse = replay(second)
    with tarfile.open(first) as prior, tarfile.open(second) as repeated:
        expected = replies(prior, "disabled")
        if any(replies(archive, mode) != expected for archive in (prior, repeated)
               for mode in ("disabled", "required")):
            raise ValueError("Prompt hashes, output IDs or finish reasons differ between orders")
    sys.stdout.buffer.write(reverse)


if __name__ == "__main__":
    main()
