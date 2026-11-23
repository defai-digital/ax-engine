"""Recompute the four M5 natural-load decode-drain receipts from original traces."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from statistics import median
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


def client_windows(root, comparison):
    report = json.loads(comparison)
    result = {}
    for arm, stats in report["arms"].items():
        observed = root / "observed" / arm
        raw = json.loads((observed / f"{arm}.json").read_bytes())
        rows = [row for row in raw["observations"] if row["phase"] == "measured"]
        cpu_path = observed / f"{arm}-cpu.jsonl"
        clock = [json.loads(line) for line in cpu_path.read_text().splitlines()]
        origin = clock[0]["monotonic_seconds"]
        slack = stats["client_cpu_window_extra_seconds"]
        assert min(row["scheduled_at_s"] for row in rows) == 0

        # The client monitor starts before the benchmark clock and ends after
        # all measurements. The difference in spans bounds the clock offset.
        intervals = []
        for older, newer in zip(clock, clock[1:]):
            deltas = {key: (newer["ticks"][key] - old) % (1 << 32) for key, old in older["ticks"].items()}
            earliest = older["monotonic_seconds"] - origin - slack
            latest = newer["monotonic_seconds"] - origin
            intervals.append((earliest, latest, deltas["idle"], sum(deltas.values())))

        windows = []
        complete_minutes = int(max(row["scheduled_at_s"] for row in rows) // 60)
        for minute in range(complete_minutes):
            start = minute * 60
            end = start + 60
            arrivals = [row for row in rows if start <= row["scheduled_at_s"] < end]
            inside = [sample for sample in intervals if start <= sample[0] and sample[1] <= end]
            assert arrivals and len(inside) >= 8
            idle = 100 * sum(sample[2] for sample in inside) / sum(sample[3] for sample in inside)
            windows.append({
                "minute": minute + 1,
                "client_lane_wait_median_s": round(median(row["queue_delay_ms"] / 1000 for row in arrivals), 3),
                "host_cpu_idle_percent": round(idle, 3),
                "whole_cpu_intervals": len(inside),
            })
        result[arm] = {"clock_offset_upper_bound_s": slack, "conservative_minute_windows": windows}
    output = {"comparison_sha256": sha256(comparison).hexdigest(),
              "scope": "Descriptive time alignment of host CPU and client queues", "arms": result}
    return (json.dumps(output, indent=2, allow_nan=False) + "\n").encode()


def replay(*, show_client_windows=False):
    with (TemporaryDirectory(prefix="flash-next-natural-drain-replay-") as scratch,
          tarfile.open(HERE / "evidence.tar.xz") as archive):
        root = Path(scratch)
        names = archive.getnames()
        assert len(set(names)) == len(names)
        for entry in archive:
            name = PurePosixPath(entry.name)
            assert entry.isfile() and not name.is_absolute() and ".." not in name.parts
            assert name.name != "ax-engine-server"
        archive.extractall(root, filter="data")
        manifest = (root / "inputs.sha256").read_bytes()
        assert (root / "bundle/inputs.sha256").read_bytes() == manifest
        entries = {name: digest for digest, name in map(str.split, manifest.decode().splitlines())}
        bundle = root / "bundle"
        assert {item.name for item in bundle.iterdir()} == entries.keys() - {"ax-engine-server"} | {"inputs.sha256"}
        for name, digest in entries.items():
            if name != "ax-engine-server":
                assert sha256((bundle / name).read_bytes()).hexdigest() == digest, name
        build = json.loads((root / "build.json").read_bytes())
        assert build["codesign_verified"] and build["server_sha256"] == entries["ax-engine-server"]
        assert build["patch_sha256"] == entries["source.patch"]
        redactions = json.loads((root / "redactions.json").read_bytes())
        assert set(redactions) == {"mixed-first", "drain-first", "drain-second", "mixed-second"}
        for arm, note in redactions.items():
            observed = root / "observed" / arm
            data = (observed / f"{arm}.json").read_bytes()
            assert sha256(data).hexdigest() == note["archived_sha256"]
            artifact = json.loads(data)
            assert artifact["target"]["base_url"] == "[loopback]"
            assert artifact["corpus"]["path"] == "mixed-corpus.jsonl"
        decision = json.loads((root / "dispatch-decision.json").read_bytes())
        targeted = HERE.parent / "flash-next-priority-short-drain-m5-20261123" / "comparison.json"
        assert decision["start"] is True
        raw_targeted = targeted.read_bytes()
        assert sha256(raw_targeted).hexdigest() == decision["crossover_sha256"]
        targeting = json.loads(raw_targeted)
        assert targeting["build"] == build and targeting["qualification"] is False
        assert decision["eligible_targeted_orders"] == [
            name for name, row in targeting["comparisons"].items() if row["timing_eligible"]
        ]
        assert decision["eligible_targeted_orders"]
        comparison = subprocess.check_output([sys.executable, "-B", root / "compare.py", root])
        return client_windows(root, comparison) if show_client_windows else comparison


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--client-window", action="store_true", help="Recompute the host CPU and queue time series")
    args = cli.parse_args()
    sys.stdout.buffer.write(replay(show_client_windows=args.client_window))
