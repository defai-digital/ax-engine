"""Recompute the fixed-tile MLX allocation comparison from measured rows."""

from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "flash-next-qsa-prefill-scratch-20261123"


def replay():
    baseline = (PRIOR / "measure_tiles.py").read_bytes()
    manifest = dict((Path(name).name, checksum) for checksum, name in
                    map(str.split, (PRIOR / "SHA256SUMS").read_text().splitlines()))
    assert sha256(baseline).hexdigest() == manifest["measure_tiles.py"]
    prior = json.loads((PRIOR / "observed-2048.json").read_bytes())
    original = next(row for row in prior["modes"] if row["mode"] == "lazy" and row["sub_batch_queries"] == 255)
    rows = json.loads((HERE / "observed.json").read_bytes())
    assert [(r["queries"], r["history"]) for r in rows] == [
        (512, 8192), (2048, 8192), (512, 32768), (2048, 32768),
    ]
    assert set(rows[0]["measurements"]) == {"0", "1", "2", "4"}
    for cell in rows:
        assert cell["exact_bf16"] is True and cell["selected"] == 2051 and cell["tile_width"] == 255
        groups = cell["measurements"]
        assert set(groups) == {"0", "1", "2", "4"}
        assert all(len(records) == 3 for records in groups.values())
        all_hashes = set()
        for group, records in groups.items():
            for record in records:
                assert math.isfinite(record["time_ms"]) and record["time_ms"] > 0
                assert type(record["peak_bytes"]) is int and record["peak_bytes"] > 0
                value = record["output_sha256"]
                assert len(bytes.fromhex(value)) == 32
                all_hashes.add(value)
            assert cell["peak_bytes_by_group"][group] == max(row["peak_bytes"] for row in records)
            assert cell["median_ms_by_group"][group] == median(row["time_ms"] for row in records)
        assert len(all_hashes) == 1
        if (cell["queries"], cell["history"]) == (2048, 8192):
            assert all_hashes == {original["output_sha256"]}
    return {
        "qualification": False,
        "branch": "synthetic selected attention",
        "mlx": "0.32.4",
        "source_input_sha256": manifest["measure_tiles.py"],
        "cases": [
            {
                "physical_rows": row["queries"], "history_rows": row["history"],
                "max_peak_bytes_by_group": row["peak_bytes_by_group"],
                "median_ms_by_group": row["median_ms_by_group"],
                "group_two_peak_fraction": row["peak_bytes_by_group"]["2"] / row["peak_bytes_by_group"]["0"],
                "all_outputs_bitwise_equal": True,
            }
            for row in rows
        ],
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
