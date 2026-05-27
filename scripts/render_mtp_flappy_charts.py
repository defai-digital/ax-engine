#!/usr/bin/env python3
"""Render the README MTP benchmark charts.

This script writes the MTP SVGs shown before the standard nine README
performance charts:

  docs/assets/perf-mtp-speed-summary.svg
  docs/assets/perf-mtp-quality-summary.svg
  docs/assets/perf-mtp-speed-tok-s.svg
  docs/assets/perf-mtp-speed-accept-rate.svg
  docs/assets/perf-mtp-quality-tok-s.svg
  docs/assets/perf-mtp-quality-accept-rate.svg

Each chart contains only rows with an artifact-backed MTPLX 0.3.7 reference.
Within each model/suite group MTPLX is rendered on the left and AX Engine MTP
on the right. The summary charts use tok/s as the primary axis and include the
accept rate in each bar label.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_readme_performance_charts import (  # noqa: E402
    ChartError,
    MTP_CHART_OUTPUTS,
    MTP_SUMMARY_CHART_OUTPUTS,
    load_mtp_rows,
    mtp_row_key,
    render_mtp_metric_chart,
    render_mtp_summary_chart,
    write_chart,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance-doc",
        type=Path,
        default=Path("docs/PERFORMANCE.md"),
        help="Performance doc containing the MTP Mode benchmark table.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated SVGs match files on disk without writing.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mtp_rows = load_mtp_rows(args.performance_doc)
    mismatches: list[Path] = []

    for row_key, output_name in MTP_SUMMARY_CHART_OUTPUTS.items():
        rows = [row for row in mtp_rows if mtp_row_key(row) == row_key]
        if not rows:
            raise ChartError(f"MTP performance table has no {row_key!r} rows")
        output_path = args.output_dir / output_name
        content = render_mtp_summary_chart(rows)
        if write_chart(output_path, content, args.check):
            if not args.check:
                print(f"wrote {output_path}")
        else:
            mismatches.append(output_path)

    for (row_key, metric), output_name in MTP_CHART_OUTPUTS.items():
        rows = [row for row in mtp_rows if mtp_row_key(row) == row_key]
        if not rows:
            raise ChartError(f"MTP performance table has no {row_key!r} rows")
        output_path = args.output_dir / output_name
        content = render_mtp_metric_chart(rows, metric)
        if write_chart(output_path, content, args.check):
            if not args.check:
                print(f"wrote {output_path}")
        else:
            mismatches.append(output_path)

    if mismatches:
        for path in mismatches:
            print(f"chart is stale: {path}", file=sys.stderr)
        return 1

    if args.check:
        print("MTP README charts are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
