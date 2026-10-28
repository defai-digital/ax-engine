#!/usr/bin/env python3
"""Independent dependency-admission checks for sliced scans and byte reductions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def run(contiguous_scans: bool = False) -> dict:
    cases = []
    failures = []
    for shape, start in [((1, 191), 173), ((5, 71), 57), ((3, 43), 39), ((1, 4097), 4081)]:
        host = (np.arange(np.prod(shape)).reshape(shape) % 23 - 11).astype(np.float32)
        base = mx.array(host)
        sentinels = [mx.full((29,), 19 + i, dtype=mx.float32) for i in range(16)]
        mx.eval(base, *sentinels)
        view = base[:, start:]
        for axis in [0, 1]:
            for reverse in [False, True]:
                for inclusive in [False, True]:
                    source = host[:, start:]
                    if reverse:
                        source = np.flip(source, axis=axis)
                    want = np.cumsum(source, axis=axis)
                    if not inclusive:
                        want -= source
                    if reverse:
                        want = np.flip(want, axis=axis)
                    if contiguous_scans:
                        moved = mx.contiguous(mx.moveaxis(view, axis, -1))
                        if moved.shape[-1] == 1:
                            dtype = mx.int32 if moved.dtype == mx.bool_ else moved.dtype
                            scanned = moved.astype(dtype) if inclusive else mx.zeros(moved.shape, dtype=dtype)
                        else:
                            scanned = mx.cumsum(moved, axis=-1, reverse=reverse, inclusive=inclusive)
                        got = mx.moveaxis(scanned, -1, axis)
                    else:
                        got = mx.cumsum(view, axis=axis, reverse=reverse, inclusive=inclusive)
                    mx.eval(got)
                    np.testing.assert_array_equal(np.array(got), want)
                    np.testing.assert_array_equal(np.array(base), host,
                        err_msg=f"source changed: {shape=} {start=} {axis=} {reverse=} {inclusive=}")
                    for i, sentinel in enumerate(sentinels):
                        np.testing.assert_array_equal(np.array(sentinel), np.full(29, 19 + i))
                    cases.append({"op": "cumsum", "shape": shape, "start": start,
                                  "axis": axis, "reverse": reverse, "inclusive": inclusive})
    for dtype in [np.int8, np.uint8]:
        for fill in [0, 1, 127, -128 if dtype == np.int8 else 255]:
            source = np.full((257, 73), fill, dtype=dtype)
            source[37, 19] = 0 if fill else 1
            base = mx.array(source)
            for name, host, device in [
                ("dense", source, base),
                ("transpose", source.T, base.T),
                ("slice", source[::3, 2::2], base[::3, 2::2]),
            ]:
                for axis in [None, 0, 1]:
                    for op in ["any", "all"]:
                        got = getattr(mx, op)(device, axis=axis)
                        mx.eval(got)
                        case = {"op": op, "dtype": np.dtype(dtype).name, "fill": fill,
                                "layout": name, "axis": axis}
                        want = getattr(np, op)(host, axis=axis)
                        if not np.array_equal(np.array(got), want):
                            failures.append(case)
                        cases.append(case)
    return {"mlx_version": mx.__version__, "status": "fail" if failures else "pass",
            "cases": cases, "failures": failures,
            "contiguous_scan_workaround": contiguous_scans,
            "scan_axis_moved_last": contiguous_scans,
            "singleton_scan_bypass": contiguous_scans,
            "scope": "Synthetic correctness only; canaries do not prove absence of out-of-bounds reads."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contiguous-scans", action="store_true",
                        help="Exercise the AX scan workaround instead of raw dependency scans")
    args = parser.parse_args()
    report = run(args.contiguous_scans)
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
