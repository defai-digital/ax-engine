"""Summarise the 15-stretch server QA records in ../raw/server-15-stretch-*.json.

Usage: python report_15_stretch.py [records-dir]   (default: ../raw)
"""
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent.parent / "raw"


def load(name):
    return json.load(open(base / f"server-15-stretch-{name}.json"))


off, default = load("f0"), load("def")
print("texts identical (default vs off):", all(a["sha"] == b["sha"] for a, b in zip(off["rows"], default["rows"])), "n =", len(default["rows"]))
print("decode off    ", [r["decode_tok_s"] for r in off["rows"]])
print("decode default", [r["decode_tok_s"] for r in default["rows"]])
print("first request ttft off/default:", off["first_hello"]["ttft"], default["first_hello"]["ttft"])
mtp_off, mtp_default = load("mtp0"), load("mtpd")
print("canonical MTP identical (default vs off):", all(a["sha"] == b["sha"] for a, b in zip(mtp_off["rows"], mtp_default["rows"])))
print("MTP decode off", [r["decode_tok_s"] for r in mtp_off["rows"]], "default", [r["decode_tok_s"] for r in mtp_default["rows"]])
rows = load("soak")["rows"]
decode = [r["decode_tok_s"] for r in rows]
print("soak n=%d distinct text SHA=%d decode min %.2f max %.2f first %.2f last %.2f rss %.0f -> %.0f MB" % (
    len(rows), len({r["sha"] for r in rows}), min(decode), max(decode), decode[0], decode[-1], rows[0]["rss_mb"], rows[-1]["rss_mb"]))
