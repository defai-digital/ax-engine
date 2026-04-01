#!/usr/bin/env python3
"""
Summarize AX kernel profile invariants and heuristic candidates.

This tool reads `perfs/*.json`, flattens profile leaves, and classifies fields:

- invariant: same value across all profiles
- varying:   different values across profiles

It writes a JSON report and can also print a compact Markdown summary for PRD
and implementation planning.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


META_FIELDS = {"model", "source", "generated"}
HEURISTIC_COVERED_FIELDS = {
    "attention_decode.splitk_threshold",
    "batch_prefill.prefer_f16_io",
    "decode_regimes.short_max_attend_len",
    "decode_regimes.long.attention_decode.splitk_threshold",
    "attention_prefill.fa2_mode",
    "attention_prefill.ax_bc64_mode",
    "attention_decode.hd128_n2_default",
    "attention_prefill.ax_bc64_min_tokens",
}


@dataclass(frozen=True)
class ProfileEntry:
    name: str
    path: str
    family: str
    size: str


def infer_family_and_size(name: str) -> tuple[str, str]:
    stem = name.removesuffix(".json")
    if stem == "default":
        return ("default", "default")
    parts = stem.split("-")
    if len(parts) >= 2:
        return ("-".join(parts[:-1]), parts[-1])
    return (stem, "unknown")


def flatten(prefix: list[str], value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(prefix + [key], child, out)
        return
    out[".".join(prefix)] = value


def delete_flat_key(tree: dict[str, Any], flat_key: str) -> None:
    parts = flat_key.split(".")
    node: Any = tree
    parents: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        parents.append((node, part))
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
        else:
            break


def load_profiles(perf_dir: str) -> tuple[list[ProfileEntry], dict[str, dict[str, Any]]]:
    files = sorted(glob.glob(os.path.join(perf_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"no profile json files found under {perf_dir}")

    entries: list[ProfileEntry] = []
    flattened: dict[str, dict[str, Any]] = {}
    for path in files:
        name = os.path.basename(path)
        family, size = infer_family_and_size(name)
        entries.append(ProfileEntry(name=name, path=path, family=family, size=size))
        data = json.load(open(path))
        flat: dict[str, Any] = {}
        flatten([], data, flat)
        flattened[name] = flat
    return entries, flattened


def classify_fields(
    entries: list[ProfileEntry], flattened: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    field_values: dict[str, dict[str, Any]] = defaultdict(dict)
    for entry in entries:
        for field, value in flattened[entry.name].items():
            if field in META_FIELDS:
                continue
            field_values[field][entry.name] = value

    invariants: dict[str, Any] = {}
    sparse_invariants: dict[str, dict[str, Any]] = {}
    varying: dict[str, dict[str, Any]] = {}
    for field, values_by_profile in sorted(field_values.items()):
        values = list(values_by_profile.values())
        first = json.dumps(values[0], sort_keys=True)
        all_same = all(json.dumps(v, sort_keys=True) == first for v in values[1:])
        if len(values_by_profile) == len(entries):
            if all_same:
                invariants[field] = values[0]
            else:
                varying[field] = values_by_profile
            continue
        if all_same:
            sparse_invariants[field] = {
                "value": values[0],
                "coverage": len(values_by_profile),
                "profiles": sorted(values_by_profile.keys()),
            }
        else:
            varying[field] = values_by_profile
    return invariants, sparse_invariants, varying


def heuristic_candidates(varying: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for field, values in sorted(varying.items()):
        distinct = []
        for value in values.values():
            rendered = json.dumps(value, sort_keys=True)
            if rendered not in distinct:
                distinct.append(rendered)
        if len(distinct) <= 6:
            candidates.append(
                {
                    "field": field,
                    "distinct_values": [json.loads(v) for v in distinct],
                    "profiles": values,
                }
            )
    return candidates


def markdown_summary(
    entries: list[ProfileEntry],
    invariants: dict[str, Any],
    sparse_invariants: dict[str, dict[str, Any]],
    varying: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# AX Kernel Profile Policy Report")
    lines.append("")
    lines.append(f"- Profiles analyzed: {len(entries)}")
    lines.append(f"- Invariant fields: {len(invariants)}")
    lines.append(f"- Sparse invariant fields: {len(sparse_invariants)}")
    lines.append(f"- Varying fields: {len(varying)}")
    lines.append("")
    lines.append("## Hardcode Candidates")
    lines.append("")
    for field, value in sorted(invariants.items()):
        lines.append(f"- `{field}` = `{json.dumps(value, sort_keys=True)}`")
    lines.append("")
    if sparse_invariants:
        lines.append("## Sparse Hardcode Candidates")
        lines.append("")
        for field, metadata in sorted(sparse_invariants.items()):
            lines.append(
                f"- `{field}` = `{json.dumps(metadata['value'], sort_keys=True)}` "
                f"(coverage={metadata['coverage']}/{len(entries)})"
            )
        lines.append("")
    lines.append("## Heuristic Candidates")
    lines.append("")
    for candidate in candidates:
        lines.append(f"- `{candidate['field']}`")
        rendered = ", ".join(json.dumps(v, sort_keys=True) for v in candidate["distinct_values"])
        lines.append(f"  values: {rendered}")
    lines.append("")
    lines.append("## Profiles")
    lines.append("")
    for entry in entries:
        lines.append(f"- `{entry.name}` family=`{entry.family}` size=`{entry.size}`")
    return "\n".join(lines)


def write_pruned_profiles(
    entries: list[ProfileEntry],
    perf_dir: str,
    pruned_dir: str,
    invariants: dict[str, Any],
) -> dict[str, Any]:
    os.makedirs(pruned_dir, exist_ok=True)
    redundant_fields = set(invariants.keys()) | HEURISTIC_COVERED_FIELDS
    pruned_summary: dict[str, Any] = {"output_dir": pruned_dir, "files": {}}

    for entry in entries:
        source_path = os.path.join(perf_dir, entry.name)
        data = json.load(open(source_path))
        removed: list[str] = []
        for field in sorted(redundant_fields):
            if field in META_FIELDS:
                continue
            before = json.dumps(data, sort_keys=True)
            delete_flat_key(data, field)
            after = json.dumps(data, sort_keys=True)
            if before != after:
                removed.append(field)
        out_path = os.path.join(pruned_dir, entry.name)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        pruned_summary["files"][entry.name] = {
            "removed_fields": removed,
            "removed_count": len(removed),
            "output_path": out_path,
        }

    return pruned_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-dir", default="perfs", help="Directory containing profile JSON files")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write machine-readable report JSON",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional path to write Markdown summary",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print Markdown summary to stdout",
    )
    parser.add_argument(
        "--write-pruned-dir",
        default=None,
        help="Optional directory to write pruned profile suggestions",
    )
    args = parser.parse_args()

    entries, flattened = load_profiles(args.perf_dir)
    invariants, sparse_invariants, varying = classify_fields(entries, flattened)
    candidates = heuristic_candidates(varying)

    report = {
        "profiles": [
            {
                "name": entry.name,
                "path": entry.path,
                "family": entry.family,
                "size": entry.size,
            }
            for entry in entries
        ],
        "invariants": invariants,
        "sparse_invariants": sparse_invariants,
        "varying": varying,
        "heuristic_candidates": candidates,
        "heuristic_covered_fields": sorted(HEURISTIC_COVERED_FIELDS),
    }

    if args.write_pruned_dir:
        report["pruned_profiles"] = write_pruned_profiles(
            entries,
            args.perf_dir,
            args.write_pruned_dir,
            invariants,
        )

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    summary = markdown_summary(
        entries,
        invariants,
        sparse_invariants,
        varying,
        candidates,
    )
    if args.markdown_output:
        with open(args.markdown_output, "w") as f:
            f.write(summary)
            f.write("\n")

    if args.print_markdown or (not args.json_output and not args.markdown_output):
        print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
