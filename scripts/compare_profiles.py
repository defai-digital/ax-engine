#!/usr/bin/env python3
"""
Compare two kernel profiles side by side.

Shows which parameters differ and by how much, useful for debugging
and for understanding what changed between profile versions.
"""

import json
import sys
from pathlib import Path


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict with dot-separated keys."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_dict(v, key))
        else:
            result[key] = v
    return result


def compare_profiles(path_a: str, path_b: str) -> dict:
    """Compare two profile JSON files."""
    with open(path_a) as f:
        profile_a = json.load(f)
    with open(path_b) as f:
        profile_b = json.load(f)
    
    flat_a = flatten_dict(profile_a)
    flat_b = flatten_dict(profile_b)
    
    all_keys = set(flat_a.keys()) | set(flat_b.keys())
    
    same = {}
    changed = {}
    only_a = {}
    only_b = {}
    
    for key in sorted(all_keys):
        if key.startswith(("model", "source", "generated", "gguf")):
            continue
        
        val_a = flat_a.get(key)
        val_b = flat_b.get(key)
        
        if val_a is None:
            only_b[key] = val_b
        elif val_b is None:
            only_a[key] = val_a
        elif val_a == val_b:
            same[key] = val_a
        else:
            changed[key] = (val_a, val_b)
    
    return {
        "same": same,
        "changed": changed,
        "only_a": only_a,
        "only_b": only_b,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare two kernel profile JSON files"
    )
    parser.add_argument("profile_a", help="First profile JSON file")
    parser.add_argument("profile_b", help="Second profile JSON file")
    parser.add_argument(
        "--show-same",
        action="store_true",
        help="Also show parameters that are the same",
    )
    
    args = parser.parse_args()
    
    result = compare_profiles(args.profile_a, args.profile_b)
    
    print(f"Comparing: {args.profile_a} vs {args.profile_b}")
    print()
    
    if result["changed"]:
        print("=== CHANGED ===")
        for key, (val_a, val_b) in result["changed"].items():
            print(f"  {key}: {val_a} -> {val_b}")
        print()
    
    if result["only_a"]:
        print("=== ONLY IN A ===")
        for key, val in result["only_a"].items():
            print(f"  {key}: {val}")
        print()
    
    if result["only_b"]:
        print("=== ONLY IN B ===")
        for key, val in result["only_b"].items():
            print(f"  {key}: {val}")
        print()
    
    if args.show_same and result["same"]:
        print("=== SAME ===")
        for key, val in result["same"].items():
            print(f"  {key}: {val}")
        print()
    
    print(f"Summary: {len(result['same'])} same, {len(result['changed'])} changed, "
          f"{len(result['only_a'])} only in A, {len(result['only_b'])} only in B")


if __name__ == "__main__":
    main()
