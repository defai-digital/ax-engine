#!/usr/bin/env python3
"""Recompute a bounded runner state diagnostic; never grants model qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ids(value):
    require(isinstance(value, list) and all(type(t) is int and 0 <= t < 2**32
                                          for t in value), "Invalid token IDs")
    return value


def read_blob(root, record, expected_name):
    require(record["file"] == expected_name, "Unexpected artifact name")
    path = root / expected_name
    require(not path.is_symlink() and path.is_file(), "Missing or linked artifact")
    raw = path.read_bytes()
    require(type(record["bytes"]) is int and len(raw) == record["bytes"]
            and hashlib.sha256(raw).hexdigest() == record["sha256"], "Artifact identity changed")
    return raw


def state_layout(raw, position, tokens, initial_histories):
    """Parse the owned AXKB v4 Flash Next format, including every layer and PLE history."""
    offset = 0

    def take(count):
        nonlocal offset
        require(0 <= count <= len(raw) - offset, "Truncated state")
        value = raw[offset:offset + count]
        offset += count
        return value

    def unpack(fmt):
        return struct.unpack(fmt, take(struct.calcsize(fmt)))

    def tag():
        value = take(1)[0]
        require(value in (0, 1), "Invalid state presence tag")
        return value

    def tensor():
        dtype, ndim = unpack("<BB")
        require(take(6) == bytes(6), "Invalid tensor padding")
        shape = unpack("<4i")
        size, = unpack("<Q")
        require(dtype in (9, 10, 12) and 1 <= ndim <= 4, "Invalid state tensor type")
        require(all(n > 0 for n in shape[:ndim]) and not any(shape[ndim:]), "Invalid tensor shape")
        require(size == math.prod(shape[:ndim]) * (4 if dtype == 10 else 2), "Tensor size mismatch")
        take(size)
        return (dtype, shape[:ndim])

    require(take(4) == b"AXKB", "Not AXKB")
    version, seq, growth, rope, layers, delta = unpack("<IQQQIi")
    require(version == 4 and seq == position and growth == rope == delta == 0
            and 0 < layers <= 1024, "State header mismatch")
    require(len(initial_histories) == layers, "Missing layer history metadata")
    layout, owner_id = [], None
    for layer in range(layers):
        require(take(8) == bytes([4]) + bytes(7), "Not Flash Next layer")
        owner, = unpack("<Q")
        if owner_id is None:
            owner_id = owner
        require(owner == owner_id, "Mixed state owners")
        kind = tag()
        require(tag() == 1, "Missing attention state")
        tensors = [tensor(), tensor()]
        if kind == 1:
            require(tag() == 1, "Missing QSA index state")
            tensors.append(tensor())
        ple, history = None, None
        if tag():
            require(tag() == 1, "Missing PLE convolution")
            ple = tensor()
            count, = unpack("<I")
            require(0 < count <= 1024, "Invalid PLE history length")
            history = list(unpack(f"<{count}I"))
            initial = ids(initial_histories[layer])
            require(len(initial) == count and len(set(initial)) == 1, "Invalid initial PLE history")
            expected = initial.copy()
            eos = initial[0]
            for token in tokens:
                expected = initial.copy() if token == eos else [token] + expected[:-1]
            require(history == expected, "PLE history differs from consumed tokens")
        else:
            require(initial_histories[layer] is None, "Missing PLE state")
        layout.append((layer, kind, tensors, ple, None if history is None else len(history)))
    require(offset == len(raw), "Trailing state payload")
    return layout


def maximum(routes, name):
    require(isinstance(routes, list), "Invalid routes")
    values = []
    for row in routes:
        require(isinstance(row, list), "Invalid route row")
        for key, value in row:
            require(isinstance(key, str) and type(value) is int and value >= 0, "Invalid route gauge")
            if key == name:
                values.append(value)
    return max(values, default=0)


def verify(root):
    root = Path(root)
    data = json.loads((root / "result.json").read_text())
    require(data["schema"] == "ax.flash_next.runner_aligned_state.v1", "Unknown schema")
    require(data["qualification"] is False and data["release_ready"] is False
            and data["mtp_certification"] == {g: "not_assessed" for g in ("MTP-S", "MTP-P", "MTP-D")},
            "Unexpected promotion")
    require(data["max_output_tokens"] == 8 and data["block_size_tokens"] == 4
            and data["target_schedule"] == "canonical_singleton", "Changed diagnostic contract")
    prompt, direct, mtp = (ids(data[k]) for k in ("prompt_ids", "direct_ids", "mtp_ids"))
    require(prompt and len(direct) <= 8 and len(mtp) <= 8, "Invalid generation length")
    positions = []
    for mode, tokens in (("direct", direct), ("mtp", mtp)):
        row = data[mode + "_positions"]
        require(isinstance(row, list) and all(type(p) is int and len(prompt) <= p <= len(prompt) + len(tokens)
                                            for p in row), "Invalid consumed position")
        require(row == sorted(set(row)), "Repeated or unordered position")
        positions.append(set(row))
    eligible = sorted(p for p in positions[0] & positions[1]
                      if 0 < p - len(prompt) <= min(len(direct), len(mtp))
                      and direct[:p - len(prompt)] == mtp[:p - len(prompt)])
    pairs = data["aligned_states"]
    require([p["position"] for p in pairs] == eligible, "Incomplete aligned-state coverage")

    def pair(row, name, position, tokens):
        require(row["position"] == row["mtp_position"] == position, "Unaligned state")
        left = read_blob(root, row["direct"], name + ".direct.axkb")
        right = read_blob(root, row["mtp"], name + ".mtp.axkb")
        require(state_layout(left, position, tokens, data["ple_initial_histories"])
                == state_layout(right, position, tokens, data["ple_initial_histories"]),
                "Different state layouts")
        exact = left == right
        require(row["byte_exact"] is exact, "False state equality claim")
        return exact

    equalities = [pair(row, f"prefix-{p - len(prompt)}", p, prompt + direct[:p - len(prompt)])
                  for row, p in zip(pairs, eligible, strict=True)]
    candidates = [p for p in eligible if p - len(prompt) + 1 < min(len(direct), len(mtp))]
    continuation = data["continuation"]
    require((continuation is not None) == bool(candidates), "Missing or spurious continuation")
    continuation_passed = False
    if continuation is not None:
        p = candidates[-1]
        consumed = p - len(prompt)
        require(continuation["from_position"] == p and continuation["input_token"] == direct[consumed],
                "Wrong continuation prefix or input")
        exact = pair(continuation["state"], "continuation", p + 1, prompt + direct[:consumed + 1])
        selected = []
        for mode, tokens in (("direct", direct), ("mtp", mtp)):
            row = continuation[mode]
            raw = read_blob(root, row["logits_f32_le"], f"continuation.{mode}.logits.f32")
            shape = row["shape"]
            require(isinstance(shape, list) and len(shape) in (1, 2, 3)
                    and all(type(n) is int and n > 0 for n in shape)
                    and math.prod(shape[:-1]) == 1 and len(raw) == 4 * math.prod(shape),
                    "Invalid decision logit shape")
            values = [v for v, in struct.iter_unpack("<f", raw)]
            require(all(math.isfinite(v) for v in values), "Nonfinite decision logits")
            token = max(range(len(values)), key=values.__getitem__)
            require(type(row["token"]) is int and token == row["token"], "Incorrect decision argmax")
            require(continuation[mode + "_expected_token"] == tokens[consumed + 1], "Wrong expected token")
            selected.append(token)
        continuation_passed = exact and selected[0] == selected[1] == direct[consumed + 1] == mtp[consumed + 1]
    passed = (len(direct) == len(mtp) == 8 and direct == mtp and len(eligible) >= 2
              and all(equalities) and continuation_passed
              and maximum(data["mtp_routes"], "ax_mlx_flash_next_mtp_verified_steps") > 0
              and maximum(data["mtp_routes"], "ax_mlx_flash_next_mtp_step_errors") == 0)
    require(data["diagnostic_passed"] is passed, "False diagnostic verdict")
    return {"collection_valid": True, "diagnostic_passed": passed, "aligned_positions": eligible,
            "qualification": False, "release_ready": False, "mtp_certification": data["mtp_certification"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
