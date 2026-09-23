#!/usr/bin/env python3
"""Assert the Flash Next MTP direct-fallback keys cannot drift between crates.

Engine side (`crates/ax-engine-mlx/src/runner/mod.rs`), which decides the
route keys the runner emits per decode step:

  * the `FlashNextMtpFallbackReason` enum variants,
  * `FLASH_NEXT_MTP_FALLBACK_REASON_COUNT`,
  * the `fn route_key` match arms.

Server side (`crates/ax-engine-server/src/flash_next_fallback_keys.rs`), which
accumulates those keys and publishes them in `/metrics`:

  * `ROUTE_KEYS: [&str; N]`,
  * `ROUTE_KEY_PREFIX`,
  * `metric_name(suffix)` -> `ax_engine_flash_next_mtp_direct_fallback_<s>_total`.

Fails when either side gains, drops, duplicates or renames a key, when the
declared counts disagree, or when a published name is not the shared prefix
rebuilt as an engine route key. Read-only: parses crate sources; no build, no
network, no weights.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_SRC = REPO_ROOT / "crates/ax-engine-mlx/src/runner/mod.rs"
SERVER_SRC = REPO_ROOT / "crates/ax-engine-server/src/flash_next_fallback_keys.rs"

ROUTE_KEY_PREFIX = "ax_mlx_flash_next_mtp_direct_fallback_"
METRIC_NAME_PREFIX = "ax_engine_flash_next_mtp_direct_fallback_"
METRIC_NAME_SUFFIX = "_total"

ENGINE_KEY_RE = re.compile(r'"(ax_mlx_flash_next_mtp_direct_fallback_[a-z0-9_]+)"')


def brace_body(text: str, start_index: int) -> str:
    """Return the text inside the first brace-delimited block after start_index."""
    open_index = text.index("{", start_index)
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index]
    raise ValueError("unbalanced braces")


def engine_reason_variants(text: str) -> list[str]:
    header = re.search(r"enum FlashNextMtpFallbackReason\b", text)
    if header is None:
        return []
    body = brace_body(text, header.end())
    return re.findall(r"^\s*([A-Z][A-Za-z0-9_]*),", body, re.MULTILINE)


def engine_route_keys(text: str) -> list[str]:
    header = re.search(r"fn route_key\(self\)", text)
    if header is None:
        return []
    return ENGINE_KEY_RE.findall(brace_body(text, header.end()))


def engine_reason_count_const(text: str) -> int | None:
    match = re.search(r"FLASH_NEXT_MTP_FALLBACK_REASON_COUNT:\s*usize\s*=\s*(\d+)", text)
    return int(match.group(1)) if match else None


def server_route_keys(text: str) -> tuple[list[str], int | None]:
    declared = re.search(r"ROUTE_KEYS:\s*\[&str;\s*(\d+)\]\s*=", text)
    keys = ENGINE_KEY_RE.findall(text)
    return keys, (int(declared.group(1)) if declared else None)


def server_route_key_prefix(text: str) -> str | None:
    match = re.search(r'ROUTE_KEY_PREFIX:\s*&str\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else None


def server_metric_name_template(text: str) -> str | None:
    match = re.search(
        r"fn metric_name\(suffix:\s*&str\)\s*->\s*String\s*\{\s*"
        r'format!\("([^"]+)"\)',
        text,
    )
    return match.group(1) if match else None


def main() -> int:
    problems: list[str] = []

    for path in (ENGINE_SRC, SERVER_SRC):
        if not path.is_file():
            print(f"missing source: {path.relative_to(REPO_ROOT)}")
            return 1

    engine_text = ENGINE_SRC.read_text(encoding="utf-8")
    server_text = SERVER_SRC.read_text(encoding="utf-8")

    engine_keys = engine_route_keys(engine_text)
    variants = engine_reason_variants(engine_text)
    count_const = engine_reason_count_const(engine_text)
    server_keys, declared_len = server_route_keys(server_text)
    server_prefix = server_route_key_prefix(server_text)
    metric_template = server_metric_name_template(server_text)

    if not engine_keys:
        problems.append("no route keys found in the engine route_key match")
    if not server_keys:
        problems.append("no route keys found in the server ROUTE_KEYS array")

    for label, keys in (("engine", engine_keys), ("server", server_keys)):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            problems.append(f"{label} has duplicate route keys: {', '.join(duplicates)}")
        for key in keys:
            if not key.startswith(ROUTE_KEY_PREFIX):
                problems.append(f"{label} key does not use the shared prefix: {key}")

    if len(engine_keys) != len(server_keys):
        problems.append(
            f"count mismatch: engine has {len(engine_keys)}, server has {len(server_keys)}"
        )
    if declared_len is not None and declared_len != len(server_keys):
        problems.append(
            f"server ROUTE_KEYS declares {declared_len} entries but lists {len(server_keys)}"
        )
    if count_const is not None and count_const != len(engine_keys):
        problems.append(
            f"FLASH_NEXT_MTP_FALLBACK_REASON_COUNT is {count_const} "
            f"but the engine emits {len(engine_keys)} keys"
        )
    if variants and len(variants) != len(engine_keys):
        problems.append(
            f"enum has {len(variants)} variants but the engine emits {len(engine_keys)} route keys"
        )

    if server_prefix != ROUTE_KEY_PREFIX:
        problems.append(f"ROUTE_KEY_PREFIX is {server_prefix!r}, expected {ROUTE_KEY_PREFIX!r}")

    expected_template = f"{METRIC_NAME_PREFIX}{{suffix}}{METRIC_NAME_SUFFIX}"
    if metric_template != expected_template:
        problems.append(
            f"metric_name template is {metric_template!r}, expected {expected_template!r}"
        )

    for engine_key, server_key in zip(engine_keys, server_keys):
        if engine_key != server_key:
            problems.append(f"route key drift: engine {engine_key!r} vs server {server_key!r}")
    only_engine = [k for k in engine_keys if k not in server_keys]
    only_server = [k for k in server_keys if k not in engine_keys]
    for key in only_engine:
        problems.append(f"engine route key missing from the server: {key}")
    for key in only_server:
        problems.append(f"server route key missing from the engine: {key}")

    if engine_keys and server_keys and not problems:
        suffixes = [k[len(ROUTE_KEY_PREFIX) :] for k in engine_keys]
        names = [f"{METRIC_NAME_PREFIX}{s}{METRIC_NAME_SUFFIX}" for s in suffixes]
        print(
            f"OK: {len(engine_keys)} fallback reasons agree across crates; "
            f"/metrics names {names[0]} .. {names[-1]}"
        )
        return 0

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1

    print("FAIL: route keys could not be compared")
    return 1


if __name__ == "__main__":
    sys.exit(main())
