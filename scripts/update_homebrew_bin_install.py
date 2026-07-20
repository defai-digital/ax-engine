#!/usr/bin/env python3
"""Normalize a Homebrew formula's bin.install payload list."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BIN_INSTALL_RE = re.compile(r"^(?P<indent>[ \t]*)bin\.install\b")
CONTINUATION_RE = re.compile(r"^[ \t]+[\"']")


def rewrite_bin_install(text: str, payloads: list[str]) -> str:
    """Replace all bin.install statements with one canonical payload list."""
    if not payloads:
        raise ValueError("at least one Homebrew payload is required")

    lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    output: list[str] = []
    found = False
    index = 0

    while index < len(lines):
        line = lines[index]
        match = BIN_INSTALL_RE.match(line)
        if match is None:
            output.append(line)
            index += 1
            continue

        if not found:
            payload_list = ", ".join(f'"{payload}"' for payload in payloads)
            output.append(f'{match.group("indent")}bin.install {payload_list}{newline}')
            found = True

        # Ruby continues an argument list when the preceding line ends in a
        # comma. Consume the entire statement so multiline formula formatting
        # cannot leave orphaned arguments behind.
        while line.rstrip().endswith(","):
            index += 1
            if index >= len(lines):
                raise ValueError("bin.install statement ends with an unterminated comma")
            line = lines[index]
            if CONTINUATION_RE.match(line) is None:
                raise ValueError("unsupported bin.install continuation line")
        index += 1

    if not found:
        raise ValueError("formula has no bin.install statements")

    return re.sub(r"\n{3,}", "\n\n", "".join(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formula", type=Path)
    parser.add_argument("payload", nargs="+")
    args = parser.parse_args()

    text = args.formula.read_text(encoding="utf-8")
    try:
        updated = rewrite_bin_install(text, args.payload)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    args.formula.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
