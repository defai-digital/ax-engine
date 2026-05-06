#!/usr/bin/env python3
"""Check public TurboQuant docs keep the experimental support boundary explicit."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SNIPPETS = {
    "docs/SERVER.md": [
        "Experimental MLX KV compression is opt-in and off by default.",
        "keeps generation on the existing full-precision MLX KV path",
        "turboquant-fused-experimental",
        "runner_not_integrated",
        "successful attempts and zero fallbacks",
        "shadow and CPU oracle rows are diagnostic only",
        "does not imply production TurboQuant support",
        "shadow sync calls",
        "wall time",
    ],
    "docs/CLI.md": [
        "This mode is optional, disabled by default",
        "full-precision shadow path",
        "turboquant-fused-experimental",
        "runner_not_integrated",
        "successful attempts and zero fallbacks",
        "shadow and CPU oracle rows are diagnostic only",
        "shadow-storage sync calls and wall time",
        "not a production support claim",
    ],
    "docs/BENCHMARKS.md": [
        "The default remains disabled.",
        "turboquant-fused-experimental",
        "runner_not_integrated",
        "full-precision MLX decode path",
        "shadow-storage sync calls and wall time",
        "fused_compressed_decode path code",
        "cpu_oracle_compressed_decode rows",
        "check_turboquant_promotion_readiness.py",
        "head_dim=128",
        "head_dim=256",
        "head_dim=512",
        "promoted TurboQuant support still requires a long-context quality artifact",
    ],
    "scripts/README.md": [
        "--experimental-mlx-kv-compression turboquant-shadow",
        "turboquant-fused-experimental",
        "fused_compressed_decode path code",
        "zero fallbacks",
        "fallback reason labels",
        "TurboQuant KV compression route counters",
        "shadow-storage sync calls and wall time",
    ],
}


def normalize(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    missing: list[str] = []
    for relative_path, snippets in REQUIRED_SNIPPETS.items():
        path = ROOT / relative_path
        text = normalize(path.read_text())
        for snippet in snippets:
            if normalize(snippet) not in text:
                missing.append(f"{relative_path}: missing {snippet!r}")

    if missing:
        print("TurboQuant public docs contract failed:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("ok: TurboQuant public docs contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
