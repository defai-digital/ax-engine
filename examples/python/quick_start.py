#!/usr/bin/env python3
"""Quick-start example: download a model and run a first generation.

This is the recommended starting point for new users. It demonstrates the
full setup flow from zero to first token without requiring any Rust toolchain
knowledge.

Steps:
  1. Download an mlx-community model via ax_engine.download_model()
  2. Generate the ax-engine manifest (one-time Rust step — see printed hint)
  3. Create a Session and run generate()

Usage:
  python examples/python/quick_start.py
  python examples/python/quick_start.py --repo-id mlx-community/Qwen3-4B-4bit
  python examples/python/quick_start.py --model-dir /path/to/already/downloaded/model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO_ID = "mlx-community/Qwen3-4B-4bit"
MANIFEST_FILE = "model-manifest.json"

SAMPLE_TOKENS = [2, 5726, 374, 264, 1296, 315]  # "This is a test of"


def main() -> int:
    parser = argparse.ArgumentParser(description="ax-engine quick-start example")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"mlx-community repo id to download (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Use an already-downloaded model directory instead of downloading",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32, help="Max output tokens (default: 32)"
    )
    args = parser.parse_args()

    import ax_engine

    # Step 1: get the model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
        print(f"Using existing model directory: {model_dir}")
    else:
        print(f"Downloading {args.repo_id} …")
        model_dir = ax_engine.download_model(args.repo_id)

    # Step 2: check manifest is present
    if not (model_dir / MANIFEST_FILE).exists():
        print(
            f"\nManifest not found. Generate it before running this example:\n"
            f"  cargo run -p ax-engine-core --bin generate-manifest -- {model_dir}\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        return 1

    # Step 3: create a session and generate
    print(f"\nLoading model from {model_dir} …")
    session = ax_engine.Session(
        model_id="quick_start",
        mlx=True,
        mlx_model_artifacts_dir=str(model_dir),
    )

    print(f"Generating {args.max_tokens} tokens …\n")
    result = session.generate(
        input_tokens=SAMPLE_TOKENS,
        max_output_tokens=args.max_tokens,
        temperature=0.0,
    )

    print("Output tokens:", result.output_tokens)
    if result.output_text:
        print("Output text: ", result.output_text)

    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
