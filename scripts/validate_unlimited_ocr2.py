#!/usr/bin/env python3
"""Bounded exact-pack OCR controls; run only on an authorized Apple Silicon host."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import re
import time
from pathlib import Path

PACK_SHA256 = {
    "model-00001-of-00002.safetensors": "34d8a55b84fc3df9edac6094c12b5f9f1a126177493b8e45bd6345f8429f7bfb",
    "model-00002-of-00002.safetensors": "e2a8ede94d2280231f93702522ce2af0310e8d63b00a75012ff3d2fde5594857",
    "config.json": "fcf27e3d579846e44d6425eecbc6f2b63e15a1b18bc652b6639341ec9e28404d",
    "tokenizer.json": "c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589",
    "model.safetensors.index.json": "45d7dba90ce15a2a5c07b03513ffbd94f3fb72bebac010b6093af1a5825bc994",
}


def ocr_text_matches(text: str, expected: list[str], pages: int) -> tuple[bool, str]:
    body = " ".join(re.findall(r"<FORK>(.*?)(?=<BLOCK>|<PAGE>|$)", text, re.S))
    normalized = " ".join(body.upper().split())
    expected_body = " ".join(" ".join(expected).upper().split())
    return normalized == expected_body and text.count("<PAGE>") == pages, body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args()

    import ax_engine
    from PIL import Image, ImageDraw, ImageFont
    from tokenizers import Tokenizer

    model = args.model_dir
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, expected in PACK_SHA256.items():
        with (model / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if hashes[name] != expected:
            raise SystemExit(f"Pinned pack hash mismatch: {name}")
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 42)
    pages = []
    for index, text in enumerate(["AX ENGINE OCR TWO\nPAGE ONE 12345", "NATIVE METAL OCR\nPAGE TWO 67890"]):
        image = Image.new("RGB", (900, 600), "white")
        ImageDraw.Draw(image).multiline_text((60, 100), text, fill="black", font=font, spacing=24)
        path = args.output / f"page-{index + 1}.png"
        image.save(path)
        pages.append(path)

    cases = [
        ("single-base", [pages[0]], False, ["AX ENGINE OCR TWO", "PAGE ONE 12345"]),
        ("single-tiles", [pages[0]], True, ["AX ENGINE OCR TWO", "PAGE ONE 12345"]),
        ("multi-base", pages, False, ["AX ENGINE OCR TWO", "PAGE ONE 12345", "NATIVE METAL OCR", "PAGE TWO 67890"]),
        ("reset-single", [pages[0]], False, ["AX ENGINE OCR TWO", "PAGE ONE 12345"]),
    ]
    records = []
    started = time.monotonic()
    with ax_engine.Session(mlx=True, model_id="unlimited_ocr2", mlx_model_artifacts_dir=str(model)) as session:
        loaded = time.monotonic()
        for name, images, cropping, expected in cases:
            prompt = "<PAGE><image>" * len(images) + "Multi page parsing."
            ids = [0, *tokenizer.encode(prompt, add_special_tokens=False).ids]
            request = ax_engine.prepare_unlimited_ocr_image_request(model, ids, images, cropping=cropping)
            before = time.monotonic()
            result = session.generate(
                request.input_tokens,
                multimodal_inputs=request.multimodal_inputs,
                max_output_tokens=512 if len(images) > 1 else 256,
                temperature=0.0,
                no_repeat_ngram_size=35,
                ngram_window=1024 if len(images) > 1 else 128,
            )
            text = tokenizer.decode(result.output_tokens, skip_special_tokens=False)
            passed, body = ocr_text_matches(text, expected, len(images))
            record = {"name": name, "prompt_tokens": len(request.input_tokens),
                      "output_tokens": list(result.output_tokens), "text": text,
                      "elapsed_seconds": time.monotonic() - before,
                      "expected": expected, "body": body,
                      "passed": passed}
            records.append(record)
            print(json.dumps(record), flush=True)
            (args.output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")

    reset_equal = records[0]["output_tokens"] == records[-1]["output_tokens"]
    summary = {"model_id": "AutomatosX/AX-Unlimited-OCR-2-AXQ-MXFP8",
               "revision": "462c40bb4f2c7eeb3e1230e912940efcbdefc8a3",
               "host": args.host_label, "os": platform.mac_ver()[0],
               "mlx": importlib.metadata.version("mlx"),
               "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
               "pack_sha256": hashes,
               "load_seconds": loaded - started,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "reset_equal": reset_equal, "cases": records,
               "passed": reset_equal and all(row["passed"] for row in records),
               "limitations": "Bounded functional controls; retained KV grows with document length. No quality, throughput or constant-memory qualification."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise SystemExit("OCR-2 functional controls failed; inspect retained outputs")


if __name__ == "__main__":
    main()
