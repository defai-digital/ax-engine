#!/usr/bin/env python3
"""Generate golden-vector fixtures for Gemma 4 unified image preprocessing.

This mirrors the reference `processing_gemma4_unified.py` /
`processing_gemma4.py` exactly (aspect-ratio-preserving resize via PIL BICUBIC +
`_convert_image_to_model_patches` + rescale/normalize) so the Rust
`multimodal::preprocess_image` path can be cross-checked against it.

Outputs, under crates/ax-engine-server/src/tests/fixtures/gemma4_golden/:
  - image_noresize.png + golden_noresize.json  (resize is a no-op -> exact match)
  - image_resize.png   + golden_resize.json    (upscaled -> compare w/ tolerance)

Run: python3 scripts/gen_gemma4_golden.py
"""
import json
import math
import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path("crates/ax-engine-server/src/tests/fixtures/gemma4_golden")

# Small synthetic config so fixtures stay tiny. model_patch_size = patch_size *
# pooling_kernel_size = 8; patch_dim = 8*8*3 = 192.
PATCH_SIZE = 4
POOLING = 2
MODEL_PATCH_SIZE = PATCH_SIZE * POOLING
MAX_SOFT_TOKENS = 4
RESCALE = 1.0 / 255.0
DO_NORMALIZE = False  # Gemma 4 image processor default
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def aspect_ratio_preserving_resize(image_chw, patch_size, max_patches, pooling):
    """Verbatim from processing_gemma4.py (channel-first input)."""
    _, height, width = image_chw.shape
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / (height * width))
    side_mult = pooling * patch_size
    target_height = int(math.floor(factor * height / side_mult)) * side_mult
    target_width = int(math.floor(factor * width / side_mult)) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("0x0 resize")
    if target_height == height and target_width == width:
        return image_chw
    img = np.transpose(image_chw, (1, 2, 0))  # HWC
    pil = Image.fromarray(img.astype(np.uint8))
    pil = pil.resize((target_width, target_height), resample=Image.BICUBIC)
    return np.transpose(np.array(pil), (2, 0, 1))  # CHW


def convert_image_to_model_patches(image_chw, model_patch_size):
    """Verbatim from processing_gemma4_unified.py."""
    channels, height, width = image_chw.shape
    ph = height // model_patch_size
    pw = width // model_patch_size
    patches = image_chw.reshape(channels, ph, model_patch_size, pw, model_patch_size)
    patches = patches.transpose(1, 3, 2, 4, 0)
    patches = patches.reshape(ph * pw, model_patch_size * model_patch_size * channels)
    grid = np.meshgrid(
        np.arange(pw, dtype=np.int64), np.arange(ph, dtype=np.int64), indexing="xy"
    )
    positions = np.stack(grid, axis=-1).reshape(-1, 2)
    return patches.astype(np.float32), positions


def preprocess(rgb_hwc):
    max_patches = MAX_SOFT_TOKENS * POOLING**2
    image = np.transpose(rgb_hwc, (2, 0, 1)).astype(np.uint8)  # CHW uint8
    image = aspect_ratio_preserving_resize(image, PATCH_SIZE, max_patches, POOLING)
    image = image.astype(np.float32) * RESCALE
    if DO_NORMALIZE:
        mean = np.array(MEAN, dtype=np.float32)[:, None, None]
        std = np.array(STD, dtype=np.float32)[:, None, None]
        image = (image - mean) / std
    patches, positions = convert_image_to_model_patches(image, MODEL_PATCH_SIZE)
    keep = min(patches.shape[0], MAX_SOFT_TOKENS)
    return patches[:keep], positions[:keep]


def gradient_image(width, height):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            rgb[y, x] = [(x * 13) % 256, (y * 17) % 256, (x * y) % 256]
    return rgb


def dump(name, width, height):
    rgb = gradient_image(width, height)
    Image.fromarray(rgb).save(OUT / f"image_{name}.png")
    patches, positions = preprocess(rgb)
    golden = {
        "config": {
            "patch_size": PATCH_SIZE,
            "pooling_kernel_size": POOLING,
            "model_patch_size": MODEL_PATCH_SIZE,
            "max_soft_tokens": MAX_SOFT_TOKENS,
        },
        "width": width,
        "height": height,
        "soft_tokens": int(patches.shape[0]),
        "pixel_values": patches.reshape(-1).tolist(),
        "positions": positions.tolist(),
    }
    (OUT / f"golden_{name}.json").write_text(json.dumps(golden))
    print(f"{name}: {width}x{height} -> {patches.shape[0]} patches, dim {patches.shape[1]}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 16x16: scale == 1 so resize is a no-op -> Rust must match exactly.
    dump("noresize", 16, 16)
    # 12x12: upscaled to 16x16 via bicubic -> Rust compares within tolerance.
    dump("resize", 12, 12)


if __name__ == "__main__":
    main()
