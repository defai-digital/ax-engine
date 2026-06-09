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
    # Reference single-axis fallback + max_side_length clamp (processing_gemma4.py
    # lines 115-125): keep extreme aspect ratios downscaling the whole image
    # rather than collapsing one axis to zero.
    max_side_length = (max_patches // pooling**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
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


AUDIO_SAMPLES_PER_TOKEN = 640
AUDIO_SAMPLE_RATE = 16000


def _write_int16_wav(path, samples_i16, sample_rate):
    import wave

    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples_i16.astype("<i2").tobytes())


def dump_audio(name, num_samples):
    # Deterministic int16 waveform; int16/32768 round-trips losslessly through
    # the Rust hound decoder, so the reshape can be compared bit-for-bit.
    samples_i16 = np.array(
        [((i * 37) % 65536) - 32768 for i in range(num_samples)], dtype=np.int16
    )
    _write_int16_wav(OUT / f"audio_{name}.wav", samples_i16, AUDIO_SAMPLE_RATE)

    # Reference Gemma4UnifiedAudioFeatureExtractor: pad to a multiple of
    # audio_samples_per_token, reshape into fixed frames.
    waveform = samples_i16.astype(np.float32) / 32768.0
    pad = (-num_samples) % AUDIO_SAMPLES_PER_TOKEN
    padded = np.pad(waveform, (0, pad), mode="constant", constant_values=0.0)
    features = padded.reshape(-1, AUDIO_SAMPLES_PER_TOKEN)
    golden = {
        "audio_samples_per_token": AUDIO_SAMPLES_PER_TOKEN,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "sample_count": int(num_samples),
        "frame_count": int(features.shape[0]),
        "feature_count": AUDIO_SAMPLES_PER_TOKEN,
        "input_features": features.reshape(-1).tolist(),
    }
    (OUT / f"golden_audio_{name}.json").write_text(json.dumps(golden))
    print(f"audio {name}: {num_samples} samples -> {features.shape[0]} frames")


def _video_frame(width, height, seed):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            rgb[y, x] = [
                (x * 13 + seed * 7) % 256,
                (y * 17 + seed * 11) % 256,
                (x * y + seed * 23) % 256,
            ]
    return rgb


def dump_video(name, sizes):
    # No-resize frames (scale == 1) so per-frame patchify is exact; the golden
    # validates frame concatenation order and per-frame position reset on top of
    # the already-golden patchify. Frames are made distinct (different seed) so a
    # "duplicate frame 0" bug would be caught.
    pixel_values = []
    positions = []
    soft_per_frame = None
    for index, (width, height) in enumerate(sizes):
        rgb = _video_frame(width, height, index)
        Image.fromarray(rgb).save(OUT / f"video_{name}_frame{index}.png")
        patches, frame_positions = preprocess(rgb)
        if soft_per_frame is None:
            soft_per_frame = int(patches.shape[0])
        pixel_values.append(patches.reshape(-1))
        positions.append(frame_positions)

    golden = {
        "config": {
            "patch_size": PATCH_SIZE,
            "pooling_kernel_size": POOLING,
            "model_patch_size": MODEL_PATCH_SIZE,
            "max_soft_tokens": MAX_SOFT_TOKENS,
        },
        "frame_count": len(sizes),
        "soft_tokens_per_frame": soft_per_frame,
        "pixel_values": np.concatenate(pixel_values).tolist(),
        "positions": np.concatenate(positions).tolist(),
    }
    (OUT / f"golden_video_{name}.json").write_text(json.dumps(golden))
    print(f"video {name}: {len(sizes)} frames -> {soft_per_frame} soft tokens/frame")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 16x16: scale == 1 so resize is a no-op -> Rust must match exactly.
    dump("noresize", 16, 16)
    # 12x12: upscaled to 16x16 via bicubic -> Rust compares within tolerance.
    dump("resize", 12, 12)
    # Audio: 1600 samples -> 3 frames of 640 (last frame zero-padded).
    dump_audio("noresize", 1600)
    # Video: two no-resize 16x16 frames -> exact concatenated patches.
    dump_video("noresize", [(16, 16), (16, 16)])


if __name__ == "__main__":
    main()
