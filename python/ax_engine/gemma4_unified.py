from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Gemma4UnifiedImageRequest:
    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    soft_token_counts: list[int]


@dataclass(frozen=True)
class _Gemma4UnifiedImageConfig:
    image_token_id: int
    boi_token_id: int
    eoi_token_id: int
    do_convert_rgb: bool
    do_resize: bool
    do_rescale: bool
    rescale_factor: float
    do_normalize: bool
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    patch_size: int
    model_patch_size: int
    pooling_kernel_size: int
    max_soft_tokens: int


def prepare_gemma4_unified_image_request(
    model_dir: str | Path,
    input_tokens: list[int],
    images: list[Any],
) -> Gemma4UnifiedImageRequest:
    """Build a native Gemma4 unified image request from tokenized placeholders.

    The returned payload is intentionally the processed tensor contract consumed
    by AX's native MLX runtime. Raw image decoding and resizing happen here,
    outside the optimized text/MTP decode path.
    """

    config = _load_image_config(Path(model_dir))
    processed = [_process_image(image, config) for image in images]
    expanded_tokens, spans = _expand_image_placeholders(
        input_tokens,
        [item["soft_token_count"] for item in processed],
        config,
    )

    runtime_images = []
    for span, item in zip(spans, processed):
        runtime_images.append(
            {
                "span": span,
                "pixel_values": item["pixel_values"],
                "pixel_position_ids": item["pixel_position_ids"],
            }
        )

    return Gemma4UnifiedImageRequest(
        input_tokens=expanded_tokens,
        multimodal_inputs={
            "gemma4_unified": {
                "images": runtime_images,
                "audios": [],
                "videos": [],
            }
        },
        soft_token_counts=[item["soft_token_count"] for item in processed],
    )


def _load_image_config(model_dir: Path) -> _Gemma4UnifiedImageConfig:
    model_config_path = model_dir / "config.json"
    processor_config_path = model_dir / "processor_config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Gemma4 unified config not found: {model_config_path}")
    if not processor_config_path.is_file():
        raise FileNotFoundError(
            f"Gemma4 unified processor config not found: {processor_config_path}"
        )

    model_config = json.loads(model_config_path.read_text())
    processor_config = json.loads(processor_config_path.read_text())
    image_config = processor_config.get("image_processor") or {}
    vision_config = model_config.get("vision_config") or {}
    if image_config.get("image_processor_type") not in (None, "Gemma4UnifiedImageProcessor"):
        raise ValueError(
            "Gemma4 unified image preprocessing requires "
            "Gemma4UnifiedImageProcessor"
        )

    return _Gemma4UnifiedImageConfig(
        image_token_id=_required_int(model_config, "image_token_id"),
        boi_token_id=_required_int(model_config, "boi_token_id"),
        eoi_token_id=_required_int(model_config, "eoi_token_id"),
        do_convert_rgb=bool(image_config.get("do_convert_rgb", True)),
        do_resize=bool(image_config.get("do_resize", True)),
        do_rescale=bool(image_config.get("do_rescale", True)),
        rescale_factor=float(image_config.get("rescale_factor", 1 / 255)),
        do_normalize=bool(image_config.get("do_normalize", False)),
        image_mean=_triple(image_config.get("image_mean", [0.5, 0.5, 0.5])),
        image_std=_triple(image_config.get("image_std", [0.5, 0.5, 0.5])),
        patch_size=_optional_int(image_config, "patch_size")
        or _required_int(vision_config, "patch_size"),
        model_patch_size=_optional_int(image_config, "model_patch_size")
        or _required_int(vision_config, "model_patch_size"),
        pooling_kernel_size=_optional_int(image_config, "pooling_kernel_size")
        or _required_int(vision_config, "pooling_kernel_size"),
        max_soft_tokens=_optional_int(image_config, "max_soft_tokens")
        or _optional_int(vision_config, "num_soft_tokens")
        or _required_int(vision_config, "default_output_length"),
    )


def _process_image(image: Any, config: _Gemma4UnifiedImageConfig) -> dict[str, Any]:
    pil_image = _load_pil_image(image)
    if config.do_convert_rgb:
        pil_image = pil_image.convert("RGB")
    elif pil_image.mode != "RGB":
        raise ValueError("Gemma4 unified image preprocessing currently requires RGB images")

    if config.do_resize:
        target_width, target_height = _resized_dimensions(
            pil_image.width,
            pil_image.height,
            config.patch_size,
            config.pooling_kernel_size,
            config.max_soft_tokens,
        )
        if target_width != pil_image.width or target_height != pil_image.height:
            from PIL import Image

            pil_image = pil_image.resize(
                (target_width, target_height),
                resample=Image.Resampling.BICUBIC,
            )

    if pil_image.width % config.model_patch_size != 0:
        raise ValueError(
            "Gemma4 unified image width must be divisible by model_patch_size "
            f"after resize: width={pil_image.width}, model_patch_size={config.model_patch_size}"
        )
    if pil_image.height % config.model_patch_size != 0:
        raise ValueError(
            "Gemma4 unified image height must be divisible by model_patch_size "
            f"after resize: height={pil_image.height}, model_patch_size={config.model_patch_size}"
        )

    pixels = _rgb_pixels(pil_image, config)
    patch_values, position_ids = _patchify_rgb(pixels, pil_image.width, pil_image.height, config)
    soft_token_count = len(position_ids)
    if soft_token_count > config.max_soft_tokens:
        patch_values = patch_values[: config.max_soft_tokens]
        position_ids = position_ids[: config.max_soft_tokens]
        soft_token_count = config.max_soft_tokens

    while len(position_ids) < config.max_soft_tokens:
        patch_values.append([0.0] * (config.model_patch_size * config.model_patch_size * 3))
        position_ids.append([-1, -1])

    return {
        "pixel_values": [value for patch in patch_values for value in patch],
        "pixel_position_ids": position_ids,
        "soft_token_count": soft_token_count,
    }


def _load_pil_image(image: Any):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Gemma4 unified image preprocessing requires Pillow; "
            "install ax-engine[multimodal] or pillow"
        ) from exc

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, bytes):
        from io import BytesIO

        return Image.open(BytesIO(image))
    return Image.open(image)


def _resized_dimensions(
    width: int,
    height: int,
    patch_size: int,
    pooling_kernel_size: int,
    max_soft_tokens: int,
) -> tuple[int, int]:
    max_patches = max_soft_tokens * pooling_kernel_size**2
    target_px = max_patches * patch_size**2
    factor = math.sqrt(target_px / (height * width))
    side_mult = pooling_kernel_size * patch_size
    target_height = math.floor(factor * height / side_mult) * side_mult
    target_width = math.floor(factor * width / side_mult) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("attempting to resize to a 0 x 0 image")

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(math.floor(width / height) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(math.floor(height / width) * side_mult, max_side_length)
    return int(target_width), int(target_height)


def _rgb_pixels(
    image: Any,
    config: _Gemma4UnifiedImageConfig,
) -> list[tuple[float, float, float]]:
    values = []
    data = image.tobytes()
    for index in range(0, len(data), 3):
        red, green, blue = data[index], data[index + 1], data[index + 2]
        pixel = [float(red), float(green), float(blue)]
        if config.do_rescale:
            pixel = [channel * config.rescale_factor for channel in pixel]
        if config.do_normalize:
            pixel = [
                (channel - mean) / std
                for channel, mean, std in zip(pixel, config.image_mean, config.image_std)
            ]
        values.append((pixel[0], pixel[1], pixel[2]))
    return values


def _patchify_rgb(
    pixels: list[tuple[float, float, float]],
    width: int,
    height: int,
    config: _Gemma4UnifiedImageConfig,
) -> tuple[list[list[float]], list[list[int]]]:
    patch = config.model_patch_size
    patch_height = height // patch
    patch_width = width // patch
    patch_values: list[list[float]] = []
    position_ids: list[list[int]] = []
    for patch_y in range(patch_height):
        for patch_x in range(patch_width):
            values: list[float] = []
            y0 = patch_y * patch
            x0 = patch_x * patch
            for dy in range(patch):
                row_offset = (y0 + dy) * width
                for dx in range(patch):
                    values.extend(pixels[row_offset + x0 + dx])
            patch_values.append(values)
            position_ids.append([patch_x, patch_y])
    return patch_values, position_ids


def _expand_image_placeholders(
    input_tokens: list[int],
    soft_token_counts: list[int],
    config: _Gemma4UnifiedImageConfig,
) -> tuple[list[int], list[dict[str, Any]]]:
    actual = sum(1 for token in input_tokens if token == config.image_token_id)
    if actual != len(soft_token_counts):
        raise ValueError(
            "Gemma4 unified image placeholder count mismatch: "
            f"expected {len(soft_token_counts)}, found {actual}"
        )

    output: list[int] = []
    spans: list[dict[str, Any]] = []
    image_index = 0
    for placeholder_index, token in enumerate(input_tokens):
        if token != config.image_token_id:
            output.append(token)
            continue

        soft_token_count = soft_token_counts[image_index]
        replacement_start = len(output)
        output.append(config.boi_token_id)
        output.extend([config.image_token_id] * soft_token_count)
        output.append(config.eoi_token_id)
        spans.append(
            {
                "modality": "image",
                "placeholder_index": placeholder_index,
                "replacement_start": replacement_start,
                "soft_token_count": soft_token_count,
                "replacement_token_count": soft_token_count + 2,
            }
        )
        image_index += 1
    return output, spans


def _required_int(config: dict[str, Any], key: str) -> int:
    value = _optional_int(config, key)
    if value is None:
        raise ValueError(f"missing Gemma4 unified config field {key}")
    return value


def _optional_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    return int(value)


def _triple(value: Any) -> tuple[float, float, float]:
    items = list(value)
    if len(items) != 3:
        raise ValueError("Gemma4 unified image mean/std must contain three values")
    return float(items[0]), float(items[1]), float(items[2])
