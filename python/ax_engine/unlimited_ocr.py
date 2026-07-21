from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT = 273
UNLIMITED_OCR_LOCAL_TILE_SIZE = 640
UNLIMITED_OCR_MAX_LOCAL_TILES = 32
UNLIMITED_OCR_LOCAL_QUERY_GRID = 10
_MAX_IMAGE_DIMENSION = 32_768
_MAX_RGB_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class UnlimitedOcrImageRequest:
    """Prepared native Unlimited-OCR request for one document image."""

    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    soft_token_count: int


def prepare_unlimited_ocr_image_request(
    model_dir: str | Path,
    input_tokens: list[int],
    images: list[Any],
    *,
    cropping: bool = True,
) -> UnlimitedOcrImageRequest:
    """Prepare the public native Unlimited-OCR image request.

    The caller tokenizes a prompt containing exactly one literal ``<image>``.
    This helper expands that placeholder to the exact soft-token positions used
    by the native global + local-tile vision path and attaches bounded RGB8
    input. Resize, tiling, letterbox, normalization, and BF16 conversion remain
    native so Python and Rust entry points share one preprocessing contract.
    """

    directory = Path(model_dir).expanduser()
    image_token_id = _load_image_token_id(directory)
    tokens = _validate_input_tokens(input_tokens)
    placeholder_count = tokens.count(image_token_id)
    if placeholder_count != 1:
        raise ValueError(
            "Unlimited-OCR input_tokens must contain exactly one <image> token; "
            f"found {placeholder_count}"
        )
    if not isinstance(images, list) or len(images) != 1:
        actual = len(images) if isinstance(images, list) else type(images).__name__
        raise ValueError(
            "native Unlimited-OCR requires exactly one source image; "
            f"found {actual}"
        )
    if not isinstance(cropping, bool):
        raise TypeError("Unlimited-OCR cropping must be a boolean")
    image = _load_rgb_image(images[0])
    try:
        width, height = image.size
        _validate_dimensions(width, height)
        rgb_bytes = image.tobytes()
    finally:
        image.close()

    soft_token_count = _soft_token_count(width, height, cropping)
    placeholder_index = tokens.index(image_token_id)
    expanded_tokens = [
        *tokens[:placeholder_index],
        *([image_token_id] * soft_token_count),
        *tokens[placeholder_index + 1 :],
    ]

    return UnlimitedOcrImageRequest(
        input_tokens=expanded_tokens,
        multimodal_inputs={
            "unlimited_ocr": {
                "image_token_id": image_token_id,
                "soft_token_count": soft_token_count,
                "cropping": cropping,
                "images": [
                    {
                        "width": width,
                        "height": height,
                        "rgb_bytes": rgb_bytes,
                    }
                ],
            }
        },
        soft_token_count=soft_token_count,
    )


def _crop_grid(width: int, height: int, cropping: bool) -> tuple[int, int]:
    if (
        not cropping
        or (width <= UNLIMITED_OCR_LOCAL_TILE_SIZE and height <= UNLIMITED_OCR_LOCAL_TILE_SIZE)
        or width <= 0
        or height <= 0
    ):
        return (1, 1)

    ratios: list[tuple[int, int]] = []
    for n in range(2, UNLIMITED_OCR_MAX_LOCAL_TILES + 1):
        for columns in range(1, n + 1):
            for rows in range(1, n + 1):
                count = columns * rows
                ratio = (columns, rows)
                if 2 <= count <= UNLIMITED_OCR_MAX_LOCAL_TILES and ratio not in ratios:
                    ratios.append(ratio)
    ratios.sort(key=lambda ratio: ratio[0] * ratio[1])

    aspect = width / height
    area = width * height
    tile_area = UNLIMITED_OCR_LOCAL_TILE_SIZE**2
    best = (1, 1)
    best_diff = float("inf")
    for columns, rows in ratios:
        difference = abs(aspect - columns / rows)
        if difference < best_diff or (
            difference == best_diff
            and area > 0.5 * tile_area * columns * rows
        ):
            best_diff = difference
            best = (columns, rows)
    return best


def _soft_token_count(width: int, height: int, cropping: bool) -> int:
    columns, rows = _crop_grid(width, height, cropping)
    if (columns, rows) == (1, 1):
        return UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT
    local_rows = rows * UNLIMITED_OCR_LOCAL_QUERY_GRID
    local_columns = columns * UNLIMITED_OCR_LOCAL_QUERY_GRID
    return UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT + local_rows * (local_columns + 1)


def _validate_input_tokens(input_tokens: list[int]) -> list[int]:
    if not isinstance(input_tokens, list) or not input_tokens:
        raise ValueError("Unlimited-OCR input_tokens must be a non-empty list")
    if any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in input_tokens
    ):
        raise ValueError("Unlimited-OCR input_tokens must contain non-negative integers")
    return list(input_tokens)


def _load_image_token_id(model_dir: Path) -> int:
    config = _load_json_object(model_dir / "config.json", "Unlimited-OCR config")
    model_type = str(config.get("model_type", "")).replace("-", "_")
    if model_type not in {"unlimited_ocr", "deepseekocr"}:
        raise ValueError(
            "Unlimited-OCR config model_type must be unlimited-ocr, unlimited_ocr, "
            "or deepseekocr"
        )
    configured = config.get("image_token_id")
    if isinstance(configured, int) and not isinstance(configured, bool) and configured >= 0:
        return configured

    tokenizer = _load_json_object(model_dir / "tokenizer.json", "tokenizer")
    matches = {
        item.get("id")
        for item in tokenizer.get("added_tokens", [])
        if isinstance(item, dict)
        and item.get("content") == "<image>"
        and isinstance(item.get("id"), int)
        and not isinstance(item.get("id"), bool)
        and item["id"] >= 0
    }
    if len(matches) != 1:
        raise ValueError("tokenizer must define exactly one non-negative <image> token ID")
    return matches.pop()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_rgb_image(value: Any):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Unlimited-OCR image requests require Pillow; install ax-engine[multimodal]"
        ) from exc

    if isinstance(value, Image.Image):
        source = value.copy()
    elif isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Unlimited-OCR image not found: {path}")
        try:
            with Image.open(path) as opened:
                opened.load()
                source = opened.copy()
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot decode Unlimited-OCR image: {path}") from exc
    else:
        raise TypeError(
            "Unlimited-OCR images must be local paths or PIL.Image.Image instances; "
            f"got {type(value).__name__}"
        )

    transposed = None
    try:
        transposed = ImageOps.exif_transpose(source)
        rgb = transposed.convert("RGB")
    finally:
        if transposed is not None and transposed is not source:
            transposed.close()
        source.close()
    return rgb


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Unlimited-OCR image dimensions must be non-zero")
    if width > _MAX_IMAGE_DIMENSION or height > _MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Unlimited-OCR image dimensions must not exceed {_MAX_IMAGE_DIMENSION} per side"
        )
    expected = width * height * 3
    if expected > _MAX_RGB_BYTES:
        raise ValueError(f"Unlimited-OCR RGB payload exceeds {_MAX_RGB_BYTES} bytes")
