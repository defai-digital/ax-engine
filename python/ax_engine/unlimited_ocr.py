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
_MAX_TOKEN_ID = 2**32 - 1


@dataclass(frozen=True)
class UnlimitedOcrImageRequest:
    """Prepared native Unlimited-OCR request for one image or ordered OCR-2 pages."""

    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    soft_token_count: int


def prepare_unlimited_ocr_image_request(
    model_dir: str | Path,
    input_tokens: list[int],
    images: list[Any],
    *,
    cropping: bool | None = None,
) -> UnlimitedOcrImageRequest:
    """Prepare the public native Unlimited-OCR image request.

    V1 accepts one ``<image>``. OCR-2 requires ``<PAGE><image>`` for each page.
    This helper expands that placeholder to the exact soft-token positions used
    by the native global + local-tile vision path and attaches bounded RGB8
    input. Resize, tiling, letterbox, normalization, and BF16 conversion remain
    native so Python and Rust entry points share one preprocessing contract.
    """

    directory = Path(model_dir).expanduser()
    config = _load_json_object(directory / "config.json", "Unlimited-OCR config")
    sparse = _sparse_attention_config(config)
    is_v2 = isinstance(sparse, dict) and sparse.get("use_sparse_attention") is True
    image_token_id = _load_image_token_id(directory)
    tokens = _validate_input_tokens(input_tokens)
    if not isinstance(images, list) or not images or len(images) > (32 if is_v2 else 1):
        raise ValueError(
            "native Unlimited-OCR requires exactly one source image"
            if not is_v2
            else "Unlimited-OCR-2 requires 1..32 ordered source images"
        )
    if tokens.count(image_token_id) != len(images):
        raise ValueError(
            "Unlimited-OCR input_tokens must contain exactly one <image> token per image"
        )
    if cropping is None:
        cropping = len(images) == 1
    if not isinstance(cropping, bool):
        raise TypeError("Unlimited-OCR cropping must be a boolean")
    page_token_id = sparse.get("page_token_id") if isinstance(sparse, dict) else None
    if is_v2:
        if (
            not isinstance(page_token_id, int)
            or isinstance(page_token_id, bool)
            or not 0 <= page_token_id <= _MAX_TOKEN_ID
            or page_token_id == image_token_id
        ):
            raise ValueError("Unlimited-OCR-2 requires a valid page_token_id")
        if tokens.count(page_token_id) != len(images) or any(
            i == 0 or tokens[i - 1] != page_token_id
            for i, token in enumerate(tokens)
            if token == image_token_id
        ):
            raise ValueError("Unlimited-OCR-2 requires <PAGE><image> for each source image")
    payloads = []
    counts = []
    total_rgb_bytes = 0
    for source in images:
        image = _load_rgb_image(source)
        try:
            width, height = image.size
            _validate_dimensions(width, height)
            total_rgb_bytes += width * height * 3
            if total_rgb_bytes > _MAX_RGB_BYTES:
                raise ValueError("Unlimited-OCR total RGB payload exceeds the request limit")
            payloads.append({"width": width, "height": height, "rgb_bytes": image.tobytes()})
            counts.append(_soft_token_count(width, height, cropping, 16 if is_v2 else 32))
        finally:
            image.close()
    expanded_tokens = []
    index = 0
    for token in tokens:
        if token == image_token_id:
            expanded_tokens.extend([token] * counts[index])
            index += 1
        else:
            expanded_tokens.append(token)
    runtime: dict[str, Any] = {
        "image_token_id": image_token_id,
        "soft_token_count": sum(counts),
        "cropping": cropping,
        "images": payloads,
    }
    if is_v2:
        runtime["page_token_id"] = page_token_id
    return UnlimitedOcrImageRequest(expanded_tokens, {"unlimited_ocr": runtime}, sum(counts))


def _crop_grid(width: int, height: int, cropping: bool, max_tiles: int = 32) -> tuple[int, int]:
    if (
        not cropping
        or (width <= UNLIMITED_OCR_LOCAL_TILE_SIZE and height <= UNLIMITED_OCR_LOCAL_TILE_SIZE)
        or width <= 0
        or height <= 0
    ):
        return (1, 1)

    ratios: list[tuple[int, int]] = []
    for n in range(2, max_tiles + 1):
        for columns in range(1, n + 1):
            for rows in range(1, n + 1):
                count = columns * rows
                ratio = (columns, rows)
                if 2 <= count <= max_tiles and ratio not in ratios:
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
            difference == best_diff and area > 0.5 * tile_area * columns * rows
        ):
            best_diff = difference
            best = (columns, rows)
    return best


def _soft_token_count(width: int, height: int, cropping: bool, max_tiles: int = 32) -> int:
    columns, rows = _crop_grid(width, height, cropping, max_tiles)
    if (columns, rows) == (1, 1):
        return UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT
    local_rows = rows * UNLIMITED_OCR_LOCAL_QUERY_GRID
    local_columns = columns * UNLIMITED_OCR_LOCAL_QUERY_GRID
    return UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT + local_rows * (local_columns + 1)


def _validate_input_tokens(input_tokens: list[int]) -> list[int]:
    if not isinstance(input_tokens, list) or not input_tokens:
        raise ValueError("Unlimited-OCR input_tokens must be a non-empty list")
    if any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0 or token > _MAX_TOKEN_ID
        for token in input_tokens
    ):
        raise ValueError("Unlimited-OCR input_tokens must contain unsigned 32-bit integers")
    return list(input_tokens)


def _same_json_value(first: Any, second: Any) -> bool:
    # JSON booleans and number representations must match the native Value comparison.
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _same_json_value(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, list):
        return len(first) == len(second) and all(
            _same_json_value(left, right) for left, right in zip(first, second, strict=True)
        )
    return bool(first == second)


def _sparse_attention_config(config: dict[str, Any]) -> object:
    # Same scopes and conflict rule as the native OCR-2 config contract.
    found = [
        scope["sparse_attention_config"]
        for scope in (config, config.get("language_config"), config.get("text_config"))
        if isinstance(scope, dict) and scope.get("sparse_attention_config") is not None
    ]
    if any(not _same_json_value(other, found[0]) for other in found[1:]):
        raise ValueError("Unlimited-OCR config has conflicting sparse attention scopes")
    return found[0] if found else None


def _load_image_token_id(model_dir: Path) -> int:
    config = _load_json_object(model_dir / "config.json", "Unlimited-OCR config")
    model_type = str(config.get("model_type", "")).replace("-", "_")
    if model_type not in {"unlimited_ocr", "deepseekocr"}:
        raise ValueError(
            "Unlimited-OCR config model_type must be unlimited-ocr, unlimited_ocr, or deepseekocr"
        )
    sparse = _sparse_attention_config(config)
    configured = config.get(
        "image_token_id", sparse.get("image_token_id") if isinstance(sparse, dict) else None
    )
    if isinstance(configured, int) and not isinstance(configured, bool):
        if configured < 0 or configured > _MAX_TOKEN_ID:
            raise ValueError("Unlimited-OCR image_token_id must be an unsigned 32-bit integer")
        return configured

    tokenizer = _load_json_object(model_dir / "tokenizer.json", "tokenizer")
    added_tokens = tokenizer.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer added_tokens must be a list")
    matches: set[int] = set()
    for item in added_tokens:
        if not isinstance(item, dict) or item.get("content") != "<image>":
            continue
        token_id = item.get("id")
        if (
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id <= _MAX_TOKEN_ID
        ):
            matches.add(token_id)
    if len(matches) != 1:
        raise ValueError("tokenizer must define exactly one unsigned 32-bit <image> token ID")
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

    # Pillow's decompression-bomb guard raises an Exception subclass that is
    # neither OSError nor ValueError.
    decode_errors = (OSError, ValueError, Image.DecompressionBombError)
    if isinstance(value, Image.Image):
        _validate_dimensions(*value.size)
        try:
            # copy() forces the lazy decode of a still-unloaded image.
            source = value.copy()
        except decode_errors as exc:
            raise ValueError("cannot decode Unlimited-OCR image") from exc
    elif isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Unlimited-OCR image not found: {path}")
        try:
            opened = Image.open(path)
        except decode_errors as exc:
            raise ValueError(f"cannot decode Unlimited-OCR image: {path}") from exc
        with opened:
            _validate_dimensions(*opened.size)
            try:
                opened.load()
                source = opened.copy()
            except decode_errors as exc:
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
    except decode_errors as exc:
        raise ValueError("cannot convert Unlimited-OCR image to RGB") from exc
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
