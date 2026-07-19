"""Multimodal QA probes against a running ``ax-engine-server``.

Best-practice layers (product inference engines, not full VLMEval):

1. **Fail-closed policy** — remote media URLs and public video rejected
2. **Capability honesty** — if ``/v1/models`` advertises image, image chat
   must return non-empty content (no soft-skip)
3. **Path smoke** — tokens flow through vision/audio when package supports it
4. **Optional content** — deterministic color / speech checks (tier ``standard``)

This is **not** a public multimodal leaderboard (MMMU / OCRBench / …). It is a
serving gate that blocks engine and contract regressions.

Matrix mode ``multimodal`` uses this module instead of the text question bank.
Deep Gemma-only operator suite remains ``scripts/qa_gemma4_multimodal.py``.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from surface_probes import (  # flat import when qa/ is on sys.path
    SurfaceProbeResult,
    _post_json,
    chat_completion_payload,
    extract_chat_content,
    fetch_model_card,
    model_advertises_image,
    probe_multimodal_image,
    probe_remote_media_rejected,
    probe_video_rejected,
    tiny_png_data_url,
)


@dataclass
class MultimodalReport:
    base_url: str
    model: str
    tier: str
    results: list[SurfaceProbeResult] = field(default_factory=list)
    package_multimodal: bool = False
    advertises_image: bool = False

    @property
    def hard_passed(self) -> bool:
        return all(r.passed or r.skipped for r in self.results if r.hard)

    @property
    def summary_line(self) -> str:
        hard = [r for r in self.results if r.hard and not r.skipped]
        ok = sum(1 for r in hard if r.passed)
        skip = sum(1 for r in self.results if r.skipped)
        return (
            f"multimodal/{self.tier} hard {ok}/{len(hard)} passed "
            f"({skip} skipped)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "multimodal_probes",
            "base_url": self.base_url,
            "model": self.model,
            "tier": self.tier,
            "package_multimodal": self.package_multimodal,
            "advertises_image": self.advertises_image,
            "hard_passed": self.hard_passed,
            "summary": self.summary_line,
            "results": [asdict(r) for r in self.results],
            "market_alignment": {
                "layers": [
                    "fail_closed_policy",
                    "capability_honesty",
                    "path_smoke",
                    "optional_content",
                ],
                "not_vendored": ["mmmu", "ocrbench", "vlmevalkit"],
            },
        }


def package_looks_like_multimodal(artifacts: Path) -> bool:
    """True when artifact config looks like Gemma 4 unified (vision_config + ids)."""
    cfg_path = artifacts / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(cfg, dict):
        return False
    if "vision_config" in cfg and (
        "image_token_id" in cfg or "boi_token_id" in cfg
    ):
        return True
    # preprocessor alone is weak; require image processor + sampling audio shape
    prep = artifacts / "preprocessor_config.json"
    if prep.is_file() and "image_token_id" in cfg:
        return True
    return False


def solid_png_rgb(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Encode a solid-color RGB PNG with stdlib only (no Pillow)."""
    if width < 1 or height < 1:
        raise ValueError("png dimensions must be positive")
    r, g, b = rgb
    raw = bytearray()
    row = bytes([r, g, b]) * width
    for _ in range(height):
        raw.append(0)  # filter none
        raw.extend(row)
    compressed = zlib.compress(bytes(raw), level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _thinking_channel_leaked(text: str) -> bool:
    stripped = text.strip()
    return stripped == "thought" or stripped.startswith("thought\n")


def probe_image_color_content(
    base_url: str,
    model: str,
    *,
    timeout: float = 120.0,
    color_name: str = "blue",
    rgb: tuple[int, int, int] = (0, 0, 255),
) -> SurfaceProbeResult:
    """Ask for solid-color recognition; hard content check at temperature 0."""
    name = "image_color_content"
    start = time.monotonic()
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    png = solid_png_rgb(64, 64, rgb)
    content = [
        {
            "type": "text",
            "text": "What color is this image? Answer in one word.",
        },
        {"type": "image_url", "image_url": {"url": png_data_url(png)}},
    ]
    status, body = _post_json(
        url,
        chat_completion_payload(model, content, max_tokens=32, temperature=0.0),
        timeout=timeout,
    )
    elapsed = (time.monotonic() - start) * 1000
    if status in (400, 415, 422):
        return SurfaceProbeResult(
            name,
            False,
            f"image content probe rejected HTTP {status}: {str(body)[:160]}",
            elapsed_ms=elapsed,
        )
    if status >= 500 or status == 0:
        return SurfaceProbeResult(
            name,
            False,
            f"server/connection error HTTP {status}: {str(body)[:160]}",
            elapsed_ms=elapsed,
        )
    if status != 200:
        return SurfaceProbeResult(
            name,
            False,
            f"unexpected HTTP {status}: {str(body)[:160]}",
            elapsed_ms=elapsed,
        )
    text = extract_chat_content(body) or ""
    if not text.strip():
        return SurfaceProbeResult(name, False, "empty content", elapsed_ms=elapsed)
    if _thinking_channel_leaked(text):
        return SurfaceProbeResult(
            name,
            False,
            f"thinking-channel leak: {text[:80]!r}",
            elapsed_ms=elapsed,
        )
    hit = (
        re.search(rf"\b{re.escape(color_name)}\b", text, re.IGNORECASE) is not None
    )
    if not hit:
        return SurfaceProbeResult(
            name,
            False,
            f"expected color {color_name!r} in {text[:80]!r}",
            elapsed_ms=elapsed,
        )
    return SurfaceProbeResult(
        name,
        True,
        f"matched {color_name!r} in {len(text)} chars",
        elapsed_ms=elapsed,
    )


def probe_image_describe_smoke(
    base_url: str,
    model: str,
    *,
    timeout: float = 120.0,
) -> SurfaceProbeResult:
    """Non-empty image description + no thinking-channel leak."""
    name = "image_describe_smoke"
    start = time.monotonic()
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    content = [
        {"type": "text", "text": "Describe this image in one short sentence."},
        {"type": "image_url", "image_url": {"url": tiny_png_data_url()}},
    ]
    status, body = _post_json(
        url,
        chat_completion_payload(model, content, max_tokens=48, temperature=0.0),
        timeout=timeout,
    )
    elapsed = (time.monotonic() - start) * 1000
    if status != 200:
        return SurfaceProbeResult(
            name,
            False,
            f"HTTP {status}: {str(body)[:160]}",
            elapsed_ms=elapsed,
        )
    text = extract_chat_content(body) or ""
    if not text.strip():
        return SurfaceProbeResult(name, False, "empty content", elapsed_ms=elapsed)
    if _thinking_channel_leaked(text):
        return SurfaceProbeResult(
            name,
            False,
            f"thinking-channel leak: {text[:80]!r}",
            elapsed_ms=elapsed,
        )
    return SurfaceProbeResult(
        name, True, f"describe returned {len(text)} chars", elapsed_ms=elapsed
    )


def run_multimodal_probes(
    base_url: str,
    model: str,
    *,
    timeout: float = 120.0,
    tier: str = "smoke",
    artifacts: Optional[Path] = None,
    require_image: Optional[bool] = None,
) -> MultimodalReport:
    """Run multimodal QA tiers.

    * ``smoke`` — fail-closed + capability-aware image path
    * ``standard`` — smoke + solid-color content + describe smoke
    """
    if tier not in ("smoke", "standard"):
        raise ValueError(f"unknown multimodal tier: {tier}")

    package_mm = (
        package_looks_like_multimodal(artifacts) if artifacts is not None else False
    )
    card = fetch_model_card(base_url, model, timeout=min(timeout, 30.0))
    advertises = model_advertises_image(card)
    if require_image is None:
        # Matrix multimodal cells and packages that look multimodal always require.
        require_image = True if package_mm else advertises

    report = MultimodalReport(
        base_url=base_url,
        model=model,
        tier=tier,
        package_multimodal=package_mm,
        advertises_image=advertises,
    )

    # Layer 1: product policy (hard for every model)
    report.results.append(
        probe_remote_media_rejected(base_url, model, timeout=timeout)
    )
    report.results.append(probe_video_rejected(base_url, model, timeout=timeout))

    # Layer 2–3: image path honesty
    report.results.append(
        probe_multimodal_image(
            base_url,
            model,
            timeout=timeout,
            require_image=require_image,
        )
    )

    if tier == "standard":
        # Content checks only when image is required (vision package / advertised).
        if require_image:
            report.results.append(
                probe_image_color_content(base_url, model, timeout=timeout)
            )
            report.results.append(
                probe_image_describe_smoke(base_url, model, timeout=timeout)
            )
        else:
            report.results.append(
                SurfaceProbeResult(
                    "image_color_content",
                    True,
                    "skipped: model does not advertise image input",
                    hard=False,
                    skipped=True,
                )
            )
            report.results.append(
                SurfaceProbeResult(
                    "image_describe_smoke",
                    True,
                    "skipped: model does not advertise image input",
                    hard=False,
                    skipped=True,
                )
            )

    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AX Engine multimodal QA probes")
    parser.add_argument("--base-url", default="http://127.0.0.1:31418")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--tier",
        default="smoke",
        choices=["smoke", "standard"],
        help="smoke = policy+path; standard = + color/describe content",
    )
    parser.add_argument(
        "--artifacts",
        default=None,
        help="Optional artifact dir for package_looks_like_multimodal",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument(
        "--require-image",
        action="store_true",
        help="Force hard image path (default: auto from capabilities/package)",
    )
    args = parser.parse_args(argv)

    report = run_multimodal_probes(
        args.base_url,
        args.model,
        timeout=args.timeout,
        tier=args.tier,
        artifacts=Path(args.artifacts) if args.artifacts else None,
        require_image=True if args.require_image else None,
    )
    for r in report.results:
        flag = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        print(f"  [{flag}] {r.name}: {r.detail} ({r.elapsed_ms:.0f}ms)")
    print(report.summary_line)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.as_dict(), indent=2))
        print(f"JSON: {path}")
    return 0 if report.hard_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
