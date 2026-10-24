#!/usr/bin/env python3
"""Build the Mac icon from the immutable user-supplied AX Engine master.

Follows AX Code Desktop's icon policy: preserve composition/aspect ratio,
90% canvas safe zone, antialiased rounded plate, transparent corners, a full
Retina iconset, and placement checks. No crop, recolor, or repeated masking.
Requires Pillow; iconutil is provided by macOS.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageChops, ImageDraw, ImageFilter

SOURCE_SHA256 = "a1cee6d646f3fc2ab3baa6cf7dd5167a4a910ab968670e88015d49d9bda18836"


def build_master(source: Image.Image, size: int = 1024) -> Image.Image:
    source = source.convert("RGBA")
    plate = source.getpixel((source.width // 20, source.height // 20))[:3] + (255,)
    square = Image.new("RGBA", (size, size), plate)
    scale = min(size / source.width, size / source.height) * 0.90
    artwork = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
    square.alpha_composite(artwork, ((size - artwork.width) // 2, (size - artwork.height) // 2))
    oversample = 4
    mask = Image.new("L", (size * oversample, size * oversample))
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, mask.width - 1, mask.height - 1), radius=round(mask.width * 0.2237), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.4)).resize((size, size), Image.Resampling.LANCZOS)
    square.putalpha(ImageChops.multiply(square.getchannel("A"), mask))
    return square


def verify(image: Image.Image) -> None:
    w, h = image.size
    assert (w, h) == (1024, 1024)
    assert all(image.getpixel(p)[3] <= 20 for p in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
    assert image.getpixel((w // 2, h // 2))[3] >= 200
    # Test the dark/green mark's bounding box, not its color-weighted centroid:
    # AX Engine's intentional left green bar has different luminance.
    points = [(x, y) for y in range(h) for x in range(w)
              if (lambda p: p[3] >= 200 and (max(p[:3]) < 100 or (p[1] > p[0] + 25 and p[1] > p[2] + 10)))(image.getpixel((x, y)))]
    assert points, "Logo mark is missing"
    xs, ys = zip(*points)
    assert abs((min(xs) + max(xs)) / 2 - w / 2) <= 24, "Logo shifted horizontally"
    assert abs((min(ys) + max(ys)) / 2 - h / 2) <= 24, "Logo shifted vertically"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "Resources/icon-source.png"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256, "Source artwork changed; review provenance first"
    master = build_master(Image.open(source))
    verify(master)
    args.output.mkdir(parents=True, exist_ok=True)
    master.save(args.output / "AppIcon.png", optimize=True)
    with tempfile.TemporaryDirectory(prefix="ax-engine-icon-") as temp:
        iconset = Path(temp) / "AppIcon.iconset"
        iconset.mkdir()
        for size in [16, 32, 128, 256, 512]:
            for multiplier in [1, 2]:
                suffix = "@2x" if multiplier == 2 else ""
                master.resize((size * multiplier, size * multiplier), Image.Resampling.LANCZOS).save(iconset / f"icon_{size}x{size}{suffix}.png")
        subprocess.run(["/usr/bin/iconutil", "-c", "icns", str(iconset), "-o", str(args.output / "AppIcon.icns")], check=True)
    print("Verified master, centering, transparent corners and 10 iconset representations.")


if __name__ == "__main__":
    main()
