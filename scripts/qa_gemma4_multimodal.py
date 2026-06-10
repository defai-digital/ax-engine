#!/usr/bin/env python3
"""Runtime smoke + QA probes for Gemma 4 12B unified multimodal chat.

Sends inline base64 image / audio / video through the OpenAI-compatible
`/v1/chat/completions` endpoint of a *running* AX Engine server and checks the
responses. This is the end-to-end layer the preprocessing unit/golden tests
cannot cover (they stop before the MLX graph runs), so it doubles as:

  - a runtime smoke test (does the vision/audio/video graph produce tokens at
    all?), and
  - a QA probe set (does the model answer plausibly?).

It needs a live server with a Gemma 4 unified artifact loaded, so it is not part
of CI. Run it against a local server:

    python3 scripts/qa_gemma4_multimodal.py --url http://127.0.0.1:8000 \
        --model gemma-4-12B-it

Exit code is 0 only if every probe returns HTTP 200 with non-empty content and
no response leaks Gemma 4 thinking-channel framing (a content prefix like
`thought\n`). Content-match checks (e.g. the answer mentions "red") are
reported but do not fail the run by default, since exact wording depends on
the quantized weights; pass `--strict` to fail on substring mismatches too.

On macOS the probe set includes a speech-transcription check synthesized with
`say`/`afconvert`. Speech transcription is the reliable audio-health signal:
the 2026-06-09 baseline showed the model transcribes speech verbatim through
both AX and llama.cpp, while synthetic tone/silence classification fails in
BOTH engines (out-of-distribution for the model), so tone probes here are
smoke-only.

Requires Pillow (`pip install pillow`); WAV uses the stdlib.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment dependent
    print("error: this script needs Pillow (pip install pillow)", file=sys.stderr)
    raise


def _png_solid(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def _png_gradient(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 5) % 256, (y * 5) % 256, 128)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _gif_frames(width: int, height: int, colors: list[tuple[int, int, int]]) -> bytes:
    frames = [Image.new("RGB", (width, height), color) for color in colors]
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    return buffer.getvalue()


def _wav_tone(seconds: float = 0.5, sample_rate: int = 16000, freq: float = 220.0) -> bytes:
    import math

    n = int(seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        samples = bytearray()
        for i in range(n):
            value = int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            samples += struct.pack("<h", value)
        writer.writeframes(bytes(samples))
    return buffer.getvalue()


SPEECH_SENTENCE = "The quick brown fox jumps over the lazy dog"


def _wav_speech() -> bytes | None:
    """Synthesize a known sentence as 16 kHz mono WAV via macOS `say`.

    Returns None when synthesis is unavailable (non-macOS host, missing
    tools), in which case the speech probe is skipped with a notice.
    """
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "speech.aiff"
        wav = Path(tmp) / "speech.wav"
        try:
            subprocess.run(
                ["say", "-o", str(aiff), SPEECH_SENTENCE],
                check=True,
                capture_output=True,
                timeout=60,
            )
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            return wav.read_bytes()
        except (OSError, subprocess.SubprocessError):
            return None


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@dataclass
class Probe:
    name: str
    content: list[dict]
    expect_substring: str | None


def _build_probes() -> list[Probe]:
    # Blue is the robust color expectation: the model answers red squares as
    # "maroon"/"red" interchangeably at temperature 0 (llama.cpp's own
    # reasoning lists both as candidates for pure red), while saturated blue
    # and green answer unambiguously.
    blue_png = _png_solid(64, 64, (0, 0, 255))
    gradient_png = _png_gradient(96, 96)
    tone_wav = _wav_tone()
    speech_wav = _wav_speech()
    three_frame_gif = _gif_frames(32, 32, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

    image_part = lambda data: {  # noqa: E731 - terse local helper
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_b64(data)}"},
    }
    probes = [
        Probe(
            name="image-color",
            content=[
                {"type": "text", "text": "What color is this image? Answer in one word."},
                image_part(blue_png),
            ],
            expect_substring="blue",
        ),
        Probe(
            name="image-describe",
            content=[
                {"type": "text", "text": "Describe this image in one sentence."},
                image_part(gradient_png),
            ],
            expect_substring=None,
        ),
        # Smoke-only: synthetic tones are out-of-distribution for the model.
        # The 2026-06-09 baseline showed tone/silence classification fails the
        # same way through llama.cpp, so no content expectation is asserted —
        # speech transcription below is the audio-health signal.
        Probe(
            name="audio-describe-tone",
            content=[
                {"type": "text", "text": "Describe this audio."},
                {"type": "input_audio", "input_audio": {"data": _b64(tone_wav), "format": "wav"}},
            ],
            expect_substring=None,
        ),
        Probe(
            name="video-frame-count",
            content=[
                {"type": "text", "text": "How many distinct frames are in this clip?"},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:image/gif;base64,{_b64(three_frame_gif)}"},
                },
            ],
            expect_substring="3",
        ),
    ]
    if speech_wav is not None:
        probes.append(
            Probe(
                name="audio-transcribe-speech",
                content=[
                    {"type": "text", "text": "Transcribe the speech in this audio."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _b64(speech_wav), "format": "wav"},
                    },
                ],
                expect_substring="quick brown fox",
            )
        )
    else:
        print(
            "note: speech synthesis unavailable (say/afconvert); "
            "skipping audio-transcribe-speech probe"
        )
    return probes


def _post_chat(url: str, model: str, content: list[dict], max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="gemma-4-12B-it")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail the run when a content-match expectation misses",
    )
    args = parser.parse_args()

    failures = 0
    print(f"Probing {args.url} (model={args.model})\n")
    for probe in _build_probes():
        try:
            answer = _post_chat(args.url, args.model, probe.content, args.max_tokens)
        except (urllib.error.URLError, KeyError, ValueError) as error:
            print(f"[FAIL] {probe.name}: request error: {error}")
            failures += 1
            continue

        text = (answer or "").strip()
        if not text:
            print(f"[FAIL] {probe.name}: empty response")
            failures += 1
            continue

        # Guard against the Gemma 4 thinking-channel header leaking into chat
        # content (`thought\n…`): the server must strip channel framing.
        if text == "thought" or text.startswith("thought\n"):
            snippet = text.replace("\n", " ")[:80]
            print(f"[FAIL] {probe.name}: thinking-channel header leaked: {snippet!r}")
            failures += 1
            continue

        match = ""
        if probe.expect_substring is not None:
            # Word-boundary match so a numeric expectation like "3" is not
            # falsely satisfied by "13"/"30", and a word like "red" is not
            # satisfied by an unrelated substring.
            hit = (
                re.search(
                    rf"\b{re.escape(probe.expect_substring)}\b",
                    text,
                    re.IGNORECASE,
                )
                is not None
            )
            match = f"  (match '{probe.expect_substring}': {'yes' if hit else 'no'})"
            if args.strict and not hit:
                snippet = text.replace("\n", " ")[:80]
                print(f"[FAIL] {probe.name}: {snippet!r}{match}")
                failures += 1
                continue
        snippet = text.replace("\n", " ")[:80]
        print(f"[PASS] {probe.name}: {snippet!r}{match}")

    print()
    if failures:
        print(f"{failures} probe(s) failed.")
        return 1
    print("All probes passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
