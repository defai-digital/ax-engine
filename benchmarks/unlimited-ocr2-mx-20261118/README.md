# Unlimited-OCR-2 MXFP4 and MXFP8 functional controls

The pinned MXFP4 and MXFP8 packs each passed the single-page global view,
the single-page tiled view, the ordered two-page request, and the repeat
of the first request. Each run reached native stop, returned the expected
text and page count, and reproduced its first request's token IDs after
the intervening cases.

The native Python extension used the `release-pyext` profile, Homebrew
Rust 1.99.1, and MLX 0.32.4 on MacBook Pro M5 Max 128 GB running macOS
27.2.1. MXFP4 weights were on network storage and MXFP8 weights were on
local storage. The
[runtime receipt](runtime.json) binds the source patch and extension.
The wheel SHA-256 is
`188e715f5445c507a9ad2ca52228182a8fe13e56119ce22d581ddddae404cd89`.

- [MXFP4 result](mxfp4.json) and [raw output](mxfp4.log.gz)
- [MXFP8 result](mxfp8.json) and [raw output](mxfp8.log.gz)
- [MXFP4 pack inventory](pack-inventory.json) and
  [native manifest checks](manifest-check.json)

The generated MXFP4 native manifest identifies 5,015 tensors, including
2,196 packed U32 weight and U8 scale pairs. The manifest controls rejected
three malformed geometries. Nine Rust OCR-2 tests and four Python
validator tests passed. These controls cover two exact revisions and
synthetic page images. Full per-request KV and routing keys remain
resident during generation.

Run the validator with the pinned artifact and the compiled native
Python extension. Repeat the command with the other pack and format.

```bash
python3 scripts/validate_unlimited_ocr2.py \
  --model-dir "$MODEL_DIR" --pack-format mxfp4 --output "$RESULT_DIR" \
  --host-label "MacBook Pro M5 Max 128 GB"
```
