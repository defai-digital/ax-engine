# Unlimited-OCR-2 MXFP8 functional controls

Date: 2026-10-29. Scope: experimental native graph functionality.
Host: MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.0.1.
MLX: published 0.32.4 wheel. Build: Homebrew Cargo/Rust 1.99.1,
`maturin develop --profile release-pyext`.

Exact pack: `AutomatosX/AX-Unlimited-OCR-2-AXQ-MXFP8` at
`462c40bb4f2c7eeb3e1230e912940efcbdefc8a3`.
Both weight shards, the config, tokenizer and tensor index are SHA-256 checked
before inference. The pack's archived Python code is not executed.

| Control | Result |
| --- | --- |
| Single page, global view | Complete normalized OCR body and one output page match |
| Single page, dynamic tiles | Complete normalized OCR body and one output page match |
| Ordered two-page input | Both complete OCR bodies and output page order/count match |
| Repeat first request after the other controls | Token output identical to the first request |

A separate text-only negative control returns `status=failed`,
`finish_reason=error` and no output tokens, retained in
`text-only-rejected.json`. The Python API reports this as a failed result;
callers should inspect result status.

`summary.json` and individual case JSON files retain raw tokens, layout text,
expected text and timings. `page-1.png` and `page-2.png` are the synthetic
inputs. Normalization permits whitespace and letter case differences; extra
or missing body text fails the witness. Layout coordinates are not scored.
`build-provenance.json` identifies the native extension and toolchain;
`source-binding.json` binds 509 build/runtime source files before and after
the build. Its base commit is combined with these file hashes to identify
the uncommitted implementation used for this run.

Reproduce with a build containing this implementation:

```bash
python scripts/validate_unlimited_ocr2.py \
  --model-dir /path/to/AX-Unlimited-OCR-2-AXQ-MXFP8 \
  --output /tmp/ax-ocr2-validation \
  --host-label "MacBook Pro M5 Max 128 GB"
```

The test uses synthetic English pages. It does not qualify general OCR
quality, multilingual documents, long contexts, cancellation/interleaving,
MTP, throughput, or constant-memory document scheduling. Retained K/V and
router keys grow with the request. Process peak RSS and request timings
describe this bounded run only; there is no baseline speedup claim.

Independent 1.5-entmax math is also compared with 16 deterministic CPU
outputs from the unchanged pinned upstream `dash_attention.py` (PyTorch
2.15.0). The source hash and oracle values are retained in
[`unlimited-ocr2-entmax.json`](../../crates/ax-engine-core/tests/fixtures/unlimited-ocr2-entmax.json).
This primitive comparison is separate from full-model BF16 parity.
