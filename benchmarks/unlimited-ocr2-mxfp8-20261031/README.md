# Unlimited-OCR-2 MXFP8 functional controls

Date: 2026-10-31. Scope: experimental native graph functionality.
Host: MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.0.1.
MLX: published 0.32.4 wheel. Build: Homebrew Cargo/Rust 1.99.1,
`maturin build --profile release-pyext` from a clean `git archive` of
[`b4a283e9e`](https://github.com/defai-digital/ax-engine/commit/b4a283e9eface0aea509df5a5a946aa57d822ba9).

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

The two-page control ends with one empty `<PAGE>` marker before EOS. The
page-count guard, which matches the released generation, lets the marker after
the last page through and then forces EOS; the witness does not count that
single trailing empty marker as a page. An empty page anywhere else, or a
repeated page with content, fails it.

A separate text-only negative control returns `status=failed`,
`finish_reason=error` and no output tokens, retained in
`text-only-rejected.json`. The Python API reports this as a failed result;
callers should inspect result status.

`summary.json` and individual case JSON files retain raw tokens, layout text,
expected text and timings. `page-1.png` and `page-2.png` are the synthetic
inputs. Normalization permits whitespace and letter case differences; extra
or missing body text fails the witness. Layout coordinates are not scored.
`build-provenance.json` identifies the native extension and toolchain;
`source-binding.json` hashes 527 build/runtime source files of the archived
commit before and after the build (`source_sha256` is the SHA-256 of the
sorted `<sha256>  <path>` lines). `inventory.json` comes from a manifest
regenerated at this commit; its 5,015 tensor entries are identical to the
2026-10-29 manifest.

Reproduce with a build of this commit:

```bash
python scripts/validate_unlimited_ocr2.py \
  --model-dir /path/to/AX-Unlimited-OCR-2-AXQ-MXFP8 \
  --output /tmp/ax-ocr2-validation \
  --host-label "MacBook Pro M5 Max 128 GB"
```

The same controls also pass on Mac mini M4 Pro 64 GB and Mac Studio M2 Ultra
192 GB at the page-guard commit (`0475d2894`). Those runs are not retained here.

The test uses synthetic English pages. It does not qualify general OCR
quality, multilingual documents, long contexts, cancellation/interleaving,
MTP, throughput, or constant-memory document scheduling. Retained K/V and
router keys grow with the request. Process peak RSS and request timings
describe this bounded run only; there is no baseline speedup claim.

The earlier encoder's run is retained in
[`unlimited-ocr2-mxfp8-20261029`](../unlimited-ocr2-mxfp8-20261029/README.md).
Independent 1.5-entmax math is compared with 16 deterministic CPU outputs from
the unchanged pinned upstream `dash_attention.py`; see
[`unlimited-ocr2-entmax.json`](../../crates/ax-engine-core/tests/fixtures/unlimited-ocr2-entmax.json).
This primitive comparison is separate from full-model BF16 parity.
