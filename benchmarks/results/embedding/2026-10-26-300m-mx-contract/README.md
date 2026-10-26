# EmbeddingGemma 300M MX format regression

Date: 2026-10-26. Hardware: MacBook Pro M5 Max 128 GB, macOS 27.0.1,
MLX 0.32.4, Homebrew Rust/Cargo 1.99.1 (no rustup). Base commit:
`1170fd4e591651cacdd2cc10675fc3804fdade40`; test source hashes are in
`source-files.json`. This extends the existing native 300M encoder's format
coverage; it introduces no inference algorithm or model conversion.

## Scope

Both MXFP4 and MXFP8 are exercised in each of two regression tests:

- Manifest conversion: `gemma3_text` stays `embeddinggemma` / `EncoderEmbed`,
  MX group-32 layer weights coexist with protected affine8/group64 token
  embeddings and FP32 Dense heads, both Dense projections are retained, and
  admission rejects a missing second projection. Fixture headers model one
  64-wide layer; this test does not load a real checkpoint.
- Metal execution: deterministic two-layer, 64-wide miniature encoders use
  affine8 token embeddings, MX packed QKV and gate/up projections, sandwich
  and Q/K norms, bidirectional attention and both unquantized Dense heads.
  Inputs of 1, 17, 33 and 129 tokens are padded to 160 in a compiled batch.
  Each single result must reach cosine 0.999 against the same weights
  explicitly dequantized with the selected MX mode; each compiled padded
  result must reach 0.999 against its single result. All outputs must be finite.

`m5-test.log` is the actual test output. These are synthetic format/dispatch
regressions, not 300M real-weight quality, throughput or checkpoint qualification.
The previously verified upstream affine8 300M pack remains available separately.
The full local Rust workspace passed 4,372 tests (88 ignored); formatting and
all-target/all-feature Clippy also passed. The GitHub integration was additionally
checked with Python, script gates, and JavaScript/Ruby/Swift SDK suites.

## Pack admission remains pending

`hub-check.json` records authenticated checks of both requested repository IDs:

- `AutomatosX/AX-EmbeddingGemma-300M-MLX-AXQ-MXFP4`
- `AutomatosX/AX-EmbeddingGemma-300M-MLX-AXQ-MXFP8`

Both returned HTTP 404 at this check. No revision is fabricated, no managed
format alias is promoted, and no source weights are substituted or requantized.
Once the snapshots are accessible, pin each revision and run the existing
`verify_embedding_models.py --model-kind embeddinggemma` and the previous
M5 direct/HTTP review harnesses against those actual packs before claiming
checkpoint support or publishing performance figures.

## Reproduction

On the recorded M5 environment, from a checkout with MLX installed:

```bash
cargo test -q -p ax-engine-core -p ax-engine-mlx --lib embeddinggemma_300m_mx \
  -- --test-threads=1 --nocapture
```

Use `rustup run 1.99.1 cargo` on hosts managed by rustup. Tests create only
small deterministic fixtures and in-memory arrays; they do not download,
convert or publish model packs.
