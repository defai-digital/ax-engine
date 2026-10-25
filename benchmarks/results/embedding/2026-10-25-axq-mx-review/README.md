# AXQ MX embedding review — 2026-10-25

Bounded correctness and AX-versus-AX optimization evidence on **MacBook Pro
M5 Max 128 GB**, macOS 27.0.1, MLX 0.32.4. This is not a retrieval-quality,
maximum-context, competitive-runtime, or SKU certification result.

## Scope and fixes

- Six pinned AutomatosX Qwen3-Embedding packs: 0.6B, 4B and 8B, each MXFP4
  and MXFP8. Separate Q/K/V avoids the observed packed-MX reduction drift.
  Short inputs (below 256 tokens) retain exact lengths and per-sequence layer
  reductions. This trades short-batch speed for stable vector directions;
  longer ingest inputs keep shared batch execution. Embedding SDPA avoids
  redundant FP32 upcasts without changing generation arithmetic.
- Both AutomatosX EmbeddingGemma 2 MX packs now use a dedicated native text
  encoder: projection-only PLE, mixed local/full bidirectional attention,
  multiplicative norms, token-wise projection and masked mean pooling. Graph
  caching, broadcast padding masks, bounded local windows, and shape-limited
  packed projections reduce dispatch and mask costs. Media and chat are rejected.
- EmbeddingGemma 300M single-item pooling now excludes compile-bucket padding.
  The previous verifier accidentally exercised a batch of one instead of the
  single-item API, hiding the problem. The verifier now exercises both.
- HTTP text follows the tokenizer's special-token processor for both Gemma
  generations. Both reviewed tokenizers supply BOS and EOS; the old plain-text
  plus EOS path omitted BOS. Records apply the processor to every chunk, with
  an explicit EOS override that neither duplicates nor accidentally retains EOS.
- Invalid IDs and unsupported media tokens fail without poisoning the session;
  HTTP returns a client error while genuine runtime errors remain server errors.

The AutomatosX 300M repository returned 404 during inventory. Its existing alias
retains its identity; the new explicit `embeddinggemma-300m:8bit` fallback pins
`mlx-community/embeddinggemma-300m-8bit`. **No real 300M MXFP4/MXFP8 pack was
available or qualified.** No weights were converted or published by this review.

## Evidence interpretation

`hub-pins.json` identifies all nine immutable snapshots. `environment-final.json`
records software/toolchain versions, binary hashes, and generated-manifest
hashes. `source-files.json` binds modified/new source files to SHA-256 hashes;
the base is `fcf6e0fc8`. The benchmark toolchain is Homebrew Rust/Cargo 1.99.1,
with no rustup on the benchmark machine. The working build used release-pyext;
raw CPU `float32` matrices are materialized inside each timed call.

Final gates are `stability-final.json` and `http-final.json`. The direct gate
covers 19 lengths from 1 to 2048, single and mixed batches, repeatability,
permutation, normalization, invalid IDs and recovery. All three normalized
cosine comparisons must be at least 0.999 (single/reference, batch/reference,
batch/single). Same-shape repeats and permutations must match exactly.
Unnormalized single/batch-of-one direction must reach 0.999 and norm ratio stay
within 1%; absolute differences are retained, not treated as a bitwise contract
between different BF16 kernels. The HTTP gate also checks parallel requests,
text/token agreement, records/chunks, EOS overrides, native dimensions,
unsupported parameters, and encoder-only chat rejection.

`final/*/report.json` retains every timing sample (two warmups, five trials)
and short-text reference cosines. The accompanying arrays are raw outputs and
references. Fixed-token timing cases are deterministic throughput probes, not
semantic-quality datasets. Baseline and final runs are sequential diagnostics;
thermal/load variation limits small cross-run differences.

`gemma-optimization-final.json` compares the final Gemma 2 build with compiled
and packed paths disabled/enabled, in off/on/on/off process order, with five
warmups and nine samples per case. `qwen-sdpa-final.json` uses the same design
for the final 0.6B SDPA switch. These are the controlled optimization comparisons;
no reference-library speed ratio is asserted.

Development artifacts are retained to make the rejected paths visible:

- `baseline/`: the original runtime's seven loadable packs. Gemma 2 was not
  supported. The 300M single-item minimum reference cosine was about 0.848.
- `candidate-v3/`: intermediate diagnostic runs, including the first Gemma 2
  implementation. These are not final support or performance gates.
- `stability-v4.json`: intentionally retained failures. Packed QKV and shared
  short-batch reductions caused Qwen boundary drift; its initial raw absolute
  equality check also overconstrained different BF16 kernels.
- `stability-v5.json`, `stability-imperative.json`: Gemma 2 controls with
  compiled/imperative paths, including heterogeneous padded local windows.
- `dispatch-ab.json`: intermediate switch isolation. Gemma packing regressed
  B=8/S=256 by about 3%, so the final dispatch only packs B*S <= 256. Its Qwen
  implementation predates the short-input stability fix and must not be quoted
  as final Qwen throughput.
- `qwen-*-diagnosis.*`: packed, split, and upstream batch probes. The upstream
  batched 8B MXFP4 oracle also showed about 0.960 cosine at 129 tokens; preserving
  per-sequence reductions was necessary beyond matching that batch oracle.

## Measured results

Nine of nine packs passed both final direct and HTTP gates. Across the 19-length
direct sweep, the minimum single/reference cosine was 0.999562, batch/reference
0.999557, and batch/single 0.999779. Same-shape repeat and permutation results
were exact. These compare each quantized pack to its own reference implementation;
they do not measure quantization loss against an unquantized model.

Final AX median latency in milliseconds (B = batch size, S = tokens per row):

| Pack | B1 S16 | B8 S16 | B1 S256 | B8 S256 | B1 S1024 | B8 S128 |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 2 MXFP4 | 3.03 | 5.12 | 6.33 | 18.93 | 13.46 | 10.96 |
| Gemma 2 MXFP8 | 3.35 | 5.19 | 6.50 | 19.99 | 14.10 | 11.60 |
| Qwen 0.6B MXFP4 | 3.67 | 21.76 | 12.76 | 38.25 | 22.87 | 62.01 |
| Qwen 0.6B MXFP8 | 4.59 | 27.90 | 13.43 | 41.31 | 24.44 | 64.78 |
| Qwen 4B MXFP4 | 15.09 | 120.58 | 44.43 | 274.20 | 143.05 | 189.67 |
| Qwen 4B MXFP8 | 18.15 | 142.36 | 48.83 | 315.96 | 170.01 | 209.51 |
| Qwen 8B MXFP4 | 26.22 | 184.17 | 84.12 | 580.85 | 304.55 | 365.61 |
| Qwen 8B MXFP8 | 32.56 | 240.07 | 97.23 | 656.73 | 336.87 | 394.56 |
| Gemma 300M 8-bit | 2.58 | 4.39 | 4.72 | 14.26 | 15.81 | 9.36 |

Same-binary controlled speedup (disabled latency / enabled latency):

| Pack / optimization | B1 S16 | B8 S16 | B1 S256 | B8 S256 | B1 S1024 | B8 S128 |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 2 MXFP4 | 1.269x | 1.443x | 1.379x | 1.360x | 1.257x | 1.285x |
| Gemma 2 MXFP8 | 1.256x | 1.441x | 1.379x | 1.340x | 1.246x | 1.271x |
| Qwen 0.6B MXFP4 | 1.040x | 1.034x | 1.049x | 1.140x | 1.159x | 1.025x |
| Qwen 0.6B MXFP8 | 1.034x | 1.027x | 1.033x | 1.075x | 1.101x | 1.033x |

Gemma 2 compilation plus shape-limited packing improved these cases by
1.246–1.443x over the same new encoder with those options disabled. Qwen 0.6B
native-dtype SDPA improved B8/S256 and B1/S1024 by 1.075–1.159x against its
same-binary FP32-SDPA control. Small differences remain sensitive to run variance.

**Correctness has a measured cost.** Compared with the original runtime,
Qwen B8/S16 latency increased by roughly 3.3–5.0x because short sequences now
retain independent MX reduction geometry. B8/S128 increased by roughly
1.2–2.6x. The original runtime failed numerical stability checks, so it is not
an interchangeable faster configuration. Long-shape changes also vary by pack:
0.6B B8/S256 and B1/S1024 improved approximately 15–19%, while the 8B cases
were approximately 2–6% slower; B1/S256 regressed approximately 9–20% across
the Qwen packs. These cross-build measurements are diagnostics, not a universal
speedup claim. Further stable short-MX batching remains an optimization opportunity.

300M timing was approximately unchanged; its improvement is correct single-item
pooling and HTTP tokenization. The final short-text single/reference minimum was
0.999936, compared with about 0.848 before the pooling fix.

Software gates are recorded in `software-gates.json`: 4,368 Rust tests passed
(88 ignored), Python 329 passed / 7 skipped, formatting, Clippy, Ruff, mypy,
script checks and both Qwen dry-run qualification checks passed. The M5
embedding unit-test filter passed all 13 tests (`unit-embedding-final.log`).
Ignored/skipped cases are not hardware qualification passes.

## Reproduction

Run on Apple Silicon with the recorded package versions and cached snapshots.
Use an isolated checkout and serialized GPU runs. From the repository root:

```bash
maturin develop --profile release-pyext
cargo build --release -p ax-engine-core --bin generate-manifest
cargo build --release -p ax-engine-server
ARTIFACT=benchmarks/results/embedding/2026-10-25-axq-mx-review
mkdir -p /tmp/ax-embedding-review
python "$ARTIFACT/review.py" --pins "$ARTIFACT/hub-pins.json" \
  --output /tmp/ax-embedding-review/review
python "$ARTIFACT/stability.py" --pins "$ARTIFACT/hub-pins.json" \
  --output /tmp/ax-embedding-review/stability.json
python "$ARTIFACT/http_check.py" --pins "$ARTIFACT/hub-pins.json" \
  --output /tmp/ax-embedding-review/http.json
python "$ARTIFACT/ab.py" --pins "$ARTIFACT/hub-pins.json" \
  --only EmbeddingGemma-2 --flag gemma2_optimized \
  --output /tmp/ax-embedding-review/gemma-ab.json
python "$ARTIFACT/ab.py" --pins "$ARTIFACT/hub-pins.json" \
  --only 0.6B --output /tmp/ax-embedding-review/qwen-ab.json
```

`review.py` generates local manifest metadata in the cached snapshots; it does
not rewrite weights. Download the exact `repo_id`/`revision` pairs beforehand
with `huggingface_hub.snapshot_download` if they are not cached. `http_check.py`
starts and terminates only its own server children on temporary loopback ports.
Diagnostic `row_probe.py` accepts `--model-index 3 --lengths 31,32,33` (4B MXFP8)
or `--model-index 3 --lengths 31,31,31`; its default is 8B MXFP4 at 127/128/129.
Those probes expect `hub-pins.json` in the working directory and are retained
for numerical diagnosis, not timed performance comparisons.
