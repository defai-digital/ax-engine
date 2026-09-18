# Qwen 3.8 27B: reproduce the server's prefix-prefill geometry

The standalone direct-history oracle's first disagreement at output index 117
was caused by omitting the production runner's prefix-cache prefill split.
For this cold 409-token prompt, block size 16 produces a cache-only head of
400 tokens followed by normal prefill of the remaining 9. Using the same
normal prefill entry without that preceding head was insufficient.

This fixes a diagnostic reproduction defect. Production arithmetic, cache
policy, sampling, reference answers and numerical tolerances are unchanged.
It does not establish release readiness, a quality gain, or MTP promotion.

## Controlled evidence

All model runs used the selected Apple M4 Pro 64 GB SKU, the same 22 pinned
pack files and bundled native libraries. Pack revision:
`3e290738e96972307c6aeb9934ab170ca0eae1c1`. Prompt, sampling and original
192-token expectation come from the preceding
[first-split case](../2026-09-18-qwen38-27b-first-split-replay/case.json).
Indices are zero-based.

| Cold control | Result |
| --- | --- |
| Unsplit standalone; dedicated GPU stream; full runner initialization/warmup; unsplit repeat | All reject output 117: expected 40278, predicted 2849; all seven cache witnesses match one another |
| Head400/tail9; full runner plus head400/tail9; head400/tail9 repeat | All 192 IDs match the actual default server; all seven complete cache witnesses match the server |
| Unsplit baseline before/after split controls | Original output-117 rejection reproduced |
| Clean installed server with only prefix-cache budget set to zero | First difference at 117, token2849; unsplit probe matches its entire 192-token output |
| Clean installed server default reversal | Original 192-token output restored |
| Final observation-free CLI, explicit block16, two cold runs | All 192 default-server IDs match |
| Final CLI without split, using cache-off server expectation | All 192 IDs match |
| Final CLI without split, using default-server expectation | Original rejection retained |
| Final CLI block size zero | Rejected before model loading |

The cache witnesses cover boundaries 409,410,411,522,524,525,526. Each contains
all 128 logical arrays (48 convolution,48 recurrent,16 key,16 value), finite
checks, shape/dtype, SHA-256 of Float32 little-endian value bits and cache
metadata. The split probe matches all of them exactly. The unsplit probe is
already different at boundary409, immediately after prefill: all128 logical
array hashes differ, before any decode-history API choice.

Observation was admitted only after clean/observer/clean server arms preserved
all192 outputs and the observer's states522/524/525 matched the prior
[actual-direct witnesses](../2026-09-18-qwen38-27b-direct-history/events.json).
Lazy cache clones are retained and read only after the existing output117
materialization. This bounds observation perturbation; it does not establish
identity at unobserved boundaries or across all prompts. The initial observer
panicked on strided Float32 data access; those runs are explicitly excluded.

## Repair and regression

The diagnostic now accepts `--prefix-cache-block-size=N`, reuses the runner's
existing boundary policy, performs its cache-only head with the same layout
and barrier, retains the snapshot, then prefills the tail. Omitting the flag
preserves the old standalone layout. This is a cold single-request diagnostic;
it does not simulate warm prefix hits, arbitrary scheduler chunking or every
runtime prefill configuration. The helper's production body is unchanged.

The initial observer also exposed a real bug in the committed comparison
helper: a Float32 cast can preserve a strided view, while raw data reads require
row-contiguous storage. A transposed real MLX array reproduces the panic before
the repair and passes afterward. Comparisons now materialize contiguous logical
values and reject unequal/empty extents, NaN/Inf, difference overflow and
asymmetric missing state. Both absent linear states remain valid for
full-attention layers. Seven focused regression tests pass.

The final probe source base is `c43bd901770f13f179b8102a83760a1c6f0f99b8` plus
`final.patch`; its release-server binary SHA-256 is
`4d4c40a5b9ffbd70be82b329ed45538c61bd4bc7de6df38ee16b3bcac1abed4a`.
Separate observation patches and receipts bind the exploratory builds. They
are evidence only and are not installed production instrumentation. The
unchanged clean server is from the previously qualified `22affab5` wheel;
no new wheel release or installation qualification is claimed here.

## Validation and limits

Pinned Rust1.97.1: focused regressions, formatting, full workspace tests and
CI-policy Clippy pass. `maturin develop`, Python tests (209 passed,26 skipped,
140 subtests), scripts, both qualification dry-runs and primary-claim checks
pass. **Strict Clippy still exits101 on existing workspace unwrap/expect
warnings**; the existing CI warning exceptions are recorded separately and
are not a strict-Clippy pass. `validation.json` binds commands, exit codes,
source-before/after hashes and logs. Dry-run success is not hardware qualification.

DeepSeek V4 Pro and MiniMax M2.7 completed bounded source/evidence reviews via
AX Code. [Disposition and reference comparison](reference-review.md) record
which claims were accepted or rejected. No reference code was extracted.

Existing direct/MTP differences at116/155, aligned broad quality evaluation,
performance certification and default promotion are not resolved by this
single-case diagnostic. Historical failures and expectations are retained.
The pack remains Candidate; this evidence does not support a ship-ready claim.

## Reproduction and artifact layout

Build with the pinned toolchain:

```sh
rustup run 1.97.1 cargo build --locked --profile release-server \
  -p ax-engine-mlx --bin linear_mtp_state_oracle_probe
python3 -B benchmarks/results/qualification/2026-09-18-qwen38-27b-prefix-prefill/verify_evidence.py
```

On the target SKU, provide the pinned pack and native libraries, and pass the
case's comma-separated `prompt` and `baseline_direct_output` to the probe:

```sh
linear_mtp_state_oracle_probe "$MODEL_DIR" "$PROMPT_IDS" "$EXPECTED_IDS" \
  --validate-history=direct-pipeline --prefix-cache-block-size=16
```

Repeat without the block-size option to reproduce the negative control. The
cache-off server control uses `AX_MLX_PREFIX_CACHE_MAX_BYTES=0`, otherwise the
same request and installed binary. This is a diagnostic setting, not a
recommended deployment change.

`live-controls.json` binds exact HTTP request/response bodies and owned-process
cleanup. `observations.json` retains per-arm exit status, raw-log hash and
extracted diagnostic lines; identical cache sets are deduplicated under
`states/`. Raw logs with private paths stay local. Final CLI logs are sanitized,
with raw and published hashes recorded separately. Long validation logs retain
first/last excerpts plus raw hashes. `publication.json` hashes the public
artifacts; the offline verifier checks token/state controls and receipt links.
