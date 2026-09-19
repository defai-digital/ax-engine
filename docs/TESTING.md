# Testing Design

AX Engine tests in four layers. The **primary optimization target** is Qwen 3.8
27B AXQ 6-bit MTP (`qwen3.8-27b:axq`). That pack does not belong in CI.

Primary optimization target. Checkpoint Tier 1. MTP Tier 2 pending. AX certification record: Candidate (gates open).

**Related:** [Contributing](../CONTRIBUTING.md) ·
[Qwen 3.8 27B AXQ certification](model-certifications/qwen3.8-27b-axq.md) ·
[Benchmarks](BENCHMARKS.md) · [Releasing](RELEASING.md)

## Layers

| Layer | What it proves | When it blocks a release |
| --- | --- | --- |
| Always-on | Format, unit tests, Clippy, script gates, alias/revision pin, public-claim contract | Every release |
| Small-pack CI smoke | Tiny Certified/Compatible checkpoints load and generate | Every release that runs model-smoke |
| Primary 27B qualification | Pinned `qwen3.8-27b:axq` on Mac mini M4 Pro, 64 GB | Minor/major, and patches that touch runtime, model, kernel, cache, scheduler, or serving |
| Secondary family regression | One representative checkpoint per other Certified family | Minor/major, and patches that touch that family |

## Always-on

```bash
cargo fmt --check
cargo test --quiet --no-fail-fast
cargo clippy --all-targets --all-features -- -D warnings
pytest python/tests
bash scripts/check-scripts.sh
python3 scripts/smoke_compatible_models.py --dry-run
python3 scripts/check_qwen38_primary_claims.py
python3 scripts/qualify_qwen38_27b.py --dry-run
```

`check_qwen38_primary_claims.py` fails if README/docs drop the canonical status
sentence, retarget the pinned revision, claim MTP Tier 2 certified for this
pack, or publish internal SSH host aliases (historical `benchmarks/` paths are
allowed).

## Small-pack CI smoke

```bash
python3 scripts/smoke_compatible_models.py --list
python3 scripts/smoke_compatible_models.py --dry-run
python3 scripts/smoke_compatible_models.py --models qwen3-0.6b,qwen3-4b,gemma4-e4b,llama3.2-1b,ministral-8b
```

These packs are architecture proxies. They do not qualify Qwen 3.8 27B.
Do not add a 27B cell here.

Live-model QA when artifacts are mounted uses `scripts/check-qa-model.sh`
(default `QA_MODEL_ID=qwen3_5_9b_q4`). That is a Qwen 3.5 hybrid proxy, not
the hero pack.

## Primary qualification (Mac mini M4 Pro, 64 GB)

Print the contract (no weights):

```bash
python3 scripts/qualify_qwen38_27b.py --dry-run
```

Preflight against the pinned snapshot (does not qualify the product):

```bash
python3 scripts/qualify_qwen38_27b.py \
  --model-dir /path/to/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/snapshots/3e290738e96972307c6aeb9934ab170ca0eae1c1
```

Execute the gates with an isolated installed release wheel:

```bash
python3 scripts/qualify_qwen38_27b.py --run \
  --model-dir "$MODEL_DIR" --output "$NEW_RESULT_DIR" \
  --build-manifest "$BUILD_MANIFEST" --wheel "$WHEEL" \
  --server-bin "$WHEEL_SERVER" --bench-bin "$WHEEL_BENCH" --cli "$WHEEL_CLI"
```

The build manifest records `source_commit`, `dirty: false`, `server_sha256`,
`bench_sha256`, `wheel_sha256`, `cli_sha256`, `model_revision`, and `model_files` (relative
filename to SHA-256). Hash executables after wheel installation. The gate
checks the exact SKU, clean checkout, build and model hashes, doctor readiness,
and default/direct/explicit-MTP surface and sampled QA. The default launch uses
no acceleration overrides and must neither request nor activate MTP. The MTP
cell explicitly selects `--mlx-mtp-policy required`; publisher speed metadata
does not authorize linear-Qwen default promotion. The two raw 16-input/64-output-token
greedy probes must be complete. Their cross-route token identity is diagnostic
only: differences are disclosed without failing product-health qualification.
Missing, skipped, partial, failed-QA or fallback results still fail. Results,
including failures, are saved in `qualification.json` (schema 4, with hashes of
the default and both paired probe requests and responses). Schema 3 did not
exercise the actual default launch. `paired_greedy` records `matched`,
`first_divergence`, `divergence_count` and every differing position with both
token IDs. It explicitly sets `release_blocking: false`; unavailable logit
margins are `null`, not an inferred zero. Historical schema-1/2 evidence keeps
its original meaning and is not reclassified. Use a new output directory for every run. Runtime
`AX_`/`DYLD_` overrides are rejected for this product-default qualification.
A passing small QA sample does not establish advanced benchmark accuracy or
MTP certification. The report marks MTP-S, MTP-P and MTP-D as `not_assessed`:
cross-route identity proves neither same-state verifier safety nor speed nor
default-promotion readiness. Separate MTP-S evidence is still required for
shipping MTP. See the [three-gate requirements](model-certifications/qwen3.8-27b-axq.md#what-mtp-tier-2-pending-means).

The [2026-09-18 product qualification](../benchmarks/results/qualification/2026-09-18-qwen38-27b-product-default/)
extends that sample with all 79 QA items in both streaming forms on the actual
default and explicit MTP routes, a uniform 1024-token cap, and completion-aware
grading. Natural `stop` is required; an answer containing the expected text
but truncated by its output budget fails. Its installed-CLI lifecycle collector
also requires measured backpressure, quiescence and unchanged recovery output.
The original 256-token failures and pressure workload without observed
backpressure remain failed records. Reproduction commands and an offline
evidence/grade verifier are included with the artifacts.

The [v7.4.0 release qualification](../benchmarks/results/qualification/2026-09-18-qwen38-27b-release/)
repeats this protocol against the exact-source hosted wheel and binds its digest
to the public PyPI artifact. Both routes again pass 158/158 hard checks, with
zero incomplete responses and six soft keyword failures each. It also verifies
fresh installation on the SKU and all three public signed standalone launch
controls (default alias, explicit MTP alias, and local directory without Python).
This closes the recorded product-health and publication scope without promoting
MTP-P or MTP-D or claiming broad accuracy or 8h/72h endurance.

The live path expects a clean worktree, `ax-engine doctor` ready, surface QA
for default, direct and explicit MTP, and a short direct + MTP check against the last published
refresh. Full stack claims still use
`scripts/bench_mlx_inference_stack.py` with `mlx_lm.benchmark` as the primary
baseline. MTP suites are `flappy`, `long_code`, and `python_modules_long`.
Silent direct-fallback on the MTP path is a fail.

Campaign-only (does not block unrelated patches): MTP-P performance certification,
MTP-D default promotion, 8h/72h
endurance, long-context decode-at-depth, peer ranking, multi-model residency,
multimodal quality, 4/8-bit/MXFP4 A/B. 27B campaign runs belong on the
Mac mini M4 Pro 64 GB SKU. Qwen 3.8 Flash Next MXFP4 MTP is a
second SKU on MacBook Pro M5 Max 128 GB
(`python3 scripts/qualify_qwen38_flash_next.py --dry-run`).
Second SKU. MXFP4 MTP target; native support and checkpoint qualification pending. MTP Tier 2 pending. AX certification record: Candidate (gates open). Existing `qwen3.8-flash-next:axq` selects affine 4-bit; MXFP4 has no CLI alias yet.
Audited affine 4-bit/group64 and 6-bit/group64 packs load with no environment
variable. The Flash Next development comparison uses a pinned MLX-VLM
reference because `mlx_lm` has no `qwen4_exp` model. Passing `--skip-mlx-lm`
to the general benchmark only skips its baseline; it does not install or run
that separate reference. These records never claim an `mlx_lm` ratio.
See the [Flash Next record](model-certifications/qwen3.8-flash-next.md).
Do not treat a campaign host as either SKU.

## Flash Next residency control (development only)

The ignored real-pack test writes four tokens, full F32 logit fingerprints,
serialized cache fingerprints, and memory/table-read counters. Use a private
copy of an audited pack; ordinary auto-conversion may create its manifest.
Run on an adequately sized Apple Silicon development host with MLX 0.32.2:

```bash
AX_STREAM_EXPERTS=on AX_STREAM_EXPERT_LAYERS=1 \
AX_FLASH_NEXT_EXPECT_STREAMING=1 \
AX_FLASH_NEXT_CANDIDATE_PACK_DIR=/path/to/private-flash-next-6bit \
AX_FLASH_NEXT_PROMPT_IDS='[760,6511,314,9338,369]' \
AX_FLASH_NEXT_SMOKE_OUTPUT=/tmp/flash-next-6bit-on.json \
cargo test -p ax-engine-mlx --profile release-server \
  model::qwen4_exp_integration_tests::qwen4_exp_real_pack_residency_fingerprint \
  -- --ignored --exact --nocapture
```

For a resident control set `AX_STREAM_EXPERTS=off` and
`AX_FLASH_NEXT_EXPECT_STREAMING=0`, using a different output filename. Auto uses
the existing full-resident estimate plus 48 GiB admission rule; set the expected
streaming value for the pack and host being tested. Audited 4-bit/group64 and
6-bit/group64 packs need no family opt-in. The 2-bit export still needs
`AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1` and `AX_ENGINE_2BIT_EXPERIMENTAL=1` in
every process.

For selected-expert controls, keep `AX_STREAM_EXPERTS=on` and add
`AX_MLX_FLASH_NEXT_SELECTED_EXPERTS=1`, writing a separate result file. The
`selected_expert_payload_bytes` field must be positive; without the flag it
must be zero. This field counts successful row reads, including successful
earlier projections if a later projection fails.

The fingerprint harness accepts 2-512 prompt tokens; broader context validation
uses the dedicated qualification and serving campaigns.

For bounded multi-token Shared prefill, additionally set
`AX_MLX_FLASH_NEXT_SELECTED_PREFILL=1`. Both selected flags are required. The
`selected_expert_payload_bytes_after_prefill` field is positive when a layer's
expert union fits the selected payload cap. With a multi-token test prefix, it
remains zero for the singleton-only control. Compare `prefix_record` as well as generated records to cover logits
and serialized state immediately after the same multi-token prefill. Capacity
misses alone retain whole-layer reads; malformed metadata and I/O errors fail.

Compare `generated_ids` and every `records` entry across modes of the same pack,
with the same prompt and prefill schedule. This test alone does not establish
checkpoint quality, long-context correctness, throughput, or SKU qualification.

When a greedy continuation differs, the ignored fingerprint test accepts
`AX_FLASH_NEXT_TEACHER_FORCE_IDS='[11751,13,271]'` to fix the three subsequent
singleton inputs. It validates the token count and vocabulary bounds and
records the forced inputs separately from the predicted IDs. Compare logits
only when the complete input history and prefill schedule match.

For independent numerical comparisons, set `AX_FLASH_NEXT_LOGITS_DIR` to a
fresh output directory. The fingerprint test writes `prefix.f32le` and
`decode-0.f32le` through `decode-3.f32le`: contiguous little-endian float32
pre-softmax logits, with shapes recorded in the corresponding JSON fingerprint.
These are diagnostic files and must not be added as model artifacts.

The test binary also accepts `AX_FLASH_NEXT_FIRST_LAYER_DUMP` for synchronized
embedding and first-layer stage dumps. Each `.f32le` file has a JSON shape and
original dtype record. This adds evaluation barriers; verify that final logits
and state still match the uninstrumented control. It is not a timing baseline.

With that explicit dump root set, `AX_FLASH_NEXT_DUMP_LAYER` selects a zero-based
layer instead of the default 0. Use a separate directory for each layer. The
checkpoint's PLE layer IDs are one-based: ID 2 corresponds to dump layer 1.
HC stages include occurrence 1 (attention) or 2 (MLP) to distinguish their
activation, injection and stream-mean boundaries. PLE stages include gate,
score, key/query/value and convolution window/weights/pre-activation/output.
These controls exist only in the test binary.

The first-layer dump also records GDN raw Q/K/V, log decay, beta, incoming and
outgoing recurrent state, FP32 recurrence output, norm gain, gate and gated
output. State uses AX's `[batch, value_heads, value_dim, key_dim]` layout; an
official Transformers comparison must transpose its final two axes. Preserve
the original dtype metadata when restoring captured BF16 values.

MoE stage dumps include router logits, original expert IDs and weights, expert
and shared gate/up activations, down projections, weighted outputs and the
complete delta. Compare activation on identical captured projections; separately
record tied top-k selections and expert accumulation order. Conditioning an
oracle on captured routing/projections does not establish full-model agreement.

The ignored `qwen4_exp_mtp_candidate_keeps_primary_tokens_and_state_exact` test
also accepts `AX_FLASH_NEXT_REAL_PACK`, `AX_FLASH_NEXT_PROMPT_IDS` (3-16 tokens),
and `AX_FLASH_NEXT_RESULT_PATH`. This real-pack mode requires both selected
expert flags, forces paging, and checks primary and draft state against a
same-schedule direct/full-head control. Its MTP-only selected payload counter
excludes reference forwards. Audited 4-bit and 6-bit packs need no family
opt-in; 2-bit needs both `AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1` and
`AX_ENGINE_2BIT_EXPERIMENTAL=1`. Without the real-pack
variable, the existing synthetic MTP oracle mode is unchanged.

## Secondary families

Keep Qwen 3.6 27B AXQ, Qwen 3.6 35B-A3B, Gemma 4 assistant-MTP, and GLM 4.7
Flash as correctness cells. Their published numbers stay attached to those
packs. See [QA](../qa/README.md) and [QA matrix catalog](../qa/matrix-catalog.md).

## Do not

- Put 27B weights in CI
- Treat a 9B or 4B smoke as 27B qualification
- Promote dirty-worktree benches to public defaults
- Claim MTP Tier 2, endurance, multi-model add, or P0 multimodal for Qwen 3.8
  27B until the matching artifacts exist
- Use Super-class Qwen 3.8 (2.4T) as the primary pack
