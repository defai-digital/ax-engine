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
| Primary 27B qualification | Pinned `qwen3.8-27b:axq` on Mac mini M5, 64 GB | Minor/major, and patches that touch runtime, model, kernel, cache, scheduler, or serving |
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

## Primary qualification (Mac mini M5, 64 GB)

Print the contract (no weights):

```bash
python3 scripts/qualify_qwen38_27b.py --dry-run
```

Live run against the pinned snapshot:

```bash
python3 scripts/qualify_qwen38_27b.py \
  --model-dir /path/to/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/snapshots/3e290738e96972307c6aeb9934ab170ca0eae1c1
```

The live path expects a clean worktree, `ax-engine doctor` ready, surface QA
for direct and MTP, and a short direct + MTP check against the last published
refresh. Full stack claims still use
`scripts/bench_mlx_inference_stack.py` with `mlx_lm.benchmark` as the primary
baseline. MTP suites are `flappy`, `long_code`, and `python_modules_long`.
Silent direct-fallback on the MTP path is a fail.

Campaign-only (does not block unrelated patches): MTP Tier 2 promotion, 8h/72h
endurance, long-context decode-at-depth, peer ranking, multi-model residency,
multimodal quality, 4/8-bit/MXFP4 A/B. 27B campaign runs belong on the
Mac mini M5 64 GB SKU. Qwen 3.8 Flash Next is incubating on Mac Studio M5 Ultra
256 GB (`python3 scripts/qualify_qwen38_flash_next.py --dry-run`). Convert may
map `qwen4_exp` metadata; default load/serve stay fail-closed. Audited affine
exports have an explicit experimental development path documented in the
[Flash Next record](model-certifications/qwen3.8-flash-next.md). Do not treat a
campaign host as either SKU.

## Flash Next residency control (development only)

The ignored real-pack test writes four tokens, full F32 logit fingerprints,
serialized cache fingerprints, and memory/table-read counters. Use a private
copy of an audited pack; ordinary auto-conversion may create its manifest.
Run on an adequately sized Apple Silicon development host with MLX 0.32.2:

```bash
AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1 \
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
streaming value for the pack and host being tested. The 2-bit export also needs
`AX_ENGINE_2BIT_EXPERIMENTAL=1` in every process.

For selected-expert controls, keep `AX_STREAM_EXPERTS=on` and add
`AX_MLX_FLASH_NEXT_SELECTED_EXPERTS=1`, writing a separate result file. The
`selected_expert_payload_bytes` field must be positive; without the flag it
must be zero. This field counts successful row reads, including successful
earlier projections if a later projection fails.

Compare `generated_ids` and every `records` entry across modes of the same pack,
with the same prompt and prefill schedule. This test alone does not establish
checkpoint quality, long-context correctness, throughput, or SKU qualification.

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
