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
and both direct/MTP surface and sampled QA. It also requires the two raw
16-input/64-output-token greedy probes to match exactly; a route-dependent
greedy divergence on this probe fails the gate. This one probe is a
regression gate, not general numerical or quality certification. Missing,
skipped, partial, divergent, or fallback results fail. Results, including failures, are saved in
`qualification.json` (schema 2, with hashes of both probe requests and responses);
historical schema-1 passes do not satisfy this paired gate. Use a new output directory for every run. Runtime
`AX_`/`DYLD_` overrides are rejected for this product-default qualification.
A passing small QA sample does not establish advanced benchmark accuracy or
MTP Tier 2 certification.

The live path expects a clean worktree, `ax-engine doctor` ready, surface QA
for direct and MTP, and a short direct + MTP check against the last published
refresh. Full stack claims still use
`scripts/bench_mlx_inference_stack.py` with `mlx_lm.benchmark` as the primary
baseline. MTP suites are `flappy`, `long_code`, and `python_modules_long`.
Silent direct-fallback on the MTP path is a fail.

Campaign-only (does not block unrelated patches): MTP Tier 2 promotion, 8h/72h
endurance, long-context decode-at-depth, peer ranking, multi-model residency,
multimodal quality, 4/8-bit/MXFP4 A/B. 27B campaign runs belong on the
Mac mini M4 Pro 64 GB SKU. Qwen 3.8 Flash Next is incubating on Mac Studio M5 Ultra
256 GB (`python3 scripts/qualify_qwen38_flash_next.py --dry-run`). Convert may
map `qwen4_exp` metadata; load/serve stay fail-closed. Do not treat a laptop
campaign host as either SKU.

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
