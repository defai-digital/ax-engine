# Codex batch inspect — Waves 1-4

Current working tree audited without source edits, weight access, or cargo builds. Result: three P1 findings; no P0 findings.

## Per family

### qwen36-27b

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: Weight-backed parity remains unavailable for `qwen3_linear::layer_forward`; linear-MTP remains fail-closed through `MtpModelPolicy::from_loaded`.

### qwen36-35b-a3b

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: Quantized MoE routing/expert arithmetic was not numerically exercised; converter metadata and linear/full-attention dispatch are structurally consistent.

### qwen35-9b

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: No logit oracle for `qwen3_linear::layer_forward`; direct inference remains the default when MTP lacks certification.

### qwen3-coder-next

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: Gated-delta state evolution and MoE outputs remain weight-dependent; static converter, validation, and dispatch contracts align.

### qwen3-dense

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `standard::layer_forward` and Qwen RoPE/QK normalization were reviewed statically only.

### gemma4-12b-unified

- Status recommendation: parked
- Findings: **DI-W1-001 (P1)** — [`model_family_for_type`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/convert/model_family.rs:68) now emits `gemma4_unified`, but [`ModelConfig::from_manifest`](/Users/akiralam/code/ax-engine/crates/ax-engine-mlx/src/model/config.rs:629), [`build_layer_configs`](/Users/akiralam/code/ax-engine/crates/ax-engine-mlx/src/model/config.rs:908), and [`architecture::uses_geglu`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/architecture.rs:317) omit that label from Gemma semantics. Direct inference consequently selects generic query scaling, SwiGLU instead of GeGLU, and non-Gemma full/sliding RoPE geometry; the MoE router flag is also wrong for any unified MoE package.
- Residual LIMIT notes: Code path divergence is deterministic; weights are needed only to quantify output drift.

### gemma4-e-series-26-31

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `ModelConfig::from_manifest`, `build_layer_configs`, and `gemma4_vl::build_vl_prefill_embeddings` retain Gemma semantics for `gemma4`/`gemma4_vl`; tower and MoE parity remain unmeasured.

### glm47-flash

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `glm4_moe_lite::layer_forward` and `validate_mla_moe_manifest` align structurally; MLA/router numerics require weights.

### qwen3-vl

- Status recommendation: parked
- Findings: **DI-W2-001 (P1)** — [`model_family_for_type`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/convert/model_family.rs:100) maps `qwen3_vl_moe` tensors as MoE, but [`moe_config`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/convert/hf_config.rs:556) never classifies that model type as MoE. The generated `manifest.moe` is empty, and [`validate_native_model_manifest`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/model.rs:1962) rejects the mapped expert tensors.
- Residual LIMIT notes: Dense `qwen3_vl` has no static P0/P1; the finding is scoped to the supported MoE variant.

### minicpmv4_6

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `minicpm_v::build_vl_prefill_embeddings` and the Qwen3.5 hybrid text route align; vision projection and gated-delta parity remain unverified.

### nemotron-omni

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `nemotron_omni::build_omni_prefill_embeddings` feeds `nemotron_h::layer_forward` consistently; RADIO/audio and Mamba-2 numerics need weights.

### unlimited-ocr

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `unlimited_ocr::build_embeddings_with_image`, protected-prefix SWA, and standard MoE dispatch are structurally coherent; dual-vision/projector parity is unavailable.

### whisper-large-v3-turbo

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `WhisperModel::load`/`WhisperModel::transcribe` are correctly isolated from generic `MlxRunner`; encoder-decoder transcription parity remains untested.

### embeddings-primary

- Status recommendation: parked
- Findings: **DI-W2-002 (P1)** — [`MlxRunner::embed`](/Users/akiralam/code/ax-engine/crates/ax-engine-mlx/src/runner/mod.rs:3575) uses [`forward_for_embedding_body`](/Users/akiralam/code/ax-engine/crates/ax-engine-mlx/src/model/mod.rs:2759), which enables bidirectional attention only for `nemotron_embed` and uses the generic dense layer without Gemma sandwich norms. In contrast, [`forward_for_embedding_batch`](/Users/akiralam/code/ax-engine/crates/ax-engine-mlx/src/model/mod.rs:3405) dispatches EmbeddingGemma to the dedicated bidirectional Gemma3 path. Direct SDK/Python single-item embedding therefore differs architecturally from batch-of-one.
- Residual LIMIT notes: The ordinary OpenAI microbatch path uses the correct batch forward; direct `embed`/`embed_bytes` and fallback paths remain affected.

### nemotron-embed

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: Both singleton and batch embedding paths explicitly construct bidirectional masks for `nemotron_embed`; embedding-vector parity remains weight-dependent.

### llama3

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: Registry routing, Llama3 RoPE frequency construction, and `standard::layer_forward` align statically; no logit oracle was available.

### llama4-scout

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `llama4::layer_forward` consistently implements iRoPE, no-weight QK norm, temperature scaling, and interleaved MoE; numerical certification remains open.

### mistral-family

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `mistral3::layer_forward` delegates through uniform-SWA geometry correctly; classic/nested checkpoint parity requires weights.

### gpt-oss

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: Native MXFP4 blocks are sanitized by `load_gpt_oss_openai_mxfp4_split_experts` before `gpt_oss::layer_forward`; packed expert and attention-sink numerics remain untested.

### deepseek-v3

- Status recommendation: closed-code-only
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: `deepseek_v3::layer_forward`, MLA conversion, sigmoid/correction-bias routing, and dense/MoE layer selection align statically.

### diffusion-gemma

- Status recommendation: parked
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: Experimental LIMIT: `open_diffusion_block` → `advance_diffusion_workspace` → `commit_diffusion_workspace` is wired, but denoise convergence and generated-quality parity cannot be assessed without weights.

### deepseek-v4

- Status recommendation: parked
- Findings: none — no open P0/P1 from static audit
- Residual LIMIT notes: Experimental LIMIT: dedicated packed hyper-connection dispatch is present before the generic layer loop, but sparse compressor/indexer, hash routing, and output parity lack weight evidence. Nextn MTP defaults to direct fallback via `MtpModelPolicy::route_safe`.

## Cross-family issues (if any)

- DI-W1-001 exposes semantic drift between the central [`ARCHITECTURE_REGISTRY`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/architecture_registry.rs:85) and duplicated family-string allowlists in runtime configuration and structural classification. Registry presence alone does not preserve inherited architecture traits.
- DI-W2-001 is converter-classifier drift: accepted model-type aliases and tensor maps are not mechanically tied to `moe_config`.
- DI-W2-002 is singleton/batch dispatch drift: embedding implementations lack a parity-enforced family strategy boundary.
