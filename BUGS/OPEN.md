# AX Engine — Open Bugs

BUG-104, BUG-103 (Rust), and BUG-102 (JS SDK) remain open.

---

## MEDIUM (2)

### BUG-104: Clippy `too_many_arguments` on Metal elementwise dispatch
**File:** `crates/ax-engine-metal/src/dispatch/elementwise.rs:1807`
`encode_softplus_bias_mul_sigmoid_pair_batch` takes 8 arguments, exceeding clippy limit of 7. Prevents `cargo clippy -- -D warnings`.

### BUG-103: Gemma4 test `test_global_layer_uses_backend_prefill_when_hd512_has_large_enough_batch` fails
**Files:** `crates/ax-engine-core/src/model/arch/gemma4/tests.rs:112-130`
`crates/ax-engine-core/src/model/arch/gemma4/forward.rs:77-82`
**Reproduction:** `cargo test -p ax-engine-core model::arch::gemma4::tests::test_global_layer_uses_backend_prefill_when_hd512_has_large_enough_batch`

Test asserts `spec.use_backend_prefill(0, 128) == true` for Global layer (head_dim=512) but impl always returns `false` (due to scale=1.0 requirement for Gemma4 QK norms; backend hardcodes 1/sqrt(head_dim)).

Mismatch between test expectation and current Gemma4Forward impl (see also fixed BUG-024/026). Causes test failure.

---

## Additional Issues
- Metal build warning: "Metal Tensor API shader not available (SDK lacks MetalPerformancePrimitives)" (crates/ax-engine-metal)

---

## LOW (1)

### BUG-102: JS SDK responseStream doesn't detect premature stream termination
**File:** `packages/ax-engine-js/index.cjs:513-576`
If the SSE stream ends without emitting `finish_reason`, the consumer receives partial deltas with no way to detect incompleteness.

---

## Fixed Bugs (complete list)

| Bug | Severity | Fix |
|---|---|---|
| BUG-024 | HIGH | Gemma4 GPU batch RoPE — use `scaled_start_step()` for global layers |
| BUG-025 | HIGH | Gemma4 GPU decode post-FFN norm |
| BUG-026 | HIGH | Gemma4 CPU batch attention — disable `use_backend_prefill` (scale=1.0) |
| BUG-027 | HIGH | Qwen3.5 restore_snapshot GPU recurrent (resolved in refactor) |
| BUG-028 | MEDIUM | Chunked prefill fallback — slice to remaining tokens |
| BUG-029 | MEDIUM | Sampling sort — added sort to `sample_filtered_logits_with_scratch` |
| BUG-030/031 | MEDIUM | GGUF tensor checked_mul overflow |
| BUG-032 | MEDIUM | tokenizer add_space_prefix default → `unwrap_or(true)` |
| BUG-033 | MEDIUM | truncate_attention_to — sync recurrent seqlen_offset |
| BUG-034 | LOW | GPT-2 word splitting — split on newline and tab chars |
| BUG-037 | LOW | CpuKv doc comment (resolved in refactor) |
| BUG-038/039/048/049 | LOW | Arena checked_add + gpt2_byte_table OnceLock |
| BUG-041 | HIGH | GGUF metadata loop capped to 4096 |
| BUG-043 | MEDIUM | Speculative clamping (was already fixed) |
| BUG-046 | MEDIUM | Gemma4 CPU decode — pre-allocate KV padding buffers |
| BUG-050 | HIGH | gpu_recurrent_buffers — bounds check via `.get()` |
| BUG-051 | MEDIUM | restore() pristine_zero flags → fill(false) |
| BUG-052 | MEDIUM | Qwen prefill plan — added quant blocker check |
| BUG-053/088 | MEDIUM | Qwen3MoE forward_batch_profiled — consult PrefillExecutionPlan |
| BUG-055 | LOW | CLI flush_prefix — proportional token count reduction |
| BUG-077 | HIGH | forward_batch — documented partial-advance contract + warn logging |
| BUG-084 | MEDIUM | Config — sliding_window_size <= context_length check |
| BUG-087 | LOW | apply_bias_to_batch — debug_assert → assert |
| BUG-097 | CRITICAL | Q8 KV append block offset (graph-IR) |
| BUG-098 | CRITICAL | Q8 KV append block offset (Gemma3) |
| BUG-099 | HIGH | sync_gpu_attention stale data guard |
| BUG-100 | MEDIUM | Gemma3 fused QKV Q8 guard |
| BUG-101 | MEDIUM | Qwen3 MoE test assertions |
| Metal MoE multitoken | HIGH | Nested execute_sync → on-encoder precompute |
