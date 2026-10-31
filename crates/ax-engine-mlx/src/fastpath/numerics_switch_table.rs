// Generated classification of every quoted AX_* name in fastpath.rs.
// A new env literal fails `numerics_switch_table_covers_every_fastpath_env`
// until it is added here and, when it changes numerics, to the suffix.

#[derive(Clone, Copy, Debug)]
enum SwitchClass {
    Token(&'static str),
    ProvenExact(&'static str),
    NonNumeric(&'static str),
}

struct SwitchRow {
    env: &'static str,
    shipped: &'static str,
    class: SwitchClass,
}

const NUMERICS_SWITCH_TABLE: &[SwitchRow] = &[
    SwitchRow { env: "AX_MLX_MTP_WARMUP_CAP", shipped: "unset", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_MTP_FIXED_DRAFT_DEPTH", shipped: "unset", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_MTP_CONSERVATIVE_DEPTH", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_MTP_DEPTH3_HYSTERESIS", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_MTP_DEPTH3_MISS_BACKOFF", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_FLASH_NEXT_NGRAM_READ_THREADS", shipped: "unset", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_FLASH_NEXT_NGRAM_PREFETCH", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_FLASH_NEXT_KV_SEGMENTS", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_GDN_METAL", shipped: "on", class: SwitchClass::ProvenExact("native_gdn_forward_matches_compiled_and_eager") },
    SwitchRow { env: "AX_FLASH_NEXT_ROPE_CACHE", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_FLASH_NEXT_POOLED_CACHE", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_FLASH_NEXT_NGRAM_WARM_AUX", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_ACTIVATION_DTYPE", shipped: "auto", class: SwitchClass::Token("act=f16") },
    SwitchRow { env: "AX_NO_SPEC", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY", shipped: "off", class: SwitchClass::NonNumeric("telemetry") },
    SwitchRow { env: "AX_MLX_DECODE_SAMPLING_GPU_TOPK", shipped: "on", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_THINK_SOFT_CLOSE", shipped: "on", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_MLX_BATCHED_PREFILL", shipped: "off", class: SwitchClass::Token("batched_prefill") },
    SwitchRow { env: "AX_MLX_BATCHED_PREFILL_TOKENS", shipped: "unset", class: SwitchClass::Token("batched_tokens=0") },
    SwitchRow { env: "AX_MLX_MTP_VERIFY_SUBMIT_LAYERS", shipped: "0", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_BATCHED_PREFILL_ROWS", shipped: "8", class: SwitchClass::Token("batched_rows=4") },
    SwitchRow { env: "AX_MLX_LOAD_KERNEL_WARMUP", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASSISTANT_MTP_COALESCED_VERIFY", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASSISTANT_MTP_SEQUENTIAL_ORACLE", shipped: "on", class: SwitchClass::Token("gemma_seq_oracle=0") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD", shipped: "on", class: SwitchClass::Token("gemma_cycle_guard=0") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASSISTANT_MTP_SENSITIVE_F32", shipped: "on", class: SwitchClass::Token("gemma_sensitive_f32=0") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASSISTANT_MTP_EARLY_GEN_PURE_DIRECT", shipped: "off", class: SwitchClass::Token("gemma_early_direct") },
    SwitchRow { env: "AX_MLX_DECODE_MTP_TARGET_PROB_WORKSPACE", shipped: "on", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_PREFILL_FFN_COMPILE_SWIGLU", shipped: "on", class: SwitchClass::ProvenExact("swiglu_compiled_matches_imperative") },
    SwitchRow { env: "AX_MLX_PACK_QKV_PROJECTIONS", shipped: "on", class: SwitchClass::Token("pack_qkv=0") },
    SwitchRow { env: "AX_MLX_PACK_DENSE_FFN_GATE_UP", shipped: "on", class: SwitchClass::Token("pack_gate_up=0") },
    SwitchRow { env: "AX_MLX_GEGLU_MUL_METAL", shipped: "on", class: SwitchClass::Token("geglu_mul=0") },
    SwitchRow { env: "AX_MLX_DENSE_GEGLU_PACKED_METAL", shipped: "on", class: SwitchClass::ProvenExact("packed_geglu_metal_matches_direct_geglu_for_bf16_packed_gate_up") },
    SwitchRow { env: "AX_MLX_DENSE_SWIGLU_PACKED_METAL", shipped: "on", class: SwitchClass::Token("dense_swiglu_packed=0") },
    SwitchRow { env: "AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL", shipped: "on", class: SwitchClass::ProvenExact("qwen_dense_ffn_gate_up_swiglu_preserves_bfloat16_intermediates") },
    SwitchRow { env: "AX_MLX_QWEN_LINEAR_MTP_EXACT", shipped: "unset", class: SwitchClass::Token("qwen_exact=0") },
    SwitchRow { env: "AX_MLX_MTP_TARGET_LAYER_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_MTP_PACKED_VERIFY_FFN", shipped: "off", class: SwitchClass::Token("mtp_packed_verify") },
    SwitchRow { env: "AX_MLX_MTP_WHOLE_VERIFY_COMPILE", shipped: "off", class: SwitchClass::Token("mtp_whole_verify") },
    SwitchRow { env: "AX_MLX_MTP_PROFITABILITY_THROUGHPUT", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_MTP_COST_MODEL_DEPTH", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_STICKY_FALLBACK", shipped: "off", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_MLX_MTP_NATIVE_GREEDY_VERIFY_LOGITS", shipped: "off", class: SwitchClass::Token("native_greedy_logits") },
    SwitchRow { env: "AX_MLX_MTP_LINEAR_LAYER_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_MTP_ASYNC_DUAL_GATE_UP", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_MTP_LA_OUT_PROJ_SILU_MUL_QMM", shipped: "on", class: SwitchClass::Token("mtp_la_out_fuse=0") },
    SwitchRow { env: "AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP_DEPTH", shipped: "3", class: SwitchClass::Token("mtp_depth=4") },
    SwitchRow { env: "AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP", shipped: "on", class: SwitchClass::Token("throughput_mtp=0") },
    SwitchRow { env: "AX_MLX_MTP_LINEAR_PROJECTED_REPLAY", shipped: "off", class: SwitchClass::Token("mtp_projected_replay") },
    SwitchRow { env: "AX_MLX_MTP_RELAXED_TARGET_VERIFY", shipped: "on", class: SwitchClass::Token("relaxed_mtp=0") },
    SwitchRow { env: "AX_MLX_MTP_SPLIT_VERIFY_HIDDEN_EVAL", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_MTP_LAZY_ADOPT_STATE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_MTP_DEVICE_GREEDY_ACCEPT", shipped: "off", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_MLX_MTP_REBIND_VERIFY_FA", shipped: "on", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_MTP_LINEAR_TAPE_CAPTURE", shipped: "off", class: SwitchClass::Token("mtp_tape") },
    SwitchRow { env: "AX_MLX_MTP_SKIP_PREFIX_CHECKPOINT", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_MTP_REUSE_PROCESSED_GDN", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_MTP_GDN_PREWORK_SIMD32", shipped: "off", class: SwitchClass::Token("mtp_gdn_prework") },
    SwitchRow { env: "AX_MLX_MTP_FUSED_GDN_VERIFY", shipped: "off", class: SwitchClass::Token("mtp_fused_gdn") },
    SwitchRow { env: "AX_MLX_MTP_REFOLD_ACCEPTED_HISTORY", shipped: "off", class: SwitchClass::Token("mtp_refold") },
    SwitchRow { env: "AX_MLX_MTP_BATCHED_COMMITTED_FOLD", shipped: "off", class: SwitchClass::Token("mtp_committed_fold") },
    SwitchRow { env: "AX_MLX_MTP_LAST_COMMITTED_QUERY", shipped: "off", class: SwitchClass::Token("mtp_last_query") },
    SwitchRow { env: "AX_MLX_INVARIANT_MXFP4_QMV_FAST", shipped: "off", class: SwitchClass::Token("invariant_mxfp4") },
    SwitchRow { env: "AX_MLX_QWEN_DENSE_FFN_MATVEC_EXT_BITS", shipped: "off", class: SwitchClass::Token("matvec_ext") },
    SwitchRow { env: "AX_MLX_EXACT_LM_HEAD_WINDOW_INVARIANT", shipped: "on", class: SwitchClass::ProvenExact("invariant_projection_window_matches_row_by_row_bit_exact") },
    SwitchRow { env: "AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV", shipped: "on", class: SwitchClass::Token("dense_gemv=0") },
    SwitchRow { env: "AX_MLX_INVARIANT_QMV_BF16_Q4", shipped: "on", class: SwitchClass::ProvenExact("invariant_affine_bf16_q4_kernel_matches_generic_bit_exact") },
    SwitchRow { env: "AX_MLX_MTP_DENSE_HEAD_DRAFT_Q4", shipped: "on", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_DFLASH2_DIR", shipped: "unset", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_DFLASH2", shipped: "on", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_UNLIMITED_OCR_PAGE_GUARD", shipped: "on", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_MLX_DFLASH2_WINDOW", shipped: "unset", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_DFLASH2_BF16", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_TREE_VERIFY", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_EXPERT_STREAM_DISCOUNT_NGRAM", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_MTP_QUANTIZED_HEAD_DRAFT", shipped: "on", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_MTP_PENALIZED_VERIFY", shipped: "on", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_MLX_QWEN4_EXP_VISION", shipped: "off", class: SwitchClass::Token("qwen4_vision") },
    SwitchRow { env: "AX_MLX_LINEAR_PREFIX_GRID_RUNG", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_MTP_DENSE_HEAD_DRAFT_BITS", shipped: "3", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_MTP_FORCE_REQUESTED", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_DENSE_WIDE_GEMV", shipped: "off", class: SwitchClass::Token("wide_gemv") },
    SwitchRow { env: "AX_MLX_EXACT_RMS_GATE_METAL", shipped: "off", class: SwitchClass::Token("rms_gate_metal") },
    SwitchRow { env: "AX_MLX_MTP_ASYNC_DRAFT", shipped: "off", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MLX_GEMMA_DUAL_GATE_UP_METAL", shipped: "off", class: SwitchClass::Token("gemma_dual_gate_up") },
    SwitchRow { env: "AX_MLX_LAYER_SCALAR_FUSED_ADD", shipped: "on", class: SwitchClass::Token("layer_scalar=0") },
    SwitchRow { env: "AX_MLX_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_FUSED_PREFILL_ATTENTION", shipped: "off", class: SwitchClass::Token("fused_prefill") },
    SwitchRow { env: "AX_MLX_QWEN_FUSED_PREFILL_ATTENTION", shipped: "off", class: SwitchClass::Token("qwen_fused_prefill") },
    SwitchRow { env: "AX_MLX_QWEN_LINEAR_ADD_RMS_NORM", shipped: "off", class: SwitchClass::Token("qwen_add_rms") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_INTERLAYER_ADD_RMS", shipped: "off", class: SwitchClass::Token("qwen_inter_add_rms") },
    SwitchRow { env: "AX_MLX_GEMMA4_FUSED_PREFILL_ATTENTION_P128", shipped: "on", class: SwitchClass::Token("gemma4_fused_p128=0") },
    SwitchRow { env: "AX_MLX_QWEN_SKIP_LINEAR_PREFILL_MASK", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_EVAL_KV_ONLY", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_EXACT_SIZE_FIRST_KV", shipped: "on", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_EXACT_SIZE_KV_GROW", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_SKIP_UNUSED_FULL_KV_VIEW_SLICE", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_SKIP_UNUSED_LA_OUT_RESHAPE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_LA_REUSE_INITIAL_STATE_ZEROS", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK", shipped: "off", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_PREFILL_TIME_DEBUG", shipped: "off", class: SwitchClass::NonNumeric("telemetry") },
    SwitchRow { env: "AX_MLX_SIBLING_PREFILL_ROTATION", shipped: "unset", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_ROTATING_SLIDING_DECODE", shipped: "on", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_ROTATING_SLIDING_PREFILL", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_ROTATING_BOUNDED_ROLLBACK", shipped: "on", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_MULTI_TOKEN_WINDOW_VIEWS", shipped: "on", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_GEMMA4_VERIFY_QMM_LM_HEAD", shipped: "off", class: SwitchClass::Token("gemma4_verify_qmm") },
    SwitchRow { env: "AX_MLX_GEMMA_MT_PERPOS_FFN", shipped: "off", class: SwitchClass::Token("gemma_mt_perpos") },
    SwitchRow { env: "AX_MLX_MULTI_TOKEN_F32_ATTENTION", shipped: "on", class: SwitchClass::Token("mt_f32=0") },
    SwitchRow { env: "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_F32_SDPA", shipped: "on", class: SwitchClass::Token("gemma4_prefill_f32=0") },
    SwitchRow { env: "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_EMBED_CLIP", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAST_RESIDUAL", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAST_FFN_PACKED", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_PREFILL_BF16_EMBED", shipped: "on", class: SwitchClass::Token("gemma4_bf16_embed=0") },
    SwitchRow { env: "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAYER_MASKS", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_PREFILL_PIPELINE_HINT_P128", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_PREFILL_LAST_QUERY_P128", shipped: "off", class: SwitchClass::Token("gemma4_last_query") },
    SwitchRow { env: "AX_MLX_DENSE_LONG_MT_BF16_FOLD", shipped: "on", class: SwitchClass::Token("dense_bf16_fold=0") },
    SwitchRow { env: "AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN", shipped: "off", class: SwitchClass::Token("cpp_gemma4_ffn") },
    SwitchRow { env: "AX_MLX_DENSE_QMATMUL_RMS_NORM", shipped: "off", class: SwitchClass::Token("dense_qmm_rms") },
    SwitchRow { env: "AX_MLX_O_PROJ_QMATMUL_RMS_NORM", shipped: "off", class: SwitchClass::Token("o_proj_rms") },
    SwitchRow { env: "AX_MLX_ATTN_NORM_QKV_FUSE", shipped: "off", class: SwitchClass::Token("attn_qkv_fuse") },
    SwitchRow { env: "AX_MLX_QWEN_ATTN_NORM_QKV_FUSE", shipped: "off", class: SwitchClass::Token("qwen_attn_qkv_fuse") },
    SwitchRow { env: "AX_MLX_GEMMA4_ATTN_NORM_QKV_FUSE_P128", shipped: "on", class: SwitchClass::Token("gemma4_attn_qkv_fuse=0") },
    SwitchRow { env: "AX_MLX_DUAL_QMM_GEGLU", shipped: "off", class: SwitchClass::Token("dual_qmm_geglu") },
    SwitchRow { env: "AX_MLX_ASYNC_DUAL_GATE_UP", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASYNC_DUAL_GATE_UP_P128", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASYNC_FIRST_KV_P128", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_DUAL_STREAM_GATE_UP_P128", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_CACHE_ONLY_CHUNK_EVAL", shipped: "off", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL", shipped: "off", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_PIPELINE_GRANULARITY", shipped: "unset", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_PIPELINE_BLOCK", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_PIPELINE_EVAL_GRANULARITY", shipped: "unset", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_PIPELINE_EVAL_TAIL_LAYERS", shipped: "unset", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_NATIVE_OFFSET_CAUSAL", shipped: "off", class: SwitchClass::Token("native_causal") },
    SwitchRow { env: "AX_MLX_DENSE_GEGLU_DOWN_FUSE", shipped: "off", class: SwitchClass::Token("geglu_down_fuse") },
    SwitchRow { env: "AX_MLX_QWEN_SWIGLU_DOWN_FUSE", shipped: "off", class: SwitchClass::Token("swiglu_down_fuse") },
    SwitchRow { env: "AX_MLX_QWEN_DUAL_QMM_SWIGLU", shipped: "off", class: SwitchClass::Token("dual_qmm_swiglu") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL", shipped: "off", class: SwitchClass::ProvenExact("qwen_prefill_dual_qmm_swiglu_metal_matches_two_qmm_silu_mul_4bit_gs32") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_FLAT_DOWN_QMM", shipped: "off", class: SwitchClass::Token("flat_down") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_FLAT_FFN", shipped: "off", class: SwitchClass::Token("flat_ffn") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_CONTIGUOUS_FFN", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_LA_OUT_PROJ_SILU_MUL_QMM", shipped: "off", class: SwitchClass::Token("la_out_proj_fuse") },
    SwitchRow { env: "AX_MLX_DIRECT_CPP_QK_NORM_ROPE", shipped: "off", class: SwitchClass::Token("cpp_qk") },
    SwitchRow { env: "AX_MLX_QWEN_DIRECT_CPP_QK_NORM_ROPE", shipped: "on", class: SwitchClass::Token("qwen_cpp_qk=0") },
    SwitchRow { env: "AX_MLX_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_LA_FUSED_QKVZ_BA_QMM", shipped: "off", class: SwitchClass::Token("la_fused_qkvz") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_DOWN_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_CHUNK_1536", shipped: "off", class: SwitchClass::Token("prefill_chunk=1536") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_CHUNK_1280", shipped: "off", class: SwitchClass::Token("prefill_chunk=1280") },
    SwitchRow { env: "AX_MLX_QWEN_COMPILED_GATED_DELTA_PREFILL", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PACKED_FFN_PREFILL_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PACKED_LA_INPUTS_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_LA_POST_INPUT_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_CONTIGUOUS_LA_INPUT", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_LA_CONTIGUOUS_QKV", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_LA_PREFILL_Q2_PROJ", shipped: "off", class: SwitchClass::Token("la_q2") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_Q2_DOWN", shipped: "off", class: SwitchClass::Token("down_q2") },
    SwitchRow { env: "AX_MLX_QWEN_GD_PREFILL_CHUNKWISE", shipped: "off", class: SwitchClass::Token("gd_chunkwise") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_FFN_GS64", shipped: "off", class: SwitchClass::Token("ffn_gs64") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_Q3_FFN", shipped: "off", class: SwitchClass::Token("ffn_q3") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_CONTIGUOUS_FFN_WEIGHTS", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_ASYNC_GATE_UP", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_FFN_F32_INPUT", shipped: "off", class: SwitchClass::Token("ffn_f32") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_EVAL_FFN_INPUT", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_EVAL_LA_INPUT", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_ASYNC_LA_OUTPUTS", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_ASYNC_PACKED_GATE_UP", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_CONTIGUOUS_LA_WEIGHTS", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_EVAL_ATTN_INPUT", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_EVAL_FFN_HIDDEN", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_CONTIGUOUS_ATTN_WEIGHTS", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_LA_OUT", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_ASYNC_DOWN", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_LAST_TOKEN_O_PROJ", shipped: "off", class: SwitchClass::Token("last_token_o") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_REUSE_ROPE", shipped: "off", class: SwitchClass::Token("reuse_rope") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_LAST_QUERY_SDPA", shipped: "off", class: SwitchClass::Token("last_query_sdpa") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_LAST_QUERY_Q_PROJ", shipped: "off", class: SwitchClass::Token("last_query_q") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_QK_NORM", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_ASYNC_SDPA", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_ASYNC_GD", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_EVAL_GD", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_CONTIGUOUS_GD", shipped: "off", class: SwitchClass::NonNumeric("placement") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_SPLIT_PACKED", shipped: "off", class: SwitchClass::Token("qwen_split_packed") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_DEQUANT_DENSE", shipped: "auto", class: SwitchClass::Token("dequant_dense=cached") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_DEQUANT_DENSE_MIN_SEQ", shipped: "auto", class: SwitchClass::Token("dequant_min=256") },
    SwitchRow { env: "AX_MLX_QWEN_LA_NORM_QKVZ_FUSE", shipped: "off", class: SwitchClass::Token("la_norm_qkvz") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_SKIP_BF16_ASTYPE", shipped: "off", class: SwitchClass::Token("qwen_skip_bf16_astype") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_FLAT_QMM", shipped: "off", class: SwitchClass::Token("flat_qmm") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_TILE_QMM", shipped: "off", class: SwitchClass::Token("tile_qmm") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_DUAL_AFFINE_QMM", shipped: "off", class: SwitchClass::Token("qwen_dual_affine") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_EMBED_CLIP", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA", shipped: "on", class: SwitchClass::Token("qwen_prefill_f32=0") },
    SwitchRow { env: "AX_MLX_HY_V3_PREFILL_SKIP_F32_SDPA", shipped: "on", class: SwitchClass::Token("hy_prefill_f32=0") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_BF16_EMBED_DEQUANT", shipped: "off", class: SwitchClass::Token("qwen_bf16_embed") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL", shipped: "off", class: SwitchClass::Token("qwen_native_causal") },
    SwitchRow { env: "AX_MLX_NAX_ATTENTION", shipped: "on", class: SwitchClass::Token("nax_attn=0") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_SWIGLU_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_ASYNC_EMBED", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA_DIRECT_CPP_QK_NORM_ROPE", shipped: "off", class: SwitchClass::Token("gemma_cpp_qk") },
    SwitchRow { env: "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS", shipped: "off", class: SwitchClass::Token("cpp_la_in") },
    SwitchRow { env: "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS", shipped: "on", class: SwitchClass::Token("qwen_cpp_la_in=0") },
    SwitchRow { env: "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT", shipped: "off", class: SwitchClass::Token("cpp_la_post") },
    SwitchRow { env: "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT", shipped: "on", class: SwitchClass::Token("qwen_cpp_la_post=0") },
    SwitchRow { env: "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL", shipped: "on", class: SwitchClass::Token("la_decode_post=0") },
    SwitchRow { env: "AX_MLX_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL", shipped: "off", class: SwitchClass::Token("la_prefill_post") },
    SwitchRow { env: "AX_MLX_QWEN_GATED_DELTA_DECODE_METAL", shipped: "on", class: SwitchClass::Token("gd_decode_metal=0") },
    SwitchRow { env: "AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX", shipped: "off", class: SwitchClass::Token("qwen_gdn_mlx") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX", shipped: "on", class: SwitchClass::Token("gdn_prefill_mlx=0") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_DEVICE_SELECT", shipped: "on", class: SwitchClass::Token("qsa_decode_select=0") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_DEVICE_SELECT_MULTI", shipped: "on", class: SwitchClass::Token("qsa_multi_select=0") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_DECODE_BLOCK_KEY", shipped: "off", class: SwitchClass::Token("decode_key") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_PREFILL_DEVICE_SELECT", shipped: "on", class: SwitchClass::ProvenExact("prefill_device_selection_matches_host_on_subnormal_scores_zeros_ties_and_normals") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_DENSE_SHORT_CONTEXT", shipped: "on", class: SwitchClass::Token("qsa_dense=0") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_BATCHED_ATTENTION", shipped: "on", class: SwitchClass::Token("qsa_batched=0") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_FUSED_GATHER", shipped: "on", class: SwitchClass::ProvenExact("fused_gather_batched_attention_matches_bit_for_bit") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL", shipped: "off", class: SwitchClass::Token("indexed") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_TILED_SCORES", shipped: "off", class: SwitchClass::ProvenExact("qsa_tiled_score_bits_match_monolithic") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_QSA_TILED_PREFILL_CHUNK", shipped: "unset", class: SwitchClass::Token("chunk=8192") },
    SwitchRow { env: "AX_MLX_FLASH_NEXT_PREFILL_EXPERT_SORT", shipped: "on", class: SwitchClass::Token("expert_sort=0") },
    SwitchRow { env: "AX_MLX_QWEN_GATED_DELTA_PREFILL_STREAMING", shipped: "off", class: SwitchClass::Token("gd_streaming") },
    SwitchRow { env: "AX_MLX_QWEN_GATED_DELTA_PREFILL_TILE_512", shipped: "off", class: SwitchClass::Token("gd_tile_512") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_SINGLE_2048", shipped: "off", class: SwitchClass::Token("single_2048") },
    SwitchRow { env: "AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS", shipped: "on", class: SwitchClass::Token("pack_la=0") },
    SwitchRow { env: "AX_MLX_LINEAR_ATTENTION_RMS_NORM_GATE_METAL", shipped: "on", class: SwitchClass::ProvenExact("rms_norm_gate_metal_is_bit_exact_vs_portable_silu_mul") },
    SwitchRow { env: "AX_MLX_MOE_FUSE_SHARED_EXPERT_ADD", shipped: "on", class: SwitchClass::Token("moe_shared_add=0") },
    SwitchRow { env: "AX_MLX_QWEN3_MOE_NARROW_SOFTMAX", shipped: "on", class: SwitchClass::Token("narrow_softmax=0") },
    SwitchRow { env: "AX_MLX_MOE_LAYER_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_DENSE_FFN_COMPILE", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_DENSE_FFN_COMPILE_PREFILL", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_COMPILED_DUAL_GATE_UP", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_SPLIT_FFN_PREFILL_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_PACKED_FFN_COMPILE_P128", shipped: "off", class: SwitchClass::Token("gemma4_p128_packed") },
    SwitchRow { env: "AX_MLX_AUTO_BUFFER_CAPS", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_MOE_ROUTER_FUSED_METAL", shipped: "off", class: SwitchClass::Token("moe_router") },
    SwitchRow { env: "AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL", shipped: "off", class: SwitchClass::Token("la_whole_layer") },
    SwitchRow { env: "AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL", shipped: "off", class: SwitchClass::Token("moe_deep_expert") },
    SwitchRow { env: "AX_MLX_MOE_FUSED_EXPERT_BLOCK", shipped: "off", class: SwitchClass::Token("moe_expert") },
    SwitchRow { env: "AX_MLX_MOE_EXPERT_PARALLEL", shipped: "off", class: SwitchClass::Token("moe_parallel") },
    SwitchRow { env: "AX_MLX_MOE_SWIGLU_PACKED_METAL", shipped: "on", class: SwitchClass::Token("moe_swiglu_packed=0") },
    SwitchRow { env: "AX_MLX_MOE_GEGLU_PACKED_METAL", shipped: "on", class: SwitchClass::ProvenExact("packed_geglu_metal_matches_direct_geglu_for_bf16_packed_gate_up") },
    SwitchRow { env: "AX_MLX_F32_PACK_BF16_NORMALIZE", shipped: "on", class: SwitchClass::Token("f32_pack_bf16=0") },
    SwitchRow { env: "AX_MLX_MOE_SWIGLU_PREFILL_MAX_SEQ", shipped: "0", class: SwitchClass::Token("moe_swiglu_prefill=512") },
    SwitchRow { env: "AX_MLX_MOE_SHARED_FUSION_SEQ_THRESHOLD", shipped: "unset", class: SwitchClass::Token("moe_shared_seq=128") },
    SwitchRow { env: "AX_MLX_MLA_PREFILL_CHUNK", shipped: "unset", class: SwitchClass::Token("mla_chunk=64") },
    SwitchRow { env: "AX_MLX_MLA_COLD_PREFILL_CHUNK", shipped: "unset", class: SwitchClass::Token("mla_cold=256") },
    SwitchRow { env: "AX_MLX_LONG_PROMPT_PREFILL_CHUNK", shipped: "512", class: SwitchClass::Token("long_chunk=768") },
    SwitchRow { env: "AX_MLX_QWEN_MOE_PREFILL_SINGLE_2048", shipped: "on", class: SwitchClass::Token("moe_single_2048=0") },
    SwitchRow { env: "AX_MLX_SKIP_COLD_PREFILL_CACHE_CLEAR", shipped: "off", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_INTERMEDIATE_ASYNC_EVAL", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_QWEN_PREFILL_LAZY_INTERMEDIATE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_PREFIX_CACHE_DIR", shipped: "unset", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_PREFIX_CACHE_DISK_DISABLED", shipped: "off", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_DISABLE_MLA_PREFIX_RESTORE", shipped: "off", class: SwitchClass::Token("mla_prefix_restore=0") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASSISTANT_COMPILE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF", shipped: "on", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_DIFFUSION_NO_EMBEDDING_CACHE", shipped: "off", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_DIFFUSION_NO_KV_CONCAT_BUFFER", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_DIFFUSION_NO_FULL_PIPELINE", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_DIFFUSION_NO_COMPILED_FORWARD", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MTP_COMPILED_HEAD", shipped: "on", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_MTP_COMPILED_HEAD_FIXED_KV", shipped: "on", class: SwitchClass::NonNumeric("draft") },
    SwitchRow { env: "AX_DIFFUSION_NO_SKIP_COMMIT", shipped: "off", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_DIFFUSION_PROFILE", shipped: "off", class: SwitchClass::NonNumeric("telemetry") },
    SwitchRow { env: "AX_MLX_BATCHED_SHARED_PROJ", shipped: "on", class: SwitchClass::Token("batched_proj=row") },
    SwitchRow { env: "AX_MLX_BATCHED_STATE_MATERIALIZE", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_DIFFUSION_ENTROPY_THRESHOLD", shipped: "0.005", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_ACCEPTANCE_RATE_THRESHOLD", shipped: "0.075", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_ENTROPY_PLATEAU_DELTA", shipped: "0.001", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_MAX_STEPS", shipped: "48", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_STEPS_PER_ENGINE_STEP", shipped: "unset", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_CHECK_INTERVAL", shipped: "unset", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_SAMPLER", shipped: "unset", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_CONFIDENCE_THRESHOLD", shipped: "0.9", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_TEMPERATURE_SCHEDULE", shipped: "unset", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_DIFFUSION_SC_SKIP_ACCEPTANCE_RATE", shipped: "0.95", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_MLX_MTP_GDN_TGY", shipped: "4", class: SwitchClass::Token("gdn_tgy=8") },
    SwitchRow { env: "AX_MLX_MULTIMODAL_PREFIX_REUSE", shipped: "off", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_EMBED_MEAN_COMPILE_THRESHOLD", shipped: "512", class: SwitchClass::NonNumeric("scheduling") },
    SwitchRow { env: "AX_MLX_MTP_LINEAR_EXACT_REPLAY", shipped: "off", class: SwitchClass::Token("mtp_exact_replay") },
    SwitchRow { env: "AX_MLX_MTP_NGRAM_CACHE_POLICY", shipped: "preserve", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_MLX_GEMMA4_MOE_LONG_MT", shipped: "off", class: SwitchClass::Token("gemma4_moe_long") },
    SwitchRow { env: "AX_GEMMA4_LOOP_DETECTION", shipped: "default", class: SwitchClass::NonNumeric("sampling") },
    SwitchRow { env: "AX_MLX_PRICED_MEMORY", shipped: "on", class: SwitchClass::NonNumeric("cache") },
    SwitchRow { env: "AX_STREAM_EXPERT_CACHE_BYTES", shipped: "unset", class: SwitchClass::NonNumeric("cache") },
];

fn apply_numerics_deviation(env: &str, switches: &mut NumericsTierSwitches) -> bool {
    match env {
        "AX_MLX_FLASH_NEXT_GDN_METAL" => {
            switches.gdn_metal = false;
            true
        }
        "AX_MLX_ACTIVATION_DTYPE" => {
            switches.activation_deviation = Some("act=f16");
            true
        }
        "AX_MLX_BATCHED_PREFILL" => {
            switches.batched_prefill = true;
            true
        }
        "AX_MLX_BATCHED_PREFILL_TOKENS" => {
            switches.batched_tokens = Some(0);
            true
        }
        "AX_MLX_BATCHED_PREFILL_ROWS" => {
            switches.batched_rows = Some(4);
            true
        }
        "AX_MLX_GEMMA4_ASSISTANT_MTP_SEQUENTIAL_ORACLE" => {
            switches.gemma_sequential_oracle = false;
            true
        }
        "AX_MLX_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD" => {
            switches.gemma_cycle_guard = false;
            true
        }
        "AX_MLX_GEMMA4_ASSISTANT_MTP_SENSITIVE_F32" => {
            switches.gemma_sensitive_f32 = false;
            true
        }
        "AX_MLX_GEMMA4_ASSISTANT_MTP_EARLY_GEN_PURE_DIRECT" => {
            switches.gemma_early_direct = true;
            true
        }
        "AX_MLX_PACK_QKV_PROJECTIONS" => {
            switches.pack_qkv = false;
            true
        }
        "AX_MLX_PACK_DENSE_FFN_GATE_UP" => {
            switches.pack_gate_up = false;
            true
        }
        "AX_MLX_GEGLU_MUL_METAL" => {
            switches.geglu_mul = false;
            true
        }
        "AX_MLX_DENSE_SWIGLU_PACKED_METAL" => {
            switches.dense_swiglu_packed = false;
            true
        }
        "AX_MLX_QWEN_LINEAR_MTP_EXACT" => {
            switches.qwen_exact_killed = true;
            true
        }
        "AX_MLX_MTP_PACKED_VERIFY_FFN" => {
            switches.mtp_packed_verify = true;
            true
        }
        "AX_MLX_MTP_NATIVE_GREEDY_VERIFY_LOGITS" => {
            switches.native_greedy_logits = true;
            true
        }
        "AX_MLX_MTP_LA_OUT_PROJ_SILU_MUL_QMM" => {
            switches.mtp_la_out_fuse = false;
            true
        }
        "AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP_DEPTH" => {
            switches.throughput_mtp_depth = 4;
            true
        }
        "AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP" => {
            switches.throughput_mtp = false;
            true
        }
        "AX_MLX_MTP_LINEAR_PROJECTED_REPLAY" => {
            switches.mtp_projected_replay = true;
            true
        }
        "AX_MLX_MTP_RELAXED_TARGET_VERIFY" => {
            switches.relaxed_mtp = false;
            true
        }
        "AX_MLX_MTP_LINEAR_TAPE_CAPTURE" => {
            switches.mtp_tape = true;
            true
        }
        "AX_MLX_MTP_GDN_PREWORK_SIMD32" => {
            switches.mtp_gdn_prework = true;
            true
        }
        "AX_MLX_MTP_FUSED_GDN_VERIFY" => {
            switches.mtp_fused_gdn = true;
            true
        }
        "AX_MLX_MTP_REFOLD_ACCEPTED_HISTORY" => {
            switches.mtp_refold = true;
            true
        }
        "AX_MLX_MTP_BATCHED_COMMITTED_FOLD" => {
            switches.mtp_committed_fold = true;
            true
        }
        "AX_MLX_MTP_LAST_COMMITTED_QUERY" => {
            switches.mtp_last_query = true;
            true
        }
        "AX_MLX_INVARIANT_MXFP4_QMV_FAST" => {
            switches.invariant_mxfp4 = true;
            true
        }
        "AX_MLX_QWEN_DENSE_FFN_MATVEC_EXT_BITS" => {
            switches.matvec_ext_bits = true;
            true
        }
        "AX_MLX_EXACT_LM_HEAD_WINDOW_INVARIANT" => {
            switches.exact_lm_head_window = false;
            true
        }
        "AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV" => {
            switches.dense_weight_t_gemv = false;
            true
        }
        "AX_MLX_INVARIANT_QMV_BF16_Q4" => {
            switches.invariant_qmv_bf16_q4 = false;
            true
        }
        "AX_MLX_QWEN4_EXP_VISION" => {
            switches.qwen4_vision = true;
            true
        }
        "AX_MLX_DENSE_WIDE_GEMV" => {
            switches.wide_gemv = true;
            true
        }
        "AX_MLX_EXACT_RMS_GATE_METAL" => {
            switches.exact_rms_gate = true;
            true
        }
        "AX_MLX_GEMMA_DUAL_GATE_UP_METAL" => {
            switches.gemma_dual_gate_up = true;
            true
        }
        "AX_MLX_LAYER_SCALAR_FUSED_ADD" => {
            switches.layer_scalar = false;
            true
        }
        "AX_MLX_FUSED_PREFILL_ATTENTION" => {
            switches.fused_prefill = true;
            true
        }
        "AX_MLX_QWEN_FUSED_PREFILL_ATTENTION" => {
            switches.qwen_fused_prefill = true;
            true
        }
        "AX_MLX_QWEN_LINEAR_ADD_RMS_NORM" => {
            switches.qwen_add_rms = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_INTERLAYER_ADD_RMS" => {
            switches.qwen_inter_add_rms = true;
            true
        }
        "AX_MLX_GEMMA4_FUSED_PREFILL_ATTENTION_P128" => {
            switches.gemma4_fused_p128 = false;
            true
        }
        "AX_MLX_GEMMA4_VERIFY_QMM_LM_HEAD" => {
            switches.gemma4_verify_qmm = true;
            true
        }
        "AX_MLX_GEMMA_MT_PERPOS_FFN" => {
            switches.gemma_mt_perpos_ffn = true;
            true
        }
        "AX_MLX_MULTI_TOKEN_F32_ATTENTION" => {
            switches.multi_token_f32_attention = false;
            true
        }
        "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_F32_SDPA" => {
            switches.gemma4_prefill_skip_f32 = false;
            true
        }
        "AX_MLX_GEMMA4_PREFILL_BF16_EMBED" => {
            switches.gemma4_bf16_embed = false;
            true
        }
        "AX_MLX_GEMMA4_PREFILL_LAST_QUERY_P128" => {
            switches.gemma4_last_query = true;
            true
        }
        "AX_MLX_DENSE_LONG_MT_BF16_FOLD" => {
            switches.dense_bf16_fold = false;
            true
        }
        "AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN" => {
            switches.cpp_gemma4_ffn = true;
            true
        }
        "AX_MLX_DENSE_QMATMUL_RMS_NORM" => {
            switches.dense_qmm_rms = true;
            true
        }
        "AX_MLX_O_PROJ_QMATMUL_RMS_NORM" => {
            switches.o_proj_rms = true;
            true
        }
        "AX_MLX_ATTN_NORM_QKV_FUSE" => {
            switches.attn_qkv_fuse = true;
            true
        }
        "AX_MLX_QWEN_ATTN_NORM_QKV_FUSE" => {
            switches.qwen_attn_qkv_fuse = true;
            true
        }
        "AX_MLX_GEMMA4_ATTN_NORM_QKV_FUSE_P128" => {
            switches.gemma4_attn_qkv_fuse = false;
            true
        }
        "AX_MLX_DUAL_QMM_GEGLU" => {
            switches.dual_qmm_geglu = true;
            true
        }
        "AX_MLX_NATIVE_OFFSET_CAUSAL" => {
            switches.native_offset_causal = true;
            true
        }
        "AX_MLX_DENSE_GEGLU_DOWN_FUSE" => {
            switches.geglu_down_fuse = true;
            true
        }
        "AX_MLX_QWEN_SWIGLU_DOWN_FUSE" => {
            switches.swiglu_down_fuse = true;
            true
        }
        "AX_MLX_QWEN_DUAL_QMM_SWIGLU" => {
            switches.dual_qmm_swiglu = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL" => {
            switches.dual_qmm_swiglu_metal = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_FLAT_DOWN_QMM" => {
            switches.flat_down = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_FLAT_FFN" => {
            switches.flat_ffn = true;
            true
        }
        "AX_MLX_QWEN_LA_OUT_PROJ_SILU_MUL_QMM" => {
            switches.la_out_proj_fuse = true;
            true
        }
        "AX_MLX_DIRECT_CPP_QK_NORM_ROPE" => {
            switches.cpp_qk = true;
            true
        }
        "AX_MLX_QWEN_DIRECT_CPP_QK_NORM_ROPE" => {
            switches.qwen_cpp_qk = false;
            true
        }
        "AX_MLX_QWEN_LA_FUSED_QKVZ_BA_QMM" => {
            switches.la_fused_qkvz = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_CHUNK_1536" => {
            switches.prefill_chunk_1536 = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_CHUNK_1280" => {
            switches.prefill_chunk_1280 = true;
            true
        }
        "AX_MLX_QWEN_LA_PREFILL_Q2_PROJ" => {
            switches.la_q2 = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_Q2_DOWN" => {
            switches.down_q2 = true;
            true
        }
        "AX_MLX_QWEN_GD_PREFILL_CHUNKWISE" => {
            switches.gd_chunkwise = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_FFN_GS64" => {
            switches.ffn_gs64 = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_Q3_FFN" => {
            switches.ffn_q3 = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_FFN_F32_INPUT" => {
            switches.ffn_f32 = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_LAST_TOKEN_O_PROJ" => {
            switches.last_token_o = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_REUSE_ROPE" => {
            switches.reuse_rope = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_LAST_QUERY_SDPA" => {
            switches.last_query_sdpa = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_LAST_QUERY_Q_PROJ" => {
            switches.last_query_q = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_SPLIT_PACKED" => {
            switches.qwen_split_packed = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_DEQUANT_DENSE" => {
            switches.dequant_dense = Some("dequant_dense=cached");
            true
        }
        "AX_MLX_QWEN_PREFILL_DEQUANT_DENSE_MIN_SEQ" => {
            switches.dequant_dense_min_seq = Some(256);
            true
        }
        "AX_MLX_QWEN_LA_NORM_QKVZ_FUSE" => {
            switches.la_norm_qkvz = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_SKIP_BF16_ASTYPE" => {
            switches.qwen_skip_bf16_astype = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_FLAT_QMM" => {
            switches.flat_qmm = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_TILE_QMM" => {
            switches.tile_qmm = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_DUAL_AFFINE_QMM" => {
            switches.qwen_dual_affine = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA" => {
            switches.qwen_prefill_skip_f32 = false;
            true
        }
        "AX_MLX_HY_V3_PREFILL_SKIP_F32_SDPA" => {
            switches.hy_v3_prefill_skip_f32 = false;
            true
        }
        "AX_MLX_QWEN_PREFILL_BF16_EMBED_DEQUANT" => {
            switches.qwen_bf16_embed = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL" => {
            switches.qwen_native_offset_causal = true;
            true
        }
        "AX_MLX_NAX_ATTENTION" => {
            switches.nax_attention_killed = true;
            true
        }
        "AX_MLX_GEMMA_DIRECT_CPP_QK_NORM_ROPE" => {
            switches.gemma_cpp_qk = true;
            true
        }
        "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS" => {
            switches.cpp_la_in = true;
            true
        }
        "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS" => {
            switches.qwen_cpp_la_in = false;
            true
        }
        "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT" => {
            switches.cpp_la_post = true;
            true
        }
        "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT" => {
            switches.qwen_cpp_la_post = false;
            true
        }
        "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL" => {
            switches.la_decode_post = false;
            true
        }
        "AX_MLX_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL" => {
            switches.la_prefill_post = true;
            true
        }
        "AX_MLX_QWEN_GATED_DELTA_DECODE_METAL" => {
            switches.gd_decode_metal = false;
            true
        }
        "AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX" => {
            switches.qwen_gdn_prefill_mlx = true;
            true
        }
        "AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX" => {
            switches.flash_gdn_prefill_mlx = false;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_DEVICE_SELECT" => {
            switches.qsa_device_select = false;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_DEVICE_SELECT_MULTI" => {
            switches.qsa_device_select_multi = false;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_DECODE_BLOCK_KEY" => {
            switches.decode_block_key = true;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_PREFILL_DEVICE_SELECT" => {
            switches.prefill_device_select = false;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_DENSE_SHORT_CONTEXT" => {
            switches.qsa_dense_short = false;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_BATCHED_ATTENTION" => {
            switches.qsa_batched_attention = false;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_FUSED_GATHER" => {
            switches.qsa_fused_gather = false;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL" => {
            switches.indexed_prefill = true;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_TILED_SCORES" => {
            switches.tiled_scores = true;
            true
        }
        "AX_MLX_FLASH_NEXT_QSA_TILED_PREFILL_CHUNK" => {
            switches.tiled_prefill_chunk = Some(8192);
            true
        }
        "AX_MLX_FLASH_NEXT_PREFILL_EXPERT_SORT" => {
            switches.prefill_expert_sort = false;
            true
        }
        "AX_MLX_QWEN_GATED_DELTA_PREFILL_STREAMING" => {
            switches.gd_streaming = true;
            true
        }
        "AX_MLX_QWEN_GATED_DELTA_PREFILL_TILE_512" => {
            switches.gd_tile_512 = true;
            true
        }
        "AX_MLX_QWEN_PREFILL_SINGLE_2048" => {
            switches.single_2048 = true;
            true
        }
        "AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS" => {
            switches.pack_la = false;
            true
        }
        "AX_MLX_MOE_FUSE_SHARED_EXPERT_ADD" => {
            switches.moe_shared_add = false;
            true
        }
        "AX_MLX_QWEN3_MOE_NARROW_SOFTMAX" => {
            switches.narrow_softmax = false;
            true
        }
        "AX_MLX_GEMMA4_PACKED_FFN_COMPILE_P128" => {
            switches.gemma4_p128_packed = true;
            true
        }
        "AX_MLX_AUTO_BUFFER_CAPS" => {
            switches.auto_buffer_caps = false;
            true
        }
        "AX_MLX_MOE_ROUTER_FUSED_METAL" => {
            switches.moe_router_fused = true;
            true
        }
        "AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL" => {
            switches.linear_attention_whole_layer = true;
            true
        }
        "AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL" => {
            switches.moe_deep_expert_block = true;
            true
        }
        "AX_MLX_MOE_FUSED_EXPERT_BLOCK" => {
            switches.moe_fused_expert_block = true;
            true
        }
        "AX_MLX_MOE_EXPERT_PARALLEL" => {
            switches.moe_expert_parallel = true;
            true
        }
        "AX_MLX_MOE_SWIGLU_PACKED_METAL" => {
            switches.moe_swiglu_packed = false;
            true
        }
        "AX_MLX_MOE_GEGLU_PACKED_METAL" => {
            switches.moe_geglu_packed = false;
            true
        }
        "AX_MLX_F32_PACK_BF16_NORMALIZE" => {
            switches.f32_pack_bf16 = false;
            true
        }
        "AX_MLX_MOE_SWIGLU_PREFILL_MAX_SEQ" => {
            switches.moe_swiglu_prefill = Some(512);
            true
        }
        "AX_MLX_MOE_SHARED_FUSION_SEQ_THRESHOLD" => {
            switches.moe_shared_seq = Some(128);
            true
        }
        "AX_MLX_MLA_PREFILL_CHUNK" => {
            switches.mla_prefill_chunk = Some(64);
            true
        }
        "AX_MLX_MLA_COLD_PREFILL_CHUNK" => {
            switches.mla_cold_chunk = Some(256);
            true
        }
        "AX_MLX_LONG_PROMPT_PREFILL_CHUNK" => {
            switches.long_prompt_chunk = 768;
            true
        }
        "AX_MLX_QWEN_MOE_PREFILL_SINGLE_2048" => {
            switches.moe_single_2048 = false;
            true
        }
        "AX_MLX_BATCHED_SHARED_PROJ" => {
            switches.batched_shared_proj = false;
            true
        }
        "AX_MLX_MTP_GDN_TGY" => {
            switches.gdn_tgy = Some(8);
            true
        }
        "AX_MLX_MTP_LINEAR_EXACT_REPLAY" => {
            switches.mtp_exact_replay = true;
            true
        }
        "AX_MLX_GEMMA4_MOE_LONG_MT" => {
            switches.gemma4_moe_long = true;
            true
        }
        "AX_MLX_MTP_WHOLE_VERIFY_COMPILE" => {
            switches.mtp_whole_verify_compile = true;
            true
        }
        "AX_DISABLE_MLA_PREFIX_RESTORE" => {
            switches.mla_prefix_restore_disabled = true;
            true
        }
        _ => false,
    }
}

fn quoted_fastpath_env_names(source: &str) -> std::collections::BTreeSet<String> {
    let bytes = source.as_bytes();
    let mut names = std::collections::BTreeSet::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'"' {
            index += 1;
            continue;
        }
        let start = index + 1;
        let mut end = start;
        while end < bytes.len() && bytes[end] != b'"' {
            end += 1;
        }
        if end >= bytes.len() {
            break;
        }
        if let Ok(literal) = std::str::from_utf8(&bytes[start..end])
            && is_ax_env_name(literal)
        {
            names.insert(literal.to_string());
        }
        index = end + 1;
    }
    names
}

fn is_ax_env_name(literal: &str) -> bool {
    let Some(rest) = literal.strip_prefix("AX_") else {
        return false;
    };
    !rest.is_empty()
        && rest
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
}

/// Same call shape as `DIRECT_ENV_CALL_RE` in
/// `scripts/check_mlx_fastpath_env_controls.py`.
///
/// Needle pieces are split so this file's source text does not contain a
/// contiguous process-environment read. The per-call allow-list scans every
/// `.rs` file as text.
fn direct_env_var_names(source: &str) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    let bytes = source.as_bytes();
    let mut index = 0;
    let std_var_os = concat!("std::env::", "var_os(");
    let std_var = concat!("std::env::", "var(");
    let var_os = concat!("env::", "var_os(");
    let var_plain = concat!("env::", "var(");
    let needles = [std_var_os, std_var, var_os, var_plain];
    while index < bytes.len() {
        let tail = &source[index..];
        let mut found: Option<(usize, &str)> = None;
        for needle in needles {
            if let Some(at) = tail.find(needle)
                && found.is_none_or(|(prev, _)| at < prev)
            {
                found = Some((at, needle));
            }
        }
        let Some((at, needle)) = found else {
            break;
        };
        let mut cursor = index + at + needle.len();
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b'"' {
            let start = cursor + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end] != b'"' {
                end += 1;
            }
            if end < bytes.len()
                && let Ok(literal) = std::str::from_utf8(&bytes[start..end])
                && is_ax_env_name(literal)
            {
                names.insert(literal.to_string());
            }
            index = end.saturating_add(1);
        } else {
            index = cursor;
        }
    }
    names
}

#[test]
fn numerics_switch_table_covers_every_fastpath_env() {
    let source = include_str!("../fastpath.rs");
    let owned = quoted_fastpath_env_names(source);
    let mut table: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut tokens = std::collections::BTreeSet::new();
    for row in NUMERICS_SWITCH_TABLE {
        assert!(
            table.insert(row.env.to_string()),
            "duplicate row {}",
            row.env
        );
        assert!(!row.shipped.is_empty(), "{}", row.env);
        let mut switches = NumericsTierSwitches::default();
        let deviated = apply_numerics_deviation(row.env, &mut switches);
        match row.class {
            SwitchClass::Token(token) => {
                assert!(!token.is_empty(), "{}", row.env);
                assert!(tokens.insert(token), "duplicate token {token}");
                assert!(deviated, "token row {} does not deviate", row.env);
                assert_eq!(
                    numerics_tier_suffix(&switches),
                    format!(";numerics={token}"),
                    "{}",
                    row.env
                );
            }
            SwitchClass::ProvenExact(test_name) => {
                assert!(!test_name.is_empty(), "{}", row.env);
                if deviated {
                    assert_eq!(
                        numerics_tier_suffix(&switches),
                        "",
                        "{test_name} must not gain a token"
                    );
                }
            }
            SwitchClass::NonNumeric(kind) => {
                assert!(
                    matches!(
                        kind,
                        "placement" | "telemetry" | "cache" | "scheduling" | "sampling" | "draft"
                    ),
                    "{kind} for {}",
                    row.env
                );
                if deviated {
                    assert_eq!(
                        numerics_tier_suffix(&switches),
                        "",
                        "{} must not gain a token",
                        row.env
                    );
                }
            }
        }
    }
    let missing: Vec<_> = owned.difference(&table).cloned().collect();
    let extra: Vec<_> = table.difference(&owned).cloned().collect();
    assert!(missing.is_empty(), "unclassified fastpath env: {missing:?}");
    assert!(extra.is_empty(), "table names absent from fastpath.rs: {extra:?}");
    let direct = direct_env_var_names(source);
    let unaudited: Vec<_> = direct.difference(&table).cloned().collect();
    assert!(unaudited.is_empty(), "env::var name missing from the table: {unaudited:?}");
    for audited in [
        "AX_DISABLE_MLA_PREFIX_RESTORE",
        "AX_MLX_PACK_DENSE_FFN_GATE_UP",
        "AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS",
        "AX_MLX_PACK_QKV_PROJECTIONS",
        "AX_MLX_PREFILL_FFN_COMPILE_SWIGLU",
    ] {
        assert!(table.contains(audited), "{audited}");
    }
    assert_eq!(numerics_tier_suffix(&NumericsTierSwitches::default()), "");
}
