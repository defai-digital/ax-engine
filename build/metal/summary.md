# AX Metal Kernel Build

- status: `compiled`
- library: `ax_phase1_dense_path`
- manifest: `/Users/akiralam/code/ax-engine-v4/metal/phase1-kernels.json`
- source: `/Users/akiralam/code/ax-engine-v4/metal/kernels/phase1_dense_path.metal`
- doctor_status: `ready`
- default_block_size_tokens: `16`
- supported_block_size_tokens: `16`
- bringup_allowed: `true`
- metal_toolchain_fully_available: `true`
- kernels: `reshape_and_cache, paged_decode_attention, gather_kv_cache, copy_blocks, swap_blocks, kv_scale_update, gather_embedding_rows_f32, gather_embedding_rows_f16, gather_embedding_rows_bf16, decode_logits_projection_f32, decode_logits_projection_f16, decode_logits_projection_bf16, decode_logits_projection_batched_f32, decode_logits_projection_batched_f16, decode_logits_projection_batched_bf16, logits_argmax_f32, logits_argmax_batched_f32, sample_argmax_logprob_f32, sample_argmax_logprob_batched_f32, apply_rope_batched_f32, rms_norm_f32, rms_norm_f16, rms_norm_bf16, rms_norm_batched_f32, rms_norm_batched_f16, rms_norm_batched_bf16, ffn_gate_silu_product_f32, ffn_gate_gelu_approx_product_f32, apply_rope_f32, expand_grouped_kv_heads_f32`
- air: `/Users/akiralam/code/ax-engine-v4/build/metal/ax_phase1_dense_path.air`
- metalar: `/Users/akiralam/code/ax-engine-v4/build/metal/ax_phase1_dense_path.metalar`
- metallib: `/Users/akiralam/code/ax-engine-v4/build/metal/ax_phase1_dense_path.metallib`
