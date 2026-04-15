use rayon::prelude::*;

/// The archived Gemma4 implementations in llama.cpp, mistral.rs, and mlx-lm all
/// classify each layer up front, then derive attention geometry, cache policy,
/// and RoPE behavior from that classification before running the math kernels.
///
/// AX does not implement Gemma4 KV-sharing or per-layer inputs yet, but this
/// batch path follows the same staged shape so those features can land without
/// rewriting one monolithic `forward_batch_chunk` again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4LayerKind {
    Sliding,
    Global,
}

#[derive(Debug, Clone)]
struct Gemma4LayerSpec {
    layer: usize,
    prefix: String,
    kind: Gemma4LayerKind,
    head_dim: usize,
    n_kv_heads: usize,
    q_dim: usize,
    kv_dim: usize,
    v_equals_k: bool,
    rope_base: f32,
    local_window: Option<usize>,
    has_moe: bool,
}

impl Gemma4LayerSpec {
    fn new(layer: usize, config: &ModelConfig, weights: &WeightStore) -> Self {
        let is_sliding = Gemma4Forward::use_sliding_window(layer, config);
        let (head_dim, n_kv_heads, q_dim, kv_dim) = Gemma4Forward::layer_dims(layer, config);
        let kind = if is_sliding {
            Gemma4LayerKind::Sliding
        } else {
            Gemma4LayerKind::Global
        };

        Self {
            layer,
            prefix: format!("blk.{layer}"),
            kind,
            head_dim,
            n_kv_heads,
            q_dim,
            kv_dim,
            v_equals_k: Gemma4Forward::layer_v_equals_k(layer, config, weights),
            rope_base: if is_sliding {
                config.rope_freq_base_local.unwrap_or(config.rope_freq_base)
            } else {
                config.rope_freq_base
            },
            local_window: if is_sliding {
                config.sliding_window_size.map(|window| window as usize)
            } else {
                None
            },
            has_moe: weights.has(&format!("blk.{layer}.ffn_gate_inp.weight")),
        }
    }

    fn is_sliding(&self) -> bool {
        self.kind == Gemma4LayerKind::Sliding
    }

    fn prefix_read_plan(&self, base_seq_len: usize) -> (usize, usize) {
        let prefix_read_len = self
            .local_window
            .map(|window| base_seq_len.min(window))
            .unwrap_or(base_seq_len);
        let prefix_start = base_seq_len.saturating_sub(prefix_read_len);
        (prefix_read_len, prefix_start)
    }

    fn use_backend_prefill(&self, prefix_read_len: usize, _n_tokens: usize) -> bool {
        // Gemma4 requires attention scale=1.0. AX's generic prefill kernels
        // apply 1/sqrt(head_dim), so we only use them for the no-prefix case
        // and compensate by pre-scaling Q with sqrt(head_dim) before dispatch.
        prefix_read_len == 0
    }
}

struct MoeScratch {
    logits: Vec<f32>,
    fused: Vec<f32>,
    down: Vec<f32>,
    expert_ids: Vec<i32>,
    expert_weights: Vec<f32>,
}

struct Gemma4SingleMoeScratch {
    router_input: Vec<f32>,
    expert_accum: Vec<f32>,
    fused: Vec<f32>,
    router_logits: Vec<f32>,
}

impl Gemma4SingleMoeScratch {
    fn new(config: &ModelConfig) -> Self {
        let dim = config.embedding_dim as usize;
        let expert_inter_dim = config.expert_intermediate_dim.unwrap_or(0) as usize;
        let n_expert = config.n_expert.unwrap_or(0) as usize;

        Self {
            router_input: vec![0.0; dim],
            expert_accum: vec![0.0; dim],
            fused: vec![0.0; 2 * expert_inter_dim],
            router_logits: vec![0.0; n_expert],
        }
    }
}

struct Gemma4BatchScratch {
    hidden: Vec<f32>,
    norm_buf: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    attn_out: Vec<f32>,
    proj_buf: Vec<f32>,
    gate_buf: Vec<f32>,
    up_buf: Vec<f32>,
    down_buf: Vec<f32>,
    k_padded_batch: Vec<f32>,
    v_padded_batch: Vec<f32>,
    prefix_k_scratch: Vec<f32>,
    prefix_v_scratch: Vec<f32>,
    padded_prefix_k_scratch: Vec<f32>,
    padded_prefix_v_scratch: Vec<f32>,
    moe_norm_buf: Vec<f32>,
    moe_accum_buf: Vec<f32>,
    moe_scratch: MoeScratch,
}

impl Gemma4BatchScratch {
    fn new(config: &ModelConfig, chunk_len: usize) -> Self {
        let dim = config.embedding_dim as usize;
        let n_heads = config.n_heads as usize;
        let max_hd = config
            .gemma4_head_dim_global
            .unwrap_or(config.head_dim)
            .max(config.gemma4_head_dim_swa.unwrap_or(config.head_dim))
            as usize;
        let max_q_dim = n_heads * max_hd;
        let swa_kv_dim = config.gemma4_n_kv_heads_swa.unwrap_or(config.n_kv_heads) as usize
            * config.gemma4_head_dim_swa.unwrap_or(config.head_dim) as usize;
        let global_kv_dim = config.gemma4_n_kv_heads_global.unwrap_or(config.n_kv_heads) as usize
            * config.gemma4_head_dim_global.unwrap_or(config.head_dim) as usize;
        let max_kv_dim = swa_kv_dim.max(global_kv_dim);
        let inter_dim = config.intermediate_dim as usize;
        let kv_stride = max_kv_dim;
        let hidden_len = chunk_len * dim;

        Self {
            hidden: vec![0.0; hidden_len],
            norm_buf: vec![0.0; hidden_len],
            q_buf: vec![0.0; chunk_len * max_q_dim],
            k_buf: vec![0.0; chunk_len * max_kv_dim],
            v_buf: vec![0.0; chunk_len * max_kv_dim],
            attn_out: vec![0.0; chunk_len * max_q_dim],
            proj_buf: vec![0.0; hidden_len],
            gate_buf: vec![0.0; chunk_len * inter_dim],
            up_buf: vec![0.0; chunk_len * inter_dim],
            down_buf: vec![0.0; hidden_len],
            k_padded_batch: vec![0.0; chunk_len * kv_stride],
            v_padded_batch: vec![0.0; chunk_len * kv_stride],
            prefix_k_scratch: Vec::new(),
            prefix_v_scratch: Vec::new(),
            padded_prefix_k_scratch: Vec::new(),
            padded_prefix_v_scratch: Vec::new(),
            moe_norm_buf: if config.n_expert.is_some() { vec![0.0; hidden_len] } else { Vec::new() },
            moe_accum_buf: if config.n_expert.is_some() { vec![0.0; hidden_len] } else { Vec::new() },
            moe_scratch: MoeScratch {
                logits: vec![0.0; config.n_expert.unwrap_or(0) as usize],
                fused: vec![0.0; 2 * config.expert_intermediate_dim.unwrap_or(0) as usize],
                down: vec![0.0; dim],
                expert_ids: vec![
                    -1;
                    chunk_len * config.n_expert_used.unwrap_or(0) as usize
                ],
                expert_weights: vec![
                    0.0;
                    chunk_len * config.n_expert_used.unwrap_or(0) as usize
                ],
            },
        }
    }
}

impl Gemma4Forward {
    const CPU_BATCH_SCRATCH_TARGET_BYTES: usize = 64 * 1024 * 1024;
    const GPU_KV_BATCH_SCRATCH_TARGET_BYTES: usize = 256 * 1024 * 1024;
    const PARALLEL_BATCH_MIN_TOKENS: usize = 64;
    const PARALLEL_FLOAT_CHUNK: usize = 16 * 1024;
    fn cpu_batch_chunk_len(
        config: &ModelConfig,
        n_tokens: usize,
        scratch_target_bytes: usize,
    ) -> usize {
        if n_tokens <= 1 {
            return n_tokens;
        }

        let dim = config.embedding_dim as usize;
        let n_heads = config.n_heads as usize;
        let max_hd = config
            .gemma4_head_dim_global
            .unwrap_or(config.head_dim)
            .max(config.gemma4_head_dim_swa.unwrap_or(config.head_dim))
            as usize;
        let max_q_dim = n_heads * max_hd;
        let max_kv_dim = config
            .gemma4_n_kv_heads_swa
            .unwrap_or(config.n_kv_heads)
            .max(config.gemma4_n_kv_heads_global.unwrap_or(config.n_kv_heads))
            as usize
            * max_hd;
        let inter_dim = config.intermediate_dim as usize;

        let per_token_floats =
            (4 * dim) + (2 * max_q_dim) + (2 * max_kv_dim) + (2 * inter_dim);
        let per_token_bytes = per_token_floats * std::mem::size_of::<f32>();
        let target_tokens = (scratch_target_bytes / per_token_bytes.max(1)).max(1);
        let sliding_window_cap = config
            .sliding_window_size
            .map(|window| window as usize)
            .unwrap_or(usize::MAX);
        n_tokens.min(target_tokens).min(sliding_window_cap).max(1)
    }

    fn append_layer_batch(
        kv: &mut ModelKv,
        layer: usize,
        k_batch: &[f32],
        v_batch: &[f32],
        n_tokens: usize,
    ) {
        match kv {
            ModelKv::Cpu(cpu_kv) => cpu_kv.append_batch(layer, k_batch, v_batch, n_tokens),
            ModelKv::Gpu(gpu_kv) => gpu_kv.append_layer_batch(layer, k_batch, v_batch, n_tokens),
            ModelKv::Qwen35(_) => unreachable!("gemma4 does not support qwen35 kv"),
        }
    }

    fn finalize_batch(kv: &mut ModelKv, n_tokens: usize) {
        match kv {
            ModelKv::Cpu(cpu_kv) => cpu_kv.finalize_batch(n_tokens),
            ModelKv::Gpu(gpu_kv) => gpu_kv.finalize_batch(n_tokens),
            ModelKv::Qwen35(_) => unreachable!("gemma4 does not support qwen35 kv"),
        }
    }

    fn copy_rows(
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        dst_stride: usize,
        copy_len: usize,
        n_rows: usize,
    ) {
        if n_rows >= Self::PARALLEL_BATCH_MIN_TOKENS {
            src[..n_rows * src_stride]
                .par_chunks(src_stride)
                .zip(dst[..n_rows * dst_stride].par_chunks_mut(dst_stride))
                .for_each(|(src_row, dst_row)| {
                    dst_row[..copy_len].copy_from_slice(&src_row[..copy_len]);
                });
        } else {
            for row in 0..n_rows {
                let src_start = row * src_stride;
                let dst_start = row * dst_stride;
                dst[dst_start..dst_start + copy_len]
                    .copy_from_slice(&src[src_start..src_start + copy_len]);
            }
        }
    }

    fn parallel_elementwise_add(dst: &mut [f32], src: &[f32]) {
        if dst.len() >= Self::PARALLEL_FLOAT_CHUNK {
            dst.par_chunks_mut(Self::PARALLEL_FLOAT_CHUNK)
                .zip(src.par_chunks(Self::PARALLEL_FLOAT_CHUNK))
                .for_each(|(dst_chunk, src_chunk)| {
                    silu::elementwise_add(dst_chunk, src_chunk);
                });
        } else {
            silu::elementwise_add(dst, src);
        }
    }

    fn parallel_scale(values: &mut [f32], scale: f32) {
        if values.len() >= Self::PARALLEL_FLOAT_CHUNK {
            values
                .par_chunks_mut(Self::PARALLEL_FLOAT_CHUNK)
                .for_each(|chunk| {
                    for value in chunk.iter_mut() {
                        *value *= scale;
                    }
                });
        } else {
            for value in values.iter_mut() {
                *value *= scale;
            }
        }
    }

    fn apply_row_rms_norm(
        buf: &mut [f32],
        weight: &[f32],
        n_tokens: usize,
        dim: usize,
        eps: f32,
    ) {
        if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
            buf[..n_tokens * dim]
                .par_chunks_mut(dim)
                .for_each(|row| rms_norm::rms_norm(row, weight, eps));
        } else {
            for row in buf[..n_tokens * dim].chunks_exact_mut(dim) {
                rms_norm::rms_norm(row, weight, eps);
            }
        }
    }

    fn embed_batch_hidden(
        token_ids: &[u32],
        hidden: &mut [f32],
        dim: usize,
        weights: &WeightStore,
        config: &ModelConfig,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        timed!(ops, dequant, {
            for (i, &tid) in token_ids.iter().enumerate() {
                let row = &mut hidden[i * dim..(i + 1) * dim];
                weights.dequantize_row("token_embd.weight", tid as usize, row)?;
            }
            anyhow::Ok(())
        })?;

        if config.embed_scale {
            let embd_scale = (dim as f32).sqrt();
            for h in hidden.iter_mut() {
                *h *= embd_scale;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_norm_batch(
        spec: &Gemma4LayerSpec,
        weights: &WeightStore,
        layer_hidden: &[f32],
        layer_norm: &mut [f32],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let attn_norm_w = timed!(
            ops,
            dequant,
            weights.f32_slice(&format!("{}.attn_norm.weight", spec.prefix))?
        );
        timed!(ops, norm, {
            if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
                layer_hidden[..n_tokens * dim]
                    .par_chunks(dim)
                    .zip(layer_norm[..n_tokens * dim].par_chunks_mut(dim))
                    .for_each(|(hidden_token, norm_token)| {
                        rms_norm::rms_norm_out(
                            hidden_token,
                            attn_norm_w,
                            norm_token,
                            rms_norm_eps,
                        );
                    });
            } else {
                for t in 0..n_tokens {
                    let start = t * dim;
                    rms_norm::rms_norm_out(
                        &layer_hidden[start..start + dim],
                        attn_norm_w,
                        &mut layer_norm[start..start + dim],
                        rms_norm_eps,
                    );
                }
            }
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn project_attention_batch(
        spec: &Gemma4LayerSpec,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        layer_norm: &[f32],
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        n_tokens: usize,
        dim: usize,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{}.attn_q.weight", spec.prefix))?;
        let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{}.attn_k.weight", spec.prefix))?;

        if spec.v_equals_k {
            timed!(ops, matmul, {
                backend.dequant_matmul_token_major(
                    wq_raw, wq_dtype, layer_norm, q_batch, n_tokens, spec.q_dim, dim,
                );
                backend.dequant_matmul_token_major(
                    wk_raw, wk_dtype, layer_norm, k_batch, n_tokens, spec.kv_dim, dim,
                );
            });
            v_batch.copy_from_slice(k_batch);
        } else {
            let (wv_raw, wv_dtype) =
                weights.raw_with_dtype(&format!("{}.attn_v.weight", spec.prefix))?;
            timed!(ops, matmul, {
                backend.dequant_matmul_token_major(
                    wq_raw, wq_dtype, layer_norm, q_batch, n_tokens, spec.q_dim, dim,
                );
                backend.dequant_matmul_token_major(
                    wk_raw, wk_dtype, layer_norm, k_batch, n_tokens, spec.kv_dim, dim,
                );
                backend.dequant_matmul_token_major(
                    wv_raw, wv_dtype, layer_norm, v_batch, n_tokens, spec.kv_dim, dim,
                );
            });
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_qk_and_v_norm_batch(
        spec: &Gemma4LayerSpec,
        weights: &WeightStore,
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        n_tokens: usize,
        rms_norm_eps: f32,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let n_heads = spec.q_dim / spec.head_dim;
        let qk_norm_weights = timed!(
            ops,
            dequant,
            crate::model::shared::maybe_attention_qk_norm_weights(weights, &spec.prefix)?
        );
        if let Some(norm_weights) = qk_norm_weights {
            timed!(ops, norm, {
                if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
                    q_batch
                        .par_chunks_mut(spec.q_dim)
                        .zip(k_batch.par_chunks_mut(spec.kv_dim))
                        .for_each(|(q_row, k_row)| {
                            crate::model::shared::apply_attention_qk_norm(
                                q_row,
                                k_row,
                                n_heads,
                                spec.n_kv_heads,
                                spec.head_dim,
                                norm_weights,
                                rms_norm_eps,
                            );
                        });
                } else {
                    for t in 0..n_tokens {
                        crate::model::shared::apply_attention_qk_norm(
                            &mut q_batch[t * spec.q_dim..(t + 1) * spec.q_dim],
                            &mut k_batch[t * spec.kv_dim..(t + 1) * spec.kv_dim],
                            n_heads,
                            spec.n_kv_heads,
                            spec.head_dim,
                            norm_weights,
                            rms_norm_eps,
                        );
                    }
                }
            });
        }

        timed!(ops, norm, {
            if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
                v_batch.par_chunks_mut(spec.head_dim).for_each(|v_head| {
                    rms_norm::rms_norm_no_weight(v_head, rms_norm_eps);
                });
            } else {
                for v_head in v_batch.chunks_exact_mut(spec.head_dim) {
                    rms_norm::rms_norm_no_weight(v_head, rms_norm_eps);
                }
            }
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_rope_batch(
        spec: &Gemma4LayerSpec,
        config: &ModelConfig,
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        n_tokens: usize,
        base_seq_len: usize,
        rope_freq_factors: Option<&[f32]>,
        mut ops: Option<&mut OpBreakdown>,
    ) {
        let n_heads = spec.q_dim / spec.head_dim;
        timed!(ops, rope, {
            if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
                q_batch
                    .par_chunks_mut(spec.q_dim)
                    .zip(k_batch.par_chunks_mut(spec.kv_dim))
                    .enumerate()
                    .for_each(|(t, (q_row, k_row))| {
                        let position = base_seq_len + t;
                        let rope_position = if spec.is_sliding() {
                            position as f32
                        } else {
                            config.rope_scaling.scaled_position(position)
                        };
                        if !spec.is_sliding() {
                            if let Some(freq_factors) = rope_freq_factors {
                                rope::apply_rope_neox_with_freq_factors(
                                    q_row,
                                    k_row,
                                    n_heads,
                                    spec.n_kv_heads,
                                    spec.head_dim,
                                    rope_position,
                                    spec.rope_base,
                                    freq_factors,
                                );
                            } else {
                                rope::apply_rope_multi_head_neox_partial_scaled(
                                    q_row,
                                    k_row,
                                    n_heads,
                                    spec.n_kv_heads,
                                    spec.head_dim,
                                    spec.head_dim,
                                    rope_position,
                                    spec.rope_base,
                                );
                            }
                        } else {
                            rope::apply_rope_multi_head_neox_partial_scaled(
                                q_row,
                                k_row,
                                n_heads,
                                spec.n_kv_heads,
                                spec.head_dim,
                                spec.head_dim,
                                rope_position,
                                spec.rope_base,
                            );
                        }
                    });
            } else {
                for t in 0..n_tokens {
                    let position = base_seq_len + t;
                    let rope_position = if spec.is_sliding() {
                        position as f32
                    } else {
                        config.rope_scaling.scaled_position(position)
                    };
                    let q_row = &mut q_batch[t * spec.q_dim..(t + 1) * spec.q_dim];
                    let k_row = &mut k_batch[t * spec.kv_dim..(t + 1) * spec.kv_dim];
                    if !spec.is_sliding() {
                        if let Some(freq_factors) = rope_freq_factors {
                            rope::apply_rope_neox_with_freq_factors(
                                q_row,
                                k_row,
                                n_heads,
                                spec.n_kv_heads,
                                spec.head_dim,
                                rope_position,
                                spec.rope_base,
                                freq_factors,
                            );
                        } else {
                            rope::apply_rope_multi_head_neox_partial_scaled(
                                q_row,
                                k_row,
                                n_heads,
                                spec.n_kv_heads,
                                spec.head_dim,
                                spec.head_dim,
                                rope_position,
                                spec.rope_base,
                            );
                        }
                    } else {
                        rope::apply_rope_multi_head_neox_partial_scaled(
                            q_row,
                            k_row,
                            n_heads,
                            spec.n_kv_heads,
                            spec.head_dim,
                            spec.head_dim,
                            rope_position,
                            spec.rope_base,
                        );
                    }
                }
            }
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn with_attention_prefix<R>(
        spec: &Gemma4LayerSpec,
        kv: &ModelKv,
        base_seq_len: usize,
        kv_stride: usize,
        prefix_k_scratch: &mut Vec<f32>,
        prefix_v_scratch: &mut Vec<f32>,
        _padded_k_scratch: &mut Vec<f32>,
        _padded_v_scratch: &mut Vec<f32>,
        f: impl FnOnce(&[f32], &[f32], usize) -> R,
    ) -> R {
        let (prefix_read_len, prefix_start) = spec.prefix_read_plan(base_seq_len);
        if prefix_read_len == 0 {
            return f(&[], &[], 0);
        }

        match kv {
            ModelKv::Cpu(cpu_kv) => {
                let k_range = &cpu_kv.k_slice(spec.layer, base_seq_len)
                    [prefix_start * kv_stride..base_seq_len * kv_stride];
                let v_range = &cpu_kv.v_slice(spec.layer, base_seq_len)
                    [prefix_start * kv_stride..base_seq_len * kv_stride];
                if spec.kv_dim == kv_stride {
                    f(k_range, v_range, prefix_read_len)
                } else {
                    prefix_k_scratch.resize(prefix_read_len * spec.kv_dim, 0.0);
                    prefix_v_scratch.resize(prefix_read_len * spec.kv_dim, 0.0);
                    Self::copy_rows(
                        k_range,
                        kv_stride,
                        prefix_k_scratch,
                        spec.kv_dim,
                        spec.kv_dim,
                        prefix_read_len,
                    );
                    Self::copy_rows(
                        v_range,
                        kv_stride,
                        prefix_v_scratch,
                        spec.kv_dim,
                        spec.kv_dim,
                        prefix_read_len,
                    );
                    f(prefix_k_scratch, prefix_v_scratch, prefix_read_len)
                }
            }
            ModelKv::Gpu(gpu_kv) => {
                prefix_k_scratch.resize(prefix_read_len * spec.kv_dim, 0.0);
                prefix_v_scratch.resize(prefix_read_len * spec.kv_dim, 0.0);
                gpu_kv.read_layer_range_into(
                    spec.layer,
                    prefix_start,
                    prefix_read_len,
                    prefix_k_scratch,
                    prefix_v_scratch,
                );
                f(prefix_k_scratch, prefix_v_scratch, prefix_read_len)
            }
            ModelKv::Qwen35(_) => unreachable!("gemma4 does not support qwen35 kv"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention_batch(
        spec: &Gemma4LayerSpec,
        backend: &dyn crate::backend::Backend,
        prefix_k: &[f32],
        prefix_v: &[f32],
        prefix_read_len: usize,
        q_batch: &mut [f32],
        k_batch: &[f32],
        v_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        mut ops: Option<&mut OpBreakdown>,
    ) {
        let n_heads = spec.q_dim / spec.head_dim;
        if spec.use_backend_prefill(prefix_read_len, n_tokens) {
            let q_attn_compensation = (spec.head_dim as f32).sqrt();
            if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
                q_batch
                    .par_chunks_mut(Self::PARALLEL_FLOAT_CHUNK)
                    .for_each(|chunk| {
                        for value in chunk {
                            *value *= q_attn_compensation;
                        }
                    });
            } else {
                for value in q_batch.iter_mut() {
                    *value *= q_attn_compensation;
                }
            }
            timed!(ops, attention, backend.attention_prefill(
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                n_heads,
                spec.n_kv_heads,
                spec.head_dim,
            ));
        } else {
            let attn_params =
                attention::AttentionParams::new(n_heads, spec.n_kv_heads, spec.head_dim);
            timed!(
                ops,
                attention,
                attention::multi_head_attention_prefill_with_prefix_scaled_window(
                    prefix_k,
                    prefix_v,
                    prefix_read_len,
                    q_batch,
                    k_batch,
                    v_batch,
                    attn_out_batch,
                    n_tokens,
                    &attn_params,
                    1.0,
                    spec.local_window,
                )
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_run_attention_batch_gpu_cached(
        spec: &Gemma4LayerSpec,
        ctx: &ForwardContext,
        kv: &mut ModelKv,
        base_seq_len: usize,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        _kv_stride: usize,
        _k_padded_batch: &mut [f32],
        _v_padded_batch: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<bool> {
        if base_seq_len == 0 {
            return Ok(false);
        }
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(false);
        };
        let Some(gpu_kv) = kv.as_gpu_mut() else {
            return Ok(false);
        };
        if gpu_kv.is_q8() {
            return Ok(false);
        }

        debug_assert_eq!(q_batch.len(), n_tokens * spec.q_dim);
        debug_assert_eq!(k_batch.len(), n_tokens * spec.kv_dim);
        debug_assert_eq!(v_batch.len(), n_tokens * spec.kv_dim);
        debug_assert_eq!(attn_out_batch.len(), n_tokens * spec.q_dim);

        gpu_kv.ensure_capacity(&metal_ops.device, base_seq_len + n_tokens)?;
        metal_ops.init_batch_scratches(ctx.config, n_tokens);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        unsafe {
            bs.q_buf.as_mut_slice::<f32>()[..n_tokens * spec.q_dim].copy_from_slice(q_batch);
        }

        gpu_kv.append_layer_batch(spec.layer, k_batch, v_batch, n_tokens);

        let attention_t = OpTimer::start();
        metal_ops.device.execute_sync(|encoder| {
            let n_heads = (spec.q_dim / spec.head_dim) as u32;
            let n_tokens_u32 = n_tokens as u32;
            let sliding_window = spec.local_window.unwrap_or(0) as u32;
            metal_ops
                .attention
                .encode_attention_prefill_cached_with_config(
                    encoder,
                    &bs.q_buf,
                    gpu_kv.k_buffer(spec.layer),
                    gpu_kv.v_buffer(spec.layer),
                    &bs.attn_out,
                    gpu_kv.is_f16(),
                    n_tokens_u32,
                    n_heads,
                    spec.n_kv_heads as u32,
                    spec.head_dim as u32,
                    base_seq_len as u32,
                    sliding_window,
                    metal_ops.attention_dispatch_config(),
                );
            Ok(())
        })?;

        unsafe {
            attn_out_batch[..n_tokens * spec.q_dim]
                .copy_from_slice(&bs.attn_out.as_slice::<f32>()[..n_tokens * spec.q_dim]);
        }
        if let Some(ops_ref) = ops.as_mut() {
            ops_ref.attention += attention_t.elapsed();
        }
        drop(bs_guard);
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn append_attention_kv_batch(
        spec: &Gemma4LayerSpec,
        kv: &mut ModelKv,
        k_batch: &[f32],
        v_batch: &[f32],
        k_padded_batch: &mut [f32],
        v_padded_batch: &mut [f32],
        n_tokens: usize,
        kv_stride: usize,
    ) {
        if matches!(kv, ModelKv::Gpu(_)) || spec.kv_dim == kv_stride {
            Self::append_layer_batch(kv, spec.layer, k_batch, v_batch, n_tokens);
        } else {
            k_padded_batch.fill(0.0);
            v_padded_batch.fill(0.0);
            Self::copy_rows(k_batch, spec.kv_dim, k_padded_batch, kv_stride, spec.kv_dim, n_tokens);
            Self::copy_rows(v_batch, spec.kv_dim, v_padded_batch, kv_stride, spec.kv_dim, n_tokens);
            Self::append_layer_batch(kv, spec.layer, k_padded_batch, v_padded_batch, n_tokens);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_and_ffn_batch(
        spec: &Gemma4LayerSpec,
        ctx: &ForwardContext,
        weights: &WeightStore,
        layer_hidden: &mut [f32],
        layer_norm: &mut [f32],
        attn_out_batch: &[f32],
        proj_batch: &mut [f32],
        gate_batch: &mut [f32],
        up_batch: &mut [f32],
        down_batch: &mut [f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let (wo_raw, wo_dtype) =
            weights.raw_with_dtype(&format!("{}.attn_output.weight", spec.prefix))?;
        timed!(ops, matmul, ctx.backend.dequant_matmul_token_major(
            wo_raw,
            wo_dtype,
            attn_out_batch,
            proj_batch,
            n_tokens,
            dim,
            spec.q_dim,
        ));

        if weights.has(&format!("{}.post_attention_norm.weight", spec.prefix)) {
            let post_attn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{}.post_attention_norm.weight", spec.prefix))?
            );
            timed!(ops, norm, {
                if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
                    proj_batch[..n_tokens * dim]
                        .par_chunks_mut(dim)
                        .for_each(|proj_token| {
                            rms_norm::rms_norm(proj_token, post_attn_norm_w, rms_norm_eps);
                        });
                } else {
                    for t in 0..n_tokens {
                        let start = t * dim;
                        rms_norm::rms_norm(
                            &mut proj_batch[start..start + dim],
                            post_attn_norm_w,
                            rms_norm_eps,
                        );
                    }
                }
            });
        }

        Self::parallel_elementwise_add(layer_hidden, proj_batch);

        let ffn_norm_w = weights.f32_slice(&format!("{}.ffn_norm.weight", spec.prefix))?;
        crate::model::layer_ops::apply_ffn_batch(
            ctx.backend,
            weights,
            &spec.prefix,
            layer_hidden,
            layer_norm,
            gate_batch,
            up_batch,
            down_batch,
            n_tokens,
            dim,
            inter_dim,
            ffn_norm_w,
            rms_norm_eps,
            crate::model::layer_ops::FfnActivation::GELU,
        );

        if weights.has(&format!("{}.layer_output_scale.weight", spec.prefix)) {
            let scale = weights.f32_slice(&format!("{}.layer_output_scale.weight", spec.prefix))?[0];
            Self::parallel_scale(layer_hidden, scale);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_and_moe_ffn_batch(
        spec: &Gemma4LayerSpec,
        ctx: &ForwardContext,
        weights: &WeightStore,
        layer_hidden: &mut [f32],
        layer_norm: &mut [f32],
        attn_out_batch: &[f32],
        proj_batch: &mut [f32],
        gate_batch: &mut [f32],
        up_batch: &mut [f32],
        down_batch: &mut [f32],
        moe_norm_buf: &mut [f32],
        moe_accum_buf: &mut [f32],
        moe_scratch: &mut MoeScratch,
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        config: &ModelConfig,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        debug_assert!(spec.has_moe, "Gemma4 MoE helper called for dense layer");

        // ── 1. Output projection ──
        let (wo_raw, wo_dtype) =
            weights.raw_with_dtype(&format!("{}.attn_output.weight", spec.prefix))?;
        timed!(ops, matmul, ctx.backend.dequant_matmul_token_major(
            wo_raw,
            wo_dtype,
            attn_out_batch,
            proj_batch,
            n_tokens,
            dim,
            spec.q_dim,
        ));

        // ── 2. Post-attention RMSNorm ──
        if weights.has(&format!("{}.post_attention_norm.weight", spec.prefix)) {
            let post_attn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{}.post_attention_norm.weight", spec.prefix))?
            );
            timed!(ops, norm, {
                for t in 0..n_tokens {
                    let start = t * dim;
                    rms_norm::rms_norm(
                        &mut proj_batch[start..start + dim],
                        post_attn_norm_w,
                        rms_norm_eps,
                    );
                }
            });
        }

        // ── 3. Residual add (attention) ──
        Self::parallel_elementwise_add(layer_hidden, proj_batch);

        let hidden_len = n_tokens * dim;
        let n_expert = config.n_expert.unwrap_or(0) as usize;
        let n_expert_used = config.n_expert_used.unwrap_or(0) as usize;
        let expert_inter_dim = config.expert_intermediate_dim.unwrap_or(0) as usize;
        anyhow::ensure!(
            n_expert > 0 && n_expert_used > 0 && expert_inter_dim > 0,
            "Gemma4 MoE layer {} missing MoE dimensions in config",
            spec.layer,
        );

        // Save the post-attention residual state. Both the shared FFN and the
        // routed experts branch from this same tensor.
        let attn_out_hidden = &mut moe_norm_buf[..hidden_len];
        attn_out_hidden.copy_from_slice(&layer_hidden[..hidden_len]);

        // ── 4. Shared FFN branch: ffn_norm -> gate/up -> GELU*mul -> down -> post_ffw_norm_1 ──
        let ffn_norm_w = weights.f32_slice(&format!("{}.ffn_norm.weight", spec.prefix))?;
        timed!(ops, norm, {
            if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
                attn_out_hidden
                    .par_chunks(dim)
                    .zip(layer_norm[..hidden_len].par_chunks_mut(dim))
                    .for_each(|(src, dst)| rms_norm::rms_norm_out(src, ffn_norm_w, dst, rms_norm_eps));
            } else {
                for t in 0..n_tokens {
                    let start = t * dim;
                    rms_norm::rms_norm_out(
                        &attn_out_hidden[start..start + dim],
                        ffn_norm_w,
                        &mut layer_norm[start..start + dim],
                        rms_norm_eps,
                    );
                }
            }
        });

        let (wg_raw, wg_dtype) = weights.raw_with_dtype(&format!("{}.ffn_gate.weight", spec.prefix))?;
        let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{}.ffn_up.weight", spec.prefix))?;
        timed!(ops, matmul, {
            ctx.backend.dequant_matmul_token_major(
                wg_raw,
                wg_dtype,
                layer_norm,
                gate_batch,
                n_tokens,
                inter_dim,
                dim,
            );
            ctx.backend.dequant_matmul_token_major(
                wu_raw,
                wu_dtype,
                layer_norm,
                up_batch,
                n_tokens,
                inter_dim,
                dim,
            );
        });
        if n_tokens >= Self::PARALLEL_BATCH_MIN_TOKENS {
            gate_batch
                .par_chunks_mut(Self::PARALLEL_FLOAT_CHUNK)
                .zip(up_batch.par_chunks(Self::PARALLEL_FLOAT_CHUNK))
                .for_each(|(gate_chunk, up_chunk)| {
                    crate::compute::gelu::gelu_elementwise_mul(gate_chunk, up_chunk);
                });
        } else {
            crate::compute::gelu::gelu_elementwise_mul(gate_batch, up_batch);
        }

        let (wd_raw, wd_dtype) = weights.raw_with_dtype(&format!("{}.ffn_down.weight", spec.prefix))?;
        timed!(ops, matmul, {
            ctx.backend.dequant_matmul_token_major(
                wd_raw,
                wd_dtype,
                gate_batch,
                proj_batch,
                n_tokens,
                dim,
                inter_dim,
            );
        });
        if let Ok(post_ff1_w) = weights.f32_slice(&format!("{}.post_ffw_norm_1.weight", spec.prefix)) {
            timed!(
                ops,
                norm,
                Self::apply_row_rms_norm(proj_batch, post_ff1_w, n_tokens, dim, rms_norm_eps)
            );
        }

        // ── 5. Routed experts branch: router(attn_out) + pre_ffw_norm_2 -> experts -> post_ffw_norm_2 ──
        let router_scale = weights.f32_slice(&format!("{}.ffn_gate_inp.scale", spec.prefix))?;
        let pre_ff2_w = weights.f32_slice(&format!("{}.pre_ffw_norm_2.weight", spec.prefix))?;
        let post_ff2_w = weights.f32_slice(&format!("{}.post_ffw_norm_2.weight", spec.prefix))?;
        let expert_scales = weights.f32_slice(&format!("{}.ffn_down_exps.scale", spec.prefix))?;
        let (router_raw, router_dtype) =
            weights.raw_with_dtype(&format!("{}.ffn_gate_inp.weight", spec.prefix))?;
        let gate_up_name = format!("{}.ffn_gate_up_exps.weight", spec.prefix);
        let down_name = format!("{}.ffn_down_exps.weight", spec.prefix);
        let (gate_up_raw, gate_up_dtype) = weights.raw_with_dtype(&gate_up_name)?;
        let (down_raw, down_dtype) = weights.raw_with_dtype(&down_name)?;
        let fused_dim = 2 * expert_inter_dim;
        let gate_up_stride =
            crate::model::moe_utils::expert_byte_stride(gate_up_dtype, fused_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
        let router_input_scale = (dim as f32).sqrt().recip();
        let cpu = crate::backend::cpu::CpuBackend;
        let moe_accum_slice = &mut moe_accum_buf[..hidden_len];
        moe_accum_slice.fill(0.0);
        let use_gpu_routed_expert_dispatch = ctx.backend.metal_ops().is_some()
            && crate::model::shared::moe_routed_expert_dtype_supported(gate_up_dtype)
            && crate::model::shared::moe_routed_expert_dtype_supported(down_dtype);
        let MoeScratch {
            logits: moe_logits,
            fused: fused_scratch,
            down: down_scratch,
            expert_ids: scratch_expert_ids,
            expert_weights: scratch_expert_weights,
        } = moe_scratch;
        let expert_ids = &mut scratch_expert_ids[..n_tokens * n_expert_used];
        let expert_weights = &mut scratch_expert_weights[..n_tokens * n_expert_used];
        expert_ids.fill(-1);
        expert_weights.fill(0.0);

        for t in 0..n_tokens {
            let start = t * dim;
            let end = start + dim;
            let attn_token = &attn_out_hidden[start..end];
            let expert_input = &mut layer_norm[start..end];
            rms_norm::rms_norm_out(attn_token, pre_ff2_w, expert_input, rms_norm_eps);

            let router_input = &mut down_batch[start..end];
            router_input.copy_from_slice(attn_token);
            rms_norm::rms_norm_no_weight(router_input, rms_norm_eps);
            for (value, &scale) in router_input.iter_mut().zip(router_scale.iter()) {
                *value *= router_input_scale * scale;
            }

            let router_logits = &mut moe_logits[..n_expert];
            timed!(ops, matmul, cpu.dequant_matmul(
                router_raw,
                router_dtype,
                router_input,
                router_logits,
                n_expert,
                1,
                dim,
            ));

            let (top_indices, mut top_weights) =
                crate::model::moe_utils::top_k_softmax(router_logits, n_expert_used);
            for (weight, &expert_idx) in top_weights.iter_mut().zip(top_indices.iter()) {
                *weight *= expert_scales[expert_idx];
            }
            let slot_base = t * n_expert_used;
            for (slot_idx, (&expert_idx, &weight)) in top_indices.iter().zip(top_weights.iter()).enumerate()
            {
                expert_ids[slot_base + slot_idx] = expert_idx as i32;
                expert_weights[slot_base + slot_idx] = weight;
            }
            if use_gpu_routed_expert_dispatch {
                continue;
            }

            let token_accum = &mut moe_accum_slice[start..end];
            for (slot, &weight) in top_indices.iter().zip(top_weights.iter()) {
                let fused_buf = &mut fused_scratch[..fused_dim];
                let expert_gate_up = crate::model::moe_utils::expert_quant_slice(
                    gate_up_raw,
                    gate_up_stride,
                    *slot,
                    &gate_up_name,
                )?;
                timed!(ops, matmul, cpu.dequant_matmul(
                    expert_gate_up,
                    gate_up_dtype,
                    expert_input,
                    fused_buf,
                    fused_dim,
                    1,
                    dim,
                ));

                let (gate_half, up_half) = fused_buf.split_at_mut(expert_inter_dim);
                crate::compute::gelu::gelu_elementwise_mul(gate_half, up_half);

                let expert_down = &mut down_scratch[..dim];
                let expert_down_slice = crate::model::moe_utils::expert_quant_slice(
                    down_raw,
                    down_stride,
                    *slot,
                    &down_name,
                )?;
                timed!(ops, matmul, cpu.dequant_matmul(
                    expert_down_slice,
                    down_dtype,
                    gate_half,
                    expert_down,
                    dim,
                    1,
                    expert_inter_dim,
                ));

                for (acc, &value) in token_accum.iter_mut().zip(expert_down.iter()) {
                    *acc += weight * value;
                }
            }
        }
        if use_gpu_routed_expert_dispatch && let Some(metal_ops) = ctx.backend.metal_ops() {
            timed!(ops, gpu, metal_ops.moe_fused_gate_up_gelu_dispatch(
                &layer_norm[..hidden_len],
                moe_accum_slice,
                expert_ids,
                expert_weights,
                gate_up_raw,
                down_raw,
                gate_up_dtype,
                down_dtype,
                n_tokens,
                n_expert,
                n_expert_used,
                dim,
                expert_inter_dim,
                gate_up_stride,
                down_stride,
            ))?;
        }

        timed!(
            ops,
            norm,
            Self::apply_row_rms_norm(moe_accum_slice, post_ff2_w, n_tokens, dim, rms_norm_eps)
        );

        // ── 6. Combine branches -> post_ffw_norm -> residual ──
        Self::parallel_elementwise_add(proj_batch, moe_accum_slice);
        if let Ok(post_ff_w) = weights.f32_slice(&format!("{}.post_ffw_norm.weight", spec.prefix)) {
            timed!(
                ops,
                norm,
                Self::apply_row_rms_norm(proj_batch, post_ff_w, n_tokens, dim, rms_norm_eps)
            );
        }
        Self::parallel_elementwise_add(layer_hidden, proj_batch);

        // ── 7. Layer output scale ──
        if weights.has(&format!("{}.layer_output_scale.weight", spec.prefix)) {
            let scale = weights.f32_slice(&format!("{}.layer_output_scale.weight", spec.prefix))?[0];
            Self::parallel_scale(layer_hidden, scale);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_and_moe_ffn_single(
        ctx: &ForwardContext,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        config: &ModelConfig,
        moe_scratch: &mut Gemma4SingleMoeScratch,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let attn_out_hidden = hidden.to_vec();
        let n_expert = config.n_expert.unwrap_or(0) as usize;
        let n_expert_used = config.n_expert_used.unwrap_or(0) as usize;
        let expert_inter_dim = config.expert_intermediate_dim.unwrap_or(0) as usize;
        anyhow::ensure!(
            n_expert > 0 && n_expert_used > 0 && expert_inter_dim > 0,
            "Gemma4 MoE layer {prefix} missing MoE dimensions in config",
        );

        // Shared FFN branch: ffn_norm -> gate/up -> GELU*mul -> down -> post_ffw_norm_1.
        let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
        timed!(ops, norm, {
            rms_norm::rms_norm_out(&attn_out_hidden, ffn_norm_w, norm_buf, rms_norm_eps)
        });
        let (wg_raw, wg_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
        let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
        timed!(ops, matmul, {
            ctx.backend.batch_dequant_matvec(
                &[(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)],
                norm_buf,
                dim,
                &mut [gate_buf, up_buf],
            );
        });
        crate::compute::gelu::gelu_elementwise_mul(&mut gate_buf[..inter_dim], &up_buf[..inter_dim]);

        let (wd_raw, wd_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
        timed!(ops, matmul, {
            ctx.backend.dequant_matmul(
                wd_raw,
                wd_dtype,
                &gate_buf[..inter_dim],
                down_buf,
                dim,
                1,
                inter_dim,
            );
        });
        if let Ok(post_ff1_w) = weights.f32_slice(&format!("{prefix}.post_ffw_norm_1.weight")) {
            timed!(ops, norm, rms_norm::rms_norm(down_buf, post_ff1_w, rms_norm_eps));
        }

        // Routed experts branch: router(attn_out) + pre_ffw_norm_2 -> experts -> post_ffw_norm_2.
        let router_scale = weights.f32_slice(&format!("{prefix}.ffn_gate_inp.scale"))?;
        let pre_ff2_w = weights.f32_slice(&format!("{prefix}.pre_ffw_norm_2.weight"))?;
        let post_ff2_w = weights.f32_slice(&format!("{prefix}.post_ffw_norm_2.weight"))?;
        let expert_scales = weights.f32_slice(&format!("{prefix}.ffn_down_exps.scale"))?;
        let (router_raw, router_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ffn_gate_inp.weight"))?;
        let gate_up_name = format!("{prefix}.ffn_gate_up_exps.weight");
        let down_name = format!("{prefix}.ffn_down_exps.weight");
        let (gate_up_raw, gate_up_dtype) = weights.raw_with_dtype(&gate_up_name)?;
        let (down_raw, down_dtype) = weights.raw_with_dtype(&down_name)?;
        let fused_dim = 2 * expert_inter_dim;
        let gate_up_stride =
            crate::model::moe_utils::expert_byte_stride(gate_up_dtype, fused_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
        let router_input_scale = (dim as f32).sqrt().recip();
        let cpu = crate::backend::cpu::CpuBackend;
        let routed_accum = &mut moe_scratch.expert_accum[..dim];
        routed_accum.fill(0.0);

        timed!(ops, norm, {
            rms_norm::rms_norm_out(&attn_out_hidden, pre_ff2_w, norm_buf, rms_norm_eps)
        });
        moe_scratch
            .router_input
            .copy_from_slice(&attn_out_hidden);
        rms_norm::rms_norm_no_weight(&mut moe_scratch.router_input, rms_norm_eps);
        for (value, &scale) in moe_scratch.router_input.iter_mut().zip(router_scale.iter()) {
            *value *= router_input_scale * scale;
        }

        let router_logits = &mut moe_scratch.router_logits[..n_expert];
        timed!(ops, matmul, {
            cpu.dequant_matmul(
                router_raw,
                router_dtype,
                &moe_scratch.router_input,
                router_logits,
                n_expert,
                1,
                dim,
            )
        });
        let (expert_ids, mut expert_weights) =
            crate::model::moe_utils::top_k_softmax(router_logits, n_expert_used);
        for (weight, &expert_idx) in expert_weights.iter_mut().zip(expert_ids.iter()) {
            *weight *= expert_scales[expert_idx];
        }

        for (&expert_idx, &weight) in expert_ids.iter().zip(expert_weights.iter()) {
            let expert_gate_up = crate::model::moe_utils::expert_quant_slice(
                gate_up_raw,
                gate_up_stride,
                expert_idx,
                &gate_up_name,
            )?;
            timed!(ops, matmul, {
                cpu.dequant_matmul(
                    expert_gate_up,
                    gate_up_dtype,
                    norm_buf,
                    &mut moe_scratch.fused[..fused_dim],
                    fused_dim,
                    1,
                    dim,
                )
            });
            let (gate_half, up_half) =
                moe_scratch.fused[..fused_dim].split_at_mut(expert_inter_dim);
            crate::compute::gelu::gelu_elementwise_mul(gate_half, up_half);

            let expert_down = crate::model::moe_utils::expert_quant_slice(
                down_raw,
                down_stride,
                expert_idx,
                &down_name,
            )?;
            timed!(ops, matmul, {
                cpu.dequant_matmul(
                    expert_down,
                    down_dtype,
                    gate_half,
                    up_buf,
                    dim,
                    1,
                    expert_inter_dim,
                )
            });

            for (acc, &value) in routed_accum.iter_mut().zip(up_buf.iter()) {
                *acc += weight * value;
            }
        }
        timed!(ops, norm, rms_norm::rms_norm(routed_accum, post_ff2_w, rms_norm_eps));

        // Combine shared + routed branches -> post_ffw_norm -> residual.
        Self::parallel_elementwise_add(down_buf, routed_accum);
        if let Ok(post_ff_w) = weights.f32_slice(&format!("{prefix}.post_ffw_norm.weight")) {
            timed!(ops, norm, rms_norm::rms_norm(down_buf, post_ff_w, rms_norm_eps));
        }
        silu::elementwise_add(hidden, down_buf);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_batch_last_logits(
        ctx: &ForwardContext,
        weights: &WeightStore,
        hidden: &mut [f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        config: &ModelConfig,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        assert!(logits.len() >= vocab_size);
        debug_assert!(n_tokens > 0, "n_tokens must be > 0 for last-token logit extraction");
        logits.fill(0.0);
        let last_hidden = &mut hidden[(n_tokens - 1) * dim..n_tokens * dim];
        apply_output_norm_single(weights, last_hidden, config.rms_norm_eps, ops.as_deref_mut())?;
        write_normalized_single_logits_with_breakdown(
            ctx.backend,
            last_hidden,
            dim,
            vocab_size,
            weights,
            logits,
            ops,
        )?;
        if let Some(cap) = config.final_logit_softcapping {
            for l in logits[..vocab_size].iter_mut() {
                *l = (*l / cap).tanh() * cap;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_token_major(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        if token_ids.is_empty() {
            return Ok(());
        }

        if let ModelKv::Gpu(gpu_kv) = kv {
            let Some(metal_ops) = ctx.backend.metal_ops() else {
                anyhow::bail!("gemma4 gpu kv batch prefill requires metal ops");
            };
            gpu_kv.ensure_capacity(&metal_ops.device, gpu_kv.seq_len() + token_ids.len())?;
        }

        let scratch_target_bytes = if matches!(kv, ModelKv::Gpu(_)) {
            Self::GPU_KV_BATCH_SCRATCH_TARGET_BYTES
        } else {
            Self::CPU_BATCH_SCRATCH_TARGET_BYTES
        };
        let chunk_len =
            Self::cpu_batch_chunk_len(ctx.config, token_ids.len(), scratch_target_bytes);
        let layer_specs: Vec<_> = (0..ctx.config.n_layers as usize)
            .map(|layer| Gemma4LayerSpec::new(layer, ctx.config, weights))
            .collect();
        let rope_freq_factors: Option<&[f32]> = if weights.has("rope_freqs.weight") {
            Some(timed!(ops, dequant, weights.f32_slice("rope_freqs.weight")?))
        } else {
            None
        };
        let mut scratch = Gemma4BatchScratch::new(ctx.config, chunk_len);
        let mut chunk_start = 0usize;
        while chunk_start < token_ids.len() {
            let chunk_end = (chunk_start + chunk_len).min(token_ids.len());
            let write_logits = chunk_end == token_ids.len();
            self.forward_batch_chunk(
                ctx,
                &token_ids[chunk_start..chunk_end],
                kv,
                weights,
                &layer_specs,
                rope_freq_factors,
                &mut scratch,
                if write_logits { Some(logits) } else { None },
                ops.as_deref_mut(),
            )?;
            chunk_start = chunk_end;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_chunk(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        layer_specs: &[Gemma4LayerSpec],
        rope_freq_factors: Option<&[f32]>,
        scratch: &mut Gemma4BatchScratch,
        logits: Option<&mut [f32]>,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        if token_ids.is_empty() {
            return Ok(());
        }

        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let vocab_size = cfg.vocab_size as usize;
        let n_tokens = token_ids.len();
        let hidden_len = n_tokens * dim;
        let inter_dim = cfg.intermediate_dim as usize;
        let swa_kv_dim = cfg.gemma4_n_kv_heads_swa.unwrap_or(cfg.n_kv_heads) as usize
            * cfg.gemma4_head_dim_swa.unwrap_or(cfg.head_dim) as usize;
        let global_kv_dim = cfg.gemma4_n_kv_heads_global.unwrap_or(cfg.n_kv_heads) as usize
            * cfg.gemma4_head_dim_global.unwrap_or(cfg.head_dim) as usize;
        let kv_stride = swa_kv_dim.max(global_kv_dim);
        let base_seq_len = kv.seq_len();

        let hidden = &mut scratch.hidden[..hidden_len];
        Self::embed_batch_hidden(token_ids, hidden, dim, weights, cfg, ops.as_deref_mut())?;

        let norm_buf = &mut scratch.norm_buf[..hidden_len];
        let proj_buf = &mut scratch.proj_buf[..hidden_len];
        let down_buf = &mut scratch.down_buf[..hidden_len];
        let gate_buf = &mut scratch.gate_buf[..n_tokens * inter_dim];
        let up_buf = &mut scratch.up_buf[..n_tokens * inter_dim];
        let k_padded_batch = &mut scratch.k_padded_batch[..n_tokens * kv_stride];
        let v_padded_batch = &mut scratch.v_padded_batch[..n_tokens * kv_stride];

        for spec in layer_specs {
            Self::apply_attention_norm_batch(
                spec,
                weights,
                hidden,
                norm_buf,
                n_tokens,
                dim,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;
            let q_batch = &mut scratch.q_buf[..n_tokens * spec.q_dim];
            let k_batch = &mut scratch.k_buf[..n_tokens * spec.kv_dim];
            let v_batch = &mut scratch.v_buf[..n_tokens * spec.kv_dim];
            Self::project_attention_batch(
                spec,
                ctx.backend,
                weights,
                norm_buf,
                q_batch,
                k_batch,
                v_batch,
                n_tokens,
                dim,
                ops.as_deref_mut(),
            )?;
            Self::apply_attention_qk_and_v_norm_batch(
                spec,
                weights,
                q_batch,
                k_batch,
                v_batch,
                n_tokens,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;
            Self::apply_attention_rope_batch(
                spec,
                cfg,
                q_batch,
                k_batch,
                n_tokens,
                base_seq_len,
                rope_freq_factors,
                ops.as_deref_mut(),
            );
            let attn_out_batch = &mut scratch.attn_out[..n_tokens * spec.q_dim];
            let used_gpu_cached_attention = Self::try_run_attention_batch_gpu_cached(
                spec,
                ctx,
                kv,
                base_seq_len,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                kv_stride,
                k_padded_batch,
                v_padded_batch,
                ops.as_deref_mut(),
            )?;
            if !used_gpu_cached_attention {
                Self::with_attention_prefix(
                    spec,
                    &*kv,
                    base_seq_len,
                    kv_stride,
                    &mut scratch.prefix_k_scratch,
                    &mut scratch.prefix_v_scratch,
                    &mut scratch.padded_prefix_k_scratch,
                    &mut scratch.padded_prefix_v_scratch,
                    |prefix_k, prefix_v, prefix_read_len| {
                        Self::run_attention_batch(
                spec,
                ctx.backend,
                prefix_k,
                prefix_v,
                prefix_read_len,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                            n_tokens,
                            ops.as_deref_mut(),
                        );
                    },
                );
                Self::append_attention_kv_batch(
                    spec,
                    kv,
                    k_batch,
                    v_batch,
                    k_padded_batch,
                    v_padded_batch,
                    n_tokens,
                    kv_stride,
                );
            }
            if spec.has_moe {
                Self::apply_post_attention_and_moe_ffn_batch(
                    spec,
                    ctx,
                    weights,
                    hidden,
                    norm_buf,
                    attn_out_batch,
                    proj_buf,
                    gate_buf,
                    up_buf,
                    down_buf,
                    &mut scratch.moe_norm_buf,
                    &mut scratch.moe_accum_buf,
                    &mut scratch.moe_scratch,
                    n_tokens,
                    dim,
                    inter_dim,
                    cfg.rms_norm_eps,
                    cfg,
                    ops.as_deref_mut(),
                )?;
            } else {
                Self::apply_post_attention_and_ffn_batch(
                    spec,
                    ctx,
                    weights,
                    hidden,
                    norm_buf,
                    attn_out_batch,
                    proj_buf,
                    gate_buf,
                    up_buf,
                    down_buf,
                    n_tokens,
                    dim,
                    inter_dim,
                    cfg.rms_norm_eps,
                    ops.as_deref_mut(),
                )?;
            }
        }

        Self::finalize_batch(kv, n_tokens);

        if let Some(logits) = logits {
            Self::write_batch_last_logits(
                ctx,
                weights,
                hidden,
                n_tokens,
                dim,
                vocab_size,
                cfg,
                logits,
                ops,
            )?;
        }

        Ok(())
    }
}

impl ForwardPass for Gemma4Forward {
    fn prepare_runtime(&self, ctx: &ForwardContext, weights: &WeightStore) -> anyhow::Result<()> {
        if ctx.backend.use_gpu_decode()
            && let Some(metal_ops) = ctx.backend.metal_ops()
            && gpu_decode_quant_supported(ctx.config, weights)
            && !metal_ops.has_cached_model_keys()
        {
            Self::build_cached_model_keys_gemma4(metal_ops, weights, ctx.config)?;
        }

        Ok(())
    }

    fn forward_single(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let moe_gpu_safe = Self::has_any_moe_layer(ctx.config, weights)
            && !Self::all_layers_q8_0_expert_down(ctx.config, weights);

        if moe_gpu_safe
            && let Some(metal_ops) = ctx.backend.metal_ops()
            && let Some(gpu_kv) = kv.as_gpu_mut()
        {
            if let Some(ops_ref) = ops {
                let t = OpTimer::start();
                let r = self.forward_batch_gpu_unified(
                    ctx,
                    metal_ops,
                    std::slice::from_ref(&token_id),
                    gpu_kv,
                    weights,
                    Some(logits),
                    None,
                    Some(ops_ref),
                );
                ops_ref.gpu += t.elapsed();
                return r;
            }
            return self.forward_batch_gpu_unified(
                ctx,
                metal_ops,
                std::slice::from_ref(&token_id),
                gpu_kv,
                weights,
                Some(logits),
                None,
                None,
            );
        } else if Self::has_any_moe_layer(ctx.config, weights) {
            return self.forward_batch_token_major(
                ctx,
                std::slice::from_ref(&token_id),
                kv,
                weights,
                logits,
                ops,
            );
        }

        if ctx.backend.use_gpu_decode()
            && let Some(metal_ops) = ctx.backend.metal_ops()
            && let Some(gpu_kv) = kv.as_gpu_mut()
            && gpu_decode_quant_supported(ctx.config, weights)
        {
            if let Some(ops_ref) = ops {
                let t = OpTimer::start();
                let r = self.forward_single_gpu_unified(
                    ctx,
                    metal_ops,
                    token_id,
                    position,
                    gpu_kv,
                    weights,
                    logits,
                    Some(ops_ref),
                );
                ops_ref.gpu += t.elapsed();
                return r;
            }
            return self.forward_single_gpu_unified(
                ctx, metal_ops, token_id, position, gpu_kv, weights, logits, None,
            );
        }

        let cfg = ctx.config;
        let dim = cfg.embedding_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let n_heads = cfg.n_heads as usize;
        let vocab_size = cfg.vocab_size as usize;

        // Max dimensions across all layer types (for scratch buffer allocation)
        let max_hd = cfg
            .gemma4_head_dim_global
            .unwrap_or(cfg.head_dim)
            .max(cfg.gemma4_head_dim_swa.unwrap_or(cfg.head_dim))
            as usize;
        let max_q_dim = n_heads * max_hd;
        let max_kv_dim = cfg
            .gemma4_n_kv_heads_swa
            .unwrap_or(cfg.n_kv_heads)
            .max(cfg.gemma4_n_kv_heads_global.unwrap_or(cfg.n_kv_heads))
            as usize
            * max_hd;
        let inter_dim = cfg.intermediate_dim as usize;

        assert!(logits.len() >= vocab_size);

        let cpu_kv = kv
            .as_cpu_mut()
            .expect("Gemma4Forward CPU path requires ModelKv::Cpu");

        // --- Step 1: Token embedding ---
        let mut hidden = vec![0.0f32; dim];
        timed!(
            ops,
            dequant,
            weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?
        );

        // Gemma: scale embeddings by sqrt(embedding_dim)
        if cfg.embed_scale {
            let embd_scale = (dim as f32).sqrt();
            for h in hidden.iter_mut() {
                *h *= embd_scale;
            }
        }

        // Scratch buffers (sized to max across layer types)
        let mut norm_buf = vec![0.0f32; dim];
        let mut q_buf = vec![0.0f32; max_q_dim];
        let mut k_buf = vec![0.0f32; max_kv_dim];
        let mut v_buf = vec![0.0f32; max_kv_dim];
        let mut attn_out = vec![0.0f32; max_q_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; dim];
        let mut single_moe_scratch = if Self::has_any_moe_layer(cfg, weights) {
            Some(Gemma4SingleMoeScratch::new(cfg))
        } else {
            None
        };

        // KV cache stride: max product across SWA and global layer types
        let swa_kv_dim = cfg.gemma4_n_kv_heads_swa.unwrap_or(cfg.n_kv_heads) as usize
            * cfg.gemma4_head_dim_swa.unwrap_or(cfg.head_dim) as usize;
        let global_kv_dim = cfg.gemma4_n_kv_heads_global.unwrap_or(cfg.n_kv_heads) as usize
            * cfg.gemma4_head_dim_global.unwrap_or(cfg.head_dim) as usize;
        let kv_stride = swa_kv_dim.max(global_kv_dim);

        // Pre-allocate KV padding scratch for global layers (avoids per-token
        // heap allocation inside the layer loop — BUG-046).
        let mut k_padded = vec![0.0f32; kv_stride];
        let mut v_padded = vec![0.0f32; kv_stride];

        // Load rope_freqs for proportional RoPE on global layers
        let rope_freq_factors: Option<Vec<f32>> = if weights.has("rope_freqs.weight") {
            Some(weights.f32_slice("rope_freqs.weight")?.to_vec())
        } else {
            None
        };

        // --- Step 2: Transformer layers ---
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let is_local = Self::use_sliding_window(layer, cfg);
            let (head_dim, n_kv_heads, q_dim, kv_dim) = Self::layer_dims(layer, cfg);
            let v_equals_k = Self::layer_v_equals_k(layer, cfg, weights);

            // 2a. Pre-attention RMSNorm
            apply_attention_norm_single(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;

            // 2b. Q/K/V projections (per-layer variable dimensions)
            let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
            let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;

            if v_equals_k {
                // Global layer: no V projection, V = K
                timed!(ops, matmul, {
                    ctx.backend.dequant_matmul(
                        wq_raw, wq_dtype, &norm_buf, &mut q_buf[..q_dim], q_dim, 1, dim,
                    );
                    ctx.backend.dequant_matmul(
                        wk_raw, wk_dtype, &norm_buf, &mut k_buf[..kv_dim], kv_dim, 1, dim,
                    );
                });
                // Copy K → V (before normalization)
                v_buf[..kv_dim].copy_from_slice(&k_buf[..kv_dim]);
            } else {
                // SWA layer: standard Q/K/V projections
                let (wv_raw, wv_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
                timed!(ops, matmul, {
                    ctx.backend.batch_dequant_matvec(
                        &[
                            (wq_raw, wq_dtype, q_dim),
                            (wk_raw, wk_dtype, kv_dim),
                            (wv_raw, wv_dtype, kv_dim),
                        ],
                        &norm_buf,
                        dim,
                        &mut [
                            &mut q_buf[..q_dim],
                            &mut k_buf[..kv_dim],
                            &mut v_buf[..kv_dim],
                        ],
                    );
                });
            }

            // 2c. Per-head QK normalization
            apply_optional_attention_qk_norm_single(
                weights,
                &prefix,
                &mut q_buf[..q_dim],
                &mut k_buf[..kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;

            // 2c'. V normalization (Gemma4: raw RMSNorm, no learned weight)
            for v_head in v_buf[..kv_dim].chunks_exact_mut(head_dim) {
                rms_norm::rms_norm_no_weight(v_head, cfg.rms_norm_eps);
            }

            // 2d. RoPE on Q and K (Gemma4 uses NeoX-style split-half pairs)
            let rope_base = if is_local {
                cfg.rope_freq_base_local.unwrap_or(cfg.rope_freq_base)
            } else {
                cfg.rope_freq_base
            };
            let rope_position = if is_local {
                position as f32
            } else {
                cfg.rope_scaling.scaled_position(position)
            };

            if !is_local && let Some(ref freq_factors) = rope_freq_factors {
                // Global layers: NeoX RoPE with freq_factors for proportional rotation
                timed!(
                    ops,
                    rope,
                    rope::apply_rope_neox_with_freq_factors(
                        &mut q_buf[..q_dim],
                        &mut k_buf[..kv_dim],
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        rope_position,
                        rope_base,
                        freq_factors,
                    )
                );
            } else {
                // SWA layers: standard NeoX RoPE with full rotation
                timed!(
                    ops,
                    rope,
                    rope::apply_rope_multi_head_neox_partial_scaled(
                        &mut q_buf[..q_dim],
                        &mut k_buf[..kv_dim],
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        head_dim,
                        rope_position,
                        rope_base,
                    )
                );
            }

            // 2e. Update KV cache (pad to full stride for global layers)
            if kv_dim < kv_stride {
                k_padded[..kv_dim].copy_from_slice(&k_buf[..kv_dim]);
                k_padded[kv_dim..kv_stride].fill(0.0);
                v_padded[..kv_dim].copy_from_slice(&v_buf[..kv_dim]);
                v_padded[kv_dim..kv_stride].fill(0.0);
                cpu_kv.append_and_advance(layer, &k_padded[..kv_stride], &v_padded[..kv_stride]);
            } else {
                cpu_kv.append_and_advance(layer, &k_buf[..kv_dim], &v_buf[..kv_dim]);
            }

            // 2f. Multi-head attention
            let full_seq_len = cpu_kv.seq_len() + 1;
            let seq_len = if is_local {
                if let Some(window) = cfg.sliding_window_size {
                    full_seq_len.min(window as usize)
                } else {
                    full_seq_len
                }
            } else {
                full_seq_len
            };

            let k_start = full_seq_len.saturating_sub(seq_len);
            let k_cache_raw = cpu_kv.k_slice_including_current(layer, full_seq_len);
            let v_cache_raw = cpu_kv.v_slice_including_current(layer, full_seq_len);

            // Repack KV data for global layers (cache stride != layer kv_dim)
            let (k_cache_slice, v_cache_slice);
            let mut k_compact;
            let mut v_compact;
            if kv_dim < kv_stride {
                k_compact = vec![0.0f32; seq_len * kv_dim];
                v_compact = vec![0.0f32; seq_len * kv_dim];
                for t in 0..seq_len {
                    let src = (k_start + t) * kv_stride;
                    k_compact[t * kv_dim..(t + 1) * kv_dim]
                        .copy_from_slice(&k_cache_raw[src..src + kv_dim]);
                    v_compact[t * kv_dim..(t + 1) * kv_dim]
                        .copy_from_slice(&v_cache_raw[src..src + kv_dim]);
                }
                k_cache_slice = &k_compact[..];
                v_cache_slice = &v_compact[..];
            } else {
                k_cache_slice = &k_cache_raw[k_start * kv_stride..full_seq_len * kv_stride];
                v_cache_slice = &v_cache_raw[k_start * kv_stride..full_seq_len * kv_stride];
            }

            // Gemma4: attention scale = 1.0 (QK norms handle scaling)
            timed!(
                ops,
                attention,
                attention::multi_head_attention_scaled(
                    &q_buf[..q_dim],
                    k_cache_slice,
                    v_cache_slice,
                    &mut attn_out[..q_dim],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_len,
                    1.0,
                )
            );

            // 2g. Output projection
            let (wo_raw, wo_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
            timed!(
                ops,
                matmul,
                ctx.backend.dequant_matmul(
                    wo_raw,
                    wo_dtype,
                    &attn_out[..q_dim],
                    &mut proj_buf,
                    dim,
                    1,
                    q_dim,
                )
            );

            // 2h. Post-attention RMSNorm
            if weights.has(&format!("{prefix}.post_attention_norm.weight")) {
                let post_attn_norm_w = timed!(
                    ops,
                    dequant,
                    weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?
                );
                timed!(
                    ops,
                    norm,
                    rms_norm::rms_norm(&mut proj_buf, post_attn_norm_w, cfg.rms_norm_eps)
                );
            }

            // 2i. Residual add
            silu::elementwise_add(&mut hidden, &proj_buf);

            if weights.has(&format!("{prefix}.ffn_gate_inp.weight")) {
                Self::apply_post_attention_and_moe_ffn_single(
                    ctx,
                    weights,
                    &prefix,
                    &mut hidden,
                    &mut norm_buf,
                    &mut gate_buf,
                    &mut up_buf,
                    &mut down_buf,
                    dim,
                    inter_dim,
                    cfg.rms_norm_eps,
                    cfg,
                    single_moe_scratch
                        .as_mut()
                        .expect("Gemma4 MoE single scratch missing"),
                    ops.as_deref_mut(),
                )?;
            } else {
                // 2j-2o. FFN: norm → gate/up → GELU → down → [post-FFN norm] → residual
                let ffn_norm_w = weights.f32_slice(&format!("{prefix}.ffn_norm.weight"))?;
                crate::model::layer_ops::apply_ffn_single(
                    ctx.backend,
                    weights,
                    &prefix,
                    &mut hidden,
                    &mut norm_buf,
                    &mut gate_buf,
                    &mut up_buf,
                    &mut down_buf,
                    dim,
                    inter_dim,
                    ffn_norm_w,
                    cfg.rms_norm_eps,
                    crate::model::layer_ops::FfnActivation::GELU,
                );
            }

            // 2p. Layer output scale (Gemma4-specific)
            if weights.has(&format!("{prefix}.layer_output_scale.weight")) {
                let scale =
                    weights.f32_slice(&format!("{prefix}.layer_output_scale.weight"))?[0];
                for h in hidden.iter_mut() {
                    *h *= scale;
                }
            }
        }

        // Advance CPU KV cache
        cpu_kv.finalize_token();

        // --- Step 3: Final RMSNorm ---
        apply_output_norm_single(weights, &mut hidden, cfg.rms_norm_eps, ops.as_deref_mut())?;

        // --- Step 4: LM head ---
        write_normalized_single_logits_with_breakdown(
            ctx.backend,
            &hidden,
            dim,
            vocab_size,
            weights,
            logits,
            ops,
        )?;

        // --- Step 5: Logit softcapping ---
        if let Some(cap) = cfg.final_logit_softcapping {
            for l in logits[..vocab_size].iter_mut() {
                *l = (*l / cap).tanh() * cap;
            }
        }

        Ok(())
    }

    fn forward_batch(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        // Q8_0 expert-down weights trip a GPU hang in `moe_mul_mat_id_q8_0`
        // (both blocked and non-blocked variants). Route to the CPU MoE path
        // only when the model is pure Q8_0 (all layers), so mixed-quant schemes
        // like Q4_K_M that use Q8_0 only on a few layers keep the GPU path.
        if Self::all_layers_q8_0_expert_down(ctx.config, weights) {
            return self.forward_batch_token_major(ctx, token_ids, kv, weights, logits, None);
        }

        let prefill_plan = crate::model::execution_plan::PrefillExecutionPlan::for_forward_batch(
            ctx,
            kv,
            weights,
            token_ids.len(),
            false,
        )?;

        if matches!(
            prefill_plan.mode,
            crate::model::execution_plan::PrefillMode::GpuBatch
                | crate::model::execution_plan::PrefillMode::GpuChunked
        ) {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
            if prefill_plan.mode == crate::model::execution_plan::PrefillMode::GpuChunked {
                let chunk_len = prefill_plan.chunk_len.unwrap();
                let initial_seq_len = gpu_kv.seq_len();
                for chunk in token_ids.chunks(chunk_len) {
                    match self.forward_batch_gpu_unified(
                        ctx,
                        metal_ops,
                        chunk,
                        gpu_kv,
                        weights,
                        Some(logits),
                        None,
                        None,
                    ) {
                        Ok(()) => {}
                        Err(e) => {
                            tracing::warn!(
                                "Gemma4 chunked GPU batch prefill failed, falling back to serial: {e}"
                            );
                            let already_processed = kv.seq_len() - initial_seq_len;
                            let remaining = &token_ids[already_processed..];
                            let start_pos = kv.seq_len();
                            for (i, &tid) in remaining.iter().enumerate() {
                                logits.fill(0.0);
                                self.forward_single(
                                    ctx,
                                    tid,
                                    start_pos + i,
                                    kv,
                                    weights,
                                    logits,
                                    None,
                                )?;
                            }
                            return Ok(());
                        }
                    }
                }
                return Ok(());
            }
            match self.forward_batch_gpu_unified(
                ctx,
                metal_ops,
                token_ids,
                gpu_kv,
                weights,
                Some(logits),
                None,
                None,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!("Gemma4 GPU batch prefill failed, falling back to serial: {e}");
                }
            }
        }

        if prefill_plan.mode == crate::model::execution_plan::PrefillMode::CpuBatch {
            return self.forward_batch_token_major(ctx, token_ids, kv, weights, logits, None);
        }

        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, None)?;
        }
        Ok(())
    }

    fn forward_batch_profiled(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        if Self::all_layers_q8_0_expert_down(ctx.config, weights) {
            return self.forward_batch_token_major(ctx, token_ids, kv, weights, logits, Some(ops));
        }

        let prefill_plan = crate::model::execution_plan::PrefillExecutionPlan::for_forward_batch(
            ctx,
            kv,
            weights,
            token_ids.len(),
            false,
        )?;

        if matches!(
            prefill_plan.mode,
            crate::model::execution_plan::PrefillMode::GpuBatch
                | crate::model::execution_plan::PrefillMode::GpuChunked
        ) {
            let metal_ops = ctx.backend.metal_ops().unwrap();
            let gpu_kv = kv.as_gpu_mut().unwrap();
            if prefill_plan.mode == crate::model::execution_plan::PrefillMode::GpuChunked {
                let total_t = OpTimer::start();
                let chunk_len = prefill_plan.chunk_len.unwrap();
                for chunk in token_ids.chunks(chunk_len) {
                    self.forward_batch_gpu_unified(
                        ctx,
                        metal_ops,
                        chunk,
                        gpu_kv,
                        weights,
                        Some(logits),
                        None,
                        Some(&mut *ops),
                    )?;
                }
                ops.gpu += total_t.elapsed();
                return Ok(());
            }

            let total_t = OpTimer::start();
            let result = self.forward_batch_gpu_unified(
                ctx,
                metal_ops,
                token_ids,
                gpu_kv,
                weights,
                Some(logits),
                None,
                Some(&mut *ops),
            );
            ops.gpu += total_t.elapsed();
            return result;
        }

        if prefill_plan.mode == crate::model::execution_plan::PrefillMode::CpuBatch {
            return self.forward_batch_token_major(ctx, token_ids, kv, weights, logits, Some(ops));
        }

        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, Some(ops))?;
        }
        Ok(())
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        if config.gate_activation != crate::model::config::GateActivation::GELU {
            tracing::warn!(
                "Gemma4Forward selected but gate_activation is {:?}, expected GELU",
                config.gate_activation
            );
        }
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "gemma4"
    }

    fn supports_pipelined_decode(&self, _ctx: &ForwardContext) -> bool {
        false
    }
}
