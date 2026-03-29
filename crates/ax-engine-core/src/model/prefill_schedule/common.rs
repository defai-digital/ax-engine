/// Whether the graph-IR prefill schedule is enabled.
///
/// Controlled by `AX_METAL_PREFILL_GRAPH_IR`:
/// - unset / `1` / `true` / `on` -> enabled (default since G23)
/// - `0` / `false` / `off` -> disabled
///
/// G23: enabled by default. The graph-IR path pre-computes the full dispatch
/// schedule (kernel variants, buffer bindings, barrier positions) before
/// encoding, eliminating per-dispatch CPU overhead (plan computation, weight
/// cache lookup, SmartBarrier conflict scanning).
pub(crate) fn prefill_graph_ir_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_GRAPH_IR") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true,
    })
}

/// Whether multi-command-buffer prefill pipelining is enabled.
///
/// Requires `AX_METAL_PREFILL_GRAPH_IR=1`. When enabled, the schedule
/// is split into 2 command buffers: the first half of layers is committed
/// immediately so the GPU starts while the second half encodes.
///
/// Controlled by `AX_METAL_PREFILL_MULTI_CB`:
/// - unset / `1` / `true` / `on` -> enabled (default since G25)
/// - `0` / `false` / `off` -> disabled
///
/// G25: enabled by default. Splits the prefill schedule into 2 command
/// buffers at the layer midpoint. CB1 is committed immediately so the GPU
/// starts executing while the CPU encodes CB2. Requires graph-IR (G23).
pub(crate) fn prefill_multi_cb_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_MULTI_CB") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true,
    })
}

// ---------------------------------------------------------------------------
// BufRef: lightweight buffer pointer for the schedule
// ---------------------------------------------------------------------------

/// Raw pointer to a `MetalBuffer`. Safe to use because the schedule lifetime
/// is bounded by the `forward_batch_gpu_unified` scope where all buffers live.
#[derive(Clone, Copy)]
pub(super) struct BufRef(*const ax_engine_metal::MetalBuffer);

// Safety: MetalBuffer wraps an MTLBuffer which is thread-safe on Apple Metal.
unsafe impl Send for BufRef {}
unsafe impl Sync for BufRef {}

impl BufRef {
    fn new(buf: &ax_engine_metal::MetalBuffer) -> Self {
        Self(buf as *const _)
    }

    /// # Safety
    /// Caller must ensure the referenced MetalBuffer is still alive.
    #[inline]
    unsafe fn get(&self) -> &ax_engine_metal::MetalBuffer {
        unsafe { &*self.0 }
    }

    /// Pointer identity for barrier conflict detection.
    fn ptr_id(&self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// PrefillOp: a single pre-resolved GPU dispatch
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
pub(super) enum PrefillOp {
    /// Memory barrier (pre-computed by offline barrier pass).
    Barrier,

    // -- Norm / residual --
    RmsNormOutBatch {
        input: BufRef,
        norm_w: BufRef,
        output: BufRef,
        dim: u32,
        n: u32,
        eps: f32,
    },
    RmsNormOutBatchF16 {
        input: BufRef,
        norm_w: BufRef,
        output: BufRef,
        dim: u32,
        n: u32,
        eps: f32,
    },
    ResidualAddRmsNormOutBatch {
        hidden: BufRef,
        proj: BufRef,
        norm_w: BufRef,
        output: BufRef,
        dim: u32,
        n: u32,
        eps: f32,
    },
    ResidualAddRmsNormOutBatchF16 {
        hidden: BufRef,
        proj: BufRef,
        norm_w: BufRef,
        output: BufRef,
        dim: u32,
        n: u32,
        eps: f32,
    },
    ElementwiseAddBatch {
        a: BufRef,
        b: BufRef,
        dim: u32,
        n: u32,
    },
    SplitQGateBatch {
        input: BufRef,
        q: BufRef,
        gate: BufRef,
        n: u32,
        q_dim: u32,
    },
    PerHeadRmsNormBatch {
        input: BufRef,
        norm_w: BufRef,
        n: u32,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    },
    SigmoidElementwiseMul {
        gate: BufRef,
        value: BufRef,
        count: u32,
    },

    // -- Matmul (batch) --
    DequantBatch {
        weight: BufRef,
        input: BufRef,
        output: BufRef,
        scratch_f16: BufRef,
        m: u32,
        n: u32,
        k: u32,
        dtype: GgmlType,
        use_simd: bool,
        use_q5k_small: bool,
    },
    DequantBatchF16In {
        weight: BufRef,
        input_f16: BufRef,
        output: BufRef,
        m: u32,
        n: u32,
        k: u32,
        dtype: GgmlType,
    },
    DequantBatchPairF16In {
        w0: BufRef,
        w1: BufRef,
        input_f16: BufRef,
        out0: BufRef,
        out1: BufRef,
        m: u32,
        n: u32,
        k: u32,
        dtype: GgmlType,
    },

    // -- QKV / RoPE / KV append --
    QkvSplitRopeBatch {
        qkv: BufRef,
        q: BufRef,
        k: BufRef,
        v: BufRef,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rope_start: f32,
        rope_step: f32,
        rope_base: f32,
    },
    QkvSplitRopeAppendKvBatch {
        qkv: BufRef,
        q: BufRef,
        k: BufRef,
        v: BufRef,
        kv_k: BufRef,
        kv_v: BufRef,
        kv_f16: bool,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rope_start: f32,
        rope_step: f32,
        rope_base: f32,
        cache_offset: u32,
        kv_dim: u32,
    },
    RopeBatch {
        q: BufRef,
        k: BufRef,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rope_start: f32,
        rope_step: f32,
        rope_base: f32,
    },
    KvAppendBatch {
        src: BufRef,
        dst: BufRef,
        kv_f16: bool,
        cache_offset: u32,
        row_stride: u32,
        kv_dim: u32,
        n: u32,
    },

    // -- Attention --
    AttentionBatchLocal {
        q: BufRef,
        k: BufRef,
        v: BufRef,
        out: BufRef,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        config: ax_engine_metal::AttentionDispatchConfig,
    },
    AttentionF16OutHd128 {
        q: BufRef,
        k: BufRef,
        v: BufRef,
        out: BufRef,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    },
    AttentionCached {
        q: BufRef,
        kv_k: BufRef,
        kv_v: BufRef,
        out: BufRef,
        kv_f16: bool,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        base_seq: u32,
        config: ax_engine_metal::AttentionDispatchConfig,
    },

    // -- Activation --
    CastF32ToF16 {
        input: BufRef,
        output: BufRef,
        count: u32,
    },
    SiluMulBatch {
        gate: BufRef,
        up: BufRef,
        dim: u32,
        n: u32,
    },
    SiluMulBatchF16 {
        gate: BufRef,
        up: BufRef,
        output: BufRef,
        dim: u32,
        n: u32,
    },

    // -- Post-loop (single-token logits path) --
    RmsNorm {
        buf: BufRef,
        norm_w: BufRef,
        dim: u32,
        eps: f32,
    },
    BufferCopy {
        src: BufRef,
        src_off: usize,
        dst: BufRef,
        count: u32,
    },
    DequantMatvec {
        weight: BufRef,
        input: BufRef,
        output: BufRef,
        m: u32,
        k: u32,
        dtype: GgmlType,
    },

    // -- Post-loop (batch logits path) --
    BatchLogits {
        lm: BufRef,
        input: BufRef,
        scratch_f16: BufRef,
        output: BufRef,
        vocab: u32,
        n: u32,
        dim: u32,
        dtype: GgmlType,
        f16_io: bool,
        use_simd: bool,
    },
}

// ---------------------------------------------------------------------------
// PrefillSchedule: the complete pre-computed dispatch list
// ---------------------------------------------------------------------------

pub(super) struct PrefillSchedule {
    ops: Vec<PrefillOp>,
    /// Index where second half of layers starts (for multi-CB split).
    split_index: usize,
}

pub(super) struct Qwen35RecurrentTailProjection<'a> {
    pub rec_out: &'a ax_engine_metal::MetalBuffer,
    pub rec_z: &'a ax_engine_metal::MetalBuffer,
    pub ssm_norm: &'a ax_engine_metal::MetalBuffer,
    pub ssm_weight: &'a ax_engine_metal::MetalBuffer,
    pub ssm_dtype: GgmlType,
    pub time_step_rank: usize,
    pub state_size: usize,
    pub inner_dim: usize,
}

// ---------------------------------------------------------------------------
// Offline barrier pass
// ---------------------------------------------------------------------------

/// Returns (reads, writes) buffer pointer IDs for a given op.
fn op_buffer_sets(op: &PrefillOp) -> (Vec<usize>, Vec<usize>) {
    match op {
        PrefillOp::Barrier => (vec![], vec![]),
        PrefillOp::RmsNormOutBatch { input, output, .. }
        | PrefillOp::RmsNormOutBatchF16 { input, output, .. } => {
            (vec![input.ptr_id()], vec![output.ptr_id()])
        }
        PrefillOp::ResidualAddRmsNormOutBatch {
            hidden,
            proj,
            output,
            ..
        }
        | PrefillOp::ResidualAddRmsNormOutBatchF16 {
            hidden,
            proj,
            output,
            ..
        } => (
            vec![hidden.ptr_id(), proj.ptr_id()],
            vec![hidden.ptr_id(), output.ptr_id()],
        ),
        PrefillOp::ElementwiseAddBatch { a, b, .. } => {
            (vec![a.ptr_id(), b.ptr_id()], vec![a.ptr_id()])
        }
        PrefillOp::SplitQGateBatch { input, q, gate, .. } => {
            (vec![input.ptr_id()], vec![q.ptr_id(), gate.ptr_id()])
        }
        PrefillOp::PerHeadRmsNormBatch { input, .. } => {
            (vec![input.ptr_id()], vec![input.ptr_id()])
        }
        PrefillOp::SigmoidElementwiseMul { gate, value, .. } => {
            (vec![gate.ptr_id(), value.ptr_id()], vec![value.ptr_id()])
        }
        PrefillOp::DequantBatch { input, output, .. } => {
            (vec![input.ptr_id()], vec![output.ptr_id()])
        }
        PrefillOp::DequantBatchF16In {
            input_f16, output, ..
        } => (vec![input_f16.ptr_id()], vec![output.ptr_id()]),
        PrefillOp::DequantBatchPairF16In {
            input_f16,
            out0,
            out1,
            ..
        } => (vec![input_f16.ptr_id()], vec![out0.ptr_id(), out1.ptr_id()]),
        PrefillOp::QkvSplitRopeBatch { qkv, q, k, v, .. } => {
            (vec![qkv.ptr_id()], vec![q.ptr_id(), k.ptr_id(), v.ptr_id()])
        }
        PrefillOp::QkvSplitRopeAppendKvBatch {
            qkv,
            q,
            k,
            v,
            kv_k,
            kv_v,
            ..
        } => (
            vec![qkv.ptr_id()],
            vec![
                q.ptr_id(),
                k.ptr_id(),
                v.ptr_id(),
                kv_k.ptr_id(),
                kv_v.ptr_id(),
            ],
        ),
        PrefillOp::RopeBatch { q, k, .. } => {
            (vec![q.ptr_id(), k.ptr_id()], vec![q.ptr_id(), k.ptr_id()])
        }
        PrefillOp::KvAppendBatch { src, dst, .. } => (vec![src.ptr_id()], vec![dst.ptr_id()]),
        PrefillOp::AttentionBatchLocal { q, k, v, out, .. } => {
            (vec![q.ptr_id(), k.ptr_id(), v.ptr_id()], vec![out.ptr_id()])
        }
        PrefillOp::AttentionF16OutHd128 { q, k, v, out, .. } => {
            (vec![q.ptr_id(), k.ptr_id(), v.ptr_id()], vec![out.ptr_id()])
        }
        PrefillOp::AttentionCached {
            q, kv_k, kv_v, out, ..
        } => (
            vec![q.ptr_id(), kv_k.ptr_id(), kv_v.ptr_id()],
            vec![out.ptr_id()],
        ),
        PrefillOp::CastF32ToF16 { input, output, .. } => {
            (vec![input.ptr_id()], vec![output.ptr_id()])
        }
        PrefillOp::SiluMulBatch { gate, up, .. } => {
            (vec![gate.ptr_id(), up.ptr_id()], vec![gate.ptr_id()])
        }
        PrefillOp::SiluMulBatchF16 {
            gate, up, output, ..
        } => (vec![gate.ptr_id(), up.ptr_id()], vec![output.ptr_id()]),
        PrefillOp::RmsNorm { buf, .. } => (vec![buf.ptr_id()], vec![buf.ptr_id()]),
        PrefillOp::BufferCopy { src, dst, .. } => (vec![src.ptr_id()], vec![dst.ptr_id()]),
        PrefillOp::DequantMatvec { input, output, .. } => {
            (vec![input.ptr_id()], vec![output.ptr_id()])
        }
        PrefillOp::BatchLogits { input, output, .. } => {
            (vec![input.ptr_id()], vec![output.ptr_id()])
        }
    }
}

/// Insert `PrefillOp::Barrier` into the op list where data hazards exist.
///
/// Uses the same conflict rules as `SmartBarrier`: a barrier is needed when
/// a new dispatch reads from a buffer with a pending write, or writes to a
/// buffer with any pending access. Resets tracking at `split_index` for
/// multi-CB correctness.
fn insert_barriers(ops: &mut Vec<PrefillOp>, split_index: usize) {
    let smart = ax_engine_metal::smart_barriers_enabled();
    let mut pending: Vec<(usize, bool)> = Vec::with_capacity(32);
    let mut i = 0;
    let mut adjusted_split = split_index;

    while i < ops.len() {
        if matches!(ops[i], PrefillOp::Barrier) {
            i += 1;
            continue;
        }

        // Reset at CB boundary.
        if i == adjusted_split {
            pending.clear();
        }

        let (reads, writes) = op_buffer_sets(&ops[i]);

        let needs_barrier = if smart {
            // Smart: only barrier on true data hazards.
            let read_conflict = reads
                .iter()
                .any(|id| pending.iter().any(|&(pid, is_w)| is_w && pid == *id));
            let write_conflict = writes
                .iter()
                .any(|id| pending.iter().any(|&(pid, _)| pid == *id));
            read_conflict || write_conflict
        } else {
            // Dumb: barrier if anything is pending.
            !pending.is_empty()
        };

        if needs_barrier {
            ops.insert(i, PrefillOp::Barrier);
            adjusted_split += if i < adjusted_split { 1 } else { 0 };
            i += 1; // skip past the barrier we just inserted
            pending.clear();
        }

        // Register this op's reads/writes.
        for id in &reads {
            pending.push((*id, false));
        }
        for id in &writes {
            pending.push((*id, true));
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Schedule builder (LLaMA)
// ---------------------------------------------------------------------------

/// Build a pre-computed prefill schedule for LLaMA / Phi architectures.
///
/// All execution-plan decisions, weight-cache lookups, and barrier positions
/// are resolved upfront. The returned schedule can be encoded into one or
/// more Metal command buffers with zero branching overhead.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_llama_prefill_schedule(
    cfg: &crate::model::config::ModelConfig,
    prefill_plan: &GpuBatchPrefillExecutionPlan,
    cached: &crate::backend::metal::CachedModelKeys,
    weight_cache: &std::sync::MutexGuard<
        '_,
        rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
    >,
    bs: &crate::backend::metal::GpuBatchScratchBuffers,
    s: &crate::backend::metal::GpuScratchBuffers,
    gpu_kv: &crate::kv::GpuKv,
    base_seq_len: usize,
    n_tokens: usize,
    all_logits_buf: Option<&ax_engine_metal::MetalBuffer>,
    fused_qkv_cache: &std::sync::MutexGuard<
        '_,
        rustc_hash::FxHashMap<(usize, usize, usize), ax_engine_metal::MetalBuffer>,
    >,
) -> PrefillSchedule {
    let dim = cfg.embedding_dim as usize;
    let n_layers = cfg.n_layers as usize;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let inter_dim = cfg.intermediate_dim as usize;
    let vocab_size = cfg.vocab_size as usize;
    let q_dim = n_heads as usize * head_dim as usize;
    let kv_dim = n_kv_heads as usize * head_dim as usize;
    let eps = cfg.rms_norm_eps;
    let nt = n_tokens as u32;

    let capacity = n_layers * 14 + 10;
    let mut ops: Vec<PrefillOp> = Vec::with_capacity(capacity);

    // Helpers to wrap MetalBuffer refs as BufRef.
    let br = BufRef::new;

    // ── Layer-0 initial RMSNorm ──
    {
        let norm_w = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
        if prefill_plan.use_f16_batch_io {
            ops.push(PrefillOp::RmsNormOutBatchF16 {
                input: br(&bs.hidden),
                norm_w: br(norm_w),
                output: br(&bs.matmul_in_f16),
                dim: dim as u32,
                n: nt,
                eps,
            });
        } else {
            ops.push(PrefillOp::RmsNormOutBatch {
                input: br(&bs.hidden),
                norm_w: br(norm_w),
                output: br(&bs.norm_buf),
                dim: dim as u32,
                n: nt,
                eps,
            });
        }
    }

    let mut split_index = 0;

    // ── Layer loop ──
    for layer in 0..n_layers {
        if layer == n_layers / 2 {
            split_index = ops.len();
        }

        let lw = &cached.layers[layer];
        let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(base_seq_len);

        let wq_buf = weight_cache.get(&lw.wq).unwrap();
        let wk_buf = weight_cache.get(&lw.wk).unwrap();
        let wv_buf = weight_cache.get(&lw.wv).unwrap();
        let fused_qkv_key = (lw.wq, lw.wk, lw.wv);
        let qkv_layer_plan = DecodeExecutionPlan::llama_prefill_qkv_layer(
            prefill_plan,
            lw.wq_dtype,
            lw.wk_dtype,
            lw.wv_dtype,
        );
        let fused_qkv_buf = if qkv_layer_plan.use_fused_projection {
            fused_qkv_cache.get(&fused_qkv_key)
        } else {
            None
        };

        // Phase 1b: QKV matmul
        let fused_qkv_m = (q_dim + 2 * kv_dim) as u32;
        if let Some(fused_w) = fused_qkv_buf {
            match qkv_layer_plan.input {
                PrefillProjectionInputPlan::MatmulScratchF16 => {
                    ops.push(PrefillOp::DequantBatchF16In {
                        weight: br(fused_w),
                        input_f16: br(&bs.matmul_in_f16),
                        output: br(&bs.qkv_buf),
                        m: fused_qkv_m,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wq_dtype,
                    });
                }
                PrefillProjectionInputPlan::NormBufF32 => {
                    ops.push(PrefillOp::DequantBatch {
                        weight: br(fused_w),
                        input: br(&bs.norm_buf),
                        output: br(&bs.qkv_buf),
                        scratch_f16: br(&bs.matmul_in_f16),
                        m: fused_qkv_m,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wq_dtype,
                        use_simd: prefill_plan.use_batch_simd,
                        use_q5k_small: prefill_plan.q5k_prefill_small_n,
                    });
                }
            }
            // Split + RoPE (+ optional KV append)
            let cache_offset = (base_seq_len * kv_dim) as u32;
            if qkv_layer_plan.llama_post == LlamaPrefillQkvPostPlan::FusedSplitRopeAppendKv {
                ops.push(PrefillOp::QkvSplitRopeAppendKvBatch {
                    qkv: br(&bs.qkv_buf),
                    q: br(&bs.q_buf),
                    k: br(&bs.k_buf),
                    v: br(&bs.v_buf),
                    kv_k: br(gpu_kv.k_buffer(layer)),
                    kv_v: br(gpu_kv.v_buffer(layer)),
                    kv_f16: prefill_plan.kv_f16,
                    n: nt,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_start,
                    rope_step,
                    rope_base: cfg.rope_freq_base,
                    cache_offset,
                    kv_dim: kv_dim as u32,
                });
            } else {
                ops.push(PrefillOp::QkvSplitRopeBatch {
                    qkv: br(&bs.qkv_buf),
                    q: br(&bs.q_buf),
                    k: br(&bs.k_buf),
                    v: br(&bs.v_buf),
                    n: nt,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_start,
                    rope_step,
                    rope_base: cfg.rope_freq_base,
                });
            }
        } else {
            // Separate Q/K/V projections.
            match qkv_layer_plan.input {
                PrefillProjectionInputPlan::MatmulScratchF16 => {
                    ops.push(PrefillOp::DequantBatchF16In {
                        weight: br(wq_buf),
                        input_f16: br(&bs.matmul_in_f16),
                        output: br(&bs.q_buf),
                        m: q_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wq_dtype,
                    });
                    ops.push(PrefillOp::DequantBatchF16In {
                        weight: br(wk_buf),
                        input_f16: br(&bs.matmul_in_f16),
                        output: br(&bs.k_buf),
                        m: kv_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wk_dtype,
                    });
                    ops.push(PrefillOp::DequantBatchF16In {
                        weight: br(wv_buf),
                        input_f16: br(&bs.matmul_in_f16),
                        output: br(&bs.v_buf),
                        m: kv_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wv_dtype,
                    });
                }
                PrefillProjectionInputPlan::NormBufF32 => {
                    ops.push(PrefillOp::DequantBatch {
                        weight: br(wq_buf),
                        input: br(&bs.norm_buf),
                        output: br(&bs.q_buf),
                        scratch_f16: br(&bs.matmul_in_f16),
                        m: q_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wq_dtype,
                        use_simd: prefill_plan.use_batch_simd,
                        use_q5k_small: prefill_plan.q5k_prefill_small_n,
                    });
                    ops.push(PrefillOp::DequantBatch {
                        weight: br(wk_buf),
                        input: br(&bs.norm_buf),
                        output: br(&bs.k_buf),
                        scratch_f16: br(&bs.matmul_in_f16),
                        m: kv_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wk_dtype,
                        use_simd: prefill_plan.use_batch_simd,
                        use_q5k_small: prefill_plan.q5k_prefill_small_n,
                    });
                    ops.push(PrefillOp::DequantBatch {
                        weight: br(wv_buf),
                        input: br(&bs.norm_buf),
                        output: br(&bs.v_buf),
                        scratch_f16: br(&bs.matmul_in_f16),
                        m: kv_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wv_dtype,
                        use_simd: prefill_plan.use_batch_simd,
                        use_q5k_small: prefill_plan.q5k_prefill_small_n,
                    });
                }
            }
        }

        // Phase 1c: Separate RoPE (if not fused above)
        if qkv_layer_plan.llama_post == LlamaPrefillQkvPostPlan::Separate {
            ops.push(PrefillOp::RopeBatch {
                q: br(&bs.q_buf),
                k: br(&bs.k_buf),
                n: nt,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_start,
                rope_step,
                rope_base: cfg.rope_freq_base,
            });
        }

        // Phase 1d: KV cache append (if not fused above)
        if qkv_layer_plan.llama_post != LlamaPrefillQkvPostPlan::FusedSplitRopeAppendKv {
            let cache_offset = (base_seq_len * kv_dim) as u32;
            ops.push(PrefillOp::KvAppendBatch {
                src: br(&bs.k_buf),
                dst: br(gpu_kv.k_buffer(layer)),
                kv_f16: prefill_plan.kv_f16,
                cache_offset,
                row_stride: kv_dim as u32,
                kv_dim: kv_dim as u32,
                n: nt,
            });
            ops.push(PrefillOp::KvAppendBatch {
                src: br(&bs.v_buf),
                dst: br(gpu_kv.v_buffer(layer)),
                kv_f16: prefill_plan.kv_f16,
                cache_offset,
                row_stride: kv_dim as u32,
                kv_dim: kv_dim as u32,
                n: nt,
            });
        }

        // Phase 2: Attention
        match prefill_plan.attention {
            PrefillAttentionPlan::BatchLocalF16OutHd128 => {
                ops.push(PrefillOp::AttentionF16OutHd128 {
                    q: br(&bs.q_buf),
                    k: br(&bs.k_buf),
                    v: br(&bs.v_buf),
                    out: br(&bs.matmul_in_f16),
                    n: nt,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                });
            }
            PrefillAttentionPlan::BatchLocal => {
                ops.push(PrefillOp::AttentionBatchLocal {
                    q: br(&bs.q_buf),
                    k: br(&bs.k_buf),
                    v: br(&bs.v_buf),
                    out: br(&bs.attn_out),
                    n: nt,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    config: prefill_plan.attention_dispatch,
                });
            }
            PrefillAttentionPlan::Cached => {
                ops.push(PrefillOp::AttentionCached {
                    q: br(&bs.q_buf),
                    kv_k: br(gpu_kv.k_buffer(layer)),
                    kv_v: br(gpu_kv.v_buffer(layer)),
                    out: br(&bs.attn_out),
                    kv_f16: prefill_plan.kv_f16,
                    n: nt,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    base_seq: base_seq_len as u32,
                    config: prefill_plan.attention_dispatch,
                });
            }
        }

        // Phase 3a: Output projection
        let wo_buf = weight_cache.get(&lw.wo).unwrap();
        if prefill_plan.use_f16_batch_io {
            if prefill_plan.wo_input == PrefillWoInputPlan::AttentionOutF32 {
                ops.push(PrefillOp::CastF32ToF16 {
                    input: br(&bs.attn_out),
                    output: br(&bs.matmul_in_f16),
                    count: nt * q_dim as u32,
                });
            }
            ops.push(PrefillOp::DequantBatchF16In {
                weight: br(wo_buf),
                input_f16: br(&bs.matmul_in_f16),
                output: br(&bs.proj_buf),
                m: dim as u32,
                n: nt,
                k: q_dim as u32,
                dtype: lw.wo_dtype,
            });
        } else {
            ops.push(PrefillOp::DequantBatch {
                weight: br(wo_buf),
                input: br(&bs.attn_out),
                output: br(&bs.proj_buf),
                scratch_f16: br(&bs.matmul_in_f16),
                m: dim as u32,
                n: nt,
                k: q_dim as u32,
                dtype: lw.wo_dtype,
                use_simd: prefill_plan.use_batch_simd,
                use_q5k_small: false,
            });
        }

        // Phase 3b: Residual + FFN norm
        let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
        if prefill_plan.use_f16_batch_io {
            ops.push(PrefillOp::ResidualAddRmsNormOutBatchF16 {
                hidden: br(&bs.hidden),
                proj: br(&bs.proj_buf),
                norm_w: br(ffn_nw),
                output: br(&bs.matmul_in_f16),
                dim: dim as u32,
                n: nt,
                eps,
            });
        } else {
            ops.push(PrefillOp::ResidualAddRmsNormOutBatch {
                hidden: br(&bs.hidden),
                proj: br(&bs.proj_buf),
                norm_w: br(ffn_nw),
                output: br(&bs.norm_buf),
                dim: dim as u32,
                n: nt,
                eps,
            });
        }

        // Phase 3c: Gate + Up
        let wg_buf = weight_cache.get(&lw.wg).unwrap();
        let wu_buf = weight_cache.get(&lw.wu).unwrap();
        let ffn_layer_plan =
            DecodeExecutionPlan::llama_prefill_ffn_layer(prefill_plan, lw.wg_dtype, lw.wu_dtype);
        match ffn_layer_plan.input {
            PrefillProjectionInputPlan::MatmulScratchF16 => {
                if ffn_layer_plan.use_pair_kernel {
                    ops.push(PrefillOp::DequantBatchPairF16In {
                        w0: br(wg_buf),
                        w1: br(wu_buf),
                        input_f16: br(&bs.matmul_in_f16),
                        out0: br(&bs.gate_buf),
                        out1: br(&bs.up_buf),
                        m: inter_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wg_dtype,
                    });
                } else {
                    ops.push(PrefillOp::DequantBatchF16In {
                        weight: br(wg_buf),
                        input_f16: br(&bs.matmul_in_f16),
                        output: br(&bs.gate_buf),
                        m: inter_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wg_dtype,
                    });
                    ops.push(PrefillOp::DequantBatchF16In {
                        weight: br(wu_buf),
                        input_f16: br(&bs.matmul_in_f16),
                        output: br(&bs.up_buf),
                        m: inter_dim as u32,
                        n: nt,
                        k: dim as u32,
                        dtype: lw.wu_dtype,
                    });
                }
            }
            PrefillProjectionInputPlan::NormBufF32 => {
                ops.push(PrefillOp::DequantBatch {
                    weight: br(wg_buf),
                    input: br(&bs.norm_buf),
                    output: br(&bs.gate_buf),
                    scratch_f16: br(&bs.matmul_in_f16),
                    m: inter_dim as u32,
                    n: nt,
                    k: dim as u32,
                    dtype: lw.wg_dtype,
                    use_simd: prefill_plan.use_batch_simd,
                    use_q5k_small: false,
                });
                ops.push(PrefillOp::DequantBatch {
                    weight: br(wu_buf),
                    input: br(&bs.norm_buf),
                    output: br(&bs.up_buf),
                    scratch_f16: br(&bs.matmul_in_f16),
                    m: inter_dim as u32,
                    n: nt,
                    k: dim as u32,
                    dtype: lw.wu_dtype,
                    use_simd: prefill_plan.use_batch_simd,
                    use_q5k_small: false,
                });
            }
        }

        // Phase 3d: Activation
        match ffn_layer_plan.activation {
            PrefillFfnActivationPlan::SiluMulScratchF16 => {
                ops.push(PrefillOp::SiluMulBatchF16 {
                    gate: br(&bs.gate_buf),
                    up: br(&bs.up_buf),
                    output: br(&bs.matmul_in_f16),
                    dim: inter_dim as u32,
                    n: nt,
                });
            }
            PrefillFfnActivationPlan::SiluMulGateF32 => {
                ops.push(PrefillOp::SiluMulBatch {
                    gate: br(&bs.gate_buf),
                    up: br(&bs.up_buf),
                    dim: inter_dim as u32,
                    n: nt,
                });
            }
            PrefillFfnActivationPlan::GeluMulGateF32 => unreachable!(),
        }

        // Phase 3e: Down projection
        let wd_buf = weight_cache.get(&lw.wd).unwrap();
        match ffn_layer_plan.activation {
            PrefillFfnActivationPlan::SiluMulScratchF16 => {
                ops.push(PrefillOp::DequantBatchF16In {
                    weight: br(wd_buf),
                    input_f16: br(&bs.matmul_in_f16),
                    output: br(&bs.proj_buf),
                    m: dim as u32,
                    n: nt,
                    k: inter_dim as u32,
                    dtype: lw.wd_dtype,
                });
            }
            PrefillFfnActivationPlan::SiluMulGateF32 => {
                ops.push(PrefillOp::DequantBatch {
                    weight: br(wd_buf),
                    input: br(&bs.gate_buf),
                    output: br(&bs.proj_buf),
                    scratch_f16: br(&bs.matmul_in_f16),
                    m: dim as u32,
                    n: nt,
                    k: inter_dim as u32,
                    dtype: lw.wd_dtype,
                    use_simd: prefill_plan.use_batch_simd,
                    use_q5k_small: false,
                });
            }
            PrefillFfnActivationPlan::GeluMulGateF32 => unreachable!(),
        }

        // Phase 3f: Residual + next-layer norm
        match DecodeExecutionPlan::llama_prefill_residual_handoff(
            prefill_plan,
            layer + 1 == n_layers,
        ) {
            PrefillResidualHandoffPlan::ResidualOnly => {
                ops.push(PrefillOp::ElementwiseAddBatch {
                    a: br(&bs.hidden),
                    b: br(&bs.proj_buf),
                    dim: dim as u32,
                    n: nt,
                });
            }
            PrefillResidualHandoffPlan::ResidualAddRmsNormF32 => {
                let next_nw = weight_cache
                    .get(&cached.layers[layer + 1].attn_norm)
                    .unwrap();
                ops.push(PrefillOp::ResidualAddRmsNormOutBatch {
                    hidden: br(&bs.hidden),
                    proj: br(&bs.proj_buf),
                    norm_w: br(next_nw),
                    output: br(&bs.norm_buf),
                    dim: dim as u32,
                    n: nt,
                    eps,
                });
            }
            PrefillResidualHandoffPlan::ResidualAddRmsNormF16 => {
                let next_nw = weight_cache
                    .get(&cached.layers[layer + 1].attn_norm)
                    .unwrap();
                ops.push(PrefillOp::ResidualAddRmsNormOutBatchF16 {
                    hidden: br(&bs.hidden),
                    proj: br(&bs.proj_buf),
                    norm_w: br(next_nw),
                    output: br(&bs.matmul_in_f16),
                    dim: dim as u32,
                    n: nt,
                    eps,
                });
            }
        }
    }

    // ── Post-loop: final norm + LM head ──
    let fnw = weight_cache.get(&cached.output_norm).unwrap();
    let lm = weight_cache.get(&cached.lm_head).unwrap();
    match DecodeExecutionPlan::prefill_logits_plan(all_logits_buf.is_some()) {
        PrefillLogitsPlan::BatchAllLogits => {
            let logits_buf = all_logits_buf.unwrap();
            ops.push(PrefillOp::RmsNormOutBatch {
                input: br(&bs.hidden),
                norm_w: br(fnw),
                output: br(&bs.norm_buf),
                dim: dim as u32,
                n: nt,
                eps,
            });
            ops.push(PrefillOp::BatchLogits {
                lm: br(lm),
                input: br(&bs.norm_buf),
                scratch_f16: br(&bs.matmul_in_f16),
                output: br(logits_buf),
                vocab: vocab_size as u32,
                n: nt,
                dim: dim as u32,
                dtype: cached.lm_head_dtype,
                f16_io: prefill_plan.use_f16_batch_io,
                use_simd: prefill_plan.use_batch_simd,
            });
        }
        PrefillLogitsPlan::LastTokenMatvec => {
            let last_off = (n_tokens - 1) * dim * 4;
            ops.push(PrefillOp::BufferCopy {
                src: br(&bs.hidden),
                src_off: last_off,
                dst: br(&s.hidden),
                count: dim as u32,
            });
            ops.push(PrefillOp::RmsNorm {
                buf: br(&s.hidden),
                norm_w: br(fnw),
                dim: dim as u32,
                eps,
            });
            ops.push(PrefillOp::DequantMatvec {
                weight: br(lm),
                input: br(&s.hidden),
                output: br(&s.logits_buf),
                m: vocab_size as u32,
                k: dim as u32,
                dtype: cached.lm_head_dtype,
            });
        }
    }

    // ── Offline barrier pass ──
    insert_barriers(&mut ops, split_index);

    // Recalculate split_index after barriers were inserted (they may have shifted it).
    // We find the split by counting: the original split was at a specific op,
    // now there may be Barrier ops before it.
    // Simple approach: recalculate from the vec.
    let final_split = ops.len(); // default to end (no split)
    // Count real ops to find where the original split point landed.
    let mut real_ops_seen = 0;
    let original_split = split_index;
    let mut new_split = final_split;
    for (idx, op) in ops.iter().enumerate() {
        if !matches!(op, PrefillOp::Barrier) {
            if real_ops_seen == original_split {
                new_split = idx;
                break;
            }
            real_ops_seen += 1;
        }
    }

    PrefillSchedule {
        ops,
        split_index: new_split,
    }
}
