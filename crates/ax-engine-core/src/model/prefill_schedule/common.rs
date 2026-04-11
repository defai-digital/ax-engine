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

/// Whether inter-step prefill pipelining is enabled for Qwen3.5.
///
/// When enabled, the prefill step loop submits GPU command buffers
/// asynchronously (recurrent tail phase) and defers K/V CPU mirror copies
/// so that CPU encoding of step N+1 overlaps GPU execution of step N.
///
/// Controlled by `AX_QWEN35_PREFILL_PIPELINED`:
/// - unset / `0` / `false` / `off` -> disabled (default)
/// - `1` / `true` / `on` -> enabled
pub(crate) fn prefill_inter_step_pipelined_enabled() -> bool {
    match std::env::var("AX_QWEN35_PREFILL_PIPELINED") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => false,
    }
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
#[allow(dead_code)]
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
        head_dim: u32,
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
    /// Pair gate+up batch matmul with f32 input (no cast needed).
    /// Reads input once, produces both gate and up outputs.
    /// Currently unused — the f32 pair kernel uses a non-blocked layout that
    /// regresses vs two separate blocked matmuls. Retained for future use
    /// when a blocked pair kernel is available.
    #[allow(dead_code)]
    DequantBatchPair {
        w0: BufRef,
        w1: BufRef,
        input: BufRef,
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
        kv_q8: bool,
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
        kv_q8: bool,
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
        kv_q8: bool,
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
    /// Fused SiLU activation + Q4_K down projection in one dispatch.
    /// Replaces SiluMulBatch + DequantBatch(down) when down weight is Q4_K.
    FusedSiluDownBatchQ4K {
        weight: BufRef,
        gate: BufRef,
        up: BufRef,
        output: BufRef,
        m: u32,
        n: u32,
        k: u32,
    },
    /// Fused SiLU activation + Q6_K down projection in one dispatch.
    /// Replaces SiluMulBatch + DequantBatch(down) when down weight is Q6_K.
    FusedSiluDownBatchQ6K {
        weight: BufRef,
        gate: BufRef,
        up: BufRef,
        output: BufRef,
        m: u32,
        n: u32,
        k: u32,
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

#[derive(Clone, Copy)]
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
        PrefillOp::DequantBatchPair {
            input, out0, out1, ..
        } => (vec![input.ptr_id()], vec![out0.ptr_id(), out1.ptr_id()]),
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
        PrefillOp::FusedSiluDownBatchQ4K {
            gate, up, output, ..
        } => (
            vec![gate.ptr_id(), up.ptr_id()],
            vec![output.ptr_id()],
        ),
        PrefillOp::FusedSiluDownBatchQ6K {
            gate, up, output, ..
        } => (
            vec![gate.ptr_id(), up.ptr_id()],
            vec![output.ptr_id()],
        ),
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
