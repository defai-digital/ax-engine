fn qwen35_prefill_projection_needs_f16_input(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::Q8_0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen35PrefillLayerRouteKind {
    FullAttention {
        uses_cached_attention: bool,
        has_graph_ir_schedule: bool,
    },
    RecurrentGdn {
        force_backend_state_batch: bool,
        has_projection_phase: bool,
        has_runtime_phase: bool,
        has_tail_graph_ir_schedule: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Qwen35PrefillLayerRoute {
    pub layer: usize,
    pub kind: Qwen35PrefillLayerRouteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen35PrefillPhaseKind {
    FullAttention,
    RecurrentProjection,
    RecurrentRuntime,
    RecurrentTail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Qwen35PrefillPhase {
    pub kind: Qwen35PrefillPhaseKind,
    pub uses_graph_ir: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Qwen35PrefillRoutePlan {
    pub batch_position: usize,
    pub n_tokens: usize,
    pub uses_graph_ir: bool,
    pub layers: Vec<Qwen35PrefillLayerRoute>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Qwen35PrefillLayerScheduleKind {
    FullAttention {
        uses_cached_attention: bool,
        phases: Vec<Qwen35PrefillPhase>,
    },
    RecurrentGdn {
        force_backend_state_batch: bool,
        phases: Vec<Qwen35PrefillPhase>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Qwen35PrefillLayerSchedule {
    pub layer: usize,
    pub kind: Qwen35PrefillLayerScheduleKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Qwen35PrefillSchedule {
    pub batch_position: usize,
    pub n_tokens: usize,
    pub uses_graph_ir: bool,
    pub layers: Vec<Qwen35PrefillLayerSchedule>,
    pub steps: Vec<Qwen35PrefillExecutionStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Qwen35PrefillScheduleSummary {
    pub full_attention_layers: usize,
    pub recurrent_layers: usize,
    pub backend_state_batch_layers: usize,
    pub full_attention_graph_ir_layers: usize,
    pub recurrent_tail_graph_ir_layers: usize,
    pub cached_attention_layers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen35PrefillExecutionStepKind {
    FullAttention {
        uses_cached_attention: bool,
        uses_graph_ir: bool,
    },
    RecurrentProjection {
        force_backend_state_batch: bool,
    },
    RecurrentRuntime {
        force_backend_state_batch: bool,
    },
    RecurrentTail {
        force_backend_state_batch: bool,
        uses_graph_ir: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Qwen35PrefillExecutionStep {
    pub layer: usize,
    pub kind: Qwen35PrefillExecutionStepKind,
}

pub(super) fn build_qwen35_prefill_route_plan(
    cfg: &crate::model::config::ModelConfig,
    batch_position: usize,
    n_tokens: usize,
    has_gpu_attention: bool,
    recurrent_layer_state_owners: Option<&[crate::kv::qwen35_kv::Qwen3_5LayerStateOwner]>,
) -> Qwen35PrefillRoutePlan {
    let uses_graph_ir = prefill_graph_ir_enabled();
    let uses_cached_attention = batch_position > 0 && has_gpu_attention;
    let layers = (0..cfg.n_layers as usize)
        .map(|layer| {
            let kind = if cfg.qwen35_is_recurrent_layer(layer) {
                let layer_state_owner = recurrent_layer_state_owners
                    .and_then(|owners| owners.get(layer).copied())
                    .unwrap_or(crate::kv::qwen35_kv::Qwen3_5LayerStateOwner::CpuMaterialized);
                Qwen35PrefillLayerRouteKind::RecurrentGdn {
                    force_backend_state_batch:
                        crate::model::qwen35::Qwen3_5Forward::qwen35_prefill_backend_state_batch_for_tokens(
                            n_tokens,
                            layer_state_owner,
                        ),
                    has_projection_phase: true,
                    has_runtime_phase: true,
                    has_tail_graph_ir_schedule: uses_graph_ir,
                }
            } else {
                Qwen35PrefillLayerRouteKind::FullAttention {
                    uses_cached_attention,
                    has_graph_ir_schedule: uses_graph_ir
                        && (batch_position == 0 || has_gpu_attention),
                }
            };
            Qwen35PrefillLayerRoute { layer, kind }
        })
        .collect();

    Qwen35PrefillRoutePlan {
        batch_position,
        n_tokens,
        uses_graph_ir,
        layers,
    }
}

pub(super) fn build_qwen35_prefill_schedule(
    cfg: &crate::model::config::ModelConfig,
    batch_position: usize,
    n_tokens: usize,
    has_gpu_attention: bool,
    recurrent_layer_state_owners: Option<&[crate::kv::qwen35_kv::Qwen3_5LayerStateOwner]>,
) -> Qwen35PrefillSchedule {
    let route_plan = build_qwen35_prefill_route_plan(
        cfg,
        batch_position,
        n_tokens,
        has_gpu_attention,
        recurrent_layer_state_owners,
    );
    let mut steps = Vec::with_capacity(route_plan.layers.len() * 3);
    let layers = route_plan
        .layers
        .iter()
        .map(|route| match route.kind {
            Qwen35PrefillLayerRouteKind::FullAttention {
                uses_cached_attention,
                has_graph_ir_schedule,
            } => {
                steps.push(Qwen35PrefillExecutionStep {
                    layer: route.layer,
                    kind: Qwen35PrefillExecutionStepKind::FullAttention {
                        uses_cached_attention,
                        uses_graph_ir: has_graph_ir_schedule,
                    },
                });
                Qwen35PrefillLayerSchedule {
                    layer: route.layer,
                    kind: Qwen35PrefillLayerScheduleKind::FullAttention {
                        uses_cached_attention,
                        phases: vec![Qwen35PrefillPhase {
                            kind: Qwen35PrefillPhaseKind::FullAttention,
                            uses_graph_ir: has_graph_ir_schedule,
                        }],
                    },
                }
            }
            Qwen35PrefillLayerRouteKind::RecurrentGdn {
                force_backend_state_batch,
                has_projection_phase,
                has_runtime_phase,
                has_tail_graph_ir_schedule,
            } => {
                let mut phases = Vec::with_capacity(3);
                if has_projection_phase {
                    steps.push(Qwen35PrefillExecutionStep {
                        layer: route.layer,
                        kind: Qwen35PrefillExecutionStepKind::RecurrentProjection {
                            force_backend_state_batch,
                        },
                    });
                    phases.push(Qwen35PrefillPhase {
                        kind: Qwen35PrefillPhaseKind::RecurrentProjection,
                        uses_graph_ir: false,
                    });
                }
                if has_runtime_phase {
                    steps.push(Qwen35PrefillExecutionStep {
                        layer: route.layer,
                        kind: Qwen35PrefillExecutionStepKind::RecurrentRuntime {
                            force_backend_state_batch,
                        },
                    });
                    phases.push(Qwen35PrefillPhase {
                        kind: Qwen35PrefillPhaseKind::RecurrentRuntime,
                        uses_graph_ir: false,
                    });
                }
                steps.push(Qwen35PrefillExecutionStep {
                    layer: route.layer,
                    kind: Qwen35PrefillExecutionStepKind::RecurrentTail {
                        force_backend_state_batch,
                        uses_graph_ir: has_tail_graph_ir_schedule,
                    },
                });
                phases.push(Qwen35PrefillPhase {
                    kind: Qwen35PrefillPhaseKind::RecurrentTail,
                    uses_graph_ir: has_tail_graph_ir_schedule,
                });
                Qwen35PrefillLayerSchedule {
                    layer: route.layer,
                    kind: Qwen35PrefillLayerScheduleKind::RecurrentGdn {
                        force_backend_state_batch,
                        phases,
                    },
                }
            }
        })
        .collect();
    Qwen35PrefillSchedule {
        batch_position: route_plan.batch_position,
        n_tokens: route_plan.n_tokens,
        uses_graph_ir: route_plan.uses_graph_ir,
        layers,
        steps,
    }
}

pub(super) fn summarize_qwen35_prefill_schedule(
    schedule: &Qwen35PrefillSchedule,
) -> Qwen35PrefillScheduleSummary {
    let mut summary = Qwen35PrefillScheduleSummary {
        full_attention_layers: 0,
        recurrent_layers: 0,
        backend_state_batch_layers: 0,
        full_attention_graph_ir_layers: 0,
        recurrent_tail_graph_ir_layers: 0,
        cached_attention_layers: 0,
    };

    for layer in &schedule.layers {
        match layer.kind {
            Qwen35PrefillLayerScheduleKind::FullAttention {
                uses_cached_attention,
                ref phases,
            } => {
                summary.full_attention_layers += 1;
                if uses_cached_attention {
                    summary.cached_attention_layers += 1;
                }
                if phases.iter().any(|phase| {
                    phase.kind == Qwen35PrefillPhaseKind::FullAttention && phase.uses_graph_ir
                }) {
                    summary.full_attention_graph_ir_layers += 1;
                }
            }
            Qwen35PrefillLayerScheduleKind::RecurrentGdn {
                force_backend_state_batch,
                ref phases,
            } => {
                summary.recurrent_layers += 1;
                if force_backend_state_batch {
                    summary.backend_state_batch_layers += 1;
                }
                if phases.iter().any(|phase| {
                    phase.kind == Qwen35PrefillPhaseKind::RecurrentTail && phase.uses_graph_ir
                }) {
                    summary.recurrent_tail_graph_ir_layers += 1;
                }
            }
        }
    }

    summary
}

#[allow(clippy::too_many_arguments)]
fn push_qwen35_batch_projection_op(
    ops: &mut Vec<PrefillOp>,
    weight: &ax_engine_metal::MetalBuffer,
    input_f32: &ax_engine_metal::MetalBuffer,
    input_f16: &ax_engine_metal::MetalBuffer,
    output: &ax_engine_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
) {
    if qwen35_prefill_projection_needs_f16_input(dtype) {
        ops.push(PrefillOp::DequantBatchF16In {
            weight: BufRef::new(weight),
            input_f16: BufRef::new(input_f16),
            output: BufRef::new(output),
            m,
            n,
            k,
            dtype,
        });
    } else {
        ops.push(PrefillOp::DequantBatch {
            weight: BufRef::new(weight),
            input: BufRef::new(input_f32),
            output: BufRef::new(output),
            scratch_f16: BufRef::new(input_f16),
            m,
            n,
            k,
            dtype,
            use_simd: false,
            use_q5k_small: false,
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_qwen35_full_attention_prefill_schedule(
    cfg: &crate::model::config::ModelConfig,
    cached_layer: &crate::backend::metal::CachedLayerKeys,
    weight_cache: &std::sync::MutexGuard<
        '_,
        rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
    >,
    bs: &crate::backend::metal::GpuBatchScratchBuffers,
    gpu_kv: Option<&crate::kv::GpuKv>,
    layer: usize,
    batch_position: usize,
    n_tokens: usize,
    q_dim: usize,
    kv_dim: usize,
    inter_dim: usize,
    attention_dispatch: ax_engine_metal::AttentionDispatchConfig,
) -> anyhow::Result<PrefillSchedule> {
    use anyhow::Context;

    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let eps = cfg.rms_norm_eps;
    let nt = n_tokens as u32;
    let br = BufRef::new;
    let mut ops = Vec::with_capacity(24);

    let attn_norm = weight_cache
        .get(&cached_layer.attn_norm)
        .context("missing qwen35 attention norm buffer")?;
    let wq = weight_cache
        .get(&cached_layer.wq)
        .context("missing qwen35 WQ buffer")?;
    let wk = weight_cache
        .get(&cached_layer.wk)
        .context("missing qwen35 WK buffer")?;
    let wv = weight_cache
        .get(&cached_layer.wv)
        .context("missing qwen35 WV buffer")?;
    let wo = weight_cache
        .get(&cached_layer.wo)
        .context("missing qwen35 WO buffer")?;
    let ffn_norm = weight_cache
        .get(&cached_layer.ffn_norm)
        .context("missing qwen35 FFN norm buffer")?;
    let qkv_uses_f16 = qwen35_prefill_projection_needs_f16_input(cached_layer.wq_dtype)
        || qwen35_prefill_projection_needs_f16_input(cached_layer.wk_dtype)
        || qwen35_prefill_projection_needs_f16_input(cached_layer.wv_dtype);
    let ffn_input_uses_f16 = qwen35_prefill_projection_needs_f16_input(cached_layer.wg_dtype)
        || qwen35_prefill_projection_needs_f16_input(cached_layer.wu_dtype);

    ops.push(PrefillOp::RmsNormOutBatch {
        input: br(&bs.hidden),
        norm_w: br(attn_norm),
        output: br(&bs.norm_buf),
        dim: dim as u32,
        n: nt,
        eps,
    });
    if qkv_uses_f16 {
        ops.push(PrefillOp::CastF32ToF16 {
            input: br(&bs.norm_buf),
            output: br(&bs.matmul_in_f16),
            count: nt * dim as u32,
        });
    }

    push_qwen35_batch_projection_op(
        &mut ops,
        wq,
        &bs.norm_buf,
        &bs.matmul_in_f16,
        &bs.gate_buf,
        (q_dim * 2) as u32,
        nt,
        dim as u32,
        cached_layer.wq_dtype,
    );
    push_qwen35_batch_projection_op(
        &mut ops,
        wk,
        &bs.norm_buf,
        &bs.matmul_in_f16,
        &bs.k_buf,
        kv_dim as u32,
        nt,
        dim as u32,
        cached_layer.wk_dtype,
    );
    push_qwen35_batch_projection_op(
        &mut ops,
        wv,
        &bs.norm_buf,
        &bs.matmul_in_f16,
        &bs.v_buf,
        kv_dim as u32,
        nt,
        dim as u32,
        cached_layer.wv_dtype,
    );
    ops.push(PrefillOp::SplitQGateBatch {
        input: br(&bs.gate_buf),
        q: br(&bs.q_buf),
        gate: br(&bs.up_buf),
        n: nt,
        q_dim: q_dim as u32,
        head_dim,
    });

    if let (Some(q_key), Some(k_key)) = (cached_layer.attn_q_norm, cached_layer.attn_k_norm) {
        let q_nw = weight_cache
            .get(&q_key)
            .context("missing qwen35 Q norm buffer")?;
        let k_nw = weight_cache
            .get(&k_key)
            .context("missing qwen35 K norm buffer")?;
        ops.push(PrefillOp::PerHeadRmsNormBatch {
            input: br(&bs.q_buf),
            norm_w: br(q_nw),
            n: nt,
            n_heads,
            head_dim,
            eps,
        });
        ops.push(PrefillOp::PerHeadRmsNormBatch {
            input: br(&bs.k_buf),
            norm_w: br(k_nw),
            n: nt,
            n_heads: n_kv_heads,
            head_dim,
            eps,
        });
    }

    let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(batch_position);
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

    if let Some(gpu_kv) = gpu_kv {
        // For Q8_0 KV, the shader interprets cache_offset as block offset
        // (1 block = 32 values), not element offset.
        let cache_offset = if gpu_kv.is_q8() {
            (batch_position * (kv_dim / crate::kv::gpu_kv::Q8_0_BLOCK_VALUES)) as u32
        } else {
            (batch_position * kv_dim) as u32
        };
        ops.push(PrefillOp::KvAppendBatch {
            src: br(&bs.k_buf),
            dst: br(gpu_kv.k_buffer(layer)),
            kv_f16: gpu_kv.is_f16(),
            kv_q8: gpu_kv.is_q8(),
            cache_offset,
            row_stride: kv_dim as u32,
            kv_dim: kv_dim as u32,
            n: nt,
        });
        ops.push(PrefillOp::KvAppendBatch {
            src: br(&bs.v_buf),
            dst: br(gpu_kv.v_buffer(layer)),
            kv_f16: gpu_kv.is_f16(),
            kv_q8: gpu_kv.is_q8(),
            cache_offset,
            row_stride: kv_dim as u32,
            kv_dim: kv_dim as u32,
            n: nt,
        });
    }

    if batch_position == 0 {
        ops.push(PrefillOp::AttentionBatchLocal {
            q: br(&bs.q_buf),
            k: br(&bs.k_buf),
            v: br(&bs.v_buf),
            out: br(&bs.attn_out),
            n: nt,
            n_heads,
            n_kv_heads,
            head_dim,
            config: attention_dispatch,
        });
    } else {
        let gpu_kv = gpu_kv.context("qwen35 cached attention schedule requires GPU KV")?;
        ops.push(PrefillOp::AttentionCached {
            q: br(&bs.q_buf),
            kv_k: br(gpu_kv.k_buffer(layer)),
            kv_v: br(gpu_kv.v_buffer(layer)),
            out: br(&bs.attn_out),
            kv_f16: gpu_kv.is_f16(),
            kv_q8: gpu_kv.is_q8(),
            n: nt,
            n_heads,
            n_kv_heads,
            head_dim,
            base_seq: batch_position as u32,
            config: attention_dispatch,
        });
    }

    ops.push(PrefillOp::SigmoidElementwiseMul {
        gate: br(&bs.up_buf),
        value: br(&bs.attn_out),
        count: (n_tokens * q_dim) as u32,
    });

    if qwen35_prefill_projection_needs_f16_input(cached_layer.wo_dtype) {
        ops.push(PrefillOp::CastF32ToF16 {
            input: br(&bs.attn_out),
            output: br(&bs.matmul_in_f16),
            count: nt * q_dim as u32,
        });
    }
    push_qwen35_batch_projection_op(
        &mut ops,
        wo,
        &bs.attn_out,
        &bs.matmul_in_f16,
        &bs.proj_buf,
        dim as u32,
        nt,
        q_dim as u32,
        cached_layer.wo_dtype,
    );

    let wg = weight_cache
        .get(&cached_layer.wg)
        .context("missing qwen35 WG buffer")?;
    let wu = weight_cache
        .get(&cached_layer.wu)
        .context("missing qwen35 WU buffer")?;
    let wd = weight_cache
        .get(&cached_layer.wd)
        .context("missing qwen35 WD buffer")?;

    ops.push(PrefillOp::ResidualAddRmsNormOutBatch {
        hidden: br(&bs.hidden),
        proj: br(&bs.proj_buf),
        norm_w: br(ffn_norm),
        output: br(&bs.norm_buf),
        dim: dim as u32,
        n: nt,
        eps,
    });
    if ffn_input_uses_f16 {
        ops.push(PrefillOp::CastF32ToF16 {
            input: br(&bs.norm_buf),
            output: br(&bs.matmul_in_f16),
            count: nt * dim as u32,
        });
    }
    push_qwen35_batch_projection_op(
        &mut ops,
        wg,
        &bs.norm_buf,
        &bs.matmul_in_f16,
        &bs.gate_buf,
        inter_dim as u32,
        nt,
        dim as u32,
        cached_layer.wg_dtype,
    );
    push_qwen35_batch_projection_op(
        &mut ops,
        wu,
        &bs.norm_buf,
        &bs.matmul_in_f16,
        &bs.up_buf,
        inter_dim as u32,
        nt,
        dim as u32,
        cached_layer.wu_dtype,
    );
    ops.push(PrefillOp::SiluMulBatch {
        gate: br(&bs.gate_buf),
        up: br(&bs.up_buf),
        dim: inter_dim as u32,
        n: nt,
    });
    if qwen35_prefill_projection_needs_f16_input(cached_layer.wd_dtype) {
        ops.push(PrefillOp::CastF32ToF16 {
            input: br(&bs.gate_buf),
            output: br(&bs.matmul_in_f16),
            count: nt * inter_dim as u32,
        });
    }
    push_qwen35_batch_projection_op(
        &mut ops,
        wd,
        &bs.gate_buf,
        &bs.matmul_in_f16,
        &bs.proj_buf,
        dim as u32,
        nt,
        inter_dim as u32,
        cached_layer.wd_dtype,
    );
    ops.push(PrefillOp::ElementwiseAddBatch {
        a: br(&bs.hidden),
        b: br(&bs.proj_buf),
        dim: dim as u32,
        n: nt,
    });

    let split_index = ops.len();
    insert_barriers(&mut ops, split_index);

    Ok(PrefillSchedule {
        split_index: ops.len(),
        ops,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_execute_qwen35_full_attention_prefill_schedule(
    device: &ax_engine_metal::MetalDevice,
    metal_ops: &MetalOps,
    allow_graph_ir_schedule: bool,
    cfg: &crate::model::config::ModelConfig,
    cached_layer: &crate::backend::metal::CachedLayerKeys,
    weight_cache: &std::sync::MutexGuard<
        '_,
        rustc_hash::FxHashMap<usize, ax_engine_metal::MetalBuffer>,
    >,
    bs: &crate::backend::metal::GpuBatchScratchBuffers,
    gpu_kv: Option<&crate::kv::GpuKv>,
    layer: usize,
    batch_position: usize,
    n_tokens: usize,
    q_dim: usize,
    kv_dim: usize,
    inter_dim: usize,
    attention_dispatch: ax_engine_metal::AttentionDispatchConfig,
) -> anyhow::Result<bool> {
    if !allow_graph_ir_schedule || (batch_position != 0 && gpu_kv.is_none()) {
        return Ok(false);
    }

    let schedule = build_qwen35_full_attention_prefill_schedule(
        cfg,
        cached_layer,
        weight_cache,
        bs,
        gpu_kv,
        layer,
        batch_position,
        n_tokens,
        q_dim,
        kv_dim,
        inter_dim,
        attention_dispatch,
    )?;
    execute_prefill_multi_cb(device, &schedule, metal_ops)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_qwen35_recurrent_tail_prefill_schedule(
    bs: &crate::backend::metal::GpuBatchScratchBuffers,
    recurrent_projection: Option<Qwen35RecurrentTailProjection<'_>>,
    ffn_norm: &ax_engine_metal::MetalBuffer,
    wg: &ax_engine_metal::MetalBuffer,
    wg_dtype: GgmlType,
    wu: &ax_engine_metal::MetalBuffer,
    wu_dtype: GgmlType,
    wd: &ax_engine_metal::MetalBuffer,
    wd_dtype: GgmlType,
    n_tokens: usize,
    dim: usize,
    inter_dim: usize,
    eps: f32,
) -> PrefillSchedule {
    let nt = n_tokens as u32;
    let br = BufRef::new;
    let mut ops = Vec::with_capacity(16);

    if let Some(proj) = recurrent_projection {
        ops.push(PrefillOp::PerHeadRmsNormBatch {
            input: br(proj.rec_out),
            norm_w: br(proj.ssm_norm),
            n: nt,
            n_heads: proj.time_step_rank as u32,
            head_dim: proj.state_size as u32,
            eps,
        });
        ops.push(PrefillOp::SiluMulBatch {
            gate: br(proj.rec_z),
            up: br(proj.rec_out),
            dim: proj.inner_dim as u32,
            n: nt,
        });
        if qwen35_prefill_projection_needs_f16_input(proj.ssm_dtype) {
            ops.push(PrefillOp::CastF32ToF16 {
                input: br(proj.rec_z),
                output: br(&bs.matmul_in_f16),
                count: nt * proj.inner_dim as u32,
            });
        }
        push_qwen35_batch_projection_op(
            &mut ops,
            proj.ssm_weight,
            proj.rec_z,
            &bs.matmul_in_f16,
            &bs.attn_out,
            dim as u32,
            nt,
            proj.inner_dim as u32,
            proj.ssm_dtype,
        );
    }

    ops.push(PrefillOp::ResidualAddRmsNormOutBatch {
        hidden: br(&bs.hidden),
        proj: br(&bs.attn_out),
        norm_w: br(ffn_norm),
        output: br(&bs.norm_buf),
        dim: dim as u32,
        n: nt,
        eps,
    });
    if qwen35_prefill_projection_needs_f16_input(wg_dtype)
        || qwen35_prefill_projection_needs_f16_input(wu_dtype)
    {
        ops.push(PrefillOp::CastF32ToF16 {
            input: br(&bs.norm_buf),
            output: br(&bs.matmul_in_f16),
            count: nt * dim as u32,
        });
    }
    push_qwen35_batch_projection_op(
        &mut ops,
        wg,
        &bs.norm_buf,
        &bs.matmul_in_f16,
        &bs.gate_buf,
        inter_dim as u32,
        nt,
        dim as u32,
        wg_dtype,
    );
    push_qwen35_batch_projection_op(
        &mut ops,
        wu,
        &bs.norm_buf,
        &bs.matmul_in_f16,
        &bs.up_buf,
        inter_dim as u32,
        nt,
        dim as u32,
        wu_dtype,
    );
    ops.push(PrefillOp::SiluMulBatch {
        gate: br(&bs.gate_buf),
        up: br(&bs.up_buf),
        dim: inter_dim as u32,
        n: nt,
    });
    if qwen35_prefill_projection_needs_f16_input(wd_dtype) {
        ops.push(PrefillOp::CastF32ToF16 {
            input: br(&bs.gate_buf),
            output: br(&bs.matmul_in_f16),
            count: nt * inter_dim as u32,
        });
    }
    push_qwen35_batch_projection_op(
        &mut ops,
        wd,
        &bs.gate_buf,
        &bs.matmul_in_f16,
        &bs.proj_buf,
        dim as u32,
        nt,
        inter_dim as u32,
        wd_dtype,
    );
    ops.push(PrefillOp::ElementwiseAddBatch {
        a: br(&bs.hidden),
        b: br(&bs.proj_buf),
        dim: dim as u32,
        n: nt,
    });

    let split_index = ops.len();
    insert_barriers(&mut ops, split_index);

    PrefillSchedule {
        split_index: ops.len(),
        ops,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_execute_qwen35_recurrent_tail_prefill_schedule(
    device: &ax_engine_metal::MetalDevice,
    metal_ops: &MetalOps,
    allow_graph_ir_schedule: bool,
    bs: &crate::backend::metal::GpuBatchScratchBuffers,
    recurrent_projection: Option<Qwen35RecurrentTailProjection<'_>>,
    ffn_norm: &ax_engine_metal::MetalBuffer,
    wg: &ax_engine_metal::MetalBuffer,
    wg_dtype: GgmlType,
    wu: &ax_engine_metal::MetalBuffer,
    wu_dtype: GgmlType,
    wd: &ax_engine_metal::MetalBuffer,
    wd_dtype: GgmlType,
    n_tokens: usize,
    dim: usize,
    inter_dim: usize,
    eps: f32,
) -> anyhow::Result<bool> {
    if !allow_graph_ir_schedule {
        return Ok(false);
    }

    let schedule = build_qwen35_recurrent_tail_prefill_schedule(
        bs,
        recurrent_projection,
        ffn_norm,
        wg,
        wg_dtype,
        wu,
        wu_dtype,
        wd,
        wd_dtype,
        n_tokens,
        dim,
        inter_dim,
        eps,
    );
    execute_prefill_multi_cb(device, &schedule, metal_ops)?;
    Ok(true)
}

/// Like [`try_execute_qwen35_recurrent_tail_prefill_schedule`] but returns an
/// [`InflightFrame`] instead of blocking. Returns `Ok(None)` when graph-IR is
/// disabled and the caller should fall back to inline encoding.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_execute_qwen35_recurrent_tail_prefill_schedule_async(
    device: &ax_engine_metal::MetalDevice,
    metal_ops: &MetalOps,
    allow_graph_ir_schedule: bool,
    bs: &crate::backend::metal::GpuBatchScratchBuffers,
    recurrent_projection: Option<Qwen35RecurrentTailProjection<'_>>,
    ffn_norm: &ax_engine_metal::MetalBuffer,
    wg: &ax_engine_metal::MetalBuffer,
    wg_dtype: GgmlType,
    wu: &ax_engine_metal::MetalBuffer,
    wu_dtype: GgmlType,
    wd: &ax_engine_metal::MetalBuffer,
    wd_dtype: GgmlType,
    n_tokens: usize,
    dim: usize,
    inter_dim: usize,
    eps: f32,
) -> anyhow::Result<Option<ax_engine_metal::InflightFrame>> {
    if !allow_graph_ir_schedule {
        return Ok(None);
    }

    let schedule = build_qwen35_recurrent_tail_prefill_schedule(
        bs,
        recurrent_projection,
        ffn_norm,
        wg,
        wg_dtype,
        wu,
        wu_dtype,
        wd,
        wd_dtype,
        n_tokens,
        dim,
        inter_dim,
        eps,
    );
    let frame = execute_prefill_multi_cb_async(device, &schedule, metal_ops)?;
    Ok(Some(frame))
}

// ---------------------------------------------------------------------------
// Encoder: dispatch ops from the pre-computed schedule
// ---------------------------------------------------------------------------

/// Encode a slice of pre-computed prefill operations into a Metal encoder.
///
/// This is the hot loop: a flat match over ~300 ops with zero HashMap lookups,
/// zero execution-plan evaluation, zero SmartBarrier conflict scanning.
pub(super) fn encode_prefill_schedule(
    ops: &[PrefillOp],
    metal_ops: &MetalOps,
    encoder: &ax_engine_metal::MetalEncoder,
) -> anyhow::Result<()> {
    for op in ops {
        // Safety: all BufRef pointers are valid because the schedule is consumed
        // within the same scope where the buffers are live.
        unsafe {
            match op {
                PrefillOp::Barrier => {
                    ax_engine_metal::barrier_buffers(encoder);
                }

                PrefillOp::RmsNormOutBatch {
                    input,
                    norm_w,
                    output,
                    dim,
                    n,
                    eps,
                } => {
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        input.get(),
                        norm_w.get(),
                        output.get(),
                        *dim,
                        *n,
                        *eps,
                    );
                }
                PrefillOp::RmsNormOutBatchF16 {
                    input,
                    norm_w,
                    output,
                    dim,
                    n,
                    eps,
                } => {
                    metal_ops.elementwise.encode_rms_norm_out_batch_f16(
                        encoder,
                        input.get(),
                        norm_w.get(),
                        output.get(),
                        *dim,
                        *n,
                        *eps,
                    );
                }
                PrefillOp::ResidualAddRmsNormOutBatch {
                    hidden,
                    proj,
                    norm_w,
                    output,
                    dim,
                    n,
                    eps,
                } => {
                    metal_ops
                        .elementwise
                        .encode_residual_add_rms_norm_out_batch(
                            encoder,
                            hidden.get(),
                            proj.get(),
                            norm_w.get(),
                            output.get(),
                            *dim,
                            *n,
                            *eps,
                        );
                }
                PrefillOp::ResidualAddRmsNormOutBatchF16 {
                    hidden,
                    proj,
                    norm_w,
                    output,
                    dim,
                    n,
                    eps,
                } => {
                    metal_ops
                        .elementwise
                        .encode_residual_add_rms_norm_out_batch_f16(
                            encoder,
                            hidden.get(),
                            proj.get(),
                            norm_w.get(),
                            output.get(),
                            *dim,
                            *n,
                            *eps,
                        );
                }
                PrefillOp::ElementwiseAddBatch { a, b, dim, n } => {
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        a.get(),
                        b.get(),
                        *dim,
                        *n,
                    );
                }
                PrefillOp::SplitQGateBatch {
                    input,
                    q,
                    gate,
                    n,
                    q_dim,
                    head_dim,
                } => {
                    metal_ops.elementwise.encode_split_qgate_batch(
                        encoder,
                        input.get(),
                        q.get(),
                        gate.get(),
                        *n,
                        *q_dim,
                        *head_dim,
                    );
                }
                PrefillOp::PerHeadRmsNormBatch {
                    input,
                    norm_w,
                    n,
                    n_heads,
                    head_dim,
                    eps,
                } => {
                    metal_ops.elementwise.encode_per_head_rms_norm_batch(
                        encoder,
                        input.get(),
                        norm_w.get(),
                        *n,
                        *n_heads,
                        *head_dim,
                        *eps,
                    );
                }
                PrefillOp::SigmoidElementwiseMul { gate, value, count } => {
                    metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                        encoder,
                        gate.get(),
                        value.get(),
                        *count,
                    );
                }

                PrefillOp::DequantBatch {
                    weight,
                    input,
                    output,
                    scratch_f16,
                    m,
                    n,
                    k,
                    dtype,
                    use_simd,
                    use_q5k_small,
                } => {
                    encode_dequant_batch(
                        &metal_ops.dequant,
                        &metal_ops.elementwise,
                        encoder,
                        weight.get(),
                        input.get(),
                        output.get(),
                        scratch_f16.get(),
                        *m,
                        *n,
                        *k,
                        *dtype,
                        false,
                        *use_simd,
                        *use_q5k_small,
                    );
                }
                PrefillOp::DequantBatchF16In {
                    weight,
                    input_f16,
                    output,
                    m,
                    n,
                    k,
                    dtype,
                } => {
                    encode_dequant_batch_f16in(
                        metal_ops,
                        encoder,
                        weight.get(),
                        input_f16.get(),
                        output.get(),
                        *m,
                        *n,
                        *k,
                        *dtype,
                    );
                }
                PrefillOp::DequantBatchPairF16In {
                    w0,
                    w1,
                    input_f16,
                    out0,
                    out1,
                    m,
                    n,
                    k,
                    dtype,
                } => {
                    encode_dequant_batch_pair_f16in(
                        &metal_ops.dequant,
                        encoder,
                        w0.get(),
                        w1.get(),
                        input_f16.get(),
                        out0.get(),
                        out1.get(),
                        *m,
                        *n,
                        *k,
                        *dtype,
                    );
                }
                PrefillOp::DequantBatchPair {
                    w0,
                    w1,
                    input,
                    out0,
                    out1,
                    m,
                    n,
                    k,
                    dtype,
                } => match dtype {
                    GgmlType::Q4K => {
                        metal_ops.dequant.encode_fused_batch_pair_q4_k(
                            encoder,
                            w0.get(),
                            w1.get(),
                            input.get(),
                            out0.get(),
                            out1.get(),
                            *m,
                            *n,
                            *k,
                        );
                    }
                    GgmlType::Q6K => {
                        metal_ops.dequant.encode_fused_batch_pair_q6_k(
                            encoder,
                            w0.get(),
                            w1.get(),
                            input.get(),
                            out0.get(),
                            out1.get(),
                            *m,
                            *n,
                            *k,
                        );
                    }
                    _ => panic!("DequantBatchPair: unsupported dtype {:?}", dtype),
                },
                PrefillOp::QkvSplitRopeBatch {
                    qkv,
                    q,
                    k,
                    v,
                    n,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_start,
                    rope_step,
                    rope_base,
                } => {
                    metal_ops.elementwise.encode_qkv_split_rope_batch(
                        encoder,
                        qkv.get(),
                        q.get(),
                        k.get(),
                        v.get(),
                        *n,
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                        *rope_start,
                        *rope_step,
                        *rope_base,
                    );
                }
                PrefillOp::QkvSplitRopeAppendKvBatch {
                    qkv,
                    q,
                    k,
                    v,
                    kv_k,
                    kv_v,
                    kv_f16,
                    kv_q8: _,
                    n,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_start,
                    rope_step,
                    rope_base,
                    cache_offset,
                    kv_dim,
                } => {
                    metal_ops.elementwise.encode_qkv_split_rope_append_kv_batch(
                        encoder,
                        qkv.get(),
                        q.get(),
                        k.get(),
                        v.get(),
                        kv_k.get(),
                        kv_v.get(),
                        *kv_f16,
                        *n,
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                        *rope_start,
                        *rope_step,
                        *rope_base,
                        *cache_offset,
                        *kv_dim,
                    );
                }
                PrefillOp::RopeBatch {
                    q,
                    k,
                    n,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_start,
                    rope_step,
                    rope_base,
                } => {
                    metal_ops.elementwise.encode_rope_batch_neox_partial(
                        encoder,
                        q.get(),
                        k.get(),
                        *n,
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                        (*head_dim).min(64),
                        *rope_start,
                        *rope_step,
                        *rope_base,
                    );
                }
                PrefillOp::KvAppendBatch {
                    src,
                    dst,
                    kv_f16,
                    kv_q8,
                    cache_offset,
                    row_stride,
                    kv_dim,
                    n,
                } => {
                    if *kv_q8 {
                        metal_ops.elementwise.encode_kv_append_batch_q8(
                            encoder,
                            src.get(),
                            dst.get(),
                            *cache_offset,
                            *kv_dim / 32,
                            *row_stride,
                            *n,
                        );
                    } else {
                        metal_ops.elementwise.encode_kv_append_batch(
                            encoder,
                            src.get(),
                            dst.get(),
                            *kv_f16,
                            *cache_offset,
                            *row_stride,
                            *kv_dim,
                            *n,
                        );
                    }
                }

                PrefillOp::AttentionBatchLocal {
                    q,
                    k,
                    v,
                    out,
                    n,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    config,
                } => {
                    metal_ops.attention.encode_attention_prefill_with_config(
                        encoder,
                        q.get(),
                        k.get(),
                        v.get(),
                        out.get(),
                        *n,
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                        *config,
                    );
                }
                PrefillOp::AttentionF16OutHd128 {
                    q,
                    k,
                    v,
                    out,
                    n,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                } => {
                    metal_ops.attention.encode_attention_prefill_f16out_hd128(
                        encoder,
                        q.get(),
                        k.get(),
                        v.get(),
                        out.get(),
                        *n,
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                    );
                }
                PrefillOp::AttentionCached {
                    q,
                    kv_k,
                    kv_v,
                    out,
                    kv_f16,
                    kv_q8,
                    n,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    base_seq,
                    config,
                } => {
                    if *kv_q8 {
                        metal_ops.attention.encode_attention_prefill_cached_q8kv(
                            encoder,
                            q.get(),
                            kv_k.get(),
                            kv_v.get(),
                            out.get(),
                            *n,
                            *n_heads,
                            *n_kv_heads,
                            *head_dim,
                            *base_seq,
                            0,
                        );
                    } else {
                        metal_ops
                            .attention
                            .encode_attention_prefill_cached_with_config(
                                encoder,
                                q.get(),
                                kv_k.get(),
                                kv_v.get(),
                                out.get(),
                                *kv_f16,
                                *n,
                                *n_heads,
                                *n_kv_heads,
                                *head_dim,
                                *base_seq,
                                0,
                                *config,
                            );
                    }
                }

                PrefillOp::CastF32ToF16 {
                    input,
                    output,
                    count,
                } => {
                    metal_ops.elementwise.encode_cast_f32_to_f16(
                        encoder,
                        input.get(),
                        output.get(),
                        *count,
                    );
                }
                PrefillOp::SiluMulBatch { gate, up, dim, n } => {
                    metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                        encoder,
                        gate.get(),
                        up.get(),
                        *dim,
                        *n,
                    );
                }
                PrefillOp::SiluMulBatchF16 {
                    gate,
                    up,
                    output,
                    dim,
                    n,
                } => {
                    metal_ops.elementwise.encode_silu_elementwise_mul_batch_f16(
                        encoder,
                        gate.get(),
                        up.get(),
                        output.get(),
                        *dim,
                        *n,
                    );
                }
                PrefillOp::FusedSiluDownBatchQ4K {
                    weight,
                    gate,
                    up,
                    output,
                    m,
                    n,
                    k,
                } => {
                    metal_ops.dequant.encode_fused_batch_q4_k_silu(
                        encoder,
                        weight.get(),
                        gate.get(),
                        up.get(),
                        output.get(),
                        *m,
                        *n,
                        *k,
                    );
                }
                PrefillOp::FusedSiluDownBatchQ6K {
                    weight,
                    gate,
                    up,
                    output,
                    m,
                    n,
                    k,
                } => {
                    metal_ops.dequant.encode_fused_batch_q6_k_silu(
                        encoder,
                        weight.get(),
                        gate.get(),
                        up.get(),
                        output.get(),
                        *m,
                        *n,
                        *k,
                    );
                }

                PrefillOp::RmsNorm {
                    buf,
                    norm_w,
                    dim,
                    eps,
                } => {
                    metal_ops.elementwise.encode_rms_norm(
                        encoder,
                        buf.get(),
                        norm_w.get(),
                        *dim,
                        *eps,
                    );
                }
                PrefillOp::BufferCopy {
                    src,
                    src_off,
                    dst,
                    count,
                } => {
                    metal_ops.elementwise.encode_buffer_copy(
                        encoder,
                        src.get(),
                        *src_off,
                        dst.get(),
                        0,
                        *count,
                    );
                }
                PrefillOp::DequantMatvec {
                    weight,
                    input,
                    output,
                    m,
                    k,
                    dtype,
                } => {
                    encode_dequant_matvec(
                        metal_ops,
                        encoder,
                        weight.get(),
                        input.get(),
                        output.get(),
                        *m,
                        *k,
                        *dtype,
                    );
                }
                PrefillOp::BatchLogits {
                    lm,
                    input,
                    scratch_f16,
                    output,
                    vocab,
                    n,
                    dim,
                    dtype,
                    f16_io,
                    use_simd,
                } => {
                    encode_batch_logits(
                        metal_ops,
                        encoder,
                        lm.get(),
                        input.get(),
                        scratch_f16.get(),
                        output.get(),
                        *vocab,
                        *n,
                        *dim,
                        *dtype,
                        *f16_io,
                        *use_simd,
                    );
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Multi-CB execution
// ---------------------------------------------------------------------------

/// Execute a prefill schedule using 2 command buffers for pipelining.
///
/// CB1 (first half of layers) is committed immediately so the GPU starts
/// while the CPU encodes CB2 (second half + logits). Metal same-queue FIFO
/// ordering guarantees CB2 executes after CB1.
pub(super) fn execute_prefill_multi_cb(
    device: &ax_engine_metal::MetalDevice,
    schedule: &PrefillSchedule,
    metal_ops: &MetalOps,
) -> anyhow::Result<()> {
    if !prefill_multi_cb_enabled() || schedule.split_index >= schedule.ops.len() {
        // Single CB fallback.
        return device.execute_sync_concurrent(|encoder| {
            encode_prefill_schedule(&schedule.ops, metal_ops, encoder)
        });
    }

    // CB1: first half of layers — commit immediately, GPU starts.
    let _cb1 = device.execute_async_concurrent(|encoder| {
        encode_prefill_schedule(&schedule.ops[..schedule.split_index], metal_ops, encoder)
    })?;

    // CB2: second half + logits. FIFO ordering: CB2 waits for CB1.
    // execute_sync_concurrent commits CB2 and waits for it, which implicitly
    // means CB1 also finished (since CB2 can't start until CB1 completes).
    device.execute_sync_concurrent(|encoder| {
        encode_prefill_schedule(&schedule.ops[schedule.split_index..], metal_ops, encoder)
    })
}

/// Like [`execute_prefill_multi_cb`] but returns an [`InflightFrame`] instead
/// of blocking. The caller must call `wait_frame` before reading GPU-written
/// buffers.
pub(super) fn execute_prefill_multi_cb_async(
    device: &ax_engine_metal::MetalDevice,
    schedule: &PrefillSchedule,
    metal_ops: &MetalOps,
) -> anyhow::Result<ax_engine_metal::InflightFrame> {
    if !prefill_multi_cb_enabled() || schedule.split_index >= schedule.ops.len() {
        return device.execute_async_concurrent(|encoder| {
            encode_prefill_schedule(&schedule.ops, metal_ops, encoder)
        });
    }

    // CB1: first half — commit immediately, GPU starts.
    let _cb1 = device.execute_async_concurrent(|encoder| {
        encode_prefill_schedule(&schedule.ops[..schedule.split_index], metal_ops, encoder)
    })?;

    // CB2: second half. FIFO ordering ensures CB2 runs after CB1.
    // Return CB2's inflight frame — waiting on it guarantees both CBs complete.
    device.execute_async_concurrent(|encoder| {
        encode_prefill_schedule(&schedule.ops[schedule.split_index..], metal_ops, encoder)
    })
}
