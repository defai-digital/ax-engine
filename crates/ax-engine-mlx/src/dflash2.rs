// SPDX-License-Identifier: Apache-2.0
//
// Architecture and numerics follow the Apache-2.0 reference implementation in
// mlx-vlm 0.7.9 (`speculative/drafters/dflash2`) and the z-lab
// `Qwen3.8-27B-DFlash2` checkpoint. The Rust code, cache handling and AX
// integration are written for this repository.

//! DFlash2 block drafter for Qwen 3.8 27B.
//!
//! A 5-layer, 8-token-block drafter. Each round it consumes the target hidden
//! states (layers 5/19/33/47/61, concatenated) of the tokens committed since
//! the previous round, extends a per-layer sliding key/value cache with their
//! projection, and drafts `block - 1` tokens for the block `[anchor, mask, ...]`
//! in one non-causal pass. It borrows the target's token embedding and LM head.
//! A chain-aware candidate selector rescored the top-k logits of each position.
//!
//! This module is the drafter only. Capturing the target hidden states and the
//! verify/rollback loop live elsewhere.

// Not called from the decode loop yet (ADR-045 increment (a)); the parity test and
// the draft-cost probe exercise it.
#![allow(dead_code)]

use std::path::Path;

use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, add, argmax, argpartition_axis, astype, concatenate,
    contiguous, matmul, multiply, reshape, rms_norm, rope, scaled_dot_product_attention, silu_mul,
    slice, split, stack, sum_axis, take, take_along_axis, transpose,
};

use mlx_sys::{KernelOutputSpec, KernelTemplateArg, LazyKernel};

use crate::model::shared::qw;
use crate::model::shared::verify_qmm::QwenMtpVerifyQmmGuard;
use crate::weights::{ModelWeights, QuantizedWeight};

/// Drafter hyper-parameters from the checkpoint's `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Dflash2Config {
    pub(crate) hidden_size: i32,
    pub(crate) num_layers: usize,
    pub(crate) num_heads: i32,
    pub(crate) num_kv_heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) intermediate_size: i32,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) sliding_window: i32,
    pub(crate) block_size: i32,
    pub(crate) mask_token_id: u32,
    pub(crate) target_layer_ids: Vec<usize>,
    pub(crate) conv_group_size: i32,
    pub(crate) conv_kernel_size: i32,
    pub(crate) selector_rank: i32,
    pub(crate) selector_top_k: i32,
    pub(crate) vocab_size: i32,
    pub(crate) input_embedding_scale: f32,
    pub(crate) output_multiplier: f32,
}

impl Dflash2Config {
    pub(crate) fn from_json(text: &str) -> Result<Self, String> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|error| format!("DFlash2 config.json: {error}"))?;
        let flash = root
            .get("dflash_config")
            .and_then(serde_json::Value::as_object)
            .ok_or("DFlash2 config.json has no dflash_config object")?;
        let int = |object: &serde_json::Map<String, serde_json::Value>, key: &str| {
            object
                .get(key)
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| format!("DFlash2 config is missing integer {key}"))
        };
        let top = root
            .as_object()
            .ok_or("DFlash2 config.json is not an object")?;
        let float =
            |object: &serde_json::Map<String, serde_json::Value>, key: &str, default: f64| {
                object
                    .get(key)
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(default)
            };
        let rope_theta = root
            .get("rope_parameters")
            .and_then(|value| value.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(10_000_000.0);
        let target_layer_ids = flash
            .get("target_layer_ids")
            .and_then(serde_json::Value::as_array)
            .ok_or("DFlash2 config is missing target_layer_ids")?
            .iter()
            .map(|value| value.as_u64().map(|id| id as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or("DFlash2 target_layer_ids must be non-negative integers")?;
        if root.get("is_causal").and_then(serde_json::Value::as_bool) == Some(true) {
            return Err("DFlash2 causal draft blocks are not supported".into());
        }
        if flash
            .get("final_logit_softcapping")
            .is_some_and(|value| !value.is_null())
        {
            return Err("DFlash2 final_logit_softcapping is not supported".into());
        }
        // The drafter applies plain full-width RoPE at `rope_theta`; a scaled
        // or partial rope would draft with the wrong positions, so decline it.
        let rope_kind = |block: &serde_json::Value| {
            block
                .get("rope_type")
                .or_else(|| block.get("type"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        };
        if let Some(kind) = root
            .get("rope_parameters")
            .and_then(rope_kind)
            .filter(|kind| kind != "default")
        {
            return Err(format!("DFlash2 rope_type {kind:?} is not supported"));
        }
        if let Some(scaling) = root.get("rope_scaling").filter(|value| !value.is_null())
            && rope_kind(scaling).as_deref() != Some("default")
        {
            return Err("DFlash2 rope_scaling is not supported".into());
        }
        let partial_rotary = root
            .get("partial_rotary_factor")
            .or_else(|| {
                root.get("rope_parameters")
                    .and_then(|v| v.get("partial_rotary_factor"))
            })
            .and_then(serde_json::Value::as_f64);
        if partial_rotary.is_some_and(|factor| factor != 1.0) {
            return Err("DFlash2 partial_rotary_factor other than 1 is not supported".into());
        }
        let config = Self {
            hidden_size: int(top, "hidden_size")? as i32,
            num_layers: int(top, "num_hidden_layers")? as usize,
            num_heads: int(top, "num_attention_heads")? as i32,
            num_kv_heads: int(top, "num_key_value_heads")? as i32,
            head_dim: int(top, "head_dim")? as i32,
            intermediate_size: int(top, "intermediate_size")? as i32,
            rms_norm_eps: float(top, "rms_norm_eps", 1e-6) as f32,
            rope_theta: rope_theta as f32,
            sliding_window: int(top, "sliding_window")? as i32,
            block_size: int(flash, "block_size")? as i32,
            mask_token_id: int(flash, "mask_token_id")? as u32,
            target_layer_ids,
            conv_group_size: int(flash, "conv_group_size")? as i32,
            conv_kernel_size: int(flash, "conv_kernel_size")? as i32,
            selector_rank: int(flash, "selector_rank")? as i32,
            selector_top_k: int(flash, "selector_top_k")? as i32,
            vocab_size: int(top, "vocab_size")? as i32,
            input_embedding_scale: float(flash, "input_embedding_scale", 1.0) as f32,
            output_multiplier: float(flash, "output_multiplier", 1.0) as f32,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        let positive = [
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.intermediate_size,
            self.sliding_window,
            self.block_size,
            self.conv_group_size,
            self.conv_kernel_size,
            self.selector_rank,
            self.selector_top_k,
            self.vocab_size,
        ];
        if self.num_layers == 0 || positive.iter().any(|value| *value <= 0) {
            return Err("DFlash2 config has a non-positive dimension".into());
        }
        if self.num_heads % self.num_kv_heads != 0 || self.hidden_size % self.conv_group_size != 0 {
            return Err("DFlash2 config has inconsistent head or convolution grouping".into());
        }
        if self.target_layer_ids.is_empty()
            || self
                .target_layer_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "DFlash2 target_layer_ids must be non-empty and strictly increasing".into(),
            );
        }
        if self.selector_top_k > self.vocab_size || self.mask_token_id as i32 >= self.vocab_size {
            return Err("DFlash2 selector_top_k or mask_token_id exceeds the vocabulary".into());
        }
        Ok(())
    }

    /// Width of the concatenated target hidden states the drafter consumes.
    pub(crate) fn target_hidden_width(&self) -> i32 {
        self.hidden_size * self.target_layer_ids.len() as i32
    }
}

/// Grouped, causal, input-dependent convolution along the block's sequence
/// axis, applied before and after attention and the MLP of every layer.
/// How the large drafter matrices are stored. 4-bit MXFP4 costs 0.3% of the
/// tokens committed per round against bf16 (5.90 vs 5.92 on the 17-prompt suite)
/// and reads a quarter of the bytes through the fused multi-row matmul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrafterPrecision {
    Bf16,
    Mxfp4,
}

/// A projection `y = x W^T`, dense (transposed bf16) or quantized.
enum Projection {
    Dense(MlxArray),
    Quantized(QuantizedWeight),
}

impl Projection {
    /// Builds a projection from an `[out, in]` bf16 weight.
    fn new(weight: &MlxArray, precision: DrafterPrecision) -> Self {
        match precision {
            DrafterPrecision::Bf16 => Self::Dense(transpose(weight, &[1, 0], None)),
            DrafterPrecision::Mxfp4 => {
                let packed = mlx_sys::quantize(
                    weight,
                    Some(32),
                    Some(4),
                    MlxQuantizationMode::Mxfp4,
                    None,
                    None,
                );
                Self::Quantized(QuantizedWeight {
                    weight: packed[0].clone(),
                    scales: Some(packed[1].clone()),
                    biases: None,
                    group_size: 32,
                    bits: 4,
                    mode: "mxfp4".to_owned(),
                    linear_bias: None,
                    decode_weight_t: None,
                    decode_q2_weight: None,
                    decode_q2_scales: None,
                    decode_q2_biases: None,
                    weight_t_view: Default::default(),
                })
            }
        }
    }

    /// `x`: `[1, rows, in]`.
    fn project(&self, x: &MlxArray) -> MlxArray {
        match self {
            Self::Dense(weight_t) => matmul(x, weight_t, None),
            Self::Quantized(weight) => qw(x, weight),
        }
    }

    fn arrays(&self) -> Vec<&MlxArray> {
        match self {
            Self::Dense(weight_t) => vec![weight_t],
            Self::Quantized(weight) => {
                let mut arrays = vec![&weight.weight];
                arrays.extend(weight.scales.as_ref());
                arrays
            }
        }
    }
}

/// A scalar array of `dtype`.
fn scalar(value: f32, dtype: MlxDtype) -> MlxArray {
    astype(&MlxArray::from_f32(value), dtype, None)
}

struct GroupedConv {
    /// `[2, kernel, hidden]`: base taps for the prepare (0) and finish (1) halves.
    base: MlxArray,
    /// Transposed `kernel_projection.weight`, `[hidden, 2 * kernel * groups]`.
    projection_t: MlxArray,
}

/// One thread per output element of [`GroupedConv::convolve`]. Each step is
/// rounded to `T` in the op chain's order (tap + base, times the shifted
/// input, then the running sum), so the result matches it value for value.
const DFLASH2_GROUPED_CONV_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    uint row = idx / uint(Width);
    if (row >= uint(Rows)) {
        return;
    }
    uint col = idx - row * uint(Width);
    uint t = row % uint(Length);
    uint group = col / uint(GroupSize);
    uint dyn_row = row * uint(2 * Taps * Groups) + uint(Half * Taps * Groups) + group;
    T acc = T(0.0f);
    for (int offset = 0; offset < Taps; ++offset) {
        float base_tap = float(base[uint(Half * Taps + offset) * uint(Width) + col]);
        float tap = float(dynamic[dyn_row + uint(offset * Groups)]);
        T weight = static_cast<T>(base_tap + tap);
        float value = (int(t) >= offset) ? float(hidden[idx - uint(offset) * uint(Width)]) : 0.0f;
        T term = static_cast<T>(float(weight) * value);
        acc = (offset == 0) ? term : static_cast<T>(float(acc) + float(term));
    }
    out[idx] = acc;
"#;

/// Per-row top-K of `logits` (`[rows, V]`, `Threads` threads per row): each
/// thread keeps a sorted top-K of its strided slice, sixteen threads merge
/// `Threads / 16` lists each, and thread 0 merges those. Order is value
/// descending, then token id ascending, so the result does not depend on the
/// thread partition.
const DFLASH2_TOPK_SOURCE: &str = r#"
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const device T* x = logits + static_cast<size_t>(row) * static_cast<size_t>(V);
    // The sorted lists live in thread memory (dynamic indexing); the scan only
    // compares against the registers `thr`/`thr_id` (the current K-th entry)
    // and touches the lists on an insert, which is rare.
    float bv[K];
    uint bi[K];
    for (int k = 0; k < K; k++) { bv[k] = -INFINITY; bi[k] = 0xffffffffu; }
    float thr = -INFINITY;
    uint thr_id = 0xffffffffu;
    for (uint i = tid; i < static_cast<uint>(V); i += static_cast<uint>(Threads)) {
        const float v = static_cast<float>(x[i]);
        if (v > thr || (v == thr && i < thr_id)) {
            int p = K - 1;
            while (p > 0 && (bv[p - 1] < v || (bv[p - 1] == v && bi[p - 1] > i))) {
                bv[p] = bv[p - 1];
                bi[p] = bi[p - 1];
                p--;
            }
            bv[p] = v;
            bi[p] = i;
            thr = bv[K - 1];
            thr_id = bi[K - 1];
        }
    }
    threadgroup float sv[Threads * K];
    threadgroup uint si[Threads * K];
    for (int k = 0; k < K; k++) { sv[tid * K + k] = bv[k]; si[tid * K + k] = bi[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int Groups = 16;
    const int Per = Threads / Groups;
    if (tid < static_cast<uint>(Groups)) {
        for (int k = 0; k < K; k++) { bv[k] = -INFINITY; bi[k] = 0xffffffffu; }
        thr = -INFINITY;
        thr_id = 0xffffffffu;
        for (int t = static_cast<int>(tid) * Per; t < (static_cast<int>(tid) + 1) * Per; t++) {
            for (int k = 0; k < K; k++) {
                const float v = sv[t * K + k];
                const uint id = si[t * K + k];
                if (id == 0xffffffffu) { break; }
                if (v < thr || (v == thr && id > thr_id)) { break; }
                int p = K - 1;
                while (p > 0 && (bv[p - 1] < v || (bv[p - 1] == v && bi[p - 1] > id))) {
                    bv[p] = bv[p - 1];
                    bi[p] = bi[p - 1];
                    p--;
                }
                bv[p] = v;
                bi[p] = id;
                thr = bv[K - 1];
                thr_id = bi[K - 1];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < static_cast<uint>(Groups)) {
        for (int k = 0; k < K; k++) { sv[tid * K + k] = bv[k]; si[tid * K + k] = bi[k]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (int k = 0; k < K; k++) { bv[k] = -INFINITY; bi[k] = 0xffffffffu; }
        thr = -INFINITY;
        thr_id = 0xffffffffu;
        for (int t = 0; t < Groups; t++) {
            for (int k = 0; k < K; k++) {
                const float v = sv[t * K + k];
                const uint id = si[t * K + k];
                if (id == 0xffffffffu) { break; }
                if (v < thr || (v == thr && id > thr_id)) { break; }
                int p = K - 1;
                while (p > 0 && (bv[p - 1] < v || (bv[p - 1] == v && bi[p - 1] > id))) {
                    bv[p] = bv[p - 1];
                    bi[p] = bi[p - 1];
                    p--;
                }
                bv[p] = v;
                bi[p] = id;
                thr = bv[K - 1];
                thr_id = bi[K - 1];
            }
        }
        for (int k = 0; k < K; k++) { out[row * K + k] = bi[k]; }
    }
"#;

static DFLASH2_TOPK: LazyKernel = LazyKernel::new(
    "ax_dflash2_topk_v2",
    &["logits"],
    &["out"],
    DFLASH2_TOPK_SOURCE,
    "",
    true,
);

const DFLASH2_TOPK_THREADS: i32 = 192;

/// Indices of the `k` largest entries along the last axis of `logits`
/// (`[..., V]` -> `[..., k]`, uint32, value descending then id ascending) with
/// one Metal kernel, or `None` when the kernel does not apply (then the
/// caller partitions). `V` must hold at least `k` entries and `k` at most 32.
fn top_k_indices(logits: &MlxArray, k: i32) -> Option<MlxArray> {
    if !crate::fastpath::dflash2_topk_kernel_enabled() {
        return None;
    }
    let shape = logits.shape();
    let vocab = *shape.last()?;
    if !(1..=32).contains(&k) || vocab < k || shape.len() < 2 {
        return None;
    }
    let dtype = logits.dtype();
    if !matches!(
        dtype,
        MlxDtype::Float32 | MlxDtype::Bfloat16 | MlxDtype::Float16
    ) {
        return None;
    }
    let rows: i32 = shape[..shape.len() - 1].iter().product();
    if rows < 1 {
        return None;
    }
    let flat = reshape(logits, &[rows, vocab], None);
    let mut outputs = DFLASH2_TOPK
        .get()?
        .try_apply_with_template(
            &[&flat],
            &[KernelOutputSpec {
                shape: vec![rows, k],
                dtype: MlxDtype::Uint32,
            }],
            &[
                KernelTemplateArg::Dtype { name: "T", dtype },
                KernelTemplateArg::Int {
                    name: "V",
                    value: vocab,
                },
                KernelTemplateArg::Int {
                    name: "K",
                    value: k,
                },
                KernelTemplateArg::Int {
                    name: "Threads",
                    value: DFLASH2_TOPK_THREADS,
                },
            ],
            (rows * DFLASH2_TOPK_THREADS, 1, 1),
            (DFLASH2_TOPK_THREADS, 1, 1),
            None,
        )
        .ok()?;
    let out = outputs.pop()?;
    let mut out_shape = shape.clone();
    *out_shape.last_mut()? = k;
    Some(reshape(&out, &out_shape, None))
}

static DFLASH2_GROUPED_CONV: LazyKernel = LazyKernel::new(
    "ax_dflash2_grouped_conv_v1",
    &["hidden", "dynamic", "base"],
    &["out"],
    DFLASH2_GROUPED_CONV_SOURCE,
    "",
    true,
);

impl GroupedConv {
    /// `hidden`: `[B, L, hidden]` -> (prepared hidden, dynamic taps `[B, L,
    /// 2 * kernel * groups]` for both halves; `finish` reads the second).
    fn prepare(&self, cfg: &Dflash2Config, hidden: &MlxArray) -> (MlxArray, MlxArray) {
        let dynamic = matmul(hidden, &self.projection_t, None);
        (self.convolve(cfg, hidden, &dynamic, 0), dynamic)
    }

    fn finish(&self, cfg: &Dflash2Config, hidden: &MlxArray, dynamic: &MlxArray) -> MlxArray {
        self.convolve(cfg, hidden, dynamic, 1)
    }

    /// `output = sum_offset (base[half, offset] + taps[.., offset]) * shift(hidden, offset)`
    /// where `shift` delays the sequence axis by `offset` with zero fill and
    /// `taps` is half `half` of `dynamic`. One fused kernel when available;
    /// it rounds each step to the activation dtype like the op chain below,
    /// so both give the same values.
    fn convolve(
        &self,
        cfg: &Dflash2Config,
        hidden: &MlxArray,
        dynamic: &MlxArray,
        half: i32,
    ) -> MlxArray {
        if let Some(out) = self.convolve_fused(cfg, hidden, dynamic, half) {
            return out;
        }
        let shape = hidden.shape();
        let (batch, length) = (shape[0], shape[1]);
        let groups = cfg.hidden_size / cfg.conv_group_size;
        let taps = reshape(
            &slice(
                &reshape(
                    dynamic,
                    &[batch, length, 2, cfg.conv_kernel_size, groups],
                    None,
                ),
                &[0, 0, half, 0, 0],
                &[batch, length, half + 1, cfg.conv_kernel_size, groups],
                &[1, 1, 1, 1, 1],
                None,
            ),
            &[batch, length, cfg.conv_kernel_size, groups],
            None,
        );
        self.convolve_ops(cfg, hidden, &taps, half)
    }

    fn convolve_fused(
        &self,
        cfg: &Dflash2Config,
        hidden: &MlxArray,
        dynamic: &MlxArray,
        half: i32,
    ) -> Option<MlxArray> {
        if !crate::fastpath::dflash2_fused_conv_enabled() {
            return None;
        }
        let shape = hidden.shape();
        let dtype = hidden.dtype();
        let groups = cfg.hidden_size / cfg.conv_group_size;
        if shape.len() != 3
            || shape[2] != cfg.hidden_size
            || !matches!(dtype, MlxDtype::Bfloat16 | MlxDtype::Float16)
            || dynamic.dtype() != dtype
            || dynamic.shape() != [shape[0], shape[1], 2 * cfg.conv_kernel_size * groups]
            || self.base.shape() != [2, cfg.conv_kernel_size, cfg.hidden_size]
        {
            return None;
        }
        let base = if self.base.dtype() == dtype {
            self.base.clone()
        } else {
            astype(&self.base, dtype, None)
        };
        let total = shape[0] * shape[1] * shape[2];
        let mut outputs = DFLASH2_GROUPED_CONV
            .get()?
            .try_apply_with_template(
                &[hidden, dynamic, &base],
                &[KernelOutputSpec {
                    shape: shape.clone(),
                    dtype,
                }],
                &[
                    KernelTemplateArg::Dtype { name: "T", dtype },
                    KernelTemplateArg::Int {
                        name: "Rows",
                        value: shape[0] * shape[1],
                    },
                    KernelTemplateArg::Int {
                        name: "Length",
                        value: shape[1],
                    },
                    KernelTemplateArg::Int {
                        name: "Width",
                        value: shape[2],
                    },
                    KernelTemplateArg::Int {
                        name: "Taps",
                        value: cfg.conv_kernel_size,
                    },
                    KernelTemplateArg::Int {
                        name: "Groups",
                        value: groups,
                    },
                    KernelTemplateArg::Int {
                        name: "GroupSize",
                        value: cfg.conv_group_size,
                    },
                    KernelTemplateArg::Int {
                        name: "Half",
                        value: half,
                    },
                ],
                (total, 1, 1),
                (256, 1, 1),
                None,
            )
            .ok()?;
        outputs.pop()
    }

    /// The op-chain form of [`Self::convolve`]; `taps` is `[B, L, kernel, groups]`.
    fn convolve_ops(
        &self,
        cfg: &Dflash2Config,
        hidden: &MlxArray,
        taps: &MlxArray,
        half: i32,
    ) -> MlxArray {
        let shape = hidden.shape();
        let (batch, length, width) = (shape[0], shape[1], shape[2]);
        let groups = width / cfg.conv_group_size;
        let group = cfg.conv_group_size;
        let blocks = reshape(hidden, &[batch, length, groups, group], None);
        let taps = reshape(
            taps,
            &[batch, length, cfg.conv_kernel_size, groups, 1],
            None,
        );
        let mut output: Option<MlxArray> = None;
        for offset in 0..cfg.conv_kernel_size {
            let values = if offset == 0 {
                blocks.clone()
            } else {
                let pad = mlx_sys::zeros(
                    &[batch, offset.min(length), groups, group],
                    hidden.dtype(),
                    None,
                );
                if offset >= length {
                    pad
                } else {
                    let kept = slice(
                        &blocks,
                        &[0, 0, 0, 0],
                        &[batch, length - offset, groups, group],
                        &[1, 1, 1, 1],
                        None,
                    );
                    concatenate(&[&pad, &kept], 1, None)
                }
            };
            let kernel = reshape(
                &slice(
                    &self.base,
                    &[half, offset, 0],
                    &[half + 1, offset + 1, width],
                    &[1, 1, 1],
                    None,
                ),
                &[1, 1, groups, group],
                None,
            );
            let tap = reshape(
                &slice(
                    &taps,
                    &[0, 0, offset, 0, 0],
                    &[batch, length, offset + 1, groups, 1],
                    &[1, 1, 1, 1, 1],
                    None,
                ),
                &[batch, length, groups, 1],
                None,
            );
            let term = multiply(
                &add(&astype(&kernel, hidden.dtype(), None), &tap, None),
                &values,
                None,
            );
            output = Some(match output {
                Some(sum) => add(&sum, &term, None),
                None => term,
            });
        }
        reshape(
            &output.expect("convolution kernel size is positive"),
            &[batch, length, width],
            None,
        )
    }
}

struct Layer {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: MlxArray,
    k_norm: MlxArray,
    gate: Projection,
    up: Projection,
    down: Projection,
    input_norm: MlxArray,
    post_norm: MlxArray,
    attention_conv: GroupedConv,
    mlp_conv: GroupedConv,
}

struct Selector {
    predecessor_codebook: MlxArray,
    successor_codebook: MlxArray,
    hidden_projection_t: MlxArray,
}

/// Immutable drafter weights.
pub(crate) struct Dflash2Weights {
    cfg: Dflash2Config,
    fc: Projection,
    hidden_norm: MlxArray,
    layers: Vec<Layer>,
    final_norm: MlxArray,
    selector: Selector,
    /// `[hidden, layers * 2 * kv_width]`: every layer's K then V projection, for
    /// one matmul over the new context rows.
    context_kv: Projection,
    precision: DrafterPrecision,
    artifact: DrafterArtifact,
}

fn take_tensor(
    tensors: &mut std::collections::HashMap<String, MlxArray>,
    name: &str,
) -> Result<MlxArray, String> {
    tensors
        .remove(name)
        .ok_or_else(|| format!("DFlash2 checkpoint is missing tensor {name}"))
}

fn weight_t(
    tensors: &mut std::collections::HashMap<String, MlxArray>,
    name: &str,
) -> Result<MlxArray, String> {
    Ok(transpose(&take_tensor(tensors, name)?, &[1, 0], None))
}

impl Dflash2Weights {
    pub(crate) fn load(dir: &Path, precision: DrafterPrecision) -> Result<Self, String> {
        let config_text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|error| format!("DFlash2 config.json: {error}"))?;
        let cfg = Dflash2Config::from_json(&config_text)?;
        let artifact = DrafterArtifact::read(dir)?;
        let mut tensors = mlx_sys::load_safetensors(&dir.join("model.safetensors"), None)?;
        let projection = |tensors: &mut std::collections::HashMap<String, MlxArray>, name: &str| {
            Ok::<_, String>(Projection::new(&take_tensor(tensors, name)?, precision))
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        // Every layer's K then V projection, stacked on the output axis, so one
        // matmul over the new context rows yields all ten projections.
        let mut context_kv = Vec::with_capacity(cfg.num_layers * 2);
        for index in 0..cfg.num_layers {
            let name = |suffix: &str| format!("layers.{index}.{suffix}");
            let k_weight = take_tensor(&mut tensors, &name("self_attn.k_proj.weight"))?;
            let v_weight = take_tensor(&mut tensors, &name("self_attn.v_proj.weight"))?;
            let conv = |tensors: &mut std::collections::HashMap<String, MlxArray>, prefix: &str| {
                Ok::<_, String>(GroupedConv {
                    base: take_tensor(tensors, &name(&format!("{prefix}.base_kernel")))?,
                    projection_t: weight_t(
                        tensors,
                        &name(&format!("{prefix}.kernel_projection.weight")),
                    )?,
                })
            };
            layers.push(Layer {
                q: projection(&mut tensors, &name("self_attn.q_proj.weight"))?,
                k: Projection::new(&k_weight, precision),
                v: Projection::new(&v_weight, precision),
                o: projection(&mut tensors, &name("self_attn.o_proj.weight"))?,
                q_norm: take_tensor(&mut tensors, &name("self_attn.q_norm.weight"))?,
                k_norm: take_tensor(&mut tensors, &name("self_attn.k_norm.weight"))?,
                gate: projection(&mut tensors, &name("mlp.gate_proj.weight"))?,
                up: projection(&mut tensors, &name("mlp.up_proj.weight"))?,
                down: projection(&mut tensors, &name("mlp.down_proj.weight"))?,
                input_norm: take_tensor(&mut tensors, &name("input_layernorm.weight"))?,
                post_norm: take_tensor(&mut tensors, &name("post_attention_layernorm.weight"))?,
                attention_conv: conv(&mut tensors, "attention_conv")?,
                mlp_conv: conv(&mut tensors, "mlp_conv")?,
            });
            context_kv.push(k_weight);
            context_kv.push(v_weight);
        }
        let kv_refs: Vec<&MlxArray> = context_kv.iter().collect();
        let context_kv = Projection::new(&concatenate(&kv_refs, 0, None), precision);
        Ok(Self {
            fc: projection(&mut tensors, "fc.weight")?,
            hidden_norm: take_tensor(&mut tensors, "hidden_norm.weight")?,
            final_norm: take_tensor(&mut tensors, "norm.weight")?,
            selector: Selector {
                predecessor_codebook: take_tensor(
                    &mut tensors,
                    "candidate_selector.predecessor_codebook",
                )?,
                successor_codebook: take_tensor(
                    &mut tensors,
                    "candidate_selector.successor_codebook",
                )?,
                hidden_projection_t: weight_t(
                    &mut tensors,
                    "candidate_selector.hidden_projection.weight",
                )?,
            },
            layers,
            context_kv,
            precision,
            cfg,
            artifact,
        })
    }

    pub(crate) fn identity(&self) -> String {
        format!(
            "{};precision={:?}",
            self.artifact.identity(),
            self.precision
        )
    }

    /// The admitted drafter this checkpoint is, by content.
    pub(crate) fn admitted(&self) -> Option<&'static AdmittedDrafter> {
        self.artifact.admitted()
    }

    pub(crate) fn config(&self) -> &Dflash2Config {
        &self.cfg
    }

    /// Evaluates every weight once so rounds read resident device memory
    /// instead of lazily loaded or mapped file pages.
    pub(crate) fn materialize(&self) {
        let mut arrays: Vec<&MlxArray> = vec![
            &self.hidden_norm,
            &self.final_norm,
            &self.selector.predecessor_codebook,
            &self.selector.successor_codebook,
            &self.selector.hidden_projection_t,
        ];
        arrays.extend(self.fc.arrays());
        arrays.extend(self.context_kv.arrays());
        for layer in &self.layers {
            for projection in [
                &layer.q,
                &layer.k,
                &layer.v,
                &layer.o,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                arrays.extend(projection.arrays());
            }
            arrays.extend([
                &layer.q_norm,
                &layer.k_norm,
                &layer.input_norm,
                &layer.post_norm,
                &layer.attention_conv.base,
                &layer.attention_conv.projection_t,
                &layer.mlp_conv.base,
                &layer.mlp_conv.projection_t,
            ]);
        }
        // Load-time residency. No request cache or drafter cursor exists yet,
        // so a failure here cannot decline a step.
        mlx_sys::eval(&arrays);
    }
}

/// Per-layer rolling key/value cache of the committed context (post-RoPE keys).
#[derive(Clone, Default)]
struct LayerCache {
    keys: Option<MlxArray>,
    values: Option<MlxArray>,
}

/// How many cached rows to keep before appending `new_rows`: a rotating cache
/// of `max_size` rows retains `max_size - 1` old rows so every appended row
/// still sees `max_size` of context.
fn retained_old_rows(cached: i32, max_size: i32) -> i32 {
    cached.min((max_size - 1).max(0))
}

/// Pre-round drafter cursor. A failed capture or verify restores this so the
/// next round drafts the same context rows again.
#[derive(Clone)]
pub(crate) struct DrafterCursor {
    offset: i32,
    caches: Vec<LayerCache>,
    tree_alternatives: [Option<u32>; 2],
}

/// The drafter and its cache state for one request.
pub(crate) struct Dflash2Drafter {
    weights: std::sync::Arc<Dflash2Weights>,
    /// Pick each draft token with the candidate selector, as the reference
    /// runtime does for greedy decoding too (`AX_MLX_DFLASH2_SELECTOR`,
    /// default on). Off, drafts take the plain argmax of the target LM head.
    use_selector: bool,
    /// When set, every stage is evaluated and timed (diagnostic; breaks overlap).
    stage_times: Option<Vec<(&'static str, f64)>>,
    caches: Vec<LayerCache>,
    /// Absolute position of the next context row (the reference's `cache.offset`).
    offset: i32,
    /// Second-best token at the first two draft positions. Filled only by
    /// [`Self::draft_block_capturing_alternatives`]; otherwise both are `None`.
    pub(crate) tree_alternatives: [Option<u32>; 2],
}

impl Dflash2Drafter {
    pub(crate) fn new(weights: std::sync::Arc<Dflash2Weights>) -> Self {
        let layers = weights.cfg.num_layers;
        Self {
            weights,
            use_selector: false,
            stage_times: None,
            caches: (0..layers).map(|_| LayerCache::default()).collect(),
            offset: 0,
            tree_alternatives: [None, None],
        }
    }

    pub(crate) fn with_selector(mut self, enabled: bool) -> Self {
        self.use_selector = enabled;
        self
    }

    /// Start (or stop and return) per-stage timing.
    pub(crate) fn set_stage_timing(&mut self, enabled: bool) {
        self.stage_times = enabled.then(Vec::new);
    }

    pub(crate) fn take_stage_times(&mut self) -> Vec<(&'static str, f64)> {
        self.stage_times
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Evaluates `array` and records the time since the previous mark.
    ///
    /// Stage timing is opt-in. The default drafter leaves it unset, so this
    /// does not evaluate on the linear request path. A profiled failure is
    /// returned so the caller can restore the pre-round cursor.
    fn mark(
        &mut self,
        stage: &'static str,
        array: &MlxArray,
        started: &mut std::time::Instant,
    ) -> Result<(), String> {
        let Some(times) = self.stage_times.as_mut() else {
            return Ok(());
        };
        if let Some(error) = take_dflash2_capture_verify_eval_error() {
            tracing::warn!(
                target: "ax_engine_mlx::dflash2",
                %error,
                "DFlash2 draft evaluation failed; declining speculation"
            );
            return Err(error);
        }
        if let Err(error) = mlx_sys::try_eval(&[array]) {
            tracing::warn!(
                target: "ax_engine_mlx::dflash2",
                %error,
                "DFlash2 draft evaluation failed; declining speculation"
            );
            return Err(error);
        }
        times.push((stage, started.elapsed().as_secs_f64() * 1e3));
        *started = std::time::Instant::now();
        Ok(())
    }

    /// Restores `cursor` after a draft read fails. The returned ids are unused.
    fn fail_draft(
        &mut self,
        cursor: DrafterCursor,
        block: i32,
        error: String,
    ) -> (MlxArray, Option<String>) {
        self.restore_cursor(cursor);
        let width = (block - 1).max(0);
        (
            mlx_sys::zeros(&[1, width], MlxDtype::Uint32, None),
            Some(error),
        )
    }

    /// Runs the block at every context-row count it will see (1 to the block
    /// size) so each custom-kernel instance is compiled before the first request,
    /// then clears the cache. Without it the first rounds pay the compile time.
    pub(crate) fn warm_up(&mut self, target: &ModelWeights) {
        let width = self.weights.cfg.target_hidden_width();
        let block = self.weights.cfg.block_size;
        for rows in 1..=block {
            let hidden = mlx_sys::zeros(&[1, rows, width], MlxDtype::Bfloat16, None);
            let drafts = match self.draft_block(0, &hidden, target, block) {
                Ok(drafts) => drafts,
                Err(error) => {
                    // Scratch cursor only. `reset` drops it; no request is live.
                    tracing::warn!(
                        target: "ax_engine_mlx::dflash2",
                        %error,
                        "DFlash2 drafter warm-up failed; the first request compiles the kernels"
                    );
                    self.reset();
                    return;
                }
            };
            if let Err(error) = mlx_sys::try_eval(&[&drafts]) {
                tracing::warn!(
                    target: "ax_engine_mlx::dflash2",
                    %error,
                    "DFlash2 drafter warm-up failed; the first request compiles the kernels"
                );
                self.reset();
                return;
            }
        }
        self.reset();
    }

    /// Absolute position of the next context row.
    pub(crate) fn cache_offset(&self) -> i32 {
        self.offset
    }

    /// Rows in the first layer's key cache. Zero before the first committed draft.
    pub(crate) fn cache_rows(&self) -> i32 {
        self.caches
            .iter()
            .find_map(|cache| cache.keys.as_ref().map(|keys| keys.shape()[2]))
            .unwrap_or(0)
    }

    /// Copies the offset, per-layer KV, and runner-up slots.
    pub(crate) fn snapshot_cursor(&self) -> DrafterCursor {
        DrafterCursor {
            offset: self.offset,
            caches: self.caches.clone(),
            tree_alternatives: self.tree_alternatives,
        }
    }

    /// Puts back a cursor taken before a draft whose capture or verify failed.
    pub(crate) fn restore_cursor(&mut self, cursor: DrafterCursor) {
        self.offset = cursor.offset;
        self.caches = cursor.caches;
        self.tree_alternatives = cursor.tree_alternatives;
    }

    /// Sets the absolute position of the first context row when the prompt rows
    /// before it were dropped by the window cap. Only valid on a fresh drafter.
    pub(crate) fn seed_offset(&mut self, skipped_rows: i32) {
        debug_assert_eq!(self.offset, 0, "seed_offset needs a fresh drafter");
        self.offset = skipped_rows;
    }

    /// Moves the next context row's position past `rows` committed rows the
    /// drafter will never see (rows dropped while the request decoded on
    /// another route). The context cache needs no positions beyond RoPE.
    pub(crate) fn skip_rows(&mut self, rows: i32) {
        self.offset += rows;
    }

    pub(crate) fn reset(&mut self) {
        for cache in &mut self.caches {
            *cache = LayerCache::default();
        }
        self.offset = 0;
        self.tree_alternatives = [None, None];
    }

    /// Drafts `block - 1` tokens after `anchor`.
    ///
    /// `new_hidden` is `[1, S, target_hidden_width]` bf16: the captured target
    /// hidden states of the tokens committed since the previous call (the whole
    /// prompt on the first call). Returns `[1, block - 1]` token ids (lazy).
    ///
    /// `Err` means an opt-in stage-timing read failed. The drafter cursor is
    /// already back to the pre-call snapshot. The default path does not time
    /// stages, so this returns the same lazy ids as before.
    pub(crate) fn draft_block(
        &mut self,
        anchor: u32,
        new_hidden: &MlxArray,
        target: &ModelWeights,
        block: i32,
    ) -> Result<MlxArray, String> {
        let (tokens, error) = self.draft_block_inner(anchor, new_hidden, target, block, false);
        match error {
            Some(error) => Err(error),
            None => Ok(tokens),
        }
    }

    /// Same block as [`Self::draft_block`], and also records the runner-up
    /// token at the first two draft positions in [`Self::tree_alternatives`].
    /// The returned primary chain matches [`Self::draft_block`].
    ///
    /// `Err` means the runner-up read failed before any request cache write.
    /// The drafter's offset and KV cache are restored to the pre-call cursor
    /// so the caller can draft the same rows again.
    pub(crate) fn draft_block_capturing_alternatives(
        &mut self,
        anchor: u32,
        new_hidden: &MlxArray,
        target: &ModelWeights,
        block: i32,
    ) -> Result<MlxArray, String> {
        let (tokens, error) = self.draft_block_inner(anchor, new_hidden, target, block, true);
        match error {
            Some(error) => Err(error),
            None => Ok(tokens),
        }
    }

    /// Drafts one block per lane in a single graph: the embedding, context
    /// and every projection, the grouped convolutions, the MLP and the LM
    /// head run once over all lanes' rows (each weight read once), and only
    /// each lane's context-cache attention runs on its own drafter. Each
    /// lane's cursor advances exactly as [`Self::draft_block`] would advance
    /// it, and each returned `[1, block - 1]` id array is lazy.
    ///
    /// `None` when the lanes cannot share a graph (fewer than two, the
    /// candidate selector or stage timing on, different drafter weights or
    /// hidden widths); nothing is changed then and the caller drafts each
    /// lane alone.
    pub(crate) fn draft_blocks_packed(
        lanes: &mut [(&mut Dflash2Drafter, u32, MlxArray)],
        target: &ModelWeights,
        block: i32,
    ) -> Option<Vec<MlxArray>> {
        let weights = lanes.first()?.0.weights.clone();
        let use_selector = lanes.first()?.0.use_selector;
        let cfg = &weights.cfg;
        let hidden = cfg.hidden_size;
        let width = lanes.first()?.2.shape().get(2).copied()?;
        if lanes.len() < 2
            || block < 2
            || lanes.iter().any(|(drafter, _, new_hidden)| {
                drafter.use_selector != use_selector
                    || drafter.stage_times.is_some()
                    || !std::sync::Arc::ptr_eq(&drafter.weights, &weights)
                    || new_hidden.shape().len() != 3
                    || new_hidden.shape()[0] != 1
                    || new_hidden.shape()[2] != width
                    || new_hidden.shape()[1] < 1
            })
        {
            return None;
        }
        let batch = lanes.len() as i32;
        let _fused_rows = QwenMtpVerifyQmmGuard::arm(true);
        // `[B, L, in]` -> `[B, L, out]` through one `[1, B·L, in]` matmul.
        let project_rows = |projection: &Projection, x: &MlxArray| {
            let shape = x.shape();
            let rows = shape[0] * shape[1];
            let out = projection.project(&reshape(x, &[1, rows, shape[2]], None));
            let out_width = out.shape()[2];
            reshape(&out, &[shape[0], shape[1], out_width], None)
        };
        let lane_rows = |x: &MlxArray, lane: i32| {
            let shape = x.shape();
            slice(
                x,
                &[lane, 0, 0],
                &[lane + 1, shape[1], shape[2]],
                &[1, 1, 1],
                None,
            )
        };

        let mut ids = Vec::with_capacity(lanes.len() * block as usize);
        for (drafter, anchor, _) in lanes.iter_mut() {
            drafter.tree_alternatives = [None, None];
            ids.push(*anchor);
            ids.extend(std::iter::repeat_n(cfg.mask_token_id, block as usize - 1));
        }
        let mut x = reshape(
            &crate::model::embed_tokens(&ids, &target.token_embedding, hidden as usize),
            &[batch, block, hidden],
            None,
        );
        if cfg.input_embedding_scale != 1.0 {
            x = multiply(&x, &scalar(cfg.input_embedding_scale, x.dtype()), None);
        }

        // Context rows of every lane through one fc + norm, cut to the window.
        let hidden_refs: Vec<&MlxArray> = lanes.iter().map(|(_, _, h)| h).collect();
        let context_all = rms_norm(
            &weights.fc.project(&concatenate(&hidden_refs, 1, None)),
            Some(&weights.hidden_norm),
            cfg.rms_norm_eps,
            None,
        );
        let keep = cfg.sliding_window - 1;
        let mut contexts = Vec::with_capacity(lanes.len());
        let mut lane_context_rows = Vec::with_capacity(lanes.len());
        let mut start = 0;
        for (drafter, _, new_hidden) in lanes.iter_mut() {
            let rows = new_hidden.shape()[1];
            let skip = (rows - keep).max(0);
            contexts.push(slice(
                &context_all,
                &[0, start + skip, 0],
                &[1, start + rows, hidden],
                &[1, 1, 1],
                None,
            ));
            drafter.offset += skip;
            lane_context_rows.push(rows - skip);
            start += rows;
        }
        let context_refs: Vec<&MlxArray> = contexts.iter().collect();
        let projected = weights
            .context_kv
            .project(&concatenate(&context_refs, 1, None));
        let projected_width = projected.shape()[2];
        let mut chunks = Vec::with_capacity(lanes.len());
        let mut start = 0;
        for &rows in &lane_context_rows {
            let lane = slice(
                &projected,
                &[0, start, 0],
                &[1, start + rows, projected_width],
                &[1, 1, 1],
                None,
            );
            chunks.push(split(&lane, (cfg.num_layers * 2) as i32, 2, None));
            start += rows;
        }
        let offsets: Vec<(i32, i32)> = lanes
            .iter()
            .zip(&lane_context_rows)
            .map(|((drafter, _, _), &rows)| (drafter.offset, drafter.offset + rows))
            .collect();

        for (index, layer) in weights.layers.iter().enumerate() {
            let residual = x.clone();
            let normed = rms_norm(&x, Some(&layer.input_norm), cfg.rms_norm_eps, None);
            let (prepared, taps) = layer.attention_conv.prepare(cfg, &normed);
            let (q, k, v) = (
                project_rows(&layer.q, &prepared),
                project_rows(&layer.k, &prepared),
                project_rows(&layer.v, &prepared),
            );
            let mut attended = Vec::with_capacity(lanes.len());
            for (lane, (drafter, _, _)) in lanes.iter_mut().enumerate() {
                let (context_offset, block_offset) = offsets[lane];
                let lane_idx = lane as i32;
                attended.push(drafter.attention_core(
                    index,
                    layer,
                    &lane_rows(&q, lane_idx),
                    &lane_rows(&k, lane_idx),
                    &lane_rows(&v, lane_idx),
                    &chunks[lane][index * 2],
                    &chunks[lane][index * 2 + 1],
                    lane_context_rows[lane],
                    context_offset,
                    block_offset,
                ));
            }
            let attended_refs: Vec<&MlxArray> = attended.iter().collect();
            let attention = project_rows(&layer.o, &concatenate(&attended_refs, 0, None));
            x = add(
                &residual,
                &layer.attention_conv.finish(cfg, &attention, &taps),
                None,
            );
            let residual = x.clone();
            let normed = rms_norm(&x, Some(&layer.post_norm), cfg.rms_norm_eps, None);
            let (prepared, taps) = layer.mlp_conv.prepare(cfg, &normed);
            let activated = silu_mul(
                &project_rows(&layer.gate, &prepared),
                &project_rows(&layer.up, &prepared),
                None,
            );
            let mlp = project_rows(&layer.down, &activated);
            x = add(&residual, &layer.mlp_conv.finish(cfg, &mlp, &taps), None);
        }
        for ((drafter, _, _), &rows) in lanes.iter_mut().zip(&lane_context_rows) {
            drafter.offset += rows;
        }
        let normed = rms_norm(&x, Some(&weights.final_norm), cfg.rms_norm_eps, None);
        let draft_rows = batch * (block - 1);
        let draft_hidden = reshape(
            &slice(
                &normed,
                &[0, 1, 0],
                &[batch, block, hidden],
                &[1, 1, 1],
                None,
            ),
            &[1, draft_rows, hidden],
            None,
        );
        let head_dtype = target
            .lm_head
            .scales
            .as_ref()
            .map(|scales| scales.dtype())
            .filter(|dtype| matches!(dtype, MlxDtype::Float16 | MlxDtype::Bfloat16));
        let head_input = match head_dtype {
            Some(dtype) if dtype != draft_hidden.dtype() => astype(&draft_hidden, dtype, None),
            _ => draft_hidden,
        };
        let logits = crate::model::lm_head_project(&head_input, target);
        let logits = if cfg.output_multiplier != 1.0 {
            multiply(
                &logits,
                &scalar(cfg.output_multiplier, logits.dtype()),
                None,
            )
        } else {
            logits
        };
        if use_selector {
            // One selector over every lane: one top-k partition of all lanes'
            // logits, one gather per codebook, and a chain that walks the
            // positions with the lanes side by side. Per-lane selectors left
            // the drafter graph ending in a dozen small command buffers (one
            // partition per lane), which kept an asynchronous submission of
            // the draft from holding any real GPU work.
            let anchors: Vec<u32> = lanes.iter().map(|(_, anchor, _)| *anchor).collect();
            return Some(Self::select_packed(
                &weights.selector,
                cfg,
                &head_input,
                &logits,
                &anchors,
                block - 1,
            ));
        }
        let tokens = astype(&argmax(&logits, None), MlxDtype::Uint32, None);
        Some(
            (0..batch)
                .map(|lane| {
                    slice(
                        &tokens,
                        &[0, lane * (block - 1)],
                        &[1, (lane + 1) * (block - 1)],
                        &[1, 1],
                        None,
                    )
                })
                .collect(),
        )
    }

    fn draft_block_inner(
        &mut self,
        anchor: u32,
        new_hidden: &MlxArray,
        target: &ModelWeights,
        block: i32,
        capture_alternatives: bool,
    ) -> (MlxArray, Option<String>) {
        // The block graph appends KV and advances `offset` before the runner-up
        // read. Keep the pre-call cursor so a failed read does not stick.
        let cursor = self.snapshot_cursor();
        self.tree_alternatives = [None, None];
        let weights = self.weights.clone();
        let cfg = &weights.cfg;
        // Arm the fused multi-row matmuls for the whole block: the projections
        // run 3 to 8 rows (the block, and the new context rows), and the LM head
        // seven.
        let _fused_rows = QwenMtpVerifyQmmGuard::arm(true);
        let mut timer = std::time::Instant::now();
        // Opt-in only. With stage timing off, `mark` does not evaluate and
        // this never restores.
        macro_rules! drafted_mark {
            ($stage:expr, $array:expr) => {
                if let Err(error) = self.mark($stage, $array, &mut timer) {
                    return self.fail_draft(cursor.clone(), block, error);
                }
            };
        }
        let hidden = cfg.hidden_size;
        let kv_width = cfg.num_kv_heads * cfg.head_dim;
        let dtype = new_hidden.dtype();

        // Block input: [anchor, mask, mask, ...] embedded with the target table.
        let mut ids = vec![cfg.mask_token_id; block as usize];
        ids[0] = anchor;
        let mut embedded =
            crate::model::embed_tokens(&ids, &target.token_embedding, hidden as usize);
        if cfg.input_embedding_scale != 1.0 {
            embedded = multiply(
                &embedded,
                &scalar(cfg.input_embedding_scale, embedded.dtype()),
                None,
            );
        }
        let mut x = embedded;
        drafted_mark!("embed", &x);

        // Context for the new rows: project, normalise, and cut to the window.
        let mut context = rms_norm(
            &weights.fc.project(new_hidden),
            Some(&weights.hidden_norm),
            cfg.rms_norm_eps,
            None,
        );
        let mut rows = context.shape()[1];
        let keep = cfg.sliding_window - 1;
        if rows > keep {
            let skip = rows - keep;
            context = slice(
                &context,
                &[0, skip, 0],
                &[1, rows, hidden],
                &[1, 1, 1],
                None,
            );
            self.offset += skip;
            rows = keep;
        }
        drafted_mark!("context fc+norm", &context);
        let projected = weights.context_kv.project(&context);
        let chunks = split(&projected, (cfg.num_layers * 2) as i32, 2, None);
        drafted_mark!("context kv projection", &projected);

        let context_offset = self.offset;
        let block_offset = self.offset + rows;
        for (index, layer) in weights.layers.iter().enumerate() {
            let residual = x.clone();
            let normed = rms_norm(&x, Some(&layer.input_norm), cfg.rms_norm_eps, None);
            let (prepared, taps) = layer.attention_conv.prepare(cfg, &normed);
            let attention = self.attention(
                index,
                layer,
                &prepared,
                &chunks[index * 2],
                &chunks[index * 2 + 1],
                rows,
                kv_width,
                context_offset,
                block_offset,
            );
            x = add(
                &residual,
                &layer.attention_conv.finish(cfg, &attention, &taps),
                None,
            );
            drafted_mark!("attention block", &x);
            let residual = x.clone();
            let normed = rms_norm(&x, Some(&layer.post_norm), cfg.rms_norm_eps, None);
            let (prepared, taps) = layer.mlp_conv.prepare(cfg, &normed);
            let activated = silu_mul(
                &layer.gate.project(&prepared),
                &layer.up.project(&prepared),
                None,
            );
            let mlp = layer.down.project(&activated);
            x = add(&residual, &layer.mlp_conv.finish(cfg, &mlp, &taps), None);
            drafted_mark!("mlp block", &x);
        }
        self.offset += rows;
        let normed = rms_norm(&x, Some(&weights.final_norm), cfg.rms_norm_eps, None);
        let draft_hidden = slice(&normed, &[0, 1, 0], &[1, block, hidden], &[1, 1, 1], None);
        drafted_mark!("final norm", &draft_hidden);

        // The target's LM head over the seven draft positions, using the fused
        // multi-row MXFP4 matmul when the head is MXFP4.
        // The head's affine scales follow the target's activation dtype (fp16 on
        // M1/M2, bf16 elsewhere); the fused route declines mixed dtypes, so meet
        // the head's dtype here.
        let head_dtype = target
            .lm_head
            .scales
            .as_ref()
            .map(|scales| scales.dtype())
            .filter(|dtype| matches!(dtype, MlxDtype::Float16 | MlxDtype::Bfloat16));
        let head_input = match head_dtype {
            Some(dtype) if dtype != draft_hidden.dtype() => astype(&draft_hidden, dtype, None),
            _ => draft_hidden.clone(),
        };
        let logits = crate::model::lm_head_project(&head_input, target);
        let logits = if cfg.output_multiplier != 1.0 {
            multiply(
                &logits,
                &scalar(cfg.output_multiplier, logits.dtype()),
                None,
            )
        } else {
            logits
        };
        let _ = dtype;
        drafted_mark!("lm head", &logits);
        let (tokens, capture_error) = if !self.use_selector {
            // Greedy path: plain argmax of the target LM head.
            let tokens = astype(&argmax(&logits, None), MlxDtype::Uint32, None);
            drafted_mark!("argmax", &tokens);
            let capture_error = if capture_alternatives {
                match greedy_runner_up(&logits, &tokens) {
                    Ok(alternatives) => {
                        self.tree_alternatives = alternatives;
                        None
                    }
                    Err(error) => Some(error),
                }
            } else {
                None
            };
            (tokens, capture_error)
        } else {
            self.select(
                &weights.selector,
                cfg,
                &draft_hidden,
                &logits,
                anchor,
                capture_alternatives,
            )
        };
        if capture_error.is_some() {
            self.restore_cursor(cursor);
        }
        (tokens, capture_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        index: usize,
        layer: &Layer,
        x: &MlxArray,
        context_k: &MlxArray,
        context_v: &MlxArray,
        rows: i32,
        kv_width: i32,
        context_offset: i32,
        block_offset: i32,
    ) -> MlxArray {
        let _ = kv_width;
        let attended = self.attention_core(
            index,
            layer,
            &layer.q.project(x),
            &layer.k.project(x),
            &layer.v.project(x),
            context_k,
            context_v,
            rows,
            context_offset,
            block_offset,
        );
        layer.o.project(&attended)
    }

    /// Attention of one lane after the Q/K/V projections: `q_proj`, `k_proj`,
    /// `v_proj` are `[1, L, ·]` block rows. Appends the new context to this
    /// layer's cache and returns `[1, L, heads * head_dim]` before `o`.
    #[allow(clippy::too_many_arguments)]
    fn attention_core(
        &mut self,
        index: usize,
        layer: &Layer,
        q_proj: &MlxArray,
        k_proj: &MlxArray,
        v_proj: &MlxArray,
        context_k: &MlxArray,
        context_v: &MlxArray,
        rows: i32,
        context_offset: i32,
        block_offset: i32,
    ) -> MlxArray {
        let cfg = &self.weights.cfg;
        let length = q_proj.shape()[1];
        let (heads, kv_heads, head_dim) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let to_heads = |array: &MlxArray, count: i32, len: i32| {
            transpose(
                &reshape(array, &[1, len, count, head_dim], None),
                &[0, 2, 1, 3],
                None,
            )
        };
        let apply_rope = |array: &MlxArray, offset: i32| {
            rope(
                array,
                head_dim,
                false,
                Some(cfg.rope_theta),
                1.0,
                offset,
                None,
                None,
            )
        };
        let norm = |array: &MlxArray, weight: &MlxArray| {
            rms_norm(array, Some(weight), cfg.rms_norm_eps, None)
        };

        let queries = apply_rope(
            &to_heads(
                &norm(
                    &reshape(q_proj, &[1, length, heads, head_dim], None),
                    &layer.q_norm,
                ),
                heads,
                length,
            ),
            block_offset,
        );
        let block_keys = apply_rope(
            &to_heads(
                &norm(
                    &reshape(k_proj, &[1, length, kv_heads, head_dim], None),
                    &layer.k_norm,
                ),
                kv_heads,
                length,
            ),
            block_offset,
        );
        let block_values = to_heads(v_proj, kv_heads, length);
        let context_keys = apply_rope(
            &to_heads(
                &norm(
                    &reshape(context_k, &[1, rows, kv_heads, head_dim], None),
                    &layer.k_norm,
                ),
                kv_heads,
                rows,
            ),
            context_offset,
        );
        let context_values = to_heads(context_v, kv_heads, rows);

        // Append the new context to this layer's cache, trimming the oldest rows.
        let cache = &mut self.caches[index];
        let (keys, values) = match (cache.keys.take(), cache.values.take()) {
            (Some(old_keys), Some(old_values)) => {
                let cached = old_keys.shape()[2];
                let keep = retained_old_rows(cached, cfg.sliding_window - 1);
                let trim = |array: &MlxArray| {
                    if keep < cached {
                        slice(
                            array,
                            &[0, 0, cached - keep, 0],
                            &[1, kv_heads, cached, head_dim],
                            &[1, 1, 1, 1],
                            None,
                        )
                    } else {
                        array.clone()
                    }
                };
                (
                    concatenate(&[&trim(&old_keys), &context_keys], 2, None),
                    concatenate(&[&trim(&old_values), &context_values], 2, None),
                )
            }
            _ => (context_keys, context_values),
        };
        cache.keys = Some(keys.clone());
        cache.values = Some(values.clone());

        let all_keys = concatenate(&[&keys, &block_keys], 2, None);
        let all_values = concatenate(&[&values, &block_values], 2, None);
        let scale = (head_dim as f32).powf(-0.5);
        let attended =
            scaled_dot_product_attention(&queries, &all_keys, &all_values, scale, false, None);
        reshape(
            &transpose(&attended, &[0, 2, 1, 3], None),
            &[1, length, heads * head_dim],
            None,
        )
    }

    /// [`Self::select`] for a packed draft: `draft_hidden` is `[1, L·P, hidden]`
    /// and `logits` `[1, L·P, vocab]` for `L` lanes of `P` proposals, `anchors`
    /// one token per lane. Returns each lane's `[1, P]` draft ids. Same
    /// candidates, scores and picks as the per-lane selector, computed with
    /// the lanes stacked on a leading axis.
    fn select_packed(
        selector: &Selector,
        cfg: &Dflash2Config,
        draft_hidden: &MlxArray,
        logits: &MlxArray,
        anchors: &[u32],
        proposals: i32,
    ) -> Vec<MlxArray> {
        let lanes = anchors.len() as i32;
        let vocab = logits.shape()[2];
        let hidden = draft_hidden.shape()[2];
        let top_k = cfg.selector_top_k;
        let rank = cfg.selector_rank;
        let logits = reshape(logits, &[lanes, proposals, vocab], None);
        let candidates = top_k_indices(&logits, top_k).unwrap_or_else(|| {
            slice(
                &argpartition_axis(&logits, vocab - top_k, 2, None),
                &[0, 0, vocab - top_k],
                &[lanes, proposals, vocab],
                &[1, 1, 1],
                None,
            )
        });
        let unary = take_along_axis(&logits, &candidates, 2, None);
        let projected = matmul(
            &reshape(draft_hidden, &[lanes, proposals, hidden], None),
            &selector.hidden_projection_t,
            None,
        );
        let anchor_arr = MlxArray::from_raw_data(
            anchors.as_ptr().cast::<u8>(),
            std::mem::size_of_val(anchors),
            &[lanes],
            MlxDtype::Uint32,
        );
        let successor_all = take(&selector.successor_codebook, &candidates, 0, None);
        let predecessor_all = take(&selector.predecessor_codebook, &candidates, 0, None);
        let mut predecessor_row = take(&selector.predecessor_codebook, &anchor_arr, 0, None);
        let mut path = Vec::with_capacity(proposals as usize);
        for position in 0..proposals {
            let slot = |array: &MlxArray, width: i32| {
                reshape(
                    &slice(
                        array,
                        &[0, position, 0],
                        &[lanes, position + 1, width],
                        &[1, 1, 1],
                        None,
                    ),
                    &[lanes, width],
                    None,
                )
            };
            let rows = |array: &MlxArray| {
                reshape(
                    &slice(
                        array,
                        &[0, position, 0, 0],
                        &[lanes, position + 1, top_k, rank],
                        &[1, 1, 1, 1],
                        None,
                    ),
                    &[lanes, top_k, rank],
                    None,
                )
            };
            let position_candidates = slot(&candidates, top_k);
            let position_hidden = slot(&projected, rank);
            let edges = sum_axis(
                &multiply(
                    &multiply(
                        &reshape(&predecessor_row, &[lanes, 1, rank], None),
                        &reshape(&position_hidden, &[lanes, 1, rank], None),
                        None,
                    ),
                    &rows(&successor_all),
                    None,
                ),
                2,
                false,
                None,
            );
            let scores = add(&slot(&unary, top_k), &edges, None);
            let chosen = reshape(&argmax(&scores, None), &[lanes, 1], None);
            path.push(take_along_axis(&position_candidates, &chosen, 1, None));
            predecessor_row = reshape(
                &take_along_axis(
                    &rows(&predecessor_all),
                    &reshape(&chosen, &[lanes, 1, 1], None),
                    1,
                    None,
                ),
                &[lanes, rank],
                None,
            );
        }
        let refs: Vec<&MlxArray> = path.iter().collect();
        let tokens = astype(&concatenate(&refs, 1, None), MlxDtype::Uint32, None);
        (0..lanes)
            .map(|lane| slice(&tokens, &[lane, 0], &[lane + 1, proposals], &[1, 1], None))
            .collect()
    }

    /// Chain-aware rescoring of each position's top-k logits: the score of a
    /// candidate is its logit plus a low-rank edge score against the previously
    /// chosen token, and the best candidate is taken position by position.
    fn select(
        &mut self,
        selector: &Selector,
        cfg: &Dflash2Config,
        draft_hidden: &MlxArray,
        logits: &MlxArray,
        anchor: u32,
        capture_alternatives: bool,
    ) -> (MlxArray, Option<String>) {
        let shape = logits.shape();
        let proposals = shape[1];
        let vocab = shape[2];
        let top_k = cfg.selector_top_k;
        let candidates = top_k_indices(logits, top_k).unwrap_or_else(|| {
            slice(
                &argpartition_axis(logits, vocab - top_k, 2, None),
                &[0, 0, vocab - top_k],
                &[1, proposals, vocab],
                &[1, 1, 1],
                None,
            )
        });
        let unary = take_along_axis(logits, &candidates, 2, None);
        let projected = matmul(draft_hidden, &selector.hidden_projection_t, None);
        let anchor_arr = MlxArray::from_raw_data(
            (&anchor as *const u32).cast::<u8>(),
            std::mem::size_of::<u32>(),
            &[1],
            MlxDtype::Uint32,
        );
        let rank = cfg.selector_rank;
        // Gather every codebook row the chain can need up front: the successor
        // rows of all candidates, and the predecessor rows of all candidates
        // (position `p`'s predecessor is the token chosen at `p - 1`, one of
        // its candidates). The per-position chain then only indexes these
        // small tensors. Each codebook is 63M elements, and MLX starts a new
        // command buffer whenever a buffer's distinct inputs exceed its byte
        // cap, so touching a codebook per position used to leave the drafter
        // graph ending in a dozen tiny command buffers; those buffers kept an
        // asynchronous draft submission from holding any real GPU work.
        let successor_all = take(&selector.successor_codebook, &candidates, 0, None);
        let predecessor_all = take(&selector.predecessor_codebook, &candidates, 0, None);
        let mut predecessor_row = take(&selector.predecessor_codebook, &anchor_arr, 0, None);
        let mut path = Vec::with_capacity(proposals as usize);
        let mut alt_slots = capture_alternatives.then(Vec::new);
        for position in 0..proposals {
            let slot = |array: &MlxArray, width: i32| {
                reshape(
                    &slice(
                        array,
                        &[0, position, 0],
                        &[1, position + 1, width],
                        &[1, 1, 1],
                        None,
                    ),
                    &[1, width],
                    None,
                )
            };
            let rows = |array: &MlxArray| {
                reshape(
                    &slice(
                        array,
                        &[0, position, 0, 0],
                        &[1, position + 1, top_k, rank],
                        &[1, 1, 1, 1],
                        None,
                    ),
                    &[1, top_k, rank],
                    None,
                )
            };
            let position_candidates = slot(&candidates, top_k);
            let position_hidden = slot(&projected, rank);
            let successor_rows = rows(&successor_all);
            let edges = sum_axis(
                &multiply(
                    &multiply(
                        &reshape(&predecessor_row, &[1, 1, rank], None),
                        &reshape(&position_hidden, &[1, 1, rank], None),
                        None,
                    ),
                    &successor_rows,
                    None,
                ),
                2,
                false,
                None,
            );
            let scores = add(&slot(&unary, top_k), &edges, None);
            if let Some(slots) = alt_slots.as_mut()
                && position < 2
            {
                slots.push((position_candidates.clone(), scores.clone()));
            }
            let chosen = argmax(&scores, None);
            let chosen_index = reshape(&chosen, &[1, 1], None);
            let predecessor = reshape(
                &take_along_axis(&position_candidates, &chosen_index, 1, None),
                &[1],
                None,
            );
            // The chosen candidate's predecessor row for the next position.
            predecessor_row = reshape(
                &take_along_axis(
                    &rows(&predecessor_all),
                    &reshape(&chosen, &[1, 1, 1], None),
                    1,
                    None,
                ),
                &[1, rank],
                None,
            );
            path.push(predecessor);
        }
        let capture_error = if let Some(slots) = alt_slots.as_ref() {
            match selector_runner_up(slots, &path) {
                Ok(alternatives) => {
                    self.tree_alternatives = alternatives;
                    None
                }
                Err(error) => Some(error),
            }
        } else {
            None
        };
        let refs: Vec<&MlxArray> = path.iter().collect();
        (
            astype(&stack(&refs, 1, None), MlxDtype::Uint32, None),
            capture_error,
        )
    }
}

/// Runner-up token at the first two greedy draft positions. The primary id is
/// read from `drafted`, the same array the caller returns, so a tie cannot
/// make the alternative equal the committed chain.
fn greedy_runner_up(logits: &MlxArray, drafted: &MlxArray) -> Result<[Option<u32>; 2], String> {
    let shape = logits.shape();
    if shape.len() != 3 || shape[1] < 1 || shape[2] < 2 {
        return Ok([None, None]);
    }
    let positions = shape[1].min(2);
    let vocab = shape[2];
    let head = slice(logits, &[0, 0, 0], &[1, positions, vocab], &[1, 1, 1], None);
    let partitioned = argpartition_axis(&head, vocab - 2, 2, None);
    // The tail of each row is a view. Copy it before reading ids, or a
    // two-row slice is not a dense `[positions, 2]` buffer.
    let top = astype(
        &contiguous(
            &slice(
                &partitioned,
                &[0, 0, vocab - 2],
                &[1, positions, vocab],
                &[1, 1, 1],
                None,
            ),
            None,
        ),
        MlxDtype::Uint32,
        None,
    );
    let primary = astype(
        &contiguous(
            &slice(drafted, &[0, 0], &[1, positions], &[1, 1], None),
            None,
        ),
        MlxDtype::Uint32,
        None,
    );
    // Draft capture only: the request cache is not written on this path.
    // The drafter KV was updated while the block graph was built; the caller
    // restores that cursor when this returns `Err`.
    if let Some(error) = take_dflash2_capture_verify_eval_error() {
        tracing::warn!(
            target: "ax_engine_mlx::dflash2",
            %error,
            "DFlash2 runner-up capture failed; declining speculation"
        );
        return Err(error);
    }
    if let Err(error) = mlx_sys::try_eval(&[&top, &primary]) {
        tracing::warn!(
            target: "ax_engine_mlx::dflash2",
            %error,
            "DFlash2 runner-up capture failed; declining speculation"
        );
        return Err(error);
    }
    let primary_ids = primary.data_u32();
    let top_ids = top.data_u32();
    let mut out = [None, None];
    if primary_ids.len() < positions as usize || top_ids.len() < (positions as usize) * 2 {
        return Ok(out);
    }
    for position in 0..positions as usize {
        let primary = primary_ids[position];
        let first = top_ids[position * 2];
        let second = top_ids[position * 2 + 1];
        out[position] = [first, second].into_iter().find(|token| *token != primary);
    }
    Ok(out)
}

/// Highest-scoring candidate that is not `primary`. Equal scores keep the
/// earlier candidate.
fn distinct_runner_up(primary: u32, candidates: &[u32], scores: &[f32]) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (&token, &score) in candidates.iter().zip(scores) {
        if token == primary || !score.is_finite() {
            continue;
        }
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((token, score));
        }
    }
    best.map(|(token, _)| token)
}

fn selector_runner_up(
    slots: &[(MlxArray, MlxArray)],
    chosen: &[MlxArray],
) -> Result<[Option<u32>; 2], String> {
    let mut out = [None, None];
    for (index, ((candidates, scores), chosen)) in slots.iter().zip(chosen).enumerate() {
        if index >= 2 {
            break;
        }
        let candidates = astype(candidates, MlxDtype::Uint32, None);
        let scores = astype(scores, MlxDtype::Float32, None);
        let chosen = astype(chosen, MlxDtype::Uint32, None);
        // Draft capture only: the request cache is not written on this path.
        // The drafter KV was updated while the block graph was built; the caller
        // restores that cursor when this returns `Err`.
        if let Some(error) = take_dflash2_capture_verify_eval_error() {
            tracing::warn!(
                target: "ax_engine_mlx::dflash2",
                %error,
                "DFlash2 runner-up capture failed; declining speculation"
            );
            return Err(error);
        }
        if let Err(error) = mlx_sys::try_eval(&[&candidates, &scores, &chosen]) {
            tracing::warn!(
                target: "ax_engine_mlx::dflash2",
                %error,
                "DFlash2 runner-up capture failed; declining speculation"
            );
            return Err(error);
        }
        let Some(primary) = chosen.data_u32().first().copied() else {
            continue;
        };
        out[index] = distinct_runner_up(primary, candidates.data_u32(), scores.data_f32());
    }
    Ok(out)
}

/// A drafter AX finds by discovery: the Hugging Face snapshot it is pinned to
/// and the source model of the target it was trained for.
#[derive(Debug, PartialEq)]
pub(crate) struct AdmittedDrafter {
    /// Hugging Face repository of the target's source model, lowercase.
    target: &'static str,
    repo: &'static str,
    revision: &'static str,
    config_sha256: &'static str,
    weights_sha256: &'static str,
    /// Step costs that let a low-acceptance request leave DFlash2, measured
    /// for this target on the primary hardware tier. `None` keeps every
    /// request on DFlash2.
    pub(crate) handoff: Option<HandoffCosts>,
}

/// Wall time of the routes a request can leave DFlash2 for, relative to a
/// four-row DFlash2 round on the same target. The ratio depends on the target:
/// a dense model verifies four rows for little more than one, an MoE model
/// activates more experts per row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HandoffCosts {
    /// One MTP step at depth three (four rows): the round less the drafter.
    pub(crate) mtp_step: f64,
    /// One direct decode step.
    pub(crate) direct_step: f64,
}

impl AdmittedDrafter {
    /// Directory name of the drafter's Hugging Face cache entry.
    fn hf_entry(&self) -> String {
        format!("models--{}", self.repo.replace('/', "--"))
    }
}

/// Every drafter discovery may load, each bound to its target's source model.
/// The drafters' shapes also fit other checkpoints (Qwen 3.6 27B and 3.8
/// fine-tunes match the 27B drafter), whose weights they never saw.
pub(crate) const ADMITTED_DRAFTERS: &[AdmittedDrafter] = &[
    AdmittedDrafter {
        target: "qwen/qwen3.8-27b",
        repo: "z-lab/Qwen3.8-27B-DFlash2",
        revision: "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
        config_sha256: "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980",
        weights_sha256: "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c",
        // DFlash2 beats direct decode on every measured prompt (dense: a direct
        // step costs 0.71 of a four-row round on the M5 Max) and ties MTP on
        // prose, so requests stay on DFlash2.
        handoff: None,
    },
    AdmittedDrafter {
        target: "qwen/qwen3.6-35b-a3b",
        repo: "incoai/Qwen3.6-35B-A3B-DFlash2",
        revision: "51ef7b6923ad6c14cb1bb41c37a9041446496ab6",
        config_sha256: "86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59",
        weights_sha256: "d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d",
        // M5 Max: an MTP step is 0.82 of a four-row round and a direct step
        // 0.47 (M4 Pro: 0.73 and 0.41; the M5 values keep requests on DFlash2
        // longer there). Prose loses up to 25% to MTP without a handoff.
        handoff: Some(HandoffCosts {
            mtp_step: 0.82,
            direct_step: 0.47,
        }),
    },
];

struct DrafterArtifact {
    config_sha256: String,
    weights_sha256: String,
}

impl DrafterArtifact {
    fn read(dir: &Path) -> Result<Self, String> {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let digest = |name: &str| -> Result<String, String> {
            let mut file = std::fs::File::open(dir.join(name)).map_err(|e| e.to_string())?;
            let mut hash = Sha256::new();
            let mut buffer = vec![0u8; 1024 * 1024];
            loop {
                let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            Ok(crate::artifact_identity::hex_digest(&hash.finalize()))
        };
        Ok(Self {
            config_sha256: digest("config.json")?,
            weights_sha256: digest("model.safetensors")?,
        })
    }

    /// The admitted drafter whose both digests this checkpoint matches.
    fn admitted(&self) -> Option<&'static AdmittedDrafter> {
        ADMITTED_DRAFTERS.iter().find(|drafter| {
            self.config_sha256 == drafter.config_sha256
                && self.weights_sha256 == drafter.weights_sha256
        })
    }

    fn identity(&self) -> String {
        let (repo, revision) = self
            .admitted()
            .map_or(("dflash2", "explicit-local"), |drafter| {
                (drafter.repo, drafter.revision)
            });
        format!(
            "{repo}@{revision};config_sha256={};weights_sha256={}",
            self.config_sha256, self.weights_sha256
        )
    }
}

/// Whether `dir` holds a drafter checkpoint.
fn is_drafter_dir(dir: &Path) -> bool {
    dir.join("config.json").is_file() && dir.join("model.safetensors").is_file()
}

/// The admitted drafter for the target in `model_dir`, decided by the source
/// model the pack records: `axquant_manifest.json` `source_model.model_id`,
/// else the repository or directory name in `config.json` `_name_or_path`.
pub(crate) fn admitted_drafter_for(model_dir: &Path) -> Option<&'static AdmittedDrafter> {
    let read = |name: &str| {
        std::fs::read_to_string(model_dir.join(name))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
    };
    if let Some(manifest) = read("axquant_manifest.json") {
        let id = manifest
            .get("source_model")
            .and_then(|source| source.get("model_id"))
            .and_then(serde_json::Value::as_str)?
            .trim()
            .to_ascii_lowercase();
        return ADMITTED_DRAFTERS
            .iter()
            .find(|drafter| drafter.target == id);
    }
    let name = read("config.json")?
        .get("_name_or_path")
        .and_then(serde_json::Value::as_str)?
        .to_ascii_lowercase();
    // Compare the whole repository or directory name: a fine-tune such as
    // `Swift-1.5-Qwen3.8-27b` merely ends with the base name.
    let repo = name.trim_end_matches('/').rsplit('/').next()?.to_owned();
    ADMITTED_DRAFTERS
        .iter()
        .find(|drafter| drafter.target.rsplit('/').next() == Some(repo.as_str()))
}

/// Finds the drafter checkpoint to load: the explicit directory (which is never
/// substituted when it is wrong), else `dflash2/` inside the model directory,
/// else the pinned snapshot of the target's admitted drafter in the first
/// Hugging Face cache root that has one.
pub(crate) fn locate_drafter(
    explicit: Option<&Path>,
    model_dir: &Path,
    admitted: Option<&AdmittedDrafter>,
    hf_cache_roots: &[std::path::PathBuf],
) -> Option<std::path::PathBuf> {
    if let Some(dir) = explicit {
        return is_drafter_dir(dir).then(|| dir.to_path_buf());
    }
    let bundled = model_dir.join("dflash2");
    if is_drafter_dir(&bundled) {
        return Some(bundled);
    }
    let admitted = admitted?;
    for root in hf_cache_roots {
        let snapshot = root
            .join(admitted.hf_entry())
            .join("snapshots")
            .join(admitted.revision);
        if is_drafter_dir(&snapshot) {
            return Some(snapshot);
        }
    }
    None
}

#[cfg(test)]
thread_local! {
    static DFLASH2_CAPTURE_VERIFY_EVAL_ERROR: std::cell::Cell<Option<&'static str>> =
        const { std::cell::Cell::new(None) };
}

/// One-shot fault for the next DFlash2 capture or verify read.
///
/// Production builds never set it. Tests use it to make the fallible read
/// return the same error a failed `try_eval` would, after the drafter cursor
/// has already moved.
pub(crate) fn take_dflash2_capture_verify_eval_error() -> Option<String> {
    #[cfg(test)]
    {
        DFLASH2_CAPTURE_VERIFY_EVAL_ERROR.with(|slot| slot.take().map(str::to_owned))
    }
    #[cfg(not(test))]
    {
        None
    }
}

/// Arms [`take_dflash2_capture_verify_eval_error`] for the next read.
#[cfg(test)]
pub(crate) fn fail_next_dflash2_capture_verify_eval_for_test(message: &'static str) {
    DFLASH2_CAPTURE_VERIFY_EVAL_ERROR.with(|slot| slot.set(Some(message)));
}

/// One-layer drafter and a dense target embedding / LM head, for the eval-failure
/// round test. Shapes satisfy the drafter's block graph; the weights are zeros.
#[cfg(test)]
pub(crate) fn dflash2_tiny_fixture() -> (std::sync::Arc<Dflash2Weights>, ModelWeights) {
    dflash2_fixture_with(&|shape| mlx_sys::zeros(shape, MlxDtype::Bfloat16, None))
}

/// [`dflash2_tiny_fixture`] with every tensor drawn from `fill`.
#[cfg(test)]
pub(crate) fn dflash2_fixture_with(
    fill: &dyn Fn(&[i32]) -> MlxArray,
) -> (std::sync::Arc<Dflash2Weights>, ModelWeights) {
    let hidden = 8i32;
    let head_dim = 4i32;
    let kv = 4i32;
    let inter = 8i32;
    let vocab = 16i32;
    let kernel = 2i32;
    let groups = 2i32;
    let rank = 4i32;
    let z = |shape: &[i32]| fill(shape);
    let proj = |out: i32, inn: i32| Projection::new(&z(&[out, inn]), DrafterPrecision::Bf16);
    let conv = || GroupedConv {
        base: z(&[2, kernel, hidden]),
        projection_t: z(&[hidden, 2 * kernel * groups]),
    };
    let cfg = Dflash2Config {
        hidden_size: hidden,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim,
        intermediate_size: inter,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        sliding_window: 8,
        block_size: 4,
        mask_token_id: 1,
        target_layer_ids: vec![0],
        conv_group_size: hidden / groups,
        conv_kernel_size: kernel,
        selector_rank: rank,
        selector_top_k: 2,
        vocab_size: vocab,
        input_embedding_scale: 1.0,
        output_multiplier: 1.0,
    };
    let k_weight = z(&[kv, hidden]);
    let v_weight = z(&[kv, hidden]);
    let context_kv = Projection::new(
        &concatenate(&[&k_weight, &v_weight], 0, None),
        DrafterPrecision::Bf16,
    );
    let layer = Layer {
        q: proj(hidden, hidden),
        k: Projection::new(&k_weight, DrafterPrecision::Bf16),
        v: Projection::new(&v_weight, DrafterPrecision::Bf16),
        o: proj(hidden, hidden),
        q_norm: z(&[head_dim]),
        k_norm: z(&[head_dim]),
        gate: proj(inter, hidden),
        up: proj(inter, hidden),
        down: proj(hidden, inter),
        input_norm: z(&[hidden]),
        post_norm: z(&[hidden]),
        attention_conv: conv(),
        mlp_conv: conv(),
    };
    let weights = std::sync::Arc::new(Dflash2Weights {
        cfg,
        fc: proj(hidden, hidden),
        hidden_norm: z(&[hidden]),
        layers: vec![layer],
        final_norm: z(&[hidden]),
        selector: Selector {
            predecessor_codebook: z(&[vocab, rank]),
            successor_codebook: z(&[vocab, rank]),
            hidden_projection_t: z(&[hidden, rank]),
        },
        context_kv,
        precision: DrafterPrecision::Bf16,
        artifact: DrafterArtifact {
            config_sha256: "fixture-config".into(),
            weights_sha256: "fixture-weights".into(),
        },
    });
    let embedding = QuantizedWeight::new(z(&[vocab, hidden]), None, None);
    let target = ModelWeights {
        token_embedding: embedding.clone(),
        final_norm: Some(z(&[hidden])),
        lm_head: embedding,
        lm_head_f32: std::sync::OnceLock::new(),
        layers: Vec::new(),
        per_layer_embed: None,
        per_layer_model_proj: None,
        per_layer_proj_norm: None,
        mtp: None,
        gemma4_assistant_mtp: Default::default(),
        assistant_pre_projection: None,
        assistant_post_projection: None,
        embedding_dense_0: None,
        embedding_dense_1: None,
        gemma4_unified_vision: None,
        gemma4_unified_audio: None,
        gemma4_vl_vision: None,
        diffusion_self_conditioning: None,
        glm_mtp: None,
        deepseek_v4_head: None,
        deepseek_v4_nextn: None,
        unlimited_ocr_vision: None,
        qwen3_vl_vision: None,
        minicpm_v46_vision: None,
        nemotron_omni: None,
        expert_stream: None,
        qwen4_exp: None,
        qwen4_exp_mtp: None,
    };
    (weights, target)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn stage_timing_eval_failure_restores_the_drafter_cursor() {
        let (weights, target) = dflash2_tiny_fixture();
        let mut drafter = Dflash2Drafter::new(weights.clone());
        drafter.set_stage_timing(true);
        let width = weights.config().target_hidden_width();
        let hidden = mlx_sys::zeros(&[1, 2, width], MlxDtype::Bfloat16, None);
        let block = weights.config().block_size;
        fail_next_dflash2_capture_verify_eval_for_test("injected draft stage eval failure");
        let failed = drafter.draft_block(3, &hidden, &target, block);
        assert!(failed.is_err(), "a stage read must fail the draft");
        assert_eq!(drafter.cache_offset(), 0);
        assert_eq!(drafter.cache_rows(), 0);

        drafter.set_stage_timing(false);
        let drafts = drafter
            .draft_block(3, &hidden, &target, block)
            .expect("next draft");
        mlx_sys::eval(&[&drafts]);
        assert_eq!(drafts.shape()[1], block - 1);
        assert!(drafter.cache_offset() > 0);
        assert!(drafter.cache_rows() > 0);
    }

    #[test]
    fn packed_drafts_match_each_lane_drafted_alone() {
        let seed = std::cell::Cell::new(0x9e37_79b9_u32);
        let fill = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    let mut x = seed.get();
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    seed.set(x);
                    (x % 2001) as f32 / 1000.0 - 1.0
                })
                .collect();
            astype(
                &reshape(&MlxArray::from_f32_slice(&values), shape, None),
                MlxDtype::Bfloat16,
                None,
            )
        };
        let (weights, target) = dflash2_fixture_with(&fill);
        let width = weights.config().target_hidden_width();
        let block = weights.config().block_size;
        // Context rows per round, one entry per lane (9 > the window cap).
        let rows = [[2, 5, 1], [3, 1, 9]];
        let rounds: Vec<Vec<MlxArray>> = rows
            .iter()
            .map(|round| round.iter().map(|&r| fill(&[1, r, width])).collect())
            .collect();
        let anchors = [3_u32, 7, 11];
        for use_selector in [false, true] {
            let mut solo: Vec<Dflash2Drafter> = (0..3)
                .map(|_| Dflash2Drafter::new(weights.clone()).with_selector(use_selector))
                .collect();
            let mut packed: Vec<Dflash2Drafter> = (0..3)
                .map(|_| Dflash2Drafter::new(weights.clone()).with_selector(use_selector))
                .collect();
            for (round, hiddens) in rounds.iter().enumerate() {
                let expected: Vec<Vec<u32>> = solo
                    .iter_mut()
                    .enumerate()
                    .map(|(lane, drafter)| {
                        let ids = drafter
                            .draft_block(anchors[lane], &hiddens[lane], &target, block)
                            .expect("solo draft");
                        mlx_sys::eval(&[&ids]);
                        ids.data_u32().to_vec()
                    })
                    .collect();
                let mut lanes: Vec<(&mut Dflash2Drafter, u32, MlxArray)> = packed
                    .iter_mut()
                    .enumerate()
                    .map(|(lane, drafter)| (drafter, anchors[lane], hiddens[lane].clone()))
                    .collect();
                let drafted = Dflash2Drafter::draft_blocks_packed(&mut lanes, &target, block)
                    .expect("packed");
                for (lane, ids) in drafted.iter().enumerate() {
                    mlx_sys::eval(&[ids]);
                    assert_eq!(ids.shape(), vec![1, block - 1]);
                    assert_eq!(
                        ids.data_u32(),
                        expected[lane].as_slice(),
                        "round {round} lane {lane}"
                    );
                }
                for (a, b) in solo.iter().zip(&packed) {
                    assert_eq!(a.cache_offset(), b.cache_offset());
                    assert_eq!(a.cache_rows(), b.cache_rows());
                }
            }
        }
    }

    #[test]
    fn top_k_kernel_matches_argpartition_sets() {
        for (dtype, vocab) in [
            (MlxDtype::Float32, 1000),
            (MlxDtype::Bfloat16, 4099),
            (MlxDtype::Float32, 248_000),
        ] {
            let mut seed = 0x2545_f491_u32;
            let n = 3 * 7 * vocab;
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    // Coarse values so exact ties occur.
                    ((seed % 4001) as f32 / 100.0) - 20.0
                })
                .collect();
            let logits = astype(
                &reshape(&MlxArray::from_f32_slice(&values), &[3, 7, vocab], None),
                dtype,
                None,
            );
            let k = 16;
            let ours = top_k_indices(&logits, k).expect("kernel applies");
            let reference = contiguous(
                &slice(
                    &argpartition_axis(&logits, vocab - k, 2, None),
                    &[0, 0, vocab - k],
                    &[3, 7, vocab],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            mlx_sys::eval(&[&ours, &reference]);
            assert_eq!(ours.shape(), vec![3, 7, k]);
            let ours_ids = ours.data_u32().to_vec();
            let reference_ids = reference.data_u32().to_vec();
            let logit_values = astype(&logits, MlxDtype::Float32, None);
            mlx_sys::eval(&[&logit_values]);
            let flat = logit_values.data_f32().to_vec();
            for row in 0..21usize {
                let base = row * vocab as usize;
                let mut a: Vec<f32> = ours_ids[row * 16..row * 16 + 16]
                    .iter()
                    .map(|&id| flat[base + id as usize])
                    .collect();
                let mut b: Vec<f32> = reference_ids[row * 16..row * 16 + 16]
                    .iter()
                    .map(|&id| flat[base + id as usize])
                    .collect();
                a.sort_by(|x, y| y.total_cmp(x));
                b.sort_by(|x, y| y.total_cmp(x));
                // Same top-k values (ties at the boundary may pick other ids).
                assert_eq!(a, b, "row {row} dtype {dtype:?} vocab {vocab}");
                // Sorted descending, ids ascending within equal values.
                let ids = &ours_ids[row * 16..row * 16 + 16];
                for w in 0..15 {
                    let (v0, v1) = (
                        flat[base + ids[w] as usize],
                        flat[base + ids[w + 1] as usize],
                    );
                    assert!(
                        v0 > v1 || (v0 == v1 && ids[w] < ids[w + 1]),
                        "row {row} order"
                    );
                }
            }
        }
    }

    #[test]
    fn fused_grouped_conv_matches_the_op_chain() {
        let seed = std::cell::Cell::new(0x2545_f491_u32);
        let fill = |shape: &[i32], dtype: MlxDtype| {
            let n: i32 = shape.iter().product();
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    let mut x = seed.get();
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    seed.set(x);
                    (x % 4001) as f32 / 1000.0 - 2.0
                })
                .collect();
            astype(
                &reshape(&MlxArray::from_f32_slice(&values), shape, None),
                dtype,
                None,
            )
        };
        let (weights, _) = dflash2_tiny_fixture();
        let mut cfg = weights.config().clone();
        cfg.hidden_size = 64;
        cfg.conv_group_size = 16;
        for kernel in [1, 2, 3] {
            cfg.conv_kernel_size = kernel;
            let groups = cfg.hidden_size / cfg.conv_group_size;
            for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
                let conv = GroupedConv {
                    base: fill(&[2, kernel, cfg.hidden_size], MlxDtype::Bfloat16),
                    projection_t: fill(&[cfg.hidden_size, 2 * kernel * groups], dtype),
                };
                for (batch, length) in [(1, 8), (3, 5), (2, 1)] {
                    let hidden = fill(&[batch, length, cfg.hidden_size], dtype);
                    let dynamic = matmul(&hidden, &conv.projection_t, None);
                    for half in [0, 1] {
                        let fused = conv
                            .convolve_fused(&cfg, &hidden, &dynamic, half)
                            .expect("fused kernel");
                        let taps = reshape(
                            &slice(
                                &reshape(&dynamic, &[batch, length, 2, kernel, groups], None),
                                &[0, 0, half, 0, 0],
                                &[batch, length, half + 1, kernel, groups],
                                &[1, 1, 1, 1, 1],
                                None,
                            ),
                            &[batch, length, kernel, groups],
                            None,
                        );
                        let ops = conv.convolve_ops(&cfg, &hidden, &taps, half);
                        let (fused, ops) = (
                            astype(&fused, MlxDtype::Float32, None),
                            astype(&ops, MlxDtype::Float32, None),
                        );
                        mlx_sys::eval(&[&fused, &ops]);
                        assert_eq!(
                            fused.data_f32(),
                            ops.data_f32(),
                            "kernel {kernel} {dtype:?} batch {batch} length {length} half {half}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn skipped_rows_move_the_next_context_position() {
        let (weights, _) = dflash2_tiny_fixture();
        let mut drafter = Dflash2Drafter::new(weights);
        drafter.skip_rows(5);
        assert_eq!(drafter.cache_offset(), 5);
        assert_eq!(drafter.cache_rows(), 0);
    }

    const CONFIG: &str = r#"{
        "hidden_size": 5120, "num_hidden_layers": 5, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 17408,
        "rms_norm_eps": 1e-06, "sliding_window": 2048, "vocab_size": 248320,
        "is_causal": false, "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 248070, "selector_rank": 256, "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]}
    }"#;

    #[test]
    fn second_best_score_skips_the_primary_token() {
        assert_eq!(distinct_runner_up(4, &[4, 9, 2], &[3.0, 2.5, 2.5]), Some(9));
        assert_eq!(distinct_runner_up(4, &[4, 4], &[1.0, 1.0]), None);
        assert_eq!(distinct_runner_up(1, &[8, 3], &[f32::NAN, 0.5]), Some(3));
    }

    #[test]
    fn runner_up_capture_reads_the_other_top_token() {
        let logits = reshape(
            &MlxArray::from_f32_slice(&[0.0, 5.0, 1.0, 3.0, 9.0, 1.0, 2.0, 0.0]),
            &[1, 2, 4],
            None,
        );
        let primary = [1_u32, 0];
        let drafted = MlxArray::from_raw_data(
            primary.as_ptr().cast(),
            std::mem::size_of_val(&primary),
            &[1, 2],
            MlxDtype::Uint32,
        );
        assert_eq!(
            greedy_runner_up(&logits, &drafted).unwrap(),
            [Some(3), Some(2)]
        );
    }

    #[test]
    fn runner_up_capture_skips_a_head_that_has_no_second_token() {
        let logits = reshape(&MlxArray::from_f32_slice(&[1.0]), &[1, 1, 1], None);
        let primary = [0_u32];
        let drafted = MlxArray::from_raw_data(
            primary.as_ptr().cast(),
            std::mem::size_of_val(&primary),
            &[1, 1],
            MlxDtype::Uint32,
        );
        assert_eq!(greedy_runner_up(&logits, &drafted).unwrap(), [None, None]);
    }

    #[test]
    fn selector_runner_up_keeps_the_distinct_candidate_after_eval() {
        let candidates = [4_u32, 9, 2];
        let scores = [3.0_f32, 2.5, 2.5];
        let chosen = [4_u32];
        let slots = [(
            MlxArray::from_raw_data(
                candidates.as_ptr().cast(),
                std::mem::size_of_val(&candidates),
                &[3],
                MlxDtype::Uint32,
            ),
            MlxArray::from_f32_slice(&scores),
        )];
        let chosen_arr = [MlxArray::from_raw_data(
            chosen.as_ptr().cast(),
            std::mem::size_of_val(&chosen),
            &[1],
            MlxDtype::Uint32,
        )];
        assert_eq!(
            selector_runner_up(&slots, &chosen_arr).unwrap(),
            [Some(9), None]
        );
    }

    #[test]
    fn config_parses_the_released_checkpoint_shape() {
        let cfg = Dflash2Config::from_json(CONFIG).unwrap();
        assert_eq!(cfg.num_layers, 5);
        assert_eq!(cfg.block_size, 8);
        assert_eq!(cfg.target_hidden_width(), 25_600);
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert_eq!(cfg.input_embedding_scale, 1.0);
    }

    #[test]
    fn config_rejects_a_scaled_or_partial_rope() {
        let yarn = CONFIG.replace(
            "\"rope_type\": \"default\"",
            "\"rope_type\": \"yarn\", \"factor\": 4.0",
        );
        assert!(Dflash2Config::from_json(&yarn).is_err());
        let legacy = CONFIG.replace(
            "\"is_causal\": false,",
            "\"is_causal\": false, \"rope_scaling\": {\"type\": \"linear\", \"factor\": 2.0},",
        );
        assert!(Dflash2Config::from_json(&legacy).is_err());
        let partial = CONFIG.replace(
            "\"is_causal\": false,",
            "\"is_causal\": false, \"partial_rotary_factor\": 0.25,",
        );
        assert!(Dflash2Config::from_json(&partial).is_err());
        // A null rope_scaling and the default type keep the released drafter.
        let explicit_none = CONFIG.replace(
            "\"is_causal\": false,",
            "\"is_causal\": false, \"rope_scaling\": null, \"partial_rotary_factor\": 1.0,",
        );
        assert!(Dflash2Config::from_json(&explicit_none).is_ok());
    }

    #[test]
    fn config_rejects_what_it_cannot_serve() {
        let causal = CONFIG.replace("\"is_causal\": false", "\"is_causal\": true");
        assert!(Dflash2Config::from_json(&causal).is_err());
        let unordered = CONFIG.replace("[5, 19, 33, 47, 61]", "[19, 5]");
        assert!(Dflash2Config::from_json(&unordered).is_err());
        let softcap = CONFIG.replace(
            "\"block_size\": 8,",
            "\"block_size\": 8, \"final_logit_softcapping\": 30.0,",
        );
        assert!(Dflash2Config::from_json(&softcap).is_err());
        assert!(Dflash2Config::from_json("{}").is_err());
    }

    #[test]
    fn rotating_cache_keeps_the_rows_the_reference_keeps() {
        // A cache of max_size rows retains max_size - 1 old rows before appending.
        assert_eq!(retained_old_rows(10, 2047), 10);
        assert_eq!(retained_old_rows(2047, 2047), 2046);
        assert_eq!(retained_old_rows(5000, 2047), 2046);
        assert_eq!(retained_old_rows(0, 2047), 0);
    }

    fn conv_for_test(group: i32, kernel: i32, width: i32) -> (Dflash2Config, GroupedConv) {
        let mut cfg = Dflash2Config::from_json(CONFIG).unwrap();
        cfg.hidden_size = width;
        cfg.conv_group_size = group;
        cfg.conv_kernel_size = kernel;
        let groups = width / group;
        let base_data: Vec<f32> = (0..(2 * kernel * width) as usize)
            .map(|i| ((i % 7) as f32 - 3.0) / 8.0)
            .collect();
        let projection: Vec<f32> = (0..(width * 2 * kernel * groups) as usize)
            .map(|i| ((i % 5) as f32 - 2.0) / 64.0)
            .collect();
        (
            cfg,
            GroupedConv {
                base: reshape(
                    &MlxArray::from_f32_slice(&base_data),
                    &[2, kernel, width],
                    None,
                ),
                projection_t: reshape(
                    &MlxArray::from_f32_slice(&projection),
                    &[width, 2 * kernel * groups],
                    None,
                ),
            },
        )
    }

    #[test]
    fn convolution_matches_a_scalar_reference() {
        let (group, kernel, width, length) = (4_i32, 2_i32, 8_i32, 5_i32);
        let (cfg, conv) = conv_for_test(group, kernel, width);
        let input: Vec<f32> = (0..(length * width) as usize)
            .map(|i| ((i * 3 % 11) as f32 - 5.0) / 7.0)
            .collect();
        let hidden = reshape(&MlxArray::from_f32_slice(&input), &[1, length, width], None);
        let (prepared, taps) = conv.prepare(&cfg, &hidden);
        let (prepared, taps) = (
            mlx_sys::contiguous(&prepared, None),
            mlx_sys::contiguous(&taps, None),
        );
        mlx_sys::eval(&[&prepared, &taps]);

        // Scalar reference of prepare: taps[l, offset, g] = projection(hidden)[l, 0, offset, g].
        let groups = (width / group) as usize;
        let proj = conv.projection_t.shape();
        let projection_t = reshape(&conv.projection_t, &[proj[0], proj[1]], None);
        mlx_sys::eval(&[&projection_t, &conv.base]);
        let (projection_t, base_c) = (
            mlx_sys::contiguous(&projection_t, None),
            mlx_sys::contiguous(&conv.base, None),
        );
        mlx_sys::eval(&[&projection_t, &base_c]);
        let (w, base) = (projection_t.data_f32(), base_c.data_f32());
        let (width_u, kernel_u, length_u, group_u) = (
            width as usize,
            kernel as usize,
            length as usize,
            group as usize,
        );
        let proj_cols = 2 * kernel_u * groups;
        let dynamic = |l: usize, half: usize, offset: usize, g: usize| -> f32 {
            let column = (half * kernel_u + offset) * groups + g;
            (0..width_u)
                .map(|c| input[l * width_u + c] * w[c * proj_cols + column])
                .sum()
        };
        let expected = |l: usize, c: usize, half: usize| -> f32 {
            let g = c / group_u;
            (0..kernel_u)
                .map(|offset| {
                    let tap = dynamic(l, half, offset, g);
                    let shifted = if l >= offset {
                        input[(l - offset) * width_u + c]
                    } else {
                        0.0
                    };
                    (base[(half * kernel_u + offset) * width_u + c] + tap) * shifted
                })
                .sum()
        };
        let got = prepared.data_f32();
        for l in 0..length_u {
            for c in 0..width_u {
                let want = expected(l, c, 0);
                assert!(
                    (got[l * width_u + c] - want).abs() < 1e-3,
                    "l={l} c={c}: {} vs {want}",
                    got[l * width_u + c]
                );
            }
        }
        // The finish half uses the second set of taps and base kernel.
        let finished = mlx_sys::contiguous(&conv.finish(&cfg, &hidden, &taps), None);
        mlx_sys::eval(&[&finished]);
        let got = finished.data_f32();
        let tap_values = taps.data_f32();
        for l in 0..length_u {
            for c in 0..width_u {
                let g = c / group_u;
                let want: f32 = (0..kernel_u)
                    .map(|offset| {
                        let shifted = if l >= offset {
                            input[(l - offset) * width_u + c]
                        } else {
                            0.0
                        };
                        // `taps` is the full dynamic projection; finish reads
                        // its second half.
                        (base[(kernel_u + offset) * width_u + c]
                            + tap_values[((l * 2 + 1) * kernel_u + offset) * groups + g])
                            * shifted
                    })
                    .sum();
                assert!(
                    (got[l * width_u + c] - want).abs() < 1e-3,
                    "finish l={l} c={c}"
                );
            }
        }
    }

    fn env_dir(name: &str) -> Option<std::path::PathBuf> {
        std::env::var_os(name).map(std::path::PathBuf::from)
    }

    /// Replays mlx-vlm's DFlash2 rounds (candidate selector active) and requires
    /// the same draft tokens, round by round, from an empty drafter cache.
    ///
    /// AX_ENGINE_DFLASH2_DIR          drafter checkpoint (config.json, model.safetensors)
    /// AX_ENGINE_DFLASH2_TARGET_DIR   Qwen 3.8 27B MXFP4 pack
    /// AX_ENGINE_DFLASH2_GOLDEN_DIR   output of dflash2_golden.py
    #[test]
    #[ignore = "needs AX_ENGINE_DFLASH2_DIR, AX_ENGINE_DFLASH2_TARGET_DIR, AX_ENGINE_DFLASH2_GOLDEN_DIR"]
    fn drafter_reproduces_mlx_vlm_draft_tokens() {
        let (Some(drafter_dir), Some(target_dir), Some(golden_dir)) = (
            env_dir("AX_ENGINE_DFLASH2_DIR"),
            env_dir("AX_ENGINE_DFLASH2_TARGET_DIR"),
            env_dir("AX_ENGINE_DFLASH2_GOLDEN_DIR"),
        ) else {
            panic!(
                "set AX_ENGINE_DFLASH2_DIR, AX_ENGINE_DFLASH2_TARGET_DIR and AX_ENGINE_DFLASH2_GOLDEN_DIR"
            );
        };
        let artifacts = ax_engine_core::NativeModelArtifacts::from_dir(&target_dir).unwrap();
        let target = crate::weights::load_weights(&artifacts).unwrap();
        eprintln!(
            "target lm_head: mode {:?} bits {} group {} biases {} shape {:?}; embedding mode {:?} bits {}",
            target.lm_head.mlx_quantization_mode(),
            target.lm_head.bits,
            target.lm_head.group_size,
            target.lm_head.biases.is_some(),
            target.lm_head.weight.shape(),
            target.token_embedding.mlx_quantization_mode(),
            target.token_embedding.bits,
        );
        let precision = if std::env::var("AX_ENGINE_DFLASH2_PRECISION").is_ok_and(|v| v == "bf16") {
            DrafterPrecision::Bf16
        } else {
            DrafterPrecision::Mxfp4
        };
        let weights = std::sync::Arc::new(Dflash2Weights::load(&drafter_dir, precision).unwrap());
        weights.materialize();
        {
            let mut warm = Dflash2Drafter::new(weights.clone());
            let started = std::time::Instant::now();
            warm.warm_up(&target);
            eprintln!(
                "warm-up (kernel compile) {:.0} ms",
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        let mut goldens: Vec<_> = std::fs::read_dir(&golden_dir)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        goldens.sort();
        assert!(
            !goldens.is_empty(),
            "no golden files in {}",
            golden_dir.display()
        );

        let selector = std::env::var("AX_ENGINE_DFLASH2_SELECTOR").is_ok_and(|v| v == "1");
        let profile = std::env::var("AX_ENGINE_DFLASH2_PROFILE").is_ok_and(|v| v == "1");
        let mut stages: std::collections::BTreeMap<&'static str, (f64, usize)> = Default::default();
        let (mut tokens, mut equal) = (0_usize, 0_usize);
        let mut draft_ms: Vec<f64> = Vec::new();
        for path in goldens {
            let tensors = mlx_sys::load_safetensors(&path, None).unwrap();
            let rounds = tensors
                .keys()
                .filter(|name| name.starts_with("hidden_"))
                .count();
            let mut drafter = Dflash2Drafter::new(weights.clone()).with_selector(selector);
            drafter.set_stage_timing(profile);
            for round in 0..rounds {
                let hidden = &tensors[&format!("hidden_{round}")];
                let anchor = astype(&tensors[&format!("anchor_{round}")], MlxDtype::Uint32, None);
                let expected = astype(&tensors[&format!("draft_{round}")], MlxDtype::Uint32, None);
                mlx_sys::eval(&[hidden, &anchor, &expected]);
                let anchor = anchor.data_u32()[0];
                let started = std::time::Instant::now();
                let drafts = drafter
                    .draft_block(anchor, hidden, &target, weights.cfg.block_size)
                    .unwrap();
                mlx_sys::eval(&[&drafts]);
                draft_ms.push(started.elapsed().as_secs_f64() * 1e3);
                for (stage, ms) in drafter.take_stage_times() {
                    if round > 0 {
                        let entry = stages.entry(stage).or_default();
                        entry.0 += ms;
                        entry.1 += 1;
                    }
                }
                let got = drafts.data_u32().to_vec();
                let want = expected.data_u32().to_vec();
                tokens += want.len();
                equal += got.iter().zip(&want).filter(|(a, b)| a == b).count();
                eprintln!(
                    "{} round {round}: {} ctx rows, {:.1} ms\n  rust {got:?}\n  ref  {want:?}",
                    path.file_name().unwrap().to_string_lossy(),
                    hidden.shape()[1],
                    draft_ms.last().copied().unwrap_or_default()
                );
            }
        }
        let later = draft_ms.clone();
        eprintln!(
            "mean round time {:.1} ms over {} rounds",
            later.iter().sum::<f64>() / later.len() as f64,
            later.len()
        );
        let rounds_timed = (later.len() - 3).max(1) as f64; // rounds after each file's first
        for (stage, (total, _)) in &stages {
            eprintln!(
                "  stage {stage:24} {:7.2} ms per round",
                total / rounds_timed
            );
        }
        eprintln!("draft tokens equal: {equal}/{tokens}");
        // The bf16 drafter matches except where a near-tie flips a token (125/126
        // measured; the fused LM-head matmul sums in a different order than MLX's).
        // 4-bit weights move more near-ties (119/126) and cost 0.3% of the tokens
        // committed per round (5.90 against 5.92), so they get a looser bar.
        let required = if precision == DrafterPrecision::Bf16 {
            95
        } else {
            90
        };
        assert!(
            equal * 100 >= tokens * required,
            "drafter diverges from the mlx-vlm reference ({equal}/{tokens}, need {required}%)"
        );
    }

    /// AX's prefill, run under the layer-capture scope, must produce the target
    /// hidden states mlx-vlm captured for the same prompt (round 0 of the
    /// golden data): cosine similarity per captured layer and per token.
    #[test]
    #[ignore = "needs AX_ENGINE_DFLASH2_DIR, AX_ENGINE_DFLASH2_TARGET_DIR, AX_ENGINE_DFLASH2_GOLDEN_DIR"]
    fn captured_target_hidden_matches_mlx_vlm() {
        let (Some(drafter_dir), Some(target_dir), Some(golden_dir)) = (
            env_dir("AX_ENGINE_DFLASH2_DIR"),
            env_dir("AX_ENGINE_DFLASH2_TARGET_DIR"),
            env_dir("AX_ENGINE_DFLASH2_GOLDEN_DIR"),
        ) else {
            panic!(
                "set AX_ENGINE_DFLASH2_DIR, AX_ENGINE_DFLASH2_TARGET_DIR and AX_ENGINE_DFLASH2_GOLDEN_DIR"
            );
        };
        let cfg_text = std::fs::read_to_string(drafter_dir.join("config.json")).unwrap();
        let drafter_cfg = Dflash2Config::from_json(&cfg_text).unwrap();
        let artifacts = ax_engine_core::NativeModelArtifacts::from_dir(&target_dir).unwrap();
        let model_cfg = crate::model::ModelConfig::from_manifest(artifacts.manifest());
        let target = crate::weights::load_weights(&artifacts).unwrap();
        let width = drafter_cfg.hidden_size;
        let mut worst_mean = 1.0_f64;
        let mut files: Vec<_> = std::fs::read_dir(&golden_dir)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
            .collect();
        files.sort();
        for json_path in files {
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&json_path).unwrap()).unwrap();
            let tokens: Vec<u32> = meta["prompt_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as u32)
                .collect();
            let golden =
                mlx_sys::load_safetensors(&json_path.with_extension("safetensors"), None).unwrap();
            let reference = astype(&golden["hidden_0"], MlxDtype::Float32, None);
            let mut cache = crate::kv_cache::MlxKVCache::new(model_cfg.layer_count);
            let mut guard =
                crate::model::layer_capture::capture_layers(&drafter_cfg.target_layer_ids);
            let _ = &mut guard;
            let sink =
                crate::model::forward_cache_only(&model_cfg, &target, &tokens, &mut cache, 0);
            mlx_sys::eval(&[&sink]);
            let captured = crate::model::layer_capture::take().expect("all five layers captured");
            drop(guard);
            assert_eq!(captured.len(), drafter_cfg.target_layer_ids.len());
            let rows = tokens.len() as i32;
            for (index, layer_hidden) in captured.iter().enumerate() {
                let ours = astype(layer_hidden, MlxDtype::Float32, None);
                let theirs = slice(
                    &reference,
                    &[0, 0, index as i32 * width],
                    &[1, rows, (index as i32 + 1) * width],
                    &[1, 1, 1],
                    None,
                );
                let dot = sum_axis(&multiply(&ours, &theirs, None), 2, false, None);
                let norm_a = sum_axis(&multiply(&ours, &ours, None), 2, false, None);
                let norm_b = sum_axis(&multiply(&theirs, &theirs, None), 2, false, None);
                let inverse_norm = mlx_sys::power(
                    &multiply(&norm_a, &norm_b, None),
                    &scalar(-0.5, MlxDtype::Float32),
                    None,
                );
                let cosine = multiply(&dot, &inverse_norm, None);
                let cosine = mlx_sys::contiguous(&cosine, None);
                mlx_sys::eval(&[&cosine]);
                let values = cosine.data_f32();
                let minimum = values.iter().copied().fold(f32::INFINITY, f32::min) as f64;
                let mean =
                    values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
                eprintln!(
                    "{} layer {:2}: mean cosine {mean:.5}, min {minimum:.5} over {rows} tokens",
                    json_path.file_name().unwrap().to_string_lossy(),
                    drafter_cfg.target_layer_ids[index]
                );
                let weak: Vec<(usize, f32)> = values
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, value)| *value < 0.99)
                    .collect();
                if !weak.is_empty() {
                    eprintln!(
                        "   tokens below 0.99 cosine: {:?}",
                        &weak[..weak.len().min(12)]
                    );
                }
                worst_mean = worst_mean.min(mean);
            }
        }
        // A few positions of a long prompt drift further at depth (recurrent state
        // sums in a different order); the mean is the contract, the minimums are
        // printed for inspection.
        assert!(
            worst_mean > 0.98,
            "captured hidden states diverge from mlx-vlm: worst mean cosine {worst_mean}"
        );
    }

    /// Times the target LM head at 1..=8 rows with and without the fused
    /// verify routes, to see which rows each kernel actually serves.
    #[test]
    #[ignore = "needs AX_ENGINE_DFLASH2_TARGET_DIR"]
    fn lm_head_projection_probe() {
        let target_dir =
            env_dir("AX_ENGINE_DFLASH2_TARGET_DIR").expect("set AX_ENGINE_DFLASH2_TARGET_DIR");
        let artifacts = ax_engine_core::NativeModelArtifacts::from_dir(&target_dir).unwrap();
        let target = crate::weights::load_weights(&artifacts).unwrap();
        let head = &target.lm_head;
        let k = head.weight.shape()[1] * 32 / head.bits;
        eprintln!(
            "lm_head {:?} bits {} group {} scales {:?} biases {:?} x dtype for probe bf16",
            head.weight.shape(),
            head.bits,
            head.group_size,
            head.scales.as_ref().map(|s| (s.dtype(), s.shape())),
            head.biases.as_ref().map(|b| (b.dtype(), b.shape())),
        );
        let time = |rows: i32, armed: bool| -> f64 {
            let dtype = head
                .scales
                .as_ref()
                .map(|scales| scales.dtype())
                .filter(|dtype| matches!(dtype, MlxDtype::Float16 | MlxDtype::Bfloat16))
                .unwrap_or(MlxDtype::Bfloat16);
            let x = astype(
                &reshape(
                    &MlxArray::from_f32_slice(&vec![0.01_f32; (rows * k) as usize]),
                    &[1, rows, k],
                    None,
                ),
                dtype,
                None,
            );
            mlx_sys::eval(&[&x]);
            let mut samples = Vec::new();
            for _ in 0..7 {
                let _guard = armed.then(|| QwenMtpVerifyQmmGuard::arm(true));
                let started = std::time::Instant::now();
                let y = qw(&x, head);
                mlx_sys::eval(&[&y]);
                samples.push(started.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        for rows in 1..=8 {
            eprintln!(
                "rows {rows}: stock {:6.2} ms, armed {:6.2} ms",
                time(rows, false),
                time(rows, true)
            );
        }
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ax-dflash2-locate-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_drafter(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
    }

    #[test]
    fn artifact_identity_tracks_content_and_requires_both_pinned_digests() {
        let root = scratch_dir("identity");
        make_drafter(&root);
        let original = DrafterArtifact::read(&root).unwrap();
        assert!(original.admitted().is_none());
        assert!(original.identity().contains("@explicit-local;"));
        std::fs::write(root.join("model.safetensors"), b"changed").unwrap();
        let changed = DrafterArtifact::read(&root).unwrap();
        assert_eq!(original.config_sha256, changed.config_sha256);
        assert_ne!(original.weights_sha256, changed.weights_sha256);
        assert_ne!(original.identity(), changed.identity());
        for drafter in ADMITTED_DRAFTERS {
            let mut pinned = DrafterArtifact {
                config_sha256: drafter.config_sha256.into(),
                weights_sha256: drafter.weights_sha256.into(),
            };
            assert_eq!(pinned.admitted(), Some(drafter));
            assert!(
                pinned
                    .identity()
                    .starts_with(&format!("{}@{};", drafter.repo, drafter.revision))
            );
            pinned.config_sha256 = original.config_sha256.clone();
            assert!(pinned.admitted().is_none());
        }
        // One drafter's config with another's weights is not admitted.
        let crossed = DrafterArtifact {
            config_sha256: ADMITTED_DRAFTERS[0].config_sha256.into(),
            weights_sha256: ADMITTED_DRAFTERS[1].weights_sha256.into(),
        };
        assert!(crossed.admitted().is_none());
        std::fs::remove_file(root.join("model.safetensors")).unwrap();
        assert!(DrafterArtifact::read(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicit_directory_is_used_or_nothing() {
        let root = scratch_dir("explicit");
        let model = root.join("model");
        make_drafter(&model.join("dflash2"));
        let good = root.join("good");
        make_drafter(&good);
        let bad = root.join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        assert_eq!(locate_drafter(Some(&good), &model, None, &[]), Some(good));
        // A wrong explicit directory is not replaced by the bundled one.
        assert_eq!(locate_drafter(Some(&bad), &model, None, &[]), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn bundled_drafter_beats_the_hub_cache() {
        let root = scratch_dir("bundled");
        let model = root.join("model");
        make_drafter(&model.join("dflash2"));
        let hub = root.join("hub");
        let admitted = &ADMITTED_DRAFTERS[0];
        make_drafter(
            &hub.join(admitted.hf_entry())
                .join("snapshots")
                .join(admitted.revision),
        );
        assert_eq!(
            locate_drafter(None, &model, Some(admitted), std::slice::from_ref(&hub)),
            Some(model.join("dflash2"))
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hub_cache_snapshot_is_found_and_incomplete_ones_are_skipped() {
        let root = scratch_dir("hub");
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let hub = root.join("hub");
        for admitted in ADMITTED_DRAFTERS {
            let snapshots = hub.join(admitted.hf_entry()).join("snapshots");
            // A snapshot without weights is not a drafter.
            std::fs::create_dir_all(snapshots.join("partial")).unwrap();
            std::fs::write(snapshots.join("partial").join("config.json"), "{}").unwrap();
            let find =
                |roots: &[std::path::PathBuf]| locate_drafter(None, &model, Some(admitted), roots);
            assert_eq!(find(std::slice::from_ref(&hub)), None);
            make_drafter(&snapshots.join("unrelated-newer"));
            assert_eq!(find(std::slice::from_ref(&hub)), None);
            make_drafter(&snapshots.join(admitted.revision));
            assert_eq!(
                find(&[root.join("missing"), hub.clone()]),
                Some(snapshots.join(admitted.revision))
            );
        }
        // Without an admitted drafter for the target, the cache is not searched.
        assert_eq!(
            locate_drafter(None, &model, None, std::slice::from_ref(&hub)),
            None
        );
        // The 27B drafter is never offered to the 35B-A3B target, or the reverse.
        assert_eq!(
            locate_drafter(
                None,
                &model,
                Some(&ADMITTED_DRAFTERS[1]),
                std::slice::from_ref(&hub)
            ),
            Some(
                hub.join(ADMITTED_DRAFTERS[1].hf_entry())
                    .join("snapshots")
                    .join(ADMITTED_DRAFTERS[1].revision)
            )
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn only_the_trained_source_model_selects_a_drafter() {
        let root = scratch_dir("target");
        let write = |dir: &std::path::Path, name: &str, text: &str| {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join(name), text).unwrap();
        };
        let manifest = |id: &str| format!(r#"{{"source_model": {{"model_id": "{id}"}}}}"#);
        let repo = |dir: &std::path::Path| admitted_drafter_for(dir).map(|drafter| drafter.repo);
        let a = root.join("a");
        write(&a, "axquant_manifest.json", &manifest("Qwen/Qwen3.8-27B"));
        assert_eq!(repo(&a), Some("z-lab/Qwen3.8-27B-DFlash2"));
        let moe = root.join("moe");
        write(
            &moe,
            "axquant_manifest.json",
            &manifest("Qwen/Qwen3.6-35B-A3B"),
        );
        assert_eq!(repo(&moe), Some("incoai/Qwen3.6-35B-A3B-DFlash2"));
        // Same shapes, different weights: Qwen 3.6 27B and a 3.8 fine-tune.
        let b = root.join("b");
        write(&b, "axquant_manifest.json", &manifest("Qwen/Qwen3.6-27B"));
        assert_eq!(repo(&b), None);
        let c = root.join("c");
        write(
            &c,
            "axquant_manifest.json",
            &manifest("ukisai/Swift-1.5-Qwen3.8-27b"),
        );
        assert_eq!(repo(&c), None);
        // A manifest that names another source wins over a matching config.
        write(
            &b,
            "config.json",
            r#"{"_name_or_path": "Qwen/Qwen3.8-27B"}"#,
        );
        assert_eq!(repo(&b), None);
        // Without a manifest the config's name decides; without either, none.
        let d = root.join("d");
        write(
            &d,
            "config.json",
            r#"{"_name_or_path": "mlx-community/Qwen3.8-27B"}"#,
        );
        assert_eq!(repo(&d), Some("z-lab/Qwen3.8-27B-DFlash2"));
        let e = root.join("e");
        std::fs::create_dir_all(&e).unwrap();
        assert_eq!(repo(&e), None);
        // A fine-tune whose name only ends with the base name does not match.
        let f = root.join("f");
        write(
            &f,
            "config.json",
            r#"{"_name_or_path": "ukisai/Swift-1.5-Qwen3.8-27b"}"#,
        );
        assert_eq!(repo(&f), None);
        let g = root.join("g");
        write(
            &g,
            "config.json",
            r#"{"_name_or_path": "/models/Qwen3.6-35B-A3B/"}"#,
        );
        assert_eq!(repo(&g), Some("incoai/Qwen3.6-35B-A3B-DFlash2"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
