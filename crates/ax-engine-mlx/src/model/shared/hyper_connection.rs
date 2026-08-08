//! Manifold-constrained hyper-connections (mHC) for DeepSeek V4 (Flash).
//!
//! The residual stream is packed as `[1, seq, hc_mult * hidden]` holding
//! `hc_mult` parallel streams of width `hidden`. Each attention/FFN branch is
//! bracketed by `hc_pre` (collapse the streams into one layer input, produce
//! the mixing coefficients) and `hc_post` (scatter the branch output back into
//! the streams). A root-level `hc_head` collapses the stream before the final
//! norm.
//!
//! Authoritative spec: vLLM `vllm/model_executor/kernels/mhc/torch.py`
//! (`mhc_pre_torch` / `mhc_post_torch`), cross-checked against llama.cpp
//! `src/models/deepseek4.cpp` (`build_hc_pre` / `build_hc_sinkhorn` /
//! `build_hc_post` / `build_hc_head`). All mixing math runs in f32; the layer
//! input and repacked stream are cast back to the residual dtype (bf16).

use mlx_sys::{
    MlxArray, MlxDtype, add, astype, divide, matmul, multiply, power, reshape, sigmoid, slice,
    slice_last_dim, softmax, sum_axis, transpose,
};

use super::super::config::DeepseekV4Config;

/// Post-mix multiplier (`hc_post_mult_value` in the reference; always 2.0).
const HC_POST_MULT_VALUE: f32 = 2.0;

/// Outputs of [`hc_pre`] consumed by [`hc_post`] after the branch runs.
pub(crate) struct HcPreOutput {
    /// `[1, seq, hc]` f32 — post-mix coefficients (`2 * sigmoid(...)`).
    pub post: MlxArray,
    /// `[1, seq, hc, hc]` f32 — Sinkhorn-normalised residual mixing matrix,
    /// indexed `[src, dst]` (out stream `dst` gathers `comb[src, dst] * stream[src]`).
    pub comb: MlxArray,
    /// `[1, seq, hidden]` — pre-mixed layer input in the residual stream dtype.
    pub layer_input: MlxArray,
}

/// Collapse the packed residual stream into one branch input.
///
/// `fn_weight` is `[2*hc + hc*hc, hc*hidden]`, `base` is `[2*hc + hc*hc]`,
/// `scale` is `[3]` (pre / post / comb scalars), matching the checkpoint
/// layout (`hc_attn_*` / `hc_ffn_*` tensors).
pub(crate) fn hc_pre(
    packed_stream: &MlxArray,
    fn_weight: &MlxArray,
    base: &MlxArray,
    scale: &MlxArray,
    cfg: &DeepseekV4Config,
    rms_eps: f32,
) -> HcPreOutput {
    let (post, comb, pre) = hc_mixes(packed_stream, fn_weight, base, scale, cfg, rms_eps);
    let layer_input = hc_pre_mix(packed_stream, &pre, cfg.hc_mult);
    HcPreOutput {
        post,
        comb,
        layer_input,
    }
}

/// Root-level hyper-connection head: pre-mix only, collapsing
/// `[1, seq, hc*hidden]` to `[1, seq, hidden]` before the final norm.
/// `fn_weight` is `[hc, hc*hidden]`, `base` is `[hc]`, `scale` is `[1]`.
pub(crate) fn hc_head(
    packed_stream: &MlxArray,
    fn_weight: &MlxArray,
    base: &MlxArray,
    scale: &MlxArray,
    cfg: &DeepseekV4Config,
    rms_eps: f32,
) -> MlxArray {
    let mixes = hc_rms_matmul(packed_stream, fn_weight, cfg.hc_mult, rms_eps);
    let hc = cfg.hc_mult as i32;
    assert_eq!(
        mixes.shape()[2],
        hc,
        "HC head fn weight must produce hc mixes per token"
    );
    let pre_logits = add(&multiply(&mixes, scale, None), base, None);
    let pre = add(
        &sigmoid(&pre_logits, None),
        &mlx_sys::ops::cached_scalar(cfg.hc_eps, MlxDtype::Float32),
        None,
    );
    hc_pre_mix(packed_stream, &pre, cfg.hc_mult)
}

/// Scatter a branch output back into the packed residual stream:
/// `out[dst] = x * post[dst] + Σ_src comb[src, dst] * stream[src]`.
pub(crate) fn hc_post(
    x: &MlxArray,
    packed_stream: &MlxArray,
    post: &MlxArray,
    comb: &MlxArray,
    cfg: &DeepseekV4Config,
) -> MlxArray {
    let shape = packed_stream.shape();
    assert_eq!(
        shape.len(),
        3,
        "HC packed residual stream must be [1, seq, hc*hidden]"
    );
    let seq = shape[1];
    let hc = cfg.hc_mult as i32;
    let hidden = shape[2] / hc;
    assert_eq!(
        shape[2],
        hidden * hc,
        "HC packed width must divide evenly into hc streams"
    );

    // einsum("...ij,...ih->...jh", comb, stream): transpose comb to
    // [dst, src] so a batched matmul gathers the source streams.
    let stream = reshape(
        &astype(packed_stream, MlxDtype::Float32, None),
        &[1, seq, hc, hidden],
        None,
    );
    let comb_dst_src = transpose(comb, &[0, 1, 3, 2], None);
    let mixed_residual = matmul(&comb_dst_src, &stream, None);

    let post_term = multiply(
        &reshape(post, &[1, seq, hc, 1], None),
        &reshape(
            &astype(x, MlxDtype::Float32, None),
            &[1, seq, 1, hidden],
            None,
        ),
        None,
    );
    let out = add(&mixed_residual, &post_term, None);
    let out = astype(&out, packed_stream.dtype(), None);
    reshape(&out, &shape, None)
}

/// Shared mixing pipeline of `hc_pre`: RMS-scaled matmul → sigmoid pre/post
/// gates → softmax + Sinkhorn comb matrix. Returns (post, comb, pre), all f32.
fn hc_mixes(
    packed_stream: &MlxArray,
    fn_weight: &MlxArray,
    base: &MlxArray,
    scale: &MlxArray,
    cfg: &DeepseekV4Config,
    rms_eps: f32,
) -> (MlxArray, MlxArray, MlxArray) {
    let hc = cfg.hc_mult;
    let hc_i = hc as i32;
    let mixes = hc_rms_matmul(packed_stream, fn_weight, hc, rms_eps);
    let eps = mlx_sys::ops::cached_scalar(cfg.hc_eps, MlxDtype::Float32);

    // scale / base slices ([1] / [hc] / [hc*hc] broadcast against [1, seq, ...]).
    let scale_pre = slice(scale, &[0], &[1], &[1], None);
    let scale_post = slice(scale, &[1], &[2], &[1], None);
    let scale_comb = slice(scale, &[2], &[3], &[1], None);
    let base_pre = slice(base, &[0], &[hc_i], &[1], None);
    let base_post = slice(base, &[hc_i], &[2 * hc_i], &[1], None);
    let base_comb = slice(base, &[2 * hc_i], &[(2 + hc) as i32 * hc_i], &[1], None);

    let pre_logits = add(
        &multiply(&slice_last_dim(&mixes, 0, hc_i, None), &scale_pre, None),
        &base_pre,
        None,
    );
    let pre = add(&sigmoid(&pre_logits, None), &eps, None);

    let post_logits = add(
        &multiply(
            &slice_last_dim(&mixes, hc_i, 2 * hc_i, None),
            &scale_post,
            None,
        ),
        &base_post,
        None,
    );
    let post = multiply(
        &sigmoid(&post_logits, None),
        &mlx_sys::ops::cached_scalar(HC_POST_MULT_VALUE, MlxDtype::Float32),
        None,
    );

    let seq = packed_stream.shape()[1];
    let comb_logits = reshape(
        &slice_last_dim(&mixes, 2 * hc_i, (2 + hc) as i32 * hc_i, None),
        &[1, seq, hc_i, hc_i],
        None,
    );
    let comb_logits = add(
        &multiply(&comb_logits, &scale_comb, None),
        &reshape(&base_comb, &[hc_i, hc_i], None),
        None,
    );
    let comb = hc_sinkhorn(&comb_logits, cfg);
    (post, comb, pre)
}

/// `rms_norm(packed_stream)` contracted with `fn_weight`: the reference forms
/// the matmul on the unnormalised f32 stream and scales the result by
/// `rsqrt(mean(square) + eps)` (`mhc_pre_torch`), which equals a weight-free
/// RMSNorm before the matmul. Returns `[1, seq, fn_weight.shape[0]]` f32.
fn hc_rms_matmul(
    packed_stream: &MlxArray,
    fn_weight: &MlxArray,
    hc_mult: usize,
    rms_eps: f32,
) -> MlxArray {
    let shape = packed_stream.shape();
    assert_eq!(
        shape.len(),
        3,
        "HC packed residual stream must be [1, seq, hc*hidden]"
    );
    let hc_hidden = shape[2];
    assert_eq!(
        hc_hidden % hc_mult as i32,
        0,
        "HC packed width must divide evenly into hc streams"
    );
    let x = astype(packed_stream, MlxDtype::Float32, None);
    let fn_t = transpose(&astype(fn_weight, MlxDtype::Float32, None), &[1, 0], None);
    let mixes = matmul(&x, &fn_t, None);
    let sqrsum = sum_axis(&multiply(&x, &x, None), 2, true, None);
    let inv_width = mlx_sys::ops::cached_scalar(1.0 / hc_hidden as f32, MlxDtype::Float32);
    let mean_sqr = multiply(&sqrsum, &inv_width, None);
    let rsqrt = power(
        &add(
            &mean_sqr,
            &mlx_sys::ops::cached_scalar(rms_eps, MlxDtype::Float32),
            None,
        ),
        &mlx_sys::ops::cached_scalar(-0.5, MlxDtype::Float32),
        None,
    );
    multiply(&mixes, &rsqrt, None)
}

/// Sinkhorn normalisation of the comb matrix (`[1, seq, src, dst]` f32):
/// softmax over `dst`, one `src`-normalisation, then `(iters - 1)` rounds of
/// `dst`- then `src`-normalisation. Every normalisation divides by
/// `sum + hc_eps`, matching `mhc_pre_torch` and `build_hc_sinkhorn`.
fn hc_sinkhorn(comb_logits: &MlxArray, cfg: &DeepseekV4Config) -> MlxArray {
    let eps = mlx_sys::ops::cached_scalar(cfg.hc_eps, MlxDtype::Float32);
    let mut comb = add(&softmax(comb_logits, -1, None), &eps, None);
    // Normalise over src (dim -2) / dst (dim -1); axes are fixed at 2 and 3
    // because comb is [1, seq, hc, hc].
    let norm_src = |m: &MlxArray| divide(m, &add(&sum_axis(m, 2, true, None), &eps, None), None);
    let norm_dst = |m: &MlxArray| divide(m, &add(&sum_axis(m, 3, true, None), &eps, None), None);
    comb = norm_src(&comb);
    for _ in 1..cfg.hc_sinkhorn_iters {
        comb = norm_dst(&comb);
        comb = norm_src(&comb);
    }
    comb
}

/// `Σ_h pre[h] ⊙ stream_h` — combine the `hc` streams into one
/// `[1, seq, hidden]` input in the residual stream dtype.
fn hc_pre_mix(packed_stream: &MlxArray, pre: &MlxArray, hc_mult: usize) -> MlxArray {
    let shape = packed_stream.shape();
    let seq = shape[1];
    let hc = hc_mult as i32;
    let hidden = shape[2] / hc;
    let stream = reshape(
        &astype(packed_stream, MlxDtype::Float32, None),
        &[1, seq, hc, hidden],
        None,
    );
    let weighted = multiply(&reshape(pre, &[1, seq, hc, 1], None), &stream, None);
    let mixed = sum_axis(&weighted, 2, false, None);
    astype(&mixed, packed_stream.dtype(), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::eval;

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn eval_f32(array: &MlxArray) -> Vec<f32> {
        eval(&[array]);
        array.data_f32().to_vec()
    }

    /// Deterministic pseudo-random fill (no external deps).
    fn fill(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5)
            .collect()
    }

    const HC: usize = 4;
    const HIDDEN: usize = 8;

    fn hc_weights(seq: usize) -> (MlxArray, MlxArray, MlxArray, MlxArray) {
        let mixes = 2 * HC + HC * HC;
        let stream = array_f32(
            &fill(seq * HC * HIDDEN, 0.7),
            &[1, seq as i32, (HC * HIDDEN) as i32],
        );
        let fn_weight = array_f32(
            &fill(mixes * HC * HIDDEN, 0.3),
            &[mixes as i32, (HC * HIDDEN) as i32],
        );
        let base = array_f32(&fill(mixes, 0.5), &[mixes as i32]);
        let scale = array_f32(&[1.0, 1.0, 1.0], &[3]);
        (stream, fn_weight, base, scale)
    }

    fn hc_pre_impl(
        stream: &MlxArray,
        fn_weight: &MlxArray,
        base: &MlxArray,
        scale: &MlxArray,
        sinkhorn_iters: usize,
    ) -> HcPreOutput {
        let cfg = test_config(1e-5, sinkhorn_iters);
        hc_pre(stream, fn_weight, base, scale, &cfg, 1e-6)
    }

    fn test_config(hc_eps: f32, sinkhorn_iters: usize) -> DeepseekV4Config {
        DeepseekV4Config {
            head_dim: 512,
            qk_rope_head_dim: 64,
            q_lora_rank: 1536,
            o_lora_rank: 512,
            o_groups: 4,
            index_topk: 2048,
            index_n_heads: 64,
            index_head_dim: 128,
            compress_rope_theta: 50000.0,
            compress_rope_scaling: None,
            has_attn_sinks: true,
            compress_ratios: Vec::new(),
            hc_mult: HC,
            hc_sinkhorn_iters: sinkhorn_iters,
            hc_eps,
            num_hash_layers: 0,
            num_nextn_predict_layers: 0,
            scoring_func: None,
            swiglu_limit: 7.0,
        }
    }

    #[test]
    fn hc_pre_output_shapes_match_reference() {
        let seq = 3;
        let (stream, fn_weight, base, scale) = hc_weights(seq);
        let out = hc_pre_impl(&stream, &fn_weight, &base, &scale, 5);
        assert_eq!(out.post.shape(), vec![1, seq as i32, HC as i32]);
        assert_eq!(out.comb.shape(), vec![1, seq as i32, HC as i32, HC as i32]);
        assert_eq!(out.layer_input.shape(), vec![1, seq as i32, HIDDEN as i32]);
        assert_eq!(out.post.dtype(), MlxDtype::Float32);
        assert_eq!(out.comb.dtype(), MlxDtype::Float32);
        assert_eq!(out.layer_input.dtype(), stream.dtype());
    }

    #[test]
    fn hc_sinkhorn_comb_is_doubly_stochastic() {
        let seq = 2;
        let (stream, fn_weight, base, scale) = hc_weights(seq);
        let out = hc_pre_impl(&stream, &fn_weight, &base, &scale, 8);
        let comb = eval_f32(&out.comb);
        for t in 0..seq {
            let tok = &comb[t * HC * HC..(t + 1) * HC * HC];
            for dst in 0..HC {
                let col_sum: f32 = (0..HC).map(|src| tok[src * HC + dst]).sum();
                assert!(
                    (col_sum - 1.0).abs() < 1e-2,
                    "comb column (dst) sum {col_sum} should be ~1"
                );
            }
            for src in 0..HC {
                let row_sum: f32 = tok[src * HC..(src + 1) * HC].iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-2,
                    "comb row (src) sum {row_sum} should be ~1"
                );
            }
        }
    }

    #[test]
    fn hc_pre_zero_weights_give_uniform_mix() {
        // fn = 0, base = 0 → all logits 0: pre = 0.5 + eps, post = 2*0.5 = 1,
        // comb = uniform 1/hc (fixed point of Sinkhorn).
        let seq = 2;
        let mixes = 2 * HC + HC * HC;
        let stream = array_f32(
            &fill(seq * HC * HIDDEN, 0.9),
            &[1, seq as i32, (HC * HIDDEN) as i32],
        );
        let fn_weight = array_f32(
            &vec![0.0; mixes * HC * HIDDEN],
            &[mixes as i32, (HC * HIDDEN) as i32],
        );
        let base = array_f32(&vec![0.0; mixes], &[mixes as i32]);
        let scale = array_f32(&[1.0, 1.0, 1.0], &[3]);
        let hc_eps = 1e-5;
        let cfg = test_config(hc_eps, 3);
        let out = hc_pre(&stream, &fn_weight, &base, &scale, &cfg, 1e-6);

        let post = eval_f32(&out.post);
        assert!(post.iter().all(|v| (*v - 1.0).abs() < 1e-5));
        let comb = eval_f32(&out.comb);
        assert!(comb.iter().all(|v| (*v - 0.25).abs() < 1e-4));

        // layer_input = Σ_h (0.5 + eps) * stream_h.
        let stream_data = eval_f32(&stream);
        let layer_input = eval_f32(&out.layer_input);
        for t in 0..seq {
            for e in 0..HIDDEN {
                let expect: f32 = (0..HC)
                    .map(|h| (0.5 + hc_eps) * stream_data[(t * HC + h) * HIDDEN + e])
                    .sum();
                assert!(
                    (layer_input[t * HIDDEN + e] - expect).abs() < 1e-3,
                    "layer_input mismatch at token {t} elem {e}"
                );
            }
        }
    }

    #[test]
    fn hc_post_round_trip_shape_and_uniform_value() {
        // With comb = uniform 1/hc, post = 1 and x = 0, the repacked stream is
        // the per-token mean of the source streams in every slot.
        let seq = 2;
        let mixes = 2 * HC + HC * HC;
        let stream = array_f32(
            &fill(seq * HC * HIDDEN, 0.4),
            &[1, seq as i32, (HC * HIDDEN) as i32],
        );
        let fn_weight = array_f32(
            &vec![0.0; mixes * HC * HIDDEN],
            &[mixes as i32, (HC * HIDDEN) as i32],
        );
        let base = array_f32(&vec![0.0; mixes], &[mixes as i32]);
        let scale = array_f32(&[1.0, 1.0, 1.0], &[3]);
        let cfg = test_config(1e-5, 3);
        let pre = hc_pre(&stream, &fn_weight, &base, &scale, &cfg, 1e-6);
        let zeros_x = array_f32(&vec![0.0; seq * HIDDEN], &[1, seq as i32, HIDDEN as i32]);

        let out = hc_post(&zeros_x, &stream, &pre.post, &pre.comb, &cfg);
        assert_eq!(out.shape(), stream.shape());
        assert_eq!(out.dtype(), stream.dtype());

        let stream_data = eval_f32(&stream);
        let out_data = eval_f32(&out);
        for t in 0..seq {
            for dst in 0..HC {
                for e in 0..HIDDEN {
                    let expect: f32 = (0..HC)
                        .map(|src| 0.25 * stream_data[(t * HC + src) * HIDDEN + e])
                        .sum();
                    let actual = out_data[(t * HC + dst) * HIDDEN + e];
                    assert!(
                        (actual - expect).abs() < 1e-3,
                        "hc_post mismatch at token {t} dst {dst} elem {e}: {actual} vs {expect}"
                    );
                }
            }
        }
    }

    #[test]
    fn hc_head_collapses_to_hidden() {
        let seq = 2;
        let stream = array_f32(
            &fill(seq * HC * HIDDEN, 0.6),
            &[1, seq as i32, (HC * HIDDEN) as i32],
        );
        let fn_weight = array_f32(
            &vec![0.0; HC * HC * HIDDEN],
            &[HC as i32, (HC * HIDDEN) as i32],
        );
        let base = array_f32(&[0.0; HC], &[HC as i32]);
        let scale = array_f32(&[1.0], &[1]);
        let hc_eps = 1e-5;
        let cfg = test_config(hc_eps, 3);
        let out = hc_head(&stream, &fn_weight, &base, &scale, &cfg, 1e-6);
        assert_eq!(out.shape(), vec![1, seq as i32, HIDDEN as i32]);

        // pre = sigmoid(0) + eps → out = Σ_h (0.5 + eps) * stream_h.
        let stream_data = eval_f32(&stream);
        let out_data = eval_f32(&out);
        for t in 0..seq {
            for e in 0..HIDDEN {
                let expect: f32 = (0..HC)
                    .map(|h| (0.5 + hc_eps) * stream_data[(t * HC + h) * HIDDEN + e])
                    .sum();
                assert!((out_data[t * HIDDEN + e] - expect).abs() < 1e-3);
            }
        }
    }
}
