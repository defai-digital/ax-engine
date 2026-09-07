//! Gated-residual hyper-connections for Qwen4-Exp (`qwen4_exp`).
//!
//! The residual stream is packed as `[1, seq, hc_count * hidden]` holding
//! `hc_count` parallel streams of width `hidden` (4 × 2560 = 10240 in the
//! Qwen 3.8 Flash-Next pack). Each attention/FFN branch is bracketed by
//! [`mixed_input`] (collapse the streams into one branch input) and
//! [`inject_write_back`] (scatter the branch output into every stream under
//! a per-stream gate); a root-level [`mixer_output`] collapses the stream to
//! `hidden` before `lm_head` (there is no separate final RMSNorm — the
//! mixer's grouped norm is it). [`expand_embedding`] tiles the token
//! embedding into the packed stream at the input.
//!
//! This is NOT DeepSeek V4 mHC: no Sinkhorn, no softmax, no exp mixing.
//! Authoritative spec: `.internal/planning/qwen38-flash-next-support.md`
//! section "C. Hyper-connections" (HF `Qwen4ExpTextGatedResidual`). All
//! mixing math runs in f32; the branch input and the repacked stream are
//! cast back to the residual stream dtype (bf16).

use mlx_sys::{
    MlxArray, MlxDtype, add, astype, broadcast_to, divide, matmul, multiply, power, reshape,
    sigmoid, sum_axis, transpose,
};

/// Post-sigmoid multiplier of the injection gate (always 2.0 in the reference).
const INJECT_MULT_VALUE: f32 = 2.0;

/// Weights of one `Qwen4ExpTextGatedResidual` instance (checkpoint layout,
/// bf16 in the pack). Three instances per model: `attn_hyper_connection` and
/// `mlp_hyper_connection` per layer (both carry `block_inject_weight`), plus
/// the final `hyper_connection_mixer` (no `block_inject_weight`).
pub(crate) struct Qwen4ExpGatedResidualWeights {
    /// Grouped-RMSNorm scale (`hc_norm.weight`), `[hc*hidden]`, `1 + γ` form.
    pub hc_norm: MlxArray,
    /// Mix down-projection (`input_mix_weight_down.weight`), `[lowrank, hc*hidden]`.
    pub input_mix_weight_down: MlxArray,
    /// Mix up-projection (`input_mix_weight_up.weight`), `[hc*hidden, lowrank]`.
    pub input_mix_weight_up: MlxArray,
    /// Write-back gate (`block_inject_weight.weight`), `[hc, hc*hidden]`;
    /// `None` only on the final mixer instance.
    pub block_inject_weight: Option<MlxArray>,
}

/// Grouped RMSNorm over the packed stream: the last dim splits into
/// `hc_count` contiguous groups of `hidden_size`, each RMS-normalised
/// independently and scaled elementwise by `1 + weight`. Runs in f32 and
/// returns f32 regardless of the input dtype.
pub(crate) fn grouped_rms_norm(
    x: &MlxArray,
    weight: &MlxArray,
    hc_count: usize,
    hidden_size: usize,
    eps: f32,
) -> MlxArray {
    let shape = x.shape();
    assert_eq!(
        shape.len(),
        3,
        "qwen4_exp packed residual stream must be [batch, seq, hc*hidden]"
    );
    let (batch, seq, width) = (shape[0], shape[1], shape[2]);
    let (hc, hidden) = (hc_count as i32, hidden_size as i32);
    assert_eq!(
        width,
        hc * hidden,
        "qwen4_exp packed width must equal hc_count * hidden_size"
    );
    assert_eq!(
        weight.shape(),
        vec![width],
        "qwen4_exp hc_norm weight must cover the packed width"
    );

    let streams = reshape(
        &astype(x, MlxDtype::Float32, None),
        &[batch, seq, hc, hidden],
        None,
    );
    let sqrsum = sum_axis(&multiply(&streams, &streams, None), 3, true, None);
    let mean_sqr = divide(
        &sqrsum,
        &mlx_sys::ops::cached_scalar(hidden_size as f32, MlxDtype::Float32),
        None,
    );
    let rstd = power(
        &add(
            &mean_sqr,
            &mlx_sys::ops::cached_scalar(eps, MlxDtype::Float32),
            None,
        ),
        &mlx_sys::ops::cached_scalar(-0.5, MlxDtype::Float32),
        None,
    );
    let normed = multiply(&streams, &rstd, None);
    let scale = add(
        &reshape(
            &astype(weight, MlxDtype::Float32, None),
            &[hc, hidden],
            None,
        ),
        &mlx_sys::ops::cached_scalar(1.0, MlxDtype::Float32),
        None,
    );
    reshape(&multiply(&normed, &scale, None), &shape, None)
}

/// Collapse the packed stream into one branch input:
/// `mean_i(mix_i * normed_i)` with `normed` the grouped-normed stream and
/// `mix = sigmoid(up(silu(down(normed) / hc)))` reshaped to streams.
/// Returns `[batch, seq, hidden]` in the residual stream dtype.
pub(crate) fn mixed_input(
    x: &MlxArray,
    weights: &Qwen4ExpGatedResidualWeights,
    hc_count: usize,
    hidden_size: usize,
    eps: f32,
) -> MlxArray {
    let normed = grouped_rms_norm(x, &weights.hc_norm, hc_count, hidden_size, eps);
    let mix = mix_gates(&normed, weights, hc_count, hidden_size);
    let shape = normed.shape();
    let (batch, seq) = (shape[0], shape[1]);
    let (hc, hidden) = (hc_count as i32, hidden_size as i32);
    let weighted = multiply(
        &mix,
        &reshape(&normed, &[batch, seq, hc, hidden], None),
        None,
    );
    let summed = sum_axis(&weighted, 2, false, None);
    let meaned = divide(
        &summed,
        &mlx_sys::ops::cached_scalar(hc_count as f32, MlxDtype::Float32),
        None,
    );
    astype(&meaned, x.dtype(), None)
}

/// Scatter a branch output back into the packed stream:
/// `H + branch_out[..., None, :] * injection[..., None]` with
/// `injection = 2 * sigmoid(block_inject(normed) / hc)` gating each stream.
/// Returns the repacked stream in the residual stream dtype.
pub(crate) fn inject_write_back(
    x: &MlxArray,
    branch_out: &MlxArray,
    weights: &Qwen4ExpGatedResidualWeights,
    hc_count: usize,
    hidden_size: usize,
    eps: f32,
) -> MlxArray {
    let Some(inject_w) = weights.block_inject_weight.as_ref() else {
        unreachable!("qwen4_exp inject_write_back requires block_inject_weight");
    };
    let normed = grouped_rms_norm(x, &weights.hc_norm, hc_count, hidden_size, eps);
    let shape = normed.shape();
    let (batch, seq) = (shape[0], shape[1]);
    let (hc, hidden) = (hc_count as i32, hidden_size as i32);
    assert_eq!(
        inject_w.shape(),
        vec![hc, hc * hidden],
        "qwen4_exp block_inject weight must be [hc, hc*hidden]"
    );
    assert_eq!(
        branch_out.shape(),
        vec![batch, seq, hidden],
        "qwen4_exp branch output must be [batch, seq, hidden]"
    );

    let inject_t = transpose(&astype(inject_w, MlxDtype::Float32, None), &[1, 0], None);
    let logits = divide(
        &matmul(&normed, &inject_t, None),
        &mlx_sys::ops::cached_scalar(hc_count as f32, MlxDtype::Float32),
        None,
    );
    let injection = multiply(
        &sigmoid(&logits, None),
        &mlx_sys::ops::cached_scalar(INJECT_MULT_VALUE, MlxDtype::Float32),
        None,
    );

    let streams = reshape(
        &astype(x, MlxDtype::Float32, None),
        &[batch, seq, hc, hidden],
        None,
    );
    let branch = reshape(
        &astype(branch_out, MlxDtype::Float32, None),
        &[batch, seq, 1, hidden],
        None,
    );
    let gated = multiply(
        &branch,
        &reshape(&injection, &[batch, seq, hc, 1], None),
        None,
    );
    let out = astype(&add(&streams, &gated, None), x.dtype(), None);
    reshape(&out, &x.shape(), None)
}

/// Root-level collapse after the last decoder layer: identical to
/// [`mixed_input`]. The mixer instance carries no `block_inject_weight` —
/// it is simply unused here — and its grouped norm doubles as the model's
/// final norm, feeding `lm_head` directly.
pub(crate) fn mixer_output(
    x: &MlxArray,
    weights: &Qwen4ExpGatedResidualWeights,
    hc_count: usize,
    hidden_size: usize,
    eps: f32,
) -> MlxArray {
    mixed_input(x, weights, hc_count, hidden_size, eps)
}

/// Expand the embedded tokens `[batch, seq, hidden]` into the packed
/// residual stream `[batch, seq, hc_count * hidden]`: stream `i` is a full
/// copy of the embedding occupying dims `[i*hidden, (i+1)*hidden]`.
pub(crate) fn expand_embedding(emb: &MlxArray, hc_count: usize) -> MlxArray {
    let shape = emb.shape();
    assert_eq!(
        shape.len(),
        3,
        "qwen4_exp expand_embedding expects [batch, seq, hidden]"
    );
    let (batch, seq, hidden) = (shape[0], shape[1], shape[2]);
    let hc = hc_count as i32;
    let tiled = broadcast_to(
        &reshape(emb, &[batch, seq, 1, hidden], None),
        &[batch, seq, hc, hidden],
        None,
    );
    reshape(&tiled, &[batch, seq, hc * hidden], None)
}

/// Per-stream mix coefficients `sigmoid(up(silu(down(normed) / hc)))` as
/// `[batch, seq, hc, hidden]` f32. `normed` is the f32 grouped-normed
/// stream; both projections are bias-free `[out, in]` matmuls, and the
/// reference divides the low-rank activations by `hc_count` (4 in this pack).
fn mix_gates(
    normed: &MlxArray,
    weights: &Qwen4ExpGatedResidualWeights,
    hc_count: usize,
    hidden_size: usize,
) -> MlxArray {
    let shape = normed.shape();
    let (batch, seq, width) = (shape[0], shape[1], shape[2]);
    let down = &weights.input_mix_weight_down;
    let up = &weights.input_mix_weight_up;
    assert_eq!(
        down.shape()[1],
        width,
        "qwen4_exp input_mix_weight_down must contract the packed width"
    );
    assert_eq!(
        up.shape(),
        vec![width, down.shape()[0]],
        "qwen4_exp input_mix_weight_up must expand back to the packed width"
    );

    let down_t = transpose(&astype(down, MlxDtype::Float32, None), &[1, 0], None);
    let low = divide(
        &matmul(normed, &down_t, None),
        &mlx_sys::ops::cached_scalar(hc_count as f32, MlxDtype::Float32),
        None,
    );
    let act = mlx_sys::ops::silu(&low, None);
    let up_t = transpose(&astype(up, MlxDtype::Float32, None), &[1, 0], None);
    let logits = matmul(&act, &up_t, None);
    reshape(
        &sigmoid(&logits, None),
        &[batch, seq, hc_count as i32, hidden_size as i32],
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::eval;

    const HC: usize = 4;
    const HIDDEN: usize = 8;
    const WIDTH: usize = HC * HIDDEN;
    const LOWRANK: usize = 2;
    const EPS: f32 = 1e-6;

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

    fn scaled_fill(len: usize, seed: f32, scale: f32) -> Vec<f32> {
        fill(len, seed).iter().map(|v| v * scale).collect()
    }

    fn test_stream(seq: usize) -> MlxArray {
        array_f32(&fill(seq * WIDTH, 0.9), &[1, seq as i32, WIDTH as i32])
    }

    fn test_weights(with_inject: bool) -> Qwen4ExpGatedResidualWeights {
        Qwen4ExpGatedResidualWeights {
            hc_norm: array_f32(&scaled_fill(WIDTH, 0.11, 0.5), &[WIDTH as i32]),
            input_mix_weight_down: array_f32(
                &scaled_fill(LOWRANK * WIDTH, 0.13, 0.3),
                &[LOWRANK as i32, WIDTH as i32],
            ),
            input_mix_weight_up: array_f32(
                &scaled_fill(WIDTH * LOWRANK, 0.17, 0.3),
                &[WIDTH as i32, LOWRANK as i32],
            ),
            block_inject_weight: with_inject.then(|| {
                array_f32(
                    &scaled_fill(HC * WIDTH, 0.19, 0.3),
                    &[HC as i32, WIDTH as i32],
                )
            }),
        }
    }

    /// CPU reference for [`grouped_rms_norm`] on row-major `[rows, WIDTH]`.
    fn manual_grouped_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let rows = x.len() / WIDTH;
        let mut out = vec![0.0; x.len()];
        for r in 0..rows {
            for g in 0..HC {
                let base = r * WIDTH + g * HIDDEN;
                let group = &x[base..base + HIDDEN];
                let mean_sqr: f32 = group.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
                let rstd = (mean_sqr + eps).powf(-0.5);
                for (e, v) in group.iter().enumerate() {
                    out[base + e] = v * rstd * (1.0 + w[g * HIDDEN + e]);
                }
            }
        }
        out
    }

    /// Row-major `x [rows, in_dim] @ w [out_dim, in_dim]^T`.
    fn manual_matmul(x: &[f32], rows: usize, w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += x[r * in_dim + i] * w[o * in_dim + i];
                }
                out[r * out_dim + o] = acc;
            }
        }
        out
    }

    fn sigmoid_f32(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// CPU reference for [`mixed_input`] on row-major `[rows, WIDTH]`.
    fn manual_mixed(x: &[f32], w: &[f32], down: &[f32], up: &[f32], eps: f32) -> Vec<f32> {
        let normed = manual_grouped_norm(x, w, eps);
        let rows = x.len() / WIDTH;
        let low = manual_matmul(&normed, rows, down, LOWRANK, WIDTH);
        let act: Vec<f32> = low
            .iter()
            .map(|v| {
                let d = v / HC as f32;
                d * sigmoid_f32(d)
            })
            .collect();
        let logits = manual_matmul(&act, rows, up, WIDTH, LOWRANK);
        let mix: Vec<f32> = logits.iter().map(|v| sigmoid_f32(*v)).collect();
        let mut out = vec![0.0; rows * HIDDEN];
        for r in 0..rows {
            for e in 0..HIDDEN {
                let mut acc = 0.0f32;
                for g in 0..HC {
                    acc += mix[r * WIDTH + g * HIDDEN + e] * normed[r * WIDTH + g * HIDDEN + e];
                }
                out[r * HIDDEN + e] = acc / HC as f32;
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{what} mismatch at {i}: {a} vs {e} (tol {tol})"
            );
        }
    }

    #[test]
    fn grouped_norm_matches_manual_per_group_rms() {
        let seq = 3;
        let x = test_stream(seq);
        let w = array_f32(&scaled_fill(WIDTH, 0.21, 0.6), &[WIDTH as i32]);
        let out = grouped_rms_norm(&x, &w, HC, HIDDEN, EPS);
        assert_eq!(out.shape(), x.shape());
        assert_eq!(out.dtype(), MlxDtype::Float32);
        let expected =
            manual_grouped_norm(&fill(seq * WIDTH, 0.9), &scaled_fill(WIDTH, 0.21, 0.6), EPS);
        assert_close(&eval_f32(&out), &expected, 1e-5, "grouped_rms_norm");
    }

    #[test]
    fn mixed_input_zero_down_gives_half_stream_mean() {
        // down = 0 → silu(0) = 0 → up(0) = 0 → mix = sigmoid(0) = 0.5, so
        // mixed_input = 0.5 * mean_i(normed_i).
        let seq = 2;
        let x = test_stream(seq);
        let mut weights = test_weights(true);
        weights.input_mix_weight_down =
            array_f32(&[0.0; LOWRANK * WIDTH], &[LOWRANK as i32, WIDTH as i32]);
        let out = mixed_input(&x, &weights, HC, HIDDEN, EPS);
        assert_eq!(out.shape(), vec![1, seq as i32, HIDDEN as i32]);

        let normed =
            manual_grouped_norm(&fill(seq * WIDTH, 0.9), &scaled_fill(WIDTH, 0.11, 0.5), EPS);
        let mut expected = vec![0.0; seq * HIDDEN];
        for r in 0..seq {
            for e in 0..HIDDEN {
                let mean: f32 = (0..HC)
                    .map(|g| normed[r * WIDTH + g * HIDDEN + e])
                    .sum::<f32>()
                    / HC as f32;
                expected[r * HIDDEN + e] = 0.5 * mean;
            }
        }
        assert_close(&eval_f32(&out), &expected, 1e-5, "mixed_input zero-down");
    }

    #[test]
    fn mixed_input_matches_manual_pipeline() {
        let seq = 3;
        let x = test_stream(seq);
        let weights = test_weights(true);
        let out = mixed_input(&x, &weights, HC, HIDDEN, EPS);
        assert_eq!(out.dtype(), x.dtype());
        let expected = manual_mixed(
            &fill(seq * WIDTH, 0.9),
            &scaled_fill(WIDTH, 0.11, 0.5),
            &scaled_fill(LOWRANK * WIDTH, 0.13, 0.3),
            &scaled_fill(WIDTH * LOWRANK, 0.17, 0.3),
            EPS,
        );
        assert_close(&eval_f32(&out), &expected, 2e-5, "mixed_input full");
    }

    #[test]
    fn inject_write_back_zero_branch_returns_x() {
        let seq = 2;
        let x = test_stream(seq);
        let weights = test_weights(true);
        let zeros = array_f32(&vec![0.0; seq * HIDDEN], &[1, seq as i32, HIDDEN as i32]);
        let out = inject_write_back(&x, &zeros, &weights, HC, HIDDEN, EPS);
        assert_eq!(out.shape(), x.shape());
        assert_eq!(out.dtype(), x.dtype());
        assert_close(
            &eval_f32(&out),
            &fill(seq * WIDTH, 0.9),
            1e-6,
            "inject_write_back zero branch",
        );
    }

    #[test]
    fn inject_write_back_zero_gate_adds_branch_to_all_streams() {
        // block_inject = 0 → injection = 2 * sigmoid(0) = 1, so every stream
        // gains an unscaled copy of the branch output.
        let seq = 2;
        let x = test_stream(seq);
        let mut weights = test_weights(true);
        weights.block_inject_weight =
            Some(array_f32(&[0.0; HC * WIDTH], &[HC as i32, WIDTH as i32]));
        let branch = array_f32(
            &scaled_fill(seq * HIDDEN, 0.31, 1.0),
            &[1, seq as i32, HIDDEN as i32],
        );
        let out = inject_write_back(&x, &branch, &weights, HC, HIDDEN, EPS);

        let x_data = fill(seq * WIDTH, 0.9);
        let branch_data = scaled_fill(seq * HIDDEN, 0.31, 1.0);
        let mut expected = x_data.clone();
        for r in 0..seq {
            for g in 0..HC {
                for e in 0..HIDDEN {
                    expected[r * WIDTH + g * HIDDEN + e] += branch_data[r * HIDDEN + e];
                }
            }
        }
        assert_close(
            &eval_f32(&out),
            &expected,
            1e-5,
            "inject_write_back zero gate",
        );
    }

    #[test]
    fn inject_write_back_matches_manual_pipeline() {
        let seq = 3;
        let x = test_stream(seq);
        let weights = test_weights(true);
        let branch = array_f32(
            &scaled_fill(seq * HIDDEN, 0.23, 1.0),
            &[1, seq as i32, HIDDEN as i32],
        );
        let out = inject_write_back(&x, &branch, &weights, HC, HIDDEN, EPS);
        assert_eq!(out.shape(), x.shape());

        let normed =
            manual_grouped_norm(&fill(seq * WIDTH, 0.9), &scaled_fill(WIDTH, 0.11, 0.5), EPS);
        let logits = manual_matmul(&normed, seq, &scaled_fill(HC * WIDTH, 0.19, 0.3), HC, WIDTH);
        let injection: Vec<f32> = logits
            .iter()
            .map(|v| INJECT_MULT_VALUE * sigmoid_f32(v / HC as f32))
            .collect();
        let x_data = fill(seq * WIDTH, 0.9);
        let branch_data = scaled_fill(seq * HIDDEN, 0.23, 1.0);
        let mut expected = x_data.clone();
        for r in 0..seq {
            for g in 0..HC {
                for e in 0..HIDDEN {
                    expected[r * WIDTH + g * HIDDEN + e] +=
                        branch_data[r * HIDDEN + e] * injection[r * HC + g];
                }
            }
        }
        assert_close(&eval_f32(&out), &expected, 2e-5, "inject_write_back full");
    }

    #[test]
    #[should_panic(expected = "requires block_inject_weight")]
    fn inject_write_back_requires_block_inject() {
        let x = test_stream(1);
        let weights = test_weights(false);
        let branch = array_f32(&[0.0; HIDDEN], &[1, 1, HIDDEN as i32]);
        let _ = inject_write_back(&x, &branch, &weights, HC, HIDDEN, EPS);
    }

    #[test]
    fn mixer_output_ignores_block_inject() {
        let seq = 2;
        let x = test_stream(seq);
        let with_inject = test_weights(true);
        let without_inject = test_weights(false);
        let mixed = mixed_input(&x, &with_inject, HC, HIDDEN, EPS);
        let mixer_some = mixer_output(&x, &with_inject, HC, HIDDEN, EPS);
        let mixer_none = mixer_output(&x, &without_inject, HC, HIDDEN, EPS);
        assert_eq!(mixer_some.shape(), vec![1, seq as i32, HIDDEN as i32]);
        eval(&[&mixed, &mixer_some, &mixer_none]);
        assert_eq!(mixed.data_f32(), mixer_some.data_f32());
        assert_eq!(mixed.data_f32(), mixer_none.data_f32());
    }

    #[test]
    fn expand_embedding_tiles_streams() {
        let seq = 2;
        let emb_data: Vec<f32> = (0..seq * HIDDEN).map(|i| 0.01 * (i + 1) as f32).collect();
        let emb = array_f32(&emb_data, &[1, seq as i32, HIDDEN as i32]);
        let out = expand_embedding(&emb, HC);
        assert_eq!(out.shape(), vec![1, seq as i32, WIDTH as i32]);
        assert_eq!(out.dtype(), emb.dtype());
        let out_data = eval_f32(&out);
        for r in 0..seq {
            for g in 0..HC {
                for e in 0..HIDDEN {
                    assert_eq!(
                        out_data[r * WIDTH + g * HIDDEN + e],
                        emb_data[r * HIDDEN + e],
                        "stream {g} of row {r} must copy the embedding"
                    );
                }
            }
        }
    }

    #[test]
    fn expand_then_mixer_round_trip() {
        // Tiled embedding through a zero-down mixer: identical streams make
        // every group norm equal, so the mixer returns 0.5 * normed(emb) and
        // the shape collapses back to [1, seq, hidden].
        let seq = 2;
        let emb = array_f32(
            &scaled_fill(seq * HIDDEN, 0.7, 1.0),
            &[1, seq as i32, HIDDEN as i32],
        );
        let packed = expand_embedding(&emb, HC);
        let mut weights = test_weights(false);
        weights.input_mix_weight_down =
            array_f32(&[0.0; LOWRANK * WIDTH], &[LOWRANK as i32, WIDTH as i32]);
        let out = mixer_output(&packed, &weights, HC, HIDDEN, EPS);
        assert_eq!(out.shape(), emb.shape());

        let mut tiled = Vec::with_capacity(seq * WIDTH);
        for r in 0..seq {
            for _ in 0..HC {
                tiled.extend_from_slice(
                    &scaled_fill(seq * HIDDEN, 0.7, 1.0)[r * HIDDEN..(r + 1) * HIDDEN],
                );
            }
        }
        let normed = manual_grouped_norm(&tiled, &scaled_fill(WIDTH, 0.11, 0.5), EPS);
        let mut expected = vec![0.0; seq * HIDDEN];
        for r in 0..seq {
            for e in 0..HIDDEN {
                let mean: f32 = (0..HC)
                    .map(|g| normed[r * WIDTH + g * HIDDEN + e])
                    .sum::<f32>()
                    / HC as f32;
                expected[r * HIDDEN + e] = 0.5 * mean;
            }
        }
        assert_close(&eval_f32(&out), &expected, 1e-5, "expand+mixer round trip");
    }

    #[test]
    fn trunk_dtype_round_trips_through_bf16() {
        let seq = 2;
        let x_f32 = test_stream(seq);
        let x = astype(&x_f32, MlxDtype::Bfloat16, None);
        let weights = test_weights(true);
        let branch = astype(
            &array_f32(
                &scaled_fill(seq * HIDDEN, 0.29, 1.0),
                &[1, seq as i32, HIDDEN as i32],
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let mixed = mixed_input(&x, &weights, HC, HIDDEN, EPS);
        let written = inject_write_back(&x, &branch, &weights, HC, HIDDEN, EPS);
        assert_eq!(mixed.dtype(), MlxDtype::Bfloat16);
        assert_eq!(written.dtype(), MlxDtype::Bfloat16);
        assert_eq!(mixed.shape(), vec![1, seq as i32, HIDDEN as i32]);
        assert_eq!(written.shape(), x.shape());
        // bf16 inputs keep the f32 math within bf16 resolution of the f32 path.
        let mixed_f32 = mixed_input(&x_f32, &weights, HC, HIDDEN, EPS);
        let mixed_back = astype(&mixed, MlxDtype::Float32, None);
        eval(&[&mixed_back, &mixed_f32]);
        let a = mixed_back.data_f32();
        let b = mixed_f32.data_f32();
        for (i, (v, r)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (v - r).abs() < 0.02,
                "bf16 mixed_input drifted at {i}: {v} vs {r}"
            );
        }
    }
}
