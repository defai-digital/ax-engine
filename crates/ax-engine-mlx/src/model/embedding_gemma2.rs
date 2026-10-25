//! Cache-free, text-only EmbeddingGemma 2 encoder.
//!
//! Unlike 300M, this encoder has projection-only per-layer inputs, local
//! bidirectional windows, scale-one attention and a token-wise output head.

use super::shared::utils::qw;
use super::{ModelConfig, build_bidirectional_padding_mask, build_embedding_batch_hidden};
use crate::weights::ModelWeights;
use mlx_sys::where_cond;
use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, MlxVectorArray, ScaledDotProductAttentionMask, add, arange,
    astype, concatenate, cos, divide, gelu_approx_mul, multiply, negative, power, reshape,
    rms_norm, scaled_dot_product_attention_with_mask, sin, slice, sum_axis, transpose,
};
use std::sync::{Arc, LazyLock};

fn array(data: &[f32], shape: &[i32]) -> MlxArray {
    reshape(&MlxArray::from_f32_slice(data), shape, None)
}

fn scalar(value: f32, dtype: MlxDtype) -> MlxArray {
    mlx_sys::ops::cached_scalar(value, dtype)
}

static PACKED_PROJECTIONS: LazyLock<bool> =
    LazyLock::new(|| crate::fastpath::env_flag_default_on("AX_MLX_EMBED_GEMMA2_PACKED"));

/// The encoder rounds the rotary cos/sin tables to the activation dtype
/// before multiplication. MLX fast.rope rounds only the final result.
fn rotary_tables(seq: i32, dim: i32, theta: f32, dtype: MlxDtype) -> (MlxArray, MlxArray) {
    let exponents = divide(
        &arange(0.0, dim as f64, 2.0, MlxDtype::Float32, None),
        &scalar(dim as f32, MlxDtype::Float32),
        None,
    );
    let denominators = power(&scalar(theta, MlxDtype::Float32), &exponents, None);
    let positions = reshape(
        &arange(0.0, seq as f64, 1.0, MlxDtype::Float32, None),
        &[seq, 1],
        None,
    );
    let angles = divide(&positions, &denominators, None);
    let angles = concatenate(&[&angles, &angles], -1, None);
    (
        astype(&cos(&angles, None), dtype, None),
        astype(&sin(&angles, None), dtype, None),
    )
}

fn rotate(x: &MlxArray, tables: &(MlxArray, MlxArray)) -> MlxArray {
    let shape = x.shape();
    let half = shape[3] / 2;
    let left = slice(
        x,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], shape[2], half],
        &[1; 4],
        None,
    );
    let right = slice(x, &[0, 0, 0, half], &shape, &[1; 4], None);
    let swapped = concatenate(&[&negative(&right, None), &left], -1, None);
    add(
        &multiply(x, &tables.0, None),
        &multiply(&swapped, &tables.1, None),
        None,
    )
}

struct Window {
    query_start: i32,
    query_end: i32,
    key_start: i32,
    key_end: i32,
    mask: MlxArray,
}

struct Context {
    padding: MlxArray,
    windows: Vec<Window>,
    local_tables: (MlxArray, MlxArray),
    full_tables: (MlxArray, MlxArray),
    pool_mask: MlxArray,
    lengths: MlxArray,
    has_padding: bool,
}

impl Context {
    fn new(cfg: &ModelConfig, hidden: &MlxArray, lengths: &[usize]) -> Self {
        let b = hidden.shape()[0];
        let seq = hidden.shape()[1];
        let dtype = hidden.dtype();
        let padding = build_bidirectional_padding_mask(b as usize, seq as usize, lengths, dtype)
            .expect("bidirectional padding mask");
        let sliding = cfg
            .layer_configs
            .iter()
            .find(|l| l.sliding_window.is_some());
        let window = sliding
            .and_then(|l| l.sliding_window)
            .unwrap_or(seq as usize);
        let mut windows = Vec::new();
        // Bound local mask storage and attention work to the receptive field.
        // Short inputs use the full mask, since every token lies in the window.
        if seq as usize > window + 1 {
            for start in (0..seq).step_by(256) {
                let end = (start + 256).min(seq);
                let key_start = (start - window as i32).max(0);
                let key_end = (end + window as i32).min(seq);
                let values: Vec<f32> = (start..end)
                    .flat_map(|q| {
                        (key_start..key_end).map(move |k| {
                            if q.abs_diff(k) as usize <= window {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect();
                let band = astype(
                    &array(&values, &[1, 1, end - start, key_end - key_start]),
                    dtype,
                    None,
                );
                let pad = slice(
                    &padding,
                    &[0, 0, 0, key_start],
                    &[b, 1, 1, key_end],
                    &[1; 4],
                    None,
                );
                windows.push(Window {
                    query_start: start,
                    query_end: end,
                    key_start,
                    key_end,
                    mask: add(&band, &pad, None),
                });
            }
        }
        let full = cfg
            .layer_configs
            .iter()
            .find(|l| l.sliding_window.is_none())
            .expect("full encoder layer");
        let local_theta = sliding.map(|l| l.rope_theta).unwrap_or(full.rope_theta);
        let pool_mask: Vec<f32> = lengths
            .iter()
            .flat_map(|&len| (0..seq as usize).map(move |i| if i < len { 1.0 } else { 0.0 }))
            .collect();
        Self {
            padding,
            windows,
            local_tables: rotary_tables(seq, cfg.head_dim as i32, local_theta, dtype),
            full_tables: rotary_tables(seq, full.head_dim as i32, full.rope_theta, dtype),
            pool_mask: astype(&array(&pool_mask, &[b, seq, 1]), MlxDtype::Bool, None),
            lengths: array(
                &lengths.iter().map(|&l| l as f32).collect::<Vec<_>>(),
                &[b, 1],
            ),
            has_padding: lengths.iter().any(|&l| l != seq as usize),
        }
    }

    fn attention(&self, q: &MlxArray, k: &MlxArray, v: &MlxArray, sliding: bool) -> MlxArray {
        if !sliding || self.windows.is_empty() {
            return scaled_dot_product_attention_with_mask(
                q,
                k,
                v,
                1.0,
                ScaledDotProductAttentionMask::Array(&self.padding),
                None,
            );
        }
        let qs = q.shape();
        let ks = k.shape();
        let mut blocks = Vec::with_capacity(self.windows.len());
        for window in &self.windows {
            let q = slice(
                q,
                &[0, 0, window.query_start, 0],
                &[qs[0], qs[1], window.query_end, qs[3]],
                &[1; 4],
                None,
            );
            let k = slice(
                k,
                &[0, 0, window.key_start, 0],
                &[ks[0], ks[1], window.key_end, ks[3]],
                &[1; 4],
                None,
            );
            let v = slice(
                v,
                &[0, 0, window.key_start, 0],
                &[ks[0], ks[1], window.key_end, ks[3]],
                &[1; 4],
                None,
            );
            blocks.push(scaled_dot_product_attention_with_mask(
                &q,
                &k,
                &v,
                1.0,
                ScaledDotProductAttentionMask::Array(&window.mask),
                None,
            ));
        }
        concatenate(&blocks.iter().collect::<Vec<_>>(), 2, None)
    }
}

pub(crate) fn forward(cfg: &ModelConfig, weights: &ModelWeights, batch: &[Vec<u32>]) -> MlxArray {
    let lengths: Vec<usize> = batch.iter().map(Vec::len).collect();
    let seq = lengths.iter().copied().max().unwrap_or(0);
    let hidden = build_embedding_batch_hidden(cfg, weights, batch, batch.len(), seq);
    let context = Context::new(cfg, &hidden, &lengths);
    body(cfg, weights, hidden, &context)
}

pub(crate) fn compile(
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    hidden: &MlxArray,
    lengths: &[usize],
) -> Result<MlxClosure, String> {
    let context = Context::new(&cfg, hidden, lengths);
    let closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        vec![body(&cfg, &weights, inputs.get(0), &context)]
    });
    // MX formats require concrete shapes; never use shapeless compilation.
    closure.compile(false)
}

fn body(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    mut hidden: MlxArray,
    context: &Context,
) -> MlxArray {
    let b = hidden.shape()[0];
    let seq = hidden.shape()[1];
    // Packing saves dispatches on small matrices; M5 Max measurements show
    // split projections win for the larger ingest shapes.
    let packed_projections = *PACKED_PROJECTIONS && i64::from(b) * i64::from(seq) <= 256;
    let per_layer = weights.per_layer_model_proj.as_ref().map(|projection| {
        let projected = qw(&hidden, projection);
        let projected = multiply(
            &projected,
            &scalar((cfg.hidden_size as f32).sqrt().recip(), projected.dtype()),
            None,
        );
        let projected = reshape(
            &projected,
            &[
                b,
                seq,
                cfg.layer_count as i32,
                cfg.hidden_size_per_layer_input as i32,
            ],
            None,
        );
        rms_norm(
            &projected,
            weights.per_layer_proj_norm.as_ref(),
            cfg.rms_norm_eps,
            None,
        )
    });
    for (index, layer) in weights.layers.iter().enumerate() {
        let geometry = &cfg.layer_configs[index];
        let dim = geometry.head_dim as i32;
        let normed = rms_norm(&hidden, Some(&layer.attn_norm), cfg.rms_norm_eps, None);
        // Keep the projections mode-aware: MXFP8 has scales without affine biases.
        let (q, k, v) = if packed_projections {
            let (q, k, v, _) = super::shared::qkv_project_embed(cfg, layer, &normed, dim as usize);
            (q, k, v)
        } else {
            (
                qw(&normed, layer.q_proj.as_ref().expect("encoder query")),
                qw(&normed, layer.k_proj.as_ref().expect("encoder key")),
                qw(&normed, layer.v_proj.as_ref().expect("encoder value")),
            )
        };
        let kv = k.shape()[2] / dim;
        let q = rms_norm(
            &reshape(&q, &[b, seq, cfg.n_heads as i32, dim], None),
            layer.q_norm.as_ref(),
            cfg.rms_norm_eps,
            None,
        );
        let k = rms_norm(
            &reshape(&k, &[b, seq, kv, dim], None),
            layer.k_norm.as_ref(),
            cfg.rms_norm_eps,
            None,
        );
        let v = rms_norm(
            &reshape(&v, &[b, seq, kv, dim], None),
            None,
            cfg.rms_norm_eps,
            None,
        );
        let tables = if geometry.sliding_window.is_some() {
            &context.local_tables
        } else {
            &context.full_tables
        };
        let q = rotate(&transpose(&q, &[0, 2, 1, 3], None), tables);
        let k = rotate(&transpose(&k, &[0, 2, 1, 3], None), tables);
        let v = transpose(&v, &[0, 2, 1, 3], None);
        let attention = context.attention(&q, &k, &v, geometry.sliding_window.is_some());
        let attention = reshape(
            &transpose(&attention, &[0, 2, 1, 3], None),
            &[b, seq, cfg.n_heads as i32 * dim],
            None,
        );
        // Fully masked padded query rows may be NaN. Remove them before
        // they can enter a later full-attention layer; multiplication by zero
        // would retain NaNs and contaminate the pooled result.
        let attention = if context.has_padding {
            where_cond(
                &context.pool_mask,
                &attention,
                &scalar(0.0, attention.dtype()),
                None,
            )
        } else {
            attention
        };
        let attention = qw(&attention, layer.o_proj.as_ref().expect("encoder output"));
        hidden = add(
            &hidden,
            &rms_norm(
                &attention,
                layer.attn_post_norm.as_ref(),
                cfg.rms_norm_eps,
                None,
            ),
            None,
        );
        let normed = rms_norm(&hidden, Some(&layer.ffn_norm), cfg.rms_norm_eps, None);
        let (gate, up) = if packed_projections && let Some(packed) = &layer.gate_up_packed {
            let projected = qw(&normed, packed);
            let width = projected.shape()[2] / 2;
            (
                mlx_sys::slice_last_dim(&projected, 0, width, None),
                mlx_sys::slice_last_dim(&projected, width, 2 * width, None),
            )
        } else {
            (
                qw(&normed, layer.gate_proj.as_ref().expect("encoder gate")),
                qw(&normed, layer.up_proj.as_ref().expect("encoder up")),
            )
        };
        let ffn = qw(
            &gelu_approx_mul(&gate, &up, None),
            layer.down_proj.as_ref().expect("encoder down"),
        );
        hidden = add(
            &hidden,
            &rms_norm(&ffn, layer.ffn_post_norm.as_ref(), cfg.rms_norm_eps, None),
            None,
        );
        if let Some(inputs) = &per_layer {
            let input = slice(
                inputs,
                &[0, 0, index as i32, 0],
                &[
                    b,
                    seq,
                    index as i32 + 1,
                    cfg.hidden_size_per_layer_input as i32,
                ],
                &[1; 4],
                None,
            );
            let input = reshape(
                &input,
                &[b, seq, cfg.hidden_size_per_layer_input as i32],
                None,
            );
            let gate = qw(
                &hidden,
                layer.per_layer_gate.as_ref().expect("encoder PLE gate"),
            );
            let projected = qw(
                &gelu_approx_mul(&gate, &input, None),
                layer
                    .per_layer_proj_w
                    .as_ref()
                    .expect("encoder PLE projection"),
            );
            hidden = add(
                &hidden,
                &rms_norm(
                    &projected,
                    layer.per_layer_post_norm.as_ref(),
                    cfg.rms_norm_eps,
                    None,
                ),
                None,
            );
        }
        if let Some(scale) = &layer.layer_scalar {
            hidden = multiply(&hidden, scale, None);
        }
    }
    let hidden = rms_norm(&hidden, Some(weights.final_norm()), cfg.rms_norm_eps, None);
    let projected = qw(
        &hidden,
        weights
            .embedding_dense_0
            .as_ref()
            .expect("embedding projection"),
    );
    let projected = astype(&projected, MlxDtype::Float32, None);
    let projected = where_cond(
        &context.pool_mask,
        &projected,
        &scalar(0.0, MlxDtype::Float32),
        None,
    );
    let summed = sum_axis(&projected, 1, false, None);
    divide(&summed, &context.lengths, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::LayerConfig;

    #[test]
    fn embedding_gemma2_window_blocks_match_dense_mask_at_boundaries() {
        let mut cfg = crate::model::tests::cfg(false);
        cfg.head_dim = 8;
        cfg.layer_configs = vec![
            LayerConfig {
                head_dim: 8,
                rope_theta: 10000.0,
                rope_dims: 8,
                rope_freqs: None,
                sliding_window: Some(16),
                kv_source_layer: None,
                v_norm_no_scale: true,
            },
            LayerConfig {
                head_dim: 8,
                rope_theta: 1000000.0,
                rope_dims: 8,
                rope_freqs: None,
                sliding_window: None,
                kv_source_layer: None,
                v_norm_no_scale: true,
            },
        ];
        let seq = 520;
        let hidden = mlx_sys::zeros(&[2, seq, 8], MlxDtype::Float32, None);
        let context = Context::new(&cfg, &hidden, &[seq as usize, 17]);
        assert_eq!(context.padding.shape(), vec![2, 1, 1, seq]);
        assert_eq!(context.windows.len(), 3);
        assert!(
            context
                .windows
                .iter()
                .all(|w| w.key_end - w.key_start <= 288)
        );
        let q = reshape(
            &sin(
                &arange(0.0, (2 * 2 * seq * 8) as f64, 1.0, MlxDtype::Float32, None),
                None,
            ),
            &[2, 2, seq, 8],
            None,
        );
        let k = reshape(
            &cos(
                &arange(0.0, (2 * seq * 8) as f64, 1.0, MlxDtype::Float32, None),
                None,
            ),
            &[2, 1, seq, 8],
            None,
        );
        let v = sin(&k, None);
        let actual = context.attention(&q, &k, &v, true);
        let mask: Vec<f32> = (0..seq)
            .flat_map(|q| {
                (0..seq).map(move |k| {
                    if q.abs_diff(k) <= 16 {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let mask = add(&array(&mask, &[1, 1, seq, seq]), &context.padding, None);
        let expected = scaled_dot_product_attention_with_mask(
            &q,
            &k,
            &v,
            1.0,
            ScaledDotProductAttentionMask::Array(&mask),
            None,
        );
        mlx_sys::eval(&[&actual, &expected]);
        for (batch, len) in [(0, seq), (1, 17)] {
            for head in 0..2 {
                let offset = ((batch * 2 + head) * seq * 8) as usize;
                for (a, e) in actual.data_f32()[offset..offset + (len * 8) as usize]
                    .iter()
                    .zip(&expected.data_f32()[offset..offset + (len * 8) as usize])
                {
                    assert!(a.is_finite() && (a - e).abs() < 1e-5, "{a} != {e}");
                }
            }
        }
    }
}
