//! DeepSeek V4 (Flash) re-parameterized MLA attention: raw sliding-window
//! path plus the Phase-3 sparse/compressed paths (CSA ratio 4, HCA ratio
//! 128).
//!
//! Single latent KV head (`wkv` → `head_dim` doubles as K **and** V), Q LoRA
//! (`wq_a` → norm → `wq_b`) with a weight-free per-head RMSNorm on the
//! produced Q, nope/pe split with RoPE on the `pe` slice only, per-head
//! learned attention sinks, an inverse-RoPE ("de-rope") of the attention
//! output's pe slice, and a grouped low-rank output projection
//! (`wo_a` per `o_groups` → `wo_b`).
//!
//! Compress layers (ratio 4/128) first run the Phase-3 compressor update
//! (`deepseek_v4_compressor`), then attend the concatenation of the raw
//! sliding-window latent K and the committed compressed-K rows with a
//! concatenated mask: CSA (ratio 4) masks compressed rows by visibility AND
//! the lightning-indexer top-k selection, HCA (ratio 128) by block
//! visibility alone. Ratio-0 layers take the raw path only.
//!
//! Authoritative references: llama.cpp `src/models/deepseek4.cpp`
//! `build_attention_impl` / `build_raw_attention` / `build_csa_lid_attention`
//! / `build_hca_attention` and vLLM `vllm/models/deepseek_v4/attention.py`
//! (`DeepseekV4Attention`). Every layer rotates with `compress_rope_theta`
//! on compress layers (ratio != 0) — YaRN-scaled on that base when the
//! manifest carries rope_scaling — plain per-layer freqs/theta otherwise.
//!
//! llama.cpp's optional hadamard `k_rot` is a cache-level orthogonal
//! involution (`k_rot² == I`) applied to Q and K before caching and undone
//! on the output; it cancels out mathematically, so the plain path used
//! here is equivalent (it only exists when `attn_rot_k` is enabled). The
//! same applies to the indexer and compressed-K rotations in Phase 3.

use mlx_sys::{
    MlxArray, concatenate, matmul, multiply, quantized_matmul, reshape, rms_norm, rope, slice,
    slice_last_dim, transpose,
};

use super::super::config::{DeepseekV4Config, ModelConfig, layer_params};
use super::attention::{attention_mask_key_len, attention_with_sinks, full_precision_attention};
use super::build_yarn_rope_freqs;
use super::deepseek_v4_compressor::{
    DeepseekV4CompFrame, deepseek_v4_compressor_update, deepseek_v4_lid_top_k_mask,
    deepseek_v4_visibility_mask,
};
use super::utils::qw;
use crate::attention_mask::create_causal_mask;
use crate::kv_cache::MlxKVCache;
use crate::weights::{DeepseekV4LayerWeights, LayerWeights};

/// DeepSeek V4 raw attention from the (already `attn_norm`-normed) HC layer
/// input `x: [1, seq, hidden]` to the branch output `[1, seq, hidden]`.
///
/// `shared_mask`: pre-computed SDPA mask for this layer from
/// `build_layer_masks_for_forward`; `None` computes the mask internally from
/// `seq` (same convention as [`full_precision_attention`] /
/// [`attention_with_sinks`]).
pub(crate) fn deepseek_v4_attention_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    shared_mask: Option<&MlxArray>,
) -> MlxArray {
    let v4_cfg = cfg.deepseek_v4.as_ref().expect("DeepSeek V4 config");
    let v4_w = w.deepseek_v4.as_ref().expect("DeepSeek V4 layer weights");
    let seq = x.shape()[1] as usize;
    let n_heads = cfg.n_heads;
    let head_dim = v4_cfg.head_dim;
    let rot = v4_cfg.qk_rope_head_dim;
    let nope = head_dim - rot;

    // Q LoRA: qr = rms_norm(wq_a(x)); q = wq_b(qr) → [1, seq, H*D] → per-head,
    // then a weight-free per-head RMSNorm (llama.cpp `q_norm` /
    // `ggml_rms_norm` after the reshape; vLLM `fused_..._qnorm_rope`).
    let qr = qw(x, &v4_w.wq_a);
    let qr = rms_norm(&qr, Some(&v4_w.q_a_norm), cfg.rms_norm_eps, None);
    let q = qw(&qr, &v4_w.wq_b);
    let q = reshape(&q, &[1, seq as i32, n_heads as i32, head_dim as i32], None);
    let q = rms_norm(&q, None, cfg.rms_norm_eps, None);
    let q = transpose(&q, &[0, 2, 1, 3], None); // [1, H, seq, D]

    // Single latent KV head: kv = rms_norm(wkv(x)) [1, seq, D].
    let kv = qw(x, &v4_w.wkv);
    let kv = rms_norm(&kv, Some(&v4_w.kv_norm), cfg.rms_norm_eps, None);
    let kv = reshape(&kv, &[1, 1, seq as i32, head_dim as i32], None);

    // RoPE base: compress layers rotate with `compress_rope_theta`; plain
    // layers follow the per-layer freqs/theta (same source the mask builder
    // and standard family use). When the manifest carries YaRN rope_scaling,
    // compress layers build a second YaRN freq set on the compress theta and
    // scale the pe slice by the llama.cpp attn factor (`compress_rope_setup`).
    let (_, layer_rope_theta, _, layer_rope_freqs, sliding_window, _, _) =
        layer_params(cfg, layer_idx);
    let (rope_base, rope_freqs_owned, rope_pe_scale) = if v4_cfg.compress_ratio(layer_idx) != 0 {
        compress_rope_setup(v4_cfg)
    } else {
        match layer_rope_freqs {
            Some(freqs) => (None, Some(freqs.clone()), 1.0),
            None => (Some(layer_rope_theta), None, 1.0),
        }
    };
    let rope_freqs = rope_freqs_owned.as_ref();

    // Split nope [0, D-rot) / pe [D-rot, D); RoPE the pe slice only
    // (GPT-J interleaved — `traditional=true`, vLLM `is_neox_style=False`).
    let q = split_rope_concat(
        &q,
        nope,
        rot,
        rope_base,
        rope_freqs,
        token_offset,
        1.0,
        rope_pe_scale,
    );
    let kv = split_rope_concat(
        &kv,
        nope,
        rot,
        rope_base,
        rope_freqs,
        token_offset,
        1.0,
        rope_pe_scale,
    );

    // Cache the post-RoPE latent K (K doubles as V — there is no V cache),
    // then read back the sliding window from the append's full logical view.
    // (`deepseek_v4_k_window` keys off `cache.seq_len`, which mid-forward is
    // still the pre-append length — the runner advances it only after the
    // forward — so the in-flight window must be sliced from the returned
    // view instead.) The window width mirrors `attention_mask_key_len` so
    // the K view and the hoisted mask width stay in lockstep.
    let full = cache.append_deepseek_v4(layer_idx, kv);

    // Phase 3: compress layers update their compressor pipelines (per-token
    // states → block compression → committed compressed-K/indexer-K rows)
    // before attention, so a query whose own position completes a block sees
    // that block's row. Runs after the raw append, which creates the layer's
    // cache entry the compressor state hangs off.
    let ratio = v4_cfg.compress_ratio(layer_idx);
    let comp = if ratio != 0 {
        deepseek_v4_compressor_update(
            cfg,
            v4_cfg,
            v4_w,
            x,
            cache,
            layer_idx,
            token_offset,
            ratio as usize,
            full.dtype(),
        )
    } else {
        DeepseekV4CompFrame::default()
    };

    let key_len = token_offset + seq;
    let window = attention_mask_key_len(seq, key_len, sliding_window);
    let k_raw = if window < key_len {
        slice(
            &full,
            &[0, 0, (key_len - window) as i32, 0],
            &[1, 1, key_len as i32, head_dim as i32],
            &[1, 1, 1, 1],
            None,
        )
    } else {
        full
    };

    // scores[h] = q[h]·K^T * (1/√D) + mask + per-head sink; out[h] = probs·K
    // (V == K). The native fused SDPA broadcasts the single latent K head
    // across all H query heads (multi-query style). Compress layers attend
    // K = [raw window | compressed rows] with mask = [raw mask | compressed
    // mask] (llama.cpp `build_csa_lid_attention` / `build_hca_attention`).
    let scale = 1.0 / (head_dim as f32).sqrt();
    let (k, mask_opt) = match comp.k_rows.as_ref() {
        Some(comp_k) if comp.n_rows > 0 => {
            let k_all = concatenate(&[&k_raw, comp_k], 2, None);
            let raw_mask = raw_window_mask(shared_mask, seq, window);
            let comp_mask = if ratio == 4 {
                let idx_w = v4_w
                    .indexer
                    .as_ref()
                    .expect("DeepSeek V4 CSA layer must carry indexer weights");
                let idx_rows = comp
                    .indexer_rows
                    .as_ref()
                    .expect("DeepSeek V4 CSA indexer rows commit in lockstep with compressed rows");
                deepseek_v4_lid_top_k_mask(
                    v4_cfg,
                    idx_w,
                    &qr,
                    x,
                    idx_rows,
                    token_offset,
                    seq,
                    ratio as usize,
                )
            } else {
                deepseek_v4_visibility_mask(seq, token_offset, ratio as usize, comp.n_rows)
            };
            let mask = concatenate(&[&raw_mask, &comp_mask], -1, None);
            (k_all, Some(mask))
        }
        _ => (k_raw, shared_mask.cloned()),
    };
    let out = match v4_w.attn_sink.as_ref() {
        Some(sinks) => attention_with_sinks(&q, &k, &k, sinks, scale, seq, &mask_opt),
        None => full_precision_attention(&q, &k, &k, scale, seq, &mask_opt),
    };

    // De-rope (llama.cpp `ggml_rope_ext_back`): inverse-rotate the pe slice
    // of each head's output at the QUERY positions. MLX `rope` multiplies
    // positions by `scale`, so `scale = -1` negates the rotation angles —
    // the exact inverse of the forward rotation above.
    let out = split_rope_concat(
        &out,
        nope,
        rot,
        rope_base,
        rope_freqs,
        token_offset,
        -1.0,
        rope_pe_scale,
    );

    // Grouped low-rank output projection.
    let out = transpose(&out, &[0, 2, 1, 3], None); // [1, seq, H, D]
    let flat = reshape(&out, &[1, seq as i32, (n_heads * head_dim) as i32], None);
    grouped_output_projection(v4_cfg, v4_w, &flat)
}

/// Boolean raw-window mask for the CSA/HCA concat: the hoisted shared mask
/// when the runner built one, otherwise the mask the fast SDPA paths would
/// have applied implicitly. `attention_mask_array` returns `None` exactly
/// when causal-with-offset (multi-token) or no-mask (single-token decode,
/// raw window pre-truncated to the sliding window) is equivalent — both are
/// `create_causal_mask(seq, raw_len - seq, None)` (a single-token row comes
/// out all-true).
fn raw_window_mask(shared_mask: Option<&MlxArray>, seq: usize, raw_len: usize) -> MlxArray {
    shared_mask
        .cloned()
        .unwrap_or_else(|| create_causal_mask(seq, raw_len - seq, None))
}

/// Compress-layer RoPE setup (llama.cpp deepseek4.cpp `build_attention_impl`
/// ~lines 928-935): with YaRN rope_scaling in the manifest, compress layers
/// rotate with YaRN freqs built on the `compress_rope_theta` base (llama.cpp
/// `freq_base_l = dsv4_compress_rope_base` with the config's `factor`,
/// `beta_fast`/`beta_slow`, `n_ctx_orig`) and scale the pe slice by
/// `dsv4_rope_attn_factor(freq_scale = 1/factor)` = `1 / (1 + 0.1*ln(factor))`
/// — the reciprocal of the mscale `build_yarn_rope_freqs` returns with
/// mscale=1/mscale_all_dim=0. Without rope_scaling the layer keeps the plain
/// `compress_rope_theta` base and unit pe scale.
///
/// Returns `(rope_base, rope_freqs, pe_scale)` for [`split_rope_concat`].
/// Also used by the compressor: compressed-K/indexer-K rows and the indexer
/// query rotate with the same compress-layer YaRN configuration (llama.cpp
/// shares one rotary cache per compress layer).
pub(crate) fn compress_rope_setup(
    v4_cfg: &DeepseekV4Config,
) -> (Option<f32>, Option<MlxArray>, f32) {
    match v4_cfg.compress_rope_scaling {
        Some(scaling) => {
            let (freqs, mscale) = build_yarn_rope_freqs(
                v4_cfg.qk_rope_head_dim,
                v4_cfg.compress_rope_theta,
                scaling.factor,
                scaling.original_context_len,
                scaling.beta_fast,
                scaling.beta_slow,
                1.0,
                0.0,
            );
            (None, Some(freqs), 1.0 / mscale)
        }
        None => (Some(v4_cfg.compress_rope_theta), None, 1.0),
    }
}

/// Split `[.., D]` into nope/pe, RoPE the pe slice at `token_offset` with the
/// given `position_scale` (+1 forward, −1 de-rope), and concat back.
/// `pe_scale` multiplies the pe slice before the rotation (YaRN attn factor;
/// 1.0 = no scale).
#[allow(clippy::too_many_arguments)]
fn split_rope_concat(
    x: &MlxArray,
    nope: usize,
    rot: usize,
    rope_base: Option<f32>,
    rope_freqs: Option<&MlxArray>,
    token_offset: usize,
    position_scale: f32,
    pe_scale: f32,
) -> MlxArray {
    let head_dim = (nope + rot) as i32;
    let x_nope = slice_last_dim(x, 0, nope as i32, None);
    let x_pe = slice_last_dim(x, nope as i32, head_dim, None);
    let x_pe = if (pe_scale - 1.0).abs() > 1e-6 {
        multiply(
            &x_pe,
            &mlx_sys::ops::cached_scalar(pe_scale, x_pe.dtype()),
            None,
        )
    } else {
        x_pe
    };
    let x_pe = rope(
        &x_pe,
        rot as i32,
        true,
        rope_base,
        position_scale,
        token_offset as i32,
        rope_freqs,
        None,
    );
    concatenate(&[&x_nope, &x_pe], -1, None)
}

/// Grouped low-rank output projection (llama.cpp `attn_wo_a` / `attn_wo_b`):
/// `flat [1, seq, H*D]` → G groups of `H*D/G` → per-group `wo_a` →
/// `[1, seq, G*R_o]` → `wo_b` → `[1, seq, hidden]`.
///
/// `wo_a` is stored as a `[G*R_o, H*D/G]` matrix whose contiguous `R_o`-row
/// blocks are the per-group projections (vLLM `wo_a` `ColumnParallelLinear`
/// with `bmm_batch_size = G`).
fn grouped_output_projection(
    v4_cfg: &DeepseekV4Config,
    v4_w: &DeepseekV4LayerWeights,
    flat: &MlxArray,
) -> MlxArray {
    let seq = flat.shape()[1];
    let groups = v4_cfg.o_groups as i32;
    let o_lora_rank = v4_cfg.o_lora_rank as i32;
    let group_dim = flat.shape()[2] / groups;
    let grouped = reshape(flat, &[1, seq, groups, group_dim], None);
    let grouped = transpose(&grouped, &[0, 2, 1, 3], None); // [1, G, seq, H*D/G]

    let wo_a = &v4_w.wo_a;
    let oa = if let Some(scales) = &wo_a.scales {
        // Quantization packs along the last dim, so the contiguous R_o-row
        // blocks split cleanly into per-group 3D stacks (same idiom as
        // `glm_mla_embed_q_decode`).
        let w3 = reshape(
            &wo_a.weight,
            &[groups, o_lora_rank, wo_a.weight.shape()[1]],
            None,
        );
        let s3 = reshape(scales, &[groups, o_lora_rank, scales.shape()[1]], None);
        let b3 = wo_a
            .biases
            .as_ref()
            .map(|b| reshape(b, &[groups, o_lora_rank, b.shape()[1]], None));
        quantized_matmul(
            &grouped,
            &w3,
            &s3,
            b3.as_ref(),
            true,
            Some(wo_a.group_size),
            Some(wo_a.bits),
            None,
        )
    } else {
        let w3 = reshape(&wo_a.weight, &[groups, o_lora_rank, group_dim], None);
        let w3 = transpose(&w3, &[0, 2, 1], None);
        matmul(&grouped, &w3, None)
    };

    // [1, G, seq, R_o] → [1, seq, G*R_o] → wo_b [hidden, G*R_o].
    let oa = transpose(&oa, &[0, 2, 1, 3], None);
    let oa = reshape(&oa, &[1, seq, groups * o_lora_rank], None);
    qw(&oa, &v4_w.wo_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::DeepseekV4CompressRopeScaling;
    use crate::weights::QuantizedWeight;
    use mlx_sys::{MlxDtype, contiguous, eval};

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    /// Deterministic pseudo-random fill (no external deps).
    fn fill(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5)
            .collect()
    }

    #[test]
    fn rope_then_derope_is_identity_on_pe_slice() {
        // [1, H, seq, rot] — rope forward at offset then de-rope (scale = -1)
        // must round-trip, validating the inverse-RoPE approach.
        let (heads, seq, rot) = (2usize, 3usize, 4usize);
        let pe = array_f32(&fill(heads * seq * rot, 0.9), &[1, 2, 3, 4]);
        let offset = 5usize;
        let roped = rope(
            &pe,
            rot as i32,
            true,
            Some(10000.0),
            1.0,
            offset as i32,
            None,
            None,
        );
        let back = rope(
            &roped,
            rot as i32,
            true,
            Some(10000.0),
            -1.0,
            offset as i32,
            None,
            None,
        );
        eval(&[&pe, &back]);
        let original = pe.data_f32().to_vec();
        let round_trip = back.data_f32().to_vec();
        for (a, b) in original.iter().zip(round_trip.iter()) {
            assert!((a - b).abs() < 1e-4, "de-rope round trip: {a} vs {b}");
        }
    }

    #[test]
    fn split_rope_concat_preserves_shape_and_nope() {
        let x = array_f32(&fill(2 * 3 * 8, 0.7), &[1, 2, 3, 8]);
        let out = split_rope_concat(&x, 4, 4, Some(10000.0), None, 2, 1.0, 1.0);
        eval(&[&out]);
        assert_eq!(out.shape(), x.shape());
        // nope slice is untouched by the rope path. Slices are strided views;
        // materialize a contiguous copy before reading raw data.
        let nope_out = contiguous(&slice_last_dim(&out, 0, 4, None), None);
        let nope_in = contiguous(&slice_last_dim(&x, 0, 4, None), None);
        eval(&[&nope_out, &nope_in]);
        assert_eq!(nope_out.data_f32().to_vec(), nope_in.data_f32().to_vec());
    }

    fn compress_rope_scaling_test_config(
        scaling: Option<DeepseekV4CompressRopeScaling>,
    ) -> DeepseekV4Config {
        DeepseekV4Config {
            head_dim: 64,
            qk_rope_head_dim: 64,
            q_lora_rank: 8,
            o_lora_rank: 4,
            o_groups: 2,
            index_topk: 8,
            index_n_heads: 2,
            index_head_dim: 4,
            compress_rope_theta: 50000.0,
            compress_rope_scaling: scaling,
            has_attn_sinks: true,
            compress_ratios: vec![4],
            hc_mult: 4,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-5,
            num_hash_layers: 0,
            num_nextn_predict_layers: 0,
            scoring_func: None,
            swiglu_limit: 7.0,
        }
    }

    #[test]
    fn compress_rope_yarn_scales_long_wavelength_divisors() {
        let v4_cfg = compress_rope_scaling_test_config(Some(DeepseekV4CompressRopeScaling {
            factor: 8.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_context_len: 4096,
        }));
        let (base, freqs, pe_scale) = compress_rope_setup(&v4_cfg);
        assert!(
            base.is_none(),
            "YaRN compress rope passes freqs, not a base"
        );
        let freqs = freqs.expect("YaRN freqs for compress layers");
        eval(&[&freqs]);
        let data = freqs.data_f32().to_vec();
        assert_eq!(data.len(), 32);
        // Short-wavelength dims keep the base divisor (i=0 → base^0 == 1).
        assert!((data[0] - 1.0).abs() < 1e-6, "got {}", data[0]);
        // Long-wavelength (high-i) divisors grow toward factor × the plain
        // compress-theta divisor (YaRN interpolation), slowing the rotation
        // rate for distant positions. i=31 sits in the fully-scaled band for
        // this config (wavelength ≫ original_context_len / beta_slow).
        let i = 31;
        let plain = 50000.0f32.powf(2.0 * i as f32 / 64.0);
        assert!(
            data[i] > plain,
            "yarn divisor {} should exceed plain {}",
            data[i],
            plain
        );
        let want = plain * 8.0;
        assert!(
            (data[i] - want).abs() <= 1e-3 * want,
            "fully-scaled band: got {}, expected ~{want}",
            data[i]
        );
        // llama.cpp dsv4_rope_attn_factor(freq_scale = 1/factor)
        // = 1 / (1 + 0.1*ln(factor)).
        let expected_scale = 1.0 / (1.0 + 0.1 * 8.0f32.ln());
        assert!(
            (pe_scale - expected_scale).abs() < 1e-6,
            "got pe_scale {pe_scale}, expected {expected_scale}"
        );
    }

    #[test]
    fn compress_rope_without_scaling_stays_plain() {
        let v4_cfg = compress_rope_scaling_test_config(None);
        let (base, freqs, pe_scale) = compress_rope_setup(&v4_cfg);
        assert_eq!(base, Some(50000.0));
        assert!(freqs.is_none());
        assert_eq!(pe_scale, 1.0);
    }

    #[test]
    fn grouped_output_projection_shapes() {
        // Tiny synthetic config: E=64 (irrelevant here), D=16, H=2, G=2,
        // R_o=4 — H*D=32, group dim 16, G*R_o=8.
        let (seq, groups, o_lora_rank, group_dim) = (3i32, 2i32, 4i32, 16i32);
        let v4_cfg = DeepseekV4Config {
            head_dim: 16,
            qk_rope_head_dim: 4,
            q_lora_rank: 8,
            o_lora_rank: o_lora_rank as usize,
            o_groups: groups as usize,
            index_topk: 8,
            index_n_heads: 2,
            index_head_dim: 4,
            compress_rope_theta: 50000.0,
            compress_rope_scaling: None,
            has_attn_sinks: true,
            compress_ratios: vec![0],
            hc_mult: 4,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-5,
            num_hash_layers: 0,
            num_nextn_predict_layers: 0,
            scoring_func: None,
            swiglu_limit: 7.0,
        };
        let hidden = 64i32;
        let wo_a = QuantizedWeight::new(
            array_f32(
                &fill((groups * o_lora_rank * group_dim) as usize, 0.3),
                &[groups * o_lora_rank, group_dim],
            ),
            None,
            None,
        );
        let wo_b = QuantizedWeight::new(
            array_f32(
                &fill((hidden * groups * o_lora_rank) as usize, 0.4),
                &[hidden, groups * o_lora_rank],
            ),
            None,
            None,
        );
        let v4_w = DeepseekV4LayerWeights {
            wq_a: wo_a.clone(),
            q_a_norm: array_f32(&fill(o_lora_rank as usize, 0.5), &[o_lora_rank]),
            wq_b: wo_a.clone(),
            wkv: wo_a.clone(),
            kv_norm: array_f32(&fill(o_lora_rank as usize, 0.5), &[o_lora_rank]),
            wo_a,
            wo_b,
            attn_sink: None,
            hc_attn_fn: array_f32(&[1.0], &[1]),
            hc_attn_base: array_f32(&[1.0], &[1]),
            hc_attn_scale: array_f32(&[1.0], &[1]),
            hc_ffn_fn: array_f32(&[1.0], &[1]),
            hc_ffn_base: array_f32(&[1.0], &[1]),
            hc_ffn_scale: array_f32(&[1.0], &[1]),
            compressor: None,
            indexer: None,
            tid2eid: None,
        };
        let flat = array_f32(
            &fill((seq * groups * group_dim) as usize, 0.6),
            &[1, seq, groups * group_dim],
        );
        let out = grouped_output_projection(&v4_cfg, &v4_w, &flat);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, seq, hidden]);

        // Reference: per group g, oa_g = flat_g @ wo_a[g*R_o..(g+1)*R_o]^T,
        // then out = oa @ wo_b^T. Check one element by hand.
        let flat_data = flat.data_f32().to_vec();
        let out_data = out.data_f32().to_vec();
        let wa = fill((groups * o_lora_rank * group_dim) as usize, 0.3);
        let wb = fill((hidden * groups * o_lora_rank) as usize, 0.4);
        let (t, e) = (1usize, 7usize);
        let mut expect = 0.0f64;
        for g in 0..groups as usize {
            for r in 0..o_lora_rank as usize {
                let mut oa = 0.0f64;
                for k in 0..group_dim as usize {
                    let x = flat_data[(t * groups as usize + g) * group_dim as usize + k] as f64;
                    oa += x * wa[(g * o_lora_rank as usize + r) * group_dim as usize + k] as f64;
                }
                expect += oa
                    * wb[e * (groups * o_lora_rank) as usize + g * o_lora_rank as usize + r] as f64;
            }
        }
        let actual = out_data[t * hidden as usize + e] as f64;
        assert!(
            (actual - expect).abs() < 1e-4,
            "grouped output mismatch: {actual} vs {expect}"
        );
    }

    // Tiny synthetic config: E=64, D=16, H=2, G=2, R_o=4, rot=4, R_q=8.
    const E: usize = 64;
    const D: usize = 16;
    const H: usize = 2;
    const G: usize = 2;
    const R_O: usize = 4;
    const ROT: usize = 4;
    const R_Q: usize = 8;

    fn attention_test_config() -> ModelConfig {
        ModelConfig {
            compile_cache_identity: 1,
            model_family: "deepseek_v4".to_string(),
            layer_count: 1,
            hidden_size: E,
            intermediate_size: 8,
            n_heads: H,
            n_kv_heads: 1,
            head_dim: D,
            vocab_size: 16,
            rope_theta: 10000.0,
            rope_dims: ROT,
            attn_output_gate: false,
            query_scale: 1.0 / (D as f32).sqrt(),
            final_logit_softcapping: None,
            final_logits_scale: None,
            post_norm_eps: 1e-6,
            embed_norm_no_weight: false,
            moe_expert_count: 0,
            moe_experts_per_token: 0,
            moe_expert_intermediate_size: 0,
            layer_configs: Vec::new(),
            global_sliding_window: None,
            protected_prefix_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            deepseek_v4: Some(DeepseekV4Config {
                head_dim: D,
                qk_rope_head_dim: ROT,
                q_lora_rank: R_Q,
                o_lora_rank: R_O,
                o_groups: G,
                index_topk: 8,
                index_n_heads: 2,
                index_head_dim: 4,
                compress_rope_theta: 50000.0,
                compress_rope_scaling: None,
                has_attn_sinks: true,
                compress_ratios: vec![0],
                hc_mult: 4,
                hc_sinkhorn_iters: 3,
                hc_eps: 1e-5,
                num_hash_layers: 0,
                num_nextn_predict_layers: 0,
                scoring_func: None,
                swiglu_limit: 7.0,
            }),
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            rope_mscale: 1.0,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 8192.0,
            attn_temperature_scale: 0.1,
            intermediate_size_mlp: 0,
            moe_layer_freq: 1,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 0,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: None,
            generation_kind: ax_engine_core::GenerationKind::Autoregressive,
            kv_cache_quant: vec![None; 1],
        }
    }

    fn attention_test_weights() -> LayerWeights {
        let dense = |rows: usize, cols: usize, seed: f32| {
            QuantizedWeight::new(
                array_f32(&fill(rows * cols, seed), &[rows as i32, cols as i32]),
                None,
                None,
            )
        };
        LayerWeights {
            attn_norm: array_f32(&fill(E, 0.9), &[E as i32]),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            attn_out_gate: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: Some(DeepseekV4LayerWeights {
                wq_a: dense(R_Q, E, 0.11),
                q_a_norm: array_f32(&fill(R_Q, 0.8), &[R_Q as i32]),
                wq_b: dense(H * D, R_Q, 0.13),
                wkv: dense(D, E, 0.17),
                kv_norm: array_f32(&fill(D, 0.8), &[D as i32]),
                wo_a: dense(G * R_O, H * D / G, 0.19),
                wo_b: dense(E, G * R_O, 0.23),
                attn_sink: Some(array_f32(&[-1.0, -2.0], &[H as i32])),
                hc_attn_fn: array_f32(&[1.0], &[1]),
                hc_attn_base: array_f32(&[1.0], &[1]),
                hc_attn_scale: array_f32(&[1.0], &[1]),
                hc_ffn_fn: array_f32(&[1.0], &[1]),
                hc_ffn_base: array_f32(&[1.0], &[1]),
                hc_ffn_scale: array_f32(&[1.0], &[1]),
                compressor: None,
                indexer: None,
                tid2eid: None,
            }),
            ffn_norm: array_f32(&fill(E, 0.9), &[E as i32]),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
            expert_stream: None,
        }
    }

    #[test]
    fn attention_forward_prefill_and_decode_shapes() {
        let cfg = attention_test_config();
        let w = attention_test_weights();
        let mut cache = MlxKVCache::new(1);

        // Prefill: seq=3 at offset 0 → [1, 3, E], cache holds 3 latent rows.
        let x = array_f32(&fill(3 * E, 0.63), &[1, 3, E as i32]);
        let out = deepseek_v4_attention_forward(&cfg, &w, &x, &mut cache, 0, 0, None);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 3, E as i32]);
        assert!(out.data_f32().iter().all(|v| v.is_finite()));
        cache.advance(3);

        // Decode: seq=1 at offset 3 → [1, 1, E], cache view spans 4 rows.
        let x = array_f32(&fill(E, 0.67), &[1, 1, E as i32]);
        let out = deepseek_v4_attention_forward(&cfg, &w, &x, &mut cache, 0, 3, None);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 1, E as i32]);
        assert!(out.data_f32().iter().all(|v| v.is_finite()));
        cache.advance(1);

        let state = cache.deepseek_v4_layer_state(0).expect("V4 cache state");
        assert_eq!(state.head_dim, D as i32);
        let full = cache
            .deepseek_v4_k_window(0, 0)
            .expect("full window read-back");
        assert_eq!(full.shape(), vec![1, 1, 4, D as i32]);
    }

    // ── Phase 3: CSA / HCA compressed paths ─────────────────────────────

    use crate::weights::{DeepseekV4CompressorWeights, DeepseekV4IndexerWeights};

    const I: usize = 8; // index_head_dim (real: 128)
    const HI: usize = 2; // index_n_heads (real: 64)

    /// Ratio-4 (CSA) variant of the synthetic config: indexer present,
    /// `index_topk = 1` so committing a second row exercises the top-k
    /// selection rather than the short-context no-op.
    fn csa_test_config() -> ModelConfig {
        let mut cfg = attention_test_config();
        let v4 = cfg.deepseek_v4.as_mut().expect("v4 cfg");
        v4.compress_ratios = vec![4];
        v4.index_head_dim = I;
        v4.index_n_heads = HI;
        v4.index_topk = 1;
        cfg
    }

    /// Ratio-8 (HCA stand-in for 128) variant: compressor with coff=1 state
    /// width, no indexer. `compress_ratios` is a plain `Vec<u32>` (no
    /// {0,4,128} validation at config level), so a small test ratio keeps
    /// the schedule observable in a handful of tokens.
    fn hca_test_config() -> ModelConfig {
        let mut cfg = attention_test_config();
        cfg.deepseek_v4.as_mut().expect("v4 cfg").compress_ratios = vec![8];
        cfg
    }

    fn add_compressor_weights(w: &mut LayerWeights, ratio: usize, indexer: bool) {
        let overlap = ratio == 4;
        let coff = if overlap { 2 } else { 1 };
        let v4 = w.deepseek_v4.as_mut().expect("v4 weights");
        v4.compressor = Some(DeepseekV4CompressorWeights {
            kv: QuantizedWeight::new(
                array_f32(&fill(coff * D * E, 0.11), &[(coff * D) as i32, E as i32]),
                None,
                None,
            ),
            gate: QuantizedWeight::new(
                array_f32(&fill(coff * D * E, 0.13), &[(coff * D) as i32, E as i32]),
                None,
                None,
            ),
            ape: array_f32(
                &fill(ratio * coff * D, 0.17),
                &[ratio as i32, (coff * D) as i32],
            ),
            norm: array_f32(&fill(D, 0.19), &[D as i32]),
        });
        if indexer {
            v4.indexer = Some(DeepseekV4IndexerWeights {
                proj: QuantizedWeight::new(
                    array_f32(&fill(HI * E, 0.23), &[HI as i32, E as i32]),
                    None,
                    None,
                ),
                qb: QuantizedWeight::new(
                    array_f32(&fill(HI * I * R_Q, 0.27), &[(HI * I) as i32, R_Q as i32]),
                    None,
                    None,
                ),
                compressor_kv: QuantizedWeight::new(
                    array_f32(&fill(2 * I * E, 0.29), &[(2 * I) as i32, E as i32]),
                    None,
                    None,
                ),
                compressor_gate: QuantizedWeight::new(
                    array_f32(&fill(2 * I * E, 0.31), &[(2 * I) as i32, E as i32]),
                    None,
                    None,
                ),
                compressor_ape: array_f32(
                    &fill(ratio * 2 * I, 0.37),
                    &[ratio as i32, (2 * I) as i32],
                ),
                compressor_norm: array_f32(&fill(I, 0.41), &[I as i32]),
            });
        }
    }

    fn forward_step(
        cfg: &ModelConfig,
        w: &LayerWeights,
        cache: &mut MlxKVCache,
        seq: usize,
        token_offset: usize,
        seed: f32,
    ) -> MlxArray {
        let x = array_f32(&fill(seq * E, seed), &[1, seq as i32, E as i32]);
        let out = deepseek_v4_attention_forward(cfg, w, &x, cache, 0, token_offset, None);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, seq as i32, E as i32]);
        assert!(out.data_f32().iter().all(|v| v.is_finite()));
        cache.advance(seq);
        out
    }

    #[test]
    fn csa_forward_prefill_then_decode_across_block_boundaries() {
        let cfg = csa_test_config();
        let mut w = attention_test_weights();
        add_compressor_weights(&mut w, 4, true);
        let mut cache = MlxKVCache::new(1);

        // Prefill 3 tokens: no block complete → raw path, nothing committed.
        forward_step(&cfg, &w, &mut cache, 3, 0, 0.61);
        assert_eq!(cache.deepseek_v4_comp_committed(0, false), 0);

        // Decode pos 3 completes block 0 → CSA concat path (1 row ≤ topk →
        // short-context, indexer mask is visibility-only).
        forward_step(&cfg, &w, &mut cache, 1, 3, 0.63);
        assert_eq!(cache.deepseek_v4_comp_committed(0, false), 1);
        assert_eq!(cache.deepseek_v4_comp_committed(0, true), 1);

        // Decode pos 4..=6: no new rows.
        for pos in 4..7 {
            forward_step(&cfg, &w, &mut cache, 1, pos, 0.01 * pos as f32);
            assert_eq!(cache.deepseek_v4_comp_committed(0, false), 1, "pos {pos}");
        }

        // Decode pos 7 completes block 1 → 2 committed rows > topk=1 → the
        // lightning-indexer top-k selection runs.
        forward_step(&cfg, &w, &mut cache, 1, 7, 0.71);
        assert_eq!(cache.deepseek_v4_comp_committed(0, false), 2);
        let rows = cache.deepseek_v4_comp_k(0, false).expect("compressed rows");
        assert_eq!(rows.shape(), vec![1, 1, 2, D as i32]);
        let idx_rows = cache.deepseek_v4_comp_k(0, true).expect("indexer rows");
        assert_eq!(idx_rows.shape(), vec![1, 1, 2, I as i32]);
    }

    #[test]
    fn hca_forward_prefill_then_decode_across_block_boundaries() {
        let cfg = hca_test_config();
        let mut w = attention_test_weights();
        add_compressor_weights(&mut w, 8, false);
        let mut cache = MlxKVCache::new(1);

        // Prefill 9 tokens (r=8): block 0 completes mid-chunk → the prefill
        // itself already takes the HCA concat path for its last query.
        forward_step(&cfg, &w, &mut cache, 9, 0, 0.61);
        assert_eq!(cache.deepseek_v4_comp_committed(0, false), 1);
        assert!(cache.deepseek_v4_comp_k(0, true).is_none());

        // Decode pos 9..=14: no new rows; pos 15 completes block 1.
        for pos in 9..16 {
            forward_step(&cfg, &w, &mut cache, 1, pos, 0.01 * pos as f32);
            let expect = if pos < 15 { 1 } else { 2 };
            assert_eq!(
                cache.deepseek_v4_comp_committed(0, false),
                expect,
                "pos {pos}"
            );
        }
        let rows = cache.deepseek_v4_comp_k(0, false).expect("compressed rows");
        assert_eq!(rows.shape(), vec![1, 1, 2, D as i32]);
    }
}
