//! DeepSeek V4 (Flash) family: owns the packed hyper-connection residual.
//!
//! The residual stream is `[1, seq, hc_mult * hidden]` (4 packed streams);
//! this family expands the token embedding into it, threads it through every
//! layer end-to-end (each layer is `hc_pre` → branch → `hc_post` for both the
//! attention and MoE-FFN branches), and collapses it with the root-level
//! `hc_head` before the final norm. The generic `layer_forward` dispatch in
//! `model/mod.rs` deliberately does NOT route here — V4 forward passes go
//! through the dedicated top-level helpers (E-wide only at the boundaries).
//!
//! References: llama.cpp `src/models/deepseek4.cpp` graph loop
//! (`hc_init` / `build_hc_pre` / `build_hc_post` / `hc_head`) and vLLM
//! `vllm/models/deepseek_v4`.

use mlx_sys::{MlxArray, add, broadcast_to, reshape, rms_norm};

use super::super::ModelConfig;
use super::super::shared::{
    deepseek_v4_attention_forward, hc_head, hc_post, hc_pre, moe_experts_forward,
    moe_router_deepseek_v4, shared_expert_forward,
};
use crate::kv_cache::MlxKVCache;
use crate::weights::{DeepseekV4HeadWeights, LayerWeights};

/// Expand the embedded tokens `[1, seq, hidden]` into the packed residual
/// stream `[1, seq, hc_mult * hidden]` by repeating the embedding into each
/// stream (llama.cpp `ggml_repeat_4d` `hc_init`, deepseek4.cpp:1286-1288).
pub(crate) fn expand_embedding(
    embedded: &MlxArray,
    cfg: &super::super::config::DeepseekV4Config,
) -> MlxArray {
    let shape = embedded.shape();
    assert_eq!(
        shape.len(),
        3,
        "V4 expand_embedding expects [1, seq, hidden]"
    );
    let seq = shape[1];
    let hidden = shape[2];
    let hc = cfg.hc_mult as i32;
    let streams = reshape(embedded, &[1, seq, 1, hidden], None);
    let tiled = broadcast_to(&streams, &[1, seq, hc, hidden], None);
    reshape(&tiled, &[1, seq, hc * hidden], None)
}

/// One DeepSeek V4 layer over the packed residual stream:
/// `[1, seq, hc*hidden]` → `[1, seq, hc*hidden]`.
///
/// Both branches are `hc_pre` → `rms_norm` → branch → `hc_post`. Attention
/// dispatches per layer on the compress ratio: raw sliding-window (0), CSA
/// (4), or HCA (128) — and the FFN is the MoE path (`sqrtsoftplus` router,
/// hash-routed on the leading `num_hash_layers` layers — `token_ids` feeds
/// the `tid2eid` lookup).
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    packed_stream: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    token_ids: Option<&MlxArray>,
    shared_mask: Option<&MlxArray>,
) -> MlxArray {
    let v4_cfg = cfg.deepseek_v4.as_ref().expect("DeepSeek V4 config");
    let v4_w = w.deepseek_v4.as_ref().expect("DeepSeek V4 layer weights");

    // Attention branch.
    let pre = hc_pre(
        packed_stream,
        &v4_w.hc_attn_fn,
        &v4_w.hc_attn_base,
        &v4_w.hc_attn_scale,
        v4_cfg,
        cfg.rms_norm_eps,
    );
    let normed = rms_norm(&pre.layer_input, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let attn_out =
        deepseek_v4_attention_forward(cfg, w, &normed, cache, layer_idx, token_offset, shared_mask);
    let packed_stream = hc_post(&attn_out, packed_stream, &pre.post, &pre.comb, v4_cfg);

    // MoE FFN branch (every V4 layer is MoE): sqrtsoftplus/hash router +
    // routed experts + shared expert (deepseek4.cpp:1316-1366).
    let pre = hc_pre(
        &packed_stream,
        &v4_w.hc_ffn_fn,
        &v4_w.hc_ffn_base,
        &v4_w.hc_ffn_scale,
        v4_cfg,
        cfg.rms_norm_eps,
    );
    let normed = rms_norm(&pre.layer_input, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let (indices, weights) = moe_router_deepseek_v4(cfg, w, &normed, token_ids);
    let ffn_out = if cfg.moe_shared_expert_count > 0 {
        let shared_out = shared_expert_forward(cfg, w, &normed);
        add(
            &moe_experts_forward(cfg, w, &normed, &indices, &weights),
            &shared_out,
            None,
        )
    } else {
        moe_experts_forward(cfg, w, &normed, &indices, &weights)
    };
    hc_post(&ffn_out, &packed_stream, &pre.post, &pre.comb, v4_cfg)
}

/// Collapse the packed residual stream to `[1, seq, hidden]` via the
/// root-level hyper-connection head, before the final RMSNorm + lm_head
/// (llama.cpp `build_hc_head`, deepseek4.cpp:1394).
pub(crate) fn collapse_for_head(
    cfg: &ModelConfig,
    head_w: &DeepseekV4HeadWeights,
    packed_stream: &MlxArray,
) -> MlxArray {
    let v4_cfg = cfg.deepseek_v4.as_ref().expect("DeepSeek V4 config");
    hc_head(
        packed_stream,
        &head_w.hc_head_fn,
        &head_w.hc_head_base,
        &head_w.hc_head_scale,
        v4_cfg,
        cfg.rms_norm_eps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::QuantizedWeight;
    use mlx_sys::{MlxDtype, eval, zeros};

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

    // Tiny synthetic config: E=64, D=16, H=2, G=2, R_o=4, rot=4, R_q=8.
    const E: usize = 64;
    const D: usize = 16;
    const H: usize = 2;
    const G: usize = 2;
    const R_O: usize = 4;
    const ROT: usize = 4;
    const R_Q: usize = 8;
    const HC: usize = 4;
    const N_EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const INTER: usize = 8;
    const VOCAB: usize = 16;

    fn test_v4_config() -> super::super::super::config::DeepseekV4Config {
        super::super::super::config::DeepseekV4Config {
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
            hc_mult: HC,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-5,
            num_hash_layers: 1,
            num_nextn_predict_layers: 0,
            scoring_func: Some("sqrtsoftplus".to_string()),
            swiglu_limit: 7.0,
        }
    }

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            compile_cache_identity: 1,
            model_family: "deepseek_v4".to_string(),
            layer_count: 1,
            hidden_size: E,
            intermediate_size: INTER,
            n_heads: H,
            n_kv_heads: 1,
            head_dim: D,
            vocab_size: VOCAB,
            rope_theta: 10000.0,
            rope_dims: ROT,
            attn_output_gate: false,
            query_scale: 1.0 / (D as f32).sqrt(),
            final_logit_softcapping: None,
            final_logits_scale: None,
            post_norm_eps: 1e-6,
            embed_norm_no_weight: false,
            moe_expert_count: N_EXPERTS,
            moe_experts_per_token: TOP_K,
            moe_expert_intermediate_size: INTER,
            layer_configs: Vec::new(),
            global_sliding_window: None,
            protected_prefix_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: true,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            deepseek_v4: Some(test_v4_config()),
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            rope_mscale: 1.0,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 8192.0,
            attn_temperature_scale: 0.1,
            intermediate_size_mlp: 0,
            moe_layer_freq: 1,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 1,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: None,
            gpt_oss_uses_mxfp4_experts: false,
            generation_kind: ax_engine_core::GenerationKind::Autoregressive,
            kv_cache_quant: vec![None; 1],
        }
    }

    fn dense_weight(rows: usize, cols: usize, seed: f32) -> QuantizedWeight {
        QuantizedWeight::new(
            array_f32(&fill(rows * cols, seed), &[rows as i32, cols as i32]),
            None,
            None,
        )
    }

    fn hc_branch_weights(mixes: usize, seed: f32) -> (MlxArray, MlxArray, MlxArray) {
        (
            array_f32(
                &fill(mixes * HC * E, seed),
                &[mixes as i32, (HC * E) as i32],
            ),
            array_f32(&fill(mixes, seed + 0.1), &[mixes as i32]),
            array_f32(&[1.0, 1.0, 1.0], &[3]),
        )
    }

    fn test_layer_weights(hash_layer: bool) -> LayerWeights {
        let mixes = 2 * HC + HC * HC;
        let (hc_attn_fn, hc_attn_base, hc_attn_scale) = hc_branch_weights(mixes, 0.31);
        let (hc_ffn_fn, hc_ffn_base, hc_ffn_scale) = hc_branch_weights(mixes, 0.47);
        // [vocab, topk] hash table used on hash-routed layers.
        let mut table = vec![0u32; VOCAB * TOP_K];
        for token in 0..VOCAB {
            table[token * TOP_K] = (token % N_EXPERTS) as u32;
            table[token * TOP_K + 1] = ((token + 1) % N_EXPERTS) as u32;
        }
        let tid2eid = MlxArray::from_raw_data(
            table.as_ptr() as *const u8,
            std::mem::size_of_val(table.as_slice()),
            &[VOCAB as i32, TOP_K as i32],
            MlxDtype::Uint32,
        );
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
            deepseek_v4: Some(crate::weights::DeepseekV4LayerWeights {
                wq_a: dense_weight(R_Q, E, 0.11),
                q_a_norm: array_f32(&fill(R_Q, 0.8), &[R_Q as i32]),
                wq_b: dense_weight(H * D, R_Q, 0.13),
                wkv: dense_weight(D, E, 0.17),
                kv_norm: array_f32(&fill(D, 0.8), &[D as i32]),
                wo_a: dense_weight(G * R_O, H * D / G, 0.19),
                wo_b: dense_weight(E, G * R_O, 0.23),
                attn_sink: Some(array_f32(&[-1.0, -2.0], &[H as i32])),
                hc_attn_fn,
                hc_attn_base,
                hc_attn_scale,
                hc_ffn_fn,
                hc_ffn_base,
                hc_ffn_scale,
                compressor: None,
                indexer: None,
                tid2eid: hash_layer.then_some(tid2eid),
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
            router_proj: Some(dense_weight(N_EXPERTS, E, 0.29)),
            router_correction_bias: (!hash_layer)
                .then(|| array_f32(&fill(N_EXPERTS, 0.05), &[N_EXPERTS as i32])),
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: Some(dense_weight(INTER, E, 0.37)),
            shared_up_proj: Some(dense_weight(INTER, E, 0.41)),
            shared_down_proj: Some(dense_weight(E, INTER, 0.43)),
            gate_up_exps_packed: None,
            gate_exps: Some(QuantizedWeight::new(
                array_f32(
                    &fill(N_EXPERTS * INTER * E, 0.53),
                    &[N_EXPERTS as i32, INTER as i32, E as i32],
                ),
                None,
                None,
            )),
            up_exps: Some(QuantizedWeight::new(
                array_f32(
                    &fill(N_EXPERTS * INTER * E, 0.59),
                    &[N_EXPERTS as i32, INTER as i32, E as i32],
                ),
                None,
                None,
            )),
            down_exps: Some(QuantizedWeight::new(
                array_f32(
                    &fill(N_EXPERTS * E * INTER, 0.61),
                    &[N_EXPERTS as i32, E as i32, INTER as i32],
                ),
                None,
                None,
            )),
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
            expert_stream: None,
        }
    }

    #[test]
    fn expand_embedding_tiles_streams() {
        let cfg = test_v4_config();
        let embedded = array_f32(&fill(2 * E, 0.7), &[1, 2, E as i32]);
        let packed = expand_embedding(&embedded, &cfg);
        eval(&[&packed]);
        assert_eq!(packed.shape(), vec![1, 2, (HC * E) as i32]);
        let data = packed.data_f32().to_vec();
        let src = embedded.data_f32().to_vec();
        for t in 0..2 {
            for h in 0..HC {
                for e in 0..E {
                    assert_eq!(data[(t * HC + h) * E + e], src[t * E + e]);
                }
            }
        }
    }

    #[test]
    fn layer_forward_packed_stream_round_trip() {
        let cfg = test_model_config();
        let w = test_layer_weights(true);
        let mut cache = MlxKVCache::new(1);
        let packed = array_f32(&fill(3 * HC * E, 0.67), &[1, 3, (HC * E) as i32]);
        let token_ids = MlxArray::from_raw_data(
            [3u32, 7, 11].as_ptr() as *const u8,
            3 * std::mem::size_of::<u32>(),
            &[1, 3],
            MlxDtype::Uint32,
        );
        let out = layer_forward(&cfg, &w, &packed, &mut cache, 0, 0, Some(&token_ids), None);
        eval(&[&out]);
        assert_eq!(out.shape(), packed.shape());
        assert!(out.data_f32().iter().all(|v| v.is_finite()));
        cache.advance(3);

        // Decode step: one more token at offset 3 through the same cache.
        let packed_1 = array_f32(&fill(HC * E, 0.71), &[1, 1, (HC * E) as i32]);
        let token_ids_1 = MlxArray::from_raw_data(
            [5u32].as_ptr() as *const u8,
            std::mem::size_of::<u32>(),
            &[1, 1],
            MlxDtype::Uint32,
        );
        let out_1 = layer_forward(
            &cfg,
            &w,
            &packed_1,
            &mut cache,
            0,
            3,
            Some(&token_ids_1),
            None,
        );
        eval(&[&out_1]);
        assert_eq!(out_1.shape(), packed_1.shape());
        assert!(out_1.data_f32().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn collapse_for_head_returns_hidden_width() {
        let cfg = test_model_config();
        let head_w = crate::weights::DeepseekV4HeadWeights {
            hc_head_fn: array_f32(&fill(HC * HC * E, 0.21), &[HC as i32, (HC * E) as i32]),
            hc_head_base: array_f32(&fill(HC, 0.22), &[HC as i32]),
            hc_head_scale: array_f32(&[1.0], &[1]),
        };
        let packed = array_f32(&fill(2 * HC * E, 0.67), &[1, 2, (HC * E) as i32]);
        let out = collapse_for_head(&cfg, &head_w, &packed);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 2, E as i32]);
        assert!(out.data_f32().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn moe_hash_path_routes_via_tid2eid() {
        // Zeroed packed stream → hc_pre layer input is 0 → router logits are
        // all-equal, so the learned path could pick any experts; the hash
        // path must still select exactly the table entries for the token.
        let cfg = test_model_config();
        let w = test_layer_weights(true);
        let mut cache = MlxKVCache::new(1);
        let packed = zeros(&[1, 1, (HC * E) as i32], MlxDtype::Float32, None);
        let token_ids = MlxArray::from_raw_data(
            [7u32].as_ptr() as *const u8,
            std::mem::size_of::<u32>(),
            &[1, 1],
            MlxDtype::Uint32,
        );
        let out = layer_forward(&cfg, &w, &packed, &mut cache, 0, 0, Some(&token_ids), None);
        eval(&[&out]);
        assert_eq!(out.shape(), packed.shape());
        assert!(out.data_f32().iter().all(|v| v.is_finite()));
    }
}
