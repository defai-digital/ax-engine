//! Bounded native Unlimited-OCR-2 attention contract and page coordinates.

use serde::{Deserialize, Serialize};

use crate::{NativeModelManifest, NativeTensorDataType, NativeTensorRole};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UnlimitedOcr2Config {
    pub block_size: usize,
    pub local_blocks: usize,
    pub remote_blocks: usize,
    pub index_dim: usize,
    pub image_token_id: u32,
    pub page_token_id: u32,
    pub scales: Vec<f32>,
    pub floors: Vec<f32>,
    pub sigma: f32,
}

impl UnlimitedOcr2Config {
    /// Admit the released head-class/page2d contract. Other sparse graphs
    /// must not silently inherit this implementation.
    pub fn from_hf(value: &serde_json::Value, layers: usize) -> Result<Option<Self>, String> {
        let scopes = [
            Some(value),
            value.get("language_config"),
            value.get("text_config"),
        ];
        let sparse: Vec<_> = scopes
            .into_iter()
            .flatten()
            .filter_map(|v| v.get("sparse_attention_config").filter(|s| !s.is_null()))
            .collect();
        let Some(s) = sparse.first() else {
            return Ok(None);
        };
        if sparse.iter().any(|other| other != s) {
            return Err("conflicting sparse attention scopes".into());
        }
        for (key, expected) in [
            ("use_sparse_attention", serde_json::json!(true)),
            ("training_stage", serde_json::json!("dash")),
            ("dash_stage0", serde_json::json!("head_cls")),
            ("dash_router", serde_json::json!("entmax_topk_perlayer")),
            ("sparse_main_rope_mode", serde_json::json!("page2d")),
            ("sparse_rope_mode", serde_json::json!("page2d")),
            (
                "sparse_main_rope_tok_freq_mode",
                serde_json::json!("resample"),
            ),
            ("sparse_rope_tok_freq_mode", serde_json::json!("resample")),
            ("sparse_rope_tok_theta", serde_json::json!(10000.0)),
            ("sparse_rope_page_theta", serde_json::json!(256.0)),
            ("sparse_rope_page_pairs", serde_json::json!(1)),
            ("sparse_rope_page_mod", serde_json::json!(2)),
            ("sparse_rope_page_span", serde_json::json!(0)),
            ("dash_image_only", serde_json::json!(true)),
            ("image_bucket_constraint", serde_json::json!(true)),
        ] {
            // `10000` and `10000.0` are distinct `Value`s but the same setting.
            let matches = match (s.get(key), &expected) {
                (Some(serde_json::Value::Number(got)), serde_json::Value::Number(want)) => {
                    got.as_f64() == want.as_f64()
                }
                (got, want) => got == Some(want),
            };
            if !matches {
                return Err(format!(
                    "unsupported Unlimited-OCR-2 {key}: expected {expected}"
                ));
            }
        }
        let every_layer = s
            .get("sparse_attention_freq")
            .and_then(|v| v.as_array())
            .is_some_and(|freq| {
                freq.len() == layers && freq.iter().all(|v| v.as_f64() == Some(1.0))
            });
        if !every_layer || s.get("sparse_inconsistency_freq").is_some() {
            return Err("Unlimited-OCR-2 requires independent DASH routing at every layer".into());
        }
        let integer = |key: &str| -> Result<usize, String> {
            s.get(key)
                .and_then(|v| v.as_u64())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("missing Unlimited-OCR-2 {key}"))
        };
        let profile = |key: &str| -> Result<Vec<f32>, String> {
            serde_json::from_value(s.get(key).cloned().unwrap_or_default())
                .map_err(|e| format!("invalid Unlimited-OCR-2 {key}: {e}"))
        };
        let cfg = Self {
            block_size: integer("sparse_block_size")?,
            local_blocks: integer("sparse_local_block")?,
            remote_blocks: integer("dash_topk_blocks")?,
            index_dim: integer("sparse_index_dim")?,
            image_token_id: u32::try_from(integer("image_token_id")?).map_err(|e| e.to_string())?,
            page_token_id: u32::try_from(integer("page_token_id")?).map_err(|e| e.to_string())?,
            scales: profile("dash_scale_per_layer")?,
            floors: profile("dash_floor_per_layer")?,
            sigma: s
                .get("dash_sigma")
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::NAN) as f32,
        };
        cfg.validate(layers)?;
        Ok(Some(cfg))
    }

    pub fn validate(&self, layers: usize) -> Result<(), String> {
        if self.block_size != 32
            || self.local_blocks != 4
            || self.remote_blocks != 12
            || self.index_dim != 128
            || self.scales.len() != layers
            || self.floors.len() != layers
            || self.image_token_id == self.page_token_id
            || !self.sigma.is_finite()
            || self.sigma <= 0.0
            || self.scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self
                .floors
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0 || *v > 1.0)
        {
            return Err("unsupported Unlimited-OCR-2 DASH geometry/profile".into());
        }
        Ok(())
    }
}

/// Source-layout tensors retain their exact names; expert ids are never
/// inferred from lexicographic order.
pub fn source_expert(name: &str) -> Option<(u32, u32, &str)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once(".mlp.experts.")?;
    let (expert, suffix) = rest.split_once('.')?;
    matches!(
        suffix,
        "gate_proj.weight" | "up_proj.weight" | "down_proj.weight"
    )
    .then_some((layer.parse().ok()?, expert.parse().ok()?, suffix))
}

pub(crate) fn validate_manifest(m: &NativeModelManifest) -> Result<(), String> {
    let cfg = m
        .unlimited_ocr2
        .as_ref()
        .ok_or("missing Unlimited-OCR-2 attention contract")?;
    cfg.validate(m.layer_count as usize)?;
    for tensor in &m.tensors {
        if crate::convert::ocr2_tensor_mapping(&tensor.name)
            != Some((tensor.role, tensor.layer_index))
        {
            return Err(format!(
                "OCR-2 tensor role differs from its source name: {}",
                tensor.name
            ));
        }
        if let Some((layer, expert, _)) = source_expert(&tensor.name)
            && (layer == 0 || layer >= m.layer_count || expert >= 64)
        {
            return Err(format!("unexpected OCR-2 expert {}", tensor.name));
        }
        for router in [".self_attn.dash_router.", ".self_attn.indexer."] {
            if let Some((_, suffix)) = tensor.name.split_once(router)
                && !matches!(
                    suffix,
                    "q_proj.weight" | "k_proj.weight" | "q_norm.weight" | "k_norm.weight"
                )
                && !(router.contains("dash_router") && suffix == "head_cls")
            {
                return Err(format!("unknown OCR-2 router state {}", tensor.name));
            }
        }
    }
    if m.attention_head_dim != 128
        || m.hidden_size != 1280
        || m.attention_head_count != 10
        || m.vocab_size != 129280
        || m.kv_head_count != m.attention_head_count
        || m.layer_count != 12
        || m.moe.expert_count != Some(64)
        || m.moe.experts_per_token != Some(6)
        || m.moe.expert_intermediate_size != Some(896)
        || m.moe.shared_expert_count != Some(2)
        || m.moe_norm_topk_prob
        || m.tie_word_embeddings
        || m.linear_attention.is_enabled()
        || m.mla_attention.is_enabled()
        || m.deepseek_v4.is_enabled()
        || !m.layer_types.is_empty()
        || m.sliding_window_size.is_some()
        || !m.kv_shared_source_layers.is_empty()
        || m.kv_cache_quantization.is_some()
    {
        return Err("unsupported Unlimited-OCR-2 language geometry".into());
    }
    let matrix = |name: &str, rows: u64, cols: u64, protected: bool| -> Result<(), String> {
        let t = m
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing {name}"))?;
        let packed_cols = if t.source_quantized {
            let q = t
                .quantization
                .as_ref()
                .ok_or_else(|| format!("missing quantization for {name}"))?;
            if protected || q.bits != 8 || q.group_size != 32 || q.mode != "mxfp8" {
                return Err(format!("unsupported or protected quantization for {name}"));
            }
            let base = name.strip_suffix(".weight").ok_or("invalid packed name")?;
            let scales = m
                .tensors
                .iter()
                .find(|s| s.name == format!("{base}.scales"))
                .ok_or_else(|| format!("missing {base}.scales"))?;
            if scales.shape != [rows, cols / 32] || scales.dtype != NativeTensorDataType::U8 {
                return Err(format!("invalid {base}.scales"));
            }
            cols * 8 / 32
        } else {
            if !matches!(
                t.dtype,
                NativeTensorDataType::Bf16 | NativeTensorDataType::F32 | NativeTensorDataType::F16
            ) {
                return Err(format!("invalid floating matrix dtype for {name}"));
            }
            cols
        };
        if t.shape != [rows, packed_cols] {
            return Err(format!("invalid {name} shape {:?}", t.shape));
        }
        Ok(())
    };
    let vector = |name: &str, len: u64| -> Result<(), String> {
        let t = m
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing {name}"))?;
        if t.source_quantized
            || t.shape != [len]
            || !matches!(
                t.dtype,
                NativeTensorDataType::Bf16 | NativeTensorDataType::F32 | NativeTensorDataType::F16
            )
        {
            return Err(format!("invalid protected {name}"));
        }
        Ok(())
    };
    let h = u64::from(m.hidden_size);
    matrix(
        "model.embed_tokens.weight",
        u64::from(m.vocab_size),
        h,
        true,
    )?;
    matrix("lm_head.weight", u64::from(m.vocab_size), h, true)?;
    vector("model.norm.weight", h)?;
    vector("model.projector.layers.bias", h)?;
    vector("model.image_newline", h)?;
    vector("model.view_seperator", h)?;
    matrix("model.projector.layers.weight", h, 2048, true)?;
    for layer in 0..m.layer_count {
        let p = format!("model.layers.{layer}");
        for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            vector(&format!("{p}.{norm}"), h)?;
        }
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            matrix(&format!("{p}.self_attn.{proj}.weight"), h, h, false)?;
        }
        for router in ["dash_router", "indexer"] {
            for proj in ["q_proj", "k_proj"] {
                matrix(
                    &format!("{p}.self_attn.{router}.{proj}.weight"),
                    128,
                    h,
                    true,
                )?;
            }
            for norm in ["q_norm", "k_norm"] {
                vector(&format!("{p}.self_attn.{router}.{norm}.weight"), 128)?;
            }
        }
        vector(&format!("{p}.self_attn.dash_router.head_cls"), 128)?;
        if layer == 0 {
            for (proj, rows, cols) in [
                ("gate_proj", 6848, h),
                ("up_proj", 6848, h),
                ("down_proj", h, 6848),
            ] {
                matrix(&format!("{p}.mlp.{proj}.weight"), rows, cols, false)?;
            }
        } else {
            matrix(&format!("{p}.mlp.gate.weight"), 64, h, true)?;
            for (proj, rows, cols) in [
                ("gate_proj", 1792, h),
                ("up_proj", 1792, h),
                ("down_proj", h, 1792),
            ] {
                matrix(
                    &format!("{p}.mlp.shared_experts.{proj}.weight"),
                    rows,
                    cols,
                    false,
                )?;
            }
            for expert in 0..64 {
                for (proj, rows, cols) in [
                    ("gate_proj", 896, h),
                    ("up_proj", 896, h),
                    ("down_proj", h, 896),
                ] {
                    matrix(
                        &format!("{p}.mlp.experts.{expert}.{proj}.weight"),
                        rows,
                        cols,
                        false,
                    )?;
                }
            }
        }
    }
    if m.dropped_tensors.count != 0
        || !m
            .tensors
            .iter()
            .any(|t| t.role == NativeTensorRole::UnlimitedOcrProjector)
    {
        return Err("Unlimited-OCR-2 requires complete source tensor coverage and vision".into());
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct Ocr2PageState {
    pub coordinates: Vec<(usize, usize)>,
    pub image_pages: Vec<Option<usize>>,
    pub query_pages: Vec<Option<usize>>,
    prompt_len: usize,
    page_count: usize,
    prefill_counts: Vec<usize>,
    response_markers: usize,
    decode_page: usize,
    decode_tok: usize,
}

impl Ocr2PageState {
    pub fn prefill(ids: &[u32], cfg: &UnlimitedOcr2Config) -> Result<Self, String> {
        let last_image = ids
            .iter()
            .rposition(|id| *id == cfg.image_token_id)
            .ok_or("Unlimited-OCR-2 requires image tokens")?;
        let page_count = ids[..=last_image]
            .iter()
            .filter(|id| **id == cfg.page_token_id)
            .count();
        if page_count == 0 {
            return Err("Unlimited-OCR-2 requires <PAGE> before each image".into());
        }
        let mut out = Self {
            prompt_len: ids.len(),
            page_count,
            prefill_counts: vec![0; page_count + 1],
            ..Self::default()
        };
        let mut page: usize = 0;
        let mut counts = vec![0; page_count + 1];
        for (pos, id) in ids.iter().enumerate() {
            if *id == cfg.page_token_id {
                page += 1;
            }
            let bucket = if pos <= last_image {
                page
            } else {
                page.saturating_sub(page_count)
            };
            if bucket > page_count || (*id == cfg.image_token_id && bucket == 0) {
                return Err("invalid Unlimited-OCR-2 prompt page layout".into());
            }
            out.coordinates.push((counts[bucket], bucket));
            counts[bucket] += 1;
            if pos <= last_image {
                out.prefill_counts[bucket] += 1;
            }
            out.image_pages
                .push((*id == cfg.image_token_id).then_some(bucket - usize::from(bucket > 0)));
            out.query_pages.push(bucket.checked_sub(1));
        }
        (out.decode_tok, out.decode_page) = *out.coordinates.last().ok_or("empty prompt")?;
        Ok(out)
    }

    pub fn append(&mut self, ids: &[u32], cfg: &UnlimitedOcr2Config) {
        for id in ids {
            if *id == cfg.page_token_id {
                self.response_markers += 1;
            }
            // A marker past the last page continues that page: restarting its
            // token index would reuse 2D RoPE coordinates already emitted.
            if *id == cfg.page_token_id && self.decode_page < self.page_count {
                self.decode_page += 1;
                self.decode_tok = self.prefill_counts[self.decode_page];
            } else {
                self.decode_tok += 1;
            }
            self.coordinates.push((self.decode_tok, self.decode_page));
            self.image_pages.push(None);
            // Same mapping as prefill: response page `p` queries image page `p - 1`.
            self.query_pages.push(self.decode_page.checked_sub(1));
        }
    }

    pub fn trim(&mut self, len: usize, generated: &[u32], cfg: &UnlimitedOcr2Config) -> bool {
        if len < self.prompt_len || generated.len() != len - self.prompt_len {
            return false;
        }
        self.coordinates.truncate(self.prompt_len);
        self.image_pages.truncate(self.prompt_len);
        self.query_pages.truncate(self.prompt_len);
        self.response_markers = 0;
        if let Some(&(tok, page)) = self.coordinates.last() {
            self.decode_tok = tok;
            self.decode_page = page;
        }
        self.append(generated, cfg);
        true
    }

    /// Candidate domain before remote eligibility is applied. Blocks that
    /// cross page boundaries belong to every image page they contain.
    pub fn candidate_blocks(&self, pos: usize, block_size: usize) -> Vec<bool> {
        let query = self.query_pages[pos];
        let mut same: Vec<bool> = self
            .image_pages
            .chunks(block_size)
            .map(|block| query.is_some() && block.contains(&query))
            .collect();
        if !same.iter().any(|v| *v) {
            same.fill(true);
        }
        same
    }
}

/// 1.5-entmax defined by p_i = max(x_i/2 - tau, 0)^2 and sum(p)=1.
/// Bisection deliberately avoids depending on a framework sort kernel's
/// threshold implementation. Translation invariance improves conditioning.
pub fn entmax15(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let x: Vec<f64> = scores.iter().map(|s| (f64::from(*s) - max) * 0.5).collect();
    let (mut lo, mut hi) = (-1.0, 0.0);
    for _ in 0..48 {
        let mid = (lo + hi) * 0.5;
        let mass: f64 = x.iter().map(|v| (v - mid).max(0.0).powi(2)).sum();
        if mass > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    x.iter()
        .map(|v| (v - (lo + hi) * 0.5).max(0.0).powi(2) as f32)
        .collect()
}

pub fn route_blocks(
    scores: &[f32],
    allowed: &[bool],
    pos: usize,
    cfg: &UnlimitedOcr2Config,
    layer: usize,
) -> Vec<(usize, f32)> {
    let remote = |b: usize| b + cfg.local_blocks <= pos / cfg.block_size;
    let restrict = allowed.iter().enumerate().any(|(b, ok)| *ok && remote(b));
    let mut candidates: Vec<_> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(b, score)| remote(*b) && score.is_finite() && (!restrict || allowed[*b]))
        .collect();
    candidates.sort_by(|(a, x), (b, y)| y.total_cmp(x).then_with(|| a.cmp(b)));
    candidates.truncate(16);
    let values: Vec<_> = candidates
        .iter()
        .map(|(_, s)| s * cfg.scales[layer] / (cfg.index_dim as f32).sqrt())
        .collect();
    let prior = entmax15(&values);
    candidates.truncate(cfg.remote_blocks);
    let logp: Vec<_> = prior
        .iter()
        .take(candidates.len())
        .map(|p| p.max(cfg.floors[layer]).ln())
        .collect();
    let mean = logp.iter().sum::<f32>() / candidates.len().max(1) as f32;
    candidates
        .iter()
        .zip(logp)
        .map(|((block, _), p)| (*block, (p - mean) / cfg.sigma))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn config() -> UnlimitedOcr2Config {
        UnlimitedOcr2Config {
            block_size: 32,
            local_blocks: 4,
            remote_blocks: 12,
            index_dim: 128,
            image_token_id: 2,
            page_token_id: 3,
            scales: vec![128.0; 12],
            floors: vec![1e-4; 12],
            sigma: 1.0,
        }
    }

    #[test]
    fn source_expert_ids_are_numeric_and_suffixes_are_strict() {
        assert_eq!(
            source_expert("model.layers.11.mlp.experts.10.up_proj.weight"),
            Some((11, 10, "up_proj.weight"))
        );
        assert!(source_expert("model.layers.11.mlp.experts.10.up_proj.scales").is_none());
        assert!(source_expert("model.layers.11.mlp.experts.bad.up_proj.weight").is_none());
    }

    #[test]
    fn config_profiles_and_scopes_fail_closed() {
        let s = serde_json::json!({
            "use_sparse_attention": true, "training_stage": "dash", "dash_stage0": "head_cls",
            "dash_router": "entmax_topk_perlayer", "sparse_main_rope_mode": "page2d",
            "sparse_rope_mode": "page2d", "sparse_main_rope_tok_freq_mode": "resample",
            "sparse_rope_tok_freq_mode": "resample", "sparse_rope_tok_theta": 10000.0,
            "sparse_rope_page_theta": 256.0, "sparse_rope_page_pairs": 1,
            "sparse_rope_page_mod": 2, "sparse_rope_page_span": 0,
            "dash_image_only": true, "image_bucket_constraint": true,
            "sparse_attention_freq": vec![1;12], "sparse_block_size": 32, "sparse_local_block": 4,
            "dash_topk_blocks": 12, "sparse_index_dim": 128, "image_token_id": 2, "page_token_id": 3,
            "dash_scale_per_layer": vec![128.0;12], "dash_floor_per_layer": vec![1e-4;12], "dash_sigma": 1.0
        });
        let value = serde_json::json!({"sparse_attention_config": s});
        assert!(UnlimitedOcr2Config::from_hf(&value, 12).unwrap().is_some());
        let mut integral = value.clone();
        integral["sparse_attention_config"]["sparse_rope_tok_theta"] = serde_json::json!(10000);
        integral["sparse_attention_config"]["sparse_rope_page_pairs"] = serde_json::json!(1.0);
        integral["sparse_attention_config"]["sparse_attention_freq"] =
            serde_json::json!(vec![1.0; 12]);
        assert!(
            UnlimitedOcr2Config::from_hf(&integral, 12)
                .unwrap()
                .is_some()
        );
        let mut wrong = value.clone();
        wrong["sparse_attention_config"]["dash_stage0"] = serde_json::json!("indexer");
        assert!(UnlimitedOcr2Config::from_hf(&wrong, 12).is_err());
        let mut conflicting = value.clone();
        conflicting["language_config"] = wrong;
        assert!(UnlimitedOcr2Config::from_hf(&conflicting, 12).is_err());
        let mut truncated = config();
        truncated.scales.pop();
        assert!(truncated.validate(12).is_err());
        truncated = config();
        truncated.floors[0] = f32::NAN;
        assert!(truncated.validate(12).is_err());
    }

    #[test]
    fn entmax_known_distribution_and_translation() {
        assert_eq!(entmax15(&[0.0, 4.0]), vec![0.0, 1.0]);
        let uniform = entmax15(&[7.0; 4]);
        for p in uniform {
            assert!((p - 0.25).abs() < 1e-6);
        }
        let expected = entmax15(&[0.0, 1.0, 2.0]);
        let shifted = entmax15(&[100.0, 101.0, 102.0]);
        assert_eq!(expected, shifted);
        assert!((expected.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn entmax_matches_pinned_upstream_cpu_oracle() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/unlimited-ocr2-entmax.json"))
                .unwrap();
        assert_eq!(
            fixture["revision"],
            "9cf077c4b206e0e2a31879df937b0b0bcba3661c"
        );
        for case in fixture["cases"].as_array().unwrap() {
            let input: Vec<f32> = serde_json::from_value(case["input"].clone()).unwrap();
            let expected: Vec<f32> = serde_json::from_value(case["expected"].clone()).unwrap();
            let actual = entmax15(&input);
            for (a, e) in actual.iter().zip(expected) {
                assert!((a - e).abs() < 2e-6, "entmax oracle mismatch {a} vs {e}");
            }
        }
    }

    #[test]
    fn page_positions_resume_and_isolate() {
        let cfg = config();
        let mut state = Ocr2PageState::prefill(&[0, 3, 2, 2, 3, 2, 2, 8], &cfg).unwrap();
        assert_eq!(
            state.coordinates,
            [
                (0, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (1, 0)
            ]
        );
        state.append(&[3, 9, 3, 10], &cfg);
        assert_eq!(&state.coordinates[8..], [(3, 1), (4, 1), (3, 2), (4, 2)]);
        assert_eq!(
            &state.query_pages[8..],
            [Some(0), Some(0), Some(1), Some(1)]
        );
        // A marker past the last page continues it without reusing positions
        // and never queries a page that has no image.
        let mut extra = state.clone();
        extra.append(&[3, 11], &cfg);
        assert_eq!(&extra.coordinates[12..], [(5, 2), (6, 2)]);
        assert_eq!(&extra.query_pages[12..], [Some(1), Some(1)]);
        let cloned = state.clone();
        assert!(state.trim(10, &[3, 9], &cfg));
        assert_eq!(state.coordinates.len(), 10);
        assert_eq!(cloned.coordinates.len(), 12);
        assert!(!state.trim(1, &[], &cfg));
    }

    #[test]
    fn routing_is_causal_with_empty_candidates_and_ties() {
        let cfg = config();
        assert!(route_blocks(&[1.0; 20], &[true; 20], 127, &cfg, 0).is_empty());
        let selected = route_blocks(&[1.0; 20], &[false; 20], 512, &cfg, 0);
        assert_eq!(selected.len(), 12);
        assert_eq!(
            selected.iter().map(|(b, _)| *b).collect::<Vec<_>>(),
            (0..12).collect::<Vec<_>>()
        );
        assert!(selected.iter().all(|(_, bias)| bias.abs() < 1e-6));
        let mut same = vec![false; 20];
        same[2] = true;
        assert_eq!(
            route_blocks(&[1.0; 20], &same, 512, &cfg, 0),
            vec![(2, 0.0)]
        );
    }
}
