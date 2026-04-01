use std::collections::HashMap;
use std::path::Path;

use crate::gguf::MappedModel;
use crate::gguf::tensor::GgmlType;

use super::config::ModelConfig;

const ACTIVE_LAYER_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantFingerprintEntry {
    pub quant: String,
    pub tensors: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFingerprint {
    pub model_name: String,
    pub architecture: String,
    pub family: String,
    pub size_label: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub embedding_dim: u32,
    pub head_dim: u32,
    pub intermediate_dim: u32,
    pub context_length: u32,
    pub sliding_window_size: Option<u32>,
    pub sliding_window_pattern: Option<u32>,
    pub n_expert: Option<u32>,
    pub n_expert_used: Option<u32>,
    pub qwen35_full_attention_interval: Option<u32>,
    pub total_tensor_bytes: u64,
    pub predominant_quant: String,
    pub predominant_layer_quant: String,
    pub lm_head_quant: Option<String>,
    pub layer_quant_histogram: Vec<QuantFingerprintEntry>,
    pub has_mixed_layer_quants: bool,
    pub has_q4k_layer_weights: bool,
    pub has_q5k_layer_weights: bool,
    pub has_q6k_layer_weights: bool,
    pub has_q8_layer_weights: bool,
    pub has_f32_layer_weights: bool,
}

impl ModelFingerprint {
    pub fn from_mapped_model(
        model_path: Option<&Path>,
        mapped: &MappedModel,
        config: &ModelConfig,
    ) -> Self {
        let model_name = model_path
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .map(str::to_owned)
            .or_else(|| mapped.header.get_str("general.name").map(str::to_owned))
            .unwrap_or_else(|| config.architecture.clone());
        let family = normalized_family_for_arch(&config.architecture);
        let size_label = size_label_for_model_name(&model_name).unwrap_or_else(|| "unknown".into());

        let mut layer_quant_counts: HashMap<GgmlType, (usize, u64)> = HashMap::new();
        for tensor in &mapped.tensors {
            if !is_active_layer_weight_name(&tensor.name) {
                continue;
            }
            let entry = layer_quant_counts.entry(tensor.dtype).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += tensor.data_size();
        }

        let mut layer_quant_histogram = layer_quant_counts
            .iter()
            .map(|(dtype, (tensors, bytes))| QuantFingerprintEntry {
                quant: dtype.to_string(),
                tensors: *tensors,
                bytes: *bytes,
            })
            .collect::<Vec<_>>();
        layer_quant_histogram.sort_by(|left, right| left.quant.cmp(&right.quant));

        let predominant_layer_quant = layer_quant_counts
            .iter()
            .max_by_key(|(_, (_, bytes))| *bytes)
            .map(|(dtype, _)| dtype.to_string())
            .unwrap_or_else(|| "default".to_string());
        let predominant_quant = mapped
            .predominant_quant()
            .map(|dtype| dtype.to_string())
            .unwrap_or_else(|| "default".to_string());
        let lm_head_quant = lm_head_quant(mapped).map(|dtype| dtype.to_string());

        Self {
            model_name,
            architecture: config.architecture.clone(),
            family,
            size_label,
            n_layers: config.n_layers,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            embedding_dim: config.embedding_dim,
            head_dim: config.head_dim,
            intermediate_dim: config.intermediate_dim,
            context_length: config.context_length,
            sliding_window_size: config.sliding_window_size,
            sliding_window_pattern: config.sliding_window_pattern,
            n_expert: config.n_expert,
            n_expert_used: config.n_expert_used,
            qwen35_full_attention_interval: config.qwen35_full_attention_interval,
            total_tensor_bytes: mapped.total_tensor_bytes(),
            predominant_quant,
            predominant_layer_quant,
            lm_head_quant,
            has_mixed_layer_quants: layer_quant_histogram.len() > 1,
            has_q4k_layer_weights: layer_quant_counts.contains_key(&GgmlType::Q4K),
            has_q5k_layer_weights: layer_quant_counts.contains_key(&GgmlType::Q5K),
            has_q6k_layer_weights: layer_quant_counts.contains_key(&GgmlType::Q6K),
            has_q8_layer_weights: layer_quant_counts.contains_key(&GgmlType::Q8_0),
            has_f32_layer_weights: layer_quant_counts.contains_key(&GgmlType::F32),
            layer_quant_histogram,
        }
    }

    pub fn cache_namespace(&self) -> String {
        if self.size_label == "unknown" {
            self.family.clone()
        } else {
            format!("{}-{}", self.family, self.size_label)
        }
    }

    pub fn canonical_descriptor(&self) -> String {
        let quant_histogram = self
            .layer_quant_histogram
            .iter()
            .map(|entry| format!("{}:{}:{}", entry.quant, entry.tensors, entry.bytes))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "model={}|arch={}|family={}|size={}|shape=l{}-h{}-kv{}-d{}-hd{}-ffn{}-ctx{}",
                "|sw={:?}|swp={:?}|experts={:?}|experts_used={:?}|full_attn={:?}",
                "|total={}|pred={}|layer_pred={}|lm={:?}|mixed={}",
                "|q4k={}|q5k={}|q6k={}|q8={}|f32={}|hist={}"
            ),
            self.model_name,
            self.architecture,
            self.family,
            self.size_label,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.embedding_dim,
            self.head_dim,
            self.intermediate_dim,
            self.context_length,
            self.sliding_window_size,
            self.sliding_window_pattern,
            self.n_expert,
            self.n_expert_used,
            self.qwen35_full_attention_interval,
            self.total_tensor_bytes,
            self.predominant_quant,
            self.predominant_layer_quant,
            self.lm_head_quant,
            self.has_mixed_layer_quants,
            self.has_q4k_layer_weights,
            self.has_q5k_layer_weights,
            self.has_q6k_layer_weights,
            self.has_q8_layer_weights,
            self.has_f32_layer_weights,
            quant_histogram,
        )
    }

    pub fn stable_id(&self) -> String {
        format!(
            "{:016x}",
            stable_hash64(self.canonical_descriptor().as_bytes())
        )
    }
}

fn is_active_layer_weight_name(name: &str) -> bool {
    name.starts_with("blk.")
        && ACTIVE_LAYER_SUFFIXES
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

fn lm_head_quant(mapped: &MappedModel) -> Option<GgmlType> {
    if let Some(info) = mapped.tensor_info("output.weight") {
        return Some(info.dtype);
    }
    mapped
        .tensor_info("token_embd.weight")
        .map(|info| info.dtype)
}

fn normalized_family_for_arch(arch: &str) -> String {
    match arch {
        "llama" => "llama3".to_string(),
        "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe" => "qwen3".to_string(),
        "qwen35" | "qwen35moe" => "qwen35".to_string(),
        "gemma" | "gemma2" | "gemma3" => "gemma3".to_string(),
        _ => arch.to_string(),
    }
}

fn size_label_for_model_name(model_name: &str) -> Option<String> {
    let size = extract_param_billions(&model_name.to_ascii_lowercase())?;
    let rounded = if approx_size(size, 4.0) {
        "4b".to_string()
    } else if approx_size(size, 6.0)
        || approx_size(size, 7.0)
        || approx_size(size, 8.0)
        || approx_size(size, 9.0)
    {
        "8b".to_string()
    } else if approx_size(size, 12.0) {
        "12b".to_string()
    } else if approx_size(size, 14.0) {
        "14b".to_string()
    } else if approx_size(size, 27.0) {
        "27b".to_string()
    } else if approx_size(size, 30.0) {
        "30b".to_string()
    } else if approx_size(size, 32.0) {
        "32b".to_string()
    } else if approx_size(size, 35.0) {
        "35b".to_string()
    } else if approx_size(size, 70.0) || approx_size(size, 72.0) {
        "70b".to_string()
    } else if approx_size(size, 397.0) {
        "397b".to_string()
    } else if (size - size.round()).abs() < 0.05 {
        format!("{}b", size.round() as i32)
    } else {
        format!("{size:.1}b")
    };
    Some(rounded)
}

fn approx_size(size: f32, target: f32) -> bool {
    (size - target).abs() < 0.11
}

fn extract_param_billions(name: &str) -> Option<f32> {
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            if i < bytes.len() && (bytes[i] == b'b' || bytes[i] == b'B') {
                let after_b = i + 1;
                let boundary = after_b >= bytes.len() || !bytes[after_b].is_ascii_alphabetic();
                if boundary && let Ok(value) = name[start..i].parse::<f32>() {
                    return Some(value);
                }
            }
        }
        i += 1;
    }
    None
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint_with_histogram(histogram: Vec<QuantFingerprintEntry>) -> ModelFingerprint {
        ModelFingerprint {
            model_name: "Qwen3-8B".to_string(),
            architecture: "qwen3".to_string(),
            family: "qwen3".to_string(),
            size_label: "8b".to_string(),
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            embedding_dim: 4096,
            head_dim: 128,
            intermediate_dim: 14336,
            context_length: 32768,
            sliding_window_size: None,
            sliding_window_pattern: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            total_tensor_bytes: 4_000_000_000,
            predominant_quant: "Q4_K".to_string(),
            predominant_layer_quant: "Q4_K".to_string(),
            lm_head_quant: Some("Q4_K".to_string()),
            has_mixed_layer_quants: histogram.len() > 1,
            has_q4k_layer_weights: true,
            has_q5k_layer_weights: false,
            has_q6k_layer_weights: false,
            has_q8_layer_weights: false,
            has_f32_layer_weights: false,
            layer_quant_histogram: histogram,
        }
    }

    #[test]
    fn test_size_label_for_model_name() {
        assert_eq!(
            size_label_for_model_name("Meta-Llama-3-8B"),
            Some("8b".into())
        );
        assert_eq!(
            size_label_for_model_name("Qwen3.5-27B-Q4_K_M"),
            Some("27b".into())
        );
        assert_eq!(size_label_for_model_name("some-unlabeled-model"), None);
    }

    #[test]
    fn test_normalized_family_for_arch() {
        assert_eq!(normalized_family_for_arch("llama"), "llama3");
        assert_eq!(normalized_family_for_arch("qwen35moe"), "qwen35");
        assert_eq!(normalized_family_for_arch("gemma2"), "gemma3");
    }

    #[test]
    fn test_stable_id_changes_with_quant_histogram() {
        let q4k = fingerprint_with_histogram(vec![QuantFingerprintEntry {
            quant: "Q4_K".to_string(),
            tensors: 224,
            bytes: 1000,
        }]);
        let mixed = fingerprint_with_histogram(vec![
            QuantFingerprintEntry {
                quant: "Q4_K".to_string(),
                tensors: 112,
                bytes: 500,
            },
            QuantFingerprintEntry {
                quant: "Q6_K".to_string(),
                tensors: 112,
                bytes: 600,
            },
        ]);
        assert_ne!(q4k.stable_id(), mixed.stable_id());
    }
}
