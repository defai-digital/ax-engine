use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION: &str = "ax.native_model.v1";
pub const AX_NATIVE_MODEL_MANIFEST_FILE: &str = "model-manifest.json";
pub const QWEN3_5_DEFAULT_FULL_ATTENTION_INTERVAL: u32 = 4;
const SUPPORTED_MLX_AFFINE_QUANTIZATION_BITS: &[u32] = &[4, 5, 6, 8];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorFormat {
    Safetensors,
    /// GGUF file, loaded directly without conversion.
    Gguf,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorDataType {
    F16,
    Bf16,
    F32,
    I8,
    U8,
    /// Packed uint32 — used by MLX affine quantization for the weight tensor.
    /// Bit width and group size are carried by per-tensor quantization metadata.
    /// Scales and biases are stored as separate bf16/f32 tensors with the same base name.
    U32,
    /// Q4_K_M quantized: 256-element super-blocks, 144 bytes each (4.5 bits/weight).
    /// Raw block_q4_K bytes stored directly in the Metal buffer; dequant happens in kernel.
    Q4Km,
    /// Q5_K quantized: 256-element super-blocks, 176 bytes each. Dequantized to F16 at load.
    Q5Km,
    /// Q6_K quantized: 256-element super-blocks, 210 bytes each. Dequantized to F16 at load.
    Q6Km,
    /// Q8_0 quantized: 32-element blocks, 34 bytes each. Dequantized to F16 at load.
    Q8Zero,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorRole {
    TokenEmbedding,
    AttentionNorm,
    AttentionPostNorm,
    AttentionQNorm,
    AttentionKNorm,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionQkvPacked,
    AttentionQa,
    AttentionQaNorm,
    AttentionQb,
    AttentionKvA,
    AttentionKvANorm,
    AttentionEmbedQ,
    AttentionUnembedOut,
    AttentionO,
    LinearAttentionInProjQkv,
    LinearAttentionInProjQkvz,
    LinearAttentionInProjZ,
    LinearAttentionInProjA,
    LinearAttentionInProjB,
    LinearAttentionInProjBa,
    LinearAttentionConv1d,
    LinearAttentionDtBias,
    LinearAttentionALog,
    LinearAttentionNorm,
    LinearAttentionOutProj,
    FfnNorm,
    FfnNorm2,
    FfnPostNorm,
    FfnPostNorm1,
    FfnPostNorm2,
    FfnGateInp,
    FfnGateInpScale,
    FfnGateInpExpertScale,
    FfnGateInpCorrectionBias,
    FfnGate,
    FfnUp,
    FfnGateUpPacked,
    FfnSharedExpertGateInp,
    FfnSharedExpertGate,
    FfnSharedExpertUp,
    FfnSharedExpertDown,
    FfnGateExps,
    FfnUpExps,
    FfnGateUpExpsPacked,
    FfnDown,
    FfnDownExps,
    FfnDownExpsScale,
    LayerScalar,
    /// Global embedding table for per-layer token inputs (Gemma4 2B/4B).
    PerLayerEmbedding,
    /// Global projection from hidden state to stacked per-layer inputs (Gemma4 2B/4B).
    PerLayerModelProjection,
    /// Global RMSNorm weight over hidden_size_per_layer_input (Gemma4 2B/4B).
    PerLayerProjectionNorm,
    /// Per-layer gate projection: hidden → hidden_size_per_layer_input (Gemma4 2B/4B).
    PerLayerInputGate,
    /// Per-layer output projection: hidden_size_per_layer_input → hidden (Gemma4 2B/4B).
    PerLayerInputProjection,
    /// Per-layer post-gating RMSNorm weight (Gemma4 2B/4B).
    PerLayerInputPostNorm,
    FinalNorm,
    LmHead,
    RopeFreqs,
    Other,
}

impl NativeTensorRole {
    const fn requires_layer_index(self) -> bool {
        matches!(
            self,
            Self::AttentionNorm
                | Self::AttentionPostNorm
                | Self::AttentionQNorm
                | Self::AttentionKNorm
                | Self::AttentionQ
                | Self::AttentionK
                | Self::AttentionV
                | Self::AttentionQkvPacked
                | Self::AttentionQa
                | Self::AttentionQaNorm
                | Self::AttentionQb
                | Self::AttentionKvA
                | Self::AttentionKvANorm
                | Self::AttentionEmbedQ
                | Self::AttentionUnembedOut
                | Self::AttentionO
                | Self::LinearAttentionInProjQkv
                | Self::LinearAttentionInProjQkvz
                | Self::LinearAttentionInProjZ
                | Self::LinearAttentionInProjA
                | Self::LinearAttentionInProjB
                | Self::LinearAttentionInProjBa
                | Self::LinearAttentionConv1d
                | Self::LinearAttentionDtBias
                | Self::LinearAttentionALog
                | Self::LinearAttentionNorm
                | Self::LinearAttentionOutProj
                | Self::FfnNorm
                | Self::FfnNorm2
                | Self::FfnPostNorm
                | Self::FfnPostNorm1
                | Self::FfnPostNorm2
                | Self::FfnGateInp
                | Self::FfnGateInpScale
                | Self::FfnGateInpExpertScale
                | Self::FfnGateInpCorrectionBias
                | Self::FfnGate
                | Self::FfnUp
                | Self::FfnGateUpPacked
                | Self::FfnSharedExpertGateInp
                | Self::FfnSharedExpertGate
                | Self::FfnSharedExpertUp
                | Self::FfnSharedExpertDown
                | Self::FfnGateExps
                | Self::FfnUpExps
                | Self::FfnGateUpExpsPacked
                | Self::FfnDown
                | Self::FfnDownExps
                | Self::FfnDownExpsScale
                | Self::LayerScalar
                | Self::PerLayerInputGate
                | Self::PerLayerInputProjection
                | Self::PerLayerInputPostNorm
        )
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeLinearAttentionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_attention_interval: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_value_heads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_key_heads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_kernel_dim: Option<u32>,
}

impl NativeLinearAttentionConfig {
    pub fn is_enabled(&self) -> bool {
        self.full_attention_interval.is_some()
            || self.num_value_heads.is_some()
            || self.num_key_heads.is_some()
            || self.key_head_dim.is_some()
            || self.value_head_dim.is_some()
            || self.conv_kernel_dim.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }

    pub fn resolved_full_attention_interval(&self, model_family: &str) -> Option<u32> {
        self.full_attention_interval.or_else(|| {
            let is_hybrid_family = matches!(model_family, "qwen3_5" | "qwen3_next");
            (self.is_enabled() && is_hybrid_family)
                .then_some(QWEN3_5_DEFAULT_FULL_ATTENTION_INTERVAL)
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NativeMlaAttentionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_lora_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_lora_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qk_nope_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qk_rope_head_dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_head_dim: Option<u32>,
}

impl NativeMlaAttentionConfig {
    pub fn is_enabled(&self) -> bool {
        self.q_lora_rank.is_some()
            || self.kv_lora_rank.is_some()
            || self.qk_nope_head_dim.is_some()
            || self.qk_rope_head_dim.is_some()
            || self.value_head_dim.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeMoeConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts_per_token: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_intermediate_size: Option<u32>,
}

impl NativeMoeConfig {
    pub fn is_enabled(&self) -> bool {
        self.expert_count.is_some()
            || self.experts_per_token.is_some()
            || self.expert_intermediate_size.is_some()
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NativeGlmRouterConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_dense_layer_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routed_scaling_factor: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_group: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topk_group: Option<u32>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub has_shared_experts: bool,
}

impl NativeGlmRouterConfig {
    pub fn is_enabled(&self) -> bool {
        self.first_dense_layer_count.is_some()
            || self.routed_scaling_factor.is_some()
            || self.n_group.is_some()
            || self.topk_group.is_some()
            || self.has_shared_experts
    }

    pub fn is_disabled(&self) -> bool {
        !self.is_enabled()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeTensorSpec {
    pub name: String,
    pub role: NativeTensorRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<u32>,
    pub dtype: NativeTensorDataType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_tensor_type: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub source_quantized: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<NativeTensorQuantization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantized_source: Option<NativeQuantizedTensorSource>,
    pub shape: Vec<u64>,
    pub file: PathBuf,
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeTensorQuantization {
    pub mode: String,
    pub group_size: u32,
    pub bits: u32,
}

impl Default for NativeTensorQuantization {
    fn default() -> Self {
        Self {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 4,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeQuantizedTensorSource {
    pub format: String,
    pub file: PathBuf,
    #[serde(default)]
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeSourceQuantization {
    pub format: String,
    #[serde(default)]
    pub tensor_type_counts: BTreeMap<String, u32>,
    #[serde(default)]
    pub quantized_tensor_count: u32,
    #[serde(default)]
    pub contains_quantized_tensors: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeRuntimeStatus {
    #[serde(default = "default_runtime_ready")]
    pub ready: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl Default for NativeRuntimeStatus {
    fn default() -> Self {
        Self {
            ready: true,
            blockers: Vec::new(),
            notes: Vec::new(),
        }
    }
}

impl NativeRuntimeStatus {
    pub fn ready_without_details(&self) -> bool {
        self.ready && self.blockers.is_empty() && self.notes.is_empty()
    }
}

fn default_runtime_ready() -> bool {
    true
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NativeModelManifest {
    pub schema_version: String,
    pub model_family: String,
    pub tensor_format: NativeTensorFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_quantization: Option<NativeSourceQuantization>,
    #[serde(
        default,
        skip_serializing_if = "NativeRuntimeStatus::ready_without_details"
    )]
    pub runtime_status: NativeRuntimeStatus,
    pub layer_count: u32,
    pub hidden_size: u32,
    #[serde(default)]
    pub intermediate_size: u32,
    pub attention_head_count: u32,
    pub attention_head_dim: u32,
    pub kv_head_count: u32,
    pub vocab_size: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<u32>,
    /// Rope theta for sliding-window attention (SWA) layers in ISWA models (e.g. Gemma4).
    /// Corresponds to GGUF key `{arch}.rope.freq_base_swa`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta_swa: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_logit_softcap: Option<u32>,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attention_value_from_key_layers: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attention_v_norm_no_scale_layers: Vec<u32>,
    /// Head dimension for full-attention layers in interleaved SWA models (e.g. Gemma4).
    /// Sliding-attention layers use `attention_head_dim`; full-attention layers use this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<u32>,
    /// Sliding-window size for SWA layers (None = global attention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_size: Option<u32>,
    /// Per-layer type annotations ("sliding_attention" / "full_attention").
    /// Empty for homogeneous models; populated for interleaved-SWA models (Gemma4).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layer_types: Vec<String>,
    /// Maps KV-shared layer index → source layer index that supplies K/V.
    /// For layers absent from this map, K/V is computed from the layer's own weights.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub kv_shared_source_layers: BTreeMap<u32, u32>,
    /// Final-logit softcapping: apply `tanh(x / cap) * cap` after lm_head (Gemma4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f32>,
    /// Scale applied to token embeddings before the first layer (Gemma4: sqrt(hidden_size)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_states_scale: Option<f32>,
    /// When true, normalise the selected top-k MoE weights to sum to 1 (Qwen3 MoE).
    #[serde(default)]
    pub moe_norm_topk_prob: bool,
    /// Dimension of per-layer token embeddings (Gemma4 2B/4B). 0 = feature disabled.
    #[serde(default)]
    pub hidden_size_per_layer_input: u32,
    /// Vocab size for the per-layer embedding table (Gemma4 2B/4B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size_per_layer_input: Option<u32>,
    #[serde(
        default,
        skip_serializing_if = "NativeLinearAttentionConfig::is_disabled"
    )]
    pub linear_attention: NativeLinearAttentionConfig,
    #[serde(default, skip_serializing_if = "NativeMlaAttentionConfig::is_disabled")]
    pub mla_attention: NativeMlaAttentionConfig,
    #[serde(default, skip_serializing_if = "NativeMoeConfig::is_disabled")]
    pub moe: NativeMoeConfig,
    #[serde(default, skip_serializing_if = "NativeGlmRouterConfig::is_disabled")]
    pub glm_router: NativeGlmRouterConfig,
    pub tensors: Vec<NativeTensorSpec>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeModelArtifacts {
    root_dir: PathBuf,
    manifest: NativeModelManifest,
}

impl NativeModelArtifacts {
    /// Build artifacts directly from a pre-parsed manifest and root directory.
    /// Used by the GGUF loader to bypass the JSON manifest file.
    pub fn from_manifest_and_root(
        root_dir: PathBuf,
        manifest: NativeModelManifest,
    ) -> Result<Self, NativeModelError> {
        validate_native_model_manifest(&root_dir, &manifest)?;
        Ok(Self { root_dir, manifest })
    }

    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, NativeModelError> {
        let root_dir = path.as_ref().to_path_buf();
        let manifest_path = root_dir.join(AX_NATIVE_MODEL_MANIFEST_FILE);
        let bytes = fs::read(&manifest_path).map_err(|source| NativeModelError::ReadManifest {
            path: manifest_path.clone(),
            source,
        })?;
        let manifest = serde_json::from_slice::<NativeModelManifest>(&bytes).map_err(|source| {
            NativeModelError::ParseManifest {
                path: manifest_path.clone(),
                source,
            }
        })?;

        validate_native_model_manifest(&root_dir, &manifest)?;

        Ok(Self { root_dir, manifest })
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn manifest(&self) -> &NativeModelManifest {
        &self.manifest
    }

    pub fn tensor_specs(&self) -> &[NativeTensorSpec] {
        &self.manifest.tensors
    }

    pub fn global_tensor(&self, role: NativeTensorRole) -> Option<&NativeTensorSpec> {
        self.manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == role && tensor.layer_index.is_none())
    }

    pub fn layer_tensor(
        &self,
        layer_index: u32,
        role: NativeTensorRole,
    ) -> Option<&NativeTensorSpec> {
        self.manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == role && tensor.layer_index == Some(layer_index))
    }

    pub fn resolve_tensor_path(&self, tensor: &NativeTensorSpec) -> PathBuf {
        self.root_dir.join(&tensor.file)
    }

    pub fn summary(&self) -> NativeModelArtifactsSummary {
        let is_hybrid_attention = self.manifest.linear_attention.is_enabled();
        NativeModelArtifactsSummary {
            model_family: self.manifest.model_family.clone(),
            tensor_format: self.manifest.tensor_format,
            source_quantization: self.manifest.source_quantization.clone(),
            runtime_status: self.manifest.runtime_status.clone(),
            layer_count: self.manifest.layer_count,
            tensor_count: self.manifest.tensors.len() as u32,
            tie_word_embeddings: self.manifest.tie_word_embeddings,
            is_moe: self.manifest.moe.is_enabled(),
            is_hybrid_attention,
            hybrid_full_attention_interval: is_hybrid_attention
                .then(|| {
                    self.manifest
                        .linear_attention
                        .resolved_full_attention_interval(&self.manifest.model_family)
                })
                .flatten(),
        }
    }

    pub fn layer_uses_attention_value_from_key(&self, layer_index: u32) -> bool {
        self.manifest
            .attention_value_from_key_layers
            .contains(&layer_index)
    }

    pub fn layer_uses_attention_v_norm_no_scale(&self, layer_index: u32) -> bool {
        self.manifest
            .attention_v_norm_no_scale_layers
            .contains(&layer_index)
    }

    pub fn linear_attention_config(&self) -> Option<&NativeLinearAttentionConfig> {
        self.manifest
            .linear_attention
            .is_enabled()
            .then_some(&self.manifest.linear_attention)
    }

    pub fn moe_config(&self) -> Option<&NativeMoeConfig> {
        self.manifest.moe.is_enabled().then_some(&self.manifest.moe)
    }

    /// Returns the number of head dimensions that receive rotary embedding.
    /// When `partial_rotary_factor` is set, only a fraction of head_dim is rotated.
    pub fn rotary_dim(&self) -> usize {
        let head_dim = self.manifest.attention_head_dim as usize;
        if let Some(factor) = self.manifest.partial_rotary_factor {
            let dim = (head_dim as f32 * factor) as usize;
            // Rotary dim must be even; round down to nearest even
            dim & !1
        } else {
            head_dim
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeModelArtifactsSummary {
    pub model_family: String,
    pub tensor_format: NativeTensorFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_quantization: Option<NativeSourceQuantization>,
    #[serde(
        default,
        skip_serializing_if = "NativeRuntimeStatus::ready_without_details"
    )]
    pub runtime_status: NativeRuntimeStatus,
    pub layer_count: u32,
    pub tensor_count: u32,
    pub tie_word_embeddings: bool,
    /// True when the model uses a mixture-of-experts FFN (e.g. Gemma 4, Qwen3-MoE).
    #[serde(default)]
    pub is_moe: bool,
    /// True when the model interleaves linear-attention layers with standard attention
    /// (e.g. Qwen3.5, Qwen3-Next).
    #[serde(default)]
    pub is_hybrid_attention: bool,
    /// For hybrid-attention models: how many layers apart the full-attention layers occur.
    /// None for pure-attention or pure-linear-attention models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid_full_attention_interval: Option<u32>,
}

#[derive(Debug, Error)]
pub enum NativeModelError {
    #[error("failed to read native model manifest {path}: {source}")]
    ReadManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse native model manifest {path}: {source}")]
    ParseManifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid native model manifest: {message}")]
    InvalidManifest { message: String },
}

fn validate_native_model_manifest(
    root_dir: &Path,
    manifest: &NativeModelManifest,
) -> Result<(), NativeModelError> {
    if manifest.schema_version != AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "schema_version must be {}, got {}",
                AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, manifest.schema_version
            ),
        });
    }
    if manifest.model_family.trim().is_empty() {
        return Err(NativeModelError::InvalidManifest {
            message: "model_family must not be empty".to_string(),
        });
    }
    if !manifest.runtime_status.ready || !manifest.runtime_status.blockers.is_empty() {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "native model manifest is not runtime ready: ready={} blockers={:?}",
                manifest.runtime_status.ready, manifest.runtime_status.blockers
            ),
        });
    }
    if manifest.layer_count == 0
        || manifest.hidden_size == 0
        || manifest.attention_head_count == 0
        || manifest.attention_head_dim == 0
        || manifest.kv_head_count == 0
        || manifest.vocab_size == 0
    {
        return Err(NativeModelError::InvalidManifest {
            message: "layer_count, hidden_size, attention_head_count, attention_head_dim, kv_head_count, and vocab_size must be greater than zero".to_string(),
        });
    }
    if manifest.tensors.is_empty() {
        return Err(NativeModelError::InvalidManifest {
            message: "tensors must not be empty".to_string(),
        });
    }
    validate_manifest_layer_index_list(
        manifest,
        &manifest.attention_value_from_key_layers,
        "attention_value_from_key_layers",
    )?;
    validate_manifest_layer_index_list(
        manifest,
        &manifest.attention_v_norm_no_scale_layers,
        "attention_v_norm_no_scale_layers",
    )?;
    validate_interleaved_attention_metadata(manifest)?;
    if let Some(rope_theta) = manifest.rope_theta {
        if rope_theta == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("rope_theta must be > 0, got {rope_theta}"),
            });
        }
    }
    if let Some(query_pre_attn_scalar) = manifest.query_pre_attn_scalar {
        if query_pre_attn_scalar == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("query_pre_attn_scalar must be > 0, got {query_pre_attn_scalar}"),
            });
        }
    }
    if let Some(attention_logit_softcap) = manifest.attention_logit_softcap {
        if attention_logit_softcap == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "attention_logit_softcap must be > 0, got {attention_logit_softcap}"
                ),
            });
        }
    }
    if let Some(factor) = manifest.partial_rotary_factor {
        if factor <= 0.0 || factor > 1.0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("partial_rotary_factor must be in (0.0, 1.0], got {factor}"),
            });
        }
        let rotary_dim = (manifest.attention_head_dim as f32 * factor) as u32;
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "partial_rotary_factor {factor} yields rotary_dim {rotary_dim} which must be even and > 0"
                ),
            });
        }
    }
    if manifest.linear_attention.is_enabled() {
        require_positive_field(
            manifest
                .linear_attention
                .resolved_full_attention_interval(&manifest.model_family),
            "linear_attention.full_attention_interval",
        )?;
        require_positive_field(
            manifest.linear_attention.num_value_heads,
            "linear_attention.num_value_heads",
        )?;
        require_positive_field(
            manifest.linear_attention.num_key_heads,
            "linear_attention.num_key_heads",
        )?;
        require_positive_field(
            manifest.linear_attention.key_head_dim,
            "linear_attention.key_head_dim",
        )?;
        require_positive_field(
            manifest.linear_attention.value_head_dim,
            "linear_attention.value_head_dim",
        )?;
        require_positive_field(
            manifest.linear_attention.conv_kernel_dim,
            "linear_attention.conv_kernel_dim",
        )?;
        if let (Some(num_value_heads), Some(num_key_heads)) = (
            manifest.linear_attention.num_value_heads,
            manifest.linear_attention.num_key_heads,
        ) {
            if !num_value_heads.is_multiple_of(num_key_heads) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "linear_attention.num_value_heads {} must be divisible by linear_attention.num_key_heads {}",
                        num_value_heads, num_key_heads
                    ),
                });
            }
        }
    }
    if manifest.mla_attention.is_enabled() {
        require_positive_field(
            manifest.mla_attention.q_lora_rank,
            "mla_attention.q_lora_rank",
        )?;
        require_positive_field(
            manifest.mla_attention.kv_lora_rank,
            "mla_attention.kv_lora_rank",
        )?;
        require_positive_field(
            manifest.mla_attention.qk_nope_head_dim,
            "mla_attention.qk_nope_head_dim",
        )?;
        require_positive_field(
            manifest.mla_attention.qk_rope_head_dim,
            "mla_attention.qk_rope_head_dim",
        )?;
        require_positive_field(
            manifest.mla_attention.value_head_dim,
            "mla_attention.value_head_dim",
        )?;
        if let (Some(nope_dim), Some(rope_dim)) = (
            manifest.mla_attention.qk_nope_head_dim,
            manifest.mla_attention.qk_rope_head_dim,
        ) {
            let expected_head_dim = nope_dim.saturating_add(rope_dim);
            if expected_head_dim != manifest.attention_head_dim {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "mla_attention qk_nope_head_dim + qk_rope_head_dim must equal attention_head_dim {}, got {} + {}",
                        manifest.attention_head_dim, nope_dim, rope_dim
                    ),
                });
            }
        }
    }
    if manifest.moe.is_enabled() {
        require_positive_field(manifest.moe.expert_count, "moe.expert_count")?;
        require_positive_field(manifest.moe.experts_per_token, "moe.experts_per_token")?;
        require_positive_field(
            manifest.moe.expert_intermediate_size,
            "moe.expert_intermediate_size",
        )?;
        if let (Some(expert_count), Some(experts_per_token)) =
            (manifest.moe.expert_count, manifest.moe.experts_per_token)
        {
            if experts_per_token > expert_count {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "moe.experts_per_token {} must be <= moe.expert_count {}",
                        experts_per_token, expert_count
                    ),
                });
            }
        }
    }
    if manifest.glm_router.is_enabled() {
        if manifest.glm_router.first_dense_layer_count.is_none() {
            return Err(NativeModelError::InvalidManifest {
                message: "glm_router.first_dense_layer_count must be configured".to_string(),
            });
        }
        if manifest
            .glm_router
            .first_dense_layer_count
            .is_some_and(|count| count > manifest.layer_count)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "glm_router.first_dense_layer_count must be <= layer_count {}, got {}",
                    manifest.layer_count,
                    manifest
                        .glm_router
                        .first_dense_layer_count
                        .unwrap_or_default()
                ),
            });
        }
        match manifest.glm_router.routed_scaling_factor {
            Some(value) if value.is_finite() && value > 0.0 => {}
            Some(value) => {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "glm_router.routed_scaling_factor must be finite and > 0, got {value}"
                    ),
                });
            }
            None => {
                return Err(NativeModelError::InvalidManifest {
                    message: "glm_router.routed_scaling_factor must be configured".to_string(),
                });
            }
        }
        require_positive_field(manifest.glm_router.n_group, "glm_router.n_group")?;
        require_positive_field(manifest.glm_router.topk_group, "glm_router.topk_group")?;
        if let (Some(n_group), Some(topk_group)) =
            (manifest.glm_router.n_group, manifest.glm_router.topk_group)
        {
            if topk_group > n_group {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "glm_router.topk_group {} must be <= glm_router.n_group {}",
                        topk_group, n_group
                    ),
                });
            }
            if manifest
                .moe
                .expert_count
                .is_some_and(|expert_count| expert_count % n_group != 0)
            {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "moe.expert_count must divide evenly across glm_router.n_group {n_group}"
                    ),
                });
            }
        }
    }

    let mut tensor_names = BTreeMap::new();
    let mut layer_roles = BTreeMap::<u32, Vec<NativeTensorRole>>::new();
    let mut global_roles = Vec::new();

    for tensor in &manifest.tensors {
        if tensor.name.trim().is_empty() {
            return Err(NativeModelError::InvalidManifest {
                message: "tensor name must not be empty".to_string(),
            });
        }
        if tensor_names.insert(tensor.name.clone(), ()).is_some() {
            return Err(NativeModelError::InvalidManifest {
                message: format!("duplicate tensor name {}", tensor.name),
            });
        }
        if tensor.shape.is_empty() || tensor.shape.contains(&0) {
            return Err(NativeModelError::InvalidManifest {
                message: format!("tensor {} must have only positive dimensions", tensor.name),
            });
        }
        if tensor.length_bytes == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("tensor {} must have positive length_bytes", tensor.name),
            });
        }
        validate_tensor_path(root_dir, tensor)?;
        validate_quantized_source_path(root_dir, tensor)?;
        validate_tensor_quantization(tensor)?;

        if tensor.role.requires_layer_index() {
            let Some(layer_index) = tensor.layer_index else {
                return Err(NativeModelError::InvalidManifest {
                    message: format!("tensor {} requires layer_index", tensor.name),
                });
            };
            if layer_index >= manifest.layer_count {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "tensor {} layer_index {} exceeds layer_count {}",
                        tensor.name, layer_index, manifest.layer_count
                    ),
                });
            }
            layer_roles
                .entry(layer_index)
                .or_default()
                .push(tensor.role);
        } else {
            if tensor.layer_index.is_some() {
                return Err(NativeModelError::InvalidManifest {
                    message: format!("tensor {} must not declare layer_index", tensor.name),
                });
            }
            global_roles.push(tensor.role);
        }
    }

    require_global_role(
        &global_roles,
        NativeTensorRole::TokenEmbedding,
        "token_embedding",
    )?;
    require_global_role(&global_roles, NativeTensorRole::FinalNorm, "final_norm")?;
    if !manifest.tie_word_embeddings {
        require_global_role(&global_roles, NativeTensorRole::LmHead, "lm_head")?;
    }

    for layer_index in 0..manifest.layer_count {
        let roles =
            layer_roles
                .get(&layer_index)
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!("missing tensors for layer {}", layer_index),
                })?;
        require_layer_role(
            roles,
            NativeTensorRole::AttentionNorm,
            layer_index,
            "attention_norm",
        )?;
        // ffn_norm is optional when attention_post_norm serves as the FFN norm
        // (e.g. Qwen3.5 linear attention layers).
        if !roles.contains(&NativeTensorRole::FfnNorm)
            && !roles.contains(&NativeTensorRole::AttentionPostNorm)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} is missing required tensor role ffn_norm or attention_post_norm",
                    layer_index
                ),
            });
        }
        let has_packed_gate_up = roles.contains(&NativeTensorRole::FfnGateUpPacked);
        let has_split_gate_up =
            roles.contains(&NativeTensorRole::FfnGate) && roles.contains(&NativeTensorRole::FfnUp);
        let has_dense_ffn =
            roles.contains(&NativeTensorRole::FfnDown) && (has_packed_gate_up || has_split_gate_up);
        let has_shared_expert_ffn = roles.contains(&NativeTensorRole::FfnSharedExpertGateInp)
            && roles.contains(&NativeTensorRole::FfnSharedExpertGate)
            && roles.contains(&NativeTensorRole::FfnSharedExpertUp)
            && roles.contains(&NativeTensorRole::FfnSharedExpertDown);
        let has_glm_shared_expert_ffn = manifest.model_family == "glm4_moe_lite"
            && roles.contains(&NativeTensorRole::FfnSharedExpertGate)
            && roles.contains(&NativeTensorRole::FfnSharedExpertUp)
            && roles.contains(&NativeTensorRole::FfnSharedExpertDown);
        let has_moe_expert_ffn = roles.contains(&NativeTensorRole::FfnGateInp)
            && roles.contains(&NativeTensorRole::FfnDownExps)
            && (roles.contains(&NativeTensorRole::FfnGateUpExpsPacked)
                || roles.contains(&NativeTensorRole::FfnGateExps)
                || roles.contains(&NativeTensorRole::FfnUpExps));
        if !(has_dense_ffn
            || has_shared_expert_ffn
            || has_glm_shared_expert_ffn
            || has_moe_expert_ffn)
        {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} must provide dense FFN tensors or MoE expert tensors",
                    layer_index
                ),
            });
        }

        // Attention QKV/O are required for full-attention layers but optional
        // for mixed-architecture models (e.g. Qwen3.5 linear_attention layers).
        let has_any_attention = roles.contains(&NativeTensorRole::AttentionO)
            || roles.contains(&NativeTensorRole::AttentionQ)
            || roles.contains(&NativeTensorRole::AttentionK)
            || roles.contains(&NativeTensorRole::AttentionQkvPacked)
            || has_any_glm_mla_attention_role(roles);
        let has_any_linear_attention = roles.contains(&NativeTensorRole::LinearAttentionInProjQkv)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjQkvz)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjZ)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjA)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjB)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjBa)
            || roles.contains(&NativeTensorRole::LinearAttentionConv1d)
            || roles.contains(&NativeTensorRole::LinearAttentionDtBias)
            || roles.contains(&NativeTensorRole::LinearAttentionALog)
            || roles.contains(&NativeTensorRole::LinearAttentionNorm)
            || roles.contains(&NativeTensorRole::LinearAttentionOutProj);
        let has_any_moe = roles.contains(&NativeTensorRole::FfnGateInp)
            || roles.contains(&NativeTensorRole::FfnGateInpScale)
            || roles.contains(&NativeTensorRole::FfnNorm2)
            || roles.contains(&NativeTensorRole::FfnPostNorm1)
            || roles.contains(&NativeTensorRole::FfnPostNorm2)
            || roles.contains(&NativeTensorRole::FfnSharedExpertGateInp)
            || roles.contains(&NativeTensorRole::FfnSharedExpertGate)
            || roles.contains(&NativeTensorRole::FfnSharedExpertUp)
            || roles.contains(&NativeTensorRole::FfnSharedExpertDown)
            || roles.contains(&NativeTensorRole::FfnGateExps)
            || roles.contains(&NativeTensorRole::FfnUpExps)
            || roles.contains(&NativeTensorRole::FfnGateUpExpsPacked)
            || roles.contains(&NativeTensorRole::FfnDownExps)
            || roles.contains(&NativeTensorRole::FfnDownExpsScale);
        if has_any_attention {
            require_layer_role(
                roles,
                NativeTensorRole::AttentionO,
                layer_index,
                "attention_o",
            )?;
            if has_any_glm_mla_attention_role(roles) {
                if manifest.model_family != "glm4_moe_lite" {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} provides GLM MLA attention tensors but model_family is {:?}",
                            layer_index, manifest.model_family
                        ),
                    });
                }
                for (role, label) in [
                    (NativeTensorRole::AttentionQa, "attention_qa"),
                    (NativeTensorRole::AttentionQaNorm, "attention_qa_norm"),
                    (NativeTensorRole::AttentionQb, "attention_qb"),
                    (NativeTensorRole::AttentionKvA, "attention_kv_a"),
                    (NativeTensorRole::AttentionKvANorm, "attention_kv_a_norm"),
                    (NativeTensorRole::AttentionEmbedQ, "attention_embed_q"),
                    (
                        NativeTensorRole::AttentionUnembedOut,
                        "attention_unembed_out",
                    ),
                ] {
                    require_layer_role(roles, role, layer_index, label)?;
                }
                if roles.contains(&NativeTensorRole::AttentionQkvPacked)
                    || roles.contains(&NativeTensorRole::AttentionQ)
                    || roles.contains(&NativeTensorRole::AttentionK)
                    || roles.contains(&NativeTensorRole::AttentionV)
                {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} must not mix GLM MLA attention with standard Q/K/V tensors",
                            layer_index
                        ),
                    });
                }
            } else {
                let uses_shared_kv = manifest.kv_shared_source_layers.contains_key(&layer_index);
                if uses_shared_kv {
                    require_layer_role(
                        roles,
                        NativeTensorRole::AttentionQ,
                        layer_index,
                        "attention_q",
                    )?;
                    if roles.contains(&NativeTensorRole::AttentionQkvPacked)
                        || roles.contains(&NativeTensorRole::AttentionK)
                        || roles.contains(&NativeTensorRole::AttentionV)
                    {
                        return Err(NativeModelError::InvalidManifest {
                            message: format!(
                                "KV-shared layer {} must provide attention_q/attention_o only and reuse source K/V",
                                layer_index
                            ),
                        });
                    }
                } else {
                    let uses_value_from_key = manifest
                        .attention_value_from_key_layers
                        .contains(&layer_index);
                    if uses_value_from_key
                        && (roles.contains(&NativeTensorRole::AttentionQkvPacked)
                            || roles.contains(&NativeTensorRole::AttentionV))
                    {
                        return Err(NativeModelError::InvalidManifest {
                            message: format!(
                                "value-from-key layer {} must provide split attention_q/attention_k without attention_v or attention_qkv_packed",
                                layer_index
                            ),
                        });
                    }
                    let has_packed_qkv = roles.contains(&NativeTensorRole::AttentionQkvPacked);
                    let has_split_qkv = roles.contains(&NativeTensorRole::AttentionQ)
                        && roles.contains(&NativeTensorRole::AttentionK)
                        && (roles.contains(&NativeTensorRole::AttentionV) || uses_value_from_key);
                    if !(has_packed_qkv || has_split_qkv) {
                        return Err(NativeModelError::InvalidManifest {
                            message: format!(
                                "layer {} must provide attention_qkv_packed or attention_q/attention_k plus attention_v (or mark the layer in attention_value_from_key_layers)",
                                layer_index
                            ),
                        });
                    }
                }
            }
        }
        if has_any_linear_attention {
            if !manifest.linear_attention.is_enabled() {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} provides linear attention tensors but manifest.linear_attention is not configured",
                        layer_index
                    ),
                });
            }
            let has_split_linear = roles.contains(&NativeTensorRole::LinearAttentionInProjQkv)
                && roles.contains(&NativeTensorRole::LinearAttentionInProjZ)
                && roles.contains(&NativeTensorRole::LinearAttentionInProjA)
                && roles.contains(&NativeTensorRole::LinearAttentionInProjB);
            let has_packed_linear = roles.contains(&NativeTensorRole::LinearAttentionInProjQkvz)
                && roles.contains(&NativeTensorRole::LinearAttentionInProjBa);
            if !(has_split_linear || has_packed_linear) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} must provide linear_attention split qkv/z/a/b or packed qkvz/ba projections",
                        layer_index
                    ),
                });
            }
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionConv1d,
                layer_index,
                "linear_attention_conv1d",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionDtBias,
                layer_index,
                "linear_attention_dt_bias",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionALog,
                layer_index,
                "linear_attention_a_log",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionNorm,
                layer_index,
                "linear_attention_norm",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionOutProj,
                layer_index,
                "linear_attention_out_proj",
            )?;
        }
        if has_any_moe {
            if !manifest.moe.is_enabled() {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} provides MoE tensors but manifest.moe is not configured",
                        layer_index
                    ),
                });
            }
            require_layer_role(
                roles,
                NativeTensorRole::FfnGateInp,
                layer_index,
                "ffn_gate_inp",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::FfnDownExps,
                layer_index,
                "ffn_down_exps",
            )?;
            let has_any_shared_expert = roles.contains(&NativeTensorRole::FfnSharedExpertGateInp)
                || roles.contains(&NativeTensorRole::FfnSharedExpertGate)
                || roles.contains(&NativeTensorRole::FfnSharedExpertUp)
                || roles.contains(&NativeTensorRole::FfnSharedExpertDown);
            if has_any_shared_expert || moe_requires_shared_expert(manifest) {
                if manifest.model_family != "glm4_moe_lite" {
                    require_layer_role(
                        roles,
                        NativeTensorRole::FfnSharedExpertGateInp,
                        layer_index,
                        "ffn_shared_expert_gate_inp",
                    )?;
                }
                require_layer_role(
                    roles,
                    NativeTensorRole::FfnSharedExpertGate,
                    layer_index,
                    "ffn_shared_expert_gate",
                )?;
                require_layer_role(
                    roles,
                    NativeTensorRole::FfnSharedExpertUp,
                    layer_index,
                    "ffn_shared_expert_up",
                )?;
                require_layer_role(
                    roles,
                    NativeTensorRole::FfnSharedExpertDown,
                    layer_index,
                    "ffn_shared_expert_down",
                )?;
            }
            let has_packed_moe = roles.contains(&NativeTensorRole::FfnGateUpExpsPacked);
            let has_gate_exps = roles.contains(&NativeTensorRole::FfnGateExps);
            let has_up_exps = roles.contains(&NativeTensorRole::FfnUpExps);
            let has_split_moe = has_gate_exps && has_up_exps;
            if has_packed_moe && (has_gate_exps || has_up_exps) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} must not mix ffn_gate_up_exps_packed with ffn_gate_exps/ffn_up_exps",
                        layer_index
                    ),
                });
            }
            if !(has_packed_moe || has_split_moe) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} must provide ffn_gate_up_exps_packed or ffn_gate_exps/ffn_up_exps",
                        layer_index
                    ),
                });
            }
        }
    }

    validate_native_model_tensor_shapes(manifest)?;

    Ok(())
}

fn has_any_glm_mla_attention_role(roles: &[NativeTensorRole]) -> bool {
    roles.contains(&NativeTensorRole::AttentionQa)
        || roles.contains(&NativeTensorRole::AttentionQaNorm)
        || roles.contains(&NativeTensorRole::AttentionQb)
        || roles.contains(&NativeTensorRole::AttentionKvA)
        || roles.contains(&NativeTensorRole::AttentionKvANorm)
        || roles.contains(&NativeTensorRole::AttentionEmbedQ)
        || roles.contains(&NativeTensorRole::AttentionUnembedOut)
}

fn moe_requires_shared_expert(manifest: &NativeModelManifest) -> bool {
    manifest.moe.is_enabled() && matches!(manifest.model_family.as_str(), "qwen3_5" | "qwen3_next")
}

fn validate_native_model_tensor_shapes(
    manifest: &NativeModelManifest,
) -> Result<(), NativeModelError> {
    if !manifest
        .attention_head_count
        .is_multiple_of(manifest.kv_head_count)
    {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "attention_head_count {} must be divisible by kv_head_count {}",
                manifest.attention_head_count, manifest.kv_head_count
            ),
        });
    }

    let hidden_size = u64::from(manifest.hidden_size);
    let vocab_size = u64::from(manifest.vocab_size);
    let token_embedding = required_global_tensor_spec(
        manifest,
        NativeTensorRole::TokenEmbedding,
        "token_embedding",
    )?;
    expect_matrix_shape(token_embedding, vocab_size, hidden_size, "token_embedding")?;

    let final_norm =
        required_global_tensor_spec(manifest, NativeTensorRole::FinalNorm, "final_norm")?;
    expect_vector_shape(final_norm, hidden_size, "final_norm")?;

    if !manifest.tie_word_embeddings {
        let lm_head = required_global_tensor_spec(manifest, NativeTensorRole::LmHead, "lm_head")?;
        expect_matrix_shape(lm_head, vocab_size, hidden_size, "lm_head")?;
    }

    for layer_index in 0..manifest.layer_count {
        let attention_norm = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::AttentionNorm,
            "attention_norm",
        )?;
        expect_vector_shape(attention_norm, hidden_size, "attention_norm")?;
        if let Some(attention_post_norm) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionPostNorm,
            Some(layer_index),
        ) {
            expect_vector_shape(attention_post_norm, hidden_size, "attention_post_norm")?;
        }
        if let Some(attention_q_norm) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionQNorm,
            Some(layer_index),
        ) {
            let head_dim = configured_attention_head_dim(manifest, layer_index);
            expect_vector_shape(attention_q_norm, head_dim, "attention_q_norm")?;
        }
        if let Some(attention_k_norm) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionKNorm,
            Some(layer_index),
        ) {
            let head_dim = configured_attention_head_dim(manifest, layer_index);
            expect_vector_shape(attention_k_norm, head_dim, "attention_k_norm")?;
        }
        // Attention O shape validation — only for layers that have attention tensors.
        // The output projection maps from attention output dim back to hidden_size.
        // For standard attention: o_proj shape is [hidden_size, num_heads * head_dim].
        // For gated attention (Qwen3.5): q_proj has 2x rows (queries + gate), but
        // o_proj still maps from num_heads * head_dim, not from q_proj rows.
        if let Some(attention_o) =
            manifest_tensor(manifest, NativeTensorRole::AttentionO, Some(layer_index))
            && manifest_tensor(manifest, NativeTensorRole::AttentionQa, Some(layer_index)).is_none()
        {
            let attention_output_cols = u64::from(manifest.attention_head_count)
                * configured_attention_head_dim(manifest, layer_index);
            expect_matrix_shape(
                attention_o,
                hidden_size,
                attention_output_cols,
                "attention_o",
            )?;
        }

        let ffn_norm = manifest_tensor(manifest, NativeTensorRole::FfnNorm, Some(layer_index))
            .or_else(|| {
                manifest_tensor(
                    manifest,
                    NativeTensorRole::AttentionPostNorm,
                    Some(layer_index),
                )
            });
        if let Some(ffn_norm) = ffn_norm {
            expect_vector_shape(ffn_norm, hidden_size, "ffn_norm")?;
        }
        if let Some(ffn_norm_2) =
            manifest_tensor(manifest, NativeTensorRole::FfnNorm2, Some(layer_index))
        {
            expect_vector_shape(ffn_norm_2, hidden_size, "ffn_norm_2")?;
        }
        if let Some(ffn_post_norm) =
            manifest_tensor(manifest, NativeTensorRole::FfnPostNorm, Some(layer_index))
        {
            expect_vector_shape(ffn_post_norm, hidden_size, "ffn_post_norm")?;
        }
        if let Some(ffn_post_norm_1) =
            manifest_tensor(manifest, NativeTensorRole::FfnPostNorm1, Some(layer_index))
        {
            expect_vector_shape(ffn_post_norm_1, hidden_size, "ffn_post_norm_1")?;
        }
        if let Some(ffn_post_norm_2) =
            manifest_tensor(manifest, NativeTensorRole::FfnPostNorm2, Some(layer_index))
        {
            expect_vector_shape(ffn_post_norm_2, hidden_size, "ffn_post_norm_2")?;
        }

        let ffn_down = manifest_tensor(manifest, NativeTensorRole::FfnDown, Some(layer_index));
        let ffn_down_shape = ffn_down
            .map(|tensor| {
                matrix_shape(tensor).ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_down must be a rank-2 matrix",
                        layer_index
                    ),
                })
            })
            .transpose()?;
        if let (Some(ffn_down), Some(ffn_down_shape)) = (ffn_down, ffn_down_shape) {
            if !ffn_down.source_quantized && ffn_down_shape.0 != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_down must have shape [{}, intermediate_dim], got {:?}",
                        layer_index, hidden_size, ffn_down.shape
                    ),
                });
            }
        }

        if let Some(attention_qkv) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionQkvPacked,
            Some(layer_index),
        ) {
            let head_dim = configured_attention_head_dim(manifest, layer_index);
            let q_rows = u64::from(manifest.attention_head_count) * head_dim;
            let packed_q_rows = if manifest.attn_output_gate {
                q_rows.saturating_mul(2)
            } else {
                q_rows
            };
            let kv_rows = u64::from(manifest.kv_head_count) * head_dim;
            expect_matrix_shape(
                attention_qkv,
                packed_q_rows + kv_rows + kv_rows,
                hidden_size,
                "attention_qkv_packed",
            )?;
        } else if manifest_tensor(manifest, NativeTensorRole::AttentionQ, Some(layer_index))
            .is_some()
        {
            let attention_q = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::AttentionQ,
                "attention_q",
            )?;
            if manifest.kv_shared_source_layers.contains_key(&layer_index) {
                validate_q_only_attention_tensor(manifest, layer_index, attention_q)?;
            } else {
                let attention_k = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::AttentionK,
                    "attention_k",
                )?;
                let split_dims = resolved_split_attention_dims(manifest, layer_index)?;
                expect_matrix_shape(attention_q, split_dims.q_rows, hidden_size, "attention_q")?;
                expect_matrix_shape(attention_k, split_dims.kv_rows, hidden_size, "attention_k")?;
                if let Some(attention_v) =
                    manifest_tensor(manifest, NativeTensorRole::AttentionV, Some(layer_index))
                {
                    expect_matrix_shape(
                        attention_v,
                        split_dims.kv_rows,
                        hidden_size,
                        "attention_v",
                    )?;
                }
            }
        } else if manifest_tensor(manifest, NativeTensorRole::AttentionQa, Some(layer_index))
            .is_some()
        {
            validate_glm_mla_attention_tensor_shapes(manifest, layer_index)?;
        }
        // Layers without any attention tensors (e.g. linear_attention) skip QKV shape validation.
        if manifest_tensor(
            manifest,
            NativeTensorRole::LinearAttentionInProjQkv,
            Some(layer_index),
        )
        .is_some()
        {
            let linear_dims = resolved_linear_attention_dims(manifest)?;
            if let Some(in_proj_qkvz) = manifest_tensor(
                manifest,
                NativeTensorRole::LinearAttentionInProjQkvz,
                Some(layer_index),
            ) {
                expect_matrix_shape(
                    in_proj_qkvz,
                    linear_dims.conv_dim + linear_dims.value_dim,
                    hidden_size,
                    "linear_attention_in_proj_qkvz",
                )?;
                let in_proj_ba = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjBa,
                    "linear_attention_in_proj_ba",
                )?;
                expect_matrix_shape(
                    in_proj_ba,
                    linear_dims.num_value_heads.saturating_mul(2),
                    hidden_size,
                    "linear_attention_in_proj_ba",
                )?;
            } else {
                let in_proj_qkv = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjQkv,
                    "linear_attention_in_proj_qkv",
                )?;
                expect_matrix_shape(
                    in_proj_qkv,
                    linear_dims.conv_dim,
                    hidden_size,
                    "linear_attention_in_proj_qkv",
                )?;
                let in_proj_z = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjZ,
                    "linear_attention_in_proj_z",
                )?;
                expect_matrix_shape(
                    in_proj_z,
                    linear_dims.value_dim,
                    hidden_size,
                    "linear_attention_in_proj_z",
                )?;
                let in_proj_a = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjA,
                    "linear_attention_in_proj_a",
                )?;
                expect_matrix_shape(
                    in_proj_a,
                    linear_dims.num_value_heads,
                    hidden_size,
                    "linear_attention_in_proj_a",
                )?;
                let in_proj_b = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::LinearAttentionInProjB,
                    "linear_attention_in_proj_b",
                )?;
                expect_matrix_shape(
                    in_proj_b,
                    linear_dims.num_value_heads,
                    hidden_size,
                    "linear_attention_in_proj_b",
                )?;
            }
            let dt_bias = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionDtBias,
                "linear_attention_dt_bias",
            )?;
            expect_vector_shape(
                dt_bias,
                linear_dims.num_value_heads,
                "linear_attention_dt_bias",
            )?;
            let a_log = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionALog,
                "linear_attention_a_log",
            )?;
            expect_vector_shape(a_log, linear_dims.num_value_heads, "linear_attention_a_log")?;
            let norm = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionNorm,
                "linear_attention_norm",
            )?;
            expect_vector_shape(norm, linear_dims.value_head_dim, "linear_attention_norm")?;
            let out_proj = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionOutProj,
                "linear_attention_out_proj",
            )?;
            expect_matrix_shape(
                out_proj,
                hidden_size,
                linear_dims.value_dim,
                "linear_attention_out_proj",
            )?;
            let conv1d = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::LinearAttentionConv1d,
                "linear_attention_conv1d",
            )?;
            validate_linear_attention_conv_tensor(
                conv1d,
                linear_dims.conv_dim,
                linear_dims.conv_kernel_dim,
            )?;
        }
        if manifest_tensor(manifest, NativeTensorRole::FfnGateInp, Some(layer_index)).is_some() {
            let moe_dims = resolved_moe_dims(manifest)?;
            let gate_inp = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnGateInp,
                "ffn_gate_inp",
            )?;
            expect_matrix_shape(gate_inp, moe_dims.expert_count, hidden_size, "ffn_gate_inp")?;
            if let Some(gate_inp_scale) = manifest_tensor(
                manifest,
                NativeTensorRole::FfnGateInpScale,
                Some(layer_index),
            ) {
                expect_vector_shape(gate_inp_scale, hidden_size, "ffn_gate_inp_scale")?;
            }
            if let Some(ffn_gate_up_exps_packed) = manifest_tensor(
                manifest,
                NativeTensorRole::FfnGateUpExpsPacked,
                Some(layer_index),
            ) {
                expect_tensor_shape(
                    ffn_gate_up_exps_packed,
                    &[
                        moe_dims.expert_count,
                        moe_dims.expert_intermediate_size.saturating_mul(2),
                        hidden_size,
                    ],
                    "ffn_gate_up_exps_packed",
                )?;
            } else {
                let ffn_gate_exps = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnGateExps,
                    "ffn_gate_exps",
                )?;
                let ffn_up_exps = required_layer_tensor_spec(
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnUpExps,
                    "ffn_up_exps",
                )?;
                expect_tensor_shape(
                    ffn_gate_exps,
                    &[
                        moe_dims.expert_count,
                        moe_dims.expert_intermediate_size,
                        hidden_size,
                    ],
                    "ffn_gate_exps",
                )?;
                expect_tensor_shape(
                    ffn_up_exps,
                    &[
                        moe_dims.expert_count,
                        moe_dims.expert_intermediate_size,
                        hidden_size,
                    ],
                    "ffn_up_exps",
                )?;
            }
            let ffn_down_exps = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnDownExps,
                "ffn_down_exps",
            )?;
            expect_tensor_shape(
                ffn_down_exps,
                &[
                    moe_dims.expert_count,
                    hidden_size,
                    moe_dims.expert_intermediate_size,
                ],
                "ffn_down_exps",
            )?;
            if let Some(ffn_down_exps_scale) = manifest_tensor(
                manifest,
                NativeTensorRole::FfnDownExpsScale,
                Some(layer_index),
            ) {
                expect_vector_shape(
                    ffn_down_exps_scale,
                    moe_dims.expert_count,
                    "ffn_down_exps_scale",
                )?;
            }
        }

        let dense_intermediate_dim = if let Some(ffn_gate_up_packed) = manifest_tensor(
            manifest,
            NativeTensorRole::FfnGateUpPacked,
            Some(layer_index),
        ) {
            let (rows, cols) = matrix_shape(ffn_gate_up_packed).ok_or_else(|| {
                NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate_up_packed must be a rank-2 matrix",
                        layer_index
                    ),
                }
            })?;
            if ffn_gate_up_packed.source_quantized {
                let expected_cols = expected_packed_cols(hidden_size, ffn_gate_up_packed)?;
                if cols != expected_cols {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} tensor ffn_gate_up_packed must have packed quantized shape [rows, {}], got {:?}",
                            layer_index, expected_cols, ffn_gate_up_packed.shape
                        ),
                    });
                }
            } else if cols != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate_up_packed must have hidden_size {} columns, got {:?}",
                        layer_index, hidden_size, ffn_gate_up_packed.shape
                    ),
                });
            }
            if !rows.is_multiple_of(2) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate_up_packed row count must be even, got {}",
                        layer_index, rows
                    ),
                });
            }
            rows / 2
        } else if manifest_tensor(manifest, NativeTensorRole::FfnGate, Some(layer_index)).is_some()
        {
            let ffn_gate = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnGate,
                "ffn_gate",
            )?;
            let gate_shape =
                matrix_shape(ffn_gate).ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate must be a rank-2 matrix",
                        layer_index
                    ),
                })?;
            let ffn_up = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnUp,
                "ffn_up",
            )?;
            let up_shape =
                matrix_shape(ffn_up).ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_up must be a rank-2 matrix",
                        layer_index
                    ),
                })?;
            if ffn_gate.source_quantized {
                let expected_cols = expected_packed_cols(hidden_size, ffn_gate)?;
                if gate_shape.1 != expected_cols {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} tensor ffn_gate must have packed quantized shape [rows, {}], got {:?}",
                            layer_index, expected_cols, ffn_gate.shape
                        ),
                    });
                }
            } else if gate_shape.1 != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate must have hidden_size {} columns, got {:?}",
                        layer_index, hidden_size, ffn_gate.shape
                    ),
                });
            }
            if ffn_up.source_quantized {
                let expected_cols = expected_packed_cols(hidden_size, ffn_up)?;
                if up_shape.1 != expected_cols {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} tensor ffn_up must have packed quantized shape [rows, {}], got {:?}",
                            layer_index, expected_cols, ffn_up.shape
                        ),
                    });
                }
            } else if up_shape.1 != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_up must have hidden_size {} columns, got {:?}",
                        layer_index, hidden_size, ffn_up.shape
                    ),
                });
            }
            if gate_shape.0 != up_shape.0 {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensors ffn_gate and ffn_up must agree on intermediate rows, got {:?} and {:?}",
                        layer_index, ffn_gate.shape, ffn_up.shape
                    ),
                });
            }
            gate_shape.0
        } else {
            0
        };

        if let (Some(ffn_down), Some(ffn_down_shape)) = (
            manifest_tensor(manifest, NativeTensorRole::FfnDown, Some(layer_index)),
            ffn_down_shape,
        ) {
            if ffn_down.source_quantized {
                let expected_cols = expected_packed_cols(dense_intermediate_dim, ffn_down)?;
                if ffn_down_shape.1 != expected_cols {
                    return Err(NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} tensor ffn_down must have packed quantized shape [rows, {}], got {:?}",
                            layer_index, expected_cols, ffn_down.shape
                        ),
                    });
                }
            } else if ffn_down_shape.1 != dense_intermediate_dim {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_down must have intermediate_dim {} columns, got {:?}",
                        layer_index, dense_intermediate_dim, ffn_down.shape
                    ),
                });
            }
        }

        if let Some(shared_gate_inp) = manifest_tensor(
            manifest,
            NativeTensorRole::FfnSharedExpertGateInp,
            Some(layer_index),
        ) {
            let moe_dims = resolved_moe_dims(manifest)?;
            expect_matrix_shape(
                shared_gate_inp,
                1,
                hidden_size,
                "ffn_shared_expert_gate_inp",
            )?;
            let shared_gate = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnSharedExpertGate,
                "ffn_shared_expert_gate",
            )?;
            expect_matrix_shape(
                shared_gate,
                moe_dims.expert_intermediate_size,
                hidden_size,
                "ffn_shared_expert_gate",
            )?;
            let shared_up = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnSharedExpertUp,
                "ffn_shared_expert_up",
            )?;
            expect_matrix_shape(
                shared_up,
                moe_dims.expert_intermediate_size,
                hidden_size,
                "ffn_shared_expert_up",
            )?;
            let shared_down = required_layer_tensor_spec(
                manifest,
                layer_index,
                NativeTensorRole::FfnSharedExpertDown,
                "ffn_shared_expert_down",
            )?;
            expect_matrix_shape(
                shared_down,
                hidden_size,
                moe_dims.expert_intermediate_size,
                "ffn_shared_expert_down",
            )?;
        }
    }

    Ok(())
}

fn validate_glm_mla_attention_tensor_shapes(
    manifest: &NativeModelManifest,
    layer_index: u32,
) -> Result<(), NativeModelError> {
    let hidden_size = u64::from(manifest.hidden_size);
    let head_count = u64::from(manifest.attention_head_count);
    let q_lora_rank = u64::from(manifest.mla_attention.q_lora_rank.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.q_lora_rank must be configured".to_string(),
        }
    })?);
    let kv_lora_rank = u64::from(manifest.mla_attention.kv_lora_rank.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.kv_lora_rank must be configured".to_string(),
        }
    })?);
    let qk_nope_head_dim = u64::from(manifest.mla_attention.qk_nope_head_dim.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.qk_nope_head_dim must be configured".to_string(),
        }
    })?);
    let qk_rope_head_dim = u64::from(manifest.mla_attention.qk_rope_head_dim.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.qk_rope_head_dim must be configured".to_string(),
        }
    })?);
    let value_head_dim = u64::from(manifest.mla_attention.value_head_dim.ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "mla_attention.value_head_dim must be configured".to_string(),
        }
    })?);
    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    let attention_qa = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQa,
        "attention_qa",
    )?;
    expect_matrix_shape(attention_qa, q_lora_rank, hidden_size, "attention_qa")?;
    let attention_qa_norm = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQaNorm,
        "attention_qa_norm",
    )?;
    expect_vector_shape(attention_qa_norm, q_lora_rank, "attention_qa_norm")?;
    let attention_qb = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQb,
        "attention_qb",
    )?;
    expect_matrix_shape(
        attention_qb,
        head_count * q_head_dim,
        q_lora_rank,
        "attention_qb",
    )?;
    let attention_kv_a = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionKvA,
        "attention_kv_a",
    )?;
    expect_matrix_shape(
        attention_kv_a,
        kv_lora_rank + qk_rope_head_dim,
        hidden_size,
        "attention_kv_a",
    )?;
    let attention_kv_a_norm = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionKvANorm,
        "attention_kv_a_norm",
    )?;
    expect_vector_shape(attention_kv_a_norm, kv_lora_rank, "attention_kv_a_norm")?;
    let attention_embed_q = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionEmbedQ,
        "attention_embed_q",
    )?;
    expect_tensor_shape(
        attention_embed_q,
        &[head_count, kv_lora_rank, qk_nope_head_dim],
        "attention_embed_q",
    )?;
    let attention_unembed_out = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionUnembedOut,
        "attention_unembed_out",
    )?;
    expect_tensor_shape(
        attention_unembed_out,
        &[head_count, value_head_dim, kv_lora_rank],
        "attention_unembed_out",
    )?;
    let attention_o = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionO,
        "attention_o",
    )?;
    expect_matrix_shape(
        attention_o,
        hidden_size,
        head_count * value_head_dim,
        "attention_o",
    )
}

fn validate_manifest_layer_index_list(
    manifest: &NativeModelManifest,
    layer_indices: &[u32],
    field_name: &str,
) -> Result<(), NativeModelError> {
    for &layer_index in layer_indices {
        if layer_index >= manifest.layer_count {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "{} contains out-of-range layer index {} (layer_count={})",
                    field_name, layer_index, manifest.layer_count
                ),
            });
        }
    }

    Ok(())
}

fn validate_interleaved_attention_metadata(
    manifest: &NativeModelManifest,
) -> Result<(), NativeModelError> {
    if let Some(rope_theta_swa) = manifest.rope_theta_swa {
        if rope_theta_swa == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!("rope_theta_swa must be > 0, got {rope_theta_swa}"),
            });
        }
    }
    if let Some(global_head_dim) = manifest.global_head_dim {
        if global_head_dim == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: "global_head_dim must be > 0".to_string(),
            });
        }
    }
    if let Some(sliding_window_size) = manifest.sliding_window_size {
        if sliding_window_size == 0 {
            return Err(NativeModelError::InvalidManifest {
                message: "sliding_window_size must be > 0".to_string(),
            });
        }
    }

    if !manifest.layer_types.is_empty() {
        if manifest.layer_types.len() != manifest.layer_count as usize {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer_types must contain one entry per layer, got {} for layer_count {}",
                    manifest.layer_types.len(),
                    manifest.layer_count
                ),
            });
        }
        for (idx, layer_type) in manifest.layer_types.iter().enumerate() {
            if layer_type != "sliding_attention" && layer_type != "full_attention" {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer_types[{idx}] must be sliding_attention or full_attention, got {layer_type:?}"
                    ),
                });
            }
        }
    }

    for (&layer_index, &source_layer) in &manifest.kv_shared_source_layers {
        if layer_index >= manifest.layer_count || source_layer >= manifest.layer_count {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "kv_shared_source_layers contains out-of-range mapping {} -> {} (layer_count={})",
                    layer_index, source_layer, manifest.layer_count
                ),
            });
        }
        if source_layer >= layer_index {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "kv_shared_source_layers layer {} must reference an earlier source layer, got {}",
                    layer_index, source_layer
                ),
            });
        }
        if !manifest.layer_types.is_empty() {
            let layer_type = &manifest.layer_types[layer_index as usize];
            let source_type = &manifest.layer_types[source_layer as usize];
            if layer_type != source_type {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "kv_shared_source_layers layer {} type {:?} cannot reuse source {} type {:?}",
                        layer_index, layer_type, source_layer, source_type
                    ),
                });
            }
        }
    }

    Ok(())
}

fn require_positive_field(value: Option<u32>, field_name: &str) -> Result<u32, NativeModelError> {
    match value {
        Some(0) => Err(NativeModelError::InvalidManifest {
            message: format!("{field_name} must be > 0"),
        }),
        None => Err(NativeModelError::InvalidManifest {
            message: format!("{field_name} is required when its feature is enabled"),
        }),
        Some(value) => Ok(value),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeSplitAttentionDims {
    q_rows: u64,
    kv_rows: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeLinearAttentionDims {
    num_value_heads: u64,
    num_key_heads: u64,
    key_head_dim: u64,
    value_head_dim: u64,
    conv_kernel_dim: u64,
    key_dim: u64,
    value_dim: u64,
    conv_dim: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeMoeDims {
    expert_count: u64,
    experts_per_token: u64,
    expert_intermediate_size: u64,
}

fn resolved_linear_attention_dims(
    manifest: &NativeModelManifest,
) -> Result<NativeLinearAttentionDims, NativeModelError> {
    let config = &manifest.linear_attention;
    let num_value_heads =
        u64::from(
            config
                .num_value_heads
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.num_value_heads must be configured".to_string(),
                })?,
        );
    let num_key_heads =
        u64::from(
            config
                .num_key_heads
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.num_key_heads must be configured".to_string(),
                })?,
        );
    let key_head_dim =
        u64::from(
            config
                .key_head_dim
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.key_head_dim must be configured".to_string(),
                })?,
        );
    let value_head_dim =
        u64::from(
            config
                .value_head_dim
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.value_head_dim must be configured".to_string(),
                })?,
        );
    let conv_kernel_dim =
        u64::from(
            config
                .conv_kernel_dim
                .ok_or_else(|| NativeModelError::InvalidManifest {
                    message: "linear_attention.conv_kernel_dim must be configured".to_string(),
                })?,
        );
    let key_dim = num_key_heads.checked_mul(key_head_dim).ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "linear attention key_dim overflowed".to_string(),
        }
    })?;
    let value_dim = num_value_heads.checked_mul(value_head_dim).ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: "linear attention value_dim overflowed".to_string(),
        }
    })?;
    let conv_dim = key_dim
        .checked_mul(2)
        .and_then(|twice_key_dim| twice_key_dim.checked_add(value_dim))
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: "linear attention conv_dim overflowed".to_string(),
        })?;

    Ok(NativeLinearAttentionDims {
        num_value_heads,
        num_key_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel_dim,
        key_dim,
        value_dim,
        conv_dim,
    })
}

fn resolved_moe_dims(manifest: &NativeModelManifest) -> Result<NativeMoeDims, NativeModelError> {
    let config = &manifest.moe;
    Ok(NativeMoeDims {
        expert_count: u64::from(config.expert_count.ok_or_else(|| {
            NativeModelError::InvalidManifest {
                message: "moe.expert_count must be configured".to_string(),
            }
        })?),
        experts_per_token: u64::from(config.experts_per_token.ok_or_else(|| {
            NativeModelError::InvalidManifest {
                message: "moe.experts_per_token must be configured".to_string(),
            }
        })?),
        expert_intermediate_size: u64::from(config.expert_intermediate_size.ok_or_else(|| {
            NativeModelError::InvalidManifest {
                message: "moe.expert_intermediate_size must be configured".to_string(),
            }
        })?),
    })
}

fn configured_attention_head_dim(manifest: &NativeModelManifest, layer_index: u32) -> u64 {
    if manifest
        .layer_types
        .get(layer_index as usize)
        .is_some_and(|layer_type| layer_type == "full_attention")
    {
        u64::from(
            manifest
                .global_head_dim
                .unwrap_or(manifest.attention_head_dim),
        )
    } else {
        u64::from(manifest.attention_head_dim)
    }
}

fn resolved_split_attention_dims(
    manifest: &NativeModelManifest,
    layer_index: u32,
) -> Result<NativeSplitAttentionDims, NativeModelError> {
    let attention_q = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionQ,
        "attention_q",
    )?;
    let attention_k = required_layer_tensor_spec(
        manifest,
        layer_index,
        NativeTensorRole::AttentionK,
        "attention_k",
    )?;
    let (q_rows, q_cols) =
        matrix_shape(attention_q).ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q must be a rank-2 matrix",
                layer_index
            ),
        })?;
    let (k_rows, k_cols) =
        matrix_shape(attention_k).ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_k must be a rank-2 matrix",
                layer_index
            ),
        })?;
    let hidden_size = u64::from(manifest.hidden_size);
    // Skip input-dimension checks for quantized tensors (columns are packed).
    if !attention_q.source_quantized {
        if q_cols != hidden_size {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_q must have shape [q_rows, {}], got {:?}",
                    layer_index, hidden_size, attention_q.shape
                ),
            });
        }
        if k_cols != hidden_size {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_k must have shape [kv_rows, {}], got {:?}",
                    layer_index, hidden_size, attention_k.shape
                ),
            });
        }
    }

    let mut head_dim = None;
    if let Some(attention_q_norm) = manifest_tensor(
        manifest,
        NativeTensorRole::AttentionQNorm,
        Some(layer_index),
    ) {
        let q_norm_dim =
            vector_shape(attention_q_norm).ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_q_norm must be a rank-1 vector",
                    layer_index
                ),
            })?;
        head_dim = Some(q_norm_dim);
    }
    if let Some(attention_k_norm) = manifest_tensor(
        manifest,
        NativeTensorRole::AttentionKNorm,
        Some(layer_index),
    ) {
        let k_norm_dim =
            vector_shape(attention_k_norm).ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_k_norm must be a rank-1 vector",
                    layer_index
                ),
            })?;
        if let Some(existing) = head_dim {
            if existing != k_norm_dim {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} attention_q_norm and attention_k_norm must agree on head_dim, got {} vs {}",
                        layer_index, existing, k_norm_dim
                    ),
                });
            }
        } else {
            head_dim = Some(k_norm_dim);
        }
    }
    let head_dim = head_dim.unwrap_or_else(|| configured_attention_head_dim(manifest, layer_index));
    if head_dim == 0 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} resolved attention head_dim must be > 0",
                layer_index
            ),
        });
    }
    // When attn_output_gate is enabled, q_proj encodes both queries and gate
    // values, so the effective row count for head derivation is halved.
    let effective_q_rows = if manifest.attn_output_gate {
        if !q_rows.is_multiple_of(2) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} attention_q rows {} must be even when attn_output_gate is enabled",
                    layer_index, q_rows
                ),
            });
        }
        q_rows / 2
    } else {
        q_rows
    };
    if !effective_q_rows.is_multiple_of(head_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} attention_q rows {} (effective {}) must be divisible by head_dim {}",
                layer_index, q_rows, effective_q_rows, head_dim
            ),
        });
    }
    if !k_rows.is_multiple_of(head_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} attention_k rows {} must be divisible by head_dim {}",
                layer_index, k_rows, head_dim
            ),
        });
    }
    let q_heads = effective_q_rows / head_dim;
    let kv_heads = k_rows / head_dim;
    if q_heads == 0 || kv_heads == 0 || q_heads < kv_heads || !q_heads.is_multiple_of(kv_heads) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} requires q_heads >= kv_heads and divisible; resolved q_heads={} kv_heads={}",
                layer_index, q_heads, kv_heads
            ),
        });
    }

    let kv_rows = if let Some(attention_v) =
        manifest_tensor(manifest, NativeTensorRole::AttentionV, Some(layer_index))
    {
        let (v_rows, v_cols) =
            matrix_shape(attention_v).ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_v must be a rank-2 matrix",
                    layer_index
                ),
            })?;
        if !attention_v.source_quantized && v_cols != hidden_size {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor attention_v must have shape [kv_rows, {}], got {:?}",
                    layer_index, hidden_size, attention_v.shape
                ),
            });
        }
        if v_rows != k_rows {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} attention_k and attention_v must agree on row count, got {} vs {}",
                    layer_index, k_rows, v_rows
                ),
            });
        }
        v_rows
    } else if manifest
        .attention_value_from_key_layers
        .contains(&layer_index)
    {
        k_rows
    } else {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} must provide attention_v or be listed in attention_value_from_key_layers",
                layer_index
            ),
        });
    };

    Ok(NativeSplitAttentionDims { q_rows, kv_rows })
}

fn validate_q_only_attention_tensor(
    manifest: &NativeModelManifest,
    layer_index: u32,
    attention_q: &NativeTensorSpec,
) -> Result<(), NativeModelError> {
    let (q_rows, q_cols) =
        matrix_shape(attention_q).ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q must be a rank-2 matrix",
                layer_index
            ),
        })?;
    let hidden_size = u64::from(manifest.hidden_size);
    if !attention_q.source_quantized && q_cols != hidden_size {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q must have shape [q_rows, {}], got {:?}",
                layer_index, hidden_size, attention_q.shape
            ),
        });
    }

    let head_dim = manifest_tensor(
        manifest,
        NativeTensorRole::AttentionQNorm,
        Some(layer_index),
    )
    .map(|q_norm| {
        vector_shape(q_norm).ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "layer {} tensor attention_q_norm must be a rank-1 vector",
                layer_index
            ),
        })
    })
    .transpose()?
    .unwrap_or_else(|| configured_attention_head_dim(manifest, layer_index));
    if head_dim == 0 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} resolved attention head_dim must be > 0",
                layer_index
            ),
        });
    }
    let effective_q_rows = if manifest.attn_output_gate {
        if !q_rows.is_multiple_of(2) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} attention_q rows {} must be even when attn_output_gate is enabled",
                    layer_index, q_rows
                ),
            });
        }
        q_rows / 2
    } else {
        q_rows
    };
    if effective_q_rows == 0 || !effective_q_rows.is_multiple_of(head_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} attention_q rows {} (effective {}) must be divisible by head_dim {}",
                layer_index, q_rows, effective_q_rows, head_dim
            ),
        });
    }
    Ok(())
}

fn manifest_tensor(
    manifest: &NativeModelManifest,
    role: NativeTensorRole,
    layer_index: Option<u32>,
) -> Option<&NativeTensorSpec> {
    manifest
        .tensors
        .iter()
        .find(|tensor| tensor.role == role && tensor.layer_index == layer_index)
}

fn required_global_tensor_spec<'a>(
    manifest: &'a NativeModelManifest,
    role: NativeTensorRole,
    label: &str,
) -> Result<&'a NativeTensorSpec, NativeModelError> {
    manifest_tensor(manifest, role, None).ok_or_else(|| NativeModelError::InvalidManifest {
        message: format!("manifest is missing required global tensor role {}", label),
    })
}

fn required_layer_tensor_spec<'a>(
    manifest: &'a NativeModelManifest,
    layer_index: u32,
    role: NativeTensorRole,
    label: &str,
) -> Result<&'a NativeTensorSpec, NativeModelError> {
    manifest_tensor(manifest, role, Some(layer_index)).ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: format!(
                "layer {} is missing required tensor role {}",
                layer_index, label
            ),
        }
    })
}

fn matrix_shape(tensor: &NativeTensorSpec) -> Option<(u64, u64)> {
    (tensor.shape.len() == 2).then_some((*tensor.shape.first()?, *tensor.shape.get(1)?))
}

fn total_elements(tensor: &NativeTensorSpec) -> Option<u64> {
    tensor
        .shape
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
}

fn vector_shape(tensor: &NativeTensorSpec) -> Option<u64> {
    (tensor.shape.len() == 1).then_some(*tensor.shape.first()?)
}

fn expect_vector_shape(
    tensor: &NativeTensorSpec,
    expected_len: u64,
    label: &str,
) -> Result<(), NativeModelError> {
    if tensor.shape == [expected_len] {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have shape [{}], got {:?}",
                label, expected_len, tensor.shape
            ),
        })
    }
}

fn expect_matrix_shape(
    tensor: &NativeTensorSpec,
    expected_rows: u64,
    expected_cols: u64,
    label: &str,
) -> Result<(), NativeModelError> {
    // Quantized tensors (e.g. MLX affine) pack columns. Validate the row count
    // and the packed column count derived from bits instead of accepting any
    // second dimension.
    if tensor.source_quantized {
        let Some((rows, cols)) = matrix_shape(tensor) else {
            return Err(NativeModelError::InvalidManifest {
                message: format!("tensor {} must be a rank-2 quantized matrix", label),
            });
        };
        let expected_packed_cols = expected_packed_cols(expected_cols, tensor)?;
        if rows == expected_rows && cols == expected_packed_cols {
            return Ok(());
        }
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have packed quantized shape [{}, {}], got {:?}",
                label, expected_rows, expected_packed_cols, tensor.shape
            ),
        });
    }
    if tensor.shape == [expected_rows, expected_cols] {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have shape [{}, {}], got {:?}",
                label, expected_rows, expected_cols, tensor.shape
            ),
        })
    }
}

fn expect_tensor_shape(
    tensor: &NativeTensorSpec,
    expected_shape: &[u64],
    label: &str,
) -> Result<(), NativeModelError> {
    if tensor.source_quantized {
        if tensor.shape.len() != expected_shape.len() {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "tensor {} must have rank {} for quantized shape, got {:?}",
                    label,
                    expected_shape.len(),
                    tensor.shape
                ),
            });
        }
        if expected_shape.is_empty() {
            return Ok(());
        }
        let expected_last = expected_packed_cols(*expected_shape.last().unwrap(), tensor)?;
        let mut expected = expected_shape.to_vec();
        *expected.last_mut().unwrap() = expected_last;
        if tensor.shape == expected {
            return Ok(());
        }
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have packed quantized shape {:?}, got {:?}",
                label, expected, tensor.shape
            ),
        });
    }
    if tensor.shape == expected_shape {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} must have shape {:?}, got {:?}",
                label, expected_shape, tensor.shape
            ),
        })
    }
}

fn tensor_quantization_or_default(tensor: &NativeTensorSpec) -> NativeTensorQuantization {
    tensor.quantization.clone().unwrap_or_default()
}

fn validate_tensor_quantization(tensor: &NativeTensorSpec) -> Result<(), NativeModelError> {
    if tensor.dtype == NativeTensorDataType::U32 && !tensor.source_quantized {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} uses dtype u32 but source_quantized is false",
                tensor.name
            ),
        });
    }
    let Some(quantization) = &tensor.quantization else {
        return Ok(());
    };
    if !tensor.source_quantized {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} declares quantization but source_quantized is false",
                tensor.name
            ),
        });
    }
    if tensor.dtype != NativeTensorDataType::U32 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} declares affine quantization but dtype is {:?}, expected u32",
                tensor.name, tensor.dtype
            ),
        });
    }
    if quantization.mode != "affine" {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantization mode {} is unsupported",
                tensor.name, quantization.mode
            ),
        });
    }
    if quantization.group_size == 0 {
        return Err(NativeModelError::InvalidManifest {
            message: format!("tensor {} quantization group_size must be > 0", tensor.name),
        });
    }
    if !SUPPORTED_MLX_AFFINE_QUANTIZATION_BITS.contains(&quantization.bits) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantization bits must be one of {:?}, got {}",
                tensor.name, SUPPORTED_MLX_AFFINE_QUANTIZATION_BITS, quantization.bits
            ),
        });
    }
    Ok(())
}

fn expected_packed_cols(
    expected_cols: u64,
    tensor: &NativeTensorSpec,
) -> Result<u64, NativeModelError> {
    let quantization = tensor_quantization_or_default(tensor);
    let packed_bits = expected_cols
        .checked_mul(u64::from(quantization.bits))
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized column count overflowed for {} columns at {} bits",
                tensor.name, expected_cols, quantization.bits
            ),
        })?;
    Ok(packed_bits.div_ceil(32))
}

fn validate_linear_attention_conv_tensor(
    tensor: &NativeTensorSpec,
    expected_channels: u64,
    expected_kernel_dim: u64,
) -> Result<(), NativeModelError> {
    if tensor.shape.is_empty() || tensor.shape[0] != expected_channels {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor linear_attention_conv1d must start with channel dimension {}, got {:?}",
                expected_channels, tensor.shape
            ),
        });
    }
    let remaining_product = tensor.shape[1..]
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: "linear_attention_conv1d shape overflowed".to_string(),
        })?;
    if remaining_product != expected_kernel_dim {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor linear_attention_conv1d must encode kernel size {}, got {:?}",
                expected_kernel_dim, tensor.shape
            ),
        });
    }
    if total_elements(tensor) != expected_channels.checked_mul(expected_kernel_dim) {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor linear_attention_conv1d total element count must be {}, got {:?}",
                expected_channels.saturating_mul(expected_kernel_dim),
                tensor.shape
            ),
        });
    }
    Ok(())
}

fn require_global_role(
    roles: &[NativeTensorRole],
    required: NativeTensorRole,
    label: &str,
) -> Result<(), NativeModelError> {
    if roles.contains(&required) {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!("manifest is missing required global tensor role {}", label),
        })
    }
}

fn require_layer_role(
    roles: &[NativeTensorRole],
    required: NativeTensorRole,
    layer_index: u32,
    label: &str,
) -> Result<(), NativeModelError> {
    if roles.contains(&required) {
        Ok(())
    } else {
        Err(NativeModelError::InvalidManifest {
            message: format!(
                "layer {} is missing required tensor role {}",
                layer_index, label
            ),
        })
    }
}

fn validate_tensor_path(
    root_dir: &Path,
    tensor: &NativeTensorSpec,
) -> Result<(), NativeModelError> {
    if tensor.file.is_absolute() {
        return Err(NativeModelError::InvalidManifest {
            message: format!("tensor {} file path must be relative", tensor.name),
        });
    }
    if tensor
        .file
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(NativeModelError::InvalidManifest {
            message: format!("tensor {} file path must not escape root_dir", tensor.name),
        });
    }

    let path = root_dir.join(&tensor.file);
    let metadata = fs::metadata(&path).map_err(|source| NativeModelError::InvalidManifest {
        message: format!(
            "tensor {} references missing file {}: {}",
            tensor.name,
            path.display(),
            source
        ),
    })?;
    if !metadata.is_file() {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} path {} is not a file",
                tensor.name,
                path.display()
            ),
        });
    }
    let file_len = metadata.len();
    let end = tensor
        .offset_bytes
        .checked_add(tensor.length_bytes)
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!("tensor {} byte range overflowed", tensor.name),
        })?;
    if end > file_len {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} byte range [{}, {}) exceeds file length {}",
                tensor.name, tensor.offset_bytes, end, file_len
            ),
        });
    }

    Ok(())
}

fn validate_quantized_source_path(
    root_dir: &Path,
    tensor: &NativeTensorSpec,
) -> Result<(), NativeModelError> {
    let Some(source) = &tensor.quantized_source else {
        return Ok(());
    };
    if !tensor.source_quantized {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} declares quantized_source but source_quantized is false",
                tensor.name
            ),
        });
    }
    if source.length_bytes == 0 {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source must have positive length_bytes",
                tensor.name
            ),
        });
    }
    if source.file.is_absolute() {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source file path must be relative",
                tensor.name
            ),
        });
    }
    if source
        .file
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source file path must not escape root_dir",
                tensor.name
            ),
        });
    }

    let path = root_dir.join(&source.file);
    let metadata =
        fs::metadata(&path).map_err(|source_error| NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} references missing quantized_source file {}: {}",
                tensor.name,
                path.display(),
                source_error
            ),
        })?;
    if !metadata.is_file() {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source path {} is not a file",
                tensor.name,
                path.display()
            ),
        });
    }
    let file_len = metadata.len();
    let end = source
        .offset_bytes
        .checked_add(source.length_bytes)
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source byte range overflowed",
                tensor.name
            ),
        })?;
    if end > file_len {
        return Err(NativeModelError::InvalidManifest {
            message: format!(
                "tensor {} quantized_source byte range [{}, {}) exceeds file length {}",
                tensor.name, source.offset_bytes, end, file_len
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_test_dir(label: &str) -> PathBuf {
        static NEXT_SUFFIX: AtomicU64 = AtomicU64::new(0);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let suffix = NEXT_SUFFIX.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ax-native-model-{label}-{}-{unique}-{suffix}",
            std::process::id()
        ))
    }

    fn write_fixture(
        mut manifest: NativeModelManifest,
        file_names: &[&str],
    ) -> (PathBuf, NativeModelManifest) {
        let dir = unique_test_dir("fixture");
        fs::create_dir_all(&dir).expect("fixture directory should create");
        for file_name in file_names {
            fs::write(dir.join(file_name), vec![0_u8; 4096]).expect("tensor file should write");
        }
        for tensor in &mut manifest.tensors {
            tensor.length_bytes = 32;
        }
        fs::write(
            dir.join(AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");
        (dir, manifest)
    }

    fn packed_layer_manifest() -> NativeModelManifest {
        NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 2,
            hidden_size: 2048,
            intermediate_size: 11008,
            attention_head_count: 16,
            attention_head_dim: 128,
            kv_head_count: 8,
            vocab_size: 151936,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: NativeMoeConfig::default(),
            glm_router: Default::default(),
            tensors: vec![
                tensor(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    vec![151936, 2048],
                ),
                tensor(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    vec![2048],
                ),
                tensor(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    vec![151936, 2048],
                ),
                tensor(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    vec![2048],
                ),
                tensor(
                    "model.layers.0.self_attn.qkv_proj.weight",
                    NativeTensorRole::AttentionQkvPacked,
                    Some(0),
                    vec![4096, 2048],
                ),
                tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    vec![2048, 2048],
                ),
                tensor(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    vec![2048],
                ),
                tensor(
                    "model.layers.0.mlp.gate_up_proj.weight",
                    NativeTensorRole::FfnGateUpPacked,
                    Some(0),
                    vec![8192, 2048],
                ),
                tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    vec![2048, 4096],
                ),
                tensor(
                    "model.layers.1.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(1),
                    vec![2048],
                ),
                tensor(
                    "model.layers.1.self_attn.qkv_proj.weight",
                    NativeTensorRole::AttentionQkvPacked,
                    Some(1),
                    vec![4096, 2048],
                ),
                tensor(
                    "model.layers.1.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(1),
                    vec![2048, 2048],
                ),
                tensor(
                    "model.layers.1.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(1),
                    vec![2048],
                ),
                tensor(
                    "model.layers.1.mlp.gate_up_proj.weight",
                    NativeTensorRole::FfnGateUpPacked,
                    Some(1),
                    vec![8192, 2048],
                ),
                tensor(
                    "model.layers.1.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(1),
                    vec![2048, 4096],
                ),
            ],
        }
    }

    fn split_layer_manifest_with_value_from_key() -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.attention_value_from_key_layers = vec![1];
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![2048, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(1),
                vec![1024, 2048],
            ),
        ]);
        manifest
    }

    fn moe_layer_manifest() -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.hidden_size = 2816;
        manifest.attention_head_count = 8;
        manifest.attention_head_dim = 256;
        manifest.kv_head_count = 2;
        manifest.vocab_size = 262144;
        manifest.tie_word_embeddings = true;
        manifest
            .tensors
            .retain(|tensor| tensor.role != NativeTensorRole::LmHead);
        for tensor in &mut manifest.tensors {
            match tensor.role {
                NativeTensorRole::TokenEmbedding => tensor.shape = vec![262144, 2816],
                NativeTensorRole::FinalNorm => tensor.shape = vec![2816],
                NativeTensorRole::AttentionNorm | NativeTensorRole::FfnNorm => {
                    tensor.shape = vec![2816]
                }
                NativeTensorRole::AttentionQkvPacked => tensor.shape = vec![3072, 2816],
                NativeTensorRole::AttentionO => tensor.shape = vec![2816, 2048],
                NativeTensorRole::FfnGateUpPacked => tensor.shape = vec![4224, 2816],
                NativeTensorRole::FfnDown => tensor.shape = vec![2816, 2112],
                _ => {}
            }
        }
        manifest.moe = NativeMoeConfig {
            expert_count: Some(128),
            experts_per_token: Some(8),
            expert_intermediate_size: Some(704),
        };
        manifest.tensors.extend([
            tensor(
                "model.layers.0.router.proj.weight",
                NativeTensorRole::FfnGateInp,
                Some(0),
                vec![128, 2816],
            ),
            tensor(
                "model.layers.0.router.scale",
                NativeTensorRole::FfnGateInpScale,
                Some(0),
                vec![2816],
            ),
            tensor(
                "model.layers.0.experts.gate_up_proj.weight",
                NativeTensorRole::FfnGateUpExpsPacked,
                Some(0),
                vec![128, 1408, 2816],
            ),
            tensor(
                "model.layers.0.experts.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                Some(0),
                vec![128, 2816, 704],
            ),
            tensor(
                "model.layers.0.experts.down_proj.scale",
                NativeTensorRole::FfnDownExpsScale,
                Some(0),
                vec![128],
            ),
            tensor(
                "model.layers.1.router.proj.weight",
                NativeTensorRole::FfnGateInp,
                Some(1),
                vec![128, 2816],
            ),
            tensor(
                "model.layers.1.router.scale",
                NativeTensorRole::FfnGateInpScale,
                Some(1),
                vec![2816],
            ),
            tensor(
                "model.layers.1.experts.gate_proj.weight",
                NativeTensorRole::FfnGateExps,
                Some(1),
                vec![128, 704, 2816],
            ),
            tensor(
                "model.layers.1.experts.up_proj.weight",
                NativeTensorRole::FfnUpExps,
                Some(1),
                vec![128, 704, 2816],
            ),
            tensor(
                "model.layers.1.experts.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                Some(1),
                vec![128, 2816, 704],
            ),
        ]);
        manifest
    }

    fn switch_moe_manifest(model_family: &str, include_shared_expert: bool) -> NativeModelManifest {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = model_family.to_string();
        manifest.layer_count = 1;
        manifest.moe = NativeMoeConfig {
            expert_count: Some(4),
            experts_per_token: Some(2),
            expert_intermediate_size: Some(512),
        };
        manifest.tensors.retain(|tensor| {
            tensor.layer_index != Some(1)
                && !matches!(
                    tensor.role,
                    NativeTensorRole::FfnGateUpPacked | NativeTensorRole::FfnDown
                )
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.0.mlp.gate.weight",
                NativeTensorRole::FfnGateInp,
                Some(0),
                vec![4, 2048],
            ),
            tensor(
                "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                NativeTensorRole::FfnGateExps,
                Some(0),
                vec![4, 512, 2048],
            ),
            tensor(
                "model.layers.0.mlp.switch_mlp.up_proj.weight",
                NativeTensorRole::FfnUpExps,
                Some(0),
                vec![4, 512, 2048],
            ),
            tensor(
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                Some(0),
                vec![4, 2048, 512],
            ),
        ]);
        if include_shared_expert {
            manifest.tensors.extend([
                tensor(
                    "model.layers.0.mlp.shared_expert_gate.weight",
                    NativeTensorRole::FfnSharedExpertGateInp,
                    Some(0),
                    vec![1, 2048],
                ),
                tensor(
                    "model.layers.0.mlp.shared_expert.gate_proj.weight",
                    NativeTensorRole::FfnSharedExpertGate,
                    Some(0),
                    vec![512, 2048],
                ),
                tensor(
                    "model.layers.0.mlp.shared_expert.up_proj.weight",
                    NativeTensorRole::FfnSharedExpertUp,
                    Some(0),
                    vec![512, 2048],
                ),
                tensor(
                    "model.layers.0.mlp.shared_expert.down_proj.weight",
                    NativeTensorRole::FfnSharedExpertDown,
                    Some(0),
                    vec![2048, 512],
                ),
            ]);
        }
        manifest
    }

    fn tensor(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: NativeTensorDataType::F16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    #[test]
    fn unique_test_dir_produces_distinct_paths_in_a_burst() {
        let paths = (0..1024)
            .map(|_| unique_test_dir("burst"))
            .collect::<Vec<_>>();
        let unique = paths.iter().cloned().collect::<BTreeSet<_>>();

        assert_eq!(paths.len(), unique.len());
    }

    #[test]
    fn native_model_artifacts_load_valid_packed_manifest() {
        let manifest = packed_layer_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts =
            NativeModelArtifacts::from_dir(&dir).expect("packed manifest should validate");

        assert_eq!(artifacts.manifest().model_family, "qwen3_dense");
        assert_eq!(
            artifacts.summary(),
            NativeModelArtifactsSummary {
                model_family: "qwen3_dense".to_string(),
                tensor_format: NativeTensorFormat::Safetensors,
                source_quantization: None,
                runtime_status: NativeRuntimeStatus::default(),
                layer_count: 2,
                tensor_count: 15,
                tie_word_embeddings: false,
                is_moe: false,
                is_hybrid_attention: false,
                hybrid_full_attention_interval: None,
            }
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn qwen35_linear_attention_defaults_missing_full_interval() {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "qwen3_5".to_string();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: None,
            num_value_heads: Some(32),
            num_key_heads: Some(16),
            key_head_dim: Some(128),
            value_head_dim: Some(128),
            conv_kernel_dim: Some(4),
        };
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts =
            NativeModelArtifacts::from_dir(&dir).expect("Qwen3.5 should inherit interval 4");

        assert_eq!(
            artifacts
                .manifest()
                .linear_attention
                .resolved_full_attention_interval(&artifacts.manifest().model_family),
            Some(QWEN3_5_DEFAULT_FULL_ATTENTION_INTERVAL)
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn non_qwen35_linear_attention_requires_full_interval() {
        let mut manifest = packed_layer_manifest();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: None,
            num_value_heads: Some(32),
            num_key_heads: Some(16),
            key_head_dim: Some(128),
            value_head_dim: Some(128),
            conv_kernel_dim: Some(4),
        };
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("non-Qwen3.5 manifests must carry an explicit interval");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };

        assert!(message.contains("linear_attention.full_attention_interval"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_runtime_not_ready_manifest() {
        let mut manifest = packed_layer_manifest();
        manifest.runtime_status = NativeRuntimeStatus {
            ready: false,
            blockers: vec!["qwen35_quantized_gguf_native_runtime_not_implemented".to_string()],
            notes: Vec::new(),
        };
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let err = NativeModelArtifacts::from_dir(&dir)
            .expect_err("runtime-not-ready manifest should fail closed");
        let message = err.to_string();
        assert!(message.contains("not runtime ready"));
        assert!(message.contains("qwen35_quantized_gguf_native_runtime_not_implemented"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_attn_output_gate_with_packed_qkv() {
        let mut manifest = packed_layer_manifest();
        manifest.attn_output_gate = true;
        for tensor in &mut manifest.tensors {
            if tensor.role == NativeTensorRole::AttentionQkvPacked {
                tensor.shape[0] = 6144;
            }
        }
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("packed attn_output_gate manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_attn_output_gate_with_split_qkv() {
        let mut manifest = packed_layer_manifest();
        manifest.attn_output_gate = true;
        for tensor in &mut manifest.tensors {
            if tensor.role == NativeTensorRole::AttentionQkvPacked {
                tensor.shape[0] = 6144;
            }
        }
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_norm.weight",
                NativeTensorRole::AttentionQNorm,
                Some(1),
                vec![128],
            ),
            tensor(
                "model.layers.1.self_attn.k_norm.weight",
                NativeTensorRole::AttentionKNorm,
                Some(1),
                vec![128],
            ),
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![4096, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(1),
                vec![1024, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(1),
                vec![1024, 2048],
            ),
        ]);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("split gated-attention manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_expose_tensor_accessors_and_resolved_paths() {
        let manifest = packed_layer_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts =
            NativeModelArtifacts::from_dir(&dir).expect("packed manifest should validate");
        let embedding = artifacts
            .global_tensor(NativeTensorRole::TokenEmbedding)
            .expect("token embedding should resolve");
        let layer_qkv = artifacts
            .layer_tensor(1, NativeTensorRole::AttentionQkvPacked)
            .expect("layer qkv should resolve");

        assert_eq!(artifacts.tensor_specs().len(), 15);
        assert_eq!(embedding.name, "model.embed_tokens.weight");
        assert_eq!(layer_qkv.name, "model.layers.1.self_attn.qkv_proj.weight");
        assert_eq!(
            artifacts.resolve_tensor_path(layer_qkv),
            dir.join("model.safetensors")
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_tied_embeddings_without_lm_head_tensor() {
        let mut manifest = packed_layer_manifest();
        manifest.tie_word_embeddings = true;
        manifest
            .tensors
            .retain(|tensor| tensor.role != NativeTensorRole::LmHead);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("tied embeddings should allow lm_head omission");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_load_valid_moe_manifest() {
        let manifest = moe_layer_manifest();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let artifacts = NativeModelArtifacts::from_dir(&dir).expect("moe manifest should validate");

        assert_eq!(
            artifacts
                .moe_config()
                .and_then(|config| config.expert_count),
            Some(128)
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_bad_quantized_packed_columns() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::U32;
        gate.source_quantized = true;
        gate.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 4,
        });
        gate.shape = vec![8192, 1024];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("wrong packed column count should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("packed quantized shape"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_5_and_6_bit_quantized_packed_columns() {
        for bits in [5, 6] {
            let mut manifest = packed_layer_manifest();
            let gate = manifest
                .tensors
                .iter_mut()
                .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
                .expect("fixture should include packed ffn gate/up");
            gate.dtype = NativeTensorDataType::U32;
            gate.source_quantized = true;
            gate.quantization = Some(NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: 64,
                bits,
            });
            gate.shape = vec![8192, (2048 * u64::from(bits)).div_ceil(32)];
            let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

            NativeModelArtifacts::from_dir(&dir).unwrap_or_else(|error| {
                panic!("{bits}-bit quantized packed columns should validate: {error}")
            });

            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn native_model_artifacts_reject_unbenchmarked_affine_quantization_bits() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::U32;
        gate.source_quantized = true;
        gate.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 7,
        });
        gate.shape = vec![8192, (2048 * 7_u64).div_ceil(32)];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("unbenchmarked affine bit widths should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("quantization bits must be one of"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_u32_tensor_without_source_quantized_flag() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::U32;
        gate.source_quantized = false;
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("u32 tensors should be declared source-quantized");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("dtype u32 but source_quantized is false"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_affine_quantization_on_non_u32_tensor() {
        let mut manifest = packed_layer_manifest();
        let gate = manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateUpPacked)
            .expect("fixture should include packed ffn gate/up");
        gate.dtype = NativeTensorDataType::Bf16;
        gate.source_quantized = true;
        gate.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 4,
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("affine quantization metadata should belong to u32 tensors");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("declares affine quantization"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_8_bit_quantized_moe_router_columns() {
        let mut manifest = moe_layer_manifest();
        let router = manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.layer_index == Some(0) && tensor.role == NativeTensorRole::FfnGateInp
            })
            .expect("fixture should include MoE router projection");
        router.dtype = NativeTensorDataType::U32;
        router.source_quantized = true;
        router.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 8,
        });
        router.shape = vec![128, 704];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("8-bit quantized MoE router should validate with 4 values per u32");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_moe_tensors_without_manifest_config() {
        let mut manifest = moe_layer_manifest();
        manifest.moe = NativeMoeConfig::default();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("missing moe config should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("manifest.moe"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_incomplete_split_moe_experts() {
        let mut manifest = moe_layer_manifest();
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::FfnUpExps)
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("split MoE expert weights should require gate and up tensors");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("ffn_gate_up_exps_packed or ffn_gate_exps/ffn_up_exps"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_mixed_packed_and_split_moe_experts() {
        let mut manifest = moe_layer_manifest();
        manifest.tensors.extend([
            tensor(
                "model.layers.0.experts.gate_proj.weight",
                NativeTensorRole::FfnGateExps,
                Some(0),
                vec![128, 704, 2816],
            ),
            tensor(
                "model.layers.0.experts.up_proj.weight",
                NativeTensorRole::FfnUpExps,
                Some(0),
                vec![128, 704, 2816],
            ),
        ]);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("MoE expert format should be unambiguous per layer");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("must not mix ffn_gate_up_exps_packed"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_qwen3_moe_without_shared_expert() {
        let manifest = switch_moe_manifest("qwen3_moe", false);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("Qwen3 MoE switch experts do not require a shared expert block");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_qwen35_moe_without_shared_expert() {
        let manifest = switch_moe_manifest("qwen3_5", false);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("Qwen3.5 MoE requires the reference shared expert block");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("ffn_shared_expert_gate_inp"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_qwen35_moe_with_shared_expert() {
        let manifest = switch_moe_manifest("qwen3_5", true);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("Qwen3.5 MoE should validate with switch experts and shared expert");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_missing_layer_qkv_roles() {
        let mut manifest = packed_layer_manifest();
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir).expect_err("missing qkv role should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_qkv_packed"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_q_only_kv_shared_layer() {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1)
                && matches!(
                    tensor.role,
                    NativeTensorRole::AttentionQkvPacked
                        | NativeTensorRole::AttentionK
                        | NativeTensorRole::AttentionV
                ))
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![2048, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.q_norm.weight",
                NativeTensorRole::AttentionQNorm,
                Some(1),
                vec![128],
            ),
        ]);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir).expect("Q-only KV-shared layer should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_kv_shared_layer_with_own_kv() {
        let mut manifest = packed_layer_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);
        manifest.tensors.retain(|tensor| {
            !(tensor.layer_index == Some(1) && tensor.role == NativeTensorRole::AttentionQkvPacked)
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![2048, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(1),
                vec![1024, 2048],
            ),
            tensor(
                "model.layers.1.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(1),
                vec![1024, 2048],
            ),
        ]);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("KV-shared layer with packed QKV should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(
            message.contains("KV-shared layer"),
            "unexpected error: {message}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_allow_missing_attention_v_when_value_comes_from_key() {
        let manifest = split_layer_manifest_with_value_from_key();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("attention_value_from_key_layers should allow missing attention_v");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_value_from_key_layers_with_attention_v() {
        let mut manifest = split_layer_manifest_with_value_from_key();
        manifest.tensors.push(tensor(
            "model.layers.1.self_attn.v_proj.weight",
            NativeTensorRole::AttentionV,
            Some(1),
            vec![1024, 2048],
        ));
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("value-from-key layer must not also provide attention_v");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("value-from-key layer 1"));
        assert!(message.contains("attention_v"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_value_from_key_layers_with_packed_qkv() {
        let mut manifest = packed_layer_manifest();
        manifest.attention_value_from_key_layers = vec![1];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("value-from-key layer must not provide packed QKV");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("value-from-key layer 1"));
        assert!(message.contains("attention_qkv_packed"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_out_of_range_attention_value_from_key_layers() {
        let mut manifest = packed_layer_manifest();
        manifest.attention_value_from_key_layers = vec![99];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("out-of-range attention_value_from_key_layers should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_value_from_key_layers"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_mismatched_layer_types_length() {
        let mut manifest = packed_layer_manifest();
        manifest.layer_types = vec!["sliding_attention".to_string()];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("layer_types length mismatch should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("layer_types"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_invalid_kv_shared_source_layer() {
        let mut manifest = packed_layer_manifest();
        manifest.kv_shared_source_layers.insert(1, 99);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("bad KV source should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("kv_shared_source_layers"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_cross_type_kv_shared_source_layer() {
        let mut manifest = packed_layer_manifest();
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("bad KV source should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("cannot reuse source"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_zero_interleaved_attention_fields() {
        let mut manifest = packed_layer_manifest();
        manifest.rope_theta_swa = Some(0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("zero rope_theta_swa should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("rope_theta_swa"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_parent_escaping_tensor_paths() {
        let mut manifest = packed_layer_manifest();
        manifest.tensors[0].file = PathBuf::from("../escape.safetensors");
        let (dir, manifest) = write_fixture(manifest, &["model.safetensors"]);
        fs::write(dir.join("..").join("escape.safetensors"), vec![0_u8; 16])
            .expect("escape file should write");
        fs::write(
            dir.join(AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should rewrite");

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("parent path traversal should fail closed");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("must not escape root_dir"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_hidden_size_shape_mismatches() {
        let mut manifest = packed_layer_manifest();
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::AttentionNorm && tensor.layer_index == Some(0)
            })
            .expect("attention norm should exist")
            .shape = vec![1024];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("hidden-size mismatch should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_norm"));
        assert!(message.contains("2048"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_ffn_intermediate_shape_mismatches() {
        let mut manifest = packed_layer_manifest();
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::FfnDown && tensor.layer_index == Some(1)
            })
            .expect("ffn down should exist")
            .shape = vec![2048, 2048];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("ffn intermediate mismatch should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("ffn_down"));
        assert!(message.contains("4096"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_attention_q_norm_shape_mismatches() {
        let mut manifest = packed_layer_manifest();
        manifest.tensors.push(tensor(
            "model.layers.0.self_attn.q_norm.weight",
            NativeTensorRole::AttentionQNorm,
            Some(0),
            vec![2048],
        ));
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("q norm shape mismatch should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_q_norm"));
        assert!(message.contains("128"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_attention_o_input_dim_mismatches() {
        let mut manifest = packed_layer_manifest();
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| {
                tensor.role == NativeTensorRole::AttentionO && tensor.layer_index == Some(0)
            })
            .expect("attention o should exist")
            .shape = vec![2048, 1024];
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("attention o mismatch should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_o"));
        assert!(message.contains("2048"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_non_positive_rope_theta() {
        let mut manifest = packed_layer_manifest();
        manifest.rope_theta = Some(0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error =
            NativeModelArtifacts::from_dir(&dir).expect_err("non-positive rope theta should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("rope_theta"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_non_positive_query_pre_attn_scalar() {
        let mut manifest = packed_layer_manifest();
        manifest.query_pre_attn_scalar = Some(0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("non-positive query pre attention scalar should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("query_pre_attn_scalar"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn native_model_artifacts_reject_non_positive_attention_logit_softcap() {
        let mut manifest = packed_layer_manifest();
        manifest.attention_logit_softcap = Some(0);
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        let error = NativeModelArtifacts::from_dir(&dir)
            .expect_err("non-positive attention softcap should fail");
        let NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("attention_logit_softcap"));

        let _ = fs::remove_dir_all(dir);
    }
}
