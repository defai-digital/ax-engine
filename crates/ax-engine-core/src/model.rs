use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION: &str = "ax.native_model.v1";
pub const AX_NATIVE_MODEL_MANIFEST_FILE: &str = "model-manifest.json";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorFormat {
    Safetensors,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorDataType {
    F16,
    Bf16,
    F32,
    I8,
    U8,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTensorRole {
    TokenEmbedding,
    AttentionNorm,
    AttentionQNorm,
    AttentionKNorm,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionQkvPacked,
    AttentionO,
    LinearAttentionInProjQkv,
    LinearAttentionInProjZ,
    LinearAttentionInProjA,
    LinearAttentionInProjB,
    LinearAttentionConv1d,
    LinearAttentionDtBias,
    LinearAttentionALog,
    LinearAttentionNorm,
    LinearAttentionOutProj,
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnGateUpPacked,
    FfnDown,
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
                | Self::AttentionQNorm
                | Self::AttentionKNorm
                | Self::AttentionQ
                | Self::AttentionK
                | Self::AttentionV
                | Self::AttentionQkvPacked
                | Self::AttentionO
                | Self::LinearAttentionInProjQkv
                | Self::LinearAttentionInProjZ
                | Self::LinearAttentionInProjA
                | Self::LinearAttentionInProjB
                | Self::LinearAttentionConv1d
                | Self::LinearAttentionDtBias
                | Self::LinearAttentionALog
                | Self::LinearAttentionNorm
                | Self::LinearAttentionOutProj
                | Self::FfnNorm
                | Self::FfnGate
                | Self::FfnUp
                | Self::FfnGateUpPacked
                | Self::FfnDown
        )
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeLinearAttentionConfig {
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
        self.num_value_heads.is_some()
            || self.num_key_heads.is_some()
            || self.key_head_dim.is_some()
            || self.value_head_dim.is_some()
            || self.conv_kernel_dim.is_some()
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
    pub shape: Vec<u64>,
    pub file: PathBuf,
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NativeModelManifest {
    pub schema_version: String,
    pub model_family: String,
    pub tensor_format: NativeTensorFormat,
    pub layer_count: u32,
    pub hidden_size: u32,
    pub attention_head_count: u32,
    pub attention_head_dim: u32,
    pub kv_head_count: u32,
    pub vocab_size: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<u32>,
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
    #[serde(
        default,
        skip_serializing_if = "NativeLinearAttentionConfig::is_disabled"
    )]
    pub linear_attention: NativeLinearAttentionConfig,
    pub tensors: Vec<NativeTensorSpec>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeModelArtifacts {
    root_dir: PathBuf,
    manifest: NativeModelManifest,
}

impl NativeModelArtifacts {
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
        NativeModelArtifactsSummary {
            model_family: self.manifest.model_family.clone(),
            tensor_format: self.manifest.tensor_format,
            layer_count: self.manifest.layer_count,
            tensor_count: self.manifest.tensors.len() as u32,
            tie_word_embeddings: self.manifest.tie_word_embeddings,
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
    pub layer_count: u32,
    pub tensor_count: u32,
    pub tie_word_embeddings: bool,
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
        if rotary_dim == 0 || rotary_dim % 2 != 0 {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "partial_rotary_factor {factor} yields rotary_dim {rotary_dim} which must be even and > 0"
                ),
            });
        }
    }
    if manifest.linear_attention.is_enabled() {
        validate_positive_optional_field(
            manifest.linear_attention.num_value_heads,
            "linear_attention.num_value_heads",
        )?;
        validate_positive_optional_field(
            manifest.linear_attention.num_key_heads,
            "linear_attention.num_key_heads",
        )?;
        validate_positive_optional_field(
            manifest.linear_attention.key_head_dim,
            "linear_attention.key_head_dim",
        )?;
        validate_positive_optional_field(
            manifest.linear_attention.value_head_dim,
            "linear_attention.value_head_dim",
        )?;
        validate_positive_optional_field(
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
        require_layer_role(roles, NativeTensorRole::FfnNorm, layer_index, "ffn_norm")?;
        require_layer_role(roles, NativeTensorRole::FfnDown, layer_index, "ffn_down")?;

        let has_packed_gate_up = roles.contains(&NativeTensorRole::FfnGateUpPacked);
        let has_split_gate_up =
            roles.contains(&NativeTensorRole::FfnGate) && roles.contains(&NativeTensorRole::FfnUp);
        if !(has_packed_gate_up || has_split_gate_up) {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} must provide ffn_gate_up_packed or ffn_gate/ffn_up",
                    layer_index
                ),
            });
        }

        // Attention QKV/O are required for full-attention layers but optional
        // for mixed-architecture models (e.g. Qwen3.5 linear_attention layers).
        let has_any_attention = roles.contains(&NativeTensorRole::AttentionO)
            || roles.contains(&NativeTensorRole::AttentionQ)
            || roles.contains(&NativeTensorRole::AttentionK)
            || roles.contains(&NativeTensorRole::AttentionQkvPacked);
        let has_any_linear_attention = roles.contains(&NativeTensorRole::LinearAttentionInProjQkv)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjZ)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjA)
            || roles.contains(&NativeTensorRole::LinearAttentionInProjB)
            || roles.contains(&NativeTensorRole::LinearAttentionConv1d)
            || roles.contains(&NativeTensorRole::LinearAttentionDtBias)
            || roles.contains(&NativeTensorRole::LinearAttentionALog)
            || roles.contains(&NativeTensorRole::LinearAttentionNorm)
            || roles.contains(&NativeTensorRole::LinearAttentionOutProj);
        if has_any_attention {
            require_layer_role(
                roles,
                NativeTensorRole::AttentionO,
                layer_index,
                "attention_o",
            )?;
            let has_packed_qkv = roles.contains(&NativeTensorRole::AttentionQkvPacked);
            let has_split_qkv = roles.contains(&NativeTensorRole::AttentionQ)
                && roles.contains(&NativeTensorRole::AttentionK)
                && (roles.contains(&NativeTensorRole::AttentionV)
                    || manifest
                        .attention_value_from_key_layers
                        .contains(&layer_index));
            if !(has_packed_qkv || has_split_qkv) {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} must provide attention_qkv_packed or attention_q/attention_k plus attention_v (or mark the layer in attention_value_from_key_layers)",
                        layer_index
                    ),
                });
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
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionInProjQkv,
                layer_index,
                "linear_attention_in_proj_qkv",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionInProjZ,
                layer_index,
                "linear_attention_in_proj_z",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionInProjA,
                layer_index,
                "linear_attention_in_proj_a",
            )?;
            require_layer_role(
                roles,
                NativeTensorRole::LinearAttentionInProjB,
                layer_index,
                "linear_attention_in_proj_b",
            )?;
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
    }

    validate_native_model_tensor_shapes(manifest)?;

    Ok(())
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
        if let Some(attention_q_norm) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionQNorm,
            Some(layer_index),
        ) {
            expect_vector_shape(
                attention_q_norm,
                u64::from(manifest.attention_head_dim),
                "attention_q_norm",
            )?;
        }
        if let Some(attention_k_norm) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionKNorm,
            Some(layer_index),
        ) {
            expect_vector_shape(
                attention_k_norm,
                u64::from(manifest.attention_head_dim),
                "attention_k_norm",
            )?;
        }
        // Attention O shape validation — only for layers that have attention tensors.
        // The output projection maps from attention output dim back to hidden_size.
        // For standard attention: o_proj shape is [hidden_size, num_heads * head_dim].
        // For gated attention (Qwen3.5): q_proj has 2x rows (queries + gate), but
        // o_proj still maps from num_heads * head_dim, not from q_proj rows.
        if let Some(attention_o) =
            manifest_tensor(manifest, NativeTensorRole::AttentionO, Some(layer_index))
        {
            let o_shape =
                matrix_shape(attention_o).ok_or_else(|| NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor attention_o must be a rank-2 matrix",
                        layer_index
                    ),
                })?;
            if o_shape.0 != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor attention_o must have {} output rows, got {}",
                        layer_index, hidden_size, o_shape.0
                    ),
                });
            }
        }

        let ffn_norm = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::FfnNorm,
            "ffn_norm",
        )?;
        expect_vector_shape(ffn_norm, hidden_size, "ffn_norm")?;

        let ffn_down = required_layer_tensor_spec(
            manifest,
            layer_index,
            NativeTensorRole::FfnDown,
            "ffn_down",
        )?;
        let ffn_down_shape =
            matrix_shape(ffn_down).ok_or_else(|| NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor ffn_down must be a rank-2 matrix",
                    layer_index
                ),
            })?;
        if ffn_down_shape.0 != hidden_size {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor ffn_down must have shape [{}, intermediate_dim], got {:?}",
                    layer_index, hidden_size, ffn_down.shape
                ),
            });
        }

        if let Some(attention_qkv) = manifest_tensor(
            manifest,
            NativeTensorRole::AttentionQkvPacked,
            Some(layer_index),
        ) {
            let q_rows =
                u64::from(manifest.attention_head_count) * u64::from(manifest.attention_head_dim);
            let kv_rows =
                u64::from(manifest.kv_head_count) * u64::from(manifest.attention_head_dim);
            expect_matrix_shape(
                attention_qkv,
                q_rows + kv_rows + kv_rows,
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
                expect_matrix_shape(attention_v, split_dims.kv_rows, hidden_size, "attention_v")?;
            }
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

        let intermediate_dim = if let Some(ffn_gate_up_packed) = manifest_tensor(
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
            if cols != hidden_size {
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
        } else {
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
            if gate_shape.1 != hidden_size {
                return Err(NativeModelError::InvalidManifest {
                    message: format!(
                        "layer {} tensor ffn_gate must have hidden_size {} columns, got {:?}",
                        layer_index, hidden_size, ffn_gate.shape
                    ),
                });
            }
            if up_shape.1 != hidden_size {
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
        };

        if ffn_down_shape.1 != intermediate_dim {
            return Err(NativeModelError::InvalidManifest {
                message: format!(
                    "layer {} tensor ffn_down must have intermediate_dim {} columns, got {:?}",
                    layer_index, intermediate_dim, ffn_down.shape
                ),
            });
        }
    }

    Ok(())
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

fn validate_positive_optional_field(
    value: Option<u32>,
    field_name: &str,
) -> Result<(), NativeModelError> {
    if matches!(value, Some(0)) {
        return Err(NativeModelError::InvalidManifest {
            message: format!("{field_name} must be > 0"),
        });
    }
    Ok(())
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
    let head_dim = head_dim.unwrap_or(u64::from(manifest.attention_head_dim));
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
        if v_cols != hidden_size {
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
            layer_count: 2,
            hidden_size: 2048,
            attention_head_count: 16,
            attention_head_dim: 128,
            kv_head_count: 8,
            vocab_size: 151936,
            tie_word_embeddings: false,
            rope_theta: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            linear_attention: NativeLinearAttentionConfig::default(),
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
                layer_count: 2,
                tensor_count: 15,
                tie_word_embeddings: false,
            }
        );

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
    fn native_model_artifacts_allow_missing_attention_v_when_value_comes_from_key() {
        let manifest = split_layer_manifest_with_value_from_key();
        let (dir, _) = write_fixture(manifest, &["model.safetensors"]);

        NativeModelArtifacts::from_dir(&dir)
            .expect("attention_value_from_key_layers should allow missing attention_v");

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
