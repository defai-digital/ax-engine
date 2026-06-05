use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ax_engine_core::NativeModelManifest;
use serde_json::Value;

pub const GEMMA4_ASSISTANT_MTP_CONTRACT_FILE: &str = "ax_gemma4_assistant_mtp.json";
pub const GEMMA4_ASSISTANT_MTP_SCHEMA_VERSION: &str = "ax.gemma4_assistant_mtp.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gemma4AssistantMtpDisableReason {
    None,
    NotGemma4Target,
    MissingConfig,
    DisabledByEnv,
    InvalidConfig,
    UnsupportedAssistantModelType,
    TokenizerMismatch,
    VocabMismatch,
    PairMismatch,
    UnsupportedAssistantConfig,
    UnsupportedKvSharingLayout,
    WeightLoadFailed,
}

impl Gemma4AssistantMtpDisableReason {
    pub fn route_code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::NotGemma4Target => 1,
            Self::MissingConfig => 2,
            Self::DisabledByEnv => 3,
            Self::InvalidConfig => 4,
            Self::UnsupportedAssistantModelType => 5,
            Self::TokenizerMismatch => 6,
            Self::VocabMismatch => 7,
            Self::PairMismatch => 8,
            Self::UnsupportedAssistantConfig => 9,
            Self::UnsupportedKvSharingLayout => 10,
            Self::WeightLoadFailed => 11,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4AssistantMtpConfig {
    pub target_model_id: String,
    pub assistant_model_id: String,
    pub assistant_path: PathBuf,
    pub max_depth: usize,
    pub exact_pair_required: bool,
    pub assistant_hidden_size: usize,
    pub backbone_hidden_size: usize,
    pub assistant_layer_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4AssistantMtpStatus {
    pub configured: bool,
    pub validated: bool,
    pub enabled: bool,
    pub attach_failed: bool,
    pub disable_reason: Gemma4AssistantMtpDisableReason,
    pub max_depth: usize,
    pub config: Option<Gemma4AssistantMtpConfig>,
}

impl Default for Gemma4AssistantMtpStatus {
    fn default() -> Self {
        Self {
            configured: false,
            validated: false,
            enabled: false,
            attach_failed: false,
            disable_reason: Gemma4AssistantMtpDisableReason::MissingConfig,
            max_depth: 0,
            config: None,
        }
    }
}

impl Gemma4AssistantMtpStatus {
    fn disabled(reason: Gemma4AssistantMtpDisableReason, configured: bool) -> Self {
        let attach_failed = configured
            && !matches!(
                reason,
                Gemma4AssistantMtpDisableReason::None
                    | Gemma4AssistantMtpDisableReason::MissingConfig
                    | Gemma4AssistantMtpDisableReason::DisabledByEnv
            );
        Self {
            configured,
            validated: false,
            enabled: false,
            attach_failed,
            disable_reason: reason,
            max_depth: 0,
            config: None,
        }
    }

    fn validated(config: Gemma4AssistantMtpConfig) -> Self {
        let max_depth = config.max_depth;
        Self {
            configured: true,
            validated: true,
            enabled: false,
            attach_failed: false,
            disable_reason: Gemma4AssistantMtpDisableReason::None,
            max_depth,
            config: Some(config),
        }
    }
}

pub fn load_gemma4_assistant_mtp_status(
    target_root: &Path,
    target_manifest: &NativeModelManifest,
) -> Gemma4AssistantMtpStatus {
    if target_manifest.model_family != "gemma4" {
        return Gemma4AssistantMtpStatus::disabled(
            Gemma4AssistantMtpDisableReason::NotGemma4Target,
            false,
        );
    }

    if !gemma4_assistant_mtp_env_enabled() {
        return Gemma4AssistantMtpStatus::disabled(
            Gemma4AssistantMtpDisableReason::DisabledByEnv,
            false,
        );
    }

    let contract_path = target_root.join(GEMMA4_ASSISTANT_MTP_CONTRACT_FILE);
    let Ok(contract_bytes) = fs::read(&contract_path) else {
        return Gemma4AssistantMtpStatus::disabled(
            Gemma4AssistantMtpDisableReason::MissingConfig,
            false,
        );
    };
    let Ok(contract) = serde_json::from_slice::<Value>(&contract_bytes) else {
        return Gemma4AssistantMtpStatus::disabled(
            Gemma4AssistantMtpDisableReason::InvalidConfig,
            true,
        );
    };

    match parse_and_validate_contract(target_root, target_manifest, &contract) {
        Ok(config) => Gemma4AssistantMtpStatus::validated(config),
        Err(reason) => Gemma4AssistantMtpStatus::disabled(reason, true),
    }
}

fn parse_and_validate_contract(
    target_root: &Path,
    target_manifest: &NativeModelManifest,
    contract: &Value,
) -> Result<Gemma4AssistantMtpConfig, Gemma4AssistantMtpDisableReason> {
    let schema = required_str(contract, "schema_version")?;
    if schema != GEMMA4_ASSISTANT_MTP_SCHEMA_VERSION {
        return Err(Gemma4AssistantMtpDisableReason::InvalidConfig);
    }
    let backend = required_str(contract, "backend")?;
    if backend != "gemma4_assistant" {
        return Err(Gemma4AssistantMtpDisableReason::InvalidConfig);
    }

    let target_model_id = required_str(contract, "target_model_id")?.to_string();
    let assistant_model_id = required_str(contract, "assistant_model_id")?.to_string();
    let assistant_path_raw = required_str(contract, "assistant_path")?;
    let exact_pair_required = contract
        .get("pairing")
        .and_then(Value::as_str)
        .map(|s| s == "exact")
        .unwrap_or_else(gemma4_assistant_mtp_require_exact_pair);

    if !is_known_gemma4_assistant_pair(&assistant_model_id, &target_model_id) {
        return Err(Gemma4AssistantMtpDisableReason::PairMismatch);
    }

    let raw_depth = contract
        .get("max_depth")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .clamp(1, usize::MAX as u64) as usize;
    let max_depth = raw_depth.min(gemma4_assistant_mtp_max_depth_cap());

    let assistant_path = resolve_assistant_path(target_root, assistant_path_raw)?;
    let assistant_config = read_json_file(&assistant_path.join("config.json"))?;
    let model_type = assistant_config_model_type(&assistant_config)
        .ok_or(Gemma4AssistantMtpDisableReason::InvalidConfig)?;
    if model_type != "gemma4_assistant" {
        return Err(Gemma4AssistantMtpDisableReason::UnsupportedAssistantModelType);
    }

    let assistant_vocab = assistant_config_vocab_size(&assistant_config)
        .ok_or(Gemma4AssistantMtpDisableReason::InvalidConfig)?;
    if assistant_vocab != target_manifest.vocab_size as u64 {
        return Err(Gemma4AssistantMtpDisableReason::VocabMismatch);
    }
    let assistant_hidden_size = assistant_config_usize(&assistant_config, "hidden_size")
        .ok_or(Gemma4AssistantMtpDisableReason::InvalidConfig)?;
    let backbone_hidden_size = assistant_config
        .get("backbone_hidden_size")
        .and_then(Value::as_u64)
        .ok_or(Gemma4AssistantMtpDisableReason::InvalidConfig)?
        as usize;
    if backbone_hidden_size != target_manifest.hidden_size as usize {
        return Err(Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig);
    }
    let assistant_layer_count = assistant_config_usize(&assistant_config, "num_hidden_layers")
        .ok_or(Gemma4AssistantMtpDisableReason::InvalidConfig)?;
    validate_assistant_architecture(&assistant_config, assistant_layer_count)?;

    validate_tokenizer_compatibility(target_root, &assistant_path)?;

    Ok(Gemma4AssistantMtpConfig {
        target_model_id,
        assistant_model_id,
        assistant_path,
        max_depth,
        exact_pair_required,
        assistant_hidden_size,
        backbone_hidden_size,
        assistant_layer_count,
    })
}

fn required_str<'a>(
    value: &'a Value,
    key: &str,
) -> Result<&'a str, Gemma4AssistantMtpDisableReason> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or(Gemma4AssistantMtpDisableReason::InvalidConfig)
}

fn read_json_file(path: &Path) -> Result<Value, Gemma4AssistantMtpDisableReason> {
    let bytes = fs::read(path).map_err(|_| Gemma4AssistantMtpDisableReason::InvalidConfig)?;
    serde_json::from_slice(&bytes).map_err(|_| Gemma4AssistantMtpDisableReason::InvalidConfig)
}

fn resolve_assistant_path(
    target_root: &Path,
    raw: &str,
) -> Result<PathBuf, Gemma4AssistantMtpDisableReason> {
    let candidate = Path::new(raw);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        target_root.join(candidate)
    };
    let canonical = joined
        .canonicalize()
        .map_err(|_| Gemma4AssistantMtpDisableReason::InvalidConfig)?;
    if !canonical.join("config.json").exists() {
        return Err(Gemma4AssistantMtpDisableReason::InvalidConfig);
    }
    Ok(canonical)
}

fn assistant_config_model_type(value: &Value) -> Option<&str> {
    value.get("model_type").and_then(Value::as_str).or_else(|| {
        value
            .get("text_config")
            .and_then(|text| text.get("model_type"))
            .and_then(Value::as_str)
    })
}

fn assistant_config_vocab_size(value: &Value) -> Option<u64> {
    value.get("vocab_size").and_then(Value::as_u64).or_else(|| {
        value
            .get("text_config")
            .and_then(|text| text.get("vocab_size"))
            .and_then(Value::as_u64)
    })
}

fn assistant_config_usize(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .or_else(|| {
            value
                .get("text_config")
                .and_then(|text| text.get(key))
                .and_then(Value::as_u64)
        })
        .and_then(|v| usize::try_from(v).ok())
}

fn assistant_config_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool).or_else(|| {
        value
            .get("text_config")
            .and_then(|text| text.get(key))
            .and_then(Value::as_bool)
    })
}

fn validate_assistant_architecture(
    value: &Value,
    assistant_layer_count: usize,
) -> Result<(), Gemma4AssistantMtpDisableReason> {
    let Some(num_kv_shared_layers) = assistant_config_usize(value, "num_kv_shared_layers") else {
        return Err(Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig);
    };
    if num_kv_shared_layers != assistant_layer_count {
        return Err(Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig);
    }
    if assistant_config_usize(value, "hidden_size_per_layer_input").unwrap_or(0) != 0 {
        return Err(Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig);
    }
    if assistant_config_usize(value, "vocab_size_per_layer_input").unwrap_or(0) != 0 {
        return Err(Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig);
    }
    if assistant_config_bool(value, "enable_moe_block").unwrap_or(false) {
        return Err(Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig);
    }
    if assistant_config_bool(value, "use_double_wide_mlp").unwrap_or(false) {
        return Err(Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig);
    }
    Ok(())
}

fn validate_tokenizer_compatibility(
    target_root: &Path,
    assistant_path: &Path,
) -> Result<(), Gemma4AssistantMtpDisableReason> {
    let target_tokenizer = fs::read(target_root.join("tokenizer.json"))
        .map_err(|_| Gemma4AssistantMtpDisableReason::TokenizerMismatch)?;
    let assistant_tokenizer = fs::read(assistant_path.join("tokenizer.json"))
        .map_err(|_| Gemma4AssistantMtpDisableReason::TokenizerMismatch)?;
    if target_tokenizer != assistant_tokenizer {
        return Err(Gemma4AssistantMtpDisableReason::TokenizerMismatch);
    }
    Ok(())
}

fn is_known_gemma4_assistant_pair(assistant_model_id: &str, target_model_id: &str) -> bool {
    const KNOWN_TARGETS: &[&str] = &[
        "gemma-4-e2b-it",
        "gemma-4-e4b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    ];
    let assistant = model_id_leaf(assistant_model_id);
    let target = model_id_leaf(target_model_id);
    if !KNOWN_TARGETS.contains(&target.as_str()) {
        return false;
    }
    assistant
        .strip_suffix("-assistant")
        .is_some_and(|prefix| prefix == target)
}

fn model_id_leaf(model_id: &str) -> String {
    model_id
        .rsplit('/')
        .next()
        .unwrap_or(model_id)
        .to_ascii_lowercase()
}

fn gemma4_assistant_mtp_env_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_GEMMA4_ASSISTANT_MTP")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn gemma4_assistant_mtp_max_depth_cap() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(1)
    })
}

fn gemma4_assistant_mtp_require_exact_pair() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_GEMMA4_ASSISTANT_MTP_REQUIRE_EXACT_PAIR")
            .map(|v| v == "1")
            .unwrap_or(true)
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use ax_engine_core::{
        NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus, NativeTensorFormat,
    };

    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-gemma4-mtp-{name}-{nonce}"));
        fs::create_dir_all(&path).expect("temp fixture should create");
        path
    }

    fn manifest(family: &str) -> NativeModelManifest {
        NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: family.to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 1,
            hidden_size: 16,
            intermediate_size: 32,
            attention_head_count: 2,
            attention_head_dim: 8,
            kv_head_count: 1,
            vocab_size: 262144,
            tie_word_embeddings: true,
            rope_theta: None,
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: BTreeMap::new(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: Default::default(),
            mla_attention: Default::default(),
            moe: NativeMoeConfig::default(),
            glm_router: Default::default(),
            weight_sanitize: ax_engine_core::WeightSanitize::None,
            think_start_token_id: None,
            think_end_token_id: None,
            tensors: Vec::new(),
        }
    }

    fn write_valid_fixture(root: &Path) -> PathBuf {
        let assistant = root.join("assistant");
        fs::create_dir_all(&assistant).expect("assistant dir should create");
        fs::write(root.join("tokenizer.json"), br#"{"model":"same"}"#)
            .expect("target tokenizer should write");
        fs::write(assistant.join("tokenizer.json"), br#"{"model":"same"}"#)
            .expect("assistant tokenizer should write");
        fs::write(
            assistant.join("config.json"),
            br#"{
              "model_type":"gemma4_assistant",
              "backbone_hidden_size":16,
              "text_config":{
                "model_type":"gemma4_assistant",
                "vocab_size":262144,
                "hidden_size":8,
                "num_hidden_layers":2,
                "num_kv_shared_layers":2,
                "hidden_size_per_layer_input":0,
                "vocab_size_per_layer_input":0,
                "enable_moe_block":false,
                "use_double_wide_mlp":false
              }
            }"#,
        )
        .expect("assistant config should write");
        fs::write(
            root.join(GEMMA4_ASSISTANT_MTP_CONTRACT_FILE),
            br#"{
              "schema_version": "ax.gemma4_assistant_mtp.v1",
              "backend": "gemma4_assistant",
              "target_model_id": "gemma-4-e2b-it",
              "assistant_model_id": "gemma-4-e2b-it-assistant",
              "assistant_path": "assistant",
              "max_depth": 1,
              "pairing": "exact"
            }"#,
        )
        .expect("contract should write");
        assistant
    }

    #[test]
    fn missing_contract_is_not_attach_failure() {
        let root = temp_root("missing");
        let status = load_gemma4_assistant_mtp_status(&root, &manifest("gemma4"));
        assert_eq!(
            status.disable_reason,
            Gemma4AssistantMtpDisableReason::MissingConfig
        );
        assert!(!status.configured);
        assert!(!status.attach_failed);
    }

    #[test]
    fn non_gemma4_target_does_not_attach() {
        let root = temp_root("nongemma");
        let status = load_gemma4_assistant_mtp_status(&root, &manifest("qwen3"));
        assert_eq!(
            status.disable_reason,
            Gemma4AssistantMtpDisableReason::NotGemma4Target
        );
        assert!(!status.configured);
    }

    #[test]
    fn valid_contract_is_validated_and_ready_for_runtime_attach() {
        let root = temp_root("valid");
        write_valid_fixture(&root);
        let status = load_gemma4_assistant_mtp_status(&root, &manifest("gemma4"));
        assert!(status.configured);
        assert!(status.validated);
        assert!(!status.enabled);
        assert!(!status.attach_failed);
        assert_eq!(status.disable_reason, Gemma4AssistantMtpDisableReason::None);
        assert_eq!(status.max_depth, 1);
        let config = status.config.expect("validated config should be present");
        assert_eq!(config.assistant_hidden_size, 8);
        assert_eq!(config.backbone_hidden_size, 16);
        assert_eq!(config.assistant_layer_count, 2);
    }

    #[test]
    fn unsupported_assistant_model_type_fails_closed() {
        let root = temp_root("badtype");
        let assistant = write_valid_fixture(&root);
        fs::write(
            assistant.join("config.json"),
            br#"{
              "model_type":"gemma4",
              "backbone_hidden_size":16,
              "text_config":{
                "vocab_size":262144,
                "hidden_size":8,
                "num_hidden_layers":2,
                "num_kv_shared_layers":2
              }
            }"#,
        )
        .expect("assistant config should overwrite");
        let status = load_gemma4_assistant_mtp_status(&root, &manifest("gemma4"));
        assert_eq!(
            status.disable_reason,
            Gemma4AssistantMtpDisableReason::UnsupportedAssistantModelType
        );
        assert!(status.attach_failed);
    }

    #[test]
    fn tokenizer_mismatch_fails_closed() {
        let root = temp_root("tok");
        let assistant = write_valid_fixture(&root);
        fs::write(assistant.join("tokenizer.json"), br#"{"model":"other"}"#)
            .expect("assistant tokenizer should overwrite");
        let status = load_gemma4_assistant_mtp_status(&root, &manifest("gemma4"));
        assert_eq!(
            status.disable_reason,
            Gemma4AssistantMtpDisableReason::TokenizerMismatch
        );
        assert!(status.attach_failed);
    }

    #[test]
    fn pair_mismatch_fails_closed() {
        let root = temp_root("pair");
        write_valid_fixture(&root);
        fs::write(
            root.join(GEMMA4_ASSISTANT_MTP_CONTRACT_FILE),
            br#"{
              "schema_version": "ax.gemma4_assistant_mtp.v1",
              "backend": "gemma4_assistant",
              "target_model_id": "gemma-4-e4b-it",
              "assistant_model_id": "gemma-4-e2b-it-assistant",
              "assistant_path": "assistant",
              "max_depth": 1,
              "pairing": "exact"
            }"#,
        )
        .expect("contract should overwrite");
        let status = load_gemma4_assistant_mtp_status(&root, &manifest("gemma4"));
        assert_eq!(
            status.disable_reason,
            Gemma4AssistantMtpDisableReason::PairMismatch
        );
        assert!(status.attach_failed);
    }

    #[test]
    fn backbone_hidden_mismatch_fails_closed() {
        let root = temp_root("backbone");
        let assistant = write_valid_fixture(&root);
        fs::write(
            assistant.join("config.json"),
            br#"{
              "model_type":"gemma4_assistant",
              "backbone_hidden_size":32,
              "text_config":{
                "model_type":"gemma4_assistant",
                "vocab_size":262144,
                "hidden_size":8,
                "num_hidden_layers":2,
                "num_kv_shared_layers":2
              }
            }"#,
        )
        .expect("assistant config should overwrite");
        let status = load_gemma4_assistant_mtp_status(&root, &manifest("gemma4"));
        assert_eq!(
            status.disable_reason,
            Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig
        );
        assert!(status.attach_failed);
    }

    #[test]
    fn non_shared_assistant_layers_fail_closed() {
        let root = temp_root("shared");
        let assistant = write_valid_fixture(&root);
        fs::write(
            assistant.join("config.json"),
            br#"{
              "model_type":"gemma4_assistant",
              "backbone_hidden_size":16,
              "text_config":{
                "model_type":"gemma4_assistant",
                "vocab_size":262144,
                "hidden_size":8,
                "num_hidden_layers":2,
                "num_kv_shared_layers":1
              }
            }"#,
        )
        .expect("assistant config should overwrite");
        let status = load_gemma4_assistant_mtp_status(&root, &manifest("gemma4"));
        assert_eq!(
            status.disable_reason,
            Gemma4AssistantMtpDisableReason::UnsupportedAssistantConfig
        );
        assert!(status.attach_failed);
    }
}
