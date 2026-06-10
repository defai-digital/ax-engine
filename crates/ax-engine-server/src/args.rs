use ax_engine_sdk::{DelegatedHttpTimeouts, KvCompressionConfig};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

mod artifacts;
mod presets;
mod session;

pub use presets::{ServerPreset, render_presets};

pub const API_KEY_ENV: &str = "AX_ENGINE_API_KEY";
const MODEL_ARTIFACTS_ENV: &str = "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PreviewSupportTier {
    MlxCertified,
    MlxPreview,
    MlxLmDelegated,
    LlamaCpp,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum PreviewMlxKvCompression {
    #[default]
    Disabled,
    #[value(name = "turboquant-shadow")]
    TurboQuantShadow,
    #[value(name = "turboquant-fused-experimental")]
    TurboQuantFusedExperimental,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ModelArtifactResolution {
    ExplicitOnly,
    HfCache,
}

/// Speculative-decode posture preset (ADR-022). Bundles the MTP + n-gram gate
/// configuration into one selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum SpeculationProfileArg {
    Auto,
    Coding,
    Agentic,
    Chatbot,
}

impl SpeculationProfileArg {
    /// Canonical lowercase name passed to the runtime resolver.
    pub fn as_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Coding => "coding",
            Self::Agentic => "agentic",
            Self::Chatbot => "chatbot",
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "ax-engine-server", version, about)]
pub struct ServerArgs {
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    #[arg(long = "port", default_value_t = 8080)]
    pub port: u16,

    #[arg(long = "model-id", default_value = "qwen3")]
    pub model_id: String,

    /// Require `Authorization: Bearer <key>` on HTTP API routes. If unset, the
    /// server falls back to AX_ENGINE_API_KEY; empty values disable auth.
    #[arg(long = "api-key")]
    pub api_key: Option<String>,

    #[arg(long = "preset", value_enum, conflicts_with_all = ["model_id", "support_tier"])]
    pub preset: Option<ServerPreset>,

    #[arg(long = "list-presets", default_value_t = false)]
    pub list_presets: bool,

    #[arg(long = "deterministic", default_value_t = true)]
    pub deterministic: bool,

    #[arg(long = "max-batch-tokens", default_value_t = 2048)]
    pub max_batch_tokens: u32,

    #[arg(long = "cache-group-id", default_value_t = 0)]
    pub cache_group_id: u16,

    #[arg(long = "block-size-tokens", default_value_t = 16)]
    pub block_size_tokens: u32,

    #[arg(long = "total-blocks", default_value_t = 1024)]
    pub total_blocks: u32,

    #[arg(long = "mlx", default_value_t = false)]
    pub mlx: bool,

    #[arg(long = "support-tier", value_enum, default_value_t = PreviewSupportTier::LlamaCpp)]
    pub support_tier: PreviewSupportTier,

    #[arg(long = "llama-cli-path", default_value = "llama-cli")]
    pub llama_cli_path: String,

    #[arg(long = "llama-model-path")]
    pub llama_model_path: Option<PathBuf>,

    #[arg(long = "llama-server-url")]
    pub llama_server_url: Option<String>,
    #[arg(long = "mlx-lm-server-url")]
    pub mlx_lm_server_url: Option<String>,

    /// Connect timeout, in seconds, for delegated llama.cpp / mlx_lm HTTP backends.
    #[arg(long = "delegated-http-connect-timeout-secs", default_value_t = DelegatedHttpTimeouts::default_connect_secs())]
    pub delegated_http_connect_timeout_secs: u64,

    /// Read timeout, in seconds, for delegated llama.cpp / mlx_lm HTTP responses.
    #[arg(long = "delegated-http-read-timeout-secs", default_value_t = DelegatedHttpTimeouts::default_io_secs())]
    pub delegated_http_read_timeout_secs: u64,

    /// Write timeout, in seconds, for delegated llama.cpp / mlx_lm HTTP requests.
    #[arg(long = "delegated-http-write-timeout-secs", default_value_t = DelegatedHttpTimeouts::default_io_secs())]
    pub delegated_http_write_timeout_secs: u64,

    #[arg(long = "mlx-model-artifacts-dir")]
    pub mlx_model_artifacts_dir: Option<PathBuf>,

    #[arg(long = "resolve-model-artifacts", value_enum, default_value_t = ModelArtifactResolution::ExplicitOnly)]
    pub resolve_model_artifacts: ModelArtifactResolution,

    #[arg(long = "hf-cache-root")]
    pub hf_cache_root: Option<PathBuf>,

    /// Disable n-gram acceleration and run the direct same-policy decode path.
    /// Useful for establishing clean benchmark comparisons against mlx_lm.
    #[arg(long = "disable-ngram-acceleration", default_value_t = false)]
    pub disable_ngram_acceleration: bool,

    /// Enable n-gram-first drafting inside the MTP verify loop. Disabled by
    /// default after the Gemma 4 12B Phase 4 sweep found pure assistant-MTP is
    /// the stronger default and n-gram stacking is workload-dependent.
    #[arg(
        long = "mlx-mtp-enable-ngram-stacking",
        default_value_t = false,
        conflicts_with = "mlx_mtp_disable_ngram_stacking"
    )]
    pub mlx_mtp_enable_ngram_stacking: bool,

    /// Keep MTP speculation enabled but disable n-gram-first drafting inside
    /// the MTP verify loop. This remains for explicit pure-MTP benchmark
    /// comparisons; it is also the default when the enable flag is absent.
    #[arg(long = "mlx-mtp-disable-ngram-stacking", default_value_t = false)]
    pub mlx_mtp_disable_ngram_stacking: bool,

    /// Speculative-decode posture preset (ADR-022): `auto` (default,
    /// temperature-driven), `coding`, `agentic`, or `chatbot`. Bundles the MTP
    /// and n-gram gate configuration into one selector. Unset falls back to the
    /// `AX_MLX_SPECULATION_PROFILE` env / built-in `auto`.
    #[arg(short = 's', long = "speculation-profile", alias = "spec", value_enum)]
    pub speculation_profile: Option<SpeculationProfileArg>,

    /// Override the MLX prefill chunk size. When unset, the runner uses
    /// `DEFAULT_PREFILL_CHUNK` (2048), matching mlx_lm's default
    /// `prefill_step_size`. Models with MLA layers clamp warm-extend prefill
    /// through `AX_MLX_MLA_PREFILL_CHUNK` / `MLA_DEFAULT_PREFILL_CHUNK` for
    /// prefix-restore equivalence; Qwen GatedDelta layers clamp to the
    /// long-prefill Metal specialization capacity.
    #[arg(long = "prefill-chunk")]
    pub prefill_chunk: Option<usize>,

    /// Experimental MLX KV compression policy. Disabled keeps the existing KV path unchanged.
    #[arg(long = "experimental-mlx-kv-compression", value_enum, default_value_t = PreviewMlxKvCompression::Disabled)]
    pub experimental_mlx_kv_compression: PreviewMlxKvCompression,

    /// Full-precision tail retained when experimental MLX KV compression is enabled.
    #[arg(long = "experimental-mlx-kv-compression-hot-window-tokens", default_value_t = KvCompressionConfig::DEFAULT_HOT_WINDOW_TOKENS)]
    pub experimental_mlx_kv_compression_hot_window_tokens: usize,

    /// Minimum context before experimental MLX KV compression becomes eligible.
    #[arg(long = "experimental-mlx-kv-compression-min-context-tokens", default_value_t = KvCompressionConfig::DEFAULT_MIN_CONTEXT_TOKENS)]
    pub experimental_mlx_kv_compression_min_context_tokens: usize,

    /// When set, also bind a tonic gRPC server at this address. Omit to run
    /// HTTP only. Format `host:port`, e.g. `127.0.0.1:50051`.
    #[arg(long = "grpc-bind-address")]
    pub grpc_bind_address: Option<String>,
}

impl ServerArgs {
    pub fn resolved_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var(API_KEY_ENV).ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }
}

#[cfg(test)]
mod tests;
