use ax_engine_sdk::DelegatedHttpTimeouts;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

mod artifacts;
mod presets;
mod session;

pub use presets::{ServerPreset, render_presets};

pub const API_KEY_ENV: &str = "AX_ENGINE_API_KEY";
pub const DEFAULT_INFERENCE_PORT: u16 = 31_418;
const MODEL_ARTIFACTS_ENV: &str = "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR";
const DEFAULT_MODEL_ID: &str = "qwen3";

const MAX_CONCURRENT_REQUESTS_ENV: &str = "AX_ENGINE_MAX_CONCURRENT_REQUESTS";
const MAX_CONCURRENT_REQUESTS_PER_MODEL_ENV: &str = "AX_ENGINE_MAX_CONCURRENT_REQUESTS_PER_MODEL";
const MAX_REQUEST_BODY_BYTES_ENV: &str = "AX_ENGINE_MAX_REQUEST_BODY_BYTES";
const REQUEST_TIMEOUT_SECS_ENV: &str = "AX_ENGINE_REQUEST_TIMEOUT_SECS";
const GRPC_REQUEST_TIMEOUT_SECS_ENV: &str = "AX_ENGINE_GRPC_REQUEST_TIMEOUT_SECS";
const RATE_LIMIT_RPS_ENV: &str = "AX_ENGINE_RATE_LIMIT_RPS";
const RATE_LIMIT_BURST_ENV: &str = "AX_ENGINE_RATE_LIMIT_BURST";
const STREAM_IDLE_TIMEOUT_SECS_ENV: &str = "AX_ENGINE_STREAM_IDLE_TIMEOUT_SECS";
const STREAM_MAX_DURATION_SECS_ENV: &str = "AX_ENGINE_STREAM_MAX_DURATION_SECS";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PreviewSupportTier {
    MlxCertified,
    MlxPreview,
    MlxLmDelegated,
    LlamaCpp,
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

    #[arg(long = "port", default_value_t = DEFAULT_INFERENCE_PORT)]
    pub port: u16,

    #[arg(long = "model-id", default_value = "", hide_default_value = true)]
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

    #[arg(long = "support-tier", value_enum, default_value_t = PreviewSupportTier::MlxPreview)]
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

    /// Opt into fair multi-prefill progress under residual budget (default OFF).
    /// Improves progress fairness among concurrent text prefills; does **not**
    /// enable GPU continuous batching or claim wall-clock overlap.
    #[arg(long = "multi-prefill-fair", default_value_t = false)]
    pub multi_prefill_fair: bool,

    /// Per-request text prefill token cap when `--multi-prefill-fair` is set.
    /// `0` (default) uses the session `block_size_tokens`.
    #[arg(long = "max-prefill-tokens-per-request-per-step", default_value_t = 0)]
    pub max_prefill_tokens_per_request_per_step: u32,

    /// Max concurrent text prefills per step when fair mode is on.
    /// `0` (default) means unlimited (still subject to free-block headroom).
    #[arg(long = "max-inflight-prefill-requests", default_value_t = 0)]
    pub max_inflight_prefill_requests: u32,

    /// When set, also bind a tonic gRPC server at this address. Omit to run
    /// HTTP only. Format `host:port`, e.g. `127.0.0.1:50051`.
    #[arg(long = "grpc-bind-address")]
    pub grpc_bind_address: Option<String>,

    /// Opt-in cap on engine jobs shared by HTTP and gRPC. A permit remains
    /// held until blocking generation, streaming, embedding, or stepwise work
    /// reaches a terminal state, even if its transport times out or disconnects.
    /// Unset (or non-positive) means no limit. Falls back to
    /// AX_ENGINE_MAX_CONCURRENT_REQUESTS. Saturated HTTP calls receive 429 and
    /// saturated gRPC calls receive RESOURCE_EXHAUSTED.
    #[arg(long = "max-concurrent-requests")]
    pub max_concurrent_requests: Option<usize>,

    /// Opt-in cap on concurrent engine jobs for each loaded model. This is
    /// enforced in addition to the process-wide cap, so one hot model cannot
    /// consume every configured slot. Unset (or non-positive) means no
    /// per-model cap. Falls back to
    /// AX_ENGINE_MAX_CONCURRENT_REQUESTS_PER_MODEL.
    #[arg(long = "max-concurrent-requests-per-model")]
    pub max_concurrent_requests_per_model: Option<usize>,

    /// Cap on request body size, in bytes. Falls back to
    /// AX_ENGINE_MAX_REQUEST_BODY_BYTES; unset (or non-positive) keeps the
    /// built-in 256 MiB default (processed multimodal payloads can be
    /// large).
    #[arg(long = "max-request-body-bytes")]
    pub max_request_body_bytes: Option<usize>,

    /// Opt-in per-request timeout in seconds, applied to the HTTP router
    /// and, by default, the gRPC server too (see
    /// --grpc-request-timeout-secs to diverge). Unset (or non-positive)
    /// means no timeout. Falls back to AX_ENGINE_REQUEST_TIMEOUT_SECS. For
    /// streaming endpoints this bounds time to the first byte, not the
    /// whole stream — see --stream-idle-timeout-secs and
    /// --stream-max-duration-secs for stream-lifetime deadlines.
    #[arg(long = "request-timeout-secs")]
    pub request_timeout_secs: Option<u64>,

    /// Per-request timeout, in seconds, for the gRPC server only. Unset
    /// falls back to AX_ENGINE_GRPC_REQUEST_TIMEOUT_SECS, then to
    /// --request-timeout-secs (today's shared-timeout behavior) — gRPC's
    /// streaming RPCs can legitimately run far longer than typical HTTP
    /// calls, so this lets an operator diverge them without losing the
    /// shared default.
    #[arg(long = "grpc-request-timeout-secs")]
    pub grpc_request_timeout_secs: Option<u64>,

    /// Per-client request-rate limit, in requests per second. Clients are
    /// keyed by bearer token when present, otherwise by peer IP. Unset
    /// (or non-positive) disables rate limiting. Falls back to
    /// AX_ENGINE_RATE_LIMIT_RPS.
    #[arg(long = "rate-limit-rps")]
    pub rate_limit_rps: Option<f64>,

    /// Burst capacity for --rate-limit-rps, in requests. Falls back to
    /// AX_ENGINE_RATE_LIMIT_BURST; when --rate-limit-rps is set but this is
    /// not, burst defaults to the rate itself (one second of headroom).
    #[arg(long = "rate-limit-burst")]
    pub rate_limit_burst: Option<f64>,

    /// Idle deadline for SSE/stream responses, in seconds: if no event is
    /// produced for this long, the stream ends with an error event instead
    /// of hanging indefinitely. Unset (or non-positive) disables it,
    /// preserving today's behavior. Falls back to
    /// AX_ENGINE_STREAM_IDLE_TIMEOUT_SECS.
    #[arg(long = "stream-idle-timeout-secs")]
    pub stream_idle_timeout_secs: Option<u64>,

    /// Hard cap on total stream duration, in seconds, regardless of event
    /// activity. Unset (or non-positive) disables it, preserving today's
    /// behavior. Falls back to AX_ENGINE_STREAM_MAX_DURATION_SECS.
    #[arg(long = "stream-max-duration-secs")]
    pub stream_max_duration_secs: Option<u64>,

    /// Idle-evict non-default resident models after this many seconds without
    /// an admitted request (multi-model serving). The default model is never
    /// evicted, and eviction only runs while the server is otherwise idle.
    /// Unset (or non-positive) disables eviction. Falls back to
    /// AX_ENGINE_MODEL_IDLE_TIMEOUT_SECS.
    #[arg(long = "model-idle-timeout-secs")]
    pub model_idle_timeout_secs: Option<u64>,

    /// Opt-in mDNS / DNS-SD advertisement so AX Serving agents can discover
    /// this server on the local LAN (`_ax-engine._tcp`). Requires a non-loopback
    /// bind host (typically `--host 0.0.0.0`). See docs/LAN-DISCOVERY.md.
    #[arg(long = "advertise-lan", default_value_t = false)]
    pub advertise_lan: bool,

    /// Optional cluster / namespace label for LAN isolation (TXT `cluster`).
    #[arg(long = "lan-cluster")]
    pub lan_cluster: Option<String>,

    /// DNS-SD instance name. Defaults to hostname or `ax-engine`.
    #[arg(long = "lan-instance-name")]
    pub lan_instance_name: Option<String>,

    /// IPv4 address published in mDNS and discovery URLs. Defaults to a
    /// detected private interface address when `--host` is unspecified.
    #[arg(long = "lan-advertise-host")]
    pub lan_advertise_host: Option<String>,

    /// Allow mDNS LAN advertisement without an API key (`auth=open`).
    /// Refused by default: keyless advertise is discoverable by anyone on
    /// the LAN. Prefer `--api-key` / `AX_ENGINE_API_KEY`.
    #[arg(long = "allow-open-lan", default_value_t = false)]
    pub allow_open_lan: bool,
}

const ADVERTISE_LAN_ENV: &str = "AX_ENGINE_ADVERTISE_LAN";
const LAN_CLUSTER_ENV: &str = "AX_ENGINE_LAN_CLUSTER";
const LAN_INSTANCE_NAME_ENV: &str = "AX_ENGINE_LAN_INSTANCE_NAME";
const LAN_ADVERTISE_HOST_ENV: &str = "AX_ENGINE_LAN_ADVERTISE_HOST";
const ALLOW_OPEN_LAN_ENV: &str = "AX_ENGINE_ALLOW_OPEN_LAN";

impl ServerArgs {
    pub fn resolved_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var(API_KEY_ENV).ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    pub fn resolved_advertise_lan(&self) -> bool {
        if self.advertise_lan {
            return true;
        }
        matches!(
            std::env::var(ADVERTISE_LAN_ENV)
                .ok()
                .as_deref()
                .map(str::trim)
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    }

    /// Whether keyless mDNS advertise is explicitly allowed.
    pub fn resolved_allow_open_lan(&self) -> bool {
        if self.allow_open_lan {
            return true;
        }
        matches!(
            std::env::var(ALLOW_OPEN_LAN_ENV)
                .ok()
                .as_deref()
                .map(str::trim)
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    }

    pub fn resolved_lan_cluster(&self) -> Option<String> {
        self.lan_cluster
            .clone()
            .or_else(|| std::env::var(LAN_CLUSTER_ENV).ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    pub fn resolved_lan_instance_name(&self) -> String {
        self.lan_instance_name
            .clone()
            .or_else(|| std::env::var(LAN_INSTANCE_NAME_ENV).ok())
            .or_else(|| std::env::var("HOSTNAME").ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "ax-engine".into())
    }

    pub fn resolved_lan_advertise_host(&self) -> Option<String> {
        self.lan_advertise_host
            .clone()
            .or_else(|| std::env::var(LAN_ADVERTISE_HOST_ENV).ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    pub(crate) fn resolved_max_concurrent_requests(&self) -> Option<usize> {
        super::routes::parse_max_concurrent_requests(
            self.max_concurrent_requests
                .map(|value| value.to_string())
                .or_else(|| std::env::var(MAX_CONCURRENT_REQUESTS_ENV).ok()),
        )
    }

    pub(crate) fn resolved_max_concurrent_requests_per_model(&self) -> Option<usize> {
        super::routes::parse_max_concurrent_requests(
            self.max_concurrent_requests_per_model
                .map(|value| value.to_string())
                .or_else(|| std::env::var(MAX_CONCURRENT_REQUESTS_PER_MODEL_ENV).ok()),
        )
    }

    pub(crate) fn resolved_max_request_body_bytes(&self) -> usize {
        super::routes::parse_max_request_body_bytes(
            self.max_request_body_bytes
                .map(|value| value.to_string())
                .or_else(|| std::env::var(MAX_REQUEST_BODY_BYTES_ENV).ok()),
        )
        .unwrap_or(super::DEFAULT_MAX_REQUEST_BODY_BYTES)
    }

    pub(crate) fn resolved_request_timeout(&self) -> Option<std::time::Duration> {
        super::routes::parse_request_timeout_secs(
            self.request_timeout_secs
                .map(|value| value.to_string())
                .or_else(|| std::env::var(REQUEST_TIMEOUT_SECS_ENV).ok()),
        )
        .map(std::time::Duration::from_secs)
    }

    /// Falls back to [`Self::resolved_request_timeout`] when unset, so the
    /// gRPC server keeps today's shared-timeout behavior by default.
    pub(crate) fn resolved_grpc_request_timeout(&self) -> Option<std::time::Duration> {
        super::routes::parse_request_timeout_secs(
            self.grpc_request_timeout_secs
                .map(|value| value.to_string())
                .or_else(|| std::env::var(GRPC_REQUEST_TIMEOUT_SECS_ENV).ok()),
        )
        .map(std::time::Duration::from_secs)
        .or_else(|| self.resolved_request_timeout())
    }

    pub(crate) fn resolved_rate_limit(&self) -> Option<crate::rate_limit::RateLimitConfig> {
        fn parse_positive_f64(value: Option<String>) -> Option<f64> {
            value
                .as_deref()
                .map(str::trim)
                .filter(|raw| !raw.is_empty())
                .and_then(|raw| raw.parse::<f64>().ok())
                .filter(|rate| *rate > 0.0)
        }

        let rps = parse_positive_f64(
            self.rate_limit_rps
                .map(|value| value.to_string())
                .or_else(|| std::env::var(RATE_LIMIT_RPS_ENV).ok()),
        )?;
        let burst = parse_positive_f64(
            self.rate_limit_burst
                .map(|value| value.to_string())
                .or_else(|| std::env::var(RATE_LIMIT_BURST_ENV).ok()),
        )
        .unwrap_or(rps);
        Some(crate::rate_limit::RateLimitConfig { rps, burst })
    }

    /// Idle timeout for evicting non-default resident models; `None` disables.
    pub(crate) fn resolved_model_idle_timeout(&self) -> Option<std::time::Duration> {
        self.model_idle_timeout_secs
            .map(|secs| secs.to_string())
            .or_else(|| std::env::var("AX_ENGINE_MODEL_IDLE_TIMEOUT_SECS").ok())
            .as_deref()
            .map(str::trim)
            .filter(|raw| !raw.is_empty())
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|secs| *secs > 0)
            .map(std::time::Duration::from_secs)
    }

    pub(crate) fn resolved_stream_deadlines(
        &self,
    ) -> crate::generation::streaming::StreamDeadlines {
        fn parse_positive_secs(value: Option<String>) -> Option<std::time::Duration> {
            value
                .as_deref()
                .map(str::trim)
                .filter(|raw| !raw.is_empty())
                .and_then(|raw| raw.parse::<u64>().ok())
                .filter(|secs| *secs > 0)
                .map(std::time::Duration::from_secs)
        }

        crate::generation::streaming::StreamDeadlines {
            idle_timeout: parse_positive_secs(
                self.stream_idle_timeout_secs
                    .map(|value| value.to_string())
                    .or_else(|| std::env::var(STREAM_IDLE_TIMEOUT_SECS_ENV).ok()),
            ),
            max_duration: parse_positive_secs(
                self.stream_max_duration_secs
                    .map(|value| value.to_string())
                    .or_else(|| std::env::var(STREAM_MAX_DURATION_SECS_ENV).ok()),
            ),
        }
    }
}

#[cfg(test)]
mod tests;
