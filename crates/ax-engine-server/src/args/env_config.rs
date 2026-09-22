//! Process-environment configuration for the ax-engine-server serving path.
//!
//! [`ServerEnvConfig`] is parsed once at process start-up (see
//! [`ServerEnvConfig::from_env`]) and carried in
//! [`crate::app_state::AppState`], so request-path code (the generation
//! worker, the embeddings handlers, and the metrics renderer) reads
//! resolved fields instead of the process environment. Tests build
//! configs without touching the process environment through
//! [`ServerEnvConfig::from_vars`].
//!
//! The accepted values, defaults, and key names here are byte-for-byte the
//! pre-config behavior of the per-site readers this module replaced; the
//! per-key doc comments on the struct fields are the single source of
//! truth for operators.

use std::time::Duration;

use crate::embeddings::microbatch::{
    embedding_microbatch_max_batch_from_raw, embedding_microbatch_queue_capacity_from_raw,
    embedding_microbatch_window_from_raw,
};
use crate::embeddings::{parse_embedding_max_tokens, parse_embedding_timeout_ms};
use crate::generation::service::{
    resolve_adaptive_prefill_latency_tokens, resolve_exec_arbiter_max_concurrent,
    resolve_long_prefill_exclusive, resolve_sibling_engine_step_burst,
    resolve_worker_recycle_after_ticks,
};
use crate::openai::embeddings::{
    DEFAULT_EMBED_MAX_BATCH_TOKENS, DEFAULT_EMBED_MAX_TOKENS, DEFAULT_EMBED_TIMEOUT_MS,
};

pub(crate) const AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT_ENV: &str =
    "AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT";
pub(crate) const AX_SERVER_LONG_PREFILL_EXCLUSIVE_ENV: &str = "AX_SERVER_LONG_PREFILL_EXCLUSIVE";
pub(crate) const AX_SERVER_LONG_PREFILL_WARM_ENV: &str = "AX_SERVER_LONG_PREFILL_WARM";
pub(crate) const AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS_ENV: &str =
    "AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS";
pub(crate) const AX_SERVER_WORKER_RECYCLE_AFTER_TICKS_ENV: &str =
    "AX_SERVER_WORKER_RECYCLE_AFTER_TICKS";
pub(crate) const AX_SERVER_SIBLING_ENGINE_STEP_BURST_ENV: &str =
    "AX_SERVER_SIBLING_ENGINE_STEP_BURST";
pub(crate) const AX_SERVER_SCHED_DEBUG_ENV: &str = "AX_SERVER_SCHED_DEBUG";
pub(crate) const AX_ENGINE_EMBED_MAX_TOKENS_ENV: &str = "AX_ENGINE_EMBED_MAX_TOKENS";
pub(crate) const AX_ENGINE_EMBED_MAX_BATCH_TOKENS_ENV: &str = "AX_ENGINE_EMBED_MAX_BATCH_TOKENS";
pub(crate) const AX_ENGINE_EMBED_TIMEOUT_MS_ENV: &str = "AX_ENGINE_EMBED_TIMEOUT_MS";
pub(crate) const AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS_ENV: &str =
    "AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS";
pub(crate) const AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH_ENV: &str =
    "AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH";
pub(crate) const AX_ENGINE_EMBED_MICROBATCH_QUEUE_CAPACITY_ENV: &str =
    "AX_ENGINE_EMBED_MICROBATCH_QUEUE_CAPACITY";
pub(crate) const AX_MLX_BATCHED_DECODE_MAX_ENV: &str = "AX_MLX_BATCHED_DECODE_MAX";

/// Start-up-resolved serving configuration sourced from the process
/// environment.
///
/// Every field documents its key, default, and accepted values. Parsing is
/// pure (see [`ServerEnvConfig::from_vars`]); the live constructor
/// [`ServerEnvConfig::from_env`] reads the process environment exactly
/// once, in `main`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ServerEnvConfig {
    /// `AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT` (default `1` = exclusive):
    /// how many distinct model workers may hold Metal/MLX arbiter turns
    /// together. Accepted: an integer `1..=8`; values `< 1` or `> 8` clamp
    /// back (values below the floor resolve to 1, above the cap to 8), and
    /// empty/unparseable values fall back to 1. Whitespace is trimmed.
    pub(crate) exec_arbiter_max_concurrent: usize,
    /// `AX_SERVER_LONG_PREFILL_EXCLUSIVE` (default **on**): multi-token
    /// prefill quanta force an exclusive arbiter window. Accepted: `0`,
    /// `false`, or `off` (words case-insensitive) disable it; anything
    /// else, including unset, keeps it on. Whitespace is trimmed.
    pub(crate) long_prefill_exclusive: bool,
    /// `AX_SERVER_LONG_PREFILL_WARM` (default **off**): opt-in long S1
    /// Gemma prefill warm-up after multi-model publish. Accepted: exactly
    /// `1`, `true`, `TRUE`, `yes`, or `YES` (case-sensitive, **no
    /// trimming**); every other value, including ` True ` or `1 `, is off.
    pub(crate) long_prefill_warm: bool,
    /// `AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS` (default `64`): the
    /// starting sibling-active prefill quantum for the adaptive isolation
    /// controller. Accepted: any positive integer; `0`, empty, or
    /// unparseable values fall back to the default. Whitespace is trimmed.
    pub(crate) adaptive_prefill_latency_tokens: u32,
    /// `AX_SERVER_WORKER_RECYCLE_AFTER_TICKS` (default `0` = off): rebuild
    /// the engine session after this many worker engine ticks at the next
    /// fully idle moment. Accepted: any `u64`; empty/unparseable values
    /// fall back to 0. Whitespace is trimmed.
    pub(crate) worker_recycle_after_ticks: u64,
    /// `AX_SERVER_SIBLING_ENGINE_STEP_BURST` (default `16`, capped at 64):
    /// engine steps per worker tick while a sibling model is active.
    /// Accepted: any positive integer, capped at the full stream burst
    /// (64); `0`, empty, or unparseable values fall back to the default.
    /// Whitespace is trimmed.
    pub(crate) sibling_engine_step_burst: usize,
    /// `AX_SERVER_SCHED_DEBUG` (default **off**): per-tick scheduler
    /// eprintln traces. Presence-based: any value (including empty or,
    /// on the live path, non-UTF-8) enables it.
    pub(crate) sched_debug: bool,
    /// `AX_ENGINE_EMBED_MAX_TOKENS` (default `8192`): per-item embedding
    /// input token cap shared by `/v1/embeddings`, the gRPC embeddings
    /// RPC, and `/v1/embeddings/records` (total chunk tokens there).
    /// Accepted: any positive integer; `0`, empty, or unparseable values
    /// fall back to the default. Whitespace is trimmed.
    pub(crate) embed_max_tokens: usize,
    /// `AX_ENGINE_EMBED_MAX_BATCH_TOKENS` (default `8192 * 64` = `524288`):
    /// maximum total tokens accepted across a whole `/v1/embeddings` batch
    /// (the sum of every item's token count). Accepted: any positive integer;
    /// `0`, empty, or unparseable values fall back to the default. Whitespace
    /// is trimmed.
    pub(crate) embed_max_batch_tokens: usize,
    /// `AX_ENGINE_EMBED_TIMEOUT_MS` for `/v1/embeddings` (default
    /// `30000`): per-request embedding timeout. Accepted: any positive
    /// integer; `0`, empty, or unparseable values fall back to the
    /// default. Whitespace is trimmed. The `/v1/embeddings/records`
    /// endpoint intentionally keeps its own longer default; see
    /// [`Self::embed_records_timeout_ms`].
    pub(crate) embed_timeout_ms: u64,
    /// `AX_ENGINE_EMBED_TIMEOUT_MS` for `/v1/embeddings/records` (default
    /// `60000`): the records endpoint chunk-embeds whole documents and
    /// keeps a longer timeout than `/v1/embeddings` from the same key.
    /// Accepted values and trimming match [`Self::embed_timeout_ms`].
    pub(crate) embed_records_timeout_ms: u64,
    /// `AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS` (default `2`, capped at
    /// 100): coalescing window for the embedding microbatcher. Accepted:
    /// any `u64`; unparseable values fall back to the default and the cap
    /// is applied after resolution. **No trimming.**
    pub(crate) embed_microbatch_window: Duration,
    /// `AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH` (default `32`, clamped to
    /// 1..=512): maximum coalesced embedding batch size. Accepted: any
    /// `usize`; unparseable values fall back to the default and the clamp
    /// is applied after resolution. **No trimming.**
    pub(crate) embed_microbatch_max_batch: usize,
    /// `AX_ENGINE_EMBED_MICROBATCH_QUEUE_CAPACITY` (default `1024`,
    /// clamped to 64..=8192): microbatch queue capacity. Accepted: any
    /// `usize`; unparseable values fall back to the default and the clamp
    /// is applied after resolution. **No trimming.**
    pub(crate) embed_microbatch_queue_capacity: usize,
    /// `AX_MLX_BATCHED_DECODE_MAX` (default `8`): the batched-decode
    /// cohort cap exported as the `ax_runtime_max_batch_size` gauge.
    /// This field MIRRORS the MLX runner's own parse of the same key (see
    /// [`mirror_mlx_batched_decode_cohort_cap`]) so the exported gauge
    /// can never disagree with the runner's effective cohort: **no
    /// trimming**, values `< 1` or unparseable fall back to 8.
    pub(crate) mlx_batched_decode_max: u64,
}

impl ServerEnvConfig {
    /// Pure constructor: resolve every field from a variable lookup, so
    /// tests build configs without touching the process environment.
    /// `vars` returns `None` for an unset key.
    pub(crate) fn from_vars(vars: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            exec_arbiter_max_concurrent: resolve_exec_arbiter_max_concurrent(
                vars(AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT_ENV).as_deref(),
            ),
            long_prefill_exclusive: resolve_long_prefill_exclusive(
                vars(AX_SERVER_LONG_PREFILL_EXCLUSIVE_ENV).as_deref(),
            ),
            long_prefill_warm: resolve_long_prefill_warm(
                vars(AX_SERVER_LONG_PREFILL_WARM_ENV).as_deref(),
            ),
            adaptive_prefill_latency_tokens: resolve_adaptive_prefill_latency_tokens(
                vars(AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS_ENV).as_deref(),
            ),
            worker_recycle_after_ticks: resolve_worker_recycle_after_ticks(
                vars(AX_SERVER_WORKER_RECYCLE_AFTER_TICKS_ENV).as_deref(),
            ),
            sibling_engine_step_burst: resolve_sibling_engine_step_burst(
                vars(AX_SERVER_SIBLING_ENGINE_STEP_BURST_ENV).as_deref(),
            ),
            sched_debug: vars(AX_SERVER_SCHED_DEBUG_ENV).is_some(),
            embed_max_tokens: parse_embedding_max_tokens(
                vars(AX_ENGINE_EMBED_MAX_TOKENS_ENV),
                DEFAULT_EMBED_MAX_TOKENS,
            ),
            embed_max_batch_tokens: parse_embedding_max_tokens(
                vars(AX_ENGINE_EMBED_MAX_BATCH_TOKENS_ENV),
                DEFAULT_EMBED_MAX_BATCH_TOKENS,
            ),
            embed_timeout_ms: parse_embedding_timeout_ms(
                vars(AX_ENGINE_EMBED_TIMEOUT_MS_ENV),
                DEFAULT_EMBED_TIMEOUT_MS,
            ),
            embed_records_timeout_ms: parse_embedding_timeout_ms(
                vars(AX_ENGINE_EMBED_TIMEOUT_MS_ENV),
                crate::embeddings::records::DEFAULT_EMBED_RECORDS_TIMEOUT_MS,
            ),
            embed_microbatch_window: embedding_microbatch_window_from_raw(
                vars(AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS_ENV).as_deref(),
            ),
            embed_microbatch_max_batch: embedding_microbatch_max_batch_from_raw(
                vars(AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH_ENV).as_deref(),
            ),
            embed_microbatch_queue_capacity: embedding_microbatch_queue_capacity_from_raw(
                vars(AX_ENGINE_EMBED_MICROBATCH_QUEUE_CAPACITY_ENV).as_deref(),
            ),
            mlx_batched_decode_max: mirror_mlx_batched_decode_cohort_cap(
                vars(AX_MLX_BATCHED_DECODE_MAX_ENV).as_deref(),
            ),
        }
    }

    /// Parse from the live process environment. Call exactly once, at
    /// start-up in `main`, and thread the result through
    /// [`crate::app_state::AppState`].
    pub(crate) fn from_env() -> Self {
        let mut config = Self::from_vars(|key| std::env::var(key).ok());
        // Presence semantics use `var_os`: any value, including a
        // non-UTF-8 one, enables the debug trace; `std::env::var` would
        // drop non-UTF-8 values and silently disable it.
        config.sched_debug = std::env::var_os(AX_SERVER_SCHED_DEBUG_ENV).is_some();
        config
    }
}

impl Default for ServerEnvConfig {
    fn default() -> Self {
        // The all-keys-unset configuration.
        Self::from_vars(|_| None)
    }
}

/// Resolve `AX_SERVER_LONG_PREFILL_WARM` from a raw value: exactly `1`,
/// `true`, `TRUE`, `yes`, or `YES` (case-sensitive, no trimming) is on;
/// unset or any other value is off.
fn resolve_long_prefill_warm(raw: Option<&str>) -> bool {
    matches!(
        raw,
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

/// Mirror of the MLX runner's `AX_MLX_BATCHED_DECODE_MAX` parse.
///
/// The server crate has no dependency on `ax-engine-mlx`, so this function
/// replicates the runner's exact resolution (`crates/ax-engine-mlx/src/
/// runner/mod.rs`, `AxEngineRunner` construction): the raw value is parsed
/// as `usize` **without trimming**, values `< 1` fall back, and the
/// default is 8. Whitespace-padded values therefore resolve to the
/// default here even though most other keys trim -- that is deliberate,
/// because the runner does not trim either, and the exported
/// `ax_runtime_max_batch_size` gauge must report the cohort the runner
/// actually built. `golden_runner_cohort_cap_parse_still_matches_the_mirror`
/// fails when the runner's parse drifts from this mirror.
pub(crate) fn mirror_mlx_batched_decode_cohort_cap(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.parse::<usize>().ok())
        .filter(|&cap| cap >= 1)
        .map(|cap| cap as u64)
        .unwrap_or(8)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Build a lookup that returns `value` for exactly one key (`None`
    /// models an unset key so tests never touch the process environment).
    /// The value lifetime is generic so closures can borrow test-local
    /// strings without requiring `'static`.
    fn single_var<'a>(
        key: &'static str,
        value: Option<&'a str>,
    ) -> impl Fn(&str) -> Option<String> + 'a {
        move |probe| {
            let value = (probe == key).then_some(value)?;
            value.map(str::to_string)
        }
    }

    #[test]
    fn all_keys_unset_matches_default_config() {
        assert_eq!(
            ServerEnvConfig::from_vars(|_| None),
            ServerEnvConfig::default()
        );
    }

    #[test]
    fn default_config_pins_every_unset_default() {
        let config = ServerEnvConfig::default();
        assert_eq!(config.exec_arbiter_max_concurrent, 1);
        assert!(config.long_prefill_exclusive);
        assert!(!config.long_prefill_warm);
        assert_eq!(config.adaptive_prefill_latency_tokens, 64);
        assert_eq!(config.worker_recycle_after_ticks, 0);
        assert_eq!(config.sibling_engine_step_burst, 16);
        assert!(!config.sched_debug);
        assert_eq!(config.embed_max_tokens, 8192);
        assert_eq!(config.embed_max_batch_tokens, 8192 * 64);
        assert_eq!(config.embed_timeout_ms, 30_000);
        assert_eq!(config.embed_records_timeout_ms, 60_000);
        assert_eq!(config.embed_microbatch_window, Duration::from_millis(2));
        assert_eq!(config.embed_microbatch_max_batch, 32);
        assert_eq!(config.embed_microbatch_queue_capacity, 1024);
        assert_eq!(config.mlx_batched_decode_max, 8);
    }

    #[test]
    fn exec_arbiter_max_concurrent_round_trips() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT_ENV, value))
                .exec_arbiter_max_concurrent
        };
        assert_eq!(resolve(None), 1);
        assert_eq!(resolve(Some("1")), 1);
        assert_eq!(resolve(Some("2")), 2);
        assert_eq!(resolve(Some("8")), 8);
        // Above the cap clamps down; below the floor or garbage falls back.
        assert_eq!(resolve(Some("99")), 8);
        assert_eq!(resolve(Some("0")), 1);
        assert_eq!(resolve(Some("-1")), 1);
        assert_eq!(resolve(Some("two")), 1);
        assert_eq!(resolve(Some("")), 1);
        // This key trims.
        assert_eq!(resolve(Some(" 4 ")), 4);
    }

    #[test]
    fn long_prefill_exclusive_round_trips() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_SERVER_LONG_PREFILL_EXCLUSIVE_ENV, value))
                .long_prefill_exclusive
        };
        assert!(resolve(None));
        assert!(resolve(Some("1")));
        assert!(resolve(Some("junk")));
        assert!(!resolve(Some("0")));
        assert!(!resolve(Some("false")));
        assert!(!resolve(Some("FALSE")));
        assert!(!resolve(Some("Off")));
        assert!(!resolve(Some(" off ")));
    }

    #[test]
    fn long_prefill_warm_round_trips_exact_forms_only() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_SERVER_LONG_PREFILL_WARM_ENV, value))
                .long_prefill_warm
        };
        assert!(resolve(Some("1")));
        assert!(resolve(Some("true")));
        assert!(resolve(Some("TRUE")));
        assert!(resolve(Some("yes")));
        assert!(resolve(Some("YES")));
        // No trimming and no other casing: these all stay off.
        assert!(!resolve(None));
        assert!(!resolve(Some("0")));
        assert!(!resolve(Some("on")));
        assert!(!resolve(Some("True")));
        assert!(!resolve(Some(" true")));
        assert!(!resolve(Some("1 ")));
    }

    #[test]
    fn adaptive_prefill_latency_tokens_round_trips() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(
                AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS_ENV,
                value,
            ))
            .adaptive_prefill_latency_tokens
        };
        assert_eq!(resolve(None), 64);
        assert_eq!(resolve(Some("96")), 96);
        assert_eq!(resolve(Some("1")), 1);
        assert_eq!(resolve(Some("0")), 64);
        assert_eq!(resolve(Some("")), 64);
        assert_eq!(resolve(Some("many")), 64);
        assert_eq!(resolve(Some(" 80 ")), 80);
    }

    #[test]
    fn worker_recycle_after_ticks_round_trips() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_SERVER_WORKER_RECYCLE_AFTER_TICKS_ENV, value))
                .worker_recycle_after_ticks
        };
        assert_eq!(resolve(None), 0);
        assert_eq!(resolve(Some("0")), 0);
        assert_eq!(resolve(Some("4096")), 4096);
        assert_eq!(resolve(Some("soon")), 0);
        assert_eq!(resolve(Some(" 8 ")), 8);
    }

    #[test]
    fn sibling_engine_step_burst_round_trips() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_SERVER_SIBLING_ENGINE_STEP_BURST_ENV, value))
                .sibling_engine_step_burst
        };
        assert_eq!(resolve(None), 16);
        assert_eq!(resolve(Some("1")), 1);
        assert_eq!(resolve(Some("16")), 16);
        assert_eq!(resolve(Some("64")), 64);
        // Capped at the full stream burst; zero/garbage falls back.
        assert_eq!(resolve(Some("512")), 64);
        assert_eq!(resolve(Some("0")), 16);
        assert_eq!(resolve(Some("big")), 16);
        assert_eq!(resolve(Some(" 2 ")), 2);
    }

    #[test]
    fn sched_debug_is_presence_based() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_SERVER_SCHED_DEBUG_ENV, value)).sched_debug
        };
        assert!(!resolve(None));
        assert!(resolve(Some("")));
        assert!(resolve(Some("1")));
        assert!(resolve(Some("anything")));
    }

    #[test]
    fn embed_max_tokens_round_trips() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_ENGINE_EMBED_MAX_TOKENS_ENV, value))
                .embed_max_tokens
        };
        assert_eq!(resolve(None), 8192);
        assert_eq!(resolve(Some("16384")), 16384);
        assert_eq!(resolve(Some(" 16384 ")), 16384);
        assert_eq!(resolve(Some("0")), 8192);
        assert_eq!(resolve(Some("-1")), 8192);
        assert_eq!(resolve(Some("many")), 8192);
        assert_eq!(resolve(Some("")), 8192);
    }

    #[test]
    fn embed_max_batch_tokens_round_trips() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_ENGINE_EMBED_MAX_BATCH_TOKENS_ENV, value))
                .embed_max_batch_tokens
        };
        assert_eq!(resolve(None), 8192 * 64);
        assert_eq!(resolve(Some("1048576")), 1048576);
        assert_eq!(resolve(Some(" 1048576 ")), 1048576);
        assert_eq!(resolve(Some("0")), 8192 * 64);
        assert_eq!(resolve(Some("-1")), 8192 * 64);
        assert_eq!(resolve(Some("many")), 8192 * 64);
        assert_eq!(resolve(Some("")), 8192 * 64);
    }

    #[test]
    fn embed_timeout_keeps_both_endpoint_defaults_from_one_key() {
        let openai = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_ENGINE_EMBED_TIMEOUT_MS_ENV, value))
                .embed_timeout_ms
        };
        let records = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_ENGINE_EMBED_TIMEOUT_MS_ENV, value))
                .embed_records_timeout_ms
        };
        // Unset: the two endpoints keep their historical distinct defaults.
        assert_eq!(openai(None), 30_000);
        assert_eq!(records(None), 60_000);
        // One shared key still drives both endpoints when set.
        assert_eq!(openai(Some("45000")), 45_000);
        assert_eq!(records(Some("45000")), 45_000);
        assert_eq!(openai(Some(" 45000 ")), 45_000);
        assert_eq!(records(Some(" 45000 ")), 45_000);
        assert_eq!(openai(Some("0")), 30_000);
        assert_eq!(records(Some("0")), 60_000);
        assert_eq!(openai(Some("fast")), 30_000);
        assert_eq!(records(Some("fast")), 60_000);
    }

    #[test]
    fn embed_microbatch_keys_round_trip_without_trimming() {
        let window = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS_ENV, value))
                .embed_microbatch_window
        };
        let max_batch = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH_ENV, value))
                .embed_microbatch_max_batch
        };
        let capacity = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(
                AX_ENGINE_EMBED_MICROBATCH_QUEUE_CAPACITY_ENV,
                value,
            ))
            .embed_microbatch_queue_capacity
        };
        assert_eq!(window(None), Duration::from_millis(2));
        assert_eq!(window(Some("5")), Duration::from_millis(5));
        // Cap after resolution, and clamps beat defaults for parseable input.
        assert_eq!(window(Some("1000")), Duration::from_millis(100));
        assert_eq!(window(Some("0")), Duration::from_millis(0));
        assert_eq!(window(Some("junk")), Duration::from_millis(2));
        // No trimming: padded values are unparseable and fall back.
        assert_eq!(window(Some(" 5 ")), Duration::from_millis(2));
        assert_eq!(max_batch(None), 32);
        assert_eq!(max_batch(Some("64")), 64);
        assert_eq!(max_batch(Some("0")), 1);
        assert_eq!(max_batch(Some("9999")), 512);
        assert_eq!(max_batch(Some(" 64 ")), 32);
        assert_eq!(capacity(None), 1024);
        assert_eq!(capacity(Some("2048")), 2048);
        assert_eq!(capacity(Some("0")), 64);
        assert_eq!(capacity(Some("99999")), 8192);
        assert_eq!(capacity(Some(" 2048 ")), 1024);
    }

    #[test]
    fn mlx_batched_decode_max_mirrors_the_runner_parse() {
        let resolve = |value: Option<&str>| {
            ServerEnvConfig::from_vars(single_var(AX_MLX_BATCHED_DECODE_MAX_ENV, value))
                .mlx_batched_decode_max
        };
        assert_eq!(resolve(None), 8);
        assert_eq!(resolve(Some("4")), 4);
        assert_eq!(resolve(Some("1")), 1);
        assert_eq!(resolve(Some("16")), 16);
        assert_eq!(resolve(Some("0")), 8);
        assert_eq!(resolve(Some("-1")), 8);
        assert_eq!(resolve(Some("junk")), 8);
        // The runner does NOT trim: padded values fail its integer parse
        // and fall back to 8. This is the exact server/runner drift the
        // mirror removes -- the gauge now reports the runner's cohort.
        assert_eq!(resolve(Some(" 4")), 8);
        assert_eq!(resolve(Some("4 ")), 8);
        assert_eq!(resolve(Some(" 4 ")), 8);
    }

    #[test]
    fn golden_runner_cohort_cap_parse_still_matches_the_mirror() {
        // Golden pin on the runner's own source: if the ax-engine-mlx
        // runner ever changes how it parses AX_MLX_BATCHED_DECODE_MAX
        // (trim, type, floor, or default), this fails and the mirror in
        // this crate must be updated in the same change.
        let runner_source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../ax-engine-mlx/src/runner/mod.rs");
        let Ok(source) = std::fs::read_to_string(&runner_source) else {
            // Standalone checkout without the sibling crate: the golden
            // source pin only applies inside the workspace.
            return;
        };
        let anchor = source
            .find(AX_MLX_BATCHED_DECODE_MAX_ENV)
            .expect("runner source should still read AX_MLX_BATCHED_DECODE_MAX");
        let window_end = source.len().min(anchor + 250);
        let parse_window = &source[anchor..window_end];
        assert!(
            parse_window.contains("parse::<usize>"),
            "runner cohort-cap parse changed type; update the mirror:\n{parse_window}"
        );
        assert!(
            parse_window.contains("unwrap_or(8)"),
            "runner cohort-cap default changed; update the mirror:\n{parse_window}"
        );
        assert!(
            !parse_window.contains(".trim()"),
            "runner cohort-cap parse now trims; update the mirror:\n{parse_window}"
        );
    }

    #[test]
    fn process_env_reads_stay_out_of_the_serving_path() {
        // Grep guard: the serving path must receive its environment knobs
        // through ServerEnvConfig, never read the process environment.
        // `env::var(` catches both `std::env::var(` and the aliased
        // `use std::env; env::var(` form (var_os is intentionally not
        // matched; no request-path var_os readers remain).
        //
        // Allow-list:
        // - `src/args.rs` and `src/args/**`: start-up arg/env fallbacks and
        //   ServerEnvConfig::from_env itself.
        // - `src/main.rs`: process start-up (tracing filter).
        // - `src/model_load.rs`: load-path readers outside the agreed
        //   request-path boundary (AX_SERVER_LOAD_MEMORY_PREFLIGHT,
        //   AX_SERVER_MULTI_MODEL_PREFILL_ISOLATION); follow-up work.
        // - `src/metadata.rs`: model-family probe (AX_MLX_GEMMA4_VIDEO)
        //   with its own OnceLock cache; follow-up work.
        let allowed = |relative: &str| {
            matches!(
                relative,
                "args.rs" | "main.rs" | "model_load.rs" | "metadata.rs"
            ) || relative.starts_with("args/")
        };
        let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut violations = Vec::new();
        let mut pending = vec![src_root.clone()];
        while let Some(dir) = pending.pop() {
            let entries = std::fs::read_dir(&dir).expect("src tree should be readable");
            for entry in entries {
                let path = entry.expect("src entry should be readable").path();
                if path.is_dir() {
                    pending.push(path);
                    continue;
                }
                if path.extension().is_none_or(|ext| ext != "rs") {
                    continue;
                }
                let relative = path
                    .strip_prefix(&src_root)
                    .expect("entry should live under src")
                    .to_string_lossy()
                    .replace('\\', "/");
                if allowed(&relative) {
                    continue;
                }
                let contents = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("{relative} should be readable: {error}"));
                for (index, line) in contents.lines().enumerate() {
                    if line.contains("env::var(") {
                        violations.push(format!("{relative}:{}", index + 1));
                    }
                }
            }
        }
        assert!(
            violations.is_empty(),
            "process-env reads outside the start-up boundary (src/args/, src/main.rs, and \
             the documented allow-list); resolve the key once in ServerEnvConfig instead: \
             {violations:?}"
        );
    }
}
