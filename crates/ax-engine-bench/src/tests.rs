use super::*;
use std::sync::{Mutex, MutexGuard, OnceLock};

use ax_engine_bench::microbench::{
    MicrobenchProfileExportBlocker, MicrobenchProfileExportBlockerKind,
    MicrobenchProfileExportState,
};

fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("ax-bench env test lock")
}

struct EnvVarRestore {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(prev) => unsafe { std::env::set_var(self.key, prev) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn with_cleared_env_vars<T>(keys: &[&'static str], f: impl FnOnce() -> T) -> T {
    let _guard = env_lock();
    let _restore: Vec<_> = keys
        .iter()
        .map(|&key| {
            let previous = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            EnvVarRestore { key, previous }
        })
        .collect();
    f()
}

#[test]
fn test_parse_duration_hours() {
    assert_eq!(
        parse_duration("24h").unwrap(),
        Duration::from_secs(24 * 3600)
    );
    assert_eq!(parse_duration("8h").unwrap(), Duration::from_secs(8 * 3600));
}

#[test]
fn test_parse_duration_minutes() {
    assert_eq!(parse_duration("5m").unwrap(), Duration::from_secs(300));
    assert_eq!(parse_duration("10m").unwrap(), Duration::from_secs(600));
}

#[test]
fn test_parse_duration_seconds() {
    assert_eq!(parse_duration("300s").unwrap(), Duration::from_secs(300));
}

#[test]
fn test_parse_duration_bare_number() {
    assert_eq!(parse_duration("60").unwrap(), Duration::from_secs(60));
}

#[test]
fn test_parse_duration_invalid() {
    assert!(parse_duration("abc").is_err());
    assert!(parse_duration("1x").is_err());
}

#[test]
fn test_determine_profile_export_action_blocks_by_default() {
    let decision = microbench::MicrobenchProfileExportDecision {
        allowed: false,
        blocker_details: vec![],
        blockers: vec!["blocked".to_string()],
    };
    assert_eq!(
        microbench::determine_profile_export_action(true, false, &decision),
        MicrobenchProfileExportAction::Blocked
    );
}

#[test]
fn test_determine_profile_export_action_allows_override() {
    let decision = microbench::MicrobenchProfileExportDecision {
        allowed: false,
        blocker_details: vec![],
        blockers: vec!["blocked".to_string()],
    };
    assert_eq!(
        microbench::determine_profile_export_action(true, true, &decision),
        MicrobenchProfileExportAction::Write {
            override_used: true
        }
    );
}

#[test]
fn test_determine_profile_export_action_skips_when_not_requested() {
    let decision = microbench::MicrobenchProfileExportDecision {
        allowed: true,
        blocker_details: vec![],
        blockers: vec![],
    };
    assert_eq!(
        microbench::determine_profile_export_action(false, false, &decision),
        MicrobenchProfileExportAction::NotRequested
    );
}

#[test]
fn test_format_export_status_blockers_uses_structured_threshold_details() {
    let status = MicrobenchProfileExportStatus {
        state: MicrobenchProfileExportState::Blocked,
        exit_code: 1,
        requested: true,
        gate_allowed: false,
        override_used: false,
        wrote_profile: false,
        blocker_details: vec![MicrobenchProfileExportBlocker {
            kind: MicrobenchProfileExportBlockerKind::BelowThreshold,
            reason: "decode_matvec.q4_k avg speedup 1.080x below export threshold 1.100x"
                .to_string(),
            rule: Some("decode_matvec.q4_k".to_string()),
            avg_speedup: Some(1.08),
            required_speedup: Some(1.10),
            forced: false,
        }],
        blockers: vec!["legacy blocker".to_string()],
        output_path: Some("/tmp/profile.json".to_string()),
    };

    let lines = format_export_status_blockers(&status);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("kind=below_threshold"));
    assert!(lines[0].contains("rule=decode_matvec.q4_k"));
    assert!(lines[0].contains("avg=1.080x"));
    assert!(lines[0].contains("required=1.100x"));
    assert!(!lines[0].contains("legacy blocker"));
}

#[test]
fn test_format_export_status_blockers_uses_structured_forced_details() {
    let status = MicrobenchProfileExportStatus {
        state: MicrobenchProfileExportState::Blocked,
        exit_code: 1,
        requested: true,
        gate_allowed: false,
        override_used: false,
        wrote_profile: false,
        blocker_details: vec![MicrobenchProfileExportBlocker {
            kind: MicrobenchProfileExportBlockerKind::Forced,
            reason: "forced export block via AX_BENCH_MICROBENCH_FORCE_EXPORT_BLOCK=1".to_string(),
            rule: None,
            avg_speedup: None,
            required_speedup: None,
            forced: true,
        }],
        blockers: vec!["legacy forced blocker".to_string()],
        output_path: None,
    };

    let lines = format_export_status_blockers(&status);
    assert_eq!(
        lines,
        vec![
            "forced export block via AX_BENCH_MICROBENCH_FORCE_EXPORT_BLOCK=1 kind=forced"
                .to_string()
        ]
    );
}

#[test]
fn test_llama_parity_preset_does_not_override_prefill_routing_modes() {
    with_cleared_env_vars(
        &[
            "AX_METAL_F16_KV_CACHE",
            "AX_METAL_PREFILL_FA2_MODE",
            "AX_METAL_PREFILL_FA2_HD128_MODE",
        ],
        || {
            apply_llama_parity_preset();
            assert_eq!(
                std::env::var("AX_METAL_F16_KV_CACHE").ok().as_deref(),
                Some("on")
            );
            assert!(std::env::var_os("AX_METAL_BATCH_F16_IO").is_none());
            assert!(std::env::var_os("AX_METAL_PREFILL_FA2_MODE").is_none());
            assert!(std::env::var_os("AX_METAL_PREFILL_FA2_HD128_MODE").is_none());
        },
    );
}

#[test]
fn test_cli_parse_prefill_gap_accepts_baseline_flags() {
    let cli = Cli::try_parse_from([
        "ax-engine-bench",
        "prefill-gap",
        "--model",
        "./models/Qwen3.5-9B-Q4_K_M.gguf",
        "--prompt-tokens",
        "512",
        "--baseline-prefill-tok-s",
        "720.5",
        "--baseline-label",
        "llama.cpp",
        "--json",
    ])
    .expect("prefill-gap CLI should parse");

    let Command::PrefillGap {
        model,
        prompt_tokens,
        baseline_prefill_tok_s,
        baseline_label,
        json,
        ..
    } = cli.command
    else {
        panic!("expected prefill-gap command");
    };

    assert_eq!(model, "./models/Qwen3.5-9B-Q4_K_M.gguf");
    assert_eq!(prompt_tokens, 512);
    assert_eq!(baseline_prefill_tok_s, Some(720.5));
    assert_eq!(baseline_label.as_deref(), Some("llama.cpp"));
    assert!(json);
}

#[test]
fn test_cli_parse_prefill_gap_rejects_conflicting_baseline_flags() {
    let err = match Cli::try_parse_from([
        "ax-engine-bench",
        "prefill-gap",
        "--model",
        "./models/Qwen3.5-9B-Q4_K_M.gguf",
        "--baseline-json",
        "./baseline.json",
        "--baseline-prefill-tok-s",
        "720.5",
    ]) {
        Ok(_) => panic!("expected conflicting baseline flags to fail"),
        Err(err) => err,
    };

    let message = err.to_string();
    assert!(message.contains("--baseline-json"));
    assert!(message.contains("--baseline-prefill-tok-s"));
}

#[test]
fn test_cli_parse_soak_rejects_zero_tokens_per_iter() {
    let err = match Cli::try_parse_from([
        "ax-engine-bench",
        "soak",
        "--model",
        "./models/Qwen3.5-9B-Q4_K_M.gguf",
        "--tokens-per-iter",
        "0",
    ]) {
        Ok(_) => panic!("expected zero tokens-per-iter to fail"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("--tokens-per-iter"));
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_env_maps_variants() {
    assert_eq!(
        qwen35_prefill_recurrent_state_mode_env(Qwen35PrefillRecurrentStateModeArg::Auto),
        None
    );
    assert_eq!(
        qwen35_prefill_recurrent_state_mode_env(Qwen35PrefillRecurrentStateModeArg::CpuAlias),
        Some("cpu_alias")
    );
    assert_eq!(
        qwen35_prefill_recurrent_state_mode_env(Qwen35PrefillRecurrentStateModeArg::SlotBuffer),
        Some("slot_buffer")
    );
    assert_eq!(
        qwen35_prefill_recurrent_state_mode_env(Qwen35PrefillRecurrentStateModeArg::BackendOwned),
        Some("backend_owned")
    );
}

#[test]
fn test_qwen35_prefill_alpha_beta_storage_mode_env_maps_variants() {
    assert_eq!(
        qwen35_prefill_alpha_beta_storage_mode_env(Qwen35PrefillAlphaBetaStorageModeArg::Auto,),
        None
    );
    assert_eq!(
        qwen35_prefill_alpha_beta_storage_mode_env(Qwen35PrefillAlphaBetaStorageModeArg::F32,),
        Some("f32")
    );
    assert_eq!(
        qwen35_prefill_alpha_beta_storage_mode_env(Qwen35PrefillAlphaBetaStorageModeArg::F16,),
        Some("f16")
    );
}

#[test]
fn test_qwen35_prefill_force_backend_state_batch_env_maps_variants() {
    assert_eq!(qwen35_prefill_force_backend_state_batch_env(false), None);
    assert_eq!(
        qwen35_prefill_force_backend_state_batch_env(true),
        Some("1")
    );
}

#[test]
fn test_local_hd128_prefill_route_env_overrides_force_ax_bc64() {
    let overrides = local_hd128_prefill_route_env_overrides(LocalPrefillHd128Route::AxBc64);
    assert!(overrides.contains(&("AX_METAL_PREFILL_BC64_MODE", Some("on"))));
    assert!(overrides.contains(&("AX_METAL_PREFILL_FA2_HD128_MODE", Some("off"))));
    assert!(overrides.contains(&("AX_METAL_PREFILL_FA2_HALF", Some("off"))));
}

#[test]
fn test_default_prefill_route_compare_routes_covers_auto_bc64_and_fa2_variants() {
    assert_eq!(
        default_prefill_route_compare_routes(),
        vec![
            LocalPrefillHd128RouteArg::Auto,
            LocalPrefillHd128RouteArg::AxBc64,
            LocalPrefillHd128RouteArg::Fa2SimdHd128,
            LocalPrefillHd128RouteArg::Fa2HalfHd128,
        ]
    );
}

#[test]
fn test_cli_parse_prefill_profile_accepts_qwen35_recurrent_state_mode() {
    let cli = Cli::try_parse_from([
        "ax-engine-bench",
        "prefill-profile",
        "--model",
        "./models/Qwen3.5-9B-Q4_K_M.gguf",
        "--qwen35-recurrent-state-mode",
        "backend-owned",
        "--qwen35-alpha-beta-storage-mode",
        "f16",
        "--qwen35-prime-slot-buffers",
        "--qwen35-prewarm-prefill-same-kv",
        "--qwen35-force-backend-state-batch",
        "--local-hd128-route",
        "fa2-half-hd128",
    ])
    .expect("prefill-profile CLI should parse");

    let Command::PrefillProfile {
        qwen35_recurrent_state_mode,
        qwen35_alpha_beta_storage_mode,
        qwen35_prime_slot_buffers,
        qwen35_prewarm_prefill_same_kv,
        qwen35_force_backend_state_batch,
        local_hd128_route,
        ..
    } = cli.command
    else {
        panic!("expected prefill-profile command");
    };

    assert_eq!(
        qwen35_recurrent_state_mode,
        Qwen35PrefillRecurrentStateModeArg::BackendOwned
    );
    assert_eq!(
        qwen35_alpha_beta_storage_mode,
        Qwen35PrefillAlphaBetaStorageModeArg::F16
    );
    assert!(qwen35_prime_slot_buffers);
    assert!(qwen35_prewarm_prefill_same_kv);
    assert!(qwen35_force_backend_state_batch);
    assert_eq!(local_hd128_route, LocalPrefillHd128RouteArg::Fa2HalfHd128);
}

#[test]
fn test_cli_parse_prefill_route_compare_accepts_multiple_routes() {
    let cli = Cli::try_parse_from([
        "ax-engine-bench",
        "prefill-route-compare",
        "--model",
        "./models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf",
        "--samples",
        "5",
        "--route",
        "ax-bc64",
        "--route",
        "fa2-simd-hd128",
    ])
    .expect("prefill-route-compare CLI should parse");

    let Command::PrefillRouteCompare { samples, route, .. } = cli.command else {
        panic!("expected prefill-route-compare command");
    };

    assert_eq!(samples, 5);
    assert_eq!(
        route,
        vec![
            LocalPrefillHd128RouteArg::AxBc64,
            LocalPrefillHd128RouteArg::Fa2SimdHd128,
        ]
    );
}
