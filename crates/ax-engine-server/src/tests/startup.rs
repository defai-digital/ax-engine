use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ax_engine_sdk::SelectedBackend;
use axum::Router;
use axum::routing::any;

use crate::app_state::startup_worker_warmup_passes;
use crate::args::ServerEnvConfig;
use crate::{should_warm_http_completions, warm_http_completions_path};

use super::fixtures::{TEST_MODEL_ID, llama_cpp_state};

#[tokio::test]
async fn delegated_startup_does_not_issue_generation_warmup_requests() {
    let state = llama_cpp_state();
    let requests = Arc::new(AtomicUsize::new(0));
    let observed = requests.clone();
    let app = Router::new().fallback(any(move || {
        let observed = observed.clone();
        async move {
            observed.fetch_add(1, Ordering::SeqCst);
        }
    }));

    for selected_backend in [SelectedBackend::MlxLmDelegated, SelectedBackend::LlamaCpp] {
        let mut live = state.snapshot();
        live.runtime_report.selected_backend = selected_backend;
        state.swap_live(live);
        warm_http_completions_path(&app, &state, TEST_MODEL_ID).await;
    }

    assert_eq!(requests.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn native_mlx_startup_issues_three_generation_warmup_requests() {
    let state = llama_cpp_state();
    let mut live = state.snapshot();
    live.runtime_report.selected_backend = SelectedBackend::Mlx;
    state.swap_live(live);

    let requests = Arc::new(AtomicUsize::new(0));
    let observed = requests.clone();
    let app = Router::new().fallback(any(move || {
        let observed = observed.clone();
        async move {
            observed.fetch_add(1, Ordering::SeqCst);
        }
    }));

    warm_http_completions_path(&app, &state, TEST_MODEL_ID).await;

    assert_eq!(requests.load(Ordering::SeqCst), 3);
}

#[test]
fn flash_next_startup_warmup_once_preserves_other_profiles_and_default() {
    assert!(!ServerEnvConfig::default().flash_next_startup_warmup_once);
    let env = ServerEnvConfig::from_vars(|key| {
        (key == "AX_SERVER_FLASH_NEXT_STARTUP_WARMUP_ONCE").then(|| " on ".to_owned())
    });
    assert!(env.flash_next_startup_warmup_once);
    for (family, slots, enabled, expected) in [
        (
            Some("qwen4_exp"),
            Some(1),
            env.flash_next_startup_warmup_once,
            1,
        ),
        (Some("qwen4_exp"), Some(1), false, 3),
        (Some("qwen4_exp"), Some(2), true, 3),
        (Some("qwen4_exp"), None, true, 3),
        (Some("qwen3_5"), Some(1), true, 3),
        (None, Some(1), true, 3),
    ] {
        assert_eq!(
            startup_worker_warmup_passes(family, slots, enabled),
            expected
        );
    }
}

#[test]
fn streamed_native_mlx_skips_http_generation_warmup() {
    assert!(should_warm_http_completions(SelectedBackend::Mlx, false));
    assert!(!should_warm_http_completions(SelectedBackend::Mlx, true));
    assert!(!should_warm_http_completions(
        SelectedBackend::LlamaCpp,
        false,
    ));
}
