use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ax_engine_sdk::SelectedBackend;
use axum::Router;
use axum::routing::any;

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
fn streamed_native_mlx_skips_http_generation_warmup() {
    assert!(should_warm_http_completions(SelectedBackend::Mlx, false));
    assert!(!should_warm_http_completions(SelectedBackend::Mlx, true));
    assert!(!should_warm_http_completions(
        SelectedBackend::LlamaCpp,
        false,
    ));
}
