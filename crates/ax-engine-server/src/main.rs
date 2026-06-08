#![allow(clippy::collapsible_if)]

use ax_engine_sdk::{EngineSession, EngineSessionConfig};
use clap::Parser;
use std::env;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

mod app_state;
mod args;
mod backends;
mod chat;
mod embeddings;
mod errors;
mod generation;
mod grpc;
mod metadata;
mod openai;
mod routes;
mod tasks;

use app_state::build_app_state;
use args::{ServerArgs, render_presets};
use routes::build_router;

// Processed Gemma4 unified image tensors are JSON-heavy; keep the server cap
// aligned with the Python helper's bounded media fetch policy.
const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

fn log_host_detection_warnings(session_config: &EngineSessionConfig) {
    let host = ax_engine_sdk::current_host_report();
    if let Some(reason) = host.detection_error.as_deref() {
        let selected = session_config.resolved_backend.selected_backend;
        warn!(
            detected_soc = host.detected_soc.as_deref().unwrap_or("unknown Apple Silicon"),
            supported_mlx_runtime = host.supported_mlx_runtime,
            selected_backend = ?selected,
            reason = %reason,
            "ax-engine host SoC detection failed; MLX backends will refuse to start in this \
             environment. Run outside the sandbox or set AX_ALLOW_UNSUPPORTED_HOST=1 only for \
             bring-up. /health surfaces this under runtime.host.detection_error."
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tracing_enabled = init_tracing();

    let args = ServerArgs::parse();
    if args.list_presets {
        println!("{}", render_presets());
        return Ok(());
    }
    let bind_address = args.bind_address();
    let model_id = args.effective_model_id().to_string();
    let support_tier = args.effective_support_tier();
    let session_config = args
        .session_config()
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    log_host_detection_warnings(&session_config);
    let session = EngineSession::new(session_config.clone())?;
    let state = build_app_state(model_id.clone(), session)?;
    let app = build_router(state.clone());
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;

    let grpc_bind_address = args.grpc_bind_address.clone();

    if tracing_enabled {
        info!(
            bind_address = %bind_address,
            grpc_bind_address = ?grpc_bind_address,
            model_id = %model_id,
            support_tier = ?support_tier,
            "ax-engine-server preview listening"
        );
    } else {
        eprintln!(
            "ax-engine-server preview listening on http://{} model_id={} support_tier={:?}",
            bind_address, model_id, support_tier
        );
        if let Some(addr) = grpc_bind_address.as_deref() {
            eprintln!("ax-engine-server gRPC listening on {addr}");
        }
    }

    let http_server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());

    if let Some(addr) = grpc_bind_address {
        let parsed: std::net::SocketAddr = addr.parse().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid --grpc-bind-address {addr}: {e}"),
            )
        })?;
        let grpc_service = grpc::AxEngineGrpcService::new(state).into_server();
        let grpc_server = tonic::transport::Server::builder()
            .add_service(grpc_service)
            .serve_with_shutdown(parsed, shutdown_signal());
        let http_handle = tokio::spawn(http_server.into_future());
        let grpc_handle = tokio::spawn(grpc_server);
        let (http_result, grpc_result) = tokio::join!(http_handle, grpc_handle);
        http_result??;
        grpc_result??;
    } else {
        http_server.await?;
    }
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn init_tracing() -> bool {
    let filter = env::var("AX_ENGINE_SERVER_LOG")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("RUST_LOG")
                .ok()
                .filter(|value| !value.trim().is_empty())
        });
    let Some(filter) = filter else {
        return false;
    };
    let Ok(env_filter) = EnvFilter::try_new(filter) else {
        return false;
    };

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_ansi(false)
        .compact()
        .try_init()
        .is_ok()
}

#[cfg(test)]
mod tests;
