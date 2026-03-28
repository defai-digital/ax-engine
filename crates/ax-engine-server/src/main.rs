#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

mod api;
mod args;
mod engine;

use std::net::SocketAddr;

use anyhow::Context;
use clap::Parser;

use crate::api::AppState;
use crate::args::ServerArgs;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();
    init_tracing(args.verbose);

    let engine = engine::ServerEngine::load(&args)?;
    let state = AppState::new(engine);
    let app = api::router(state);
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .with_context(|| format!("invalid listen address {}:{}", args.host, args.port))?;

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    tracing::info!("ax-engine-server listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server exited with error")?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}

fn init_tracing(verbose: bool) {
    let filter = if verbose {
        "ax_engine_core=debug,ax_engine_sdk=debug,ax_engine_server=debug"
    } else {
        "ax_engine_core=warn,ax_engine_sdk=info,ax_engine_server=info"
    };

    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}
