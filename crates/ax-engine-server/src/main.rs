#![allow(clippy::collapsible_if)]

use ax_engine_sdk::EngineSessionConfig;
use clap::Parser;
use std::env;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

mod admission;
mod anthropic;
mod app_state;
mod args;
mod backends;
mod chat;
mod embeddings;
mod errors;
mod generation;
mod grpc;
mod grpc_auth;
mod grpc_metrics;
mod lan_advertise;
mod metadata;
mod metrics;
mod model_load;
mod multimodal;
mod ollama;
mod openai;
mod rate_limit;
mod routes;
mod tasks;

use app_state::{ServerLimits, build_app_state};
use args::{ServerArgs, render_presets};
use routes::build_router;

// Processed Gemma4 unified media tensors are JSON-heavy: multi-image and audio
// payloads can exceed 100 MiB before transport compression.
const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 256 * 1024 * 1024;

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
    let model_id = args
        .effective_model_id()
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    let support_tier = args.effective_support_tier();
    let session_config = args
        .session_config()
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    log_host_detection_warnings(&session_config);
    let limits = ServerLimits {
        max_concurrent_requests: args.resolved_max_concurrent_requests(),
        max_request_body_bytes: args.resolved_max_request_body_bytes(),
        request_timeout: args.resolved_request_timeout(),
        grpc_request_timeout: args.resolved_grpc_request_timeout(),
        rate_limit: args.resolved_rate_limit(),
        stream_deadlines: args.resolved_stream_deadlines(),
    };
    let api_key = args.resolved_api_key();
    let discovery_instance_id = new_instance_id();
    let lan_cluster = args.resolved_lan_cluster();
    let state = build_app_state(model_id.clone(), session_config)?
        .with_api_key(api_key.clone())
        .with_limits(limits)
        .with_discovery(app_state::DiscoveryMeta {
            instance_id: discovery_instance_id.clone(),
            cluster: lan_cluster.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        });
    if let Some(idle_timeout) = args.resolved_model_idle_timeout() {
        model_load::spawn_model_idle_evictor(state.clone(), idle_timeout);
    }
    let app = build_router(state.clone());
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;
    // Use the OS-assigned port when `--port 0` so mDNS advertises a reachable endpoint.
    let bound_port = listener.local_addr()?.port();

    let grpc_bind_address = args.grpc_bind_address.clone();

    // Keep the advertiser alive for the process lifetime.
    let _lan_advertiser = if args.resolved_advertise_lan() {
        if is_loopback_bind_host(&args.host) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--advertise-lan requires a non-loopback --host (use 0.0.0.0 or a LAN IP)",
            )
            .into());
        }
        let advertise_ip = lan_advertise::pick_advertise_ipv4(
            args.resolved_lan_advertise_host().as_deref(),
            &args.host,
        )
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
        if api_key.is_none() {
            warn!(
                "LAN advertise is enabled without --api-key / AX_ENGINE_API_KEY; \
                 peers will see auth=open. Prefer requiring a bearer token on non-loopback binds."
            );
        }
        let advertiser = lan_advertise::LanAdvertiser::start(lan_advertise::LanAdvertiseConfig {
            instance_name: args.resolved_lan_instance_name(),
            port: bound_port,
            advertise_ip,
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_id: model_id.clone(),
            auth_required: api_key.is_some(),
            cluster: lan_cluster,
            instance_id: discovery_instance_id,
        })
        .map_err(std::io::Error::other)?;
        if !tracing_enabled {
            eprintln!(
                "ax-engine-server LAN mDNS advertise on {}:{} (_ax-engine._tcp)",
                advertise_ip, bound_port
            );
        }
        Some(advertiser)
    } else {
        None
    };

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
        let grpc_api_key = state.api_key.clone();
        let grpc_metrics = state.metrics.clone();
        let grpc_request_timeout = state.limits.grpc_request_timeout;
        let grpc_service = grpc::AxEngineGrpcService::new(state).into_server();
        // tower/tonic's `.layer()` composes with the FIRST call as the
        // OUTERMOST layer (opposite of axum's `Router::layer`, where the
        // LAST call is outermost) — see `tower::layer::util::Stack` and
        // `ServiceBuilder::layer`. Metrics is added first so it wraps auth
        // and observes auth rejections too, matching the HTTP layer
        // ordering in `routes.rs`.
        let mut grpc_builder = tonic::transport::Server::builder()
            .layer(grpc_metrics::GrpcMetricsLayer::new(grpc_metrics))
            .layer(grpc_auth::GrpcAuthLayer::new(grpc_api_key));
        if let Some(timeout) = grpc_request_timeout {
            grpc_builder = grpc_builder.timeout(timeout);
        }
        let grpc_server = grpc_builder
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

/// Per-process instance identity for discovery/mDNS. Mixes wall time, pid, and
/// OS entropy (via [`std::hash::RandomState`]) so parallel starts do not share IDs.
fn new_instance_id() -> String {
    use std::hash::{BuildHasher, Hasher, RandomState};
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u128(nanos);
    hasher.write_u32(pid);
    let entropy = hasher.finish();
    format!("axeng-{pid:x}-{entropy:016x}")
}

fn is_loopback_bind_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        // Non-IP hostnames (e.g. machine.local) are treated as non-loopback;
        // advertise still requires a resolvable private IPv4 via pick_advertise_ipv4.
        Err(_) => false,
    }
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
