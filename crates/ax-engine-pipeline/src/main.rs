use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ax_engine_core::{AX_NATIVE_MODEL_MANIFEST_FILE, NativeModelArtifacts, PipelineTopology};
use ax_engine_mlx::model::ModelConfig;
use ax_engine_mlx::pipeline::PipelineRankExecutor;
use ax_engine_mlx::weights::{load_pipeline_stage_weights, pipeline_stage_required_files};
use ax_engine_pipeline::artifacts::RankBootstrapPlan;
use ax_engine_pipeline::{MlxRankProcessor, router};
use clap::Parser;
use serde::Serialize;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(about = "Run one authenticated AX Engine static pipeline rank")]
struct Args {
    #[arg(long, env = "AX_ENGINE_PIPELINE_TOPOLOGY")]
    topology: PathBuf,
    #[arg(long, env = "AX_ENGINE_PIPELINE_MODEL_DIR")]
    model_dir: PathBuf,
    #[arg(long, env = "AX_ENGINE_PIPELINE_BOOTSTRAP_PLAN")]
    bootstrap_plan: Option<PathBuf>,
    #[arg(
        long,
        env = "AX_ENGINE_PIPELINE_ARTIFACT_BASE_URL",
        requires = "bootstrap_plan"
    )]
    artifact_base_url: Option<String>,
    #[arg(
        long,
        env = "AX_ENGINE_PIPELINE_ARTIFACT_TOKEN",
        hide_env_values = true,
        requires = "artifact_base_url"
    )]
    artifact_token: Option<String>,
    #[arg(long, env = "AX_ENGINE_PIPELINE_RANK")]
    rank: u16,
    #[arg(long, env = "AX_ENGINE_PIPELINE_WORKER_TOKEN", hide_env_values = true)]
    worker_token: String,
    #[arg(long, default_value = "127.0.0.1:9300")]
    listen: String,
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    maximum_activation_bytes: u64,
    #[arg(long, env = "AX_ENGINE_PIPELINE_COORDINATOR_URL")]
    coordinator_url: Option<String>,
    #[arg(
        long,
        env = "AX_ENGINE_PIPELINE_CONTROL_TOKEN",
        hide_env_values = true,
        requires = "coordinator_url"
    )]
    control_token: Option<String>,
    #[arg(long, requires = "coordinator_url")]
    peer_bandwidth_bytes_per_second: Option<u64>,
    #[arg(long, requires = "coordinator_url")]
    peer_latency_micros: Option<u64>,
    #[arg(long, default_value_t = 5)]
    heartbeat_interval_seconds: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let topology_bytes = std::fs::read(&args.topology)?;
    let topology = serde_json::from_slice::<PipelineTopology>(&topology_bytes)?;
    topology.validate()?;
    let assignment = topology
        .assignment(args.rank)
        .ok_or_else(|| format!("rank {} is absent from topology", args.rank))?
        .clone();
    let bootstrap_plan = match &args.bootstrap_plan {
        Some(path) => {
            let plan = RankBootstrapPlan::load(path)?;
            if let Some(base_url) = &args.artifact_base_url {
                let prepared = plan
                    .prepare_from_base_url(
                        &args.model_dir,
                        &topology,
                        args.rank,
                        base_url,
                        args.artifact_token.as_deref(),
                    )
                    .await?;
                tracing::info!(
                    rank = args.rank,
                    downloaded_files = prepared.downloaded_files,
                    downloaded_bytes = prepared.downloaded_bytes,
                    reused_files = prepared.reused_files,
                    "rank artifacts prepared"
                );
            } else {
                plan.verify(&args.model_dir, &topology, args.rank)?;
            }
            Some(plan)
        }
        None if args.coordinator_url.is_some() => {
            return Err(
                "--bootstrap-plan is required when coordinator integration is enabled".into(),
            );
        }
        None => None,
    };
    let artifacts = NativeModelArtifacts::from_dir(&args.model_dir)?;
    if artifacts.manifest().layer_count != topology.total_layers {
        return Err("model manifest layer_count does not match pipeline topology".into());
    }
    if let Some(plan) = &bootstrap_plan {
        let mut required = pipeline_stage_required_files(&artifacts, &assignment);
        required.insert(PathBuf::from(AX_NATIVE_MODEL_MANIFEST_FILE));
        plan.require_artifacts(required)?;
    }
    let config = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_pipeline_stage_weights(&artifacts, assignment)?;
    let executor = PipelineRankExecutor::new(topology.clone(), args.rank, config, weights)?;
    let processor = Arc::new(MlxRankProcessor::new(
        topology.clone(),
        args.rank,
        executor,
        args.maximum_activation_bytes,
    ));
    let body_limit = usize::try_from(args.maximum_activation_bytes)
        .unwrap_or(usize::MAX)
        .saturating_add(64 * 1024);
    let app = router(processor, args.worker_token, body_limit)?;
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    tracing::info!(rank = args.rank, listen = %args.listen, "AX Engine pipeline rank ready");
    let heartbeat = match args.coordinator_url {
        Some(coordinator_url) => {
            let control_token = args.control_token.ok_or_else(|| {
                std::io::Error::other("--control-token is required with --coordinator-url")
            })?;
            let bandwidth = args.peer_bandwidth_bytes_per_second.ok_or_else(|| {
                std::io::Error::other(
                    "--peer-bandwidth-bytes-per-second is required with --coordinator-url",
                )
            })?;
            let latency = args.peer_latency_micros.ok_or_else(|| {
                std::io::Error::other("--peer-latency-micros is required with --coordinator-url")
            })?;
            let config = HeartbeatConfig {
                coordinator_url,
                control_token,
                topology: topology.clone(),
                rank: args.rank,
                bandwidth,
                latency,
                interval: Duration::from_secs(args.heartbeat_interval_seconds.max(1)),
            };
            let handle = tokio::spawn(heartbeat_loop(config.clone()));
            Some((handle, config))
        }
        None => None,
    };
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await;
    if let Some((heartbeat, config)) = heartbeat {
        heartbeat.abort();
        let _ = heartbeat.await;
        let state = if serve_result.is_ok() {
            "draining"
        } else {
            "failed"
        };
        match heartbeat_client() {
            Ok(client) => {
                if let Err(error) = post_rank_observation(&client, &config, state).await {
                    tracing::warn!(rank = config.rank, %error, state, "final rank heartbeat failed");
                }
            }
            Err(error) => {
                tracing::warn!(rank = args.rank, %error, "failed to construct final heartbeat client");
            }
        }
    }
    serve_result?;
    Ok(())
}

#[derive(Clone)]
struct HeartbeatConfig {
    coordinator_url: String,
    control_token: String,
    topology: PipelineTopology,
    rank: u16,
    bandwidth: u64,
    latency: u64,
    interval: Duration,
}

#[derive(Serialize)]
struct RankObservation {
    cluster_id: String,
    generation: u64,
    manifest_digest: String,
    rank: u16,
    state: &'static str,
    observed_at: String,
    peer_bandwidth_bytes_per_second: u64,
    peer_latency_micros: u64,
}

async fn heartbeat_loop(config: HeartbeatConfig) {
    let client = match heartbeat_client() {
        Ok(client) => client,
        Err(error) => {
            tracing::error!(%error, "failed to construct rank heartbeat client");
            return;
        }
    };
    loop {
        if let Err(error) = post_rank_observation(&client, &config, "ready").await {
            tracing::warn!(rank = config.rank, %error, "rank heartbeat failed");
        }
        tokio::time::sleep(config.interval).await;
    }
}

fn heartbeat_client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(10))
        .build()
}

async fn post_rank_observation(
    client: &reqwest::Client,
    config: &HeartbeatConfig,
    state: &'static str,
) -> Result<(), String> {
    let observed_at = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into());
    let observation = RankObservation {
        cluster_id: config.topology.cluster_id.clone(),
        generation: config.topology.generation,
        manifest_digest: config.topology.manifest_digest.clone(),
        rank: config.rank,
        state,
        observed_at,
        peer_bandwidth_bytes_per_second: config.bandwidth,
        peer_latency_micros: config.latency,
    };
    let endpoint = format!(
        "{}/internal/cluster/ranks/{}/heartbeat",
        config.coordinator_url.trim_end_matches('/'),
        config.rank
    );
    let response = client
        .post(endpoint)
        .header("x-ax-cluster-control-token", &config.control_token)
        .json(&observation)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!("coordinator returned HTTP {}", response.status()))
    }
}
