use std::path::PathBuf;
use std::time::Duration;

use ax_engine_core::PipelineTopology;
use ax_engine_pipeline::client::PipelineChainClient;
use ax_engine_pipeline::gateway::{GatewayState, router};
use ax_engine_sdk::EngineTokenizer;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Serve OpenAI greedy completions through an AX Engine Mac pipeline")]
struct Args {
    #[arg(long, env = "AX_ENGINE_PIPELINE_TOPOLOGY")]
    topology: PathBuf,
    #[arg(long, env = "AX_ENGINE_PIPELINE_MODEL_DIR")]
    model_dir: PathBuf,
    #[arg(long, value_delimiter = ',', env = "AX_ENGINE_PIPELINE_ENDPOINTS")]
    endpoints: Vec<String>,
    #[arg(long, env = "AX_ENGINE_PIPELINE_WORKER_TOKEN", hide_env_values = true)]
    worker_token: String,
    #[arg(long, env = "AX_ENGINE_PIPELINE_API_KEY", hide_env_values = true)]
    api_key: String,
    #[arg(long)]
    model_id: String,
    #[arg(long, default_value = "127.0.0.1:9400")]
    listen: String,
    #[arg(long, default_value_t = 4)]
    maximum_concurrent_requests: usize,
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    maximum_activation_bytes: u64,
    #[arg(long, default_value_t = 300)]
    request_timeout_seconds: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let topology = serde_json::from_slice::<PipelineTopology>(&std::fs::read(&args.topology)?)?;
    let client = PipelineChainClient::new(
        topology,
        args.endpoints,
        args.worker_token,
        args.maximum_activation_bytes,
    )?;
    client.preflight().await?;
    let tokenizer = EngineTokenizer::from_model_dir(&args.model_dir)?;
    let state = GatewayState::new(
        client,
        tokenizer,
        args.model_id,
        args.api_key,
        args.maximum_concurrent_requests,
        Duration::from_secs(args.request_timeout_seconds),
    )?;
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
