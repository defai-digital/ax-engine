use std::io::{self, Write as _};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_core::PipelineTopology;
use ax_engine_pipeline::TokenStepRequest;
use ax_engine_pipeline::client::PipelineChainClient;
use ax_engine_sdk::EngineTokenizer;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Run one greedy prompt through an AX Engine Mac pipeline")]
struct Args {
    #[arg(long, env = "AX_ENGINE_PIPELINE_TOPOLOGY")]
    topology: PathBuf,
    #[arg(long, env = "AX_ENGINE_PIPELINE_MODEL_DIR")]
    model_dir: PathBuf,
    #[arg(long, value_delimiter = ',', env = "AX_ENGINE_PIPELINE_ENDPOINTS")]
    endpoints: Vec<String>,
    #[arg(long, env = "AX_ENGINE_PIPELINE_WORKER_TOKEN", hide_env_values = true)]
    worker_token: String,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 128)]
    maximum_output_tokens: usize,
    #[arg(long, value_delimiter = ',')]
    stop_token_ids: Vec<u32>,
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    maximum_activation_bytes: u64,
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
    let prompt_tokens = tokenizer.encode_with_special_tokens(&args.prompt, true)?;
    if prompt_tokens.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }
    let request_id = request_id();
    let mut generated = Vec::<u32>::new();
    // Codepoint-complete text already printed. Incomplete multi-byte tails
    // (decoded as U+FFFD) are held back — never printed as � and never fully
    // re-emitted with a carriage return after a prefix mismatch.
    let mut emitted = String::new();
    let mut sequence = 1_u64;
    let mut token_offset = 0_u64;
    let mut input = prompt_tokens.clone();
    let generation_result = async {
        while generated.len() < args.maximum_output_tokens {
            let token = client
                .step(TokenStepRequest {
                    request_id,
                    request_sequence: sequence,
                    token_offset,
                    token_ids: input.clone(),
                })
                .await?
                .token_id;
            generated.push(token);
            let next_rendered = tokenizer.decode(&generated, true)?;
            let complete = next_rendered
                .strip_suffix('\u{FFFD}')
                .unwrap_or(next_rendered.as_str());
            if complete.len() > emitted.len()
                && complete.starts_with(emitted.as_str())
                && complete.is_char_boundary(emitted.len())
            {
                let delta = &complete[emitted.len()..];
                if !delta.is_empty() {
                    print!("{delta}");
                    io::stdout().flush()?;
                }
                emitted = complete.to_string();
            }
            if args.stop_token_ids.contains(&token)
                || tokenizer.eos_token_id().is_some_and(|eos| eos == token)
            {
                break;
            }
            token_offset = token_offset
                .checked_add(if sequence == 1 {
                    prompt_tokens.len() as u64
                } else {
                    1
                })
                .ok_or_else(|| io::Error::other("token offset overflow"))?;
            sequence = sequence
                .checked_add(1)
                .ok_or_else(|| io::Error::other("request sequence overflow"))?;
            input.clear();
            input.push(token);
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    }
    .await;
    let close_result = client.close_request(request_id).await;
    println!();
    generation_result?;
    // Successful generation already printed tokens; close failures are logged
    // but must not discard the exit status of a completed run.
    if let Err(error) = close_result {
        eprintln!("warning: pipeline close_request failed after generation: {error}");
    }
    Ok(())
}

fn request_id() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(1);
    (nanos as u64).max(1)
}
