use std::io::{self, Write as _};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_core::PipelineTopology;
use ax_engine_pipeline::TokenStepRequest;
use ax_engine_pipeline::client::PipelineChainClient;
use ax_engine_pipeline::text_stream::{complete_decode_prefix, final_stream_delta, stream_delta};
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
    #[arg(long, default_value_t = 128, value_parser = parse_maximum_output_tokens)]
    maximum_output_tokens: usize,
    #[arg(long, value_delimiter = ',')]
    stop_token_ids: Vec<u32>,
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    maximum_activation_bytes: u64,
}

fn parse_maximum_output_tokens(raw: &str) -> Result<usize, String> {
    let parsed = raw.parse::<usize>().map_err(|error| error.to_string())?;
    if parsed == 0 {
        return Err("maximum output tokens must be at least 1".to_string());
    }
    Ok(parsed)
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
            if let Some(delta) = stream_delta(&emitted, &next_rendered) {
                print!("{delta}");
                io::stdout().flush()?;
                emitted = complete_decode_prefix(&next_rendered).to_string();
            }
            if args.stop_token_ids.contains(&token) || tokenizer.eos_token_ids().contains(&token) {
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
        // Normal termination: flush the held-back trailing suffix exactly
        // once so printed text equals the final decode.
        let final_rendered = tokenizer.decode(&generated, true)?;
        if let Some(delta) = final_stream_delta(&emitted, &final_rendered) {
            print!("{delta}");
            io::stdout().flush()?;
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use clap::Parser as _;

    use super::Args;

    fn parse_args(maximum_output_tokens: &str) -> Result<Args, clap::Error> {
        Args::try_parse_from([
            "ax-engine-pipeline-generate",
            "--topology",
            "topology.json",
            "--model-dir",
            "model",
            "--endpoints",
            "127.0.0.1:9401",
            "--worker-token",
            "0123456789abcdef",
            "--prompt",
            "hello",
            "--maximum-output-tokens",
            maximum_output_tokens,
        ])
    }

    #[test]
    fn maximum_output_tokens_rejects_zero_and_accepts_positive() {
        assert!(parse_args("0").is_err(), "zero must be rejected by clap");
        assert_eq!(
            parse_args("1")
                .expect("positive parses")
                .maximum_output_tokens,
            1
        );
    }
}
