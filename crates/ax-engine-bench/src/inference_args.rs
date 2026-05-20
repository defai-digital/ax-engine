use ax_engine_core::CacheGroupId;
use ax_engine_sdk::{
    EngineSession, EngineSessionConfig, GenerateRequest, GenerateSampling, PreviewBackendRequest,
    SupportTier, preview_support_tier_from_label,
};
use std::path::PathBuf;

use crate::args::{next_flag_value, parse_flag_value, parse_token_list};
use crate::cli::usage;
use crate::error::CliError;

/// Bench-default MLX prefill chunk size.
///
/// Matches `mlx-lm`'s `prefill_step_size=2048` default (see
/// `.internal/reference/mlx-lm/mlx_lm/generate.py::generate`). The AX runner
/// auto-clamps this value per model family:
///
/// - Dense / full-attention models (Qwen3, Llama-style): use 2048 as-is.
/// - Models with GatedDelta linear-attention layers (Qwen3.5 / Qwen 3.6):
///   clamped to `GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY` (1024) — a
///   2048-token prompt processes as two 1024-token sub-chunks, recovering
///   SM occupancy lost by the larger threadgroup-cache allocation.
/// - MLA models (GLM 4 Flash MLA): forced to
///   `MLA_DEFAULT_PREFILL_CHUNK = 16` by
///   `crate::fastpath::resolve_prefill_chunk`.
///
/// Using `None` here (the previous default) caused the runner to fall back
/// to the runtime default, which may differ by model family. Keep this
/// explicit so long-prompt comparisons remain aligned with `mlx_lm.benchmark`.
/// Override with `--prefill-chunk N` when needed.
pub(crate) const BENCH_DEFAULT_MLX_PREFILL_CHUNK: usize = 2048;

#[derive(Clone, Debug)]
pub(crate) struct InferenceArgs {
    pub(crate) model_id: String,
    pub(crate) input_tokens: Vec<u32>,
    pub(crate) input_text: Option<String>,
    pub(crate) max_output_tokens: u32,
    pub(crate) sampling: GenerateSampling,
    pub(crate) metadata: Option<String>,
    pub(crate) deterministic: bool,
    pub(crate) mlx: bool,
    pub(crate) support_tier: SupportTier,
    pub(crate) llama_cli_path: PathBuf,
    pub(crate) llama_model_path: Option<PathBuf>,
    pub(crate) llama_server_url: Option<String>,
    pub(crate) mlx_lm_server_url: Option<String>,
    pub(crate) mlx_model_artifacts_dir: Option<PathBuf>,
    pub(crate) mlx_prefill_chunk: Option<usize>,
    pub(crate) json: bool,
}

impl Default for InferenceArgs {
    fn default() -> Self {
        Self {
            model_id: "qwen3".to_string(),
            input_tokens: Vec::new(),
            input_text: None,
            max_output_tokens: 32,
            sampling: GenerateSampling::default(),
            metadata: None,
            deterministic: true,
            mlx: false,
            support_tier: SupportTier::LlamaCpp,
            llama_cli_path: PathBuf::from("llama-cli"),
            llama_model_path: None,
            llama_server_url: None,
            mlx_lm_server_url: None,
            mlx_model_artifacts_dir: None,
            mlx_prefill_chunk: Some(BENCH_DEFAULT_MLX_PREFILL_CHUNK),
            json: false,
        }
    }
}

impl InferenceArgs {
    pub(crate) fn generate_request(&self) -> GenerateRequest {
        GenerateRequest {
            model_id: self.model_id.clone(),
            input_tokens: self.input_tokens.clone(),
            input_text: self.input_text.clone(),
            max_output_tokens: self.max_output_tokens,
            sampling: self.sampling.clone(),
            stop_sequences: Vec::new(),
            metadata: self.metadata.clone(),
        }
    }
}

pub(crate) fn parse_inference_args(
    args: &[String],
    command: &str,
) -> Result<InferenceArgs, CliError> {
    let mut parsed = InferenceArgs::default();
    let mut support_tier_label = "llama_cpp".to_string();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model-id" => parsed.model_id = next_flag_value(&mut iter, "--model-id")?.to_string(),
            "--prompt" => {
                parsed.input_text = Some(next_flag_value(&mut iter, "--prompt")?.to_string())
            }
            "--tokens" => {
                parsed.input_tokens = parse_token_list(next_flag_value(&mut iter, "--tokens")?)?
            }
            "--max-output-tokens" => {
                parsed.max_output_tokens = parse_flag_value::<u32>(
                    next_flag_value(&mut iter, "--max-output-tokens")?,
                    "--max-output-tokens",
                )?
            }
            "--temperature" => {
                parsed.sampling.temperature = parse_flag_value::<f32>(
                    next_flag_value(&mut iter, "--temperature")?,
                    "--temperature",
                )?
            }
            "--top-p" => {
                parsed.sampling.top_p =
                    parse_flag_value::<f32>(next_flag_value(&mut iter, "--top-p")?, "--top-p")?
            }
            "--top-k" => {
                parsed.sampling.top_k =
                    parse_flag_value::<u32>(next_flag_value(&mut iter, "--top-k")?, "--top-k")?
            }
            "--repetition-penalty" => {
                parsed.sampling.repetition_penalty = parse_flag_value::<f32>(
                    next_flag_value(&mut iter, "--repetition-penalty")?,
                    "--repetition-penalty",
                )?
            }
            "--seed" => {
                parsed.sampling.seed =
                    parse_flag_value::<u64>(next_flag_value(&mut iter, "--seed")?, "--seed")?
            }
            "--metadata" => {
                parsed.metadata = Some(next_flag_value(&mut iter, "--metadata")?.to_string())
            }
            "--deterministic" => {
                parsed.deterministic = parse_flag_value::<bool>(
                    next_flag_value(&mut iter, "--deterministic")?,
                    "--deterministic",
                )?
            }
            "--mlx" => parsed.mlx = true,
            "--support-tier" => {
                support_tier_label = next_flag_value(&mut iter, "--support-tier")?.to_string()
            }
            "--llama-cli-path" => {
                parsed.llama_cli_path =
                    PathBuf::from(next_flag_value(&mut iter, "--llama-cli-path")?)
            }
            "--llama-model-path" => {
                parsed.llama_model_path = Some(PathBuf::from(next_flag_value(
                    &mut iter,
                    "--llama-model-path",
                )?))
            }
            "--llama-server-url" => {
                parsed.llama_server_url =
                    Some(next_flag_value(&mut iter, "--llama-server-url")?.to_string())
            }
            "--mlx-lm-server-url" => {
                parsed.mlx_lm_server_url =
                    Some(next_flag_value(&mut iter, "--mlx-lm-server-url")?.to_string())
            }
            "--mlx-model-artifacts-dir" => {
                parsed.mlx_model_artifacts_dir = Some(PathBuf::from(next_flag_value(
                    &mut iter,
                    "--mlx-model-artifacts-dir",
                )?))
            }
            "--prefill-chunk" => {
                parsed.mlx_prefill_chunk = Some(parse_flag_value::<usize>(
                    next_flag_value(&mut iter, "--prefill-chunk")?,
                    "--prefill-chunk",
                )?)
            }
            "--json" => parsed.json = true,
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag for {command}: {other}\n\n{}",
                    usage()
                )));
            }
        }
    }

    if parsed.input_text.is_some() && !parsed.input_tokens.is_empty() {
        return Err(CliError::Usage(format!(
            "{command} accepts either --prompt or --tokens, not both"
        )));
    }

    if parsed.input_text.is_none() && parsed.input_tokens.is_empty() {
        return Err(CliError::Usage(format!(
            "{command} requires either --prompt or --tokens"
        )));
    }

    parsed.support_tier = preview_support_tier_from_label(&support_tier_label)
        .map_err(|error| CliError::Usage(format!("invalid --support-tier: {error}")))?;
    parsed.sampling.deterministic = Some(parsed.deterministic);

    Ok(parsed)
}

pub(crate) fn build_inference_session(args: &InferenceArgs) -> Result<EngineSession, CliError> {
    let backend_request = if args.mlx {
        PreviewBackendRequest::shipping_mlx()
    } else if args.support_tier == SupportTier::LlamaCpp {
        PreviewBackendRequest::shipping_default_llama_cpp(
            args.llama_cli_path.clone(),
            args.llama_model_path.clone(),
            args.llama_server_url.clone(),
        )
    } else if args.support_tier == SupportTier::MlxLmDelegated {
        PreviewBackendRequest {
            support_tier: SupportTier::MlxLmDelegated,
            mlx_lm_server_url: args.mlx_lm_server_url.clone(),
            ..PreviewBackendRequest::default()
        }
    } else {
        return Err(CliError::Usage(
            "non-MLX inference routes to explicit delegated backends: llama_cpp or mlx_lm_delegated; pass --mlx for AX-owned MLX inference"
                .to_string(),
        ));
    };

    let mlx_model_artifacts_dir = if args.mlx {
        args.mlx_model_artifacts_dir
            .clone()
            .or_else(|| args.llama_model_path.clone())
    } else {
        args.mlx_model_artifacts_dir.clone()
    };

    // Honor AX_NO_SPEC=1 so decode-profile runs can route through the direct
    // double-buffer path instead of n-gram acceleration.
    let disable_ngram = matches!(
        std::env::var("AX_NO_SPEC").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    );
    let config =
        EngineSessionConfig::from_preview_request(ax_engine_sdk::PreviewSessionConfigRequest {
            cache_group_id: CacheGroupId(0),
            block_size_tokens: 16,
            total_blocks: 1024,
            deterministic: args.deterministic,
            max_batch_tokens: 2048,
            mlx_runtime_artifacts_dir: None,
            backend_request,
            mlx_model_artifacts_dir,
            mlx_disable_ngram_acceleration: disable_ngram,
            mlx_kv_compression: ax_engine_sdk::KvCompressionConfig::disabled(),
            mlx_prefill_chunk: args.mlx_prefill_chunk,
        })
        .map_err(|error| CliError::Usage(format!("invalid inference configuration: {error}")))?;

    EngineSession::new(config)
        .map_err(|error| CliError::Runtime(format!("failed to start AX Engine session: {error}")))
}
