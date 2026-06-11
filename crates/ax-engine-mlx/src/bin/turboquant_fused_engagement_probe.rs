//! One-off probe: does TurboQuant fused decode actually engage on a real
//! model, and does it preserve greedy decode parity?
//!
//! Drives the same layer-level path the runner uses — eligibility mask from
//! `turboquant_support_report`, shadow sync, then single-token decode steps
//! with a `TurboQuantModelDecodeContext` — and prints the per-status decode
//! candidate counters plus a token-parity comparison against a context-free
//! run.
//!
//! Usage:
//!   cargo run -p ax-engine-mlx --bin turboquant-fused-engagement-probe -- \
//!     <model_dir> [prompt_tokens] [decode_steps]

use std::env;
use std::path::Path;

use ax_engine_core::{KvCompressionConfig, NativeModelArtifacts};
use ax_engine_mlx::{
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill},
    kv_cache::{MlxKVCache, MlxKvCompressionDecodeUsage},
    model::{ModelConfig, TurboQuantModelDecodeContext},
    ngram_accel::{NgramTable, single_decode_with_turboquant_context},
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    turboquant::turboquant_support_report,
    weights::{ModelWeights, load_weights},
};

fn layer_windows(cfg: &ModelConfig) -> Vec<Option<usize>> {
    let mut windows = vec![None; cfg.layer_count];
    for (idx, layer) in cfg.layer_configs.iter().enumerate().take(cfg.layer_count) {
        windows[idx] = layer.sliding_window;
    }
    windows
}

#[allow(clippy::too_many_arguments)]
fn decode_tokens(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt: &[u32],
    steps: usize,
    compression: Option<KvCompressionConfig>,
    eligible: &[bool],
    windows: &[Option<usize>],
    usage_sink: &mut MlxKvCompressionDecodeUsage,
) -> Vec<u32> {
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let mut ngram = NgramTable::new();
    let sampling = MlxSamplingParams::greedy();

    let bootstrap = chunked_prefill(
        cfg,
        weights,
        prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(sampling, prompt),
        &mut rng,
    );

    let mut tokens = vec![bootstrap];
    let mut last = bootstrap;
    let mut probs_buf = Vec::new();
    let mut logits_buf = Vec::new();
    let mut candidates_buf = Vec::new();
    for _ in 0..steps {
        if let Some(compression) = compression {
            // Mirror the runner cadence: sync the shadow storage when due so
            // cold tokens advance during decode.
            if cache.turboquant_shadow_storage_sync_due(windows, compression, Some(eligible)) {
                cache.sync_turboquant_shadow_storage(windows, compression, Some(eligible));
            }
        }
        let context = compression.map(|config| TurboQuantModelDecodeContext {
            config,
            layer_eligible: eligible,
        });
        let out = single_decode_with_turboquant_context(
            cfg,
            weights,
            &mut cache,
            &mut ngram,
            last,
            sampling,
            &[],
            &mut rng,
            context.as_ref(),
            &mut probs_buf,
            &mut logits_buf,
            &mut candidates_buf,
        );
        last = *out.last().expect("decode step yields a token");
        tokens.extend(out);
        usage_sink.apply(cache.take_turboquant_decode_usage());
    }
    tokens
}

trait ApplyUsage {
    fn apply(&mut self, other: MlxKvCompressionDecodeUsage);
}

impl ApplyUsage for MlxKvCompressionDecodeUsage {
    fn apply(&mut self, other: MlxKvCompressionDecodeUsage) {
        self.fused_decode_attempts += other.fused_decode_attempts;
        self.fused_decode_successes += other.fused_decode_successes;
        self.fused_decode_metal_successes += other.fused_decode_metal_successes;
        self.fused_decode_fallbacks += other.fused_decode_fallbacks;
        self.fused_decode_ready_candidates += other.fused_decode_ready_candidates;
        self.fused_decode_blocked_prefill_only += other.fused_decode_blocked_prefill_only;
        self.fused_decode_blocked_attention_kind += other.fused_decode_blocked_attention_kind;
        self.fused_decode_blocked_linear_attention += other.fused_decode_blocked_linear_attention;
        self.fused_decode_blocked_sliding_window += other.fused_decode_blocked_sliding_window;
        self.fused_decode_blocked_kv_shared += other.fused_decode_blocked_kv_shared;
        self.fused_decode_blocked_ineligible_layer += other.fused_decode_blocked_ineligible_layer;
        self.fused_decode_blocked_unsupported_preset +=
            other.fused_decode_blocked_unsupported_preset;
        self.fused_decode_blocked_unsupported_head_dim +=
            other.fused_decode_blocked_unsupported_head_dim;
        self.fused_decode_blocked_gqa += other.fused_decode_blocked_gqa;
        self.fused_decode_blocked_missing_storage += other.fused_decode_blocked_missing_storage;
        self.fused_decode_query_readback_wall_us += other.fused_decode_query_readback_wall_us;
        self.fused_decode_cold_metal_wall_us += other.fused_decode_cold_metal_wall_us;
        self.fused_decode_hot_tail_merge_wall_us += other.fused_decode_hot_tail_merge_wall_us;
        self.fused_decode_output_staging_wall_us += other.fused_decode_output_staging_wall_us;
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let model_dir = args.next().expect(
        "Usage: turboquant-fused-engagement-probe <model_dir> [prompt_tokens|@tokens_file] [steps]",
    );
    let prompt_arg = args.next().unwrap_or_else(|| "700".to_string());
    let steps: usize = args
        .next()
        .map(|s| s.parse().expect("steps must be an integer"))
        .unwrap_or(16);
    // `@path` loads a comma/whitespace-separated token-id file (real prompt);
    // a bare integer generates that many synthetic tokens.
    let explicit_prompt: Option<Vec<u32>> = prompt_arg.strip_prefix('@').map(|path| {
        std::fs::read_to_string(path)
            .expect("tokens file should be readable")
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse().expect("token ids must be u32"))
            .collect()
    });
    let prompt_tokens: usize = explicit_prompt
        .as_ref()
        .map(|tokens| tokens.len())
        .unwrap_or_else(|| {
            prompt_arg
                .parse()
                .expect("prompt_tokens must be an integer")
        });

    println!("loading {model_dir}");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .expect("failed to load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("failed to load weights");

    let compression = KvCompressionConfig::turboquant_fused_experimental();
    let support = turboquant_support_report(&cfg, compression.preset)
        .expect("TurboQuant support report should build");
    let eligible = support.eligible_layer_mask();
    let windows = layer_windows(&cfg);
    println!(
        "model_family={} layers={} head_dim={} heads={}/{} eligible_layers={}",
        cfg.model_family,
        cfg.layer_count,
        cfg.head_dim,
        cfg.n_heads,
        cfg.n_kv_heads,
        eligible.iter().filter(|e| **e).count(),
    );

    let prompt: Vec<u32> = explicit_prompt.unwrap_or_else(|| {
        (0..prompt_tokens)
            .map(|i| 2 + ((i * 37) % 4000) as u32)
            .collect()
    });

    let mut fused_usage = MlxKvCompressionDecodeUsage::default();
    let fused_tokens = decode_tokens(
        &cfg,
        &weights,
        &prompt,
        steps,
        Some(compression),
        &eligible,
        &windows,
        &mut fused_usage,
    );

    let mut baseline_usage = MlxKvCompressionDecodeUsage::default();
    let baseline_tokens = decode_tokens(
        &cfg,
        &weights,
        &prompt,
        steps,
        None,
        &eligible,
        &windows,
        &mut baseline_usage,
    );

    println!("--- fused decode candidate counters over {steps} steps ---");
    println!(
        "ready_candidates        = {}",
        fused_usage.fused_decode_ready_candidates
    );
    println!(
        "attempts                = {}",
        fused_usage.fused_decode_attempts
    );
    println!(
        "successes               = {}",
        fused_usage.fused_decode_successes
    );
    println!(
        "metal_successes         = {}",
        fused_usage.fused_decode_metal_successes
    );
    println!(
        "fallbacks               = {}",
        fused_usage.fused_decode_fallbacks
    );
    println!(
        "blocked_sliding_window  = {}",
        fused_usage.fused_decode_blocked_sliding_window
    );
    println!(
        "blocked_kv_shared       = {}",
        fused_usage.fused_decode_blocked_kv_shared
    );
    println!(
        "blocked_ineligible      = {}",
        fused_usage.fused_decode_blocked_ineligible_layer
    );
    println!(
        "blocked_missing_storage = {}",
        fused_usage.fused_decode_blocked_missing_storage
    );
    println!(
        "blocked_head_dim        = {}",
        fused_usage.fused_decode_blocked_unsupported_head_dim
    );
    println!(
        "blocked_gqa             = {}",
        fused_usage.fused_decode_blocked_gqa
    );
    println!(
        "blocked_prefill_only    = {}",
        fused_usage.fused_decode_blocked_prefill_only
    );
    println!(
        "cold_metal_wall_us      = {}",
        fused_usage.fused_decode_cold_metal_wall_us
    );

    let parity = fused_tokens == baseline_tokens;
    println!(
        "--- greedy parity vs full precision: {} ---",
        if parity { "MATCH" } else { "MISMATCH" }
    );
    if !parity {
        let first_diff = fused_tokens
            .iter()
            .zip(&baseline_tokens)
            .position(|(a, b)| a != b);
        println!("first divergence at token index {first_diff:?}");
    }
    println!("fused_tokens={fused_tokens:?}");
    println!("baseline_tokens={baseline_tokens:?}");
}
