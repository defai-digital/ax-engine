use std::fmt;
use std::time::Duration;

use crate::kv::ModelKv;
use crate::metrics::counters::OpTimer;
use crate::model::{LlamaModel, WeightStore};
use crate::sampling::Sampler;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeIntent {
    Throughput,
    Latency,
}

impl fmt::Display for DecodeIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Throughput => f.write_str("throughput"),
            Self::Latency => f.write_str("latency"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    Sequential,
    SingleCb,
    Pipelined,
}

impl fmt::Display for DecodeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => f.write_str("sequential"),
            Self::SingleCb => f.write_str("single_cb"),
            Self::Pipelined => f.write_str("pipelined"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodeSelection {
    pub intent: DecodeIntent,
    pub mode: DecodeMode,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeRunConfig {
    pub intent: DecodeIntent,
    pub allow_pipelined: bool,
}

impl Default for DecodeRunConfig {
    fn default() -> Self {
        Self {
            intent: DecodeIntent::Throughput,
            allow_pipelined: true,
        }
    }
}

#[derive(Debug)]
pub struct DecodeRunResult {
    pub selection: DecodeSelection,
    pub generated_tokens: u64,
    pub decode_duration: Duration,
    pub latencies: Vec<Duration>,
}

pub fn select_decode_mode(
    model: &LlamaModel,
    kv: &ModelKv,
    intent: DecodeIntent,
    allow_pipelined: bool,
) -> DecodeSelection {
    let has_gpu_decode = kv.is_gpu() && model.metal_device().is_some();

    if intent == DecodeIntent::Throughput && allow_pipelined {
        if has_gpu_decode && model.supports_pipelined_decode() {
            return DecodeSelection {
                intent,
                mode: DecodeMode::Pipelined,
                fallback_reason: None,
            };
        }
        if has_gpu_decode {
            return DecodeSelection {
                intent,
                mode: DecodeMode::SingleCb,
                fallback_reason: Some(
                    "pipelined decode is unavailable for this model/backend combination"
                        .to_string(),
                ),
            };
        }
        return DecodeSelection {
            intent,
            mode: DecodeMode::Sequential,
            fallback_reason: Some("GPU decode unavailable; using sequential decode".to_string()),
        };
    }

    DecodeSelection {
        intent,
        mode: if has_gpu_decode {
            DecodeMode::SingleCb
        } else {
            DecodeMode::Sequential
        },
        fallback_reason: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_decode<F>(
    model: &LlamaModel,
    weights: &WeightStore<'_>,
    tokenizer: &Tokenizer,
    kv: &mut ModelKv,
    sampler: &mut Sampler,
    history: &mut Vec<u32>,
    first_token: u32,
    position: usize,
    max_tokens: usize,
    config: DecodeRunConfig,
    on_token: F,
) -> anyhow::Result<DecodeRunResult>
where
    F: FnMut(u32) -> anyhow::Result<()>,
{
    let selection = select_decode_mode(model, kv, config.intent, config.allow_pipelined);
    match selection.mode {
        DecodeMode::Pipelined => run_pipelined_decode(
            model,
            weights,
            tokenizer,
            kv,
            sampler,
            history,
            first_token,
            position,
            max_tokens,
            selection,
            on_token,
        ),
        DecodeMode::Sequential | DecodeMode::SingleCb => run_sequential_decode(
            model,
            weights,
            tokenizer,
            kv,
            sampler,
            history,
            first_token,
            position,
            max_tokens,
            selection,
            on_token,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sequential_decode<F>(
    model: &LlamaModel,
    weights: &WeightStore<'_>,
    tokenizer: &Tokenizer,
    kv: &mut ModelKv,
    sampler: &mut Sampler,
    history: &mut Vec<u32>,
    first_token: u32,
    mut position: usize,
    max_tokens: usize,
    selection: DecodeSelection,
    mut on_token: F,
) -> anyhow::Result<DecodeRunResult>
where
    F: FnMut(u32) -> anyhow::Result<()>,
{
    let mut logits = vec![0.0f32; model.config.vocab_size as usize];
    let mut next_token = first_token;
    let mut generated_tokens = 0u64;
    let mut latencies = Vec::new();
    let decode_timer = OpTimer::start();

    for _ in 0..max_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }

        on_token(next_token)?;
        history.push(next_token);
        generated_tokens += 1;

        let tok_timer = OpTimer::start();
        logits.fill(0.0);
        model.forward_single(next_token, position, kv, weights, &mut logits)?;
        let tok_latency = tok_timer.elapsed();
        if selection.intent == DecodeIntent::Latency {
            latencies.push(tok_latency);
        }
        position += 1;

        next_token = sampler.sample(&mut logits, history);
    }

    Ok(DecodeRunResult {
        selection,
        generated_tokens,
        decode_duration: decode_timer.elapsed(),
        latencies,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_pipelined_decode<F>(
    model: &LlamaModel,
    weights: &WeightStore<'_>,
    tokenizer: &Tokenizer,
    kv: &mut ModelKv,
    sampler: &mut Sampler,
    history: &mut Vec<u32>,
    first_token: u32,
    start_position: usize,
    max_tokens: usize,
    selection: DecodeSelection,
    mut on_token: F,
) -> anyhow::Result<DecodeRunResult>
where
    F: FnMut(u32) -> anyhow::Result<()>,
{
    if tokenizer.is_eos(first_token) || max_tokens == 0 {
        return Ok(DecodeRunResult {
            selection,
            generated_tokens: 0,
            decode_duration: Duration::ZERO,
            latencies: Vec::new(),
        });
    }

    let metal_dev = model
        .metal_device()
        .ok_or_else(|| anyhow::anyhow!("pipelined decode requested without a Metal device"))?;
    let dim = model.config.embedding_dim as usize;
    let hidden_bytes = dim * std::mem::size_of::<f32>();
    let mut hidden_a = model.alloc_metal_buf(hidden_bytes)?;
    let mut hidden_b = model.alloc_metal_buf(hidden_bytes)?;

    model.prewarm_kv_capacity(kv, start_position + max_tokens + 2)?;

    let mut logits = vec![0.0f32; model.config.vocab_size as usize];
    let mut position = start_position;
    let mut next_token = first_token;
    let mut generated_tokens = 0u64;
    let decode_timer = OpTimer::start();

    model.embed_token_into(first_token, &hidden_a, weights)?;
    let pending = model
        .encode_pending_decode_step(&hidden_a, position, kv, weights)?
        .ok_or_else(|| anyhow::anyhow!("pipelined decode became unavailable during setup"))?;
    let mut inflight = Some(metal_dev.commit_frame(pending));

    for step_idx in 0..max_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }

        let should_prepare_next = step_idx + 1 < max_tokens;
        let next_position = position + 1;
        let pending_next = if should_prepare_next {
            model.encode_pending_decode_step(&hidden_b, next_position, kv, weights)?
        } else {
            None
        };

        if let Some(frame) = inflight.take() {
            metal_dev.wait_frame(frame)?;
        }
        model.advance_gpu_kv_token(kv);

        model.read_gpu_logits(&mut logits)?;
        let sampled = sampler.sample(&mut logits, history);

        on_token(next_token)?;
        history.push(next_token);
        generated_tokens += 1;

        let stop_after_emit = !should_prepare_next || tokenizer.is_eos(sampled);
        if stop_after_emit {
            break;
        }

        model.embed_token_into(sampled, &hidden_b, weights)?;
        let pending_next = pending_next.ok_or_else(|| {
            anyhow::anyhow!("pipelined decode became unavailable while preparing the next frame")
        })?;
        inflight = Some(metal_dev.commit_frame(pending_next));

        next_token = sampled;
        position = next_position;
        std::mem::swap(&mut hidden_a, &mut hidden_b);
    }

    Ok(DecodeRunResult {
        selection,
        generated_tokens,
        decode_duration: decode_timer.elapsed(),
        latencies: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::{HybridCpuDecodeBackend, MetalBackend};
    use crate::model::config::{GateActivation, RopeScaling};

    fn tiny_config(arch: &str) -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            architecture: arch.into(),
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            context_length: 32,
            vocab_size: 8,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: arch == "qwen3",
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: if arch == "gemma3" {
                GateActivation::GELU
            } else {
                GateActivation::SiLU
            },
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: RopeScaling::None,
            embed_scale: arch == "gemma3",
            rope_freq_base_local: None,
        }
    }

    #[test]
    fn test_select_decode_mode_cpu_uses_sequential_decode() {
        let model = LlamaModel::new(tiny_config("llama"));
        let kv = model.create_model_kv();
        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);

        assert_eq!(selection.mode, DecodeMode::Sequential);
        assert_eq!(
            selection.fallback_reason.as_deref(),
            Some("GPU decode unavailable; using sequential decode")
        );
    }

    #[test]
    fn test_select_decode_mode_hybrid_cpu_decode_uses_sequential_decode() {
        let Ok(backend) = HybridCpuDecodeBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("llama"), Box::new(backend));
        let kv = model.create_model_kv();

        assert!(!kv.is_gpu(), "HybridCpuDecode should allocate CPU KV");

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::Sequential);
    }

    #[test]
    fn test_select_decode_mode_llama_metal_prefers_pipelined() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("llama"), Box::new(backend));
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::Pipelined);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_select_decode_mode_qwen3_metal_prefers_pipelined() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("qwen3"), Box::new(backend));
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::Pipelined);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_select_decode_mode_gemma3_metal_prefers_pipelined() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("gemma3"), Box::new(backend));
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Throughput, true);
        assert_eq!(selection.mode, DecodeMode::Pipelined);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_select_decode_mode_latency_gpu_uses_single_cb() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = LlamaModel::with_backend(tiny_config("llama"), Box::new(backend));
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let selection = select_decode_mode(&model, &kv, DecodeIntent::Latency, false);
        assert_eq!(selection.mode, DecodeMode::SingleCb);
        assert!(selection.fallback_reason.is_none());
    }
}
