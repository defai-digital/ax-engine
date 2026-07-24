//! Native MLX inference for OpenAI Whisper multilingual checkpoints.
//!
//! The primary certified target is
//! `mlx-community/whisper-large-v3-turbo`: 32 encoder blocks, four decoder
//! blocks, 128 mel bins, and the native mlx-whisper tensor layout.

use ax_engine_core::NativeModelArtifacts;
use mlx_sys::{
    MlxArray, MlxDtype, add, argmax, astype, concatenate, contiguous, conv1d, gelu, layer_norm,
    load_safetensors, matmul, multiply, reshape, slice, softmax_precise, take, transpose, try_eval,
};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use thiserror::Error;

use crate::whisper_mel::{N_FRAMES, N_SAMPLES, WhisperMel};
use crate::whisper_tokenizer::WhisperTokenizer;

#[derive(Debug, Error)]
pub enum WhisperError {
    #[error("invalid Whisper config: {0}")]
    Config(String),
    #[error("failed to load Whisper weights: {0}")]
    Weights(String),
    #[error("invalid Whisper tokenizer contract: {0}")]
    Tokenizer(String),
    #[error("Whisper MLX evaluation failed: {0}")]
    Runtime(String),
    #[error("unsupported Whisper language: {0}")]
    Language(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperTranscription {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Clone, Debug)]
struct WhisperDims {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
}

impl WhisperDims {
    fn from_config(config: &Value) -> Result<Self, WhisperError> {
        let field = |name: &'static str| -> Result<usize, WhisperError> {
            config
                .get(name)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| WhisperError::Config(format!("missing positive field {name}")))
        };
        if config.get("model_type").and_then(Value::as_str) != Some("whisper") {
            return Err(WhisperError::Config(
                "model_type must be whisper".to_string(),
            ));
        }
        let dims = Self {
            n_mels: field("n_mels")?,
            n_audio_ctx: field("n_audio_ctx")?,
            n_audio_state: field("n_audio_state")?,
            n_audio_head: field("n_audio_head")?,
            n_audio_layer: field("n_audio_layer")?,
            n_vocab: field("n_vocab")?,
            n_text_ctx: field("n_text_ctx")?,
            n_text_state: field("n_text_state")?,
            n_text_head: field("n_text_head")?,
            n_text_layer: field("n_text_layer")?,
        };
        if dims.n_audio_ctx != 1_500
            || dims.n_audio_state != dims.n_text_state
            || !dims.n_audio_state.is_multiple_of(dims.n_audio_head)
            || !dims.n_text_state.is_multiple_of(dims.n_text_head)
            || dims.n_text_ctx > 448
        {
            return Err(WhisperError::Config(format!(
                "unsupported dimensions {dims:?}"
            )));
        }
        Ok(dims)
    }
}

pub struct WhisperModel {
    dims: WhisperDims,
    dtype: MlxDtype,
    encoder: AudioEncoder,
    decoder: TextDecoder,
    tokenizer: WhisperTokenizer,
    mel: WhisperMel,
}

impl fmt::Debug for WhisperModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WhisperModel")
            .field("dims", &self.dims)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl WhisperModel {
    pub fn load(artifacts: &NativeModelArtifacts) -> Result<Self, WhisperError> {
        if artifacts.manifest().model_family != "whisper" {
            return Err(WhisperError::Config(format!(
                "manifest family must be whisper, got {:?}",
                artifacts.manifest().model_family
            )));
        }
        let config_path = artifacts.root_dir().join("config.json");
        let config_bytes = std::fs::read(&config_path).map_err(|error| {
            WhisperError::Config(format!("read {}: {error}", config_path.display()))
        })?;
        let config: Value = serde_json::from_slice(&config_bytes)
            .map_err(|error| WhisperError::Config(format!("parse config.json: {error}")))?;
        let dims = WhisperDims::from_config(&config)?;
        let mut weights = load_checkpoint_tensors(artifacts)?;
        weights.remove("alignment_heads");

        let dtype = weights
            .get("decoder.token_embedding.weight")
            .ok_or_else(|| {
                WhisperError::Weights(
                    "missing decoder.token_embedding.weight in native checkpoint".to_string(),
                )
            })?
            .dtype();
        if !matches!(
            dtype,
            MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::Bfloat16
        ) {
            return Err(WhisperError::Weights(format!(
                "decoder token embedding must be floating point, got {dtype:?}"
            )));
        }

        let encoder = AudioEncoder::load(&mut weights, &dims, dtype)?;
        let decoder = TextDecoder::load(&mut weights, &dims, dtype)?;
        let tokenizer = WhisperTokenizer::new(dims.n_vocab).map_err(WhisperError::Tokenizer)?;
        let mel = WhisperMel::new(dims.n_mels).map_err(WhisperError::Config)?;

        Ok(Self {
            dims,
            dtype,
            encoder,
            decoder,
            tokenizer,
            mel,
        })
    }

    pub fn transcribe(
        &self,
        audio_16k: &[f32],
        language: Option<&str>,
        translate: bool,
    ) -> Result<WhisperTranscription, WhisperError> {
        if audio_16k.is_empty() {
            return Ok(WhisperTranscription {
                text: String::new(),
                language: language.map(str::to_string),
            });
        }
        if let Some(code) = language
            && self.tokenizer.language_token(code).is_none()
        {
            return Err(WhisperError::Language(code.to_string()));
        }

        // Whisper's reference frontend pads before the global log clamp. Doing
        // the same avoids feeding literal zero-valued mel frames for a short
        // clip, which the encoder interprets as mid-band energy.
        let windows = audio_16k.len().div_ceil(N_SAMPLES);
        let mut padded = Vec::with_capacity(windows.saturating_mul(N_SAMPLES));
        padded.extend_from_slice(audio_16k);
        padded.resize(windows.saturating_mul(N_SAMPLES), 0.0);
        let (mel, frames) = self.mel.extract(&padded);
        if frames == 0 {
            return Ok(WhisperTranscription {
                text: String::new(),
                language: language.map(str::to_string),
            });
        }

        let mut text = String::new();
        let mut resolved_language = language.map(str::to_string);
        let mut seek = 0usize;
        while seek < frames {
            let mut window = vec![0.0f32; N_FRAMES * self.dims.n_mels];
            let available = (frames - seek).min(N_FRAMES);
            let source_start = seek * self.dims.n_mels;
            let source_end = source_start + available * self.dims.n_mels;
            window[..available * self.dims.n_mels].copy_from_slice(&mel[source_start..source_end]);
            let mel_array = array_f32(&window, &[1, N_FRAMES as i32, self.dims.n_mels as i32]);
            let mel_array = astype(&mel_array, self.dtype, None);
            let audio_features = self.encoder.forward(&mel_array);
            let (tokens, detected) = transcribe_segment(
                &self.decoder,
                &audio_features,
                &self.tokenizer,
                self.dims.n_vocab,
                self.dims.n_text_ctx,
                resolved_language.as_deref(),
                translate,
            )?;
            if resolved_language.is_none() {
                resolved_language = detected;
            }
            text.push_str(&self.tokenizer.decode_text(&tokens));
            seek += N_FRAMES;
        }

        Ok(WhisperTranscription {
            text: text.trim().to_string(),
            language: resolved_language,
        })
    }
}

fn load_checkpoint_tensors(
    artifacts: &NativeModelArtifacts,
) -> Result<HashMap<String, MlxArray>, WhisperError> {
    let files = artifacts
        .tensor_specs()
        .iter()
        .map(|tensor| artifacts.root_dir().join(&tensor.file))
        .collect::<BTreeSet<_>>();
    let mut tensors = HashMap::new();
    for path in files {
        let loaded = load_safetensors(&path, None)
            .map_err(|error| WhisperError::Weights(format!("load {}: {error}", path.display())))?;
        let references = loaded.values().collect::<Vec<_>>();
        try_eval(&references).map_err(|error| {
            WhisperError::Runtime(format!("materialize {}: {error}", path.display()))
        })?;
        tensors.extend(loaded);
    }
    Ok(tensors)
}

#[derive(Clone)]
struct DenseLinear {
    weight: MlxArray,
    bias: Option<MlxArray>,
}

impl DenseLinear {
    fn load(weights: &mut HashMap<String, MlxArray>, prefix: &str) -> Result<Self, WhisperError> {
        Ok(Self {
            weight: take_required(weights, &format!("{prefix}.weight"))?,
            bias: weights.remove(&format!("{prefix}.bias")),
        })
    }

    fn forward(&self, input: &MlxArray) -> MlxArray {
        let input = if input.dtype() == self.weight.dtype() {
            input.clone()
        } else {
            astype(input, self.weight.dtype(), None)
        };
        let output = matmul(&input, &transpose(&self.weight, &[1, 0], None), None);
        self.bias
            .as_ref()
            .map_or_else(|| output.clone(), |bias| add(&output, bias, None))
    }
}

#[derive(Clone)]
struct LayerNormWeights {
    weight: MlxArray,
    bias: MlxArray,
}

impl LayerNormWeights {
    fn load(weights: &mut HashMap<String, MlxArray>, prefix: &str) -> Result<Self, WhisperError> {
        Ok(Self {
            weight: take_required(weights, &format!("{prefix}.weight"))?,
            bias: take_required(weights, &format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, input: &MlxArray) -> MlxArray {
        layer_norm(input, &self.weight, &self.bias, 1.0e-5, None)
    }
}

#[derive(Clone)]
struct KvCache {
    key: MlxArray,
    value: MlxArray,
}

#[derive(Clone)]
struct MultiHeadAttention {
    n_head: usize,
    head_dim: usize,
    qk_scale: f32,
    query: DenseLinear,
    key: DenseLinear,
    value: DenseLinear,
    output: DenseLinear,
}

impl MultiHeadAttention {
    fn load(
        weights: &mut HashMap<String, MlxArray>,
        prefix: &str,
        n_state: usize,
        n_head: usize,
    ) -> Result<Self, WhisperError> {
        let head_dim = n_state / n_head;
        Ok(Self {
            n_head,
            head_dim,
            // mlx-whisper scales Q and K separately before their fp16
            // matmul. Preserve that rounding contract (rather than using
            // fused SDPA's post-matmul scale) and request precise softmax.
            qk_scale: (head_dim as f32).powf(-0.25),
            query: DenseLinear::load(weights, &format!("{prefix}.query"))?,
            key: DenseLinear::load(weights, &format!("{prefix}.key"))?,
            value: DenseLinear::load(weights, &format!("{prefix}.value"))?,
            output: DenseLinear::load(weights, &format!("{prefix}.out"))?,
        })
    }

    fn self_attention(
        &self,
        input: &MlxArray,
        cache: &mut Option<KvCache>,
        causal: bool,
    ) -> MlxArray {
        let query = self.query.forward(input);
        let key_new = self.key.forward(input);
        let value_new = self.value.forward(input);
        let had_cache = cache.is_some();
        let (key, value) = match cache.take() {
            Some(previous) => (
                concatenate(&[&previous.key, &key_new], 1, None),
                concatenate(&[&previous.value, &value_new], 1, None),
            ),
            None => (key_new, value_new),
        };
        let causal = causal && !had_cache && query.shape()[1] > 1;
        let attended = self.attend(&query, &key, &value, causal);
        *cache = Some(KvCache { key, value });
        self.output.forward(&attended)
    }

    fn cross_attention(
        &self,
        input: &MlxArray,
        audio_features: &MlxArray,
        cache: &mut Option<KvCache>,
    ) -> MlxArray {
        let query = self.query.forward(input);
        let (key, value) = match cache.take() {
            Some(previous) => (previous.key, previous.value),
            None => (
                self.key.forward(audio_features),
                self.value.forward(audio_features),
            ),
        };
        let attended = self.attend(&query, &key, &value, false);
        *cache = Some(KvCache { key, value });
        self.output.forward(&attended)
    }

    fn attend(&self, query: &MlxArray, key: &MlxArray, value: &MlxArray, causal: bool) -> MlxArray {
        let query_shape = query.shape();
        let key_shape = key.shape();
        let batch = query_shape[0];
        let query_length = query_shape[1];
        let key_length = key_shape[1];
        let state = query_shape[2];
        let query = reshape(
            query,
            &[
                batch,
                query_length,
                self.n_head as i32,
                self.head_dim as i32,
            ],
            None,
        );
        let key = reshape(
            key,
            &[batch, key_length, self.n_head as i32, self.head_dim as i32],
            None,
        );
        let value = reshape(
            value,
            &[batch, key_length, self.n_head as i32, self.head_dim as i32],
            None,
        );
        let query = transpose(&query, &[0, 2, 1, 3], None);
        let key = transpose(&key, &[0, 2, 3, 1], None);
        let value = transpose(&value, &[0, 2, 1, 3], None);
        let scale = astype(&array_f32(&[self.qk_scale], &[1]), query.dtype(), None);
        let query = multiply(&query, &scale, None);
        let key = multiply(&key, &scale, None);
        let mut scores = matmul(&query, &key, None);
        if causal {
            let mut mask = vec![0.0f32; query_length as usize * key_length as usize];
            for row in 0..query_length as usize {
                for column in row + 1..key_length as usize {
                    mask[row * key_length as usize + column] = f32::NEG_INFINITY;
                }
            }
            let mask = astype(
                &array_f32(&mask, &[query_length, key_length]),
                scores.dtype(),
                None,
            );
            scores = add(&scores, &mask, None);
        }
        let probabilities = softmax_precise(&scores, -1, None);
        let attended = matmul(&probabilities, &value, None);
        let attended = transpose(&attended, &[0, 2, 1, 3], None);
        contiguous(
            &reshape(&attended, &[batch, query_length, state], None),
            None,
        )
    }
}

#[derive(Clone)]
struct ResidualAttentionBlock {
    attention: MultiHeadAttention,
    attention_norm: LayerNormWeights,
    cross_attention: Option<MultiHeadAttention>,
    cross_attention_norm: Option<LayerNormWeights>,
    mlp1: DenseLinear,
    mlp2: DenseLinear,
    mlp_norm: LayerNormWeights,
}

impl ResidualAttentionBlock {
    fn load(
        weights: &mut HashMap<String, MlxArray>,
        prefix: &str,
        n_state: usize,
        n_head: usize,
        cross_attention: bool,
    ) -> Result<Self, WhisperError> {
        Ok(Self {
            attention: MultiHeadAttention::load(
                weights,
                &format!("{prefix}.attn"),
                n_state,
                n_head,
            )?,
            attention_norm: LayerNormWeights::load(weights, &format!("{prefix}.attn_ln"))?,
            cross_attention: cross_attention
                .then(|| {
                    MultiHeadAttention::load(
                        weights,
                        &format!("{prefix}.cross_attn"),
                        n_state,
                        n_head,
                    )
                })
                .transpose()?,
            cross_attention_norm: cross_attention
                .then(|| LayerNormWeights::load(weights, &format!("{prefix}.cross_attn_ln")))
                .transpose()?,
            mlp1: DenseLinear::load(weights, &format!("{prefix}.mlp1"))?,
            mlp2: DenseLinear::load(weights, &format!("{prefix}.mlp2"))?,
            mlp_norm: LayerNormWeights::load(weights, &format!("{prefix}.mlp_ln"))?,
        })
    }

    fn forward(
        &self,
        input: &MlxArray,
        audio_features: Option<&MlxArray>,
        self_cache: &mut Option<KvCache>,
        cross_cache: &mut Option<KvCache>,
        causal: bool,
    ) -> MlxArray {
        let attention =
            self.attention
                .self_attention(&self.attention_norm.forward(input), self_cache, causal);
        let mut hidden = add(input, &attention, None);
        if let (Some(cross_attention), Some(cross_norm), Some(audio_features)) = (
            &self.cross_attention,
            &self.cross_attention_norm,
            audio_features,
        ) {
            let attention = cross_attention.cross_attention(
                &cross_norm.forward(&hidden),
                audio_features,
                cross_cache,
            );
            hidden = add(&hidden, &attention, None);
        }
        let mlp = self.mlp2.forward(&gelu(
            &self.mlp1.forward(&self.mlp_norm.forward(&hidden)),
            None,
        ));
        add(&hidden, &mlp, None)
    }
}

struct AudioEncoder {
    conv1_weight: MlxArray,
    conv1_bias: MlxArray,
    conv2_weight: MlxArray,
    conv2_bias: MlxArray,
    positional_embedding: MlxArray,
    blocks: Vec<ResidualAttentionBlock>,
    final_norm: LayerNormWeights,
}

impl AudioEncoder {
    fn load(
        weights: &mut HashMap<String, MlxArray>,
        dims: &WhisperDims,
        dtype: MlxDtype,
    ) -> Result<Self, WhisperError> {
        let positional = sinusoids(dims.n_audio_ctx, dims.n_audio_state);
        let positional = astype(
            &array_f32(
                &positional,
                &[dims.n_audio_ctx as i32, dims.n_audio_state as i32],
            ),
            dtype,
            None,
        );
        let blocks = (0..dims.n_audio_layer)
            .map(|layer| {
                ResidualAttentionBlock::load(
                    weights,
                    &format!("encoder.blocks.{layer}"),
                    dims.n_audio_state,
                    dims.n_audio_head,
                    false,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            conv1_weight: take_required(weights, "encoder.conv1.weight")?,
            conv1_bias: take_required(weights, "encoder.conv1.bias")?,
            conv2_weight: take_required(weights, "encoder.conv2.weight")?,
            conv2_bias: take_required(weights, "encoder.conv2.bias")?,
            positional_embedding: positional,
            blocks,
            final_norm: LayerNormWeights::load(weights, "encoder.ln_post")?,
        })
    }

    fn forward(&self, mel: &MlxArray) -> MlxArray {
        let hidden = conv1d(mel, &self.conv1_weight, 1, 1, 1, 1, None);
        let hidden = gelu(&add(&hidden, &self.conv1_bias, None), None);
        let hidden = conv1d(&hidden, &self.conv2_weight, 2, 1, 1, 1, None);
        let hidden = gelu(&add(&hidden, &self.conv2_bias, None), None);
        let mut hidden = add(&hidden, &self.positional_embedding, None);
        for block in &self.blocks {
            let mut self_cache = None;
            let mut cross_cache = None;
            hidden = block.forward(&hidden, None, &mut self_cache, &mut cross_cache, false);
        }
        self.final_norm.forward(&hidden)
    }
}

struct TextDecoder {
    token_embedding: MlxArray,
    positional_embedding: MlxArray,
    blocks: Vec<ResidualAttentionBlock>,
    final_norm: LayerNormWeights,
    n_state: usize,
}

impl TextDecoder {
    fn load(
        weights: &mut HashMap<String, MlxArray>,
        dims: &WhisperDims,
        _dtype: MlxDtype,
    ) -> Result<Self, WhisperError> {
        let blocks = (0..dims.n_text_layer)
            .map(|layer| {
                ResidualAttentionBlock::load(
                    weights,
                    &format!("decoder.blocks.{layer}"),
                    dims.n_text_state,
                    dims.n_text_head,
                    true,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            token_embedding: take_required(weights, "decoder.token_embedding.weight")?,
            positional_embedding: take_required(weights, "decoder.positional_embedding")?,
            blocks,
            final_norm: LayerNormWeights::load(weights, "decoder.ln")?,
            n_state: dims.n_text_state,
        })
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn forward(
        &self,
        tokens: &MlxArray,
        audio_features: &MlxArray,
        offset: usize,
        self_caches: &mut [Option<KvCache>],
        cross_caches: &mut [Option<KvCache>],
    ) -> MlxArray {
        let sequence = tokens.shape()[1] as usize;
        let embeddings = take(&self.token_embedding, tokens, 0, None);
        let positions = slice(
            &self.positional_embedding,
            &[offset as i32, 0],
            &[offset.saturating_add(sequence) as i32, self.n_state as i32],
            &[1, 1],
            None,
        );
        let mut hidden = add(&embeddings, &positions, None);
        for (layer, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(
                &hidden,
                Some(audio_features),
                &mut self_caches[layer],
                &mut cross_caches[layer],
                true,
            );
        }
        let hidden = self.final_norm.forward(&hidden);
        matmul(
            &hidden,
            &transpose(&self.token_embedding, &[1, 0], None),
            None,
        )
    }
}

fn transcribe_segment(
    decoder: &TextDecoder,
    audio_features: &MlxArray,
    tokenizer: &WhisperTokenizer,
    n_vocab: usize,
    n_text_ctx: usize,
    language: Option<&str>,
    translate: bool,
) -> Result<(Vec<u32>, Option<String>), WhisperError> {
    let resolved = match language {
        Some(language) => Some(language.to_string()),
        None => detect_language(decoder, audio_features, tokenizer, n_vocab)?,
    };
    let initial = tokenizer
        .initial_tokens(resolved.as_deref(), translate)
        .map_err(WhisperError::Language)?;
    let always_mask = array_f32(&suppression_mask(tokenizer, n_vocab), &[1, n_vocab as i32]);
    let first_mask = array_f32(&first_step_mask(tokenizer, n_vocab), &[1, n_vocab as i32]);
    let mut self_caches = empty_caches(decoder.num_layers());
    let mut cross_caches = empty_caches(decoder.num_layers());
    let initial_array = array_u32(&initial, &[1, initial.len() as i32]);
    let logits = decoder.forward(
        &initial_array,
        audio_features,
        0,
        &mut self_caches,
        &mut cross_caches,
    );
    let mut next = argmax_with_masks(
        &logits,
        initial.len() - 1,
        n_vocab,
        &always_mask,
        Some(&first_mask),
    )?;
    let mut generated = Vec::new();
    let mut offset = initial.len();
    let max_new = n_text_ctx / 2;
    while next != tokenizer.eot && generated.len() < max_new && offset < n_text_ctx {
        generated.push(next);
        let token = array_u32(&[next], &[1, 1]);
        let logits = decoder.forward(
            &token,
            audio_features,
            offset,
            &mut self_caches,
            &mut cross_caches,
        );
        offset += 1;
        next = argmax_with_masks(&logits, 0, n_vocab, &always_mask, None)?;
    }
    Ok((generated, resolved))
}

fn detect_language(
    decoder: &TextDecoder,
    audio_features: &MlxArray,
    tokenizer: &WhisperTokenizer,
    n_vocab: usize,
) -> Result<Option<String>, WhisperError> {
    let mut self_caches = empty_caches(decoder.num_layers());
    let mut cross_caches = empty_caches(decoder.num_layers());
    let start = array_u32(&[tokenizer.sot], &[1, 1]);
    let logits = decoder.forward(
        &start,
        audio_features,
        0,
        &mut self_caches,
        &mut cross_caches,
    );
    let mut mask = vec![f32::NEG_INFINITY; n_vocab];
    for (_, token) in &tokenizer.language_ids {
        if let Some(value) = mask.get_mut(*token as usize) {
            *value = 0.0;
        }
    }
    let mask = array_f32(&mask, &[1, n_vocab as i32]);
    let token = argmax_with_masks(&logits, 0, n_vocab, &mask, None)?;
    Ok(tokenizer.language_for_token(token).map(str::to_string))
}

fn suppression_mask(tokenizer: &WhisperTokenizer, n_vocab: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; n_vocab];
    for token in &tokenizer.suppress {
        if let Some(value) = mask.get_mut(*token as usize) {
            *value = f32::NEG_INFINITY;
        }
    }
    for value in mask.iter_mut().skip(tokenizer.timestamp_begin as usize) {
        *value = f32::NEG_INFINITY;
    }
    mask
}

fn first_step_mask(tokenizer: &WhisperTokenizer, n_vocab: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; n_vocab];
    for token in [tokenizer.blank, tokenizer.eot] {
        if let Some(value) = mask.get_mut(token as usize) {
            *value = f32::NEG_INFINITY;
        }
    }
    mask
}

fn argmax_with_masks(
    logits: &MlxArray,
    position: usize,
    n_vocab: usize,
    always_mask: &MlxArray,
    first_mask: Option<&MlxArray>,
) -> Result<u32, WhisperError> {
    let logits = slice(
        logits,
        &[0, position as i32, 0],
        &[1, position.saturating_add(1) as i32, n_vocab as i32],
        &[1, 1, 1],
        None,
    );
    let logits = reshape(&logits, &[1, n_vocab as i32], None);
    let mut logits = add(&astype(&logits, MlxDtype::Float32, None), always_mask, None);
    if let Some(first_mask) = first_mask {
        logits = add(&logits, first_mask, None);
    }
    let token = argmax(&logits, None);
    try_eval(&[&token]).map_err(WhisperError::Runtime)?;
    token
        .data_u32()
        .first()
        .copied()
        .ok_or_else(|| WhisperError::Runtime("argmax returned no token".to_string()))
}

fn empty_caches(size: usize) -> Vec<Option<KvCache>> {
    (0..size).map(|_| None).collect()
}

fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let log_increment = 10_000.0f64.ln() / (half.saturating_sub(1).max(1)) as f64;
    let inverse_timescales = (0..half)
        .map(|index| (-log_increment * index as f64).exp())
        .collect::<Vec<_>>();
    let mut output = vec![0.0f32; length * channels];
    for position in 0..length {
        for (index, inverse) in inverse_timescales.iter().copied().enumerate() {
            let scaled = position as f64 * inverse;
            output[position * channels + index] = scaled.sin() as f32;
            output[position * channels + half + index] = scaled.cos() as f32;
        }
    }
    output
}

fn take_required(
    weights: &mut HashMap<String, MlxArray>,
    name: &str,
) -> Result<MlxArray, WhisperError> {
    weights
        .remove(name)
        .ok_or_else(|| WhisperError::Weights(format!("missing tensor {name}")))
}

fn array_f32(values: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Float32,
    )
}

fn array_u32(values: &[u32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Uint32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_dimensions_parse_exact_native_config() {
        let dims = WhisperDims::from_config(&serde_json::json!({
            "model_type": "whisper",
            "n_mels": 128,
            "n_audio_ctx": 1500,
            "n_audio_state": 1280,
            "n_audio_head": 20,
            "n_audio_layer": 32,
            "n_vocab": 51866,
            "n_text_ctx": 448,
            "n_text_state": 1280,
            "n_text_head": 20,
            "n_text_layer": 4
        }))
        .expect("large-v3-turbo dimensions should parse");
        assert_eq!(dims.n_audio_layer, 32);
        assert_eq!(dims.n_text_layer, 4);
        assert_eq!(dims.n_mels, 128);
    }

    #[test]
    fn no_timestamp_mask_keeps_eot_and_blocks_timestamp_range() {
        let tokenizer = WhisperTokenizer::new(51_866).expect("tokenizer should load");
        let mask = suppression_mask(&tokenizer, 51_866);
        assert_eq!(mask[tokenizer.eot as usize], 0.0);
        assert_eq!(mask[tokenizer.timestamp_begin as usize], f32::NEG_INFINITY);
        assert_eq!(mask[tokenizer.sot as usize], f32::NEG_INFINITY);
    }
}
