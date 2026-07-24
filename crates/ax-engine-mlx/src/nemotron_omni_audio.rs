//! Parakeet/Conformer audio tower for Nemotron H Nano Omni.
//!
//! This follows the released checkpoint and mlxcel implementation: a
//! pre-emphasized Slaney log-mel frontend, three-stage Conv2D subsampling,
//! Transformer-XL relative-position Conformer blocks, and an RMSNorm/MLP
//! projection into the Nemotron-H text width.
//!
//! Portions of the mathematical implementation are derived from mlxcel's
//! Apache-2.0 Nemotron H Nano Omni audio implementation.

use std::collections::HashMap;
use std::f64::consts::PI;

use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, astype, broadcast_to, conv1d, conv2d,
    cos, expand_dims, layer_norm, matmul, multiply, pad, reshape, rms_norm,
    scaled_dot_product_attention_with_mask, sigmoid, sin, slice, stack, subtract, transpose, zeros,
};
use serde_json::Value;

use crate::model::shared::qw;
use crate::nemotron_omni::{load_quantized_linear, take_optional, take_required};
use crate::weights::{QuantizedWeight, WeightLoadError};

#[derive(Clone, Debug)]
pub(crate) struct NemotronOmniAudioConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub conv_kernel_size: usize,
    pub convolution_bias: bool,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub num_mel_bins: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub sampling_rate: u32,
    pub hop_length: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub preemphasis: f32,
}

impl NemotronOmniAudioConfig {
    pub(crate) fn from_json(root: &Value) -> Result<Self, WeightLoadError> {
        let sound = root.get("sound_config").ok_or_else(|| {
            WeightLoadError::InvalidLayer(
                "Nemotron H Nano Omni config has no sound_config".to_string(),
            )
        })?;
        let usize_field = |key: &str, fallback: usize| {
            sound
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(fallback)
        };
        let config = Self {
            hidden_size: usize_field("hidden_size", 1024),
            num_attention_heads: usize_field("num_attention_heads", 8),
            num_hidden_layers: usize_field("num_hidden_layers", 24),
            conv_kernel_size: usize_field("conv_kernel_size", 9),
            convolution_bias: sound
                .get("convolution_bias")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            subsampling_factor: usize_field("subsampling_factor", 8),
            subsampling_conv_channels: usize_field("subsampling_conv_channels", 256),
            num_mel_bins: usize_field("num_mel_bins", 128),
            subsampling_conv_kernel_size: usize_field("subsampling_conv_kernel_size", 3),
            subsampling_conv_stride: usize_field("subsampling_conv_stride", 2),
            sampling_rate: sound
                .get("sampling_rate")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(16_000),
            hop_length: usize_field("hop_length", 160),
            n_fft: usize_field("n_fft", 512),
            win_length: usize_field("win_length", 400),
            preemphasis: sound
                .get("preemphasis")
                .and_then(Value::as_f64)
                .unwrap_or(0.97) as f32,
        };
        if config.hidden_size == 0
            || config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
            || config.num_hidden_layers == 0
            || !config.subsampling_factor.is_power_of_two()
            || config.subsampling_conv_channels == 0
            || config.num_mel_bins == 0
            || config.subsampling_conv_kernel_size == 0
            || config.subsampling_conv_stride == 0
            || config.sampling_rate == 0
            || config.hop_length == 0
            || config.n_fft == 0
            || config.win_length == 0
            || config.win_length > config.n_fft
            || !config.preemphasis.is_finite()
        {
            return Err(WeightLoadError::InvalidLayer(format!(
                "invalid Nemotron Omni sound_config: {config:?}"
            )));
        }
        Ok(config)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn num_subsampling_layers(&self) -> usize {
        self.subsampling_factor.ilog2() as usize
    }

    pub(crate) fn subsampling_output_length(&self, mut length: usize) -> usize {
        let kernel = self.subsampling_conv_kernel_size as i64;
        let stride = self.subsampling_conv_stride as i64;
        let add_pad = ((kernel - 1) / 2) * 2 - kernel;
        for _ in 0..self.num_subsampling_layers() {
            let signed = length as i64 + add_pad;
            length = if signed < 0 {
                0
            } else {
                (signed / stride + 1) as usize
            };
        }
        length
    }
}

#[derive(Clone)]
pub(crate) struct NemotronOmniFeatureExtractor {
    config: NemotronOmniAudioConfig,
    window: Vec<f32>,
    mel_filters: Vec<f32>,
    /// `(cos(angle), sin(angle))` for `[frequency_bin, fft_sample]`.
    fft_twiddles: Vec<(f64, f64)>,
}

pub(crate) struct ExtractedAudio {
    pub features: Vec<f32>,
    pub frames: usize,
    pub valid_frames: usize,
}

impl NemotronOmniFeatureExtractor {
    pub(crate) fn new(config: NemotronOmniAudioConfig) -> Self {
        let window = centered_symmetric_hann(config.win_length, config.n_fft);
        let mel_filters =
            slaney_mel_filterbank(config.sampling_rate, config.n_fft, config.num_mel_bins);
        let bins = config.n_fft / 2 + 1;
        let mut fft_twiddles = Vec::with_capacity(bins * config.n_fft);
        for k in 0..bins {
            for t in 0..config.n_fft {
                let angle = -2.0 * PI * k as f64 * t as f64 / config.n_fft as f64;
                fft_twiddles.push((angle.cos(), angle.sin()));
            }
        }
        Self {
            config,
            window,
            mel_filters,
            fft_twiddles,
        }
    }

    pub(crate) fn extract_clip(&self, waveform: &[f32]) -> ExtractedAudio {
        let bins = self.config.n_fft / 2 + 1;
        let half_window = self.config.n_fft / 2;
        let padded_len = waveform.len().saturating_add(2 * half_window);
        // torch.stft defaults to center=true: n_fft / 2 zeros on each
        // side yield exactly `1 + floor(samples / hop)` frames.
        let frames = waveform.len() / self.config.hop_length + 1;
        let mut preemphasized = Vec::with_capacity(waveform.len());
        if let Some(first) = waveform.first() {
            preemphasized.push(*first);
            preemphasized.extend(
                waveform
                    .windows(2)
                    .map(|pair| pair[1] - self.config.preemphasis * pair[0]),
            );
        }
        let mut padded = vec![0.0f32; padded_len];
        if !preemphasized.is_empty() {
            padded[half_window..half_window + preemphasized.len()].copy_from_slice(&preemphasized);
        }

        let mut fft_input = vec![0.0f64; self.config.n_fft];
        let mut power = vec![0.0f64; bins];
        let mut features = vec![0.0f32; frames * self.config.num_mel_bins];
        let mel_floor = 2.0f64.powi(-24);
        for frame in 0..frames {
            let start = frame * self.config.hop_length;
            for (index, value) in fft_input.iter_mut().enumerate() {
                *value = padded.get(start + index).copied().unwrap_or(0.0) as f64
                    * self.window[index] as f64;
            }
            for (bin, output) in power.iter_mut().enumerate() {
                let mut real = 0.0f64;
                let mut imaginary = 0.0f64;
                let twiddles =
                    &self.fft_twiddles[bin * self.config.n_fft..(bin + 1) * self.config.n_fft];
                for (sample, (cosine, sine)) in fft_input.iter().zip(twiddles) {
                    real += sample * cosine;
                    imaginary += sample * sine;
                }
                *output = real * real + imaginary * imaginary;
            }
            for mel in 0..self.config.num_mel_bins {
                let filter = &self.mel_filters[mel * bins..(mel + 1) * bins];
                let energy = filter
                    .iter()
                    .zip(&power)
                    .map(|(weight, value)| f64::from(*weight) * value)
                    .sum::<f64>();
                features[frame * self.config.num_mel_bins + mel] = (energy + mel_floor).ln() as f32;
            }
        }
        let valid_frames = (waveform.len() / self.config.hop_length).min(frames);
        normalize_log_mel(
            &mut features,
            frames,
            self.config.num_mel_bins,
            valid_frames,
        );
        ExtractedAudio {
            features,
            frames,
            valid_frames,
        }
    }
}

fn centered_symmetric_hann(win_length: usize, n_fft: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n_fft];
    let left = (n_fft - win_length) / 2;
    if win_length == 1 {
        output[left] = 1.0;
        return output;
    }
    for index in 0..win_length {
        // Parakeet uses torch.hann_window(periodic=false).
        output[left + index] =
            (0.5 - 0.5 * (2.0 * PI * index as f64 / (win_length - 1) as f64).cos()) as f32;
    }
    output
}

fn slaney_mel_filterbank(sample_rate: u32, n_fft: usize, mel_bins: usize) -> Vec<f32> {
    let frequency_bins = n_fft / 2 + 1;
    let mel_min = hz_to_slaney_mel(0.0);
    let mel_max = hz_to_slaney_mel(sample_rate as f64 / 2.0);
    let edges = (0..mel_bins + 2)
        .map(|index| {
            let fraction = index as f64 / (mel_bins + 1) as f64;
            slaney_mel_to_hz(mel_min + fraction * (mel_max - mel_min))
        })
        .collect::<Vec<_>>();
    let mut filters = vec![0.0f32; mel_bins * frequency_bins];
    for mel in 0..mel_bins {
        let lower = edges[mel];
        let center = edges[mel + 1];
        let upper = edges[mel + 2];
        let scale = 2.0 / (upper - lower).max(1.0e-12);
        for bin in 0..frequency_bins {
            let frequency = bin as f64 * sample_rate as f64 / n_fft as f64;
            let rising = (frequency - lower) / (center - lower).max(1.0e-12);
            let falling = (upper - frequency) / (upper - center).max(1.0e-12);
            filters[mel * frequency_bins + bin] = (rising.min(falling).max(0.0) * scale) as f32;
        }
    }
    filters
}

fn hz_to_slaney_mel(frequency: f64) -> f64 {
    const FREQ_STEP: f64 = 200.0 / 3.0;
    const LOG_START_HZ: f64 = 1000.0;
    const LOG_START_MEL: f64 = LOG_START_HZ / FREQ_STEP;
    const LOG_STEP: f64 = 0.068_751_777_56;
    if frequency >= LOG_START_HZ {
        LOG_START_MEL + (frequency / LOG_START_HZ).ln() / LOG_STEP
    } else {
        frequency / FREQ_STEP
    }
}

fn slaney_mel_to_hz(mel: f64) -> f64 {
    const FREQ_STEP: f64 = 200.0 / 3.0;
    const LOG_START_HZ: f64 = 1000.0;
    const LOG_START_MEL: f64 = LOG_START_HZ / FREQ_STEP;
    const LOG_STEP: f64 = 0.068_751_777_56;
    if mel >= LOG_START_MEL {
        LOG_START_HZ * (LOG_STEP * (mel - LOG_START_MEL)).exp()
    } else {
        FREQ_STEP * mel
    }
}

fn normalize_log_mel(features: &mut [f32], frames: usize, mel_bins: usize, valid_frames: usize) {
    let mean_denominator = valid_frames.max(1) as f64;
    let variance_denominator = valid_frames.saturating_sub(1).max(1) as f64;
    let mut means = vec![0.0f64; mel_bins];
    for frame in 0..valid_frames.min(frames) {
        for mel in 0..mel_bins {
            means[mel] += f64::from(features[frame * mel_bins + mel]);
        }
    }
    for mean in &mut means {
        *mean /= mean_denominator;
    }
    let mut variances = vec![0.0f64; mel_bins];
    for frame in 0..valid_frames.min(frames) {
        for mel in 0..mel_bins {
            let delta = f64::from(features[frame * mel_bins + mel]) - means[mel];
            variances[mel] += delta * delta;
        }
    }
    for variance in &mut variances {
        *variance /= variance_denominator;
    }
    for frame in 0..frames {
        for mel in 0..mel_bins {
            features[frame * mel_bins + mel] = if frame < valid_frames {
                (features[frame * mel_bins + mel] - means[mel] as f32)
                    / (variances[mel].sqrt() as f32 + 1.0e-5)
            } else {
                0.0
            };
        }
    }
}

#[derive(Clone)]
struct AudioLayerNorm {
    weight: MlxArray,
    bias: MlxArray,
    eps: f32,
}

impl AudioLayerNorm {
    fn load(map: &mut HashMap<String, MlxArray>, base: &str) -> Result<Self, WeightLoadError> {
        let weight = take_required(map, &format!("{base}.weight"))?;
        let bias = take_optional(map, &format!("{base}.bias"))
            .unwrap_or_else(|| zeros(&weight.shape(), weight.dtype(), None));
        Ok(Self {
            weight,
            bias,
            eps: 1.0e-5,
        })
    }

    fn forward(&self, input: &MlxArray) -> MlxArray {
        layer_norm(input, &self.weight, &self.bias, self.eps, None)
    }
}

#[derive(Clone)]
struct AudioBatchNorm {
    weight: MlxArray,
    bias: MlxArray,
    running_mean: MlxArray,
    running_var: MlxArray,
}

impl AudioBatchNorm {
    fn load(map: &mut HashMap<String, MlxArray>, base: &str) -> Result<Self, WeightLoadError> {
        Ok(Self {
            weight: take_required(map, &format!("{base}.weight"))?,
            bias: take_required(map, &format!("{base}.bias"))?,
            running_mean: take_required(map, &format!("{base}.running_mean"))?,
            running_var: take_required(map, &format!("{base}.running_var"))?,
        })
    }

    fn forward(&self, input: &MlxArray) -> MlxArray {
        let dtype = input.dtype();
        let epsilon = scalar(1.0e-5, dtype);
        let variance = add(&self.running_var, &epsilon, None);
        let half = scalar(-0.5, dtype);
        let inverse_std = mlx_sys::power(&variance, &half, None);
        let normalized = multiply(
            &subtract(input, &self.running_mean, None),
            &inverse_std,
            None,
        );
        add(&multiply(&normalized, &self.weight, None), &self.bias, None)
    }
}

#[derive(Clone)]
struct SubsamplingConv {
    weight: MlxArray,
    bias: MlxArray,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl SubsamplingConv {
    fn forward(&self, input: &MlxArray) -> MlxArray {
        add(
            &conv2d(
                input,
                &self.weight,
                self.stride as i32,
                self.padding as i32,
                1,
                self.groups as i32,
                None,
            ),
            &self.bias,
            None,
        )
    }
}

#[derive(Clone)]
struct AudioSubsampling {
    convs: Vec<SubsamplingConv>,
    relu_after: Vec<bool>,
    linear: QuantizedWeight,
    kernel: usize,
    layer_strides: Vec<usize>,
}

impl AudioSubsampling {
    fn load(
        map: &mut HashMap<String, MlxArray>,
        config: &NemotronOmniAudioConfig,
        transpose_convs: bool,
    ) -> Result<Self, WeightLoadError> {
        let base = "sound_encoder.encoder.subsampling";
        let stride = config.subsampling_conv_stride;
        let padding = (config.subsampling_conv_kernel_size - 1) / 2;
        let mut convs = vec![SubsamplingConv {
            weight: take_audio_conv(map, &format!("{base}.layers.0.weight"), transpose_convs)?,
            bias: take_required(map, &format!("{base}.layers.0.bias"))?,
            stride,
            padding,
            groups: 1,
        }];
        let mut relu_after = vec![true];
        let mut layer_strides = vec![stride];
        let mut checkpoint_index = 2usize;
        for _ in 0..config.num_subsampling_layers().saturating_sub(1) {
            convs.push(SubsamplingConv {
                weight: take_audio_conv(
                    map,
                    &format!("{base}.layers.{checkpoint_index}.weight"),
                    transpose_convs,
                )?,
                bias: take_required(map, &format!("{base}.layers.{checkpoint_index}.bias"))?,
                stride,
                padding,
                groups: config.subsampling_conv_channels,
            });
            relu_after.push(false);
            layer_strides.push(stride);
            checkpoint_index += 1;
            convs.push(SubsamplingConv {
                weight: take_audio_conv(
                    map,
                    &format!("{base}.layers.{checkpoint_index}.weight"),
                    transpose_convs,
                )?,
                bias: take_required(map, &format!("{base}.layers.{checkpoint_index}.bias"))?,
                stride: 1,
                padding: 0,
                groups: 1,
            });
            relu_after.push(true);
            layer_strides.push(1);
            checkpoint_index += 2;
        }
        Ok(Self {
            convs,
            relu_after,
            linear: load_quantized_linear(map, &format!("{base}.linear"), 64, 4)?,
            kernel: config.subsampling_conv_kernel_size,
            layer_strides,
        })
    }

    fn forward(&self, input: &MlxArray, mut valid_length: usize) -> (MlxArray, usize) {
        let mut hidden = expand_dims(input, -1, None);
        for (index, conv) in self.convs.iter().enumerate() {
            hidden = conv.forward(&hidden);
            if self.layer_strides[index] != 1 {
                valid_length =
                    conv_output_length(valid_length, self.kernel, self.layer_strides[index]);
            }
            hidden = multiply(
                &hidden,
                &time_mask_4d(hidden.shape()[1] as usize, valid_length, hidden.dtype()),
                None,
            );
            if self.relu_after[index] {
                hidden = relu(&hidden);
            }
        }
        let shape = hidden.shape();
        let transposed = transpose(&hidden, &[0, 1, 3, 2], None);
        let flattened = reshape(&transposed, &[shape[0], shape[1], -1], None);
        (qw(&flattened, &self.linear), valid_length)
    }
}

#[derive(Clone)]
struct AudioFeedForward {
    linear1: QuantizedWeight,
    linear2: QuantizedWeight,
}

impl AudioFeedForward {
    fn load(map: &mut HashMap<String, MlxArray>, base: &str) -> Result<Self, WeightLoadError> {
        Ok(Self {
            linear1: load_quantized_linear(map, &format!("{base}.linear1"), 64, 4)?,
            linear2: load_quantized_linear(map, &format!("{base}.linear2"), 64, 4)?,
        })
    }

    fn forward(&self, input: &MlxArray) -> MlxArray {
        qw(&silu(&qw(input, &self.linear1), None), &self.linear2)
    }
}

#[derive(Clone)]
struct AudioConvModule {
    pointwise1: MlxArray,
    pointwise1_bias: Option<MlxArray>,
    depthwise: MlxArray,
    depthwise_bias: Option<MlxArray>,
    norm: AudioBatchNorm,
    pointwise2: MlxArray,
    pointwise2_bias: Option<MlxArray>,
    kernel: usize,
    channels: usize,
}

impl AudioConvModule {
    fn load(
        map: &mut HashMap<String, MlxArray>,
        base: &str,
        config: &NemotronOmniAudioConfig,
        transpose_convs: bool,
    ) -> Result<Self, WeightLoadError> {
        let required_bias = |map: &mut HashMap<String, MlxArray>, name: String| {
            if config.convolution_bias {
                take_required(map, &name).map(Some)
            } else {
                Ok(take_optional(map, &name))
            }
        };
        Ok(Self {
            pointwise1: take_audio_conv(
                map,
                &format!("{base}.pointwise_conv1.weight"),
                transpose_convs,
            )?,
            pointwise1_bias: required_bias(map, format!("{base}.pointwise_conv1.bias"))?,
            depthwise: take_audio_conv(
                map,
                &format!("{base}.depthwise_conv.weight"),
                transpose_convs,
            )?,
            depthwise_bias: required_bias(map, format!("{base}.depthwise_conv.bias"))?,
            norm: AudioBatchNorm::load(map, &format!("{base}.norm"))?,
            pointwise2: take_audio_conv(
                map,
                &format!("{base}.pointwise_conv2.weight"),
                transpose_convs,
            )?,
            pointwise2_bias: required_bias(map, format!("{base}.pointwise_conv2.bias"))?,
            kernel: config.conv_kernel_size,
            channels: config.hidden_size,
        })
    }

    fn forward(&self, input: &MlxArray, valid_length: usize) -> MlxArray {
        let hidden = conv1d(input, &self.pointwise1, 1, 0, 1, 1, None);
        let hidden = add_optional_bias(hidden, self.pointwise1_bias.as_ref());
        let split = hidden.shape()[2] / 2;
        let first = slice(
            &hidden,
            &[0, 0, 0],
            &[hidden.shape()[0], hidden.shape()[1], split],
            &[1, 1, 1],
            None,
        );
        let second = slice(
            &hidden,
            &[0, 0, split],
            &[hidden.shape()[0], hidden.shape()[1], hidden.shape()[2]],
            &[1, 1, 1],
            None,
        );
        let hidden = multiply(&first, &sigmoid(&second, None), None);
        let hidden = multiply(
            &hidden,
            &time_mask_3d(hidden.shape()[1] as usize, valid_length, hidden.dtype()),
            None,
        );
        let hidden = conv1d(
            &hidden,
            &self.depthwise,
            1,
            ((self.kernel - 1) / 2) as i32,
            1,
            self.channels as i32,
            None,
        );
        let hidden = add_optional_bias(hidden, self.depthwise_bias.as_ref());
        let hidden = silu(&self.norm.forward(&hidden), None);
        let hidden = conv1d(&hidden, &self.pointwise2, 1, 0, 1, 1, None);
        add_optional_bias(hidden, self.pointwise2_bias.as_ref())
    }
}

#[derive(Clone)]
struct AudioAttention {
    q_proj: QuantizedWeight,
    k_proj: QuantizedWeight,
    v_proj: QuantizedWeight,
    o_proj: QuantizedWeight,
    relative_k_proj: QuantizedWeight,
    bias_u: MlxArray,
    bias_v: MlxArray,
    heads: usize,
    head_dim: usize,
    scale: f32,
}

impl AudioAttention {
    fn load(
        map: &mut HashMap<String, MlxArray>,
        base: &str,
        config: &NemotronOmniAudioConfig,
    ) -> Result<Self, WeightLoadError> {
        Ok(Self {
            q_proj: load_quantized_linear(map, &format!("{base}.q_proj"), 64, 4)?,
            k_proj: load_quantized_linear(map, &format!("{base}.k_proj"), 64, 4)?,
            v_proj: load_quantized_linear(map, &format!("{base}.v_proj"), 64, 4)?,
            o_proj: load_quantized_linear(map, &format!("{base}.o_proj"), 64, 4)?,
            relative_k_proj: load_quantized_linear(map, &format!("{base}.relative_k_proj"), 64, 4)?,
            bias_u: take_required(map, &format!("{base}.bias_u"))?,
            bias_v: take_required(map, &format!("{base}.bias_v"))?,
            heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            scale: (config.head_dim() as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        input: &MlxArray,
        position_embeddings: &MlxArray,
        valid_length: usize,
    ) -> MlxArray {
        let shape = input.shape();
        let batch = shape[0];
        let seq = shape[1];
        let project = |weight: &QuantizedWeight| {
            transpose(
                &reshape(
                    &qw(input, weight),
                    &[batch, seq, self.heads as i32, self.head_dim as i32],
                    None,
                ),
                &[0, 2, 1, 3],
                None,
            )
        };
        let query = project(&self.q_proj);
        let key = project(&self.k_proj);
        let value = project(&self.v_proj);
        let bias_shape = [1, self.heads as i32, 1, self.head_dim as i32];
        let query_u = add(&query, &reshape(&self.bias_u, &bias_shape, None), None);
        let query_v = add(&query, &reshape(&self.bias_v, &bias_shape, None), None);
        let relative_key = qw(position_embeddings, &self.relative_k_proj);
        let relative_key = reshape(
            &relative_key,
            &[batch, -1, self.heads as i32, self.head_dim as i32],
            None,
        );
        let relative_key = transpose(&relative_key, &[0, 2, 3, 1], None);
        let relative_scores = matmul(&query_v, &relative_key, None);
        let relative_scores = relative_shift(&relative_scores);
        let relative_scores = slice(
            &relative_scores,
            &[0, 0, 0, 0],
            &[batch, self.heads as i32, seq, seq],
            &[1, 1, 1, 1],
            None,
        );
        let relative_scores = multiply(
            &relative_scores,
            &scalar(self.scale, relative_scores.dtype()),
            None,
        );
        let relative_scores = add(
            &relative_scores,
            &attention_padding_bias(seq as usize, valid_length, relative_scores.dtype()),
            None,
        );
        let attention = scaled_dot_product_attention_with_mask(
            &query_u,
            &key,
            &value,
            self.scale,
            ScaledDotProductAttentionMask::Array(&relative_scores),
            None,
        );
        let attention = multiply(
            &attention,
            &query_mask_4d(seq as usize, valid_length, attention.dtype()),
            None,
        );
        let attention = transpose(&attention, &[0, 2, 1, 3], None);
        qw(
            &reshape(
                &attention,
                &[batch, seq, (self.heads * self.head_dim) as i32],
                None,
            ),
            &self.o_proj,
        )
    }
}

#[derive(Clone)]
struct AudioEncoderBlock {
    feed_forward1: AudioFeedForward,
    attention: AudioAttention,
    conv: AudioConvModule,
    feed_forward2: AudioFeedForward,
    norm_feed_forward1: AudioLayerNorm,
    norm_attention: AudioLayerNorm,
    norm_conv: AudioLayerNorm,
    norm_feed_forward2: AudioLayerNorm,
    norm_out: AudioLayerNorm,
}

impl AudioEncoderBlock {
    fn load(
        map: &mut HashMap<String, MlxArray>,
        base: &str,
        config: &NemotronOmniAudioConfig,
        transpose_convs: bool,
    ) -> Result<Self, WeightLoadError> {
        Ok(Self {
            feed_forward1: AudioFeedForward::load(map, &format!("{base}.feed_forward1"))?,
            attention: AudioAttention::load(map, &format!("{base}.self_attn"), config)?,
            conv: AudioConvModule::load(map, &format!("{base}.conv"), config, transpose_convs)?,
            feed_forward2: AudioFeedForward::load(map, &format!("{base}.feed_forward2"))?,
            norm_feed_forward1: AudioLayerNorm::load(map, &format!("{base}.norm_feed_forward1"))?,
            norm_attention: AudioLayerNorm::load(map, &format!("{base}.norm_self_att"))?,
            norm_conv: AudioLayerNorm::load(map, &format!("{base}.norm_conv"))?,
            norm_feed_forward2: AudioLayerNorm::load(map, &format!("{base}.norm_feed_forward2"))?,
            norm_out: AudioLayerNorm::load(map, &format!("{base}.norm_out"))?,
        })
    }

    fn forward(
        &self,
        input: &MlxArray,
        position_embeddings: &MlxArray,
        valid_length: usize,
    ) -> MlxArray {
        let first_ffn = self
            .feed_forward1
            .forward(&self.norm_feed_forward1.forward(input));
        let mut hidden = add(
            input,
            &multiply(&first_ffn, &scalar(0.5, first_ffn.dtype()), None),
            None,
        );
        let attention = self.attention.forward(
            &self.norm_attention.forward(&hidden),
            position_embeddings,
            valid_length,
        );
        hidden = add(&hidden, &attention, None);
        let conv = self
            .conv
            .forward(&self.norm_conv.forward(&hidden), valid_length);
        hidden = add(&hidden, &conv, None);
        let second_ffn = self
            .feed_forward2
            .forward(&self.norm_feed_forward2.forward(&hidden));
        hidden = add(
            &hidden,
            &multiply(&second_ffn, &scalar(0.5, second_ffn.dtype()), None),
            None,
        );
        self.norm_out.forward(&hidden)
    }
}

#[derive(Clone)]
struct AudioEncoder {
    config: NemotronOmniAudioConfig,
    subsampling: AudioSubsampling,
    layers: Vec<AudioEncoderBlock>,
}

impl AudioEncoder {
    fn load(
        map: &mut HashMap<String, MlxArray>,
        config: &NemotronOmniAudioConfig,
        transpose_convs: bool,
    ) -> Result<Self, WeightLoadError> {
        let subsampling = AudioSubsampling::load(map, config, transpose_convs)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(AudioEncoderBlock::load(
                map,
                &format!("sound_encoder.encoder.layers.{index}"),
                config,
                transpose_convs,
            )?);
        }
        Ok(Self {
            config: config.clone(),
            subsampling,
            layers,
        })
    }

    fn forward(&self, input: &MlxArray, valid_frames: usize) -> MlxArray {
        let (mut hidden, valid_length) = self.subsampling.forward(input, valid_frames);
        let seq = hidden.shape()[1] as usize;
        let positions = relative_position_embeddings(seq, self.config.hidden_size, hidden.dtype());
        let positions = broadcast_to(
            &reshape(
                &positions,
                &[1, (2 * seq - 1) as i32, self.config.hidden_size as i32],
                None,
            ),
            &[
                hidden.shape()[0],
                (2 * seq - 1) as i32,
                self.config.hidden_size as i32,
            ],
            None,
        );
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &positions, valid_length);
        }
        hidden
    }
}

#[derive(Clone)]
struct AudioProjection {
    norm: MlxArray,
    linear1: QuantizedWeight,
    linear2: QuantizedWeight,
}

impl AudioProjection {
    fn load(map: &mut HashMap<String, MlxArray>) -> Result<Self, WeightLoadError> {
        Ok(Self {
            norm: take_required(map, "sound_projection.norm.weight")?,
            linear1: load_quantized_linear(map, "sound_projection.linear1", 64, 4)?,
            linear2: load_quantized_linear(map, "sound_projection.linear2", 64, 4)?,
        })
    }

    fn forward(&self, input: &MlxArray) -> MlxArray {
        let normalized = rms_norm(input, Some(&self.norm), 1.0e-5, None);
        let hidden = qw(&normalized, &self.linear1);
        let relu = relu(&hidden);
        let activated = multiply(&relu, &relu, None);
        qw(&activated, &self.linear2)
    }
}

#[derive(Clone)]
pub(crate) struct NemotronOmniAudioWeights {
    extractor: NemotronOmniFeatureExtractor,
    encoder: AudioEncoder,
    projection: AudioProjection,
}

impl NemotronOmniAudioWeights {
    pub(crate) fn load(
        map: &mut HashMap<String, MlxArray>,
        root_config: &Value,
    ) -> Result<Self, WeightLoadError> {
        let config = NemotronOmniAudioConfig::from_json(root_config)?;
        let transpose_convs = audio_convs_need_transpose(map);
        let extractor = NemotronOmniFeatureExtractor::new(config.clone());
        let encoder = AudioEncoder::load(map, &config, transpose_convs)?;
        let projection = AudioProjection::load(map)?;
        // Training-only frontend and BatchNorm counters are intentionally not
        // part of the inference graph.
        map.retain(|name, _| {
            !name.starts_with("sound_encoder.encoder.feature_extractor.")
                && !name.ends_with(".num_batches_tracked")
        });
        Ok(Self {
            extractor,
            encoder,
            projection,
        })
    }

    pub(crate) fn soft_token_count(&self, sample_count: usize) -> usize {
        let frames = sample_count / self.extractor.config.hop_length + 1;
        self.extractor.config.subsampling_output_length(frames)
    }

    pub(crate) fn forward(&self, samples: &[f32], sample_rate: u32) -> Result<MlxArray, String> {
        if sample_rate != self.sampling_rate() {
            return Err(format!(
                "audio sample_rate {sample_rate} != checkpoint rate {}",
                self.sampling_rate()
            ));
        }
        let expected_tokens = self.soft_token_count(samples.len());
        let extracted = self.extractor.extract_clip(samples);
        let features = MlxArray::from_raw_data(
            extracted.features.as_ptr().cast(),
            std::mem::size_of_val(extracted.features.as_slice()),
            &[
                1,
                extracted.frames as i32,
                self.extractor.config.num_mel_bins as i32,
            ],
            MlxDtype::Float32,
        );
        let features = astype(&features, self.projection.norm.dtype(), None);
        let encoded = self.encoder.forward(&features, extracted.valid_frames);
        let projected = self.projection.forward(&encoded);
        if projected.shape()[1] as usize != expected_tokens {
            return Err(format!(
                "Parakeet produced {} tokens, expected {expected_tokens}",
                projected.shape()[1]
            ));
        }
        Ok(projected)
    }

    pub(crate) fn sampling_rate(&self) -> u32 {
        self.extractor.config.sampling_rate
    }
}

fn audio_convs_need_transpose(map: &HashMap<String, MlxArray>) -> bool {
    map.iter()
        .find(|(name, weight)| {
            name.starts_with("sound_encoder.encoder.")
                && name.contains("depthwise")
                && name.ends_with(".weight")
                && weight.shape().len() == 3
        })
        .is_some_and(|(_, weight)| weight.shape()[1] == 1)
}

fn take_audio_conv(
    map: &mut HashMap<String, MlxArray>,
    name: &str,
    transpose_convs: bool,
) -> Result<MlxArray, WeightLoadError> {
    let weight = take_required(map, name)?;
    if !transpose_convs {
        return Ok(weight);
    }
    match weight.shape().len() {
        3 => Ok(transpose(&weight, &[0, 2, 1], None)),
        4 => Ok(transpose(&weight, &[0, 2, 3, 1], None)),
        rank => Err(WeightLoadError::InvalidLayer(format!(
            "Nemotron Omni audio conv {name} has rank {rank}"
        ))),
    }
}

fn conv_output_length(length: usize, kernel: usize, stride: usize) -> usize {
    let padding = (kernel - 1) / 2;
    let numerator = length as i64 + (2 * padding) as i64 - kernel as i64;
    if numerator < 0 {
        0
    } else {
        numerator as usize / stride + 1
    }
}

fn relative_position_embeddings(seq: usize, hidden: usize, dtype: MlxDtype) -> MlxArray {
    let positions = (0..2 * seq - 1)
        .map(|index| (seq as i64 - 1 - index as i64) as f32)
        .collect::<Vec<_>>();
    let inverse_frequency = (0..hidden / 2)
        .map(|index| 1.0 / 10_000f32.powf((2 * index) as f32 / hidden as f32))
        .collect::<Vec<_>>();
    let positions = f32_array(&positions, &[(2 * seq - 1) as i32, 1]);
    let inverse_frequency = f32_array(&inverse_frequency, &[1, (hidden / 2) as i32]);
    let frequencies = multiply(&positions, &inverse_frequency, None);
    let sine = sin(&frequencies, None);
    let cosine = cos(&frequencies, None);
    astype(
        &reshape(
            &stack(&[&sine, &cosine], -1, None),
            &[(2 * seq - 1) as i32, hidden as i32],
            None,
        ),
        dtype,
        None,
    )
}

fn relative_shift(scores: &MlxArray) -> MlxArray {
    let shape = scores.shape();
    let zero = scalar(0.0, scores.dtype());
    let padded = pad(scores, &[3], &[1], &[0], &zero, None);
    let reshaped = reshape(&padded, &[shape[0], shape[1], shape[3] + 1, shape[2]], None);
    let cropped = slice(
        &reshaped,
        &[0, 0, 1, 0],
        &[shape[0], shape[1], shape[3] + 1, shape[2]],
        &[1, 1, 1, 1],
        None,
    );
    reshape(&cropped, &[shape[0], shape[1], shape[2], shape[3]], None)
}

fn time_mask_4d(seq: usize, valid: usize, dtype: MlxDtype) -> MlxArray {
    let values = (0..seq)
        .map(|index| f32::from(index < valid))
        .collect::<Vec<_>>();
    astype(&f32_array(&values, &[1, seq as i32, 1, 1]), dtype, None)
}

fn time_mask_3d(seq: usize, valid: usize, dtype: MlxDtype) -> MlxArray {
    let values = (0..seq)
        .map(|index| f32::from(index < valid))
        .collect::<Vec<_>>();
    astype(&f32_array(&values, &[1, seq as i32, 1]), dtype, None)
}

fn query_mask_4d(seq: usize, valid: usize, dtype: MlxDtype) -> MlxArray {
    let values = (0..seq)
        .map(|index| f32::from(index < valid))
        .collect::<Vec<_>>();
    astype(&f32_array(&values, &[1, 1, seq as i32, 1]), dtype, None)
}

fn attention_padding_bias(seq: usize, valid: usize, dtype: MlxDtype) -> MlxArray {
    let invalid = match dtype {
        MlxDtype::Float16 => -65_504.0,
        MlxDtype::Bfloat16 => -3.38e38,
        _ => f32::MIN,
    };
    let mut values = Vec::with_capacity(seq * seq);
    for query in 0..seq {
        for key in 0..seq {
            values.push(if query < valid && key < valid {
                0.0
            } else {
                invalid
            });
        }
    }
    astype(
        &f32_array(&values, &[1, 1, seq as i32, seq as i32]),
        dtype,
        None,
    )
}

fn add_optional_bias(output: MlxArray, bias: Option<&MlxArray>) -> MlxArray {
    bias.map_or(output.clone(), |bias| add(&output, bias, None))
}

fn relu(input: &MlxArray) -> MlxArray {
    mlx_sys::maximum(input, &scalar(0.0, input.dtype()), None)
}

fn silu(input: &MlxArray, stream: Option<&mlx_sys::MlxStream>) -> MlxArray {
    multiply(input, &sigmoid(input, stream), stream)
}

fn scalar(value: f32, dtype: MlxDtype) -> MlxArray {
    let value = MlxArray::from_raw_data(
        (&value as *const f32).cast(),
        std::mem::size_of::<f32>(),
        &[],
        MlxDtype::Float32,
    );
    if dtype == MlxDtype::Float32 {
        value
    } else {
        astype(&value, dtype, None)
    }
}

fn f32_array(values: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Float32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_and_subsampling_lengths_match_checkpoint_contract() {
        let config = NemotronOmniAudioConfig {
            hidden_size: 1024,
            num_attention_heads: 8,
            num_hidden_layers: 24,
            conv_kernel_size: 9,
            convolution_bias: false,
            subsampling_factor: 8,
            subsampling_conv_channels: 256,
            num_mel_bins: 128,
            subsampling_conv_kernel_size: 3,
            subsampling_conv_stride: 2,
            sampling_rate: 16_000,
            hop_length: 160,
            n_fft: 512,
            win_length: 400,
            preemphasis: 0.97,
        };
        let extractor = NemotronOmniFeatureExtractor::new(config.clone());
        let audio = vec![0.0f32; 16_000];
        let extracted = extractor.extract_clip(&audio);
        assert_eq!(extracted.frames, 101);
        assert_eq!(extracted.valid_frames, 100);
        assert_eq!(config.subsampling_output_length(101), 13);
    }

    #[test]
    fn target_length_uses_three_stride_two_stages() {
        assert_eq!(conv_output_length(100, 3, 2), 50);
        assert_eq!(conv_output_length(50, 3, 2), 25);
        assert_eq!(conv_output_length(25, 3, 2), 13);
    }

    #[test]
    fn relative_position_vector_is_descending() {
        let array = relative_position_embeddings(3, 4, MlxDtype::Float32);
        assert_eq!(array.shape(), vec![5, 4]);
    }
}
