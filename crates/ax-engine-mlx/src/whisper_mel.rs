//! Whisper-compatible 16 kHz log-mel frontend.
//!
//! The geometry and Slaney filterbank match OpenAI Whisper and mlx-whisper:
//! centered 400-point periodic-Hann STFT, 160-sample hop, trailing frame
//! dropped, power projection, global 8-dB log clamp, then `(x + 4) / 4`.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::Fft;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

pub(crate) const SAMPLE_RATE: usize = 16_000;
pub(crate) const N_FFT: usize = 400;
pub(crate) const HOP_LENGTH: usize = 160;
pub(crate) const CHUNK_LENGTH_SECONDS: usize = 30;
pub(crate) const N_SAMPLES: usize = CHUNK_LENGTH_SECONDS * SAMPLE_RATE;
pub(crate) const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;

pub(crate) struct WhisperMel {
    n_mels: usize,
    window: Vec<f32>,
    filters: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl WhisperMel {
    pub(crate) fn new(n_mels: usize) -> Result<Self, String> {
        if !matches!(n_mels, 80 | 128) {
            return Err(format!(
                "Whisper supports 80- or 128-bin mel inputs, got {n_mels}"
            ));
        }
        let mut planner = FftPlanner::<f32>::new();
        Ok(Self {
            n_mels,
            window: periodic_hann(N_FFT),
            filters: slaney_mel_filterbank(n_mels),
            fft: planner.plan_fft_forward(N_FFT),
        })
    }

    /// Return row-major `[frames, n_mels]` normalized log-mel features.
    pub(crate) fn extract(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let padded = reflect_pad(audio, N_FFT / 2);
        if padded.len() < N_FFT {
            return (Vec::new(), 0);
        }
        let total_frames = 1 + (padded.len() - N_FFT) / HOP_LENGTH;
        // mlx-whisper computes `freqs[:-1]`: the centered trailing analysis
        // frame is deliberately omitted.
        let frames = total_frames.saturating_sub(1);
        if frames == 0 {
            return (Vec::new(), 0);
        }

        let bins = N_FFT / 2 + 1;
        let mut fft_buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = vec![0.0f32; bins];
        let mut features = vec![0.0f32; frames * self.n_mels];
        for frame in 0..frames {
            let start = frame * HOP_LENGTH;
            for (index, value) in fft_buffer.iter_mut().enumerate() {
                *value = Complex32::new(padded[start + index] * self.window[index], 0.0);
            }
            self.fft.process(&mut fft_buffer);
            for (output, value) in power.iter_mut().zip(&fft_buffer[..bins]) {
                *output = value.norm_sqr();
            }
            for mel in 0..self.n_mels {
                let filter = &self.filters[mel * bins..(mel + 1) * bins];
                let energy = filter
                    .iter()
                    .zip(&power)
                    .map(|(weight, power)| weight * power)
                    .sum::<f32>();
                features[frame * self.n_mels + mel] = energy.max(1.0e-10).log10();
            }
        }

        let global_max = features.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = global_max - 8.0;
        for feature in &mut features {
            *feature = (feature.max(floor) + 4.0) / 4.0;
        }
        (features, frames)
    }
}

fn periodic_hann(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| (0.5 - 0.5 * (2.0 * PI * index as f64 / size.max(1) as f64).cos()) as f32)
        .collect()
}

fn reflect_pad(audio: &[f32], padding: usize) -> Vec<f32> {
    if audio.is_empty() {
        return vec![0.0; 2 * padding];
    }
    let mut output = Vec::with_capacity(audio.len() + 2 * padding);
    for index in (1..=padding).rev() {
        output.push(audio[index.min(audio.len() - 1)]);
    }
    output.extend_from_slice(audio);
    for index in 1..=padding {
        output.push(audio[(audio.len() - 1).saturating_sub(index)]);
    }
    output
}

fn slaney_mel_filterbank(n_mels: usize) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let mel_min = hz_to_slaney_mel(0.0);
    let mel_max = hz_to_slaney_mel(SAMPLE_RATE as f64 / 2.0);
    let edges = (0..n_mels + 2)
        .map(|index| {
            let fraction = index as f64 / (n_mels + 1) as f64;
            slaney_mel_to_hz(mel_min + fraction * (mel_max - mel_min))
        })
        .collect::<Vec<_>>();
    let mut filters = vec![0.0f32; n_mels * bins];
    for mel in 0..n_mels {
        let lower = edges[mel];
        let center = edges[mel + 1];
        let upper = edges[mel + 2];
        let scale = 2.0 / (upper - lower).max(f64::EPSILON);
        for bin in 0..bins {
            let frequency = bin as f64 * SAMPLE_RATE as f64 / N_FFT as f64;
            let rising = (frequency - lower) / (center - lower).max(f64::EPSILON);
            let falling = (upper - frequency) / (upper - center).max(f64::EPSILON);
            filters[mel * bins + bin] = (rising.min(falling).max(0.0) * scale) as f32;
        }
    }
    filters
}

fn hz_to_slaney_mel(frequency: f64) -> f64 {
    let frequency_step = 200.0 / 3.0;
    let log_start_hz = 1_000.0;
    let log_start_mel = log_start_hz / frequency_step;
    let log_step = 6.4f64.ln() / 27.0;
    if frequency >= log_start_hz {
        log_start_mel + (frequency / log_start_hz).ln() / log_step
    } else {
        frequency / frequency_step
    }
}

fn slaney_mel_to_hz(mel: f64) -> f64 {
    let frequency_step = 200.0 / 3.0;
    let log_start_hz = 1_000.0;
    let log_start_mel = log_start_hz / frequency_step;
    let log_step = 6.4f64.ln() / 27.0;
    if mel >= log_start_mel {
        log_start_hz * (log_step * (mel - log_start_mel)).exp()
    } else {
        frequency_step * mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_window_produces_reference_frame_count() {
        let mel = WhisperMel::new(128).expect("mel frontend should build");
        let (features, frames) = mel.extract(&vec![0.0; N_SAMPLES]);
        assert_eq!(frames, N_FRAMES);
        assert_eq!(features.len(), N_FRAMES * 128);
        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn periodic_hann_has_reference_endpoints() {
        let window = periodic_hann(N_FFT);
        assert!(window[0].abs() < 1.0e-7);
        assert!(window[N_FFT / 2] > 0.999);
        assert!(window[N_FFT - 1] > 0.0);
    }

    #[test]
    fn filterbank_has_a_nonempty_filter_per_mel_bin() {
        let filters = slaney_mel_filterbank(128);
        let bins = N_FFT / 2 + 1;
        assert_eq!(filters.len(), 128 * bins);
        for filter in filters.chunks_exact(bins) {
            assert!(filter.iter().any(|weight| *weight > 0.0));
        }
    }
}
