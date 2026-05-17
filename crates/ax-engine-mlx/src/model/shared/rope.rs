use std::f32::consts::PI;

use mlx_sys::MlxArray;

/// Precompute LLaMA-3 corrected RoPE frequencies.
///
/// LLaMA-3 applies a smooth correction to the standard RoPE frequencies that
/// gradually scales down low-frequency components (long wavelengths) while
/// leaving high-frequency components unchanged. The result is a `[dims/2]`
/// f32 array suitable for passing as `freqs` to `mlx_sys::rope`.
///
/// Reference: `transformers/models/llama/modeling_llama.py` and mlx-lm `llama.py`.
pub(crate) fn build_llama3_rope_freqs(
    dims: usize,
    theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_context_len: u32,
) -> MlxArray {
    let orig = original_context_len as f32;
    let low_wavelen = orig / low_freq_factor;
    let high_wavelen = orig / high_freq_factor;

    let freqs: Vec<f32> = (0..dims / 2)
        .map(|i| {
            let base_freq = 1.0 / theta.powf(2.0 * i as f32 / dims as f32);
            let wavelen = 2.0 * PI / base_freq;
            if wavelen < high_wavelen {
                base_freq
            } else if wavelen > low_wavelen {
                base_freq / factor
            } else {
                let smooth =
                    (orig / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                (1.0 - smooth) * base_freq / factor + smooth * base_freq
            }
        })
        .collect();

    MlxArray::from_f32_slice(&freqs)
}
