use std::f32::consts::PI;

use mlx_sys::MlxArray;

/// Per-dimension LLaMA-3 RoPE wavelength divisor.
///
/// `mlx_fast_rope` computes `theta = position / freqs` (it reciprocates the
/// `freqs` array internally), so each entry must be the wavelength **divisor**
/// `base^(2i/dims)`, exactly like mlx-lm's `Llama3RoPE` and this crate's
/// `build_gemma4_proportional_rope_freqs` — NOT its reciprocal. The smooth
/// LLaMA-3 correction scales down the low-frequency (long-wavelength) divisors
/// while leaving the high-frequency ones unchanged.
fn llama3_rope_freq(
    i: usize,
    dims: usize,
    theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    orig: f32,
) -> f32 {
    let low_wavelen = orig / low_freq_factor;
    let high_wavelen = orig / high_freq_factor;
    let freq = theta.powf(2.0 * i as f32 / dims as f32);
    let wavelen = 2.0 * PI * freq;
    if wavelen < high_wavelen {
        // High frequency / short wavelength: leave the divisor unchanged.
        freq
    } else if wavelen > low_wavelen {
        // Low frequency / long wavelength: divide positions further by `factor`.
        freq * factor
    } else {
        // Medium band: smoothly interpolate between the two regimes.
        let smooth = (orig / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        freq / ((1.0 - smooth) / factor + smooth)
    }
}

/// Precompute LLaMA-3 corrected RoPE frequencies.
///
/// LLaMA-3 applies a smooth correction to the standard RoPE frequencies that
/// gradually scales down low-frequency components (long wavelengths) while
/// leaving high-frequency components unchanged. The result is a `[dims/2]`
/// f32 array suitable for passing as `freqs` to `mlx_sys::rope`.
///
/// Reference: `transformers/models/llama/modeling_llama.py` and mlx-lm
/// `models/rope_utils.py::Llama3RoPE` (which passes the same divisor array to
/// `mx.fast.rope`).
pub(crate) fn build_llama3_rope_freqs(
    dims: usize,
    theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_context_len: u32,
) -> MlxArray {
    let orig = original_context_len as f32;

    let freqs: Vec<f32> = (0..dims / 2)
        .map(|i| {
            llama3_rope_freq(
                i,
                dims,
                theta,
                factor,
                low_freq_factor,
                high_freq_factor,
                orig,
            )
        })
        .collect();

    MlxArray::from_f32_slice(&freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // mlx-lm Llama3 8B scaling config.
    const THETA: f32 = 500_000.0;
    const DIMS: usize = 128;
    const FACTOR: f32 = 8.0;
    const LOW_FF: f32 = 1.0;
    const HIGH_FF: f32 = 4.0;
    const ORIG: f32 = 8192.0;

    #[test]
    fn divisor_convention_for_unchanged_high_frequency_dim() {
        // `mlx_fast_rope` divides by `freqs`, so an unchanged high-frequency
        // dimension must equal base^(2i/dims) (> 1 for i>0), not its reciprocal
        // (which the previous implementation returned).
        let i = 10; // short wavelength -> "unchanged" band
        let f = llama3_rope_freq(i, DIMS, THETA, FACTOR, LOW_FF, HIGH_FF, ORIG);
        let expected = THETA.powf(2.0 * i as f32 / DIMS as f32);
        assert!(
            (f - expected).abs() <= 1e-3 * expected,
            "got {f}, expected divisor {expected}"
        );
        assert!(f > 1.0, "divisor for i>0 must exceed 1, got {f}");
    }

    #[test]
    fn dim_zero_is_unit_divisor() {
        // base^0 == 1 regardless of convention; the short wavelength keeps it
        // in the unchanged band.
        let f = llama3_rope_freq(0, DIMS, THETA, FACTOR, LOW_FF, HIGH_FF, ORIG);
        assert!((f - 1.0).abs() <= 1e-6, "got {f}");
    }

    #[test]
    fn low_frequency_dim_scaled_by_factor() {
        // A long-wavelength dimension has its divisor multiplied by `factor`
        // (mlx-lm: `freqs * factor`), shrinking the rotation rate.
        let i = 60;
        let freq = THETA.powf(2.0 * i as f32 / DIMS as f32);
        let wavelen = 2.0 * PI * freq;
        assert!(wavelen > ORIG / LOW_FF, "expected low-frequency band");
        let f = llama3_rope_freq(i, DIMS, THETA, FACTOR, LOW_FF, HIGH_FF, ORIG);
        let want = freq * FACTOR;
        assert!((f - want).abs() <= 1e-3 * want, "got {f}, expected {want}");
    }

    #[test]
    fn medium_band_interpolates_between_regimes() {
        // Pick a dim in the medium band and confirm the result lies strictly
        // between the unchanged divisor and the fully-scaled divisor.
        let i = (0..DIMS / 2)
            .find(|&i| {
                let wl = 2.0 * PI * THETA.powf(2.0 * i as f32 / DIMS as f32);
                (ORIG / HIGH_FF..=ORIG / LOW_FF).contains(&wl)
            })
            .expect("a medium-band dim should exist for this config");
        let freq = THETA.powf(2.0 * i as f32 / DIMS as f32);
        let f = llama3_rope_freq(i, DIMS, THETA, FACTOR, LOW_FF, HIGH_FF, ORIG);
        assert!(
            f > freq,
            "medium divisor should exceed the unchanged divisor"
        );
        assert!(
            f < freq * FACTOR,
            "medium divisor should be below the fully-scaled divisor"
        );
    }
}
