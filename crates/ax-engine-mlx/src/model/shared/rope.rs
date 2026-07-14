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

/// YaRN RoPE frequency divisors + attention mscale (mlx-lm `YarnRoPE`).
///
/// Returns `(freqs[dims/2], mscale)` where `freqs` are wavelength **divisors**
/// for `mlx_sys::rope` (same convention as [`build_llama3_rope_freqs`]), and
/// `mscale` multiplies Q/K head dims before RoPE when ≠ 1.
///
/// Defaults match openai/gpt-oss (`beta_fast=32`, `beta_slow=1`, `mscale=1`,
/// `mscale_all_dim=0`).
pub(crate) fn build_yarn_rope_freqs(
    dims: usize,
    theta: f32,
    factor: f32,
    original_context_len: u32,
    beta_fast: f32,
    beta_slow: f32,
    mscale: f32,
    mscale_all_dim: f32,
) -> (MlxArray, f32) {
    let orig = original_context_len as f32;
    let dims_f = dims as f32;

    let yarn_find_correction_dim = |num_rotations: f32| -> f32 {
        (dims_f * (orig / (num_rotations * 2.0 * PI)).ln()) / (2.0 * theta.ln())
    };
    let low = yarn_find_correction_dim(beta_fast).floor().max(0.0);
    let high = yarn_find_correction_dim(beta_slow)
        .ceil()
        .min((dims.saturating_sub(1)) as f32);

    let n_freqs = dims / 2;
    let mut freqs = Vec::with_capacity(n_freqs);
    for i in 0..n_freqs {
        let freq_extra = theta.powf(2.0 * i as f32 / dims_f);
        let freq_inter = factor * freq_extra;
        // yarn_linear_ramp_mask over dims/2 entries, then freq_mask = 1 - ramp
        let ramp = if (high - low).abs() < 1e-6 {
            let max_val = low + 0.001;
            ((i as f32 - low) / (max_val - low)).clamp(0.0, 1.0)
        } else {
            ((i as f32 - low) / (high - low)).clamp(0.0, 1.0)
        };
        let freq_mask = 1.0 - ramp;
        // mlx-lm: (freq_inter * freq_extra) / (freq_inter * mask + freq_extra * (1-mask))
        let denom = freq_inter * freq_mask + freq_extra * (1.0 - freq_mask);
        freqs.push((freq_inter * freq_extra) / denom);
    }

    let yarn_get_mscale = |scale: f32, ms: f32| -> f32 {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * ms * scale.ln() + 1.0
        }
    };
    let attention_mscale =
        yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim);

    (MlxArray::from_f32_slice(&freqs), attention_mscale)
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

    #[test]
    fn yarn_gpt_oss_defaults_produce_mscale_and_freqs() {
        // openai/gpt-oss-20b: head_dim=64, rope_theta=150000, factor=32, orig=4096.
        let (freqs, mscale) =
            build_yarn_rope_freqs(64, 150_000.0, 32.0, 4096, 32.0, 1.0, 1.0, 0.0);
        assert_eq!(freqs.shape(), vec![32]);
        // mscale = (0.1 * ln(32) + 1) / 1 ≈ 1.34657
        let expected = 0.1 * 32.0f32.ln() + 1.0;
        assert!(
            (mscale - expected).abs() < 1e-4,
            "got mscale {mscale}, expected ~{expected}"
        );
        let data = freqs.data_f32();
        assert!(data[0] > 0.0);
        // Low-frequency (high-i) divisors grow with factor vs base RoPE.
        let base_last = 150_000.0f32.powf(2.0 * 31.0 / 64.0);
        assert!(
            data[31] > base_last,
            "yarn low-freq divisor should exceed base, got {} vs {}",
            data[31],
            base_last
        );
    }
}
