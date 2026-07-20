use std::time::Instant;

#[cfg(test)]
pub(crate) fn percentile_f64(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let quantile = quantile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values.get(index).copied()
}

#[cfg(test)]
pub(crate) fn percentile_u64(values: &[u64], quantile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let quantile = quantile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values.get(index).copied()
}

pub(crate) fn percentage_delta(baseline: f64, candidate: f64) -> f64 {
    if baseline.abs() < f64::EPSILON {
        if candidate.abs() < f64::EPSILON {
            0.0
        } else {
            f64::INFINITY.copysign(candidate)
        }
    } else {
        ((candidate - baseline) / baseline) * 100.0
    }
}

pub(crate) fn proportional_time_us(total_us: u64, part_tokens: u64, total_tokens: u64) -> u64 {
    if total_us == 0 || part_tokens == 0 || total_tokens == 0 {
        return 0;
    }
    let value =
        u128::from(total_us).saturating_mul(u128::from(part_tokens)) / u128::from(total_tokens);
    value.min(u128::from(u64::MAX)) as u64
}

pub(crate) fn tokens_per_second_from_micros(tokens: u64, elapsed_us: u64) -> f64 {
    if tokens == 0 || elapsed_us == 0 {
        0.0
    } else {
        (tokens as f64 * 1_000_000.0) / elapsed_us as f64
    }
}

pub(crate) fn elapsed_ms_since(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64 + 1
}
