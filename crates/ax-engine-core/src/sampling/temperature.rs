//! Temperature scaling for logits.
//!
//! Divides logits by temperature before softmax. Higher temperature
//! makes the distribution more uniform (more random), lower temperature
//! makes it sharper (more deterministic). Temperature = 0 is greedy.

/// Apply temperature scaling to logits (in-place).
///
/// `logits[i] /= temperature`
///
/// Temperature must be > 0. For greedy decoding (temperature = 0),
/// skip temperature and use argmax directly.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    assert!(
        temperature > 0.0,
        "temperature must be > 0, got {temperature}"
    );
    let inv_temp = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_one() {
        // Temperature 1.0 should not change logits
        let mut logits = [1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_temperature_high() {
        // High temperature compresses logits toward zero
        let mut logits = [2.0, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        assert_eq!(logits, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_temperature_low() {
        // Low temperature amplifies differences
        let mut logits = [1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 0.5);
        assert_eq!(logits, [2.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "temperature must be > 0")]
    fn test_temperature_zero_panics() {
        let mut logits = [1.0];
        apply_temperature(&mut logits, 0.0);
    }
}
