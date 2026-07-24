//! Shared media identity digests for vision-feature and prefix caches (WS-M2/M3).
//!
//! Cache keys must be identical across server, runner, and tests. Digests are
//! domain-separated by soft-token budget and model fingerprint so a budget or
//! weight change cannot serve stale features.

use sha2::{Digest, Sha256};

/// Compute a hex SHA-256 digest over encoded media bytes + soft-token budget +
/// model fingerprint. Empty fingerprint is allowed (still domain-separated by budget).
pub fn media_digest(
    encoded_bytes: &[u8],
    soft_token_budget: u32,
    model_fingerprint: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"ax.media.v1\0");
    hasher.update((encoded_bytes.len() as u64).to_le_bytes());
    hasher.update(encoded_bytes);
    hasher.update(soft_token_budget.to_le_bytes());
    hasher.update((model_fingerprint.len() as u64).to_le_bytes());
    hasher.update(model_fingerprint.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Ordered multi-media digest for prefix-cache keys (BLAKE3 not required; SHA-256 hex).
pub fn ordered_media_digests_key(digests: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"ax.media.prefix.v1\0");
    hasher.update((digests.len() as u64).to_le_bytes());
    for d in digests {
        hasher.update((d.len() as u64).to_le_bytes());
        hasher.update(d.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Digest f32 tensor bytes (little-endian) for runtime media identity.
pub fn media_digest_f32(values: &[f32], soft_token_budget: u32, model_fingerprint: &str) -> String {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    media_digest(&bytes, soft_token_budget, model_fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_inputs_same_digest() {
        let a = media_digest(b"png-bytes", 280, "model-fp");
        let b = media_digest(b"png-bytes", 280, "model-fp");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn budget_change_is_miss() {
        let a = media_digest(b"png-bytes", 280, "model-fp");
        let b = media_digest(b"png-bytes", 560, "model-fp");
        assert_ne!(a, b);
    }

    #[test]
    fn model_change_is_miss() {
        let a = media_digest(b"png-bytes", 280, "fp-a");
        let b = media_digest(b"png-bytes", 280, "fp-b");
        assert_ne!(a, b);
    }

    #[test]
    fn ordered_key_order_sensitive() {
        let d1 = media_digest(b"a", 70, "m");
        let d2 = media_digest(b"b", 70, "m");
        assert_ne!(
            ordered_media_digests_key(&[d1.clone(), d2.clone()]),
            ordered_media_digests_key(&[d2, d1])
        );
    }
}
