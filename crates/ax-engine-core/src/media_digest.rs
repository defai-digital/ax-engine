//! Shared media identity digests for vision-feature and prefix caches (WS-M2/M3).
//!
//! Cache keys must be identical across server, runner, and tests. Digests are
//! domain-separated by soft-token budget and model fingerprint so a budget or
//! weight change cannot serve stale features. Runtime prefix identity additionally
//! binds the family, modality, geometry, and placement of each tensor.

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

/// Ordered multi-media digest for prefix-cache keys, or an empty string when
/// there is no media (the "no media" cache-key sentinel).
pub(crate) fn ordered_media_digests_key_or_empty(digests: &[String]) -> String {
    if digests.is_empty() {
        String::new()
    } else {
        ordered_media_digests_key(digests)
    }
}

/// Digest raw f32 tensor bytes (little-endian). This v1 helper omits geometry
/// and placement; runtime prefix keys must use the schema-specific v2 identity.
pub fn media_digest_f32(values: &[f32], soft_token_budget: u32, model_fingerprint: &str) -> String {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    media_digest(&bytes, soft_token_budget, model_fingerprint)
}

/// Versioned runtime identity. Each schema supplies fields in a fixed order;
/// lengths precede variable fields, and usize is encoded as u64 on every host.
/// Keep the public v1 encoded-media helpers stable for feature-cache callers.
pub(crate) struct RuntimeMediaDigest(Sha256);

impl RuntimeMediaDigest {
    pub(crate) fn new(schema: &str, model_fingerprint: &str) -> Self {
        let mut digest = Self(Sha256::new());
        digest.0.update(b"ax.media.runtime.v2\0");
        for value in [schema, model_fingerprint] {
            digest.usize(value.len());
            digest.0.update(value.as_bytes());
        }
        digest
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.0.update(value.to_le_bytes());
    }

    pub(crate) fn usize(&mut self, value: usize) {
        self.0.update((value as u64).to_le_bytes());
    }

    pub(crate) fn positions(&mut self, values: &[[i32; 2]]) {
        self.usize(values.len());
        for [row, column] in values {
            self.0.update(row.to_le_bytes());
            self.0.update(column.to_le_bytes());
        }
    }

    pub(crate) fn f32s(&mut self, values: &[f32]) {
        self.usize(values.len());
        // Hash LE bits without allocating another copy of a large media tensor.
        let mut bytes = [0u8; 1024];
        for chunk in values.chunks(bytes.len() / 4) {
            for (value, output) in chunk.iter().zip(bytes.as_chunks_mut::<4>().0) {
                output.copy_from_slice(&value.to_le_bytes());
            }
            self.0.update(&bytes[..chunk.len() * 4]);
        }
    }

    pub(crate) fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
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
