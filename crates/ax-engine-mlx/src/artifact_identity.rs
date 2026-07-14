//! Content-derived model artifact identity.
//!
//! A filesystem path (or its mtime) does not identify a checkpoint: an
//! in-place replacement keeps the path while changing every weight, and a
//! stale prefix-cache entry restored under the old path would silently
//! produce wrong-model KV. This module computes a fingerprint from the
//! artifact **content**:
//!
//! 1. the exact native model manifest bytes;
//! 2. every referenced tensor filename, in sorted order;
//! 3. each tensor's content-addressed store hash (the SHA-256 the tensor
//!    symlink resolves to) and byte length.
//!
//! The fingerprint is domain-separated per consumer so evidence formats
//! stay independently versioned: batched-decode certification keeps its
//! historical `ax.mlx.batched_decode.artifact.v1` domain, while the
//! durable prefix cache uses `ax.mlx.prefix_cache.artifact.v1`
//! (TECH-SPEC-DURABLE-TIERED-PREFIX-CACHE §6).
//!
//! Fail-closed: when any tensor is not a symlink into the content-addressed
//! store (or the manifest is unreadable), no fingerprint is produced and the
//! consumer must treat the artifact identity as unavailable — for the prefix
//! cache that makes L2 ineligible rather than falling back to a path-only
//! identity.

use std::collections::BTreeSet;
use std::fs;

use ax_engine_core::{AX_NATIVE_MODEL_MANIFEST_FILE, NativeModelArtifacts, NativeTensorSpec};
use sha2::{Digest, Sha256};

/// Domain string for the durable prefix cache's artifact identity
/// (canonical key schema v3).
pub(crate) const PREFIX_CACHE_ARTIFACT_DOMAIN: &[u8] = b"ax.mlx.prefix_cache.artifact.v1\0";

/// Compute the content fingerprint under a consumer-specific domain.
/// Returns `None` when the artifact source cannot supply a stable content
/// identity (unreadable manifest, tensor not content-addressed).
pub(crate) fn artifact_fingerprint_sha256_with_domain(
    artifacts: &NativeModelArtifacts,
    domain: &[u8],
) -> Option<String> {
    let manifest_path = artifacts.root_dir().join(AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest = fs::read(manifest_path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&manifest);

    let files = artifacts
        .tensor_specs()
        .iter()
        .map(tensor_file)
        .collect::<BTreeSet<_>>();
    for file in files {
        let path = artifacts.root_dir().join(file);
        let target = fs::read_link(&path).ok()?;
        let content_hash = target
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| is_sha256_hex(name))?
            .to_string();
        let byte_len = fs::metadata(&path).ok()?.len();
        hasher.update(file.as_bytes());
        hasher.update(b"\0");
        hasher.update(content_hash.as_bytes());
        hasher.update(b"\0");
        hasher.update(byte_len.to_le_bytes());
    }
    Some(hex_digest(&hasher.finalize()))
}

fn tensor_file(tensor: &NativeTensorSpec) -> &str {
    tensor.file.to_str().unwrap_or("")
}

pub(crate) fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
