//! Rank bootstrap artifact integrity verification.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read as _;
use std::path::{Component, Path, PathBuf};

use ax_engine_core::PipelineTopology;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use thiserror::Error;

#[derive(Debug, Deserialize)]
pub struct RankBootstrapPlan {
    pub cluster_id: String,
    pub generation: u64,
    pub manifest_digest: String,
    pub rank: BootstrapRank,
    pub artifacts: Vec<BootstrapArtifact>,
}

#[derive(Debug, Deserialize)]
pub struct BootstrapRank {
    pub rank: u16,
}

#[derive(Debug, Deserialize)]
pub struct BootstrapArtifact {
    pub relative_path: PathBuf,
    pub digest: String,
    pub size_bytes: u64,
}

impl RankBootstrapPlan {
    pub fn load_and_verify(
        path: &Path,
        model_root: &Path,
        topology: &PipelineTopology,
        expected_rank: u16,
    ) -> Result<Self, ArtifactVerificationError> {
        let bytes = std::fs::read(path).map_err(ArtifactVerificationError::ReadPlan)?;
        let plan =
            serde_json::from_slice::<Self>(&bytes).map_err(ArtifactVerificationError::ParsePlan)?;
        plan.verify(model_root, topology, expected_rank)?;
        Ok(plan)
    }

    pub fn verify(
        &self,
        model_root: &Path,
        topology: &PipelineTopology,
        expected_rank: u16,
    ) -> Result<(), ArtifactVerificationError> {
        if self.cluster_id != topology.cluster_id
            || self.generation != topology.generation
            || self.manifest_digest != topology.manifest_digest
            || self.rank.rank != expected_rank
        {
            return Err(ArtifactVerificationError::BootstrapIdentityMismatch);
        }
        if self.artifacts.is_empty() {
            return Err(ArtifactVerificationError::EmptyArtifactPlan);
        }
        let canonical_root = model_root
            .canonicalize()
            .map_err(ArtifactVerificationError::ReadModelRoot)?;
        let mut paths = BTreeSet::new();
        let mut digests = BTreeSet::new();
        for artifact in &self.artifacts {
            validate_relative_path(&artifact.relative_path)?;
            if !paths.insert(artifact.relative_path.clone()) {
                return Err(ArtifactVerificationError::DuplicateArtifactPath(
                    artifact.relative_path.clone(),
                ));
            }
            if !digests.insert(artifact.digest.clone()) {
                return Err(ArtifactVerificationError::DuplicateArtifactDigest(
                    artifact.digest.clone(),
                ));
            }
            let expected_digest = parse_sha256(&artifact.digest)?;
            let path = model_root.join(&artifact.relative_path);
            let canonical_path =
                path.canonicalize()
                    .map_err(|source| ArtifactVerificationError::ReadArtifact {
                        path: artifact.relative_path.clone(),
                        source,
                    })?;
            if !canonical_path.starts_with(&canonical_root) {
                return Err(ArtifactVerificationError::ArtifactEscapesModelRoot(
                    artifact.relative_path.clone(),
                ));
            }
            let metadata = std::fs::metadata(&canonical_path).map_err(|source| {
                ArtifactVerificationError::ReadArtifact {
                    path: artifact.relative_path.clone(),
                    source,
                }
            })?;
            if !metadata.is_file() || metadata.len() != artifact.size_bytes {
                return Err(ArtifactVerificationError::ArtifactSizeMismatch {
                    path: artifact.relative_path.clone(),
                    expected: artifact.size_bytes,
                    actual: metadata.len(),
                });
            }
            let actual_digest = sha256_file(&canonical_path, &artifact.relative_path)?;
            if actual_digest != expected_digest {
                return Err(ArtifactVerificationError::ArtifactDigestMismatch(
                    artifact.relative_path.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Fail closed when the runtime would read a file not covered by the
    /// integrity-bound bootstrap plan.
    pub fn require_artifacts(
        &self,
        required_paths: impl IntoIterator<Item = PathBuf>,
    ) -> Result<(), ArtifactVerificationError> {
        let planned = self
            .artifacts
            .iter()
            .map(|artifact| artifact.relative_path.as_path())
            .collect::<BTreeSet<_>>();
        for required in required_paths {
            validate_relative_path(&required)?;
            if !planned.contains(required.as_path()) {
                return Err(ArtifactVerificationError::MissingRequiredArtifact(required));
            }
        }
        Ok(())
    }
}

fn validate_relative_path(path: &Path) -> Result<(), ArtifactVerificationError> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(ArtifactVerificationError::UnsafeArtifactPath(
            path.to_path_buf(),
        ));
    }
    Ok(())
}

fn parse_sha256(value: &str) -> Result<[u8; 32], ArtifactVerificationError> {
    let hex = value
        .strip_prefix("sha256:")
        .ok_or_else(|| ArtifactVerificationError::InvalidDigest(value.to_string()))?;
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ArtifactVerificationError::InvalidDigest(value.to_string()));
    }
    let mut digest = [0_u8; 32];
    for (index, slot) in digest.iter_mut().enumerate() {
        let offset = index * 2;
        *slot = u8::from_str_radix(&hex[offset..offset + 2], 16)
            .map_err(|_| ArtifactVerificationError::InvalidDigest(value.to_string()))?;
    }
    Ok(digest)
}

fn sha256_file(path: &Path, display_path: &Path) -> Result<[u8; 32], ArtifactVerificationError> {
    let mut file = File::open(path).map_err(|source| ArtifactVerificationError::ReadArtifact {
        path: display_path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read =
            file.read(&mut buffer)
                .map_err(|source| ArtifactVerificationError::ReadArtifact {
                    path: display_path.to_path_buf(),
                    source,
                })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

#[derive(Debug, Error)]
pub enum ArtifactVerificationError {
    #[error("failed to read rank bootstrap plan: {0}")]
    ReadPlan(std::io::Error),
    #[error("failed to parse rank bootstrap plan: {0}")]
    ParsePlan(serde_json::Error),
    #[error("failed to access model artifact root: {0}")]
    ReadModelRoot(std::io::Error),
    #[error("bootstrap plan does not match pipeline cluster, generation, manifest, or rank")]
    BootstrapIdentityMismatch,
    #[error("bootstrap plan contains no artifacts")]
    EmptyArtifactPlan,
    #[error("bootstrap artifact path is duplicated: {0}")]
    DuplicateArtifactPath(PathBuf),
    #[error("bootstrap artifact digest is duplicated: {0}")]
    DuplicateArtifactDigest(String),
    #[error("bootstrap artifact path is unsafe: {0}")]
    UnsafeArtifactPath(PathBuf),
    #[error("bootstrap artifact resolves outside the model root: {0}")]
    ArtifactEscapesModelRoot(PathBuf),
    #[error("runtime-required artifact is absent from the bootstrap plan: {0}")]
    MissingRequiredArtifact(PathBuf),
    #[error("bootstrap artifact digest must be sha256:<64 hex characters>: {0}")]
    InvalidDigest(String),
    #[error("failed to read bootstrap artifact {path}: {source}")]
    ReadArtifact {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("bootstrap artifact {path} size mismatch: expected {expected}, got {actual}")]
    ArtifactSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("bootstrap artifact digest mismatch: {0}")]
    ArtifactDigestMismatch(PathBuf),
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use ax_engine_core::{PipelineLayerRange, PipelineRankAssignment};

    use super::*;

    fn topology() -> PipelineTopology {
        PipelineTopology {
            cluster_id: "cluster".into(),
            generation: 1,
            manifest_digest: "manifest".into(),
            model_artifact_digest: "model".into(),
            total_layers: 2,
            ranks: vec![
                PipelineRankAssignment {
                    rank: 0,
                    node_identity_digest: "node-a".into(),
                    layers: PipelineLayerRange { start: 0, end: 1 },
                    owns_embeddings: true,
                    owns_output_head: false,
                },
                PipelineRankAssignment {
                    rank: 1,
                    node_identity_digest: "node-b".into(),
                    layers: PipelineLayerRange { start: 1, end: 2 },
                    owns_embeddings: false,
                    owns_output_head: true,
                },
            ],
        }
    }

    #[test]
    fn digest_parser_requires_explicit_sha256_scheme() {
        assert_eq!(
            parse_sha256(&format!("sha256:{}", "ab".repeat(32))).expect("valid digest"),
            [0xab; 32]
        );
        assert!(parse_sha256(&"ab".repeat(32)).is_err());
        assert!(parse_sha256("sha256:xyz").is_err());
    }

    #[test]
    fn artifact_paths_reject_parent_and_absolute_components() {
        assert!(validate_relative_path(Path::new("weights/rank-0.safetensors")).is_ok());
        assert!(validate_relative_path(Path::new("../rank-1.safetensors")).is_err());
        assert!(validate_relative_path(Path::new("/tmp/model")).is_err());
    }

    #[test]
    fn required_runtime_files_must_be_declared_by_plan() {
        let plan = RankBootstrapPlan {
            cluster_id: "cluster".into(),
            generation: 1,
            manifest_digest: "manifest".into(),
            rank: BootstrapRank { rank: 0 },
            artifacts: vec![BootstrapArtifact {
                relative_path: PathBuf::from("model-manifest.json"),
                digest: format!("sha256:{}", "ab".repeat(32)),
                size_bytes: 1,
            }],
        };
        assert!(
            plan.require_artifacts([PathBuf::from("model-manifest.json")])
                .is_ok()
        );
        assert!(
            plan.require_artifacts([PathBuf::from("weights/rank-0.safetensors")])
                .is_err()
        );
    }

    #[test]
    fn artifact_verification_hashes_files_and_detects_tampering() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ax-pipeline-artifact-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("temporary model root");
        let artifact_path = root.join("model-manifest.json");
        std::fs::write(&artifact_path, b"verified").expect("write artifact");
        let plan = RankBootstrapPlan {
            cluster_id: "cluster".into(),
            generation: 1,
            manifest_digest: "manifest".into(),
            rank: BootstrapRank { rank: 0 },
            artifacts: vec![BootstrapArtifact {
                relative_path: PathBuf::from("model-manifest.json"),
                digest: format!("sha256:{}", ax_engine_core::sha256_hex(b"verified")),
                size_bytes: 8,
            }],
        };
        plan.verify(&root, &topology(), 0)
            .expect("matching artifact verifies");
        std::fs::write(&artifact_path, b"tampered").expect("tamper artifact");
        assert!(matches!(
            plan.verify(&root, &topology(), 0),
            Err(ArtifactVerificationError::ArtifactDigestMismatch(_))
        ));
        std::fs::remove_dir_all(root).expect("remove temporary model root");
    }
}
