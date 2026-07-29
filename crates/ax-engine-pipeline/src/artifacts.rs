//! Rank bootstrap artifact preparation and integrity verification.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read as _;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use ax_engine_core::PipelineTopology;
use reqwest::Url;
use reqwest::header::AUTHORIZATION;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use thiserror::Error;
use tokio::io::AsyncWriteExt as _;

const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const MAX_ARTIFACT_COUNT: usize = 16_384;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Outcome of preparing the exact artifact subset assigned to one rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactPreparation {
    pub downloaded_files: usize,
    pub downloaded_bytes: u64,
    pub reused_files: usize,
}

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
    pub fn load(path: &Path) -> Result<Self, ArtifactVerificationError> {
        let bytes = std::fs::read(path).map_err(ArtifactVerificationError::ReadPlan)?;
        serde_json::from_slice::<Self>(&bytes).map_err(ArtifactVerificationError::ParsePlan)
    }

    pub fn load_and_verify(
        path: &Path,
        model_root: &Path,
        topology: &PipelineTopology,
        expected_rank: u16,
    ) -> Result<Self, ArtifactVerificationError> {
        let plan = Self::load(path)?;
        plan.verify(model_root, topology, expected_rank)?;
        Ok(plan)
    }

    /// Download only artifacts explicitly assigned by the integrity-bound rank
    /// plan, then verify the complete local subset before returning.
    pub async fn prepare_from_base_url(
        &self,
        model_root: &Path,
        topology: &PipelineTopology,
        expected_rank: u16,
        artifact_base_url: &str,
        bearer_token: Option<&str>,
    ) -> Result<ArtifactPreparation, ArtifactVerificationError> {
        self.validate_contract(topology, expected_rank)?;
        std::fs::create_dir_all(model_root).map_err(ArtifactVerificationError::CreateModelRoot)?;
        let canonical_root = model_root
            .canonicalize()
            .map_err(ArtifactVerificationError::ReadModelRoot)?;
        let base_url = parse_base_url(artifact_base_url)?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(DOWNLOAD_TIMEOUT)
            .build()
            .map_err(ArtifactVerificationError::BuildDownloadClient)?;

        let mut result = ArtifactPreparation {
            downloaded_files: 0,
            downloaded_bytes: 0,
            reused_files: 0,
        };
        for artifact in &self.artifacts {
            let expected_digest = parse_sha256(&artifact.digest)?;
            let destination = checked_destination(&canonical_root, &artifact.relative_path, true)?;
            if artifact_matches(&destination, artifact, expected_digest)? {
                result.reused_files += 1;
                continue;
            }

            let artifact_url = artifact_url(&base_url, &artifact.relative_path)?;
            let mut request = client.get(artifact_url);
            if let Some(token) = bearer_token {
                if token.len() < 16 || token.bytes().any(|byte| byte.is_ascii_control()) {
                    return Err(ArtifactVerificationError::InvalidArtifactCredential);
                }
                request = request.header(AUTHORIZATION, format!("Bearer {token}"));
            }
            let response = request.send().await.map_err(|source| {
                ArtifactVerificationError::DownloadRequest {
                    path: artifact.relative_path.clone(),
                    source,
                }
            })?;
            if !response.status().is_success() {
                return Err(ArtifactVerificationError::DownloadStatus {
                    path: artifact.relative_path.clone(),
                    status: response.status(),
                });
            }
            if response
                .content_length()
                .is_some_and(|length| length != artifact.size_bytes)
            {
                return Err(ArtifactVerificationError::DownloadLength {
                    path: artifact.relative_path.clone(),
                    expected: artifact.size_bytes,
                    actual: response.content_length(),
                });
            }

            let temporary = temporary_path(&destination)?;
            let download =
                download_to_temporary(response, &temporary, artifact, expected_digest).await;
            if let Err(error) = download {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(error);
            }
            tokio::fs::rename(&temporary, &destination)
                .await
                .map_err(|source| ArtifactVerificationError::InstallArtifact {
                    path: artifact.relative_path.clone(),
                    source,
                })?;
            result.downloaded_files += 1;
            result.downloaded_bytes = result
                .downloaded_bytes
                .checked_add(artifact.size_bytes)
                .ok_or(ArtifactVerificationError::ArtifactPlanSizeOverflow)?;
        }
        self.verify(&canonical_root, topology, expected_rank)?;
        Ok(result)
    }

    pub fn verify(
        &self,
        model_root: &Path,
        topology: &PipelineTopology,
        expected_rank: u16,
    ) -> Result<(), ArtifactVerificationError> {
        self.validate_contract(topology, expected_rank)?;
        let canonical_root = model_root
            .canonicalize()
            .map_err(ArtifactVerificationError::ReadModelRoot)?;
        for artifact in &self.artifacts {
            let expected_digest = parse_sha256(&artifact.digest)?;
            let canonical_path =
                checked_destination(&canonical_root, &artifact.relative_path, false)?;
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

    fn validate_contract(
        &self,
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
        if self.artifacts.len() > MAX_ARTIFACT_COUNT {
            return Err(ArtifactVerificationError::TooManyArtifacts {
                maximum: MAX_ARTIFACT_COUNT,
                actual: self.artifacts.len(),
            });
        }
        let mut paths = BTreeSet::new();
        let mut digests = BTreeSet::new();
        let mut total_size = 0_u64;
        for artifact in &self.artifacts {
            validate_relative_path(&artifact.relative_path)?;
            if artifact.size_bytes == 0 {
                return Err(ArtifactVerificationError::ZeroArtifactSize(
                    artifact.relative_path.clone(),
                ));
            }
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
            parse_sha256(&artifact.digest)?;
            total_size = total_size
                .checked_add(artifact.size_bytes)
                .ok_or(ArtifactVerificationError::ArtifactPlanSizeOverflow)?;
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

fn parse_base_url(value: &str) -> Result<Url, ArtifactVerificationError> {
    let mut url =
        Url::parse(value).map_err(|_| ArtifactVerificationError::InvalidArtifactBaseUrl)?;
    if !matches!(url.scheme(), "http" | "https")
        || url.cannot_be_a_base()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(ArtifactVerificationError::InvalidArtifactBaseUrl);
    }
    if !url.path().ends_with('/') {
        url.path_segments_mut()
            .map_err(|_| ArtifactVerificationError::InvalidArtifactBaseUrl)?
            .push("");
    }
    Ok(url)
}

fn artifact_url(base: &Url, path: &Path) -> Result<Url, ArtifactVerificationError> {
    validate_relative_path(path)?;
    let mut url = base.clone();
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| ArtifactVerificationError::InvalidArtifactBaseUrl)?;
        segments.pop_if_empty();
        for component in path.components() {
            let Component::Normal(value) = component else {
                return Err(ArtifactVerificationError::UnsafeArtifactPath(
                    path.to_path_buf(),
                ));
            };
            let value = value
                .to_str()
                .ok_or_else(|| ArtifactVerificationError::UnsafeArtifactPath(path.to_path_buf()))?;
            segments.push(value);
        }
    }
    Ok(url)
}

fn checked_destination(
    canonical_root: &Path,
    relative_path: &Path,
    create_parent: bool,
) -> Result<PathBuf, ArtifactVerificationError> {
    validate_relative_path(relative_path)?;
    let destination = canonical_root.join(relative_path);
    let parent = destination.parent().ok_or_else(|| {
        ArtifactVerificationError::UnsafeArtifactPath(relative_path.to_path_buf())
    })?;
    if create_parent {
        std::fs::create_dir_all(parent).map_err(|source| {
            ArtifactVerificationError::CreateArtifactDirectory {
                path: relative_path.to_path_buf(),
                source,
            }
        })?;
    }
    let canonical_parent =
        parent
            .canonicalize()
            .map_err(|source| ArtifactVerificationError::ReadArtifact {
                path: relative_path.to_path_buf(),
                source,
            })?;
    if !canonical_parent.starts_with(canonical_root) {
        return Err(ArtifactVerificationError::ArtifactEscapesModelRoot(
            relative_path.to_path_buf(),
        ));
    }
    let file_name = destination.file_name().ok_or_else(|| {
        ArtifactVerificationError::UnsafeArtifactPath(relative_path.to_path_buf())
    })?;
    let checked = canonical_parent.join(file_name);
    if checked.exists() {
        let canonical =
            checked
                .canonicalize()
                .map_err(|source| ArtifactVerificationError::ReadArtifact {
                    path: relative_path.to_path_buf(),
                    source,
                })?;
        if !canonical.starts_with(canonical_root) {
            return Err(ArtifactVerificationError::ArtifactEscapesModelRoot(
                relative_path.to_path_buf(),
            ));
        }
        return Ok(canonical);
    }
    Ok(checked)
}

fn artifact_matches(
    path: &Path,
    artifact: &BootstrapArtifact,
    expected_digest: [u8; 32],
) -> Result<bool, ArtifactVerificationError> {
    let Ok(metadata) = std::fs::metadata(path) else {
        return Ok(false);
    };
    if !metadata.is_file() || metadata.len() != artifact.size_bytes {
        return Ok(false);
    }
    Ok(sha256_file(path, &artifact.relative_path)? == expected_digest)
}

fn temporary_path(destination: &Path) -> Result<PathBuf, ArtifactVerificationError> {
    let file_name = destination
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| ArtifactVerificationError::UnsafeArtifactPath(destination.to_path_buf()))?;
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    Ok(destination.with_file_name(format!(
        ".{file_name}.ax-download-{}-{sequence}",
        std::process::id()
    )))
}

async fn download_to_temporary(
    mut response: reqwest::Response,
    temporary: &Path,
    artifact: &BootstrapArtifact,
    expected_digest: [u8; 32],
) -> Result<(), ArtifactVerificationError> {
    let mut file = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(temporary)
        .await
        .map_err(|source| ArtifactVerificationError::WriteArtifact {
            path: artifact.relative_path.clone(),
            source,
        })?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0_u64;
    while let Some(chunk) =
        response
            .chunk()
            .await
            .map_err(|source| ArtifactVerificationError::DownloadRequest {
                path: artifact.relative_path.clone(),
                source,
            })?
    {
        downloaded = downloaded
            .checked_add(chunk.len() as u64)
            .ok_or(ArtifactVerificationError::ArtifactPlanSizeOverflow)?;
        if downloaded > artifact.size_bytes {
            return Err(ArtifactVerificationError::DownloadLength {
                path: artifact.relative_path.clone(),
                expected: artifact.size_bytes,
                actual: Some(downloaded),
            });
        }
        hasher.update(&chunk);
        file.write_all(&chunk).await.map_err(|source| {
            ArtifactVerificationError::WriteArtifact {
                path: artifact.relative_path.clone(),
                source,
            }
        })?;
    }
    file.flush()
        .await
        .map_err(|source| ArtifactVerificationError::WriteArtifact {
            path: artifact.relative_path.clone(),
            source,
        })?;
    drop(file);
    if downloaded != artifact.size_bytes {
        return Err(ArtifactVerificationError::DownloadLength {
            path: artifact.relative_path.clone(),
            expected: artifact.size_bytes,
            actual: Some(downloaded),
        });
    }
    let actual_digest: [u8; 32] = hasher.finalize().into();
    if actual_digest != expected_digest {
        return Err(ArtifactVerificationError::ArtifactDigestMismatch(
            artifact.relative_path.clone(),
        ));
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<(), ArtifactVerificationError> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.as_os_str().len() > 512
        || path.components().any(|component| {
            matches!(
                component,
                Component::CurDir
                    | Component::ParentDir
                    | Component::RootDir
                    | Component::Prefix(_)
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
    #[error("failed to create model artifact root: {0}")]
    CreateModelRoot(std::io::Error),
    #[error("bootstrap plan does not match pipeline cluster, generation, manifest, or rank")]
    BootstrapIdentityMismatch,
    #[error("bootstrap plan contains no artifacts")]
    EmptyArtifactPlan,
    #[error("bootstrap plan has {actual} artifacts; maximum is {maximum}")]
    TooManyArtifacts { maximum: usize, actual: usize },
    #[error("bootstrap artifact size must be greater than zero: {0}")]
    ZeroArtifactSize(PathBuf),
    #[error("bootstrap artifact byte total overflowed u64")]
    ArtifactPlanSizeOverflow,
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
    #[error("artifact base URL must be an explicit credential-free http(s) base URL")]
    InvalidArtifactBaseUrl,
    #[error("artifact bearer token must contain at least 16 non-control bytes")]
    InvalidArtifactCredential,
    #[error("failed to construct artifact download client: {0}")]
    BuildDownloadClient(reqwest::Error),
    #[error("artifact download request failed for {path}: {source}")]
    DownloadRequest {
        path: PathBuf,
        source: reqwest::Error,
    },
    #[error("artifact download returned HTTP {status} for {path}")]
    DownloadStatus {
        path: PathBuf,
        status: reqwest::StatusCode,
    },
    #[error("artifact download length mismatch for {path}: expected {expected}, got {actual:?}")]
    DownloadLength {
        path: PathBuf,
        expected: u64,
        actual: Option<u64>,
    },
    #[error("failed to create artifact directory for {path}: {source}")]
    CreateArtifactDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write bootstrap artifact {path}: {source}")]
    WriteArtifact {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to atomically install bootstrap artifact {path}: {source}")]
    InstallArtifact {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use ax_engine_core::{PipelineLayerRange, PipelineRankAssignment};
    use axum::Router;
    use axum::body::Body;
    use axum::http::StatusCode;
    use axum::routing::get;

    use super::*;

    fn topology() -> PipelineTopology {
        PipelineTopology {
            cluster_id: "cluster".into(),
            generation: 1,
            manifest_digest: "manifest".into(),
            model_artifact_digest: "model".into(),
            total_layers: 2,
            micro_batch_limit: 2,
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
        assert!(validate_relative_path(Path::new("./weights")).is_err());
        assert!(validate_relative_path(Path::new("../rank-1.safetensors")).is_err());
        assert!(validate_relative_path(Path::new("/tmp/model")).is_err());
    }

    #[test]
    fn artifact_url_encodes_each_safe_path_segment() {
        let base = parse_base_url("https://models.example/base").expect("base URL");
        let url = artifact_url(&base, Path::new("weights/model shard.safetensors"))
            .expect("artifact URL");
        assert_eq!(
            url.as_str(),
            "https://models.example/base/weights/model%20shard.safetensors"
        );
        assert!(parse_base_url("https://user:secret@models.example/base").is_err());
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

    #[tokio::test]
    async fn artifact_preparation_downloads_only_rank_plan_and_reuses_verified_files() {
        let body = b"rank-zero-weights".to_vec();
        let app = Router::new().route(
            "/models/weights/rank-0.safetensors",
            get({
                let body = body.clone();
                move || {
                    let body = body.clone();
                    async move { (StatusCode::OK, Body::from(body)) }
                }
            }),
        );
        let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(_) => return,
        };
        let address = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("artifact server");
        });

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ax-pipeline-download-{}-{nonce}",
            std::process::id()
        ));
        let plan = RankBootstrapPlan {
            cluster_id: "cluster".into(),
            generation: 1,
            manifest_digest: "manifest".into(),
            rank: BootstrapRank { rank: 0 },
            artifacts: vec![BootstrapArtifact {
                relative_path: PathBuf::from("weights/rank-0.safetensors"),
                digest: format!("sha256:{}", ax_engine_core::sha256_hex(&body)),
                size_bytes: body.len() as u64,
            }],
        };
        let first = plan
            .prepare_from_base_url(
                &root,
                &topology(),
                0,
                &format!("http://{address}/models"),
                None,
            )
            .await
            .expect("download rank artifacts");
        assert_eq!(
            first,
            ArtifactPreparation {
                downloaded_files: 1,
                downloaded_bytes: body.len() as u64,
                reused_files: 0,
            }
        );
        assert_eq!(
            std::fs::read(root.join("weights/rank-0.safetensors")).expect("prepared artifact"),
            body
        );

        server.abort();
        let second = plan
            .prepare_from_base_url(&root, &topology(), 0, "http://127.0.0.1:1/models", None)
            .await
            .expect("reuse already verified artifact without a network request");
        assert_eq!(
            second,
            ArtifactPreparation {
                downloaded_files: 0,
                downloaded_bytes: 0,
                reused_files: 1,
            }
        );
        std::fs::remove_dir_all(root).expect("remove temporary model root");
    }

    #[tokio::test]
    async fn artifact_preparation_rejects_oversized_or_tampered_downloads() {
        let body = b"larger-than-certified".to_vec();
        let app = Router::new().fallback(move || {
            let body = body.clone();
            async move { (StatusCode::OK, Body::from(body)) }
        });
        let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(_) => return,
        };
        let address = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("artifact server");
        });
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ax-pipeline-download-reject-{}-{nonce}",
            std::process::id()
        ));
        let plan = RankBootstrapPlan {
            cluster_id: "cluster".into(),
            generation: 1,
            manifest_digest: "manifest".into(),
            rank: BootstrapRank { rank: 0 },
            artifacts: vec![BootstrapArtifact {
                relative_path: PathBuf::from("weights/rank-0.safetensors"),
                digest: format!("sha256:{}", ax_engine_core::sha256_hex(b"expected")),
                size_bytes: 8,
            }],
        };
        let error = plan
            .prepare_from_base_url(
                &root,
                &topology(),
                0,
                &format!("http://{address}/models"),
                None,
            )
            .await
            .expect_err("oversized artifact must fail closed");
        assert!(matches!(
            error,
            ArtifactVerificationError::DownloadLength { .. }
        ));
        assert!(!root.join("weights/rank-0.safetensors").exists());
        server.abort();
        std::fs::remove_dir_all(root).expect("remove temporary model root");
    }
}
