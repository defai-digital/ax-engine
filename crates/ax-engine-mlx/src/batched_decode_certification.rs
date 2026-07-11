use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;

use ax_engine_core::{AX_NATIVE_MODEL_MANIFEST_FILE, NativeModelArtifacts, NativeTensorSpec};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const BATCHED_DECODE_CERTIFICATION_FILE: &str = "batched-decode-certification.json";
pub const BATCHED_DECODE_CERTIFICATION_SCHEMA: &str = "ax.mlx.batched_decode_certification.v1";
pub const BATCHED_DECODE_RUNTIME_CONTRACT: &str = "ax.mlx.batched_decode.runtime.v2";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BatchedDecodeCertificationScenario {
    pub batch: u32,
    pub prompt_len: u32,
    pub gen_len: u32,
    pub prompt_seed: u64,
    pub ragged: bool,
    pub sampling: String,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BatchedDecodeCertificationEvidence {
    pub schema_version: String,
    pub verdict: String,
    pub model_family: String,
    pub artifact_fingerprint_sha256: String,
    pub engine_version: String,
    pub mlx_version: String,
    pub device_architecture: String,
    pub runtime_contract: String,
    pub numerics_env_sha256: String,
    pub scenarios: Vec<BatchedDecodeCertificationScenario>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BatchedDecodeCertificationContext {
    pub schema_version: String,
    pub model_family: String,
    pub artifact_fingerprint_sha256: String,
    pub engine_version: String,
    pub mlx_version: String,
    pub device_architecture: String,
    pub runtime_contract: String,
    pub numerics_env_sha256: String,
    pub required_scenarios: Vec<BatchedDecodeCertificationScenario>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchedDecodeCertificationStatus {
    Certified,
    Missing,
    Unreadable,
    Invalid,
    EvidenceFailed,
    ModelMismatch,
    ArtifactIdentityUnavailable,
    ArtifactMismatch,
    EngineMismatch,
    MlxMismatch,
    DeviceMismatch,
    RuntimeMismatch,
    EnvironmentMismatch,
    CoverageIncomplete,
}

impl BatchedDecodeCertificationStatus {
    pub const fn is_certified(self) -> bool {
        matches!(self, Self::Certified)
    }

    pub const fn route_reason(self) -> &'static str {
        match self {
            Self::Certified => "certified",
            Self::Missing => "certification_missing",
            Self::Unreadable => "certification_unreadable",
            Self::Invalid => "certification_invalid",
            Self::EvidenceFailed => "certification_failed",
            Self::ModelMismatch => "certification_model_mismatch",
            Self::ArtifactIdentityUnavailable => "certification_artifact_identity_unavailable",
            Self::ArtifactMismatch => "certification_artifact_mismatch",
            Self::EngineMismatch => "certification_engine_mismatch",
            Self::MlxMismatch => "certification_mlx_mismatch",
            Self::DeviceMismatch => "certification_device_mismatch",
            Self::RuntimeMismatch => "certification_runtime_mismatch",
            Self::EnvironmentMismatch => "certification_environment_mismatch",
            Self::CoverageIncomplete => "certification_coverage_incomplete",
        }
    }
}

pub fn required_batched_decode_certification_scenarios() -> Vec<BatchedDecodeCertificationScenario>
{
    [
        (2, 32, 64, 0, false),
        (4, 32, 64, 0, false),
        (4, 128, 64, 0, false),
        (4, 128, 64, 1, false),
        (4, 512, 64, 0, false),
        (4, 128, 64, 0, true),
    ]
    .into_iter()
    .map(
        |(batch, prompt_len, gen_len, prompt_seed, ragged)| BatchedDecodeCertificationScenario {
            batch,
            prompt_len,
            gen_len,
            prompt_seed,
            ragged,
            sampling: String::from("greedy"),
            passed: true,
        },
    )
    .collect()
}

pub fn batched_decode_certification_context(
    artifacts: &NativeModelArtifacts,
) -> Result<BatchedDecodeCertificationContext, BatchedDecodeCertificationStatus> {
    let artifact_fingerprint_sha256 = artifact_fingerprint_sha256(artifacts)?;
    let mlx_version =
        mlx_sys::runtime_version().map_err(|_| BatchedDecodeCertificationStatus::MlxMismatch)?;
    let device_architecture = mlx_sys::gpu_device_architecture()
        .map_err(|_| BatchedDecodeCertificationStatus::DeviceMismatch)?;
    Ok(BatchedDecodeCertificationContext {
        schema_version: String::from(BATCHED_DECODE_CERTIFICATION_SCHEMA),
        model_family: artifacts.manifest().model_family.clone(),
        artifact_fingerprint_sha256,
        engine_version: String::from(env!("CARGO_PKG_VERSION")),
        mlx_version,
        device_architecture,
        runtime_contract: String::from(BATCHED_DECODE_RUNTIME_CONTRACT),
        numerics_env_sha256: batched_decode_numerics_env_sha256(),
        required_scenarios: required_batched_decode_certification_scenarios(),
    })
}

pub(crate) fn load_batched_decode_certification(
    artifacts: &NativeModelArtifacts,
) -> BatchedDecodeCertificationStatus {
    let path = artifacts.root_dir().join(BATCHED_DECODE_CERTIFICATION_FILE);
    if !path.is_file() {
        return BatchedDecodeCertificationStatus::Missing;
    }
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => return BatchedDecodeCertificationStatus::Unreadable,
    };
    let evidence = match serde_json::from_slice::<BatchedDecodeCertificationEvidence>(&bytes) {
        Ok(evidence) => evidence,
        Err(_) => return BatchedDecodeCertificationStatus::Invalid,
    };
    let context = match batched_decode_certification_context(artifacts) {
        Ok(context) => context,
        Err(status) => return status,
    };
    validate_batched_decode_certification(&evidence, &context)
}

pub fn batched_decode_numerics_env_sha256() -> String {
    numerics_env_sha256_from_iter(env::vars())
}

fn validate_batched_decode_certification(
    evidence: &BatchedDecodeCertificationEvidence,
    context: &BatchedDecodeCertificationContext,
) -> BatchedDecodeCertificationStatus {
    if evidence.schema_version != BATCHED_DECODE_CERTIFICATION_SCHEMA {
        return BatchedDecodeCertificationStatus::Invalid;
    }
    if evidence.verdict != "pass" || evidence.scenarios.iter().any(|scenario| !scenario.passed) {
        return BatchedDecodeCertificationStatus::EvidenceFailed;
    }
    if evidence.model_family != context.model_family {
        return BatchedDecodeCertificationStatus::ModelMismatch;
    }
    if evidence.artifact_fingerprint_sha256 != context.artifact_fingerprint_sha256 {
        return BatchedDecodeCertificationStatus::ArtifactMismatch;
    }
    if evidence.engine_version != context.engine_version {
        return BatchedDecodeCertificationStatus::EngineMismatch;
    }
    if evidence.mlx_version != context.mlx_version {
        return BatchedDecodeCertificationStatus::MlxMismatch;
    }
    if evidence.device_architecture != context.device_architecture {
        return BatchedDecodeCertificationStatus::DeviceMismatch;
    }
    if evidence.runtime_contract != context.runtime_contract {
        return BatchedDecodeCertificationStatus::RuntimeMismatch;
    }
    if evidence.numerics_env_sha256 != context.numerics_env_sha256 {
        return BatchedDecodeCertificationStatus::EnvironmentMismatch;
    }
    let covers_required = context.required_scenarios.iter().all(|required| {
        evidence.scenarios.iter().any(|observed| {
            observed.batch == required.batch
                && observed.prompt_len == required.prompt_len
                && observed.gen_len == required.gen_len
                && observed.prompt_seed == required.prompt_seed
                && observed.ragged == required.ragged
                && observed.sampling == required.sampling
                && observed.passed
        })
    });
    if !covers_required {
        return BatchedDecodeCertificationStatus::CoverageIncomplete;
    }
    BatchedDecodeCertificationStatus::Certified
}

fn artifact_fingerprint_sha256(
    artifacts: &NativeModelArtifacts,
) -> Result<String, BatchedDecodeCertificationStatus> {
    let manifest_path = artifacts.root_dir().join(AX_NATIVE_MODEL_MANIFEST_FILE);
    let manifest = fs::read(manifest_path)
        .map_err(|_| BatchedDecodeCertificationStatus::ArtifactIdentityUnavailable)?;
    let mut hasher = Sha256::new();
    hasher.update(b"ax.mlx.batched_decode.artifact.v1\0");
    hasher.update(&manifest);

    let files = artifacts
        .tensor_specs()
        .iter()
        .map(tensor_file)
        .collect::<BTreeSet<_>>();
    for file in files {
        let path = artifacts.root_dir().join(file);
        let target = fs::read_link(&path)
            .map_err(|_| BatchedDecodeCertificationStatus::ArtifactIdentityUnavailable)?;
        let content_hash = target
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| is_sha256_hex(name))
            .ok_or(BatchedDecodeCertificationStatus::ArtifactIdentityUnavailable)?;
        let byte_len = fs::metadata(&path)
            .map_err(|_| BatchedDecodeCertificationStatus::ArtifactIdentityUnavailable)?
            .len();
        hasher.update(file.as_bytes());
        hasher.update(b"\0");
        hasher.update(content_hash.as_bytes());
        hasher.update(b"\0");
        hasher.update(byte_len.to_le_bytes());
    }
    let digest = hasher.finalize();
    Ok(hex_digest(&digest))
}

fn tensor_file(tensor: &NativeTensorSpec) -> &str {
    tensor.file.to_str().unwrap_or("")
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn numerics_env_sha256_from_iter<I>(vars: I) -> String
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut relevant = vars
        .into_iter()
        .filter(|(key, _)| numerics_env_key_relevant(key))
        .collect::<Vec<_>>();
    relevant.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(b"ax.mlx.batched_decode.numerics_env.v1\0");
    for (key, value) in relevant {
        hasher.update(key.as_bytes());
        hasher.update(b"=");
        hasher.update(value.as_bytes());
        hasher.update(b"\n");
    }
    let digest = hasher.finalize();
    hex_digest(&digest)
}

fn numerics_env_key_relevant(key: &str) -> bool {
    if !key.starts_with("AX_MLX_") && !key.starts_with("MLX_") {
        return false;
    }
    !matches!(
        key,
        "AX_MLX_BATCHED_DECODE"
            | "AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED"
            | "AX_MLX_BATCHED_DECODE_MAX"
            | "AX_MLX_BATCHED_DECODE_SAMPLING"
    )
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> BatchedDecodeCertificationContext {
        BatchedDecodeCertificationContext {
            schema_version: String::from(BATCHED_DECODE_CERTIFICATION_SCHEMA),
            model_family: String::from("qwen3"),
            artifact_fingerprint_sha256: String::from("artifact"),
            engine_version: String::from("6.8.2"),
            mlx_version: String::from("0.29.3"),
            device_architecture: String::from("applegpu_test"),
            runtime_contract: String::from(BATCHED_DECODE_RUNTIME_CONTRACT),
            numerics_env_sha256: String::from("environment"),
            required_scenarios: required_batched_decode_certification_scenarios(),
        }
    }

    fn evidence() -> BatchedDecodeCertificationEvidence {
        let context = context();
        BatchedDecodeCertificationEvidence {
            schema_version: context.schema_version,
            verdict: String::from("pass"),
            model_family: context.model_family,
            artifact_fingerprint_sha256: context.artifact_fingerprint_sha256,
            engine_version: context.engine_version,
            mlx_version: context.mlx_version,
            device_architecture: context.device_architecture,
            runtime_contract: context.runtime_contract,
            numerics_env_sha256: context.numerics_env_sha256,
            scenarios: context.required_scenarios,
        }
    }

    #[test]
    fn complete_matching_evidence_certifies() {
        assert_eq!(
            validate_batched_decode_certification(&evidence(), &context()),
            BatchedDecodeCertificationStatus::Certified
        );
    }

    #[test]
    fn weak_or_failed_evidence_fails_closed() {
        let mut weak = evidence();
        weak.scenarios.pop();
        assert_eq!(
            validate_batched_decode_certification(&weak, &context()),
            BatchedDecodeCertificationStatus::CoverageIncomplete
        );

        let mut failed = evidence();
        failed.scenarios[0].passed = false;
        assert_eq!(
            validate_batched_decode_certification(&failed, &context()),
            BatchedDecodeCertificationStatus::EvidenceFailed
        );
    }

    #[test]
    fn runtime_device_and_environment_drift_fail_closed() {
        let mut runtime = evidence();
        runtime.runtime_contract.push_str(".changed");
        assert_eq!(
            validate_batched_decode_certification(&runtime, &context()),
            BatchedDecodeCertificationStatus::RuntimeMismatch
        );

        let mut device = evidence();
        device.device_architecture.push_str(".changed");
        assert_eq!(
            validate_batched_decode_certification(&device, &context()),
            BatchedDecodeCertificationStatus::DeviceMismatch
        );

        let mut environment = evidence();
        environment.numerics_env_sha256.push_str("changed");
        assert_eq!(
            validate_batched_decode_certification(&environment, &context()),
            BatchedDecodeCertificationStatus::EnvironmentMismatch
        );
    }

    #[test]
    fn numerics_environment_ignores_batching_controls_but_tracks_fastpaths() {
        let baseline = numerics_env_sha256_from_iter(Vec::<(String, String)>::new());
        let controls = numerics_env_sha256_from_iter(vec![
            (String::from("AX_MLX_BATCHED_DECODE"), String::from("1")),
            (
                String::from("AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED"),
                String::from("1"),
            ),
        ]);
        assert_eq!(baseline, controls);

        let fastpath = numerics_env_sha256_from_iter(vec![(
            String::from("AX_MLX_DENSE_FFN_COMPILE"),
            String::from("0"),
        )]);
        assert_ne!(baseline, fastpath);
    }
}
