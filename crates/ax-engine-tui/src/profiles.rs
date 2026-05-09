use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const PROFILE_SCHEMA_VERSION: &str = "ax.engine_manager.profile.v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ManagerProfile {
    pub schema_version: String,
    pub name: String,
    pub model_dir: Option<String>,
    pub server_url: Option<String>,
    pub artifact_root: Option<String>,
}

impl ManagerProfile {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: PROFILE_SCHEMA_VERSION.to_string(),
            name: name.into(),
            model_dir: None,
            server_url: None,
            artifact_root: None,
        }
    }
}

pub fn default_profile_root() -> Result<PathBuf, ProfileError> {
    if let Some(path) = std::env::var_os("AX_ENGINE_MANAGER_CONFIG_DIR") {
        return Ok(PathBuf::from(path).join("profiles"));
    }
    if let Some(path) = std::env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(path)
            .join("ax-engine")
            .join("manager")
            .join("profiles"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(home)
            .join(".config")
            .join("ax-engine")
            .join("manager")
            .join("profiles"));
    }
    Err(ProfileError::MissingConfigRoot)
}

pub fn profile_path(root: &Path, name: &str) -> Result<PathBuf, ProfileError> {
    if !name
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(ProfileError::InvalidName(name.to_string()));
    }
    Ok(root.join(format!("{name}.json")))
}

pub fn write_profile(root: &Path, profile: &ManagerProfile) -> Result<PathBuf, ProfileError> {
    fs::create_dir_all(root).map_err(|source| ProfileError::Io {
        path: root.display().to_string(),
        source,
    })?;
    let path = profile_path(root, &profile.name)?;
    let text = serde_json::to_string_pretty(profile).map_err(ProfileError::Serialize)?;
    fs::write(&path, format!("{text}\n")).map_err(|source| ProfileError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(path)
}

pub fn read_profile(path: &Path) -> Result<ManagerProfile, ProfileError> {
    let text = fs::read_to_string(path).map_err(|source| ProfileError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let profile: ManagerProfile = serde_json::from_str(&text).map_err(ProfileError::Parse)?;
    if profile.schema_version != PROFILE_SCHEMA_VERSION {
        return Err(ProfileError::Schema {
            expected: PROFILE_SCHEMA_VERSION,
            actual: profile.schema_version,
        });
    }
    Ok(profile)
}

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("AX Engine Manager config root could not be resolved")]
    MissingConfigRoot,
    #[error("profile name must contain only ASCII letters, digits, '-' or '_': {0}")]
    InvalidName(String),
    #[error("profile IO failed for {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("profile JSON parse failed: {0}")]
    Parse(serde_json::Error),
    #[error("profile JSON serialization failed: {0}")]
    Serialize(serde_json::Error),
    #[error("unsupported profile schema_version: expected {expected}, got {actual}")]
    Schema {
        expected: &'static str,
        actual: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn writes_and_reads_profile_contract() {
        let root = tempdir().expect("tempdir should create");
        let mut profile = ManagerProfile::new("local-dev");
        profile.model_dir = Some("/models/qwen".to_string());
        profile.server_url = Some("http://127.0.0.1:8080".to_string());

        let path = write_profile(root.path(), &profile).expect("profile should write");
        let read = read_profile(&path).expect("profile should read");

        assert_eq!(read, profile);
    }

    #[test]
    fn rejects_path_like_profile_names() {
        assert!(profile_path(Path::new("/tmp"), "../local").is_err());
        assert!(profile_path(Path::new("/tmp"), "local dev").is_err());
    }
}
