use std::env;
use std::path::{Component, Path, PathBuf};

use crate::error::CliError;

pub(crate) fn path_string(path: &Path) -> String {
    path.display().to_string()
}

pub(crate) fn absolutize_path(path: PathBuf, current_dir: &Path) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        current_dir.join(path)
    }
}

pub(crate) fn expand_manifest_path_env(path: &Path) -> Result<PathBuf, CliError> {
    let raw = path.to_string_lossy();
    if let Some(variable) = raw
        .strip_prefix("${")
        .and_then(|value| value.strip_suffix('}'))
    {
        return env::var_os(variable).map(PathBuf::from).ok_or_else(|| {
            CliError::Contract(format!(
                "manifest path references unset environment variable {variable}"
            ))
        });
    }

    if let Some(stripped) = raw.strip_prefix('$') {
        let variable_len = stripped
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .count();
        if variable_len > 0 {
            let (variable, remainder) = stripped.split_at(variable_len);
            let value = env::var_os(variable).ok_or_else(|| {
                CliError::Contract(format!(
                    "manifest path references unset environment variable {variable}"
                ))
            })?;
            let mut resolved = PathBuf::from(value);
            if !remainder.is_empty() {
                resolved.push(remainder.trim_start_matches(['/', '\\']));
            }
            return Ok(resolved);
        }
    }

    Ok(path.to_path_buf())
}

pub(crate) fn normalize_path_lexically(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                ) {
                    normalized.pop();
                } else if !normalized.has_root() {
                    normalized.push(component.as_os_str());
                }
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}
