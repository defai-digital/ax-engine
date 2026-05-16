use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use crate::error::CliError;
use crate::json_io::{load_json_value, nested_value};

pub(crate) fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), CliError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        CliError::Runtime(format!(
            "failed to serialize JSON for {}: {error}",
            path.display()
        ))
    })?;
    fs::write(path, bytes)
        .map_err(|error| CliError::Runtime(format!("failed to write {}: {error}", path.display())))
}

pub(crate) fn create_unique_result_dir(
    output_root: &Path,
    label: Option<&str>,
    component: &str,
) -> Result<(String, PathBuf), CliError> {
    let timestamp = unix_timestamp_secs()?;
    let component = sanitize_component(component);

    for attempt in 0..32 {
        let suffix = unique_run_suffix(attempt)?;
        let run_id = match label {
            Some(label) => format!("{timestamp}-{suffix}-{label}-{component}"),
            None => format!("{timestamp}-{suffix}-{component}"),
        };
        let result_dir = output_root.join(&run_id);
        match fs::create_dir(&result_dir) {
            Ok(()) => return Ok((run_id, result_dir)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(CliError::Runtime(format!(
                    "failed to create result directory {}: {error}",
                    result_dir.display()
                )));
            }
        }
    }

    Err(CliError::Runtime(format!(
        "failed to create unique result directory under {} after repeated collisions",
        output_root.display()
    )))
}

pub(crate) fn unique_run_suffix(attempt: u32) -> Result<String, CliError> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .map_err(|error| CliError::Runtime(format!("system clock error: {error}")))?;
    let pid = u64::from(std::process::id());
    let mixed = (nanos as u64)
        ^ pid.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(attempt).rotate_left(17)
        ^ ((nanos >> 64) as u64);
    Ok(format!("{:012x}", mixed & 0xFFFF_FFFF_FFFF))
}

pub(crate) fn unix_timestamp_secs() -> Result<u64, CliError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| CliError::Runtime(format!("system clock error: {error}")))
}

pub(crate) fn sanitize_component(input: &str) -> String {
    input
        .chars()
        .map(|char| match char {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => char,
            _ => '-',
        })
        .collect()
}

pub(crate) fn copy_required_artifact_file(
    source_dir: &Path,
    destination_dir: &Path,
    name: &str,
) -> Result<(), CliError> {
    let source = source_dir.join(name);
    if !source.is_file() {
        return Err(CliError::Contract(format!(
            "benchmark artifact missing {name}: {}",
            source_dir.display()
        )));
    }
    fs::copy(&source, destination_dir.join(name)).map_err(|error| {
        CliError::Runtime(format!(
            "failed to copy {} into {}: {error}",
            source.display(),
            destination_dir.display()
        ))
    })?;
    Ok(())
}

pub(crate) fn copy_optional_artifact_file(
    source_dir: &Path,
    destination_dir: &Path,
    name: &str,
) -> Result<(), CliError> {
    let source = source_dir.join(name);
    if !source.is_file() {
        return Ok(());
    }
    fs::copy(&source, destination_dir.join(name)).map_err(|error| {
        CliError::Runtime(format!(
            "failed to copy {} into {}: {error}",
            source.display(),
            destination_dir.display()
        ))
    })?;
    Ok(())
}

pub(crate) fn reject_contract_failure_artifact_dir(
    path: &Path,
    label: &str,
) -> Result<(), CliError> {
    let failure_path = path.join("contract_failure.json");
    if !failure_path.is_file() {
        return Ok(());
    }

    let failure = load_json_value(&failure_path)?;
    let code = nested_value(&failure, &["failure", "code"])
        .and_then(Value::as_str)
        .unwrap_or("contract_validation_failed");
    let message = nested_value(&failure, &["failure", "message"])
        .and_then(Value::as_str)
        .unwrap_or("contract failure artifact present");

    Err(CliError::Contract(format!(
        "compare requires successful execution artifacts; {label} points to a contract-failure result at {} ({code}): {message}",
        path.display()
    )))
}
