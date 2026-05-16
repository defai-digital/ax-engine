use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::error::CliError;

pub(crate) fn load_json_value(path: &Path) -> Result<Value, CliError> {
    let raw = fs::read_to_string(path).map_err(|error| {
        CliError::Runtime(format!("failed to read {}: {error}", path.display()))
    })?;

    serde_json::from_str(&raw).map_err(|error| {
        CliError::Contract(format!(
            "failed to parse {} as JSON: {error}",
            path.display()
        ))
    })
}

pub(crate) fn load_optional_json_value(path: &Path) -> Result<Option<Value>, CliError> {
    if !path.is_file() {
        return Ok(None);
    }
    load_json_value(path).map(Some)
}

pub(crate) fn nested_value<'a>(json: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = json;
    for component in path {
        current = current.get(*component)?;
    }
    Some(current)
}

pub(crate) fn nested_string<'a>(json: &'a Value, path: &[&str]) -> Result<&'a str, CliError> {
    nested_value(json, path)
        .and_then(Value::as_str)
        .ok_or_else(|| CliError::Contract(format!("missing string field {}", path.join("."))))
}

pub(crate) fn validate_matching_json_field(
    baseline: &Value,
    candidate: &Value,
    field: &[&str],
) -> Result<(), CliError> {
    let baseline_value = nested_value(baseline, field)
        .ok_or_else(|| CliError::Contract(format!("baseline missing {}", field.join("."))))?;
    let candidate_value = nested_value(candidate, field)
        .ok_or_else(|| CliError::Contract(format!("candidate missing {}", field.join("."))))?;
    validate_matching_values(field, baseline_value, candidate_value)
}

pub(crate) fn validate_matching_optional_json_field(
    baseline: &Value,
    candidate: &Value,
    field: &[&str],
) -> Result<(), CliError> {
    match (
        nested_value(baseline, field),
        nested_value(candidate, field),
    ) {
        (None, None) => Ok(()),
        (Some(baseline_value), Some(candidate_value)) => {
            validate_matching_values(field, baseline_value, candidate_value)
        }
        (Some(_), None) => Err(CliError::Contract(format!(
            "candidate missing {}",
            field.join(".")
        ))),
        (None, Some(_)) => Err(CliError::Contract(format!(
            "baseline missing {}",
            field.join(".")
        ))),
    }
}

fn validate_matching_values(
    field: &[&str],
    baseline_value: &Value,
    candidate_value: &Value,
) -> Result<(), CliError> {
    if baseline_value != candidate_value {
        return Err(CliError::Contract(format!(
            "benchmark contract mismatch for {}: baseline={}, candidate={}",
            field.join("."),
            json_value_label(baseline_value),
            json_value_label(candidate_value)
        )));
    }

    Ok(())
}

pub(crate) fn json_value_label(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}
