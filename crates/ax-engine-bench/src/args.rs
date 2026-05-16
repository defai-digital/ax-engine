use std::fs;
use std::path::{Path, PathBuf};

use crate::error::CliError;

pub(crate) fn next_flag_value<'a>(
    iter: &mut std::slice::Iter<'a, String>,
    name: &str,
) -> Result<&'a str, CliError> {
    iter.next()
        .map(|value| value.as_str())
        .ok_or_else(|| CliError::Usage(format!("missing value for required flag {name}")))
}

pub(crate) fn parse_flag_value<T>(value: &str, name: &str) -> Result<T, CliError>
where
    T: std::str::FromStr,
{
    value.parse::<T>().map_err(|_| {
        CliError::Usage(format!(
            "invalid value for {name}: expected {}, got {value}",
            std::any::type_name::<T>()
        ))
    })
}

pub(crate) fn parse_u32_list(value: &str, name: &str) -> Result<Vec<u32>, CliError> {
    let parts = split_list_parts(value);
    if parts.is_empty() {
        return Err(CliError::Usage(format!(
            "{name} expects a comma- or space-separated list"
        )));
    }
    Ok(unique_sorted_u32(
        parts
            .into_iter()
            .map(|part| parse_flag_value::<u32>(part, name))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

pub(crate) fn parse_optional_u32_list(
    value: &str,
    name: &str,
) -> Result<Vec<Option<u32>>, CliError> {
    let parts = split_list_parts(value);
    if parts.is_empty() {
        return Err(CliError::Usage(format!(
            "{name} expects a comma- or space-separated list"
        )));
    }
    Ok(unique_sorted_option_u32(
        parts
            .into_iter()
            .map(|part| {
                if part.eq_ignore_ascii_case("none") {
                    Ok(None)
                } else {
                    Ok(Some(parse_flag_value::<u32>(part, name)?))
                }
            })
            .collect::<Result<Vec<_>, CliError>>()?,
    ))
}

pub(crate) fn parse_bool_list(value: &str, name: &str) -> Result<Vec<bool>, CliError> {
    let parts = split_list_parts(value);
    if parts.is_empty() {
        return Err(CliError::Usage(format!(
            "{name} expects a comma- or space-separated list"
        )));
    }
    Ok(unique_sorted_bool(
        parts
            .into_iter()
            .map(|part| parse_flag_value::<bool>(part, name))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

pub(crate) fn split_list_parts(value: &str) -> Vec<&str> {
    value
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect()
}

pub(crate) fn parse_token_list(value: &str) -> Result<Vec<u32>, CliError> {
    split_list_parts(value)
        .into_iter()
        .map(|part| {
            part.parse::<u32>().map_err(|_| {
                CliError::Usage(format!(
                    "invalid token id {part}; --tokens expects a comma- or space-separated u32 list"
                ))
            })
        })
        .collect()
}

pub(crate) fn unique_sorted_u32(values: Vec<u32>) -> Vec<u32> {
    let mut values = values;
    values.sort_unstable();
    values.dedup();
    values
}

pub(crate) fn unique_sorted_option_u32(values: Vec<Option<u32>>) -> Vec<Option<u32>> {
    let mut values = values;
    values.sort_by_key(|value| value.unwrap_or(0));
    values.dedup();
    values
}

pub(crate) fn unique_sorted_bool(values: Vec<bool>) -> Vec<bool> {
    let mut values = values;
    values.sort_unstable();
    values.dedup();
    values
}

pub(crate) fn required_flag(args: &[String], name: &str) -> Result<PathBuf, CliError> {
    Ok(PathBuf::from(required_string_flag(args, name)?))
}

pub(crate) fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

pub(crate) fn required_string_flag(args: &[String], name: &str) -> Result<String, CliError> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            let Some(value) = iter.next() else {
                return Err(CliError::Usage(format!(
                    "missing value for required flag {name}"
                )));
            };
            return Ok(value.clone());
        }
    }

    Err(CliError::Usage(format!("missing required flag {name}")))
}

pub(crate) fn require_existing_file(path: &Path) -> Result<(), CliError> {
    if !path.is_file() {
        return Err(CliError::Contract(format!(
            "manifest file does not exist: {}",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn require_existing_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::Contract(format!(
            "required path does not exist: {}",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn require_existing_dir(path: &Path) -> Result<(), CliError> {
    if !path.is_dir() {
        return Err(CliError::Contract(format!(
            "required directory does not exist or is not a directory: {}",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn ensure_output_root(path: &Path) -> Result<(), CliError> {
    if path.exists() && !path.is_dir() {
        return Err(CliError::Contract(format!(
            "output root exists but is not a directory: {}",
            path.display()
        )));
    }

    if !path.exists() {
        fs::create_dir_all(path).map_err(|error| {
            CliError::Runtime(format!(
                "failed to create output root {}: {error}",
                path.display()
            ))
        })?;
    }

    Ok(())
}
