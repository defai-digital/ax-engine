//! Helpers shared by the `src/bin/*.rs` probe binaries.
//!
//! Every probe is its own crate target and historically carried its own copy
//! of these helpers. Error strings and median shapes are part of each probe's
//! printed contract, so where the copies disagreed the exact variants are kept
//! side by side instead of being collapsed.

#![allow(dead_code)]

use std::env;

/// Read `name` when present: unset is `Ok(None)`; an unreadable (non-Unicode)
/// value is an error.
pub fn optional_env(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed to read {name}: {error}")),
    }
}

/// Parse `name` as a `usize`, falling back to `default` for any read failure
/// (unset or non-Unicode); a present but unparsable value is an error.
pub fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a non-negative integer, got {value:?}")),
        Err(_) => Ok(default),
    }
}

/// [`env_usize`] with strict read handling: a present-but-unreadable value is
/// an error instead of falling back to `default`.
pub fn env_usize_strict(name: &str, default: usize) -> Result<usize, String> {
    optional_env(name)?
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{name} must be a non-negative integer, got {value:?}"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

/// Median of the samples: the mean of the two middle samples for even
/// lengths. Empty input yields 0.0.
pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return 0.0;
    }
    let middle = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[middle - 1] + v[middle]) * 0.5
    } else {
        v[middle]
    }
}

/// Upper median: the sample at index `len / 2` after sorting (the higher of
/// the two middle samples for even lengths). Empty input yields 0.0.
pub fn median_upper(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return 0.0;
    }
    v[v.len() / 2]
}

/// Parse a comma- or whitespace-separated list of `u32` token ids; blank
/// entries are skipped and an empty list is rejected.
pub fn parse_token_ids(spec: &str) -> Result<Vec<u32>, String> {
    let ids = spec
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.trim().is_empty())
        .map(|token| {
            token
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("invalid token id {token:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("token id list must not be empty".to_string());
    }
    Ok(ids)
}

/// [`parse_token_ids`] with prompt-specific wording in the error strings.
pub fn parse_prompt_token_ids(raw: &str) -> Result<Vec<u32>, String> {
    let ids = raw
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.trim().is_empty())
        .map(|token| {
            token
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("invalid prompt token id {token:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("prompt token list must not be empty".to_string());
    }
    Ok(ids)
}

/// Parse `value` as a `usize` of at least 1.
pub fn parse_positive_usize(label: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{label} must be a positive integer, got {value:?}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be greater than zero"));
    }
    Ok(parsed)
}

/// Slice `post_norm_all: [1, seq, hidden]` to `[1, 1, hidden]` at `row`.
pub fn slice_hidden_row(
    post_norm_all: &mlx_sys::MlxArray,
    row: usize,
    hidden: usize,
) -> mlx_sys::MlxArray {
    let r = row as i32;
    let h = hidden as i32;
    mlx_sys::slice(post_norm_all, &[0, r, 0], &[1, r + 1, h], &[1, 1, 1], None)
}
