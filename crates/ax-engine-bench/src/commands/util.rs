use super::*;

/// Reject flags a subcommand does not define, and value-taking flags with a
/// missing value (or a value that is itself a flag). The pluck-style parsers
/// below never see a typo'd flag otherwise, so `--jsn` would silently change
/// behavior and `--model-id --json` would swallow `--json` as the model id.
pub(crate) fn reject_unknown_flags(
    command: &str,
    args: &[String],
    value_flags: &[&str],
    bool_flags: &[&str],
) -> Result<(), CliError> {
    let mut iter = args.iter();
    while let Some(token) = iter.next() {
        if !token.starts_with("--") {
            return Err(CliError::Usage(format!(
                "unexpected argument for {command}: {token}\n\n{}",
                usage()
            )));
        }
        if bool_flags.contains(&token.as_str()) {
            continue;
        }
        if value_flags.contains(&token.as_str()) {
            match iter.next() {
                Some(value) if !value.starts_with("--") => {}
                Some(value) => {
                    return Err(CliError::Usage(format!(
                        "{token} expects a value for {command}, found flag {value}"
                    )));
                }
                None => {
                    return Err(CliError::Usage(format!(
                        "{token} expects a value for {command}"
                    )));
                }
            }
            continue;
        }
        return Err(CliError::Usage(format!(
            "unknown flag for {command}: {token}\n\n{}",
            usage()
        )));
    }
    Ok(())
}

pub(crate) fn optional_named_flag(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(candidate) = iter.next() {
        if candidate == name {
            return iter.next().cloned();
        }
    }
    None
}

pub(crate) fn parse_optional_u32_flag(
    args: &[String],
    name: &str,
) -> Result<Option<u32>, CliError> {
    optional_named_flag(args, name)
        .map(|raw| {
            raw.parse::<u32>().map_err(|error| {
                CliError::Usage(format!("invalid value for {name}: {raw} ({error})"))
            })
        })
        .transpose()
}

pub(crate) fn parse_optional_u64_flag(
    args: &[String],
    name: &str,
) -> Result<Option<u64>, CliError> {
    optional_named_flag(args, name)
        .map(|raw| {
            raw.parse::<u64>().map_err(|error| {
                CliError::Usage(format!("invalid value for {name}: {raw} ({error})"))
            })
        })
        .transpose()
}

pub(crate) fn run_inference_generate(args: &InferenceArgs) -> Result<GenerateResponse, CliError> {
    let mut session = build_inference_session(args)?;
    session
        .generate(args.generate_request())
        .map_err(|error| CliError::Runtime(format!("generate request failed: {error}")))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn collect_inference_stream_events(
    args: &InferenceArgs,
) -> Result<Vec<GenerateStreamEvent>, CliError> {
    let mut session = build_inference_session(args)?;
    session
        .stream_generate(args.generate_request())
        .map_err(|error| CliError::Runtime(format!("stream request failed: {error}")))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| CliError::Runtime(format!("stream request failed: {error}")))
}
