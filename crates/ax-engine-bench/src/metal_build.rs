use std::env;
use std::path::PathBuf;

use ax_engine_core::MetalBuildStatus;

use crate::cli::usage;
use crate::error::CliError;
use crate::path_utils::absolutize_path;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MetalBuildArgs {
    pub(crate) manifest_path: PathBuf,
    pub(crate) output_dir: PathBuf,
}

pub(crate) fn parse_metal_build_args(args: &[String]) -> Result<MetalBuildArgs, CliError> {
    let current_dir = env::current_dir().map_err(|error| {
        CliError::Runtime(format!(
            "failed to resolve current working directory: {error}"
        ))
    })?;
    let mut manifest_path = env::var_os("AX_METAL_MANIFEST_PATH").map(PathBuf::from);
    let mut output_dir = env::var_os("AX_METAL_OUTPUT_DIR").map(PathBuf::from);

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--manifest" => {
                let Some(value) = iter.next() else {
                    return Err(CliError::Usage(
                        "missing value for required flag --manifest".to_string(),
                    ));
                };
                manifest_path = Some(PathBuf::from(value));
            }
            "--output-dir" => {
                let Some(value) = iter.next() else {
                    return Err(CliError::Usage(
                        "missing value for required flag --output-dir".to_string(),
                    ));
                };
                output_dir = Some(PathBuf::from(value));
            }
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag for metal-build: {other}\n\n{}",
                    usage()
                )));
            }
        }
    }

    let manifest_path = absolutize_path(
        manifest_path.unwrap_or_else(|| current_dir.join("metal/phase1-kernels.json")),
        &current_dir,
    );
    let output_dir = absolutize_path(
        output_dir.unwrap_or_else(|| current_dir.join("build/metal")),
        &current_dir,
    );

    Ok(MetalBuildArgs {
        manifest_path,
        output_dir,
    })
}

pub(crate) fn metal_build_status_label(status: MetalBuildStatus) -> &'static str {
    match status {
        MetalBuildStatus::Unknown => "unknown",
        MetalBuildStatus::Compiled => "compiled",
        MetalBuildStatus::SkippedToolchainUnavailable => "skipped_toolchain_unavailable",
        MetalBuildStatus::SkippedNotReady => "skipped_not_ready",
        MetalBuildStatus::FailedCompile => "failed_compile",
    }
}

pub(crate) fn map_metal_build_error(error: ax_engine_core::MetalRuntimeError) -> CliError {
    match error {
        ax_engine_core::MetalRuntimeError::InvalidManifest { .. }
        | ax_engine_core::MetalRuntimeError::InvalidBuildReport { .. }
        | ax_engine_core::MetalRuntimeError::MissingBuildArtifact { .. } => {
            CliError::Contract(error.to_string())
        }
        _ => CliError::Runtime(error.to_string()),
    }
}
