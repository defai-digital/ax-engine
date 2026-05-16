use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::cli::generate_manifest_usage;
use crate::error::CliError;

pub(crate) const GENERATE_MANIFEST_SCHEMA_VERSION: &str = "ax.generate_manifest.v1";

pub(crate) fn handle_generate_manifest(args: &[String]) -> Result<(), CliError> {
    let args = parse_generate_manifest_args(args)?;
    let model_dir = args.model_dir;
    if !model_dir.is_dir() {
        return Err(CliError::Runtime(format!(
            "model directory not found: {}",
            model_dir.display()
        )));
    }
    let manifest_path = model_dir.join(ax_engine_core::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let status = if manifest_path.exists() {
        GenerateManifestStatus::AlreadyExists
    } else {
        let manifest = ax_engine_core::convert::convert_hf_model_dir(&model_dir)
            .map_err(|e| CliError::Runtime(format!("error converting model: {e}")))?;
        ax_engine_core::convert::write_manifest(&model_dir, &manifest)
            .map_err(|e| CliError::Runtime(format!("error writing manifest: {e}")))?;
        GenerateManifestStatus::Written
    };
    let validation = if args.validate {
        validate_native_model_artifacts(&model_dir)?;
        Some(GenerateManifestValidationSummary { passed: true })
    } else {
        None
    };
    let summary = GenerateManifestSummary {
        schema_version: GENERATE_MANIFEST_SCHEMA_VERSION,
        model_dir: model_dir.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        status,
        manifest_present: manifest_path.exists(),
        validation,
    };

    if args.json {
        let json = serde_json::to_string_pretty(&summary).map_err(|error| {
            CliError::Runtime(format!(
                "failed to serialize generate-manifest summary: {error}"
            ))
        })?;
        println!("{json}");
    } else if status == GenerateManifestStatus::AlreadyExists {
        println!("manifest already exists: {}", manifest_path.display());
        if args.validate {
            println!("validated {}", manifest_path.display());
        }
    } else {
        println!("wrote {}", manifest_path.display());
        if args.validate {
            println!("validated {}", manifest_path.display());
        }
    }
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct GenerateManifestArgs {
    pub(crate) model_dir: PathBuf,
    pub(crate) json: bool,
    pub(crate) validate: bool,
}

pub(crate) fn parse_generate_manifest_args(
    args: &[String],
) -> Result<GenerateManifestArgs, CliError> {
    let mut model_dir = None;
    let mut json = false;
    let mut validate = false;

    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "--validate" => validate = true,
            value if value.starts_with('-') => {
                return Err(CliError::Usage(format!(
                    "unknown generate-manifest option: {value}\n\n{}",
                    generate_manifest_usage()
                )));
            }
            value => {
                if model_dir.is_some() {
                    return Err(CliError::Usage(format!(
                        "unexpected generate-manifest argument: {value}\n\n{}",
                        generate_manifest_usage()
                    )));
                }
                model_dir = Some(PathBuf::from(value));
            }
        }
    }

    let Some(model_dir) = model_dir else {
        return Err(CliError::Usage(generate_manifest_usage()));
    };

    Ok(GenerateManifestArgs {
        model_dir,
        json,
        validate,
    })
}

fn validate_native_model_artifacts(model_dir: &Path) -> Result<(), CliError> {
    ax_engine_core::model::NativeModelArtifacts::from_dir(model_dir)
        .map(|_| ())
        .map_err(|error| {
            CliError::Runtime(format!(
                "generated manifest validation failed for {}: {error}",
                model_dir.display()
            ))
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GenerateManifestStatus {
    AlreadyExists,
    Written,
}

#[derive(Debug, Serialize)]
pub(crate) struct GenerateManifestSummary {
    pub(crate) schema_version: &'static str,
    pub(crate) model_dir: String,
    pub(crate) manifest_path: String,
    pub(crate) status: GenerateManifestStatus,
    pub(crate) manifest_present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) validation: Option<GenerateManifestValidationSummary>,
}

#[derive(Debug, Serialize)]
pub(crate) struct GenerateManifestValidationSummary {
    pub(crate) passed: bool,
}
