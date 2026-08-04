use std::path::PathBuf;
use std::process;

#[derive(Debug, Eq, PartialEq)]
struct GenerateManifestArgs {
    model_dir: PathBuf,
    force: bool,
    validate: bool,
}

fn main() {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    let args = match parse_generate_manifest_args(&raw_args) {
        Ok(Some(args)) => args,
        Ok(None) => {
            print_usage();
            return;
        }
        Err(error) => {
            eprintln!("{error}");
            print_usage();
            process::exit(1);
        }
    };
    let GenerateManifestArgs {
        model_dir,
        force,
        validate,
    } = args;

    let manifest_path = model_dir.join(ax_engine_core::model::AX_NATIVE_MODEL_MANIFEST_FILE);

    if manifest_path.exists() && !force {
        eprintln!("manifest already exists: {}", manifest_path.display());
        if validate {
            validate_manifest(&model_dir);
        }
        process::exit(0);
    }

    match ax_engine_core::convert::convert_hf_model_dir(&model_dir) {
        Ok(manifest) => match ax_engine_core::convert::write_manifest(&model_dir, &manifest) {
            Ok(()) => {
                println!("wrote {}", manifest_path.display());
                if validate {
                    validate_manifest(&model_dir);
                }
            }
            Err(e) => {
                eprintln!("error writing manifest: {e}");
                process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("error converting model: {e}");
            process::exit(1);
        }
    }
}

fn parse_generate_manifest_args(args: &[String]) -> Result<Option<GenerateManifestArgs>, String> {
    let mut model_dir = None;
    let mut force = false;
    let mut validate = false;
    let mut options_ended = false;
    for arg in args {
        if !options_ended && arg == "--" {
            options_ended = true;
            continue;
        }
        match arg.as_str() {
            "--force" if !options_ended => force = true,
            "--validate" if !options_ended => validate = true,
            "--help" | "-h" if !options_ended => return Ok(None),
            _ if model_dir.is_none() && (options_ended || !arg.starts_with('-')) => {
                model_dir = Some(PathBuf::from(arg));
            }
            _ => return Err(format!("unexpected argument: {arg}")),
        }
    }
    let model_dir = model_dir.ok_or_else(|| "missing model directory".to_string())?;
    Ok(Some(GenerateManifestArgs {
        model_dir,
        force,
        validate,
    }))
}

fn print_usage() {
    eprintln!(
        "Usage: generate-manifest [--force] [--validate] <model-dir>\n\
                generate-manifest [--force] [--validate] -- <model-dir>"
    );
}

fn validate_manifest(model_dir: &std::path::Path) {
    match ax_engine_core::model::NativeModelArtifacts::from_dir(model_dir) {
        Ok(_) => println!("validated {}", model_dir.display()),
        Err(error) => {
            eprintln!("error validating manifest: {error}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<GenerateManifestArgs>, String> {
        parse_generate_manifest_args(
            &args
                .iter()
                .map(|arg| (*arg).to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn accepts_dash_prefixed_model_dir_after_end_of_options() {
        let parsed = parse(&["--force", "--", "-model"])
            .expect("arguments should parse")
            .expect("arguments should request generation");

        assert_eq!(
            parsed,
            GenerateManifestArgs {
                model_dir: PathBuf::from("-model"),
                force: true,
                validate: false,
            }
        );
    }

    #[test]
    fn treats_option_like_value_as_model_dir_after_end_of_options() {
        let parsed = parse(&["--", "--validate"])
            .expect("arguments should parse")
            .expect("arguments should request generation");

        assert_eq!(parsed.model_dir, PathBuf::from("--validate"));
        assert!(!parsed.validate);
    }

    #[test]
    fn rejects_dash_prefixed_model_dir_without_end_of_options() {
        let error = parse(&["-model"]).expect_err("unknown option should be rejected");

        assert_eq!(error, "unexpected argument: -model");
    }
}
