use std::path::PathBuf;
use std::process;

fn main() {
    let mut model_dir = None;
    let mut force = false;
    let mut validate = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--force" => force = true,
            "--validate" => validate = true,
            "--help" | "-h" => {
                print_usage();
                return;
            }
            _ if model_dir.is_none() && !arg.starts_with('-') => {
                model_dir = Some(PathBuf::from(arg))
            }
            _ => {
                eprintln!("unexpected argument: {arg}");
                print_usage();
                process::exit(1);
            }
        }
    }
    let Some(model_dir) = model_dir else {
        print_usage();
        process::exit(1);
    };

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

fn print_usage() {
    eprintln!("Usage: generate-manifest [--force] [--validate] <model-dir>");
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
