use std::path::PathBuf;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: generate-manifest <model-dir>");
        process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let manifest_path = model_dir.join(ax_engine_core::model::AX_NATIVE_MODEL_MANIFEST_FILE);

    if manifest_path.exists() {
        eprintln!("manifest already exists: {}", manifest_path.display());
        process::exit(0);
    }

    match ax_engine_core::convert::convert_hf_model_dir(&model_dir) {
        Ok(manifest) => match ax_engine_core::convert::write_manifest(&model_dir, &manifest) {
            Ok(()) => {
                println!("wrote {}", manifest_path.display());
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
