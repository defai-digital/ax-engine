use std::path::PathBuf;

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    let python_lib_dir = pyo3_build_config::get().lib_dir().map(str::to_owned);

    if std::env::var_os("CARGO_FEATURE_PYTHON_EXTENSION").is_some() {
        pyo3_build_config::add_extension_module_link_args();
    }

    // Leave room for install_name_tool and absolute MLX rpaths on the cdylib.
    println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");

    if let Some(lib_dir) = python_lib_dir.as_deref() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }

    // Embed the same libmlx rpath the extension was built against so we do not
    // accidentally load a slower Homebrew bottle at runtime (see mlx-sys/build.rs).
    let mlx_lib = find_mlx_lib_dir();
    if let Some(mlx_lib) = mlx_lib.as_deref() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx_lib}");
        println!("cargo:rustc-link-search=native={mlx_lib}");
    }

    // The primary rpath can dangle: scripts/check-python-preview.sh runs
    // `maturin develop` inside an ephemeral venv that is deleted after the
    // check, while the built extension stays in python/ax_engine/ for the
    // editable install. Embed the repo .venv MLX as a fallback rpath so the
    // extension keeps loading on dev machines after that venv is gone.
    if let Some(fallback) = repo_venv_mlx_lib_dir()
        && Some(fallback.as_str()) != mlx_lib.as_deref()
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{fallback}");
    }
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DEP_MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
}

/// The repo `.venv`'s pip-installed MLX lib dir, if present (canonicalized so
/// the embedded rpath is a clean absolute path rather than `../../.venv/...`).
fn repo_venv_mlx_lib_dir() -> Option<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let venv_lib = PathBuf::from(manifest_dir).join("../../.venv/lib");
    for entry in std::fs::read_dir(venv_lib).ok()?.flatten() {
        let lib = entry.path().join("site-packages/mlx/lib");
        if lib.join("libmlx.dylib").is_file()
            && let Ok(lib) = lib.canonicalize()
        {
            return Some(lib.display().to_string());
        }
    }
    None
}

fn find_mlx_lib_dir() -> Option<String> {
    // mlx-sys is a direct dependency specifically so its `links = "mlx"`
    // metadata reaches this build script. Reuse that exact directory instead
    // of resolving MLX independently through the Python selected by pyo3:
    // PYO3_PYTHON may point at Conda while mlx-sys correctly selects the
    // repository .venv, which previously left test binaries with no usable
    // libmlx LC_RPATH.
    if let Ok(lib_dir) = std::env::var("DEP_MLX_LIB_DIR") {
        let p = PathBuf::from(&lib_dir);
        if p.join("libmlx.dylib").is_file() || p.join("libmlx.so").is_file() {
            return Some(lib_dir);
        }
    }

    if let Ok(lib_dir) = std::env::var("MLX_LIB_DIR") {
        let p = PathBuf::from(&lib_dir);
        if p.join("libmlx.dylib").is_file() || p.join("libmlx.so").is_file() {
            return Some(lib_dir);
        }
    }

    let python = ["PYO3_PYTHON", "PYTHON", "PYTHON_SYS_EXECUTABLE"]
        .iter()
        .find_map(std::env::var_os)
        .unwrap_or_else(|| "python3".into());
    let out = std::process::Command::new(python)
        .args([
            "-c",
            "import mlx, pathlib; print(pathlib.Path(list(mlx.__path__)[0]) / 'lib')",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let lib = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    if lib.join("libmlx.dylib").is_file() || lib.join("libmlx.so").is_file() {
        Some(lib.display().to_string())
    } else {
        None
    }
}
