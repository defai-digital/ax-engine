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
    if let Some(mlx_lib) = find_mlx_lib_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx_lib}");
        println!("cargo:rustc-link-search=native={mlx_lib}");
    }
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
}

fn find_mlx_lib_dir() -> Option<String> {
    if let Ok(lib_dir) = std::env::var("MLX_LIB_DIR") {
        let p = PathBuf::from(&lib_dir);
        if p.join("libmlx.dylib").is_file() || p.join("libmlx.so").is_file() {
            return Some(lib_dir);
        }
    }

    let python = ["PYO3_PYTHON", "PYTHON", "PYTHON_SYS_EXECUTABLE"]
        .iter()
        .find_map(|k| std::env::var_os(k))
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
