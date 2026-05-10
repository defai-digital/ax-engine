use std::path::PathBuf;

fn homebrew_prefix() -> String {
    std::process::Command::new("brew")
        .arg("--prefix")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "/opt/homebrew".to_string())
}

fn main() {
    let prefix = homebrew_prefix();
    let lib_dir = format!("{prefix}/lib");
    let include_dir = format!("{prefix}/include");

    println!("cargo:rustc-link-lib=mlxc");
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={include_dir}/mlx/c/mlx.h");
    println!("cargo:rerun-if-changed={include_dir}/mlx/c");
    println!("cargo:rerun-if-env-changed=HOMEBREW_PREFIX");

    let mlx_c_header_pattern = format!(r".*{}/mlx/c/.*\.h", regex::escape(&include_dir));

    let bindings = bindgen::Builder::default()
        .header(format!("{include_dir}/mlx/c/mlx.h"))
        .clang_arg(format!("-I{include_dir}"))
        // Only generate bindings for mlx/c headers
        .allowlist_file(mlx_c_header_pattern)
        // Represent C enums as Rust enums
        .rustified_enum("mlx_dtype_")
        // Don't derive Default for opaque types that need explicit init
        .no_default("mlx_array_")
        .no_default("mlx_stream_")
        .no_default("mlx_vector_array_")
        .no_default("mlx_vector_string_")
        .no_default("mlx_map_string_to_array_")
        .no_default("mlx_fast_metal_kernel_")
        .no_default("mlx_fast_metal_kernel_config_")
        .generate()
        .expect("Unable to generate mlx-c bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
