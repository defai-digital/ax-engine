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

// Resolve (lib_dir, include_dir, mlx_cpp_include_dir) for MLX.
//
// activation.cpp includes both mlx-c C wrapper headers (mlx/c/*.h) and the
// underlying MLX C++ headers (mlx/fast.h, mlx/ops.h, …).  We therefore need
// two include roots: one for mlx-c and one for the MLX C++ library itself.
//
// Priority:
//   1. MLX_LIB_DIR env var (set in CI wheel builds to point at the mlx-c
//      library; MLX_INCLUDE_DIR can optionally override the header location;
//      MLX_CPP_INCLUDE_DIR can override the C++ header location)
//   2. `brew --prefix mlx-c` for the wrapper + `brew --prefix mlx` for C++
//   3. General Homebrew prefix — fallback for environments where both are
//      installed under the top-level Homebrew tree
// Returns (mlxc_lib_dir, mlxc_include_dir, mlx_cpp_include_dir, mlx_lib_dir).
// mlx_lib_dir may equal mlxc_lib_dir when both live under the same prefix.
fn find_mlx_dirs() -> (String, String, String, String) {
    if let Ok(lib_dir) = std::env::var("MLX_LIB_DIR") {
        let include_dir = std::env::var("MLX_INCLUDE_DIR").unwrap_or_else(|_| {
            PathBuf::from(&lib_dir)
                .parent()
                .map(|p| p.join("include").display().to_string())
                .unwrap_or_else(|| format!("{lib_dir}/../include"))
        });
        let cpp_include_dir =
            std::env::var("MLX_CPP_INCLUDE_DIR").unwrap_or_else(|_| include_dir.clone());
        return (lib_dir.clone(), include_dir, cpp_include_dir, lib_dir);
    }

    let mlxc_prefix = std::process::Command::new("brew")
        .args(["--prefix", "mlx-c"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let mlx_prefix = std::process::Command::new("brew")
        .args(["--prefix", "mlx"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(mlxc) = mlxc_prefix {
        let lib_dir = format!("{mlxc}/lib");
        let include_dir = format!("{mlxc}/include");
        // Use mlx C++ headers and lib from its own prefix if available,
        // otherwise fall back to the general Homebrew prefix.
        let (cpp_include_dir, mlx_lib_dir) = mlx_prefix
            .map(|p| (format!("{p}/include"), format!("{p}/lib")))
            .unwrap_or_else(|| {
                let prefix = homebrew_prefix();
                (format!("{prefix}/include"), format!("{prefix}/lib"))
            });
        return (lib_dir, include_dir, cpp_include_dir, mlx_lib_dir);
    }

    let prefix = homebrew_prefix();
    (
        format!("{prefix}/lib"),
        format!("{prefix}/include"),
        format!("{prefix}/include"),
        format!("{prefix}/lib"),
    )
}

fn main() {
    let (lib_dir, include_dir, cpp_include_dir, mlx_lib_dir) = find_mlx_dirs();

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&include_dir)
        .include(&cpp_include_dir)
        .file("native/activation.cpp")
        .compile("ax_mlx_direct");

    println!("cargo:rustc-link-lib=mlx");
    println!("cargo:rustc-link-lib=mlxc");
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-search=native={mlx_lib_dir}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native/activation.cpp");
    println!("cargo:rerun-if-changed={include_dir}/mlx/c/mlx.h");
    println!("cargo:rerun-if-changed={include_dir}/mlx/c");
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MLX_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MLX_CPP_INCLUDE_DIR");
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
