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

/// Resolve the MLX C++ include and library directories.
///
/// ax_shim.cpp and activation.cpp include MLX C++ headers (`mlx/fast.h`,
/// `mlx/ops.h`, …) and link against `libmlx`. The C wrapper layer (mlx-c /
/// libmlxc) is no longer needed — our shim replaces it.
///
/// Priority:
///   1. `MLX_LIB_DIR` env var (CI wheel builds)
///   2. `brew --prefix mlx` for the C++ library
///   3. General Homebrew prefix fallback
struct MlxDirs {
    mlx_include_dir: String,
    mlx_lib_dir: String,
}

fn find_mlx_dirs() -> MlxDirs {
    // --- Priority 1: MLX_LIB_DIR env var ---
    if let Ok(lib_dir) = std::env::var("MLX_LIB_DIR") {
        let include_dir = std::env::var("MLX_INCLUDE_DIR").unwrap_or_else(|_| {
            PathBuf::from(&lib_dir)
                .parent()
                .map(|p| p.join("include").display().to_string())
                .unwrap_or_else(|| format!("{lib_dir}/../include"))
        });
        return MlxDirs {
            mlx_include_dir: include_dir,
            mlx_lib_dir: lib_dir,
        };
    }

    // --- Priority 2: brew --prefix mlx ---
    let mlx_prefix = std::process::Command::new("brew")
        .args(["--prefix", "mlx"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(p) = mlx_prefix {
        return MlxDirs {
            mlx_include_dir: format!("{p}/include"),
            mlx_lib_dir: format!("{p}/lib"),
        };
    }

    // --- Priority 3: general Homebrew fallback ---
    let prefix = homebrew_prefix();
    MlxDirs {
        mlx_include_dir: format!("{prefix}/include"),
        mlx_lib_dir: format!("{prefix}/lib"),
    }
}

fn main() {
    let dirs = find_mlx_dirs();
    let native_dir = PathBuf::from("native");

    // --- Compile native C++ shims (ax_shim.cpp + activation.cpp) ---
    // Both files include MLX C++ headers and call mlx::core directly.
    // ax_shim.cpp provides the C ABI entry points; activation.cpp provides
    // fused multi-op shims for performance-critical paths.
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .warnings(false)
        .include(&native_dir)
        .include(&dirs.mlx_include_dir)
        .file("native/ax_shim.cpp")
        .file("native/activation.cpp")
        .compile("ax_shim");

    // --- Link MLX C++ library ---
    println!("cargo:rustc-link-lib=mlx");
    println!("cargo:rustc-link-search=native={}", dirs.mlx_lib_dir);

    // --- Rerun triggers ---
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native/ax_shim.h");
    println!("cargo:rerun-if-changed=native/ax_shim.cpp");
    println!("cargo:rerun-if-changed=native/activation.cpp");
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MLX_INCLUDE_DIR");

    // --- Generate bindgen FFI bindings from ax_shim.h ---
    let bindings = bindgen::Builder::default()
        .header("native/ax_shim.h")
        .clang_arg(format!("-I{}", native_dir.display()))
        .allowlist_file(r".*native/ax_shim\.h")
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
        .expect("Unable to generate ax_shim bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
