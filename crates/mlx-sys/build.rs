use std::path::{Path, PathBuf};

fn homebrew_prefix() -> String {
    std::process::Command::new("brew")
        .arg("--prefix")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "/opt/homebrew".to_string())
}

/// Resolve include/lib directories for the MLX C wrapper and C++ core.
///
/// `activation.cpp` includes both mlx-c C wrapper headers (`mlx/c/*.h`) and the
/// underlying MLX C++ headers (`mlx/fast.h`, `mlx/ops.h`, …). We therefore need
/// two include roots: one for mlx-c and one for the MLX C++ library itself.
///
/// Priority:
///   1. Vendored `vendor/mlx-c/` (built from source via cmake)
///   2. `MLX_LIB_DIR` env var (CI wheel builds)
///   3. `brew --prefix mlx-c` for the wrapper + `brew --prefix mlx` for C++
///   4. General Homebrew prefix fallback
struct MlxDirs {
    mlxc_lib_dir: String,
    mlxc_include_dir: String,
    mlx_cpp_include_dir: String,
    mlx_lib_dir: String,
    vendored: bool,
}

fn find_mlx_dirs() -> MlxDirs {
    // --- Priority 1: vendored mlx-c ---
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendored_path = manifest_dir.join("../../vendor/mlx-c");
    if vendored_path.join("CMakeLists.txt").exists() {
        return build_vendored_mlx_c(&manifest_dir, &vendored_path);
    }

    // --- Priority 2: MLX_LIB_DIR env var ---
    if let Ok(lib_dir) = std::env::var("MLX_LIB_DIR") {
        let include_dir = std::env::var("MLX_INCLUDE_DIR").unwrap_or_else(|_| {
            PathBuf::from(&lib_dir)
                .parent()
                .map(|p| p.join("include").display().to_string())
                .unwrap_or_else(|| format!("{lib_dir}/../include"))
        });
        let cpp_include_dir =
            std::env::var("MLX_CPP_INCLUDE_DIR").unwrap_or_else(|_| include_dir.clone());
        return MlxDirs {
            mlxc_lib_dir: lib_dir.clone(),
            mlxc_include_dir: include_dir,
            mlx_cpp_include_dir: cpp_include_dir,
            mlx_lib_dir: lib_dir,
            vendored: false,
        };
    }

    // --- Priority 3: brew --prefix mlx-c ---
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
        let (cpp_include_dir, mlx_lib_dir) = mlx_prefix
            .map(|p| (format!("{p}/include"), format!("{p}/lib")))
            .unwrap_or_else(|| {
                let prefix = homebrew_prefix();
                (format!("{prefix}/include"), format!("{prefix}/lib"))
            });
        return MlxDirs {
            mlxc_lib_dir: lib_dir,
            mlxc_include_dir: include_dir,
            mlx_cpp_include_dir: cpp_include_dir,
            mlx_lib_dir,
            vendored: false,
        };
    }

    // --- Priority 4: general Homebrew fallback ---
    let prefix = homebrew_prefix();
    MlxDirs {
        mlxc_lib_dir: format!("{prefix}/lib"),
        mlxc_include_dir: format!("{prefix}/include"),
        mlx_cpp_include_dir: format!("{prefix}/include"),
        mlx_lib_dir: format!("{prefix}/lib"),
        vendored: false,
    }
}

/// Build mlx-c from the vendored source tree using cmake.
///
/// This produces `libmlxc.a` and gives us direct access to the vendored
/// headers for bindgen and the `activation.cpp` cc build. The underlying
/// MLX C++ library is resolved via the system (brew) — we do NOT vendor
/// the MLX C++ core itself.
fn build_vendored_mlx_c(_manifest_dir: &Path, vendored_path: &Path) -> MlxDirs {
    eprintln!(
        "mlx-sys: building vendored mlx-c from {}",
        vendored_path.display()
    );

    // Resolve MLX C++ include/lib from system (brew) for cmake's
    // `find_package(MLX)` or the FetchContent fallback.
    let mlx_prefix = std::process::Command::new("brew")
        .args(["--prefix", "mlx"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let (mlx_cpp_include_dir, mlx_lib_dir) = mlx_prefix
        .map(|p| (format!("{p}/include"), format!("{p}/lib")))
        .unwrap_or_else(|| {
            let prefix = homebrew_prefix();
            (format!("{prefix}/include"), format!("{prefix}/lib"))
        });

    // Build vendored mlx-c with cmake.
    // MLX_C_USE_SYSTEM_MLX=ON tells cmake to find the system-installed MLX
    // C++ library instead of fetching it from git.
    let mut cfg = cmake::Config::new(vendored_path);
    cfg.define("MLX_C_USE_SYSTEM_MLX", "ON")
        .define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        // Point cmake at the MLX C++ installation for find_package(MLX).
        .define("CMAKE_PREFIX_PATH", &mlx_lib_dir);

    // If find_package(MLX) fails, cmake will fall back to FetchContent which
    // clones MLX from git. To avoid that in CI, we also set MLX_DIR if the
    // brew install provides a cmake config.
    // Homebrew installs cmake configs under share/cmake/MLX/ (not lib/cmake/).
    let mlx_cmake_share = format!("{}/../../share/cmake/MLX", mlx_lib_dir);
    let mlx_cmake_share = Path::new(&mlx_cmake_share)
        .canonicalize()
        .map(|p| p.display().to_string())
        .unwrap_or(mlx_cmake_share);
    if Path::new(&mlx_cmake_share).exists() {
        cfg.define("MLX_DIR", &mlx_cmake_share);
    }

    let dst = cfg.build();

    // cmake outputs the static library under dst/lib/ (or dst/build/).
    let mlxc_lib_dir = format!("{}/lib", dst.display());
    let mlxc_include_dir = vendored_path.display().to_string();

    eprintln!("mlx-sys: vendored mlx-c built at {}", dst.display());
    eprintln!("mlx-sys: vendored include: {mlxc_include_dir}");
    eprintln!("mlx-sys: vendored lib: {mlxc_lib_dir}");

    MlxDirs {
        mlxc_lib_dir,
        mlxc_include_dir,
        mlx_cpp_include_dir,
        mlx_lib_dir,
        vendored: true,
    }
}

fn main() {
    let dirs = find_mlx_dirs();

    // --- Compile native C++ shims (activation.cpp, etc.) ---
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&dirs.mlxc_include_dir)
        .include(&dirs.mlx_cpp_include_dir)
        .file("native/activation.cpp")
        .compile("ax_mlx_direct");

    // --- Link libraries ---
    // When vendored, mlxc is built as a static lib in the cmake output dir.
    // When not vendored, we link against the system-installed shared lib.
    if dirs.vendored {
        // Static link against the vendored libmlxc.a
        println!("cargo:rustc-link-search=native={}", dirs.mlxc_lib_dir);
        println!("cargo:rustc-link-lib=static=mlxc");
    } else {
        println!("cargo:rustc-link-lib=mlxc");
        println!("cargo:rustc-link-search=native={}", dirs.mlxc_lib_dir);
    }
    println!("cargo:rustc-link-lib=mlx");
    println!("cargo:rustc-link-search=native={}", dirs.mlx_lib_dir);

    // --- Rerun triggers ---
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native/activation.cpp");
    println!(
        "cargo:rerun-if-changed={}/mlx/c/mlx.h",
        dirs.mlxc_include_dir
    );
    println!("cargo:rerun-if-changed={}/mlx/c", dirs.mlxc_include_dir);
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MLX_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MLX_CPP_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=HOMEBREW_PREFIX");

    // For vendored builds, also rerun when the vendored source changes.
    if dirs.vendored {
        let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let vendored_src = manifest_dir.join("../../vendor/mlx-c/mlx/c");
        if vendored_src.exists() {
            println!("cargo:rerun-if-changed={}", vendored_src.display());
        }
    }

    // --- Generate bindgen FFI bindings ---
    let mlx_c_header_pattern = format!(r".*{}/mlx/c/.*\.h", regex::escape(&dirs.mlxc_include_dir));

    let bindings = bindgen::Builder::default()
        .header(format!("{}/mlx/c/mlx.h", dirs.mlxc_include_dir))
        .clang_arg(format!("-I{}", dirs.mlxc_include_dir))
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
