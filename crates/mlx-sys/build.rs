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

    // --- Priority 2: active Python venv / site-packages mlx (matches mlx-lm) ---
    // Prefer the pip wheel when present so native AX and mlx-lm share one MLX
    // build. Homebrew bottles have lagged the wheel's Metal backend quality.
    if let Some(dirs) = find_pip_mlx_dirs() {
        return dirs;
    }

    // --- Priority 3: brew --prefix mlx ---
    // `brew --prefix mlx` succeeds and prints the opt path even when the
    // formula is NOT installed, so validate the directories before trusting
    // them; otherwise fall through to the generic prefix.
    let mlx_prefix = std::process::Command::new("brew")
        .args(["--prefix", "mlx"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(p) = mlx_prefix {
        let dirs = MlxDirs {
            mlx_include_dir: format!("{p}/include"),
            mlx_lib_dir: format!("{p}/lib"),
        };
        if mlx_dirs_valid(&dirs) {
            return dirs;
        }
    }

    // --- Priority 4: general Homebrew fallback ---
    let prefix = homebrew_prefix();
    let dirs = MlxDirs {
        mlx_include_dir: format!("{prefix}/include"),
        mlx_lib_dir: format!("{prefix}/lib"),
    };
    if !mlx_dirs_valid(&dirs) {
        panic!(
            "No usable MLX installation found: checked MLX_LIB_DIR, the active \
             Python's mlx package, `brew --prefix mlx`, and {prefix}. Install \
             MLX (`brew install mlx` or `pip install mlx`) or point \
             MLX_LIB_DIR/MLX_INCLUDE_DIR at an MLX build."
        );
    }
    dirs
}

/// A candidate is usable only if it has both the C++ headers and the dylib.
fn mlx_dirs_valid(dirs: &MlxDirs) -> bool {
    PathBuf::from(&dirs.mlx_lib_dir)
        .join("libmlx.dylib")
        .is_file()
        && PathBuf::from(&dirs.mlx_include_dir).join("mlx").is_dir()
}

/// Locate a pip-installed `mlx` package that ships `lib/libmlx.dylib` + headers.
fn find_pip_mlx_dirs() -> Option<MlxDirs> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        candidates.push(PathBuf::from(venv).join("lib"));
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        candidates.push(PathBuf::from(prefix).join("lib"));
    }
    // Prefer the Python that is building us (maturin/pyo3 set PYO3_PYTHON /
    // PYTHON). Falling back to bare `python3` can miss the active venv.
    let python = ["PYO3_PYTHON", "PYTHON", "PYTHON_SYS_EXECUTABLE"]
        .iter()
        .find_map(std::env::var_os)
        .unwrap_or_else(|| "python3".into());
    if let Ok(out) = std::process::Command::new(&python)
        .args([
            "-c",
            "import mlx, pathlib; print(pathlib.Path(list(mlx.__path__)[0]))",
        ])
        .output()
        && out.status.success()
        && let Ok(s) = String::from_utf8(out.stdout)
    {
        let p = PathBuf::from(s.trim());
        if p.is_dir() {
            candidates.push(p);
        }
    }

    for base in candidates {
        // base may be .../site-packages/mlx or .../lib
        let mlx_root = if base.join("lib/libmlx.dylib").is_file() {
            base.clone()
        } else if base
            .join("python3.14/site-packages/mlx/lib/libmlx.dylib")
            .is_file()
        {
            base.join("python3.14/site-packages/mlx")
        } else if base.join("site-packages/mlx/lib/libmlx.dylib").is_file() {
            base.join("site-packages/mlx")
        } else {
            // search one level of python* site-packages
            let mut found = None;
            if let Ok(entries) = std::fs::read_dir(&base) {
                for ent in entries.flatten() {
                    let sp = ent.path().join("site-packages/mlx");
                    if sp.join("lib/libmlx.dylib").is_file() {
                        found = Some(sp);
                        break;
                    }
                }
            }
            match found {
                Some(p) => p,
                None => continue,
            }
        };
        let lib_dir = mlx_root.join("lib");
        let include_dir = mlx_root.join("include");
        if lib_dir.join("libmlx.dylib").is_file() && include_dir.join("mlx").is_dir() {
            return Some(MlxDirs {
                mlx_include_dir: include_dir.display().to_string(),
                mlx_lib_dir: lib_dir.display().to_string(),
            });
        }
    }
    None
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
    // Embed an absolute rpath so the extension loads the same libmlx we
    // compiled against. Without this, `@rpath/libmlx.dylib` often resolves to
    // a stale Homebrew bottle that can be ~3× slower than the pip/wheel MLX
    // build used by mlx-lm (observed on 0.32.0: brew 15 MB dylib vs pip 22 MB).
    // headerpad leaves room for install_name_tool rewrites if needed later.
    println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dirs.mlx_lib_dir);
    // `rustc-link-arg` only reaches this package's own targets. Downstream
    // packages that produce mlx-linking binaries or test executables must
    // embed the rpath themselves; export the resolved directory as
    // DEP_MLX_LIB_DIR (via the `links = "mlx"` key) so they stay in lockstep
    // with the library actually linked here.
    println!("cargo:lib_dir={}", dirs.mlx_lib_dir);

    // --- Rerun triggers ---
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native/ax_shim.h");
    println!("cargo:rerun-if-changed=native/ax_shim_internal.h");
    println!("cargo:rerun-if-changed=native/ax_shim.cpp");
    println!("cargo:rerun-if-changed=native/activation.cpp");
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MLX_INCLUDE_DIR");
    // A Homebrew `mlx` upgrade swaps the dylib and headers under the
    // version-stable `/opt/homebrew/opt/mlx` symlink without touching any
    // tracked file, which would leave a stale shim object running against a
    // new C++ ABI. Track the MLX version header so the shim recompiles when
    // the installed MLX changes. (Guarded: registering a missing path would
    // force a rebuild on every run.)
    let mlx_version_header = PathBuf::from(&dirs.mlx_include_dir).join("mlx/version.h");
    if mlx_version_header.exists() {
        println!("cargo:rerun-if-changed={}", mlx_version_header.display());
    }

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
