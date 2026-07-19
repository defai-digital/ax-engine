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

/// Where the resolved MLX build came from — drives the provenance gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MlxProvenance {
    /// `MLX_LIB_DIR` env var: an explicit operator/CI choice.
    Explicit,
    /// A pip-installed wheel (repo `.venv`, active venv, or python3's
    /// site-packages) — the only supported source on macOS 26.x hosts.
    PipWheel,
    /// Homebrew formula. Refused by default: its build derives the
    /// deployment target from `MacOS.version.major.minor`, which truncates
    /// to 26.0 on every macOS 26.x host, silently compiling out the NAX
    /// GEMM/attention kernels (~3-4x slower prefill, no build error).
    Homebrew,
}

fn find_mlx_dirs() -> (MlxDirs, MlxProvenance) {
    // --- Priority 1: MLX_LIB_DIR env var ---
    if let Ok(lib_dir) = std::env::var("MLX_LIB_DIR") {
        let include_dir = std::env::var("MLX_INCLUDE_DIR").unwrap_or_else(|_| {
            PathBuf::from(&lib_dir)
                .parent()
                .map(|p| p.join("include").display().to_string())
                .unwrap_or_else(|| format!("{lib_dir}/../include"))
        });
        return (
            MlxDirs {
                mlx_include_dir: include_dir,
                mlx_lib_dir: lib_dir,
            },
            MlxProvenance::Explicit,
        );
    }

    // --- Priority 2: pip-installed mlx (repo .venv, active venv, python3) ---
    // Prefer the pip wheel when present so native AX and mlx-lm share one MLX
    // build. Homebrew bottles have lagged the wheel's Metal backend quality.
    if let Some(dirs) = find_pip_mlx_dirs() {
        return (dirs, MlxProvenance::PipWheel);
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
            return (dirs, MlxProvenance::Homebrew);
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
            "No usable MLX installation found: checked MLX_LIB_DIR, the repo \
             .venv, the active Python's mlx package, `brew --prefix mlx`, and \
             {prefix}. Install MLX with `pip install mlx=={}` (preferred — \
             Homebrew's formula builds without NAX acceleration on macOS 26.x \
             hosts) or point MLX_LIB_DIR/MLX_INCLUDE_DIR at an MLX build.",
            pinned_mlx_version().unwrap_or_else(|| "<see mlx.version>".to_string())
        );
    }
    (dirs, MlxProvenance::Homebrew)
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

    // An explicitly selected build Python must win over ambient virtualenv or
    // Conda paths. Studio uses PYO3_PYTHON to keep runtime preparation and the
    // Rust link target on the same pinned MLX wheel.
    let explicit_python_root = ["PYO3_PYTHON", "PYTHON", "PYTHON_SYS_EXECUTABLE"]
        .iter()
        .filter_map(std::env::var_os)
        .find_map(|python| mlx_root_from_python(&python));
    if let Some(root) = &explicit_python_root {
        candidates.push(root.clone());
    }

    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        candidates.push(PathBuf::from(venv).join("lib"));
    }
    // The repo-local `.venv` is the canonical dev environment: consult it
    // even when the shell forgot to activate it, so a bare `cargo build`
    // cannot silently drift to another MLX install on dev machines.
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let repo_venv = PathBuf::from(manifest_dir).join("../../.venv/lib");
        if repo_venv.is_dir() {
            candidates.push(repo_venv);
        }
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        candidates.push(PathBuf::from(prefix).join("lib"));
    }

    // Fall back to the shell's python3 only when none of the explicit Python
    // selectors resolved a usable MLX package.
    if explicit_python_root.is_none() {
        if let Some(root) = mlx_root_from_python(std::ffi::OsStr::new("python3")) {
            candidates.push(root);
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

fn mlx_root_from_python(python: &std::ffi::OsStr) -> Option<PathBuf> {
    let out = std::process::Command::new(python)
        .args([
            "-c",
            "import mlx, pathlib; print(pathlib.Path(list(mlx.__path__)[0]))",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let root = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    root.is_dir().then_some(root)
}

/// The admitted MLX version, pinned in `mlx.version` at the repo root.
/// Bumping it is a deliberate act: rerun the qmm microbench parity gate and
/// the bit-exactness suites first (see docs/GETTING-STARTED.md).
fn pinned_mlx_version() -> Option<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let pin_path = PathBuf::from(manifest_dir).join("../../mlx.version");
    std::fs::read_to_string(pin_path)
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|pin| !pin.is_empty())
}

/// Parse `MLX_VERSION_MAJOR/MINOR/PATCH` out of `mlx/version.h`.
fn resolved_mlx_version(include_dir: &str) -> Option<String> {
    let header = PathBuf::from(include_dir).join("mlx/version.h");
    let text = std::fs::read_to_string(header).ok()?;
    let field = |name: &str| -> Option<u32> {
        text.lines()
            .find(|line| line.contains(name))
            .and_then(|line| line.split_whitespace().last())
            .and_then(|value| value.trim().parse::<u32>().ok())
    };
    Some(format!(
        "{}.{}.{}",
        field("MLX_VERSION_MAJOR")?,
        field("MLX_VERSION_MINOR")?,
        field("MLX_VERSION_PATCH")?
    ))
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.trim(), "1" | "true" | "on" | "yes"))
        .unwrap_or(false)
}

/// Fail the build (loudly, with remediation) when the resolved MLX is a
/// source or version the repo has not admitted. Deployment correctness
/// depends on this: a Homebrew fallback or a version drift changes kernels
/// silently and invalidates every certified benchmark and bit-exactness gate.
fn enforce_mlx_provenance_and_version(dirs: &MlxDirs, provenance: MlxProvenance) {
    let version = resolved_mlx_version(&dirs.mlx_include_dir);
    let pin = pinned_mlx_version();
    println!(
        "cargo:warning=mlx-sys: linking MLX {} from {} ({:?})",
        version.as_deref().unwrap_or("<unknown version>"),
        dirs.mlx_lib_dir,
        provenance
    );

    if provenance == MlxProvenance::Homebrew {
        if env_flag("AX_MLX_ALLOW_HOMEBREW") {
            println!(
                "cargo:warning=mlx-sys: AX_MLX_ALLOW_HOMEBREW=1 — linking Homebrew MLX; NAX \
                 kernels may be silently disabled and certified benchmarks do not apply"
            );
        } else {
            panic!(
                "mlx-sys resolved MLX to the Homebrew formula at {} — refusing to link it. \
                 Homebrew's build derives its deployment target from the macOS minor version, \
                 which truncates to 26.0 on macOS 26.x hosts and silently compiles out the NAX \
                 GEMM/attention kernels. Use the pip wheel instead: activate the repo .venv or \
                 `python3 -m pip install mlx=={}`. Set AX_MLX_ALLOW_HOMEBREW=1 only for \
                 bring-up experiments.",
                dirs.mlx_lib_dir,
                pin.as_deref().unwrap_or("<see mlx.version>")
            );
        }
    }

    if let (Some(version), Some(pin)) = (version.as_deref(), pin.as_deref())
        && version != pin
    {
        if env_flag("AX_MLX_VERSION_OVERRIDE") {
            println!(
                "cargo:warning=mlx-sys: AX_MLX_VERSION_OVERRIDE=1 — MLX {version} does not \
                 match the admitted pin {pin}; benchmarks and bit-exactness gates were \
                 certified against the pin"
            );
        } else {
            panic!(
                "mlx-sys resolved MLX {version} from {} but the repo pins {pin} \
                 (mlx.version). Install the pinned wheel (`python3 -m pip install \
                 mlx=={pin}`), or — to deliberately test a different MLX — set \
                 AX_MLX_VERSION_OVERRIDE=1 and rerun the qmm microbench parity gate and \
                 bit-exactness suites before trusting any results.",
                dirs.mlx_lib_dir
            );
        }
    }
}

fn main() {
    let (dirs, provenance) = find_mlx_dirs();
    enforce_mlx_provenance_and_version(&dirs, provenance);
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
    // Environment switches that change which MLX gets resolved (or whether
    // the provenance/version gate applies) must re-trigger resolution —
    // otherwise a stale link survives an env change.
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=AX_MLX_ALLOW_HOMEBREW");
    println!("cargo:rerun-if-env-changed=AX_MLX_VERSION_OVERRIDE");
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let pin = PathBuf::from(manifest_dir).join("../../mlx.version");
        if pin.exists() {
            println!("cargo:rerun-if-changed={}", pin.display());
        }
    }
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
