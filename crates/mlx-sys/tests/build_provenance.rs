#[path = "../build_provenance.rs"]
mod build_provenance;

use std::path::Path;

use build_provenance::is_homebrew_mlx_path;

#[test]
fn explicit_homebrew_paths_are_detected() {
    assert!(is_homebrew_mlx_path(
        Path::new("/opt/homebrew/Cellar/mlx/0.32.0/lib"),
        None,
    ));
    assert!(is_homebrew_mlx_path(
        Path::new("/opt/homebrew/opt/mlx/lib"),
        None,
    ));
    assert!(is_homebrew_mlx_path(
        Path::new("/usr/local/Cellar/mlx/0.32.0/lib"),
        None,
    ));
}

#[test]
fn pip_wheel_path_is_not_homebrew() {
    assert!(!is_homebrew_mlx_path(
        Path::new("/Users/dev/.venv/lib/python3.12/site-packages/mlx/lib"),
        None,
    ));
}

#[test]
fn detected_formula_prefix_is_rejected_outside_conventional_roots() {
    assert!(is_homebrew_mlx_path(
        Path::new("/custom/packages/mlx/lib"),
        Some(Path::new("/custom/packages/mlx")),
    ));
}

#[cfg(unix)]
#[test]
fn symlink_cannot_disguise_a_homebrew_cellar_path() -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = std::env::temp_dir().join(format!(
        "ax-mlx-build-provenance-{}-{nonce}",
        std::process::id()
    ));
    let cellar_lib = root.join("Cellar/mlx/0.32.0/lib");
    std::fs::create_dir_all(&cellar_lib)?;
    let disguised = root.join("explicit-mlx-lib");
    symlink(&cellar_lib, &disguised)?;

    assert!(is_homebrew_mlx_path(&disguised, None));

    let custom_lib = root.join("custom/lib");
    std::fs::create_dir_all(&custom_lib)?;
    let cellar_dylib = cellar_lib.join("libmlx.dylib");
    std::fs::write(&cellar_dylib, [])?;
    symlink(&cellar_dylib, custom_lib.join("libmlx.dylib"))?;

    assert!(is_homebrew_mlx_path(&custom_lib, None));

    std::fs::remove_dir_all(&root)?;
    Ok(())
}
