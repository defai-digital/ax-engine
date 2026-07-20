use std::path::{Component, Path};

/// Return whether an MLX library directory resolves to a Homebrew formula.
///
/// Check both the spelling supplied by the caller and the canonical path. The
/// latter matters for Homebrew's stable `opt/mlx` symlink, and also prevents an
/// arbitrary symlink from disguising a Cellar directory as an explicit build.
pub(crate) fn is_homebrew_mlx_path(path: &Path, brew_mlx_prefix: Option<&Path>) -> bool {
    let mut candidates = vec![path.to_path_buf()];
    if let Ok(canonical) = path.canonicalize()
        && canonical != path
    {
        candidates.push(canonical);
    }
    let dylib = path.join("libmlx.dylib");
    if let Ok(canonical) = dylib.canonicalize()
        && canonical != dylib
    {
        candidates.push(canonical);
    }

    let mut prefixes = Vec::new();
    if let Some(prefix) = brew_mlx_prefix {
        prefixes.push(prefix.to_path_buf());
        if let Ok(canonical) = prefix.canonicalize()
            && canonical != prefix
        {
            prefixes.push(canonical);
        }
    }

    candidates.iter().any(|candidate| {
        has_component_pair(candidate, "Cellar", "mlx")
            || has_component_pair(candidate, "opt", "mlx")
            || prefixes.iter().any(|prefix| candidate.starts_with(prefix))
    })
}

fn has_component_pair(path: &Path, first: &str, second: &str) -> bool {
    let components = normal_components(path).collect::<Vec<_>>();
    components
        .windows(2)
        .any(|pair| pair[0] == first && pair[1] == second)
}

fn normal_components(path: &Path) -> impl Iterator<Item = &str> {
    path.components().filter_map(|component| match component {
        Component::Normal(value) => value.to_str(),
        _ => None,
    })
}
