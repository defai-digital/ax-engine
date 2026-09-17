/// The admitted MLX runtime, pinned in `mlx.version` at the repo root.
///
/// Two formats are accepted:
///   - `0.32.2`               — a published wheel (install via pip);
///   - `git:<sha>@<version>`  — an admitted source build of upstream MLX at
///     `<sha>` whose `mlx/version.h` reports `<version>` (build recipe and
///     admission evidence: docs/performance/mlx-main-admission-2026-07-28.md).
///
/// Bumping either form is a deliberate act: rerun the qmm microbench parity
/// gate and the bit-exactness suites first (see docs/GETTING-STARTED.md).
pub(crate) enum MlxPin {
    Wheel(String),
    Source { commit: String, version: String },
}

impl MlxPin {
    pub(crate) fn expected_header_version(&self) -> &str {
        match self {
            Self::Wheel(version) => version,
            Self::Source { version, .. } => version,
        }
    }

    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Wheel(version) => version.clone(),
            Self::Source { commit, version } => format!("source build {commit} ({version})"),
        }
    }

    pub(crate) fn install_hint(&self) -> String {
        match self {
            Self::Wheel(version) => format!("python3 -m pip install mlx=={version}"),
            Self::Source { commit, .. } => format!(
                "build upstream MLX at {commit} per \
                 docs/performance/mlx-main-admission-2026-07-28.md and point \
                 MLX_LIB_DIR/MLX_INCLUDE_DIR at its install prefix"
            ),
        }
    }
}

pub(crate) fn parse_mlx_pin(raw: &str) -> Option<MlxPin> {
    let raw = raw.trim();
    fn valid_version(version: &str) -> bool {
        let fields: Vec<_> = version.split('.').collect();
        fields.len() == 3
            && fields.iter().all(|field| {
                !field.is_empty()
                    && field.bytes().all(|byte| byte.is_ascii_digit())
                    && field.parse::<u32>().is_ok()
            })
    }
    if let Some(rest) = raw.strip_prefix("git:") {
        let (commit, version) = rest.split_once('@')?;
        let (commit, version) = (commit.trim(), version.trim());
        if !(7..=40).contains(&commit.len())
            || !commit.bytes().all(|byte| byte.is_ascii_hexdigit())
            || !valid_version(version)
        {
            return None;
        }
        return Some(MlxPin::Source {
            commit: commit.to_string(),
            version: version.to_string(),
        });
    }
    valid_version(raw).then(|| MlxPin::Wheel(raw.to_string()))
}

pub(crate) fn read_mlx_pin(path: &std::path::Path) -> Result<MlxPin, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read required MLX pin {}: {error}", path.display()))?;
    parse_mlx_pin(&raw).ok_or_else(|| {
        format!(
            "invalid MLX pin {}: expected MAJOR.MINOR.PATCH or git:<hex commit>@MAJOR.MINOR.PATCH",
            path.display()
        )
    })
}
