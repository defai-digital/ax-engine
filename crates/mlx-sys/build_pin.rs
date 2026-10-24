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

/// `major.minor.patch` from `MLX_VERSION_MAJOR` / `MINOR` / `PATCH`.
///
/// A dev build keeps those three macros and additionally embeds
/// `major.minor.patch.devYYYYMMDD+<hex>` (the `mlx::core::version()` form
/// documented on `MLX_VERSION`). The suffix is not part of this numeric parse.
pub(crate) fn header_numeric_version(text: &str) -> Option<String> {
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

/// One dev-build identity: `numeric.devYYYYMMDD+commit`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DevMlxVersion {
    pub numeric: String,
    pub date: String,
    pub commit: String,
}

impl DevMlxVersion {
    pub(crate) fn token(&self) -> String {
        format!("{}.dev{}+{}", self.numeric, self.date, self.commit)
    }

    fn full_commit(&self) -> bool {
        self.commit.len() == 40
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DevHeader {
    Absent,
    One(DevMlxVersion),
    Conflict,
}

/// How a resolved MLX may be linked against the numeric pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlxLinkGate {
    /// Header is the pinned release. Experiment commit is ignored.
    PinnedRelease,
    /// Numeric release mismatch with no dev suffix. The historical
    /// `AX_MLX_VERSION_OVERRIDE` check still decides this case.
    NumericMismatch,
    /// Dev header whose commit matches `AX_MLX_EXPERIMENT_COMMIT` and the
    /// bytes of the built library.
    Experiment,
}

/// Parse every well-formed `x.y.z.devYYYYMMDD+hex` token in `bytes`.
///
/// The hex run is 7 to 40 digits so a short upstream hash and a full commit
/// both parse. Tokens that differ only by hex case collapse to one record.
fn scan_dev_versions(bytes: &[u8]) -> Vec<DevMlxVersion> {
    let mut found = Vec::new();
    let mut index = 0;
    while index + 4 < bytes.len() {
        if bytes[index..].starts_with(b".dev") {
            if let Some(dev) = parse_dev_token(bytes, index)
                && !found.contains(&dev)
            {
                found.push(dev);
            }
            index += 4;
        } else {
            index += 1;
        }
    }
    found
}

fn parse_dev_token(bytes: &[u8], dev_dot: usize) -> Option<DevMlxVersion> {
    let rest = bytes.get(dev_dot + 4..)?;
    let date_bytes = rest.get(..8)?;
    if !date_bytes.iter().all(|byte| byte.is_ascii_digit()) || rest.get(8) != Some(&b'+') {
        return None;
    }
    let hex = rest.get(9..)?;
    let hex_len = hex
        .iter()
        .position(|byte| !byte.is_ascii_hexdigit())
        .unwrap_or(hex.len());
    if !(7..=40).contains(&hex_len) {
        return None;
    }
    let commit = std::str::from_utf8(hex.get(..hex_len)?)
        .ok()?
        .to_ascii_lowercase();
    let date = std::str::from_utf8(date_bytes).ok()?.to_string();
    let numeric = numeric_before(bytes, dev_dot)?;
    Some(DevMlxVersion {
        numeric,
        date,
        commit,
    })
}

fn numeric_before(bytes: &[u8], dev_dot: usize) -> Option<String> {
    let mut start = dev_dot;
    while start > 0 {
        let byte = bytes.get(start - 1)?;
        if byte.is_ascii_digit() || *byte == b'.' {
            start -= 1;
        } else {
            break;
        }
    }
    let text = std::str::from_utf8(bytes.get(start..dev_dot)?).ok()?;
    let fields: Vec<_> = text.split('.').collect();
    if fields.len() != 3
        || fields
            .iter()
            .any(|field| field.is_empty() || !field.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return None;
    }
    Some(text.to_string())
}

pub(crate) fn classify_dev_header(bytes: &[u8]) -> DevHeader {
    match scan_dev_versions(bytes).as_slice() {
        [] => DevHeader::Absent,
        [one] => DevHeader::One(one.clone()),
        _ => DevHeader::Conflict,
    }
}

pub(crate) fn library_contains_dev_version(bytes: &[u8], dev: &DevMlxVersion) -> bool {
    let needle = dev.token();
    let needle = needle.as_bytes();
    if needle.is_empty() || bytes.len() < needle.len() {
        return false;
    }
    bytes.windows(needle.len()).any(|window| window == needle)
}

fn normalize_full_commit(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.len() == 40 && raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Some(raw.to_ascii_lowercase())
    } else {
        None
    }
}

/// Decide whether this MLX is the pinned release, a numeric override, or an
/// explicitly selected dev experiment.
///
/// `experiment_commit` is `AX_MLX_EXPERIMENT_COMMIT`. It is required only when
/// the header carries a dev token. A pinned numeric header does not consult it.
pub(crate) fn mlx_link_gate(
    pin_numeric: &str,
    header_numeric: &str,
    header_dev: Option<&DevMlxVersion>,
    library_bytes: Option<&[u8]>,
    experiment_commit: Option<&str>,
) -> Result<MlxLinkGate, String> {
    let Some(dev) = header_dev else {
        return if header_numeric == pin_numeric {
            Ok(MlxLinkGate::PinnedRelease)
        } else {
            Ok(MlxLinkGate::NumericMismatch)
        };
    };
    if dev.numeric != header_numeric {
        return Err(format!(
            "dev MLX version {} does not match header macros {header_numeric}",
            dev.token()
        ));
    }
    let Some(expected) = normalize_full_commit(experiment_commit.unwrap_or("")) else {
        return Err(format!(
            "dev MLX {} is not the admitted pin {pin_numeric}; set AX_MLX_EXPERIMENT_COMMIT to the full 40-hex commit embedded in that build",
            dev.token()
        ));
    };
    if !dev.full_commit() || dev.commit != expected {
        return Err(format!(
            "AX_MLX_EXPERIMENT_COMMIT {expected} does not match dev MLX provenance {}",
            dev.token()
        ));
    }
    let Some(library) = library_bytes else {
        return Err(format!(
            "dev MLX {} requires the built libmlx bytes to confirm provenance",
            dev.token()
        ));
    };
    if !library_contains_dev_version(library, dev) {
        return Err(format!(
            "AX_MLX_EXPERIMENT_COMMIT {expected} does not match the provenance string in the built libmlx ({})",
            dev.token()
        ));
    }
    Ok(MlxLinkGate::Experiment)
}
