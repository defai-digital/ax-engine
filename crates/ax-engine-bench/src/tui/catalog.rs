//! Catalog view over `crate::MODEL_PROFILES`: families, precision variants,
//! size estimates, and the RAM-fit heuristic used by the wizard.

use std::path::{Path, PathBuf};

/// One precision variant of a model, with cached install state.
pub(super) struct Variant {
    pub profile: &'static crate::ModelProfile,
    pub bits: Option<u32>,
    /// `download-mtp` alias when this variant has an MTP accelerator.
    pub mtp_alias: Option<&'static str>,
    pub installed: bool,
    /// On-disk size when installed, 0 otherwise.
    pub size: u64,
}

impl Variant {
    pub fn precision(&self) -> String {
        let lower = self.profile.repo_id.to_ascii_lowercase();
        if lower.contains("mxfp4") {
            if lower.contains("q4") || lower.contains("-q4") {
                return "MXFP4-Q4".into();
            }
            return "MXFP4".into();
        }
        self.bits
            .map(|b| format!("{b}-bit"))
            .unwrap_or_else(|| self.profile.label.to_string())
    }

    /// Best size estimate for display: real on-disk bytes when installed,
    /// otherwise the static catalog estimate.
    pub fn size_estimate(&self) -> Option<u64> {
        if self.installed {
            Some(self.size)
        } else {
            self.profile.approx_size_bytes
        }
    }

    /// (base, extra) download estimate for the MTP package of this variant.
    /// The MTP path may fetch a different-precision base repo than the direct
    /// variant, so this reads the `download-mtp` target's own numbers.
    pub fn mtp_size_estimate(&self) -> Option<(Option<u64>, Option<u64>)> {
        let alias = self.mtp_alias?;
        let target = crate::mtp_download_target_for_model(alias)?;
        Some((target.approx_base_bytes, target.approx_extra_bytes))
    }
}

/// A model and the precision variants it is published in.
pub(super) struct Family {
    pub key: String,
    pub variants: Vec<Variant>,
}

impl Family {
    pub fn has_mtp(&self) -> bool {
        self.variants.iter().any(|v| v.mtp_alias.is_some())
    }

    /// Human-readable family name for UI (alias `key` stays for filter/CLI).
    pub fn display_name(&self) -> String {
        family_display_name(&self.key)
    }

    /// Primary productivity stack vs secondary preview families.
    pub fn is_primary(&self) -> bool {
        is_primary_family_key(&self.key)
    }

    pub fn installed_count(&self) -> usize {
        self.variants.iter().filter(|v| v.installed).count()
    }
}

/// Primary catalog families (deepest performance + product focus).
pub(super) fn is_primary_family_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    k.starts_with("gemma") || k.starts_with("qwen") || k.starts_with("glm")
}

/// Friendly display name for a catalog family key.
pub(super) fn family_display_name(key: &str) -> String {
    match key {
        "gemma4-e2b" => "Gemma 4 E2B".into(),
        "gemma4-12b" => "Gemma 4 12B".into(),
        "gemma4-26b" => "Gemma 4 26B".into(),
        "gemma4-31b" => "Gemma 4 31B".into(),
        "glm4.7-flash" => "GLM 4.7 Flash".into(),
        "qwen3.5-9b" => "Qwen 3.5 9B".into(),
        "qwen3.6-27b" => "Qwen 3.6 27B".into(),
        "qwen3.6-35b" => "Qwen 3.6 35B".into(),
        "llama3.1-8b" => "Llama 3.1 8B".into(),
        "llama3.3-70b" => "Llama 3.3 70B".into(),
        "llama4-scout" => "Llama 4 Scout".into(),
        "mistral-small" => "Mistral Small".into(),
        "ministral-8b" => "Ministral 8B".into(),
        "devstral-small" => "Devstral Small".into(),
        "gpt-oss-20b" => "GPT-OSS 20B".into(),
        "gpt-oss-120b" => "GPT-OSS 120B".into(),
        other => {
            // Fallback: turn `foo-bar` into title-ish text without inventing facts.
            other
                .split(['-', '_'])
                .map(|part| {
                    let mut chars = part.chars();
                    match chars.next() {
                        Some(first) => {
                            format!("{}{}", first.to_uppercase(), chars.as_str())
                        }
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

/// Quantization bit-width parsed from a repo id (e.g. `...-4bit` -> 4).
///
/// Also maps GPT-OSS product tags (`MXFP4-Q4`, bare `MXFP4`) to 4-bit so the
/// wizard can sort and badge those variants.
pub(super) fn quant_bits(repo_id: &str) -> Option<u32> {
    let lower = repo_id.to_ascii_lowercase();
    if lower.contains("mxfp4") {
        if let Some(idx) = lower.rfind('q') {
            let digits: String = lower[idx + 1..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(bits) = digits.parse::<u32>()
                && bits > 0
            {
                return Some(bits);
            }
        }
        return Some(4);
    }
    let idx = lower.find("bit")?;
    let digits: String = lower[..idx]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse().ok()
}

/// Family key: the label with any trailing `-Nbit` precision suffix removed.
pub(super) fn family_key(label: &str) -> String {
    let lower = label.to_ascii_lowercase();
    if let Some(idx) = lower.rfind("-")
        && lower[idx + 1..].ends_with("bit")
        && lower[idx + 1..idx + 2].chars().all(|c| c.is_ascii_digit())
    {
        return label[..idx].to_string();
    }
    label.to_string()
}

/// `download-mtp` trigger alias for a model label.
///
/// Mirrors the Python catalog's per-variant `mtp_target` exactly.
pub(super) fn mtp_trigger_alias(label: &str) -> Option<&'static str> {
    match label {
        "gemma4-12b" => Some("gemma-4-12b-4bit"),
        "gemma4-12b-6bit" => Some("gemma-4-12b"),
        "gemma4-26b" => Some("gemma-4-26b"),
        "gemma4-31b" => Some("gemma-4-31b"),
        "qwen3.6-27b" => Some("qwen3.6-27b-6bit"),
        "qwen3.6-27b-6bit" => Some("qwen3.6-27b-6bit"),
        "qwen3.6-35b" => Some("qwen3.6-35b-a3b"),
        _ => None,
    }
}

/// HF hub cache directory for a repo id (`.../models--org--name`).
pub(super) fn repo_cache_dir(repo_id: &str) -> PathBuf {
    crate::default_hf_cache_root().join(format!("models--{}", repo_id.replace('/', "--")))
}

/// The actual on-disk snapshot directory for a downloaded repo (containing
/// `config.json`/`*.safetensors`), not just the top-level HF cache wrapper.
/// Picks the most recently modified snapshot when a repo has more than one
/// cached revision. This is what the server needs for `--mlx-model-artifacts-dir`
/// — passing the wrapper dir directly would miss the actual model files, which
/// live one level down under `snapshots/<hash>/`.
pub(super) fn repo_snapshot_dir(repo_id: &str) -> Option<PathBuf> {
    most_recent_subdir(&repo_cache_dir(repo_id).join("snapshots"))
}

/// The most recently modified immediate subdirectory of `dir`, if any.
pub(super) fn most_recent_subdir(dir: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<(PathBuf, std::time::SystemTime)> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((path, modified))
        })
        .collect();
    dirs.sort_by_key(|(_, modified)| *modified);
    dirs.pop().map(|(path, _)| path)
}

pub(super) fn dir_has_content(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|mut it| it.next().is_some())
        .unwrap_or(false)
}

/// Recursive on-disk size, following directories but not chasing symlinks twice.
pub(super) fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
}

pub(super) fn build_families() -> Vec<Family> {
    let mut families: Vec<Family> = Vec::new();
    for profile in crate::MODEL_PROFILES.iter().filter(|p| p.downloadable) {
        let key = family_key(profile.label);
        let cache = repo_cache_dir(profile.repo_id);
        let installed = cache.is_dir() && dir_has_content(&cache);
        let variant = Variant {
            profile,
            bits: quant_bits(profile.repo_id),
            mtp_alias: mtp_trigger_alias(profile.label),
            installed,
            size: if installed { dir_size(&cache) } else { 0 },
        };
        match families.iter_mut().find(|f| f.key == key) {
            Some(f) => f.variants.push(variant),
            None => families.push(Family {
                key,
                variants: vec![variant],
            }),
        }
    }
    for family in &mut families {
        family.variants.sort_by_key(|v| v.bits.unwrap_or(99));
    }
    families
}

/// Flattened installed (family, variant) index pairs for the Serve list.
pub(super) fn installed_variants(families: &[Family]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (fi, family) in families.iter().enumerate() {
        for (vi, variant) in family.variants.iter().enumerate() {
            if variant.installed {
                out.push((fi, vi));
            }
        }
    }
    out
}

pub(super) fn format_bytes(num: u64) -> String {
    let mut value = num as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if value < 1024.0 {
            return format!("{value:.1} {unit}");
        }
        value /= 1024.0;
    }
    format!("{value:.1} PB")
}

/// `format_bytes` with the "estimate" marker used for catalog sizes.
pub(super) fn format_approx_bytes(num: Option<u64>) -> String {
    match num {
        Some(num) => format!("~{}", format_bytes(num)),
        None => "size varies".into(),
    }
}

// ---------------------------------------------------------------------------
// RAM fit heuristic
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RamFit {
    Fits,
    Tight,
    TooLarge,
    Unknown,
}

impl RamFit {
    /// Short badge text for list rows (keep "fits" so scanners stay familiar).
    pub fn label(self) -> &'static str {
        match self {
            RamFit::Fits => "fits",
            RamFit::Tight => "tight",
            RamFit::TooLarge => "too large",
            RamFit::Unknown => "",
        }
    }

    /// Plain-language outcome for home / confirm copy.
    pub fn plain(self) -> &'static str {
        match self {
            RamFit::Fits => "good for this Mac",
            RamFit::Tight => "may be slow under load",
            RamFit::TooLarge => "likely won't fit in memory",
            RamFit::Unknown => "",
        }
    }
}

/// Rough serving-footprint check against unified memory.
///
/// Heuristic, not a promise: estimated footprint = weight bytes x 1.2 (runtime
/// graph + KV cache headroom) + 1.5 GiB fixed overhead.  Below 70% of total
/// RAM counts as a comfortable fit, below 85% as tight (may page under load),
/// and above that as too large.  macOS wires GPU memory out of the same
/// unified pool, so anything past ~85% starts fighting the OS.
pub(super) fn ram_fit(model_bytes: Option<u64>, total_ram: Option<u64>) -> RamFit {
    let (Some(bytes), Some(ram)) = (model_bytes, total_ram) else {
        return RamFit::Unknown;
    };
    if ram == 0 {
        return RamFit::Unknown;
    }
    let footprint = bytes as f64 * 1.2 + 1.5 * 1024.0 * 1024.0 * 1024.0;
    let ratio = footprint / ram as f64;
    if ratio < 0.70 {
        RamFit::Fits
    } else if ratio < 0.85 {
        RamFit::Tight
    } else {
        RamFit::TooLarge
    }
}
