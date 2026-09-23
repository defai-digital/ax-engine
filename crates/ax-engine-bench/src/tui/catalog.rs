//! Catalog view for the Models wizard: families, precision variants, size
//! estimates, and the RAM-fit heuristic.
//!
//! Remote membership comes from the live AutomatosX listing on Hugging Face.
//! The local snapshot library is scanned separately for every publisher. Built-in profiles only enrich known repos with labels,
//! presets, size estimates, and qualified revision pins.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Owned catalog row. Live Hub repos are not `'static` profiles, and known
/// repos still copy the fields the wizard, downloader, and server need.
pub(super) struct CatalogModel {
    pub label: String,
    /// Server preset, when this repo is a built-in profile that has one.
    pub preset: Option<String>,
    pub repo_id: String,
    /// Explicit repo id passed to `ax-engine download`, including a pinned
    /// revision when one exists. Never depends on a CLI alias.
    pub download_target: String,
    pub aliases: Vec<String>,
    pub approx_size_bytes: Option<u64>,
    /// Known metadata may enrich a row but never determines Hub membership.
    pub known_profile: bool,
}

/// One precision variant of a model, with cached install state.
pub(super) struct Variant {
    pub model: CatalogModel,
    pub bits: Option<u32>,
    /// The published Hugging Face snapshot already includes its MTP package.
    pub mtp_included: bool,
    pub installed: bool,
    /// On-disk size when installed, 0 otherwise.
    pub size: u64,
}

/// One locally cached repository, independent of publisher or Hub membership.
#[derive(Clone, Debug)]
pub(super) struct LocalModel {
    pub repo_id: String,
    pub snapshot: PathBuf,
    pub cache_dir: PathBuf,
    pub size: u64,
    pub revisions: usize,
    pub ready: bool,
}

pub(super) fn delete_local_cache(model: &LocalModel, cache_root: &Path) -> std::io::Result<()> {
    let expected = cache_root.join(format!("models--{}", model.repo_id.replace('/', "--")));
    if ax_engine_core::repo_ref::parse_repo_ref(&model.repo_id).is_err()
        || model.repo_id.matches('/').count() != 1
        || expected != model.cache_dir
        || !std::fs::symlink_metadata(&expected)?.file_type().is_dir()
    {
        return Err(std::io::Error::other(
            "Model cache path changed; refresh before deleting",
        ));
    }
    std::fs::remove_dir_all(expected)
}

pub(super) type LocalScan = Result<Vec<LocalModel>, String>;

pub(super) fn scan_local_models(cache_root: &Path) -> LocalScan {
    let entries = match std::fs::read_dir(cache_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("Cannot read snapshot cache: {error}")),
    };
    let mut models = Vec::new();
    for entry in entries.flatten() {
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some((owner, name)) = name
            .strip_prefix("models--")
            .and_then(|name| name.split_once("--"))
        else {
            continue;
        };
        let repo_id = format!("{owner}/{name}");
        if ax_engine_core::repo_ref::parse_repo_ref(&repo_id).is_err() {
            continue;
        }
        let cache_dir = entry.path();
        let Ok(snapshots) = std::fs::read_dir(cache_dir.join("snapshots")) else {
            continue;
        };
        let mut snapshots: Vec<_> = snapshots
            .flatten()
            .filter_map(|snapshot| {
                if !snapshot.file_type().is_ok_and(|kind| kind.is_dir()) {
                    return None;
                }
                let path = snapshot.path();
                let ready = (path.join("config.json").is_file()
                    || path.join("model-manifest.json").is_file())
                    && crate::snapshot_has_complete_weights(&path);
                let modified = snapshot.metadata().ok()?.modified().ok()?;
                Some((ready, modified, path))
            })
            .collect();
        snapshots.sort();
        let revisions = snapshots.len();
        if let Some((ready, _, snapshot)) = snapshots.pop() {
            models.push(LocalModel {
                repo_id,
                snapshot,
                size: dir_size(&cache_dir),
                cache_dir,
                revisions,
                ready,
            });
        }
    }
    models.sort_by(|left, right| left.repo_id.cmp(&right.repo_id));
    Ok(models)
}

impl Variant {
    pub fn precision(&self) -> String {
        let lower = self.model.repo_id.to_ascii_lowercase();
        if lower.contains("mxfp4") {
            if lower.contains("q4") || lower.contains("-q4") {
                return "MXFP4-Q4".into();
            }
            return "MXFP4".into();
        }
        let base = self
            .bits
            .map(|b| format!("{b}-bit"))
            .unwrap_or_else(|| self.model.label.clone());
        // AutomatosX pack recipe tags: same bit width can ship as a plain
        // quant, a QAT build, mixed-precision OptiQ, or DWQ — surface the
        // recipe so equal-bit variants stay distinguishable in lists.
        let mut tags = Vec::new();
        if lower.contains("-qat-") {
            tags.push("QAT");
        }
        if lower.contains("optiq") {
            tags.push("OptiQ");
        }
        if lower.contains("-dwq") {
            tags.push("DWQ");
        }
        if lower.contains("-axq-") {
            tags.push("AXQ candidate");
        }
        if tags.is_empty() {
            base
        } else {
            format!("{} {base}", tags.join(" "))
        }
    }

    /// Best size estimate for display: real on-disk bytes when installed,
    /// otherwise the static catalog estimate.
    pub fn size_estimate(&self) -> Option<u64> {
        if self.installed {
            Some(self.size)
        } else {
            self.model.approx_size_bytes
        }
    }
}

/// A model and the precision variants it is published in.
pub(super) struct Family {
    pub key: String,
    pub variants: Vec<Variant>,
}

impl Family {
    pub fn has_mtp(&self) -> bool {
        self.variants.iter().any(|v| v.mtp_included)
    }

    /// Human-readable family name for UI (alias `key` stays for filter/CLI).
    pub fn display_name(&self) -> String {
        family_display_name(&self.key)
    }

    /// Primary productivity stack (Gemma, Qwen, GLM, EmbeddingGemma).
    pub fn is_primary(&self) -> bool {
        is_primary_family_key(&self.key)
    }

    /// Three-tier quality grade for the registry family behind this catalog
    /// key (see `ax_engine_core::support_tier`).
    pub fn support_tier(&self) -> ax_engine_core::ModelSupportTier {
        // Checkpoint-level AXQ candidates must not inherit architecture
        // certification until their quality/runtime/memory gates pass.
        if self.key == "ax-qwen3.6-27b-axq" || self.key == "ax-qwen3-vl-30b-a3b-axq" {
            return ax_engine_core::ModelSupportTier::Compatible;
        }
        ax_engine_core::support_tier_for_family(registry_family_label(&self.key))
    }

    pub fn installed_count(&self) -> usize {
        self.variants.iter().filter(|v| v.installed).count()
    }
}

/// Map a catalog family key (`gemma4-e2b`, `ax-qwen3.6-27b`, …) to the
/// canonical manifest `model_family` label used by the architecture
/// registry. Unknown keys pass through unchanged, so they resolve to
/// `ModelSupportTier::Compatible` (manifest-probing caveat) by default.
pub(super) fn registry_family_label(key: &str) -> &str {
    let k = key.to_ascii_lowercase();
    let k = k.strip_prefix("ax-").unwrap_or(&k);
    if k.starts_with("gemma4") {
        "gemma4"
    } else if k.starts_with("qwen3.5") {
        "qwen3_5"
    } else if k.starts_with("qwen3.6") {
        // The converter canonicalizes qwen3_6 / qwen3.6 model types to the
        // qwen3_next family (convert/model_family.rs) — grade against the
        // registry row that actually runs these models.
        "qwen3_next"
    } else if k.starts_with("qwen3-vl") {
        // Catalog rows currently cover the 30B-A3B MoE Instruct AXQ packs
        // (`model_type=qwen3_vl_moe`). Dense `qwen3_vl` remains a valid
        // runtime family; grade against the MoE row these packs use.
        "qwen3_vl_moe"
    } else if k.starts_with("qwen3-coder-next") {
        "qwen3_next"
    } else if k.starts_with("qwen3-embedding") {
        "qwen3"
    } else if k.starts_with("glm") {
        "glm4_moe_lite"
    } else if k.starts_with("gpt-oss") {
        "gpt_oss"
    } else if k.starts_with("llama3") {
        "llama3"
    } else if k.starts_with("llama4") {
        "llama4"
    } else if k.starts_with("mistral") || k.starts_with("ministral") || k.starts_with("devstral") {
        "mistral3"
    } else if k.starts_with("embeddinggemma") {
        "embeddinggemma"
    } else if k.starts_with("diffusiongemma") {
        "diffusion_gemma"
    } else {
        key
    }
}

/// Primary catalog families (deepest performance + product focus). The
/// AutomatosX packs (`ax-` prefix) are branded builds of the same primary
/// stack, embeddings included.
pub(super) fn is_primary_family_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    let k = k.strip_prefix("ax-").unwrap_or(&k);
    k.starts_with("gemma")
        || k.starts_with("qwen")
        || k.starts_with("glm")
        || k.starts_with("embeddinggemma")
}

/// Embedding-only catalog families (served via /v1/embeddings, not chat).
pub(super) fn is_embedding_family_key(key: &str) -> bool {
    key.to_ascii_lowercase().contains("embedding")
}

/// Whether a catalog family can back the first-run text chat experience.
///
/// Quick Start must not select a smaller task-specific pack (ASR, OCR,
/// embeddings, diffusion, or a reward model) merely because its download is
/// smaller than the smallest general chat model.
pub(super) fn is_chat_family_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    !is_embedding_family_key(&key)
        && !key.contains("diffusiongemma")
        && !key.contains("asr")
        && !key.contains("ocr")
        && !key.contains("genrm")
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
        "ax-qwen3.5-9b" => "AX Qwen 3.5 9B".into(),
        "ax-qwen3.6-27b" => "AX Qwen 3.6 27B".into(),
        "ax-qwen3.6-27b-axq" => "AX Qwen 3.6 27B AXQ candidates".into(),
        "ax-qwen3.6-35b" => "AX Qwen 3.6 35B".into(),
        "ax-qwen3-vl-30b-a3b-axq" => "AX Qwen3-VL 30B-A3B Instruct AXQ".into(),
        "ax-gemma4-12b" => "AX Gemma 4 12B".into(),
        "ax-gemma4-26b" => "AX Gemma 4 26B".into(),
        "ax-gemma4-31b" => "AX Gemma 4 31B".into(),
        "ax-qwen3-coder-next" => "AX Qwen3 Coder Next".into(),
        "ax-embeddinggemma-300m" => "AX EmbeddingGemma 300M".into(),
        "ax-qwen3-embedding-0.6b" => "AX Qwen3 Embedding 0.6B".into(),
        "ax-qwen3-embedding-4b" => "AX Qwen3 Embedding 4B".into(),
        "ax-qwen3-embedding-8b" => "AX Qwen3 Embedding 8B".into(),
        "ax-diffusiongemma-26b" => "AX DiffusionGemma 26B".into(),
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

/// HF hub cache directory for a repo id (`.../models--org--name`).
pub(super) fn repo_cache_dir(repo_id: &str) -> PathBuf {
    crate::default_hf_cache_root().join(format!("models--{}", repo_id.replace('/', "--")))
}

/// The actual on-disk snapshot directory for a downloaded repo (containing
/// `config.json`/`*.safetensors`), not just the top-level HF cache wrapper.
/// Picks the most recently modified **usable** snapshot when a repo has more
/// than one cached revision. This is what the server needs for
/// `--mlx-model-artifacts-dir` — passing the wrapper dir directly would miss
/// the actual model files, which live one level down under `snapshots/<hash>/`.
pub(super) fn repo_snapshot_dir(repo_id: &str) -> Option<PathBuf> {
    let snapshots = repo_cache_dir(repo_id).join("snapshots");
    let mut dirs: Vec<(PathBuf, std::time::SystemTime)> = std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() || !artifact_dir_usable(&path) {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((path, modified))
        })
        .collect();
    dirs.sort_by_key(|(_, modified)| *modified);
    dirs.pop().map(|(path, _)| path)
}

/// True when a directory looks like a loadable AX/MLX artifact tree (not an
/// empty or partial HF cache stub).
pub(super) fn artifact_dir_usable(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    if dir.join("config.json").is_file() || dir.join("model-manifest.json").is_file() {
        return true;
    }
    // Some incomplete snapshots only have tensors; still treat as present so
    // the user can open the path, but prefer config/manifest when available.
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            })
        })
        .unwrap_or(false)
}

/// Whether the HF hub cache for `repo_id` has a usable snapshot (not merely a
/// wrapper directory or incomplete download stubs).
pub(super) fn repo_is_installed(repo_id: &str) -> bool {
    repo_snapshot_dir(repo_id).is_some()
}

/// The most recently modified immediate subdirectory of `dir`, if any.
#[cfg(test)]
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

/// Families for an explicit repo-id list from the live Hub listing. Install state comes
/// from disk. Families are sorted by display name so the wizard does not
/// jump around with Hub page order.
pub(super) fn build_families_from_repo_ids(repo_ids: &[String]) -> Vec<Family> {
    let models = repo_ids
        .iter()
        .map(|repo_id| model_for_repo(repo_id))
        .collect();
    build_families_from_models(models, true, true)
}

/// Same grouping as [`build_families_from_repo_ids`] without reading the
/// Hugging Face cache. Tests use this so a machine with real snapshots
/// does not pay `dir_size` on multi-gigabyte packs.
#[cfg(test)]
pub(super) fn build_families_from_repo_ids_uninstalled(repo_ids: &[String]) -> Vec<Family> {
    let models = repo_ids
        .iter()
        .map(|repo_id| model_for_repo(repo_id))
        .collect();
    build_families_from_models(models, false, true)
}

/// Catalog shape (grouping/sorting/labels) without touching disk for
/// installed/size state. Every downloadable profile reports `installed:
/// false, size: 0` unconditionally.
///
/// Production catalog construction walks the real HF cache on disk (`repo_is_installed`,
/// `dir_size`) for every downloadable profile, which is appropriately real
/// work for production use but is unrelated, environment-dependent I/O for
/// tests that only assert on catalog shape (grouping, labels, precision,
/// support tier) — on a developer machine with many real cached model
/// downloads this made the TUI test suite take tens of seconds to minutes
/// instead of running near-instantly.
#[cfg(test)]
pub(super) fn build_families_uninstalled() -> Vec<Family> {
    build_families_from_models(builtin_downloadable_models(), false, false)
}

#[cfg(test)]
fn builtin_downloadable_models() -> Vec<(String, CatalogModel)> {
    crate::MODEL_PROFILES
        .iter()
        .filter(|profile| profile.is_downloadable())
        .map(|profile| {
            (
                family_key(profile.label),
                catalog_model_from_profile(profile),
            )
        })
        .collect()
}

fn catalog_model_from_profile(profile: &crate::ModelProfile) -> CatalogModel {
    CatalogModel {
        label: profile.label.to_string(),
        preset: profile.preset.map(str::to_string),
        repo_id: profile.repo_id.to_string(),
        download_target: match crate::profile_revision(*profile) {
            Some(revision) => format!("{}@{}", profile.repo_id, revision.replace('%', "%25")),
            None => profile.repo_id.to_string(),
        },
        aliases: profile
            .aliases
            .iter()
            .map(|alias| (*alias).to_string())
            .collect(),
        approx_size_bytes: profile.approx_size_bytes,
        known_profile: true,
    }
}

/// Known profiles keep their label, preset, size estimate, and family key.
/// Anything else published on the Hub becomes its own row keyed from the
/// repo name, and downloads by repo id.
fn model_for_repo(repo_id: &str) -> (String, CatalogModel) {
    if let Some(profile) = profile_for_repo(repo_id) {
        return (
            family_key(profile.label),
            catalog_model_from_profile(profile),
        );
    }
    let name = repo_id.rsplit('/').next().unwrap_or(repo_id);
    (
        family_key_for_repo(repo_id),
        CatalogModel {
            label: name.to_string(),
            preset: None,
            repo_id: repo_id.to_string(),
            download_target: repo_id.to_string(),
            aliases: Vec::new(),
            approx_size_bytes: None,
            known_profile: false,
        },
    )
}

fn profile_for_repo(repo_id: &str) -> Option<&'static crate::ModelProfile> {
    let mut fallback = None;
    for profile in crate::MODEL_PROFILES {
        if profile.repo_id == repo_id {
            if profile.is_downloadable() {
                return Some(profile);
            }
            fallback = Some(profile);
        }
    }
    fallback
}

/// Group an unknown repo with its other bit-widths: strip MTP, quant, and
/// recipe suffixes, then keep the `ax-` family prefix the wizard already uses.
pub(super) fn family_key_for_repo(repo_id: &str) -> String {
    let name = repo_id.rsplit('/').next().unwrap_or(repo_id);
    let mut key = name.to_ascii_lowercase();
    if let Some(rest) = key.strip_suffix("-mtp") {
        key = rest.to_string();
    }
    let suffixes = [
        "-mxfp8", "-mxfp4", "-4bit", "-5bit", "-6bit", "-8bit", "-optiq", "-axq", "-qat", "-dwq",
        "-mlx", "-q4",
    ];
    loop {
        let stripped = suffixes
            .iter()
            .find_map(|suffix| key.strip_suffix(suffix).map(str::to_string));
        match stripped {
            Some(rest) if !rest.is_empty() => key = rest,
            _ => break,
        }
    }
    let stripped_bits = key.rfind('-').and_then(|idx| {
        let tail = &key[idx + 1..];
        let digits = tail.strip_suffix("bit")?;
        if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
            Some(key[..idx].to_string())
        } else {
            None
        }
    });
    if let Some(stripped) = stripped_bits {
        key = stripped;
    }
    if key.is_empty() {
        name.to_ascii_lowercase()
    } else if key.starts_with("ax-") {
        key
    } else {
        format!("ax-{key}")
    }
}

fn install_state_from_disk(repo_id: &str) -> (bool, u64) {
    let installed = repo_is_installed(repo_id);
    // Require a usable snapshot (config/manifest/safetensors), not merely
    // an HF wrapper dir left by a cancelled or incomplete download.
    let size = if installed {
        dir_size(&repo_cache_dir(repo_id))
    } else {
        0
    };
    (installed, size)
}

fn build_families_from_models(
    models: Vec<(String, CatalogModel)>,
    scan_disk: bool,
    sort_families: bool,
) -> Vec<Family> {
    let mut families: Vec<Family> = Vec::new();
    for (key, model) in models {
        let (installed, size) = if scan_disk {
            install_state_from_disk(&model.repo_id)
        } else {
            (false, 0)
        };
        let variant = Variant {
            bits: quant_bits(&model.repo_id),
            mtp_included: model.known_profile
                && model.repo_id.to_ascii_lowercase().contains("-mtp"),
            installed,
            size,
            model,
        };
        match families.iter_mut().find(|family| family.key == key) {
            Some(family) => family.variants.push(variant),
            None => families.push(Family {
                key,
                variants: vec![variant],
            }),
        }
    }
    for family in &mut families {
        family
            .variants
            .sort_by_key(|variant| variant.bits.unwrap_or(99));
    }
    if sort_families {
        families.sort_by(|left, right| {
            left.display_name()
                .cmp(&right.display_name())
                .then_with(|| left.key.cmp(&right.key))
        });
    }
    families
}

/// Result of a background catalog refresh.
pub(super) enum CatalogRefresh {
    /// Live Hub membership, including local AutomatosX installs missing
    /// from the listing.
    Hub {
        ids: Vec<String>,
        families: Vec<Family>,
    },
    /// Install-state rescan of whatever membership is already on screen.
    Ready(Vec<Family>),
    /// The Hub listing failed. `families` contains local installs only.
    Fallback {
        error: String,
        families: Vec<Family>,
    },
}

const HUB_MODELS_URL: &str = "https://huggingface.co/api/models?author=AutomatosX&limit=1000";

/// Public AutomatosX model ids, following Hub `Link: rel="next"` pages.
/// Sends `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` when one is set so private
/// repos that token can see are included. The token is never logged.
pub(super) fn fetch_automatosx_repo_ids() -> Result<Vec<String>, String> {
    if std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    }) {
        return Err("Hugging Face offline mode is enabled".into());
    }
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(30))
        .timeout(Duration::from_secs(45))
        .build();
    let token = hf_token();
    let mut url = HUB_MODELS_URL.to_string();
    let mut ids = Vec::new();
    let mut seen = HashSet::new();
    let mut pages = HashSet::new();
    let mut complete = false;
    for _ in 0..20 {
        if !url.starts_with("https://huggingface.co/api/models?") || !pages.insert(url.clone()) {
            return Err("Hugging Face returned an invalid pagination link".into());
        }
        let (next, body) = http_get(&agent, &url, token.as_deref())?;
        for id in parse_hf_model_ids(&body)? {
            if seen.insert(id.clone()) {
                ids.push(id);
            }
        }
        match next {
            Some(next) => url = next,
            None => {
                complete = true;
                break;
            }
        }
    }
    if !complete {
        return Err("Hugging Face model list exceeded the pagination limit".into());
    }
    if ids.is_empty() {
        return Err("Hugging Face returned no AutomatosX models".into());
    }
    Ok(ids)
}

fn hf_token() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(value) = std::env::var(key) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn http_get(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
) -> Result<(Option<String>, String), String> {
    let mut request = agent
        .get(url)
        .set("User-Agent", "ax-engine")
        .set("Accept", "application/json");
    if let Some(token) = token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    match request.call() {
        Ok(response) => {
            let next = next_link(response.header("link"));
            let body = response
                .into_string()
                .map_err(|err| format!("reading Hugging Face response: {err}"))?;
            Ok((next, body))
        }
        Err(ureq::Error::Status(code, _)) => {
            Err(format!("Hugging Face model list returned HTTP {code}"))
        }
        Err(err) => Err(format!("Hugging Face model list request failed: {err}")),
    }
}

/// Repo ids from one Hugging Face `/api/models` page.
pub(super) fn parse_hf_model_ids(body: &str) -> Result<Vec<String>, String> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|err| format!("Hugging Face model list was not JSON: {err}"))?;
    let items = value
        .as_array()
        .ok_or_else(|| "Hugging Face model list was not a JSON array".to_string())?;
    let mut ids = Vec::new();
    for item in items {
        let Some(id) = item.get("id").and_then(|value| value.as_str()) else {
            continue;
        };
        if !id.starts_with("AutomatosX/")
            || ax_engine_core::repo_ref::parse_repo_ref(id).is_err()
            || id.contains('@')
            || id.matches('/').count() != 1
        {
            continue;
        }
        ids.push(id.to_string());
    }
    Ok(ids)
}

/// `Link` header URL whose relation is `next`.
pub(super) fn next_link(header: Option<&str>) -> Option<String> {
    let header = header?;
    for part in header.split(',') {
        let part = part.trim();
        let (url, params) = part.split_once(';')?;
        let is_next = params.split(';').any(|param| {
            let param = param.trim();
            param.eq_ignore_ascii_case("rel=\"next\"") || param.eq_ignore_ascii_case("rel=next")
        });
        if !is_next {
            continue;
        }
        let url = url
            .trim()
            .trim_start_matches('<')
            .trim_end_matches('>')
            .trim();
        if !url.is_empty() {
            return Some(url.to_string());
        }
    }
    None
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
