//! Catalog and host-hardware parsing: family grouping, precision metadata,
//! RAM-fit thresholds, snapshot-dir discovery, and `df` parsing.
use super::super::catalog::{
    self, RamFit, build_families_uninstalled, family_key, most_recent_subdir, quant_bits,
};
use super::super::hardware::parse_df_available_kib;
use std::path::Path;
use std::process;

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

#[test]
fn grouping_collapses_variants_into_families() {
    let families = build_families_uninstalled();
    let keys: Vec<&str> = families.iter().map(|f| f.key.as_str()).collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        keys.len(),
        "family keys must be unique: {keys:?}"
    );
    let flat = crate::MODEL_PROFILES
        .iter()
        .filter(|p| p.is_downloadable())
        .count();
    assert!(
        families.len() < flat,
        "{} families from {flat} profiles",
        families.len()
    );

    let q9 = families.iter().find(|f| f.key == "ax-qwen3.5-9b").unwrap();
    assert_eq!(q9.variants.len(), 3); // OptiQ-4bit / 4-bit / 6-bit
    assert!(q9.has_mtp(), "AX Qwen packs bundle mtp.safetensors");
    let axq = families
        .iter()
        .find(|f| f.key == "ax-qwen3.6-27b-axq")
        .unwrap();
    assert_eq!(axq.variants.len(), 2);
    assert!(
        axq.variants
            .iter()
            .all(|variant| variant.precision().contains("AXQ candidate"))
    );
    let vl_axq = families
        .iter()
        .find(|f| f.key == "ax-qwen3-vl-30b-a3b-axq")
        .unwrap();
    assert_eq!(vl_axq.variants.len(), 2);
    assert!(
        !vl_axq.has_mtp(),
        "Qwen3-VL 30B AXQ Instruct packs do not ship MTP"
    );
    assert!(
        vl_axq
            .variants
            .iter()
            .all(|variant| variant.precision().contains("AXQ candidate"))
    );
    let embed = families
        .iter()
        .find(|f| f.key == "ax-embeddinggemma-300m")
        .unwrap();
    assert!(!embed.has_mtp());
    let g12 = families.iter().find(|f| f.key == "ax-gemma4-12b").unwrap();
    assert!(g12.has_mtp());
    // Recommended (first) variant is the lowest bit-width.
    assert_eq!(g12.variants[0].bits, Some(4));
}

#[test]
fn quant_and_family_parsing() {
    assert_eq!(quant_bits("mlx-community/gemma-4-12B-it-4bit"), Some(4));
    assert_eq!(quant_bits("mlx-community/Qwen3.6-27B-8bit"), Some(8));
    assert_eq!(quant_bits("mlx-community/gpt-oss-20b-MXFP4-Q4"), Some(4));
    assert_eq!(quant_bits("mlx-community/gpt-oss-120b-MXFP4-Q4"), Some(4));
    assert_eq!(
        quant_bits("AutomatosX/AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP"),
        Some(4)
    );
    assert_eq!(
        quant_bits("AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP"),
        Some(4)
    );
    assert_eq!(
        quant_bits("AutomatosX/AX-Qwen3-Embedding-8B-MLX-4bit-DWQ"),
        Some(4)
    );
    assert_eq!(family_key("gemma4-e2b-8bit"), "gemma4-e2b");
    assert_eq!(family_key("glm4.7-flash-4bit"), "glm4.7-flash");
    assert_eq!(family_key("qwen3.6-35b"), "qwen3.6-35b");
    assert_eq!(family_key("gpt-oss-20b"), "gpt-oss-20b");
    assert_eq!(family_key("ax-qwen3.6-27b-6bit"), "ax-qwen3.6-27b");
    assert_eq!(
        family_key("ax-embeddinggemma-300m"),
        "ax-embeddinggemma-300m"
    );
}

#[test]
fn catalog_families_map_to_registry_support_tiers() {
    use ax_engine_core::ModelSupportTier;

    assert_eq!(catalog::registry_family_label("gemma4-e2b"), "gemma4");
    assert_eq!(
        catalog::registry_family_label("ax-qwen3.6-27b"),
        "qwen3_next"
    );
    assert_eq!(catalog::registry_family_label("ax-qwen3.5-9b"), "qwen3_5");
    assert_eq!(
        catalog::registry_family_label("ax-qwen3-coder-next"),
        "qwen3_next"
    );
    assert_eq!(
        catalog::registry_family_label("ax-qwen3-vl-30b-a3b-axq"),
        "qwen3_vl_moe"
    );
    assert_eq!(catalog::registry_family_label("gpt-oss-20b"), "gpt_oss");
    assert_eq!(catalog::registry_family_label("llama3.1-8b"), "llama3");
    assert_eq!(catalog::registry_family_label("ministral-8b"), "mistral3");
    assert_eq!(
        catalog::registry_family_label("ax-diffusiongemma-26b"),
        "diffusion_gemma"
    );
    // Unknown catalog keys pass through and resolve to Compatible.
    assert_eq!(catalog::registry_family_label("mystery-7b"), "mystery-7b");

    let families = build_families_uninstalled();
    for family in &families {
        let tier = family.support_tier();
        if family.key == "ax-qwen3.6-27b-axq" || family.key == "ax-qwen3-vl-30b-a3b-axq" {
            assert_eq!(
                tier,
                ModelSupportTier::Compatible,
                "checkpoint candidates must not inherit family certification"
            );
        } else if family.key.starts_with("ax-diffusiongemma") {
            assert_eq!(
                tier,
                ModelSupportTier::Experimental,
                "{} must surface as experimental",
                family.key
            );
        } else {
            assert_ne!(
                tier,
                ModelSupportTier::Experimental,
                "{} must not surface as experimental",
                family.key
            );
        }
    }
}

#[test]
fn automatosx_packs_are_primary_families_with_recipe_precisions() {
    let families = build_families_uninstalled();
    for key in [
        "ax-qwen3.5-9b",
        "ax-qwen3.6-27b",
        "ax-qwen3.6-35b",
        "ax-gemma4-12b",
        "ax-gemma4-26b",
        "ax-gemma4-31b",
        "ax-qwen3-coder-next",
        "ax-embeddinggemma-300m",
        "ax-qwen3-embedding-0.6b",
        "ax-qwen3-embedding-4b",
        "ax-qwen3-embedding-8b",
    ] {
        let family = families
            .iter()
            .find(|f| f.key == key)
            .unwrap_or_else(|| panic!("missing TUI family {key}"));
        assert!(family.is_primary(), "{key} must group as primary");
        assert!(
            family.variants.iter().all(|v| v.profile.downloadable),
            "{key} variants must be downloadable"
        );
        assert!(
            family
                .variants
                .iter()
                .all(|v| v.profile.approx_size_bytes.is_some()),
            "{key} variants must carry size estimates"
        );
        assert!(
            family
                .variants
                .iter()
                .all(|v| v.profile.repo_id.starts_with("AutomatosX/")),
            "{key} variants must resolve to AutomatosX repos"
        );
    }

    let q27 = families
        .iter()
        .find(|f| f.key == "ax-qwen3.6-27b")
        .expect("ax-qwen3.6-27b family");
    // OptiQ flagship sorts first among equal-bit variants and keeps its
    // recipe tag distinguishable from the plain 4-bit build.
    assert_eq!(q27.variants[0].precision(), "OptiQ 4-bit");
    assert!(
        q27.variants
            .iter()
            .any(|v| v.precision() == "4-bit" && v.bits == Some(4)),
        "plain 4-bit variant should stay distinguishable"
    );
    let g12 = families
        .iter()
        .find(|f| f.key == "ax-gemma4-12b")
        .expect("ax-gemma4-12b family");
    assert_eq!(g12.variants[0].precision(), "QAT OptiQ 4-bit");
    let e8 = families
        .iter()
        .find(|f| f.key == "ax-qwen3-embedding-8b")
        .expect("ax-qwen3-embedding-8b family");
    assert_eq!(e8.variants[0].precision(), "DWQ 4-bit");
}

#[test]
fn legacy_profiles_stay_serve_aliases_but_leave_the_download_catalog() {
    // Managed downloads are restricted to the AutomatosX org; the older
    // mlx-community-backed profiles disappear from the TUI catalog but keep
    // resolving as serve aliases for already-downloaded artifacts.
    let families = build_families_uninstalled();
    for key in [
        "qwen3.5-9b",
        "glm4.7-flash",
        "llama3.1-8b",
        "llama3.3-70b",
        "llama4-scout",
        "mistral-small",
        "ministral-8b",
        "devstral-small",
        "gpt-oss-20b",
        "gpt-oss-120b",
    ] {
        assert!(
            !families.iter().any(|f| f.key == key),
            "legacy family {key} must not appear in the download catalog"
        );
    }
    for alias in ["qwen3.5-9b", "gpt-oss-20b", "llama3.3-70b"] {
        let profile = crate::profile_for_model(alias)
            .unwrap_or_else(|| panic!("legacy serve alias {alias} must keep resolving"));
        assert!(
            !profile.is_downloadable(),
            "{alias} must not be download-managed"
        );
    }
    assert!(
        families
            .iter()
            .all(|f| f.variants.iter().all(|v| v.profile.is_downloadable())),
        "every catalog variant must be an AutomatosX-managed download"
    );
}

#[test]
fn every_downloadable_profile_has_a_size_estimate() {
    for profile in crate::MODEL_PROFILES.iter().filter(|p| p.downloadable) {
        assert!(
            profile.approx_size_bytes.is_some(),
            "{} is downloadable but has no approx_size_bytes",
            profile.label
        );
    }
}

#[test]
fn ram_fit_thresholds() {
    let gib = 1024u64 * 1024 * 1024;
    // 3 GB model on 64 GB RAM: comfortable.
    assert_eq!(
        catalog::ram_fit(Some(3 * gib), Some(64 * gib)),
        RamFit::Fits
    );
    // 40 GB model on 64 GB RAM: footprint ~49.5/64 = 77% -> tight.
    assert_eq!(
        catalog::ram_fit(Some(40 * gib), Some(64 * gib)),
        RamFit::Tight
    );
    // 60 GB model on 64 GB RAM: too large.
    assert_eq!(
        catalog::ram_fit(Some(60 * gib), Some(64 * gib)),
        RamFit::TooLarge
    );
    assert_eq!(catalog::ram_fit(None, Some(64 * gib)), RamFit::Unknown);
    assert_eq!(catalog::ram_fit(Some(gib), None), RamFit::Unknown);
}

#[test]
fn most_recent_subdir_is_none_for_missing_dir() {
    assert_eq!(
        most_recent_subdir(Path::new("/definitely/does/not/exist")),
        None
    );
}

#[test]
fn most_recent_subdir_picks_the_only_entry() {
    let base = std::env::temp_dir().join(format!("ax-engine-tui-test-{}", process::id()));
    let snapshots = base.join("snapshots");
    let snapshot = snapshots.join("abc123");
    std::fs::create_dir_all(&snapshot).unwrap();
    assert_eq!(most_recent_subdir(&snapshots), Some(snapshot));
    std::fs::remove_dir_all(&base).unwrap();
}

// ---------------------------------------------------------------------------
// Hardware
// ---------------------------------------------------------------------------

#[test]
fn df_available_column_parses() {
    let output = "\
Filesystem   1024-blocks       Used Available Capacity iused ifree %iused  Mounted on
/dev/disk3s5  971350180  530692820 419382948    56%  915272 4193829480    0%   /System/Volumes/Data
";
    assert_eq!(parse_df_available_kib(output), Some(419_382_948));
    assert_eq!(parse_df_available_kib("garbage"), None);
}
