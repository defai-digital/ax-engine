use ax_engine_core::speculation::{CommitError, plan_greedy_commit};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    drafts: Vec<u32>,
    predicted: Vec<u32>,
    budget: usize,
    eos: Vec<u32>,
    accepted_drafts: Option<usize>,
    emitted: Option<Vec<u32>>,
    retained_draft_rows: Option<usize>,
    error: Option<String>,
}

#[test]
fn versioned_speculative_commit_contract() -> Result<(), Box<dyn std::error::Error>> {
    let corpus: Corpus = serde_json::from_str(include_str!("fixtures/speculation-v1.json"))?;
    assert_eq!(corpus.schema_version, "ax.engine.speculation.v1");
    for case in corpus.cases {
        let actual = plan_greedy_commit(&case.drafts, &case.predicted, case.budget, &case.eos);
        if let Some(error) = case.error {
            assert_eq!(error, "prediction_count", "{}", case.name);
            assert_eq!(actual, Err(CommitError::PredictionCount), "{}", case.name);
        } else {
            let actual = actual?;
            assert_eq!(
                Some(actual.accepted_drafts),
                case.accepted_drafts,
                "{}",
                case.name
            );
            assert_eq!(Some(actual.emitted), case.emitted, "{}", case.name);
            assert_eq!(
                Some(actual.retained_draft_rows),
                case.retained_draft_rows,
                "{}",
                case.name
            );
        }
    }
    Ok(())
}

/// Mirrored assets, keyed by their names in `engine-contract-manifest.json`.
const MIRRORED_ASSETS: [(&str, &[u8]); 2] = [
    (
        "speculation-v1.json",
        include_bytes!("fixtures/speculation-v1.json"),
    ),
    (
        "benchmark-result-v1.schema.json",
        include_bytes!("fixtures/benchmark-result-v1.schema.json"),
    ),
];

fn local_manifest() -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_str(include_str!("fixtures/engine-contract-manifest.json"))
}

#[test]
fn mirrored_contract_assets_match_their_manifest() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = local_manifest()?;
    assert_eq!(manifest["schema_version"], "ax.engine.contract-assets.v1");
    let pinned = manifest["sha256"]
        .as_object()
        .ok_or("manifest has no sha256 map")?;
    let mut pinned_names: Vec<&str> = pinned.keys().map(String::as_str).collect();
    let mut mirrored_names: Vec<&str> = MIRRORED_ASSETS.iter().map(|(name, _)| *name).collect();
    pinned_names.sort_unstable();
    mirrored_names.sort_unstable();
    assert_eq!(
        pinned_names, mirrored_names,
        "every pinned asset is mirrored and checked"
    );
    for (name, bytes) in MIRRORED_ASSETS {
        let actual = format!("{:x}", Sha256::digest(bytes));
        assert_eq!(pinned[name].as_str(), Some(actual.as_str()), "{name}");
    }
    Ok(())
}

/// Compares the mirror with the upstream `ax-engine-sglang/contracts` checkout
/// when one is available: `AX_ENGINE_SGLANG_CONTRACTS`, else a sibling clone.
/// Set `AX_ENGINE_REQUIRE_SGLANG_CONTRACTS=1` to fail instead of skipping.
#[test]
fn mirrored_contract_assets_match_upstream_when_available() -> Result<(), Box<dyn std::error::Error>>
{
    let upstream = std::env::var_os("AX_ENGINE_SGLANG_CONTRACTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../ax-engine-sglang/contracts")
        });
    if !upstream.join("manifest.json").is_file() {
        let required = std::env::var("AX_ENGINE_REQUIRE_SGLANG_CONTRACTS").is_ok_and(|v| v == "1");
        assert!(
            !required,
            "upstream contracts not found at {}",
            upstream.display()
        );
        eprintln!(
            "skipping upstream contract drift check: {} not found",
            upstream.display()
        );
        return Ok(());
    }
    let upstream_manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(upstream.join("manifest.json"))?)?;
    assert_eq!(
        local_manifest()?,
        upstream_manifest,
        "manifest drifted from upstream"
    );
    for (name, bytes) in MIRRORED_ASSETS {
        assert_eq!(
            std::fs::read(upstream.join(name))?.as_slice(),
            bytes,
            "{name} drifted from upstream"
        );
    }
    Ok(())
}
