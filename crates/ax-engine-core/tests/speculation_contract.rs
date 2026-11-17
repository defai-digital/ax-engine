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

#[test]
fn mirrored_contract_assets_match_their_manifest() -> Result<(), Box<dyn std::error::Error>> {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/engine-contract-manifest.json"))?;
    let assets = [
        (
            "speculation-v1.json",
            include_bytes!("fixtures/speculation-v1.json").as_slice(),
        ),
        (
            "benchmark-result-v1.schema.json",
            include_bytes!("fixtures/benchmark-result-v1.schema.json").as_slice(),
        ),
    ];
    for (name, bytes) in assets {
        let actual = format!("{:x}", Sha256::digest(bytes));
        assert_eq!(
            manifest["sha256"][name].as_str(),
            Some(actual.as_str()),
            "{name}"
        );
    }
    Ok(())
}
