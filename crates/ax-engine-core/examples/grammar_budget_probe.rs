//! Tokenizer-only completion-budget diagnostic; does not execute a model.

use ax_engine_core::grammar::budget::BudgetWorkspace;
use ax_engine_core::grammar::{JsonGrammar, TokenVocab};
use serde_json::json;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected tokenizer.json path")?;
    let tokenizer: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let base = tokenizer["model"]["vocab"]
        .as_object()
        .ok_or("expected BPE vocabulary")?;
    let max_id = base
        .values()
        .filter_map(|v| v.as_u64())
        .chain(
            tokenizer["added_tokens"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v["id"].as_u64()),
        )
        .max()
        .ok_or("empty vocabulary")?;
    let vocab = TokenVocab::from_tokenizer_json(&tokenizer, max_id as usize + 1)?;
    let cases = [
        ("empty_object", json!({"const":{}})),
        ("object", json!({"type":"object"})),
        (
            "answer",
            json!({"type":"object", "properties":{"answer":{"type":"boolean"}},
            "required":["answer"], "additionalProperties":false}),
        ),
        ("nested", json!({"const":{"data":[true, "\u{6c34}"]}})),
        (
            "five_booleans",
            json!({"type":"object", "properties":{"items":{
            "type":"array", "items":{"type":"boolean"}, "minItems":5, "maxItems":5
        }}, "required":["items"], "additionalProperties":false}),
        ),
    ];
    for (name, schema) in cases {
        let grammar = JsonGrammar::from_schema(&schema)?;
        let mut workspace = BudgetWorkspace::default();
        let mut matcher = grammar.matcher();
        if name == "five_booleans" {
            assert!(matcher.advance_bytes(&grammar, b"{"));
        }
        for remaining in [1, 4, 16, 64] {
            let start = Instant::now();
            let result = workspace.mask(&vocab, &grammar, &matcher, &[], remaining);
            let (status, allowed) = match result {
                Ok(mask) => (
                    "ok".to_string(),
                    mask.iter().map(|w| w.count_ones()).sum::<u32>(),
                ),
                Err(error) => (error.to_string(), 0),
            };
            println!(
                "{}",
                json!({"case":name, "remaining":remaining, "status":status,
                "allowed":allowed, "elapsed_ms":start.elapsed().as_secs_f64()*1000.0})
            );
        }
    }
    Ok(())
}
