//! Injected-code TinyStories codec diagnostic; no model-family admission.

mod ncm_probe;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mlx_sys::{MlxDtype, add, eval, load_safetensors, zeros};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use ncm_probe::*;

fn execute(model: &Path, trace_path: &Path, reference_path: &Path) -> Result<Value, String> {
    let bytes = read_json(trace_path)?;
    let trace: Trace = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    trace.validate()?;
    let reference_sha256 = reference_digest(reference_path)?;
    let references = load_safetensors(reference_path, None)?;
    let shapes = codec_reference_shapes();
    if references.len() != shapes.len() {
        return Err("unexpected reference tensor set".into());
    }
    for (name, shape) in shapes {
        let value = references
            .get(&name)
            .ok_or_else(|| format!("missing reference {name}"))?;
        validate_tensor(&name, value, &shape)?;
    }
    let reference = |name: &str| {
        references
            .get(name)
            .ok_or_else(|| format!("missing reference {name}"))
    };
    let bounds = reference("numerical.max_abs_bounds")?;
    eval(&[bounds]);
    let bounds = bounds.data_f32().to_vec();
    if bounds.iter().any(|&bound| bound <= 0.0) {
        return Err("numerical bounds must be finite and positive".into());
    }
    if token_ids(reference("decoder.logits")?)? != trace.tokens[0] {
        return Err("reference logits and trace BPE IDs do not identify the same case".into());
    }
    let stored_tokens = reference("decoder.tokens")?;
    eval(&[stored_tokens]);
    if !stored_tokens
        .data_f32()
        .iter()
        .zip(&trace.tokens[0])
        .all(|(&a, &b)| a == b as f32)
    {
        return Err("reference token IDs and trace disagree".into());
    }
    let codec = Codec::load(model)?;
    let mut latent = zeros(&[1, LENGTH, WIDTH], MlxDtype::Float32, None);
    let mut errors = Vec::new();
    for (level, codes) in trace.codes.iter().enumerate() {
        let contribution = codec.contribution(&codes[0], level);
        let name = format!("contribution.{level}");
        errors.push(compare(
            &name,
            &contribution,
            reference(&name)?,
            bounds[2 * level],
        )?);
        latent = add(&latent, &contribution, None);
        let name = format!("accumulated.{level}");
        errors.push(compare(
            &name,
            &latent,
            reference(&name)?,
            bounds[2 * level + 1],
        )?);
    }
    let logits = codec.decode_logits(&latent);
    errors.push(compare(
        "decoder.logits",
        &logits,
        reference("decoder.logits")?,
        bounds[24],
    )?);
    let tokens = token_ids(&logits)?;
    let mismatches = tokens
        .iter()
        .zip(&trace.tokens[0])
        .filter(|(a, b)| a != b)
        .count();
    let numerical_match = errors.iter().all(|value| value["within_bound"] == true);
    Ok(json!({
        "schema_version": "ax.ncm_codec_probe.v1",
        "scope": "native injected-code codec diagnostic only",
        "codec_sha256": CODEC_SHA256, "spec_sha256": SPEC_SHA256,
        "trace_sha256": format!("{:x}", Sha256::digest(&bytes)),
        "reference_sha256": reference_sha256,
        "artifact_revision": "2d5f137713fe047abb6fcffe780aad419486f3dd",
        "mlx_version": mlx_sys::runtime_version()?,
        "token_mismatches": mismatches, "tokens": [tokens],
        "exact_token_match": mismatches == 0,
        "numerical_match": numerical_match,
        "passed": mismatches == 0 && numerical_match,
        "all_compared_values_finite": true,
        "comparisons": errors,
    }))
}

fn run() -> Result<bool, String> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "Usage: ncm_codec_probe <model-dir> <trace.json> <reference.safetensors> <output.json>"
        );
        return Ok(true);
    }
    if args.len() != 4 {
        return Err(
            "expected model directory, trace, reference and output paths; see --help".into(),
        );
    }
    let paths = args.iter().map(PathBuf::from).collect::<Vec<_>>();
    if paths[3].exists() {
        return Err("output path already exists".into());
    }
    let report = execute(&paths[0], &paths[1], &paths[2])?;
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&paths[3])
        .map_err(|e| e.to_string())?;
    output.write_all(&bytes).map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|e| e.to_string())?
    );
    Ok(report["passed"] == true)
}

fn main() -> ExitCode {
    match std::panic::catch_unwind(run) {
        Ok(Ok(true)) => ExitCode::SUCCESS,
        Ok(Ok(false)) => ExitCode::FAILURE,
        Ok(Err(error)) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
        Err(_) => ExitCode::from(3),
    }
}
