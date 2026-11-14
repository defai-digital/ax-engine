//! Pinned TinyStories NCM teacher-forced levels and bounded greedy diagnostic.
//! No model registration, user prompting, RNG sampling or serving integration.

mod ncm_probe;

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mlx_sys::{MlxArray, add, contiguous, eval, load_safetensors};
#[cfg(test)]
use mlx_sys::{MlxDtype, astype, reshape};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use ax_engine_mlx::ncm::generator::*;
use ncm_probe::*;

fn reference_shapes() -> BTreeMap<String, Vec<i32>> {
    let mut shapes = BTreeMap::new();
    shapes.insert("trace.sha256".into(), vec![32]);
    shapes.insert("prefix.embedding".into(), vec![1, 1, HIDDEN]);
    shapes.insert("first.input".into(), vec![1, 9, HIDDEN]);
    shapes.insert("medium.input".into(), vec![1, 25, HIDDEN]);
    shapes.insert("medium.block.0".into(), vec![1, 25, HIDDEN]);
    for i in 0..DEPTH {
        shapes.insert(format!("first.block.{i}"), vec![1, 9, HIDDEN]);
    }
    for (level, &length) in LEVELS.iter().enumerate() {
        shapes.insert(format!("greedy.codes.{level}"), vec![1, length as i32]);
        if level > 0 {
            shapes.insert(
                format!("teacher.pool.{level}"),
                vec![1, length as i32, WIDTH],
            );
            shapes.insert(
                format!("teacher.logits.{level}"),
                vec![1, length as i32, 4096],
            );
            shapes.insert(
                format!("greedy.logits.{level}"),
                vec![1, length as i32, 4096],
            );
        }
    }
    for mode in ["teacher", "greedy"] {
        shapes.insert(format!("{mode}.tokens"), vec![1, LENGTH]);
        shapes.insert(format!("{mode}.decoder"), vec![1, LENGTH, 8192]);
    }
    shapes
}

fn ids(array: &MlxArray, length: usize, vocabulary: u32) -> Result<Vec<u32>, String> {
    if array.shape() != [1, length as i32] {
        return Err("wrong ID shape".into());
    }
    let array = contiguous(array, None);
    eval(&[&array]);
    let values = array.data_f32();
    if values
        .iter()
        .any(|&v| !v.is_finite() || v < 0.0 || v >= vocabulary as f32 || v.fract() != 0.0)
    {
        return Err("reference IDs are not bounded integers".into());
    }
    Ok(values.iter().map(|&v| v as u32).collect())
}

fn differences(a: &[u32], b: &[u32]) -> Result<usize, String> {
    if a.len() != b.len() {
        return Err("ID sequence lengths differ".into());
    }
    Ok(a.iter().zip(b).filter(|(x, y)| x != y).count())
}

fn execute(directory: &Path, trace_path: &Path, reference_path: &Path) -> Result<Value, String> {
    let raw = read_json(trace_path)?;
    let metadata: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if metadata["prefix_tokens"] != json!([[0]]) {
        return Err("only the fixed unconditional prefix [[0]] is supported".into());
    }
    let trace: Trace = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    trace.validate()?;
    let file = File::open(reference_path).map_err(|e| e.to_string())?;
    let mut reference_bytes = Vec::new();
    file.take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut reference_bytes)
        .map_err(|e| e.to_string())?;
    if reference_bytes.len() > 64 * 1024 * 1024 {
        return Err("generator reference exceeds 64 MiB".into());
    }
    let reference_hash = format!("{:x}", Sha256::digest(&reference_bytes));
    let refs = load_safetensors(reference_path, None)?;
    let shapes = reference_shapes();
    if refs.len() != shapes.len() {
        return Err("unexpected generator reference tensor set".into());
    }
    for (name, shape) in shapes {
        validate_tensor(
            &name,
            refs.get(&name).ok_or_else(|| format!("missing {name}"))?,
            &shape,
        )?;
    }
    let digest = Sha256::digest(&raw);
    eval(&[&refs["trace.sha256"]]);
    if !refs["trace.sha256"]
        .data_f32()
        .iter()
        .zip(digest.iter())
        .all(|(&a, &b)| a == b as f32)
    {
        return Err("reference trace hash mismatch".into());
    }
    for (level, &length) in LEVELS.iter().enumerate() {
        let codes = ids(&refs[&format!("greedy.codes.{level}")], length, 4096)?;
        if level > 0 && codes != chosen(&refs[&format!("greedy.logits.{level}")])? {
            return Err(format!(
                "greedy reference logits and codes disagree at level {level}"
            ));
        }
    }
    if ids(&refs["greedy.codes.0"], 1, 4096)? != trace.codes[0][0] {
        return Err("greedy initial code differs from the trace".into());
    }
    if ids(&refs["teacher.tokens"], 128, 8192)? != trace.tokens[0] {
        return Err("teacher output is not bound to trace tokens".into());
    }
    ids(&refs["greedy.tokens"], 128, 8192)?;
    for mode in ["teacher", "greedy"] {
        if ids(&refs[&format!("{mode}.tokens")], 128, 8192)?
            != token_ids(&refs[&format!("{mode}.decoder")])?
        {
            return Err(format!("{mode} reference decoder and tokens disagree"));
        }
    }
    let codec = Codec::load(directory)?;
    let generator = Generator::load(directory, &codec)?;
    let mut comparisons = Vec::new();
    let mut check =
        |name: &str, actual: &MlxArray, reference: &str, depth: f32| -> Result<(), String> {
            comparisons.push(compare_with_budget(
                name,
                actual,
                &refs[reference],
                f32::MAX,
                depth,
            )?);
            Ok(())
        };
    check(
        "prefix.embedding",
        generator.prefix_embedding(),
        "prefix.embedding",
        32.0,
    )?;
    let (cosine, sine) = rotary_tables(9);
    let mask = prefix_mask(9);
    for i in 0..DEPTH {
        let input = if i == 0 {
            &refs["first.input"]
        } else {
            &refs[&format!("first.block.{}", i - 1)]
        };
        check(
            &format!("local.block.{i}"),
            &generator.block(input, i, &cosine, &sine, &mask),
            &format!("first.block.{i}"),
            32.0,
        )?;
    }
    let (medium_cosine, medium_sine) = rotary_tables(25);
    check(
        "local.medium.block.0",
        &generator.block(
            &refs["medium.input"],
            0,
            &medium_cosine,
            &medium_sine,
            &prefix_mask(25),
        ),
        "medium.block.0",
        32.0,
    )?;
    let mut latent = codec.contribution(&trace.codes[0][0], 0);
    let mut teacher_mismatches = Vec::new();
    for level in 1..12 {
        let pool = area_pool(&latent, LEVELS[level]);
        check(
            &format!("teacher.pool.{level}"),
            &pool,
            &format!("teacher.pool.{level}"),
            32.0,
        )?;
        let forward = generator.forward(&codec, level, &pool);
        if level == 1 {
            check("first.input", &forward.input, "first.input", 32.0)?;
            for (i, block) in forward.blocks.iter().enumerate() {
                check(
                    &format!("chain.block.{i}"),
                    block,
                    &format!("first.block.{i}"),
                    32.0 * (i + 1) as f32,
                )?;
            }
        }
        if level == 3 {
            check("medium.input", &forward.input, "medium.input", 32.0)?;
            check(
                "chain.medium.block.0",
                &forward.blocks[0],
                "medium.block.0",
                32.0,
            )?;
        }
        check(
            &format!("teacher.logits.{level}"),
            &forward.logits,
            &format!("teacher.logits.{level}"),
            448.0,
        )?;
        teacher_mismatches.push(differences(
            &chosen(&forward.logits)?,
            &chosen(&refs[&format!("teacher.logits.{level}")])?,
        )?);
        latent = add(
            &latent,
            &codec.contribution(&trace.codes[level][0], level),
            None,
        );
    }
    let teacher_decoder = codec.decode_logits(&latent);
    check("teacher.decoder", &teacher_decoder, "teacher.decoder", 32.0)?;
    let teacher_tokens = differences(&token_ids(&teacher_decoder)?, &trace.tokens[0])?;
    latent = codec.contribution(&trace.codes[0][0], 0);
    let mut greedy_codes = vec![trace.codes[0][0].clone()];
    let mut greedy_mismatches = Vec::new();
    for level in 1..12 {
        let forward = generator.forward(&codec, level, &area_pool(&latent, LEVELS[level]));
        check(
            &format!("greedy.logits.{level}"),
            &forward.logits,
            &format!("greedy.logits.{level}"),
            448.0,
        )?;
        let codes = chosen(&forward.logits)?;
        greedy_mismatches.push(differences(
            &codes,
            &ids(&refs[&format!("greedy.codes.{level}")], LEVELS[level], 4096)?,
        )?);
        latent = add(&latent, &codec.contribution(&codes, level), None);
        greedy_codes.push(codes);
    }
    let decoder = codec.decode_logits(&latent);
    check("greedy.decoder", &decoder, "greedy.decoder", 32.0)?;
    let tokens = token_ids(&decoder)?;
    let greedy_tokens = differences(&tokens, &ids(&refs["greedy.tokens"], 128, 8192)?)?;
    let numerical_match = comparisons
        .iter()
        .all(|entry| entry["within_bound"] == true);
    let exact_match = teacher_tokens == 0
        && greedy_tokens == 0
        && teacher_mismatches
            .iter()
            .chain(&greedy_mismatches)
            .all(|&count| count == 0);
    Ok(json!({
        "schema_version":"ax.ncm_generator_probe.v1", "scope":"native teacher-forced levels and bounded greedy diagnostic only",
        "generator_sha256":GENERATOR_SHA256,"codec_sha256":CODEC_SHA256,"spec_sha256":SPEC_SHA256,
        "levels_compared":(1..12).collect::<Vec<_>>(),
        "numerical_policy":{"epsilon":f32::EPSILON,"peak_floor":1.0,"local_pool_input":32,"chained_block":"32 * depth","generator_logits":448,"codec_decoder":32},
        "trace_sha256":format!("{:x}",digest),"reference_sha256":reference_hash,"mlx_version":mlx_sys::runtime_version()?,
        "numerical_match":numerical_match,"exact_match":exact_match,"passed":numerical_match && exact_match,
        "teacher_argmax_mismatches":teacher_mismatches,"greedy_code_mismatches":greedy_mismatches,
        "teacher_token_mismatches":teacher_tokens,"greedy_token_mismatches":greedy_tokens,
        "greedy_codes":greedy_codes,"greedy_tokens":[tokens],"comparisons":comparisons,
    }))
}

fn run() -> Result<bool, String> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "Usage: ncm_generator_probe <model-dir> <trace.json> <reference.safetensors> <output.json>"
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
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&paths[3])
        .map_err(|e| e.to_string())?;
    output
        .write_all(&serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn area_pool_uses_overlapping_bins_for_noninteger_scales() {
        let input = float_array(&[0., 3., 6., 9., 12.], &[1, 5, 1]);
        let output = contiguous(&area_pool(&input, 3), None);
        eval(&[&output]);
        assert_eq!(output.data_f32(), &[1.5, 6., 10.5]);
    }

    #[test]
    fn integer_reference_ids_reject_fractional_and_out_of_range_values() {
        assert!(ids(&float_array(&[0.5], &[1, 1]), 1, 4096).is_err());
        assert!(ids(&float_array(&[4096.], &[1, 1]), 1, 4096).is_err());
        assert!(differences(&[0], &[0, 1]).is_err());
    }

    #[test]
    fn prefix_mask_seals_the_prefix_but_preserves_target_bidirectionality() {
        let mask = astype(&prefix_mask(3), MlxDtype::Float32, None);
        eval(&[&mask]);
        assert_eq!(mask.data_f32(), &[1., 0., 0., 1., 1., 1., 1., 1., 1.]);
    }

    #[test]
    fn rotary_prefix_and_target_origins_both_start_at_zero() {
        let (cosine, sine) = rotary_tables(3);
        eval(&[&cosine, &sine]);
        assert_eq!(&cosine.data_f32()[..128], &[1.; 128]);
        assert_eq!(&sine.data_f32()[..128], &[0.; 128]);
        assert!((cosine.data_f32()[128] - 1.0f32.cos()).abs() < 1e-7);
    }

    #[test]
    fn code_argmax_rejects_flattened_layouts_and_preserves_positions() {
        let x = float_array(&vec![0.; 8192], &[1, 2, 4096]);
        assert_eq!(chosen(&x).unwrap(), vec![0, 0]);
        let mut tied = vec![0.; 8192];
        tied[3] = 1.;
        tied[7] = 1.;
        tied[4096 + 5] = 2.;
        tied[4096 + 10] = 2.;
        assert_eq!(
            chosen(&float_array(&tied, &[1, 2, 4096])).unwrap(),
            vec![3, 5]
        );
        assert!(chosen(&reshape(&x, &[2, 4096], None)).is_err());
    }

    #[test]
    fn tied_head_standardizes_distance_before_inverse_temperature() {
        let q = float_array(&[2., 1.], &[1, 1, 2]);
        let book = float_array(&[1., 0., 0., 2.], &[2, 2]);
        let norms = MlxArray::from_f32_slice(&[1., 4.]);
        let out = contiguous(
            &tied_logits(
                &q,
                &book,
                &norms,
                &MlxArray::from_f32(2.5),
                &MlxArray::from_f32(2.0f32.ln()),
            ),
            None,
        );
        eval(&[&out]);
        assert!((out.data_f32()[0] - 2.4).abs() < 1e-6);
        assert_eq!(out.data_f32()[1], 0.);
    }

    #[test]
    fn split_half_rotation_preserves_sign_and_pairing() {
        let mut values = vec![0.; 12 * 3 * 64];
        for row in 0..3 {
            values[row * 64] = 1.;
        }
        let x = float_array(&values, &[1, 12, 3, 64]);
        let (cosine, sine) = rotary_tables(3);
        let output = contiguous(&rotate(&x, &cosine, &sine), None);
        eval(&[&output]);
        assert_eq!(output.data_f32()[0], 1.);
        assert_eq!(output.data_f32()[64], 1.);
        assert!((output.data_f32()[128] - 1.0f32.cos()).abs() < 1e-7);
        assert!((output.data_f32()[160] - 1.0f32.sin()).abs() < 1e-7);
    }

    #[test]
    fn integer_pooling_uses_disjoint_bins() {
        let input = float_array(&[0., 1., 2., 3., 4., 5., 6., 7.], &[1, 8, 1]);
        let output = contiguous(&area_pool(&input, 4), None);
        eval(&[&output]);
        assert_eq!(output.data_f32(), &[0.5, 2.5, 4.5, 6.5]);
    }
}
