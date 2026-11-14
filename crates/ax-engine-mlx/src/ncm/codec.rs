//! Independently authored pinned TinyStories codec and diagnostic primitives.

//!
//! The graph is independently expressed through AX's MLX primitives from the
//! pinned tensor contract and additive reconstruction equations in arXiv:2610.14162.
//! External codec outputs are comparison data only, never a computation fallback.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[cfg(test)]
use mlx_sys::eval;

use mlx_sys::{
    MlxArray, MlxDtype, add, argmax, astype, contiguous, conv1d, gelu, layer_norm,
    load_safetensors, matmul, multiply, reshape, rms_norm, scaled_dot_product_attention, slice,
    take, transpose, try_eval, zeros,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const LEVELS: [usize; 12] = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128];
pub const WIDTH: i32 = 384;
pub const LENGTH: i32 = 128;
const EPS: f32 = 1e-6;
pub const CODEC_SHA256: &str = "b28e6c7dc804fcd056ca98ff8dc818484a8bb17d52169c88205eb3d27b74c593";
pub const SPEC_SHA256: &str = "e9cf868d06141400d061a8bee2e8d95df541fb90548d0b3e0d43e314f1479b1f";

#[derive(Deserialize)]
pub struct Trace {
    pub codes: Vec<Vec<Vec<u32>>>,
    pub tokens: Vec<Vec<u32>>,
}

impl Trace {
    pub fn validate(&self) -> Result<(), String> {
        if self.codes.len() != LEVELS.len() {
            return Err("expected exactly 12 latent levels".into());
        }
        for (level, (rows, length)) in self.codes.iter().zip(LEVELS).enumerate() {
            if rows.len() != 1 || rows[0].len() != length {
                return Err(format!("level {level}: expected one row of {length} codes"));
            }
            if rows[0].iter().any(|&code| code >= 4096) {
                return Err(format!("level {level}: code outside the latent vocabulary"));
            }
        }
        if self.tokens.len() != 1
            || self.tokens[0].len() != LENGTH as usize
            || self.tokens[0].iter().any(|&token| token >= 8192)
        {
            return Err("expected one row of 128 BPE IDs in 0..8192".into());
        }
        Ok(())
    }
}

pub fn read_json(path: &Path) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(2 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() > 2 * 1024 * 1024 {
        return Err("JSON input exceeds 2 MiB".into());
    }
    Ok(bytes)
}

pub fn verify_file(path: &Path, size: u64, expected: &str) -> Result<(), String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    if file.metadata().map_err(|e| e.to_string())?.len() != size {
        return Err(format!("{}: pinned byte length mismatch", path.display()));
    }
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let length = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if length == 0 {
            break;
        }
        hash.update(&buffer[..length]);
    }
    if format!("{:x}", hash.finalize()) != expected {
        return Err(format!("{}: pinned SHA-256 mismatch", path.display()));
    }
    Ok(())
}

fn expected_shapes() -> BTreeMap<String, Vec<i32>> {
    let mut shapes = BTreeMap::new();
    for k in 0..12 {
        shapes.insert(format!("codebooks.{k}"), vec![4096, WIDTH]);
        shapes.insert(format!("phi.{k}.weight"), vec![WIDTH, 3, WIDTH]);
        shapes.insert(format!("phi.{k}.bias"), vec![WIDTH]);
    }
    for (direction, strides) in [("up0", [2, 4, 4, 4]), ("down0", [4, 4, 4, 2])] {
        for (i, stride) in strides.iter().enumerate() {
            let shape = if direction == "up0" {
                vec![WIDTH, stride * WIDTH]
            } else {
                vec![stride * WIDTH, WIDTH]
            };
            shapes.insert(format!("{direction}.{i}.weight"), shape);
            shapes.insert(format!("{direction}.{i}.bias"), vec![WIDTH]);
        }
    }
    for norm in ["dec_block.ln1", "dec_block.ln2"] {
        shapes.insert(format!("{norm}.weight"), vec![WIDTH]);
        shapes.insert(format!("{norm}.bias"), vec![WIDTH]);
    }
    for norm in ["q_norm", "k_norm"] {
        shapes.insert(format!("dec_block.attn.{norm}.weight"), vec![64]);
    }
    for (name, output, input) in [
        ("dec_block.attn.qkv", 1152, WIDTH),
        ("dec_block.attn.proj", WIDTH, WIDTH),
        ("dec_block.fc", 1536, WIDTH),
        ("dec_block.out", WIDTH, 1536),
    ] {
        shapes.insert(format!("{name}.weight"), vec![output, input]);
        shapes.insert(format!("{name}.bias"), vec![output]);
    }
    shapes.insert("tok.weight".into(), vec![8192, WIDTH]);
    shapes.insert("pos.weight".into(), vec![LENGTH, WIDTH]);
    shapes.insert("dec_ln.weight".into(), vec![WIDTH]);
    shapes.insert("dec_head.weight".into(), vec![8192, WIDTH]);
    shapes
}

pub fn validate_tensor(name: &str, tensor: &MlxArray, shape: &[i32]) -> Result<(), String> {
    if tensor.dtype() != MlxDtype::Float32 || tensor.shape() != shape {
        return Err(format!("{name}: expected F32 {shape:?}"));
    }
    let tensor = contiguous(tensor, None);
    try_eval(&[&tensor])?;
    if tensor.data_f32().iter().any(|value| !value.is_finite()) {
        return Err(format!("{name}: non-finite value"));
    }
    Ok(())
}

pub struct Codec {
    weights: HashMap<String, MlxArray>,
}

impl Codec {
    pub fn load(directory: &Path) -> Result<Self, String> {
        Self::load_verified(
            &directory.join("codec.safetensors"),
            &directory.join("spec.json"),
        )
    }

    pub fn load_verified(codec: &Path, spec: &Path) -> Result<Self, String> {
        verify_file(spec, 3067, SPEC_SHA256)?;
        verify_file(codec, 145745190, CODEC_SHA256)?;
        let weights = load_safetensors(codec, None)?;
        let shapes = expected_shapes();
        if weights.len() != shapes.len() {
            return Err("unexpected codec tensor set".into());
        }
        for (name, shape) in shapes {
            let tensor = weights
                .get(&name)
                .ok_or_else(|| format!("missing {name}"))?;
            validate_tensor(&name, tensor, &shape)?;
        }
        Ok(Self { weights })
    }

    pub fn weight(&self, name: &str) -> &MlxArray {
        &self.weights[name]
    }

    pub fn linear(&self, x: &MlxArray, prefix: &str) -> MlxArray {
        let weight = self.weight(&format!("{prefix}.weight"));
        let y = matmul(x, &transpose(weight, &[1, 0], None), None);
        match self.weights.get(&format!("{prefix}.bias")) {
            Some(bias) => add(&y, bias, None),
            None => y,
        }
    }

    pub fn norm(&self, x: &MlxArray, prefix: &str) -> MlxArray {
        let zero = zeros(&[WIDTH], MlxDtype::Float32, None);
        layer_norm(
            x,
            self.weight(&format!("{prefix}.weight")),
            self.weights.get(&format!("{prefix}.bias")).unwrap_or(&zero),
            EPS,
            None,
        )
    }

    pub fn contribution(&self, codes: &[u32], level: usize) -> MlxArray {
        let ids = float_array(
            &codes.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &[1, codes.len() as i32],
        );
        let ids = astype(&ids, MlxDtype::Uint32, None);
        let mut x = take(self.weight(&format!("codebooks.{level}")), &ids, 0, None);
        if level == 0 {
            for (i, stride) in [2, 4, 4, 4].iter().enumerate() {
                x = expand_sequence(
                    &x,
                    self.weight(&format!("up0.{i}.weight")),
                    self.weight(&format!("up0.{i}.bias")),
                    *stride,
                );
            }
        } else {
            x = resize_linear(&x, LENGTH);
        }
        let convolution = conv1d(
            &x,
            self.weight(&format!("phi.{level}.weight")),
            1,
            1,
            1,
            1,
            None,
        );
        let convolution = add(
            &convolution,
            self.weight(&format!("phi.{level}.bias")),
            None,
        );
        let half = MlxArray::from_f32(0.5);
        add(
            &multiply(&x, &half, None),
            &multiply(&convolution, &half, None),
            None,
        )
    }

    pub fn decode_logits(&self, latent: &MlxArray) -> MlxArray {
        let normed = self.norm(latent, "dec_block.ln1");
        let qkv = self.linear(&normed, "dec_block.attn.qkv");
        let head = |offset: i32| {
            let x = slice(
                &qkv,
                &[0, 0, offset],
                &[1, LENGTH, offset + WIDTH],
                &[1, 1, 1],
                None,
            );
            reshape(&x, &[1, LENGTH, 6, 64], None)
        };
        let q = rms_norm(
            &head(0),
            Some(self.weight("dec_block.attn.q_norm.weight")),
            EPS,
            None,
        );
        let k = rms_norm(
            &head(WIDTH),
            Some(self.weight("dec_block.attn.k_norm.weight")),
            EPS,
            None,
        );
        let q = transpose(&q, &[0, 2, 1, 3], None);
        let k = transpose(&k, &[0, 2, 1, 3], None);
        let v = transpose(&head(2 * WIDTH), &[0, 2, 1, 3], None);
        let attended = scaled_dot_product_attention(&q, &k, &v, 0.125, false, None);
        let attended = reshape(
            &transpose(&attended, &[0, 2, 1, 3], None),
            &[1, LENGTH, WIDTH],
            None,
        );
        let residual = add(latent, &self.linear(&attended, "dec_block.attn.proj"), None);
        let ff = self.linear(&self.norm(&residual, "dec_block.ln2"), "dec_block.fc");
        let ff = self.linear(&gelu(&ff, None), "dec_block.out");
        let hidden = self.norm(&add(&residual, &ff, None), "dec_ln");
        self.linear(&hidden, "dec_head")
    }
}

pub fn float_array(values: &[f32], shape: &[i32]) -> MlxArray {
    reshape(&MlxArray::from_f32_slice(values), shape, None)
}

fn expand_sequence(x: &MlxArray, weight: &MlxArray, bias: &MlxArray, stride: i32) -> MlxArray {
    let shape = x.shape();
    let expanded = matmul(x, weight, None);
    add(
        &reshape(&expanded, &[shape[0], shape[1] * stride, shape[2]], None),
        bias,
        None,
    )
}

/// Half-pixel centers with clamped endpoints; weights are computed in F32.
fn resize_linear(x: &MlxArray, output: i32) -> MlxArray {
    let input = x.shape()[1];
    if input == output {
        return x.clone();
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut weight = Vec::new();
    for i in 0..output {
        let p = (i as f32 + 0.5) * (input as f32 / output as f32) - 0.5;
        let lo = p.floor();
        left.push(lo.clamp(0.0, (input - 1) as f32));
        right.push((lo + 1.0).clamp(0.0, (input - 1) as f32));
        weight.push(p - lo);
    }
    let ids = |v: &[f32]| astype(&MlxArray::from_f32_slice(v), MlxDtype::Uint32, None);
    let lower = take(x, &ids(&left), 1, None);
    let upper = take(x, &ids(&right), 1, None);
    let complement = weight.iter().map(|value| 1.0 - value).collect::<Vec<_>>();
    let complement = float_array(&complement, &[1, output, 1]);
    let weight = float_array(&weight, &[1, output, 1]);
    add(
        &multiply(&lower, &complement, None),
        &multiply(&upper, &weight, None),
        None,
    )
}

pub fn compare(
    name: &str,
    actual: &MlxArray,
    reference: &MlxArray,
    bound: f32,
) -> Result<Value, String> {
    compare_with_budget(name, actual, reference, bound, 32.0)
}

pub fn compare_with_budget(
    name: &str,
    actual: &MlxArray,
    reference: &MlxArray,
    bound: f32,
    depth_budget: f32,
) -> Result<Value, String> {
    let shape = actual.shape();
    validate_tensor(name, reference, &shape)?;
    validate_tensor(name, actual, &shape)?;
    let actual = contiguous(actual, None);
    let reference = contiguous(reference, None);
    try_eval(&[&actual, &reference])?;
    let mut maximum = 0.0f64;
    let mut squared = 0.0f64;
    let mut reference_peak = 0.0f64;
    for (&a, &b) in actual.data_f32().iter().zip(reference.data_f32()) {
        let difference = (a as f64 - b as f64).abs();
        maximum = maximum.max(difference);
        squared += difference * difference;
        reference_peak = reference_peak.max((b as f64).abs());
    }
    // Reference metadata may tighten, but cannot widen this same-GPU F32 budget.
    let bound = bound.min(depth_budget * f32::EPSILON * (reference_peak as f32).max(1.0));
    Ok(json!({
        "name": name, "shape": shape, "max_abs_error": maximum,
        "rmse": (squared / actual.data_f32().len() as f64).sqrt(),
        "reference_peak": reference_peak,
        "max_abs_bound": bound, "within_bound": maximum <= bound as f64,
    }))
}

fn comparison_names() -> Vec<String> {
    (0..12)
        .flat_map(|level| {
            [
                format!("contribution.{level}"),
                format!("accumulated.{level}"),
            ]
        })
        .chain(std::iter::once("decoder.logits".into()))
        .collect()
}

pub fn codec_reference_shapes() -> BTreeMap<String, Vec<i32>> {
    let mut shapes = comparison_names()
        .into_iter()
        .map(|name| {
            let width = if name == "decoder.logits" {
                8192
            } else {
                WIDTH
            };
            (name, vec![1, LENGTH, width])
        })
        .collect::<BTreeMap<_, _>>();
    shapes.insert("decoder.tokens".into(), vec![1, LENGTH]);
    shapes.insert("numerical.max_abs_bounds".into(), vec![25]);
    shapes
}

pub fn token_ids(logits: &MlxArray) -> Result<Vec<u32>, String> {
    let tokens = astype(&argmax(logits, None), MlxDtype::Uint32, None);
    if tokens.shape() != [1, LENGTH] {
        return Err("last-axis argmax must return [1, 128] tokens".into());
    }
    let tokens = contiguous(&tokens, None);
    try_eval(&[&tokens])?;
    let tokens = tokens.data_u32().to_vec();
    if tokens.len() != LENGTH as usize {
        return Err("argmax token count differs from the fixed output length".into());
    }
    Ok(tokens)
}

pub fn reference_digest(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("reference file exceeds 16 MiB".into());
    }
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn trace() -> Trace {
        Trace {
            codes: LEVELS.iter().map(|&length| vec![vec![0; length]]).collect(),
            tokens: vec![vec![0; 128]],
        }
    }

    #[test]
    fn latent_and_text_vocabularies_have_separate_bounds() {
        let mut input = trace();
        input.codes[0][0][0] = 4095;
        input.tokens[0][0] = 8191;
        assert!(input.validate().is_ok());
        input.codes[0][0][0] = 4096;
        assert!(input.validate().is_err());
        input.codes[0][0][0] = 0;
        input.tokens[0][0] = 8192;
        assert!(input.validate().is_err());
    }

    #[test]
    fn rejects_missing_levels_wrong_lengths_and_batches() {
        let mut input = trace();
        input.codes.pop();
        assert!(input.validate().is_err());
        let mut input = trace();
        input.codes[1][0].pop();
        assert!(input.validate().is_err());
        let mut input = trace();
        input.codes[0].push(vec![0]);
        assert!(input.validate().is_err());
        let mut input = trace();
        input.tokens[0].pop();
        assert!(input.validate().is_err());
    }

    #[test]
    fn half_pixel_resize_preserves_boundaries() {
        let input = float_array(&[0.0, 10.0], &[1, 2, 1]);
        let output = contiguous(&resize_linear(&input, 4), None);
        eval(&[&output]);
        assert_eq!(output.data_f32(), &[0.0, 2.5, 7.5, 10.0]);
    }

    #[test]
    fn learned_expansion_interleaves_positions_before_channels() {
        let input = float_array(&[2.0, 3.0], &[1, 1, 2]);
        let weight = float_array(&[1.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0, 10.0], &[2, 4]);
        let bias = MlxArray::from_f32_slice(&[1.0, -1.0]);
        let output = contiguous(&expand_sequence(&input, &weight, &bias, 2), None);
        eval(&[&output]);
        assert_eq!(output.shape(), vec![1, 2, 2]);
        assert_eq!(output.data_f32(), &[3.0, 2.0, 21.0, 29.0]);
    }

    #[test]
    fn tensor_validation_rejects_shape_dtype_and_nonfinite_values() {
        let good = MlxArray::from_f32_slice(&[1.0, 2.0]);
        assert!(validate_tensor("test", &good, &[2]).is_ok());
        assert!(validate_tensor("test", &good, &[1, 2]).is_err());
        let integer = astype(&good, MlxDtype::Uint32, None);
        assert!(validate_tensor("test", &integer, &[2]).is_err());
        let bad = MlxArray::from_f32_slice(&[f32::NAN]);
        assert!(validate_tensor("test", &bad, &[1]).is_err());
    }

    #[test]
    fn last_axis_argmax_retains_all_output_positions() {
        let logits = zeros(&[1, LENGTH, 8192], MlxDtype::Float32, None);
        assert_eq!(token_ids(&logits).unwrap(), vec![0; LENGTH as usize]);
        let wrong_rank = zeros(&[LENGTH, 8192], MlxDtype::Float32, None);
        assert!(token_ids(&wrong_rank).is_err());
    }

    #[test]
    fn numeric_gate_cannot_be_relaxed_by_reference_metadata() {
        let actual = MlxArray::from_f32_slice(&[1.001]);
        let reference = MlxArray::from_f32_slice(&[1.0]);
        let result = compare("test", &actual, &reference, 100.0).unwrap();
        assert_eq!(result["within_bound"], false);
    }

    #[test]
    fn exact_gelu_keeps_the_erf_value() {
        let output = contiguous(&gelu(&MlxArray::from_f32_slice(&[1.0]), None), None);
        eval(&[&output]);
        assert!((output.data_f32()[0] - 0.841_344_7).abs() < 1e-7);
    }

    #[test]
    fn interpolation_handles_noninteger_scale_and_identity() {
        let input = float_array(&[0.0, 10.0, 20.0], &[1, 3, 1]);
        let resized = contiguous(&resize_linear(&input, 5), None);
        eval(&[&resized]);
        for (&actual, expected) in resized.data_f32().iter().zip([0.0, 4.0, 10.0, 16.0, 20.0]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        let identity = contiguous(&resize_linear(&input, 3), None);
        eval(&[&identity]);
        assert_eq!(identity.data_f32(), &[0.0, 10.0, 20.0]);
    }
}
