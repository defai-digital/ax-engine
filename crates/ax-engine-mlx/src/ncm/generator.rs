//! Native NCM generator graph, independently authored from pinned metadata.

use super::codec::*;
use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, argmax, astype, concatenate,
    contiguous, divide, exp, gelu, layer_norm, load_safetensors, matmul, multiply, negative,
    reshape, rms_norm, scaled_dot_product_attention_with_mask, slice, subtract, sum_axis, take,
    transpose, try_eval,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

pub const GENERATOR_SHA256: &str =
    "e24e9ddde6b616add427420d7da97d86e92fbf1199a187dfc2e005fdeff5c49d";
pub const HIDDEN: i32 = 768;
pub const DEPTH: usize = 12;

pub fn generator_shapes() -> BTreeMap<String, Vec<i32>> {
    let mut shapes = BTreeMap::new();
    for i in 0..DEPTH {
        for norm in ["ln1", "ln2"] {
            shapes.insert(format!("blocks.{i}.{norm}.weight"), vec![HIDDEN]);
            shapes.insert(format!("blocks.{i}.{norm}.bias"), vec![HIDDEN]);
        }
        for norm in ["q_norm", "k_norm"] {
            shapes.insert(format!("blocks.{i}.attn.{norm}.weight"), vec![64]);
        }
        for (name, output, input) in [
            ("attn.qkv", 3 * HIDDEN, HIDDEN),
            ("attn.proj", HIDDEN, HIDDEN),
            ("fc", 4 * HIDDEN, HIDDEN),
            ("out", HIDDEN, 4 * HIDDEN),
        ] {
            shapes.insert(format!("blocks.{i}.{name}.weight"), vec![output, input]);
            shapes.insert(format!("blocks.{i}.{name}.bias"), vec![output]);
        }
    }
    for i in 0..11 {
        shapes.insert(format!("heads.{i}.proj.weight"), vec![WIDTH, HIDDEN]);
        shapes.insert(format!("heads.{i}.log_inv_t"), vec![]);
    }
    shapes.insert("in_proj.weight".into(), vec![HIDDEN, WIDTH]);
    shapes.insert("in_proj.bias".into(), vec![HIDDEN]);
    shapes.insert("level_emb.weight".into(), vec![11, HIDDEN]);
    shapes.insert("ln_f.weight".into(), vec![HIDDEN]);
    shapes.insert("ln_f.bias".into(), vec![HIDDEN]);
    shapes
}

pub struct Generator {
    weights: HashMap<String, MlxArray>,
    book_norms: Vec<MlxArray>,
    book_energies: Vec<MlxArray>,
    prefix: MlxArray,
}

pub struct Forward {
    pub input: MlxArray,
    pub blocks: Vec<MlxArray>,
    pub logits: MlxArray,
}

impl Generator {
    pub fn prefix_embedding(&self) -> &MlxArray {
        &self.prefix
    }

    pub fn load(directory: &Path, codec: &Codec) -> Result<Self, String> {
        Self::load_verified(&directory.join("generator.safetensors"), codec)
    }

    pub fn load_verified(path: &Path, codec: &Codec) -> Result<Self, String> {
        verify_file(path, 354440547, GENERATOR_SHA256)?;
        let weights = load_safetensors(path, None)?;
        let shapes = generator_shapes();
        if weights.len() != shapes.len() {
            return Err("unexpected generator tensor set".into());
        }
        for (name, shape) in shapes {
            validate_tensor(
                &name,
                weights
                    .get(&name)
                    .ok_or_else(|| format!("missing {name}"))?,
                &shape,
            )?;
        }
        let mut norms = Vec::new();
        let mut energies = Vec::new();
        for level in 1..12 {
            let book = codec.weight(&format!("codebooks.{level}"));
            let norm = sum_axis(&multiply(book, book, None), -1, false, None);
            let energy = multiply(
                &sum_axis(&norm, 0, false, None),
                &MlxArray::from_f32(1.0 / 4096.0),
                None,
            );
            try_eval(&[&energy])?;
            if !energy.data_f32()[0].is_finite() || energy.data_f32()[0] <= 0.0 {
                return Err("invalid codebook energy".into());
            }
            norms.push(norm);
            energies.push(energy);
        }
        let zero = astype(&float_array(&[0.0], &[1, 1]), MlxDtype::Uint32, None);
        let embedding = take(codec.weight("tok.weight"), &zero, 0, None);
        let projected = matmul(
            &embedding,
            &transpose(&weights["in_proj.weight"], &[1, 0], None),
            None,
        );
        let prefix = add(&projected, &weights["in_proj.bias"], None);
        Ok(Self {
            weights,
            book_norms: norms,
            book_energies: energies,
            prefix,
        })
    }

    fn linear(&self, x: &MlxArray, name: &str) -> MlxArray {
        let weight = &self.weights[&format!("{name}.weight")];
        let projected = matmul(x, &transpose(weight, &[1, 0], None), None);
        match self.weights.get(&format!("{name}.bias")) {
            Some(bias) => add(&projected, bias, None),
            None => projected,
        }
    }

    fn norm(&self, x: &MlxArray, name: &str) -> MlxArray {
        layer_norm(
            x,
            &self.weights[&format!("{name}.weight")],
            &self.weights[&format!("{name}.bias")],
            1e-6,
            None,
        )
    }

    pub fn block(
        &self,
        x: &MlxArray,
        i: usize,
        cosine: &MlxArray,
        sine: &MlxArray,
        mask: &MlxArray,
    ) -> MlxArray {
        let length = x.shape()[1];
        let name = format!("blocks.{i}");
        let qkv = self.linear(
            &self.norm(x, &format!("{name}.ln1")),
            &format!("{name}.attn.qkv"),
        );
        let head = |offset: i32| {
            let part = slice(
                &qkv,
                &[0, 0, offset],
                &[1, length, offset + HIDDEN],
                &[1, 1, 1],
                None,
            );
            reshape(&part, &[1, length, 12, 64], None)
        };
        let q = rms_norm(
            &head(0),
            Some(&self.weights[&format!("{name}.attn.q_norm.weight")]),
            1e-6,
            None,
        );
        let k = rms_norm(
            &head(HIDDEN),
            Some(&self.weights[&format!("{name}.attn.k_norm.weight")]),
            1e-6,
            None,
        );
        let q = rotate(&transpose(&q, &[0, 2, 1, 3], None), cosine, sine);
        let k = rotate(&transpose(&k, &[0, 2, 1, 3], None), cosine, sine);
        let v = transpose(&head(2 * HIDDEN), &[0, 2, 1, 3], None);
        let attended = scaled_dot_product_attention_with_mask(
            &q,
            &k,
            &v,
            0.125,
            ScaledDotProductAttentionMask::Array(mask),
            None,
        );
        let attended = reshape(
            &transpose(&attended, &[0, 2, 1, 3], None),
            &[1, length, HIDDEN],
            None,
        );
        let residual = add(
            x,
            &self.linear(&attended, &format!("{name}.attn.proj")),
            None,
        );
        let hidden = self.linear(
            &self.norm(&residual, &format!("{name}.ln2")),
            &format!("{name}.fc"),
        );
        add(
            &residual,
            &self.linear(&gelu(&hidden, None), &format!("{name}.out")),
            None,
        )
    }

    pub fn forward(&self, codec: &Codec, level: usize, pooled: &MlxArray) -> Forward {
        self.forward_with_capture(codec, level, pooled, true)
    }

    pub fn forward_with_capture(
        &self,
        codec: &Codec,
        level: usize,
        pooled: &MlxArray,
        capture: bool,
    ) -> Forward {
        let length = LEVELS[level] as i32;
        let embedding = slice(
            &self.weights["level_emb.weight"],
            &[(level - 1) as i32, 0],
            &[level as i32, HIDDEN],
            &[1, 1],
            None,
        );
        let target = add(
            &self.linear(pooled, "in_proj"),
            &reshape(&embedding, &[1, 1, HIDDEN], None),
            None,
        );
        let input = concatenate(&[&self.prefix, &target], 1, None);
        let (cosine, sine) = rotary_tables(length + 1);
        let mask = prefix_mask(length + 1);
        let mut hidden = input.clone();
        let mut blocks = Vec::new();
        for i in 0..DEPTH {
            hidden = self.block(&hidden, i, &cosine, &sine, &mask);
            if capture {
                blocks.push(hidden.clone());
            }
        }
        let hidden = self.norm(&hidden, "ln_f");
        let target = slice(
            &hidden,
            &[0, 1, 0],
            &[1, length + 1, HIDDEN],
            &[1, 1, 1],
            None,
        );
        let query = self.linear(&target, &format!("heads.{}.proj", level - 1));
        let book = codec.weight(&format!("codebooks.{level}"));
        let logits = tied_logits(
            &query,
            book,
            &self.book_norms[level - 1],
            &self.book_energies[level - 1],
            &self.weights[&format!("heads.{}.log_inv_t", level - 1)],
        );
        Forward {
            input,
            blocks,
            logits,
        }
    }
}

pub fn tied_logits(
    query: &MlxArray,
    book: &MlxArray,
    norms: &MlxArray,
    energy: &MlxArray,
    log_inv_t: &MlxArray,
) -> MlxArray {
    let dots = multiply(
        &matmul(query, &transpose(book, &[1, 0], None), None),
        &MlxArray::from_f32(2.0),
        None,
    );
    let distance = subtract(&dots, norms, None);
    multiply(
        &divide(&distance, energy, None),
        &exp(log_inv_t, None),
        None,
    )
}

pub fn area_pool(x: &MlxArray, output: usize) -> MlxArray {
    let input = x.shape()[1] as usize;
    let mut rows = Vec::new();
    for i in 0..output {
        let start = i * input / output;
        let end = ((i + 1) * input).div_ceil(output);
        let chunk = slice(
            x,
            &[0, start as i32, 0],
            &[1, end as i32, x.shape()[2]],
            &[1, 1, 1],
            None,
        );
        rows.push(multiply(
            &sum_axis(&chunk, 1, true, None),
            &MlxArray::from_f32(1.0 / (end - start) as f32),
            None,
        ));
    }
    concatenate(&rows.iter().collect::<Vec<_>>(), 1, None)
}

pub fn prefix_mask(length: i32) -> MlxArray {
    let values = (0..length)
        .flat_map(|query| (0..length).map(move |key| if query > 0 || key == 0 { 1.0 } else { 0.0 }))
        .collect::<Vec<_>>();
    astype(
        &float_array(&values, &[1, 1, length, length]),
        MlxDtype::Bool,
        None,
    )
}

pub fn rotary_tables(length: i32) -> (MlxArray, MlxArray) {
    let mut cosine = Vec::new();
    let mut sine = Vec::new();
    for row in 0..length {
        let position = (row - 1).max(0) as f32;
        for column in 0..64 {
            let inverse = 100_000.0f32.powf(-((2 * (column % 32)) as f32) / 64.0);
            let angle = position * inverse;
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
    }
    (
        float_array(&cosine, &[1, 1, length, 64]),
        float_array(&sine, &[1, 1, length, 64]),
    )
}

pub fn rotate(x: &MlxArray, cosine: &MlxArray, sine: &MlxArray) -> MlxArray {
    let shape = x.shape();
    let first = slice(
        x,
        &[0, 0, 0, 0],
        &[1, 12, shape[2], 32],
        &[1, 1, 1, 1],
        None,
    );
    let second = slice(
        x,
        &[0, 0, 0, 32],
        &[1, 12, shape[2], 64],
        &[1, 1, 1, 1],
        None,
    );
    let rotated = concatenate(&[&negative(&second, None), &first], -1, None);
    add(
        &multiply(x, cosine, None),
        &multiply(&rotated, sine, None),
        None,
    )
}

pub fn chosen(logits: &MlxArray) -> Result<Vec<u32>, String> {
    choose(logits, None)
}

pub(super) fn choose(logits: &MlxArray, key: Option<&MlxArray>) -> Result<Vec<u32>, String> {
    let shape = logits.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] != 4096 {
        return Err("code logits must have shape [1, length, 4096]".into());
    }
    let ids = match key {
        Some(key) => mlx_sys::random_categorical_with_key(logits, key, None),
        None => argmax(logits, None),
    };
    let array = contiguous(&astype(&ids, MlxDtype::Uint32, None), None);
    try_eval(&[&array])?;
    if array.shape() != [1, shape[1]] {
        return Err("code selection must preserve all positions".into());
    }
    let values = array.data_u32().to_vec();
    if values.len() != shape[1] as usize || values.iter().any(|&v| v >= 4096) {
        return Err("invalid selected code".into());
    }
    Ok(values)
}
