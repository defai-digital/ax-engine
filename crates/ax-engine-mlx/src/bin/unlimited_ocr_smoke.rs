//! Minimal Unlimited-OCR image+text smoke:
//!   cargo run -p ax-engine-mlx --release --bin unlimited_ocr_smoke -- \
//!     <model_dir> <image.png> [max_new_tokens]
//!
//! Builds a Free OCR prompt with the correct base-mode soft-token count,
//! runs dual-vision encode + language generate, prints decoded text.

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ax_engine_core::{NativeModelArtifacts, unlimited_ocr_soft_token_count};
use ax_engine_mlx::{
    generate::{
        DEFAULT_PREFILL_CHUNK, advance_direct_pipeline_with_timings,
        chunked_prefill_unlimited_ocr_with_sampling_buffers, start_direct_pipeline,
    },
    kv_cache::MlxKVCache,
    model::ModelConfig,
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    unlimited_ocr::{
        DEFAULT_IMAGE_TOKEN_ID, build_free_ocr_token_ids, default_base_soft_token_count,
        preprocess_document_rgb_u8,
    },
    weights::load_weights,
};

fn load_png_rgb(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    // Minimal PNG reader via pure-Rust `image` is not a dep of this crate.
    // Use a tiny uncompressed path: accept raw PPM (P6) or a hand-rolled PNG via
    // external preprocess. Prefer PPM for zero deps; also accept a pre-baked
    // .rgb dump of HxWx3 u8 with size encoded in the filename `WxH.rgb`.
    let bytes = std::fs::read(path).map_err(|e| format!("read image: {e}"))?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "ppm" || (bytes.len() >= 2 && &bytes[..2] == b"P6") {
        return parse_ppm_p6(&bytes);
    }
    if ext == "rgb" {
        // Expect stem like `text_320x80.rgb` or companion `.size` is not required —
        // parse trailing `_WxH` from stem.
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| "rgb path has no stem".to_string())?;
        let (w, h) = parse_wxh_suffix(stem)
            .ok_or_else(|| format!("rgb file stem must end with _WxH, got {stem}"))?;
        let expected = (w as usize) * (h as usize) * 3;
        if bytes.len() != expected {
            return Err(format!(
                "rgb length {} != {}x{}x3={}",
                bytes.len(),
                w,
                h,
                expected
            ));
        }
        return Ok((bytes, w, h));
    }
    // Fallback: try PPM magic regardless of extension.
    if bytes.len() >= 2 && &bytes[..2] == b"P6" {
        return parse_ppm_p6(&bytes);
    }
    Err(format!(
        "unsupported image format for {path:?}; provide a binary PPM (P6) or _WxH.rgb dump"
    ))
}

fn parse_wxh_suffix(stem: &str) -> Option<(u32, u32)> {
    let idx = stem.rfind('_')?;
    let dims = &stem[idx + 1..];
    let (w, h) = dims.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

fn parse_ppm_p6(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    // P6\nW H\n255\n<data>
    let skip_ws_comments = |buf: &[u8], mut pos: usize| -> usize {
        loop {
            while pos < buf.len() && buf[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos < buf.len() && buf[pos] == b'#' {
                while pos < buf.len() && buf[pos] != b'\n' {
                    pos += 1;
                }
                continue;
            }
            break pos;
        }
    };
    if bytes.len() < 3 || &bytes[..2] != b"P6" {
        return Err("not a P6 PPM".into());
    }
    let mut i = skip_ws_comments(bytes, 2);
    let start_w = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let w: u32 = std::str::from_utf8(&bytes[start_w..i])
        .map_err(|_| "bad width")?
        .parse()
        .map_err(|_| "bad width")?;
    i = skip_ws_comments(bytes, i);
    let start_h = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let h: u32 = std::str::from_utf8(&bytes[start_h..i])
        .map_err(|_| "bad height")?
        .parse()
        .map_err(|_| "bad height")?;
    i = skip_ws_comments(bytes, i);
    let start_max = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let maxv: u32 = std::str::from_utf8(&bytes[start_max..i])
        .map_err(|_| "bad maxval")?
        .parse()
        .map_err(|_| "bad maxval")?;
    if maxv != 255 {
        return Err(format!("only maxval 255 supported, got {maxv}"));
    }
    if i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let expected = (w as usize) * (h as usize) * 3;
    let data = bytes
        .get(i..i + expected)
        .ok_or_else(|| format!("ppm truncated: need {expected} bytes"))?
        .to_vec();
    Ok((data, w, h))
}

/// Very small BPE-free decode: map token ids via tokenizer.json string vocab if present.
fn try_decode_tokens(model_dir: &Path, ids: &[u32]) -> String {
    let tok_path = model_dir.join("tokenizer.json");
    let Ok(raw) = std::fs::read_to_string(&tok_path) else {
        return ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return format!("{ids:?}");
    };
    // Build id→token from model.vocab (string→id) when present.
    let mut id_to_tok: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
    if let Some(obj) = v.pointer("/model/vocab").and_then(|x| x.as_object()) {
        for (tok, idv) in obj {
            if let Some(id) = idv.as_u64() {
                id_to_tok.insert(id as u32, tok.clone());
            }
        }
    }
    if id_to_tok.is_empty() {
        return ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
    }
    let mut out = String::new();
    for &id in ids {
        if let Some(tok) = id_to_tok.get(&id) {
            // SentencePiece-style ▁ → space
            let piece = tok.replace('▁', " ").replace("<0x0A>", "\n");
            out.push_str(&piece);
        } else {
            out.push_str(&format!("<{id}>"));
        }
    }
    out
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let model_dir = args.next().ok_or_else(|| {
        "usage: unlimited_ocr_smoke <model_dir> <image.ppm|_WxH.rgb> [steps]".to_string()
    })?;
    let image_path = args.next().ok_or_else(|| {
        "usage: unlimited_ocr_smoke <model_dir> <image.ppm|_WxH.rgb> [steps]".to_string()
    })?;
    let steps: usize = args
        .next()
        .map(|s| s.parse().map_err(|e| format!("steps: {e}")))
        .transpose()?
        .unwrap_or(64);

    let model_path = Path::new(&model_dir);
    eprintln!("loading model {model_dir}");
    let artifacts =
        NativeModelArtifacts::from_dir(model_path).map_err(|e| format!("artifacts: {e}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).map_err(|e| format!("weights: {e}"))?;
    if weights.unlimited_ocr_vision.is_none() {
        return Err("model has no unlimited_ocr_vision weights loaded".into());
    }
    eprintln!(
        "vision loaded; soft tokens (base) = {}",
        default_base_soft_token_count()
    );

    let (rgb, w, h) = load_png_rgb(Path::new(&image_path))?;
    eprintln!("image {w}x{h}");
    let cropping = !matches!(
        env::var("AX_OCR_CROPPING").as_deref(),
        Ok("0") | Ok("false") | Ok("no")
    );
    let image = preprocess_document_rgb_u8(&rgb, w, h, cropping).map_err(|e| e.to_string())?;

    // Prompt: bos + image soft tokens + "Free OCR."
    // Matches mlx-vlm with base_size=1024, image_size=1024, cropping=False:
    //   Free=21431, ĠOCR=126041, .=16  (no leading newline in default Free OCR.)
    let suffix_ids: Vec<u32> = env::var("AX_OCR_SUFFIX_IDS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| vec![21431, 126041, 16]);
    eprintln!("suffix ids ({} tokens): {:?}", suffix_ids.len(), suffix_ids);
    let soft = unlimited_ocr_soft_token_count(w, h, cropping) as usize;
    let prompt = build_free_ocr_token_ids(&[], &suffix_ids, DEFAULT_IMAGE_TOKEN_ID, soft, Some(0));
    eprintln!("prompt len {}", prompt.len());

    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let mut probs = Vec::new();
    let mut logits = Vec::new();
    let mut cands = Vec::new();
    let t0 = Instant::now();
    let bootstrap = chunked_prefill_unlimited_ocr_with_sampling_buffers(
        &cfg,
        &weights,
        &prompt,
        &image,
        &mut cache,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), &prompt),
        &mut rng,
        &mut probs,
        &mut logits,
        &mut cands,
    )
    .map_err(|e| format!("prefill: {e}"))?;
    eprintln!(
        "prefill done in {:.1} ms, first token {bootstrap}",
        t0.elapsed().as_secs_f64() * 1e3
    );

    let mut generated = vec![bootstrap];
    let mut pending = start_direct_pipeline(&cfg, &weights, bootstrap, &mut cache);
    // DeepSeek-family EOS + Unlimited-OCR det end. Stop early once a complete
    // Free-OCR det block has closed so short smokes do not loop on repeats.
    const EOS: u32 = 1;
    const BOS: u32 = 0;
    const DET_END: u32 = 128_819; // <|/det|>
    while generated.len() < steps {
        let advanced = advance_direct_pipeline_with_timings(&cfg, &weights, &pending, &mut cache);
        generated.push(advanced.token);
        if advanced.token == EOS || advanced.token == BOS {
            break;
        }
        // After <|/det|>, keep generating until EOS or a small text budget.
        // If we already emitted det-end and enough trailing text, stop.
        if let Some(det_end_at) = generated.iter().position(|&t| t == DET_END) {
            let after = generated.len() - det_end_at - 1;
            // Typical Free OCR trail: "HELLO … 123" then EOS (~10 tokens).
            if after >= 16 {
                break;
            }
        }
        pending = advanced.next_pending;
        let _ = DEFAULT_PREFILL_CHUNK;
    }

    eprintln!("generated ids: {:?}", generated);
    let text = try_decode_tokens(model_path, &generated);
    // Prefer printable body: strip sentencepiece markers for the final line.
    let printable = text
        .replace('Ġ', " ")
        .replace('Ċ', "\n")
        .replace("<｜end▁of▁sentence｜>", "")
        .replace("<｜begin▁of▁sentence｜>", "");
    println!("{printable}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
