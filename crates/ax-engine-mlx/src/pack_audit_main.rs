//! Load a Qwen-family MLX model and print per-layer pack state for
//! linear-attention QKVZ/BA and dense FFN gate/up projections.
//!
//! Usage: cargo run --release --bin pack-audit -- <model_dir>
//!
//! Exits 0 after printing one summary block. No graph evaluation, no decode —
//! the only purpose is to confirm whether `pack_split_linear_attention_projections`
//! and `pack_dense_ffn_gate_up_projection` actually engage at load time on a
//! given checkpoint, so an investigation can distinguish a missing pack from a
//! gap that lives in the decode hot path.

use std::env;
use std::path::Path;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::weights::load_weights;

fn main() {
    let model_dir = env::args().nth(1).expect("Usage: pack-audit <model_dir>");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .expect("failed to load model artifacts");
    let manifest = artifacts.manifest();
    let model_family = manifest.model_family.clone();
    let layer_count = manifest.layer_count;
    let weights = load_weights(&artifacts).expect("failed to load weights");

    let mut la_packed = 0u32;
    let mut la_split = 0u32;
    let mut la_partial = 0u32;
    let mut la_absent = 0u32;
    let mut ffn_packed = 0u32;
    let mut ffn_split = 0u32;
    let mut ffn_absent = 0u32;

    for (idx, layer) in weights.layers.iter().enumerate() {
        match layer.linear_attn.as_ref() {
            Some(la) => {
                let is_packed = la.in_proj_qkvz.is_some() && la.in_proj_ba.is_some();
                let is_split = la.in_proj_qkv.is_some()
                    && la.in_proj_z.is_some()
                    && la.in_proj_a.is_some()
                    && la.in_proj_b.is_some();
                if is_packed {
                    la_packed += 1;
                } else if is_split {
                    la_split += 1;
                    println!("  layer {idx}: linear-attn SPLIT (qkv+z+a+b)");
                } else {
                    la_partial += 1;
                    println!(
                        "  layer {idx}: linear-attn PARTIAL qkv={} z={} a={} b={} qkvz={} ba={}",
                        la.in_proj_qkv.is_some(),
                        la.in_proj_z.is_some(),
                        la.in_proj_a.is_some(),
                        la.in_proj_b.is_some(),
                        la.in_proj_qkvz.is_some(),
                        la.in_proj_ba.is_some(),
                    );
                }
            }
            None => la_absent += 1,
        }

        if layer.down_proj.is_some() {
            if layer.gate_up_packed.is_some() {
                ffn_packed += 1;
            } else if layer.gate_proj.is_some() && layer.up_proj.is_some() {
                ffn_split += 1;
                println!("  layer {idx}: dense FFN SPLIT (gate+up)");
            } else {
                ffn_absent += 1;
            }
        }
    }

    println!();
    println!("model_family: {model_family}");
    println!("manifest_layer_count: {layer_count}");
    println!("layers_loaded: {}", weights.layers.len());
    println!();
    println!("linear_attention:");
    println!("  qkvz_ba_packed_layers: {la_packed}");
    println!("  split_qkvba_layers:    {la_split}");
    println!("  partial_layers:        {la_partial}");
    println!("  absent_layers:         {la_absent}");
    println!();
    println!("dense_ffn:");
    println!("  gate_up_packed_layers: {ffn_packed}");
    println!("  split_gate_up_layers:  {ffn_split}");
    println!("  absent_layers:         {ffn_absent}");
}
