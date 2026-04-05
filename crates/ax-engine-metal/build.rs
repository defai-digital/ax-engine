//! Build script: precompile Metal shaders to .metallib
//!
//! Compiles all .metal shader files to Metal AIR (intermediate representation)
//! using `xcrun metal`, then links them into a single `.metallib` binary using
//! `xcrun metallib`. The resulting `.metallib` is emitted into `OUT_DIR` and
//! loaded by `pipeline.rs` at runtime.
//!
//! Reference: mistral.rs build.rs:145-318, llama.cpp embeds ggml.metallib.
//!
//! Set AX_METAL_SKIP_PRECOMPILE=1 to skip precompilation and use runtime
//! shader compilation instead.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const METAL_SOURCES: [&str; 4] = ["attention", "elementwise", "gdn", "matmul"];

const DEQUANT_SHADER_FRAGMENTS: [&str; 7] = [
    "shaders/dequant/common.metal",
    "shaders/dequant/q4_0.metal",
    "shaders/dequant/q4_k.metal",
    "shaders/dequant/q5_k.metal",
    "shaders/dequant/q6_k.metal",
    "shaders/dequant/q8_0.metal",
    "shaders/dequant/misc.metal",
];

const DEQUANT_RUNTIME_SOURCE: &str = "dequant_runtime.metal";

/// Optional shader that requires Metal 4+ SDK (tensor API).
/// Compiled separately — if it fails, the tensor kernel is unavailable at runtime.
const METAL_TENSOR_SOURCE: &str = "matmul_tensor";

fn main() {
    // Track changes to shader sources.
    for src in METAL_SOURCES {
        println!("cargo:rerun-if-changed=shaders/{src}.metal");
    }
    for fragment in DEQUANT_SHADER_FRAGMENTS {
        println!("cargo:rerun-if-changed={fragment}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=AX_METAL_SKIP_PRECOMPILE");

    println!("cargo::rustc-check-cfg=cfg(metal_tensor_api)");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let shader_dir = PathBuf::from("shaders");
    let dequant_runtime_source =
        write_dequant_runtime_shader(&out_dir).expect("failed to prepare dequant shader source");

    // Skip precompilation if requested (useful for CI without Xcode).
    let skip = env::var("AX_METAL_SKIP_PRECOMPILE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if skip {
        println!("cargo:warning=Skipping Metal shader precompilation (AX_METAL_SKIP_PRECOMPILE=1)");
        std::fs::write(out_dir.join("ax_engine_metal.metallib"), []).unwrap();
        return;
    }

    // Check if xcrun is available.
    if Command::new("xcrun").arg("--version").output().is_err() {
        println!("cargo:warning=xcrun not found, skipping Metal shader precompilation");
        std::fs::write(out_dir.join("ax_engine_metal.metallib"), []).unwrap();
        return;
    }

    // Step 1: Compile each .metal to .air
    for src in METAL_SOURCES {
        let input = shader_dir.join(format!("{src}.metal"));
        let output = out_dir.join(format!("{src}.air"));

        if !compile_shader_to_air(&input, &output) {
            println!(
                "cargo:warning=Failed to compile {src}.metal to AIR, falling back to runtime compilation"
            );
            std::fs::write(out_dir.join("ax_engine_metal.metallib"), []).unwrap();
            return;
        }
    }

    let dequant_output = out_dir.join("dequant.air");
    if !compile_shader_to_air(&dequant_runtime_source, &dequant_output) {
        println!(
            "cargo:warning=Failed to compile generated dequant shader to AIR, falling back to runtime compilation"
        );
        std::fs::write(out_dir.join("ax_engine_metal.metallib"), []).unwrap();
        return;
    }

    // Step 2: Try compiling the optional tensor API shader (Metal 4+ SDK).
    // This requires <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    // which is only available in newer Xcode SDKs.
    println!("cargo:rerun-if-changed=shaders/{METAL_TENSOR_SOURCE}.metal");
    let tensor_input = shader_dir.join(format!("{METAL_TENSOR_SOURCE}.metal"));
    let tensor_output = out_dir.join(format!("{METAL_TENSOR_SOURCE}.air"));
    let tensor_available = Command::new("xcrun")
        .arg("--sdk")
        .arg("macosx")
        .arg("metal")
        .arg("-std=metal3.1")
        .arg("-O3")
        .arg("-w")
        .arg("-c")
        .arg(&tensor_input)
        .arg("-o")
        .arg(&tensor_output)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if tensor_available {
        println!("cargo:rustc-cfg=metal_tensor_api");
        println!("cargo:warning=Metal Tensor API shader compiled (Metal 4+ SDK detected)");
    } else {
        println!(
            "cargo:warning=Metal Tensor API shader not available (SDK lacks MetalPerformancePrimitives)"
        );
    }

    // Step 3: Link all .air into one .metallib
    let metallib_path = out_dir.join("ax_engine_metal.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.arg("--sdk")
        .arg("macosx")
        .arg("metallib")
        .arg("-o")
        .arg(&metallib_path);

    for src in METAL_SOURCES {
        cmd.arg(out_dir.join(format!("{src}.air")));
    }
    cmd.arg(&dequant_output);
    if tensor_available {
        cmd.arg(&tensor_output);
    }

    let status = cmd.status().expect("Failed to run xcrun metallib");
    if !status.success() {
        println!("cargo:warning=Failed to link metallib, falling back to runtime compilation");
        std::fs::write(out_dir.join("ax_engine_metal.metallib"), []).unwrap();
    }
}

fn write_dequant_runtime_shader(out_dir: &Path) -> std::io::Result<PathBuf> {
    let mut combined = String::new();
    for fragment in DEQUANT_SHADER_FRAGMENTS {
        let content = fs::read_to_string(fragment)?;
        combined.push_str(&content);
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }
    let output = out_dir.join(DEQUANT_RUNTIME_SOURCE);
    fs::write(&output, combined)?;
    Ok(output)
}

fn compile_shader_to_air(input: &Path, output: &Path) -> bool {
    Command::new("xcrun")
        .arg("--sdk")
        .arg("macosx")
        .arg("metal")
        .arg("-std=metal3.1")
        .arg("-O3")
        .arg("-w")
        .arg("-c")
        .arg(input)
        .arg("-o")
        .arg(output)
        .status()
        .is_ok_and(|status| status.success())
}
