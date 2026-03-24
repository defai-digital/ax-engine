//! Build script: precompile Metal shaders to .metallib
//!
//! Compiles all .metal shader files to Metal AIR (intermediate representation)
//! using `xcrun metal`, then links them into a single `.metallib` binary using
//! `xcrun metallib`. The .metallib is embedded into the binary at compile time
//! via `include_bytes!` in pipeline.rs.
//!
//! Reference: mistral.rs build.rs:145-318, llama.cpp embeds ggml.metallib.
//!
//! Set AX_METAL_SKIP_PRECOMPILE=1 to skip precompilation and use runtime
//! shader compilation instead.

use std::env;
use std::path::PathBuf;
use std::process::Command;

const METAL_SOURCES: [&str; 5] = [
    "attention",
    "dequant",
    "elementwise",
    "gdn",
    "matmul",
];

fn main() {
    // Track changes to shader sources.
    for src in METAL_SOURCES {
        println!("cargo:rerun-if-changed=shaders/{src}.metal");
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=AX_METAL_SKIP_PRECOMPILE");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let shader_dir = PathBuf::from("shaders");

    // Skip precompilation if requested (useful for CI without Xcode).
    let skip = env::var("AX_METAL_SKIP_PRECOMPILE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if skip {
        println!("cargo:warning=Skipping Metal shader precompilation (AX_METAL_SKIP_PRECOMPILE=1)");
        std::fs::write(out_dir.join("ax_metal.metallib"), []).unwrap();
        return;
    }

    // Check if xcrun is available.
    if Command::new("xcrun").arg("--version").output().is_err() {
        println!("cargo:warning=xcrun not found, skipping Metal shader precompilation");
        std::fs::write(out_dir.join("ax_metal.metallib"), []).unwrap();
        return;
    }

    // Step 1: Compile each .metal to .air
    for src in METAL_SOURCES {
        let input = shader_dir.join(format!("{src}.metal"));
        let output = out_dir.join(format!("{src}.air"));

        let status = Command::new("xcrun")
            .arg("--sdk")
            .arg("macosx")
            .arg("metal")
            .arg("-std=metal3.1")
            .arg("-O3")
            .arg("-w") // suppress warnings (unused variables etc.)
            .arg("-c")
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .status()
            .expect("Failed to run xcrun metal");

        if !status.success() {
            println!(
                "cargo:warning=Failed to compile {src}.metal to AIR, falling back to runtime compilation"
            );
            std::fs::write(out_dir.join("ax_metal.metallib"), []).unwrap();
            return;
        }
    }

    // Step 2: Link all .air into one .metallib
    let metallib_path = out_dir.join("ax_metal.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.arg("--sdk")
        .arg("macosx")
        .arg("metallib")
        .arg("-o")
        .arg(&metallib_path);

    for src in METAL_SOURCES {
        cmd.arg(out_dir.join(format!("{src}.air")));
    }

    let status = cmd.status().expect("Failed to run xcrun metallib");
    if !status.success() {
        println!("cargo:warning=Failed to link metallib, falling back to runtime compilation");
        std::fs::write(out_dir.join("ax_metal.metallib"), []).unwrap();
    }
}
