#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn ax_engine_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ax-engine"))
}

fn run_generation(model_rel: &str, prompt: &str, disable_q6k_nr2: bool) -> Option<Vec<u8>> {
    let root = workspace_root();
    let model_path = root.join(model_rel);
    if !model_path.exists() {
        eprintln!("skipping: model not found at {}", model_path.display());
        return None;
    }

    let mut cmd = Command::new(ax_engine_bin());
    cmd.current_dir(&root)
        .arg("--model")
        .arg(model_path)
        .arg("--prompt")
        .arg(prompt)
        .arg("--n-predict")
        .arg("8")
        .arg("--temp")
        .arg("0")
        .arg("--top-k")
        .arg("1")
        .arg("--top-p")
        .arg("1")
        .arg("--seed")
        .arg("0");

    if disable_q6k_nr2 {
        cmd.env("AX_METAL_MATVEC_Q6K_NR2", "0");
    } else {
        cmd.env_remove("AX_METAL_MATVEC_Q6K_NR2");
    }

    let output = cmd.output().expect("run ax-engine");
    assert!(
        output.status.success(),
        "ax-engine failed: status={:?}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );
    Some(output.stdout)
}

fn assert_same_output_with_and_without_q6k_nr2(model_rel: &str, prompt: &str) {
    let Some(default_out) = run_generation(model_rel, prompt, false) else {
        return;
    };
    let Some(base_out) = run_generation(model_rel, prompt, true) else {
        return;
    };

    let default_trimmed = String::from_utf8_lossy(&default_out).trim().to_string();
    let base_trimmed = String::from_utf8_lossy(&base_out).trim().to_string();

    assert!(
        !default_trimmed.is_empty(),
        "default-path generation was empty for model {model_rel}"
    );
    assert_eq!(
        default_trimmed, base_trimmed,
        "Q6_K NR2 changed deterministic output for model {model_rel}\ndefault={default_trimmed:?}\nbase={base_trimmed:?}"
    );
}

#[test]
#[ignore = "requires local GGUF model and Metal GPU"]
fn test_qwen3_q6k_nr2_matches_base_generation() {
    assert_same_output_with_and_without_q6k_nr2(
        "models/Qwen3-8B-Q4_K_M.gguf",
        "The capital of France is",
    );
}

#[test]
#[ignore = "requires local GGUF model and Metal GPU"]
fn test_gemma3_q6k_nr2_matches_base_generation() {
    assert_same_output_with_and_without_q6k_nr2(
        "models/gemma-3-4b-it-Q4_K_M.gguf",
        "The capital of France is",
    );
}

#[test]
#[ignore = "requires local GGUF model and Metal GPU"]
fn test_llama3_q6k_nr2_matches_base_generation() {
    assert_same_output_with_and_without_q6k_nr2(
        "models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf",
        "The capital of France is",
    );
}
