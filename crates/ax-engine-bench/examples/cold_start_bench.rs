//! Measure session-construction (cold start) time for the embedding
//! loader path. Times `EngineSession::new(...)` start-to-ready, which
//! includes safetensors load + MLX wired-limit setup + GPU stream init.
//!
//! Usage:
//!   ./target/release/examples/cold_start_bench --model-dir <path>
//!
//! Loader is selected via `AX_MMAP_WEIGHTS`:
//!   AX_MMAP_WEIGHTS=0  → C loader (default, copies into MLX-owned buffer)
//!   AX_MMAP_WEIGHTS=1  → Rust mmap loader (zero-copy, lazy paging)
//!
//! Re-runs in the same process are warm (page cache) — to see true cold
//! start, drop OS page cache between runs (`sudo purge` on macOS) or
//! reboot. The point of the comparison is the steady-state work each
//! loader does per cold tensor, not the absolute timing on a cached
//! disk.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ax_engine_core::CacheGroupId;
use ax_engine_mlx::weights::mmap_weights_enabled;
use ax_engine_sdk::{
    EngineSession, EngineSessionConfig, PreviewBackendRequest, PreviewSessionConfigRequest,
};

fn parse_args() -> Result<PathBuf, String> {
    let mut args = std::env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--model-dir requires a path".to_string())?,
                ));
            }
            other => {
                return Err(format!("unexpected argument: {other}"));
            }
        }
    }
    model_dir.ok_or_else(|| "--model-dir <path> is required".to_string())
}

fn run() -> Result<(), String> {
    let model_dir = parse_args()?;
    let mmap = mmap_weights_enabled();
    eprintln!(
        "[cold-start-bench] model={} loader={}",
        model_dir.display(),
        if mmap { "mmap (R4)" } else { "C loader" }
    );

    let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
        cache_group_id: CacheGroupId(0),
        block_size_tokens: 16,
        total_blocks: 1024,
        deterministic: true,
        max_batch_tokens: 2048,
        mlx_runtime_artifacts_dir: None,
        backend_request: PreviewBackendRequest::shipping_mlx(),
        mlx_model_artifacts_dir: Some(model_dir.clone()),
        mlx_disable_ngram_acceleration: false,
        mlx_mtp_disable_ngram_stacking: true,
        mlx_speculation_profile: None,
        mlx_prefill_chunk: None,
        ..PreviewSessionConfigRequest::default()
    })
    .map_err(|error| format!("invalid session configuration: {error}"))?;

    let t0 = Instant::now();
    let _session = EngineSession::new(config)
        .map_err(|error| format!("failed to create engine session: {error}"))?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("session_new_ms {:.2}", ms);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}
