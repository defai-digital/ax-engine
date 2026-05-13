//! R2 demo: overlap tokenization (CPU) with embedding (GPU) using two
//! threads connected by a bounded channel. Reads raw text lines from
//! stdin (one sentence per line), tokenizes them in chunks, and embeds
//! each chunk via `EngineSession::embed_batch_flat`. Reports total wall
//! time for serial vs pipelined execution on the same input.
//!
//! Usage:
//!   echo -e "sentence one\nsentence two\n..." | \
//!     cargo run -p ax-engine-bench --example embed_pipeline_demo \
//!       --release -- \
//!       --model-dir .internal/models/qwen3-embedding-0.6b-8bit \
//!       --batch-size 8
//!
//! Or feed a file:
//!   cargo run -p ax-engine-bench --example embed_pipeline_demo \
//!       --release -- --model-dir <path> --batch-size 8 < texts.txt
//!
//! Notes
//!  - Requires the model dir to contain `tokenizer.json` + `config.json`
//!    so the EngineTokenizer can find an EOS token (Qwen3-Embedding
//!    convention).
//!  - The serial path tokenizes a chunk, then embeds, then tokenizes
//!    the next chunk. The pipelined path runs tokenize on a separate
//!    thread feeding a channel; the embed thread drains the channel
//!    while GPU work is in flight. On Apple Silicon (unified memory,
//!    one GPU stream) the win comes from overlapping CPU tokenization
//!    with the GPU's forward pass — visible when chunks are CPU-heavy
//!    (long sentences, many BPE merges) or when GPU pass is short
//!    (small models / small batches).

use std::io::BufRead;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use ax_engine_core::{CacheGroupId, EmbeddingPooling};
use ax_engine_sdk::{
    EngineSession, EngineSessionConfig, EngineTokenizer, PreviewBackendRequest,
    PreviewSessionConfigRequest,
};

struct CliArgs {
    model_dir: PathBuf,
    batch_size: usize,
    warmup_chunks: usize,
}

fn parse_args() -> CliArgs {
    let mut args = std::env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    let mut batch_size: usize = 8;
    let mut warmup_chunks: usize = 2;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => model_dir = args.next().map(PathBuf::from),
            "--batch-size" => {
                batch_size = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--batch-size <usize>")
            }
            "--warmup-chunks" => {
                warmup_chunks = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--warmup-chunks <usize>")
            }
            other => {
                eprintln!("unexpected argument: {other}");
                std::process::exit(2);
            }
        }
    }
    CliArgs {
        model_dir: model_dir.expect("--model-dir <path> is required"),
        batch_size,
        warmup_chunks,
    }
}

fn build_session(model_dir: &PathBuf) -> EngineSession {
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
        mlx_kv_compression: ax_engine_sdk::MlxKvCompressionConfig::disabled(),
    })
    .expect("config");
    EngineSession::new(config).expect("session")
}

fn run_serial(
    session: &EngineSession,
    tokenizer: &EngineTokenizer,
    chunks: &[Vec<String>],
) -> (usize, usize) {
    let mut total_sentences = 0;
    let mut total_tokens = 0;
    for chunk in chunks {
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        let token_ids = tokenizer.encode_batch(&refs, true).expect("tokenize");
        total_tokens += token_ids.iter().map(Vec::len).sum::<usize>();
        let _ = session
            .embed_batch_flat(&token_ids, EmbeddingPooling::Last, true)
            .expect("embed_batch_flat");
        total_sentences += chunk.len();
    }
    (total_sentences, total_tokens)
}

fn run_pipelined(
    session: &EngineSession,
    tokenizer: &EngineTokenizer,
    chunks: &[Vec<String>],
) -> (usize, usize) {
    // Channel capacity = 1 means the tokenizer thread can stay one chunk
    // ahead of the GPU thread; raising it just buffers more chunks but
    // doesn't help once the GPU is the bottleneck (typical at 4B+).
    let (tx, rx) = mpsc::sync_channel::<Vec<Vec<u32>>>(1);
    let chunks_clone: Vec<Vec<String>> = chunks.to_vec();
    let tokenizer_clone = tokenizer.clone();
    let tok_thread = thread::spawn(move || {
        for chunk in chunks_clone {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            let token_ids = tokenizer_clone.encode_batch(&refs, true).expect("tokenize");
            if tx.send(token_ids).is_err() {
                break;
            }
        }
    });

    let mut total_sentences = 0;
    let mut total_tokens = 0;
    while let Ok(token_ids) = rx.recv() {
        total_tokens += token_ids.iter().map(Vec::len).sum::<usize>();
        total_sentences += token_ids.len();
        let _ = session
            .embed_batch_flat(&token_ids, EmbeddingPooling::Last, true)
            .expect("embed_batch_flat");
    }
    tok_thread.join().expect("tokenizer thread");
    (total_sentences, total_tokens)
}

fn main() {
    let args = parse_args();
    eprintln!(
        "[embed-pipeline-demo] model={} batch_size={} warmup_chunks={}",
        args.model_dir.display(),
        args.batch_size,
        args.warmup_chunks,
    );

    eprintln!("[embed-pipeline-demo] reading sentences from stdin (one per line) …");
    let stdin = std::io::stdin();
    let lines: Vec<String> = stdin
        .lock()
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .collect();
    if lines.is_empty() {
        eprintln!("no input sentences on stdin; aborting");
        std::process::exit(2);
    }

    // Chunk inputs into batch_size groups. The last chunk may be smaller.
    let mut chunks: Vec<Vec<String>> = Vec::new();
    for window in lines.chunks(args.batch_size) {
        chunks.push(window.to_vec());
    }
    eprintln!(
        "[embed-pipeline-demo] read {} sentences in {} chunks",
        lines.len(),
        chunks.len()
    );

    let session = build_session(&args.model_dir);
    let tokenizer = EngineTokenizer::from_model_dir(&args.model_dir).expect("tokenizer");
    eprintln!(
        "[embed-pipeline-demo] tokenizer loaded; eos_token_id={:?}",
        tokenizer.eos_token_id()
    );

    // Warmup both paths so GPU compile cache / clocks are hot.
    let warmup_chunks: Vec<Vec<String>> = chunks
        .iter()
        .take(args.warmup_chunks.min(chunks.len()))
        .cloned()
        .collect();
    eprintln!("[embed-pipeline-demo] warmup × {} chunks", warmup_chunks.len());
    run_serial(&session, &tokenizer, &warmup_chunks);

    eprintln!("[embed-pipeline-demo] running serial …");
    let t0 = Instant::now();
    let (n_serial, tokens_serial) = run_serial(&session, &tokenizer, &chunks);
    let serial_ms = t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("[embed-pipeline-demo] running pipelined …");
    let t0 = Instant::now();
    let (n_pipe, tokens_pipe) = run_pipelined(&session, &tokenizer, &chunks);
    let pipe_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!();
    println!("Serial      {:>3} sentences, {:>5} tokens   {:.2} ms total   {:.0} tok/s   {:.2} ms/sent",
        n_serial, tokens_serial, serial_ms, tokens_serial as f64 / (serial_ms / 1000.0), serial_ms / n_serial as f64);
    println!("Pipelined   {:>3} sentences, {:>5} tokens   {:.2} ms total   {:.0} tok/s   {:.2} ms/sent",
        n_pipe, tokens_pipe, pipe_ms, tokens_pipe as f64 / (pipe_ms / 1000.0), pipe_ms / n_pipe as f64);
    println!(
        "Pipeline speedup: {:+.1}%  (lower ms = better)",
        (serial_ms / pipe_ms - 1.0) * 100.0
    );
}
