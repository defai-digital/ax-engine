use clap::Parser;

/// AX Engine — llama.cpp-compatible CLI
///
/// Drop-in replacement for llama-cli on Apple Silicon M3+.
#[derive(Parser, Debug)]
#[command(name = "ax-llama", version, about)]
pub struct CliArgs {
    /// Path to GGUF model file
    #[arg(short = 'm', long = "model")]
    pub model: String,

    /// Input prompt
    #[arg(short = 'p', long = "prompt")]
    pub prompt: Option<String>,

    /// Maximum number of tokens to predict (-1 = infinite)
    #[arg(short = 'n', long = "n-predict", default_value_t = -1)]
    pub n_predict: i32,

    /// Context size
    #[arg(short = 'c', long = "ctx-size", default_value_t = 4096)]
    pub ctx_size: u32,

    /// Number of threads to use (0 = auto)
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    pub threads: u32,

    /// Temperature for sampling
    #[arg(long = "temp", default_value_t = 0.8)]
    pub temperature: f32,

    /// Top-K sampling (0 = disabled)
    #[arg(long = "top-k", default_value_t = 40)]
    pub top_k: i32,

    /// Top-P (nucleus) sampling (1.0 = disabled)
    #[arg(long = "top-p", default_value_t = 0.9)]
    pub top_p: f32,

    /// Minimum probability relative to the most likely token (0.0 = disabled)
    #[arg(long = "min-p", default_value_t = 0.0)]
    pub min_p: f32,

    /// Random seed (-1 = random)
    #[arg(long = "seed", default_value_t = -1)]
    pub seed: i64,

    /// Repeat penalty
    #[arg(long = "repeat-penalty", default_value_t = 1.0)]
    pub repeat_penalty: f32,

    /// Penalize tokens in proportion to the number of prior occurrences.
    #[arg(long = "frequency-penalty", default_value_t = 0.0)]
    pub frequency_penalty: f32,

    /// Penalize tokens that have appeared at least once.
    #[arg(long = "presence-penalty", default_value_t = 0.0)]
    pub presence_penalty: f32,

    /// Interactive mode (multi-turn REPL)
    #[arg(long = "interactive")]
    pub interactive: bool,

    /// Verbose output (print metrics, timing)
    #[arg(long = "verbose")]
    pub verbose: bool,

    /// Enable experimental features that are not yet considered production-ready.
    #[arg(long = "experimental")]
    pub experimental: bool,

    /// Wrap prompt in model-specific chat template (required for instruction-tuned models)
    #[arg(long = "chat")]
    pub chat: bool,

    /// Run a 2-pass repeated-prefix prefill benchmark and print reuse delta.
    #[arg(long = "reuse-bench")]
    pub reuse_bench: bool,

    /// Emit --reuse-bench output as JSON (machine-readable).
    #[arg(long = "reuse-bench-json")]
    pub reuse_bench_json: bool,

    /// Experimental: path to a small draft model GGUF file for speculative decoding.
    /// The draft model runs on CPU; the main model verifies on GPU.
    #[arg(long = "speculative-draft")]
    pub speculative_draft: Option<String>,

    /// Number of tokens to speculate per step (speculative decoding lookahead K).
    #[arg(long = "speculative-k", default_value_t = 4)]
    pub speculative_k: usize,
}

/// Known llama.cpp flags that AX Engine does not support, with human-readable reasons.
const UNSUPPORTED_FLAGS: &[(&str, &str)] = &[
    ("--mirostat", "Mirostat sampling"),
    ("--mirostat-lr", "Mirostat learning rate"),
    ("--mirostat-ent", "Mirostat target entropy"),
    ("--tfs", "tail-free sampling"),
    ("--typical", "locally typical sampling"),
    ("--grammar", "grammar-constrained generation"),
    ("--grammar-file", "grammar-constrained generation"),
    ("--lora", "LoRA adapters"),
    ("--lora-scaled", "LoRA adapters"),
    ("--lora-base", "LoRA adapters"),
    ("-ngl", "GPU layer offloading (CPU-only engine)"),
    ("--n-gpu-layers", "GPU layer offloading (CPU-only engine)"),
    ("--flash-attn", "flash attention"),
    ("--cont-batching", "continuous batching"),
    ("--parallel", "parallel request handling"),
    ("--embedding", "embedding extraction mode"),
    ("--perplexity", "perplexity evaluation mode"),
    ("--mlock", "memory locking"),
    ("--no-mmap", "disabling mmap (always uses mmap)"),
    ("--rope-scaling", "RoPE scaling (auto-detected from GGUF)"),
    ("--rope-freq-scale", "RoPE frequency scaling"),
    ("--chatml", "ChatML template"),
    ("--instruct", "instruction mode"),
    ("--multiline-input", "multiline input"),
    ("--reverse-prompt", "reverse prompt / antiprompt"),
    ("--color", "colored output"),
    ("--log-disable", "log control"),
    ("--log-enable", "log control"),
    ("--hellaswag", "HellaSwag evaluation"),
    ("--numa", "NUMA optimization"),
    ("--batch-size", "batch size (use -n for token count)"),
];

/// Check CLI arguments for known unsupported llama.cpp flags.
///
/// If an unsupported flag is found, prints a clear error message and exits.
/// Call this before `CliArgs::parse()` so users get a targeted message
/// instead of clap's generic "unexpected argument" error.
pub fn check_unsupported_flags() {
    let args: Vec<String> = std::env::args().collect();

    for arg in &args[1..] {
        // Extract the flag name (before any '=' for --flag=value form)
        let flag = arg.split('=').next().unwrap_or(arg);

        for &(unsupported, feature) in UNSUPPORTED_FLAGS {
            if flag == unsupported {
                eprintln!("error: flag '{unsupported}' is not supported by AX Engine ({feature})");
                eprintln!();
                eprintln!("Run with --help to see supported options.");
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_flags_list_not_empty() {
        assert!(!UNSUPPORTED_FLAGS.is_empty());
    }

    #[test]
    fn test_unsupported_flags_all_start_with_dash() {
        for &(flag, _) in UNSUPPORTED_FLAGS {
            assert!(flag.starts_with('-'), "flag must start with -: {flag}");
        }
    }

    #[test]
    fn test_unsupported_flags_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for &(flag, _) in UNSUPPORTED_FLAGS {
            assert!(seen.insert(flag), "duplicate unsupported flag: {flag}");
        }
    }

    #[test]
    fn test_unsupported_flags_have_descriptions() {
        for &(flag, desc) in UNSUPPORTED_FLAGS {
            assert!(!desc.is_empty(), "missing description for flag: {flag}");
        }
    }
}
