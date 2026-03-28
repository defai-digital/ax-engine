use clap::Parser;

/// AX Engine basic inference server.
#[derive(Parser, Debug, Clone)]
#[command(name = "ax-engine-server", version, about)]
pub struct ServerArgs {
    /// Path to GGUF model file
    #[arg(short = 'm', long = "model")]
    pub model: String,

    /// Bind host
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    /// Bind port
    #[arg(long = "port", default_value_t = 3000)]
    pub port: u16,

    /// Context size override
    #[arg(short = 'c', long = "ctx-size", default_value_t = 4096)]
    pub ctx_size: u32,

    /// Backend selection: auto, cpu, metal, hybrid, hybrid_cpu_decode
    #[arg(long = "backend", default_value = "auto")]
    pub backend: String,

    /// Default max tokens per request
    #[arg(long = "max-tokens", default_value_t = 256)]
    pub max_tokens: usize,

    /// Default random seed (-1 = random)
    #[arg(long = "seed", default_value_t = -1)]
    pub seed: i64,

    /// Verbose logging
    #[arg(long = "verbose")]
    pub verbose: bool,
}
