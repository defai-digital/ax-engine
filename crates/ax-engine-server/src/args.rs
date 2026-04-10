use clap::{ArgAction, Parser};

#[derive(Parser, Debug, Clone)]
#[command(name = "ax-engine-server", version, about)]
pub struct ServerArgs {
    /// Path to GGUF model file
    #[arg(short = 'm', long = "model")]
    pub model: String,

    /// Override the model id exposed through /models and /v1/models
    #[arg(short = 'a', long = "alias")]
    pub alias: Option<String>,

    /// Context size (0 = use model default)
    #[arg(short = 'c', long = "ctx-size", default_value_t = 0)]
    pub ctx_size: u32,

    /// Number of CPU threads to use (0 = auto)
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    pub threads: u32,

    /// Default number of tokens to predict (-1 = until context is full)
    #[arg(short = 'n', long = "n-predict", alias = "predict", default_value_t = -1)]
    pub n_predict: i32,

    /// Default sampling temperature
    #[arg(long = "temp", default_value_t = 0.8)]
    pub temperature: f32,

    /// Default top-k
    #[arg(long = "top-k", default_value_t = 40)]
    pub top_k: i32,

    /// Default top-p
    #[arg(long = "top-p", default_value_t = 0.95)]
    pub top_p: f32,

    /// Default min-p
    #[arg(long = "min-p", default_value_t = 0.05)]
    pub min_p: f32,

    /// Default repeat penalty
    #[arg(long = "repeat-penalty", default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// Default repeat-last-n
    #[arg(long = "repeat-last-n", default_value_t = 64)]
    pub repeat_last_n: i32,

    /// Default frequency penalty
    #[arg(long = "frequency-penalty", default_value_t = 0.0)]
    pub frequency_penalty: f32,

    /// Default presence penalty
    #[arg(long = "presence-penalty", default_value_t = 0.0)]
    pub presence_penalty: f32,

    /// Default RNG seed (-1 = random)
    #[arg(long = "seed", default_value_t = -1)]
    pub seed: i64,

    /// Bind host
    #[arg(long = "host", default_value = "127.0.0.1")]
    pub host: String,

    /// Bind port
    #[arg(long = "port", default_value_t = 8080)]
    pub port: u16,

    /// Optional API key accepted via Authorization: Bearer or x-api-key
    #[arg(long = "api-key")]
    pub api_key: Option<String>,

    /// Enable verbose logs
    #[arg(long = "verbose", action = ArgAction::SetTrue)]
    pub verbose: bool,

    /// Accept llama-server's tool-call mode flag. Tool calls are not implemented yet.
    #[arg(long = "jinja", action = ArgAction::SetTrue)]
    pub jinja: bool,

    /// Accept llama-server compatibility flag but do not enable true parallel decoding yet.
    #[arg(long = "parallel", default_value_t = 1)]
    pub parallel: usize,

    /// Accept llama-server compatibility flag but do not enable continuous batching yet.
    #[arg(long = "cont-batching", action = ArgAction::SetTrue)]
    pub cont_batching: bool,

    /// Reserved for future mutable props support.
    #[arg(long = "props", action = ArgAction::SetTrue)]
    pub props: bool,

    /// Reserved for future metrics endpoint support.
    #[arg(long = "metrics", action = ArgAction::SetTrue)]
    pub metrics: bool,

    /// Reserved for future template override support.
    #[arg(long = "chat-template")]
    pub chat_template: Option<String>,

    /// Reserved for future template override support.
    #[arg(long = "chat-template-file")]
    pub chat_template_file: Option<String>,

    /// Accept llama-server's flag to simplify switching scripts.
    #[arg(long = "no-webui", action = ArgAction::SetTrue)]
    pub no_webui: bool,

    /// Disable slot inspection endpoints.
    #[arg(long = "no-slots", action = ArgAction::SetTrue)]
    pub no_slots: bool,

    /// Directory used by /slots save/restore persistence.
    #[arg(long = "slot-save-path")]
    pub slot_save_path: Option<String>,
}

impl ServerArgs {
    pub fn model_alias(&self) -> String {
        self.alias.clone().unwrap_or_else(|| self.model.clone())
    }

    pub fn status_args(&self) -> Vec<String> {
        let mut args = vec!["ax-engine-server".to_string()];
        args.push("-m".to_string());
        args.push(self.model.clone());
        if self.ctx_size > 0 {
            args.push("-c".to_string());
            args.push(self.ctx_size.to_string());
        }
        args
    }

    pub fn compatibility_warnings(&self) -> Vec<&'static str> {
        let mut warnings = Vec::new();

        if self.parallel > 1 {
            warnings.push("--parallel is accepted but requests still run one session per task");
        }
        if self.cont_batching {
            warnings.push("--cont-batching is accepted but continuous batching is not implemented");
        }
        if self.props {
            warnings.push("--props is accepted but POST /props is not implemented");
        }
        if self.chat_template.is_some() || self.chat_template_file.is_some() {
            warnings.push(
                "--chat-template* flags are accepted but template override is not implemented",
            );
        }
        if self.jinja {
            warnings.push("--jinja is accepted but tool calling is not implemented");
        }
        warnings
    }
}
