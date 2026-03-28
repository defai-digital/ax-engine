use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::{
    BackendKind, ChatMessage as SdkChatMessage, ChatRole as SdkChatRole,
    FinishReason as SdkFinishReason, GenerationOptions, LoadOptions, Model, SessionOptions,
};

use crate::args::ServerArgs;

pub(crate) struct ServerEngine {
    model: Model,
    default_max_tokens: usize,
    default_seed: Option<u64>,
    created: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct ModelInfo {
    pub(crate) id: String,
    pub(crate) created: i64,
    pub(crate) architecture: String,
    pub(crate) backend: String,
    pub(crate) routing: Option<String>,
    pub(crate) context_length: usize,
    pub(crate) vocab_size: usize,
    pub(crate) support_note: Option<String>,
    pub(crate) model_name: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct GenerateRequest {
    pub(crate) prompt: PromptInput,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<i32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) min_p: Option<f32>,
    pub(crate) repeat_penalty: Option<f32>,
    pub(crate) repeat_last_n: Option<i32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) stop_strings: Vec<String>,
    pub(crate) seed: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) enum PromptInput {
    Completion(String),
    Chat(Vec<RenderedChatMessage>),
}

#[derive(Debug, Clone)]
pub(crate) struct RenderedChatMessage {
    pub(crate) role: RenderedChatRole,
    pub(crate) content: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RenderedChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone)]
pub(crate) struct GenerateOutput {
    pub(crate) text: String,
    pub(crate) finish_reason: FinishReason,
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
}

impl ServerEngine {
    pub(crate) fn load(args: &ServerArgs) -> anyhow::Result<Self> {
        let backend = BackendKind::from_str(&args.backend)?;
        let model = Model::load(
            &args.model,
            LoadOptions {
                backend,
                context_length: (args.ctx_size > 0).then_some(args.ctx_size),
            },
        )?;

        Ok(Self {
            model,
            default_max_tokens: args.max_tokens,
            default_seed: parse_seed(args.seed),
            created: unix_timestamp(),
        })
    }

    pub(crate) fn info(&self) -> ModelInfo {
        let info = self.model.info();
        ModelInfo {
            id: info.id,
            created: self.created,
            architecture: info.architecture,
            backend: info.backend.as_str().to_string(),
            routing: info.routing,
            context_length: info.context_length,
            vocab_size: info.vocab_size,
            support_note: info.support_note,
            model_name: info.model_name,
        }
    }

    pub(crate) fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateOutput> {
        let session = self.model.session(SessionOptions {
            seed: self.default_seed,
            ..SessionOptions::default()
        })?;
        let options = self.generation_options(&request);

        let output = match request.prompt {
            PromptInput::Completion(prompt) => session.generate(&prompt, options)?,
            PromptInput::Chat(messages) => {
                let messages = render_chat_messages(messages);
                session.chat(&messages, options)?
            }
        };

        Ok(convert_output(output))
    }

    pub(crate) fn stream<F>(
        &mut self,
        request: GenerateRequest,
        mut on_chunk: F,
    ) -> anyhow::Result<GenerateOutput>
    where
        F: FnMut(&str) -> anyhow::Result<()>,
    {
        let session = self.model.session(SessionOptions {
            seed: self.default_seed,
            ..SessionOptions::default()
        })?;
        let options = self.generation_options(&request);

        let mut stream = match request.prompt {
            PromptInput::Completion(prompt) => session.stream(&prompt, options)?,
            PromptInput::Chat(messages) => {
                let messages = render_chat_messages(messages);
                session.stream_chat(&messages, options)?
            }
        };

        while let Some(chunk) = stream.next_chunk()? {
            on_chunk(&chunk)?;
        }

        Ok(convert_output(stream.into_output()?))
    }

    fn generation_options(&self, request: &GenerateRequest) -> GenerationOptions {
        GenerationOptions {
            max_tokens: request.max_tokens.unwrap_or(self.default_max_tokens),
            temperature: request.temperature.unwrap_or(0.8),
            top_k: request.top_k.unwrap_or(40),
            top_p: request.top_p.unwrap_or(0.9),
            min_p: request.min_p.unwrap_or(0.0),
            repeat_penalty: request.repeat_penalty.unwrap_or(1.0),
            repeat_last_n: request.repeat_last_n.unwrap_or(64),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            stop_strings: request.stop_strings.clone(),
            seed: request.seed,
        }
    }
}

fn render_chat_messages(messages: Vec<RenderedChatMessage>) -> Vec<SdkChatMessage> {
    messages
        .into_iter()
        .map(|message| {
            let role = match message.role {
                RenderedChatRole::System => SdkChatRole::System,
                RenderedChatRole::User => SdkChatRole::User,
                RenderedChatRole::Assistant => SdkChatRole::Assistant,
            };
            SdkChatMessage::new(role, message.content)
        })
        .collect()
}

fn convert_output(output: ax_engine_sdk::GenerationOutput) -> GenerateOutput {
    GenerateOutput {
        text: output.text,
        finish_reason: match output.finish_reason {
            SdkFinishReason::Stop => FinishReason::Stop,
            SdkFinishReason::Length => FinishReason::Length,
        },
        prompt_tokens: output.usage.prompt_tokens,
        completion_tokens: output.usage.completion_tokens,
    }
}

fn parse_seed(seed: i64) -> Option<u64> {
    if seed < 0 { None } else { Some(seed as u64) }
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}
