use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use reqwest::blocking::{Client, Response};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::session::{ChatMessage, FinishReason, GenerationOptions};

#[derive(Debug, Clone)]
pub(crate) struct RemoteGenerationOutput {
    pub(crate) text: String,
    pub(crate) finish_reason: FinishReason,
}

pub(crate) struct LlamaCppProcess {
    model_path: PathBuf,
    port: u16,
    client: Client,
    child: Mutex<Option<Child>>,
}

impl LlamaCppProcess {
    pub(crate) fn spawn(model_path: &Path, context_length: u32) -> anyhow::Result<Self> {
        let binary_path = find_llama_server()?;
        let port = allocate_port()?;
        let client = Client::builder()
            .build()
            .context("failed to construct llama.cpp HTTP client")?;

        let mut command = Command::new(&binary_path);
        command
            .arg("-m")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("-ngl")
            .arg("99")
            .arg("-fa")
            .arg("--ctx-size")
            .arg(context_length.to_string())
            .arg("--log-disable")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        tracing::info!(
            binary = %binary_path.display(),
            model = %model_path.display(),
            port,
            "starting llama-server subprocess"
        );
        let child = command.spawn().with_context(|| {
            format!(
                "failed to spawn llama-server for model {}",
                model_path.display()
            )
        })?;

        let process = Self {
            model_path: model_path.to_path_buf(),
            port,
            client,
            child: Mutex::new(Some(child)),
        };
        process.wait_until_ready()?;
        Ok(process)
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn stream_completion<F>(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_chunk: F,
    ) -> anyhow::Result<RemoteGenerationOutput>
    where
        F: FnMut(&str) -> anyhow::Result<()>,
    {
        let mut payload = base_payload(options, true);
        payload.insert("prompt".to_string(), Value::String(prompt.to_string()));
        let response = self
            .client
            .post(self.endpoint("/v1/completions"))
            .json(&payload)
            .send()
            .context("failed to send completion request to llama-server")?;
        let response = ensure_success(response, "completion request")?;
        parse_completion_sse(response, &mut on_chunk)
    }

    pub(crate) fn stream_chat<F>(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        mut on_chunk: F,
    ) -> anyhow::Result<RemoteGenerationOutput>
    where
        F: FnMut(&str) -> anyhow::Result<()>,
    {
        let mut payload = base_payload(options, true);
        payload.insert(
            "messages".to_string(),
            Value::Array(
                messages
                    .iter()
                    .map(|message| {
                        json!({
                            "role": chat_role_label(message.role),
                            "content": message.content,
                        })
                    })
                    .collect(),
            ),
        );
        let response = self
            .client
            .post(self.endpoint("/v1/chat/completions"))
            .json(&payload)
            .send()
            .context("failed to send chat completion request to llama-server")?;
        let response = ensure_success(response, "chat completion request")?;
        parse_chat_sse(response, &mut on_chunk)
    }

    #[allow(dead_code)]
    pub(crate) fn shutdown(&self) -> anyhow::Result<()> {
        let mut child_guard = self
            .child
            .lock()
            .map_err(|_| anyhow!("llama-server child lock poisoned"))?;
        let Some(mut child) = child_guard.take() else {
            return Ok(());
        };

        let pid = child.id();
        let _ = Command::new("/bin/kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .status();

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if child.try_wait()?.is_some() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                child.kill().context("failed to kill llama-server")?;
                let _ = child.wait();
                return Ok(());
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn wait_until_ready(&self) -> anyhow::Result<()> {
        let timeout = startup_timeout();
        let deadline = Instant::now() + timeout;

        loop {
            if let Ok(mut child) = self.child.lock()
                && let Some(child) = child.as_mut()
                && let Some(status) = child.try_wait()?
            {
                bail!(
                    "llama-server exited before becoming ready (status: {status}) for model {}",
                    self.model_path.display()
                );
            }

            match self.client.get(self.endpoint("/health")).send() {
                Ok(response) if response.status().is_success() => {
                    tracing::info!(
                        model = %self.model_path.display(),
                        port = self.port,
                        "llama-server ready"
                    );
                    return Ok(());
                }
                Ok(_) | Err(_) => {}
            }

            if Instant::now() >= deadline {
                bail!(
                    "llama-server failed to start within {}s for model {}",
                    timeout.as_secs(),
                    self.model_path.display()
                );
            }

            thread::sleep(Duration::from_millis(500));
        }
    }

    fn endpoint(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }
}

impl Drop for LlamaCppProcess {
    fn drop(&mut self) {
        if let Ok(mut child_guard) = self.child.lock()
            && let Some(mut child) = child_guard.take()
        {
            let _ = Command::new("/bin/kill")
                .arg("-TERM")
                .arg(child.id().to_string())
                .status();
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Ok(None) | Err(_) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                }
            }
        }
    }
}

fn ensure_success(response: Response, context: &str) -> anyhow::Result<Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response.text().unwrap_or_default();
    bail!("{context} failed with {status}: {body}");
}

fn parse_completion_sse<F>(
    response: Response,
    on_chunk: &mut F,
) -> anyhow::Result<RemoteGenerationOutput>
where
    F: FnMut(&str) -> anyhow::Result<()>,
{
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;
    parse_sse(response, |payload| {
        if payload == "[DONE]" {
            return Ok(());
        }
        let chunk: CompletionChunk = serde_json::from_str(payload)
            .with_context(|| format!("invalid llama-server completion chunk: {payload}"))?;
        if let Some(error) = chunk.error {
            bail!("{}", error.message);
        }
        if let Some(choice) = chunk.choices.into_iter().next() {
            if !choice.text.is_empty() {
                text.push_str(&choice.text);
                on_chunk(&choice.text)?;
            }
            if let Some(reason) = choice.finish_reason.as_deref() {
                finish_reason = map_finish_reason(reason);
            }
        }
        Ok(())
    })?;

    Ok(RemoteGenerationOutput {
        text,
        finish_reason,
    })
}

fn parse_chat_sse<F>(response: Response, on_chunk: &mut F) -> anyhow::Result<RemoteGenerationOutput>
where
    F: FnMut(&str) -> anyhow::Result<()>,
{
    let mut text = String::new();
    let mut finish_reason = FinishReason::Stop;
    parse_sse(response, |payload| {
        if payload == "[DONE]" {
            return Ok(());
        }
        let chunk: ChatChunk = serde_json::from_str(payload)
            .with_context(|| format!("invalid llama-server chat chunk: {payload}"))?;
        if let Some(error) = chunk.error {
            bail!("{}", error.message);
        }
        if let Some(choice) = chunk.choices.into_iter().next() {
            if let Some(content) = choice.delta.content {
                text.push_str(&content);
                on_chunk(&content)?;
            }
            if let Some(reason) = choice.finish_reason.as_deref() {
                finish_reason = map_finish_reason(reason);
            }
        }
        Ok(())
    })?;

    Ok(RemoteGenerationOutput {
        text,
        finish_reason,
    })
}

fn parse_sse(
    response: Response,
    mut on_event: impl FnMut(&str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let mut reader = BufReader::new(response);
    let mut data_lines = Vec::<String>::new();

    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            if !data_lines.is_empty() {
                on_event(&data_lines.join("\n"))?;
            }
            return Ok(());
        }

        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            if !data_lines.is_empty() {
                on_event(&data_lines.join("\n"))?;
                data_lines.clear();
            }
            continue;
        }

        if let Some(data) = trimmed.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }
}

fn base_payload(options: &GenerationOptions, stream: bool) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("stream".to_string(), Value::Bool(stream));
    payload.insert("max_tokens".to_string(), json!(options.max_tokens));
    payload.insert("temperature".to_string(), json!(options.temperature));
    payload.insert("top_p".to_string(), json!(options.top_p));
    payload.insert("top_k".to_string(), json!(options.top_k));
    payload.insert("min_p".to_string(), json!(options.min_p));
    payload.insert("repeat_penalty".to_string(), json!(options.repeat_penalty));
    payload.insert("repeat_last_n".to_string(), json!(options.repeat_last_n));
    payload.insert(
        "frequency_penalty".to_string(),
        json!(options.frequency_penalty),
    );
    payload.insert(
        "presence_penalty".to_string(),
        json!(options.presence_penalty),
    );
    if !options.stop_strings.is_empty() {
        payload.insert("stop".to_string(), json!(options.stop_strings));
    }
    if let Some(seed) = options.seed {
        payload.insert("seed".to_string(), json!(seed));
    }
    payload
}

fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

fn startup_timeout() -> Duration {
    std::env::var("AX_LLAMA_SERVER_TIMEOUT")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(120))
}

fn allocate_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").context("failed to allocate local port")?;
    Ok(listener.local_addr()?.port())
}

fn find_llama_server() -> anyhow::Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("AX_LLAMA_SERVER_PATH") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Ok(path);
        }
        bail!(
            "AX_LLAMA_SERVER_PATH is set but not executable: {}",
            path.display()
        );
    }

    if let Some(path) = find_on_path("llama-server") {
        return Ok(path);
    }

    for fallback in [
        "/opt/homebrew/bin/llama-server",
        "/usr/local/bin/llama-server",
    ] {
        let path = PathBuf::from(fallback);
        if path.is_file() {
            return Ok(path);
        }
    }

    bail!("llama-server not found. Install via: brew install llama.cpp")
}

fn find_on_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn chat_role_label(role: crate::session::ChatRole) -> &'static str {
    match role {
        crate::session::ChatRole::System => "system",
        crate::session::ChatRole::User => "user",
        crate::session::ChatRole::Assistant => "assistant",
    }
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    message: String,
}

#[derive(Debug, Deserialize)]
struct CompletionChunk {
    #[serde(default)]
    choices: Vec<CompletionChunkChoice>,
    #[serde(default)]
    error: Option<ErrorBody>,
}

#[derive(Debug, Deserialize)]
struct CompletionChunkChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChatChunkChoice>,
    #[serde(default)]
    error: Option<ErrorBody>,
}

#[derive(Debug, Deserialize)]
struct ChatChunkChoice {
    delta: ChatDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatDelta {
    #[serde(default)]
    content: Option<String>,
}
