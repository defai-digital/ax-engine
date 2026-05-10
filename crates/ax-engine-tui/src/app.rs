use crate::contracts::{
    ArtifactEntry, BenchmarkArtifactSummary, DoctorReport, HealthReport, ModelsReport,
    RuntimeInfoReport,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelKind {
    Text,
    Embedding,
}

impl ModelKind {
    pub const ALL: [Self; 2] = [Self::Text, Self::Embedding];

    pub fn label(self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Embedding => "Embedding",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    Qwen,
    Gemma,
    Glm,
}

impl ModelFamily {
    pub const ALL: [Self; 3] = [Self::Qwen, Self::Gemma, Self::Glm];

    pub fn label(self) -> &'static str {
        match self {
            Self::Qwen => "Qwen",
            Self::Gemma => "Gemma",
            Self::Glm => "GLM",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelCatalogEntry {
    pub kind: ModelKind,
    pub family: ModelFamily,
    pub label: &'static str,
    pub repo_id: &'static str,
    pub note: &'static str,
}

pub const MODEL_CATALOG: &[ModelCatalogEntry] = &[
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Qwen,
        label: "Qwen3-4B-4bit",
        repo_id: "mlx-community/Qwen3-4B-4bit",
        note: "quickstart dense model",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Qwen,
        label: "Qwen3.5-9B-MLX-4bit",
        repo_id: "mlx-community/Qwen3.5-9B-MLX-4bit",
        note: "linear attention + MoE preview",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Qwen,
        label: "Qwen3.6-35B-A3B-UD-MLX-4bit",
        repo_id: "mlx-community/Qwen3.6-35B-A3B-UD-MLX-4bit",
        note: "large MoE, 4-bit",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Qwen,
        label: "Qwen3.6-35B-A3B-5bit",
        repo_id: "mlx-community/Qwen3.6-35B-A3B-5bit",
        note: "large MoE, 5-bit",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Qwen,
        label: "Qwen3.6-35B-A3B-6bit",
        repo_id: "mlx-community/Qwen3.6-35B-A3B-6bit",
        note: "large MoE, 6-bit",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Qwen,
        label: "Qwen3.6-35B-A3B-8bit",
        repo_id: "mlx-community/Qwen3.6-35B-A3B-8bit",
        note: "large MoE, 8-bit",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Qwen,
        label: "Qwen3-Coder-Next-4bit",
        repo_id: "mlx-community/Qwen3-Coder-Next-4bit",
        note: "coder next preview",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Gemma,
        label: "gemma-4-e2b-it-4bit",
        repo_id: "mlx-community/gemma-4-e2b-it-4bit",
        note: "small Gemma 4 dense",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Gemma,
        label: "gemma-4-e2b-it-5bit",
        repo_id: "mlx-community/gemma-4-e2b-it-5bit",
        note: "small Gemma 4 dense",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Gemma,
        label: "gemma-4-e2b-it-6bit",
        repo_id: "mlx-community/gemma-4-e2b-it-6bit",
        note: "small Gemma 4 dense",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Gemma,
        label: "gemma-4-e2b-it-8bit",
        repo_id: "mlx-community/gemma-4-e2b-it-8bit",
        note: "small Gemma 4 dense",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Gemma,
        label: "gemma-4-e4b-it-4bit",
        repo_id: "mlx-community/gemma-4-e4b-it-4bit",
        note: "Gemma 4 E4B",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Gemma,
        label: "gemma-4-26b-a4b-it-4bit",
        repo_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
        note: "large Gemma 4 MoE",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Gemma,
        label: "gemma-4-31b-it-4bit",
        repo_id: "mlx-community/gemma-4-31b-it-4bit",
        note: "large Gemma 4 dense",
    },
    ModelCatalogEntry {
        kind: ModelKind::Text,
        family: ModelFamily::Glm,
        label: "GLM-4.7-Flash-4bit",
        repo_id: "mlx-community/GLM-4.7-Flash-4bit",
        note: "GLM runtime-ready preview",
    },
    ModelCatalogEntry {
        kind: ModelKind::Embedding,
        family: ModelFamily::Qwen,
        label: "Qwen3-Embedding-0.6B-8bit",
        repo_id: "mlx-community/Qwen3-Embedding-0.6B-8bit",
        note: "small embedding model",
    },
    ModelCatalogEntry {
        kind: ModelKind::Embedding,
        family: ModelFamily::Qwen,
        label: "Qwen3-Embedding-4B-4bit-DWQ",
        repo_id: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        note: "balanced embedding model",
    },
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelDownloadStatus {
    Idle,
    Running(String),
    Succeeded { repo_id: String, model_dir: String },
    Failed { repo_id: String, message: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelDownloadState {
    pub kind: ModelKind,
    pub family: ModelFamily,
    pub size_index: usize,
    pub status: ModelDownloadStatus,
}

impl Default for ModelDownloadState {
    fn default() -> Self {
        Self {
            kind: ModelKind::Text,
            family: ModelFamily::Qwen,
            size_index: 0,
            status: ModelDownloadStatus::Idle,
        }
    }
}

impl ModelDownloadState {
    pub fn entries(&self) -> Vec<ModelCatalogEntry> {
        MODEL_CATALOG
            .iter()
            .copied()
            .filter(|entry| entry.kind == self.kind && entry.family == self.family)
            .collect()
    }

    pub fn selected_entry(&self) -> ModelCatalogEntry {
        let entries = self.entries();
        entries.get(self.size_index).copied().unwrap_or_else(|| {
            MODEL_CATALOG
                .iter()
                .copied()
                .find(|entry| entry.kind == self.kind)
                .unwrap_or(MODEL_CATALOG[0])
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoadState<T> {
    NotLoaded(String),
    Ready(T),
    Unavailable(String),
}

impl<T> LoadState<T> {
    pub fn not_loaded(label: &str) -> Self {
        Self::NotLoaded(label.to_string())
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::Unavailable(message.into())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerState {
    pub base_url: Option<String>,
    pub health: LoadState<HealthReport>,
    pub runtime: LoadState<RuntimeInfoReport>,
    pub models: LoadState<ModelsReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServerUrlKind {
    Health,
    Runtime,
    Models,
    Embeddings,
    Generate,
    GenerateStream,
    ChatCompletions,
    Completions,
}

impl ServerUrlKind {
    pub const ALL: [Self; 8] = [
        Self::Health,
        Self::Runtime,
        Self::Models,
        Self::Embeddings,
        Self::Generate,
        Self::GenerateStream,
        Self::ChatCompletions,
        Self::Completions,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Health => "Health",
            Self::Runtime => "Runtime",
            Self::Models => "Models",
            Self::Embeddings => "Embeddings",
            Self::Generate => "Generate",
            Self::GenerateStream => "Generate stream",
            Self::ChatCompletions => "Chat completions",
            Self::Completions => "Completions",
        }
    }

    pub fn path(self) -> &'static str {
        match self {
            Self::Health => "/health",
            Self::Runtime => "/v1/runtime",
            Self::Models => "/v1/models",
            Self::Embeddings => "/v1/embeddings",
            Self::Generate => "/v1/generate",
            Self::GenerateStream => "/v1/generate/stream",
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerEndpoint {
    pub kind: ServerUrlKind,
    pub url: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerControlState {
    pub scheme: String,
    pub host: String,
    pub port: u16,
}

impl Default for ServerControlState {
    fn default() -> Self {
        Self {
            scheme: "http".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8080,
        }
    }
}

impl ServerControlState {
    pub fn from_base_url(base_url: &str) -> Option<Self> {
        let trimmed = base_url.trim().trim_end_matches('/');
        if trimmed.is_empty() {
            return None;
        }

        let (scheme, rest) = if let Some(rest) = trimmed.strip_prefix("http://") {
            ("http", rest)
        } else if let Some(rest) = trimmed.strip_prefix("https://") {
            ("https", rest)
        } else {
            ("http", trimmed)
        };
        let authority = rest.split('/').next().unwrap_or_default();
        if authority.is_empty() {
            return None;
        }

        let default_port = if scheme == "https" { 443 } else { 80 };
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port))
                if !host.is_empty() && port.chars().all(|ch| ch.is_ascii_digit()) =>
            {
                (host.to_string(), port.parse().ok()?)
            }
            _ => (authority.to_string(), default_port),
        };

        Some(Self {
            scheme: scheme.to_string(),
            host,
            port,
        })
    }

    pub fn base_url(&self) -> String {
        format!("{}://{}:{}", self.scheme, self.host, self.port)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppState {
    pub doctor: LoadState<DoctorReport>,
    pub model_download: ModelDownloadState,
    pub server: ServerState,
    pub server_control: ServerControlState,
    pub benchmark_summary: LoadState<BenchmarkArtifactSummary>,
    pub artifacts_root: Option<String>,
    pub artifacts: LoadState<Vec<ArtifactEntry>>,
    pub status_message: String,
}

impl AppState {
    pub fn empty() -> Self {
        Self {
            doctor: LoadState::not_loaded("doctor has not run"),
            model_download: ModelDownloadState::default(),
            server: ServerState {
                base_url: None,
                health: LoadState::not_loaded("server URL not configured"),
                runtime: LoadState::not_loaded("server URL not configured"),
                models: LoadState::not_loaded("server URL not configured"),
            },
            server_control: ServerControlState::default(),
            benchmark_summary: LoadState::not_loaded("benchmark artifact summary not configured"),
            artifacts_root: None,
            artifacts: LoadState::not_loaded("artifact root not configured"),
            status_message: "Web manager ready.".to_string(),
        }
    }

    pub fn select_model_family(&mut self, family: ModelFamily) {
        self.model_download.family = family;
        self.model_download.size_index = 0;
        let entry = self.model_download.selected_entry();
        self.status_message = format!(
            "Selected model family {}: {}",
            family.label(),
            entry.repo_id
        );
    }

    pub fn select_model_size(&mut self, size_index: usize) {
        let max_index = self.model_download.entries().len().saturating_sub(1);
        self.model_download.size_index = size_index.min(max_index);
        let entry = self.model_download.selected_entry();
        self.status_message = format!("Selected model size {}: {}", entry.label, entry.repo_id);
    }

    pub fn select_model_download(&mut self) {
        let entry = self.model_download.selected_entry();
        self.status_message = format!("Ready to download {}", entry.repo_id);
    }

    pub fn mark_model_download_running(&mut self) {
        let repo_id = self.model_download.selected_entry().repo_id.to_string();
        self.model_download.status = ModelDownloadStatus::Running(repo_id.clone());
        self.status_message = format!("Downloading {repo_id}...");
    }

    pub fn mark_model_download_succeeded(&mut self, model_dir: String) {
        let repo_id = self.model_download.selected_entry().repo_id.to_string();
        self.model_download.status = ModelDownloadStatus::Succeeded {
            repo_id: repo_id.clone(),
            model_dir: model_dir.clone(),
        };
        self.status_message = format!("Downloaded {repo_id} to {model_dir}");
    }

    pub fn mark_model_download_failed(&mut self, message: impl Into<String>) {
        let repo_id = self.model_download.selected_entry().repo_id.to_string();
        let message = message.into();
        self.model_download.status = ModelDownloadStatus::Failed {
            repo_id: repo_id.clone(),
            message: message.clone(),
        };
        self.status_message = format!("Download failed for {repo_id}: {message}");
    }

    pub fn server_base_url(&self) -> String {
        self.server
            .base_url
            .clone()
            .unwrap_or_else(|| self.server_control.base_url())
    }

    pub fn server_url(&self, kind: ServerUrlKind) -> String {
        format!(
            "{}{}",
            self.server_base_url().trim_end_matches('/'),
            kind.path()
        )
    }

    pub fn server_endpoints(&self) -> Vec<ServerEndpoint> {
        ServerUrlKind::ALL
            .into_iter()
            .map(|kind| ServerEndpoint {
                kind,
                url: self.server_url(kind),
            })
            .collect()
    }
}
