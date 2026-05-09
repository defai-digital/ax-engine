use crate::contracts::{
    ArtifactEntry, BenchmarkArtifactSummary, DoctorReport, HealthReport, ModelsReport,
    RuntimeInfoReport,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AppTab {
    Readiness,
    Models,
    Server,
    Jobs,
    Benchmarks,
    Artifacts,
}

impl AppTab {
    pub const ALL: [Self; 6] = [
        Self::Readiness,
        Self::Models,
        Self::Server,
        Self::Jobs,
        Self::Benchmarks,
        Self::Artifacts,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Readiness => "Readiness",
            Self::Models => "Models",
            Self::Server => "Server",
            Self::Jobs => "Jobs",
            Self::Benchmarks => "Benchmarks",
            Self::Artifacts => "Artifacts",
        }
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppState {
    pub selected_tab: AppTab,
    pub should_quit: bool,
    pub doctor: LoadState<DoctorReport>,
    pub server: ServerState,
    pub benchmark_summary: LoadState<BenchmarkArtifactSummary>,
    pub artifacts_root: Option<String>,
    pub artifacts: LoadState<Vec<ArtifactEntry>>,
    pub status_message: String,
}

impl AppState {
    pub fn empty() -> Self {
        Self {
            selected_tab: AppTab::Readiness,
            should_quit: false,
            doctor: LoadState::not_loaded("doctor has not run"),
            server: ServerState {
                base_url: None,
                health: LoadState::not_loaded("server URL not configured"),
                runtime: LoadState::not_loaded("server URL not configured"),
                models: LoadState::not_loaded("server URL not configured"),
            },
            benchmark_summary: LoadState::not_loaded("benchmark artifact summary not configured"),
            artifacts_root: None,
            artifacts: LoadState::not_loaded("artifact root not configured"),
            status_message: "Read-only cockpit. Jobs tab shows guarded Phase 2 plans; q exits."
                .to_string(),
        }
    }

    pub fn next_tab(&mut self) {
        let index = self.tab_index();
        self.selected_tab = AppTab::ALL[(index + 1) % AppTab::ALL.len()];
    }

    pub fn previous_tab(&mut self) {
        let index = self.tab_index();
        self.selected_tab = AppTab::ALL[(index + AppTab::ALL.len() - 1) % AppTab::ALL.len()];
    }

    fn tab_index(&self) -> usize {
        AppTab::ALL
            .iter()
            .position(|tab| *tab == self.selected_tab)
            .unwrap_or(0)
    }
}
