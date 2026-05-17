use std::path::{Path, PathBuf};

use ax_engine_core::{EngineCore, MetalKernelAssets};

use crate::backend::{NativeModelArtifactsSource, NativeModelReport, NativeRuntimeArtifactsSource};

use super::config::EngineSessionConfig;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MlxRuntimeArtifactsSelection {
    pub(super) dir: PathBuf,
    pub(super) source: NativeRuntimeArtifactsSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MlxModelArtifactsSelection {
    pub(super) dir: PathBuf,
    pub(super) source: NativeModelArtifactsSource,
}

pub(super) fn resolve_default_mlx_runtime_artifacts_selection(
    explicit_dir: Option<PathBuf>,
    current_dir: Option<&Path>,
) -> Option<MlxRuntimeArtifactsSelection> {
    explicit_dir
        .map(|dir| MlxRuntimeArtifactsSelection {
            dir,
            source: NativeRuntimeArtifactsSource::ExplicitEnv,
        })
        .or_else(|| current_dir.and_then(detect_repo_owned_mlx_runtime_artifacts_dir_from))
}

fn detect_repo_owned_mlx_runtime_artifacts_dir_from(
    start_dir: &Path,
) -> Option<MlxRuntimeArtifactsSelection> {
    for candidate_root in start_dir.ancestors().take(20) {
        let manifest_path = candidate_root.join("metal/phase1-kernels.json");
        let build_dir = candidate_root.join("build/metal");
        let build_report_path = build_dir.join("build_report.json");

        if !manifest_path.is_file() || !build_report_path.is_file() {
            continue;
        }

        // Repo auto-detect should stay conservative: only opt into the Metal
        // bring-up path when the checked-in asset contract validates end to end.
        if MetalKernelAssets::from_build_dir(&build_dir).is_ok() {
            return Some(MlxRuntimeArtifactsSelection {
                dir: build_dir,
                source: NativeRuntimeArtifactsSource::RepoAutoDetect,
            });
        }
    }

    None
}

pub(super) fn resolve_native_model_report(
    config: &EngineSessionConfig,
    core: &EngineCore,
) -> Option<NativeModelReport> {
    let source = config.mlx_model_artifacts_source?;
    let summary = core.native_model_artifacts_summary()?;
    let binding = core.native_model_binding_summary();
    Some(NativeModelReport::from_summary(source, summary, binding))
}
