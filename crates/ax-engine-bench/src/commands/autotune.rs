#[cfg(test)]
#[cfg(test)]
use super::*;

#[cfg(test)]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct AutotuneCandidateConfig {
    pub(crate) max_batch_tokens: u32,
    pub(crate) kv_total_blocks: Option<u32>,
    pub(crate) prefix_cache: bool,
}

#[cfg(test)]
impl AutotuneCandidateConfig {
    pub(crate) fn label(&self) -> String {
        format!(
            "max_batch_tokens={}; kv_total_blocks={}; prefix_cache={}",
            self.max_batch_tokens,
            optional_u32_label(self.kv_total_blocks),
            self.prefix_cache
        )
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AutotuneSearchSpace {
    pub(crate) max_batch_token_options: Vec<u32>,
    pub(crate) kv_total_block_options: Vec<Option<u32>>,
    pub(crate) prefix_cache_options: Vec<bool>,
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct AutotuneWarmStartHistory {
    pub(crate) trials: Vec<AutotuneTrialRecord>,
    pub(crate) source_dirs: Vec<PathBuf>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct AutotuneHistoryIndexEntry {
    pub(crate) result_dir: PathBuf,
    pub(crate) manifest_id: String,
    pub(crate) selected_backend: String,
    pub(crate) trials: Vec<Value>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct AutotuneHistoryIndexSummary {
    pub(crate) manifest_id: String,
    pub(crate) selected_backend: String,
    pub(crate) source_dirs: Vec<PathBuf>,
    pub(crate) best_trials: Vec<Value>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct AutotuneSelectionDiagnostics {
    pub(crate) strategy: String,
    pub(crate) predicted_mean: Option<f64>,
    pub(crate) uncertainty: Option<f64>,
    pub(crate) acquisition: Option<f64>,
    pub(crate) good_density: Option<f64>,
    pub(crate) bad_density: Option<f64>,
    pub(crate) density_ratio: Option<f64>,
    pub(crate) novelty_bonus: Option<f64>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct AutotuneCandidateSelection {
    pub(crate) candidate_index: usize,
    pub(crate) diagnostics: AutotuneSelectionDiagnostics,
}

#[cfg(test)]
#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub(crate) struct AutotuneTrialMetrics {
    pub(crate) ttft_ms: Option<u64>,
    pub(crate) prefill_tok_s: f64,
    pub(crate) decode_tok_s: f64,
    pub(crate) prefix_hit_rate: f64,
    pub(crate) prefix_native_dispatch_count: u32,
    pub(crate) prefix_cpu_reference_dispatch_count: u32,
    pub(crate) prefix_cpu_projection_row_count: u32,
    pub(crate) prefix_cpu_rms_norm_element_count: u32,
    pub(crate) prefix_cpu_ffn_activation_element_count: u32,
    pub(crate) prefix_cpu_residual_add_element_count: u32,
    pub(crate) prefix_cpu_scale_element_count: u32,
    pub(crate) direct_decode_batched_group_fallback_count: u32,
    pub(crate) direct_decode_cpu_projection_row_count: u32,
    pub(crate) direct_decode_cpu_rms_norm_element_count: u32,
    pub(crate) direct_decode_cpu_ffn_activation_element_count: u32,
    pub(crate) direct_decode_cpu_residual_add_element_count: u32,
    pub(crate) direct_decode_cpu_scale_element_count: u32,
    pub(crate) mlx_metal_hot_path_cpu_fallback_free: bool,
    pub(crate) real_model_forward: bool,
    pub(crate) model_bound_ffn_decode: bool,
}

#[cfg(test)]
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct AutotuneTrialRecord {
    pub(crate) trial_index: usize,
    pub(crate) candidate: AutotuneCandidateConfig,
    pub(crate) selection: AutotuneSelectionDiagnostics,
    pub(crate) probe: Option<AutotuneProbeObservation>,
    pub(crate) score: f64,
    pub(crate) status: String,
    pub(crate) result_dir: Option<PathBuf>,
    pub(crate) error: Option<String>,
    pub(crate) metrics: Option<AutotuneTrialMetrics>,
}

#[cfg(test)]
#[allow(dead_code)]
impl AutotuneTrialRecord {
    pub(crate) fn label(&self) -> String {
        format!("trial-{:03}", self.trial_index + 1)
    }
}

#[cfg(test)]
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct AutotuneProbeObservation {
    pub(crate) shape: ScenarioShape,
    pub(crate) status: String,
    pub(crate) score: f64,
    pub(crate) skipped_full_trial: bool,
    pub(crate) skip_reason: Option<String>,
    pub(crate) result_dir: Option<PathBuf>,
    pub(crate) baseline: Option<AutotuneProbeBaseline>,
    pub(crate) metrics: AutotuneTrialMetrics,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct AutotuneProbeBaseline {
    pub(crate) observed_trial_count: usize,
    pub(crate) incumbent_score: Option<f64>,
    pub(crate) incumbent_decode_tok_s: Option<f64>,
    pub(crate) score_floor: Option<f64>,
    pub(crate) decode_tok_s_floor: Option<f64>,
    pub(crate) fallback_free_decode_tok_s_floor: Option<f64>,
    pub(crate) ttft_ceiling_ms: Option<u64>,
}

#[cfg(test)]
pub(crate) const AUTOTUNE_SCHEMA_VERSION: &str = "ax.engine_bench.autotune.v1";
#[cfg(test)]
pub(crate) const AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION: &str =
    "ax.engine_bench.autotune_history_index.v1";
#[cfg(test)]
#[allow(dead_code)]
pub(crate) const AUTOTUNE_FAILURE_SCORE: f64 = -1_000_000.0;

#[cfg(test)]
pub(crate) fn resolve_autotune_search_space(
    manifest: &BenchmarkManifest,
    args: &AutotuneArgs,
) -> AutotuneSearchSpace {
    let base_max_batch_tokens = manifest.runtime.max_batch_tokens.max(1);
    let max_batch_token_options = args.max_batch_token_options.clone().unwrap_or_else(|| {
        unique_sorted_u32(vec![
            (base_max_batch_tokens / 2).max(1),
            base_max_batch_tokens,
            base_max_batch_tokens.saturating_mul(2).max(1),
        ])
    });
    let kv_total_block_options = args.kv_total_block_options.clone().unwrap_or_else(|| {
        if let Some(base_kv_total_blocks) = manifest.runtime.kv_total_blocks {
            unique_sorted_option_u32(vec![
                Some((base_kv_total_blocks / 2).max(1)),
                Some(base_kv_total_blocks.max(1)),
                Some(base_kv_total_blocks.saturating_mul(2).max(1)),
            ])
        } else {
            unique_sorted_option_u32(vec![None, Some(32), Some(64)])
        }
    });
    let prefix_cache_options = args.prefix_cache_options.clone().unwrap_or_else(|| {
        if manifest.runtime.flags.prefix_cache {
            vec![true, false]
        } else {
            vec![false, true]
        }
    });

    AutotuneSearchSpace {
        max_batch_token_options,
        kv_total_block_options,
        prefix_cache_options,
    }
}

#[cfg(test)]
pub(crate) fn autotune_candidate_configs(
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
) -> Vec<AutotuneCandidateConfig> {
    let base_candidate = AutotuneCandidateConfig {
        max_batch_tokens: manifest.runtime.max_batch_tokens.max(1),
        kv_total_blocks: manifest.runtime.kv_total_blocks,
        prefix_cache: manifest.runtime.flags.prefix_cache,
    };

    let mut candidates = Vec::new();
    for max_batch_tokens in &search_space.max_batch_token_options {
        for kv_total_blocks in &search_space.kv_total_block_options {
            for prefix_cache in &search_space.prefix_cache_options {
                candidates.push(AutotuneCandidateConfig {
                    max_batch_tokens: *max_batch_tokens,
                    kv_total_blocks: *kv_total_blocks,
                    prefix_cache: *prefix_cache,
                });
            }
        }
    }

    candidates.sort_by(|left, right| {
        (
            left != &base_candidate,
            left.max_batch_tokens,
            left.kv_total_blocks.unwrap_or(0),
            left.prefix_cache,
        )
            .cmp(&(
                right != &base_candidate,
                right.max_batch_tokens,
                right.kv_total_blocks.unwrap_or(0),
                right.prefix_cache,
            ))
    });
    candidates
}

#[cfg(test)]
pub(crate) fn select_next_autotune_candidate(
    candidates: &[AutotuneCandidateConfig],
    search_space: &AutotuneSearchSpace,
    trials: &[AutotuneTrialRecord],
    exploration_weight: f64,
) -> AutotuneCandidateSelection {
    if trials.is_empty() {
        return AutotuneCandidateSelection {
            candidate_index: 0,
            diagnostics: AutotuneSelectionDiagnostics {
                strategy: "base_config_seed".to_string(),
                predicted_mean: None,
                uncertainty: None,
                acquisition: None,
                good_density: None,
                bad_density: None,
                density_ratio: None,
                novelty_bonus: None,
            },
        };
    }
    if trials.len() == 1 {
        return select_next_autotune_candidate_by_coverage(candidates, trials);
    }

    select_next_autotune_candidate_by_tpe(candidates, search_space, trials, exploration_weight)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn execute_autotune_trial(
    manifest_path: &Path,
    base_manifest: &BenchmarkManifest,
    trials_dir: &Path,
    trial_index: usize,
    candidate: &AutotuneCandidateConfig,
    selection: AutotuneSelectionDiagnostics,
    observed_trials: &[AutotuneTrialRecord],
    started_at_unix_s: u64,
) -> Result<AutotuneTrialRecord, CliError> {
    let manifest = autotune_trial_manifest(base_manifest, trial_index, candidate);
    let probe = execute_autotune_probe_trial(
        manifest_path,
        base_manifest,
        trials_dir,
        trial_index,
        candidate,
        observed_trials,
        started_at_unix_s,
    )?;
    if probe.as_ref().is_some_and(|probe| probe.skipped_full_trial) {
        let probe = probe.expect("checked above");
        return Ok(AutotuneTrialRecord {
            trial_index,
            candidate: candidate.clone(),
            selection,
            probe: Some(probe.clone()),
            score: probe.score,
            status: "early_stopped".to_string(),
            result_dir: probe.result_dir.clone(),
            error: probe.skip_reason.clone(),
            metrics: Some(probe.metrics.clone()),
        });
    }
    match execute_manifest_runtime(&manifest) {
        Ok(execution) => {
            let result_dir = write_execution_artifacts(
                "autotune",
                manifest_path,
                &manifest,
                trials_dir,
                started_at_unix_s,
                &execution,
                true,
            )?;
            let metrics = autotune_trial_metrics(&execution);
            let score = autotune_score(&execution, &metrics);
            Ok(AutotuneTrialRecord {
                trial_index,
                candidate: candidate.clone(),
                selection,
                probe,
                score,
                status: execution.status_label().to_string(),
                result_dir: Some(result_dir),
                error: None,
                metrics: Some(metrics),
            })
        }
        Err(CliError::Contract(message)) => {
            let result_dir = write_contract_failure_artifacts(
                "autotune",
                manifest_path,
                &manifest,
                trials_dir,
                started_at_unix_s,
                &message,
            )?;
            Ok(AutotuneTrialRecord {
                trial_index,
                candidate: candidate.clone(),
                selection,
                probe,
                score: AUTOTUNE_FAILURE_SCORE,
                status: "contract_failure".to_string(),
                result_dir: Some(result_dir),
                error: Some(message),
                metrics: None,
            })
        }
        Err(error) => Ok(AutotuneTrialRecord {
            trial_index,
            candidate: candidate.clone(),
            selection,
            probe,
            score: AUTOTUNE_FAILURE_SCORE,
            status: "runtime_error".to_string(),
            result_dir: None,
            error: Some(error.to_string()),
            metrics: None,
        }),
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn execute_autotune_probe_trial(
    manifest_path: &Path,
    base_manifest: &BenchmarkManifest,
    trials_dir: &Path,
    trial_index: usize,
    candidate: &AutotuneCandidateConfig,
    observed_trials: &[AutotuneTrialRecord],
    started_at_unix_s: u64,
) -> Result<Option<AutotuneProbeObservation>, CliError> {
    let Some(probe_manifest) = autotune_probe_manifest(base_manifest, trial_index, candidate)
    else {
        return Ok(None);
    };
    let baseline = autotune_probe_baseline(observed_trials);
    let probe_shape = probe_manifest.shape.clone().ok_or_else(|| {
        CliError::Contract("autotune probe manifest missing scenario shape".to_string())
    })?;
    let probes_dir = trials_dir.join("probes");
    fs::create_dir_all(&probes_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create autotune probe directory {}: {error}",
            probes_dir.display()
        ))
    })?;

    match execute_manifest_runtime(&probe_manifest) {
        Ok(execution) => {
            let result_dir = write_execution_artifacts(
                "autotune-probe",
                manifest_path,
                &probe_manifest,
                &probes_dir,
                started_at_unix_s,
                &execution,
                true,
            )?;
            let metrics = autotune_trial_metrics(&execution);
            let score = autotune_score(&execution, &metrics);
            let skip_reason =
                autotune_probe_skip_reason(&execution, &metrics, baseline.as_ref(), score);
            Ok(Some(AutotuneProbeObservation {
                shape: probe_shape,
                status: execution.status_label().to_string(),
                score,
                skipped_full_trial: skip_reason.is_some(),
                skip_reason,
                result_dir: Some(result_dir),
                baseline,
                metrics,
            }))
        }
        Err(CliError::Contract(message)) => {
            let result_dir = write_contract_failure_artifacts(
                "autotune-probe",
                manifest_path,
                &probe_manifest,
                &probes_dir,
                started_at_unix_s,
                &message,
            )?;
            Ok(Some(AutotuneProbeObservation {
                shape: probe_shape,
                status: "contract_failure".to_string(),
                score: AUTOTUNE_FAILURE_SCORE,
                skipped_full_trial: true,
                skip_reason: Some(format!("probe contract failure: {message}")),
                result_dir: Some(result_dir),
                baseline,
                metrics: AutotuneTrialMetrics::default(),
            }))
        }
        Err(error) => Ok(Some(AutotuneProbeObservation {
            shape: probe_shape,
            status: "runtime_error".to_string(),
            score: AUTOTUNE_FAILURE_SCORE,
            skipped_full_trial: true,
            skip_reason: Some(format!("probe runtime error: {error}")),
            result_dir: None,
            baseline,
            metrics: AutotuneTrialMetrics::default(),
        })),
    }
}

#[cfg(test)]
pub(crate) fn autotune_probe_manifest(
    base_manifest: &BenchmarkManifest,
    trial_index: usize,
    candidate: &AutotuneCandidateConfig,
) -> Option<BenchmarkManifest> {
    let mut manifest = autotune_trial_manifest(base_manifest, trial_index, candidate);
    let shape = manifest.shape.as_mut()?;
    shape.input_tokens_target = shape.input_tokens_target.clamp(1, 128);
    shape.output_tokens_target = shape.output_tokens_target.clamp(1, 8);
    shape.concurrency = shape.concurrency.clamp(1, 2);
    manifest.id = format!("{}-probe", manifest.id);
    manifest.checks.expect_deterministic = false;
    manifest.notes = Some(format!(
        "autotune probe {} using max_batch_tokens={}, kv_total_blocks={}, prefix_cache={}",
        trial_index + 1,
        candidate.max_batch_tokens,
        optional_u32_label(candidate.kv_total_blocks),
        candidate.prefix_cache
    ));
    Some(manifest)
}

#[cfg(test)]
pub(crate) fn autotune_probe_skip_reason(
    execution: &RuntimeResult,
    metrics: &AutotuneTrialMetrics,
    baseline: Option<&AutotuneProbeBaseline>,
    probe_score: f64,
) -> Option<String> {
    if !execution.correctness.passed {
        return Some(gate_failure_reason(
            &execution.correctness,
            "probe correctness gate failed",
        ));
    }
    if !execution.determinism.passed {
        return Some(gate_failure_reason(
            &execution.determinism,
            "probe determinism gate failed",
        ));
    }
    if !metrics.real_model_forward {
        return Some("probe did not reach real-model forward coverage".to_string());
    }
    if metrics.decode_tok_s <= 0.0 {
        return Some("probe decode throughput was zero".to_string());
    }
    let has_prefix_cpu_fallback = metrics.prefix_cpu_reference_dispatch_count > 0;
    let has_decode_cpu_fallback = metrics.direct_decode_batched_group_fallback_count > 0
        || metrics.direct_decode_cpu_projection_row_count > 0
        || metrics.direct_decode_cpu_scale_element_count > 0;
    let has_hot_path_cpu_fallback = has_prefix_cpu_fallback || has_decode_cpu_fallback;
    let has_incumbent_cpu_fallback = has_prefix_cpu_fallback
        || metrics.direct_decode_batched_group_fallback_count > 0
        || metrics.direct_decode_cpu_scale_element_count > 0;
    if has_prefix_cpu_fallback && has_decode_cpu_fallback {
        return Some("probe hit CPU fallback in both prefix and decode hot paths".to_string());
    }

    let baseline = baseline?;
    if !metrics.mlx_metal_hot_path_cpu_fallback_free
        && baseline
            .fallback_free_decode_tok_s_floor
            .is_some_and(|floor| floor > 0.0 && metrics.decode_tok_s < floor * 0.75)
    {
        return Some(format!(
            "probe decode tok/s {:.2} stayed below fallback-free floor {:.2} with CPU fallback",
            metrics.decode_tok_s,
            baseline
                .fallback_free_decode_tok_s_floor
                .unwrap_or_default()
        ));
    }
    if baseline
        .score_floor
        .is_some_and(|floor| floor > 0.0 && probe_score < floor * 0.7 && has_hot_path_cpu_fallback)
    {
        return Some(format!(
            "probe score {:.3} stayed below adaptive floor {:.3}",
            probe_score,
            baseline.score_floor.unwrap_or_default()
        ));
    }
    if baseline.ttft_ceiling_ms.is_some_and(|ceiling| {
        metrics.ttft_ms.is_some_and(|ttft_ms| {
            ttft_ms > ceiling.saturating_mul(2)
                && baseline
                    .decode_tok_s_floor
                    .is_some_and(|floor| floor > 0.0 && metrics.decode_tok_s < floor)
        })
    }) {
        return Some(format!(
            "probe TTFT {}ms exceeded adaptive ceiling {}ms while decode tok/s stayed low",
            metrics.ttft_ms.unwrap_or_default(),
            baseline.ttft_ceiling_ms.unwrap_or_default()
        ));
    }
    if !metrics.mlx_metal_hot_path_cpu_fallback_free
        && baseline
            .incumbent_decode_tok_s
            .is_some_and(|decode| decode > 0.0 && metrics.decode_tok_s < decode * 0.6)
    {
        return Some(format!(
            "probe decode tok/s {:.2} fell well below incumbent {:.2} with extra CPU fallback",
            metrics.decode_tok_s,
            baseline.incumbent_decode_tok_s.unwrap_or_default()
        ));
    }
    if baseline.incumbent_score.is_some_and(|score| {
        score > 0.0 && probe_score < score * 0.55 && has_incumbent_cpu_fallback
    }) {
        return Some(format!(
            "probe score {:.3} stayed far below incumbent {:.3}",
            probe_score,
            baseline.incumbent_score.unwrap_or_default()
        ));
    }

    None
}

#[cfg(test)]
pub(crate) fn autotune_probe_baseline(
    observed_trials: &[AutotuneTrialRecord],
) -> Option<AutotuneProbeBaseline> {
    let incumbent = best_autotune_trial(observed_trials.iter())?;
    let score_floor = percentile_f64(
        &observed_trials
            .iter()
            .map(|trial| trial.score)
            .collect::<Vec<_>>(),
        0.35,
    );
    let decode_tok_s_floor = percentile_f64(
        &observed_trials
            .iter()
            .filter_map(|trial| trial.metrics.as_ref().map(|metrics| metrics.decode_tok_s))
            .collect::<Vec<_>>(),
        0.35,
    );
    let fallback_free_decode_tok_s_floor = percentile_f64(
        &observed_trials
            .iter()
            .filter_map(|trial| {
                trial.metrics.as_ref().and_then(|metrics| {
                    metrics
                        .mlx_metal_hot_path_cpu_fallback_free
                        .then_some(metrics.decode_tok_s)
                })
            })
            .collect::<Vec<_>>(),
        0.5,
    );
    let ttft_ceiling_ms = percentile_u64(
        &observed_trials
            .iter()
            .filter_map(|trial| trial.metrics.as_ref().and_then(|metrics| metrics.ttft_ms))
            .collect::<Vec<_>>(),
        0.75,
    );

    Some(AutotuneProbeBaseline {
        observed_trial_count: observed_trials.len(),
        incumbent_score: Some(incumbent.score),
        incumbent_decode_tok_s: incumbent
            .metrics
            .as_ref()
            .map(|metrics| metrics.decode_tok_s),
        score_floor,
        decode_tok_s_floor,
        fallback_free_decode_tok_s_floor,
        ttft_ceiling_ms,
    })
}

#[cfg(test)]
pub(crate) fn select_next_autotune_candidate_by_coverage(
    candidates: &[AutotuneCandidateConfig],
    trials: &[AutotuneTrialRecord],
) -> AutotuneCandidateSelection {
    let tried = collect_tried_autotune_candidates(trials);
    let mut best_index = None;
    let mut best_novelty = f64::NEG_INFINITY;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        if tried.contains(candidate) {
            continue;
        }
        let novelty = autotune_candidate_novelty(candidate, trials);
        if novelty > best_novelty {
            best_novelty = novelty;
            best_index = Some(candidate_index);
        }
    }

    AutotuneCandidateSelection {
        candidate_index: best_index.unwrap_or(0),
        diagnostics: AutotuneSelectionDiagnostics {
            strategy: "coverage_explore".to_string(),
            predicted_mean: None,
            uncertainty: Some(best_novelty.max(0.0)),
            acquisition: Some(best_novelty.max(0.0)),
            good_density: None,
            bad_density: None,
            density_ratio: None,
            novelty_bonus: Some(best_novelty.max(0.0)),
        },
    }
}

#[cfg(test)]
pub(crate) fn select_next_autotune_candidate_by_tpe(
    candidates: &[AutotuneCandidateConfig],
    search_space: &AutotuneSearchSpace,
    trials: &[AutotuneTrialRecord],
    exploration_weight: f64,
) -> AutotuneCandidateSelection {
    let tried = collect_tried_autotune_candidates(trials);
    let (good_trials, bad_trials) = split_autotune_trials_by_score(trials);

    let mut best_index = None;
    let mut best_diagnostics = None;
    let mut best_acquisition = f64::NEG_INFINITY;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        if tried.contains(candidate) {
            continue;
        }

        let good_density =
            autotune_candidate_density(candidate, search_space, &good_trials).max(f64::EPSILON);
        let bad_density =
            autotune_candidate_density(candidate, search_space, &bad_trials).max(f64::EPSILON);
        let density_ratio = good_density / bad_density.max(f64::EPSILON);
        let novelty = autotune_candidate_novelty(candidate, trials);
        let posterior_good = good_density / (good_density + bad_density).max(f64::EPSILON);
        let acquisition = density_ratio.ln() + exploration_weight * novelty;

        if acquisition > best_acquisition {
            best_acquisition = acquisition;
            best_index = Some(candidate_index);
            best_diagnostics = Some(AutotuneSelectionDiagnostics {
                strategy: "tpe_ratio".to_string(),
                predicted_mean: Some(posterior_good),
                uncertainty: Some(novelty),
                acquisition: Some(acquisition),
                good_density: Some(good_density),
                bad_density: Some(bad_density),
                density_ratio: Some(density_ratio),
                novelty_bonus: Some(novelty),
            });
        }
    }

    AutotuneCandidateSelection {
        candidate_index: best_index.unwrap_or(0),
        diagnostics: best_diagnostics.unwrap_or(AutotuneSelectionDiagnostics {
            strategy: "tpe_ratio".to_string(),
            predicted_mean: None,
            uncertainty: Some(0.0),
            acquisition: Some(f64::NEG_INFINITY),
            good_density: None,
            bad_density: None,
            density_ratio: None,
            novelty_bonus: Some(0.0),
        }),
    }
}

#[cfg(test)]
pub(crate) fn collect_tried_autotune_candidates(
    trials: &[AutotuneTrialRecord],
) -> BTreeSet<AutotuneCandidateConfig> {
    trials
        .iter()
        .map(|trial| trial.candidate.clone())
        .collect::<BTreeSet<_>>()
}

#[cfg(test)]
pub(crate) fn split_autotune_trials_by_score(
    trials: &[AutotuneTrialRecord],
) -> (Vec<&AutotuneTrialRecord>, Vec<&AutotuneTrialRecord>) {
    let mut ordered = trials.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let good_count = ((ordered.len() as f64) * 0.35).ceil() as usize;
    let good_count = good_count
        .max(1)
        .min(ordered.len().saturating_sub(1).max(1));
    let good_trials = ordered.iter().take(good_count).copied().collect::<Vec<_>>();
    let bad_trials = ordered.iter().skip(good_count).copied().collect::<Vec<_>>();
    let bad_trials = if bad_trials.is_empty() {
        good_trials.clone()
    } else {
        bad_trials
    };
    (good_trials, bad_trials)
}

#[cfg(test)]
pub(crate) fn autotune_candidate_density(
    candidate: &AutotuneCandidateConfig,
    search_space: &AutotuneSearchSpace,
    trials: &[&AutotuneTrialRecord],
) -> f64 {
    smoothed_probability(
        candidate.max_batch_tokens,
        search_space.max_batch_token_options.len(),
        trials,
        |config| config.max_batch_tokens,
    ) * smoothed_probability(
        candidate.kv_total_blocks,
        search_space.kv_total_block_options.len(),
        trials,
        |config| config.kv_total_blocks,
    ) * smoothed_probability(
        candidate.prefix_cache,
        search_space.prefix_cache_options.len(),
        trials,
        |config| config.prefix_cache,
    )
}

#[cfg(test)]
pub(crate) fn autotune_candidate_novelty(
    candidate: &AutotuneCandidateConfig,
    trials: &[AutotuneTrialRecord],
) -> f64 {
    let max_batch_rarity = rarity_score(trials, candidate, |config| config.max_batch_tokens);
    let kv_total_blocks_rarity = rarity_score(trials, candidate, |config| config.kv_total_blocks);
    let prefix_cache_rarity = rarity_score(trials, candidate, |config| config.prefix_cache);
    (max_batch_rarity + kv_total_blocks_rarity + prefix_cache_rarity) / 3.0
}

#[cfg(test)]
pub(crate) fn rarity_score<T>(
    trials: &[AutotuneTrialRecord],
    candidate: &AutotuneCandidateConfig,
    accessor: impl Fn(&AutotuneCandidateConfig) -> T,
) -> f64
where
    T: PartialEq,
{
    1.0 / (1.0 + matching_trial_count(trials, candidate, accessor) as f64)
}

#[cfg(test)]
pub(crate) fn matching_trial_count<T>(
    trials: &[AutotuneTrialRecord],
    candidate: &AutotuneCandidateConfig,
    accessor: impl Fn(&AutotuneCandidateConfig) -> T,
) -> usize
where
    T: PartialEq,
{
    let expected = accessor(candidate);
    trials
        .iter()
        .filter(|trial| accessor(&trial.candidate) == expected)
        .count()
}

#[cfg(test)]
pub(crate) fn smoothed_probability<T>(
    value: T,
    option_count: usize,
    trials: &[&AutotuneTrialRecord],
    accessor: impl Fn(&AutotuneCandidateConfig) -> T,
) -> f64
where
    T: Copy + PartialEq,
{
    let matching = trials
        .iter()
        .filter(|trial| accessor(&trial.candidate) == value)
        .count() as f64;
    (matching + 1.0) / (trials.len() as f64 + option_count as f64)
}

#[cfg(test)]
pub(crate) fn load_autotune_warm_start_history(
    output_root: &Path,
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
) -> Result<AutotuneWarmStartHistory, CliError> {
    if !output_root.is_dir() {
        return Ok(AutotuneWarmStartHistory::default());
    }

    if let Some(history) =
        load_autotune_warm_start_history_from_index(output_root, manifest, search_space)?
    {
        return Ok(history);
    }

    let entries = collect_autotune_history_index_entries_from_output_root(output_root)?;
    Ok(autotune_warm_start_history_from_index_entries(
        &entries,
        manifest,
        search_space,
    ))
}

#[cfg(test)]
pub(crate) fn autotune_history_index_path(output_root: &Path) -> PathBuf {
    output_root.join("autotune-history-index.json")
}

#[cfg(test)]
pub(crate) fn load_autotune_warm_start_history_from_index(
    output_root: &Path,
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
) -> Result<Option<AutotuneWarmStartHistory>, CliError> {
    let Some(index_json) = load_autotune_history_index_json(output_root)? else {
        return Ok(None);
    };

    if let Some(summary) = load_matching_autotune_history_index_summary(&index_json, manifest)? {
        return Ok(Some(autotune_warm_start_history_from_index_summary(
            &summary,
            search_space,
        )));
    }

    let Some(entries) = load_autotune_history_index_entries_from_json(&index_json)? else {
        return Ok(None);
    };
    Ok(Some(autotune_warm_start_history_from_index_entries(
        &entries,
        manifest,
        search_space,
    )))
}

#[cfg(test)]
pub(crate) fn load_autotune_history_index_json(
    output_root: &Path,
) -> Result<Option<Value>, CliError> {
    let index_path = autotune_history_index_path(output_root);
    let Some(index_json) = load_optional_json_value(&index_path)? else {
        return Ok(None);
    };
    if index_json.get("schema_version").and_then(Value::as_str)
        != Some(AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION)
    {
        return Ok(None);
    }
    Ok(Some(index_json))
}

#[cfg(test)]
pub(crate) fn load_matching_autotune_history_index_summary(
    index_json: &Value,
    manifest: &BenchmarkManifest,
) -> Result<Option<AutotuneHistoryIndexSummary>, CliError> {
    let Some(summaries_json) = index_json.get("summaries").and_then(Value::as_array) else {
        return Ok(None);
    };
    let summaries = summaries_json
        .iter()
        .filter_map(autotune_history_index_summary_from_json)
        .collect::<Vec<_>>();
    if summaries.is_empty() && !summaries_json.is_empty() {
        return Ok(None);
    }
    Ok(summaries.into_iter().find(|summary| {
        summary.manifest_id == manifest.id
            && summary.selected_backend == selected_backend_label(manifest.runtime.selected_backend)
    }))
}

#[cfg(test)]
pub(crate) fn load_autotune_history_index_entries_from_json(
    index_json: &Value,
) -> Result<Option<Vec<AutotuneHistoryIndexEntry>>, CliError> {
    let Some(entries_json) = index_json.get("entries").and_then(Value::as_array) else {
        return Ok(None);
    };
    let entries = entries_json
        .iter()
        .filter_map(autotune_history_index_entry_from_json)
        .collect::<Vec<_>>();
    if entries.is_empty() && !entries_json.is_empty() {
        return Ok(None);
    }
    Ok(Some(entries))
}

#[cfg(test)]
pub(crate) fn collect_autotune_history_index_entries_from_output_root(
    output_root: &Path,
) -> Result<Vec<AutotuneHistoryIndexEntry>, CliError> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(output_root).map_err(|error| {
        CliError::Runtime(format!(
            "failed to read autotune output root {}: {error}",
            output_root.display()
        ))
    })? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let autotune_json_path = path.join("autotune.json");
        let history_json = match load_optional_json_value(&autotune_json_path) {
            Ok(Some(json)) => json,
            Ok(None) | Err(_) => continue,
        };
        let Some(entry) = autotune_history_index_entry_from_result_json(&path, &history_json)
        else {
            continue;
        };
        entries.push(entry);
    }
    entries.sort_by(|left, right| left.result_dir.cmp(&right.result_dir));
    entries.dedup_by(|left, right| left.result_dir == right.result_dir);
    Ok(entries)
}

#[cfg(test)]
pub(crate) fn autotune_history_index_entry_from_result_json(
    result_dir: &Path,
    history_json: &Value,
) -> Option<AutotuneHistoryIndexEntry> {
    if history_json.get("schema_version").and_then(Value::as_str) != Some(AUTOTUNE_SCHEMA_VERSION) {
        return None;
    }
    Some(AutotuneHistoryIndexEntry {
        result_dir: result_dir.to_path_buf(),
        manifest_id: history_json.get("manifest_id")?.as_str()?.to_string(),
        selected_backend: history_json.get("selected_backend")?.as_str()?.to_string(),
        trials: history_json.get("trials")?.as_array()?.clone(),
    })
}

#[cfg(test)]
pub(crate) fn autotune_history_index_entry_from_json(
    entry_json: &Value,
) -> Option<AutotuneHistoryIndexEntry> {
    Some(AutotuneHistoryIndexEntry {
        result_dir: PathBuf::from(entry_json.get("result_dir")?.as_str()?),
        manifest_id: entry_json.get("manifest_id")?.as_str()?.to_string(),
        selected_backend: entry_json.get("selected_backend")?.as_str()?.to_string(),
        trials: entry_json.get("trials")?.as_array()?.clone(),
    })
}

#[cfg(test)]
pub(crate) fn autotune_history_index_summary_from_json(
    summary_json: &Value,
) -> Option<AutotuneHistoryIndexSummary> {
    let source_dirs_json = summary_json.get("source_dirs")?.as_array()?;
    let best_trials_json = summary_json.get("best_trials")?.as_array()?;
    let source_dirs = source_dirs_json
        .iter()
        .filter_map(Value::as_str)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if source_dirs.is_empty() && !source_dirs_json.is_empty() {
        return None;
    }
    Some(AutotuneHistoryIndexSummary {
        manifest_id: summary_json.get("manifest_id")?.as_str()?.to_string(),
        selected_backend: summary_json.get("selected_backend")?.as_str()?.to_string(),
        source_dirs,
        best_trials: best_trials_json.clone(),
    })
}

#[cfg(test)]
pub(crate) fn autotune_warm_start_history_from_index_entries(
    entries: &[AutotuneHistoryIndexEntry],
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
) -> AutotuneWarmStartHistory {
    let mut source_dirs = Vec::new();
    let mut best_trials_by_candidate =
        BTreeMap::<AutotuneCandidateConfig, AutotuneTrialRecord>::new();
    for entry in entries {
        if entry.manifest_id != manifest.id {
            continue;
        }
        if entry.selected_backend != selected_backend_label(manifest.runtime.selected_backend) {
            continue;
        }
        let mut loaded_any = false;
        for (trial_index, trial_json) in entry.trials.iter().enumerate() {
            let Some(trial) =
                autotune_trial_record_from_history_json(trial_index, trial_json, search_space)
            else {
                continue;
            };
            loaded_any = true;
            match best_trials_by_candidate.get(&trial.candidate) {
                Some(existing) if existing.score >= trial.score => {}
                _ => {
                    best_trials_by_candidate.insert(trial.candidate.clone(), trial);
                }
            }
        }
        if loaded_any {
            source_dirs.push(entry.result_dir.clone());
        }
    }

    source_dirs.sort();
    source_dirs.dedup();
    let trials = best_trials_by_candidate.into_values().collect::<Vec<_>>();
    AutotuneWarmStartHistory {
        trials,
        source_dirs,
    }
}

#[cfg(test)]
pub(crate) fn autotune_warm_start_history_from_index_summary(
    summary: &AutotuneHistoryIndexSummary,
    search_space: &AutotuneSearchSpace,
) -> AutotuneWarmStartHistory {
    let trials = summary
        .best_trials
        .iter()
        .enumerate()
        .filter_map(|(trial_index, trial_json)| {
            autotune_trial_record_from_history_json(trial_index, trial_json, search_space)
        })
        .collect::<Vec<_>>();
    AutotuneWarmStartHistory {
        trials,
        source_dirs: summary.source_dirs.clone(),
    }
}

#[cfg(test)]
pub(crate) fn write_autotune_history_index_incremental(
    output_root: &Path,
    result_dir: &Path,
    result_json: &Value,
) -> Result<(), CliError> {
    let Some(current_entry) =
        autotune_history_index_entry_from_result_json(result_dir, result_json)
    else {
        return write_autotune_history_index(output_root);
    };
    let mut entries = match load_autotune_history_index_json(output_root)? {
        Some(index_json) => match load_autotune_history_index_entries_from_json(&index_json)? {
            Some(entries) => entries,
            None => collect_autotune_history_index_entries_from_output_root(output_root)?,
        },
        None => collect_autotune_history_index_entries_from_output_root(output_root)?,
    };
    upsert_autotune_history_index_entry(&mut entries, current_entry);
    write_autotune_history_index_entries(output_root, &entries)
}

#[cfg(test)]
pub(crate) fn write_autotune_history_index(output_root: &Path) -> Result<(), CliError> {
    let entries = collect_autotune_history_index_entries_from_output_root(output_root)?;
    write_autotune_history_index_entries(output_root, &entries)
}

#[cfg(test)]
pub(crate) fn write_autotune_history_index_entries(
    output_root: &Path,
    entries: &[AutotuneHistoryIndexEntry],
) -> Result<(), CliError> {
    let summaries = summarize_autotune_history_index_entries(entries);
    let index_json = json!({
        "schema_version": AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION,
        "entry_count": entries.len(),
        "summary_count": summaries.len(),
        "entries": entries.iter().map(|entry| json!({
            "result_dir": entry.result_dir.display().to_string(),
            "manifest_id": entry.manifest_id.clone(),
            "selected_backend": entry.selected_backend.clone(),
            "trials": entry.trials.clone(),
        })).collect::<Vec<_>>(),
        "summaries": summaries.iter().map(|summary| json!({
            "manifest_id": summary.manifest_id.clone(),
            "selected_backend": summary.selected_backend.clone(),
            "source_dirs": summary
                .source_dirs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>(),
            "best_trials": summary.best_trials.clone(),
        })).collect::<Vec<_>>(),
    });
    write_json_file(&autotune_history_index_path(output_root), &index_json)
}

#[cfg(test)]
pub(crate) fn upsert_autotune_history_index_entry(
    entries: &mut Vec<AutotuneHistoryIndexEntry>,
    entry: AutotuneHistoryIndexEntry,
) {
    if let Some(existing) = entries
        .iter_mut()
        .find(|existing| existing.result_dir == entry.result_dir)
    {
        *existing = entry;
    } else {
        entries.push(entry);
    }
    entries.sort_by(|left, right| left.result_dir.cmp(&right.result_dir));
    entries.dedup_by(|left, right| left.result_dir == right.result_dir);
}

#[cfg(test)]
pub(crate) fn summarize_autotune_history_index_entries(
    entries: &[AutotuneHistoryIndexEntry],
) -> Vec<AutotuneHistoryIndexSummary> {
    let mut grouped_best_trials = BTreeMap::<
        (String, String),
        (
            BTreeSet<PathBuf>,
            BTreeMap<AutotuneCandidateConfig, (f64, Value)>,
        ),
    >::new();
    for entry in entries {
        let grouped = grouped_best_trials
            .entry((entry.manifest_id.clone(), entry.selected_backend.clone()))
            .or_insert_with(|| (BTreeSet::new(), BTreeMap::new()));
        grouped.0.insert(entry.result_dir.clone());
        for trial_json in &entry.trials {
            let Some(candidate) = autotune_candidate_config_from_history_json(trial_json) else {
                continue;
            };
            let Some(score) = trial_json.get("score").and_then(Value::as_f64) else {
                continue;
            };
            match grouped.1.get(&candidate) {
                Some((existing_score, _)) if *existing_score >= score => {}
                _ => {
                    grouped.1.insert(candidate, (score, trial_json.clone()));
                }
            }
        }
    }

    grouped_best_trials
        .into_iter()
        .map(
            |((manifest_id, selected_backend), (source_dirs, best_trials))| {
                AutotuneHistoryIndexSummary {
                    manifest_id,
                    selected_backend,
                    source_dirs: source_dirs.into_iter().collect(),
                    best_trials: best_trials
                        .into_values()
                        .map(|(_, trial_json)| trial_json)
                        .collect(),
                }
            },
        )
        .collect()
}

#[cfg(test)]
pub(crate) fn autotune_trial_record_from_history_json(
    trial_index: usize,
    trial_json: &Value,
    search_space: &AutotuneSearchSpace,
) -> Option<AutotuneTrialRecord> {
    let candidate = autotune_candidate_config_from_history_json(trial_json)?;
    if !search_space
        .max_batch_token_options
        .contains(&candidate.max_batch_tokens)
        || !search_space
            .kv_total_block_options
            .contains(&candidate.kv_total_blocks)
        || !search_space
            .prefix_cache_options
            .contains(&candidate.prefix_cache)
    {
        return None;
    }

    Some(AutotuneTrialRecord {
        trial_index,
        candidate,
        selection: autotune_selection_from_history_json(trial_json.get("selection")),
        probe: None,
        score: trial_json.get("score")?.as_f64()?,
        status: trial_json
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("historical")
            .to_string(),
        result_dir: trial_json
            .get("result_dir")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        error: trial_json
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_string),
        metrics: autotune_metrics_from_history_json(trial_json.get("metrics")),
    })
}

#[cfg(test)]
pub(crate) fn autotune_candidate_config_from_history_json(
    trial_json: &Value,
) -> Option<AutotuneCandidateConfig> {
    let config = trial_json.get("config")?;
    let max_batch_tokens = config.get("max_batch_tokens")?.as_u64()? as u32;
    let kv_total_blocks = match config.get("kv_total_blocks") {
        Some(Value::Null) | None => None,
        Some(value) => Some(value.as_u64()? as u32),
    };
    let prefix_cache = config.get("prefix_cache")?.as_bool()?;
    Some(AutotuneCandidateConfig {
        max_batch_tokens,
        kv_total_blocks,
        prefix_cache,
    })
}

#[cfg(test)]
pub(crate) fn autotune_selection_from_history_json(
    selection_json: Option<&Value>,
) -> AutotuneSelectionDiagnostics {
    let Some(selection_json) = selection_json else {
        return AutotuneSelectionDiagnostics {
            strategy: "historical_import".to_string(),
            predicted_mean: None,
            uncertainty: None,
            acquisition: None,
            good_density: None,
            bad_density: None,
            density_ratio: None,
            novelty_bonus: None,
        };
    };
    AutotuneSelectionDiagnostics {
        strategy: selection_json
            .get("strategy")
            .and_then(Value::as_str)
            .unwrap_or("historical_import")
            .to_string(),
        predicted_mean: selection_json.get("predicted_mean").and_then(Value::as_f64),
        uncertainty: selection_json.get("uncertainty").and_then(Value::as_f64),
        acquisition: selection_json.get("acquisition").and_then(Value::as_f64),
        good_density: selection_json.get("good_density").and_then(Value::as_f64),
        bad_density: selection_json.get("bad_density").and_then(Value::as_f64),
        density_ratio: selection_json.get("density_ratio").and_then(Value::as_f64),
        novelty_bonus: selection_json.get("novelty_bonus").and_then(Value::as_f64),
    }
}

#[cfg(test)]
pub(crate) fn autotune_metrics_from_history_json(
    metrics_json: Option<&Value>,
) -> Option<AutotuneTrialMetrics> {
    let metrics_json = metrics_json?;
    Some(AutotuneTrialMetrics {
        ttft_ms: metrics_json.get("ttft_ms").and_then(Value::as_u64),
        prefill_tok_s: metrics_json.get("prefill_tok_s").and_then(Value::as_f64)?,
        decode_tok_s: metrics_json.get("decode_tok_s").and_then(Value::as_f64)?,
        prefix_hit_rate: metrics_json
            .get("prefix_hit_rate")
            .and_then(Value::as_f64)?,
        prefix_native_dispatch_count: metrics_json
            .get("prefix_native_dispatch_count")
            .and_then(Value::as_u64)? as u32,
        prefix_cpu_reference_dispatch_count: metrics_json
            .get("prefix_cpu_reference_dispatch_count")
            .and_then(Value::as_u64)? as u32,
        prefix_cpu_projection_row_count: metrics_json
            .get("prefix_cpu_projection_row_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        prefix_cpu_rms_norm_element_count: metrics_json
            .get("prefix_cpu_rms_norm_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        prefix_cpu_ffn_activation_element_count: metrics_json
            .get("prefix_cpu_ffn_activation_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        prefix_cpu_residual_add_element_count: metrics_json
            .get("prefix_cpu_residual_add_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        prefix_cpu_scale_element_count: metrics_json
            .get("prefix_cpu_scale_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_batched_group_fallback_count: metrics_json
            .get("direct_decode_batched_group_fallback_count")
            .and_then(Value::as_u64)? as u32,
        direct_decode_cpu_projection_row_count: metrics_json
            .get("direct_decode_cpu_projection_row_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_cpu_rms_norm_element_count: metrics_json
            .get("direct_decode_cpu_rms_norm_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_cpu_ffn_activation_element_count: metrics_json
            .get("direct_decode_cpu_ffn_activation_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_cpu_residual_add_element_count: metrics_json
            .get("direct_decode_cpu_residual_add_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_cpu_scale_element_count: metrics_json
            .get("direct_decode_cpu_scale_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        mlx_metal_hot_path_cpu_fallback_free: metrics_json
            .get("mlx_metal_hot_path_cpu_fallback_free")
            .and_then(Value::as_bool)?,
        real_model_forward: metrics_json
            .get("real_model_forward")
            .and_then(Value::as_bool)?,
        model_bound_ffn_decode: metrics_json
            .get("model_bound_ffn_decode")
            .and_then(Value::as_bool)?,
    })
}

#[cfg(test)]
pub(crate) fn autotune_trial_manifest(
    base_manifest: &BenchmarkManifest,
    trial_index: usize,
    candidate: &AutotuneCandidateConfig,
) -> BenchmarkManifest {
    let mut manifest = base_manifest.clone();
    manifest.id = format!("{}-autotune-{:03}", base_manifest.id, trial_index + 1);
    manifest.runtime.max_batch_tokens = candidate.max_batch_tokens;
    manifest.runtime.kv_total_blocks = candidate.kv_total_blocks;
    manifest.runtime.flags.prefix_cache = candidate.prefix_cache;
    manifest.notes = Some(format!(
        "autotune trial {} using {}",
        trial_index + 1,
        candidate.label()
    ));
    manifest
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn autotune_trial_metrics(execution: &RuntimeResult) -> AutotuneTrialMetrics {
    let route = &execution.observation.route_metadata;
    AutotuneTrialMetrics {
        ttft_ms: execution.observation.ttft_ms,
        prefill_tok_s: execution.observation.prefill_tok_s(),
        decode_tok_s: execution.observation.decode_tok_s(),
        prefix_hit_rate: execution.observation.prefix_hit_rate(),
        prefix_native_dispatch_count: route_prefix_native_dispatch_count(route),
        prefix_cpu_reference_dispatch_count: route_prefix_cpu_reference_dispatch_count(route),
        prefix_cpu_projection_row_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_projection_row_count",
        ),
        prefix_cpu_rms_norm_element_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_rms_norm_element_count",
        ),
        prefix_cpu_ffn_activation_element_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_ffn_activation_element_count",
        ),
        prefix_cpu_residual_add_element_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_residual_add_element_count",
        ),
        prefix_cpu_scale_element_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_scale_element_count",
        ),
        direct_decode_batched_group_fallback_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_group_fallback_count",
        ),
        direct_decode_cpu_projection_row_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_projection_row_count",
        ),
        direct_decode_cpu_rms_norm_element_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_rms_norm_element_count",
        ),
        direct_decode_cpu_ffn_activation_element_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_ffn_activation_element_count",
        ),
        direct_decode_cpu_residual_add_element_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_residual_add_element_count",
        ),
        direct_decode_cpu_scale_element_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_scale_element_count",
        ),
        mlx_metal_hot_path_cpu_fallback_free: route_prefix_cpu_reference_dispatch_count(route) == 0
            && route_decision_value(route, "metal_dispatch_prefix_cpu_projection_row_count") == 0
            && route_decision_value(route, "metal_dispatch_prefix_cpu_rms_norm_element_count") == 0
            && route_decision_value(
                route,
                "metal_dispatch_prefix_cpu_ffn_activation_element_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_prefix_cpu_residual_add_element_count",
            ) == 0
            && route_decision_value(route, "metal_dispatch_prefix_cpu_scale_element_count") == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_projection_row_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_rms_norm_element_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_ffn_activation_element_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_residual_add_element_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_scale_element_count",
            ) == 0,
        real_model_forward: route_decision_flag(route, "metal_dispatch_real_model_forward"),
        model_bound_ffn_decode: route_decision_flag(
            route,
            "metal_dispatch_direct_decode_model_bound_ffn",
        ),
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn autotune_score(execution: &RuntimeResult, metrics: &AutotuneTrialMetrics) -> f64 {
    let throughput_reward = metrics.decode_tok_s + (metrics.prefill_tok_s * 0.25);
    let prefix_reward = metrics.prefix_hit_rate * 2.0;
    let native_dispatch_reward = f64::from(metrics.prefix_native_dispatch_count) * 50.0;
    let ttft_penalty = metrics.ttft_ms.unwrap_or_default() as f64 * 0.25;
    let cpu_fallback_penalty = f64::from(metrics.prefix_cpu_reference_dispatch_count) * 1500.0
        + f64::from(metrics.direct_decode_batched_group_fallback_count) * 750.0
        + f64::from(metrics.prefix_cpu_projection_row_count) * 0.05
        + f64::from(metrics.prefix_cpu_rms_norm_element_count) * 0.001
        + f64::from(metrics.prefix_cpu_ffn_activation_element_count) * 0.001
        + f64::from(metrics.prefix_cpu_residual_add_element_count) * 0.001
        + f64::from(metrics.prefix_cpu_scale_element_count) * 0.001
        + f64::from(metrics.direct_decode_cpu_projection_row_count) * 0.05
        + f64::from(metrics.direct_decode_cpu_rms_norm_element_count) * 0.001
        + f64::from(metrics.direct_decode_cpu_ffn_activation_element_count) * 0.001
        + f64::from(metrics.direct_decode_cpu_residual_add_element_count) * 0.001
        + f64::from(metrics.direct_decode_cpu_scale_element_count) * 0.001;
    let readiness_reward = if metrics.mlx_metal_hot_path_cpu_fallback_free {
        500.0
    } else {
        0.0
    } + if metrics.real_model_forward {
        250.0
    } else {
        -250.0
    } + if metrics.model_bound_ffn_decode {
        100.0
    } else {
        -100.0
    };
    let gate_penalty = if execution.correctness.passed && execution.determinism.passed {
        0.0
    } else {
        100_000.0
    };

    throughput_reward + prefix_reward + native_dispatch_reward + readiness_reward
        - ttft_penalty
        - cpu_fallback_penalty
        - gate_penalty
}

#[cfg(test)]
pub(crate) fn best_autotune_trial<'a>(
    trials: impl IntoIterator<Item = &'a AutotuneTrialRecord>,
) -> Option<&'a AutotuneTrialRecord> {
    trials.into_iter().max_by(|left, right| {
        left.score
            .partial_cmp(&right.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn best_autotune_trial_with_history<'a>(
    trials: &'a [AutotuneTrialRecord],
    warm_start_history: &'a AutotuneWarmStartHistory,
) -> Option<&'a AutotuneTrialRecord> {
    best_autotune_trial(trials.iter().chain(warm_start_history.trials.iter()))
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn autotune_probe_baseline_json(baseline: &AutotuneProbeBaseline) -> Value {
    json!({
        "observed_trial_count": baseline.observed_trial_count,
        "incumbent_score": baseline.incumbent_score,
        "incumbent_decode_tok_s": baseline.incumbent_decode_tok_s,
        "score_floor": baseline.score_floor,
        "decode_tok_s_floor": baseline.decode_tok_s_floor,
        "fallback_free_decode_tok_s_floor": baseline.fallback_free_decode_tok_s_floor,
        "ttft_ceiling_ms": baseline.ttft_ceiling_ms,
    })
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn autotune_probe_json(probe: &AutotuneProbeObservation) -> Value {
    json!({
        "status": probe.status,
        "score": probe.score,
        "skipped_full_trial": probe.skipped_full_trial,
        "skip_reason": probe.skip_reason,
        "result_dir": probe.result_dir.as_ref().map(|path| path.display().to_string()),
        "baseline": probe.baseline.as_ref().map(autotune_probe_baseline_json),
        "shape": {
            "input_tokens_target": probe.shape.input_tokens_target,
            "output_tokens_target": probe.shape.output_tokens_target,
            "concurrency": probe.shape.concurrency,
        },
    })
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn autotune_candidate_config_json(candidate: &AutotuneCandidateConfig) -> Value {
    json!({
        "max_batch_tokens": candidate.max_batch_tokens,
        "kv_total_blocks": candidate.kv_total_blocks,
        "prefix_cache": candidate.prefix_cache,
    })
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn autotune_selection_json(selection: &AutotuneSelectionDiagnostics) -> Value {
    json!({
        "strategy": selection.strategy,
        "predicted_mean": selection.predicted_mean,
        "uncertainty": selection.uncertainty,
        "acquisition": selection.acquisition,
        "good_density": selection.good_density,
        "bad_density": selection.bad_density,
        "density_ratio": selection.density_ratio,
        "novelty_bonus": selection.novelty_bonus,
    })
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn autotune_trial_base_json(
    trial: &AutotuneTrialRecord,
) -> serde_json::Map<String, Value> {
    let mut object = serde_json::Map::new();
    object.insert("label".to_string(), json!(trial.label()));
    object.insert("score".to_string(), json!(trial.score));
    object.insert("status".to_string(), json!(trial.status));
    object.insert(
        "config".to_string(),
        autotune_candidate_config_json(&trial.candidate),
    );
    object.insert(
        "selection".to_string(),
        autotune_selection_json(&trial.selection),
    );
    object.insert(
        "result_dir".to_string(),
        json!(
            trial
                .result_dir
                .as_ref()
                .map(|path| path.display().to_string())
        ),
    );
    object.insert(
        "probe".to_string(),
        json!(trial.probe.as_ref().map(autotune_probe_json)),
    );
    object
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn autotune_probe_run_label(probe: Option<&AutotuneProbeObservation>) -> &'static str {
    match probe {
        Some(probe) if probe.skipped_full_trial => "early-stop",
        Some(_) => "full-run",
        None => "none",
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn format_optional_f64_3(value: Option<f64>, fallback: &str) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| fallback.to_string())
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn format_optional_to_string<T: ToString>(value: Option<T>, fallback: &str) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| fallback.to_string())
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn join_display(values: impl IntoIterator<Item = String>) -> String {
    values.into_iter().collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn build_autotune_result_json(
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    args: &AutotuneArgs,
    search_space: &AutotuneSearchSpace,
    warm_start_history: &AutotuneWarmStartHistory,
    trials: &[AutotuneTrialRecord],
) -> Value {
    let best_trial = best_autotune_trial_with_history(trials, warm_start_history);
    json!({
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "manifest_path": manifest_path.display().to_string(),
        "manifest_id": manifest.id,
        "selected_backend": manifest.runtime.selected_backend,
        "iterations_requested": args.iterations,
        "iterations_completed": trials.len(),
        "history_only": trials.is_empty() && !warm_start_history.trials.is_empty(),
        "exploration_weight": args.exploration_weight,
        "warm_start_history": {
            "enabled": !args.disable_history,
            "loaded_trial_count": warm_start_history.trials.len(),
            "loaded_result_count": warm_start_history.source_dirs.len(),
            "source_dirs": warm_start_history
                .source_dirs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>(),
        },
        "search_space": {
            "max_batch_tokens": search_space.max_batch_token_options,
            "kv_total_blocks": search_space.kv_total_block_options,
            "prefix_cache": search_space.prefix_cache_options,
            "candidate_count": search_space.max_batch_token_options.len()
                * search_space.kv_total_block_options.len()
                * search_space.prefix_cache_options.len(),
        },
        "best_trial": best_trial.map(|trial| Value::Object(autotune_trial_base_json(trial))),
        "trials": trials.iter().map(|trial| {
            let mut object = autotune_trial_base_json(trial);
            object.insert("error".to_string(), json!(trial.error));
            object.insert("metrics".to_string(), json!(trial.metrics.as_ref().map(|metrics| json!({
                "ttft_ms": metrics.ttft_ms,
                "prefill_tok_s": metrics.prefill_tok_s,
                "decode_tok_s": metrics.decode_tok_s,
                "prefix_hit_rate": metrics.prefix_hit_rate,
                "prefix_native_dispatch_count": metrics.prefix_native_dispatch_count,
                "prefix_cpu_reference_dispatch_count": metrics.prefix_cpu_reference_dispatch_count,
                "prefix_cpu_projection_row_count": metrics.prefix_cpu_projection_row_count,
                "prefix_cpu_rms_norm_element_count": metrics.prefix_cpu_rms_norm_element_count,
                "prefix_cpu_ffn_activation_element_count": metrics.prefix_cpu_ffn_activation_element_count,
                "prefix_cpu_residual_add_element_count": metrics.prefix_cpu_residual_add_element_count,
                "prefix_cpu_scale_element_count": metrics.prefix_cpu_scale_element_count,
                "direct_decode_batched_group_fallback_count": metrics.direct_decode_batched_group_fallback_count,
                "direct_decode_cpu_projection_row_count": metrics.direct_decode_cpu_projection_row_count,
                "direct_decode_cpu_rms_norm_element_count": metrics.direct_decode_cpu_rms_norm_element_count,
                "direct_decode_cpu_ffn_activation_element_count": metrics.direct_decode_cpu_ffn_activation_element_count,
                "direct_decode_cpu_residual_add_element_count": metrics.direct_decode_cpu_residual_add_element_count,
                "direct_decode_cpu_scale_element_count": metrics.direct_decode_cpu_scale_element_count,
                "mlx_metal_hot_path_cpu_fallback_free": metrics.mlx_metal_hot_path_cpu_fallback_free,
                "real_model_forward": metrics.real_model_forward,
                "model_bound_ffn_decode": metrics.model_bound_ffn_decode,
            }))));
            Value::Object(object)
        }).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn build_autotune_summary_markdown(
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
    warm_start_history: &AutotuneWarmStartHistory,
    trials: &[AutotuneTrialRecord],
) -> String {
    let mut lines = vec![
        "# Benchmark Autotune".to_string(),
        String::new(),
        format!("- manifest: `{}`", manifest_path.display()),
        format!("- manifest_id: `{}`", manifest.id),
        format!(
            "- selected_backend: `{}`",
            selected_backend_label(manifest.runtime.selected_backend)
        ),
        format!("- trials_completed: `{}`", trials.len()),
        format!(
            "- warm_start_history.loaded_trials: `{}`",
            warm_start_history.trials.len()
        ),
        format!(
            "- warm_start_history.loaded_results: `{}`",
            warm_start_history.source_dirs.len()
        ),
        format!(
            "- history_only: `{}`",
            trials.is_empty() && !warm_start_history.trials.is_empty()
        ),
        format!(
            "- search_space.max_batch_tokens: `{}`",
            join_display(
                search_space
                    .max_batch_token_options
                    .iter()
                    .map(|value| value.to_string()),
            )
        ),
        format!(
            "- search_space.kv_total_blocks: `{}`",
            join_display(
                search_space
                    .kv_total_block_options
                    .iter()
                    .map(|value| optional_u32_label(*value)),
            )
        ),
        format!(
            "- search_space.prefix_cache: `{}`",
            join_display(
                search_space
                    .prefix_cache_options
                    .iter()
                    .map(|value| value.to_string()),
            )
        ),
    ];
    if let Some(best_trial) = best_autotune_trial_with_history(trials, warm_start_history) {
        lines.push(format!("- best_trial: `{}`", best_trial.label()));
        lines.push(format!("- best_score: `{:.3}`", best_trial.score));
        lines.push(format!("- best_config: `{}`", best_trial.candidate.label()));
    }
    lines.push(String::new());
    lines.push("| Trial | Score | Status | Probe | Strategy | Ratio | Acquisition | Max Batch | KV Blocks | Prefix Cache | Decode tok/s | TTFT ms | Prefix CPU Fallbacks |".to_string());
    lines.push(
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"
            .to_string(),
    );
    for trial in trials {
        let metrics = trial.metrics.as_ref();
        lines.push(format!(
            "| {} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {} | {} |",
            trial.label(),
            trial.score,
            trial.status,
            autotune_probe_run_label(trial.probe.as_ref()),
            trial.selection.strategy,
            format_optional_f64_3(trial.selection.density_ratio, "n/a"),
            format_optional_f64_3(trial.selection.acquisition, "seed"),
            trial.candidate.max_batch_tokens,
            optional_u32_label(trial.candidate.kv_total_blocks),
            trial.candidate.prefix_cache,
            metrics
                .map(|metrics| metrics.decode_tok_s)
                .unwrap_or_default(),
            format_optional_to_string(metrics.and_then(|metrics| metrics.ttft_ms), "n/a"),
            format_optional_to_string(
                metrics.map(|metrics| metrics.prefix_cpu_reference_dispatch_count),
                "n/a",
            ),
        ));
    }
    lines.join("\n")
}
