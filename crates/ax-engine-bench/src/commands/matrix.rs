use super::*;

#[derive(Clone, Debug)]
pub(crate) struct MatrixMemberResult {
    pub(crate) label: String,
    pub(crate) manifest_id: String,
    pub(crate) manifest_path: PathBuf,
    pub(crate) scenario: String,
    pub(crate) model_family: String,
    pub(crate) status: String,
    pub(crate) tool_mode: String,
    pub(crate) selected_backend: String,
    pub(crate) support_tier: String,
    pub(crate) resolution_policy: String,
    pub(crate) result_dir: PathBuf,
    pub(crate) correctness_passed: Option<bool>,
    pub(crate) determinism_passed: Option<bool>,
    pub(crate) ttft_ms: Option<f64>,
    pub(crate) decode_tok_s: Option<f64>,
    pub(crate) prefix_hit_rate: Option<f64>,
    pub(crate) failure_code: Option<String>,
    pub(crate) failure_message: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct MatrixExecutionResult {
    pub(crate) result_dir: PathBuf,
    pub(crate) overall_status: String,
    pub(crate) members: Vec<MatrixMemberResult>,
}

#[derive(Clone, Debug)]
pub(crate) struct MatrixCompareMemberResult {
    pub(crate) label: String,
    pub(crate) manifest_id: String,
    pub(crate) compare_result_dir: PathBuf,
    pub(crate) compare_mode: String,
    pub(crate) tool_mode: String,
    pub(crate) execution_semantics: String,
    pub(crate) selected_backend: String,
    pub(crate) support_tier: String,
    pub(crate) resolution_policy: String,
    pub(crate) ttft_ms_pct: f64,
    pub(crate) decode_tok_s_pct: f64,
    pub(crate) memory_peak_mb_pct: f64,
    pub(crate) prefix_hit_rate_pct: f64,
}

pub(crate) fn execute_matrix_manifest(
    matrix_manifest_path: &Path,
    matrix_manifest: &BenchmarkMatrixManifest,
    output_root: &Path,
    write_trace: bool,
) -> Result<MatrixExecutionResult, CliError> {
    let (run_id, result_dir) =
        create_unique_result_dir(output_root, Some("matrix"), &matrix_manifest.id)?;
    let cases_dir = result_dir.join("cases");
    fs::create_dir_all(&cases_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create matrix result directory {}: {error}",
            result_dir.display()
        ))
    })?;

    let mut members = Vec::with_capacity(matrix_manifest.members.len());
    for member in &matrix_manifest.members {
        let manifest_path = resolve_matrix_member_manifest_path(matrix_manifest_path, member);
        require_existing_file(&manifest_path)?;
        let manifest = load_manifest(&manifest_path)?;
        validate_manifest(&manifest, ManifestClass::Scenario)?;
        let started_at_unix_s = unix_timestamp_secs()?;
        let member_output_root = cases_dir.join(sanitize_component(&manifest.id));
        fs::create_dir_all(&member_output_root).map_err(|error| {
            CliError::Runtime(format!(
                "failed to create matrix member output root {}: {error}",
                member_output_root.display()
            ))
        })?;

        let member_result = match execute_manifest_runtime(&manifest) {
            Ok(execution) => {
                let artifact_dir = write_execution_artifacts(
                    "scenario",
                    &manifest_path,
                    &manifest,
                    &member_output_root,
                    started_at_unix_s,
                    &execution,
                    write_trace,
                )?;
                MatrixMemberResult {
                    label: member.label.clone().unwrap_or_else(|| manifest.id.clone()),
                    manifest_id: manifest.id.clone(),
                    manifest_path: manifest_path.clone(),
                    scenario: manifest.scenario.clone(),
                    model_family: manifest.model.family.clone(),
                    status: execution.status_label().to_string(),
                    tool_mode: execution.tool_mode.to_string(),
                    selected_backend: json_string_label(
                        execution.runtime.resolved_backend.selected_backend,
                    ),
                    support_tier: json_string_label(
                        execution.runtime.resolved_backend.support_tier,
                    ),
                    resolution_policy: json_string_label(
                        execution.runtime.backend_policy.resolution_policy,
                    ),
                    result_dir: artifact_dir,
                    correctness_passed: Some(execution.correctness.passed),
                    determinism_passed: Some(execution.determinism.passed),
                    ttft_ms: Some(execution.observation.ttft_ms.unwrap_or_default() as f64),
                    decode_tok_s: Some(execution.observation.decode_tok_s()),
                    prefix_hit_rate: Some(execution.observation.prefix_hit_rate()),
                    failure_code: None,
                    failure_message: execution
                        .correctness
                        .reason
                        .clone()
                        .or(execution.determinism.reason.clone()),
                }
            }
            Err(CliError::Contract(message)) => {
                let artifact_dir = write_contract_failure_artifacts(
                    "scenario",
                    &manifest_path,
                    &manifest,
                    &member_output_root,
                    started_at_unix_s,
                    &message,
                )?;
                let failure = classify_contract_failure(&manifest, &message);
                MatrixMemberResult {
                    label: member.label.clone().unwrap_or_else(|| manifest.id.clone()),
                    manifest_id: manifest.id.clone(),
                    manifest_path: manifest_path.clone(),
                    scenario: manifest.scenario.clone(),
                    model_family: manifest.model.family.clone(),
                    status: "contract_failure".to_string(),
                    tool_mode: contract_failure_tool_mode(&manifest).to_string(),
                    selected_backend: json_string_label(manifest.runtime.selected_backend),
                    support_tier: json_string_label(manifest.runtime.support_tier),
                    resolution_policy: json_string_label(manifest.runtime.resolution_policy),
                    result_dir: artifact_dir,
                    correctness_passed: None,
                    determinism_passed: None,
                    ttft_ms: None,
                    decode_tok_s: None,
                    prefix_hit_rate: None,
                    failure_code: Some(failure.code.to_string()),
                    failure_message: Some(message),
                }
            }
            Err(error) => return Err(error),
        };
        members.push(member_result);
    }

    let overall_status = matrix_overall_status(&members).to_string();
    write_json_file(&result_dir.join("matrix_manifest.json"), matrix_manifest)?;
    write_json_file(
        &result_dir.join("matrix.json"),
        &build_matrix_json(
            &run_id,
            matrix_manifest_path,
            matrix_manifest,
            &result_dir,
            &members,
        )?,
    )?;
    fs::write(
        result_dir.join("summary.md"),
        build_matrix_summary_markdown(matrix_manifest, &members, &overall_status),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write matrix summary.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(MatrixExecutionResult {
        result_dir,
        overall_status,
        members,
    })
}

pub(crate) fn resolve_matrix_member_manifest_path(
    matrix_manifest_path: &Path,
    member: &BenchmarkMatrixMember,
) -> PathBuf {
    let manifest_path = PathBuf::from(&member.manifest);
    if manifest_path.is_absolute() {
        manifest_path
    } else {
        matrix_manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(manifest_path)
    }
}

pub(crate) fn matrix_overall_status(members: &[MatrixMemberResult]) -> &'static str {
    if members
        .iter()
        .any(|member| member.status == "contract_failure")
    {
        "contract_failure"
    } else if members.iter().any(|member| member.status != "ok") {
        "completed_with_failures"
    } else {
        "ok"
    }
}

pub(crate) fn build_matrix_json(
    run_id: &str,
    matrix_manifest_path: &Path,
    matrix_manifest: &BenchmarkMatrixManifest,
    result_dir: &Path,
    members: &[MatrixMemberResult],
) -> Result<Value, CliError> {
    let contract_failure_count = members
        .iter()
        .filter(|member| member.status == "contract_failure")
        .count();
    let completed_with_failures_count = members
        .iter()
        .filter(|member| member.status == "completed_with_failures")
        .count();
    let ok_count = members
        .iter()
        .filter(|member| member.status == "ok")
        .count();
    Ok(json!({
        "schema_version": "ax.engine_bench.matrix_result.v1",
        "run_id": run_id,
        "id": matrix_manifest.id,
        "class": matrix_manifest.class,
        "status": matrix_overall_status(members),
        "matrix_manifest_path": matrix_manifest_path.display().to_string(),
        "matrix_manifest_fingerprint_fnv1a64": file_fingerprint_fnv1a64(matrix_manifest_path)?,
        "result_dir": result_dir.display().to_string(),
        "summary": {
            "member_count": members.len(),
            "ok_count": ok_count,
            "completed_with_failures_count": completed_with_failures_count,
            "contract_failure_count": contract_failure_count
        },
        "members": members.iter().map(|member| {
            let mut value = json!({
                "label": member.label,
                "manifest_id": member.manifest_id,
                "manifest_path": member.manifest_path.display().to_string(),
                "scenario": member.scenario,
                "model_family": member.model_family,
                "status": member.status,
                "tool_mode": member.tool_mode,
                "selected_backend": member.selected_backend,
                "support_tier": member.support_tier,
                "resolution_policy": member.resolution_policy,
                "result_dir": member.result_dir.display().to_string()
            });
            if let Some(value_ttft_ms) = member.ttft_ms {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("ttft_ms".to_string(), json!(value_ttft_ms));
            }
            if let Some(value_decode_tok_s) = member.decode_tok_s {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("decode_tok_s".to_string(), json!(value_decode_tok_s));
            }
            if let Some(value_prefix_hit_rate) = member.prefix_hit_rate {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("prefix_hit_rate".to_string(), json!(value_prefix_hit_rate));
            }
            if let Some(correctness_passed) = member.correctness_passed {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("correctness_passed".to_string(), json!(correctness_passed));
            }
            if let Some(determinism_passed) = member.determinism_passed {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("determinism_passed".to_string(), json!(determinism_passed));
            }
            if let Some(failure_code) = &member.failure_code {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("failure_code".to_string(), json!(failure_code));
            }
            if let Some(failure_message) = &member.failure_message {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("failure_message".to_string(), json!(failure_message));
            }
            value
        }).collect::<Vec<_>>()
    }))
}

pub(crate) fn build_matrix_summary_markdown(
    matrix_manifest: &BenchmarkMatrixManifest,
    members: &[MatrixMemberResult],
    overall_status: &str,
) -> String {
    let mut lines = vec![
        "# Benchmark Matrix".to_string(),
        String::new(),
        format!("- id: `{}`", matrix_manifest.id),
        format!("- status: `{overall_status}`"),
        format!("- member_count: `{}`", members.len()),
        String::new(),
        "| Label | Scenario | Model | Status | TTFT ms | Decode tok/s | Prefix hit rate |"
            .to_string(),
        "| --- | --- | --- | --- | ---: | ---: | ---: |".to_string(),
    ];

    for member in members {
        let ttft_ms = member
            .ttft_ms
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let decode_tok_s = member
            .decode_tok_s
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let prefix_hit_rate = member
            .prefix_hit_rate
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            member.label,
            member.scenario,
            member.model_family,
            member.status,
            ttft_ms,
            decode_tok_s,
            prefix_hit_rate
        ));
        lines.push(format!("result_dir: `{}`", member.result_dir.display()));
        if let Some(failure_code) = &member.failure_code {
            lines.push(format!("failure_code: `{failure_code}`"));
        }
    }

    lines.push(String::new());
    lines.push(
        "This summary is the frozen scenario matrix roll-up for Tier 2 dense-path benchmarking."
            .to_string(),
    );
    lines.join("\n")
}

pub(crate) fn validate_comparable_matrix_results(
    baseline: &Value,
    candidate: &Value,
) -> Result<(), CliError> {
    validate_matching_json_field(baseline, candidate, &["schema_version"])?;
    validate_matching_json_field(baseline, candidate, &["id"])?;
    validate_matching_json_field(baseline, candidate, &["class"])?;
    validate_matching_json_field(
        baseline,
        candidate,
        &["matrix_manifest_fingerprint_fnv1a64"],
    )?;

    let baseline_status = baseline
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let candidate_status = candidate
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if baseline_status != "ok" {
        return Err(CliError::Contract(format!(
            "matrix compare requires successful matrix execution artifacts; baseline status is {baseline_status}"
        )));
    }
    if candidate_status != "ok" {
        return Err(CliError::Contract(format!(
            "matrix compare requires successful matrix execution artifacts; candidate status is {candidate_status}"
        )));
    }

    let baseline_members = matrix_members_by_manifest_id(baseline)?;
    let candidate_members = matrix_members_by_manifest_id(candidate)?;
    if baseline_members.len() != candidate_members.len() {
        return Err(CliError::Contract(format!(
            "matrix compare member-count mismatch: baseline={}, candidate={}",
            baseline_members.len(),
            candidate_members.len()
        )));
    }
    for manifest_id in baseline_members.keys() {
        if !candidate_members.contains_key(manifest_id) {
            return Err(CliError::Contract(format!(
                "candidate matrix missing member manifest_id={manifest_id}"
            )));
        }
    }

    Ok(())
}

pub(crate) fn matrix_members_by_manifest_id(
    matrix: &Value,
) -> Result<BTreeMap<String, &Value>, CliError> {
    let members = matrix
        .get("members")
        .and_then(Value::as_array)
        .ok_or_else(|| CliError::Contract("matrix result missing members array".to_string()))?;
    let mut by_manifest_id = BTreeMap::new();
    for member in members {
        let manifest_id = member
            .get("manifest_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                CliError::Contract("matrix result member missing manifest_id".to_string())
            })?
            .to_string();
        if by_manifest_id.insert(manifest_id.clone(), member).is_some() {
            return Err(CliError::Contract(format!(
                "matrix result contains duplicate manifest_id={manifest_id}"
            )));
        }
    }
    Ok(by_manifest_id)
}

pub(crate) fn build_matrix_compare_member_result(
    baseline_member: &Value,
    manifest_id: &str,
    compare_result_dir: &Path,
    regression: &Value,
) -> Result<MatrixCompareMemberResult, CliError> {
    Ok(MatrixCompareMemberResult {
        label: baseline_member
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or(manifest_id)
            .to_string(),
        manifest_id: manifest_id.to_string(),
        compare_result_dir: compare_result_dir.to_path_buf(),
        compare_mode: explicit_or_inferred_compare_mode(regression),
        tool_mode: explicit_or_inferred_tool_mode(
            regression,
            &["summary", "tool_mode"],
            "engine_bringup_runtime",
        ),
        execution_semantics: nested_value(regression, &["summary", "execution_semantics"])
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        selected_backend: nested_value(regression, &["summary", "selected_backend"])
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        support_tier: nested_value(regression, &["summary", "support_tier"])
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        resolution_policy: nested_value(regression, &["summary", "resolution_policy"])
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        ttft_ms_pct: nested_value(regression, &["comparison", "ttft_ms_pct"])
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                CliError::Contract(format!(
                    "matrix compare member {manifest_id} missing ttft_ms_pct"
                ))
            })?,
        decode_tok_s_pct: nested_value(regression, &["comparison", "decode_tok_s_pct"])
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                CliError::Contract(format!(
                    "matrix compare member {manifest_id} missing decode_tok_s_pct"
                ))
            })?,
        memory_peak_mb_pct: nested_value(regression, &["comparison", "memory_peak_mb_pct"])
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                CliError::Contract(format!(
                    "matrix compare member {manifest_id} missing memory_peak_mb_pct"
                ))
            })?,
        prefix_hit_rate_pct: nested_value(regression, &["comparison", "prefix_hit_rate_pct"])
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                CliError::Contract(format!(
                    "matrix compare member {manifest_id} missing prefix_hit_rate_pct"
                ))
            })?,
    })
}

pub(crate) fn build_matrix_regression_json(
    matrix_id: &str,
    baseline_dir: &Path,
    candidate_dir: &Path,
    baseline_matrix: &Value,
    candidate_matrix: &Value,
    members: &[MatrixCompareMemberResult],
) -> Value {
    json!({
        "schema_version": "ax.engine_bench.matrix_regression.v1",
        "id": matrix_id,
        "baseline_matrix_run_id": baseline_matrix.get("run_id").cloned().unwrap_or(Value::String("baseline".to_string())),
        "candidate_matrix_run_id": candidate_matrix.get("run_id").cloned().unwrap_or(Value::String("candidate".to_string())),
        "baseline_dir": baseline_dir.display().to_string(),
        "candidate_dir": candidate_dir.display().to_string(),
        "summary": {
            "member_count": members.len(),
            "requires_human_review": true
        },
        "members": members.iter().map(|member| json!({
            "label": member.label,
            "manifest_id": member.manifest_id,
            "compare_result_dir": member.compare_result_dir.display().to_string(),
            "compare_mode": member.compare_mode,
            "tool_mode": member.tool_mode,
            "execution_semantics": member.execution_semantics,
            "selected_backend": member.selected_backend,
            "support_tier": member.support_tier,
            "resolution_policy": member.resolution_policy,
            "comparison": {
                "ttft_ms_pct": member.ttft_ms_pct,
                "decode_tok_s_pct": member.decode_tok_s_pct,
                "memory_peak_mb_pct": member.memory_peak_mb_pct,
                "prefix_hit_rate_pct": member.prefix_hit_rate_pct
            }
        })).collect::<Vec<_>>()
    })
}

pub(crate) fn build_matrix_compare_summary_markdown(
    matrix_id: &str,
    baseline_matrix: &Value,
    candidate_matrix: &Value,
    members: &[MatrixCompareMemberResult],
) -> String {
    let baseline_run_id = baseline_matrix
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("baseline");
    let candidate_run_id = candidate_matrix
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("candidate");
    let mut lines = vec![
        "# Benchmark Matrix Compare".to_string(),
        String::new(),
        format!("- id: `{matrix_id}`"),
        format!("- baseline_matrix_run_id: `{baseline_run_id}`"),
        format!("- candidate_matrix_run_id: `{candidate_run_id}`"),
        format!("- member_count: `{}`", members.len()),
        String::new(),
        "| Label | Mode | TTFT % | Decode tok/s % | Memory peak MB % | Prefix hit rate % |"
            .to_string(),
        "| --- | --- | ---: | ---: | ---: | ---: |".to_string(),
    ];

    for member in members {
        lines.push(format!(
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} |",
            member.label,
            member.compare_mode,
            member.ttft_ms_pct,
            member.decode_tok_s_pct,
            member.memory_peak_mb_pct,
            member.prefix_hit_rate_pct
        ));
        lines.push(format!(
            "compare_result_dir: `{}`",
            member.compare_result_dir.display()
        ));
    }

    lines.push(String::new());
    lines.push(
        "This summary is the frozen matrix regression roll-up. Read per-member compare artifacts before drawing performance conclusions."
            .to_string(),
    );
    lines.join("\n")
}

pub(crate) fn enforce_matrix_gates(execution: &MatrixExecutionResult) -> Result<(), CliError> {
    if execution.overall_status == "contract_failure" {
        let failed_members = execution
            .members
            .iter()
            .filter(|member| member.status == "contract_failure")
            .map(|member| member.manifest_id.clone())
            .collect::<Vec<_>>();
        return Err(CliError::Contract(format!(
            "matrix completed with contract failures in {} member(s): {}",
            failed_members.len(),
            failed_members.join(", ")
        )));
    }

    if execution.overall_status == "completed_with_failures" {
        let failed_members = execution
            .members
            .iter()
            .filter(|member| member.status != "ok")
            .map(|member| member.manifest_id.clone())
            .collect::<Vec<_>>();
        return Err(CliError::Correctness(format!(
            "matrix completed with failed correctness or determinism gates in {} member(s): {}",
            failed_members.len(),
            failed_members.join(", ")
        )));
    }

    Ok(())
}
