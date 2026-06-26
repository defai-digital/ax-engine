use super::*;

pub(crate) fn write_execution_artifacts(
    command: &str,
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    output_root: &Path,
    started_at_unix_s: u64,
    execution: &RuntimeResult,
    write_trace: bool,
) -> Result<PathBuf, CliError> {
    let (run_id, result_dir) = create_unique_result_dir(output_root, None, &manifest.id)?;

    write_json_file(&result_dir.join("manifest.json"), manifest)?;
    write_json_file(
        &result_dir.join("environment.json"),
        &build_environment_json(
            &run_id,
            command,
            manifest_path,
            output_root,
            started_at_unix_s,
            execution,
        )?,
    )?;
    write_json_file(
        &result_dir.join("metrics.json"),
        &build_metrics_json(&run_id, execution),
    )?;
    write_json_file(
        &result_dir.join("routes.json"),
        &build_routes_json(&run_id, execution),
    )?;
    if write_trace {
        write_json_file(
            &result_dir.join("trace.json"),
            &build_trace_json(&run_id, execution),
        )?;
    }
    fs::write(
        result_dir.join("summary.md"),
        build_execution_summary_markdown(&run_id, command, manifest_path, execution),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write summary.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(result_dir)
}

pub(crate) fn write_contract_failure_artifacts(
    command: &str,
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    output_root: &Path,
    started_at_unix_s: u64,
    message: &str,
) -> Result<PathBuf, CliError> {
    let (run_id, result_dir) =
        create_unique_result_dir(output_root, Some("contract-failure"), &manifest.id)?;

    write_json_file(&result_dir.join("manifest.json"), manifest)?;
    write_json_file(
        &result_dir.join("contract_failure.json"),
        &build_contract_failure_json(
            &run_id,
            command,
            manifest_path,
            manifest,
            output_root,
            started_at_unix_s,
            message,
        )?,
    )?;
    fs::write(
        result_dir.join("summary.md"),
        build_contract_failure_summary_markdown(&run_id, command, manifest_path, manifest, message),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write contract-failure summary.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(result_dir)
}

pub(crate) fn write_compare_artifacts(
    baseline_dir: &Path,
    candidate_dir: &Path,
    output_root: &Path,
) -> Result<PathBuf, CliError> {
    let baseline_manifest = load_json_value(&baseline_dir.join("manifest.json"))?;
    let candidate_manifest = load_json_value(&candidate_dir.join("manifest.json"))?;
    validate_comparable_manifests(&baseline_manifest, &candidate_manifest)?;
    let baseline_environment = load_json_value(&baseline_dir.join("environment.json"))?;
    let candidate_environment = load_json_value(&candidate_dir.join("environment.json"))?;
    validate_comparable_environments(&baseline_environment, &candidate_environment)?;
    let trusted_baseline = load_optional_json_value(&baseline_dir.join("trusted_baseline.json"))?;

    let baseline_metrics = load_json_value(&baseline_dir.join("metrics.json"))?;
    let candidate_metrics = load_json_value(&candidate_dir.join("metrics.json"))?;

    let manifest_id = baseline_manifest
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("compare");
    let (_compare_id, result_dir) =
        create_unique_result_dir(output_root, Some("compare"), manifest_id)?;

    let regression_json = build_regression_json(
        &baseline_metrics,
        &candidate_metrics,
        &baseline_environment,
        &candidate_environment,
        trusted_baseline.as_ref(),
    )?;
    write_json_file(&result_dir.join("baseline.json"), &baseline_metrics)?;
    write_json_file(&result_dir.join("candidate.json"), &candidate_metrics)?;
    write_json_file(&result_dir.join("regression.json"), &regression_json)?;
    fs::write(
        result_dir.join("comparison.md"),
        build_compare_summary_markdown(
            &baseline_metrics,
            &candidate_metrics,
            &regression_json,
            trusted_baseline.as_ref(),
        ),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write comparison.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(result_dir)
}

pub(crate) fn write_trusted_baseline_artifacts(
    source_dir: &Path,
    name: &str,
    output_root: &Path,
) -> Result<PathBuf, CliError> {
    let manifest = load_json_value(&source_dir.join("manifest.json"))?;
    let environment = load_json_value(&source_dir.join("environment.json"))?;
    let metrics = load_json_value(&source_dir.join("metrics.json"))?;
    let summary_path = source_dir.join("summary.md");
    if !summary_path.is_file() {
        return Err(CliError::Contract(format!(
            "benchmark artifact missing summary.md: {}",
            source_dir.display()
        )));
    }

    let slug = sanitize_component(name.trim())
        .trim_matches('-')
        .to_string();
    if slug.is_empty() {
        return Err(CliError::Usage(
            "baseline name must contain at least one alphanumeric character".to_string(),
        ));
    }

    let baseline_dir = output_root.join(&slug);
    if baseline_dir.exists() {
        return Err(CliError::Contract(format!(
            "trusted baseline already exists and will not be overwritten: {}",
            baseline_dir.display()
        )));
    }
    fs::create_dir_all(&baseline_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create trusted baseline directory {}: {error}",
            baseline_dir.display()
        ))
    })?;

    let trusted_baseline =
        build_trusted_baseline_json(name, &slug, source_dir, &manifest, &environment, &metrics)?;
    write_json_file(
        &baseline_dir.join("trusted_baseline.json"),
        &trusted_baseline,
    )?;
    fs::write(
        baseline_dir.join("trusted_baseline.md"),
        build_trusted_baseline_summary_markdown(&trusted_baseline),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write trusted_baseline.md in {}: {error}",
            baseline_dir.display()
        ))
    })?;

    copy_required_artifact_file(source_dir, &baseline_dir, "manifest.json")?;
    copy_required_artifact_file(source_dir, &baseline_dir, "environment.json")?;
    copy_required_artifact_file(source_dir, &baseline_dir, "metrics.json")?;
    copy_required_artifact_file(source_dir, &baseline_dir, "summary.md")?;
    copy_optional_artifact_file(source_dir, &baseline_dir, "routes.json")?;
    copy_optional_artifact_file(source_dir, &baseline_dir, "trace.json")?;

    Ok(baseline_dir)
}

pub(crate) fn write_matrix_compare_artifacts(
    baseline_dir: &Path,
    candidate_dir: &Path,
    output_root: &Path,
) -> Result<PathBuf, CliError> {
    let baseline_matrix = load_json_value(&baseline_dir.join("matrix.json"))?;
    let candidate_matrix = load_json_value(&candidate_dir.join("matrix.json"))?;
    validate_comparable_matrix_results(&baseline_matrix, &candidate_matrix)?;

    let matrix_id = baseline_matrix
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("matrix-compare");
    let result_dir = output_root.join(format!(
        "{}-matrix-compare-{}",
        unix_timestamp_secs()?,
        sanitize_component(matrix_id)
    ));
    let cases_dir = result_dir.join("cases");
    fs::create_dir_all(&cases_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create matrix compare directory {}: {error}",
            result_dir.display()
        ))
    })?;

    let baseline_members = matrix_members_by_manifest_id(&baseline_matrix)?;
    let candidate_members = matrix_members_by_manifest_id(&candidate_matrix)?;
    let mut member_results = Vec::with_capacity(baseline_members.len());

    for (manifest_id, baseline_member) in baseline_members {
        let candidate_member = candidate_members.get(&manifest_id).ok_or_else(|| {
            CliError::Contract(format!(
                "candidate matrix missing member manifest_id={manifest_id}"
            ))
        })?;
        let baseline_result_dir = PathBuf::from(
            baseline_member
                .get("result_dir")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    CliError::Contract(format!(
                        "baseline matrix member {manifest_id} missing result_dir"
                    ))
                })?,
        );
        let candidate_result_dir = PathBuf::from(
            candidate_member
                .get("result_dir")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    CliError::Contract(format!(
                        "candidate matrix member {manifest_id} missing result_dir"
                    ))
                })?,
        );
        require_existing_dir(&baseline_result_dir)?;
        require_existing_dir(&candidate_result_dir)?;
        let member_compare_root = cases_dir.join(sanitize_component(&manifest_id));
        fs::create_dir_all(&member_compare_root).map_err(|error| {
            CliError::Runtime(format!(
                "failed to create matrix member compare directory {}: {error}",
                member_compare_root.display()
            ))
        })?;
        let member_compare_dir = write_compare_artifacts(
            &baseline_result_dir,
            &candidate_result_dir,
            &member_compare_root,
        )?;
        let regression = load_json_value(&member_compare_dir.join("regression.json"))?;
        member_results.push(build_matrix_compare_member_result(
            baseline_member,
            &manifest_id,
            &member_compare_dir,
            &regression,
        )?);
    }

    write_json_file(&result_dir.join("baseline_matrix.json"), &baseline_matrix)?;
    write_json_file(&result_dir.join("candidate_matrix.json"), &candidate_matrix)?;
    write_json_file(
        &result_dir.join("matrix_regression.json"),
        &build_matrix_regression_json(
            matrix_id,
            baseline_dir,
            candidate_dir,
            &baseline_matrix,
            &candidate_matrix,
            &member_results,
        ),
    )?;
    fs::write(
        result_dir.join("summary.md"),
        build_matrix_compare_summary_markdown(
            matrix_id,
            &baseline_matrix,
            &candidate_matrix,
            &member_results,
        ),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write matrix compare summary.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(result_dir)
}
