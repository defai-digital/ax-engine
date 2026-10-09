
use super::*;

#[test]
fn gated_delta_threadgroup_y_env_parser_parity() {
    // Unset and invalid values keep the historical default of 4.
    assert_eq!(gated_delta_verify_threadgroup_y_env_for(None), 4);
    for value in ["3", "0", "-1", "64", "100", "abc", ""] {
        assert_eq!(
            gated_delta_verify_threadgroup_y_env_for(Some(value)),
            4,
            "invalid value {value:?} must default to 4"
        );
    }
    // Valid values pass through.
    assert_eq!(gated_delta_verify_threadgroup_y_env_for(Some("4")), 4);
    assert_eq!(gated_delta_verify_threadgroup_y_env_for(Some("8")), 8);
    assert_eq!(gated_delta_verify_threadgroup_y_env_for(Some("16")), 16);
    assert_eq!(gated_delta_verify_threadgroup_y_env_for(Some("32")), 32);
    // No trimming: surrounding whitespace fails the i32 parse.
    assert_eq!(gated_delta_verify_threadgroup_y_env_for(Some(" 8")), 4);
    assert_eq!(gated_delta_verify_threadgroup_y_env_for(Some("8 ")), 4);
}

#[test]
fn multimodal_prefix_reuse_parser_parity() {
    assert!(!multimodal_prefix_reuse_for(None));
    for value in [
        "1", "true", "TRUE", "True", "on", "ON", "yes", "YES", " 1 ", "\ttrue\n",
    ] {
        assert!(
            multimodal_prefix_reuse_for(Some(value)),
            "expected truthy for {value:?}"
        );
    }
    // The site's exact accepted set: "on" engages, "off" does not.
    for value in ["0", "false", "no", "off", "", "  ", "anything"] {
        assert!(
            !multimodal_prefix_reuse_for(Some(value)),
            "expected falsy for {value:?}"
        );
    }
}

#[test]
fn embed_mean_compile_threshold_parser_parity() {
    assert_eq!(embed_mean_compile_threshold_for(None), 512);
    assert_eq!(embed_mean_compile_threshold_for(Some("256")), 256);
    assert_eq!(embed_mean_compile_threshold_for(Some("1024")), 1024);
    for value in ["abc", "-1", "1.5", ""] {
        assert_eq!(
            embed_mean_compile_threshold_for(Some(value)),
            512,
            "invalid value {value:?} must default to 512"
        );
    }
    // No trimming: surrounding whitespace fails the usize parse.
    assert_eq!(embed_mean_compile_threshold_for(Some(" 256")), 512);
    assert_eq!(embed_mean_compile_threshold_for(Some("256 ")), 512);
}

#[test]
fn mtp_linear_exact_replay_parser_parity() {
    assert!(!mtp_linear_exact_replay_for(None));
    assert!(!mtp_linear_exact_replay_for(Some("0")));
    // Any byte sequence other than exactly "0" engages (no trimming).
    for value in ["1", "true", "off", "", "no", " 0"] {
        assert!(
            mtp_linear_exact_replay_for(Some(value)),
            "expected engaged for {value:?}"
        );
    }
}

#[test]
fn mtp_ngram_cache_preserved_parser_parity() {
    // Default (unset) preserves the cache.
    assert!(mtp_ngram_cache_preserved_for(None));
    assert!(!mtp_ngram_cache_preserved_for(Some("reset")));
    // Only the exact byte string "reset" disables preservation (case and
    // whitespace sensitive).
    for value in [
        "Reset", "RESET", " reset", "reset ", "", "0", "1", "anything",
    ] {
        assert!(
            mtp_ngram_cache_preserved_for(Some(value)),
            "expected preserved for {value:?}"
        );
    }
}

#[test]
fn gemma4_moe_long_mt_parser_parity() {
    assert!(!gemma4_moe_long_mt_for(None));
    assert!(gemma4_moe_long_mt_for(Some("1")));
    assert!(gemma4_moe_long_mt_for(Some("true")));
    assert!(gemma4_moe_long_mt_for(Some("TRUE")));
    assert!(gemma4_moe_long_mt_for(Some("True")));
    for value in ["0", "false", "yes", "on", "", " 1", "true "] {
        assert!(
            !gemma4_moe_long_mt_for(Some(value)),
            "expected off for {value:?}"
        );
    }
}

#[test]
fn gemma4_loop_detection_mode_parser_parity() {
    assert_eq!(
        gemma4_loop_detection_mode_for(None),
        Gemma4LoopDetectionMode::Default
    );
    assert_eq!(
        gemma4_loop_detection_mode_for(Some("on")),
        Gemma4LoopDetectionMode::Default
    );
    assert_eq!(
        gemma4_loop_detection_mode_for(Some("anything")),
        Gemma4LoopDetectionMode::Default
    );
    for value in ["off", "0", "false", "no", " OFF ", "\tfalse\n"] {
        assert_eq!(
            gemma4_loop_detection_mode_for(Some(value)),
            Gemma4LoopDetectionMode::Off,
            "expected Off for {value:?}"
        );
    }
    assert_eq!(
        gemma4_loop_detection_mode_for(Some("force")),
        Gemma4LoopDetectionMode::Force
    );
    assert_eq!(
        gemma4_loop_detection_mode_for(Some("FORCE")),
        Gemma4LoopDetectionMode::Force
    );
}

/// Files where every `std::env::var(` read lives behind a cached accessor
/// or is a documented once-per-process read. `fastpath.rs` is the house
/// home for cached accessors.
const ENV_VAR_WHOLE_FILE_ALLOWED: &[&str] = &["src/fastpath.rs"];

/// `(file, owner)` pairs for the remaining load-time / once-per-process /
/// cached-in-place `std::env::var(` reads outside `fastpath.rs`. Adding a
/// new per-call read on a dispatch path will fail this test.
const ENV_VAR_ALLOWED_OWNERS: &[(&str, &str)] = &[
    ("src/batched_decode_policy.rs", "row_exact_moe_enabled"),
    ("src/batched_decode_session.rs", "batched_decode_enabled"),
    (
        "src/batched_decode_session.rs",
        "batched_decode_sampling_enabled",
    ),
    (
        "src/batched_decode_session.rs",
        "batched_decode_allow_uncertified",
    ),
    (
        "src/batched_decode_session.rs",
        "mtp_multirow_batch_enabled",
    ),
    ("src/batched_decode_session.rs", "decode_batch_bucket"),
    ("src/disk_prefix_cache.rs", "parsed"),
    ("src/disk_prefix_cache.rs", "enabled"),
    ("src/disk_prefix_cache.rs", "from_env"),
    ("src/expert_stream.rs", "stream_experts_env_enabled"),
    ("src/expert_stream.rs", "expert_layer_budget"),
    ("src/expert_stream.rs", "stream_experts_mode_checked"),
    (
        "src/gemma4_assistant_mtp.rs",
        "gemma4_assistant_mtp_env_enabled",
    ),
    (
        "src/gemma4_assistant_mtp.rs",
        "gemma4_assistant_mtp_max_depth_cap",
    ),
    (
        "src/gemma4_assistant_mtp.rs",
        "gemma4_assistant_mtp_require_exact_pair",
    ),
    (
        "src/gemma4_assistant_mtp.rs",
        "gemma4_assistant_mtp_debug_enabled",
    ),
    (
        "src/gemma4_assistant_mtp.rs",
        "gemma4_assistant_mtp_draft_min_confidence",
    ),
    (
        "src/gemma4_assistant_mtp.rs",
        "gemma4_assistant_mtp_deep_draft_min_confidence",
    ),
    ("src/gemma4_assistant_mtp.rs", "gate_env_explicit"),
    ("src/generate.rs", "direct_pipeline_barrier_enabled"),
    ("src/generate.rs", "direct_pipeline_stage_profile_enabled"),
    ("src/generate.rs", "prefill_time_debug_enabled"),
    ("src/kv_block_pool.rs", "fa_kv_block_pool_enabled"),
    ("src/kv_block_pool.rs", "fa_kv_block_sharing_enabled"),
    ("src/kv_block_pool.rs", "fa_native_paged_attention_enabled"),
    ("src/kv_block_pool.rs", "fa_block_pool_max_blocks_override"),
    ("src/kv_cache.rs", "kv_quant_env_disabled"),
    ("src/model/families/standard.rs", "enabled"),
    ("src/model/mod.rs", "embed_length_bucket"),
    ("src/model/mod.rs", "embed_length_split_enabled"),
    ("src/model/profile.rs", "profile_env_enabled"),
    ("src/model/profile.rs", "take_prefill_profile_snapshot"),
    (
        "src/model/shared/mlp.rs",
        "gemma4_split_prefill_ffn_enabled",
    ),
    ("src/model/shared/mlp.rs", "maybe_trace_moe_router"),
    ("src/model/shared/norm.rs", "use_flat_qk_norm_path"),
    ("src/model/shared/qwen4_exp_gdn_metal.rs", "enabled"),
    (
        "src/model/shared/utils.rs",
        "dense_wide_gemv_vec8_max_leading",
    ),
    ("src/model/shared/utils.rs", "trace_invariant_qmv_shape"),
    ("src/model/shared/verify_qmm.rs", "min_route_n"),
    ("src/model/shared/verify_qmm.rs", "msg_pad_m4_enabled"),
    ("src/model/shared/verify_qmm.rs", "msg_simdgroups"),
    ("src/model/shared/verify_qmm.rs", "split_k_huge_enabled"),
    (
        "src/model/whole_verify.rs",
        "try_whole_compiled_qwen_verify",
    ),
    ("src/mtp.rs", "mtp_draft_mode_from_env"),
    ("src/mtp.rs", "mtp_draft_min_confidence_from_env"),
    ("src/mtp.rs", "mtp_draft_min_confidence_explicit"),
    ("src/mtp_adaptive_gate.rs", "truthy_env"),
    ("src/mtp_adaptive_gate.rs", "parse_f32_env"),
    ("src/mtp_adaptive_gate.rs", "residual_window_from_env"),
    ("src/mtp_adaptive_gate.rs", "bins_from_env"),
    ("src/ngram_accel.rs", "effective_draft_confidence_threshold"),
    (
        "src/ngram_accel.rs",
        "effective_speculative_accept_threshold",
    ),
    (
        "src/per_layer_compile.rs",
        "COMPILE_CACHE_REFRESH_THRESHOLD",
    ),
    ("src/runner/mod.rs", "EMBED_GPU_NORMALIZE"),
    ("src/runner/mod.rs", "EMBED_NO_COMPILE"),
    ("src/runner/mod.rs", "prefix_demote_on_evict_enabled"),
    (
        "src/runner/mod.rs",
        "gemma4_assistant_mtp_confidence_mode_from_env",
    ),
    ("src/runner/mod.rs", "from_artifacts_inner"),
    (
        "src/runner/mtp_model_policy.rs",
        "qwen_linear_mtp_certification_candidate_from_env",
    ),
    (
        "src/runner/mtp_model_policy.rs",
        "deepseek_v4_mtp_certification_candidate_from_env",
    ),
    ("src/runner/mtp_ngram_gates.rs", "mtp_ngram_hurt_gate_mode"),
    ("src/runner/mtp_ngram_gates.rs", "mtp_ngram_min_support"),
    (
        "src/runner/mtp_ngram_gates.rs",
        "mtp_ngram_confidence_threshold",
    ),
    ("src/runner/mtp_ngram_gates.rs", "mtp_ngram_min_context_len"),
    (
        "src/runner/mtp_depth_controller.rs",
        "mtp_cost_depth_config_from_env",
    ),
    ("src/runner/mtp_profitability.rs", "env_u32"),
    ("src/runner/mtp_profitability.rs", "env_f64"),
    (
        "src/runner/mtp_profitability.rs",
        "mtp_profitability_config_from_env",
    ),
    ("src/runner/mtp_tuning.rs", "mtp_ngram_gate_min_samples"),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_auto_optimistic_deactivate_threshold",
    ),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_auto_optimistic_min_samples",
    ),
    ("src/runner/mtp_tuning.rs", "cached_env_f32"),
    ("src/runner/mtp_tuning.rs", "mtp_bypass_min_samples"),
    ("src/runner/mtp_tuning.rs", "mtp_min_remaining_tokens"),
    ("src/runner/mtp_tuning.rs", "mtp_bypass_threshold"),
    ("src/runner/mtp_tuning.rs", "cached_env_u32"),
    ("src/runner/mtp_tuning.rs", "cached_env_f64"),
    ("src/runner/mtp_tuning.rs", "mtp_ngram_hurt_margin"),
    ("src/runner/mtp_tuning.rs", "mtp_ngram_gate_policy_from_env"),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_ngram_utility_min_emitted_tokens",
    ),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_ngram_utility_min_ngram_tokens",
    ),
    ("src/runner/mtp_tuning.rs", "mtp_ngram_utility_margin_ratio"),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_ngram_utility_hysteresis_steps",
    ),
    ("src/runner/mtp_tuning.rs", "mtp_ngram_safety_mode"),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_ngram_auto_disable_mtp_threshold",
    ),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_ngram_auto_disable_min_ngram",
    ),
    ("src/runner/mtp_tuning.rs", "mtp_ngram_self_tune_threshold"),
    ("src/runner/mtp_tuning.rs", "mtp_ngram_self_tune_warmup"),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_ngram_auto_disable_mtp_warmup",
    ),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_ngram_auto_disable_ngram_warmup",
    ),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_ngram_acceptance_mode_from_env",
    ),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_model_acceptance_mode_from_env",
    ),
    ("src/runner/mtp_tuning.rs", "mtp_ngram_stacking_env"),
    ("src/runner/mtp_tuning.rs", "mtp_optimistic_from_env"),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_auto_optimistic_enabled_from_env",
    ),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_optimistic_draft_min_confidence_override",
    ),
    ("src/runner/mtp_tuning.rs", "mtp_skip_state_from_env"),
    (
        "src/runner/mtp_tuning.rs",
        "mtp_target_softmax_topk_from_env",
    ),
    ("src/runner/mtp_tuning.rs", "ngram_policy_variant_from_env"),
    (
        "src/runner/mtp_tuning.rs",
        "adaptive_ngram_saturation_threshold",
    ),
    ("src/runner/prefix_cache.rs", "from_env"),
    ("src/runner/runner_telemetry.rs", "from_specs"),
    ("src/speculation_profile.rs", "speculation_profile_from_env"),
    ("src/tiel_memory_policy.rs", "wired_limit_scale_override"),
    ("src/weight_rotation.rs", "weight_rotation_mode"),
    ("src/weight_rotation.rs", "shadow_log_rotation_candidates"),
    ("src/weights.rs", "mmap_weights_enabled"),
    ("src/weights.rs", "draft_lm_head_spec_from_env"),
    ("src/weights.rs", "runtime_draft_lm_head_spec_enabled"),
    ("src/weights.rs", "apply_draft_temperature_override"),
    ("src/weights.rs", "apply_mtp_depth_policy"),
    ("src/weights.rs", "apply_mtp_max_depth_cap"),
    ("src/weights.rs", "skip_vision_sidecar"),
    ("src/weights.rs", "skip_mtp_sidecar"),
];

/// Strip a leading visibility prefix so `pub fn`, `pub(crate) fn`,
/// `pub(super) fn`, and `pub(in path) fn` all reduce to the bare keyword.
fn strip_visibility(mut s: &str) -> &str {
    loop {
        let next = s
            .strip_prefix("pub(crate) ")
            .or_else(|| s.strip_prefix("pub(super) "))
            .or_else(|| s.strip_prefix("pub(self) "))
            .or_else(|| s.strip_prefix("pub "));
        match next {
            Some(rest) => s = rest,
            None => break,
        }
    }
    match s.strip_prefix("pub(in ") {
        Some(after) => match after.find(") ") {
            Some(index) => &after[index + 2..],
            None => s,
        },
        None => s,
    }
}

/// Identifier at the start of a declaration (`fn`, `static`, `const`,
/// `mod`).
fn leading_ident(s: &str) -> Option<String> {
    let name: String = s
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}

#[derive(Clone, Copy)]
enum StringKind {
    Normal,
    Raw(usize),
}

/// Return each line with string-literal and comment contents replaced by
/// spaces, so structural analysis (brace counting and `fn`/`static`/`const`
/// detection) never mistakes MSL source text inside a `r#" ... "#` kernel
/// constant (or a `//` comment) for real code.
fn code_only(lines: &[&str]) -> Vec<String> {
    let mut result = Vec::with_capacity(lines.len());
    let mut in_block_comment = false;
    let mut in_string: Option<StringKind> = None;
    for line in lines {
        let chars: Vec<char> = line.chars().collect();
        let mut out = String::with_capacity(chars.len());
        let mut i = 0usize;
        while i < chars.len() {
            let c = chars[i];
            if in_block_comment {
                out.push(' ');
                if c == '*' && i + 1 < chars.len() && chars[i + 1] == '/' {
                    in_block_comment = false;
                    out.push(' ');
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }
            match in_string {
                Some(StringKind::Raw(hashes)) => {
                    if c == '"' {
                        let mut j = i + 1;
                        let mut seen = 0usize;
                        while j < chars.len() && chars[j] == '#' && seen < hashes {
                            seen += 1;
                            j += 1;
                        }
                        if seen == hashes {
                            in_string = None;
                            for _ in i..j {
                                out.push(' ');
                            }
                            i = j;
                            continue;
                        }
                    }
                    out.push(' ');
                    i += 1;
                    continue;
                }
                Some(StringKind::Normal) => {
                    if c == '\\' {
                        out.push(' ');
                        if i + 1 < chars.len() {
                            out.push(' ');
                            i += 2;
                        } else {
                            i += 1;
                        }
                    } else if c == '"' {
                        in_string = None;
                        out.push(' ');
                        i += 1;
                    } else {
                        out.push(' ');
                        i += 1;
                    }
                    continue;
                }
                None => {}
            }
            if c == '/' && i + 1 < chars.len() {
                if chars[i + 1] == '/' {
                    while out.len() < chars.len() {
                        out.push(' ');
                    }
                    break;
                }
                if chars[i + 1] == '*' {
                    in_block_comment = true;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                    continue;
                }
            }
            if (c == 'r' || c == 'R') && i + 1 < chars.len() {
                let mut j = i + 1;
                let mut hashes = 0usize;
                while j < chars.len() && chars[j] == '#' {
                    hashes += 1;
                    j += 1;
                }
                if j < chars.len() && chars[j] == '"' {
                    in_string = Some(StringKind::Raw(hashes));
                    for _ in i..=j {
                        out.push(' ');
                    }
                    i = j + 1;
                    continue;
                }
            }
            if c == '"' {
                in_string = Some(StringKind::Normal);
                out.push(' ');
                i += 1;
                continue;
            }
            out.push(c);
            i += 1;
        }
        result.push(out);
    }
    result
}

/// Map every line to its enclosing owner: the innermost `fn`, a module-
/// scope `static`/`const` initializer, or `<module>` at the top level.
/// Lines inside a `mod tests` / `mod *_tests` block map to `<test>` so the
/// guard ignores test fixtures.
fn owners_for_lines(lines: &[&str]) -> Vec<String> {
    let code = code_only(lines);
    let mut owners = vec![String::from("<module>"); lines.len()];
    let mut depth: i64 = 0;
    let mut fn_stack: Vec<(String, i64)> = Vec::new();
    // A `fn` seen before its opening brace (multi-line signature).
    let mut pending_fn: Option<(String, i64)> = None;
    let mut module_item: Option<String> = None;
    let mut test_depth: Option<i64> = None;

    for (index, line) in code.iter().enumerate() {
        let trimmed = line.trim_start();
        let stripped = strip_visibility(trimmed);

        // Owner before this line's declarations take effect.
        owners[index] = if test_depth.is_some() {
            String::from("<test>")
        } else if let Some((name, _)) = fn_stack.last() {
            name.clone()
        } else {
            module_item
                .clone()
                .unwrap_or_else(|| String::from("<module>"))
        };

        // Function declarations nest at any depth (free functions, methods,
        // and nested fns inside a config builder). The body may open on a
        // later line, so the fn is promoted once its brace appears.
        if let Some(rest) = stripped.strip_prefix("fn ")
            && let Some(name) = leading_ident(rest)
            && pending_fn.is_none()
        {
            pending_fn = Some((name, depth));
        } else if depth == 0
            && let Some(rest) = stripped.strip_prefix("static ")
            && let Some(name) = leading_ident(rest)
        {
            module_item = Some(name);
        } else if depth == 0
            && let Some(rest) = stripped.strip_prefix("const ")
            && let Some(name) = leading_ident(rest)
        {
            module_item = Some(name);
        }

        // Track brace depth, then open/close any `mod tests` block.
        depth += line.matches('{').count() as i64;
        depth -= line.matches('}').count() as i64;

        if let Some((name, entry)) = pending_fn.take() {
            if depth > entry {
                fn_stack.push((name, depth));
            } else {
                pending_fn = Some((name, entry));
            }
        }

        if test_depth.is_none()
            && let Some(rest) = stripped.strip_prefix("mod ")
            && let Some(name) = leading_ident(rest)
            && (name == "tests" || name.ends_with("_tests"))
        {
            test_depth = Some(depth);
        } else if test_depth.is_some() && depth < test_depth.unwrap() {
            test_depth = None;
        }

        while let Some(&(_, entry)) = fn_stack.last() {
            if depth < entry {
                fn_stack.pop();
            } else {
                break;
            }
        }
    }
    owners
}

#[test]
fn env_var_reads_are_allow_listed() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.clone()];
    let mut rust_files: Vec<std::path::PathBuf> = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                rust_files.push(path);
            }
        }
    }
    rust_files.sort();
    for path in rust_files {
        let rel = path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap();
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        // Skip standalone probe/tool binaries and test-only files.
        if rel_str.starts_with("src/bin/")
            || rel_str.contains("_tests.rs")
            || rel_str.contains("_tests/")
            || rel_str.ends_with("tests.rs")
            || rel_str.contains("qwen4_exp")
            || rel_str.contains("flash_next")
            || rel_str.ends_with("vl_fixture_tests.rs")
        {
            continue;
        }
        let Ok(contents) = std::fs::read_to_string(&path) else {
            continue;
        };
        let lines: Vec<&str> = contents.lines().collect();
        let owners = owners_for_lines(&lines);
        for (index, line) in lines.iter().enumerate() {
            if !line.contains("std::env::var(") {
                continue;
            }
            if ENV_VAR_WHOLE_FILE_ALLOWED.contains(&rel_str.as_str()) {
                continue;
            }
            let owner = owners[index].as_str();
            if owner == "<test>" {
                continue;
            }
            if ENV_VAR_ALLOWED_OWNERS.contains(&(rel_str.as_str(), owner)) {
                continue;
            }
            offenders.push(format!("{rel_str}:{} in `{owner}`", index + 1));
        }
    }
    assert!(
        offenders.is_empty(),
        "std::env::var( reads outside the allow-list:\n{}",
        offenders.join("\n")
    );
}
