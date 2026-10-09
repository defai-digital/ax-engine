
use super::*;

fn probe(name: &str, value: &str) -> bool {
    // SAFETY: each test owns a disjoint set of env-var names. Remove
    // before asserting so a failing assert does not leak the var.
    unsafe {
        std::env::set_var(name, value);
    }
    let observed = parse_bool_env(name);
    unsafe {
        std::env::remove_var(name);
    }
    observed
}

#[test]
fn parse_bool_env_treats_truthy_values_as_engaged() {
    // Exercises canonical casing, all-upper, mixed case, and surrounding
    // whitespace to lock in the parser contract documented at the module
    // level.
    for value in [
        "1", "true", "TRUE", "True", "tRuE", "yes", "YES", "Yes", " 1 ", "\ttrue\n",
    ] {
        let name = format!("AX_FASTPATH_TEST_TRUTHY_{}", value.trim());
        assert!(probe(&name, value), "expected truthy for {value:?}");
    }
}

#[test]
fn parse_bool_env_rejects_other_values() {
    for value in ["0", "false", "no", "off", "on", "", "anything", "  "] {
        let name = format!("AX_FASTPATH_TEST_FALSY_{}", value.trim());
        assert!(!probe(&name, value), "expected falsy for {value:?}");
    }
}

#[test]
fn parse_bool_env_unset_is_false() {
    assert!(!parse_bool_env("AX_FASTPATH_TEST_DEFINITELY_UNSET"));
}

#[test]
fn qwen_mtp_opt_in_gates_stay_off_when_unset() {
    for var in [
        "AX_MLX_MTP_PROFITABILITY_THROUGHPUT",
        "AX_MLX_MTP_NATIVE_GREEDY_VERIFY_LOGITS",
        "AX_MLX_MTP_WHOLE_VERIFY_COMPILE",
    ] {
        unsafe {
            std::env::remove_var(var);
        }
        assert!(
            !parse_bool_env(var),
            "{var} must stay default-off when unset"
        );
    }
}

#[test]
fn qwen_mtp_eager_gates_default_on_unless_kill_switched() {
    for var in [
        "AX_MLX_MTP_SKIP_PREFIX_CHECKPOINT",
        "AX_MLX_MTP_REBIND_VERIFY_FA",
        "AX_MTP_COMPILED_HEAD_FIXED_KV",
    ] {
        unsafe {
            std::env::remove_var(var);
        }
        assert!(
            parse_bool_env_default_on(var),
            "{var} must default on when unset"
        );
        unsafe {
            std::env::set_var(var, "0");
        }
        assert!(!parse_bool_env_default_on(var), "{var}=0 must kill-switch");
        unsafe {
            std::env::remove_var(var);
        }
    }
}

#[test]
fn qwen_linear_mtp_exact_resolution_is_capability_bounded() {
    assert_eq!(
        resolve_qwen_linear_mtp_exact_with_override(false, None),
        (false, QwenLinearMtpExactSelection::Ineligible)
    );
    assert_eq!(
        resolve_qwen_linear_mtp_exact_with_override(false, Some(true)),
        (false, QwenLinearMtpExactSelection::Ineligible),
        "an env opt-in must not bypass the hard model gate"
    );
    assert_eq!(
        resolve_qwen_linear_mtp_exact_with_override(true, None),
        (true, QwenLinearMtpExactSelection::Auto)
    );
    assert_eq!(
        resolve_qwen_linear_mtp_exact_with_override(true, Some(true)),
        (true, QwenLinearMtpExactSelection::ExplicitEnabled)
    );
    assert_eq!(
        resolve_qwen_linear_mtp_exact_with_override(true, Some(false)),
        (false, QwenLinearMtpExactSelection::ExplicitDisabled)
    );
}

#[test]
fn qwen_linear_mtp_exact_scope_is_nested_and_restored() {
    let baseline = qwen_linear_mtp_exact_enabled();
    {
        let _outer = scoped_qwen_linear_mtp_exact(true);
        assert!(qwen_linear_mtp_exact_enabled());
        {
            let _inner = scoped_qwen_linear_mtp_exact(false);
            assert!(!qwen_linear_mtp_exact_enabled());
        }
        assert!(qwen_linear_mtp_exact_enabled());
    }
    assert_eq!(qwen_linear_mtp_exact_enabled(), baseline);
}

#[test]
fn relaxed_target_verify_enables_only_verify_fast_kernel_marker() {
    let exact_baseline = qwen_linear_mtp_exact_enabled();
    let fast_baseline = qwen_linear_mtp_verify_fast_kernels_enabled();
    let target_baseline = qwen_linear_mtp_target_verify_enabled();
    {
        let _exact = scoped_qwen_linear_mtp_exact(false);
        assert!(!qwen_linear_mtp_exact_enabled());
        assert!(!qwen_linear_mtp_verify_fast_kernels_enabled());
        assert!(!qwen_linear_mtp_target_verify_enabled());
        {
            let _verify = scoped_qwen_linear_mtp_target_verify(true);
            assert!(!qwen_linear_mtp_exact_enabled());
            assert!(qwen_linear_mtp_verify_fast_kernels_enabled());
            assert!(qwen_linear_mtp_target_verify_enabled());
        }
        assert!(!qwen_linear_mtp_verify_fast_kernels_enabled());
        assert!(!qwen_linear_mtp_target_verify_enabled());
    }
    assert_eq!(qwen_linear_mtp_exact_enabled(), exact_baseline);
    assert_eq!(qwen_linear_mtp_verify_fast_kernels_enabled(), fast_baseline);
    assert_eq!(qwen_linear_mtp_target_verify_enabled(), target_baseline);
}

#[test]
fn relaxed_mtp_async_dual_gate_up_is_scope_family_and_window_gated() {
    assert!(should_mtp_async_dual_gate_up_for(true, true, "qwen3_5", 3));
    assert!(should_mtp_async_dual_gate_up_for(true, true, "QWEN3_5", 4));
    assert!(!should_mtp_async_dual_gate_up_for(
        false, true, "qwen3_5", 3
    ));
    assert!(!should_mtp_async_dual_gate_up_for(
        true, false, "qwen3_5", 3
    ));
    assert!(!should_mtp_async_dual_gate_up_for(true, true, "qwen3_5", 1));
    assert!(!should_mtp_async_dual_gate_up_for(true, true, "gemma4", 3));
}

#[test]
fn relaxed_mtp_session_scope_restores_nested_state() {
    let baseline = qwen_linear_mtp_relaxed_session_enabled();
    {
        let _outer = scoped_qwen_linear_mtp_relaxed_session(true);
        assert!(qwen_linear_mtp_relaxed_session_enabled());
        {
            let _inner = scoped_qwen_linear_mtp_relaxed_session(false);
            assert!(!qwen_linear_mtp_relaxed_session_enabled());
        }
        assert!(qwen_linear_mtp_relaxed_session_enabled());
    }
    assert_eq!(qwen_linear_mtp_relaxed_session_enabled(), baseline);
}

#[test]
fn exact_verify_async_kernel_boundary_is_s2_to_s4_only() {
    let baseline = qwen_linear_mtp_exact_enabled();
    {
        let _exact = scoped_qwen_linear_mtp_exact(true);
        assert!(!should_exact_verify_async_kernel_boundary(1));
        assert!(should_exact_verify_async_kernel_boundary(2));
        assert!(should_exact_verify_async_kernel_boundary(3));
        assert!(should_exact_verify_async_kernel_boundary(4));
        assert!(!should_exact_verify_async_kernel_boundary(5));
    }
    {
        let _off = scoped_qwen_linear_mtp_exact(false);
        assert!(!should_exact_verify_async_kernel_boundary(2));
    }
    assert_eq!(qwen_linear_mtp_exact_enabled(), baseline);
}

fn probe_default_on(name: &str, value: &str) -> bool {
    // SAFETY: each test owns a disjoint set of env-var names. Remove
    // before asserting so a failing assert does not leak the var.
    unsafe {
        std::env::set_var(name, value);
    }
    let observed = parse_bool_env_default_on(name);
    unsafe {
        std::env::remove_var(name);
    }
    observed
}

#[test]
fn parse_bool_env_default_on_only_rejects_explicit_falsy_values() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_DEFAULT_ON_UNSET"
    ));
    for value in [
        "0", "false", "FALSE", "False", "no", "NO", "No", "off", "OFF",
    ] {
        let name = format!("AX_FASTPATH_TEST_DEFAULT_ON_FALSY_{}", value.trim());
        assert!(
            !probe_default_on(&name, value),
            "expected explicit falsy for {value:?}"
        );
    }
    for value in ["", " ", "1", "true", "yes", "anything"] {
        let name = format!(
            "AX_FASTPATH_TEST_DEFAULT_ON_TRUTHY_{}",
            value.trim().replace(' ', "space")
        );
        assert!(
            probe_default_on(&name, value),
            "expected default-on truthy for {value:?}"
        );
    }
}

#[test]
fn linear_attention_projection_packing_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_PACK_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_PACK_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_PACK_ENABLED",
        "1"
    ));
}

#[test]
fn direct_cpp_linear_attention_inputs_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_INPUTS_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_INPUTS_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_INPUTS_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_direct_cpp_linear_attention_inputs_uses_default_on_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_INPUTS_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_INPUTS_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_INPUTS_ENABLED",
        "1"
    ));
}

#[test]
fn direct_cpp_linear_attention_post_input_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_POST_INPUT_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_POST_INPUT_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_POST_INPUT_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_direct_cpp_linear_attention_post_input_uses_default_on_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_POST_INPUT_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_POST_INPUT_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_POST_INPUT_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_linear_attention_prefill_post_input_metal_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_linear_attention_decode_post_input_metal_uses_default_on_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn fused_prefill_attention_qwen_is_family_scoped_and_default_on() {
    assert!(super::fused_prefill_attention_family_supported("qwen3_5"));
    assert!(super::fused_prefill_attention_family_supported(
        "qwen3_next"
    ));
    assert!(super::fused_prefill_attention_family_supported("gemma4"));
    assert!(!super::fused_prefill_attention_family_supported(
        "glm4_moe_lite"
    ));
    assert!(!super::fused_prefill_qwen_skip_offset("qwen3_5", false));
    assert!(super::fused_prefill_qwen_skip_offset("qwen3_5", true));
    assert!(!super::fused_prefill_qwen_skip_offset("gemma4", true));
    assert!(
        !super::fused_prefill_attention_should_try_for_seq("gemma4", 512),
        "Gemma p512 fused prefill stays default-OFF"
    );
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_FUSED_PREFILL_ATTENTION_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_FUSED_PREFILL_ATTENTION_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_FUSED_PREFILL_ATTENTION_ENABLED",
        "1"
    ));
}

#[test]
fn gemma4_fused_prefill_p128_is_seq_and_family_gated() {
    assert!(should_gemma4_fused_prefill_p128_for(true, "gemma4", 128));
    assert!(should_gemma4_fused_prefill_p128_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(super::fused_prefill_attention_should_try_for_seq(
        "gemma4", 128
    ));
    assert!(
        !should_gemma4_fused_prefill_p128_for(true, "gemma4", 512),
        "p512 must stay on the portable attention path"
    );
    assert!(!should_gemma4_fused_prefill_p128_for(true, "gemma4", 2048));
    assert!(!should_gemma4_fused_prefill_p128_for(true, "gemma4", 1));
    assert!(!should_gemma4_fused_prefill_p128_for(true, "qwen3_5", 128));
    assert!(!should_gemma4_fused_prefill_p128_for(false, "gemma4", 128));
}

#[test]
fn gemma4_fused_prefill_fold_post_norm_requires_p128_and_weight() {
    assert!(should_gemma4_fused_prefill_fold_post_norm_for(
        true, "gemma4", 128, true
    ));
    assert!(should_gemma4_fused_prefill_fold_post_norm_for(
        true,
        "gemma4_unified",
        128,
        true
    ));
    assert!(
        !should_gemma4_fused_prefill_fold_post_norm_for(true, "gemma4", 128, false),
        "no sandwich post-norm means the fused call stays o-proj only"
    );
    assert!(!should_gemma4_fused_prefill_fold_post_norm_for(
        true, "gemma4", 512, true
    ));
    assert!(!should_gemma4_fused_prefill_fold_post_norm_for(
        true, "qwen3_5", 128, true
    ));
    assert!(!should_gemma4_fused_prefill_fold_post_norm_for(
        false, "gemma4", 128, true
    ));
}

#[test]
fn gemma4_async_dual_gate_up_p128_is_seq_and_family_gated() {
    assert!(should_gemma4_async_dual_gate_up_p128_for(
        true, "gemma4", 128
    ));
    assert!(should_gemma4_async_dual_gate_up_p128_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(
        !should_gemma4_async_dual_gate_up_p128_for(true, "gemma4", 512),
        "p512 stays on the serial gate/up submit"
    );
    assert!(!should_gemma4_async_dual_gate_up_p128_for(
        true, "gemma4", 2048
    ));
    assert!(!should_gemma4_async_dual_gate_up_p128_for(
        true, "gemma4", 1
    ));
    assert!(!should_gemma4_async_dual_gate_up_p128_for(
        true, "qwen3_5", 128
    ));
    assert!(!should_gemma4_async_dual_gate_up_p128_for(
        false, "gemma4", 128
    ));
}

#[test]
fn gemma4_async_first_kv_p128_is_seq_and_family_gated() {
    assert!(should_gemma4_async_first_kv_p128_for(true, "gemma4", 128));
    assert!(should_gemma4_async_first_kv_p128_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(
        !should_gemma4_async_first_kv_p128_for(true, "gemma4", 512),
        "p512 stays on the lazy first-KV submit"
    );
    assert!(!should_gemma4_async_first_kv_p128_for(true, "gemma4", 2048));
    assert!(!should_gemma4_async_first_kv_p128_for(true, "gemma4", 1));
    assert!(!should_gemma4_async_first_kv_p128_for(true, "qwen3_5", 128));
    assert!(!should_gemma4_async_first_kv_p128_for(false, "gemma4", 128));
}

#[test]
fn gemma4_dual_stream_gate_up_p128_is_seq_and_family_gated() {
    assert!(should_gemma4_dual_stream_gate_up_p128_for(
        true, "gemma4", 128
    ));
    assert!(should_gemma4_dual_stream_gate_up_p128_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(
        !should_gemma4_dual_stream_gate_up_p128_for(true, "gemma4", 512),
        "p512 stays on the serial / compiled split-MLP path"
    );
    assert!(!should_gemma4_dual_stream_gate_up_p128_for(
        true, "gemma4", 2048
    ));
    assert!(!should_gemma4_dual_stream_gate_up_p128_for(
        true, "gemma4", 1
    ));
    assert!(!should_gemma4_dual_stream_gate_up_p128_for(
        true, "qwen3_5", 128
    ));
    assert!(!should_gemma4_dual_stream_gate_up_p128_for(
        false, "gemma4", 128
    ));
}

#[test]
fn gemma4_packed_ffn_compile_p128_is_seq_and_family_gated() {
    assert!(should_gemma4_packed_ffn_compile_p128_for(
        true, "gemma4", 128
    ));
    assert!(should_gemma4_packed_ffn_compile_p128_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(
        !should_gemma4_packed_ffn_compile_p128_for(true, "gemma4", 512),
        "p512 keeps split gate/up and the 256-leading compile floor"
    );
    assert!(!should_gemma4_packed_ffn_compile_p128_for(
        true, "gemma4", 2048
    ));
    assert!(!should_gemma4_packed_ffn_compile_p128_for(
        true, "gemma4", 1
    ));
    assert!(!should_gemma4_packed_ffn_compile_p128_for(
        true, "qwen3_5", 128
    ));
    assert!(!should_gemma4_packed_ffn_compile_p128_for(
        false, "gemma4", 128
    ));
}

#[test]
fn qwen_gated_delta_decode_metal_uses_default_on_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_DECODE_METAL_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_DECODE_METAL_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_DECODE_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_gated_delta_prefill_contiguous_is_seq_gated() {
    assert!(should_qwen_gated_delta_prefill_contiguous_for(true, 1024));
    assert!(should_qwen_gated_delta_prefill_contiguous_for(true, 2));
    assert!(
        !should_qwen_gated_delta_prefill_contiguous_for(true, 1),
        "decode already uses a contiguous row-0 path"
    );
    assert!(!should_qwen_gated_delta_prefill_contiguous_for(false, 1024));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_la_fused_qkvz_ba_qmm_is_seq_and_quant_gated() {
    assert!(should_qwen_la_fused_qkvz_ba_qmm_for(true, 1024, true));
    assert!(should_qwen_la_fused_qkvz_ba_qmm_for(true, 2, true));
    assert!(
        !should_qwen_la_fused_qkvz_ba_qmm_for(true, 1, true),
        "decode keeps matching-bits two-qmm packing"
    );
    assert!(!should_qwen_la_fused_qkvz_ba_qmm_for(true, 1024, false));
    assert!(!should_qwen_la_fused_qkvz_ba_qmm_for(false, 1024, true));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_LA_FUSED_QKVZ_BA_QMM_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_LA_FUSED_QKVZ_BA_QMM_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_LA_FUSED_QKVZ_BA_QMM_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_prefill_down_compile_is_seq_and_leading_gated() {
    assert!(should_qwen_prefill_down_compile_for(true, 1024, 128));
    assert!(should_qwen_prefill_down_compile_for(true, 2, 2048));
    assert!(!should_qwen_prefill_down_compile_for(true, 1, 128));
    assert!(!should_qwen_prefill_down_compile_for(true, 1024, 64));
    assert!(!should_qwen_prefill_down_compile_for(false, 1024, 128));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_PREFILL_DOWN_COMPILE_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_DOWN_COMPILE_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_DOWN_COMPILE_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_prefill_chunk_1536_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1536_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1536_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1536_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_prefill_chunk_1280_uses_opt_in_contract() {
    assert!(
        !qwen_prefill_chunk_1280_enabled(),
        "closed 1280 chunk stays default-off"
    );
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1280_ENABLED",
        "1"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1280_DISABLED",
        "0"
    ));
}

#[test]
fn qwen_compiled_gated_delta_prefill_is_seq_gated() {
    assert!(should_qwen_compiled_gated_delta_prefill_for(true, 1024));
    assert!(should_qwen_compiled_gated_delta_prefill_for(true, 2));
    assert!(!should_qwen_compiled_gated_delta_prefill_for(true, 1));
    assert!(!should_qwen_compiled_gated_delta_prefill_for(false, 1024));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_COMPILED_GATED_DELTA_PREFILL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_COMPILED_GATED_DELTA_PREFILL_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_COMPILED_GATED_DELTA_PREFILL_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_packed_la_inputs_compile_is_seq_gated() {
    assert!(should_qwen_packed_la_inputs_compile_for(true, 1024));
    assert!(should_qwen_packed_la_inputs_compile_for(true, 2048));
    assert!(
        !should_qwen_packed_la_inputs_compile_for(true, 512),
        "512-token packed LA compile stays closed"
    );
    assert!(!should_qwen_packed_la_inputs_compile_for(true, 1));
    assert!(!should_qwen_packed_la_inputs_compile_for(false, 1024));
}

#[test]
fn qwen_la_post_input_compile_is_seq_gated() {
    assert!(should_qwen_la_post_input_compile_for(true, 1024));
    assert!(should_qwen_la_post_input_compile_for(true, 2048));
    assert!(
        !should_qwen_la_post_input_compile_for(true, 512),
        "512-token post-input compile stays closed"
    );
    assert!(!should_qwen_la_post_input_compile_for(true, 1));
    assert!(!should_qwen_la_post_input_compile_for(false, 1024));
}

#[test]
fn qwen_prefill_contiguous_la_input_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_contiguous_la_input_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_contiguous_la_input_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(
        !should_qwen_prefill_contiguous_la_input_for(true, "qwen3_5", 512),
        "512-token LA input contiguous stays closed"
    );
    assert!(!should_qwen_prefill_contiguous_la_input_for(
        true, "qwen3_5", 1
    ));
    assert!(!should_qwen_prefill_contiguous_la_input_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_contiguous_la_input_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_la_contiguous_qkv_is_seq_gated() {
    assert!(should_qwen_la_contiguous_qkv_for(true, 1024));
    assert!(should_qwen_la_contiguous_qkv_for(true, 2048));
    assert!(
        !should_qwen_la_contiguous_qkv_for(true, 512),
        "512-token LA qkv contiguous stays closed"
    );
    assert!(!should_qwen_la_contiguous_qkv_for(true, 1));
    assert!(!should_qwen_la_contiguous_qkv_for(false, 1024));
}

#[test]
fn qwen_la_prefill_q2_is_seq_gated() {
    assert!(should_qwen_la_prefill_q2_for(true, 1024));
    assert!(should_qwen_la_prefill_q2_for(true, 2048));
    assert!(
        !should_qwen_la_prefill_q2_for(true, 512),
        "512-token LA q2 overlay stays closed"
    );
    assert!(!should_qwen_la_prefill_q2_for(true, 1));
    assert!(!should_qwen_la_prefill_q2_for(false, 1024));
}

#[test]
fn qwen_prefill_q2_down_is_seq_gated() {
    assert!(should_qwen_prefill_q2_down_for(true, 1024));
    assert!(should_qwen_prefill_q2_down_for(true, 2048));
    assert!(
        !should_qwen_prefill_q2_down_for(true, 512),
        "512-token FFN down q2 overlay stays closed"
    );
    assert!(!should_qwen_prefill_q2_down_for(true, 1));
    assert!(!should_qwen_prefill_q2_down_for(false, 1024));
}

#[test]
fn qwen_gd_prefill_chunkwise_is_seq_gated() {
    assert!(should_qwen_gd_prefill_chunkwise_for(true, 1024));
    assert!(should_qwen_gd_prefill_chunkwise_for(true, 2048));
    assert!(
        !should_qwen_gd_prefill_chunkwise_for(true, 512),
        "512-token GD chunkwise stays on the oneshot TG path"
    );
    assert!(!should_qwen_gd_prefill_chunkwise_for(true, 1));
    assert!(!should_qwen_gd_prefill_chunkwise_for(false, 1024));
}

#[test]
fn qwen_prefill_ffn_gs64_is_seq_gated() {
    assert!(should_qwen_prefill_ffn_gs64_for(true, 1024));
    assert!(should_qwen_prefill_ffn_gs64_for(true, 2048));
    assert!(
        !should_qwen_prefill_ffn_gs64_for(true, 512),
        "512-token FFN gs64 overlay stays closed"
    );
    assert!(!should_qwen_prefill_ffn_gs64_for(true, 1));
    assert!(!should_qwen_prefill_ffn_gs64_for(false, 1024));
}

#[test]
fn qwen_prefill_q3_ffn_is_seq_gated() {
    assert!(should_qwen_prefill_q3_ffn_for(true, 1024));
    assert!(should_qwen_prefill_q3_ffn_for(true, 2048));
    assert!(
        !should_qwen_prefill_q3_ffn_for(true, 512),
        "512-token FFN q3 overlay stays closed"
    );
    assert!(!should_qwen_prefill_q3_ffn_for(true, 1));
    assert!(!should_qwen_prefill_q3_ffn_for(false, 1024));
}

#[test]
fn qwen_prefill_contiguous_ffn_weights_is_seq_gated() {
    assert!(should_qwen_prefill_contiguous_ffn_weights_for(true, 1024));
    assert!(should_qwen_prefill_contiguous_ffn_weights_for(true, 2048));
    assert!(
        !should_qwen_prefill_contiguous_ffn_weights_for(true, 512),
        "512-token FFN weight contiguous stays closed"
    );
    assert!(!should_qwen_prefill_contiguous_ffn_weights_for(true, 1));
    assert!(!should_qwen_prefill_contiguous_ffn_weights_for(false, 1024));
}

#[test]
fn qwen_prefill_async_gate_up_is_seq_gated() {
    assert!(should_qwen_prefill_async_gate_up_for(true, 1024));
    assert!(should_qwen_prefill_async_gate_up_for(true, 2048));
    assert!(
        !should_qwen_prefill_async_gate_up_for(true, 512),
        "512-token async gate/up stays closed"
    );
    assert!(!should_qwen_prefill_async_gate_up_for(true, 1));
    assert!(!should_qwen_prefill_async_gate_up_for(false, 1024));
}

#[test]
fn qwen_prefill_ffn_f32_input_is_seq_gated() {
    assert!(should_qwen_prefill_ffn_f32_input_for(true, 1024));
    assert!(should_qwen_prefill_ffn_f32_input_for(true, 2048));
    assert!(
        !should_qwen_prefill_ffn_f32_input_for(true, 512),
        "512-token FFN f32 input stays closed"
    );
    assert!(!should_qwen_prefill_ffn_f32_input_for(true, 1));
    assert!(!should_qwen_prefill_ffn_f32_input_for(false, 1024));
}

#[test]
fn qwen_prefill_eval_ffn_input_is_seq_gated() {
    assert!(should_qwen_prefill_eval_ffn_input_for(true, 1024));
    assert!(should_qwen_prefill_eval_ffn_input_for(true, 2048));
    assert!(
        !should_qwen_prefill_eval_ffn_input_for(true, 512),
        "512-token FFN input eval stays closed"
    );
    assert!(!should_qwen_prefill_eval_ffn_input_for(true, 1));
    assert!(!should_qwen_prefill_eval_ffn_input_for(false, 1024));
}

#[test]
fn qwen_prefill_eval_la_input_is_seq_gated() {
    assert!(should_qwen_prefill_eval_la_input_for(true, 1024));
    assert!(should_qwen_prefill_eval_la_input_for(true, 2048));
    assert!(
        !should_qwen_prefill_eval_la_input_for(true, 512),
        "512-token LA input eval stays closed"
    );
    assert!(!should_qwen_prefill_eval_la_input_for(true, 1));
    assert!(!should_qwen_prefill_eval_la_input_for(false, 1024));
}

#[test]
fn qwen_prefill_async_la_outputs_is_seq_gated() {
    assert!(should_qwen_prefill_async_la_outputs_for(true, 1024));
    assert!(should_qwen_prefill_async_la_outputs_for(true, 2048));
    assert!(
        !should_qwen_prefill_async_la_outputs_for(true, 512),
        "512-token async LA outputs stays closed"
    );
    assert!(!should_qwen_prefill_async_la_outputs_for(true, 1));
    assert!(!should_qwen_prefill_async_la_outputs_for(false, 1024));
}

#[test]
fn qwen_prefill_async_packed_gate_up_is_seq_gated() {
    assert!(should_qwen_prefill_async_packed_gate_up_for(true, 1024));
    assert!(should_qwen_prefill_async_packed_gate_up_for(true, 2048));
    assert!(
        !should_qwen_prefill_async_packed_gate_up_for(true, 512),
        "512-token async packed gate/up stays closed"
    );
    assert!(!should_qwen_prefill_async_packed_gate_up_for(true, 1));
    assert!(!should_qwen_prefill_async_packed_gate_up_for(false, 1024));
}

#[test]
fn qwen_prefill_contiguous_la_weights_is_seq_gated() {
    assert!(should_qwen_prefill_contiguous_la_weights_for(true, 1024));
    assert!(should_qwen_prefill_contiguous_la_weights_for(true, 2048));
    assert!(
        !should_qwen_prefill_contiguous_la_weights_for(true, 512),
        "512-token LA weight contiguous stays closed"
    );
    assert!(!should_qwen_prefill_contiguous_la_weights_for(true, 1));
    assert!(!should_qwen_prefill_contiguous_la_weights_for(false, 1024));
}

#[test]
fn qwen_prefill_eval_attn_input_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_eval_attn_input_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_eval_attn_input_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(
        !should_qwen_prefill_eval_attn_input_for(true, "qwen3_5", 512),
        "512-token attn input eval stays closed"
    );
    assert!(!should_qwen_prefill_eval_attn_input_for(true, "qwen3_5", 1));
    assert!(!should_qwen_prefill_eval_attn_input_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_eval_attn_input_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_prefill_eval_ffn_hidden_is_seq_gated() {
    assert!(should_qwen_prefill_eval_ffn_hidden_for(true, 1024));
    assert!(should_qwen_prefill_eval_ffn_hidden_for(true, 2048));
    assert!(
        !should_qwen_prefill_eval_ffn_hidden_for(true, 512),
        "512-token FFN hidden eval stays closed"
    );
    assert!(!should_qwen_prefill_eval_ffn_hidden_for(true, 1));
    assert!(!should_qwen_prefill_eval_ffn_hidden_for(false, 1024));
}

#[test]
fn qwen_prefill_contiguous_attn_weights_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_contiguous_attn_weights_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_contiguous_attn_weights_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(
        !should_qwen_prefill_contiguous_attn_weights_for(true, "qwen3_5", 512),
        "512-token attn weight contiguous stays closed"
    );
    assert!(!should_qwen_prefill_contiguous_attn_weights_for(
        true, "qwen3_5", 1
    ));
    assert!(!should_qwen_prefill_contiguous_attn_weights_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_contiguous_attn_weights_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_prefill_skip_unused_la_out_is_seq_family_and_skip_gated() {
    assert!(should_qwen_prefill_skip_unused_la_out_for(
        true, "qwen3_5", true, 1024
    ));
    assert!(should_qwen_prefill_skip_unused_la_out_for(
        true,
        "qwen3_next",
        true,
        2048
    ));
    assert!(
        !should_qwen_prefill_skip_unused_la_out_for(true, "qwen3_5", true, 512),
        "512-token unused LA out skip stays closed"
    );
    assert!(!should_qwen_prefill_skip_unused_la_out_for(
        true, "qwen3_5", false, 1024
    ));
    assert!(!should_qwen_prefill_skip_unused_la_out_for(
        true, "gemma4", true, 1024
    ));
    assert!(!should_qwen_prefill_skip_unused_la_out_for(
        false, "qwen3_5", true, 1024
    ));
}

#[test]
fn qwen_prefill_async_down_is_seq_gated() {
    assert!(should_qwen_prefill_async_down_for(true, 1024));
    assert!(should_qwen_prefill_async_down_for(true, 2048));
    assert!(
        !should_qwen_prefill_async_down_for(true, 512),
        "512-token async down stays closed"
    );
    assert!(!should_qwen_prefill_async_down_for(true, 1));
    assert!(!should_qwen_prefill_async_down_for(false, 1024));
}

#[test]
fn qwen_prefill_last_query_q_proj_is_seq_family_and_last_only_gated() {
    assert!(should_qwen_prefill_last_query_q_proj_for(
        true, "qwen3_5", true, 1024
    ));
    assert!(should_qwen_prefill_last_query_q_proj_for(
        true,
        "qwen3_next",
        true,
        2048
    ));
    assert!(!should_qwen_prefill_last_query_q_proj_for(
        true, "qwen3_5", true, 512
    ));
    assert!(!should_qwen_prefill_last_query_q_proj_for(
        true, "qwen3_5", false, 1024
    ));
    assert!(!should_qwen_prefill_last_query_q_proj_for(
        true, "gemma4", true, 1024
    ));
    assert!(!should_qwen_prefill_last_query_q_proj_for(
        false, "qwen3_5", true, 1024
    ));
    assert!(
        should_qwen_prefill_last_query_sdpa_for(
            should_qwen_prefill_last_query_q_proj_for(true, "qwen3_5", true, 1024),
            "qwen3_5",
            true,
            1024,
        ),
        "last-token Q is already S=1; SDPA length must follow Q"
    );
}

#[test]
fn qwen_prefill_skip_unused_qk_norm_is_seq_family_and_last_only_gated() {
    assert!(should_qwen_prefill_skip_unused_qk_norm_for(
        true, "qwen3_5", true, 1024
    ));
    assert!(should_qwen_prefill_skip_unused_qk_norm_for(
        true,
        "qwen3_next",
        true,
        2048
    ));
    assert!(!should_qwen_prefill_skip_unused_qk_norm_for(
        true, "qwen3_5", true, 512
    ));
    assert!(!should_qwen_prefill_skip_unused_qk_norm_for(
        true, "qwen3_5", false, 1024
    ));
    assert!(!should_qwen_prefill_skip_unused_qk_norm_for(
        true, "gemma4", true, 1024
    ));
    assert!(!should_qwen_prefill_skip_unused_qk_norm_for(
        false, "qwen3_5", true, 1024
    ));
    assert!(
        should_qwen_prefill_last_query_sdpa_for(
            should_qwen_prefill_skip_unused_qk_norm_for(true, "qwen3_5", true, 1024),
            "qwen3_5",
            true,
            1024,
        ),
        "prefix QK-norm skip leaves S=1 Q; SDPA length must follow Q"
    );
}

#[test]
fn qwen_prefill_last_query_sdpa_is_seq_family_and_last_only_gated() {
    assert!(should_qwen_prefill_last_query_sdpa_for(
        true, "qwen3_5", true, 1024
    ));
    assert!(should_qwen_prefill_last_query_sdpa_for(
        true,
        "qwen3_next",
        true,
        2048
    ));
    assert!(
        !should_qwen_prefill_last_query_sdpa_for(true, "qwen3_5", true, 512),
        "512-token last-query SDPA stays closed"
    );
    assert!(!should_qwen_prefill_last_query_sdpa_for(
        true, "qwen3_5", false, 1024
    ));
    assert!(!should_qwen_prefill_last_query_sdpa_for(
        true, "gemma4", true, 1024
    ));
    assert!(!should_qwen_prefill_last_query_sdpa_for(
        false, "qwen3_5", true, 1024
    ));
}

#[test]
fn qwen_prefill_last_token_o_proj_is_seq_family_and_last_only_gated() {
    assert!(should_qwen_prefill_last_token_o_proj_for(
        true, "qwen3_5", true, 1024
    ));
    assert!(should_qwen_prefill_last_token_o_proj_for(
        true,
        "qwen3_next",
        true,
        2048
    ));
    assert!(
        !should_qwen_prefill_last_token_o_proj_for(true, "qwen3_5", true, 512),
        "512-token last-token o_proj stays closed"
    );
    assert!(!should_qwen_prefill_last_token_o_proj_for(
        true, "qwen3_5", false, 1024
    ));
    assert!(!should_qwen_prefill_last_token_o_proj_for(
        true, "gemma4", true, 1024
    ));
    assert!(!should_qwen_prefill_last_token_o_proj_for(
        false, "qwen3_5", true, 1024
    ));
}

#[test]
fn qwen_prefill_reuse_rope_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_reuse_rope_for(true, "qwen3_5", 1024));
    assert!(should_qwen_prefill_reuse_rope_for(true, "qwen3_next", 2048));
    assert!(
        !should_qwen_prefill_reuse_rope_for(true, "qwen3_5", 512),
        "512-token rope reuse stays closed"
    );
    assert!(!should_qwen_prefill_reuse_rope_for(true, "qwen3_5", 1));
    assert!(!should_qwen_prefill_reuse_rope_for(true, "gemma4", 1024));
    assert!(!should_qwen_prefill_reuse_rope_for(false, "qwen3_5", 1024));
}

#[test]
fn qwen_prefill_async_sdpa_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_async_sdpa_for(true, "qwen3_5", 1024));
    assert!(should_qwen_prefill_async_sdpa_for(true, "qwen3_next", 2048));
    assert!(
        !should_qwen_prefill_async_sdpa_for(true, "qwen3_5", 512),
        "512-token async SDPA stays closed"
    );
    assert!(!should_qwen_prefill_async_sdpa_for(true, "qwen3_5", 1));
    assert!(!should_qwen_prefill_async_sdpa_for(true, "gemma4", 1024));
    assert!(!should_qwen_prefill_async_sdpa_for(false, "qwen3_5", 1024));
}

#[test]
fn qwen_prefill_async_gd_is_seq_gated() {
    assert!(should_qwen_prefill_async_gd_for(true, 1024));
    assert!(should_qwen_prefill_async_gd_for(true, 2048));
    assert!(
        !should_qwen_prefill_async_gd_for(true, 512),
        "512-token async GD stays closed"
    );
    assert!(!should_qwen_prefill_async_gd_for(true, 1));
    assert!(!should_qwen_prefill_async_gd_for(false, 1024));
}

#[test]
fn qwen_prefill_eval_gd_is_seq_gated() {
    assert!(should_qwen_prefill_eval_gd_for(true, 1024));
    assert!(should_qwen_prefill_eval_gd_for(true, 2048));
    assert!(
        !should_qwen_prefill_eval_gd_for(true, 512),
        "512-token eval GD stays closed"
    );
    assert!(!should_qwen_prefill_eval_gd_for(true, 1));
    assert!(!should_qwen_prefill_eval_gd_for(false, 1024));
}

#[test]
fn qwen_prefill_contiguous_gd_is_seq_gated() {
    assert!(should_qwen_prefill_contiguous_gd_for(true, 1024));
    assert!(should_qwen_prefill_contiguous_gd_for(true, 2048));
    assert!(
        !should_qwen_prefill_contiguous_gd_for(true, 512),
        "512-token contiguous GD stays closed"
    );
    assert!(!should_qwen_prefill_contiguous_gd_for(true, 1));
    assert!(!should_qwen_prefill_contiguous_gd_for(false, 1024));
}

#[test]
fn qwen_prefill_split_packed_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_split_packed_for(true, "qwen3_5", 1024));
    assert!(should_qwen_prefill_split_packed_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(
        !should_qwen_prefill_split_packed_for(true, "qwen3_5", 512),
        "512-token split-packed stays closed"
    );
    assert!(!should_qwen_prefill_split_packed_for(true, "qwen3_5", 1));
    assert!(!should_qwen_prefill_split_packed_for(true, "gemma4", 1024));
    assert!(!should_qwen_prefill_split_packed_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_prefill_dequant_dense_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_dequant_dense_for(true, "qwen3_5", 1024));
    assert!(should_qwen_prefill_dequant_dense_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(
        !should_qwen_prefill_dequant_dense_for(true, "qwen3_5", 512),
        "512-token dequant-dense stays closed"
    );
    assert!(!should_qwen_prefill_dequant_dense_for(true, "qwen3_5", 1));
    assert!(!should_qwen_prefill_dequant_dense_for(true, "gemma4", 1024));
    assert!(!should_qwen_prefill_dequant_dense_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_la_norm_qkvz_fuse_is_seq_and_family_gated() {
    assert!(should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_5", 1024));
    assert!(should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_next", 2048));
    assert!(
        !should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_5", 512),
        "512-token LA norm fuse stays closed"
    );
    assert!(!should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_5", 1));
    assert!(!should_qwen_la_norm_qkvz_fuse_for(true, "gemma4", 1024));
    assert!(!should_qwen_la_norm_qkvz_fuse_for(false, "qwen3_5", 1024));
}

#[test]
fn qwen_prefill_skip_bf16_astype_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_skip_bf16_astype_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_skip_bf16_astype_for(
        true,
        "qwen3_next",
        2
    ));
    assert!(!should_qwen_prefill_skip_bf16_astype_for(
        true, "qwen3_5", 1
    ));
    assert!(!should_qwen_prefill_skip_bf16_astype_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_skip_bf16_astype_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_prefill_flat_qmm_is_seq_and_rank_gated() {
    assert!(should_qwen_prefill_flat_qmm_for(true, 1024, 3));
    assert!(should_qwen_prefill_flat_qmm_for(true, 2048, 3));
    assert!(!should_qwen_prefill_flat_qmm_for(true, 512, 3));
    assert!(!should_qwen_prefill_flat_qmm_for(true, 1024, 2));
    assert!(!should_qwen_prefill_flat_qmm_for(true, 1, 3));
    assert!(!should_qwen_prefill_flat_qmm_for(false, 1024, 3));
}

#[test]
fn qwen_prefill_tile_qmm_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_tile_qmm_for(true, "qwen3_5", 1024));
    assert!(should_qwen_prefill_tile_qmm_for(true, "qwen3_next", 2048));
    assert!(!should_qwen_prefill_tile_qmm_for(true, "qwen3_5", 512));
    assert!(!should_qwen_prefill_tile_qmm_for(true, "qwen3_5", 1));
    assert!(!should_qwen_prefill_tile_qmm_for(true, "gemma4", 1024));
    assert!(!should_qwen_prefill_tile_qmm_for(false, "qwen3_5", 1024));
}

#[test]
fn qwen_prefill_dual_affine_qmm_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_dual_affine_qmm_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_dual_affine_qmm_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(!should_qwen_prefill_dual_affine_qmm_for(
        true, "qwen3_5", 512
    ));
    assert!(!should_qwen_prefill_dual_affine_qmm_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_dual_affine_qmm_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_prefill_skip_unused_embed_clip_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_skip_unused_embed_clip_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_skip_unused_embed_clip_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(!should_qwen_prefill_skip_unused_embed_clip_for(
        true, "qwen3_5", 512
    ));
    assert!(!should_qwen_prefill_skip_unused_embed_clip_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_skip_unused_embed_clip_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_prefill_skip_unused_f32_sdpa_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_skip_unused_f32_sdpa_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_skip_unused_f32_sdpa_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(
        should_qwen_prefill_skip_unused_f32_sdpa_for(true, "qwen3_5", 128),
        "contract p128 prefill must skip the f32 upcast"
    );
    assert!(should_qwen_prefill_skip_unused_f32_sdpa_for(
        true, "qwen3_5", 512
    ));
    assert!(
        !should_qwen_prefill_skip_unused_f32_sdpa_for(true, "qwen3_5", 8),
        "short MTP verify must keep f32 SDPA"
    );
    assert!(!should_qwen_prefill_skip_unused_f32_sdpa_for(
        true, "qwen3_5", 127
    ));
    assert!(
        should_qwen_prefill_skip_unused_f32_sdpa_for(true, "qwen3_vl_moe", 128),
        "VL-MoE text prefill shares the qwen full-attention graphs"
    );
    assert!(should_qwen_prefill_skip_unused_f32_sdpa_for(
        true, "qwen3_vl", 128
    ));
    assert!(!should_qwen_prefill_skip_unused_f32_sdpa_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_skip_unused_f32_sdpa_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn gemma4_prefill_skip_unused_f32_sdpa_is_seq_and_family_gated() {
    assert!(should_gemma4_prefill_skip_unused_f32_sdpa_for(
        true, "gemma4", 128
    ));
    assert!(should_gemma4_prefill_skip_unused_f32_sdpa_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(should_gemma4_prefill_skip_unused_f32_sdpa_for(
        true, "gemma4", 512
    ));
    assert!(
        !should_gemma4_prefill_skip_unused_f32_sdpa_for(true, "gemma4", 8),
        "short MTP verify must keep f32 SDPA"
    );
    assert!(!should_gemma4_prefill_skip_unused_f32_sdpa_for(
        true, "gemma4", 1
    ));
    assert!(!should_gemma4_prefill_skip_unused_f32_sdpa_for(
        true, "qwen3_5", 128
    ));
    assert!(!should_gemma4_prefill_skip_unused_f32_sdpa_for(
        false, "gemma4", 128
    ));
}

#[test]
fn gemma4_prefill_skip_unused_embed_clip_is_seq_and_family_gated() {
    assert!(should_gemma4_prefill_skip_unused_embed_clip_for(
        true, "gemma4", 128
    ));
    assert!(should_gemma4_prefill_skip_unused_embed_clip_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(should_gemma4_prefill_skip_unused_embed_clip_for(
        true, "gemma4", 2048
    ));
    assert!(
        !should_gemma4_prefill_skip_unused_embed_clip_for(true, "gemma4", 8),
        "short MTP verify must keep the embed clip"
    );
    assert!(!should_gemma4_prefill_skip_unused_embed_clip_for(
        true, "gemma4", 1
    ));
    assert!(!should_gemma4_prefill_skip_unused_embed_clip_for(
        true, "qwen3_5", 128
    ));
    assert!(!should_gemma4_prefill_skip_unused_embed_clip_for(
        false, "gemma4", 128
    ));
}

#[test]
fn gemma4_prefill_bf16_embed_is_seq_and_family_gated() {
    assert!(should_gemma4_prefill_bf16_embed_for(true, "gemma4", 128));
    assert!(should_gemma4_prefill_bf16_embed_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(should_gemma4_prefill_bf16_embed_for(true, "gemma4", 2048));
    assert!(
        !should_gemma4_prefill_bf16_embed_for(true, "gemma4", 8),
        "short MTP verify must keep f32 embed dequant"
    );
    assert!(!should_gemma4_prefill_bf16_embed_for(true, "gemma4", 1));
    assert!(!should_gemma4_prefill_bf16_embed_for(true, "qwen3_5", 128));
    assert!(!should_gemma4_prefill_bf16_embed_for(false, "gemma4", 128));
}

#[test]
fn gemma4_prefill_skip_unused_last_residual_is_seq_last_layer_and_family_gated() {
    assert!(
        should_gemma4_prefill_skip_unused_last_residual_for(true, "gemma4", true, 128),
        "shipped skip-unused-last-residual must accept contract p128 last layer"
    );
    assert!(should_gemma4_prefill_skip_unused_last_residual_for(
        true,
        "gemma4_unified",
        true,
        128
    ));
    assert!(should_gemma4_prefill_skip_unused_last_residual_for(
        true, "gemma4", true, 2048
    ));
    assert!(
        !should_gemma4_prefill_skip_unused_last_residual_for(true, "gemma4", false, 128),
        "non-final layers keep full-seq add_rms"
    );
    assert!(
        !should_gemma4_prefill_skip_unused_last_residual_for(true, "gemma4", true, 8),
        "short MTP verify keeps add-then-slice"
    );
    assert!(!should_gemma4_prefill_skip_unused_last_residual_for(
        true, "gemma4", true, 1
    ));
    assert!(!should_gemma4_prefill_skip_unused_last_residual_for(
        true, "qwen3_5", true, 128
    ));
    assert!(!should_gemma4_prefill_skip_unused_last_residual_for(
        false, "gemma4", true, 128
    ));
}

#[test]
fn gemma4_prefill_skip_unused_last_ffn_packed_is_seq_last_layer_and_family_gated() {
    assert!(
        should_gemma4_prefill_skip_unused_last_ffn_packed_for(true, "gemma4", true, 128),
        "shipped skip-unused-last-ffn-packed must accept contract p128 last layer"
    );
    assert!(should_gemma4_prefill_skip_unused_last_ffn_packed_for(
        true,
        "gemma4_unified",
        true,
        128
    ));
    assert!(should_gemma4_prefill_skip_unused_last_ffn_packed_for(
        true, "gemma4", true, 2048
    ));
    assert!(
        !should_gemma4_prefill_skip_unused_last_ffn_packed_for(true, "gemma4", false, 128),
        "non-final layers keep packed/split prefill policy"
    );
    assert!(
        !should_gemma4_prefill_skip_unused_last_ffn_packed_for(true, "gemma4", true, 8),
        "short MTP verify keeps packed last-layer FFN"
    );
    assert!(!should_gemma4_prefill_skip_unused_last_ffn_packed_for(
        true, "gemma4", true, 1
    ));
    assert!(!should_gemma4_prefill_skip_unused_last_ffn_packed_for(
        true, "qwen3_5", true, 128
    ));
    assert!(!should_gemma4_prefill_skip_unused_last_ffn_packed_for(
        false, "gemma4", true, 128
    ));
}

#[test]
fn gemma4_prefill_skip_unused_layer_masks_is_seq_window_and_family_gated() {
    assert!(should_gemma4_prefill_skip_unused_layer_masks_for(
        true,
        "gemma4",
        128,
        128,
        Some(1024),
        0
    ));
    assert!(should_gemma4_prefill_skip_unused_layer_masks_for(
        true,
        "gemma4_unified",
        512,
        512,
        Some(1024),
        0
    ));
    assert!(
        should_gemma4_prefill_skip_unused_layer_masks_for(true, "gemma4", 128, 128, None, 0),
        "full-attn offset-0 prefill is already maskless"
    );
    assert!(
        !should_gemma4_prefill_skip_unused_layer_masks_for(
            true,
            "gemma4",
            2048,
            2048,
            Some(1024),
            0
        ),
        "p2048 exceeds the 1024-token window and must keep the hoist"
    );
    assert!(
        !should_gemma4_prefill_skip_unused_layer_masks_for(true, "gemma4", 8, 8, Some(1024), 0),
        "short MTP verify must keep the hoist"
    );
    assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
        true,
        "gemma4",
        1,
        1,
        Some(1024),
        0
    ));
    assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
        true,
        "gemma4",
        128,
        256,
        Some(1024),
        0
    ));
    assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
        true,
        "gemma4",
        128,
        128,
        Some(1024),
        4
    ));
    assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
        true,
        "qwen3_5",
        128,
        128,
        Some(1024),
        0
    ));
    assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
        false,
        "gemma4",
        128,
        128,
        Some(1024),
        0
    ));
}

#[test]
fn gemma4_prefill_pipeline_hint_p128_is_seq_layer_and_family_gated() {
    assert!(
        should_gemma4_prefill_pipeline_hint_p128_for(true, "gemma4", 128, 0, 48),
        "shipped p128 pipeline hint must fire after non-final layers"
    );
    assert!(should_gemma4_prefill_pipeline_hint_p128_for(
        true,
        "gemma4_unified",
        128,
        46,
        48
    ));
    assert!(
        !should_gemma4_prefill_pipeline_hint_p128_for(true, "gemma4", 128, 47, 48),
        "final layer stays lazy so logits eval owns the last barrier"
    );
    assert!(
        !should_gemma4_prefill_pipeline_hint_p128_for(true, "gemma4", 512, 0, 48),
        "p512 keeps full-graph fusion"
    );
    assert!(!should_gemma4_prefill_pipeline_hint_p128_for(
        true, "gemma4", 2048, 0, 48
    ));
    assert!(
        !should_gemma4_prefill_pipeline_hint_p128_for(true, "gemma4", 8, 0, 48),
        "short MTP verify stays lazy"
    );
    assert!(!should_gemma4_prefill_pipeline_hint_p128_for(
        true, "gemma4", 1, 0, 48
    ));
    assert!(!should_gemma4_prefill_pipeline_hint_p128_for(
        true, "qwen3_5", 128, 0, 48
    ));
    assert!(!should_gemma4_prefill_pipeline_hint_p128_for(
        false, "gemma4", 128, 0, 48
    ));
    assert!(
        !pipeline_hint_should_fire(0, 48),
        "global AX_MLX_PIPELINE_GRANULARITY stays off; only the Gemma p128 predicate fires"
    );
}

#[test]
fn gemma4_prefill_last_query_p128_is_seq_last_layer_and_family_gated() {
    assert!(
        should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 128),
        "shipped last-query must accept contract p128 last layer"
    );
    assert!(should_gemma4_prefill_last_query_p128_for(
        true,
        "gemma4_unified",
        true,
        128
    ));
    assert!(
        !should_gemma4_prefill_last_query_p128_for(true, "gemma4", false, 128),
        "non-final layers must keep full-seq fused attention"
    );
    assert!(
        !should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 512),
        "p512 last layer stays on fused full-seq"
    );
    assert!(!should_gemma4_prefill_last_query_p128_for(
        true, "gemma4", true, 2048
    ));
    assert!(
        !should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 8),
        "short MTP verify keeps full-seq last-layer attention"
    );
    assert!(!should_gemma4_prefill_last_query_p128_for(
        true, "gemma4", true, 1
    ));
    assert!(!should_gemma4_prefill_last_query_p128_for(
        true, "qwen3_5", true, 128
    ));
    assert!(!should_gemma4_prefill_last_query_p128_for(
        false, "gemma4", true, 128
    ));
}

#[test]
fn qwen_prefill_skip_unused_swiglu_compile_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_skip_unused_swiglu_compile_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_skip_unused_swiglu_compile_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(!should_qwen_prefill_skip_unused_swiglu_compile_for(
        true, "qwen3_5", 512
    ));
    assert!(!should_qwen_prefill_skip_unused_swiglu_compile_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_skip_unused_swiglu_compile_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn nax_native_offset_causal_family_uses_explicit_allowlist() {
    let allowed = [
        "qwen3",
        "qwen3_5",
        "qwen3_next",
        "qwen3_vl",
        "qwen3_vl_moe",
        "llama3",
        "llama4",
        "mistral3",
        "mixtral",
        "glm4_moe_lite",
        "gpt_oss",
        "muse_glimmer",
        "minimax_m3",
        "minicpmv4_6",
        "nemotron_h",
    ];
    for family in allowed {
        assert!(
            nax_native_offset_causal_family(family),
            "expected {family} to be allowlisted"
        );
        assert!(
            nax_native_offset_causal_family(&family.to_ascii_uppercase()),
            "expected {family} matching to be case-insensitive"
        );
    }

    let denied = [
        "gemma4",
        "gemma4_vl",
        "gemma4_unified",
        "gemma4_assistant",
        "embeddinggemma",
        "diffusion_gemma",
        "gemma_future",
        "unlimited_ocr",
        "whisper",
        "nemotron_embed",
        "deepseek_v3",
        "deepseek_v4",
    ];
    for family in denied {
        assert!(
            !nax_native_offset_causal_family(family),
            "expected {family} to be denied"
        );
        assert!(
            !nax_native_offset_causal_family(&family.to_ascii_uppercase()),
            "expected {family} denial to be case-insensitive"
        );
    }
}

#[test]
fn qwen_prefill_native_offset_causal_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_native_offset_causal_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_native_offset_causal_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(!should_qwen_prefill_native_offset_causal_for(
        true, "qwen3_5", 512
    ));
    assert!(!should_qwen_prefill_native_offset_causal_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_native_offset_causal_for(
        false, "qwen3_5", 1024
    ));
    assert!(should_prefill_native_offset_causal_for(
        true, false, "qwen3_5", 1024
    ));
    assert!(!should_prefill_native_offset_causal_for(
        true, false, "llama3", 1024
    ));
}

#[test]
fn nax_attention_is_off_switch_only() {
    assert!(nax_attention_enabled_for(true, true));
    assert!(!nax_attention_enabled_for(false, true));
    assert!(!nax_attention_enabled_for(true, false));
    assert!(!nax_attention_enabled_for(false, false));
}

#[test]
fn nax_attention_arms_allowlisted_native_offset_causal_on_m5() {
    let _hw = crate::hardware::override_hardware(crate::hardware::HardwareCapabilities::m5_na());
    assert!(nax_attention_enabled_for(
        true,
        crate::hardware::neural_accelerator_active()
    ));
    for family in ["llama3", "mistral3", "qwen3", "qwen3_vl"] {
        assert!(
            should_prefill_native_offset_causal(family, 1024),
            "M5 NAX should arm native offset causal for {family}"
        );
    }
    assert!(should_prefill_native_offset_causal("qwen3_5", 1024));
    assert!(should_prefill_native_offset_causal("qwen3_next", 2048));
    assert!(!should_prefill_native_offset_causal("llama3", 512));
    assert!(!should_prefill_native_offset_causal("gemma4", 1024));
}

#[test]
fn nax_attention_does_not_arm_on_m4() {
    let _hw = crate::hardware::override_hardware(crate::hardware::HardwareCapabilities::m4());
    assert!(!nax_attention_enabled());
    assert!(!should_prefill_native_offset_causal("llama3", 1024));
    if !qwen_prefill_native_offset_causal_enabled() {
        assert!(!should_prefill_native_offset_causal("qwen3_5", 1024));
    }
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn live_nax_attention_follows_detected_hardware() {
    let active = crate::hardware::neural_accelerator_active();
    eprintln!(
        "live nax_attention: enabled={} allowed={} neural_accelerator_active={}",
        nax_attention_enabled(),
        nax_attention_allowed(),
        active
    );
    assert_eq!(
        nax_attention_enabled(),
        nax_attention_allowed() && active,
        "kill-switch AND hardware must both be true to arm NAX attention"
    );
    if nax_attention_enabled() {
        assert!(should_prefill_native_offset_causal("qwen3_5", 1024));
        assert!(should_prefill_native_offset_causal("qwen3_next", 2048));
        assert!(should_prefill_native_offset_causal("llama3", 1024));
        assert!(should_prefill_native_offset_causal("qwen3_vl", 1024));
        assert!(!should_prefill_native_offset_causal("qwen3_5", 512));
        assert!(!should_prefill_native_offset_causal("gemma4", 1024));
    }
}

#[test]
fn qwen_prefill_bf16_embed_dequant_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_bf16_embed_dequant_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_bf16_embed_dequant_for(
        true,
        "qwen3_next",
        2048
    ));
    assert!(!should_qwen_prefill_bf16_embed_dequant_for(
        true, "qwen3_5", 512
    ));
    assert!(!should_qwen_prefill_bf16_embed_dequant_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_bf16_embed_dequant_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_prefill_async_embed_is_seq_and_family_gated() {
    assert!(should_qwen_prefill_async_embed_for(true, "qwen3_5", 1024));
    assert!(should_qwen_prefill_async_embed_for(
        true,
        "qwen3_next",
        1024
    ));
    assert!(!should_qwen_prefill_async_embed_for(true, "qwen3_5", 512));
    assert!(!should_qwen_prefill_async_embed_for(true, "qwen3_5", 1));
    assert!(!should_qwen_prefill_async_embed_for(true, "gemma4", 1024));
    assert!(!should_qwen_prefill_async_embed_for(false, "qwen3_5", 1024));
}

#[test]
fn qwen_packed_ffn_prefill_compile_is_leading_gated() {
    assert!(should_qwen_packed_ffn_prefill_compile_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_packed_ffn_prefill_compile_for(
        true, "QWEN3_5", 2048
    ));
    assert!(
        !should_qwen_packed_ffn_prefill_compile_for(true, "qwen3_5", 512),
        "512-token packed compile stays closed"
    );
    assert!(!should_qwen_packed_ffn_prefill_compile_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_packed_ffn_prefill_compile_for(
        false, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_compiled_qk_norm_rope_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_COMPILED_QK_NORM_ROPE_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_COMPILED_QK_NORM_ROPE_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_COMPILED_QK_NORM_ROPE_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_direct_cpp_qk_norm_rope_uses_default_on_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_CPP_QK_NORM_ROPE_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_CPP_QK_NORM_ROPE_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_QWEN_DIRECT_CPP_QK_NORM_ROPE_ENABLED",
        "1"
    ));
}

#[test]
fn gemma_direct_cpp_qk_norm_rope_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_GEMMA_DIRECT_CPP_QK_NORM_ROPE_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_GEMMA_DIRECT_CPP_QK_NORM_ROPE_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_GEMMA_DIRECT_CPP_QK_NORM_ROPE_ENABLED",
        "1"
    ));
}

#[test]
fn gemma_dual_gate_up_metal_uses_opt_in_contract() {
    // Pure-wall A/B on mbp-m5 measured ~8.5× regression when default-on;
    // production remains opt-in only.
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_GEMMA_DUAL_GATE_UP_METAL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_GEMMA_DUAL_GATE_UP_METAL_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_GEMMA_DUAL_GATE_UP_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn o_proj_qmatmul_rms_norm_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_O_PROJ_QMATMUL_RMS_NORM_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_O_PROJ_QMATMUL_RMS_NORM_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_O_PROJ_QMATMUL_RMS_NORM_ENABLED",
        "1"
    ));
}

#[test]
fn attn_norm_qkv_fuse_uses_opt_in_contract() {
    assert!(!parse_bool_env("AX_FASTPATH_TEST_ATTN_NORM_QKV_FUSE_UNSET"));
    assert!(!probe("AX_FASTPATH_TEST_ATTN_NORM_QKV_FUSE_DISABLED", "0"));
    assert!(probe("AX_FASTPATH_TEST_ATTN_NORM_QKV_FUSE_ENABLED", "1"));
}

#[test]
fn qwen_attn_norm_qkv_fuse_is_family_scoped_and_opt_in() {
    assert!(should_attn_norm_qkv_fuse_for(
        true, false, false, "qwen3_5", 128
    ));
    assert!(should_attn_norm_qkv_fuse_for(
        true, false, false, "QWEN3_5", 2048
    ));
    assert!(
        !should_attn_norm_qkv_fuse_for(false, false, false, "qwen3_5", 128),
        "Qwen kill-switch must disable the fuse"
    );
    assert!(
        !should_attn_norm_qkv_fuse_for(true, false, false, "gemma4", 512),
        "Gemma p512 stays on the global default-OFF flag"
    );
    assert!(should_attn_norm_qkv_fuse_for(
        false, true, false, "gemma4", 512
    ));
    assert!(should_call_attn_norm_qkv_fuse(true, true, false, false));
    assert!(
        !should_call_attn_norm_qkv_fuse(true, true, false, true),
        "exact / moe-mt skip must keep standalone attn_norm"
    );
    assert!(!should_call_attn_norm_qkv_fuse(true, false, false, false));
    assert!(!should_call_attn_norm_qkv_fuse(true, true, true, false));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_ATTN_NORM_QKV_FUSE_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_ATTN_NORM_QKV_FUSE_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_ATTN_NORM_QKV_FUSE_ENABLED",
        "1"
    ));
}

#[test]
fn gemma4_attn_norm_qkv_fuse_p128_is_seq_and_family_gated() {
    assert!(should_gemma4_attn_norm_qkv_fuse_p128_for(
        true, "gemma4", 128
    ));
    assert!(should_gemma4_attn_norm_qkv_fuse_p128_for(
        true,
        "gemma4_unified",
        128
    ));
    assert!(should_attn_norm_qkv_fuse_for(
        false, false, true, "gemma4", 128
    ));
    assert!(
        !should_gemma4_attn_norm_qkv_fuse_p128_for(true, "gemma4", 512),
        "p512 must stay portable so the p128 fuse cannot regress longer cells"
    );
    assert!(!should_gemma4_attn_norm_qkv_fuse_p128_for(
        true, "gemma4", 2048
    ));
    assert!(!should_gemma4_attn_norm_qkv_fuse_p128_for(
        true, "gemma4", 1
    ));
    assert!(!should_gemma4_attn_norm_qkv_fuse_p128_for(
        true, "qwen3_5", 128
    ));
    assert!(!should_gemma4_attn_norm_qkv_fuse_p128_for(
        false, "gemma4", 128
    ));
    assert!(!should_attn_norm_qkv_fuse_for(
        false, false, true, "gemma4", 512
    ));
}

#[test]
fn native_offset_causal_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_NATIVE_OFFSET_CAUSAL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_NATIVE_OFFSET_CAUSAL_DISABLED",
        "0"
    ));
    assert!(probe("AX_FASTPATH_TEST_NATIVE_OFFSET_CAUSAL_ENABLED", "1"));
}

#[test]
fn dual_qmm_geglu_uses_opt_in_contract() {
    assert!(!parse_bool_env("AX_FASTPATH_TEST_DUAL_QMM_GEGLU_UNSET"));
    assert!(!probe("AX_FASTPATH_TEST_DUAL_QMM_GEGLU_DISABLED", "0"));
    assert!(probe("AX_FASTPATH_TEST_DUAL_QMM_GEGLU_ENABLED", "1"));
}

#[test]
fn cache_only_chunk_eval_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_CACHE_ONLY_CHUNK_EVAL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_CACHE_ONLY_CHUNK_EVAL_DISABLED",
        "0"
    ));
    assert!(probe("AX_FASTPATH_TEST_CACHE_ONLY_CHUNK_EVAL_ENABLED", "1"));
}

#[test]
fn cache_only_chunk_async_eval_only_for_non_final_under_both_flags() {
    // Both off / either off → never async.
    assert!(!cache_only_chunk_should_async_eval_for(false, false, false));
    assert!(!cache_only_chunk_should_async_eval_for(true, false, false));
    assert!(!cache_only_chunk_should_async_eval_for(false, true, false));
    // Final chunk always blocks even when both flags are on.
    assert!(!cache_only_chunk_should_async_eval_for(true, true, true));
    // Intermediate chunk under both flags → async.
    assert!(cache_only_chunk_should_async_eval_for(true, true, false));
}

#[test]
fn prefill_clear_cache_per_chunk_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_PREFILL_CLEAR_CACHE_PER_CHUNK_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_PREFILL_CLEAR_CACHE_PER_CHUNK_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_PREFILL_CLEAR_CACHE_PER_CHUNK_ENABLED",
        "1"
    ));
}

#[test]
fn parse_pipeline_granularity_matches_mlxcel_contract() {
    assert_eq!(parse_pipeline_granularity(""), PipelineGranularity::Off);
    assert_eq!(parse_pipeline_granularity("off"), PipelineGranularity::Off);
    assert_eq!(parse_pipeline_granularity("OFF"), PipelineGranularity::Off);
    assert_eq!(
        parse_pipeline_granularity("layer"),
        PipelineGranularity::PerLayer
    );
    assert_eq!(
        parse_pipeline_granularity("LAYER"),
        PipelineGranularity::PerLayer
    );
    assert_eq!(
        parse_pipeline_granularity("block:4"),
        PipelineGranularity::PerBlock(4)
    );
    // Prefix casing follows the eval-granularity parser.
    assert_eq!(
        parse_pipeline_granularity("bLoCk:2"),
        PipelineGranularity::PerBlock(2)
    );
    assert_eq!(
        parse_pipeline_granularity("block:1"),
        PipelineGranularity::PerBlock(1)
    );
    assert_eq!(
        parse_pipeline_granularity("block:0"),
        PipelineGranularity::PerBlock(1),
        "N=0 clamps to 1"
    );
    assert_eq!(
        parse_pipeline_granularity("block:xyz"),
        PipelineGranularity::PerBlock(4),
        "invalid N falls back to 4"
    );
    assert_eq!(
        parse_pipeline_granularity("garbage"),
        PipelineGranularity::Off
    );
}

#[test]
fn parse_pipeline_eval_granularity_is_strict_and_case_insensitive() {
    assert_eq!(
        parse_pipeline_eval_granularity(""),
        PipelineEvalGranularity::Off
    );
    assert_eq!(
        parse_pipeline_eval_granularity(" OFF "),
        PipelineEvalGranularity::Off
    );
    assert_eq!(
        parse_pipeline_eval_granularity("layer"),
        PipelineEvalGranularity::PerLayer
    );
    assert_eq!(
        parse_pipeline_eval_granularity("LAYER"),
        PipelineEvalGranularity::PerLayer
    );
    assert_eq!(
        parse_pipeline_eval_granularity(" sublayer "),
        PipelineEvalGranularity::Sublayer
    );
    assert_eq!(
        parse_pipeline_eval_granularity("SUBLAYER"),
        PipelineEvalGranularity::Sublayer
    );
    assert_eq!(
        parse_pipeline_eval_granularity("block:4"),
        PipelineEvalGranularity::PerBlock(4)
    );
    assert_eq!(
        parse_pipeline_eval_granularity(" BLOCK:1 "),
        PipelineEvalGranularity::PerBlock(1)
    );
    assert_eq!(
        parse_pipeline_eval_granularity("yield:16"),
        PipelineEvalGranularity::YieldMs(16)
    );
    assert_eq!(
        parse_pipeline_eval_granularity(" YIELD:8 "),
        PipelineEvalGranularity::YieldMs(8)
    );
    assert_eq!(
        parse_pipeline_eval_granularity("block:0"),
        PipelineEvalGranularity::Off
    );
    assert_eq!(
        parse_pipeline_eval_granularity("yield:0"),
        PipelineEvalGranularity::Off
    );
    assert_eq!(
        parse_pipeline_eval_granularity("block:xyz"),
        PipelineEvalGranularity::Off
    );
    assert_eq!(
        parse_pipeline_eval_granularity("yield:xyz"),
        PipelineEvalGranularity::Off
    );
    assert_eq!(
        parse_pipeline_eval_granularity("garbage"),
        PipelineEvalGranularity::Off
    );
}

#[test]
fn pipeline_eval_granularity_only_blocks_multi_token_non_final_layers() {
    use PipelineEvalGranularity::{Off, PerBlock, PerLayer, Sublayer, YieldMs};

    assert!(!pipeline_eval_should_fire_for(Off, 8, 0, 4));
    assert!(!pipeline_eval_should_fire_for(PerLayer, 1, 0, 4));
    assert!(pipeline_eval_should_fire_for(PerLayer, 8, 0, 4));
    assert!(pipeline_eval_should_fire_for(PerLayer, 8, 2, 4));
    assert!(!pipeline_eval_should_fire_for(PerLayer, 8, 3, 4));
    assert!(!pipeline_eval_should_fire_for(PerLayer, 8, 0, 0));
    assert!(pipeline_eval_should_fire_for(Sublayer, 8, 0, 4));
    assert!(!pipeline_eval_should_fire_for(Sublayer, 1, 0, 4));
    assert!(!pipeline_eval_should_fire_for(Sublayer, 8, 3, 4));

    assert!(!pipeline_eval_should_fire_for(PerBlock(2), 8, 0, 6));
    assert!(pipeline_eval_should_fire_for(PerBlock(2), 8, 1, 6));
    assert!(pipeline_eval_should_fire_for(PerBlock(2), 8, 3, 6));
    assert!(
        !pipeline_eval_should_fire_for(PerBlock(2), 8, 5, 6),
        "final layer remains exempt even when it closes a block"
    );
    // YieldMs pure path is wall-clock only; layer filters still apply via
    // pipeline_eval_yield_should_fire, not the layer-index matcher.
    assert!(!pipeline_eval_should_fire_for(YieldMs(16), 8, 0, 4));
}

#[test]
fn pipeline_eval_yield_predicate_is_wall_clock_and_fail_closed() {
    // First eligible boundary always fires.
    assert!(pipeline_eval_yield_should_fire(
        None,
        1_000_000_000,
        16,
        8,
        0,
        4
    ));
    // Within window: no fire.
    assert!(!pipeline_eval_yield_should_fire(
        Some(1_000_000_000),
        1_000_000_000 + 15_000_000,
        16,
        8,
        1,
        4
    ));
    // At/after window: fire.
    assert!(pipeline_eval_yield_should_fire(
        Some(1_000_000_000),
        1_000_000_000 + 16_000_000,
        16,
        8,
        1,
        4
    ));
    // Decode / final layer / zero ms never fire.
    assert!(!pipeline_eval_yield_should_fire(None, 1, 16, 1, 0, 4));
    assert!(!pipeline_eval_yield_should_fire(None, 1, 16, 8, 3, 4));
    assert!(!pipeline_eval_yield_should_fire(None, 1, 0, 8, 0, 4));
}

#[test]
fn parse_pipeline_eval_tail_layers_is_fail_closed() {
    assert_eq!(parse_pipeline_eval_tail_layers(""), 0);
    assert_eq!(parse_pipeline_eval_tail_layers("off"), 0);
    assert_eq!(parse_pipeline_eval_tail_layers("OFF"), 0);
    assert_eq!(parse_pipeline_eval_tail_layers("12"), 12);
    assert_eq!(parse_pipeline_eval_tail_layers(" 8 "), 8);
    assert_eq!(parse_pipeline_eval_tail_layers("0"), 0);
    assert_eq!(parse_pipeline_eval_tail_layers("xyz"), 0);
    assert_eq!(parse_pipeline_eval_tail_layers("-1"), 0);
}

#[test]
fn pipeline_eval_layer_in_tail_covers_last_n_before_final() {
    // total=40 layers (0..39); final=39 exempt; tail=8 → layers 31..38.
    assert!(!pipeline_eval_layer_in_tail(30, 40, 8));
    assert!(pipeline_eval_layer_in_tail(31, 40, 8));
    assert!(pipeline_eval_layer_in_tail(38, 40, 8));
    assert!(!pipeline_eval_layer_in_tail(39, 40, 8));
    // Off / tiny models.
    assert!(!pipeline_eval_layer_in_tail(0, 40, 0));
    assert!(!pipeline_eval_layer_in_tail(0, 1, 8));
    // Tail larger than eligible set: all non-final layers.
    assert!(pipeline_eval_layer_in_tail(0, 4, 100));
    assert!(pipeline_eval_layer_in_tail(2, 4, 100));
    assert!(!pipeline_eval_layer_in_tail(3, 4, 100));
}

#[test]
fn pipeline_sublayer_eval_is_limited_to_gemma4_multi_token_prefill() {
    use PipelineEvalGranularity::{Off, PerLayer, Sublayer};

    assert!(!pipeline_sublayer_eval_should_fire_for(Off, 8, "gemma4"));
    assert!(!pipeline_sublayer_eval_should_fire_for(
        PerLayer, 8, "gemma4"
    ));
    assert!(!pipeline_sublayer_eval_should_fire_for(
        Sublayer, 1, "gemma4"
    ));
    assert!(pipeline_sublayer_eval_should_fire_for(
        Sublayer, 8, "gemma4"
    ));
    assert!(!pipeline_sublayer_eval_should_fire_for(
        Sublayer, 8, "qwen3_5"
    ));
    assert!(!pipeline_sublayer_eval_should_fire_for(
        Sublayer,
        8,
        "gemma4_vl"
    ));
    assert!(!pipeline_sublayer_eval_should_fire_for(
        Sublayer,
        8,
        "gemma4_unified"
    ));
}

#[test]
fn direct_cpp_gemma4_post_attn_ffn_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_DIRECT_GEMMA4_POST_ATTN_FFN_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_DIRECT_GEMMA4_POST_ATTN_FFN_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_DIRECT_GEMMA4_POST_ATTN_FFN_ENABLED",
        "1"
    ));
}

#[test]
fn dense_swiglu_packed_metal_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_DENSE_SWIGLU_PACKED_METAL_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_DENSE_SWIGLU_PACKED_METAL_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_DENSE_SWIGLU_PACKED_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_gated_delta_prefill_streaming_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_STREAMING_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_STREAMING_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_STREAMING_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_gated_delta_prefill_tile_512_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_TILE_512_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_TILE_512_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_TILE_512_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_prefill_single_2048_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_PREFILL_SINGLE_2048_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_SINGLE_2048_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_SINGLE_2048_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_prefill_flat_ffn_is_family_seq_and_rank_gated() {
    assert!(should_qwen_prefill_flat_ffn_for(true, "qwen3_5", 1024, 3));
    assert!(should_qwen_prefill_flat_ffn_for(true, "QWEN3_5", 128, 3));
    assert!(
        !should_qwen_prefill_flat_ffn_for(true, "qwen3_5", 1, 3),
        "decode must stay on the 3-D qw path"
    );
    assert!(!should_qwen_prefill_flat_ffn_for(true, "qwen3_5", 1024, 2));
    assert!(!should_qwen_prefill_flat_ffn_for(false, "qwen3_5", 1024, 3));
    assert!(!should_qwen_prefill_flat_ffn_for(true, "gemma4", 1024, 3));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_FFN_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_FFN_DISABLED",
        "0"
    ));
    assert!(probe("AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_FFN_ENABLED", "1"));
}

#[test]
fn qwen_prefill_contiguous_ffn_is_family_seq_and_rank_gated() {
    assert!(should_qwen_prefill_contiguous_ffn_for(
        true, "qwen3_5", 1024, 3
    ));
    assert!(should_qwen_prefill_contiguous_ffn_for(
        true, "QWEN3_5", 128, 3
    ));
    assert!(
        !should_qwen_prefill_contiguous_ffn_for(true, "qwen3_5", 1, 3),
        "decode must not pay a contiguous copy"
    );
    assert!(!should_qwen_prefill_contiguous_ffn_for(
        true, "qwen3_5", 1024, 2
    ));
    assert!(!should_qwen_prefill_contiguous_ffn_for(
        false, "qwen3_5", 1024, 3
    ));
    assert!(!should_qwen_prefill_contiguous_ffn_for(
        true, "gemma4", 1024, 3
    ));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_PREFILL_CONTIGUOUS_FFN_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_CONTIGUOUS_FFN_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_CONTIGUOUS_FFN_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_la_out_proj_silu_mul_qmm_is_family_and_seq_gated() {
    assert!(should_qwen_la_out_proj_silu_mul_qmm_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_la_out_proj_silu_mul_qmm_for(
        true, "QWEN3_5", 128
    ));
    assert!(
        !should_qwen_la_out_proj_silu_mul_qmm_for(true, "qwen3_5", 1),
        "decode keeps rms_norm_gated + qw"
    );
    assert!(!should_qwen_la_out_proj_silu_mul_qmm_for(
        false, "qwen3_5", 1024
    ));
    assert!(!should_qwen_la_out_proj_silu_mul_qmm_for(
        true, "gemma4", 1024
    ));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_LA_OUT_PROJ_SILU_MUL_QMM_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_LA_OUT_PROJ_SILU_MUL_QMM_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_LA_OUT_PROJ_SILU_MUL_QMM_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_dense_ffn_gate_up_matvec_metal_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_linear_mtp_exact_env_override_uses_truthy_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_LINEAR_MTP_EXACT_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_LINEAR_MTP_EXACT_DISABLED",
        "0"
    ));
    assert!(probe("AX_FASTPATH_TEST_QWEN_LINEAR_MTP_EXACT_ENABLED", "1"));
}

#[test]
fn invariant_mxfp4_qmv_fast_is_opt_in() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_INVARIANT_MXFP4_QMV_FAST_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_INVARIANT_MXFP4_QMV_FAST_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_INVARIANT_MXFP4_QMV_FAST_ENABLED",
        "1"
    ));
}

#[test]
fn dense_ffn_compile_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_ENABLED",
        "1"
    ));
}

#[test]
fn dense_ffn_compile_prefill_uses_default_on_with_min_leading() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_PREFILL_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_PREFILL_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_PREFILL_ENABLED",
        "1"
    ));
    assert_eq!(super::DENSE_FFN_PREFILL_COMPILE_MIN_LEADING, 256);
    assert_eq!(super::MOE_PACKED_GEGLU_PREFILL_MAX_SEQ, 512);
}

#[test]
fn qwen_compiled_dual_gate_up_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_COMPILED_DUAL_GATE_UP_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_COMPILED_DUAL_GATE_UP_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_COMPILED_DUAL_GATE_UP_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_split_ffn_prefill_compile_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_SPLIT_FFN_PREFILL_COMPILE_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_SPLIT_FFN_PREFILL_COMPILE_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_SPLIT_FFN_PREFILL_COMPILE_ENABLED",
        "1"
    ));
    assert_eq!(super::QWEN_SPLIT_FFN_PREFILL_COMPILE_MIN_LEADING, 128);
}

#[test]
fn qwen_linear_add_rms_norm_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ADD_RMS_NORM_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ADD_RMS_NORM_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_LINEAR_ADD_RMS_NORM_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_prefill_pipeline_block_is_family_seq_and_stride_gated() {
    assert!(should_qwen_prefill_pipeline_block_for(
        true, "qwen3_5", 1024, 7, 64, 8
    ));
    assert!(should_qwen_prefill_pipeline_block_for(
        true,
        "QWEN3_NEXT",
        2048,
        15,
        64,
        8
    ));
    assert!(
        !should_qwen_prefill_pipeline_block_for(true, "qwen3_5", 512, 7, 64, 8),
        "short prefill stays one lazy graph"
    );
    assert!(!should_qwen_prefill_pipeline_block_for(
        true, "qwen3_5", 1024, 6, 64, 8
    ));
    assert!(
        !should_qwen_prefill_pipeline_block_for(true, "qwen3_5", 1024, 63, 64, 8),
        "never fire after the final layer"
    );
    assert!(!should_qwen_prefill_pipeline_block_for(
        true, "gemma4", 1024, 7, 64, 8
    ));
    assert!(!should_qwen_prefill_pipeline_block_for(
        false, "qwen3_5", 1024, 7, 64, 8
    ));
    assert_eq!(super::QWEN_PREFILL_PIPELINE_BLOCK, 8);
}

#[test]
fn qwen_prefill_interlayer_add_rms_is_family_and_seq_gated() {
    assert!(should_qwen_prefill_interlayer_add_rms_for(
        true, "qwen3_5", 1024
    ));
    assert!(should_qwen_prefill_interlayer_add_rms_for(
        true,
        "QWEN3_NEXT",
        128
    ));
    assert!(!should_qwen_prefill_interlayer_add_rms_for(
        true, "qwen3_5", 1
    ));
    assert!(!should_qwen_prefill_interlayer_add_rms_for(
        true, "gemma4", 1024
    ));
    assert!(!should_qwen_prefill_interlayer_add_rms_for(
        false, "qwen3_5", 1024
    ));
    assert!(should_defer_qwen_prefill_ffn_residual_for(
        true, true, false, 3
    ));
    assert!(
        !should_defer_qwen_prefill_ffn_residual_for(true, false, false, 3),
        "do not defer into a full-attn layer"
    );
    assert!(!should_defer_qwen_prefill_ffn_residual_for(
        true, true, true, 3
    ));
}

#[test]
fn qwen_swiglu_down_fuse_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_SWIGLU_DOWN_FUSE_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_SWIGLU_DOWN_FUSE_DISABLED",
        "0"
    ));
    assert!(probe("AX_FASTPATH_TEST_QWEN_SWIGLU_DOWN_FUSE_ENABLED", "1"));
}

#[test]
fn qwen_prefill_dual_qmm_swiglu_metal_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_prefill_flat_down_qmm_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_DOWN_QMM_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_DOWN_QMM_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_DOWN_QMM_ENABLED",
        "1"
    ));
}

#[test]
fn qwen_dual_qmm_swiglu_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_DUAL_QMM_SWIGLU_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_DUAL_QMM_SWIGLU_DISABLED",
        "0"
    ));
    assert!(probe("AX_FASTPATH_TEST_QWEN_DUAL_QMM_SWIGLU_ENABLED", "1"));
}

#[test]
fn gemma4_assistant_compile_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_COMPILE_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_COMPILE_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_COMPILE_ENABLED",
        "1"
    ));
}

#[test]
fn gemma4_assistant_mtp_cycle_guard_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD_ENABLED",
        "1"
    ));
}

#[test]
fn gemma4_assistant_lazy_multi_depth_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH_ENABLED",
        "1"
    ));
}

#[test]
fn gemma4_assistant_deep_needs_first_conf_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF_ENABLED",
        "1"
    ));
}

#[test]
fn verify_chunked_submit_is_opt_in_and_multi_position_only() {
    // Non-throughput family: the certified S=2..=4 short-verify band
    // applies whatever the throughput depth env says.
    const GEMMA: &str = "gemma";
    // Default (unset env resolves to 0) never splits a build.
    assert_eq!(verify_submit_interval_for_build(2, 40, 0, GEMMA), 0);
    // A single-position build belongs to the direct pipeline, which
    // already double-buffers; splitting it would only add submits.
    assert_eq!(verify_submit_interval_for_build(1, 40, 8, GEMMA), 0);
    // A speculative verify build splits at the configured interval.
    assert_eq!(verify_submit_interval_for_build(2, 40, 8, GEMMA), 0);
    assert_eq!(verify_submit_interval_for_build(4, 40, 8, GEMMA), 0);
    assert_eq!(verify_submit_interval_for_build(5, 40, 4, GEMMA), 4);
    // An interval that cannot produce a submit before the caller's own
    // terminating eval is pure overhead, so it is refused.
    assert_eq!(verify_submit_interval_for_build(2, 40, 40, GEMMA), 0);
    assert_eq!(verify_submit_interval_for_build(2, 40, 64, GEMMA), 0);
}

/// Finding B: the Qwen throughput depth only widens the verify window for
/// the Qwen linear families it is measured on; every other family keeps the
/// certified depth-3 width.
#[test]
fn qwen_throughput_depth_is_family_scoped() {
    // Widened throughput depth for a Qwen linear family.
    assert_eq!(
        qwen_linear_mtp_max_verify_drafts_for_family(true, 7, true),
        7
    );
    assert_eq!(qwen_linear_mtp_max_verify_seq_for_family(true, 7, true), 8);
    // A Gemma (non-Qwen-linear) family keeps the certified width.
    assert_eq!(
        qwen_linear_mtp_max_verify_drafts_for_family(true, 7, false),
        QWEN_LINEAR_EXACT_MAX_VERIFY_DRAFTS_CERTIFIED
    );
    assert_eq!(
        qwen_linear_mtp_max_verify_seq_for_family(true, 7, false),
        QWEN_LINEAR_EXACT_MAX_VERIFY_DRAFTS_CERTIFIED as i32 + 1
    );
    // The family predicate matches exactly the two served families.
    assert!(qwen_linear_throughput_family("qwen3_5"));
    assert!(qwen_linear_throughput_family("qwen3_next"));
    assert!(!qwen_linear_throughput_family("gemma"));
    assert!(!qwen_linear_throughput_family("deepseek_v4"));
    // seq containment follows the family-scoped window.
    assert!(qwen_linear_mtp_verify_seq_contains_for_family(
        8, true, 7, true
    ));
    assert!(!qwen_linear_mtp_verify_seq_contains_for_family(
        8, true, 7, false
    ));
    assert!(qwen_linear_mtp_verify_seq_contains_for_family(
        4, true, 7, false
    ));
}

#[test]
fn gemma_family_keeps_certified_verify_window_under_widened_depth() {
    // Sequence-classification sites must be family scoped: a widened
    // throughput depth (7 drafts -> verify seq 2..=8) must not widen a
    // Gemma family, which stays at the certified S=2..=4 window.
    let gemma = qwen_linear_throughput_family("gemma");
    assert!(!gemma);
    // seq 5 is outside Gemma's certified window even at depth 7.
    assert!(!qwen_linear_mtp_verify_seq_contains_for_family(
        5, true, 7, gemma
    ));
    // seq 4 remains a verify shape for Gemma.
    assert!(qwen_linear_mtp_verify_seq_contains_for_family(
        4, true, 7, gemma
    ));
    // The same widened depth does widen a Qwen linear family to seq 5.
    assert!(qwen_linear_mtp_verify_seq_contains_for_family(
        5,
        true,
        7,
        qwen_linear_throughput_family("qwen3_5")
    ));
}

#[test]
fn exact_short_verify_uses_configured_interval_instead_of_zero() {
    // Official harness sets VERIFY_SUBMIT_LAYERS=8; honor it as the
    // sole mid-loop stride (not stacked on PIPELINE=layer).
    assert_eq!(exact_short_verify_submit_interval(2, 64, 8), 8);
    assert_eq!(exact_short_verify_submit_interval(4, 64, 8), 8);
    assert_eq!(
        exact_short_verify_submit_interval(2, 64, 0),
        EXACT_SHORT_VERIFY_SUBMIT_DEFAULT
    );
    assert_eq!(exact_short_verify_submit_interval(1, 64, 8), 0);
    assert_eq!(exact_short_verify_submit_interval(5, 64, 8), 0);
    assert_eq!(exact_short_verify_submit_interval(2, 8, 8), 0);
    assert_eq!(exact_short_verify_submit_interval(2, 0, 8), 0);
}

#[test]
fn moe_router_fused_metal_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_MOE_ROUTER_FUSED_METAL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_MOE_ROUTER_FUSED_METAL_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_MOE_ROUTER_FUSED_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn linear_attention_whole_layer_metal_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_WHOLE_LAYER_METAL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_WHOLE_LAYER_METAL_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_WHOLE_LAYER_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn moe_deep_expert_block_metal_uses_opt_in_contract() {
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_MOE_DEEP_EXPERT_BLOCK_METAL_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_MOE_DEEP_EXPERT_BLOCK_METAL_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_MOE_DEEP_EXPERT_BLOCK_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn geglu_mul_metal_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_GEGLU_MUL_METAL_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_GEGLU_MUL_METAL_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_GEGLU_MUL_METAL_ENABLED",
        "1"
    ));
}

#[test]
fn gemma4_per_layer_input_gate_compile_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE_ENABLED",
        "1"
    ));
}

#[test]
fn linear_attention_rms_norm_gate_metal_uses_default_on_kill_switch_contract() {
    assert!(parse_bool_env_default_on(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_RMS_NORM_GATE_METAL_UNSET"
    ));
    assert!(!probe_default_on(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_RMS_NORM_GATE_METAL_DISABLED",
        "0"
    ));
    assert!(probe_default_on(
        "AX_FASTPATH_TEST_LINEAR_ATTENTION_RMS_NORM_GATE_METAL_ENABLED",
        "1"
    ));
}

fn probe_usize(name: &str, value: &str) -> Option<usize> {
    // SAFETY: each test owns a disjoint set of env-var names. Remove
    // before asserting so a failing assert does not leak the var.
    unsafe {
        std::env::set_var(name, value);
    }
    let observed = parse_positive_usize_env(name);
    unsafe {
        std::env::remove_var(name);
    }
    observed
}

#[test]
fn parse_positive_usize_env_accepts_positive_values() {
    assert_eq!(probe_usize("AX_FASTPATH_TEST_USIZE_16", "16"), Some(16));
    assert_eq!(
        probe_usize("AX_FASTPATH_TEST_USIZE_TRIMMED", " 32 "),
        Some(32)
    );
}

#[test]
fn parse_positive_usize_env_rejects_unset_zero_and_invalid_values() {
    assert_eq!(
        parse_positive_usize_env("AX_FASTPATH_TEST_USIZE_UNSET"),
        None
    );
    for value in ["0", "", "no", "-1", "1.5"] {
        let name = format!("AX_FASTPATH_TEST_BAD_USIZE_{}", value.replace('-', "neg"));
        assert_eq!(
            probe_usize(&name, value),
            None,
            "expected None for {value:?}"
        );
    }
}

#[test]
fn shared_fusion_threshold_preserves_each_callers_default_when_unset() {
    if std::env::var_os("AX_MLX_MOE_SHARED_FUSION_SEQ_THRESHOLD").is_some() {
        return;
    }

    assert_eq!(moe_shared_fusion_seq_threshold(64), 64);
    assert_eq!(moe_shared_fusion_seq_threshold(128), 128);
}

#[test]
fn parse_nonnegative_f32_accepts_finite_zero_and_positive_values() {
    assert_eq!(parse_nonnegative_f32("0"), Some(0.0));
    assert_eq!(parse_nonnegative_f32("1e-5"), Some(1.0e-5));
    assert_eq!(parse_nonnegative_f32(" 0.25 "), Some(0.25));
}

#[test]
fn parse_nonnegative_f32_rejects_negative_invalid_and_nonfinite_values() {
    for value in ["-0.1", "NaN", "inf", "-inf", "", "no"] {
        assert_eq!(
            parse_nonnegative_f32(value),
            None,
            "expected invalid sparse threshold for {value:?}"
        );
    }
}

#[test]
fn mtp_dense_head_draft_bits_defaults_to_three_unless_three_or_four() {
    assert_eq!(mtp_dense_head_draft_bits_for(None), 3);
    assert_eq!(mtp_dense_head_draft_bits_for(Some("3")), 3);
    assert_eq!(mtp_dense_head_draft_bits_for(Some("4")), 4);
    assert_eq!(mtp_dense_head_draft_bits_for(Some("2")), 3);
    assert_eq!(mtp_dense_head_draft_bits_for(Some("garbage")), 3);
    assert_eq!(mtp_dense_head_draft_bits_for(Some(" 4 ")), 4);
}

#[test]
fn long_prompt_prefill_clamp_skips_qwen3_5() {
    assert!(!long_prompt_prefill_clamp_applies("qwen3_5"));
    assert!(!long_prompt_prefill_clamp_applies("QWEN3_5"));
    assert!(!long_prompt_prefill_clamp_applies("muse_glimmer"));
    assert!(!long_prompt_prefill_clamp_applies("qwen3_vl_moe"));
    assert!(!long_prompt_prefill_clamp_applies("qwen3_vl"));
    assert!(long_prompt_prefill_clamp_applies("gemma4"));
    assert!(long_prompt_prefill_clamp_applies("qwen3_next"));
    assert_eq!(
        scale_prefill_chunk_for_remaining_in_family(2048, 2048, "qwen3_5"),
        2048
    );
    assert_eq!(
        scale_prefill_chunk_for_remaining_in_family(2048, 2048, "muse_glimmer"),
        2048
    );
    assert_eq!(
        scale_prefill_chunk_for_remaining_in_family(2048, 2048, "gemma4"),
        long_prompt_prefill_chunk()
    );
}

#[test]
fn cold_prefill_clears_mlx_cache_only_on_empty_kv() {
    assert!(should_clear_mlx_cache_before_cold_prefill(0));
    assert!(!should_clear_mlx_cache_before_cold_prefill(1));
    assert!(!should_clear_mlx_cache_before_cold_prefill(2048));
    assert!(should_clear_mlx_cache_before_cold_prefill_for(false, 0));
    assert!(
        !should_clear_mlx_cache_before_cold_prefill_for(true, 0),
        "skip flag must keep the buffer pool warm on cold prefill"
    );
    assert!(!should_clear_mlx_cache_before_cold_prefill_for(true, 1));
}

#[test]
fn qwen_prefill_intermediate_async_eval_is_family_and_chunk_gated() {
    assert!(should_async_eval_intermediate_qwen_prefill_for(
        true, "qwen3_5", false
    ));
    assert!(should_async_eval_intermediate_qwen_prefill_for(
        true, "QWEN3_5", false
    ));
    assert!(
        !should_async_eval_intermediate_qwen_prefill_for(true, "qwen3_5", true),
        "final chunk must still block so decode sees settled KV"
    );
    assert!(!should_async_eval_intermediate_qwen_prefill_for(
        false, "qwen3_5", false
    ));
    assert!(!should_async_eval_intermediate_qwen_prefill_for(
        true, "gemma4", false
    ));
    assert!(!should_async_eval_intermediate_qwen_prefill_for(
        true,
        "qwen3_next",
        false
    ));
}

#[test]
fn qwen_prefill_lazy_intermediate_is_family_total_and_chunk_gated() {
    assert!(should_keep_lazy_intermediate_qwen_prefill_for(
        true, "qwen3_5", false, 2048
    ));
    assert!(should_keep_lazy_intermediate_qwen_prefill_for(
        true, "QWEN3_5", false, 2048
    ));
    assert!(
        should_keep_lazy_intermediate_qwen_prefill_for(true, "qwen3_5", false, 128),
        "single-chunk contract totals still match the skip_cache_only gate"
    );
    assert!(
        !should_keep_lazy_intermediate_qwen_prefill_for(true, "qwen3_5", true, 2048),
        "final chunk must still eval so decode sees settled KV"
    );
    assert!(!should_keep_lazy_intermediate_qwen_prefill_for(
        false, "qwen3_5", false, 2048
    ));
    assert!(!should_keep_lazy_intermediate_qwen_prefill_for(
        true, "qwen3_5", false, 2049
    ));
    assert!(!should_keep_lazy_intermediate_qwen_prefill_for(
        true, "gemma4", false, 2048
    ));
    assert!(!should_keep_lazy_intermediate_qwen_prefill_for(
        true,
        "qwen3_next",
        false,
        2048
    ));
    assert!(!parse_bool_env(
        "AX_FASTPATH_TEST_QWEN_PREFILL_LAZY_INTERMEDIATE_UNSET"
    ));
    assert!(!probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_LAZY_INTERMEDIATE_DISABLED",
        "0"
    ));
    assert!(probe(
        "AX_FASTPATH_TEST_QWEN_PREFILL_LAZY_INTERMEDIATE_ENABLED",
        "1"
    ));
}

#[test]
fn certified_non_deepseek_skips_cache_only_split_on_contract_shapes() {
    for family in [
        "qwen3_5",
        "qwen3_next",
        "qwen3",
        "gemma4",
        "glm4_moe_lite",
        "gpt_oss",
        "muse_glimmer",
        "qwen3_vl",
        "qwen3_vl_moe",
    ] {
        assert!(
            skip_cache_only_split_for_family(family, 128),
            "{family} p128"
        );
        assert!(skip_cache_only_split_for_family(family, 2048));
        assert!(!skip_cache_only_split_for_family(family, 2049));
    }
    assert!(!skip_cache_only_split_for_family("qwen3_5", 0));
    assert!(!skip_cache_only_split_for_family("deepseek_v32", 128));
    assert!(!skip_cache_only_split_for_family("gemma4_vl", 128));
}

#[test]
fn qwen_skip_linear_prefill_mask_is_family_and_layer_gated() {
    assert!(should_skip_linear_prefill_mask_for(true, "qwen3_5", true));
    assert!(should_skip_linear_prefill_mask_for(
        true,
        "QWEN3_NEXT",
        true
    ));
    assert!(
        !should_skip_linear_prefill_mask_for(true, "qwen3_5", false),
        "full-attn layers still need the offset mask"
    );
    assert!(!should_skip_linear_prefill_mask_for(true, "gemma4", true));
    assert!(!should_skip_linear_prefill_mask_for(false, "qwen3_5", true));
}

#[test]
fn qwen_prefill_eval_kv_only_is_intermediate_and_family_gated() {
    assert!(should_qwen_prefill_eval_kv_only_for(
        true, "qwen3_5", false, 2048
    ));
    assert!(
        !should_qwen_prefill_eval_kv_only_for(true, "qwen3_5", true, 2048),
        "final chunk still evals logits + KV"
    );
    assert!(!should_qwen_prefill_eval_kv_only_for(
        true, "qwen3_5", false, 2049
    ));
    assert!(!should_qwen_prefill_eval_kv_only_for(
        true, "gemma4", false, 2048
    ));
    assert!(!should_qwen_prefill_eval_kv_only_for(
        false, "qwen3_5", false, 2048
    ));
}

#[test]
fn exact_size_first_kv_is_write_start_gated() {
    assert!(should_exact_size_first_kv_for(true, 0));
    assert!(
        !should_exact_size_first_kv_for(true, 128),
        "append after the first write still grows in KV_CHUNK_TOKENS"
    );
    assert!(!should_exact_size_first_kv_for(false, 0));
}

#[test]
fn exact_size_first_kv_targets_unaligned_contract_p128() {
    // The product flag only changes the first write. Of the fleet
    // contract prompts, only 128 is not a KV_CHUNK_TOKENS multiple, so
    // p512/p2048 already skip zeros+slice_update without the flag.
    const CHUNK: usize = crate::kv_cache::KV_CHUNK_TOKENS;
    assert_eq!(CHUNK, 256);
    assert_ne!(128 % CHUNK, 0, "p128 must take the exact-size first write");
    assert_eq!(512 % CHUNK, 0);
    assert_eq!(2048 % CHUNK, 0);
    assert!(
        should_exact_size_first_kv_for(true, 0),
        "fresh-layer first write is the only exact-size site"
    );
    assert!(!should_exact_size_first_kv_for(true, 128));
}

#[test]
fn exact_size_kv_grow_is_aligned_tight_append() {
    assert!(should_exact_size_kv_grow_for(true, 1024, 1024, 2048, 2048));
    assert!(
        !should_exact_size_kv_grow_for(true, 128, 128, 129, 256),
        "decode +1 must keep the padded zeros grow"
    );
    assert!(!should_exact_size_kv_grow_for(true, 512, 1024, 1536, 1536));
    assert!(!should_exact_size_kv_grow_for(
        false, 1024, 1024, 2048, 2048
    ));
}

#[test]
fn skip_unused_full_kv_view_slice_is_full_buffer_gated() {
    assert!(should_skip_unused_full_kv_view_slice_for(
        true, 0, 2048, 2048
    ));
    assert!(
        !should_skip_unused_full_kv_view_slice_for(true, 0, 128, 256),
        "padded first write still needs the live-token slice"
    );
    assert!(!should_skip_unused_full_kv_view_slice_for(
        true, 1024, 2048, 2048
    ));
    assert!(!should_skip_unused_full_kv_view_slice_for(
        false, 0, 2048, 2048
    ));
}

#[test]
fn skip_unused_la_out_reshape_is_shape_gated() {
    assert!(should_skip_unused_la_out_reshape_for(
        true,
        &[1, 1024, 2048],
        1024,
        2048
    ));
    assert!(
        !should_skip_unused_la_out_reshape_for(true, &[1, 1024, 32, 64], 1024, 2048),
        "BHSD still needs the flatten into [1,S,V]"
    );
    assert!(!should_skip_unused_la_out_reshape_for(
        false,
        &[1, 1024, 2048],
        1024,
        2048
    ));
}

#[test]
fn reuse_la_initial_state_zeros_is_flag_gated() {
    assert!(should_reuse_la_initial_state_zeros_for(true));
    assert!(!should_reuse_la_initial_state_zeros_for(false));
}

#[test]
fn dense_long_mt_sensitive_f32_range_only_disables_selected_layers() {
    let range = Some((28, 8));
    assert!(dense_long_mt_bf16_fold_enabled_for(true, Some(27), range));
    assert!(!dense_long_mt_bf16_fold_enabled_for(true, Some(28), range));
    assert!(!dense_long_mt_bf16_fold_enabled_for(true, Some(35), range));
    assert!(dense_long_mt_bf16_fold_enabled_for(true, Some(36), range));
    assert!(!dense_long_mt_bf16_fold_enabled_for(false, Some(27), range));
    assert!(dense_long_mt_bf16_fold_enabled_for(true, None, range));
}

#[test]
fn long_prompt_prefill_chunk_defaults_to_512() {
    // Process-cached via OnceLock; assert the default constant and that
    // the live helper never returns zero.
    assert_eq!(LONG_PROMPT_PREFILL_CHUNK, 512);
    assert!(long_prompt_prefill_chunk() >= 1);
}

#[test]
fn resolve_prefill_chunk_defaults_mla_to_chunk_aligned_size() {
    assert_eq!(
        resolve_prefill_chunk(true, 256, None),
        MLA_DEFAULT_PREFILL_CHUNK
    );
}

#[test]
fn resolve_prefill_chunk_allows_mla_override() {
    assert_eq!(resolve_prefill_chunk(true, 256, Some(32)), 32);
}

#[test]
fn resolve_prefill_chunk_preserves_non_mla_request() {
    assert_eq!(resolve_prefill_chunk(false, 256, Some(32)), 256);
}

#[test]
fn resolve_prefill_chunk_clamps_zero_for_all_models() {
    assert_eq!(resolve_prefill_chunk(false, 0, None), 1);
    assert_eq!(resolve_prefill_chunk(true, 0, Some(0)), 1);
}

#[test]
fn resolve_mla_cold_prefill_chunk_defaults_to_warm_trail() {
    assert_eq!(
        resolve_mla_cold_prefill_chunk(MLA_DEFAULT_PREFILL_CHUNK, None),
        MLA_DEFAULT_PREFILL_CHUNK
    );
}

#[test]
fn resolve_mla_cold_prefill_chunk_allows_throughput_override() {
    assert_eq!(resolve_mla_cold_prefill_chunk(16, Some(2048)), 2048);
}

#[test]
fn select_prefill_chunk_for_request_matrix_cold_vs_warm() {
    // Empty cache always uses cold field (even if larger than warm).
    assert_eq!(
        select_prefill_chunk_for_request(0, 2048, 16),
        (2048, PrefillChunkMode::Cold)
    );
    // Restored / partial cache always uses warm field.
    assert_eq!(
        select_prefill_chunk_for_request(1, 2048, 16),
        (16, PrefillChunkMode::WarmExtend)
    );
    assert_eq!(
        select_prefill_chunk_for_request(128, 16, 16),
        (16, PrefillChunkMode::WarmExtend)
    );
    // R2 default: both fields equal → mode still distinguishes occupancy.
    assert_eq!(
        select_prefill_chunk_for_request(0, 16, 16),
        (16, PrefillChunkMode::Cold)
    );
    // Zero chunks clamp to 1 so the loop cannot stall.
    assert_eq!(
        select_prefill_chunk_for_request(0, 0, 0),
        (1, PrefillChunkMode::Cold)
    );
}

#[test]
fn select_prefill_chunk_recompute_after_reset_is_cold() {
    // After cache.reset() / failed restore, seq_len is 0 → cold trail.
    let after_reset_seq = 0usize;
    assert_eq!(
        select_prefill_chunk_for_request(after_reset_seq, 16, 16).1,
        PrefillChunkMode::Cold
    );
}

#[test]
fn prefill_warmup_token_count_preserves_non_mla_lightweight_warmup() {
    assert_eq!(prefill_warmup_token_count(false, 256), 8);
}

#[test]
fn prefill_warmup_token_count_uses_effective_mla_chunk() {
    assert_eq!(
        prefill_warmup_token_count(true, MLA_DEFAULT_PREFILL_CHUNK),
        MLA_DEFAULT_PREFILL_CHUNK
    );
    assert_eq!(prefill_warmup_token_count(true, 32), 32);
}

#[test]
fn prefill_warmup_token_count_clamps_mla_zero() {
    assert_eq!(prefill_warmup_token_count(true, 0), 1);
}

#[test]
fn prefill_warmup_token_lengths_cover_short_prompt_serving_shapes() {
    let lengths = prefill_warmup_token_lengths(false, 256);
    assert!(lengths.contains(&8), "historical lightweight warm-up");
    assert!(lengths.contains(&32));
    assert!(lengths.contains(&34), "flip S0 prompt length must be warm");
    assert!(lengths.contains(&64));
    // Sorted unique
    let mut sorted = lengths.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(lengths, sorted);
}
