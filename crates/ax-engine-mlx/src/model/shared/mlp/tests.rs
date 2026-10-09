
use super::*;
use mlx_sys::{
    MlxQuantizationMode, add, astype, concatenate, eval, quantize, quantized_matmul, slice,
    slice_last_dim,
};

#[test]
fn dense_ffn_prefill_compile_keeps_qwen_imperative_by_default() {
    assert!(!dense_ffn_prefill_compile_supported("qwen3_5", 512));
    assert!(!dense_ffn_prefill_compile_supported("qwen3_next", 128));
    assert!(dense_ffn_prefill_compile_supported("gemma4", 512));
    assert!(
        !dense_ffn_prefill_compile_supported("qwen3_5", 1024),
        "Qwen packed prefill compile stays opt-in after community 3d wash"
    );
    assert!(fastpath::should_qwen_packed_ffn_prefill_compile_for(
        true, "qwen3_5", 1024
    ));
}

#[test]
fn qwen_compiled_split_prefill_gate_up_stays_opt_in_after_wash() {
    let x = array_f32(&vec![0.1; 128], &[1, 2, 64]);
    let w = QuantizedWeight {
        weight: array_f32(&vec![0.0; 64], &[1, 64]),
        scales: Some(array_f32(&[1.0], &[1])),
        biases: Some(array_f32(&[0.0], &[1])),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    assert!(
        qwen_compiled_split_prefill_gate_up(&x, &w, &w).is_none(),
        "dual-qmm compile must stay default-OFF after the 890.96 vs 891 wash"
    );
}

#[test]
fn qwen_prefill_dual_affine_gate_up_matches_two_qmm() {
    let seq = 1024i32;
    let hidden = 64i32;
    let inter = 32i32;
    let x_data: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) - 2048.0) * 0.000_244_140_63)
        .collect();
    let gate_data: Vec<f32> = (0..inter * hidden)
        .map(|i| ((i as f32) - 1024.0) * 0.0005)
        .collect();
    let up_data: Vec<f32> = (0..inter * hidden)
        .map(|i| ((i as f32) - 512.0) * -0.0004)
        .collect();
    let x = array_f32(&x_data, &[1, seq, hidden]);
    let gq = quantize(
        &array_f32(&gate_data, &[inter, hidden]),
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let uq = quantize(
        &array_f32(&up_data, &[inter, hidden]),
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let gate_w = QuantizedWeight {
        weight: gq[0].clone(),
        scales: Some(gq[1].clone()),
        biases: Some(gq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let up_w = QuantizedWeight {
        weight: uq[0].clone(),
        scales: Some(uq[1].clone()),
        biases: Some(uq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    assert!(
        qwen_prefill_maybe_dual_affine_gate_up_for(false, "qwen3_5", seq, &x, &gate_w, &up_w)
            .is_none()
    );
    let (g_dual, u_dual) =
        qwen_prefill_maybe_dual_affine_gate_up_for(true, "qwen3_5", seq, &x, &gate_w, &up_w)
            .expect("Qwen dual-affine qmm must engage at the p2048 chunk length");
    let g_ref = qw(&x, &gate_w);
    let u_ref = qw(&x, &up_w);
    eval(&[&g_dual, &u_dual, &g_ref, &u_ref]);
    let a = g_dual.data_f32();
    let b = g_ref.data_f32();
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    for (l, r) in u_dual.data_f32().iter().zip(u_ref.data_f32().iter()) {
        max_abs = max_abs.max((l - r).abs());
    }
    assert!(
        max_abs < 1e-4,
        "dual-affine qmm must match two steel qw, max_abs={max_abs}"
    );
    assert!(
        fastpath::should_qwen_prefill_dual_affine_qmm_for(true, "qwen3_5", 1024),
        "shipped dual-affine gate must accept the p2048 chunk length"
    );
}

#[test]
fn qwen_compiled_split_prefill_ffn_matches_two_qmm_4bit_gs32() {
    // Contract p128 leading=128. AXQ language FFN is 4-bit gs32.
    let x_data: Vec<f32> = (0..128 * 64)
        .map(|i| ((i as f32) - 4096.0) * 0.000_244_140_63)
        .collect();
    let gate_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0005)
        .collect();
    let up_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 512.0) * -0.0004)
        .collect();
    let down_data: Vec<f32> = (0..64 * 32)
        .map(|i| ((i as f32) - 768.0) * 0.0003)
        .collect();
    let x = array_f32(&x_data, &[1, 128, 64]);
    let gate_w = array_f32(&gate_data, &[32, 64]);
    let up_w = array_f32(&up_data, &[32, 64]);
    let down_w = array_f32(&down_data, &[64, 32]);
    let gq = quantize(
        &gate_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let uq = quantize(
        &up_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(gq.len(), 3);
    assert_eq!(uq.len(), 3);
    assert_eq!(dq.len(), 3);
    let qweight = |q: &[MlxArray]| QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let gate = qweight(&gq);
    let up = qweight(&uq);
    let down = qweight(&dq);
    let compiled = qwen_compiled_split_prefill_ffn(
        0x5157_454e_5052_4546,
        0,
        &x,
        &gate,
        &up,
        Some(&down),
        None,
        1e-6,
        ProjectionBatchPolicy::Shared,
    )
    .expect("Qwen 4-bit gs32 split FFN prefill compile should engage at leading=128");
    let p_gate = quantized_matmul(
        &x,
        &gq[0],
        &gq[1],
        Some(&gq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let p_up = quantized_matmul(
        &x,
        &uq[0],
        &uq[1],
        Some(&uq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let hidden = silu_mul(&p_gate, &p_up, None);
    let portable = quantized_matmul(
        &hidden,
        &dq[0],
        &dq[1],
        Some(&dq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    assert_close(compiled.data_f32(), portable.data_f32(), 3.0e-2);
    let decode = array_f32(&x_data[..64], &[1, 1, 64]);
    assert!(
        qwen_compiled_split_prefill_ffn(
            0x5157_454e_5052_4546,
            0,
            &decode,
            &gate,
            &up,
            Some(&down),
            None,
            1e-6,
            ProjectionBatchPolicy::Shared,
        )
        .is_none(),
        "split FFN prefill compile must reject decode seq==1"
    );
}

#[test]
fn qwen_compiled_split_verify_ffn_mxfp4_s2_matches_imperative() {
    let seq = 2i32;
    let hidden = 64i32;
    let intermediate = 32i32;
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 16.0) * 0.01)
        .collect();
    let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 8.0) * -0.008)
        .collect();
    let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
        .map(|i| ((i as f32) - 12.0) * 0.006)
        .collect();
    let x = array_f32(&x_data, &[1, seq, hidden]);
    let qmx = |w: &MlxArray| {
        let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
        assert_eq!(q.len(), 2);
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    };
    let gate = qmx(&array_f32(&gate_data, &[intermediate, hidden]));
    let up = qmx(&array_f32(&up_data, &[intermediate, hidden]));
    let down = qmx(&array_f32(&down_data, &[hidden, intermediate]));
    let compiled = qwen_compiled_split_verify_ffn(
        0x5652_4659_4d58_5034,
        0,
        &x,
        &gate,
        &up,
        Some(&down),
        None,
        1e-6,
        ProjectionBatchPolicy::Shared,
    )
    .expect("exact S=2 MXFP4 split FFN compile must engage");
    let g = qw(&x, &gate);
    let u = qw(&x, &up);
    let hidden_act = silu_mul(&g, &u, None);
    let portable = qw(&hidden_act, &down);
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    let a = compiled.data_f32();
    let b = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "compiled exact S=2 MXFP4 FFN must match imperative, max_abs={max_abs}"
    );
}

fn mxfp4_quant_weight(w: &MlxArray) -> QuantizedWeight {
    let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
    assert_eq!(q.len(), 2);
    QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: None,
        group_size: 32,
        bits: 4,
        mode: "mxfp4".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    }
}

#[test]
fn qwen_swiglu_down_fuse_mxfp4_matches_imperative() {
    let seq = 8i32;
    let hidden = 64i32;
    let intermediate = 64i32;
    let gate_data: Vec<f32> = (0..(seq * intermediate) as usize)
        .map(|i| ((i as f32) - 40.0) * 0.011)
        .collect();
    let up_data: Vec<f32> = (0..(seq * intermediate) as usize)
        .map(|i| ((i as f32) - 20.0) * -0.007)
        .collect();
    let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
        .map(|i| ((i as f32) - 12.0) * 0.006)
        .collect();
    let gate = array_f32(&gate_data, &[1, seq, intermediate]);
    let up = array_f32(&up_data, &[1, seq, intermediate]);
    let down = mxfp4_quant_weight(&array_f32(&down_data, &[hidden, intermediate]));
    let fused = qwen_swiglu_down_fuse(&gate, &up, &down)
        .expect("scales-only MXFP4 must take the fused SwiGLU-down path");
    let portable = qw(&silu_mul(&gate, &up, None), &down);
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    let a = fused.data_f32();
    let b = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "fused MXFP4 SwiGLU-down must match imperative, max_abs={max_abs}"
    );
}

#[test]
fn quantized_matmul_rms_norm_mxfp4_matches_imperative() {
    let seq = 4i32;
    let hidden = 64i32;
    let out_dim = 64i32;
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 30.0) * 0.013)
        .collect();
    let w_data: Vec<f32> = (0..(out_dim * hidden) as usize)
        .map(|i| ((i as f32) - 18.0) * 0.005)
        .collect();
    let norm_data: Vec<f32> = (0..out_dim as usize)
        .map(|i| 0.5 + (i as f32) * 0.01)
        .collect();
    let x = array_f32(&x_data, &[1, seq, hidden]);
    let proj = mxfp4_quant_weight(&array_f32(&w_data, &[out_dim, hidden]));
    let norm_w = array_f32(&norm_data, &[out_dim]);
    let fused = mlx_sys::ops::quantized_matmul_rms_norm(
        &x,
        &proj.weight,
        proj.scales.as_ref().expect("scales"),
        proj.biases.as_ref(),
        proj.group_size,
        proj.bits,
        &norm_w,
        1e-6,
        None,
    );
    let portable = rms_norm(&qw(&x, &proj), Some(&norm_w), 1e-6, None);
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    let a = fused.data_f32();
    let b = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "fused MXFP4 qmm+rms_norm must match imperative, max_abs={max_abs}"
    );
}

#[test]
fn switch_gather_sorted_mxfp4_experts_match_unsorted() {
    let experts = 4i32;
    let hidden = 64i32;
    let expert_out = 32i32;
    let seq = 4i32;
    let top_k = 2i32;
    let w_data: Vec<f32> = (0..(experts * expert_out * hidden) as usize)
        .map(|i| (((i % 613) as f32) - 300.0) * 0.003)
        .collect();
    let exps = mxfp4_quant_weight(&array_f32(&w_data, &[experts, expert_out, hidden]));
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| (((i % 251) as f32) - 120.0) * 0.008)
        .collect();
    let x = array_f32(&x_data, &[1, seq, hidden]);
    let x_exp = expand_dims_axes(&x, &[-2, -3], None);
    let idx_data: Vec<u32> = vec![3, 0, 1, 2, 2, 1, 0, 3];
    let indices = MlxArray::from_raw_data(
        idx_data.as_ptr() as *const u8,
        idx_data.len() * std::mem::size_of::<u32>(),
        &[1, seq, top_k],
        MlxDtype::Uint32,
    );
    let gather_inputs = switch_gather_inputs(&x_exp, &indices);
    assert!(
        gather_inputs.sorted_indices,
        "seq>1 with {} selections must take the sorted path",
        seq * top_k
    );
    let sorted = gather_inputs.unsort(squeeze_switch_singleton(&qw_gather(
        &gather_inputs.x,
        &exps,
        &gather_inputs.indices,
        true,
    )));
    let unsorted = squeeze_switch_singleton(&qw_gather(&x_exp, &exps, &indices, false));
    let unsorted = reshape(&unsorted, &[1, seq, top_k, expert_out], None);
    eval(&[&sorted, &unsorted]);
    assert_eq!(sorted.shape(), unsorted.shape());
    let a = sorted.data_f32();
    let b = unsorted.data_f32();
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "sorted MXFP4 expert gather must be bit-identical after unsort (row {i})"
        );
    }
}

#[test]
fn qwen_compiled_split_verify_ffn_plus_residual_s2_matches_imperative() {
    let seq = 2i32;
    let hidden = 64i32;
    let intermediate = 32i32;
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let attn_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 8.0) * -0.0078125)
        .collect();
    let norm_data: Vec<f32> = (0..hidden as usize)
        .map(|i| 0.75 + (i as f32) * 0.004)
        .collect();
    let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 16.0) * 0.01)
        .collect();
    let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 8.0) * -0.008)
        .collect();
    let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
        .map(|i| ((i as f32) - 12.0) * 0.006)
        .collect();
    let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
    let attn = array_f32(&attn_data, &[1, seq, hidden]);
    let ffn_norm = array_f32(&norm_data, &[hidden]);
    let qmx = |w: &MlxArray| {
        let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    };
    let mut cfg = v4_test_config(1, 1);
    cfg.model_family = "qwen3_5".to_string();
    cfg.compile_cache_identity = 0x5245_5349_4446_464E;
    let dummy = array_f32(&[0.0], &[1]);
    let mut w = v4_layer_weights(dummy, &hidden_x);
    w.router_proj = None;
    w.ffn_norm = ffn_norm.clone();
    w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
    w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
    w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let compiled = qwen_compiled_split_verify_ffn_plus_residual(&cfg, &w, &hidden_x, &attn, 0)
        .expect("exact S=2 residual+FFN compile must engage");
    let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
    let g = qw(&normed, w.gate_proj.as_ref().unwrap());
    let u = qw(&normed, w.up_proj.as_ref().unwrap());
    let act = silu_mul(&g, &u, None);
    let ffn = qw(&act, w.down_proj.as_ref().unwrap());
    let portable = add(&residual, &ffn, None);
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    let a = compiled.data_f32();
    let b = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "compiled residual+FFN must match add_rms+ffn+add, max_abs={max_abs}"
    );
    let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    assert!(
        qwen_compiled_split_verify_ffn_plus_residual(&cfg, &w, &hidden_x, &attn, 0).is_none(),
        "residual+FFN compile must stay off when exact MTP is scoped off"
    );
}

#[test]
fn qwen_compiled_split_verify_ffn_bf16_affine_matches_imperative() {
    let seq = 4i32;
    let hidden = 64i32;
    let intermediate = 32i32;
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.13)
        .collect();
    let attn_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i % 17) as f32 - 8.0) * -0.071)
        .collect();
    let norm_data: Vec<f32> = (0..hidden as usize)
        .map(|i| 0.75 + (i as f32) * 0.004)
        .collect();
    let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
        .collect();
    let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i % 29) as f32 - 14.0) * -0.023)
        .collect();
    let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.017)
        .collect();
    let hidden_x = astype(
        &array_f32(&x_data, &[1, seq, hidden]),
        MlxDtype::Bfloat16,
        None,
    );
    let attn = astype(
        &array_f32(&attn_data, &[1, seq, hidden]),
        MlxDtype::Bfloat16,
        None,
    );
    let ffn_norm = astype(&array_f32(&norm_data, &[hidden]), MlxDtype::Bfloat16, None);
    let qmx = |w: &MlxArray| {
        let w = astype(w, MlxDtype::Bfloat16, None);
        let q = quantize(
            &w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    };
    let mut cfg = v4_test_config(1, 1);
    cfg.model_family = "qwen3_5".to_string();
    cfg.compile_cache_identity = 0x4246_3136_4146_4634;
    let dummy = array_f32(&[0.0], &[1]);
    let mut w = v4_layer_weights(dummy, &hidden_x);
    w.router_proj = None;
    w.ffn_norm = ffn_norm.clone();
    w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
    w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
    w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let _verify = crate::fastpath::scoped_qwen_linear_mtp_target_verify(true);
    let compiled = qwen_compiled_split_verify_ffn_plus_residual(&cfg, &w, &hidden_x, &attn, 0)
        .expect("relaxed BF16 S=4 residual+FFN compile must engage");
    let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
    let g = qw(&normed, w.gate_proj.as_ref().unwrap());
    let u = qw(&normed, w.up_proj.as_ref().unwrap());
    let act = silu_mul(&g, &u, None);
    let ffn = qw(&act, w.down_proj.as_ref().unwrap());
    let portable = add(&residual, &ffn, None);
    let compiled = astype(&compiled, MlxDtype::Float32, None);
    let portable = astype(&portable, MlxDtype::Float32, None);
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    let a = compiled.data_f32();
    let b = portable.data_f32();
    assert_eq!(
        a, b,
        "BF16 compiled FFN must preserve imperative tensor rounding"
    );
}

#[test]
fn qwen_compiled_split_verify_o_proj_ffn_plus_residual_s2_matches_imperative() {
    let seq = 2i32;
    let hidden = 64i32;
    let value_dim = 64i32;
    let intermediate = 32i32;
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let gated_data: Vec<f32> = (0..(seq * value_dim) as usize)
        .map(|i| ((i as f32) - 6.0) * 0.03125)
        .collect();
    let o_data: Vec<f32> = (0..(hidden * value_dim) as usize)
        .map(|i| ((i as f32) - 20.0) * 0.004)
        .collect();
    let norm_data: Vec<f32> = (0..hidden as usize)
        .map(|i| 0.75 + (i as f32) * 0.004)
        .collect();
    let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 16.0) * 0.01)
        .collect();
    let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 8.0) * -0.008)
        .collect();
    let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
        .map(|i| ((i as f32) - 12.0) * 0.006)
        .collect();
    let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
    let gated = array_f32(&gated_data, &[1, seq, value_dim]);
    let ffn_norm = array_f32(&norm_data, &[hidden]);
    let qmx = |w: &MlxArray| {
        let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    };
    let mut cfg = v4_test_config(1, 1);
    cfg.model_family = "qwen3_5".to_string();
    cfg.compile_cache_identity = 0x4F50_524F_4A46_464E;
    let dummy = array_f32(&[0.0], &[1]);
    let mut w = v4_layer_weights(dummy.clone(), &hidden_x);
    w.router_proj = None;
    w.ffn_norm = ffn_norm.clone();
    w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
    w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
    w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
    let out_proj = qmx(&array_f32(&o_data, &[hidden, value_dim]));
    w.linear_attn = Some(crate::weights::LinearAttentionWeights {
        in_proj_qkv: None,
        in_proj_z: None,
        in_proj_a: None,
        in_proj_b: None,
        in_proj_qkvz: None,
        in_proj_ba: None,
        fused_qkvz_ba: None,
        prefill_q2_qkvz: None,
        prefill_q2_ba: None,
        conv1d_dense: dummy.clone(),
        conv1d_bias: None,
        dt_bias: dummy.clone(),
        a_log: dummy,
        d: None,
        norm: ffn_norm.clone(),
        out_proj,
    });
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let compiled =
        qwen_compiled_split_verify_o_proj_ffn_plus_residual(&cfg, &w, &hidden_x, &gated, 0)
            .expect("exact S=2 o_proj+residual+FFN compile must engage");
    let attn = qw(&gated, &w.linear_attn.as_ref().unwrap().out_proj);
    let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
    let g = qw(&normed, w.gate_proj.as_ref().unwrap());
    let u = qw(&normed, w.up_proj.as_ref().unwrap());
    let act = silu_mul(&g, &u, None);
    let ffn = qw(&act, w.down_proj.as_ref().unwrap());
    let portable = add(&residual, &ffn, None);
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    let a = compiled.data_f32();
    let b = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "compiled o_proj+residual+FFN must match qw+add_rms+ffn+add, max_abs={max_abs}"
    );
    let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    assert!(
        qwen_compiled_split_verify_o_proj_ffn_plus_residual(&cfg, &w, &hidden_x, &gated, 0)
            .is_none(),
        "o_proj+FFN compile must stay off when exact MTP is scoped off"
    );
}

#[test]
fn qwen_compiled_split_verify_fa_o_proj_ffn_s2_matches_imperative() {
    let seq = 2i32;
    let n_heads = 2usize;
    let head_dim = 32usize;
    let value_dim = (n_heads * head_dim) as i32;
    let hidden = 64i32;
    let intermediate = 32i32;
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let sdpa_data: Vec<f32> = (0..(n_heads * seq as usize * head_dim))
        .map(|i| ((i as f32) - 10.0) * 0.0234375)
        .collect();
    let o_data: Vec<f32> = (0..(hidden * value_dim) as usize)
        .map(|i| ((i as f32) - 20.0) * 0.004)
        .collect();
    let norm_data: Vec<f32> = (0..hidden as usize)
        .map(|i| 0.75 + (i as f32) * 0.004)
        .collect();
    let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 16.0) * 0.01)
        .collect();
    let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 8.0) * -0.008)
        .collect();
    let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
        .map(|i| ((i as f32) - 12.0) * 0.006)
        .collect();
    let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
    let attn_sdpa = array_f32(&sdpa_data, &[1, n_heads as i32, seq, head_dim as i32]);
    let ffn_norm = array_f32(&norm_data, &[hidden]);
    let qmx = |w: &MlxArray| {
        let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    };
    let mut cfg = v4_test_config(1, 1);
    cfg.model_family = "qwen3_5".to_string();
    cfg.compile_cache_identity = 0x4641_4F50_4646_4E32;
    cfg.n_heads = n_heads;
    cfg.head_dim = head_dim;
    let dummy = array_f32(&[0.0], &[1]);
    let mut w = v4_layer_weights(dummy, &hidden_x);
    w.router_proj = None;
    w.ffn_norm = ffn_norm.clone();
    w.o_proj = Some(qmx(&array_f32(&o_data, &[hidden, value_dim])));
    w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
    w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
    w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let compiled = qwen_compiled_split_verify_fa_o_proj_ffn(
        &cfg,
        &w,
        &hidden_x,
        &attn_sdpa,
        0,
        seq as usize,
        n_heads,
        head_dim,
    )
    .expect("exact S=2 FA flatten+o_proj+FFN compile must engage");
    let transposed = transpose(&attn_sdpa, &[0, 2, 1, 3], None);
    let flat = reshape(&transposed, &[1, seq, value_dim], None);
    let attn = qw(&flat, w.o_proj.as_ref().unwrap());
    let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
    let g = qw(&normed, w.gate_proj.as_ref().unwrap());
    let u = qw(&normed, w.up_proj.as_ref().unwrap());
    let act = silu_mul(&g, &u, None);
    let ffn = qw(&act, w.down_proj.as_ref().unwrap());
    let portable = add(&residual, &ffn, None);
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    let a = compiled.data_f32();
    let b = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "compiled FA flatten+o_proj+FFN must match transpose+qw+add_rms+ffn, max_abs={max_abs}"
    );
    let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    assert!(
        qwen_compiled_split_verify_fa_o_proj_ffn(
            &cfg,
            &w,
            &hidden_x,
            &attn_sdpa,
            0,
            seq as usize,
            n_heads,
            head_dim,
        )
        .is_none(),
        "FA o_proj+FFN compile must stay off when exact MTP is scoped off"
    );
}

#[test]
fn qwen_compiled_split_verify_la_gate_o_proj_s2_matches_imperative() {
    let seq = 2i32;
    let hv = 2i32;
    let dv = 32i32;
    let value_dim = hv * dv;
    let n = (seq * hv * dv) as usize;
    let gd_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
    let z_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 8.0) * 0.015625).collect();
    let norm_data: Vec<f32> = (0..dv as usize)
        .map(|i| 0.75 + (i as f32) * 0.004)
        .collect();
    let o_data: Vec<f32> = (0..(value_dim * value_dim) as usize)
        .map(|i| ((i as f32) - 20.0) * 0.004)
        .collect();
    let gd = astype(
        &array_f32(&gd_data, &[1, seq, hv, dv]),
        MlxDtype::Bfloat16,
        None,
    );
    let z = astype(
        &array_f32(&z_data, &[1, seq, hv, dv]),
        MlxDtype::Bfloat16,
        None,
    );
    let la_norm = astype(&array_f32(&norm_data, &[dv]), MlxDtype::Bfloat16, None);
    let qmx = |w: &MlxArray| {
        let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    };
    let mut cfg = v4_test_config(1, 1);
    cfg.model_family = "qwen3_5".to_string();
    cfg.compile_cache_identity = 0x4C41_4741_5445_4F50;
    let dummy = array_f32(&[0.0], &[1]);
    let mut w = v4_layer_weights(dummy.clone(), &array_f32(&[0.0; 2], &[1, 2]));
    w.router_proj = None;
    let out_proj = qmx(&array_f32(&o_data, &[value_dim, value_dim]));
    w.linear_attn = Some(crate::weights::LinearAttentionWeights {
        in_proj_qkv: None,
        in_proj_z: None,
        in_proj_a: None,
        in_proj_b: None,
        in_proj_qkvz: None,
        in_proj_ba: None,
        fused_qkvz_ba: None,
        prefill_q2_qkvz: None,
        prefill_q2_ba: None,
        conv1d_dense: dummy.clone(),
        conv1d_bias: None,
        dt_bias: dummy.clone(),
        a_log: dummy,
        d: None,
        norm: la_norm.clone(),
        out_proj: out_proj.clone(),
    });
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let compiled = qwen_compiled_split_verify_la_gate_o_proj(&cfg, &w, &gd, &z, 0, seq, value_dim)
        .expect("exact S=2 LA gate+o_proj compile must engage");
    let normed = rms_norm(&gd, Some(&la_norm), 1e-6, None);
    let gated = astype(
        &silu_mul(
            &astype(&z, MlxDtype::Float32, None),
            &astype(&normed, MlxDtype::Float32, None),
            None,
        ),
        gd.dtype(),
        None,
    );
    let flat = reshape(&gated, &[1, seq, value_dim], None);
    let portable = qw(&flat, &out_proj);
    eval(&[&compiled, &portable]);
    let a = astype(&compiled, MlxDtype::Float32, None);
    let b = astype(&portable, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    let mut max_abs = 0.0f32;
    for (l, r) in a.data_f32().iter().zip(b.data_f32().iter()) {
        max_abs = max_abs.max((l - r).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "compiled LA gate+o_proj must match rms+silu+reshape+qw, max_abs={max_abs}"
    );
    let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    assert!(
        qwen_compiled_split_verify_la_gate_o_proj(&cfg, &w, &gd, &z, 0, seq, value_dim).is_none(),
        "LA gate+o_proj compile must stay off when exact MTP is scoped off"
    );
}

#[test]
fn qwen_compiled_split_verify_la_gate_o_proj_ffn_s2_matches_imperative() {
    let seq = 2i32;
    let hv = 2i32;
    let dv = 32i32;
    let value_dim = hv * dv;
    let hidden = 64i32;
    let intermediate = 32i32;
    let n = (seq * hv * dv) as usize;
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let gd_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
    let z_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 8.0) * 0.015625).collect();
    let la_norm_data: Vec<f32> = (0..dv as usize)
        .map(|i| 0.75 + (i as f32) * 0.004)
        .collect();
    let ffn_norm_data: Vec<f32> = (0..hidden as usize)
        .map(|i| 0.8 + (i as f32) * 0.003)
        .collect();
    let o_data: Vec<f32> = (0..(hidden * value_dim) as usize)
        .map(|i| ((i as f32) - 20.0) * 0.004)
        .collect();
    let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 16.0) * 0.01)
        .collect();
    let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
        .map(|i| ((i as f32) - 8.0) * -0.008)
        .collect();
    let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
        .map(|i| ((i as f32) - 12.0) * 0.006)
        .collect();
    let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
    let gd = astype(
        &array_f32(&gd_data, &[1, seq, hv, dv]),
        MlxDtype::Bfloat16,
        None,
    );
    let z = astype(
        &array_f32(&z_data, &[1, seq, hv, dv]),
        MlxDtype::Bfloat16,
        None,
    );
    let la_norm = astype(&array_f32(&la_norm_data, &[dv]), MlxDtype::Bfloat16, None);
    let ffn_norm = array_f32(&ffn_norm_data, &[hidden]);
    let qmx = |w: &MlxArray| {
        let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    };
    let mut cfg = v4_test_config(1, 1);
    cfg.model_family = "qwen3_5".to_string();
    cfg.compile_cache_identity = 0x4C41_474F_4646_4E32;
    let dummy = array_f32(&[0.0], &[1]);
    let mut w = v4_layer_weights(dummy.clone(), &hidden_x);
    w.router_proj = None;
    w.ffn_norm = ffn_norm.clone();
    w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
    w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
    w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
    let out_proj = qmx(&array_f32(&o_data, &[hidden, value_dim]));
    w.linear_attn = Some(crate::weights::LinearAttentionWeights {
        in_proj_qkv: None,
        in_proj_z: None,
        in_proj_a: None,
        in_proj_b: None,
        in_proj_qkvz: None,
        in_proj_ba: None,
        fused_qkvz_ba: None,
        prefill_q2_qkvz: None,
        prefill_q2_ba: None,
        conv1d_dense: dummy.clone(),
        conv1d_bias: None,
        dt_bias: dummy.clone(),
        a_log: dummy,
        d: None,
        norm: la_norm.clone(),
        out_proj: out_proj.clone(),
    });
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let compiled = qwen_compiled_split_verify_la_gate_o_proj_ffn(
        &cfg, &w, &hidden_x, &gd, &z, 0, seq, value_dim,
    )
    .expect("exact S=2 LA gate+o_proj+FFN compile must engage");
    let normed = rms_norm(&gd, Some(&la_norm), 1e-6, None);
    let gated = astype(
        &silu_mul(
            &astype(&z, MlxDtype::Float32, None),
            &astype(&normed, MlxDtype::Float32, None),
            None,
        ),
        gd.dtype(),
        None,
    );
    let flat = reshape(&gated, &[1, seq, value_dim], None);
    let attn = qw(&flat, &out_proj);
    let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
    let g = qw(&normed, w.gate_proj.as_ref().unwrap());
    let u = qw(&normed, w.up_proj.as_ref().unwrap());
    let act = silu_mul(&g, &u, None);
    let ffn = qw(&act, w.down_proj.as_ref().unwrap());
    let portable = add(&residual, &ffn, None);
    eval(&[&compiled, &portable]);
    let a = astype(&compiled, MlxDtype::Float32, None);
    let b = astype(&portable, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    let mut max_abs = 0.0f32;
    for (l, r) in a.data_f32().iter().zip(b.data_f32().iter()) {
        max_abs = max_abs.max((l - r).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "compiled LA gate+o_proj+FFN must match imperative, max_abs={max_abs}"
    );
}

#[test]
fn qwen_compiled_split_verify_fa_attn_norm_qkv_s2_matches_imperative() {
    let seq = 2i32;
    let hidden = 64i32;
    let q_out = 64i32;
    let kv_out = 32i32;
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let norm_data: Vec<f32> = (0..hidden as usize)
        .map(|i| 0.75 + (i as f32) * 0.004)
        .collect();
    let q_data: Vec<f32> = (0..(q_out * hidden) as usize)
        .map(|i| ((i as f32) - 10.0) * 0.004)
        .collect();
    let k_data: Vec<f32> = (0..(kv_out * hidden) as usize)
        .map(|i| ((i as f32) - 6.0) * 0.005)
        .collect();
    let v_data: Vec<f32> = (0..(kv_out * hidden) as usize)
        .map(|i| ((i as f32) - 4.0) * -0.003)
        .collect();
    let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
    let attn_norm = array_f32(&norm_data, &[hidden]);
    let qmx = |w: &MlxArray| {
        let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    };
    let mut cfg = v4_test_config(1, 1);
    cfg.model_family = "qwen3_5".to_string();
    cfg.compile_cache_identity = 0x4641_514B_5652_4D53;
    cfg.attn_output_gate = false;
    let dummy = array_f32(&[0.0], &[1]);
    let mut w = v4_layer_weights(dummy, &hidden_x);
    w.router_proj = None;
    w.attn_norm = attn_norm.clone();
    w.q_proj = Some(qmx(&array_f32(&q_data, &[q_out, hidden])));
    w.k_proj = Some(qmx(&array_f32(&k_data, &[kv_out, hidden])));
    w.v_proj = Some(qmx(&array_f32(&v_data, &[kv_out, hidden])));
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let (cq, ck, cv) =
        qwen_compiled_split_verify_fa_attn_norm_qkv(&cfg, &w, &hidden_x, 0, seq as usize)
            .expect("exact S=2 FA attn_norm+QKV compile must engage");
    let normed = rms_norm(&hidden_x, Some(&attn_norm), 1e-6, None);
    let pq = qw(&normed, w.q_proj.as_ref().unwrap());
    let pk = qw(&normed, w.k_proj.as_ref().unwrap());
    let pv = qw(&normed, w.v_proj.as_ref().unwrap());
    eval(&[&cq, &ck, &cv, &pq, &pk, &pv]);
    let max_abs = |a: &MlxArray, b: &MlxArray| {
        a.data_f32()
            .iter()
            .zip(b.data_f32().iter())
            .fold(0.0f32, |m, (l, r)| m.max((l - r).abs()))
    };
    assert!(
        max_abs(&cq, &pq) < 1.0e-5 && max_abs(&ck, &pk) < 1.0e-5 && max_abs(&cv, &pv) < 1.0e-5,
        "compiled FA attn_norm+QKV must match rms+qw, q={} k={} v={}",
        max_abs(&cq, &pq),
        max_abs(&ck, &pk),
        max_abs(&cv, &pv)
    );
    let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    assert!(
        qwen_compiled_split_verify_fa_attn_norm_qkv(&cfg, &w, &hidden_x, 0, seq as usize).is_none(),
        "FA attn_norm+QKV compile must stay off when exact MTP is scoped off"
    );
}

#[test]
fn qwen_compiled_prefill_down_qmm_matches_qw_4bit_gs32() {
    let hidden_data: Vec<f32> = (0..128 * 32)
        .map(|i| ((i as f32) - 2048.0) * 0.00048828125)
        .collect();
    let down_data: Vec<f32> = (0..64 * 32)
        .map(|i| ((i as f32) - 768.0) * 0.0003)
        .collect();
    let hidden = array_f32(&hidden_data, &[1, 128, 32]);
    let down_w = array_f32(&down_data, &[64, 32]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let compiled =
        qwen_compiled_prefill_down_qmm_for(true, 0x444F_574E_5052_4546, 3, &hidden, &down)
            .expect("down-only prefill compile should engage at leading=128");
    let portable = qw(&hidden, &down);
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    assert_close(compiled.data_f32(), portable.data_f32(), 3.0e-2);
    let decode = array_f32(&hidden_data[..32], &[1, 1, 32]);
    assert!(
        qwen_compiled_prefill_down_qmm_for(true, 0x444F_574E_5052_4546, 3, &decode, &down)
            .is_none(),
        "down-only prefill compile must reject decode seq==1"
    );
    assert!(
        qwen_compiled_prefill_down_qmm(0x444F_574E_5052_4546, 3, &hidden, &down).is_none(),
        "default-off compile flag must keep the imperative down qmm"
    );
}

#[test]
fn cached_prefill_q2_down_requants_4bit_and_qws() {
    let hidden_data: Vec<f32> = (0..32 * 32)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let down_data: Vec<f32> = (0..64 * 32)
        .map(|i| ((i as f32) - 768.0) * 0.0003)
        .collect();
    let hidden = array_f32(&hidden_data, &[1, 32, 32]);
    let down_w = array_f32(&down_data, &[64, 32]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let q2 = cached_prefill_q2_down(0x5132_444F_574E, 7, &down)
        .expect("4-bit gs32 down must grow a 2-bit overlay");
    assert_eq!(q2.bits, crate::weights::PREFILL_LA_Q2_BITS);
    assert_eq!(q2.group_size, crate::weights::PREFILL_LA_Q2_GROUP_SIZE);
    let again = cached_prefill_q2_down(0x5132_444F_574E, 7, &down)
        .expect("second lookup must hit the overlay cache");
    assert_eq!(again.bits, q2.bits);
    let out = qw(&hidden, &q2);
    eval(&[&out]);
    assert_eq!(out.shape(), vec![1, 32, 64]);
    assert!(
        out.data_f32().iter().all(|v| v.is_finite()),
        "2-bit down qmm must produce finite values"
    );
    assert!(
        fastpath::should_qwen_prefill_q2_down_for(true, 1024),
        "shipped down q2 gate must accept the p2048 chunk length"
    );
}

#[test]
fn cached_prefill_ffn_gs64_requants_4bit_gs32_and_qws() {
    let hidden_data: Vec<f32> = (0..64 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let down_data: Vec<f32> = (0..64 * 64)
        .map(|i| ((i as f32) - 768.0) * 0.0003)
        .collect();
    let hidden = array_f32(&hidden_data, &[1, 64, 64]);
    let down_w = array_f32(&down_data, &[64, 64]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let gs64 = cached_prefill_ffn_gs64(0x4753_3634, 3, PREFILL_FFN_GS64_DOWN, &down)
        .expect("4-bit gs32 down must grow a gs64 overlay");
    assert_eq!(gs64.bits, 4);
    assert_eq!(gs64.group_size, crate::weights::PREFILL_FFN_GS64_GROUP_SIZE);
    let again = cached_prefill_ffn_gs64(0x4753_3634, 3, PREFILL_FFN_GS64_DOWN, &down)
        .expect("second lookup must hit the overlay cache");
    assert_eq!(again.group_size, gs64.group_size);
    let out = qw(&hidden, &gs64);
    eval(&[&out]);
    assert_eq!(out.shape(), vec![1, 64, 64]);
    assert!(
        out.data_f32().iter().all(|v| v.is_finite()),
        "gs64 down qmm must produce finite values"
    );
    assert!(
        fastpath::should_qwen_prefill_ffn_gs64_for(true, 1024),
        "shipped FFN gs64 gate must accept the p2048 chunk length"
    );
}

#[test]
fn cached_prefill_ffn_q3_requants_4bit_and_qws() {
    let hidden_data: Vec<f32> = (0..32 * 32)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let down_data: Vec<f32> = (0..64 * 32)
        .map(|i| ((i as f32) - 768.0) * 0.0003)
        .collect();
    let hidden = array_f32(&hidden_data, &[1, 32, 32]);
    let down_w = array_f32(&down_data, &[64, 32]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let q3 = cached_prefill_ffn_q3(0x5133_4646_4e51, 5, PREFILL_FFN_GS64_DOWN, &down)
        .expect("4-bit gs32 down must grow a 3-bit overlay");
    assert_eq!(q3.bits, crate::weights::PREFILL_FFN_Q3_BITS);
    assert_eq!(q3.group_size, crate::weights::PREFILL_FFN_Q3_GROUP_SIZE);
    let again = cached_prefill_ffn_q3(0x5133_4646_4e51, 5, PREFILL_FFN_GS64_DOWN, &down)
        .expect("second lookup must hit the overlay cache");
    assert_eq!(again.bits, q3.bits);
    let out = qw(&hidden, &q3);
    eval(&[&out]);
    assert_eq!(out.shape(), vec![1, 32, 64]);
    assert!(
        out.data_f32().iter().all(|v| v.is_finite()),
        "3-bit down qmm must produce finite values"
    );
    assert!(
        fastpath::should_qwen_prefill_q3_ffn_for(true, 1024),
        "shipped FFN q3 gate must accept the p2048 chunk length"
    );
}

#[test]
fn cached_prefill_ffn_contiguous_weight_keeps_bits_and_qws() {
    let hidden_data: Vec<f32> = (0..32 * 32)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let down_data: Vec<f32> = (0..64 * 32)
        .map(|i| ((i as f32) - 768.0) * 0.0003)
        .collect();
    let hidden = array_f32(&hidden_data, &[1, 32, 32]);
    let down_w = array_f32(&down_data, &[64, 32]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let contig = cached_prefill_ffn_contiguous_weight(0x434f_4e54, 2, PREFILL_FFN_GS64_DOWN, &down)
        .expect("affine down must grow a contiguous overlay");
    assert_eq!(contig.bits, 4);
    assert_eq!(contig.group_size, 32);
    let again = cached_prefill_ffn_contiguous_weight(0x434f_4e54, 2, PREFILL_FFN_GS64_DOWN, &down)
        .expect("second lookup must hit the overlay cache");
    assert_eq!(again.bits, contig.bits);
    let out = qw(&hidden, &contig);
    eval(&[&out]);
    assert_eq!(out.shape(), vec![1, 32, 64]);
    assert!(
        out.data_f32().iter().all(|v| v.is_finite()),
        "contiguous-weight qmm must produce finite values"
    );
    assert!(
        fastpath::should_qwen_prefill_contiguous_ffn_weights_for(true, 1024),
        "shipped FFN contiguous-weight gate must accept the p2048 chunk length"
    );
}

#[test]
fn qwen_prefill_maybe_async_gate_up_submits_pair_at_min_seq() {
    let gate_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
    let up_data: Vec<f32> = (0..32).map(|i| (i as f32) * -0.01).collect();
    let gate = array_f32(&gate_data, &[1, 32, 1]);
    let up = array_f32(&up_data, &[1, 32, 1]);
    qwen_prefill_maybe_async_gate_up(&gate, &up, true, 1024);
    eval(&[&gate, &up]);
    assert_eq!(gate.shape(), vec![1, 32, 1]);
    assert_eq!(up.shape(), vec![1, 32, 1]);
    assert!(
        fastpath::should_qwen_prefill_async_gate_up_for(true, 1024),
        "shipped async gate/up gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_async_gate_up(&gate, &up, false, 1024);
    qwen_prefill_maybe_async_gate_up(&gate, &up, true, 512);
}

#[test]
fn qwen_prefill_maybe_async_packed_gate_up_submits_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let packed = array_f32(&data, &[1, 32, 1]);
    qwen_prefill_maybe_async_packed_gate_up_for(&packed, true, true, 1024);
    eval(&[&packed]);
    assert_eq!(packed.shape(), vec![1, 32, 1]);
    assert!(
        packed.data_f32().iter().all(|v| v.is_finite()),
        "async packed gate/up must leave a finite materialized tensor"
    );
    assert!(
        fastpath::should_qwen_prefill_async_packed_gate_up_for(true, 1024),
        "shipped async packed-gate/up gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_async_packed_gate_up_for(&packed, false, true, 1024);
    qwen_prefill_maybe_async_packed_gate_up_for(&packed, true, false, 1024);
    qwen_prefill_maybe_async_packed_gate_up_for(&packed, true, true, 512);
}

#[test]
fn qwen_prefill_maybe_eval_ffn_hidden_materializes_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let h = array_f32(&data, &[1, 32, 1]);
    qwen_prefill_maybe_eval_ffn_hidden_for(&h, true, true, 1024);
    eval(&[&h]);
    assert_eq!(h.shape(), vec![1, 32, 1]);
    assert!(
        h.data_f32().iter().all(|v| v.is_finite()),
        "eval-ffn-hidden must leave a finite materialized activation"
    );
    assert!(
        fastpath::should_qwen_prefill_eval_ffn_hidden_for(true, 1024),
        "shipped FFN hidden-eval gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_eval_ffn_hidden_for(&h, false, true, 1024);
    qwen_prefill_maybe_eval_ffn_hidden_for(&h, true, false, 1024);
    qwen_prefill_maybe_eval_ffn_hidden_for(&h, true, true, 512);
}

#[test]
fn qwen_prefill_maybe_async_down_submits_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let down = array_f32(&data, &[1, 32, 1]);
    qwen_prefill_maybe_async_down_for(&down, true, true, 1024);
    eval(&[&down]);
    assert_eq!(down.shape(), vec![1, 32, 1]);
    assert!(
        down.data_f32().iter().all(|v| v.is_finite()),
        "async down must leave a finite materialized tensor"
    );
    assert!(
        fastpath::should_qwen_prefill_async_down_for(true, 1024),
        "shipped async-down gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_async_down_for(&down, false, true, 1024);
    qwen_prefill_maybe_async_down_for(&down, true, false, 1024);
    qwen_prefill_maybe_async_down_for(&down, true, true, 512);
}

#[test]
fn cached_prefill_attn_contiguous_weight_keeps_bits_and_qws() {
    let hidden_data: Vec<f32> = (0..32 * 32)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let proj_data: Vec<f32> = (0..64 * 32)
        .map(|i| ((i as f32) - 768.0) * 0.0003)
        .collect();
    let hidden = array_f32(&hidden_data, &[1, 32, 32]);
    let proj_w = array_f32(&proj_data, &[64, 32]);
    let dq = quantize(
        &proj_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let src = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let contig = cached_prefill_attn_contiguous_weight(&src);
    assert_eq!(contig.bits, 4);
    assert_eq!(contig.group_size, 32);
    let again = cached_prefill_attn_contiguous_weight(&src);
    assert_eq!(again.bits, contig.bits);
    let out = qw(&hidden, &contig);
    eval(&[&out]);
    assert_eq!(out.shape(), vec![1, 32, 64]);
    assert!(
        out.data_f32().iter().all(|v| v.is_finite()),
        "contiguous attn-weight qmm must produce finite values"
    );
    assert!(
        fastpath::should_qwen_prefill_contiguous_attn_weights_for(true, "qwen3_5", 1024),
        "shipped attn contiguous-weight gate must accept the p2048 chunk length"
    );
}

#[test]
fn qwen_prefill_ffn_f32_input_promotes_bf16_at_min_seq() {
    let data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.125).collect();
    let bf16 = astype(&array_f32(&data, &[1, 8, 1]), MlxDtype::Bfloat16, None);
    eval(&[&bf16]);
    assert_eq!(bf16.dtype(), MlxDtype::Bfloat16);
    let (promoted, restore) = qwen_prefill_ffn_f32_input_for(&bf16, true, true, 1024);
    eval(&[&promoted]);
    assert_eq!(promoted.dtype(), MlxDtype::Float32);
    assert_eq!(restore, Some(MlxDtype::Bfloat16));
    let back = qwen_prefill_ffn_restore_dtype(&promoted, restore);
    eval(&[&back]);
    assert_eq!(back.dtype(), MlxDtype::Bfloat16);
    let (short, short_restore) = qwen_prefill_ffn_f32_input_for(&bf16, true, true, 512);
    assert!(short_restore.is_none());
    assert_eq!(short.dtype(), MlxDtype::Bfloat16);
    let (other, other_restore) = qwen_prefill_ffn_f32_input_for(&bf16, true, false, 1024);
    assert!(other_restore.is_none());
    assert_eq!(other.dtype(), MlxDtype::Bfloat16);
    assert!(
        fastpath::should_qwen_prefill_ffn_f32_input_for(true, 1024),
        "shipped FFN f32-input gate must accept the p2048 chunk length"
    );
}

#[test]
fn qwen_prefill_maybe_eval_ffn_input_materializes_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let x = array_f32(&data, &[1, 32, 1]);
    qwen_prefill_maybe_eval_ffn_input_for(&x, true, true, 1024);
    eval(&[&x]);
    assert_eq!(x.shape(), vec![1, 32, 1]);
    assert!(
        x.data_f32().iter().all(|v| v.is_finite()),
        "eval-ffn-input must leave a finite materialized activation"
    );
    assert!(
        fastpath::should_qwen_prefill_eval_ffn_input_for(true, 1024),
        "shipped FFN input-eval gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_eval_ffn_input_for(&x, false, true, 1024);
    qwen_prefill_maybe_eval_ffn_input_for(&x, true, false, 1024);
    qwen_prefill_maybe_eval_ffn_input_for(&x, true, true, 512);
}

#[test]
fn qwen_prefill_dual_qmm_swiglu_metal_matches_two_qmm_silu_mul_4bit_gs32() {
    let x_data: Vec<f32> = (0..8 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let gate_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0005)
        .collect();
    let up_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 512.0) * -0.0004)
        .collect();
    let x = array_f32(&x_data, &[1, 8, 64]);
    let gate_w = array_f32(&gate_data, &[32, 64]);
    let up_w = array_f32(&up_data, &[32, 64]);
    let gq = quantize(
        &gate_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let uq = quantize(
        &up_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(gq.len(), 3);
    assert_eq!(uq.len(), 3);
    let qweight = |q: &[MlxArray]| QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let gate = qweight(&gq);
    let up = qweight(&uq);
    let metal = qwen_prefill_dual_qmm_swiglu_metal(&x, &gate, &up)
        .expect("Qwen 4-bit gs32 prefill dual qmm Metal should engage");
    let p_gate = quantized_matmul(
        &x,
        &gq[0],
        &gq[1],
        Some(&gq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let p_up = quantized_matmul(
        &x,
        &uq[0],
        &uq[1],
        Some(&uq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let portable = silu_mul(&p_gate, &p_up, None);
    eval(&[&metal, &portable]);
    assert_eq!(metal.shape(), portable.shape());
    assert_close(metal.data_f32(), portable.data_f32(), 5.0e-2);

    // Low-precision rows must be bit-exact against the split path. One-hot
    // tokens and binary-exact affine scales remove reduction-order effects,
    // so any difference would come from the activation boundaries.
    let onehot: Vec<f32> = (0..8 * 64)
        .map(|i| if i % 64 == (i / 64) * 7 + 2 { 1.0 } else { 0.0 })
        .collect();
    let exact_gate: Vec<f32> = (0..32 * 64)
        .map(|i| (((i + i / 64) % 16) as f32 - 8.0) * 0.125)
        .collect();
    let exact_up: Vec<f32> = (0..32 * 64)
        .map(|i| (7.0 - ((i + 3 * (i / 64)) % 16) as f32) * 0.125)
        .collect();
    for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
        let x = astype(&array_f32(&onehot, &[1, 8, 64]), dtype, None);
        let gq = quantize(
            &astype(&array_f32(&exact_gate, &[32, 64]), dtype, None),
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let uq = quantize(
            &astype(&array_f32(&exact_up, &[32, 64]), dtype, None),
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let metal = qwen_prefill_dual_qmm_swiglu_metal(&x, &qweight(&gq), &qweight(&uq))
            .expect("Qwen prefill dual qmm Metal should engage for low-precision rows");
        let p_gate = quantized_matmul(
            &x,
            &gq[0],
            &gq[1],
            Some(&gq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let p_up = quantized_matmul(
            &x,
            &uq[0],
            &uq[1],
            Some(&uq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let portable = silu_mul(&p_gate, &p_up, None);
        assert_eq!(metal.dtype(), dtype);
        let metal = astype(&metal, MlxDtype::Float32, None);
        let portable = astype(&portable, MlxDtype::Float32, None);
        eval(&[&metal, &portable]);
        assert_eq!(metal.data_f32(), portable.data_f32());
    }
    let decode = array_f32(&x_data[..64], &[1, 1, 64]);
    assert!(
        qwen_prefill_dual_qmm_swiglu_metal(&decode, &gate, &up).is_none(),
        "prefill dual qmm Metal must reject decode seq==1"
    );
}

#[test]
fn qwen_attn_norm_qkv_fuse_matches_rms_then_qw_4bit_gs32() {
    // Shipped Qwen full-attn fuse: rms_norm_quantized_matmul vs rms + qw.
    let x_data: Vec<f32> = (0..8 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let w_data: Vec<f32> = (0..96 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0004)
        .collect();
    let x = array_f32(&x_data, &[1, 8, 64]);
    let weight = array_f32(&w_data, &[96, 64]);
    let qw_q = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let norm_w = array_f32(&vec![1.0f32; 64], &[64]);
    let fused = rms_norm_quantized_matmul(
        &x,
        &norm_w,
        1e-6,
        &qw_q[0],
        &qw_q[1],
        Some(&qw_q[2]),
        32,
        4,
        None,
    );
    let normed = rms_norm(&x, Some(&norm_w), 1e-6, None);
    let portable = quantized_matmul(
        &normed,
        &qw_q[0],
        &qw_q[1],
        Some(&qw_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
    assert!(fastpath::should_attn_norm_qkv_fuse_for(
        true, false, false, "qwen3_5", 128
    ));
    assert!(!fastpath::should_attn_norm_qkv_fuse_for(
        true, false, false, "gemma4", 512
    ));
    assert!(
        fastpath::should_gemma4_attn_norm_qkv_fuse_p128_for(true, "gemma4", 128),
        "AXQ p128 packed QKV must take the attn-norm fuse"
    );
}

#[test]
fn gemma4_attn_norm_qkv_fuse_p128_matches_rms_then_qw_4bit_gs32() {
    // Shipped Gemma 4 p128 fuse: same C++ rms_norm_quantized_matmul as
    // the Qwen path, on the contract seq=128 / AXQ gs=32 layout.
    let hidden = 64;
    let seq = 128;
    let x_data: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) - 4096.0) * 0.0009765625)
        .collect();
    let w_data: Vec<f32> = (0..96 * hidden)
        .map(|i| ((i as f32) - 2048.0) * 0.0004)
        .collect();
    let x = array_f32(&x_data, &[1, seq as i32, hidden as i32]);
    let weight = array_f32(&w_data, &[96, hidden as i32]);
    let qw_q = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let norm_w = array_f32(&vec![1.0f32; hidden], &[hidden as i32]);
    let fused = rms_norm_quantized_matmul(
        &x,
        &norm_w,
        1e-6,
        &qw_q[0],
        &qw_q[1],
        Some(&qw_q[2]),
        32,
        4,
        None,
    );
    let normed = rms_norm(&x, Some(&norm_w), 1e-6, None);
    let portable = quantized_matmul(
        &normed,
        &qw_q[0],
        &qw_q[1],
        Some(&qw_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
    assert!(fastpath::should_call_attn_norm_qkv_fuse(
        fastpath::should_attn_norm_qkv_fuse_for(false, false, true, "gemma4", 128),
        true,
        false,
        false,
    ));
}

#[test]
fn qwen_prefill_contiguous_ffn_qw_matches_view_4bit_gs32() {
    // Shipped path: contiguous([B,S,H]) then qw must match qw on the view.
    let full: Vec<f32> = (0..16 * 64)
        .map(|i| ((i as f32) - 512.0) * 0.0009765625)
        .collect();
    let down_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0004)
        .collect();
    let wide = array_f32(&full, &[1, 16, 64]);
    let view = slice(&wide, &[0, 4, 0], &[1, 12, 64], &[1, 1, 1], None);
    assert_eq!(view.shape(), vec![1, 8, 64]);
    let down_w = array_f32(&down_data, &[32, 64]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let packed = qw(&contiguous(&view, None), &down);
    let portable = qw(&view, &down);
    eval(&[&packed, &portable]);
    assert_eq!(packed.shape(), portable.shape());
    assert_close(packed.data_f32(), portable.data_f32(), 3.0e-2);
    assert!(fastpath::should_qwen_prefill_contiguous_ffn_for(
        true, "qwen3_5", 8, 3
    ));
    assert!(!fastpath::should_qwen_prefill_contiguous_ffn_for(
        true, "qwen3_5", 1, 3
    ));
}

#[test]
fn qwen_prefill_flat_ffn_activation_qw_matches_3d_4bit_gs32() {
    // Drives the shipped flatten/restore used by ffn_swiglu_with_policy
    // so gate/up/down qmm see [B*S,H] but the layer output stays [B,S,H'].
    let hidden_data: Vec<f32> = (0..8 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let down_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0004)
        .collect();
    let hidden = array_f32(&hidden_data, &[1, 8, 64]);
    let down_w = array_f32(&down_data, &[32, 64]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let (flat, orig) = flatten_qwen_prefill_ffn_activation(&hidden);
    assert_eq!(flat.shape(), vec![8, 64]);
    assert_eq!(orig, [1, 8, 64]);
    let flat_out = restore_qwen_prefill_ffn_activation(&qw(&flat, &down), orig);
    let portable = qw(&hidden, &down);
    eval(&[&flat_out, &portable]);
    assert_eq!(flat_out.shape(), portable.shape());
    assert_eq!(flat_out.shape(), vec![1, 8, 32]);
    assert_close(flat_out.data_f32(), portable.data_f32(), 3.0e-2);
    assert!(!fastpath::should_qwen_prefill_flat_ffn_for(
        true, "qwen3_5", 1, 3
    ));
}

#[test]
fn qwen_prefill_flat_down_qmm_matches_3d_qw_4bit_gs32() {
    let hidden_data: Vec<f32> = (0..8 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let down_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0004)
        .collect();
    let hidden = array_f32(&hidden_data, &[1, 8, 64]);
    let down_w = array_f32(&down_data, &[32, 64]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(dq.len(), 3);
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let flat = qwen_prefill_flat_down_qmm(&hidden, &down)
        .expect("Qwen 4-bit gs32 flat down qmm should engage");
    let portable = qw(&hidden, &down);
    eval(&[&flat, &portable]);
    assert_eq!(flat.shape(), portable.shape());
    assert_eq!(flat.shape(), vec![1, 8, 32]);
    assert_close(flat.data_f32(), portable.data_f32(), 3.0e-2);
    let decode = array_f32(&hidden_data[..64], &[1, 1, 64]);
    assert!(
        qwen_prefill_flat_down_qmm(&decode, &down).is_none(),
        "flat down qmm must reject decode seq==1"
    );
}

#[test]
fn qwen_dual_qmm_swiglu_matches_two_qmm_silu_mul_4bit_gs32() {
    // AXQ language FFN is 4-bit gs32. Drive the C++ body directly so a
    // later wash flip of the call-site flag does not skip this test.
    let x_data: Vec<f32> = (0..8 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let gate_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0005)
        .collect();
    let up_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 512.0) * -0.0004)
        .collect();
    let x = array_f32(&x_data, &[1, 8, 64]);
    let gate_w = array_f32(&gate_data, &[32, 64]);
    let up_w = array_f32(&up_data, &[32, 64]);
    let gq = quantize(
        &gate_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let uq = quantize(
        &up_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(gq.len(), 3);
    assert_eq!(uq.len(), 3);
    let qweight = |q: &[MlxArray]| QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let gate = qweight(&gq);
    let up = qweight(&uq);
    let fused = qwen_dual_qmm_swiglu(&x, &gate, &up)
        .expect("Qwen 4-bit gs32 dual qmm + SwiGLU should engage");
    let p_gate = quantized_matmul(
        &x,
        &gq[0],
        &gq[1],
        Some(&gq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let p_up = quantized_matmul(
        &x,
        &uq[0],
        &uq[1],
        Some(&uq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let portable = silu_mul(&p_gate, &p_up, None);
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
    let mut bad_gate = gate.clone();
    bad_gate.group_size = 0;
    assert!(
        qwen_dual_qmm_swiglu(&x, &bad_gate, &up).is_none(),
        "dual qmm + SwiGLU must reject group_size<=0"
    );
}

#[test]
fn qwen_swiglu_down_fuse_matches_silu_mul_then_qmm_4bit_gs32() {
    let gate_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
    let up_data: Vec<f32> = (0..512).map(|i| ((i as f32) + 1.0) * 0.0078125).collect();
    let down_data: Vec<f32> = (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0004).collect();
    let gate = array_f32(&gate_data, &[1, 8, 64]);
    let up = array_f32(&up_data, &[1, 8, 64]);
    let down_w = array_f32(&down_data, &[32, 64]);
    let dq = quantize(
        &down_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(dq.len(), 3);
    let down = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: Some(dq[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let fused =
        qwen_swiglu_down_fuse(&gate, &up, &down).expect("Qwen SwiGLU+down fuse should engage");
    let hidden = silu_mul(&gate, &up, None);
    let portable = quantized_matmul(
        &hidden,
        &dq[0],
        &dq[1],
        Some(&dq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
}

#[test]
fn qwen_la_out_proj_silu_mul_qmm_matches_rms_silu_then_qw_4bit_gs32() {
    // Shipped LA output fuse: rms_norm(hidden) then silu(z)*normed @ out_proj.
    let hidden_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
    let gate_data: Vec<f32> = (0..512).map(|i| ((i as f32) + 1.0) * 0.0078125).collect();
    let proj_data: Vec<f32> = (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0004).collect();
    let hidden = array_f32(&hidden_data, &[1, 8, 64]);
    let gate = array_f32(&gate_data, &[1, 8, 64]);
    let proj_w = array_f32(&proj_data, &[32, 64]);
    let pq = quantize(
        &proj_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let norm_w = array_f32(&vec![1.0f32; 64], &[64]);
    let normed = rms_norm(&hidden, Some(&norm_w), 1e-6, None);
    let fused =
        silu_mul_quantized_matmul(&gate, &normed, &pq[0], &pq[1], Some(&pq[2]), 32, 4, None)
            .expect("LA out_proj silu_mul qmm should engage on 4-bit gs32");
    let gated = silu_mul(&gate, &normed, None);
    let portable = quantized_matmul(
        &gated,
        &pq[0],
        &pq[1],
        Some(&pq[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
    assert!(fastpath::should_qwen_la_out_proj_silu_mul_qmm_for(
        true, "qwen3_5", 8
    ));
    assert!(!fastpath::should_qwen_la_out_proj_silu_mul_qmm_for(
        true, "qwen3_5", 1
    ));
}

#[test]
fn dense_ffn_split_gate_up_policy_is_shape_and_family_scoped() {
    assert!(prefer_split_dense_ffn_gate_up(
        "gemma4", false, 127, 127, true
    ));
    assert!(!prefer_split_dense_ffn_gate_up("gemma4", false, 1, 1, true));
    assert!(!prefer_split_dense_ffn_gate_up("gemma4", false, 4, 4, true));
    assert!(!prefer_split_dense_ffn_gate_up(
        "gemma4", false, 126, 126, true
    ));
    assert!(prefer_split_dense_ffn_gate_up(
        "qwen3_next",
        true,
        1,
        1,
        true
    ));
    assert!(!prefer_split_dense_ffn_gate_up(
        "qwen3_next",
        true,
        128,
        128,
        true
    ));
    assert!(!prefer_split_dense_ffn_gate_up(
        "gemma4", false, 128, 128, false
    ));
    assert!(
        prefer_split_dense_ffn_gate_up("gemma4", false, 128, 128, true),
        "default-off packed compile leaves p128 on split gate/up"
    );
}

#[test]
fn use_packed_dense_ffn_prefill_skips_last_only_packed() {
    assert!(
        super::use_packed_dense_ffn_prefill(false, true, false),
        "packed prefill stays on when last-only skip is off"
    );
    assert!(
        !super::use_packed_dense_ffn_prefill(false, true, true),
        "last-only 1-token FFN must skip unused packed prefill qmm"
    );
    assert!(!super::use_packed_dense_ffn_prefill(true, true, false));
    assert!(!super::use_packed_dense_ffn_prefill(false, false, false));
    assert!(
        crate::fastpath::should_gemma4_prefill_skip_unused_last_ffn_packed_for(
            true, "gemma4", true, 128
        ),
        "shipped skip-unused-last-ffn-packed must accept contract p128 last layer"
    );
}

#[test]
fn gemma4_split_qkv_policy_is_shape_and_family_scoped() {
    assert!(prefer_split_qkv_projection(
        "gemma4",
        false,
        ProjectionBatchPolicy::Shared,
        1,
        127,
        true,
    ));
    assert!(prefer_split_qkv_projection(
        "gemma4",
        false,
        ProjectionBatchPolicy::Shared,
        1,
        511,
        true,
    ));
    assert!(!prefer_split_qkv_projection(
        "gemma4",
        false,
        ProjectionBatchPolicy::Shared,
        1,
        1,
        true,
    ));
    assert!(!prefer_split_qkv_projection(
        "gemma4",
        false,
        ProjectionBatchPolicy::Shared,
        1,
        4,
        true,
    ));
    // Chunk-512 pure: packed is faster than split (mbp-m5 A/B ~1.03×).
    assert!(!prefer_split_qkv_projection(
        "gemma4",
        false,
        ProjectionBatchPolicy::Shared,
        1,
        512,
        true,
    ));
    assert!(!prefer_split_qkv_projection(
        "gemma4",
        false,
        ProjectionBatchPolicy::Shared,
        1,
        2_048,
        true,
    ));
    assert!(!prefer_split_qkv_projection(
        "qwen3_next",
        false,
        ProjectionBatchPolicy::Shared,
        1,
        128,
        true,
    ));
    assert!(prefer_split_qkv_projection(
        "qwen3_next",
        false,
        ProjectionBatchPolicy::Shared,
        2,
        128,
        true,
    ));
    assert!(!prefer_split_qkv_projection(
        "gemma4",
        true,
        ProjectionBatchPolicy::Shared,
        1,
        128,
        true,
    ));
    assert!(!prefer_split_qkv_projection(
        "gemma4",
        false,
        ProjectionBatchPolicy::RowExact,
        1,
        128,
        true,
    ));
    assert!(!prefer_split_qkv_projection(
        "gemma4",
        false,
        ProjectionBatchPolicy::Shared,
        1,
        128,
        false,
    ));
}

fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Float32,
    )
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let max_abs_diff = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff <= tolerance,
        "max_abs_diff {max_abs_diff} exceeds tolerance {tolerance}"
    );
}

#[test]
fn moe_router_fused_metal_kernel_compiles_and_matches_fallback() {
    let logits_data: Vec<f32> = vec![0.1, 2.0, -1.0, 0.5, 3.0, 0.0, -2.0, 1.5];
    let num_experts = logits_data.len();
    let top_k = 3usize;
    let logits = array_f32(&logits_data, &[1, 1, num_experts as i32]);

    let (indices, weights) = moe_router_fused_metal_apply(&logits, num_experts, top_k)
        .expect("fused router kernel dispatch should be eligible");
    // try_eval (not eval) so a kernel-source compile error fails the test
    // with the Metal diagnostic instead of aborting the process.
    mlx_sys::transforms::try_eval(&[&indices, &weights])
        .expect("fused router kernel must compile and evaluate");

    let (ref_indices, ref_weights) = top_k_by_argpartition(&logits, num_experts, top_k, true);
    eval(&[&ref_indices, &ref_weights]);

    let mut fused: Vec<(u32, f32)> = indices
        .data_u32()
        .iter()
        .copied()
        .zip(weights.data_f32().iter().copied())
        .collect();
    let mut reference: Vec<(u32, f32)> = ref_indices
        .data_u32()
        .iter()
        .copied()
        .zip(ref_weights.data_f32().iter().copied())
        .collect();
    // argpartition returns the top-k unordered; the kernel returns them
    // max-first. Compare as (index, weight) pairs sorted by expert index.
    fused.sort_by_key(|(index, _)| *index);
    reference.sort_by_key(|(index, _)| *index);

    assert_eq!(
        fused.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        reference
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>()
    );
    let fused_weights: Vec<f32> = fused.iter().map(|(_, weight)| *weight).collect();
    let reference_weights: Vec<f32> = reference.iter().map(|(_, weight)| *weight).collect();
    assert_close(&fused_weights, &reference_weights, 1.0e-5);
    let weight_sum: f32 = fused_weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1.0e-5,
        "weights must be a softmax"
    );
}

#[test]
fn packed_geglu_metal_matches_direct_geglu_for_bf16_packed_gate_up() {
    let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.083).collect();
    let up_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.037).collect();
    let gate = astype(&array_f32(&gate_data, &[1, 3, 8]), MlxDtype::Bfloat16, None);
    let up = astype(&array_f32(&up_data, &[1, 3, 8]), MlxDtype::Bfloat16, None);
    let packed = concatenate(&[&gate, &up], -1, None);

    let direct = astype(&geglu(&gate, &up), MlxDtype::Float32, None);
    let metal = packed_geglu_metal_impl(&packed, 8)
        .expect("packed GEGLU Metal kernel should support bf16 packed gate/up");
    let metal = astype(&metal, MlxDtype::Float32, None);
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), vec![1, 3, 8]);
    assert_eq!(
        metal.data_f32(),
        direct.data_f32(),
        "packed GEGLU shim must produce bit-identical output to the imperative reference"
    );
}

#[test]
fn packed_swiglu_metal_matches_direct_swiglu_for_bf16_packed_gate_up() {
    let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.071).collect();
    let up_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.041).collect();
    let gate = astype(&array_f32(&gate_data, &[1, 3, 8]), MlxDtype::Bfloat16, None);
    let up = astype(&array_f32(&up_data, &[1, 3, 8]), MlxDtype::Bfloat16, None);
    let packed = concatenate(&[&gate, &up], -1, None);

    let direct = astype(
        &multiply(&mlx_sys::ops::silu(&gate, None), &up, None),
        MlxDtype::Float32,
        None,
    );
    let metal = packed_swiglu_metal_impl(&packed, 8)
        .expect("packed SwiGLU Metal kernel should support bf16 packed gate/up");
    let metal = astype(&metal, MlxDtype::Float32, None);
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), vec![1, 3, 8]);
    assert_close(metal.data_f32(), direct.data_f32(), 2.0e-2);
}

fn check_qwen_gate_up_swiglu_precision(dtype: MlxDtype) {
    let x_data: Vec<f32> = if dtype == MlxDtype::Float32 {
        (0..32).map(|i| ((i as f32) - 16.0) * 0.03125).collect()
    } else {
        // Isolate dtype boundaries from reduction-order differences.
        (0..32).map(|i| if i == 7 { 1.0 } else { 0.0 }).collect()
    };
    let gate_weight_data: Vec<f32> = (0..512)
        .map(|i| {
            if dtype == MlxDtype::Float32 {
                ((i as f32) - 180.0) * 0.0025
            } else {
                (((i + i / 32) % 16) as f32 - 8.0) * 0.125
            }
        })
        .collect();
    let up_weight_data: Vec<f32> = (0..512)
        .map(|i| {
            if dtype == MlxDtype::Float32 {
                ((i as f32) - 96.0) * -0.001875
            } else {
                (7.0 - ((i + 3 * (i / 32)) % 16) as f32) * 0.125
            }
        })
        .collect();
    let x = astype(&array_f32(&x_data, &[1, 1, 32]), dtype, None);
    let gate_weight = astype(&array_f32(&gate_weight_data, &[16, 32]), dtype, None);
    let up_weight = astype(&array_f32(&up_weight_data, &[16, 32]), dtype, None);
    let gate_q = quantize(
        &gate_weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let up_q = quantize(
        &up_weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(gate_q.len(), 3);
    assert_eq!(up_q.len(), 3);
    let gate = QuantizedWeight {
        weight: gate_q[0].clone(),
        scales: Some(gate_q[1].clone()),
        biases: Some(gate_q[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let up = QuantizedWeight {
        weight: up_q[0].clone(),
        scales: Some(up_q[1].clone()),
        biases: Some(up_q[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };

    let metal = qwen_dense_ffn_gate_up_swiglu_metal_impl(&x, &gate, &up)
        .expect("4-bit affine gate/up SwiGLU matvec should be eligible");
    assert_eq!(metal.shape(), vec![1, 1, 16]);
    let gate_ref = quantized_matmul(
        &x,
        &gate_q[0],
        &gate_q[1],
        Some(&gate_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let up_ref = quantized_matmul(
        &x,
        &up_q[0],
        &up_q[1],
        Some(&up_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let reference = silu_mul(&gate_ref, &up_ref, None);
    mlx_sys::transforms::try_eval(&[&metal, &reference])
        .expect("Qwen dense FFN SwiGLU matvec Metal kernel must compile and evaluate");

    assert_eq!(metal.shape(), vec![1, 1, 16]);
    let metal = astype(&metal, MlxDtype::Float32, None);
    let reference = astype(&reference, MlxDtype::Float32, None);
    eval(&[&metal, &reference]);
    if dtype == MlxDtype::Float32 {
        assert_close(metal.data_f32(), reference.data_f32(), 1.0e-4);
    } else {
        assert_eq!(
            metal
                .data_f32()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            reference
                .data_f32()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn qwen_dense_ffn_gate_up_swiglu_metal_matches_split_quantized_matmuls() {
    check_qwen_gate_up_swiglu_precision(MlxDtype::Float32);
}

#[test]
fn qwen_dense_ffn_gate_up_swiglu_preserves_bfloat16_intermediates() {
    check_qwen_gate_up_swiglu_precision(MlxDtype::Bfloat16);
}

#[test]
fn qwen_dense_ffn_gate_up_swiglu_preserves_float16_intermediates() {
    check_qwen_gate_up_swiglu_precision(MlxDtype::Float16);
}

#[test]
fn qwen_dense_ffn_gate_up_swiglu_metal_rejects_non_decode_shapes() {
    let weight = QuantizedWeight {
        weight: mlx_sys::zeros(&[16, 4], MlxDtype::Uint32, None),
        scales: Some(mlx_sys::zeros(&[16, 1], MlxDtype::Bfloat16, None)),
        biases: Some(mlx_sys::zeros(&[16, 1], MlxDtype::Bfloat16, None)),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let batched = mlx_sys::zeros(&[2, 1, 32], MlxDtype::Float32, None);
    let prefill = mlx_sys::zeros(&[1, 2, 32], MlxDtype::Float32, None);

    assert!(qwen_dense_ffn_gate_up_swiglu_metal_impl(&batched, &weight, &weight).is_none());
    assert!(qwen_dense_ffn_gate_up_swiglu_metal_impl(&prefill, &weight, &weight).is_none());
}

#[test]
fn qwen_dense_ffn_gate_up_matvec_metal_skips_only_regressed_27b_quantization() {
    assert!(qwen_dense_ffn_gate_up_matvec_metal_regresses(
        "qwen3_5", 64, 5120, 17_408, 0, 4, 64
    ));
    assert!(
        !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 32, 4096, 12_288, 0, 4, 64),
        "the measured Qwen3.5-9B win must remain eligible"
    );
    assert!(
        !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 64, 5120, 17_408, 256, 4, 64),
        "the dense-model exception must not mask MoE configurations"
    );
    assert!(
        !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 40, 2048, 0, 256, 4, 64),
        "the Qwen3.6-35B-A3B geometry must remain unaffected"
    );
    assert!(
        !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 64, 5120, 17_408, 0, 6, 64),
        "the unmeasured 6-bit configuration must not inherit the 4-bit exception"
    );
    assert!(
        !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 64, 5120, 17_408, 0, 4, 32),
        "the exception must remain specific to the measured group size"
    );
}

#[test]
fn qwen_dense_ffn_down_matvec_metal_matches_quantized_matmul() {
    // Intermediate → hidden decode matvec (tiled TG path for InputDim > 4096).
    // Use intermediate=64, out=16, group_size=32 to exercise multi-tile packing.
    let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
    let weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 400.0) * 0.00125).collect();
    let x = array_f32(&x_data, &[1, 1, 64]);
    let weight = array_f32(&weight_data, &[16, 64]);
    let q = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let down = QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };

    let metal = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, None)
        .expect("4-bit affine down matvec should be eligible");
    let reference = quantized_matmul(&x, &q[0], &q[1], Some(&q[2]), true, Some(32), Some(4), None);
    mlx_sys::transforms::try_eval(&[&metal, &reference])
        .expect("Qwen dense FFN down matvec Metal kernel must compile and evaluate");

    assert_eq!(metal.shape(), vec![1, 1, 16]);
    assert_close(metal.data_f32(), reference.data_f32(), 1.0e-4);
}

#[test]
fn qwen_dense_ffn_down_residual_metal_matches_add() {
    let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
    let weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 400.0) * 0.00125).collect();
    let residual_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let x = array_f32(&x_data, &[1, 1, 64]);
    let residual = array_f32(&residual_data, &[1, 1, 16]);
    let weight = array_f32(&weight_data, &[16, 64]);
    let q = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down = QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };

    let fused = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, Some(&residual))
        .expect("residual down matvec should be eligible");
    let split = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, None)
        .expect("plain down matvec should be eligible");
    let reference = add(&residual, &split, None);
    mlx_sys::transforms::try_eval(&[&fused, &reference])
        .expect("residual down Metal kernel must compile and evaluate");
    assert_eq!(fused.shape(), vec![1, 1, 16]);
    assert_close(fused.data_f32(), reference.data_f32(), 1.0e-4);
}

fn affine_6bit_weight(weight: &MlxArray, group_size: i32) -> (QuantizedWeight, Vec<MlxArray>) {
    let q = quantize(
        weight,
        Some(group_size),
        Some(6),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    (
        QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits: 6,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        },
        q,
    )
}

#[test]
fn qwen_dense_ffn_gate_up_swiglu_metal_6bit_matches_split_quantized_matmuls() {
    // 6-bit packs 4 values into 3 bytes: input_dim must be a multiple of
    // 16 so the packed row is whole uint32 columns (64 * 6 / 32 = 12).
    let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 30.0) * 0.017).collect();
    let gate_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 500.0) * 0.0011).collect();
    let up_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 300.0) * -0.0009).collect();
    let x = array_f32(&x_data, &[1, 1, 64]);
    let (gate, gate_q) = affine_6bit_weight(&array_f32(&gate_data, &[16, 64]), 32);
    let (up, up_q) = affine_6bit_weight(&array_f32(&up_data, &[16, 64]), 32);
    assert_eq!(gate.weight.shape(), vec![16, 12]);

    let metal = qwen_dense_ffn_gate_up_swiglu_metal_impl(&x, &gate, &up)
        .expect("6-bit affine gate/up SwiGLU matvec should be eligible");
    assert_eq!(metal.shape(), vec![1, 1, 16]);
    let gate_ref = quantized_matmul(
        &x,
        &gate_q[0],
        &gate_q[1],
        Some(&gate_q[2]),
        true,
        Some(32),
        Some(6),
        None,
    );
    let up_ref = quantized_matmul(
        &x,
        &up_q[0],
        &up_q[1],
        Some(&up_q[2]),
        true,
        Some(32),
        Some(6),
        None,
    );
    let reference = silu_mul(&gate_ref, &up_ref, None);
    mlx_sys::transforms::try_eval(&[&metal, &reference])
        .expect("6-bit gate/up SwiGLU matvec Metal kernel must compile and evaluate");
    assert_close(metal.data_f32(), reference.data_f32(), 1.0e-4);
}

#[test]
fn qwen_dense_ffn_down_matvec_metal_6bit_matches_quantized_matmul() {
    let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
    let weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 400.0) * 0.00125).collect();
    let residual_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let x = array_f32(&x_data, &[1, 1, 64]);
    let residual = array_f32(&residual_data, &[1, 1, 16]);
    let (down, q) = affine_6bit_weight(&array_f32(&weight_data, &[16, 64]), 64);

    let metal = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, None)
        .expect("6-bit affine down matvec should be eligible");
    let fused = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, Some(&residual))
        .expect("6-bit residual down matvec should be eligible");
    let reference = quantized_matmul(&x, &q[0], &q[1], Some(&q[2]), true, Some(64), Some(6), None);
    let fused_reference = add(&residual, &reference, None);
    mlx_sys::transforms::try_eval(&[&metal, &fused, &reference, &fused_reference])
        .expect("6-bit down matvec Metal kernels must compile and evaluate");
    assert_eq!(metal.shape(), vec![1, 1, 16]);
    assert_close(metal.data_f32(), reference.data_f32(), 1.0e-4);
    assert_close(fused.data_f32(), fused_reference.data_f32(), 1.0e-4);
}

#[test]
fn qwen_dense_ffn_down_matvec_metal_8bit_matches_quantized_matmul() {
    let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
    let weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 400.0) * 0.00125).collect();
    let x = array_f32(&x_data, &[1, 1, 64]);
    let weight = array_f32(&weight_data, &[16, 64]);
    let q = quantize(
        &weight,
        Some(32),
        Some(8),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let down = QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size: 32,
        bits: 8,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let metal = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, None)
        .expect("8-bit affine down matvec should be eligible");
    let reference = quantized_matmul(&x, &q[0], &q[1], Some(&q[2]), true, Some(32), Some(8), None);
    mlx_sys::transforms::try_eval(&[&metal, &reference])
        .expect("8-bit down matvec Metal kernel must compile and evaluate");
    assert_eq!(metal.shape(), vec![1, 1, 16]);
    assert_close(metal.data_f32(), reference.data_f32(), 1.0e-4);
}

#[test]
#[ignore = "micro-bench: run with --ignored --nocapture for kernel-vs-MLX timings"]
fn bench_qwen_dense_ffn_matvec_6bit_vs_mlx() {
    // Qwen3.5-9B-ish decode FFN shape: hidden 4096 -> intermediate 12288.
    let hidden = 4096_i32;
    let intermediate = 12288_i32;
    let x_data: Vec<f32> = (0..hidden as usize)
        .map(|i| (((i % 509) as f32) - 250.0) * 0.004)
        .collect();
    let x = astype(
        &array_f32(&x_data, &[1, 1, hidden]),
        MlxDtype::Bfloat16,
        None,
    );
    let mk = |rows: i32, cols: i32, seed: usize| -> MlxArray {
        let data: Vec<f32> = (0..(rows * cols) as usize)
            .map(|i| ((((i + seed) % 977) as f32) - 480.0) * 0.001)
            .collect();
        array_f32(&data, &[rows, cols])
    };
    let (gate, gate_q) = affine_6bit_weight(&mk(intermediate, hidden, 0), 64);
    let (up, up_q) = affine_6bit_weight(&mk(intermediate, hidden, 7), 64);
    let (down, down_q) = affine_6bit_weight(&mk(hidden, intermediate, 13), 64);
    let iters = 200;
    let warm = 20;
    for _ in 0..warm {
        let h = qwen_dense_ffn_gate_up_swiglu_metal_impl(&x, &gate, &up).unwrap();
        let o = qwen_dense_ffn_down_matvec_metal_impl(&h, &down, None).unwrap();
        eval(&[&o]);
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let h = qwen_dense_ffn_gate_up_swiglu_metal_impl(&x, &gate, &up).unwrap();
        let o = qwen_dense_ffn_down_matvec_metal_impl(&h, &down, None).unwrap();
        eval(&[&o]);
    }
    let kernel_us = start.elapsed().as_micros() as f64 / iters as f64;
    let mlx_ffn = |x: &MlxArray| {
        let g = quantized_matmul(
            x,
            &gate_q[0],
            &gate_q[1],
            Some(&gate_q[2]),
            true,
            Some(64),
            Some(6),
            None,
        );
        let u = quantized_matmul(
            x,
            &up_q[0],
            &up_q[1],
            Some(&up_q[2]),
            true,
            Some(64),
            Some(6),
            None,
        );
        let h = silu_mul(&g, &u, None);
        quantized_matmul(
            &h,
            &down_q[0],
            &down_q[1],
            Some(&down_q[2]),
            true,
            Some(64),
            Some(6),
            None,
        )
    };
    for _ in 0..warm {
        let o = mlx_ffn(&x);
        eval(&[&o]);
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let o = mlx_ffn(&x);
        eval(&[&o]);
    }
    let mlx_us = start.elapsed().as_micros() as f64 / iters as f64;
    println!(
        "6-bit decode FFN {hidden}->{intermediate}: kernel {kernel_us:.1}us vs MLX {mlx_us:.1}us ({:.3}x)",
        mlx_us / kernel_us
    );
}

#[test]
fn qwen_dense_ffn_matvec_rejects_unsupported_bit_widths() {
    let x = mlx_sys::zeros(&[1, 1, 64], MlxDtype::Float32, None);
    for bits in [2_i32, 3, 5] {
        let packed_cols = 64 * bits / 32;
        let weight = QuantizedWeight {
            weight: mlx_sys::zeros(&[16, packed_cols.max(1)], MlxDtype::Uint32, None),
            scales: Some(mlx_sys::zeros(&[16, 2], MlxDtype::Float32, None)),
            biases: Some(mlx_sys::zeros(&[16, 2], MlxDtype::Float32, None)),
            group_size: 32,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        assert!(
            qwen_dense_ffn_gate_up_swiglu_metal_impl(&x, &weight, &weight).is_none(),
            "{bits}-bit must not reach the gate/up matvec kernel"
        );
        assert!(
            qwen_dense_ffn_down_matvec_metal_impl(&x, &weight, None).is_none(),
            "{bits}-bit must not reach the down matvec kernel"
        );
    }
}

#[test]
fn split_geglu_metal_matches_direct_geglu_for_bf16_gate_up() {
    let gate_data: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.059).collect();
    let up_data: Vec<f32> = (0..32).map(|i| ((i as f32) + 3.0) * 0.031).collect();
    let gate = astype(
        &array_f32(&gate_data, &[1, 1, 4, 8]),
        MlxDtype::Bfloat16,
        None,
    );
    let up = astype(
        &array_f32(&up_data, &[1, 1, 4, 8]),
        MlxDtype::Bfloat16,
        None,
    );

    let direct = astype(&gelu_approx_mul(&gate, &up, None), MlxDtype::Float32, None);
    let metal = gelu_approx_mul_metal(&gate, &up, true)
        .expect("split GEGLU Metal kernel should support bf16 gate/up");
    let metal = astype(&metal, MlxDtype::Float32, None);
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), vec![1, 1, 4, 8]);
    assert_close(metal.data_f32(), direct.data_f32(), 2.0e-2);
}

#[test]
fn gemma4_moe_weighted_sum_metal_matches_mlx_ops() {
    let down_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 8.0) * 0.037).collect();
    let weight_data: Vec<f32> = vec![0.1, 0.25, 0.65, 0.5, 0.125, 0.375];
    let down = array_f32(&down_data, &[1, 2, 3, 4]);
    let weights = array_f32(&weight_data, &[1, 2, 3]);

    let scores_exp = expand_dims(&weights, weights.ndim() as i32, None);
    let weighted = multiply(&down, &scores_exp, None);
    let direct = sum_axis(&weighted, 2, false, None);
    let metal = gemma4_moe_weighted_sum_metal(&down, &weights, MlxDtype::Float32)
        .expect("weighted-sum Metal kernel should support f32 inputs");
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), vec![1, 2, 4]);
    assert_close(metal.data_f32(), direct.data_f32(), 1.0e-5);
}

#[test]
fn combine_gemma4_dual_path_fused_post_norm_matches_unfused() {
    // h1 + h2 then post-RMSNorm must match add_rms_norm_pair path.
    let h1 = array_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let h2 = array_f32(&[0.5, -0.5, 1.5, -1.5], &[1, 1, 4]);
    let post = array_f32(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let eps = 1.0e-6_f32;

    let unfused = {
        let combined = add(&h1, &h2, None);
        rms_norm(&combined, Some(&post), eps, None)
    };
    let fused = combine_gemma4_dual_path_outputs(&h1, &h2, None, Some(&post), eps);
    eval(&[&unfused, &fused]);
    assert_eq!(fused.shape(), unfused.shape());
    assert_close(fused.data_f32(), unfused.data_f32(), 1.0e-5);
}

#[test]
fn combine_gemma4_dual_path_with_expert_post_norm_stays_unfused_order() {
    let h1 = array_f32(&[1.0, 0.0, -1.0, 2.0], &[1, 1, 4]);
    let h2 = array_f32(&[0.25, 0.25, 0.25, 0.25], &[1, 1, 4]);
    let post2 = array_f32(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let post = array_f32(&[0.5, 0.5, 0.5, 0.5], &[4]);
    let eps = 1.0e-6_f32;

    let expected = {
        let h2n = rms_norm(&h2, Some(&post2), eps, None);
        let combined = add(&h1, &h2n, None);
        rms_norm(&combined, Some(&post), eps, None)
    };
    let got = combine_gemma4_dual_path_outputs(&h1, &h2, Some(&post2), Some(&post), eps);
    eval(&[&expected, &got]);
    assert_close(got.data_f32(), expected.data_f32(), 1.0e-5);
}

#[test]
fn gemma4_moe_weighted_scaled_sum_metal_matches_mlx_ops() {
    let down_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 8.0) * 0.037).collect();
    let weight_data: Vec<f32> = vec![0.1, 0.25, 0.65, 0.5, 0.125, 0.375];
    let indices_data: Vec<u32> = vec![2, 0, 3, 1, 3, 0];
    let scale_data: Vec<f32> = vec![0.75, 1.25, 0.5, 1.5];
    let down = array_f32(&down_data, &[1, 2, 3, 4]);
    let weights = array_f32(&weight_data, &[1, 2, 3]);
    let indices = MlxArray::from_raw_data(
        indices_data.as_ptr() as *const u8,
        std::mem::size_of_val(indices_data.as_slice()),
        &[1, 2, 3],
        MlxDtype::Uint32,
    );
    let scale = array_f32(&scale_data, &[4]);

    let gathered = take(&scale, &indices, 0, None);
    let scaled_weights = multiply(&weights, &gathered, None);
    let scores_exp = expand_dims(&scaled_weights, scaled_weights.ndim() as i32, None);
    let weighted = multiply(&down, &scores_exp, None);
    let direct = sum_axis(&weighted, 2, false, None);
    let metal =
        gemma4_moe_weighted_scaled_sum_metal(&down, &weights, &indices, &scale, MlxDtype::Float32)
            .expect("weighted scaled-sum Metal kernel should support f32 inputs");
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), vec![1, 2, 4]);
    assert_close(metal.data_f32(), direct.data_f32(), 1.0e-5);
}

#[test]
fn moe_fused_activation_unsort_metal_matches_direct_geglu_for_bf16() {
    let hidden_dim = 8;
    let top_k = 3;
    let gate_data: Vec<f32> = (0..hidden_dim * top_k)
        .map(|i| ((i as f32) - 12.0) * 0.083)
        .collect();
    let up_data: Vec<f32> = (0..hidden_dim * top_k)
        .map(|i| ((i as f32) + 1.0) * 0.037)
        .collect();
    // Sorted-order gate/up, shape [top_k, hidden_dim].
    let gate = astype(
        &array_f32(&gate_data, &[top_k, hidden_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let up = astype(
        &array_f32(&up_data, &[top_k, hidden_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let packed_sorted = concatenate(&[&gate, &up], -1, None);
    let packed = reshape(&packed_sorted, &[1, 1, top_k, hidden_dim * 2], None);

    // original_k -> sorted_k: original position 0 reads sorted row 2, etc.
    let inv_order_data: Vec<u32> = vec![2, 0, 1];
    let inv_order = MlxArray::from_raw_data(
        inv_order_data.as_ptr() as *const u8,
        std::mem::size_of_val(inv_order_data.as_slice()),
        &[top_k],
        MlxDtype::Uint32,
    );

    // Reference: apply geglu in sorted order, then unsort via `take`
    // (mirrors SwitchGatherInputs::unsort's flatten + take pattern).
    let direct_sorted = geglu(&gate, &up);
    let direct = astype(
        &take(&direct_sorted, &inv_order, 0, None),
        MlxDtype::Float32,
        None,
    );

    let metal = moe_fused_activation_unsort_metal(
        &packed,
        &inv_order,
        hidden_dim,
        top_k,
        MlxDtype::Bfloat16,
        true,
    )
    .expect("MoE fused activation+unsort Metal kernel should support bf16 GEGLU inputs");
    let metal = astype(
        &reshape(&metal, &[top_k, hidden_dim], None),
        MlxDtype::Float32,
        None,
    );
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), vec![top_k, hidden_dim]);
    assert_eq!(
        metal.data_f32(),
        direct.data_f32(),
        "MoE fused activation+unsort GEGLU branch must be bit-identical to the imperative reference"
    );
}

#[test]
fn moe_fused_activation_unsort_metal_matches_direct_swiglu_for_bf16() {
    // Regression guard for the sibling `uses_geglu=false` branch: proves
    // the `if constexpr (USE_GEGLU)` specialization still selects SwiGLU
    // (not just that GEGLU no longer silently falls through to it).
    let hidden_dim = 8;
    let top_k = 3;
    let gate_data: Vec<f32> = (0..hidden_dim * top_k)
        .map(|i| ((i as f32) - 12.0) * 0.071)
        .collect();
    let up_data: Vec<f32> = (0..hidden_dim * top_k)
        .map(|i| ((i as f32) + 1.0) * 0.041)
        .collect();
    let gate = astype(
        &array_f32(&gate_data, &[top_k, hidden_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let up = astype(
        &array_f32(&up_data, &[top_k, hidden_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let packed_sorted = concatenate(&[&gate, &up], -1, None);
    let packed = reshape(&packed_sorted, &[1, 1, top_k, hidden_dim * 2], None);

    let inv_order_data: Vec<u32> = vec![2, 0, 1];
    let inv_order = MlxArray::from_raw_data(
        inv_order_data.as_ptr() as *const u8,
        std::mem::size_of_val(inv_order_data.as_slice()),
        &[top_k],
        MlxDtype::Uint32,
    );

    let direct_sorted = silu_mul(&gate, &up, None);
    let direct = astype(
        &take(&direct_sorted, &inv_order, 0, None),
        MlxDtype::Float32,
        None,
    );

    let metal = moe_fused_activation_unsort_metal(
        &packed,
        &inv_order,
        hidden_dim,
        top_k,
        MlxDtype::Bfloat16,
        false,
    )
    .expect("MoE fused activation+unsort Metal kernel should support bf16 SwiGLU inputs");
    let metal = astype(
        &reshape(&metal, &[top_k, hidden_dim], None),
        MlxDtype::Float32,
        None,
    );
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), vec![top_k, hidden_dim]);
    assert_eq!(metal.data_f32(), direct.data_f32());
}

#[test]
fn moe_fused_activation_unsort_metal_preserves_low_precision_swiglu() {
    let hidden_dim = 16;
    let top_k = 4;
    let gate_data: Vec<f32> = (0..hidden_dim * top_k)
        .map(|i| ((i as f32) - 30.0) * 0.137)
        .collect();
    let up_data: Vec<f32> = (0..hidden_dim * top_k)
        .map(|i| ((i as f32) + 3.0) * -0.059)
        .collect();
    let inv_order_data: Vec<u32> = vec![3, 1, 0, 2];
    let inv_order = MlxArray::from_raw_data(
        inv_order_data.as_ptr() as *const u8,
        std::mem::size_of_val(inv_order_data.as_slice()),
        &[top_k],
        MlxDtype::Uint32,
    );
    for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
        let gate = astype(&array_f32(&gate_data, &[top_k, hidden_dim]), dtype, None);
        let up = astype(&array_f32(&up_data, &[top_k, hidden_dim]), dtype, None);
        let packed = reshape(
            &concatenate(&[&gate, &up], -1, None),
            &[1, 1, top_k, hidden_dim * 2],
            None,
        );
        let direct = astype(
            &take(&silu_mul(&gate, &up, None), &inv_order, 0, None),
            MlxDtype::Float32,
            None,
        );
        let metal =
            moe_fused_activation_unsort_metal(&packed, &inv_order, hidden_dim, top_k, dtype, false)
                .expect("fused SwiGLU unsort must engage for low-precision rows");
        assert_eq!(metal.dtype(), dtype);
        let metal = astype(
            &reshape(&metal, &[top_k, hidden_dim], None),
            MlxDtype::Float32,
            None,
        );
        eval(&[&direct, &metal]);
        assert_eq!(metal.data_f32(), direct.data_f32());
    }
}

#[test]
fn identity_moe_unsort_order_is_the_natural_ordering() {
    let ids: Vec<u32> = (0..5u32).collect();
    let order = identity_moe_unsort_order(5);
    assert_eq!(order.dtype(), MlxDtype::Uint32);
    assert_eq!(order.shape(), vec![5]);
    eval(&[&order]);
    assert_eq!(order.data_u32(), ids.as_slice());
}

#[test]
fn moe_fused_activation_unsort_metal_with_identity_order_matches_unpermuted_gather() {
    // Regression test: `switch_gather_inputs` only ever pairs
    // `sorted_indices: false` with `inv_order: None` (an unsorted
    // gather leaves the top_k rows in original expert-selection
    // order). The two call sites in this file used to fall back to
    // the caller's expert-id array instead of a real identity order
    // when `inv_order` was `None` — `inv_order[k]` must select a row
    // in `[0, top_k)` of the top_k-row gate/up buffer, but expert IDs
    // range far past `top_k` on any real model, so that was an
    // out-of-bounds Metal buffer read. This proves the fix's
    // `identity_moe_unsort_order(top_k)` correctly leaves an
    // already-unpermuted gather's rows in place: activating in place
    // must equal activating then "unsorting" through the identity.
    let hidden_dim = 8;
    let top_k = 3;
    let gate_data: Vec<f32> = (0..hidden_dim * top_k)
        .map(|i| ((i as f32) - 12.0) * 0.083)
        .collect();
    let up_data: Vec<f32> = (0..hidden_dim * top_k)
        .map(|i| ((i as f32) + 1.0) * 0.037)
        .collect();
    let gate = astype(
        &array_f32(&gate_data, &[top_k, hidden_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let up = astype(
        &array_f32(&up_data, &[top_k, hidden_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let packed_unsorted = concatenate(&[&gate, &up], -1, None);
    let packed = reshape(&packed_unsorted, &[1, 1, top_k, hidden_dim * 2], None);

    // Reference: activate in place, no permutation.
    let direct = astype(&geglu(&gate, &up), MlxDtype::Float32, None);

    let metal = moe_fused_activation_unsort_metal(
        &packed,
        &identity_moe_unsort_order(top_k),
        hidden_dim,
        top_k,
        MlxDtype::Bfloat16,
        true,
    )
    .expect("MoE fused activation+unsort Metal kernel should support the identity order");
    let metal = astype(
        &reshape(&metal, &[top_k, hidden_dim], None),
        MlxDtype::Float32,
        None,
    );
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), vec![top_k, hidden_dim]);
    assert_eq!(
        metal.data_f32(),
        direct.data_f32(),
        "identity unsort order must leave an already-unpermuted gather unchanged"
    );
}

#[test]
fn packed_geglu_metal_rejects_unexpected_packed_width() {
    let data = vec![0.0_f32; 12];
    let packed = array_f32(&data, &[1, 1, 12]);
    assert!(
        packed_geglu_metal_impl(&packed, 5).is_none(),
        "packed width must be exactly 2 * hidden_dim"
    );

    let gate = slice_last_dim(&packed, 0, 6, None);
    assert!(
        packed_geglu_metal_impl(&gate, 6).is_none(),
        "already-split gate tensors must stay on the normal GEGLU path"
    );
}

#[test]
fn packed_swiglu_metal_rejects_unexpected_packed_width() {
    let data = vec![0.0_f32; 12];
    let packed = array_f32(&data, &[1, 1, 12]);
    assert!(
        packed_swiglu_metal_impl(&packed, 5).is_none(),
        "packed width must be exactly 2 * hidden_dim"
    );

    let gate = slice_last_dim(&packed, 0, 6, None);
    assert!(
        packed_swiglu_metal_impl(&gate, 6).is_none(),
        "already-split gate tensors must stay on the normal SwiGLU path"
    );
}

#[test]
fn qwen3_moe_weighted_sum_with_shared_metal_matches_unfused() {
    // down_out: [batch=1, seq=2, top_k=3, hidden=4]
    let down_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 8.0) * 0.037).collect();
    let weight_data: Vec<f32> = vec![0.1, 0.25, 0.65, 0.5, 0.125, 0.375];
    let shared_data: Vec<f32> = (0..8).map(|i| ((i as f32) + 1.0) * 0.053).collect();

    let down = array_f32(&down_data, &[1, 2, 3, 4]);
    let weights = array_f32(&weight_data, &[1, 2, 3]);
    let shared = array_f32(&shared_data, &[1, 2, 4]);

    // Unfused reference: weighted_sum(down, weights) + shared
    let scores_exp = expand_dims(&weights, weights.ndim() as i32, None);
    let weighted = multiply(&down, &scores_exp, None);
    let expert_sum = sum_axis(&weighted, 2, false, None);
    let unfused = add(&expert_sum, &shared, None);

    // Fused kernel
    let fused =
        qwen3_moe_weighted_sum_with_shared_metal(&down, &weights, &shared, MlxDtype::Float32)
            .expect("fused weighted-sum-with-shared kernel should support f32 inputs");
    eval(&[&unfused, &fused]);

    assert_eq!(fused.shape(), vec![1, 2, 4]);
    assert_close(fused.data_f32(), unfused.data_f32(), 1.0e-5);
}

#[test]
fn qwen3_moe_weighted_sum_with_shared_metal_rejects_shape_mismatch() {
    let down = array_f32(&[0.0; 24], &[1, 2, 3, 4]);
    let weights = array_f32(&[0.0; 6], &[1, 2, 3]);
    // Wrong shared shape: [1, 3, 4] instead of [1, 2, 4]
    let shared = array_f32(&[0.0; 12], &[1, 3, 4]);
    assert!(
        qwen3_moe_weighted_sum_with_shared_metal(&down, &weights, &shared, MlxDtype::Float32)
            .is_none(),
        "kernel must reject mismatched shared_expert shape"
    );
}

fn check_packed_swiglu_precision(dtype: MlxDtype, shape: &[i32]) {
    let gate_data: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.053).collect();
    let up_data: Vec<f32> = (0..32).map(|i| ((i as f32) + 2.0) * 0.031).collect();
    let gate = astype(&array_f32(&gate_data, shape), dtype, None);
    let up = astype(&array_f32(&up_data, shape), dtype, None);
    let packed = concatenate(&[&gate, &up], -1, None);

    // Unfused reference: slice + silu_mul (matches the MoE fallback path)
    let half = *shape.last().unwrap();
    let gate_slice = mlx_slice_last_dim(&packed, 0, half);
    let up_slice = mlx_slice_last_dim(&packed, half, half * 2);
    let direct = astype(
        &silu_mul(&gate_slice, &up_slice, None),
        MlxDtype::Float32,
        None,
    );

    // The packed path must retain the split activation tensor semantics.
    let metal = packed_swiglu_metal_impl(&packed, half)
        .expect("packed SwiGLU Metal kernel should support MoE-shaped gate_up");
    let metal = astype(&metal, MlxDtype::Float32, None);
    eval(&[&direct, &metal]);

    assert_eq!(metal.shape(), shape);
    assert_eq!(metal.data_f32(), direct.data_f32());
}

#[test]
fn packed_swiglu_preserves_low_precision_dense_and_moe_rows() {
    for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
        for shape in [&[1, 1, 4, 8][..], &[1, 2, 16][..], &[1, 4, 8][..]] {
            check_packed_swiglu_precision(dtype, shape);
        }
    }
}

#[test]
fn packed_swiglu_rejects_invalid_low_precision_width() {
    let packed = zeros(&[1, 2, 16], MlxDtype::Bfloat16, None);
    for width in [0, -1, 7, i32::MAX] {
        assert!(packed_swiglu_metal_impl(&packed, width).is_none());
    }
    let scalar = zeros(&[], MlxDtype::Bfloat16, None);
    assert!(packed_swiglu_metal_impl(&scalar, i32::MAX).is_none());
    assert!(packed_swiglu_metal_impl(&scalar, 1).is_none());
}

/// Admission probe for shapeless compiled linear closures.
///
/// The core risk is that `shapeless=true` compilation with
/// `quantized_matmul` is untested in this codebase (the existing compile
/// caches use either elementwise ops with shapeless, or quantized_matmul
/// with per-shape compilation). This probe builds a small quantized weight,
/// compiles a shapeless closure doing `quantized_matmul -> sigmoid ->
/// multiply` (the shared-expert gate path), and records the current
/// MLX 0.32.1 fixes the previous cross-shape divergence, so both the traced
/// decode shape and a different sequence length must now match the
/// imperative graph. This does not promote the dormant/opt-in Tier 3A MoE
/// route by itself; its separate stream-registry and performance gates
/// remain in force.
#[test]
fn shapeless_compiled_linear_closure_is_shape_polymorphic() {
    use mlx_sys::{MlxClosure, MlxVectorArray, quantized_matmul, sigmoid};

    // Build a small non-quantized weight mimicking a shared-expert
    // projection: shape [hidden=8, out=16]. (The probe's goal is to verify
    // the shapeless compilation contract for a graph with a linear op +
    // elementwise ops across two input shapes. Quantized_matmul's packed
    // uint32 format is well-exercised by the production weight loader and
    // existing tests; the real unknown here is whether shapeless=true
    // preserves correctness for a linear graph, so a plain matmul suffices.)
    let weight_data: Vec<f32> = (0..128).map(|i| ((i as f32) - 64.0) * 0.01).collect();
    let weight = array_f32(&weight_data, &[8, 16]);
    let qw_captured = QuantizedWeight {
        weight: weight.clone(),
        scales: None,
        biases: None,
        group_size: 64,
        bits: 32,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };

    // Capture a *clone* of the weight into the closure body. Per
    // closure.rs:191, captured MlxArrays become constants in the compiled
    // graph — this is the same mechanism the embedding closures use.
    let body_factory = || {
        let qw = qw_captured.clone();
        MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
            let x = inputs.get(0);
            let h = qw_inner(&qw, &x);
            let gate = inputs.get(1);
            let sig = sigmoid(&gate, None);
            vec![multiply(&h, &sig, None)]
        })
    };

    // Helper that mirrors qw() but takes QuantizedWeight by ref (avoids
    // lifetime issues with the closure capturing qw by value).
    fn qw_inner(qw: &QuantizedWeight, x: &MlxArray) -> MlxArray {
        if let Some(scales) = &qw.scales {
            quantized_matmul(
                x,
                &qw.weight,
                scales,
                qw.biases.as_ref(),
                true,
                Some(qw.group_size),
                Some(qw.bits),
                None,
            )
        } else {
            mlx_sys::matmul(x, &qw.weight, None)
        }
    }

    let compiled = body_factory()
        .compile(true)
        .expect("shapeless compile of quantized_matmul closure must succeed");

    // Shape 1: [1, 1, 8] (decode shape).
    let x1 = array_f32(
        &(0..8).map(|i| (i as f32) * 0.1).collect::<Vec<_>>(),
        &[1, 1, 8],
    );
    let gate1 = array_f32(&[0.3; 16], &[1, 1, 16]);

    let imperative_out_1 = {
        let h = qw_inner(&qw_captured, &x1);
        let sig = sigmoid(&gate1, None);
        multiply(&h, &sig, None)
    };
    let compiled_out_1 = compiled.apply(&[&x1, &gate1]);
    eval(&[&imperative_out_1, &compiled_out_1[0]]);
    assert_eq!(compiled_out_1[0].shape(), vec![1, 1, 16]);
    // Bit-identical: compiled graph must produce exactly the same result.
    let imp = imperative_out_1.data_f32().to_vec();
    let comp = compiled_out_1[0].data_f32().to_vec();
    let max_diff = imp
        .iter()
        .zip(&comp)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1.0e-6,
        "shapeless compiled quantized_matmul closure must match imperative (shape 1): max_diff={max_diff}"
    );

    // Shape 2: [1, 4, 8] (prefill shape). The admitted MLX runtime must
    // preserve the same graph across this sequence-length change.
    let x2 = array_f32(
        &(0..32).map(|i| (i as f32) * 0.05).collect::<Vec<_>>(),
        &[1, 4, 8],
    );
    let gate2 = array_f32(&[0.7; 64], &[1, 4, 16]);

    let imperative_out_2 = {
        let h = qw_inner(&qw_captured, &x2);
        let sig = sigmoid(&gate2, None);
        multiply(&h, &sig, None)
    };
    let compiled_out_2 = compiled.apply(&[&x2, &gate2]);
    eval(&[&imperative_out_2, &compiled_out_2[0]]);
    assert_eq!(compiled_out_2[0].shape(), vec![1, 4, 16]);
    let imp = imperative_out_2.data_f32().to_vec();
    let comp = compiled_out_2[0].data_f32().to_vec();
    let max_diff = imp
        .iter()
        .zip(&comp)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1.0e-6,
        "shapeless compiled linear closure must match imperative after a sequence-length change: max_diff={max_diff}"
    );
}

fn v4_test_config(experts: usize, top_k: usize) -> ModelConfig {
    ModelConfig {
        compile_cache_identity: 1,
        model_family: "deepseek_v4".to_string(),
        layer_count: 1,
        hidden_size: 4,
        intermediate_size: 8,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 8,
        vocab_size: 16,
        rope_theta: 10000.0,
        rope_dims: 8,
        attn_output_gate: false,
        query_scale: 1.0,
        final_logit_softcapping: None,
        final_logits_scale: None,
        post_norm_eps: 1e-6,
        embed_norm_no_weight: false,
        moe_expert_count: experts,
        moe_experts_per_token: top_k,
        moe_expert_intermediate_size: 8,
        layer_configs: Vec::new(),
        global_sliding_window: None,
        protected_prefix_sliding_window: None,
        gemma4_moe_router: false,
        uses_geglu: false,
        hidden_states_scale: None,
        moe_norm_topk_prob: true,
        hidden_size_per_layer_input: 0,
        linear_attention: None,
        mla_attention: None,
        glm_router: None,
        deepseek_v4: None,
        rms_norm_eps: 1e-6,
        rope_freqs: None,
        rope_mscale: 1.0,
        no_rope_layer_interval: 0,
        attn_temperature_floor: 8192.0,
        attn_temperature_scale: 0.1,
        intermediate_size_mlp: 0,
        moe_layer_freq: 1,
        moe_first_dense_layers: 0,
        moe_shared_expert_count: 0,
        moe_sigmoid_routing: false,
        moe_routed_scaling_factor: 2.5,
        moe_n_group: 1,
        moe_topk_group: 1,
        think_start_token_id: None,
        think_end_token_id: None,
        diffusion: None,
        generation_kind: ax_engine_core::GenerationKind::Autoregressive,
        kv_cache_quant: vec![None; 1],
    }
}

fn v4_layer_weights(router: MlxArray, x: &MlxArray) -> LayerWeights {
    LayerWeights {
        attn_norm: x.clone(),
        attn_post_norm: None,
        q_norm: None,
        k_norm: None,
        q_proj: None,
        k_proj: None,
        v_proj: None,
        qkv_packed: None,
        attn_out_gate: None,
        o_proj: None,
        linear_attn: None,
        glm_mla_attn: None,
        deepseek_v4: None,
        ffn_norm: x.clone(),
        ffn_post_norm: None,
        gate_proj: None,
        up_proj: None,
        gate_up_packed: None,
        down_proj: None,
        ffn_norm2: None,
        ffn_post_norm1: None,
        ffn_post_norm2: None,
        router_proj: Some(QuantizedWeight::new(router, None, None)),
        router_correction_bias: None,
        router_scale: None,
        router_combined_scale: None,
        router_expert_scale: None,
        layer_scalar: None,
        per_layer_gate: None,
        per_layer_proj_w: None,
        per_layer_post_norm: None,
        shared_expert_gate: None,
        shared_gate_up_proj: None,
        shared_gate_proj: None,
        shared_up_proj: None,
        shared_down_proj: None,
        gate_up_exps_packed: None,
        gate_exps: None,
        up_exps: None,
        down_exps: None,
        attn_sink: None,
        rotation_smoothing_inverse: None,
        expert_stream: None,
    }
}

#[test]
fn compiled_v4_clamped_swiglu_matches_imperative() {
    let half = mlx_sys::ops::cached_scalar(0.5_f32, MlxDtype::Bfloat16);
    let one_half = mlx_sys::ops::cached_scalar(1.5_f32, MlxDtype::Bfloat16);
    let zeros_g = zeros(&[1, 1, 8], MlxDtype::Bfloat16, None);
    let gate = add(&zeros_g, &half, None);
    let up = add(&zeros_g, &one_half, None);
    let compiled = try_compiled_v4_clamped_swiglu(&gate, &up, 7.0).expect("V4 SwiGLU must compile");
    let imperative = v4_clamped_swiglu_imperative(&gate, &up, 7.0);
    mlx_sys::eval(&[&compiled, &imperative]);
    let compiled_f = astype(&compiled, MlxDtype::Float32, None);
    let imperative_f = astype(&imperative, MlxDtype::Float32, None);
    mlx_sys::eval(&[&compiled_f, &imperative_f]);
    let a = compiled_f.data_f32();
    let b = imperative_f.data_f32();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}

/// Manual `sqrt(softplus(x))` in plain f64 arithmetic for comparisons.
fn manual_sqrt_softplus(x: f32) -> f64 {
    let x = x as f64;
    let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
    softplus.sqrt()
}

#[test]
fn sqrt_softplus_scores_matches_manual() {
    let logits = array_f32(&[-20.0, -1.5, 0.0, 0.5, 3.0, 20.0], &[1, 1, 6]);
    let scores = sqrt_softplus_scores(&logits);
    eval(&[&scores]);
    assert_eq!(scores.dtype(), MlxDtype::Float32);
    let actual = scores.data_f32().to_vec();
    let expected: Vec<f32> = [-20.0, -1.5, 0.0, 0.5, 3.0, 20.0]
        .iter()
        .map(|x| manual_sqrt_softplus(*x) as f32)
        .collect();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn moe_router_deepseek_v4_learned_path_matches_manual() {
    // Identity gate: logits == x. top-2 of 4 experts, norm_topk_prob on,
    // routed_scaling_factor 2.5 (from v4_test_config).
    let cfg = v4_test_config(4, 2);
    let router = array_f32(
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
    );
    let x = array_f32(&[0.5, 3.0, -1.0, 1.5], &[1, 1, 4]);
    let mut w = v4_layer_weights(router, &x);
    // Correction bias flips the selection: expert 0 beats expert 3.
    w.router_correction_bias = Some(array_f32(&[3.0, 0.0, 0.0, 0.0], &[4]));

    let (indices, weights) = moe_router_deepseek_v4(&cfg, &w, &x, None);
    let indices = astype(&indices, MlxDtype::Uint32, None);
    eval(&[&indices, &weights]);
    assert_eq!(indices.shape(), vec![1, 1, 2]);
    assert_eq!(weights.shape(), vec![1, 1, 2]);

    // Selection scores: probs + bias → experts 1 (prob √softplus(3) ≈ 1.74)
    // and 0 (≈0.93 + 3.0) win. Weights come from the UNBIASED probs,
    // renormalised then scaled by 2.5.
    let p = [0.5_f32, 3.0, -1.0, 1.5].map(manual_sqrt_softplus);
    let sel: Vec<f64> = [p[0] + 3.0, p[1], p[2], p[3]].to_vec();
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by(|a, b| sel[*b].total_cmp(&sel[*a]));
    let (e0, e1) = (order[0], order[1]);
    let w0 = p[e0] / (p[e0] + p[e1]) * 2.5;
    let w1 = p[e1] / (p[e0] + p[e1]) * 2.5;

    let idx = indices.data_u32().to_vec();
    let got = weights.data_f32().to_vec();
    let mut got_pairs: Vec<(u32, f32)> = idx.into_iter().zip(got).collect();
    got_pairs.sort_by_key(|(i, _)| *i);
    let mut expect_pairs: Vec<(u32, f32)> = [(e0 as u32, w0 as f32), (e1 as u32, w1 as f32)]
        .into_iter()
        .collect();
    expect_pairs.sort_by_key(|(i, _)| *i);
    assert_eq!(got_pairs.len(), expect_pairs.len());
    for ((gi, gw), (ei, ew)) in got_pairs.iter().zip(expect_pairs.iter()) {
        assert_eq!(gi, ei, "selected expert mismatch");
        assert!((gw - ew).abs() < 1e-4, "weight {gw} vs expected {ew}");
    }
}

#[test]
fn moe_router_deepseek_v4_hash_path_uses_tid2eid_indices() {
    // Hash routing: indices come from the tid2eid table at token_ids;
    // weights still come from the unbiased sqrtsoftplus probs.
    let cfg = v4_test_config(4, 2);
    let router = array_f32(
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
    );
    let x = array_f32(&[0.5, 3.0, -1.0, 1.5], &[1, 1, 4]);
    let mut w = v4_layer_weights(router, &x);
    // [vocab=16, topk=2] table; token 7 routes to experts (2, 0).
    let mut table = vec![0u32; 16 * 2];
    table[7 * 2] = 2;
    table[7 * 2 + 1] = 0;
    let tid2eid = MlxArray::from_raw_data(
        table.as_ptr() as *const u8,
        std::mem::size_of_val(table.as_slice()),
        &[16, 2],
        MlxDtype::Uint32,
    );
    w.deepseek_v4 = Some(crate::weights::DeepseekV4LayerWeights {
        wq_a: QuantizedWeight::new(x.clone(), None, None),
        q_a_norm: x.clone(),
        wq_b: QuantizedWeight::new(x.clone(), None, None),
        wkv: QuantizedWeight::new(x.clone(), None, None),
        kv_norm: x.clone(),
        wo_a: QuantizedWeight::new(x.clone(), None, None),
        wo_b: QuantizedWeight::new(x.clone(), None, None),
        attn_sink: None,
        hc_attn_fn: x.clone(),
        hc_attn_base: x.clone(),
        hc_attn_scale: x.clone(),
        hc_ffn_fn: x.clone(),
        hc_ffn_base: x.clone(),
        hc_ffn_scale: x.clone(),
        compressor: None,
        indexer: None,
        tid2eid: Some(tid2eid),
    });
    // Deliberately WRONG correction bias: hash routing must ignore it.
    w.router_correction_bias = Some(array_f32(&[0.0, 0.0, 100.0, 0.0], &[4]));

    let token_ids = MlxArray::from_raw_data(
        [7u32].as_ptr() as *const u8,
        std::mem::size_of::<u32>(),
        &[1, 1],
        MlxDtype::Uint32,
    );
    let (indices, weights) = moe_router_deepseek_v4(&cfg, &w, &x, Some(&token_ids));
    let indices = astype(&indices, MlxDtype::Uint32, None);
    eval(&[&indices, &weights]);
    assert_eq!(indices.shape(), vec![1, 1, 2]);

    let p = [0.5_f32, 3.0, -1.0, 1.5].map(manual_sqrt_softplus);
    let w2 = p[2] / (p[2] + p[0]) * 2.5;
    let w0 = p[0] / (p[2] + p[0]) * 2.5;

    let idx = indices.data_u32().to_vec();
    let got = weights.data_f32().to_vec();
    let mut got_pairs: Vec<(u32, f32)> = idx.into_iter().zip(got).collect();
    got_pairs.sort_by_key(|(i, _)| *i);
    let expect_pairs: Vec<(u32, f32)> = vec![(0, w0 as f32), (2, w2 as f32)];
    for ((gi, gw), (ei, ew)) in got_pairs.iter().zip(expect_pairs.iter()) {
        assert_eq!(gi, ei, "hash-routed expert mismatch");
        assert!((gw - ew).abs() < 1e-4, "weight {gw} vs expected {ew}");
    }
}
