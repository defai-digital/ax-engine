use super::*;

#[test]
fn qwen_la_projection_preserves_float32_gate_before_output_cast() {
    use mlx_sys::{astype, quantize, quantized_matmul, silu_mul};
    let _exact = fastpath::scoped_qwen_linear_mtp_exact(true);
    for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
        for seq in [2, 4, 8] {
            let array = |values: Vec<f32>, shape: &[i32]| {
                astype(
                    &MlxArray::from_raw_data(
                        values.as_ptr() as *const u8,
                        std::mem::size_of_val(values.as_slice()),
                        shape,
                        MlxDtype::Float32,
                    ),
                    dtype,
                    None,
                )
            };
            let hidden = array(
                (0..seq * 64)
                    .map(|i| ((i % 73) as f32 - 35.0) * 0.053)
                    .collect(),
                &[1, seq, 2, 32],
            );
            let gate = array(
                (0..seq * 64)
                    .map(|i| ((i % 97) as f32 - 47.0) * 0.031)
                    .collect(),
                &[1, seq, 2, 32],
            );
            let norm = array(vec![1.0; 32], &[32]);
            // Identity rows keep the activation precision observable.
            let weight = array(
                (0..64 * 64)
                    .map(|i| if i / 64 == i % 64 { 1.0 } else { 0.0 })
                    .collect(),
                &[64, 64],
            );
            let q = quantize(
                &weight,
                Some(32),
                Some(4),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            let out_proj = QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: Some(q[2].clone()),
                group_size: 32,
                bits: 4,
                mode: "affine".into(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
            };
            let normed = rms_norm(&hidden, Some(&norm), 1e-6, None);
            let expected_gate = astype(
                &silu_mul(
                    &astype(&gate, MlxDtype::Float32, None),
                    &astype(&normed, MlxDtype::Float32, None),
                    None,
                ),
                dtype,
                None,
            );
            let expected = quantized_matmul(
                &reshape(&expected_gate, &[1, seq, 64], None),
                &q[0],
                &q[1],
                Some(&q[2]),
                true,
                Some(32),
                Some(4),
                None,
            );
            let actual = qwen_la_gated_out_projection(
                &hidden, &gate, &norm, &out_proj, 1e-6, seq, 64, false,
            )
            .expect("affine projection must engage");
            let a = astype(&actual, MlxDtype::Float32, None);
            let b = astype(&expected, MlxDtype::Float32, None);
            eval(&[&a, &b]);
            let max_abs = a
                .data_f32()
                .iter()
                .zip(b.data_f32())
                .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()));
            assert_eq!(
                a.data_f32(),
                b.data_f32(),
                "dtype={dtype:?} seq={seq} max_abs={max_abs}"
            );
            let _relaxed = fastpath::scoped_qwen_linear_mtp_exact(false);
            for allow_full in [false, true] {
                let gated =
                    rms_norm_gated_with_full_gate_policy(&hidden, &gate, &norm, 1e-6, allow_full);
                let expected = qw(&reshape(&gated, &[1, seq, 64], None), &out_proj);
                let actual = qwen_la_gated_out_projection(
                    &hidden, &gate, &norm, &out_proj, 1e-6, seq, 64, allow_full,
                )
                .expect("relaxed projection must engage");
                let a = astype(&actual, MlxDtype::Float32, None);
                let b = astype(&expected, MlxDtype::Float32, None);
                eval(&[&a, &b]);
                assert_eq!(
                    a.data_f32(),
                    b.data_f32(),
                    "relaxed dtype={dtype:?} seq={seq} full={allow_full}"
                );
            }
        }
    }
}

const TEST_COMPILE_IDENTITY: u64 = 0x5445_5354_4C41_4348;

#[test]
fn qwen_five_bit_full_gate_policy_keeps_only_early_layers() {
    assert!(linear_attention_full_gate_metal_allowed_for_bits(
        "qwen3_5", true, 5, 15
    ));
    assert!(!linear_attention_full_gate_metal_allowed_for_bits(
        "qwen3_5", true, 5, 16
    ));
    assert!(linear_attention_full_gate_metal_allowed_for_bits(
        "qwen3_5", true, 4, 63
    ));
    assert!(linear_attention_full_gate_metal_allowed_for_bits(
        "glm4_moe_lite",
        true,
        5,
        63
    ));
    assert!(linear_attention_full_gate_metal_allowed_for_bits(
        "qwen3_5", false, 5, 63
    ));
}

#[test]
fn exact_s2_full_gate_metal_follows_early_layer_policy() {
    assert!(exact_s2_full_gate_metal_allowed(2, 0, true));
    assert!(exact_s2_full_gate_metal_allowed(2, 15, true));
    assert!(!exact_s2_full_gate_metal_allowed(2, 16, true));
    assert!(!exact_s2_full_gate_metal_allowed(1, 0, true));
    assert!(!exact_s2_full_gate_metal_allowed(2, 0, false));
    assert!(exact_s2_full_gate_metal_allowed(4, 7, true));
}

#[test]
fn exact_s2_s1_metal_gate_o_proj_matches_two_s1_rows() {
    let seq = 2i32;
    let hv = 2i32;
    let dv = 32i32;
    let value_dim = hv * dv;
    let n = (seq * hv * dv) as usize;
    let hidden_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
    let gate_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
    let norm_data: Vec<f32> = (0..dv as usize)
        .map(|i| 0.75 + (i as f32) * 0.004)
        .collect();
    let o_data: Vec<f32> = (0..(value_dim * value_dim) as usize)
        .map(|i| ((i as f32) - 20.0) * 0.004)
        .collect();
    let to_bf16 = |data: &[f32], shape: &[i32]| {
        mlx_sys::astype(
            &MlxArray::from_raw_data(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data),
                shape,
                MlxDtype::Float32,
            ),
            MlxDtype::Bfloat16,
            None,
        )
    };
    let hidden = to_bf16(&hidden_data, &[1, seq, hv, dv]);
    let gate = to_bf16(&gate_data, &[1, seq, hv, dv]);
    let norm = to_bf16(&norm_data, &[dv]);
    let o_w = MlxArray::from_raw_data(
        o_data.as_ptr() as *const u8,
        std::mem::size_of_val(o_data.as_slice()),
        &[value_dim, value_dim],
        MlxDtype::Float32,
    );
    let q = mlx_sys::quantize(
        &o_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    let out_proj = QuantizedWeight {
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
    };
    let dummy = zeros(&[1], MlxDtype::Float32, None);
    let linear_w = LinearAttentionWeights {
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
        norm: norm.clone(),
        out_proj: out_proj.clone(),
    };
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let shipped =
        exact_verify_s1_metal_gate_o_proj(&hidden, &gate, &linear_w, 1e-6, value_dim, seq, true)
            .expect("exact S=2 per-row S=1 Metal gate+o_proj must engage");
    let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let r0 = {
        let h = slice_seq_row_4d(&hidden, 0);
        let g = slice_seq_row_4d(&gate, 0);
        let gated = rms_norm_gated_with_full_gate_policy(&h, &g, &norm, 1e-6, true);
        qw(&reshape(&gated, &[1, 1, value_dim], None), &out_proj)
    };
    let r1 = {
        let h = slice_seq_row_4d(&hidden, 1);
        let g = slice_seq_row_4d(&gate, 1);
        let gated = rms_norm_gated_with_full_gate_policy(&h, &g, &norm, 1e-6, true);
        qw(&reshape(&gated, &[1, 1, value_dim], None), &out_proj)
    };
    let expected = concatenate(&[&r0, &r1], 1, None);
    eval(&[&shipped, &expected]);
    let a = mlx_sys::astype(&shipped, MlxDtype::Float32, None);
    let b = mlx_sys::astype(&expected, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    let mut max_abs = 0.0f32;
    for (l, r) in a.data_f32().iter().zip(b.data_f32().iter()) {
        max_abs = max_abs.max((l - r).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "per-row S=1 Metal gate+o_proj must match two decode rows, max_abs={max_abs}"
    );
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    assert!(
        exact_verify_s1_metal_gate_o_proj(&hidden, &gate, &linear_w, 1e-6, value_dim, seq, true)
            .is_none(),
        "per-row S=1 path must stay off when exact MTP is scoped off"
    );
}

#[test]
fn mixed_pack_is_prefill_only() {
    assert!(linear_attention_prefill_allows_mixed_pack(128, true));
    assert!(linear_attention_prefill_allows_mixed_pack(512, true));
    assert!(linear_attention_prefill_allows_mixed_pack(2048, true));
    assert!(!linear_attention_prefill_allows_mixed_pack(1, true));
    assert!(linear_attention_prefill_allows_mixed_pack(1, false));
    assert!(linear_attention_prefill_allows_mixed_pack(128, false));
}

#[test]
fn qwen_prefill_maybe_skip_unused_la_out_returns_input_when_set() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let x = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(data.as_slice()),
        &[1, 32, 1],
        MlxDtype::Float32,
    );
    let skipped = qwen_prefill_maybe_skip_unused_la_out(&x, true)
        .expect("skip must return the unused residual placeholder");
    mlx_sys::eval(&[&skipped]);
    assert_eq!(skipped.shape(), x.shape());
    assert!(
        skipped.data_f32().iter().all(|v| v.is_finite()),
        "skipped unused LA out must leave a finite placeholder"
    );
    assert!(qwen_prefill_maybe_skip_unused_la_out(&x, false).is_none());
    assert!(
        fastpath::should_qwen_prefill_skip_unused_la_out_for(true, "qwen3_5", true, 1024),
        "shipped unused-LA-out skip must accept the p2048 cache-only last layer"
    );
}

#[test]
fn qwen_prefill_maybe_last_token_la_out_slices_seq_when_set() {
    let out_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let z_data: Vec<f32> = (0..16).map(|i| (i as f32) + 100.0).collect();
    let out = MlxArray::from_raw_data(
        out_data.as_ptr() as *const u8,
        std::mem::size_of_val(out_data.as_slice()),
        &[1, 4, 2, 2],
        MlxDtype::Float32,
    );
    let z = MlxArray::from_raw_data(
        z_data.as_ptr() as *const u8,
        std::mem::size_of_val(z_data.as_slice()),
        &[1, 4, 2, 2],
        MlxDtype::Float32,
    );
    let (sliced_out, sliced_z, seq) = qwen_prefill_maybe_last_token_la_out(&out, &z, true)
        .expect("last-token LA out must slice when set");
    mlx_sys::eval(&[&sliced_out, &sliced_z]);
    assert_eq!(seq, 1);
    assert_eq!(sliced_out.shape(), vec![1, 1, 2, 2]);
    assert_eq!(sliced_z.shape(), vec![1, 1, 2, 2]);
    assert_eq!(sliced_out.data_f32(), vec![12.0, 13.0, 14.0, 15.0]);
    assert_eq!(sliced_z.data_f32(), vec![112.0, 113.0, 114.0, 115.0]);
    assert!(qwen_prefill_maybe_last_token_la_out(&out, &z, false).is_none());
    assert!(
        fastpath::should_qwen_prefill_last_token_o_proj_for(true, "qwen3_5", true, 1024),
        "shipped last-token o_proj must accept the p2048 generate last layer"
    );
}

#[test]
fn qwen_prefill_maybe_async_gd_submits_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let gd = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(data.as_slice()),
        &[1, 4, 2, 4],
        MlxDtype::Float32,
    );
    qwen_prefill_maybe_async_gd_for(&gd, true, 1024);
    mlx_sys::eval(&[&gd]);
    assert_eq!(gd.shape(), vec![1, 4, 2, 4]);
    assert!(
        gd.data_f32().iter().all(|v| v.is_finite()),
        "async GD must leave a finite materialized tensor"
    );
    assert!(
        fastpath::should_qwen_prefill_async_gd_for(true, 1024),
        "shipped async-GD gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_async_gd_for(&gd, false, 1024);
    qwen_prefill_maybe_async_gd_for(&gd, true, 512);
}

#[test]
fn qwen_prefill_maybe_eval_gd_materializes_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let gd = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(data.as_slice()),
        &[1, 4, 2, 4],
        MlxDtype::Float32,
    );
    qwen_prefill_maybe_eval_gd_for(&gd, true, 1024);
    mlx_sys::eval(&[&gd]);
    assert_eq!(gd.shape(), vec![1, 4, 2, 4]);
    assert!(
        gd.data_f32().iter().all(|v| v.is_finite()),
        "eval GD must leave a finite materialized tensor"
    );
    assert!(
        fastpath::should_qwen_prefill_eval_gd_for(true, 1024),
        "shipped eval-GD gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_eval_gd_for(&gd, false, 1024);
    qwen_prefill_maybe_eval_gd_for(&gd, true, 512);
}

#[test]
fn qwen_prefill_maybe_contiguous_gd_packs_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let gd = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(data.as_slice()),
        &[1, 4, 2, 4],
        MlxDtype::Float32,
    );
    let packed = qwen_prefill_maybe_contiguous_gd_for(gd, true, 1024);
    mlx_sys::eval(&[&packed]);
    assert_eq!(packed.shape(), vec![1, 4, 2, 4]);
    assert!(
        packed.data_f32().iter().all(|v| v.is_finite()),
        "contiguous GD must leave a finite packed tensor"
    );
    assert!(
        fastpath::should_qwen_prefill_contiguous_gd_for(true, 1024),
        "shipped contiguous-GD gate must accept the p2048 chunk length"
    );
    let data2: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let gd2 = MlxArray::from_raw_data(
        data2.as_ptr() as *const u8,
        std::mem::size_of_val(data2.as_slice()),
        &[1, 4, 2, 4],
        MlxDtype::Float32,
    );
    let kept = qwen_prefill_maybe_contiguous_gd_for(gd2, false, 1024);
    mlx_sys::eval(&[&kept]);
    assert_eq!(kept.shape(), vec![1, 4, 2, 4]);
}

#[test]
fn cached_prefill_la_contiguous_weight_keeps_bits_and_qws() {
    let hidden_data: Vec<f32> = (0..32 * 32)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let proj_data: Vec<f32> = (0..64 * 32)
        .map(|i| ((i as f32) - 768.0) * 0.0003)
        .collect();
    let hidden = MlxArray::from_raw_data(
        hidden_data.as_ptr() as *const u8,
        std::mem::size_of_val(hidden_data.as_slice()),
        &[1, 32, 32],
        MlxDtype::Float32,
    );
    let proj_w = MlxArray::from_raw_data(
        proj_data.as_ptr() as *const u8,
        std::mem::size_of_val(proj_data.as_slice()),
        &[64, 32],
        MlxDtype::Float32,
    );
    let dq = mlx_sys::quantize(
        &proj_w,
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Affine,
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
    let contig = cached_prefill_la_contiguous_weight(&src);
    assert_eq!(contig.bits, 4);
    assert_eq!(contig.group_size, 32);
    let again = cached_prefill_la_contiguous_weight(&src);
    assert_eq!(again.bits, contig.bits);
    let out = qw(&hidden, &contig);
    mlx_sys::eval(&[&out]);
    assert_eq!(out.shape(), vec![1, 32, 64]);
    assert!(
        out.data_f32().iter().all(|v| v.is_finite()),
        "contiguous LA-weight qmm must produce finite values"
    );
    assert!(
        fastpath::should_qwen_prefill_contiguous_la_weights_for(true, 1024),
        "shipped LA contiguous-weight gate must accept the p2048 chunk length"
    );
}

#[test]
fn qwen_prefill_maybe_eval_la_input_materializes_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let x = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(data.as_slice()),
        &[1, 32, 1],
        MlxDtype::Float32,
    );
    qwen_prefill_maybe_eval_la_input_for(&x, true, 1024);
    mlx_sys::eval(&[&x]);
    assert_eq!(x.shape(), vec![1, 32, 1]);
    assert!(
        x.data_f32().iter().all(|v| v.is_finite()),
        "eval-la-input must leave a finite materialized activation"
    );
    assert!(
        fastpath::should_qwen_prefill_eval_la_input_for(true, 1024),
        "shipped LA input-eval gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_eval_la_input_for(&x, false, 1024);
    qwen_prefill_maybe_eval_la_input_for(&x, true, 512);
}

#[test]
fn qwen_prefill_maybe_async_la_outputs_submits_at_min_seq() {
    let qkv_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let z_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125 + 0.1).collect();
    let a_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125 + 0.2).collect();
    let b_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125 + 0.3).collect();
    let from = |data: &[f32]| {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            &[1, 32, 1],
            MlxDtype::Float32,
        )
    };
    let qkv = from(&qkv_data);
    let z = from(&z_data);
    let a = from(&a_data);
    let b = from(&b_data);
    qwen_prefill_maybe_async_la_outputs_for(&qkv, &z, &a, &b, true, 1024);
    mlx_sys::eval(&[&qkv, &z, &a, &b]);
    assert_eq!(qkv.shape(), vec![1, 32, 1]);
    assert!(
        qkv.data_f32()
            .iter()
            .chain(z.data_f32().iter())
            .chain(a.data_f32().iter())
            .chain(b.data_f32().iter())
            .all(|v| v.is_finite()),
        "async LA outputs must leave finite materialized tensors"
    );
    assert!(
        fastpath::should_qwen_prefill_async_la_outputs_for(true, 1024),
        "shipped async LA-outputs gate must accept the p2048 chunk length"
    );
    qwen_prefill_maybe_async_la_outputs_for(&qkv, &z, &a, &b, false, 1024);
    qwen_prefill_maybe_async_la_outputs_for(&qkv, &z, &a, &b, true, 512);
}

#[test]
fn packed_la_inputs_compile_matches_imperative_at_min_seq() {
    let seq = fastpath::QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ;
    let hidden = 32_i32;
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 4,
        num_key_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        conv_kernel_dim: 4,
        q_scale: 0.25,
        k_scale: 0.5,
    };
    let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
    let x_data: Vec<f32> = (0..(seq * hidden))
        .map(|i| ((i as f32) - 31.0) * 0.015625)
        .collect();
    let qkvz_data: Vec<f32> = (0..(qkvz_out * hidden))
        .map(|i| ((i as f32) - 400.0) * 0.0005)
        .collect();
    let ba_data: Vec<f32> = (0..(ba_out * hidden))
        .map(|i| ((i as f32) - 80.0) * 0.001)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let qkvz_w = MlxArray::from_raw_data(
        qkvz_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkvz_data.as_slice()),
        &[qkvz_out, hidden],
        MlxDtype::Float32,
    );
    let ba_w = MlxArray::from_raw_data(
        ba_data.as_ptr() as *const u8,
        std::mem::size_of_val(ba_data.as_slice()),
        &[ba_out, hidden],
        MlxDtype::Float32,
    );
    let qkvz_q = mlx_sys::quantize(
        &qkvz_w,
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Affine,
        None,
        None,
    );
    let ba_q = mlx_sys::quantize(
        &ba_w,
        Some(32),
        Some(6),
        mlx_sys::MlxQuantizationMode::Affine,
        None,
        None,
    );
    let qkvz_qw = affine_quant_weight(
        qkvz_q[0].clone(),
        qkvz_q[1].clone(),
        qkvz_q[2].clone(),
        4,
        32,
    );
    let ba_qw = affine_quant_weight(ba_q[0].clone(), ba_q[1].clone(), ba_q[2].clone(), 6, 32);
    assert!(
        fastpath::should_qwen_packed_la_inputs_compile_for(true, seq),
        "shipped compile gate must accept the p2048 chunk length"
    );
    let (compiled_qkv, compiled_z, compiled_a, compiled_b) =
        qwen_linear_attention_inputs_packed_compiled(
            &x,
            &qkvz_qw.weight,
            qkvz_qw.scales.as_ref().expect("qkvz scales"),
            qkvz_qw.biases.as_ref(),
            &ba_qw.weight,
            ba_qw.scales.as_ref().expect("ba scales"),
            ba_qw.biases.as_ref(),
            cfg.num_key_heads as i32,
            cfg.num_value_heads as i32,
            cfg.key_head_dim as i32,
            cfg.value_head_dim as i32,
            32,
            4,
            32,
            6,
            None,
        )
        .expect("compiled packed LA inputs must engage at seq>=1024");
    let (imp_qkv, imp_z, imp_a, imp_b) = qwen_linear_attention_inputs_packed(
        &x,
        &qkvz_qw.weight,
        qkvz_qw.scales.as_ref(),
        qkvz_qw.biases.as_ref(),
        &ba_qw.weight,
        ba_qw.scales.as_ref(),
        ba_qw.biases.as_ref(),
        cfg.num_key_heads as i32,
        cfg.num_value_heads as i32,
        cfg.key_head_dim as i32,
        cfg.value_head_dim as i32,
        32,
        4,
        32,
        6,
        None,
    )
    .expect("imperative packed LA inputs must stay the fallback");
    mlx_sys::eval(&[
        &compiled_qkv,
        &compiled_z,
        &compiled_a,
        &compiled_b,
        &imp_qkv,
        &imp_z,
        &imp_a,
        &imp_b,
    ]);
    for (got, want, name) in [
        (&compiled_qkv, &imp_qkv, "qkv"),
        (&compiled_z, &imp_z, "z"),
        (&compiled_a, &imp_a, "a"),
        (&compiled_b, &imp_b, "b"),
    ] {
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let g = got.data_f32();
        let w = want.data_f32();
        assert_eq!(g.len(), w.len(), "{name} len");
        for i in 0..g.len() {
            let err = (g[i] - w[i]).abs();
            assert!(
                err < 2.0e-4,
                "{name}[{i}] compiled {} imperative {} err {err}",
                g[i],
                w[i]
            );
        }
    }
}

#[test]
fn contiguous_packed_qkv_post_input_matches_view() {
    let seq = 2_i32;
    let num_key_heads = 2_i32;
    let key_head_dim = 4_i32;
    let num_value_heads = 4_i32;
    let value_head_dim = 3_i32;
    let conv_kernel_dim = 4_i32;
    let conv_dim = num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim;
    let tail = conv_kernel_dim - 1;
    let qkv_data: Vec<f32> = (0..(seq * conv_dim))
        .map(|i| ((i as f32) - 16.0) * 0.0625)
        .collect();
    let conv_data: Vec<f32> = (0..(conv_dim * conv_kernel_dim))
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let state_data: Vec<f32> = (0..(tail * conv_dim))
        .map(|i| ((i as f32) - 8.0) * 0.03125)
        .collect();
    let dense = MlxArray::from_raw_data(
        qkv_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkv_data.as_slice()),
        &[1, seq, conv_dim],
        MlxDtype::Float32,
    );
    let half = conv_dim / 2;
    let view = concatenate(
        &[
            &slice_last_dim(&dense, 0, half, None),
            &slice_last_dim(&dense, half, conv_dim, None),
        ],
        2,
        None,
    );
    let compact = contiguous(&view, None);
    let conv_weight = MlxArray::from_raw_data(
        conv_data.as_ptr() as *const u8,
        std::mem::size_of_val(conv_data.as_slice()),
        &[conv_dim, conv_kernel_dim, 1],
        MlxDtype::Float32,
    );
    let state = MlxArray::from_raw_data(
        state_data.as_ptr() as *const u8,
        std::mem::size_of_val(state_data.as_slice()),
        &[1, tail, conv_dim],
        MlxDtype::Float32,
    );
    let (q_s, k_s, v_s, st_s) = qwen_linear_attention_post_input(
        &view,
        &conv_weight,
        Some(&state),
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        conv_kernel_dim,
        0.5,
        0.5,
        1.0e-6,
        None,
    )
    .expect("view qkv post-input");
    let (q_c, k_c, v_c, st_c) = qwen_linear_attention_post_input(
        &compact,
        &conv_weight,
        Some(&state),
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        conv_kernel_dim,
        0.5,
        0.5,
        1.0e-6,
        None,
    )
    .expect("contiguous qkv post-input");
    mlx_sys::eval(&[&q_s, &k_s, &v_s, &st_s, &q_c, &k_c, &v_c, &st_c]);
    assert_eq!(q_s.shape(), q_c.shape());
    assert_eq!(k_s.shape(), k_c.shape());
    assert_eq!(v_s.shape(), v_c.shape());
    assert_eq!(st_s.shape(), st_c.shape());
    for (got, want, name) in [
        (&q_c, &q_s, "q"),
        (&k_c, &k_s, "k"),
        (&v_c, &v_s, "v"),
        (&st_c, &st_s, "state"),
    ] {
        let g = contiguous(got, None);
        let w = contiguous(want, None);
        mlx_sys::eval(&[&g, &w]);
        let gd = g.data_f32();
        let wd = w.data_f32();
        assert_eq!(gd.len(), wd.len(), "{name} len");
        for i in 0..gd.len() {
            let err = (gd[i] - wd[i]).abs();
            assert!(
                err < 2.0e-4,
                "{name}[{i}] contiguous {} view {} err {err}",
                gd[i],
                wd[i]
            );
        }
    }
    assert!(
        fastpath::should_qwen_la_contiguous_qkv_for(true, 1024),
        "shipped contiguous gate must accept the p2048 chunk length"
    );
}

fn affine_quant_weight(
    weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    bits: i32,
    group_size: i32,
) -> QuantizedWeight {
    QuantizedWeight {
        weight,
        scales: Some(scales),
        biases: Some(biases),
        group_size,
        bits,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    }
}

#[test]
fn matching_affine_quant_rejects_mxfp4_even_when_bits_match() {
    let w = MlxArray::from_raw_data(
        [0.1f32; 64].as_ptr() as *const u8,
        64 * std::mem::size_of::<f32>(),
        &[2, 32],
        MlxDtype::Float32,
    );
    let affine = mlx_sys::quantize(
        &w,
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Affine,
        None,
        None,
    );
    let mxfp4 = mlx_sys::quantize(
        &w,
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    let affine_qw = QuantizedWeight {
        weight: affine[0].clone(),
        scales: Some(affine[1].clone()),
        biases: Some(affine[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let mxfp4_qw = QuantizedWeight {
        weight: mxfp4[0].clone(),
        scales: Some(mxfp4[1].clone()),
        biases: None,
        group_size: 32,
        bits: 4,
        mode: "mxfp4".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    assert!(affine_qw.is_affine_quantized());
    assert!(!mxfp4_qw.is_affine_quantized());
    assert!(mxfp4_qw.is_mxfp4_quantized());
    assert!(!affine_qw.is_mxfp4_quantized());
    assert!(affine_qw.is_fused_qmm_quantized());
    assert!(mxfp4_qw.is_fused_qmm_quantized());
    assert!(affine_qw.matching_affine_quant(&affine_qw));
    assert!(!mxfp4_qw.matching_affine_quant(&mxfp4_qw));
    assert!(!affine_qw.matching_affine_quant(&mxfp4_qw));
    assert!(mxfp4_qw.matching_mxfp4_quant(&mxfp4_qw));
    assert!(!affine_qw.matching_mxfp4_quant(&mxfp4_qw));
    assert!(mxfp4_qw.concat_output_rows(&mxfp4_qw).is_some());
    let mut mislabeled = mxfp4_qw.clone();
    mislabeled.mode = "affine".to_string();
    assert!(!mislabeled.is_affine_quantized());
    assert!(
        mislabeled.is_fused_qmm_quantized(),
        "mislabeled 4/32 no-bias resolves to MXFP4 and stays fused-eligible"
    );
    assert!(matches!(
        mislabeled.mlx_quantization_mode(),
        mlx_sys::MlxQuantizationMode::Mxfp4
    ));
    assert_ne!(
        affine_qw.compile_contract_word(),
        mxfp4_qw.compile_contract_word()
    );
    assert_eq!(
        mxfp4_qw.compile_contract_word(),
        mislabeled.compile_contract_word(),
        "mislabeled affine 4/32 no-bias must share the MXFP4 compile contract"
    );
    assert_eq!(
        compile_quant_contract_salt(&[&mxfp4_qw, &mxfp4_qw]),
        compile_quant_contract_salt(&[&mislabeled, &mislabeled])
    );
    let mut with_linear_bias = mxfp4_qw.clone();
    with_linear_bias.linear_bias = Some(mlx_sys::zeros(&[2], MlxDtype::Float32, None));
    assert_ne!(
        mxfp4_qw.compile_contract_word(),
        with_linear_bias.compile_contract_word(),
        "dense linear-bias presence changes the compiled input layout"
    );
    assert_ne!(
        compile_quant_contract_salt(&[&mxfp4_qw]),
        compile_quant_contract_salt(&[&with_linear_bias])
    );
}

#[test]
fn matching_affine_quant_rejects_mixed_bits() {
    let w = mlx_sys::zeros(&[4, 8], MlxDtype::Uint32, None);
    let s = mlx_sys::zeros(&[4, 1], MlxDtype::Float32, None);
    let b = mlx_sys::zeros(&[4, 1], MlxDtype::Float32, None);
    let q4 = affine_quant_weight(w.clone(), s.clone(), b.clone(), 4, 32);
    let q6 = affine_quant_weight(w, s, b, 6, 32);
    assert!(q4.matching_affine_quant(&q4));
    assert!(!q4.matching_affine_quant(&q6));
    assert!(!should_fuse_qkvz_ba_qmm(&q4, &q6, 1024));
    assert!(
        !fastpath::should_qwen_la_fused_qkvz_ba_qmm_for(true, 1, true),
        "decode must not take the fused prefill qmm"
    );
}

#[test]
fn exact_s2_mxfp4_fused_qkvz_ba_matches_split_qw() {
    let hidden = 32i32;
    let qkvz_out = 64i32;
    let ba_out = 16i32;
    let seq = 2i32;
    let mk = |rows: i32, seed: f32| {
        let n = (rows * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) - seed) * 0.015625).collect();
        let w = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[rows, hidden],
            MlxDtype::Float32,
        );
        let q = mlx_sys::quantize(
            &w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
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
    let qkvz = mk(qkvz_out, 8.0);
    let ba = mk(ba_out, 3.0);
    assert!(qkvz.matching_mxfp4_quant(&ba));
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 16.0) * 0.03125)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let fused_w = qkvz.concat_output_rows(&ba).expect("mxfp4 concat");
    let fused = qw(&x, &fused_w);
    let split_q = qw(&x, &qkvz);
    let split_b = qw(&x, &ba);
    let expected = concatenate(&[&split_q, &split_b], 2, None);
    eval(&[&fused, &expected]);
    let a = fused.data_f32();
    let b = expected.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "fused MXFP4 QKVZ+BA qmm must match split qw, max_abs={max_abs}"
    );
}

#[test]
fn exact_s2_compiled_qkvz_ba_unpack_matches_imperative() {
    let seq = 2i32;
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 4,
        num_key_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        conv_kernel_dim: 4,
        q_scale: 0.25,
        k_scale: 0.5,
    };
    let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
    let qkvz_data: Vec<f32> = (0..(seq * qkvz_out) as usize)
        .map(|i| ((i as f32) - 12.0) * 0.03125)
        .collect();
    let ba_data: Vec<f32> = (0..(seq * ba_out) as usize)
        .map(|i| ((i as f32) - 4.0) * 0.0625)
        .collect();
    let mixed_qkvz = MlxArray::from_raw_data(
        qkvz_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkvz_data.as_slice()),
        &[1, seq, qkvz_out],
        MlxDtype::Float32,
    );
    let mixed_ba = MlxArray::from_raw_data(
        ba_data.as_ptr() as *const u8,
        std::mem::size_of_val(ba_data.as_slice()),
        &[1, seq, ba_out],
        MlxDtype::Float32,
    );
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let compiled = compiled_split_packed_qkvz_ba_projection(
        TEST_COMPILE_IDENTITY,
        &cfg,
        &mixed_qkvz,
        &mixed_ba,
        1,
        seq,
    )
    .expect("exact S=2 unpack compile must engage");
    let imperative = split_packed_qkvz_ba_projection(&cfg, &mixed_qkvz, &mixed_ba, 1, seq);
    eval(&[
        &compiled.0,
        &compiled.1,
        &compiled.2,
        &compiled.3,
        &imperative.0,
        &imperative.1,
        &imperative.2,
        &imperative.3,
    ]);
    for (got, want, name) in [
        (&compiled.0, &imperative.0, "qkv"),
        (&compiled.1, &imperative.1, "z"),
        (&compiled.2, &imperative.2, "a"),
        (&compiled.3, &imperative.3, "b"),
    ] {
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let g = got.data_f32();
        let w = want.data_f32();
        assert_eq!(g.len(), w.len(), "{name} len");
        for i in 0..g.len() {
            let err = (g[i] - w[i]).abs();
            assert!(
                err < 1.0e-6,
                "{name}[{i}] compiled {} imperative {} err {err}",
                g[i],
                w[i]
            );
        }
    }
    let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    assert!(
        compiled_split_packed_qkvz_ba_projection(
            TEST_COMPILE_IDENTITY,
            &cfg,
            &mixed_qkvz,
            &mixed_ba,
            1,
            seq,
        )
        .is_none(),
        "unpack compile must stay off when exact MTP is scoped off"
    );
}

#[test]
fn exact_s2_compiled_rms_qmm_unpack_matches_rms_then_fused() {
    let seq = 2i32;
    let hidden = 32i32;
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 4,
        num_key_heads: 2,
        key_head_dim: 8,
        value_head_dim: 8,
        conv_kernel_dim: 4,
        q_scale: 0.125,
        k_scale: 0.35355338,
    };
    let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
    let mk = |rows: i32, seed: f32| {
        let n = (rows * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) - seed) * 0.015625).collect();
        let w = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[rows, hidden],
            MlxDtype::Float32,
        );
        let q = mlx_sys::quantize(
            &w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
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
    let qkvz = mk(qkvz_out, 8.0);
    let ba = mk(ba_out, 3.0);
    let x_data: Vec<f32> = (0..(seq * hidden) as usize)
        .map(|i| ((i as f32) - 16.0) * 0.03125)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let norm_data: Vec<f32> = (0..hidden as usize)
        .map(|i| 0.8 + (i as f32) * 0.004)
        .collect();
    let norm_w = MlxArray::from_raw_data(
        norm_data.as_ptr() as *const u8,
        std::mem::size_of_val(norm_data.as_slice()),
        &[hidden],
        MlxDtype::Float32,
    );
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    set_qwen_la_exact_attn_norm(Some((norm_w.clone(), 1e-6)));
    let compiled =
        linear_attention_inputs_fused_qmm(TEST_COMPILE_IDENTITY, &cfg, &x, &qkvz, &ba, None)
            .expect("exact S=2 rms+qmm+unpack compile must engage");
    set_qwen_la_exact_attn_norm(None);
    let normed = rms_norm(&x, Some(&norm_w), 1e-6, None);
    let imperative =
        linear_attention_inputs_fused_qmm(TEST_COMPILE_IDENTITY, &cfg, &normed, &qkvz, &ba, None)
            .expect("exact S=2 qmm+unpack compile must engage");
    eval(&[
        &compiled.0,
        &compiled.1,
        &compiled.2,
        &compiled.3,
        &imperative.0,
        &imperative.1,
        &imperative.2,
        &imperative.3,
    ]);
    for (got, want, name) in [
        (&compiled.0, &imperative.0, "qkv"),
        (&compiled.1, &imperative.1, "z"),
        (&compiled.2, &imperative.2, "a"),
        (&compiled.3, &imperative.3, "b"),
    ] {
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let g = got.data_f32();
        let w = want.data_f32();
        for i in 0..g.len() {
            let err = (g[i] - w[i]).abs();
            assert!(
                err < 1.0e-5,
                "{name}[{i}] rms-folded {} split {} err {err}",
                g[i],
                w[i]
            );
        }
    }
}

#[test]
fn exact_attn_norm_fallback_applies_rms_when_compile_does_not() {
    // batch=2 makes compiled_fused_qkvz_ba_qmm_unpack return None.
    // The layer shell still skips outer RMS whenever the TLS is set.
    let seq = 2i32;
    let batch = 2i32;
    let hidden = 32i32;
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 4,
        num_key_heads: 2,
        key_head_dim: 8,
        value_head_dim: 8,
        conv_kernel_dim: 4,
        q_scale: 0.125,
        k_scale: 0.35355338,
    };
    let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
    let mk = |rows: i32, seed: f32| {
        let n = (rows * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) - seed) * 0.015625).collect();
        let w = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[rows, hidden],
            MlxDtype::Float32,
        );
        let q = mlx_sys::quantize(
            &w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
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
    let qkvz = mk(qkvz_out, 8.0);
    let ba = mk(ba_out, 3.0);
    let x_data: Vec<f32> = (0..(batch * seq * hidden) as usize)
        .map(|i| ((i as f32) - 16.0) * 0.03125)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[batch, seq, hidden],
        MlxDtype::Float32,
    );
    let norm_data: Vec<f32> = (0..hidden as usize)
        .map(|i| 0.8 + (i as f32) * 0.004)
        .collect();
    let norm_w = MlxArray::from_raw_data(
        norm_data.as_ptr() as *const u8,
        std::mem::size_of_val(norm_data.as_slice()),
        &[hidden],
        MlxDtype::Float32,
    );
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    set_qwen_la_exact_attn_norm(Some((norm_w.clone(), 1e-6)));
    let got = linear_attention_inputs_fused_qmm(TEST_COMPILE_IDENTITY, &cfg, &x, &qkvz, &ba, None)
        .expect("fallback fused qmm must still run");
    set_qwen_la_exact_attn_norm(None);
    let normed = rms_norm(&x, Some(&norm_w), 1e-6, None);
    let want =
        linear_attention_inputs_fused_qmm(TEST_COMPILE_IDENTITY, &cfg, &normed, &qkvz, &ba, None)
            .expect("normed fused qmm");
    eval(&[
        &got.0, &got.1, &got.2, &got.3, &want.0, &want.1, &want.2, &want.3,
    ]);
    let g = got.0.data_f32();
    let w = want.0.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..g.len() {
        max_abs = max_abs.max((g[i] - w[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "compile-miss fallback must still apply bound attn_norm, max_abs={max_abs}"
    );
}

#[test]
fn initial_recurrent_state_zeros_reuses_shape() {
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 4,
        num_key_heads: 2,
        key_head_dim: 8,
        value_head_dim: 4,
        conv_kernel_dim: 4,
        q_scale: 0.125,
        k_scale: 0.35355338,
    };
    let a = initial_recurrent_state_zeros(&cfg);
    let b = initial_recurrent_state_zeros(&cfg);
    assert_eq!(a.shape(), vec![1, 4, 4, 8]);
    assert_eq!(b.shape(), a.shape());
    assert_eq!(a.dtype(), MlxDtype::Float32);
}

#[test]
fn fused_qkvz_ba_qmm_matches_split_two_qmm() {
    let seq = 2_i32;
    let hidden = 32_i32;
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 4,
        num_key_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        conv_kernel_dim: 4,
        q_scale: 0.25,
        k_scale: 0.5,
    };
    let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
    let x_data: Vec<f32> = (0..(seq * hidden))
        .map(|i| ((i as f32) - 31.0) * 0.03125)
        .collect();
    let qkvz_data: Vec<f32> = (0..(qkvz_out * hidden))
        .map(|i| ((i as f32) - 400.0) * 0.0005)
        .collect();
    let ba_data: Vec<f32> = (0..(ba_out * hidden))
        .map(|i| ((i as f32) - 80.0) * 0.001)
        .collect();
    let from_f32 = |data: &[f32], shape: &[i32]| {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    };
    let x = from_f32(&x_data, &[1, seq, hidden]);
    let qkvz_q = mlx_sys::quantize(
        &from_f32(&qkvz_data, &[qkvz_out, hidden]),
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Affine,
        None,
        None,
    );
    let ba_q = mlx_sys::quantize(
        &from_f32(&ba_data, &[ba_out, hidden]),
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Affine,
        None,
        None,
    );
    let qkvz_w = affine_quant_weight(
        qkvz_q[0].clone(),
        qkvz_q[1].clone(),
        qkvz_q[2].clone(),
        4,
        32,
    );
    let ba_w = affine_quant_weight(ba_q[0].clone(), ba_q[1].clone(), ba_q[2].clone(), 4, 32);
    assert!(
        fastpath::should_qwen_la_fused_qkvz_ba_qmm_for(
            true,
            seq,
            qkvz_w.matching_affine_quant(&ba_w)
        ),
        "matching 4-bit qkvz/ba must be eligible when the fuse flag is on"
    );

    let mut packed = LinearAttentionWeights {
        in_proj_qkv: None,
        in_proj_z: None,
        in_proj_a: None,
        in_proj_b: None,
        in_proj_qkvz: Some(qkvz_w.clone()),
        in_proj_ba: Some(ba_w.clone()),
        fused_qkvz_ba: None,
        prefill_q2_qkvz: None,
        prefill_q2_ba: None,
        conv1d_dense: zeros(&[1, 1, 1], MlxDtype::Float32, None),
        conv1d_bias: None,
        dt_bias: zeros(&[1], MlxDtype::Float32, None),
        a_log: zeros(&[1], MlxDtype::Float32, None),
        d: None,
        norm: zeros(&[1], MlxDtype::Float32, None),
        out_proj: qkvz_w.clone(),
    };
    packed.prepare_fused_qkvz_ba_prefill();
    packed.prepare_prefill_q2_projections();
    let q2 = packed
        .prefill_q2_qkvz
        .as_ref()
        .expect("4-bit qkvz must grow a 2-bit prefill overlay");
    let b2 = packed
        .prefill_q2_ba
        .as_ref()
        .expect("4-bit ba must grow a 2-bit prefill overlay");
    assert_eq!(q2.bits, crate::weights::PREFILL_LA_Q2_BITS);
    assert_eq!(b2.bits, crate::weights::PREFILL_LA_Q2_BITS);
    assert_eq!(q2.group_size, crate::weights::PREFILL_LA_Q2_GROUP_SIZE);
    assert!(
        fastpath::should_qwen_la_prefill_q2_for(true, 1024),
        "shipped q2 gate must accept the p2048 chunk length"
    );
    let (q2_qkv, q2_z, q2_a, q2_b) = linear_attention_inputs_packed_direct(&cfg, &x, q2, b2)
        .expect("2-bit packed LA inputs must engage");
    mlx_sys::eval(&[&q2_qkv, &q2_z, &q2_a, &q2_b]);
    assert_eq!(q2_qkv.shape()[1], seq);
    assert!(
        packed.fused_qkvz_ba.is_some(),
        "load-time matching-bit concat must populate fused_qkvz_ba"
    );
    let (fused_qkv, fused_z, fused_a, fused_b) = linear_attention_inputs_fused_qmm(
        TEST_COMPILE_IDENTITY,
        &cfg,
        &x,
        &qkvz_w,
        &ba_w,
        packed.fused_qkvz_ba.as_ref(),
    )
    .expect("matching 4-bit qkvz/ba should fuse");
    let split_qkvz = qw(&x, &qkvz_w);
    let split_ba = qw(&x, &ba_w);
    let (split_qkv, split_z, split_a, split_b) =
        split_packed_qkvz_ba_projection(&cfg, &split_qkvz, &split_ba, 1, seq);
    mlx_sys::eval(&[
        &fused_qkv, &fused_z, &fused_a, &fused_b, &split_qkv, &split_z, &split_a, &split_b,
    ]);
    for (got, want, name) in [
        (&fused_qkv, &split_qkv, "qkv"),
        (&fused_z, &split_z, "z"),
        (&fused_a, &split_a, "a"),
        (&fused_b, &split_b, "b"),
    ] {
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let g = got.data_f32();
        let w = want.data_f32();
        assert_eq!(g.len(), w.len(), "{name} len");
        for i in 0..g.len() {
            let err = (g[i] - w[i]).abs();
            assert!(
                err < 2.0e-5,
                "{name}[{i}] fused {} split {} err {err}",
                g[i],
                w[i]
            );
        }
    }
}

#[test]
fn qw_rms_norm_qmm_matches_rms_then_qw() {
    let hidden_data: Vec<f32> = (0..8 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let proj_data: Vec<f32> = (0..96 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0004)
        .collect();
    let x = MlxArray::from_raw_data(
        hidden_data.as_ptr() as *const u8,
        std::mem::size_of_val(hidden_data.as_slice()),
        &[1, 8, 64],
        MlxDtype::Float32,
    );
    let proj_w = MlxArray::from_raw_data(
        proj_data.as_ptr() as *const u8,
        std::mem::size_of_val(proj_data.as_slice()),
        &[96, 64],
        MlxDtype::Float32,
    );
    let dq = mlx_sys::quantize(
        &proj_w,
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Affine,
        None,
        None,
    );
    let proj = QuantizedWeight {
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
    let norm_data = vec![1.0f32; 64];
    let norm_w = MlxArray::from_raw_data(
        norm_data.as_ptr() as *const u8,
        std::mem::size_of_val(norm_data.as_slice()),
        &[64],
        MlxDtype::Float32,
    );
    let fused = qw_rms_norm_qmm(&x, &norm_w, 1e-6, &proj);
    let portable = qw(&rms_norm(&x, Some(&norm_w), 1e-6, None), &proj);
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    let g = fused.data_f32();
    let w = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..g.len() {
        max_abs = max_abs.max((g[i] - w[i]).abs());
    }
    assert!(
        max_abs < 3.0e-2,
        "LA norm+qmm fuse must match rms then qw, max_abs={max_abs}"
    );
    assert!(
        fastpath::should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_5", 1024),
        "shipped LA norm fuse gate must accept the p2048 chunk length"
    );
}

#[test]
fn qw_rms_norm_qmm_mxfp4_matches_rms_then_qw() {
    let hidden_data: Vec<f32> = (0..8 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let proj_data: Vec<f32> = (0..96 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0004)
        .collect();
    let x = MlxArray::from_raw_data(
        hidden_data.as_ptr() as *const u8,
        std::mem::size_of_val(hidden_data.as_slice()),
        &[1, 8, 64],
        MlxDtype::Float32,
    );
    let proj_w = MlxArray::from_raw_data(
        proj_data.as_ptr() as *const u8,
        std::mem::size_of_val(proj_data.as_slice()),
        &[96, 64],
        MlxDtype::Float32,
    );
    let dq = mlx_sys::quantize(
        &proj_w,
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    assert_eq!(dq.len(), 2, "mxfp4 quant returns [packed, scales]");
    let proj = QuantizedWeight {
        weight: dq[0].clone(),
        scales: Some(dq[1].clone()),
        biases: None,
        group_size: 32,
        bits: 4,
        mode: "mxfp4".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let norm_data = vec![1.0f32; 64];
    let norm_w = MlxArray::from_raw_data(
        norm_data.as_ptr() as *const u8,
        std::mem::size_of_val(norm_data.as_slice()),
        &[64],
        MlxDtype::Float32,
    );
    let fused = qw_rms_norm_qmm(&x, &norm_w, 1e-6, &proj);
    let portable = qw(&rms_norm(&x, Some(&norm_w), 1e-6, None), &proj);
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    let g = fused.data_f32();
    let w = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..g.len() {
        max_abs = max_abs.max((g[i] - w[i]).abs());
    }
    assert!(
        max_abs < 3.0e-2,
        "MXFP4 LA norm+qmm must use portable qw, max_abs={max_abs}"
    );
}

#[test]
fn qwen_prefill_contiguous_la_input_qw_matches_view() {
    let full: Vec<f32> = (0..16 * 64)
        .map(|i| ((i as f32) - 512.0) * 0.0009765625)
        .collect();
    let proj_data: Vec<f32> = (0..96 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0004)
        .collect();
    let wide = MlxArray::from_raw_data(
        full.as_ptr() as *const u8,
        std::mem::size_of_val(full.as_slice()),
        &[1, 16, 64],
        MlxDtype::Float32,
    );
    let view = slice(&wide, &[0, 4, 0], &[1, 12, 64], &[1, 1, 1], None);
    let proj_w = MlxArray::from_raw_data(
        proj_data.as_ptr() as *const u8,
        std::mem::size_of_val(proj_data.as_slice()),
        &[96, 64],
        MlxDtype::Float32,
    );
    let dq = mlx_sys::quantize(
        &proj_w,
        Some(32),
        Some(4),
        mlx_sys::MlxQuantizationMode::Affine,
        None,
        None,
    );
    let proj = QuantizedWeight {
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
    let packed = qw(&contiguous(&view, None), &proj);
    let portable = qw(&view, &proj);
    eval(&[&packed, &portable]);
    assert_eq!(packed.shape(), portable.shape());
    let g = packed.data_f32();
    let w = portable.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..g.len() {
        max_abs = max_abs.max((g[i] - w[i]).abs());
    }
    assert!(
        max_abs < 3.0e-2,
        "contiguous LA input qmm must match view qmm, max_abs={max_abs}"
    );
    assert!(
        fastpath::should_qwen_prefill_contiguous_la_input_for(true, "qwen3_5", 1024),
        "shipped LA input contiguous gate must accept the p2048 chunk length"
    );
}
