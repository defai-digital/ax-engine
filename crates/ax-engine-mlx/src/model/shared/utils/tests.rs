use super::*;
use mlx_sys::{
    MlxQuantizationMode, clear_cache, contiguous, eval, get_peak_memory, matmul, quantize,
    quantized_matmul, reset_peak_memory, transpose,
};

#[test]
fn qw_applies_dense_linear_bias() {
    // Dense Linear: y = x @ W^T + b  (mlx-lm nn.Linear with bias=True).
    let w_data = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
    let weight = array_f32(&w_data, &[2, 2]);
    let bias = array_f32(&[0.5, -0.25], &[2]);
    let qw = QuantizedWeight {
        weight,
        scales: None,
        biases: None,
        group_size: 1,
        bits: 32,
        mode: "affine".to_string(),
        linear_bias: Some(bias),
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let x = array_f32(&[1.0, 2.0], &[1, 1, 2]);
    let out = super::qw(&x, &qw);
    eval(&[&out]);
    let got = out.data_f32();
    // identity + bias → [1.5, 1.75]
    assert!((got[0] - 1.5).abs() < 1e-5, "got {}", got[0]);
    assert!((got[1] - 1.75).abs() < 1e-5, "got {}", got[1]);
}

fn verify_qmm_contract_fixture(
    input_dtype: MlxDtype,
    scale_dtype: MlxDtype,
    group_bias_dtype: MlxDtype,
    input_value: f32,
) -> (MlxArray, QuantizedWeight) {
    let (m, n, k) = (4, 16_384, 64);
    let input = astype(
        &array_f32(&vec![input_value; (m * k) as usize], &[1, m, k]),
        input_dtype,
        None,
    );
    let metadata = array_f32(&vec![1.0 / 64.0; (n * 2) as usize], &[n, 2]);
    let weight = QuantizedWeight {
        weight: mlx_sys::zeros(&[n, k / 8], MlxDtype::Uint32, None),
        scales: Some(astype(&metadata, scale_dtype, None)),
        biases: Some(astype(&metadata, group_bias_dtype, None)),
        group_size: 32,
        bits: 4,
        mode: "affine".to_owned(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    (input, weight)
}

#[test]
fn qw_verify_qmm_contract_applies_dense_bias_once() {
    use super::super::verify_qmm::{QwenMtpVerifyQmmGuard, try_qwen_mtp_verify_qmm};

    let _exact = fastpath::scoped_qwen_linear_mtp_exact(false);
    let _guard = QwenMtpVerifyQmmGuard::arm(true);
    for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
        let (input, mut weight) = verify_qmm_contract_fixture(dtype, dtype, dtype, 0.0);
        weight.linear_bias = Some(astype(
            &array_f32(&vec![0.5; 16_384], &[16_384]),
            dtype,
            None,
        ));
        let projection =
            try_qwen_mtp_verify_qmm(&input, &weight).expect("matching dtypes remain eligible");
        assert_eq!(projection.dtype(), dtype);
        let output = super::qw(&input, &weight);
        assert_eq!(output.dtype(), dtype);
        let projection = astype(&projection, MlxDtype::Float32, None);
        let output = astype(&output, MlxDtype::Float32, None);
        eval(&[&projection, &output]);
        assert!(projection.data_f32().iter().all(|value| *value == 0.0));
        let mismatch = output
            .data_f32()
            .iter()
            .copied()
            .find(|value| *value != 0.5);
        assert_eq!(mismatch, None, "{dtype:?}: zero input must return one bias");
    }
}

#[test]
fn qw_verify_qmm_contract_preserves_affine_dtype_promotion() {
    use super::super::verify_qmm::{QwenMtpVerifyQmmGuard, try_qwen_mtp_verify_qmm};

    let _exact = fastpath::scoped_qwen_linear_mtp_exact(false);
    let _guard = QwenMtpVerifyQmmGuard::arm(true);
    for (input_dtype, scale_dtype, group_bias_dtype) in [
        (MlxDtype::Bfloat16, MlxDtype::Float32, MlxDtype::Bfloat16),
        (MlxDtype::Bfloat16, MlxDtype::Bfloat16, MlxDtype::Float32),
        (MlxDtype::Float16, MlxDtype::Bfloat16, MlxDtype::Bfloat16),
        (MlxDtype::Bfloat16, MlxDtype::Float16, MlxDtype::Float16),
    ] {
        let (input, weight) =
            verify_qmm_contract_fixture(input_dtype, scale_dtype, group_bias_dtype, 1.0);
        let reference = quantized_matmul(
            &input,
            &weight.weight,
            weight.scales.as_ref().unwrap(),
            weight.biases.as_ref(),
            true,
            Some(weight.group_size),
            Some(weight.bits),
            None,
        );
        let output = super::qw(&input, &weight);
        assert_eq!(reference.dtype(), MlxDtype::Float32);
        assert_eq!(
            output.dtype(),
            reference.dtype(),
            "activation={input_dtype:?}, scale={scale_dtype:?}, group bias={group_bias_dtype:?}"
        );
        assert!(try_qwen_mtp_verify_qmm(&input, &weight).is_none());
        eval(&[&output, &reference]);
        assert_eq!(output.data_f32(), reference.data_f32());
    }
}

#[test]
fn project_unquantized_decode_matches_x_at_weight_t() {
    let hidden = 8;
    let vocab = 16;
    let mut w_data = vec![0.0f32; (vocab * hidden) as usize];
    for row in 0..vocab {
        for col in 0..hidden {
            w_data[(row * hidden + col) as usize] = (row + 1) as f32 * 0.1 + col as f32 * 0.01;
        }
    }
    let weight = array_f32(&w_data, &[vocab, hidden]);
    let x_data: Vec<f32> = (0..hidden).map(|i| (i + 1) as f32 * 0.25).collect();
    let x = array_f32(&x_data, &[1, 1, hidden]);
    let hits_before = super::unquantized_decode_projection_hits();
    let metal = super::project_unquantized_decode(&x, &weight).expect("decode GEMV eligible");
    let lm = QuantizedWeight {
        weight: weight.clone(),
        scales: None,
        biases: None,
        group_size: 1,
        bits: 32,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let shipped = super::project_lm_head(&x, &lm);
    let reference = matmul(&x, &transpose(&weight, &[1, 0], None), None);
    eval(&[&metal, &shipped, &reference]);
    let got = metal.data_f32();
    let shipped_got = shipped.data_f32();
    let want = reference.data_f32();
    assert_eq!(got.len(), want.len());
    for i in 0..got.len() {
        assert!(
            (got[i] - want[i]).abs() < 1e-4,
            "idx {i}: metal {} vs ref {}",
            got[i],
            want[i]
        );
        assert!(
            (shipped_got[i] - want[i]).abs() < 1e-4,
            "idx {i}: shipped {} vs ref {}",
            shipped_got[i],
            want[i]
        );
    }
    assert!(
        super::unquantized_decode_projection_hits() > hits_before,
        "shipped lm_head must take the no-copy GEMV"
    );
}

#[test]
fn project_unquantized_decode_skips_full_weight_transpose_buffer() {
    // Large enough that a materialized W.T is visible in peak memory.
    let hidden = 128i32;
    let vocab = 4096i32;
    let w_data: Vec<f32> = (0..vocab * hidden)
        .map(|i| (i % 17) as f32 * 0.01)
        .collect();
    let x_data: Vec<f32> = (0..hidden).map(|i| (i % 5) as f32 * 0.1).collect();
    let weight = array_f32(&w_data, &[vocab, hidden]);
    let x = array_f32(&x_data, &[1, hidden]);
    eval(&[&x, &weight]);

    clear_cache();
    reset_peak_memory();
    let metal = super::project_unquantized_decode(&x, &weight).expect("decode GEMV eligible");
    eval(&[&metal]);
    let peak_metal = get_peak_memory();

    clear_cache();
    reset_peak_memory();
    let transposed = contiguous(&transpose(&weight, &[1, 0], None), None);
    let reference = matmul(&x, &transposed, None);
    eval(&[&reference]);
    let peak_transpose = get_peak_memory();

    let weight_bytes = (vocab as usize) * (hidden as usize) * 4;
    assert!(
        peak_metal + weight_bytes / 2 < peak_transpose,
        "no-copy GEMV peak {peak_metal} should beat materialized transpose peak {peak_transpose} by ~half of {weight_bytes} weight bytes"
    );
}

#[test]
fn qw_pretransposed_lm_head_matches_lazy_transpose() {
    let hidden = 8i32;
    let vocab = 16i32;
    let w_data: Vec<f32> = (0..vocab * hidden)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let x_data: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.25).collect();
    let weight = array_f32(&w_data, &[vocab, hidden]);
    let x = array_f32(&x_data, &[1, 1, hidden]);
    let mut prepared = QuantizedWeight {
        weight: weight.clone(),
        scales: None,
        biases: None,
        group_size: 1,
        bits: 32,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    prepared.prepare_contiguous_decode_weight_t();
    let weight_t = prepared
        .decode_weight_t
        .as_ref()
        .expect("unquantized rank-2 lm_head must materialize W_t once");
    assert_eq!(weight_t.shape(), vec![hidden, vocab]);
    assert_eq!(
        prepared.weight.shape(),
        vec![vocab, hidden],
        "original [out, in] layout must remain a lazy view of W_t"
    );

    let lazy = QuantizedWeight {
        weight,
        scales: None,
        biases: None,
        group_size: 1,
        bits: 32,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let got = super::qw(&x, &prepared);
    let want = super::qw(&x, &lazy);
    eval(&[&got, &want]);
    let got = got.data_f32();
    let want = want.data_f32();
    assert_eq!(got.len(), want.len());
    for i in 0..got.len() {
        assert!(
            (got[i] - want[i]).abs() < 1e-4,
            "idx {i}: prepared {} vs lazy {}",
            got[i],
            want[i]
        );
    }
}

#[test]
fn prepare_contiguous_decode_weight_t_skips_quantized() {
    let weight = array_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let scales = array_f32(&[1.0, 1.0], &[2, 1]);
    let biases = array_f32(&[0.0, 0.0], &[2, 1]);
    let mut quantized = QuantizedWeight {
        weight,
        scales: Some(scales),
        biases: Some(biases),
        group_size: 2,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    quantized.prepare_contiguous_decode_weight_t();
    quantized.prepare_decode_q2_lm_head();
    quantized.prepare_lm_head_for_inference();
    assert!(
        quantized.decode_weight_t.is_none(),
        "quantized lm_head must not grow a dense W_t copy"
    );
    assert!(
        quantized.decode_q2_weight.is_none(),
        "already-quantized tensors must not grow a decode q2 cache"
    );
}

#[test]
fn prepared_target_head_preserves_singleton_and_verifier_logits() {
    let hidden = 64i32;
    let vocab = 32i32;
    let w_data: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i % 13) as f32) * 0.05 - 0.3)
        .collect();
    let x_data: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let weight = array_f32(&w_data, &[vocab, hidden]);
    let original_weight_t = transpose(&weight, &[1, 0], None);
    let mut head = QuantizedWeight::new(weight, None, None);
    // Even an explicitly prepared draft cache must not change target logits.
    head.prepare_decode_q2_lm_head();
    head.prepare_lm_head_for_inference();
    for rows in [1, 4] {
        let data: Vec<f32> = (0..rows).flat_map(|_| x_data.iter().copied()).collect();
        let x = array_f32(&data, &[1, rows, hidden]);
        let expected = matmul(&x, &original_weight_t, None);
        for actual in [super::qw(&x, &head), super::qw_direct_mlx(&x, &head)] {
            eval(&[&actual, &expected]);
            let error = actual
                .data_f32()
                .iter()
                .zip(expected.data_f32())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                error < 1e-4,
                "target head changed at rows={rows}: max_abs={error}"
            );
        }
    }
    assert!(head.decode_q2_weight.is_none());
    assert!(head.decode_q2_scales.is_none());
    assert!(head.decode_q2_biases.is_none());
}

#[test]
fn prepare_decode_q2_lm_head_is_decode_only() {
    let hidden = 64i32;
    let vocab = 32i32;
    let seq = 4i32;
    let w_data: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i % 13) as f32) * 0.05 - 0.3)
        .collect();
    let x_decode_data: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let x_prefill_data: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let weight = array_f32(&w_data, &[vocab, hidden]);
    let x_decode = array_f32(&x_decode_data, &[1, 1, hidden]);
    let x_prefill = array_f32(&x_prefill_data, &[1, seq, hidden]);
    let mut prepared = QuantizedWeight {
        weight: weight.clone(),
        scales: None,
        biases: None,
        group_size: 1,
        bits: 32,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    prepared.prepare_decode_q2_lm_head();
    prepared.prepare_contiguous_decode_weight_t();
    assert!(
        prepared.decode_q2_weight.is_some(),
        "unquantized rank-2 hidden%64==0 must build a decode quant cache"
    );
    assert!(
        prepared.decode_weight_t.is_some(),
        "prefill keeps a contiguous BF16 W_t"
    );
    let got_decode = super::qw(&x_decode, &prepared);
    let got_prefill = super::qw(&x_prefill, &prepared);
    let want_prefill = matmul(&x_prefill, prepared.decode_weight_t.as_ref().unwrap(), None);
    eval(&[&got_decode, &got_prefill, &want_prefill]);
    let got_decode = got_decode.data_f32();
    assert!(
        got_decode.iter().all(|v| v.is_finite()),
        "2-bit decode lm_head must produce finite logits"
    );
    let got_prefill = got_prefill.data_f32();
    let want_prefill = want_prefill.data_f32();
    assert_eq!(got_prefill.len(), want_prefill.len());
    let mut max_abs = 0.0f32;
    for i in 0..got_prefill.len() {
        max_abs = max_abs.max((got_prefill[i] - want_prefill[i]).abs());
    }
    assert!(
        max_abs < 1e-4,
        "prefill must use BF16 W_t, not the 2-bit decode cache, max_abs={max_abs}"
    );
}

#[test]
fn qwen_prefill_maybe_skip_bf16_astype_skips_when_already_bf16() {
    let data = [1.0f32, -0.5, 0.25, 2.0];
    let x = array_f32(&data, &[1, 2, 2]);
    let bf = astype(&x, MlxDtype::Bfloat16, None);
    eval(&[&bf]);
    let skipped = super::qwen_prefill_maybe_skip_bf16_astype(&bf, "qwen3_5", 1024);
    let forced = astype(&bf, MlxDtype::Bfloat16, None);
    eval(&[&skipped, &forced]);
    assert_eq!(skipped.dtype(), MlxDtype::Bfloat16);
    assert_eq!(skipped.shape(), forced.shape());
    let cast = super::qwen_prefill_maybe_skip_bf16_astype(&x, "qwen3_5", 1024);
    eval(&[&cast]);
    assert_eq!(cast.dtype(), MlxDtype::Bfloat16);
    assert!(
        fastpath::should_qwen_prefill_skip_bf16_astype_for(true, "qwen3_5", 1024),
        "shipped skip-astype gate must accept prefill seq"
    );
    let gemma_skipped = super::qwen_prefill_maybe_skip_bf16_astype(&bf, "gemma4", 128);
    eval(&[&gemma_skipped]);
    assert_eq!(gemma_skipped.dtype(), MlxDtype::Bfloat16);
    assert_eq!(gemma_skipped.shape(), forced.shape());
    assert!(
        fastpath::should_gemma4_prefill_bf16_embed_for(true, "gemma4", 128),
        "shipped Gemma 4 bf16 embed must skip unused astype at p128"
    );
}

#[test]
fn qwen_prefill_maybe_async_embed_submits_at_min_seq() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
    let hidden = array_f32(&data, &[1, 4, 8]);
    super::qwen_prefill_maybe_async_embed_for(&hidden, true, "qwen3_5", 1024);
    eval(&[&hidden]);
    assert_eq!(hidden.shape(), vec![1, 4, 8]);
    assert!(
        hidden.data_f32().iter().all(|v| v.is_finite()),
        "async embed must leave a finite materialized tensor"
    );
    assert!(
        fastpath::should_qwen_prefill_async_embed_for(true, "qwen3_5", 1024),
        "shipped async-embed gate must accept the p2048 chunk length"
    );
    super::qwen_prefill_maybe_async_embed_for(&hidden, false, "qwen3_5", 1024);
    super::qwen_prefill_maybe_async_embed_for(&hidden, true, "qwen3_5", 512);
    super::qwen_prefill_maybe_async_embed_for(&hidden, true, "gemma4", 1024);
}

#[test]
fn qwen_prefill_dequant_dense_matches_quantized_matmul() {
    let input_dim = 64i32;
    let output_dim = 128i32;
    let seq = 1024i32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 127) as f32 - 63.0) * 0.015625)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let qw = QuantizedWeight {
        weight: quantized[0].clone(),
        scales: Some(quantized[1].clone()),
        biases: Some(quantized[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let input_data: Vec<f32> = (0..seq * input_dim)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.03125)
        .collect();
    let x = array_f32(&input_data, &[1, seq, input_dim]);
    super::set_qwen_prefill_dequant_dense_family(false);
    let via_qmm = super::qw(&x, &qw);
    let weight_t =
        super::cached_prefill_dequant_weight_t(&qw).expect("affine 4-bit weight must dequantize");
    let via_dense = matmul(&x, &weight_t, None);
    eval(&[&via_qmm, &via_dense]);
    let qmm = via_qmm.data_f32();
    let dense = via_dense.data_f32();
    assert_eq!(qmm.len(), dense.len());
    let mut max_abs = 0.0f32;
    for i in 0..qmm.len() {
        max_abs = max_abs.max((qmm[i] - dense[i]).abs());
    }
    assert!(
        max_abs < 2e-2,
        "dequant+dense must match steel qmm, max_abs={max_abs}"
    );
    let x_decode = array_f32(&input_data[..input_dim as usize], &[1, 1, input_dim]);
    super::set_qwen_prefill_dequant_dense_family(true);
    let decode_out = super::qw(&x_decode, &qw);
    super::set_qwen_prefill_dequant_dense_family(false);
    let decode_qmm = super::qw(&x_decode, &qw);
    eval(&[&decode_out, &decode_qmm]);
    assert_eq!(
        decode_out.data_f32(),
        decode_qmm.data_f32(),
        "seq=1 must stay on steel qmm"
    );
}

#[test]
fn qwen_prefill_maybe_flat_qmm_matches_3d_quantized_matmul() {
    let input_dim = 32i32;
    let output_dim = 64i32;
    let seq = 1024i32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 63) as f32 - 31.0) * 0.015625)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let input_data: Vec<f32> = (0..seq * input_dim)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let x = array_f32(&input_data, &[1, seq, input_dim]);
    let qmm_3d = mlx_sys::quantized_matmul_with_mode(
        &x,
        &quantized[0],
        &quantized[1],
        Some(&quantized[2]),
        true,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
    );
    let qmm_flat = super::qwen_prefill_maybe_flat_qmm_for(&x, true, |flat| {
        mlx_sys::quantized_matmul_with_mode(
            flat,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            true,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
        )
    });
    eval(&[&qmm_3d, &qmm_flat]);
    assert_eq!(qmm_flat.shape(), vec![1, seq, output_dim]);
    let a = qmm_3d.data_f32();
    let b = qmm_flat.data_f32();
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1e-4,
        "flat 2-D qmm must match 3-D steel qmm, max_abs={max_abs}"
    );
    assert!(
        fastpath::should_qwen_prefill_flat_qmm_for(true, 1024, 3),
        "shipped flat-qmm gate must accept the p2048 chunk length"
    );
    let skipped = super::qwen_prefill_maybe_flat_qmm_for(&x, false, |inner| inner.clone());
    eval(&[&skipped]);
    assert_eq!(skipped.shape(), x.shape());
}

#[test]
fn qwen_prefill_tile_qmm_matches_oneshot() {
    let input_dim = 32i32;
    let output_dim = 64i32;
    let seq = 1024i32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 63) as f32 - 31.0) * 0.015625)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let input_data: Vec<f32> = (0..seq * input_dim)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let x = array_f32(&input_data, &[1, seq, input_dim]);
    let oneshot = mlx_sys::quantized_matmul_with_mode(
        &x,
        &quantized[0],
        &quantized[1],
        Some(&quantized[2]),
        true,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
    );
    let tiled = super::qwen_prefill_maybe_tile_qmm_for(&x, true, "qwen3_5", |chunk| {
        mlx_sys::quantized_matmul_with_mode(
            chunk,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            true,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
        )
    });
    eval(&[&oneshot, &tiled]);
    assert_eq!(tiled.shape(), vec![1, seq, output_dim]);
    let a = oneshot.data_f32();
    let b = tiled.data_f32();
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1e-4,
        "tiled qmm must match oneshot steel qmm, max_abs={max_abs}"
    );
    assert!(
        fastpath::should_qwen_prefill_tile_qmm_for(true, "qwen3_5", 1024),
        "shipped tile-qmm gate must accept the p2048 chunk length"
    );
    let skipped =
        super::qwen_prefill_maybe_tile_qmm_for(&x, false, "qwen3_5", |inner| inner.clone());
    eval(&[&skipped]);
    assert_eq!(skipped.shape(), x.shape());
}

#[test]
fn prepare_contiguous_decode_weight_t_keeps_one_physical_buffer() {
    // Large enough that a second full copy would show in peak memory.
    let hidden = 128i32;
    let vocab = 4096i32;
    let w_data: Vec<f32> = (0..vocab * hidden)
        .map(|i| (i % 17) as f32 * 0.01)
        .collect();
    let weight = array_f32(&w_data, &[vocab, hidden]);
    eval(&[&weight]);
    let mut prepared = QuantizedWeight {
        weight,
        scales: None,
        biases: None,
        group_size: 1,
        bits: 32,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    prepared.prepare_contiguous_decode_weight_t();
    let weight_t = prepared.decode_weight_t.as_ref().expect("prepared W_t");
    clear_cache();
    reset_peak_memory();
    eval(&[&prepared.weight, weight_t]);
    let peak = get_peak_memory();
    let one_copy = (vocab as usize) * (hidden as usize) * 4;
    assert!(
        peak < one_copy.saturating_mul(2),
        "eval of W view + W_t must not materialize two full copies (peak {peak}, one copy {one_copy})"
    );
    assert_eq!(prepared.weight.shape(), vec![vocab, hidden]);
    assert_eq!(weight_t.shape(), vec![hidden, vocab]);
}

#[test]
fn project_unquantized_decode_matches_bf16_x_at_weight_t() {
    let hidden = 8;
    let vocab = 16;
    let w_data: Vec<f32> = (0..vocab * hidden)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let x_data: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.25).collect();
    let weight = astype(
        &array_f32(&w_data, &[vocab, hidden]),
        MlxDtype::Bfloat16,
        None,
    );
    let x = astype(
        &array_f32(&x_data, &[1, 1, hidden]),
        MlxDtype::Bfloat16,
        None,
    );
    let metal = super::project_unquantized_decode(&x, &weight).expect("bf16 GEMV eligible");
    let reference = matmul(&x, &transpose(&weight, &[1, 0], None), None);
    eval(&[&metal, &reference]);
    let got = astype(&metal, MlxDtype::Float32, None);
    let want = astype(&reference, MlxDtype::Float32, None);
    eval(&[&got, &want]);
    for (g, w) in got.data_f32().iter().zip(want.data_f32().iter()) {
        assert!((g - w).abs() < 2.0e-2, "bf16 gemv {g} vs ref {w}");
    }
}

#[test]
fn project_unquantized_decode_rejects_multi_token() {
    let weight = array_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let x = array_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    assert!(super::project_unquantized_decode(&x, &weight).is_none());
}

#[test]
fn expert_linear_bias_matches_mlx_lm_index_add() {
    // bias: [E=3, out=2], indices: [1, 2] → bias[[1,2]] = [[2,3],[4,5]]
    let bias_data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let bias = MlxArray::from_raw_data(
        bias_data.as_ptr() as *const u8,
        bias_data.len() * 4,
        &[3, 2],
        MlxDtype::Float32,
    );
    let y_data = [10.0f32, 20.0, 30.0, 40.0];
    let y = MlxArray::from_raw_data(
        y_data.as_ptr() as *const u8,
        y_data.len() * 4,
        &[1, 2, 2],
        MlxDtype::Float32,
    );
    let idx_data = [1u32, 2];
    let indices = MlxArray::from_raw_data(
        idx_data.as_ptr() as *const u8,
        idx_data.len() * 4,
        &[1, 2],
        MlxDtype::Uint32,
    );
    let out = apply_expert_linear_bias(&y, &bias, &indices);
    eval(&[&out]);
    let got = out.data_f32();
    // y[0] + bias[1] = [10,20]+[2,3] = [12,23]
    // y[1] + bias[2] = [30,40]+[4,5] = [34,45]
    assert!((got[0] - 12.0).abs() < 1e-5);
    assert!((got[1] - 23.0).abs() < 1e-5);
    assert!((got[2] - 34.0).abs() < 1e-5);
    assert!((got[3] - 45.0).abs() < 1e-5);
}

fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Float32,
    )
}

#[test]
fn dense_wide_gemv_rows_are_leading_invariant_bit_exact() {
    // The verify-window projection relies on this: row s of a Leading=S
    // launch must be bit-identical to a Leading=1 launch of that row, so
    // batched dense-head verify matches the exact profile's singleton
    // decode arithmetic.
    let input_dim = 96_i32;
    let out_dim = 40_i32;
    let weight_t_data: Vec<f32> = (0..(input_dim * out_dim) as usize)
        .map(|i| (((i % 733) as f32) - 350.0) * 0.002)
        .collect();
    let weight_t = astype(
        &array_f32(&weight_t_data, &[input_dim, out_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    for leading in [2_i32, 3, 4] {
        let x_data: Vec<f32> = (0..(leading * input_dim) as usize)
            .map(|i| (((i % 259) as f32) - 120.0) * 0.01)
            .collect();
        let x = astype(
            &array_f32(&x_data, &[1, leading, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let batched =
            dense_wide_gemv_weight_t(&x, &weight_t).expect("batched dense wide GEMV must engage");
        let mut rows = Vec::new();
        for t in 0..leading {
            let row = contiguous(
                &slice(&x, &[0, t, 0], &[1, t + 1, input_dim], &[1, 1, 1], None),
                None,
            );
            rows.push(
                dense_wide_gemv_weight_t(&row, &weight_t)
                    .expect("singleton dense wide GEMV must engage"),
            );
        }
        let refs: Vec<&MlxArray> = rows.iter().collect();
        let singles = concatenate(&refs, 1, None);
        let a = astype(&batched, MlxDtype::Float32, None);
        let b = astype(&singles, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let av = a.data_f32();
        let bv = b.data_f32();
        assert_eq!(av.len(), bv.len());
        for i in 0..av.len() {
            assert_eq!(
                av[i].to_bits(),
                bv[i].to_bits(),
                "Leading={leading} row projection diverged at {i}"
            );
        }
    }
}

#[test]
fn dense_wide_gemv_matches_matmul_f32_and_bf16() {
    let input_dim = 96_i32;
    let out_dim = 40_i32;
    let weight_t_data: Vec<f32> = (0..(input_dim * out_dim) as usize)
        .map(|i| (((i % 731) as f32) - 350.0) * 0.002)
        .collect();
    let weight_t = array_f32(&weight_t_data, &[input_dim, out_dim]);
    for leading in [1_i32, 2, 3, 4, 8] {
        let x_data: Vec<f32> = (0..(leading * input_dim) as usize)
            .map(|i| (((i % 257) as f32) - 120.0) * 0.01)
            .collect();
        let x = array_f32(&x_data, &[1, leading, input_dim]);
        let wide = dense_wide_gemv_weight_t(&x, &weight_t)
            .expect("f32 dense wide GEMV must be eligible for 1..=8 rows");
        let reference = matmul(&x, &weight_t, None);
        eval(&[&wide, &reference]);
        assert_eq!(wide.shape(), vec![1, leading, out_dim]);
        let a = wide.data_f32();
        let b = reference.data_f32();
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() < 1.0e-4,
                "f32 wide GEMV row set {leading} diverged at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }

        let x_bf16 = astype(&x, MlxDtype::Bfloat16, None);
        let weight_t_bf16 = astype(&weight_t, MlxDtype::Bfloat16, None);
        let wide_bf16 = dense_wide_gemv_weight_t(&x_bf16, &weight_t_bf16)
            .expect("bf16 dense wide GEMV must be eligible");
        let reference_bf16 = matmul(&x_bf16, &weight_t_bf16, None);
        eval(&[&wide_bf16, &reference_bf16]);
        let a = astype(&wide_bf16, MlxDtype::Float32, None);
        let b = astype(&reference_bf16, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let a = a.data_f32();
        let b = b.data_f32();
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() < 5.0e-2,
                "bf16 wide GEMV row set {leading} diverged at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }
}

#[test]
#[ignore = "micro-bench: run with --ignored --nocapture for kernel-vs-MLX timings"]
fn bench_dense_wide_gemv_vs_matmul() {
    // lm_head-ish verify shape: hidden 4096 -> a 32k vocab slice.
    let input_dim = 4096_i32;
    let out_dim = 32_768_i32;
    let weight_t_data: Vec<f32> = (0..(input_dim as usize) * (out_dim as usize))
        .map(|i| (((i % 977) as f32) - 480.0) * 0.001)
        .collect();
    let weight_t = astype(
        &array_f32(&weight_t_data, &[input_dim, out_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    for leading in [2_i32, 4] {
        let x_data: Vec<f32> = (0..(leading * input_dim) as usize)
            .map(|i| (((i % 509) as f32) - 250.0) * 0.004)
            .collect();
        let x = astype(
            &array_f32(&x_data, &[1, leading, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let iters = 100;
        let warm = 10;
        for _ in 0..warm {
            let o = dense_wide_gemv_weight_t(&x, &weight_t).unwrap();
            eval(&[&o]);
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let o = dense_wide_gemv_weight_t(&x, &weight_t).unwrap();
            eval(&[&o]);
        }
        let kernel_us = start.elapsed().as_micros() as f64 / iters as f64;
        for _ in 0..warm {
            let o = matmul(&x, &weight_t, None);
            eval(&[&o]);
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let o = matmul(&x, &weight_t, None);
            eval(&[&o]);
        }
        let mlx_us = start.elapsed().as_micros() as f64 / iters as f64;
        println!(
            "dense wide GEMV S={leading} {input_dim}x{out_dim}: kernel {kernel_us:.1}us vs matmul {mlx_us:.1}us ({:.3}x)",
            mlx_us / kernel_us
        );
    }
}

fn quantized_expert_stack(
    num_experts: i32,
    out_dim: i32,
    input_dim: i32,
    group_size: i32,
) -> (QuantizedWeight, Vec<MlxArray>) {
    let data: Vec<f32> = (0..(num_experts * out_dim * input_dim) as usize)
        .map(|i| (((i % 883) as f32) - 440.0) * 0.0015)
        .collect();
    let dense = array_f32(&data, &[num_experts, out_dim, input_dim]);
    let q = mlx_sys::quantize(
        &dense,
        Some(group_size),
        Some(4),
        mlx_sys::MlxQuantizationMode::Affine,
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
            bits: 4,
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

fn sorted_gather_reference(
    x_sorted: &MlxArray,
    exps: &QuantizedWeight,
    ids: &[u32],
    out_dim: i32,
) -> MlxArray {
    let rows = ids.len() as i32;
    let input_dim = *x_sorted.shape().last().unwrap();
    let x3 = reshape(x_sorted, &[rows, 1, input_dim], None);
    let idx = MlxArray::from_raw_data(
        ids.as_ptr() as *const u8,
        std::mem::size_of_val(ids),
        &[rows],
        MlxDtype::Uint32,
    );
    let y = qw_gather(&x3, exps, &idx, true);
    reshape(&y, &[rows, out_dim], None)
}

#[test]
fn gather_qmv_wide_sorted_matches_gather_qmm() {
    let (num_experts, out_dim, input_dim, group_size) = (8, 48, 128, 64);
    let (exps, _q) = quantized_expert_stack(num_experts, out_dim, input_dim, group_size);
    // 20 sorted selections over 6 experts with segment sizes 1..=9
    // (the 9-run splits into 8+1 host-side).
    let ids: Vec<u32> = [
        0u32, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 5, 7, 7, 7, 7,
    ]
    .to_vec();
    let rows = ids.len() as i32;
    let x_data: Vec<f32> = (0..(rows * input_dim) as usize)
        .map(|i| (((i % 311) as f32) - 150.0) * 0.006)
        .collect();
    let x = array_f32(&x_data, &[rows, input_dim]);
    let wide =
        gather_qmv_wide_sorted(&x, &exps, &ids).expect("sorted 4-bit gather qmv wide must engage");
    let reference = sorted_gather_reference(&x, &exps, &ids, out_dim);
    eval(&[&wide, &reference]);
    assert_eq!(wide.shape(), vec![rows, out_dim]);
    let a = wide.data_f32();
    let b = reference.data_f32();
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < 1.0e-4,
            "gather qmv wide diverged at {i}: {} vs {}",
            a[i],
            b[i]
        );
    }
}

#[test]
fn gather_wide_segments_caps_runs_at_eight() {
    let ids = [3u32; 19];
    let (expert, start, count) = gather_wide_segments(&ids);
    assert_eq!(expert, vec![3, 3, 3]);
    assert_eq!(start, vec![0, 8, 16]);
    assert_eq!(count, vec![8, 8, 3]);
}

#[test]
#[ignore = "micro-bench: run with --ignored --nocapture for kernel-vs-MLX timings"]
fn bench_gather_qmv_wide_vs_gather_qmm() {
    // MoE verify-ish shape: 32 stacked selections (S=4 x top-8) over a
    // 64-expert 4-bit stack, 2048 -> 768 per expert.
    let (num_experts, out_dim, input_dim, group_size) = (64, 768, 2048, 64);
    let (exps, _q) = quantized_expert_stack(num_experts, out_dim, input_dim, group_size);
    let mut ids: Vec<u32> = Vec::new();
    for e in [3u32, 9, 17, 22, 31, 40, 52, 61] {
        for _ in 0..4 {
            ids.push(e);
        }
    }
    let rows = ids.len() as i32;
    let x_data: Vec<f32> = (0..(rows * input_dim) as usize)
        .map(|i| (((i % 509) as f32) - 250.0) * 0.004)
        .collect();
    let x = astype(
        &array_f32(&x_data, &[rows, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let iters = 200;
    let warm = 20;
    for _ in 0..warm {
        let o = gather_qmv_wide_sorted(&x, &exps, &ids).unwrap();
        eval(&[&o]);
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let o = gather_qmv_wide_sorted(&x, &exps, &ids).unwrap();
        eval(&[&o]);
    }
    let kernel_us = start.elapsed().as_micros() as f64 / iters as f64;
    let x3 = reshape(&x, &[rows, 1, input_dim], None);
    let idx = MlxArray::from_raw_data(
        ids.as_ptr() as *const u8,
        std::mem::size_of_val(ids.as_slice()),
        &[rows],
        MlxDtype::Uint32,
    );
    for sorted in [true, false] {
        for _ in 0..warm {
            let o = qw_gather(&x3, &exps, &idx, sorted);
            eval(&[&o]);
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let o = qw_gather(&x3, &exps, &idx, sorted);
            eval(&[&o]);
        }
        let mlx_us = start.elapsed().as_micros() as f64 / iters as f64;
        println!(
            "gather qmv wide rows={rows} E={num_experts} {input_dim}->{out_dim}: kernel {kernel_us:.1}us vs gather_qmm(sorted={sorted}) {mlx_us:.1}us ({:.3}x)",
            mlx_us / kernel_us
        );
    }
}

#[test]
fn dense_wide_gemv_bf16_vector_tiles_match_scalar_bit_exact() {
    // The 8- and 4-column bf16 tiles must reproduce the scalar form bit
    // for bit (same sequential-k f32 fma per column), for every row count
    // the verify window can present and for column counts that select
    // each tile. Column counts that divide by neither fall back to the
    // scalar form through the public entry.
    let input_dim = 96_i32;
    for (out_dim, expected_cols) in [(40_i32, 8_i32), (44, 4), (42, 1), (264, 8)] {
        for leading in [1, 4, 6] {
            assert_eq!(
                dense_wide_gemv_cols(MlxDtype::Bfloat16, out_dim, leading),
                expected_cols,
                "tile selection for out_dim {out_dim} at {leading} rows"
            );
        }
        // Past the register budget the 8-column tile is never selected.
        for leading in [7, 8] {
            assert_eq!(
                dense_wide_gemv_cols(MlxDtype::Bfloat16, out_dim, leading),
                expected_cols.min(4),
                "tile selection for out_dim {out_dim} at {leading} rows"
            );
        }
        assert_eq!(dense_wide_gemv_cols(MlxDtype::Float32, out_dim, 1), 1);
        let weight_t_data: Vec<f32> = (0..(input_dim * out_dim) as usize)
            .map(|i| (((i % 739) as f32) - 360.0) * 0.002)
            .collect();
        let weight_t = astype(
            &array_f32(&weight_t_data, &[input_dim, out_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        for leading in 1_i32..=8 {
            let x_data: Vec<f32> = (0..(leading * input_dim) as usize)
                .map(|i| (((i % 263) as f32) - 130.0) * 0.01)
                .collect();
            let x = astype(
                &array_f32(&x_data, &[1, leading, input_dim]),
                MlxDtype::Bfloat16,
                None,
            );
            let scalar = dense_wide_gemv_weight_t_with_cols(&x, &weight_t, 1)
                .expect("scalar dense wide GEMV must engage");
            let tiled =
                dense_wide_gemv_weight_t(&x, &weight_t).expect("dense wide GEMV must engage");
            let a = astype(&scalar, MlxDtype::Float32, None);
            let b = astype(&tiled, MlxDtype::Float32, None);
            eval(&[&a, &b]);
            assert_eq!(tiled.shape(), vec![1, leading, out_dim]);
            let av = a.data_f32();
            let bv = b.data_f32();
            assert_eq!(av.len(), bv.len());
            for i in 0..av.len() {
                assert_eq!(
                    av[i].to_bits(),
                    bv[i].to_bits(),
                    "out_dim {out_dim} cols {expected_cols} Leading {leading} diverged at {i}"
                );
            }
            if expected_cols == 8 {
                // Both vector tiles are valid for a multiple of 8; pin
                // each explicitly, whichever the row count selects.
                for cols in [4, 8] {
                    let tile = dense_wide_gemv_weight_t_with_cols(&x, &weight_t, cols)
                        .expect("vector tile must engage on a multiple of 8");
                    let c = astype(&tile, MlxDtype::Float32, None);
                    eval(&[&c]);
                    let cv = c.data_f32();
                    for i in 0..av.len() {
                        assert_eq!(
                            av[i].to_bits(),
                            cv[i].to_bits(),
                            "{cols}-col tile Leading {leading} diverged at {i}"
                        );
                    }
                }
            }
        }
        // f16 / f32 activations share the bf16-weight vector path (only
        // the weight word is reinterpreted); pin them to the scalar form
        // as well.
        for x_dtype in [MlxDtype::Float16, MlxDtype::Float32] {
            let x_data: Vec<f32> = (0..(4 * input_dim) as usize)
                .map(|i| (((i % 271) as f32) - 135.0) * 0.01)
                .collect();
            let x = astype(&array_f32(&x_data, &[1, 4, input_dim]), x_dtype, None);
            let scalar = dense_wide_gemv_weight_t_with_cols(&x, &weight_t, 1)
                .expect("scalar dense wide GEMV must engage");
            let tiled =
                dense_wide_gemv_weight_t(&x, &weight_t).expect("dense wide GEMV must engage");
            assert_eq!(tiled.dtype(), x_dtype);
            let a = astype(&scalar, MlxDtype::Float32, None);
            let b = astype(&tiled, MlxDtype::Float32, None);
            eval(&[&a, &b]);
            let av = a.data_f32();
            let bv = b.data_f32();
            for i in 0..av.len() {
                assert_eq!(
                    av[i].to_bits(),
                    bv[i].to_bits(),
                    "out_dim {out_dim} x dtype {x_dtype:?} diverged at {i}"
                );
            }
        }
        // A row-offset view of a larger buffer is row-contiguous and
        // reaches the kernel uncopied; its offset is whole words, so the
        // vector tiles must still match the scalar form bit for bit.
        if expected_cols != 1 {
            let padded_data: Vec<f32> = (0..((input_dim + 3) * out_dim) as usize)
                .map(|i| (((i % 743) as f32) - 370.0) * 0.002)
                .collect();
            let padded = astype(
                &array_f32(&padded_data, &[input_dim + 3, out_dim]),
                MlxDtype::Bfloat16,
                None,
            );
            let view = slice(&padded, &[3, 0], &[input_dim + 3, out_dim], &[1, 1], None);
            let x_data: Vec<f32> = (0..(2 * input_dim) as usize)
                .map(|i| (((i % 251) as f32) - 125.0) * 0.01)
                .collect();
            let x = astype(
                &array_f32(&x_data, &[1, 2, input_dim]),
                MlxDtype::Bfloat16,
                None,
            );
            let scalar = dense_wide_gemv_weight_t_with_cols(&x, &view, 1)
                .expect("scalar dense wide GEMV must engage on a view");
            let tiled =
                dense_wide_gemv_weight_t(&x, &view).expect("dense wide GEMV must engage on a view");
            let a = astype(&scalar, MlxDtype::Float32, None);
            let b = astype(&tiled, MlxDtype::Float32, None);
            eval(&[&a, &b]);
            let av = a.data_f32();
            let bv = b.data_f32();
            for i in 0..av.len() {
                assert_eq!(
                    av[i].to_bits(),
                    bv[i].to_bits(),
                    "row-offset view out_dim {out_dim} diverged at {i}"
                );
            }
        }
        // A tile the weight cannot admit is refused rather than misread.
        let x = astype(
            &array_f32(&vec![0.5; input_dim as usize], &[1, 1, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        if out_dim % 8 != 0 {
            assert!(dense_wide_gemv_weight_t_with_cols(&x, &weight_t, 8).is_none());
        }
        if out_dim % 4 != 0 {
            assert!(dense_wide_gemv_weight_t_with_cols(&x, &weight_t, 4).is_none());
        }
    }
}

#[test]
fn dense_wide_gemv_rejects_out_of_window_shapes() {
    let weight_t = array_f32(&vec![0.0; 96 * 40], &[96, 40]);
    let single = array_f32(&vec![0.0; 96], &[1, 1, 96]);
    let nine = array_f32(&vec![0.0; 9 * 96], &[1, 9, 96]);
    let mismatched = array_f32(&vec![0.0; 2 * 64], &[1, 2, 64]);
    assert!(
        dense_wide_gemv_weight_t(&single, &weight_t).is_some(),
        "S=1 must be hosted so exact MTP-off singles share the verify arithmetic"
    );
    assert!(dense_wide_gemv_weight_t(&nine, &weight_t).is_none());
    assert!(dense_wide_gemv_weight_t(&mismatched, &weight_t).is_none());
}

#[test]
fn invariant_dense_projection_routes_prepared_head_through_weight_t() {
    let out_dim = 48_i32;
    let input_dim = 96_i32;
    let weight_data: Vec<f32> = (0..(out_dim * input_dim) as usize)
        .map(|i| (((i % 641) as f32) - 320.0) * 0.002)
        .collect();
    let mut head = QuantizedWeight::new(array_f32(&weight_data, &[out_dim, input_dim]), None, None);
    head.prepare_contiguous_decode_weight_t();
    assert!(head.decode_weight_t.is_some());
    for leading in [1_i32, 2, 4] {
        let x_data: Vec<f32> = (0..(leading * input_dim) as usize)
            .map(|i| (((i % 253) as f32) - 120.0) * 0.008)
            .collect();
        let x = array_f32(&x_data, &[1, leading, input_dim]);
        let routed = invariant_projection_metal_impl(&x, &head)
            .expect("prepared dense head must stay hosted under the exact profile");
        let reference = matmul(&x, head.decode_weight_t.as_ref().unwrap(), None);
        eval(&[&routed, &reference]);
        assert_eq!(routed.shape(), vec![1, leading, out_dim]);
        let a = routed.data_f32();
        let b = reference.data_f32();
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() < 1.0e-4,
                "prepared-head routing diverged at S={leading}, {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }
}

#[test]
fn layer_scalar_fused_add_is_decode_only() {
    assert!(layer_scalar_fused_add_shape_supported(&[1, 1, 4]));
    assert!(layer_scalar_fused_add_shape_supported(&[1, 1, 35, 4]));
    assert!(!layer_scalar_fused_add_shape_supported(&[1, 2, 4]));
    assert!(!layer_scalar_fused_add_shape_supported(&[1, 2048, 4]));
}

#[test]
fn add_then_multiply_scalar_metal_matches_unfused_float32() {
    let a = array_f32(&[0.5, -1.0, 2.0, 3.5, -4.0, 8.0], &[2, 3]);
    let b = array_f32(&[1.0, 4.0, -2.0, 0.5, 3.0, -8.0], &[2, 3]);
    let scalar = array_f32(&[0.25], &[1]);

    let direct = add_then_multiply_scalar_metal_impl(&a, &b, &scalar)
        .expect("scalar fused add should support float32 inputs");
    let reference = multiply(&add(&a, &b, None), &scalar, None);
    eval(&[&direct, &reference]);

    assert_eq!(direct.shape(), vec![2, 3]);
    assert_eq!(direct.data_f32(), reference.data_f32());
}

#[test]
fn row_exact_projection_matches_independent_quantized_rows() {
    let input_dim = 64;
    let output_dim = 64;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 127) as f32 - 63.0) * 0.015625)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(64),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let weight = QuantizedWeight {
        weight: quantized[0].clone(),
        scales: Some(quantized[1].clone()),
        biases: Some(quantized[2].clone()),
        group_size: 64,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let input_data: Vec<f32> = (0..2 * input_dim)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.03125)
        .collect();
    let input = array_f32(&input_data, &[2, 1, input_dim]);
    let batched = qw_with_policy(&input, &weight, ProjectionBatchPolicy::RowExact);

    for row in 0..2i32 {
        let row_start = (row as usize) * (input_dim as usize);
        let single_input = array_f32(
            &input_data[row_start..row_start + input_dim as usize],
            &[1, 1, input_dim],
        );
        let expected = qw(&single_input, &weight);
        let actual = contiguous(
            &slice(
                &batched,
                &[row, 0, 0],
                &[row + 1, 1, output_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        eval(&[&actual, &expected]);
        assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
    }
}

#[test]
fn batched_mxfp4_multi_token_without_exact_vs_singleton() {
    let input_dim = 64i32;
    let output_dim = 32i32;
    let seq = 2i32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.01171875)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    let weight = QuantizedWeight {
        weight: quantized[0].clone(),
        scales: Some(quantized[1].clone()),
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
    let input_data: Vec<f32> = (0..(seq * input_dim) as usize)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let input = array_f32(&input_data, &[1, seq, input_dim]);
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let batched = qw(&input, &weight);
    let mut max_abs = 0.0f32;
    for t in 0..seq {
        let start = (t * input_dim) as usize;
        let single = array_f32(
            &input_data[start..start + input_dim as usize],
            &[1, 1, input_dim],
        );
        let expected = qw(&single, &weight);
        let actual = contiguous(
            &slice(
                &batched,
                &[0, t, 0],
                &[1, t + 1, output_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        eval(&[&actual, &expected]);
        let got = actual.data_f32();
        let exp = expected.data_f32();
        for i in 0..got.len() {
            max_abs = max_abs.max((got[i] - exp[i]).abs());
        }
    }
    eprintln!("batched MXFP4 S=2 vs singleton max_abs={max_abs}");
    assert!(
        max_abs < 1.0e-5,
        "small F32 MXFP4 batch/singleton control exceeds tolerance (max_abs={max_abs})"
    );
}

#[test]
fn exact_shared_mxfp4_multi_token_matches_singleton_rows() {
    let input_dim = 64i32;
    let output_dim = 32i32;
    let seq = 2i32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.01171875)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    assert_eq!(quantized.len(), 2, "mxfp4 quant returns [packed, scales]");
    let weight = QuantizedWeight {
        weight: quantized[0].clone(),
        scales: Some(quantized[1].clone()),
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
    let input_data: Vec<f32> = (0..(seq * input_dim) as usize)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let input = array_f32(&input_data, &[1, seq, input_dim]);
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let batched = qw(&input, &weight);
    for t in 0..seq {
        let start = (t * input_dim) as usize;
        let single = array_f32(
            &input_data[start..start + input_dim as usize],
            &[1, 1, input_dim],
        );
        let expected = qw(&single, &weight);
        let actual = contiguous(
            &slice(
                &batched,
                &[0, t, 0],
                &[1, t + 1, output_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        eval(&[&actual, &expected]);
        let got = actual.data_f32();
        let exp = expected.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..got.len() {
            max_abs = max_abs.max((got[i] - exp[i]).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "MXFP4 exact S>1 Shared qw must match singleton row {t}, max_abs={max_abs}"
        );
    }
}

#[test]
fn exact_mxfp4_short_qmm_flatten_matches_3d() {
    let input_dim = 64i32;
    let output_dim = 32i32;
    let seq = 2i32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.01171875)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    let input_data: Vec<f32> = (0..(seq * input_dim) as usize)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let x = array_f32(&input_data, &[1, seq, input_dim]);
    let qmm_3d = mlx_sys::quantized_matmul_with_mode(
        &x,
        &quantized[0],
        &quantized[1],
        None,
        true,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
    );
    let qmm_flat = super::exact_mxfp4_short_qmm(&x, MlxQuantizationMode::Mxfp4, |flat| {
        mlx_sys::quantized_matmul_with_mode(
            flat,
            &quantized[0],
            &quantized[1],
            None,
            true,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
        )
    });
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let qmm_flat_exact = super::exact_mxfp4_short_qmm(&x, MlxQuantizationMode::Mxfp4, |flat| {
        mlx_sys::quantized_matmul_with_mode(
            flat,
            &quantized[0],
            &quantized[1],
            None,
            true,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
        )
    });
    eval(&[&qmm_3d, &qmm_flat, &qmm_flat_exact]);
    assert_eq!(qmm_flat.shape(), qmm_3d.shape());
    assert_eq!(qmm_flat.data_f32(), qmm_3d.data_f32());
    assert_eq!(qmm_flat_exact.shape(), qmm_3d.shape());
    let mut max_abs = 0.0f32;
    let a = qmm_3d.data_f32();
    let b = qmm_flat_exact.data_f32();
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "flattened exact MXFP4 S=2 qmm must match 3-D, max_abs={max_abs}"
    );
}

#[test]
fn exact_affine_s2_qmm_flatten_matches_3d() {
    let input_dim = 64i32;
    let output_dim = 32i32;
    let seq = 2i32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.01171875)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(8),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(
        quantized.len(),
        3,
        "affine 8-bit returns [w, scales, biases]"
    );
    let input_data: Vec<f32> = (0..(seq * input_dim) as usize)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let x = array_f32(&input_data, &[1, seq, input_dim]);
    let qmm_3d = mlx_sys::quantized_matmul_with_mode(
        &x,
        &quantized[0],
        &quantized[1],
        Some(&quantized[2]),
        true,
        Some(32),
        Some(8),
        MlxQuantizationMode::Affine,
        None,
    );
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let qmm_flat = super::exact_mxfp4_short_qmm(&x, MlxQuantizationMode::Affine, |flat| {
        mlx_sys::quantized_matmul_with_mode(
            flat,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            true,
            Some(32),
            Some(8),
            MlxQuantizationMode::Affine,
            None,
        )
    });
    eval(&[&qmm_3d, &qmm_flat]);
    assert_eq!(qmm_flat.shape(), qmm_3d.shape());
    let a = qmm_3d.data_f32();
    let b = qmm_flat.data_f32();
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        max_abs = max_abs.max((a[i] - b[i]).abs());
    }
    assert!(
        max_abs < 1.0e-5,
        "flattened exact affine S=2 qmm must match 3-D, max_abs={max_abs}"
    );
}

fn mxfp4_quantized_weight(output_dim: i32, input_dim: i32) -> QuantizedWeight {
    let weight_data: Vec<f32> = (0..(input_dim * output_dim) as usize)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.01171875)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    assert_eq!(quantized.len(), 2, "mxfp4 quant returns [packed, scales]");
    QuantizedWeight {
        weight: quantized[0].clone(),
        scales: Some(quantized[1].clone()),
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

fn max_abs_f32(left: &MlxArray, right: &MlxArray) -> f32 {
    let a = astype(left, MlxDtype::Float32, None);
    let b = astype(right, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    a.data_f32()
        .iter()
        .zip(b.data_f32().iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn invariant_mxfp4_qmv_fast_matches_mlx_singleton() {
    let input_dim = 512i32;
    let output_dim = 32i32;
    let weight = mxfp4_quantized_weight(output_dim, input_dim);
    let input_data: Vec<f32> = (0..input_dim as usize)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let x = astype(
        &array_f32(&input_data, &[1, 1, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let metal = super::invariant_mxfp4_qmv_fast_impl(&x, &weight)
        .expect("512-wide MXFP4 qmv_fast must engage");
    let mlx = mlx_sys::quantized_matmul_with_mode(
        &x,
        &weight.weight,
        weight.scales.as_ref().unwrap(),
        None,
        true,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
    );
    eval(&[&metal, &mlx]);
    assert_eq!(metal.shape(), mlx.shape());
    let max_abs = max_abs_f32(&metal, &mlx);
    assert!(
        max_abs == 0.0,
        "MXFP4 qmv_fast microbatch must match MLX S=1 fp_qmv_fast, max_abs={max_abs}"
    );
}

#[test]
fn invariant_mxfp4_qmv_fast_s2_rows_match_singleton() {
    let input_dim = 512i32;
    let output_dim = 32i32;
    let seq = 2i32;
    let weight = mxfp4_quantized_weight(output_dim, input_dim);
    let input_data: Vec<f32> = (0..(seq * input_dim) as usize)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let x = astype(
        &array_f32(&input_data, &[1, seq, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let metal =
        super::invariant_mxfp4_qmv_fast_impl(&x, &weight).expect("S=2 MXFP4 qmv_fast must engage");
    eval(&[&metal]);
    for t in 0..seq {
        let start = (t * input_dim) as usize;
        let single = astype(
            &array_f32(
                &input_data[start..start + input_dim as usize],
                &[1, 1, input_dim],
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let expected = mlx_sys::quantized_matmul_with_mode(
            &single,
            &weight.weight,
            weight.scales.as_ref().unwrap(),
            None,
            true,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
        );
        let actual = contiguous(
            &slice(
                &metal,
                &[0, t, 0],
                &[1, t + 1, output_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        eval(&[&actual, &expected]);
        let max_abs = max_abs_f32(&actual, &expected);
        assert!(
            max_abs == 0.0,
            "MXFP4 S=2 row {t} must match MLX singleton, max_abs={max_abs}"
        );
    }
}

#[test]
fn exact_qw_mxfp4_qmv_fast_s2_matches_singleton() {
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let input_dim = 512i32;
    let output_dim = 32i32;
    let seq = 2i32;
    let weight = mxfp4_quantized_weight(output_dim, input_dim);
    let input_data: Vec<f32> = (0..(seq * input_dim) as usize)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let x = astype(
        &array_f32(&input_data, &[1, seq, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let batched = qw(&x, &weight);
    for t in 0..seq {
        let start = (t * input_dim) as usize;
        let single = astype(
            &array_f32(
                &input_data[start..start + input_dim as usize],
                &[1, 1, input_dim],
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let expected = qw(&single, &weight);
        let actual = contiguous(
            &slice(
                &batched,
                &[0, t, 0],
                &[1, t + 1, output_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        eval(&[&actual, &expected]);
        let max_abs = max_abs_f32(&actual, &expected);
        assert!(
            max_abs == 0.0,
            "exact qw() MXFP4 S=2 row {t} must match singleton, max_abs={max_abs}"
        );
    }
}

#[test]
fn invariant_affine_projection_is_bit_exact_across_microbatch_shapes() {
    let input_dim = 64;
    let output_dim = 64;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 113) as f32 - 56.0) * 0.01171875)
        .collect();
    let weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let weight = QuantizedWeight {
        weight: quantized[0].clone(),
        scales: Some(quantized[1].clone()),
        biases: Some(quantized[2].clone()),
        group_size: 32,
        bits: 4,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let input_data: Vec<f32> = (0..2 * input_dim)
        .map(|index| ((index % 43) as f32 - 21.0) * 0.02734375)
        .collect();
    let input = astype(
        &array_f32(&input_data, &[1, 2, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let microbatch = invariant_projection_metal_impl(&input, &weight)
        .expect("affine invariant projection should support two rows");

    for row in 0..2 {
        let single = contiguous(
            &slice(
                &input,
                &[0, row, 0],
                &[1, row + 1, input_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        let expected = invariant_projection_metal_impl(&single, &weight)
            .expect("affine invariant projection should support one row");
        let actual = contiguous(
            &slice(
                &microbatch,
                &[0, row, 0],
                &[1, row + 1, output_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        let actual = astype(&actual, MlxDtype::Float32, None);
        let expected = astype(&expected, MlxDtype::Float32, None);
        eval(&[&actual, &expected]);
        assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
    }
}

#[test]
fn invariant_split_2816_matches_mlx_and_microbatch() {
    // Split path: 2560 qmv_fast + 256 MLX rem. Must match full MLX and microbatch.
    let input_dim = 2816;
    let output_dim = 64;
    let bits = 6;
    let group_size = 32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
        .collect();
    let source = array_f32(&weight_data, &[output_dim, input_dim]);
    let q = quantize(
        &source,
        Some(group_size),
        Some(bits),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let weight = QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size,
        bits,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let input_data: Vec<f32> = (0..3 * input_dim)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
        .collect();
    let input = astype(
        &array_f32(&input_data, &[1, 3, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let mb = invariant_projection_metal_impl(&input, &weight).expect("split 2816");
    for row in 0..3 {
        let single = contiguous(
            &slice(
                &input,
                &[0, row, 0],
                &[1, row + 1, input_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        let inv_s = invariant_projection_metal_impl(&single, &weight).expect("single");
        let mlx_s = quantized_matmul(
            &single,
            &q[0],
            &q[1],
            Some(&q[2]),
            true,
            Some(group_size),
            Some(bits),
            None,
        );
        let actual = contiguous(
            &slice(
                &mb,
                &[0, row, 0],
                &[1, row + 1, output_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        let a = astype(&actual, MlxDtype::Float32, None);
        let b = astype(&inv_s, MlxDtype::Float32, None);
        let c = astype(&mlx_s, MlxDtype::Float32, None);
        eval(&[&a, &b, &c]);
        assert_eq!(a.data_f32(), b.data_f32(), "microbatch vs single row {row}");
        // Split sum may have tiny float assoc vs full MLX — allow ulp-level.
        let da = a.data_f32();
        let dc = c.data_f32();
        let mut maxd = 0.0f32;
        for i in 0..da.len() {
            maxd = maxd.max((da[i] - dc[i]).abs());
        }
        eprintln!("row {row}: max|split-mlx|={maxd} bitexact={}", da == dc);
        assert!(maxd < 1e-2, "split vs full MLX too far: {maxd}");
    }
}

#[test]
fn invariant_qmv_2816_matches_mlx_singleton() {
    // Gemma hidden=2816 is eleven complete 6-bit qmv lane blocks.
    let input_dim = 2816;
    let output_dim = 64;
    let bits = 6;
    let group_size = 32;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
        .collect();
    let source = array_f32(&weight_data, &[output_dim, input_dim]);
    let q = quantize(
        &source,
        Some(group_size),
        Some(bits),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let weight = QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size,
        bits,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let input_data: Vec<f32> = (0..input_dim)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
        .collect();
    let input = astype(
        &array_f32(&input_data, &[1, 1, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let inv = invariant_projection_metal_impl(&input, &weight).expect("qmv 2816");
    let mlx = quantized_matmul(
        &input,
        &q[0],
        &q[1],
        Some(&q[2]),
        true,
        Some(group_size),
        Some(bits),
        None,
    );
    let a = astype(&inv, MlxDtype::Float32, None);
    let b = astype(&mlx, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    let da = a.data_f32();
    let db = b.data_f32();
    let mut maxd = 0.0f32;
    for i in 0..da.len() {
        maxd = maxd.max((da[i] - db[i]).abs());
    }
    assert!(maxd < 1e-3, "qmv 2816 vs MLX maxΔ={maxd}");
}

#[test]
fn invariant_gemma26_shapes_self_consistent_and_mlx() {
    // Realistic Gemma 26B-A4B projection shapes under Shared microbatch S=3.
    let cases: &[(i32, i32, i32, i32)] = &[
        // (input_dim, output_dim, bits, group_size)
        (2816, 4096, 6, 32), // q_proj
        (2816, 2048, 6, 32), // k/v
        (4096, 2816, 6, 32), // o_proj (512-aligned in)
        (2816, 2112, 6, 64), // dense gate/up
        (2112, 2816, 6, 64), // dense down
        (2816, 2112, 6, 32),
        (2112, 2816, 6, 32),
        (2816, 128, 8, 64), // router-ish
        (2816, 128, 8, 32),
        (704, 2816, 4, 32), // expert down-like (if used as linear)
        (2816, 704, 4, 32),
    ];
    for &(input_dim, output_dim, bits, group_size) in cases {
        let n = (input_dim * output_dim) as usize;
        let weight_data: Vec<f32> = (0..n)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim, input_dim]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let s = 3_i32;
        let input_data: Vec<f32> = (0..(s * input_dim) as usize)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, s, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let Some(mb) = invariant_projection_metal_impl(&input, &weight) else {
            eprintln!("SKIP no invariant in={input_dim} out={output_dim} b={bits} gs={group_size}");
            continue;
        };
        let mut max_self = 0.0f32;
        let mut max_mlx = 0.0f32;
        let mut bitexact_self = true;
        let mut bitexact_mlx = true;
        for row in 0..s {
            let single = contiguous(
                &slice(
                    &input,
                    &[0, row, 0],
                    &[1, row + 1, input_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let inv_s =
                invariant_projection_metal_impl(&single, &weight).expect("singleton invariant");
            let mlx_s = quantized_matmul(
                &single,
                &q[0],
                &q[1],
                Some(&q[2]),
                true,
                Some(group_size),
                Some(bits),
                None,
            );
            let actual = contiguous(
                &slice(
                    &mb,
                    &[0, row, 0],
                    &[1, row + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let a = astype(&actual, MlxDtype::Float32, None);
            let b = astype(&inv_s, MlxDtype::Float32, None);
            let c = astype(&mlx_s, MlxDtype::Float32, None);
            eval(&[&a, &b, &c]);
            let da = a.data_f32();
            let db = b.data_f32();
            let dc = c.data_f32();
            if da != db {
                bitexact_self = false;
            }
            if da != dc {
                bitexact_mlx = false;
            }
            for i in 0..da.len() {
                max_self = max_self.max((da[i] - db[i]).abs());
                max_mlx = max_mlx.max((da[i] - dc[i]).abs());
            }
        }
        eprintln!(
            "in={input_dim} out={output_dim} b={bits} gs={group_size}: self_exact={bitexact_self} max_self={max_self:.6e} mlx_exact={bitexact_mlx} max_mlx={max_mlx:.6e}"
        );
        assert!(
            max_self == 0.0,
            "SELF FAIL in={input_dim} out={output_dim}: max_self={max_self}"
        );
    }
}

#[test]
fn invariant_2d_vs_3d_singleton_shape() {
    let input_dim = 2816i32;
    let output_dim = 4096i32;
    let bits = 6i32;
    let group_size = 32i32;
    let n = (input_dim * output_dim) as usize;
    let weight_data: Vec<f32> = (0..n)
        .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
        .collect();
    let source = array_f32(&weight_data, &[output_dim, input_dim]);
    let q = quantize(
        &source,
        Some(group_size),
        Some(bits),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let weight = QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size,
        bits,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let input_data: Vec<f32> = (0..input_dim as usize)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
        .collect();
    let x3 = astype(
        &array_f32(&input_data, &[1, 1, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let x2 = astype(
        &array_f32(&input_data, &[1, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let y3 = invariant_projection_metal_impl(&x3, &weight).expect("3d");
    let y2 = invariant_projection_metal_impl(&x2, &weight).expect("2d");
    let a = astype(&y3, MlxDtype::Float32, None);
    let b = astype(&y2, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    let da = a.data_f32();
    let db = b.data_f32();
    assert_eq!(da.len(), db.len());
    let mut maxd = 0.0f32;
    for i in 0..da.len() {
        maxd = maxd.max((da[i] - db[i]).abs());
    }
    eprintln!("2d vs 3d maxd={maxd} bitexact={}", da == db);
    assert_eq!(da, db, "2d vs 3d shape mismatch maxd={maxd}");
}

#[test]
fn invariant_nonfast_2112_matches_mlx_singleton() {
    // intermediate_size-like: non-fast path must match MLX bitexact.
    let input_dim: i32 = 2112;
    let output_dim: i32 = 64;
    let bits = 6;
    let group_size = 32;
    let weight_data: Vec<f32> = (0..(input_dim * output_dim) as usize)
        .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
        .collect();
    let source = array_f32(&weight_data, &[output_dim, input_dim]);
    let q = quantize(
        &source,
        Some(group_size),
        Some(bits),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let weight = QuantizedWeight {
        weight: q[0].clone(),
        scales: Some(q[1].clone()),
        biases: Some(q[2].clone()),
        group_size,
        bits,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let input_data: Vec<f32> = (0..input_dim as usize)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
        .collect();
    let input = astype(
        &array_f32(&input_data, &[1, 1, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let inv = invariant_projection_metal_impl(&input, &weight).expect("non-fast MLX");
    let mlx = quantized_matmul(
        &input,
        &q[0],
        &q[1],
        Some(&q[2]),
        true,
        Some(group_size),
        Some(bits),
        None,
    );
    let a = astype(&inv, MlxDtype::Float32, None);
    let b = astype(&mlx, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    // 2112 uses split (2048 qmv_fast + 64 MLX); absolute MLX match is ulp-level.
    let da = a.data_f32();
    let db = b.data_f32();
    let mut maxd = 0.0f32;
    for i in 0..da.len() {
        maxd = maxd.max((da[i] - db[i]).abs());
    }
    assert!(maxd < 1e-3, "2112 split vs MLX maxΔ={maxd}");
}

#[test]
fn invariant_affine_bf16_q4_kernel_matches_generic_bit_exact() {
    // The bf16 / 4-bit specialisation must reproduce the generic invariant
    // kernel bit for bit at every verify width, for the production group
    // sizes and for both the aligned and the split (K % 512 != 0) routes.
    for (input_dim, output_dim, group_size, bf16_sidecars) in [
        (512, 64, 32, false),
        (1024, 64, 64, false),
        (1536, 32, 32, false),
        (768, 64, 64, false),
        (1024, 64, 128, false),
        (512, 64, 32, true),
        (1024, 64, 64, true),
    ] {
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source_weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &source_weight,
            Some(group_size),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        // Production packs carry bf16 scales/biases, which makes OutT bf16;
        // f32 sidecars exercise the f32 OutT path.
        let sidecar = |array: &MlxArray| {
            if bf16_sidecars {
                astype(array, MlxDtype::Bfloat16, None)
            } else {
                array.clone()
            }
        };
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(sidecar(&quantized[1])),
            biases: Some(sidecar(&quantized[2])),
            group_size,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        for leading in 1..=4 {
            let input_data: Vec<f32> = (0..leading * input_dim)
                .map(|index| match index % 97 {
                    // Special values: NaN, +/-inf and a bf16 subnormal must
                    // widen identically in both kernels.
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    3 => 1.0e-39,
                    _ => ((index % 89) as f32 - 44.0) * 0.015625 + (index % 7) as f32 * 0.001,
                })
                .collect();
            let input = astype(
                &array_f32(&input_data, &[1, leading, input_dim]),
                MlxDtype::Bfloat16,
                None,
            );
            let generic = invariant_projection_metal_impl_with_kernel(&input, &weight, false)
                .expect("generic invariant affine projection");
            let specialised = invariant_projection_metal_impl_with_kernel(&input, &weight, true)
                .expect("bf16 q4 invariant affine projection");
            assert_eq!(generic.dtype(), specialised.dtype());
            let a = astype(&generic, MlxDtype::Float32, None);
            let b = astype(&specialised, MlxDtype::Float32, None);
            eval(&[&a, &b]);
            let av = a.data_f32();
            let bv = b.data_f32();
            assert_eq!(av.len(), bv.len());
            for i in 0..av.len() {
                assert_eq!(
                    av[i].to_bits(),
                    bv[i].to_bits(),
                    "K={input_dim} N={output_dim} gs={group_size} leading={leading} diverged at {i}"
                );
            }
        }
    }
}

#[test]
fn invariant_affine_qmv_fast_matches_mlx_singleton_and_microbatch() {
    let input_dim = 512;
    let output_dim = 64;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 251) as f32 - 125.0) * 0.00390625)
        .collect();
    let source_weight = array_f32(&weight_data, &[output_dim, input_dim]);
    let input_data: Vec<f32> = (0..2 * input_dim)
        .map(|index| ((index % 89) as f32 - 44.0) * 0.015625)
        .collect();
    let input = astype(
        &array_f32(&input_data, &[1, 2, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );

    for bits in [4, 6, 8] {
        let quantized = quantize(
            &source_weight,
            Some(64),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: Some(quantized[2].clone()),
            group_size: 64,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let microbatch = invariant_projection_metal_impl(&input, &weight)
            .expect("fast invariant affine projection should support two rows");

        for row in 0..2 {
            let single = contiguous(
                &slice(
                    &input,
                    &[0, row, 0],
                    &[1, row + 1, input_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let invariant_single = invariant_projection_metal_impl(&single, &weight)
                .expect("fast invariant affine projection should support one row");
            let mlx_single = quantized_matmul(
                &single,
                &quantized[0],
                &quantized[1],
                Some(&quantized[2]),
                true,
                Some(64),
                Some(bits),
                None,
            );
            let actual = contiguous(
                &slice(
                    &microbatch,
                    &[0, row, 0],
                    &[1, row + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let actual = astype(&actual, MlxDtype::Float32, None);
            let invariant_single = astype(&invariant_single, MlxDtype::Float32, None);
            let mlx_single = astype(&mlx_single, MlxDtype::Float32, None);
            eval(&[&actual, &invariant_single, &mlx_single]);
            assert_eq!(
                actual.data_f32(),
                invariant_single.data_f32(),
                "microbatch row {row}, bits={bits}"
            );
            assert_eq!(
                invariant_single.data_f32(),
                mlx_single.data_f32(),
                "MLX singleton parity, bits={bits}"
            );
        }
    }
}

#[test]
fn invariant_dense_projection_is_bit_exact_across_microbatch_shapes() {
    let input_dim = 64;
    let output_dim = 48;
    let weight_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|index| ((index % 97) as f32 - 48.0) * 0.009765625)
        .collect();
    let weight = QuantizedWeight {
        weight: astype(
            &array_f32(&weight_data, &[output_dim, input_dim]),
            MlxDtype::Bfloat16,
            None,
        ),
        scales: None,
        biases: None,
        group_size: 0,
        bits: 16,
        mode: "affine".to_string(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    };
    let input_data: Vec<f32> = (0..2 * input_dim)
        .map(|index| ((index % 37) as f32 - 18.0) * 0.0234375)
        .collect();
    let input = astype(
        &array_f32(&input_data, &[1, 2, input_dim]),
        MlxDtype::Bfloat16,
        None,
    );
    let microbatch = invariant_projection_metal_impl(&input, &weight)
        .expect("dense invariant projection should support two rows");

    for row in 0..2 {
        let single = contiguous(
            &slice(
                &input,
                &[0, row, 0],
                &[1, row + 1, input_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        let expected = invariant_projection_metal_impl(&single, &weight)
            .expect("dense invariant projection should support one row");
        let actual = contiguous(
            &slice(
                &microbatch,
                &[0, row, 0],
                &[1, row + 1, output_dim],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        let actual = astype(&actual, MlxDtype::Float32, None);
        let expected = astype(&expected, MlxDtype::Float32, None);
        eval(&[&actual, &expected]);
        assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
    }
}

#[test]
fn add_then_multiply_scalar_metal_matches_unfused_bf16_rounding() {
    let a = astype(
        &array_f32(&[0.333, -1.125, 2.75, 3.125], &[1, 4]),
        MlxDtype::Bfloat16,
        None,
    );
    let b = astype(
        &array_f32(&[1.875, 4.25, -2.375, 0.625], &[1, 4]),
        MlxDtype::Bfloat16,
        None,
    );
    let scalar = astype(&array_f32(&[0.3125], &[1]), MlxDtype::Bfloat16, None);

    let direct = add_then_multiply_scalar_metal_impl(&a, &b, &scalar)
        .expect("scalar fused add should support bf16 inputs");
    let reference = multiply(&add(&a, &b, None), &scalar, None);
    let direct = astype(&direct, MlxDtype::Float32, None);
    let reference = astype(&reference, MlxDtype::Float32, None);
    eval(&[&direct, &reference]);

    assert_eq!(direct.shape(), vec![1, 4]);
    assert_eq!(direct.data_f32(), reference.data_f32());
}

#[test]
fn add_then_multiply_scalar_metal_rejects_broadcast_vector_scale() {
    let a = array_f32(&[1.0, 2.0], &[1, 2]);
    let b = array_f32(&[3.0, 4.0], &[1, 2]);
    let vector_scale = array_f32(&[0.5, 0.25], &[2]);

    assert!(
        add_then_multiply_scalar_metal_impl(&a, &b, &vector_scale).is_none(),
        "only exact scalar layer-scale tensors are fused"
    );
}
