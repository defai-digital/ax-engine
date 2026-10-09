
use super::*;
use crate::transforms::{eval, eval_first_u32};

#[test]
fn conv1d_depthwise_reports_reference_shape() {
    let input = zeros(&[1, 6, 2], MlxDtype::Float32, None);
    let weight = zeros(&[2, 3, 1], MlxDtype::Float32, None);
    let out = conv1d(&input, &weight, 1, 0, 1, 2, None);

    assert_eq!(out.shape(), vec![1, 4, 2]);
}

#[test]
fn stack_and_where_preserve_expected_shapes() {
    let a = zeros(&[2, 3], MlxDtype::Float32, None);
    let b = zeros(&[2, 3], MlxDtype::Float32, None);
    let condition = zeros(&[2, 3], MlxDtype::Bool, None);

    assert_eq!(stack(&[&a, &b], 1, None).shape(), vec![2, 2, 3]);
    assert_eq!(where_cond(&condition, &a, &b, None).shape(), vec![2, 3]);
}

#[test]
fn scalar_math_wrappers_keep_input_shape() {
    let a = zeros(&[2, 3], MlxDtype::Float32, None);
    let min_value = 0.0_f32;
    let max_value = 1.0_f32;
    let min = MlxArray::from_raw_data(
        &min_value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1],
        MlxDtype::Float32,
    );
    let max = MlxArray::from_raw_data(
        &max_value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1],
        MlxDtype::Float32,
    );

    assert_eq!(
        log1p(&exp(&negative(&a, None), None), None).shape(),
        vec![2, 3]
    );
    assert_eq!(clip(&a, &min, &max, None).shape(), vec![2, 3]);
}

#[test]
fn gelu_approx_mul_direct_matches_portable_composition() {
    let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.125).collect();
    let x_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.03125).collect();
    let gate = MlxArray::from_raw_data(
        gate_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_data.as_slice()),
        &[2, 3, 4],
        MlxDtype::Float32,
    );
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 3, 4],
        MlxDtype::Float32,
    );

    let prev = crate::op_count::op_count_snapshot();
    let direct = gelu_approx_mul(&gate, &x, None);
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ activation shim should count as one Rust FFI dispatch"
    );
    let portable = multiply(&gelu_approx(&gate, None), &x, None);
    let direct_f32 = astype(&direct, MlxDtype::Float32, None);
    let portable_f32 = astype(&portable, MlxDtype::Float32, None);
    eval(&[&direct_f32, &portable_f32]);

    assert_eq!(direct_f32.shape(), vec![2, 3, 4]);
    assert_eq!(
        direct_f32.data_f32().to_vec(),
        portable_f32.data_f32().to_vec(),
        "direct C++ activation shim must preserve mlx-lm GEGLU math"
    );
}

#[test]
fn silu_mul_direct_matches_portable_composition() {
    let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.125).collect();
    let x_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.03125).collect();
    let gate = MlxArray::from_raw_data(
        gate_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_data.as_slice()),
        &[2, 3, 4],
        MlxDtype::Float32,
    );
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 3, 4],
        MlxDtype::Float32,
    );

    let prev = crate::op_count::op_count_snapshot();
    let direct = silu_mul(&gate, &x, None);
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ SwiGLU activation shim should count as one Rust FFI dispatch"
    );
    let portable = multiply(&silu(&gate, None), &x, None);
    let direct_f32 = astype(&direct, MlxDtype::Float32, None);
    let portable_f32 = astype(&portable, MlxDtype::Float32, None);
    eval(&[&direct_f32, &portable_f32]);

    assert_eq!(direct_f32.shape(), vec![2, 3, 4]);
    assert_eq!(
        direct_f32.data_f32().to_vec(),
        portable_f32.data_f32().to_vec(),
        "direct C++ SwiGLU activation shim must preserve silu(gate) * x math"
    );
}

#[test]
fn rms_norm_silu_mul_normed_matches_five_op_chain() {
    let hidden_data: Vec<f32> = (0..256).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
    let gate_data: Vec<f32> = (0..256).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
    let weight_data: Vec<f32> = (0..64).map(|i| 0.75 + (i as f32) * 0.004).collect();
    let hidden = astype(
        &MlxArray::from_raw_data(
            hidden_data.as_ptr() as *const u8,
            std::mem::size_of_val(hidden_data.as_slice()),
            &[1, 2, 2, 64],
            MlxDtype::Float32,
        ),
        MlxDtype::Bfloat16,
        None,
    );
    let gate = astype(
        &MlxArray::from_raw_data(
            gate_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_data.as_slice()),
            &[1, 2, 2, 64],
            MlxDtype::Float32,
        ),
        MlxDtype::Bfloat16,
        None,
    );
    let weight = astype(
        &MlxArray::from_raw_data(
            weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(weight_data.as_slice()),
            &[64],
            MlxDtype::Float32,
        ),
        MlxDtype::Bfloat16,
        None,
    );
    let prev = crate::op_count::op_count_snapshot();
    let direct = rms_norm_silu_mul_normed(&hidden, &gate, &weight, 1e-6, None);
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "portable-gate C++ composite should count as one Rust FFI dispatch"
    );
    let normed = crate::fast::rms_norm(&hidden, Some(&weight), 1e-6, None);
    let portable = astype(
        &silu_mul(
            &astype(&gate, MlxDtype::Float32, None),
            &astype(&normed, MlxDtype::Float32, None),
            None,
        ),
        hidden.dtype(),
        None,
    );
    eval(&[&direct, &portable]);
    assert_eq!(direct.shape(), portable.shape());
    assert_eq!(direct.dtype(), portable.dtype());
    let a = astype(&direct, MlxDtype::Float32, None);
    let b = astype(&portable, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    assert_eq!(
        a.data_f32().to_vec(),
        b.data_f32().to_vec(),
        "C++ rms+silu_mul composite must match the five-op portable gate"
    );
}

#[test]
fn gelu_approx_mul_matmul_direct_matches_portable_composition() {
    let gate_data: Vec<f32> = (0..6).map(|i| ((i as f32) - 3.0) * 0.25).collect();
    let x_data: Vec<f32> = (0..6).map(|i| ((i as f32) + 1.0) * 0.125).collect();
    let weight_data: Vec<f32> = (0..12).map(|i| ((i as f32) - 6.0) * 0.0625).collect();
    let gate = MlxArray::from_raw_data(
        gate_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_data.as_slice()),
        &[2, 3],
        MlxDtype::Float32,
    );
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 3],
        MlxDtype::Float32,
    );
    let weight = MlxArray::from_raw_data(
        weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(weight_data.as_slice()),
        &[3, 4],
        MlxDtype::Float32,
    );

    let prev = crate::op_count::op_count_snapshot();
    let direct = gelu_approx_mul_matmul(&gate, &x, &weight, None);
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ activation+matmul shim should count as one Rust FFI dispatch"
    );
    let portable = matmul(
        &multiply(&gelu_approx(&gate, None), &x, None),
        &weight,
        None,
    );
    eval(&[&direct, &portable]);

    assert_eq!(direct.shape(), vec![2, 4]);
    assert_eq!(
        direct.data_f32().to_vec(),
        portable.data_f32().to_vec(),
        "direct C++ activation+matmul shim must preserve portable math"
    );
}

#[test]
fn qk_norm_rope_direct_matches_portable_composition() {
    let seq = 3_i32;
    let n_heads = 2_i32;
    let head_dim = 4_i32;
    let width = n_heads * head_dim;
    let proj_data: Vec<f32> = (0..(seq * width))
        .map(|i| ((i as f32) - 11.0) * 0.03125)
        .collect();
    let norm_data: Vec<f32> = (0..head_dim).map(|i| 0.75 + (i as f32) * 0.125).collect();
    let proj = MlxArray::from_raw_data(
        proj_data.as_ptr() as *const u8,
        std::mem::size_of_val(proj_data.as_slice()),
        &[1, seq, width],
        MlxDtype::Float32,
    );
    let norm = MlxArray::from_raw_data(
        norm_data.as_ptr() as *const u8,
        std::mem::size_of_val(norm_data.as_slice()),
        &[head_dim],
        MlxDtype::Float32,
    );

    let prev = crate::op_count::op_count_snapshot();
    let direct = qk_norm_rope_bhsd_from_proj(
        &proj,
        Some(&norm),
        n_heads,
        head_dim,
        1.0e-6,
        head_dim,
        false,
        Some(10_000.0),
        2,
        None,
        None,
    );
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ QK-norm+RoPE probe should count as one Rust FFI dispatch"
    );

    let bhsd = as_strided(
        &proj,
        &[1, n_heads, seq, head_dim],
        &[
            i64::from(seq * width),
            i64::from(head_dim),
            i64::from(width),
            1,
        ],
        0,
        None,
    );
    let normed = crate::fast::rms_norm(&bhsd, Some(&norm), 1.0e-6, None);
    let reference = crate::fast::rope(&normed, head_dim, false, Some(10_000.0), 1.0, 2, None, None);
    let direct = contiguous(&direct, None);
    let reference = contiguous(&reference, None);
    eval(&[&direct, &reference]);

    assert_eq!(direct.shape(), vec![1, n_heads, seq, head_dim]);
    assert_eq!(reference.shape(), direct.shape());
    assert_close_f32(direct.data_f32(), reference.data_f32(), 1.0e-6);
}

#[test]
fn qwen_linear_attention_inputs_packed_direct_matches_portable_composition() {
    let seq = 2_i32;
    let hidden = 32_i32;
    let num_key_heads = 2_i32;
    let num_value_heads = 4_i32;
    let key_head_dim = 3_i32;
    let value_head_dim = 2_i32;
    let value_heads_per_key = num_value_heads / num_key_heads;
    let value_dim_per_key = value_heads_per_key * value_head_dim;
    let qkvz_per_key = key_head_dim * 2 + value_dim_per_key * 2;
    let qkvz_out = num_key_heads * qkvz_per_key;
    let ba_out = num_key_heads * value_heads_per_key * 2;

    let x_data: Vec<f32> = (0..(seq * hidden))
        .map(|i| ((i as f32) - 31.0) * 0.03125)
        .collect();
    let qkvz_weight_data: Vec<f32> = (0..(qkvz_out * hidden))
        .map(|i| ((i as f32) - 448.0) * 0.0005)
        .collect();
    let ba_weight_data: Vec<f32> = (0..(ba_out * hidden))
        .map(|i| ((i as f32) - 128.0) * 0.001)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let qkvz_weight = MlxArray::from_raw_data(
        qkvz_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkvz_weight_data.as_slice()),
        &[qkvz_out, hidden],
        MlxDtype::Float32,
    );
    let ba_weight = MlxArray::from_raw_data(
        ba_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(ba_weight_data.as_slice()),
        &[ba_out, hidden],
        MlxDtype::Float32,
    );
    let qkvz_q = quantize(
        &qkvz_weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let ba_q = quantize(
        &ba_weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(qkvz_q.len(), 3);
    assert_eq!(ba_q.len(), 3);

    let prev = crate::op_count::op_count_snapshot();
    let (direct_qkv, direct_z, direct_a, direct_b) = qwen_linear_attention_inputs_packed(
        &x,
        &qkvz_q[0],
        Some(&qkvz_q[1]),
        Some(&qkvz_q[2]),
        &ba_q[0],
        Some(&ba_q[1]),
        Some(&ba_q[2]),
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        32,
        4,
        32,
        4,
        None,
    )
    .expect("direct packed linear-attention shim should accept qwen-compatible shapes");
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ linear-attention input shim should count as one Rust FFI dispatch"
    );

    let mixed_qkvz = quantized_matmul(
        &x,
        &qkvz_q[0],
        &qkvz_q[1],
        Some(&qkvz_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let mixed_qkvz = reshape(&mixed_qkvz, &[1, seq, num_key_heads, qkvz_per_key], None);
    let q = slice_last_dim(&mixed_qkvz, 0, key_head_dim, None);
    let k = slice_last_dim(&mixed_qkvz, key_head_dim, key_head_dim * 2, None);
    let v = slice_last_dim(
        &mixed_qkvz,
        key_head_dim * 2,
        key_head_dim * 2 + value_dim_per_key,
        None,
    );
    let z = slice_last_dim(
        &mixed_qkvz,
        key_head_dim * 2 + value_dim_per_key,
        qkvz_per_key,
        None,
    );
    let portable_qkv = concatenate(
        &[
            &reshape(&q, &[1, seq, num_key_heads * key_head_dim], None),
            &reshape(&k, &[1, seq, num_key_heads * key_head_dim], None),
            &reshape(&v, &[1, seq, num_value_heads * value_head_dim], None),
        ],
        2,
        None,
    );
    let portable_z = reshape(&z, &[1, seq, num_value_heads, value_head_dim], None);

    let mixed_ba = quantized_matmul(
        &x,
        &ba_q[0],
        &ba_q[1],
        Some(&ba_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let ba = reshape(
        &mixed_ba,
        &[1, seq, num_key_heads, value_heads_per_key * 2],
        None,
    );
    let portable_b = reshape(
        &slice_last_dim(&ba, 0, value_heads_per_key, None),
        &[1, seq, num_value_heads],
        None,
    );
    let portable_a = reshape(
        &slice_last_dim(&ba, value_heads_per_key, value_heads_per_key * 2, None),
        &[1, seq, num_value_heads],
        None,
    );

    let direct_qkv = contiguous(&direct_qkv, None);
    let direct_z = contiguous(&direct_z, None);
    let direct_a = contiguous(&direct_a, None);
    let direct_b = contiguous(&direct_b, None);
    let portable_qkv = contiguous(&portable_qkv, None);
    let portable_z = contiguous(&portable_z, None);
    let portable_a = contiguous(&portable_a, None);
    let portable_b = contiguous(&portable_b, None);
    eval(&[
        &direct_qkv,
        &direct_z,
        &direct_a,
        &direct_b,
        &portable_qkv,
        &portable_z,
        &portable_a,
        &portable_b,
    ]);

    assert_eq!(
        direct_qkv.shape(),
        vec![
            1,
            seq,
            num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim
        ]
    );
    assert_eq!(
        direct_z.shape(),
        vec![1, seq, num_value_heads, value_head_dim]
    );
    assert_eq!(direct_a.shape(), vec![1, seq, num_value_heads]);
    assert_eq!(direct_b.shape(), vec![1, seq, num_value_heads]);
    assert_close_f32(direct_qkv.data_f32(), portable_qkv.data_f32(), 1.0e-6);
    assert_close_f32(direct_z.data_f32(), portable_z.data_f32(), 1.0e-6);
    assert_close_f32(direct_a.data_f32(), portable_a.data_f32(), 1.0e-6);
    assert_close_f32(direct_b.data_f32(), portable_b.data_f32(), 1.0e-6);
}

#[test]
fn qwen_linear_attention_inputs_packed_accepts_mixed_qkvz_ba_bits() {
    let seq = 2_i32;
    let hidden = 64_i32;
    let num_key_heads = 2_i32;
    let num_value_heads = 4_i32;
    let key_head_dim = 4_i32;
    let value_head_dim = 4_i32;
    let value_heads_per_key = num_value_heads / num_key_heads;
    let qkvz_per_key = key_head_dim * 2 + value_heads_per_key * value_head_dim * 2;
    let qkvz_out = num_key_heads * qkvz_per_key;
    let ba_out = num_key_heads * value_heads_per_key * 2;
    let x_data: Vec<f32> = (0..(seq * hidden))
        .map(|i| ((i as f32) - 31.0) * 0.015625)
        .collect();
    let qkvz_weight_data: Vec<f32> = (0..(qkvz_out * hidden))
        .map(|i| ((i as f32) - 200.0) * 0.0005)
        .collect();
    let ba_weight_data: Vec<f32> = (0..(ba_out * hidden))
        .map(|i| ((i as f32) - 80.0) * 0.001)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let qkvz_w = MlxArray::from_raw_data(
        qkvz_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkvz_weight_data.as_slice()),
        &[qkvz_out, hidden],
        MlxDtype::Float32,
    );
    let ba_w = MlxArray::from_raw_data(
        ba_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(ba_weight_data.as_slice()),
        &[ba_out, hidden],
        MlxDtype::Float32,
    );
    let qkvz_q = quantize(
        &qkvz_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let ba_q = quantize(
        &ba_w,
        Some(32),
        Some(6),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let (qkv, z, a, b) = qwen_linear_attention_inputs_packed(
        &x,
        &qkvz_q[0],
        Some(&qkvz_q[1]),
        Some(&qkvz_q[2]),
        &ba_q[0],
        Some(&ba_q[1]),
        Some(&ba_q[2]),
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        32,
        4,
        32,
        6,
        None,
    )
    .expect("mixed 4-bit qkvz + 6-bit ba must pack on prefill seq");
    eval(&[&qkv, &z, &a, &b]);
    assert_eq!(
        qkv.shape(),
        vec![
            1,
            seq,
            num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim
        ]
    );
    assert_eq!(z.shape(), vec![1, seq, num_value_heads, value_head_dim]);
    assert_eq!(a.shape(), vec![1, seq, num_value_heads]);
    assert_eq!(b.shape(), vec![1, seq, num_value_heads]);
}

#[test]
fn qwen_linear_attention_inputs_packed_flat_matches_portable_at_min_seq() {
    // Shipped packed C++ flattens [1,S,H]→[S,H] before the two qmm at
    // seq>=1024. Dual-stream stays default-OFF. Numerics must match the
    // portable two-qmm path.
    let seq = 1024_i32;
    let hidden = 32_i32;
    let num_key_heads = 2_i32;
    let num_value_heads = 4_i32;
    let key_head_dim = 4_i32;
    let value_head_dim = 4_i32;
    let value_heads_per_key = num_value_heads / num_key_heads;
    let qkvz_per_key = key_head_dim * 2 + value_heads_per_key * value_head_dim * 2;
    let qkvz_out = num_key_heads * qkvz_per_key;
    let ba_out = num_key_heads * value_heads_per_key * 2;
    let x_data: Vec<f32> = (0..(seq * hidden))
        .map(|i| ((i as f32) - 31.0) * 0.015625)
        .collect();
    let qkvz_weight_data: Vec<f32> = (0..(qkvz_out * hidden))
        .map(|i| ((i as f32) - 200.0) * 0.0005)
        .collect();
    let ba_weight_data: Vec<f32> = (0..(ba_out * hidden))
        .map(|i| ((i as f32) - 80.0) * 0.001)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let qkvz_w = MlxArray::from_raw_data(
        qkvz_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkvz_weight_data.as_slice()),
        &[qkvz_out, hidden],
        MlxDtype::Float32,
    );
    let ba_w = MlxArray::from_raw_data(
        ba_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(ba_weight_data.as_slice()),
        &[ba_out, hidden],
        MlxDtype::Float32,
    );
    let qkvz_q = quantize(
        &qkvz_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let ba_q = quantize(
        &ba_w,
        Some(32),
        Some(6),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let (d_qkv, d_z, d_a, d_b) = qwen_linear_attention_inputs_packed(
        &x,
        &qkvz_q[0],
        Some(&qkvz_q[1]),
        Some(&qkvz_q[2]),
        &ba_q[0],
        Some(&ba_q[1]),
        Some(&ba_q[2]),
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        32,
        4,
        32,
        6,
        None,
    )
    .expect("dual-stream packed LA must engage at seq>=1024");
    let p_qkvz = quantized_matmul(
        &x,
        &qkvz_q[0],
        &qkvz_q[1],
        Some(&qkvz_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let p_ba = quantized_matmul(
        &x,
        &ba_q[0],
        &ba_q[1],
        Some(&ba_q[2]),
        true,
        Some(32),
        Some(6),
        None,
    );
    eval(&[&d_qkv, &d_z, &d_a, &d_b, &p_qkvz, &p_ba]);
    assert_eq!(d_qkv.shape()[1], seq);
    assert_eq!(d_z.shape(), vec![1, seq, num_value_heads, value_head_dim]);
    assert_eq!(d_a.shape(), vec![1, seq, num_value_heads]);
    assert_eq!(d_b.shape(), vec![1, seq, num_value_heads]);
    assert_eq!(p_qkvz.shape(), vec![1, seq, qkvz_out]);
    assert_eq!(p_ba.shape(), vec![1, seq, ba_out]);
}

#[test]
fn qwen_linear_attention_inputs_packed_compiled_matches_imperative() {
    let seq = 2_i32;
    let hidden = 64_i32;
    let num_key_heads = 2_i32;
    let num_value_heads = 4_i32;
    let key_head_dim = 4_i32;
    let value_head_dim = 4_i32;
    let value_heads_per_key = num_value_heads / num_key_heads;
    let qkvz_per_key = key_head_dim * 2 + value_heads_per_key * value_head_dim * 2;
    let qkvz_out = num_key_heads * qkvz_per_key;
    let ba_out = num_key_heads * value_heads_per_key * 2;
    let x_data: Vec<f32> = (0..(seq * hidden))
        .map(|i| ((i as f32) - 31.0) * 0.015625)
        .collect();
    let qkvz_weight_data: Vec<f32> = (0..(qkvz_out * hidden))
        .map(|i| ((i as f32) - 200.0) * 0.0005)
        .collect();
    let ba_weight_data: Vec<f32> = (0..(ba_out * hidden))
        .map(|i| ((i as f32) - 80.0) * 0.001)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let qkvz_w = MlxArray::from_raw_data(
        qkvz_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkvz_weight_data.as_slice()),
        &[qkvz_out, hidden],
        MlxDtype::Float32,
    );
    let ba_w = MlxArray::from_raw_data(
        ba_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(ba_weight_data.as_slice()),
        &[ba_out, hidden],
        MlxDtype::Float32,
    );
    let qkvz_q = quantize(
        &qkvz_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let ba_q = quantize(
        &ba_w,
        Some(32),
        Some(6),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let (c_qkv, c_z, c_a, c_b) = qwen_linear_attention_inputs_packed_compiled(
        &x,
        &qkvz_q[0],
        &qkvz_q[1],
        Some(&qkvz_q[2]),
        &ba_q[0],
        &ba_q[1],
        Some(&ba_q[2]),
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        32,
        4,
        32,
        6,
        None,
    )
    .expect("compiled packed LA inputs must engage for mixed 4/6-bit");
    let (p_qkv, p_z, p_a, p_b) = qwen_linear_attention_inputs_packed(
        &x,
        &qkvz_q[0],
        Some(&qkvz_q[1]),
        Some(&qkvz_q[2]),
        &ba_q[0],
        Some(&ba_q[1]),
        Some(&ba_q[2]),
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        32,
        4,
        32,
        6,
        None,
    )
    .expect("imperative packed LA inputs");
    eval(&[&c_qkv, &c_z, &c_a, &c_b, &p_qkv, &p_z, &p_a, &p_b]);
    assert_eq!(c_qkv.shape(), p_qkv.shape());
    assert_eq!(c_z.shape(), p_z.shape());
    assert_eq!(c_a.shape(), p_a.shape());
    assert_eq!(c_b.shape(), p_b.shape());
    assert_close_f32(c_qkv.data_f32(), p_qkv.data_f32(), 1.0e-5);
    assert_close_f32(c_z.data_f32(), p_z.data_f32(), 1.0e-5);
    assert_close_f32(c_a.data_f32(), p_a.data_f32(), 1.0e-5);
    assert_close_f32(c_b.data_f32(), p_b.data_f32(), 1.0e-5);
}

#[test]
fn qwen_linear_attention_inputs_packed_compiled_mxfp4_matches_imperative() {
    let seq = 2_i32;
    let hidden = 64_i32;
    let num_key_heads = 2_i32;
    let num_value_heads = 4_i32;
    let key_head_dim = 4_i32;
    let value_head_dim = 4_i32;
    let value_heads_per_key = num_value_heads / num_key_heads;
    let qkvz_per_key = key_head_dim * 2 + value_heads_per_key * value_head_dim * 2;
    let qkvz_out = num_key_heads * qkvz_per_key;
    let ba_out = num_key_heads * value_heads_per_key * 2;
    let x_data: Vec<f32> = (0..(seq * hidden))
        .map(|i| ((i as f32) - 31.0) * 0.015625)
        .collect();
    let qkvz_weight_data: Vec<f32> = (0..(qkvz_out * hidden))
        .map(|i| ((i as f32) - 200.0) * 0.0005)
        .collect();
    let ba_weight_data: Vec<f32> = (0..(ba_out * hidden))
        .map(|i| ((i as f32) - 80.0) * 0.001)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let qkvz_w = MlxArray::from_raw_data(
        qkvz_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkvz_weight_data.as_slice()),
        &[qkvz_out, hidden],
        MlxDtype::Float32,
    );
    let ba_w = MlxArray::from_raw_data(
        ba_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(ba_weight_data.as_slice()),
        &[ba_out, hidden],
        MlxDtype::Float32,
    );
    let qkvz_q = quantize(
        &qkvz_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    let ba_q = quantize(
        &ba_w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    assert_eq!(qkvz_q.len(), 2, "MXFP4 quantize returns weight + scales");
    assert_eq!(ba_q.len(), 2);
    let (c_qkv, c_z, c_a, c_b) = qwen_linear_attention_inputs_packed_compiled(
        &x,
        &qkvz_q[0],
        &qkvz_q[1],
        None,
        &ba_q[0],
        &ba_q[1],
        None,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        32,
        4,
        32,
        4,
        None,
    )
    .expect("compiled packed LA inputs must engage for scales-only MXFP4");
    let (p_qkv, p_z, p_a, p_b) = qwen_linear_attention_inputs_packed(
        &x,
        &qkvz_q[0],
        Some(&qkvz_q[1]),
        None,
        &ba_q[0],
        Some(&ba_q[1]),
        None,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        32,
        4,
        32,
        4,
        None,
    )
    .expect("imperative packed LA inputs must host scales-only MXFP4");
    eval(&[&c_qkv, &c_z, &c_a, &c_b, &p_qkv, &p_z, &p_a, &p_b]);
    assert_eq!(c_qkv.shape(), p_qkv.shape());
    assert_eq!(c_z.shape(), p_z.shape());
    assert_eq!(c_a.shape(), p_a.shape());
    assert_eq!(c_b.shape(), p_b.shape());
    assert_close_f32(c_qkv.data_f32(), p_qkv.data_f32(), 1.0e-5);
    assert_close_f32(c_z.data_f32(), p_z.data_f32(), 1.0e-5);
    assert_close_f32(c_a.data_f32(), p_a.data_f32(), 1.0e-5);
    assert_close_f32(c_b.data_f32(), p_b.data_f32(), 1.0e-5);
}

#[test]
fn qwen_linear_attention_post_input_direct_matches_portable_composition() {
    let batch = 1_i32;
    let seq = 2_i32;
    let num_key_heads = 2_i32;
    let key_head_dim = 4_i32;
    let num_value_heads = 4_i32;
    let value_head_dim = 3_i32;
    let conv_kernel_dim = 4_i32;
    let tail_len = conv_kernel_dim - 1;
    let key_dim = num_key_heads * key_head_dim;
    let value_dim = num_value_heads * value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let (q_scale, k_scale) = (1.0_f32 / (key_head_dim as f32).sqrt(), 0.5_f32);
    let eps = 1.0e-6_f32;

    let qkv_data: Vec<f32> = (0..(batch * seq * conv_dim))
        .map(|i| ((i as f32) - 16.0) * 0.0625)
        .collect();
    let conv_weight_data: Vec<f32> = (0..(conv_dim * conv_kernel_dim))
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let cached_state_data: Vec<f32> = (0..(batch * tail_len * conv_dim))
        .map(|i| ((i as f32) - 8.0) * 0.03125)
        .collect();
    let qkv = MlxArray::from_raw_data(
        qkv_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkv_data.as_slice()),
        &[batch, seq, conv_dim],
        MlxDtype::Float32,
    );
    // conv1d weight layout is [out_channels, kernel_w, in_channels_per_group].
    // Depthwise (`groups = conv_dim`) → in_channels_per_group = 1.
    let conv_weight = MlxArray::from_raw_data(
        conv_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(conv_weight_data.as_slice()),
        &[conv_dim, conv_kernel_dim, 1],
        MlxDtype::Float32,
    );
    let cached_state = MlxArray::from_raw_data(
        cached_state_data.as_ptr() as *const u8,
        std::mem::size_of_val(cached_state_data.as_slice()),
        &[batch, tail_len, conv_dim],
        MlxDtype::Float32,
    );

    let prev = crate::op_count::op_count_snapshot();
    let (direct_q, direct_k, direct_v, direct_state) = qwen_linear_attention_post_input(
        &qkv,
        &conv_weight,
        Some(&cached_state),
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        conv_kernel_dim,
        q_scale,
        k_scale,
        eps,
        None,
    )
    .expect("post-input shim should accept qwen-compatible shapes");
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ post-input shim should count as one Rust FFI dispatch"
    );

    // Portable composition: matches `linear_attention_conv1d` +
    // `split_linear_attention_qkv` + `normalize_linear_attention_qk` in
    // `crates/ax-engine-mlx/src/linear_attention_ops.rs`.
    let conv_input = concatenate(&[&cached_state, &qkv], 1, None);
    let total = conv_input.shape()[1];
    let portable_state = slice(
        &conv_input,
        &[0, total - tail_len, 0],
        &[batch, total, conv_dim],
        &[1, 1, 1],
        None,
    );
    let conv_out = conv1d(&conv_input, &conv_weight, 1, 0, 1, conv_dim, None);
    let silued = multiply(&conv_out, &sigmoid(&conv_out, None), None);
    let q_flat = slice_last_dim(&silued, 0, key_dim, None);
    let k_flat = slice_last_dim(&silued, key_dim, 2 * key_dim, None);
    let v_flat = slice_last_dim(&silued, 2 * key_dim, 2 * key_dim + value_dim, None);
    let q_heads = reshape(&q_flat, &[batch, seq, num_key_heads, key_head_dim], None);
    let k_heads = reshape(&k_flat, &[batch, seq, num_key_heads, key_head_dim], None);
    let v_heads = reshape(
        &v_flat,
        &[batch, seq, num_value_heads, value_head_dim],
        None,
    );
    let q_normed = crate::fast::rms_norm(&q_heads, None, eps, None);
    let k_normed = crate::fast::rms_norm(&k_heads, None, eps, None);
    let q_scale_arr = MlxArray::from_raw_data(
        &q_scale as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1],
        MlxDtype::Float32,
    );
    let k_scale_arr = MlxArray::from_raw_data(
        &k_scale as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1],
        MlxDtype::Float32,
    );
    let portable_q = multiply(&q_normed, &q_scale_arr, None);
    let portable_k = multiply(&k_normed, &k_scale_arr, None);
    let portable_v = v_heads;

    eval(&[
        &direct_q,
        &direct_k,
        &direct_v,
        &direct_state,
        &portable_q,
        &portable_k,
        &portable_v,
        &portable_state,
    ]);

    assert_eq!(
        direct_q.shape(),
        vec![batch, seq, num_key_heads, key_head_dim]
    );
    assert_eq!(
        direct_k.shape(),
        vec![batch, seq, num_key_heads, key_head_dim]
    );
    assert_eq!(
        direct_v.shape(),
        vec![batch, seq, num_value_heads, value_head_dim]
    );
    assert_eq!(direct_state.shape(), vec![batch, tail_len, conv_dim]);
    // Raw data reads require dense row-major layout; some outputs are
    // strided views (e.g. the V split), so materialize each side and
    // compare logical values rather than raw buffer prefixes.
    let dense = |arr: &MlxArray| {
        let materialized = contiguous(arr, None);
        eval(&[&materialized]);
        materialized
    };
    assert_close_f32(
        dense(&direct_q).data_f32(),
        dense(&portable_q).data_f32(),
        1.0e-6,
    );
    assert_close_f32(
        dense(&direct_k).data_f32(),
        dense(&portable_k).data_f32(),
        1.0e-6,
    );
    assert_close_f32(
        dense(&direct_v).data_f32(),
        dense(&portable_v).data_f32(),
        1.0e-6,
    );
    assert_close_f32(
        dense(&direct_state).data_f32(),
        dense(&portable_state).data_f32(),
        1.0e-6,
    );
}

#[test]
fn qwen_linear_attention_post_input_compiled_matches_imperative() {
    let batch = 1_i32;
    let seq = 2_i32;
    let num_key_heads = 2_i32;
    let key_head_dim = 4_i32;
    let num_value_heads = 4_i32;
    let value_head_dim = 3_i32;
    let conv_kernel_dim = 4_i32;
    let tail_len = conv_kernel_dim - 1;
    let key_dim = num_key_heads * key_head_dim;
    let value_dim = num_value_heads * value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let (q_scale, k_scale) = (1.0_f32 / (key_head_dim as f32).sqrt(), 0.5_f32);
    let eps = 1.0e-6_f32;
    let qkv_data: Vec<f32> = (0..(batch * seq * conv_dim))
        .map(|i| ((i as f32) - 16.0) * 0.0625)
        .collect();
    let conv_weight_data: Vec<f32> = (0..(conv_dim * conv_kernel_dim))
        .map(|i| ((i as f32) - 32.0) * 0.015625)
        .collect();
    let cached_state_data: Vec<f32> = (0..(batch * tail_len * conv_dim))
        .map(|i| ((i as f32) - 8.0) * 0.03125)
        .collect();
    let qkv = MlxArray::from_raw_data(
        qkv_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkv_data.as_slice()),
        &[batch, seq, conv_dim],
        MlxDtype::Float32,
    );
    let conv_weight = MlxArray::from_raw_data(
        conv_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(conv_weight_data.as_slice()),
        &[conv_dim, conv_kernel_dim, 1],
        MlxDtype::Float32,
    );
    let cached_state = MlxArray::from_raw_data(
        cached_state_data.as_ptr() as *const u8,
        std::mem::size_of_val(cached_state_data.as_slice()),
        &[batch, tail_len, conv_dim],
        MlxDtype::Float32,
    );
    let (c_q, c_k, c_v, c_state) = qwen_linear_attention_post_input_compiled(
        &qkv,
        &conv_weight,
        &cached_state,
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        conv_kernel_dim,
        q_scale,
        k_scale,
        eps,
        None,
    )
    .expect("compiled post-input must engage with explicit conv state");
    let (p_q, p_k, p_v, p_state) = qwen_linear_attention_post_input(
        &qkv,
        &conv_weight,
        Some(&cached_state),
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        conv_kernel_dim,
        q_scale,
        k_scale,
        eps,
        None,
    )
    .expect("imperative post-input");
    eval(&[&c_q, &c_k, &c_v, &c_state, &p_q, &p_k, &p_v, &p_state]);
    assert_eq!(c_q.shape(), p_q.shape());
    assert_eq!(c_k.shape(), p_k.shape());
    assert_eq!(c_v.shape(), p_v.shape());
    assert_eq!(c_state.shape(), p_state.shape());
    let dense = |arr: &MlxArray| {
        let materialized = contiguous(arr, None);
        eval(&[&materialized]);
        materialized
    };
    assert_close_f32(dense(&c_q).data_f32(), dense(&p_q).data_f32(), 1.0e-5);
    assert_close_f32(dense(&c_k).data_f32(), dense(&p_k).data_f32(), 1.0e-5);
    assert_close_f32(dense(&c_v).data_f32(), dense(&p_v).data_f32(), 1.0e-5);
    assert_close_f32(
        dense(&c_state).data_f32(),
        dense(&p_state).data_f32(),
        1.0e-5,
    );
}

#[test]
fn qwen_linear_attention_post_input_direct_handles_empty_cached_state() {
    // First decode step in a fresh sequence: cached_conv_state is None,
    // shim must materialise zeros internally and still return the right
    // tail slice.
    let batch = 1_i32;
    let seq = 1_i32;
    let num_key_heads = 1_i32;
    let key_head_dim = 32_i32; // gated_delta kernel requires Dk % 32 == 0
    let num_value_heads = 1_i32;
    let value_head_dim = 4_i32;
    let conv_kernel_dim = 4_i32;
    let key_dim = num_key_heads * key_head_dim;
    let value_dim = num_value_heads * value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;

    let qkv_data: Vec<f32> = (0..(batch * seq * conv_dim))
        .map(|i| i as f32 * 0.01)
        .collect();
    let conv_weight_data: Vec<f32> = (0..(conv_dim * conv_kernel_dim))
        .map(|i| ((i as f32) - 16.0) * 0.0078125)
        .collect();
    let qkv = MlxArray::from_raw_data(
        qkv_data.as_ptr() as *const u8,
        std::mem::size_of_val(qkv_data.as_slice()),
        &[batch, seq, conv_dim],
        MlxDtype::Float32,
    );
    let conv_weight = MlxArray::from_raw_data(
        conv_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(conv_weight_data.as_slice()),
        &[conv_dim, conv_kernel_dim, 1],
        MlxDtype::Float32,
    );

    let result = qwen_linear_attention_post_input(
        &qkv,
        &conv_weight,
        None,
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        conv_kernel_dim,
        1.0,
        1.0,
        1.0e-6,
        None,
    );
    assert!(
        result.is_some(),
        "shim must accept None cached_conv_state by materialising zeros internally"
    );
    let (_, _, _, new_state) = result.unwrap();
    assert_eq!(
        new_state.shape(),
        vec![batch, conv_kernel_dim - 1, conv_dim]
    );
}

#[test]
fn gelu_approx_mul_quantized_matmul_direct_matches_portable_composition() {
    let gate_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.03125).collect();
    let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) + 1.0) * 0.015625).collect();
    let weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.0005).collect();
    let gate = MlxArray::from_raw_data(
        gate_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_data.as_slice()),
        &[2, 32],
        MlxDtype::Float32,
    );
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 32],
        MlxDtype::Float32,
    );
    let weight = MlxArray::from_raw_data(
        weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(weight_data.as_slice()),
        &[32, 32],
        MlxDtype::Float32,
    );
    let quantized = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(quantized.len(), 3);

    let prev = crate::op_count::op_count_snapshot();
    let direct = gelu_approx_mul_quantized_matmul(
        &gate,
        &x,
        &quantized[0],
        &quantized[1],
        Some(&quantized[2]),
        32,
        4,
        None,
    );
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ activation+quantized-matmul shim should count as one Rust FFI dispatch"
    );

    let hidden = gelu_approx_mul(&gate, &x, None);
    let portable = quantized_matmul(
        &hidden,
        &quantized[0],
        &quantized[1],
        Some(&quantized[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&direct, &portable]);

    assert_eq!(direct.shape(), vec![2, 32]);
    assert_eq!(
        direct.data_f32().to_vec(),
        portable.data_f32().to_vec(),
        "direct C++ activation+quantized-matmul shim must preserve portable math"
    );
}

#[test]
fn silu_mul_quantized_matmul_matches_portable_4bit_gs32() {
    // AXQ 27B down_proj is 4-bit gs32. Fuse must match silu_mul + qmm.
    let gate_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
    let up_data: Vec<f32> = (0..512).map(|i| ((i as f32) + 1.0) * 0.0078125).collect();
    let weight_data: Vec<f32> = (0..4096).map(|i| ((i as f32) - 2048.0) * 0.00025).collect();
    let gate = MlxArray::from_raw_data(
        gate_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_data.as_slice()),
        &[1, 8, 64],
        MlxDtype::Float32,
    );
    let up = MlxArray::from_raw_data(
        up_data.as_ptr() as *const u8,
        std::mem::size_of_val(up_data.as_slice()),
        &[1, 8, 64],
        MlxDtype::Float32,
    );
    let weight = MlxArray::from_raw_data(
        weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(weight_data.as_slice()),
        &[64, 64],
        MlxDtype::Float32,
    );
    let q = quantize(
        &weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let fused = silu_mul_quantized_matmul(&gate, &up, &q[0], &q[1], Some(&q[2]), 32, 4, None)
        .expect("silu+down qmm fuse should engage for 4-bit gs32");
    let hidden = silu_mul(&gate, &up, None);
    let portable = quantized_matmul(
        &hidden,
        &q[0],
        &q[1],
        Some(&q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&fused, &portable]);
    assert_eq!(fused.shape(), portable.shape());
    for (a, b) in fused.data_f32().iter().zip(portable.data_f32().iter()) {
        assert!(
            (a - b).abs() < 2e-2 || (a - b).abs() / (b.abs().max(1e-6)) < 2e-2,
            "silu+down fuse must match portable: {a} vs {b}"
        );
    }
}

#[test]
fn dual_qmm_swiglu_matches_two_qmm_silu_mul_4bit_gs32() {
    // AXQ 27B language FFN is 4-bit gs32. One C++ call must match two
    // affine qmm + silu_mul on a multi-token prefill shape.
    let x_data: Vec<f32> = (0..8 * 64)
        .map(|i| ((i as f32) - 256.0) * 0.0009765625)
        .collect();
    let gate_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 1024.0) * 0.0005)
        .collect();
    let up_data: Vec<f32> = (0..32 * 64)
        .map(|i| ((i as f32) - 512.0) * -0.0004)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, 8, 64],
        MlxDtype::Float32,
    );
    let gate_w = MlxArray::from_raw_data(
        gate_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_data.as_slice()),
        &[32, 64],
        MlxDtype::Float32,
    );
    let up_w = MlxArray::from_raw_data(
        up_data.as_ptr() as *const u8,
        std::mem::size_of_val(up_data.as_slice()),
        &[32, 64],
        MlxDtype::Float32,
    );
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
    let fused = dual_qmm_swiglu(
        &x, &gq[0], &gq[1], &gq[2], &uq[0], &uq[1], &uq[2], 32, 4, None,
    )
    .expect("dual qmm + SwiGLU should engage for 4-bit gs32");
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
    for (a, b) in fused.data_f32().iter().zip(portable.data_f32().iter()) {
        assert!(
            (a - b).abs() < 2e-2 || (a - b).abs() / (b.abs().max(1e-6)) < 2e-2,
            "dual qmm + SwiGLU must match portable: {a} vs {b}"
        );
    }
    assert!(
        dual_qmm_swiglu(
            &x, &gq[0], &gq[1], &gq[2], &uq[0], &uq[1], &uq[2], 0, 4, None,
        )
        .is_none(),
        "dual qmm + SwiGLU must reject group_size<=0"
    );
}

fn compiled_split_mlp_matches_portable(bits: i32) {
    // x [2,4,64]; weight [out=64, in=64] so group_size=64 divides last dim.
    // bits=4 → shapeless (#680); bits=8 multi-token → shape-specific (#705).
    let x_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
    let w_data: Vec<f32> = (0..4096).map(|i| ((i as f32) - 2048.0) * 0.00025).collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 4, 64],
        MlxDtype::Float32,
    );
    let w = MlxArray::from_raw_data(
        w_data.as_ptr() as *const u8,
        std::mem::size_of_val(w_data.as_slice()),
        &[64, 64],
        MlxDtype::Float32,
    );
    let q = quantize(
        &w,
        Some(64),
        Some(bits),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let compiled = compiled_gelu_approx_split_mlp(
        &x, &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], 64, bits, None,
    )
    .unwrap_or_else(|| panic!("gs64/bits={bits} multi-token split MLP compile should engage"));
    let gate = quantized_matmul(
        &x,
        &q[0],
        &q[1],
        Some(&q[2]),
        true,
        Some(64),
        Some(bits),
        None,
    );
    let up = quantized_matmul(
        &x,
        &q[0],
        &q[1],
        Some(&q[2]),
        true,
        Some(64),
        Some(bits),
        None,
    );
    let hidden = gelu_approx_mul(&gate, &up, None);
    let portable = quantized_matmul(
        &hidden,
        &q[0],
        &q[1],
        Some(&q[2]),
        true,
        Some(64),
        Some(bits),
        None,
    );
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    let c = compiled.data_f32().to_vec();
    let p = portable.data_f32().to_vec();
    assert_eq!(c.len(), p.len());
    for (a, b) in c.iter().zip(p.iter()) {
        assert!(
            (a - b).abs() < 1e-3 || (a - b).abs() / (b.abs().max(1e-6)) < 1e-3,
            "compiled split MLP must match portable within tol: {a} vs {b}"
        );
    }
}

#[test]
fn compiled_dual_gate_up_qmm_matches_portable_multi_token() {
    // Profile residual: multi-token dual gate/up compile vs two qmm.
    // Path is default-off after pure wall reject; enable for parity test.
    unsafe {
        std::env::set_var("AX_MLX_COMPILED_DUAL_GATE_UP", "1");
    }
    let x_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
    let w_data: Vec<f32> = (0..4096).map(|i| ((i as f32) - 2048.0) * 0.00025).collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 4, 64],
        MlxDtype::Float32,
    );
    let w = MlxArray::from_raw_data(
        w_data.as_ptr() as *const u8,
        std::mem::size_of_val(w_data.as_slice()),
        &[64, 64],
        MlxDtype::Float32,
    );
    let q = quantize(
        &w,
        Some(64),
        Some(8),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let (c_gate, c_up) =
        compiled_dual_gate_up_qmm(&x, &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], 64, 8, None)
            .expect("multi-token dual gate/up compile should engage for bits=8");
    let p_gate = quantized_matmul(&x, &q[0], &q[1], Some(&q[2]), true, Some(64), Some(8), None);
    let p_up = quantized_matmul(&x, &q[0], &q[1], Some(&q[2]), true, Some(64), Some(8), None);
    eval(&[&c_gate, &c_up, &p_gate, &p_up]);
    assert_eq!(c_gate.shape(), p_gate.shape());
    assert_eq!(c_up.shape(), p_up.shape());
    for (a, b) in c_gate.data_f32().iter().zip(p_gate.data_f32().iter()) {
        assert!(
            (a - b).abs() < 1e-3 || (a - b).abs() / (b.abs().max(1e-6)) < 1e-3,
            "compiled dual gate must match portable: {a} vs {b}"
        );
    }
    for (a, b) in c_up.data_f32().iter().zip(p_up.data_f32().iter()) {
        assert!(
            (a - b).abs() < 1e-3 || (a - b).abs() / (b.abs().max(1e-6)) < 1e-3,
            "compiled dual up must match portable: {a} vs {b}"
        );
    }
}

#[test]
fn compiled_dual_gate_up_qmm_forced_matches_portable_4bit_gs32() {
    // AXQ 27B language FFN is 4-bit gs32. Forced compile must match two
    // affine qmms without AX_MLX_COMPILED_DUAL_GATE_UP (Gemma stays off).
    let x_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
    let gate_data: Vec<f32> = (0..4096).map(|i| ((i as f32) - 2048.0) * 0.00025).collect();
    let up_data: Vec<f32> = (0..4096).map(|i| ((i as f32) - 1024.0) * -0.0002).collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, 8, 64],
        MlxDtype::Float32,
    );
    let gate_w = MlxArray::from_raw_data(
        gate_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_data.as_slice()),
        &[64, 64],
        MlxDtype::Float32,
    );
    let up_w = MlxArray::from_raw_data(
        up_data.as_ptr() as *const u8,
        std::mem::size_of_val(up_data.as_slice()),
        &[64, 64],
        MlxDtype::Float32,
    );
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
    let (c_gate, c_up) = compiled_dual_gate_up_qmm_forced(
        &x, &gq[0], &gq[1], &gq[2], &uq[0], &uq[1], &uq[2], 32, 4, None,
    )
    .expect("forced 4-bit gs32 dual gate/up compile should engage without env");
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
    eval(&[&c_gate, &c_up, &p_gate, &p_up]);
    assert_eq!(c_gate.shape(), p_gate.shape());
    assert_eq!(c_up.shape(), p_up.shape());
    for (a, b) in c_gate.data_f32().iter().zip(p_gate.data_f32().iter()) {
        assert!(
            (a - b).abs() < 2e-2 || (a - b).abs() / (b.abs().max(1e-6)) < 2e-2,
            "forced compiled dual gate must match portable 4-bit: {a} vs {b}"
        );
    }
    for (a, b) in c_up.data_f32().iter().zip(p_up.data_f32().iter()) {
        assert!(
            (a - b).abs() < 2e-2 || (a - b).abs() / (b.abs().max(1e-6)) < 2e-2,
            "forced compiled dual up must match portable 4-bit: {a} vs {b}"
        );
    }
    // Env-gated entry stays off unless AX_MLX_COMPILED_DUAL_GATE_UP=1.
    unsafe {
        std::env::remove_var("AX_MLX_COMPILED_DUAL_GATE_UP");
    }
    assert!(
        compiled_dual_gate_up_qmm(
            &x, &gq[0], &gq[1], &gq[2], &uq[0], &uq[1], &uq[2], 32, 4, None
        )
        .is_none(),
        "Gemma env-gated compile must stay default-OFF"
    );
}

#[test]
fn compiled_gelu_approx_split_mlp_matches_portable_split_ffn() {
    compiled_split_mlp_matches_portable(4);
}

#[test]
fn compiled_gelu_approx_split_mlp_matches_portable_bits8_multitoken() {
    // Flip Gemma MLP is bits=8; multi-token shape-specific path is opt-in
    // (default OFF after mbp-m5 pure wall +2% reject).
    // SAFETY: test process only; restored after assertion path.
    // SAFETY: single-threaded test env mutation for kill-switch coverage.
    unsafe {
        std::env::set_var("AX_MLX_COMPILED_QGELU_PREFILL_SHAPED", "1");
    }
    compiled_split_mlp_matches_portable(8);
    unsafe {
        std::env::remove_var("AX_MLX_COMPILED_QGELU_PREFILL_SHAPED");
    }
}

#[test]
fn compiled_gelu_approx_split_mlp_bits8_multitoken_default_off() {
    unsafe {
        std::env::remove_var("AX_MLX_COMPILED_QGELU_PREFILL_SHAPED");
    }
    let x_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
    let w_data: Vec<f32> = (0..4096).map(|i| ((i as f32) - 2048.0) * 0.00025).collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 4, 64],
        MlxDtype::Float32,
    );
    let w = MlxArray::from_raw_data(
        w_data.as_ptr() as *const u8,
        std::mem::size_of_val(w_data.as_slice()),
        &[64, 64],
        MlxDtype::Float32,
    );
    let q = quantize(
        &w,
        Some(64),
        Some(8),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let compiled = compiled_gelu_approx_split_mlp(
        &x, &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], 64, 8, None,
    );
    assert!(
        compiled.is_none(),
        "bits=8 multi-token must stay portable when PREFILL_SHAPED is unset"
    );
}

#[test]
fn compiled_gelu_approx_split_mlp_axq_p128_gs32_unset_stays_portable() {
    unsafe {
        std::env::remove_var("AX_MLX_COMPILED_QGELU_AXQ_P128");
        std::env::remove_var("AX_MLX_COMPILED_QGELU_PREFILL_SHAPED");
    }
    let x_data: Vec<f32> = (0..128 * 64)
        .map(|i| ((i as f32) - 4096.0) * 0.015625)
        .collect();
    let w_data: Vec<f32> = (0..64 * 64)
        .map(|i| ((i as f32) - 2048.0) * 0.00025)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, 128, 64],
        MlxDtype::Float32,
    );
    let w = MlxArray::from_raw_data(
        w_data.as_ptr() as *const u8,
        std::mem::size_of_val(w_data.as_slice()),
        &[64, 64],
        MlxDtype::Float32,
    );
    let q = quantize(
        &w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let compiled = compiled_gelu_approx_split_mlp(
        &x, &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], 32, 4, None,
    );
    assert!(
        compiled.is_none(),
        "AXQ gs32/bits=4 seq=128 must stay portable when COMPILED_QGELU_AXQ_P128 is unset"
    );
}

#[test]
fn compiled_gelu_approx_split_mlp_axq_p128_gs32_matches_portable() {
    // AXQ root 4-bit is gs=32. Contract p128 is the only seq that takes
    // the shape-specific compile; portable dual qmm is the oracle.
    unsafe {
        std::env::set_var("AX_MLX_COMPILED_QGELU_AXQ_P128", "1");
        std::env::remove_var("AX_MLX_COMPILED_QGELU_PREFILL_SHAPED");
    }
    let hidden = 64;
    let seq = 128;
    let x_data: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) - 4096.0) * 0.015625)
        .collect();
    let w_data: Vec<f32> = (0..hidden * hidden)
        .map(|i| ((i as f32) - 2048.0) * 0.00025)
        .collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[1, seq, hidden],
        MlxDtype::Float32,
    );
    let w = MlxArray::from_raw_data(
        w_data.as_ptr() as *const u8,
        std::mem::size_of_val(w_data.as_slice()),
        &[hidden, hidden],
        MlxDtype::Float32,
    );
    let q = quantize(
        &w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let compiled = compiled_gelu_approx_split_mlp(
        &x, &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], 32, 4, None,
    )
    .expect("AXQ gs32/bits=4 seq=128 split MLP compile should engage");
    let gate = quantized_matmul(&x, &q[0], &q[1], Some(&q[2]), true, Some(32), Some(4), None);
    let up = quantized_matmul(&x, &q[0], &q[1], Some(&q[2]), true, Some(32), Some(4), None);
    let activated = gelu_approx_mul(&gate, &up, None);
    let portable = quantized_matmul(
        &activated,
        &q[0],
        &q[1],
        Some(&q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&compiled, &portable]);
    assert_eq!(compiled.shape(), portable.shape());
    let c = compiled.data_f32();
    let p = portable.data_f32();
    assert_eq!(c.len(), p.len());
    for (a, b) in c.iter().zip(p.iter()) {
        assert!(
            (a - b).abs() < 3e-2 || (a - b).abs() / (b.abs().max(1e-6)) < 3e-2,
            "AXQ p128 compiled split MLP must match portable: {a} vs {b}"
        );
    }
    unsafe {
        std::env::remove_var("AX_MLX_COMPILED_QGELU_AXQ_P128");
    }
}

#[test]
fn compiled_gelu_approx_split_mlp_axq_gs32_non_p128_stays_portable() {
    unsafe {
        std::env::remove_var("AX_MLX_COMPILED_QGELU_AXQ_P128");
        std::env::remove_var("AX_MLX_COMPILED_QGELU_PREFILL_SHAPED");
    }
    let x_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
    let w_data: Vec<f32> = (0..4096).map(|i| ((i as f32) - 2048.0) * 0.00025).collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 4, 64],
        MlxDtype::Float32,
    );
    let w = MlxArray::from_raw_data(
        w_data.as_ptr() as *const u8,
        std::mem::size_of_val(w_data.as_slice()),
        &[64, 64],
        MlxDtype::Float32,
    );
    let q = quantize(
        &w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(q.len(), 3);
    let compiled = compiled_gelu_approx_split_mlp(
        &x, &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], 32, 4, None,
    );
    assert!(
        compiled.is_none(),
        "AXQ gs32/bits=4 must stay portable off contract p128"
    );
}

#[test]
fn gelu_approx_quantized_ffn_direct_matches_portable_composition() {
    let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.03125).collect();
    let gate_up_weight_data: Vec<f32> = (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0005).collect();
    let down_weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.0005).collect();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr() as *const u8,
        std::mem::size_of_val(x_data.as_slice()),
        &[2, 32],
        MlxDtype::Float32,
    );
    let gate_up_weight = MlxArray::from_raw_data(
        gate_up_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_up_weight_data.as_slice()),
        &[64, 32],
        MlxDtype::Float32,
    );
    let down_weight = MlxArray::from_raw_data(
        down_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(down_weight_data.as_slice()),
        &[32, 32],
        MlxDtype::Float32,
    );
    let gate_up_q = quantize(
        &gate_up_weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down_q = quantize(
        &down_weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(gate_up_q.len(), 3);
    assert_eq!(down_q.len(), 3);

    let prev = crate::op_count::op_count_snapshot();
    let direct = gelu_approx_quantized_ffn(
        &x,
        &gate_up_q[0],
        &gate_up_q[1],
        Some(&gate_up_q[2]),
        &down_q[0],
        &down_q[1],
        Some(&down_q[2]),
        32,
        4,
        None,
    );
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ quantized FFN shim should count as one Rust FFI dispatch"
    );

    let gate_up = quantized_matmul(
        &x,
        &gate_up_q[0],
        &gate_up_q[1],
        Some(&gate_up_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let gate = slice_last_dim(&gate_up, 0, 32, None);
    let up = slice_last_dim(&gate_up, 32, 64, None);
    let hidden = gelu_approx_mul(&gate, &up, None);
    let portable = quantized_matmul(
        &hidden,
        &down_q[0],
        &down_q[1],
        Some(&down_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    eval(&[&direct, &portable]);

    assert_eq!(direct.shape(), vec![2, 32]);
    assert_eq!(
        direct.data_f32().to_vec(),
        portable.data_f32().to_vec(),
        "direct C++ quantized FFN shim must preserve portable math"
    );
}

#[test]
fn gemma4_post_attn_ffn_block_direct_matches_portable_composition() {
    let hidden_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.03125).collect();
    let attn_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 18.0) * 0.015625).collect();
    let norm_data: Vec<f32> = (0..32).map(|i| 0.75 + (i as f32) * 0.0078125).collect();
    let post_norm_data: Vec<f32> = (0..32).map(|i| 0.875 + (i as f32) * 0.00390625).collect();
    let layer_scalar_data = vec![0.9375_f32];
    let gate_up_weight_data: Vec<f32> = (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0005).collect();
    let down_weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.0005).collect();
    let hidden = MlxArray::from_raw_data(
        hidden_data.as_ptr() as *const u8,
        std::mem::size_of_val(hidden_data.as_slice()),
        &[1, 2, 32],
        MlxDtype::Float32,
    );
    let attn = MlxArray::from_raw_data(
        attn_data.as_ptr() as *const u8,
        std::mem::size_of_val(attn_data.as_slice()),
        &[1, 2, 32],
        MlxDtype::Float32,
    );
    let norm = MlxArray::from_raw_data(
        norm_data.as_ptr() as *const u8,
        std::mem::size_of_val(norm_data.as_slice()),
        &[32],
        MlxDtype::Float32,
    );
    let post_norm = MlxArray::from_raw_data(
        post_norm_data.as_ptr() as *const u8,
        std::mem::size_of_val(post_norm_data.as_slice()),
        &[32],
        MlxDtype::Float32,
    );
    let layer_scalar = MlxArray::from_raw_data(
        layer_scalar_data.as_ptr() as *const u8,
        std::mem::size_of_val(layer_scalar_data.as_slice()),
        &[1],
        MlxDtype::Float32,
    );
    let gate_up_weight = MlxArray::from_raw_data(
        gate_up_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(gate_up_weight_data.as_slice()),
        &[64, 32],
        MlxDtype::Float32,
    );
    let down_weight = MlxArray::from_raw_data(
        down_weight_data.as_ptr() as *const u8,
        std::mem::size_of_val(down_weight_data.as_slice()),
        &[32, 32],
        MlxDtype::Float32,
    );
    let gate_up_q = quantize(
        &gate_up_weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    let down_q = quantize(
        &down_weight,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );

    let prev = crate::op_count::op_count_snapshot();
    let direct = gemma4_post_attn_ffn_block(
        &hidden,
        &attn,
        &norm,
        Some(&post_norm),
        Some(&layer_scalar),
        &gate_up_q[0],
        &gate_up_q[1],
        Some(&gate_up_q[2]),
        &down_q[0],
        &down_q[1],
        Some(&down_q[2]),
        32,
        4,
        1.0e-6,
        None,
    );
    assert_eq!(
        crate::op_count::op_count_take(prev),
        1,
        "direct C++ Gemma4 post-attention FFN block should count as one Rust FFI dispatch"
    );

    let residual = add(&hidden, &attn, None);
    let normed = crate::fast::rms_norm(&residual, Some(&norm), 1.0e-6, None);
    let gate_up = quantized_matmul(
        &normed,
        &gate_up_q[0],
        &gate_up_q[1],
        Some(&gate_up_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let gate = slice_last_dim(&gate_up, 0, 32, None);
    let up = slice_last_dim(&gate_up, 32, 64, None);
    let ffn_hidden = gelu_approx_mul(&gate, &up, None);
    let ffn = quantized_matmul(
        &ffn_hidden,
        &down_q[0],
        &down_q[1],
        Some(&down_q[2]),
        true,
        Some(32),
        Some(4),
        None,
    );
    let ffn = crate::fast::rms_norm(&ffn, Some(&post_norm), 1.0e-6, None);
    let portable = multiply(&add(&residual, &ffn, None), &layer_scalar, None);
    eval(&[&direct, &portable]);

    assert_eq!(direct.shape(), vec![1, 2, 32]);
    assert_close_f32(direct.data_f32(), portable.data_f32(), 1.0e-6);
}

#[test]
fn quantize_affine_round_trip_reports_expected_shapes() {
    let values = (0..128)
        .map(|i| (i as f32 - 64.0) / 64.0)
        .collect::<Vec<_>>();
    let w = MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values.as_slice()),
        &[2, 64],
        MlxDtype::Float32,
    );

    let quantized = quantize(
        &w,
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    );

    assert_eq!(quantized.len(), 3);
    assert_eq!(quantized[0].shape(), vec![2, 8]);
    assert_eq!(quantized[0].dtype(), MlxDtype::Uint32);
    assert_eq!(quantized[1].shape(), vec![2, 2]);
    assert_eq!(quantized[1].dtype(), MlxDtype::Float32);
    assert_eq!(quantized[2].shape(), vec![2, 2]);
    assert_eq!(quantized[2].dtype(), MlxDtype::Float32);

    let restored = dequantize_with_mode(
        &quantized[0],
        &quantized[1],
        Some(&quantized[2]),
        Some(32),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        Some(MlxDtype::Float32),
        None,
    );
    eval(&[&restored]);

    assert_eq!(restored.shape(), vec![2, 64]);
    assert_eq!(restored.dtype(), MlxDtype::Float32);
}

#[test]
fn fp8_conversion_wrappers_preserve_shape_contract() {
    let values = [0.0_f32, 1.0, -2.0, 4.0];
    let x = MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(&values),
        &[2, 2],
        MlxDtype::Float32,
    );

    let fp8 = to_fp8(&x, None);
    eval(&[&fp8]);
    assert_eq!(fp8.shape(), vec![2, 2]);
    assert_eq!(fp8.dtype(), MlxDtype::Uint8);

    let restored = from_fp8(&fp8, MlxDtype::Float32, None);
    eval(&[&restored]);
    assert_eq!(restored.shape(), vec![2, 2]);
    assert_eq!(restored.dtype(), MlxDtype::Float32);
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (a - e).abs();
        assert!(
            delta <= tolerance,
            "mismatch at {idx}: actual={a}, expected={e}, delta={delta}, tolerance={tolerance}"
        );
    }
}

// ── Task 2: Shape / structural ops ──────────────────────────────

#[test]
fn as_strided_preserves_data_under_valid_strides() {
    // Create a [2,3] array and reshape via as_strided with identity strides.
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(&data[..]),
        &[2, 3],
        MlxDtype::Float32,
    );
    // Reshape [2,3] -> [3,2] with row-major strides [2,1].
    let b = as_strided(&a, &[3, 2], &[2, 1], 0, None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![3, 2]);
    assert_eq!(b.data_f32().len(), 6);
}

#[test]
fn broadcast_to_expands_dims() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
    let b = broadcast_to(&a, &[4, 3], None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![4, 3]);
}

#[test]
fn expand_dims_inserts_singleton() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
    let b = expand_dims(&a, 0, None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![1, 3]);
}

#[test]
fn expand_dims_axes_inserts_multiple() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
    let b = expand_dims_axes(&a, &[0, 2], None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![1, 3, 1]);
}

#[test]
fn flatten_collapses_dims() {
    let a = zeros(&[2, 3, 4], MlxDtype::Float32, None);
    let b = flatten(&a, 0, -1, None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![24]);
}

#[test]
fn pad_extends_with_value() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
    let pad_val = MlxArray::from_f32(0.0);
    let b = pad(&a, &[0], &[2], &[1], &pad_val, None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![6]);
    assert_eq!(b.data_f32(), &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn repeat_along_axis() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
    let b = repeat(&a, 2, None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![6]);
    assert_eq!(b.data_f32(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn repeat_axis_explicit() {
    let a = zeros(&[2, 3], MlxDtype::Float32, None);
    let b = repeat_axis(&a, 4, 1, None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![2, 12]);
}

#[test]
fn unflatten_splits_one_dim() {
    let a = zeros(&[24], MlxDtype::Float32, None);
    let b = unflatten(&a, 0, &[2, 3, 4], None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![2, 3, 4]);
}

#[test]
fn transpose_with_axes() {
    let a = zeros(&[2, 3, 4], MlxDtype::Float32, None);
    let b = transpose(&a, &[2, 0, 1], None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![4, 2, 3]);
}

#[test]
fn slice_update_replaces_region() {
    let src = MlxArray::from_f32_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let upd = MlxArray::from_f32_slice(&[99.0, 98.0]);
    let b = slice_update(&src, &upd, &[1], &[3], &[1], None);
    eval(&[&b]);
    assert_eq!(b.shape(), vec![5]);
    assert_eq!(b.data_f32(), &[0.0, 99.0, 98.0, 3.0, 4.0]);
}

#[test]
fn dynamic_slice_and_update_use_array_start() {
    let src = MlxArray::from_f32_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let start_value = [2_i32];
    let start = MlxArray::from_raw_data(
        start_value.as_ptr().cast(),
        std::mem::size_of_val(&start_value),
        &[1],
        MlxDtype::Int32,
    );
    let selected = slice_dynamic(&src, &start, &[0], &[2], None);
    let update = MlxArray::from_f32_slice(&[90.0, 91.0]);
    let replaced = slice_update_dynamic(&src, &update, &start, &[0], None);
    eval(&[&selected, &replaced]);
    assert_eq!(selected.data_f32(), &[2.0, 3.0]);
    assert_eq!(replaced.data_f32(), &[0.0, 1.0, 90.0, 91.0, 4.0]);
}

#[test]
fn contiguous_makes_non_contiguous_contiguous() {
    // transpose produces a non-contiguous view
    let a = zeros(&[2, 3], MlxDtype::Float32, None);
    let t = transpose(&a, &[1, 0], None);
    let c = contiguous(&t, None);
    eval(&[&c]);
    assert_eq!(c.shape(), vec![3, 2]);
}

#[test]
fn view_reinterprets_dtype() {
    let data: Vec<u32> = vec![0x3F800000, 0x40000000]; // 1.0f32, 2.0f32 bit patterns
    let a = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(&data[..]),
        &[2],
        MlxDtype::Uint32,
    );
    let b = view(&a, MlxDtype::Float32, None);
    eval(&[&b]);
    assert_eq!(b.dtype(), MlxDtype::Float32);
    assert_eq!(b.data_f32(), &[1.0, 2.0]);
}

// ── Task 3: Reduction / sort ops ───────────────────────────────

#[test]
fn argmax_returns_correct_index() {
    let a = MlxArray::from_f32_slice(&[0.1, 0.5, 0.3, 0.9]);
    let idx = argmax(&a, None);
    assert_eq!(eval_first_u32(&idx), 3);
}

#[test]
fn argpartition_splits_at_kth() {
    let a = MlxArray::from_f32_slice(&[3.0, 1.0, 4.0, 1.5, 2.0]);
    let idx = argpartition_axis(&a, 2, 0, None);
    let idx_f = astype(&idx, MlxDtype::Float32, None);
    eval(&[&idx_f]);
    assert_eq!(idx.shape(), vec![5]);
}

#[test]
fn argsort_produces_sorted_indices() {
    let a = MlxArray::from_f32_slice(&[3.0, 1.0, 2.0]);
    let idx = argsort_axis(&a, 0, None);
    let idx_f = astype(&idx, MlxDtype::Float32, None);
    eval(&[&idx_f]);
    assert_eq!(idx_f.data_f32(), &[1.0, 2.0, 0.0]);
}

#[test]
fn cumsum_inclusive_and_exclusive() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0]);
    let inc = cumsum(&a, 0, false, true, None);
    let exc = cumsum(&a, 0, false, false, None);
    eval(&[&inc, &exc]);
    assert_eq!(inc.data_f32(), &[1.0, 3.0, 6.0, 10.0]);
    assert_eq!(exc.data_f32(), &[0.0, 1.0, 3.0, 6.0]);
}

#[test]
fn softmax_sums_to_one() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
    let s = softmax(&a, 0, None);
    let total = sum_axis(&s, 0, false, None);
    eval(&[&total]);
    assert_close_f32(total.data_f32(), &[1.0], 1e-5);
}

#[test]
fn sum_along_axis() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(&data[..]),
        &[2, 3],
        MlxDtype::Float32,
    );
    let s0 = sum_axis(&a, 0, false, None);
    let s1 = sum_axis(&a, 1, false, None);
    eval(&[&s0, &s1]);
    assert_eq!(s0.data_f32(), &[3.0, 5.0, 7.0]); // sum over rows
    assert_eq!(s1.data_f32(), &[3.0, 12.0]); // sum over cols
}

#[test]
fn topk_returns_top_k_values_and_indices() {
    let a = MlxArray::from_f32_slice(&[1.0, 5.0, 3.0, 4.0, 2.0]);
    let result = topk(&a, 3, None);
    eval(&[&result]);
    assert_eq!(result.shape(), vec![3]);
    // MLX topk returns values in ascending order
    assert_eq!(result.data_f32(), &[3.0, 4.0, 5.0]);
}

#[test]
fn topk_axis_works_on_non_last() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(&data[..]),
        &[3, 2],
        MlxDtype::Float32,
    );
    let result = topk_axis(&a, 2, 0, None);
    eval(&[&result]);
    assert_eq!(result.shape(), vec![2, 2]);
}

#[test]
fn take_along_axis_gathers() {
    let a = MlxArray::from_f32_slice(&[10.0, 20.0, 30.0, 40.0]);
    let indices = MlxArray::from_raw_data(
        &[2u32, 0u32] as *const u32 as *const u8,
        std::mem::size_of::<u32>() * 2,
        &[2],
        MlxDtype::Uint32,
    );
    let result = take_along_axis(&a, &indices, 0, None);
    eval(&[&result]);
    assert_eq!(result.data_f32(), &[30.0, 10.0]);
}

#[test]
fn take_axis_gathers_single_axis() {
    let a = MlxArray::from_f32_slice(&[10.0, 20.0, 30.0, 40.0]);
    let indices = MlxArray::from_raw_data(
        &[3u32, 1u32] as *const u32 as *const u8,
        std::mem::size_of::<u32>() * 2,
        &[2],
        MlxDtype::Uint32,
    );
    let result = take(&a, &indices, 0, None);
    eval(&[&result]);
    assert_eq!(result.data_f32(), &[40.0, 20.0]);
}

#[test]
fn put_along_axis_scatters() {
    let a = MlxArray::from_f32_slice(&[0.0, 0.0, 0.0, 0.0]);
    let indices = MlxArray::from_raw_data(
        &[1u32, 3u32] as *const u32 as *const u8,
        std::mem::size_of::<u32>() * 2,
        &[2],
        MlxDtype::Uint32,
    );
    let values = MlxArray::from_f32_slice(&[99.0, 88.0]);
    let result = put_along_axis(&a, &indices, &values, 0, None);
    eval(&[&result]);
    assert_eq!(result.data_f32(), &[0.0, 99.0, 0.0, 88.0]);
}

// ── Task 4: Creation + binary/unary ops ──────────────────────

#[test]
fn zeros_creates_all_zero_array() {
    let a = zeros(&[3, 4], MlxDtype::Float32, None);
    eval(&[&a]);
    assert_eq!(a.shape(), vec![3, 4]);
    assert!(a.data_f32().iter().all(|&v| v == 0.0));

    let b = zeros(&[5], MlxDtype::Int32, None);
    let bf = astype(&b, MlxDtype::Float32, None);
    eval(&[&bf]);
    assert!(bf.data_f32().iter().all(|&v| v == 0.0));
}

#[test]
fn arange_produces_sequence() {
    let a = arange(0.0, 5.0, 1.0, MlxDtype::Float32, None);
    eval(&[&a]);
    assert_eq!(a.data_f32(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn random_categorical_produces_valid_indices() {
    let logits = MlxArray::from_f32_slice(&[0.0, 10.0, 0.0, 0.0]);
    let token = random_categorical(&logits, None);
    assert_eq!(eval_first_u32(&token), 1); // heavily biased toward index 1
}

#[test]
fn binary_ops_shape_and_value_correctness() {
    let a = MlxArray::from_f32_slice(&[4.0, 6.0, 8.0]);
    let b = MlxArray::from_f32_slice(&[2.0, 3.0, 4.0]);

    let d = divide(&a, &b, None);
    eval(&[&d]);
    assert_eq!(d.data_f32(), &[2.0, 2.0, 2.0]);

    let s = subtract(&a, &b, None);
    eval(&[&s]);
    assert_eq!(s.data_f32(), &[2.0, 3.0, 4.0]);

    let p = power(&b, &MlxArray::from_f32(2.0), None);
    eval(&[&p]);
    assert_eq!(p.data_f32(), &[4.0, 9.0, 16.0]);

    let mx = maximum(&a, &b, None);
    eval(&[&mx]);
    assert_eq!(mx.data_f32(), &[4.0, 6.0, 8.0]);

    let mn = minimum(&a, &b, None);
    eval(&[&mn]);
    assert_eq!(mn.data_f32(), &[2.0, 3.0, 4.0]);
}

#[test]
fn comparison_ops_return_bool() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
    let b = MlxArray::from_f32_slice(&[2.0, 2.0, 1.0]);

    let lt = less(&a, &b, None);
    eval(&[&lt]);
    assert_eq!(lt.dtype(), MlxDtype::Bool);
    let lt_f = astype(&lt, MlxDtype::Float32, None);
    eval(&[&lt_f]);
    assert_eq!(lt_f.data_f32(), &[1.0, 0.0, 0.0]);

    let eq = equal(&a, &b, None);
    let eq_f = astype(&eq, MlxDtype::Float32, None);
    eval(&[&eq_f]);
    assert_eq!(eq_f.data_f32(), &[0.0, 1.0, 0.0]);

    let ne = not_equal(&a, &b, None);
    let ne_f = astype(&ne, MlxDtype::Float32, None);
    eval(&[&ne_f]);
    assert_eq!(ne_f.data_f32(), &[1.0, 0.0, 1.0]);

    let ge = greater_equal(&a, &b, None);
    let ge_f = astype(&ge, MlxDtype::Float32, None);
    eval(&[&ge_f]);
    assert_eq!(ge_f.data_f32(), &[0.0, 1.0, 1.0]);

    let le = less_equal(&a, &b, None);
    let le_f = astype(&le, MlxDtype::Float32, None);
    eval(&[&le_f]);
    assert_eq!(le_f.data_f32(), &[1.0, 1.0, 0.0]);
}

#[test]
fn logical_and_on_bool_arrays() {
    let t: Vec<u8> = vec![1, 1, 0, 0];
    let f: Vec<u8> = vec![1, 0, 1, 0];
    let a = MlxArray::from_raw_data(t.as_ptr(), 4, &[4], MlxDtype::Bool);
    let b = MlxArray::from_raw_data(f.as_ptr(), 4, &[4], MlxDtype::Bool);
    let r = logical_and(&a, &b, None);
    let rf = astype(&r, MlxDtype::Float32, None);
    eval(&[&rf]);
    assert_eq!(rf.data_f32(), &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn outer_product_shape_and_values() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
    let b = MlxArray::from_f32_slice(&[10.0, 20.0]);
    let r = outer(&a, &b, None);
    eval(&[&r]);
    assert_eq!(r.shape(), vec![3, 2]);
    assert_eq!(r.data_f32(), &[10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
}

#[test]
fn unary_ops_correctness() {
    let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);

    let n = negative(&a, None);
    eval(&[&n]);
    assert_eq!(n.data_f32(), &[-1.0, -2.0, -3.0]);

    let e = exp(&MlxArray::from_f32(0.0), None);
    eval(&[&e]);
    assert_eq!(e.data_f32(), &[1.0]);

    let l = log(&MlxArray::from_f32(1.0), None);
    eval(&[&l]);
    assert_eq!(l.data_f32(), &[0.0]);

    let lp = log1p(&MlxArray::from_f32(0.0), None);
    eval(&[&lp]);
    assert_eq!(lp.data_f32(), &[0.0]);

    let c = cos(&MlxArray::from_f32(0.0), None);
    eval(&[&c]);
    assert_close_f32(c.data_f32(), &[1.0], 1e-6);

    let s = sin(&MlxArray::from_f32(0.0), None);
    eval(&[&s]);
    assert_close_f32(s.data_f32(), &[0.0], 1e-6);

    let fl = floor(&MlxArray::from_f32(2.7), None);
    eval(&[&fl]);
    assert_eq!(fl.data_f32(), &[2.0]);
}

#[test]
fn clip_clamps_values() {
    let a = MlxArray::from_f32_slice(&[-5.0, 0.5, 1.5, 5.0]);
    let lo = MlxArray::from_f32(0.0);
    let hi = MlxArray::from_f32(1.0);
    let r = clip(&a, &lo, &hi, None);
    eval(&[&r]);
    assert_eq!(r.data_f32(), &[0.0, 0.5, 1.0, 1.0]);
}

// ── Task 9: Integration smoke test ───────────────────────────

#[test]
fn mini_transformer_forward_pass_chains_many_ops() {
    // Simulate a mini transformer step:
    // input -> reshape -> matmul -> softmax -> topk -> take_along_axis
    let batch: i32 = 2;
    let seq: i32 = 4;
    let dim: i32 = 8;
    let vocab: i32 = 16;

    // Input embeddings [batch, seq, dim]
    let input_data: Vec<f32> = (0..batch * seq * dim).map(|i| (i as f32) * 0.01).collect();
    let input = MlxArray::from_raw_data(
        input_data.as_ptr() as *const u8,
        std::mem::size_of_val(&input_data[..]),
        &[batch, seq, dim],
        MlxDtype::Float32,
    );

    // Weight matrix [dim, vocab]
    let w_data: Vec<f32> = (0..dim * vocab)
        .map(|i| ((i as f32) - 64.0) * 0.01)
        .collect();
    let weight = MlxArray::from_raw_data(
        w_data.as_ptr() as *const u8,
        std::mem::size_of_val(&w_data[..]),
        &[dim, vocab],
        MlxDtype::Float32,
    );

    // flatten batch*seq for matmul -> [batch*seq, dim]
    let flat = reshape(&input, &[batch * seq, dim], None);
    // logits = flat @ weight -> [batch*seq, vocab]
    let logits = matmul(&flat, &weight, None);
    // softmax over vocab dim
    let probs = softmax(&logits, -1, None);
    // topk=3 from logits along last axis
    let top_vals = topk_axis(&logits, 3, -1, None);
    // argmax for greedy decode
    let token = argmax(&logits, None);

    eval(&[&probs, &top_vals, &token]);

    assert_eq!(probs.shape(), vec![batch * seq, vocab]);
    assert_eq!(top_vals.shape(), vec![batch * seq, 3]);

    // Verify softmax sums to ~1 for each row
    let row_sum = sum_axis(&probs, -1, false, None);
    eval(&[&row_sum]);
    for &s in row_sum.data_f32() {
        assert_close_f32(&[s], &[1.0], 1e-4);
    }
}
