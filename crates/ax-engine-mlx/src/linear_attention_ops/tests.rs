use super::*;

fn cfg() -> LinearAttentionConfig {
    let (q_scale, k_scale) = linear_attention_qk_scale(4);
    LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 4,
        value_head_dim: 3,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    }
}

fn f32_array(data: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Float32,
    )
}

fn stable_softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} length mismatch: actual={}, expected={}",
        actual.len(),
        expected.len()
    );

    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "{label}[{idx}] mismatch: actual={actual}, expected={expected}, diff={diff}, tolerance={tolerance}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    a_log: &[f32],
    a_raw: &[f32],
    dt_bias: &[f32],
    b_raw: &[f32],
    initial_state: &[f32],
    seq: usize,
    key_head_dim: usize,
    value_head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut state = initial_state.to_vec();
    let mut y = vec![0.0; seq * value_head_dim];
    let decay_rate = a_log[0].exp();

    for t in 0..seq {
        let g = (-decay_rate * stable_softplus(a_raw[t] + dt_bias[0])).exp();
        let beta = sigmoid(b_raw[t]);
        for dv in 0..value_head_dim {
            let state_offset = dv * key_head_dim;
            let mut kv_mem = 0.0;
            for dk in 0..key_head_dim {
                let state_idx = state_offset + dk;
                state[state_idx] *= g;
                kv_mem += state[state_idx] * k[t * key_head_dim + dk];
            }

            let delta = (v[t * value_head_dim + dv] - kv_mem) * beta;
            let mut out = 0.0;
            for dk in 0..key_head_dim {
                let state_idx = state_offset + dk;
                state[state_idx] += k[t * key_head_dim + dk] * delta;
                out += state[state_idx] * q[t * key_head_dim + dk];
            }
            y[t * value_head_dim + dv] = out;
        }
    }

    (y, state)
}

#[test]
fn warm_gated_delta_decode_kernels_compiles_decode_specializations() {
    // Exercises the exact load-time warm path — decode-shape gated
    // delta (seq=1 specialization) plus conv1d — at kernel-eligible
    // head dims (key_head_dim divisible by 32, like every production
    // linear-attention config). Must succeed on any Metal host that
    // can serve the model at all.
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let production_like = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 4,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    warm_gated_delta_decode_kernels(&production_like).expect("warm-up must compile decode kernels");

    // Sub-threshold dims never dispatch the custom kernel; the warm
    // path must be a no-op success, not a panic.
    warm_gated_delta_decode_kernels(&cfg()).expect("ineligible dims must be a no-op");
}

#[test]
fn compute_gated_delta_g_preserves_shape_and_float32_dtype() {
    let cfg = cfg();
    let a_log = zeros(&[cfg.num_value_heads as i32], MlxDtype::Float32, None);
    let a = zeros(
        &[1, 5, cfg.num_value_heads as i32],
        MlxDtype::Bfloat16,
        None,
    );
    let dt_bias = zeros(&[cfg.num_value_heads as i32], MlxDtype::Bfloat16, None);

    let g = compute_gated_delta_g(&a_log, &a, &dt_bias);

    assert_eq!(g.shape(), vec![1, 5, 2]);
    // Streaming prefill keeps g in float32 for the recurrent state update
    // (matches mlx_lm's compute_g → float32 contract).
    assert_eq!(g.dtype(), MlxDtype::Float32);
}

#[test]
fn compute_gated_delta_g_uses_stable_softplus_for_large_positive_values() {
    let a_log = f32_array(&[0.0], &[1]);
    let a = f32_array(&[25.0], &[1, 1, 1]);
    let dt_bias = f32_array(&[0.0], &[1]);

    let g = compute_gated_delta_g(&a_log, &a, &dt_bias);
    mlx_sys::eval(&[&g]);

    let actual = g.data_f32()[0];
    let expected = (-25.0_f32).exp();
    assert!(actual.is_finite(), "g should stay finite, got {actual}");
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn linear_attention_conv1d_returns_prompt_output_and_tail() {
    let cfg = cfg();
    let qkv = zeros(&[1, 5, cfg.conv_dim() as i32], MlxDtype::Float32, None);
    let weight = zeros(
        &[cfg.conv_dim() as i32, cfg.conv_kernel_dim as i32, 1_i32],
        MlxDtype::Float32,
        None,
    );

    let (conv_out, new_state) = linear_attention_conv1d(&cfg, &qkv, &weight, None);

    assert_eq!(conv_out.shape(), vec![1, 5, 14]);
    assert_eq!(new_state.shape(), vec![1, 3, 14]);
}

#[test]
fn split_linear_attention_qkv_matches_config_dims() {
    let cfg = cfg();
    let conv_out = zeros(&[1, 5, cfg.conv_dim() as i32], MlxDtype::Float32, None);

    let qkv = split_linear_attention_qkv(&cfg, &conv_out);

    assert_eq!(qkv.q.shape(), vec![1, 5, 1, 4]);
    assert_eq!(qkv.k.shape(), vec![1, 5, 1, 4]);
    assert_eq!(qkv.v.shape(), vec![1, 5, 2, 3]);
}

#[test]
fn gated_delta_kernel_reports_reference_shapes() {
    // B=1, T=2, Hk=1, Dk=32, Hv=1, Dv=4
    let q = zeros(&[1, 2, 1, 32], MlxDtype::Float32, None);
    let k = zeros(&[1, 2, 1, 32], MlxDtype::Float32, None);
    let v = zeros(&[1, 2, 1, 4], MlxDtype::Float32, None);
    // a_log, dt_bias: [Hv] float32 (StT)
    let a_log = zeros(&[1], MlxDtype::Float32, None);
    let a_raw = zeros(&[1, 2, 1], MlxDtype::Float32, None);
    let dt_bias = zeros(&[1], MlxDtype::Float32, None);
    let b_raw = zeros(&[1, 2, 1], MlxDtype::Float32, None);
    let state = zeros(&[1, 1, 4, 32], MlxDtype::Float32, None);

    let (y, new_state) = gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);

    assert_eq!(y.shape(), vec![1, 2, 1, 4]);
    assert_eq!(new_state.shape(), vec![1, 1, 4, 32]);
}

#[test]
fn gated_delta_prefill_short_seq_matches_cpu_on_default_path() {
    // Hybrid default: seq <= 512 uses the legacy short TG-cache kernel.
    const SEQ: usize = 8;
    const KEY_HEAD_DIM: usize = 32;
    const VALUE_HEAD_DIM: usize = 4;
    let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data: Vec<f32> = (0..SEQ * VALUE_HEAD_DIM)
        .map(|idx| ((idx % 3) as f32 - 1.0) * 0.04)
        .collect();
    let a_log_data = vec![-0.2];
    let a_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.01 - 0.05).collect();
    let dt_bias_data = vec![0.05];
    let b_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.02 - 0.1).collect();
    let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
        .collect();
    let (expected_y, expected_state) = gated_delta_cpu_reference(
        &q_data,
        &k_data,
        &v_data,
        &a_log_data,
        &a_raw_data,
        &dt_bias_data,
        &b_raw_data,
        &state_data,
        SEQ,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
    );
    let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
    let a_log = f32_array(&a_log_data, &[1]);
    let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
    let dt_bias = f32_array(&dt_bias_data, &[1]);
    let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
    let state = f32_array(
        &state_data,
        &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
    );
    let (y, new_state) = gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    mlx_sys::eval(&[&y, &new_state]);
    assert_close("y", y.data_f32(), &expected_y, 1e-5);
    assert_close("state", new_state.data_f32(), &expected_state, 1e-5);
}

#[test]
fn gated_delta_kernel_accepts_medium_prefill_specialization() {
    // seq=1024 is the production p2048 chunk. Default-ON tile-512 splits
    // this into two 512 TG kernels; this test drives that shipped path.
    let seq = (GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY) as i32;
    let q = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
    let k = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
    let v = zeros(&[1, seq, 1, 4], MlxDtype::Float32, None);
    let a_log = zeros(&[1], MlxDtype::Float32, None);
    let a_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
    let dt_bias = zeros(&[1], MlxDtype::Float32, None);
    let b_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
    let state = zeros(&[1, 1, 4, 32], MlxDtype::Float32, None);

    let (y, new_state) = gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    mlx_sys::eval(&[&y, &new_state]);

    assert_eq!(y.shape(), vec![1, seq, 1, 4]);
    assert_eq!(new_state.shape(), vec![1, 1, 4, 32]);
}

#[test]
fn linear_attention_prefill_chunk_cap_follows_streaming_and_moe() {
    assert_eq!(
        linear_attention_prefill_chunk_cap(true, false),
        GATED_DELTA_THREADGROUP_CACHE_CAPACITY
    );
    // Default-OFF after the 1280 wash: dense-hybrid cap is 1024.
    assert_eq!(
        linear_attention_prefill_chunk_cap(false, false),
        GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY
    );
    // MoE hybrids default to one 2048 chunk (M5 +11.9% p2048, 2026-08-17).
    assert_eq!(
        linear_attention_prefill_chunk_cap(false, true),
        GATED_DELTA_THREADGROUP_CACHE_CAPACITY
    );
    assert!(
        !fastpath::qwen_prefill_chunk_1280_enabled(),
        "closed 1280 chunk stays opt-in"
    );
    assert!(
        fastpath::qwen_moe_prefill_single_2048_enabled(),
        "MoE single-2048 ships default-ON"
    );
}

#[test]
fn gated_delta_prefill_tile_512_is_seq_gated() {
    assert!(!gated_delta_prefill_tile_512_seq_eligible(1));
    assert!(!gated_delta_prefill_tile_512_seq_eligible(
        GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32
    ));
    assert!(gated_delta_prefill_tile_512_seq_eligible(
        GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32 + 1
    ));
    assert!(gated_delta_prefill_tile_512_seq_eligible(
        GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY as i32
    ));
}

#[test]
fn gated_delta_prefill_tiled_matches_oneshot_short_seq() {
    const SEQ: usize = 16;
    const TILE: i32 = 8;
    const KEY_HEAD_DIM: usize = 32;
    const VALUE_HEAD_DIM: usize = 4;
    let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data: Vec<f32> = (0..SEQ * VALUE_HEAD_DIM)
        .map(|idx| ((idx % 3) as f32 - 1.0) * 0.04)
        .collect();
    let a_log_data = vec![-0.2];
    let a_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.01 - 0.05).collect();
    let dt_bias_data = vec![0.05];
    let b_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.02 - 0.1).collect();
    let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
        .collect();
    let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
    let a_log = f32_array(&a_log_data, &[1]);
    let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
    let dt_bias = f32_array(&dt_bias_data, &[1]);
    let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
    let state = f32_array(
        &state_data,
        &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
    );
    let (want_y, want_state) =
        gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    let (got_y, got_state) =
        gated_delta_prefill_tiled(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, TILE);
    mlx_sys::eval(&[&got_y, &got_state, &want_y, &want_state]);
    assert_close("y", got_y.data_f32(), want_y.data_f32(), 1e-5);
    assert_close("state", got_state.data_f32(), want_state.data_f32(), 1e-5);
}

#[test]
fn gated_delta_prefill_chunkwise_matches_oneshot() {
    const SEQ: usize = 16;
    const TILE: i32 = 8;
    const KEY_HEAD_DIM: usize = 32;
    const VALUE_HEAD_DIM: usize = 4;
    let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data: Vec<f32> = (0..SEQ * VALUE_HEAD_DIM)
        .map(|idx| ((idx % 3) as f32 - 1.0) * 0.04)
        .collect();
    let a_log_data = vec![-0.2];
    let a_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.01 - 0.05).collect();
    let dt_bias_data = vec![0.05];
    let b_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.02 - 0.1).collect();
    let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
        .collect();
    let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
    let a_log = f32_array(&a_log_data, &[1]);
    let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
    let dt_bias = f32_array(&dt_bias_data, &[1]);
    let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
    let state = f32_array(
        &state_data,
        &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
    );
    let (want_y, want_state) =
        gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    let (got_y, got_state) =
        gated_delta_prefill_chunkwise(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, TILE);
    mlx_sys::eval(&[&got_y, &got_state, &want_y, &want_state]);
    assert_close("y", got_y.data_f32(), want_y.data_f32(), 1e-5);
    assert_close("state", got_state.data_f32(), want_state.data_f32(), 1e-5);
    assert!(
        fastpath::should_qwen_gd_prefill_chunkwise_for(true, 1024),
        "shipped chunkwise gate must accept the p2048 chunk length"
    );
}

#[test]
fn gated_delta_prefill_streaming_matches_cpu_reference() {
    // Regression test: the streaming kernel reuses a single
    // threadgroup-shared g_t/beta_t scalar across loop iterations
    // (unlike the cached-array kernels above), which needs a barrier
    // both after the leader-thread write and after every thread's
    // read, or a fast SIMD-group can race into the next iteration's
    // write while a slower one is still reading. VALUE_HEAD_DIM=4
    // matches the kernel's fixed (32, 4, 1) threadgroup so all four
    // SIMD-groups in the threadgroup are exercised, and SEQ is long
    // enough to give real divergence opportunity across iterations.
    const SEQ: usize = 128;
    const KEY_HEAD_DIM: usize = 32;
    const VALUE_HEAD_DIM: usize = 4;
    let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data: Vec<f32> = (0..SEQ * VALUE_HEAD_DIM)
        .map(|idx| ((idx % 3) as f32 - 1.0) * 0.04)
        .collect();
    let a_log_data = vec![-0.2];
    let a_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.01 - 0.05).collect();
    let dt_bias_data = vec![0.05];
    let b_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.02 - 0.1).collect();
    let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
        .collect();
    let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
    let a_log = f32_array(&a_log_data, &[1]);
    let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
    let dt_bias = f32_array(&dt_bias_data, &[1]);
    let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
    let state = f32_array(
        &state_data,
        &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
    );
    let (want_y, want_state) = gated_delta_cpu_reference(
        &q_data,
        &k_data,
        &v_data,
        &a_log_data,
        &a_raw_data,
        &dt_bias_data,
        &b_raw_data,
        &state_data,
        SEQ,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
    );
    let (got_y, got_state) = gated_delta_prefill_streaming_kernel(
        &q,
        &k,
        &v,
        &a_log,
        &a_raw,
        &dt_bias,
        &b_raw,
        &state,
        1,
        SEQ as i32,
        1,
        KEY_HEAD_DIM as i32,
        1,
        VALUE_HEAD_DIM as i32,
        vec![1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
    );
    mlx_sys::eval(&[&got_y, &got_state]);
    assert_close("y", got_y.data_f32(), &want_y, 1e-4);
    assert_close("state", got_state.data_f32(), &want_state, 1e-4);
}

#[test]
fn gated_delta_kernel_accepts_long_prefill_specialization() {
    let seq = (GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY + 1) as i32;
    let q = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
    let k = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
    let v = zeros(&[1, seq, 1, 4], MlxDtype::Float32, None);
    let a_log = zeros(&[1], MlxDtype::Float32, None);
    let a_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
    let dt_bias = zeros(&[1], MlxDtype::Float32, None);
    let b_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
    let state = zeros(&[1, 1, 4, 32], MlxDtype::Float32, None);

    let (y, new_state) = gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    mlx_sys::eval(&[&y, &new_state]);

    assert_eq!(y.shape(), vec![1, seq, 1, 4]);
    assert_eq!(new_state.shape(), vec![1, 1, 4, 32]);
}

#[test]
fn gated_delta_kernel_matches_cpu_reference_for_small_sequence() {
    const SEQ: usize = 2;
    const KEY_HEAD_DIM: usize = 32;
    const VALUE_HEAD_DIM: usize = 4;

    let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data = vec![0.10, -0.05, 0.07, 0.03, -0.02, 0.04, 0.08, -0.06];
    let a_log_data = vec![-0.2];
    let a_raw_data = vec![0.1, -0.15];
    let dt_bias_data = vec![0.05];
    let b_raw_data = vec![0.25, -0.1];
    let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
        .collect();
    let (expected_y, expected_state) = gated_delta_cpu_reference(
        &q_data,
        &k_data,
        &v_data,
        &a_log_data,
        &a_raw_data,
        &dt_bias_data,
        &b_raw_data,
        &state_data,
        SEQ,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
    );

    let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
    let a_log = f32_array(&a_log_data, &[1]);
    let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
    let dt_bias = f32_array(&dt_bias_data, &[1]);
    let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
    let state = f32_array(
        &state_data,
        &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
    );

    let (y, new_state) = gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    mlx_sys::eval(&[&y, &new_state]);

    assert_close("y", y.data_f32(), &expected_y, 1e-6);
    assert_close("state", new_state.data_f32(), &expected_state, 1e-6);
}

#[test]
fn gated_delta_prefix_checkpoint_matches_decode_kernel_row0() {
    const SEQ: i32 = 2;
    const KEY_HEAD_DIM: i32 = 32;
    const VALUE_HEAD_DIM: i32 = 4;
    let q_data: Vec<f32> = (0..(SEQ * KEY_HEAD_DIM) as usize)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..(SEQ * KEY_HEAD_DIM) as usize)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data = vec![0.10, -0.05, 0.07, 0.03, -0.02, 0.04, 0.08, -0.06];
    let a_log = f32_array(&[-0.2], &[1]);
    let a_raw = f32_array(&[0.1, -0.15], &[1, SEQ, 1]);
    let dt_bias = f32_array(&[0.05], &[1]);
    let b_raw = f32_array(&[0.25, -0.1], &[1, SEQ, 1]);
    let q = f32_array(&q_data, &[1, SEQ, 1, KEY_HEAD_DIM]);
    let k = f32_array(&k_data, &[1, SEQ, 1, KEY_HEAD_DIM]);
    let v = f32_array(&v_data, &[1, SEQ, 1, VALUE_HEAD_DIM]);
    let state = f32_array(
        &(0..(VALUE_HEAD_DIM * KEY_HEAD_DIM) as usize)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect::<Vec<_>>(),
        &[1, 1, VALUE_HEAD_DIM, KEY_HEAD_DIM],
    );

    let (_y_ck, _final_ck, prefix_ck) = gated_delta_kernel_with_prefix_checkpoint(
        &q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, 1,
    );
    let q0 = contiguous(
        &slice(
            &q,
            &[0, 0, 0, 0],
            &[1, 1, 1, KEY_HEAD_DIM],
            &[1, 1, 1, 1],
            None,
        ),
        None,
    );
    let k0 = contiguous(
        &slice(
            &k,
            &[0, 0, 0, 0],
            &[1, 1, 1, KEY_HEAD_DIM],
            &[1, 1, 1, 1],
            None,
        ),
        None,
    );
    let v0 = contiguous(
        &slice(
            &v,
            &[0, 0, 0, 0],
            &[1, 1, 1, VALUE_HEAD_DIM],
            &[1, 1, 1, 1],
            None,
        ),
        None,
    );
    let a0 = contiguous(
        &slice(&a_raw, &[0, 0, 0], &[1, 1, 1], &[1, 1, 1], None),
        None,
    );
    let b0 = contiguous(
        &slice(&b_raw, &[0, 0, 0], &[1, 1, 1], &[1, 1, 1], None),
        None,
    );
    let (_y0, state1) = gated_delta_kernel(&q0, &k0, &v0, &a_log, &a0, &dt_bias, &b0, &state);
    mlx_sys::eval(&[&prefix_ck, &state1]);
    assert_eq!(prefix_ck.data_f32(), state1.data_f32());
}

#[test]
fn gated_delta_decode_kernel_matches_cpu_reference_for_single_token() {
    const SEQ: usize = 1;
    const KEY_HEAD_DIM: usize = 32;
    const VALUE_HEAD_DIM: usize = 4;

    let q_data: Vec<f32> = (0..KEY_HEAD_DIM)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..KEY_HEAD_DIM)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data = vec![0.10, -0.05, 0.07, 0.03];
    let a_log_data = vec![-0.2];
    let a_raw_data = vec![0.1];
    let dt_bias_data = vec![0.05];
    let b_raw_data = vec![0.25];
    let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
        .collect();
    let (expected_y, expected_state) = gated_delta_cpu_reference(
        &q_data,
        &k_data,
        &v_data,
        &a_log_data,
        &a_raw_data,
        &dt_bias_data,
        &b_raw_data,
        &state_data,
        SEQ,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
    );

    let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
    let a_log = f32_array(&a_log_data, &[1]);
    let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
    let dt_bias = f32_array(&dt_bias_data, &[1]);
    let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
    let state = f32_array(
        &state_data,
        &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
    );

    let (y, new_state) = gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    mlx_sys::eval(&[&y, &new_state]);

    assert_close("decode_y", y.data_f32(), &expected_y, 1e-6);
    assert_close("decode_state", new_state.data_f32(), &expected_state, 1e-6);
}

#[test]
fn gated_delta_prefix_checkpoint_matches_first_singleton_state() {
    const SEQ: usize = 2;
    const KEY_HEAD_DIM: usize = 32;
    const VALUE_HEAD_DIM: usize = 4;
    let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data = vec![0.10, -0.05, 0.07, 0.03, -0.02, 0.04, 0.08, -0.06];
    let a_log_data = vec![-0.2];
    let a_raw_data = vec![0.1, -0.15];
    let dt_bias_data = vec![0.05];
    let b_raw_data = vec![0.25, -0.1];
    let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
        .collect();
    let (_, expected_checkpoint) = gated_delta_cpu_reference(
        &q_data[..KEY_HEAD_DIM],
        &k_data[..KEY_HEAD_DIM],
        &v_data[..VALUE_HEAD_DIM],
        &a_log_data,
        &a_raw_data[..1],
        &dt_bias_data,
        &b_raw_data[..1],
        &state_data,
        1,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
    );
    let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
    let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
    let a_log = f32_array(&a_log_data, &[1]);
    let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
    let dt_bias = f32_array(&dt_bias_data, &[1]);
    let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
    let state = f32_array(
        &state_data,
        &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
    );

    let (output, final_state, checkpoint) = gated_delta_kernel_with_prefix_checkpoint(
        &q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, 1,
    );
    mlx_sys::eval(&[&output, &final_state, &checkpoint]);

    assert_close(
        "prefix_checkpoint",
        checkpoint.data_f32(),
        &expected_checkpoint,
        1e-6,
    );
}

#[test]
fn gated_delta_prefix_checkpoint_output_matches_sequential_singletons() {
    const SEQ: i32 = 2;
    const KEY_HEAD_DIM: i32 = 32;
    const VALUE_HEAD_DIM: i32 = 4;
    let q_data: Vec<f32> = (0..(SEQ * KEY_HEAD_DIM) as usize)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..(SEQ * KEY_HEAD_DIM) as usize)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data = vec![0.10, -0.05, 0.07, 0.03, -0.02, 0.04, 0.08, -0.06];
    let a_log = f32_array(&[-0.2], &[1]);
    let a_raw = f32_array(&[0.1, -0.15], &[1, SEQ, 1]);
    let dt_bias = f32_array(&[0.05], &[1]);
    let b_raw = f32_array(&[0.25, -0.1], &[1, SEQ, 1]);
    let q = f32_array(&q_data, &[1, SEQ, 1, KEY_HEAD_DIM]);
    let k = f32_array(&k_data, &[1, SEQ, 1, KEY_HEAD_DIM]);
    let v = f32_array(&v_data, &[1, SEQ, 1, VALUE_HEAD_DIM]);
    let state = f32_array(
        &(0..(VALUE_HEAD_DIM * KEY_HEAD_DIM) as usize)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect::<Vec<_>>(),
        &[1, 1, VALUE_HEAD_DIM, KEY_HEAD_DIM],
    );

    let (y_ck, final_ck, prefix_ck) = gated_delta_kernel_with_prefix_checkpoint(
        &q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, 1,
    );

    let q0 = slice_seq_row_4d(&q, 0);
    let k0 = slice_seq_row_4d(&k, 0);
    let v0 = slice_seq_row_4d(&v, 0);
    let a0 = slice_seq_row_3d(&a_raw, 0);
    let b0 = slice_seq_row_3d(&b_raw, 0);
    let (y0, state1) = gated_delta_kernel(&q0, &k0, &v0, &a_log, &a0, &dt_bias, &b0, &state);
    let q1 = slice_seq_row_4d(&q, 1);
    let k1 = slice_seq_row_4d(&k, 1);
    let v1 = slice_seq_row_4d(&v, 1);
    let a1 = slice_seq_row_3d(&a_raw, 1);
    let b1 = slice_seq_row_3d(&b_raw, 1);
    let (y1, state2) = gated_delta_kernel(&q1, &k1, &v1, &a_log, &a1, &dt_bias, &b1, &state1);
    let expected_y = concatenate(&[&y0, &y1], 1, None);
    mlx_sys::eval(&[&y_ck, &final_ck, &prefix_ck, &expected_y, &state2, &state1]);
    assert_eq!(prefix_ck.data_f32(), state1.data_f32());
    assert_eq!(final_ck.data_f32(), state2.data_f32());
    assert_eq!(y_ck.data_f32(), expected_y.data_f32());
}

#[test]
fn gated_delta_decode_seq_bf16_matches_sequential_singletons() {
    const SEQ: i32 = 2;
    const KEY_HEAD_DIM: i32 = 32;
    const VALUE_HEAD_DIM: i32 = 4;
    let q_data: Vec<f32> = (0..(SEQ * 2 * KEY_HEAD_DIM) as usize)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
        .collect();
    let k_data: Vec<f32> = (0..(SEQ * 2 * KEY_HEAD_DIM) as usize)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
        .collect();
    let v_data: Vec<f32> = (0..(SEQ * 2 * VALUE_HEAD_DIM) as usize)
        .map(|idx| ((idx % 9) as f32 - 4.0) * 0.04)
        .collect();
    let q = astype(
        &f32_array(&q_data, &[1, SEQ, 2, KEY_HEAD_DIM]),
        MlxDtype::Bfloat16,
        None,
    );
    let k = astype(
        &f32_array(&k_data, &[1, SEQ, 2, KEY_HEAD_DIM]),
        MlxDtype::Bfloat16,
        None,
    );
    let v = astype(
        &f32_array(&v_data, &[1, SEQ, 2, VALUE_HEAD_DIM]),
        MlxDtype::Bfloat16,
        None,
    );
    let a_log = f32_array(&[-0.2, -0.15], &[2]);
    let a_raw = astype(
        &f32_array(&[0.1, -0.15, 0.05, 0.2], &[1, SEQ, 2]),
        MlxDtype::Bfloat16,
        None,
    );
    let dt_bias = f32_array(&[0.05, -0.02], &[2]);
    let b_raw = astype(
        &f32_array(&[0.25, -0.1, 0.3, -0.05], &[1, SEQ, 2]),
        MlxDtype::Bfloat16,
        None,
    );
    let state = f32_array(
        &(0..(2 * VALUE_HEAD_DIM * KEY_HEAD_DIM) as usize)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect::<Vec<_>>(),
        &[1, 2, VALUE_HEAD_DIM, KEY_HEAD_DIM],
    );

    let (y_ck, final_ck, prefix_ck) = gated_delta_kernel_with_prefix_checkpoint(
        &q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, 1,
    );
    let q0 = slice_seq_row_4d(&q, 0);
    let k0 = slice_seq_row_4d(&k, 0);
    let v0 = slice_seq_row_4d(&v, 0);
    let a0 = slice_seq_row_3d(&a_raw, 0);
    let b0 = slice_seq_row_3d(&b_raw, 0);
    let (y0, state1) = gated_delta_kernel(&q0, &k0, &v0, &a_log, &a0, &dt_bias, &b0, &state);
    let q1 = slice_seq_row_4d(&q, 1);
    let k1 = slice_seq_row_4d(&k, 1);
    let v1 = slice_seq_row_4d(&v, 1);
    let a1 = slice_seq_row_3d(&a_raw, 1);
    let b1 = slice_seq_row_3d(&b_raw, 1);
    let (y1, state2) = gated_delta_kernel(&q1, &k1, &v1, &a_log, &a1, &dt_bias, &b1, &state1);
    let expected_y = concatenate(&[&y0, &y1], 1, None);
    mlx_sys::eval(&[&y_ck, &final_ck, &prefix_ck, &expected_y, &state2, &state1]);
    let y_a = astype(&y_ck, MlxDtype::Float32, None);
    let y_b = astype(&expected_y, MlxDtype::Float32, None);
    mlx_sys::eval(&[&y_a, &y_b]);
    assert_eq!(prefix_ck.data_f32(), state1.data_f32());
    assert_eq!(final_ck.data_f32(), state2.data_f32());
    assert_eq!(y_a.data_f32(), y_b.data_f32());
}

#[test]
fn compiled_gated_delta_decode_seq_matches_eager_kernel() {
    const BATCH: i32 = 1;
    const SEQ: i32 = 3;
    const NUM_KEY_HEADS: i32 = 1;
    const NUM_VALUE_HEADS: i32 = 2;
    const KEY_HEAD_DIM: i32 = 32;
    const VALUE_HEAD_DIM: i32 = 4;
    let state_shape = [BATCH, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM];
    let qk_len = (BATCH * SEQ * NUM_KEY_HEADS * KEY_HEAD_DIM) as usize;
    let v_len = (BATCH * SEQ * NUM_VALUE_HEADS * VALUE_HEAD_DIM) as usize;
    let ab_len = (BATCH * SEQ * NUM_VALUE_HEADS) as usize;
    let state_len = state_shape.iter().product::<i32>() as usize;
    let q_data: Vec<f32> = (0..qk_len)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.017)
        .collect();
    let k_data: Vec<f32> = (0..qk_len)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.013)
        .collect();
    let v_data: Vec<f32> = (0..v_len)
        .map(|idx| ((idx % 9) as f32 - 4.0) * 0.021)
        .collect();
    let a_data: Vec<f32> = (0..ab_len)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.031)
        .collect();
    let b_data: Vec<f32> = (0..ab_len)
        .map(|idx| ((idx % 5) as f32 - 2.0) * 0.027)
        .collect();
    let state_data: Vec<f32> = (0..state_len)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.003)
        .collect();
    let q = astype(
        &f32_array(&q_data, &[BATCH, SEQ, NUM_KEY_HEADS, KEY_HEAD_DIM]),
        MlxDtype::Bfloat16,
        None,
    );
    let k = astype(
        &f32_array(&k_data, &[BATCH, SEQ, NUM_KEY_HEADS, KEY_HEAD_DIM]),
        MlxDtype::Bfloat16,
        None,
    );
    let v = astype(
        &f32_array(&v_data, &[BATCH, SEQ, NUM_VALUE_HEADS, VALUE_HEAD_DIM]),
        MlxDtype::Bfloat16,
        None,
    );
    let a_log = f32_array(&[-0.2, -0.15], &[NUM_VALUE_HEADS]);
    let a_raw = astype(
        &f32_array(&a_data, &[BATCH, SEQ, NUM_VALUE_HEADS]),
        MlxDtype::Bfloat16,
        None,
    );
    let dt_bias = f32_array(&[0.05, -0.02], &[NUM_VALUE_HEADS]);
    let b_raw = astype(
        &f32_array(&b_data, &[BATCH, SEQ, NUM_VALUE_HEADS]),
        MlxDtype::Bfloat16,
        None,
    );
    let state = f32_array(&state_data, &state_shape);

    let (eager_y, eager_state, eager_checkpoint) = gated_delta_decode_seq_kernel(
        &q,
        &k,
        &v,
        &a_log,
        &a_raw,
        &dt_bias,
        &b_raw,
        &state,
        BATCH,
        SEQ,
        NUM_KEY_HEADS,
        KEY_HEAD_DIM,
        NUM_VALUE_HEADS,
        VALUE_HEAD_DIM,
        &state_shape,
    )
    .expect("production decode-sequence kernel must accept the test shape");

    let closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        let outputs = gated_delta_decode_seq_kernel(
            &inputs.get(0),
            &inputs.get(1),
            &inputs.get(2),
            &inputs.get(3),
            &inputs.get(4),
            &inputs.get(5),
            &inputs.get(6),
            &inputs.get(7),
            BATCH,
            SEQ,
            NUM_KEY_HEADS,
            KEY_HEAD_DIM,
            NUM_VALUE_HEADS,
            VALUE_HEAD_DIM,
            &state_shape,
        );
        match outputs {
            Some((y, state, checkpoint)) => vec![y, state, checkpoint],
            None => Vec::new(),
        }
    })
    .compile(false)
    .expect("MLX must trace the production custom Metal kernel");
    let compiled = closure
        .try_apply(&[&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state])
        .expect("compiled production custom Metal kernel must execute");
    assert_eq!(compiled.len(), 3);

    let eager_y_f32 = astype(&eager_y, MlxDtype::Float32, None);
    let compiled_y_f32 = astype(&compiled[0], MlxDtype::Float32, None);
    mlx_sys::eval(&[
        &eager_y_f32,
        &compiled_y_f32,
        &eager_state,
        &compiled[1],
        &eager_checkpoint,
        &compiled[2],
    ]);
    assert_eq!(compiled_y_f32.data_f32(), eager_y_f32.data_f32());
    assert_eq!(compiled[1].data_f32(), eager_state.data_f32());
    assert_eq!(compiled[2].data_f32(), eager_checkpoint.data_f32());

    let (tape_y, tape_final, tape) =
        gated_delta_kernel_with_tape(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state)
            .expect("short verifier tape kernel must accept the test shape");
    let prefix = |array: &MlxArray, keep: i32| {
        let shape = array.shape();
        let starts = vec![0; shape.len()];
        let mut stops = shape;
        stops[1] = keep;
        let strides = vec![1; stops.len()];
        slice(array, &starts, &stops, &strides, None)
    };
    let k1 = prefix(&k, 1);
    let a1 = prefix(&a_raw, 1);
    let tape1 = prefix(&tape, 1);
    let replay1 = replay_gated_delta_tape(&k1, &a_log, &a1, &dt_bias, &tape1, &state, 1)
        .expect("one-row tape replay");
    let q2 = prefix(&q, 2);
    let k2 = prefix(&k, 2);
    let v2 = prefix(&v, 2);
    let a2 = prefix(&a_raw, 2);
    let b2 = prefix(&b_raw, 2);
    let tape2 = prefix(&tape, 2);
    let (_, eager2, _) = gated_delta_kernel_with_prefix_checkpoint(
        &q2, &k2, &v2, &a_log, &a2, &dt_bias, &b2, &state, 1,
    );
    let replay2 = replay_gated_delta_tape(&k2, &a_log, &a2, &dt_bias, &tape2, &state, 2)
        .expect("two-row tape replay");
    let replay3 =
        replay_gated_delta_tape(&k, &a_log, &a_raw, &dt_bias, &tape, &state, SEQ as usize)
            .expect("full tape replay");
    let tape_y_f32 = astype(&tape_y, MlxDtype::Float32, None);
    mlx_sys::eval(&[
        &tape_y_f32,
        &tape_final,
        &replay1,
        &replay2,
        &replay3,
        &eager_checkpoint,
        &eager2,
        &eager_state,
    ]);
    assert_eq!(tape_y_f32.data_f32(), eager_y_f32.data_f32());
    assert_eq!(tape_final.data_f32(), eager_state.data_f32());
    assert_eq!(replay1.data_f32(), eager_checkpoint.data_f32());
    assert_eq!(replay2.data_f32(), eager2.data_f32());
    assert_eq!(replay3.data_f32(), eager_state.data_f32());
}

#[test]
fn normalize_linear_attention_qk_preserves_reference_shapes() {
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 1,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 4,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let q = zeros(&[1, 2, 1, 32], MlxDtype::Bfloat16, None);
    let k = zeros(&[1, 2, 1, 32], MlxDtype::Bfloat16, None);

    let (q, k) = normalize_linear_attention_qk(&cfg, &q, &k, 1e-6);

    assert_eq!(q.shape(), vec![1, 2, 1, 32]);
    assert_eq!(k.shape(), vec![1, 2, 1, 32]);
    assert_eq!(q.dtype(), MlxDtype::Bfloat16);
    assert_eq!(k.dtype(), MlxDtype::Bfloat16);
}

#[test]
fn decode_post_input_metal_matches_portable_composition_for_short_sequences() {
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 32,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let conv_dim = cfg.conv_dim();
    let state_data: Vec<f32> = (0..3 * conv_dim)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.005)
        .collect();
    let weight_data: Vec<f32> = (0..conv_dim * cfg.conv_kernel_dim)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
        .collect();
    let weight = f32_array(
        &weight_data,
        &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
    );

    for seq in 1..=4 {
        let qkv_data: Vec<f32> = (0..seq * conv_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
            .collect();
        let qkv = f32_array(&qkv_data, &[1, seq as i32, conv_dim as i32]);
        let state = f32_array(&state_data, &[1, 3, conv_dim as i32]);
        let (conv_out, portable_state) = linear_attention_conv1d(&cfg, &qkv, &weight, Some(&state));
        let split = split_linear_attention_qkv(&cfg, &conv_out);
        let (portable_q, portable_k) =
            normalize_linear_attention_qk(&cfg, &split.q, &split.k, 1e-6);
        let (metal_q, metal_k, metal_v, metal_state, _prefix) =
            linear_attention_decode_post_input_metal(
                &cfg,
                &qkv,
                &weight,
                Some(&state),
                q_scale,
                k_scale,
                1e-6,
            )
            .expect("decode post-input Metal path should accept Qwen-like shape");
        let portable_q = mlx_sys::contiguous(&portable_q, None);
        let portable_k = mlx_sys::contiguous(&portable_k, None);
        let portable_v = mlx_sys::contiguous(&split.v, None);
        let portable_state = mlx_sys::contiguous(&portable_state, None);
        let metal_q = mlx_sys::contiguous(&metal_q, None);
        let metal_k = mlx_sys::contiguous(&metal_k, None);
        let metal_v = mlx_sys::contiguous(&metal_v, None);
        let metal_state = mlx_sys::contiguous(&metal_state, None);
        mlx_sys::eval(&[
            &portable_q,
            &portable_k,
            &portable_v,
            &portable_state,
            &metal_q,
            &metal_k,
            &metal_v,
            &metal_state,
        ]);

        assert_close(
            &format!("decode_post_input_q_seq{seq}"),
            metal_q.data_f32(),
            portable_q.data_f32(),
            1e-5,
        );
        assert_close(
            &format!("decode_post_input_k_seq{seq}"),
            metal_k.data_f32(),
            portable_k.data_f32(),
            1e-5,
        );
        assert_close(
            &format!("decode_post_input_v_seq{seq}"),
            metal_v.data_f32(),
            portable_v.data_f32(),
            1e-5,
        );
        assert_close(
            &format!("decode_post_input_state_seq{seq}"),
            metal_state.data_f32(),
            portable_state.data_f32(),
            1e-6,
        );
    }
}

fn post_input_rounding_fixture(
    seq: usize,
    dtype: MlxDtype,
) -> (LinearAttentionConfig, MlxArray, MlxArray, MlxArray) {
    let (q_scale, k_scale) = linear_attention_qk_scale(128);
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 3,
        num_key_heads: 1,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let conv_dim = cfg.conv_dim();
    // Small dyadic products and sums are exact, isolating activation and
    // normalization rounding from convolution accumulation order.
    let state_data: Vec<f32> = (0..3 * conv_dim)
        .map(|idx| ((idx % 13) as f32 - 6.0) / 16.0)
        .collect();
    let weight_data: Vec<f32> = (0..conv_dim * cfg.conv_kernel_dim)
        .map(|idx| ((idx % 7) as f32 - 3.0) / 8.0)
        .collect();
    let weight = astype(
        &f32_array(
            &weight_data,
            &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
        ),
        dtype,
        None,
    );
    let qkv_data: Vec<f32> = (0..seq * conv_dim)
        .map(|idx| ((idx % 17) as f32 - 8.0) / 16.0)
        .collect();
    let qkv = astype(
        &f32_array(&qkv_data, &[1, seq as i32, conv_dim as i32]),
        dtype,
        None,
    );
    let state = astype(
        &f32_array(&state_data, &[1, 3, conv_dim as i32]),
        dtype,
        None,
    );
    (cfg, qkv, weight, state)
}

fn assert_post_input_rounding_boundaries(dtype: MlxDtype) {
    let mut mismatches = Vec::new();

    for (seq, target_verify) in [(1, false), (4, false), (1, true), (4, true)] {
        let _target = fastpath::scoped_qwen_linear_mtp_target_verify(target_verify);
        let (cfg, qkv, weight, state) = post_input_rounding_fixture(seq, dtype);
        let (conv_out, portable_state) = linear_attention_conv1d(&cfg, &qkv, &weight, Some(&state));
        let split = split_linear_attention_qkv(&cfg, &conv_out);
        let (portable_q, portable_k) =
            normalize_linear_attention_qk(&cfg, &split.q, &split.k, 1e-6);
        let (metal_q, metal_k, metal_v, metal_state, _) = linear_attention_decode_post_input_metal(
            &cfg,
            &qkv,
            &weight,
            Some(&state),
            cfg.q_scale,
            cfg.k_scale,
            1e-6,
        )
        .expect("post-input rounding fixture must use the Metal path");
        for (label, actual, expected) in [
            ("q", metal_q, portable_q),
            ("k", metal_k, portable_k),
            ("v", metal_v, split.v),
            ("conv_state", metal_state, portable_state),
        ] {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(expected.dtype(), dtype);
            let actual = astype(&contiguous(&actual, None), MlxDtype::Float32, None);
            let expected = astype(&contiguous(&expected, None), MlxDtype::Float32, None);
            mlx_sys::eval(&[&actual, &expected]);
            let unequal = actual
                .data_f32()
                .iter()
                .zip(expected.data_f32())
                .filter(|(a, b)| a != b)
                .count();
            if unequal != 0 {
                mismatches.push(format!(
                    "seq={seq} target_verify={target_verify} {label}: {unequal} unequal elements"
                ));
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("; "));
}

#[test]
fn decode_post_input_metal_preserves_bf16_rounding_boundaries() {
    assert_post_input_rounding_boundaries(MlxDtype::Bfloat16);
}

#[test]
fn decode_post_input_metal_preserves_fp16_rounding_boundaries() {
    assert_post_input_rounding_boundaries(MlxDtype::Float16);
}

#[test]
fn fused_gated_delta_verify_bf16_128d_matches_portable_prework() {
    for seq in [2, 4] {
        let (cfg, qkv, weight, conv_state) = post_input_rounding_fixture(seq, MlxDtype::Bfloat16);
        let heads = cfg.num_value_heads;
        let a_log = f32_array(&[-0.25, -0.125, 0.0], &[heads as i32]);
        let dt_bias = f32_array(&[0.0625, 0.0, -0.03125], &[heads as i32]);
        let ab_data: Vec<f32> = (0..seq * heads)
            .map(|idx| ((idx % 9) as f32 - 4.0) / 16.0)
            .collect();
        let a_raw = astype(
            &f32_array(&ab_data, &[1, seq as i32, heads as i32]),
            MlxDtype::Bfloat16,
            None,
        );
        let b_raw = astype(
            &f32_array(
                &ab_data.iter().rev().copied().collect::<Vec<_>>(),
                &[1, seq as i32, heads as i32],
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let recurrent_state = f32_array(
            &(0..heads * 128 * 128)
                .map(|idx| ((idx % 19) as f32 - 9.0) / 512.0)
                .collect::<Vec<_>>(),
            &[1, heads as i32, 128, 128],
        );
        let (conv_out, expected_conv) =
            linear_attention_conv1d(&cfg, &qkv, &weight, Some(&conv_state));
        let split = split_linear_attention_qkv(&cfg, &conv_out);
        let (q, k) = normalize_linear_attention_qk(&cfg, &split.q, &split.k, 1e-6);
        let (expected_y, expected_state, expected_checkpoint) =
            gated_delta_kernel_with_prefix_checkpoint(
                &q,
                &k,
                &split.v,
                &a_log,
                &a_raw,
                &dt_bias,
                &b_raw,
                &recurrent_state,
                1,
            );
        let first_qkv = slice(
            &qkv,
            &[0, 0, 0],
            &[1, 1, cfg.conv_dim() as i32],
            &[1, 1, 1],
            None,
        );
        let (_, expected_prefix_conv) =
            linear_attention_conv1d(&cfg, &first_qkv, &weight, Some(&conv_state));
        let (y, state, checkpoint, conv, prefix_conv) = gated_delta_fused_verify_from_qkv(
            &cfg,
            &qkv,
            &weight,
            Some(&conv_state),
            &a_log,
            &a_raw,
            &dt_bias,
            &b_raw,
            &recurrent_state,
            cfg.q_scale,
            cfg.k_scale,
            1e-6,
        )
        .expect("BF16 D128 checkpoint verifier must execute");
        let (no_checkpoint_y, no_checkpoint_state, no_checkpoint_conv) =
            gated_delta_fused_verify_no_checkpoint_from_qkv(
                &cfg,
                &qkv,
                &weight,
                Some(&conv_state),
                &a_log,
                &a_raw,
                &dt_bias,
                &b_raw,
                &recurrent_state,
                cfg.q_scale,
                cfg.k_scale,
                1e-6,
            )
            .expect("BF16 D128 no-checkpoint verifier must execute");
        for (label, actual, expected) in [
            ("y", &y, &expected_y),
            ("state", &state, &expected_state),
            ("checkpoint", &checkpoint, &expected_checkpoint),
            ("conv", &conv, &expected_conv),
            ("prefix_conv", &prefix_conv, &expected_prefix_conv),
            ("no_checkpoint_y", &no_checkpoint_y, &expected_y),
            ("no_checkpoint_state", &no_checkpoint_state, &expected_state),
            ("no_checkpoint_conv", &no_checkpoint_conv, &expected_conv),
        ] {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.dtype(), expected.dtype());
            let actual = astype(&contiguous(actual, None), MlxDtype::Float32, None);
            let expected = astype(&contiguous(expected, None), MlxDtype::Float32, None);
            mlx_sys::eval(&[&actual, &expected]);
            assert_close(
                &format!("fused_bf16_portable_seq{seq}_{label}"),
                actual.data_f32(),
                expected.data_f32(),
                0.0,
            );
        }
    }
}

#[test]
fn fused_gated_delta_verify_matches_two_kernel_composition() {
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 32,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let conv_dim = cfg.conv_dim();
    let conv_state_data: Vec<f32> = (0..3 * conv_dim)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.005)
        .collect();
    let recurrent_data: Vec<f32> = (0..cfg.num_value_heads * 32 * 32)
        .map(|idx| ((idx % 19) as f32 - 9.0) * 0.002)
        .collect();
    let weight_data: Vec<f32> = (0..conv_dim * cfg.conv_kernel_dim)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
        .collect();
    let weight = f32_array(
        &weight_data,
        &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
    );
    let conv_state = f32_array(&conv_state_data, &[1, 3, conv_dim as i32]);
    let recurrent_state = f32_array(&recurrent_data, &[1, 2, 32, 32]);
    let a_log = f32_array(&[-0.2, -0.15], &[2]);
    let dt_bias = f32_array(&[0.05, -0.02], &[2]);

    for seq in 2..=4 {
        let qkv_data: Vec<f32> = (0..seq * conv_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
            .collect();
        let ab_data: Vec<f32> = (0..seq * cfg.num_value_heads)
            .map(|idx| ((idx % 9) as f32 - 4.0) * 0.03)
            .collect();
        let qkv = f32_array(&qkv_data, &[1, seq as i32, conv_dim as i32]);
        let a_raw = f32_array(&ab_data, &[1, seq as i32, 2]);
        let b_data: Vec<f32> = ab_data.iter().rev().copied().collect();
        let b_raw = f32_array(&b_data, &[1, seq as i32, 2]);

        let (q, k, v, expected_conv, expected_prefix_conv) =
            linear_attention_decode_post_input_metal(
                &cfg,
                &qkv,
                &weight,
                Some(&conv_state),
                q_scale,
                k_scale,
                1e-6,
            )
            .expect("two-kernel post-input must accept the verifier shape");
        let (expected_y, expected_state, expected_checkpoint) =
            gated_delta_kernel_with_prefix_checkpoint(
                &q,
                &k,
                &v,
                &a_log,
                &a_raw,
                &dt_bias,
                &b_raw,
                &recurrent_state,
                1,
            );
        let (actual_y, actual_state, actual_checkpoint, actual_conv, actual_prefix_conv) =
            gated_delta_fused_verify_from_qkv(
                &cfg,
                &qkv,
                &weight,
                Some(&conv_state),
                &a_log,
                &a_raw,
                &dt_bias,
                &b_raw,
                &recurrent_state,
                q_scale,
                k_scale,
                1e-6,
            )
            .expect("fused verifier must accept the Qwen-like shape");
        mlx_sys::eval(&[
            &expected_y,
            &expected_state,
            &expected_checkpoint,
            &expected_conv,
            &expected_prefix_conv,
            &actual_y,
            &actual_state,
            &actual_checkpoint,
            &actual_conv,
            &actual_prefix_conv,
        ]);
        assert_close(
            &format!("fused_verify_y_seq{seq}"),
            actual_y.data_f32(),
            expected_y.data_f32(),
            1e-6,
        );
        assert_close(
            &format!("fused_verify_state_seq{seq}"),
            actual_state.data_f32(),
            expected_state.data_f32(),
            1e-6,
        );
        assert_close(
            &format!("fused_verify_checkpoint_seq{seq}"),
            actual_checkpoint.data_f32(),
            expected_checkpoint.data_f32(),
            1e-6,
        );
        assert_eq!(actual_conv.data_f32(), expected_conv.data_f32());
        assert_eq!(
            actual_prefix_conv.data_f32(),
            expected_prefix_conv.data_f32()
        );
    }
}

#[test]
fn fused_gated_delta_verify_bf16_matches_two_kernel_composition() {
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 32,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let seq = 3_i32;
    let conv_dim = cfg.conv_dim() as i32;
    let bf16 = |array: MlxArray| astype(&array, MlxDtype::Bfloat16, None);
    let qkv = bf16(f32_array(
        &(0..seq * conv_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
            .collect::<Vec<_>>(),
        &[1, seq, conv_dim],
    ));
    let conv_state = bf16(f32_array(
        &(0..3 * conv_dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.005)
            .collect::<Vec<_>>(),
        &[1, 3, conv_dim],
    ));
    let weight = bf16(f32_array(
        &(0..conv_dim * cfg.conv_kernel_dim as i32)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
            .collect::<Vec<_>>(),
        &[conv_dim, cfg.conv_kernel_dim as i32, 1],
    ));
    let a_raw = bf16(f32_array(
        &(0..seq * 2)
            .map(|idx| ((idx % 9) as f32 - 4.0) * 0.03)
            .collect::<Vec<_>>(),
        &[1, seq, 2],
    ));
    let b_raw = bf16(f32_array(
        &(0..seq * 2)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.025)
            .collect::<Vec<_>>(),
        &[1, seq, 2],
    ));
    let a_log = f32_array(&[-0.2, -0.15], &[2]);
    let dt_bias = f32_array(&[0.05, -0.02], &[2]);
    let recurrent_state = f32_array(
        &(0..2 * 32 * 32)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.002)
            .collect::<Vec<_>>(),
        &[1, 2, 32, 32],
    );

    let (q, k, v, expected_conv, expected_prefix_conv) = linear_attention_decode_post_input_metal(
        &cfg,
        &qkv,
        &weight,
        Some(&conv_state),
        q_scale,
        k_scale,
        1e-6,
    )
    .expect("BF16 post-input composition");
    let (expected_y, expected_state, expected_checkpoint) =
        gated_delta_kernel_with_prefix_checkpoint(
            &q,
            &k,
            &v,
            &a_log,
            &a_raw,
            &dt_bias,
            &b_raw,
            &recurrent_state,
            1,
        );
    let (actual_y, actual_state, actual_checkpoint, actual_conv, actual_prefix_conv) =
        gated_delta_fused_verify_from_qkv(
            &cfg,
            &qkv,
            &weight,
            Some(&conv_state),
            &a_log,
            &a_raw,
            &dt_bias,
            &b_raw,
            &recurrent_state,
            q_scale,
            k_scale,
            1e-6,
        )
        .expect("BF16 fused verifier");
    let outputs = [
        astype(&expected_y, MlxDtype::Float32, None),
        expected_state,
        expected_checkpoint,
        astype(&expected_conv, MlxDtype::Float32, None),
        astype(&expected_prefix_conv, MlxDtype::Float32, None),
        astype(&actual_y, MlxDtype::Float32, None),
        actual_state,
        actual_checkpoint,
        astype(&actual_conv, MlxDtype::Float32, None),
        astype(&actual_prefix_conv, MlxDtype::Float32, None),
    ];
    let refs: Vec<&MlxArray> = outputs.iter().collect();
    mlx_sys::eval(&refs);
    for (name, expected, actual) in [
        ("y", &outputs[0], &outputs[5]),
        ("state", &outputs[1], &outputs[6]),
        ("checkpoint", &outputs[2], &outputs[7]),
        ("conv", &outputs[3], &outputs[8]),
        ("prefix_conv", &outputs[4], &outputs[9]),
    ] {
        assert_close(
            &format!("fused_verify_bf16_{name}"),
            actual.data_f32(),
            expected.data_f32(),
            2e-5,
        );
    }
}

#[test]
fn fused_gated_delta_verify_no_checkpoint_matches_checkpoint_kernel() {
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 32,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let conv_dim = cfg.conv_dim();
    let conv_state = f32_array(
        &(0..3 * conv_dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.005)
            .collect::<Vec<_>>(),
        &[1, 3, conv_dim as i32],
    );
    let recurrent_state = f32_array(
        &(0..cfg.num_value_heads * 32 * 32)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.002)
            .collect::<Vec<_>>(),
        &[1, 2, 32, 32],
    );
    let weight = f32_array(
        &(0..conv_dim * cfg.conv_kernel_dim)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
            .collect::<Vec<_>>(),
        &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
    );
    let a_log = f32_array(&[-0.2, -0.15], &[2]);
    let dt_bias = f32_array(&[0.05, -0.02], &[2]);
    for seq in 2..=4 {
        let qkv = f32_array(
            &(0..seq * conv_dim)
                .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
                .collect::<Vec<_>>(),
            &[1, seq as i32, conv_dim as i32],
        );
        let ab_data: Vec<f32> = (0..seq * 2)
            .map(|idx| ((idx % 9) as f32 - 4.0) * 0.03)
            .collect();
        let a_raw = f32_array(&ab_data, &[1, seq as i32, 2]);
        let b_data: Vec<f32> = ab_data.iter().rev().copied().collect();
        let b_raw = f32_array(&b_data, &[1, seq as i32, 2]);
        let (expected_y, expected_state, _checkpoint, expected_conv, _prefix_conv) =
            gated_delta_fused_verify_from_qkv(
                &cfg,
                &qkv,
                &weight,
                Some(&conv_state),
                &a_log,
                &a_raw,
                &dt_bias,
                &b_raw,
                &recurrent_state,
                q_scale,
                k_scale,
                1e-6,
            )
            .expect("checkpoint fused verifier");
        let (actual_y, actual_state, actual_conv) =
            gated_delta_fused_verify_no_checkpoint_from_qkv(
                &cfg,
                &qkv,
                &weight,
                Some(&conv_state),
                &a_log,
                &a_raw,
                &dt_bias,
                &b_raw,
                &recurrent_state,
                q_scale,
                k_scale,
                1e-6,
            )
            .expect("no-checkpoint fused verifier");
        mlx_sys::eval(&[
            &expected_y,
            &expected_state,
            &expected_conv,
            &actual_y,
            &actual_state,
            &actual_conv,
        ]);
        assert_close(
            &format!("fused_verify_no_ckpt_y_seq{seq}"),
            actual_y.data_f32(),
            expected_y.data_f32(),
            1e-6,
        );
        assert_close(
            &format!("fused_verify_no_ckpt_state_seq{seq}"),
            actual_state.data_f32(),
            expected_state.data_f32(),
            1e-6,
        );
        assert_eq!(actual_conv.data_f32(), expected_conv.data_f32());
    }
    assert!(
        gated_delta_fused_verify_no_checkpoint_from_qkv(
            &cfg,
            &f32_array(
                &(0..conv_dim)
                    .map(|idx| (idx as f32) * 0.01)
                    .collect::<Vec<_>>(),
                &[1, 1, conv_dim as i32],
            ),
            &weight,
            Some(&conv_state),
            &a_log,
            &f32_array(&[0.1, -0.1], &[1, 1, 2]),
            &dt_bias,
            &f32_array(&[0.2, -0.2], &[1, 1, 2]),
            &recurrent_state,
            q_scale,
            k_scale,
            1e-6,
        )
        .is_none(),
        "seq=1 must fall through"
    );
}

#[test]
fn fused_gated_delta_verify_no_checkpoint_bf16_and_128d_match() {
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let cfg32 = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 32,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let seq = 3_i32;
    let conv_dim = cfg32.conv_dim() as i32;
    let bf16 = |array: MlxArray| astype(&array, MlxDtype::Bfloat16, None);
    let qkv = bf16(f32_array(
        &(0..seq * conv_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
            .collect::<Vec<_>>(),
        &[1, seq, conv_dim],
    ));
    let conv_state = bf16(f32_array(
        &(0..3 * conv_dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.005)
            .collect::<Vec<_>>(),
        &[1, 3, conv_dim],
    ));
    let weight = bf16(f32_array(
        &(0..conv_dim * cfg32.conv_kernel_dim as i32)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
            .collect::<Vec<_>>(),
        &[conv_dim, cfg32.conv_kernel_dim as i32, 1],
    ));
    let a_raw = bf16(f32_array(
        &(0..seq * 2)
            .map(|idx| ((idx % 9) as f32 - 4.0) * 0.03)
            .collect::<Vec<_>>(),
        &[1, seq, 2],
    ));
    let b_raw = bf16(f32_array(
        &(0..seq * 2)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.025)
            .collect::<Vec<_>>(),
        &[1, seq, 2],
    ));
    let a_log = f32_array(&[-0.2, -0.15], &[2]);
    let dt_bias = f32_array(&[0.05, -0.02], &[2]);
    let recurrent_state = f32_array(
        &(0..2 * 32 * 32)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.002)
            .collect::<Vec<_>>(),
        &[1, 2, 32, 32],
    );
    let (expected_y, expected_state, _, expected_conv, _) = gated_delta_fused_verify_from_qkv(
        &cfg32,
        &qkv,
        &weight,
        Some(&conv_state),
        &a_log,
        &a_raw,
        &dt_bias,
        &b_raw,
        &recurrent_state,
        q_scale,
        k_scale,
        1e-6,
    )
    .expect("bf16 checkpoint fused verifier");
    let (actual_y, actual_state, actual_conv) = gated_delta_fused_verify_no_checkpoint_from_qkv(
        &cfg32,
        &qkv,
        &weight,
        Some(&conv_state),
        &a_log,
        &a_raw,
        &dt_bias,
        &b_raw,
        &recurrent_state,
        q_scale,
        k_scale,
        1e-6,
    )
    .expect("bf16 no-checkpoint fused verifier");
    let outputs = [
        astype(&expected_y, MlxDtype::Float32, None),
        expected_state,
        astype(&expected_conv, MlxDtype::Float32, None),
        astype(&actual_y, MlxDtype::Float32, None),
        actual_state,
        astype(&actual_conv, MlxDtype::Float32, None),
    ];
    let refs: Vec<&MlxArray> = outputs.iter().collect();
    mlx_sys::eval(&refs);
    for (name, expected, actual) in [
        ("y", &outputs[0], &outputs[3]),
        ("state", &outputs[1], &outputs[4]),
        ("conv", &outputs[2], &outputs[5]),
    ] {
        assert_close(
            &format!("fused_verify_no_ckpt_bf16_{name}"),
            actual.data_f32(),
            expected.data_f32(),
            2e-5,
        );
    }

    let (q_scale, k_scale) = linear_attention_qk_scale(128);
    let cfg128 = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let conv_dim = cfg128.conv_dim();
    let qkv = f32_array(
        &(0..3 * conv_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
            .collect::<Vec<_>>(),
        &[1, 3, conv_dim as i32],
    );
    let conv_state = f32_array(
        &(0..3 * conv_dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.005)
            .collect::<Vec<_>>(),
        &[1, 3, conv_dim as i32],
    );
    let weight = f32_array(
        &(0..conv_dim * 4)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
            .collect::<Vec<_>>(),
        &[conv_dim as i32, 4, 1],
    );
    let a_raw = f32_array(
        &(0..6)
            .map(|idx| ((idx % 9) as f32 - 4.0) * 0.03)
            .collect::<Vec<_>>(),
        &[1, 3, 2],
    );
    let b_raw = f32_array(
        &(0..6)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.025)
            .collect::<Vec<_>>(),
        &[1, 3, 2],
    );
    let recurrent_state = f32_array(
        &(0..2 * 128 * 128)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.002)
            .collect::<Vec<_>>(),
        &[1, 2, 128, 128],
    );
    let (expected_y, expected_state, _, expected_conv, _) = gated_delta_fused_verify_from_qkv(
        &cfg128,
        &qkv,
        &weight,
        Some(&conv_state),
        &a_log,
        &a_raw,
        &dt_bias,
        &b_raw,
        &recurrent_state,
        q_scale,
        k_scale,
        1e-6,
    )
    .expect("128d checkpoint fused verifier");
    let (actual_y, actual_state, actual_conv) = gated_delta_fused_verify_no_checkpoint_from_qkv(
        &cfg128,
        &qkv,
        &weight,
        Some(&conv_state),
        &a_log,
        &a_raw,
        &dt_bias,
        &b_raw,
        &recurrent_state,
        q_scale,
        k_scale,
        1e-6,
    )
    .expect("128d no-checkpoint fused verifier");
    mlx_sys::eval(&[
        &expected_y,
        &expected_state,
        &expected_conv,
        &actual_y,
        &actual_state,
        &actual_conv,
    ]);
    assert_close(
        "fused_verify_no_ckpt_y_128d",
        actual_y.data_f32(),
        expected_y.data_f32(),
        1e-5,
    );
    assert_close(
        "fused_verify_no_ckpt_state_128d",
        actual_state.data_f32(),
        expected_state.data_f32(),
        1e-5,
    );
    assert_eq!(actual_conv.data_f32(), expected_conv.data_f32());
}

#[test]
fn prefill_post_input_metal_matches_portable_seq8_cold_start() {
    // p2048 first chunk has no cached conv state. Metal must match
    // portable conv1d cold start at a multi-token prefill shape.
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 32,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let conv_dim = cfg.conv_dim();
    let seq = 8_i32;
    let qkv_data: Vec<f32> = (0..seq as usize * conv_dim)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
        .collect();
    let weight_data: Vec<f32> = (0..conv_dim * cfg.conv_kernel_dim)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
        .collect();
    let qkv = f32_array(&qkv_data, &[1, seq, conv_dim as i32]);
    let weight = f32_array(
        &weight_data,
        &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
    );
    let (conv_out, portable_state) = linear_attention_conv1d(&cfg, &qkv, &weight, None);
    let split = split_linear_attention_qkv(&cfg, &conv_out);
    let (portable_q, portable_k) = normalize_linear_attention_qk(&cfg, &split.q, &split.k, 1e-6);
    let (metal_q, metal_k, metal_v, metal_state, _prefix) =
        linear_attention_decode_post_input_metal(&cfg, &qkv, &weight, None, q_scale, k_scale, 1e-6)
            .expect("prefill post-input Metal path should accept seq=8 cold start");
    let portable_q = mlx_sys::contiguous(&portable_q, None);
    let portable_k = mlx_sys::contiguous(&portable_k, None);
    let portable_v = mlx_sys::contiguous(&split.v, None);
    let portable_state = mlx_sys::contiguous(&portable_state, None);
    let metal_q = mlx_sys::contiguous(&metal_q, None);
    let metal_k = mlx_sys::contiguous(&metal_k, None);
    let metal_v = mlx_sys::contiguous(&metal_v, None);
    let metal_state = mlx_sys::contiguous(&metal_state, None);
    mlx_sys::eval(&[
        &portable_q,
        &portable_k,
        &portable_v,
        &portable_state,
        &metal_q,
        &metal_k,
        &metal_v,
        &metal_state,
    ]);
    assert_close(
        "prefill_post_input_q",
        metal_q.data_f32(),
        portable_q.data_f32(),
        2e-5,
    );
    assert_close(
        "prefill_post_input_k",
        metal_k.data_f32(),
        portable_k.data_f32(),
        2e-5,
    );
    assert_close(
        "prefill_post_input_v",
        metal_v.data_f32(),
        portable_v.data_f32(),
        2e-5,
    );
    assert_close(
        "prefill_post_input_state",
        metal_state.data_f32(),
        portable_state.data_f32(),
        1e-6,
    );
}

#[test]
fn decode_post_input_metal_prefix_conv_matches_first_token_tail() {
    let (q_scale, k_scale) = linear_attention_qk_scale(32);
    let cfg = LinearAttentionConfig {
        full_attention_interval: 4,
        num_value_heads: 2,
        num_key_heads: 1,
        key_head_dim: 32,
        value_head_dim: 32,
        conv_kernel_dim: 4,
        q_scale,
        k_scale,
    };
    let conv_dim = cfg.conv_dim();
    let seq = 2i32;
    let qkv_data: Vec<f32> = (0..(seq as usize) * conv_dim)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
        .collect();
    let state_data: Vec<f32> = (0..3 * conv_dim)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.02)
        .collect();
    let weight_data: Vec<f32> = (0..conv_dim * cfg.conv_kernel_dim)
        .map(|idx| ((idx % 9) as f32 - 4.0) * 0.03)
        .collect();
    let qkv = f32_array(&qkv_data, &[1, seq, conv_dim as i32]);
    let state = f32_array(&state_data, &[1, 3, conv_dim as i32]);
    let weight = f32_array(
        &weight_data,
        &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
    );
    let (_q, _k, _v, _new_state, prefix) = linear_attention_decode_post_input_metal(
        &cfg,
        &qkv,
        &weight,
        Some(&state),
        q_scale,
        k_scale,
        1e-6,
    )
    .expect("post-input Metal must emit prefix conv");
    let cached_tail = contiguous(
        &slice(
            &state,
            &[0, 1, 0],
            &[1, 3, conv_dim as i32],
            &[1, 1, 1],
            None,
        ),
        None,
    );
    let first = contiguous(
        &slice(&qkv, &[0, 0, 0], &[1, 1, conv_dim as i32], &[1, 1, 1], None),
        None,
    );
    let expected = concatenate(&[&cached_tail, &first], 1, None);
    mlx_sys::eval(&[&prefix, &expected]);
    assert_eq!(prefix.data_f32(), expected.data_f32());
}

#[test]
fn normalize_linear_attention_qk_q_uses_inv_scale_squared() {
    // mlx-lm/Swift: q_scale = Dk^(-1), k_scale = Dk^(-0.5)
    let (q_scale, k_scale) = linear_attention_qk_scale(4);

    assert!((q_scale - 0.25).abs() < f32::EPSILON, "q_scale={q_scale}");
    assert!((k_scale - 0.5).abs() < f32::EPSILON, "k_scale={k_scale}");
}

#[test]
fn rms_norm_gate_metal_matches_direct_chain_for_bf16() {
    let normed_data: Vec<f32> = (0..16)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.125)
        .collect();
    let gate_data: Vec<f32> = (0..16).map(|idx| ((idx % 5) as f32 - 2.0) * 0.25).collect();
    let normed = astype(
        &f32_array(&normed_data, &[1, 2, 2, 4]),
        MlxDtype::Bfloat16,
        None,
    );
    let gate = astype(
        &f32_array(&gate_data, &[1, 2, 2, 4]),
        MlxDtype::Bfloat16,
        None,
    );
    let direct = astype(
        &multiply(
            &mlx_sys::ops::silu(&astype(&gate, MlxDtype::Float32, None), None),
            &astype(&normed, MlxDtype::Float32, None),
            None,
        ),
        MlxDtype::Bfloat16,
        None,
    );
    let metal = rms_norm_gate_metal_impl(&normed, &gate, MlxDtype::Bfloat16)
        .expect("bf16 linear-attention RMSNorm gate Metal fast path");
    let direct = astype(&direct, MlxDtype::Float32, None);
    let metal = astype(&metal, MlxDtype::Float32, None);
    mlx_sys::eval(&[&direct, &metal]);

    assert_close("rms_norm_gate", metal.data_f32(), direct.data_f32(), 2.0e-2);
}

#[test]
fn rms_norm_full_gate_metal_matches_direct_chain_for_bf16() {
    let hidden_data: Vec<f32> = (0..16)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.125)
        .collect();
    let gate_data: Vec<f32> = (0..16).map(|idx| ((idx % 5) as f32 - 2.0) * 0.25).collect();
    let weight_data = vec![0.8_f32, 1.0, 1.2, 1.4];
    let hidden = astype(
        &f32_array(&hidden_data, &[1, 2, 2, 4]),
        MlxDtype::Bfloat16,
        None,
    );
    let gate = astype(
        &f32_array(&gate_data, &[1, 2, 2, 4]),
        MlxDtype::Bfloat16,
        None,
    );
    let weight = astype(&f32_array(&weight_data, &[4]), MlxDtype::Bfloat16, None);

    let normed = rms_norm(&hidden, Some(&weight), 1e-6, None);
    let direct = astype(
        &multiply(
            &mlx_sys::ops::silu(&astype(&gate, MlxDtype::Float32, None), None),
            &astype(&normed, MlxDtype::Float32, None),
            None,
        ),
        MlxDtype::Bfloat16,
        None,
    );
    let metal = rms_norm_full_gate_metal_impl(&hidden, &gate, &weight, 1e-6)
        .expect("bf16 linear-attention full RMSNorm gate Metal fast path");
    let direct = astype(&direct, MlxDtype::Float32, None);
    let metal = astype(&metal, MlxDtype::Float32, None);
    mlx_sys::eval(&[&direct, &metal]);

    assert_close(
        "rms_norm_full_gate",
        metal.data_f32(),
        direct.data_f32(),
        2.0e-2,
    );
}

#[test]
fn rms_norm_gate_metal_rejects_shape_mismatch() {
    let normed = zeros(&[1, 2, 2, 4], MlxDtype::Bfloat16, None);
    let gate = zeros(&[1, 2, 1, 4], MlxDtype::Bfloat16, None);

    assert!(rms_norm_gate_metal_impl(&normed, &gate, MlxDtype::Bfloat16).is_none());
}

#[test]
fn rms_norm_gated_preserves_hidden_shape_and_dtype() {
    let hidden = zeros(&[1, 5, 2, 3], MlxDtype::Bfloat16, None);
    let gate = zeros(&[1, 5, 2, 3], MlxDtype::Bfloat16, None);
    let weight = zeros(&[3], MlxDtype::Bfloat16, None);

    let out = rms_norm_gated(&hidden, &gate, &weight, 1e-6);

    assert_eq!(out.shape(), vec![1, 5, 2, 3]);
    assert_eq!(out.dtype(), MlxDtype::Bfloat16);
}

#[test]
fn rms_norm_gate_metal_is_bit_exact_vs_portable_silu_mul() {
    let normed_data: Vec<f32> = (0..256)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let gate_data: Vec<f32> = (0..256)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.0625)
        .collect();
    let normed = astype(
        &f32_array(&normed_data, &[1, 2, 2, 64]),
        MlxDtype::Bfloat16,
        None,
    );
    let gate = astype(
        &f32_array(&gate_data, &[1, 2, 2, 64]),
        MlxDtype::Bfloat16,
        None,
    );
    let portable =
        super::portable_rms_norm_gated(&normed, &gate, &ones(&[64], MlxDtype::Bfloat16), 1e-6);
    // Compare only the silu*norm step: Metal gate on already-normed input
    // versus portable's silu*mul (portable also re-applies rms).
    let direct = astype(
        &multiply(
            &mlx_sys::ops::silu(&astype(&gate, MlxDtype::Float32, None), None),
            &astype(&normed, MlxDtype::Float32, None),
            None,
        ),
        MlxDtype::Bfloat16,
        None,
    );
    let metal = rms_norm_gate_metal_impl(&normed, &gate, MlxDtype::Bfloat16)
        .expect("elementwise gate Metal");
    mlx_sys::eval(&[&direct, &metal, &portable]);
    let a = astype(&metal, MlxDtype::Float32, None);
    let b = astype(&direct, MlxDtype::Float32, None);
    mlx_sys::eval(&[&a, &b]);
    let mut max_abs = 0.0f32;
    for (l, r) in a.data_f32().iter().zip(b.data_f32().iter()) {
        max_abs = max_abs.max((l - r).abs());
    }
    eprintln!("elementwise Metal vs portable silu*mul max_abs={max_abs}");
    assert_eq!(
        a.data_f32(),
        b.data_f32(),
        "elementwise Metal silu*mul must be bit-exact vs MLX portable, max_abs={max_abs}"
    );
    let _ = portable;
}

fn max_abs_f32(a: &MlxArray, b: &MlxArray) -> f32 {
    let a = astype(a, MlxDtype::Float32, None);
    let b = astype(b, MlxDtype::Float32, None);
    mlx_sys::eval(&[&a, &b]);
    a.data_f32()
        .iter()
        .zip(b.data_f32().iter())
        .fold(0.0f32, |m, (l, r)| m.max((l - r).abs()))
}

/// Qwen3.8-27B LA gate is `[1, S, Hv=48, Dv=128]` bf16. Factory Metal-on-exact
/// flips trial-2 to `f4b5490d`; this reproduces Metal S=2 vs two Metal S=1
/// and vs the portable rms+silu chain on that shape (plus a wide range).
#[test]
fn rms_norm_gate_metal_s2_27b_matches_s1_rows_and_portable() {
    const HV: i32 = 48;
    const DV: i32 = 128;
    let n = (2 * HV * DV) as usize;
    let hidden_data: Vec<f32> = (0..n)
        .map(|i| {
            let wave = ((i % 97) as f32 - 48.0) * 0.0625;
            let spike = if i % 111 == 0 { 12.0 } else { 0.0 };
            wave + spike
        })
        .collect();
    let gate_data: Vec<f32> = (0..n)
        .map(|i| {
            let wave = ((i % 89) as f32 - 44.0) * 0.078125;
            let spike = if i % 107 == 0 { -15.0 } else { 0.0 };
            wave + spike
        })
        .collect();
    let weight_data: Vec<f32> = (0..DV as usize).map(|i| 0.5 + (i as f32) * 0.004).collect();
    let hidden = astype(
        &f32_array(&hidden_data, &[1, 2, HV, DV]),
        MlxDtype::Bfloat16,
        None,
    );
    let gate = astype(
        &f32_array(&gate_data, &[1, 2, HV, DV]),
        MlxDtype::Bfloat16,
        None,
    );
    let weight = astype(&f32_array(&weight_data, &[DV]), MlxDtype::Bfloat16, None);

    let metal_s2 = rms_norm_full_gate_metal_impl(&hidden, &gate, &weight, 1e-6)
        .expect("full-gate Metal S=2 27B shape");
    let h0 = super::slice_seq_row_4d(&hidden, 0);
    let h1 = super::slice_seq_row_4d(&hidden, 1);
    let g0 = super::slice_seq_row_4d(&gate, 0);
    let g1 = super::slice_seq_row_4d(&gate, 1);
    let m0 =
        rms_norm_full_gate_metal_impl(&h0, &g0, &weight, 1e-6).expect("full-gate Metal S=1 row0");
    let m1 =
        rms_norm_full_gate_metal_impl(&h1, &g1, &weight, 1e-6).expect("full-gate Metal S=1 row1");
    let metal_s1s = concatenate(&[&m0, &m1], 1, None);
    let portable = super::portable_rms_norm_gated(&hidden, &gate, &weight, 1e-6);
    mlx_sys::eval(&[&metal_s2, &metal_s1s, &portable]);

    let vs_s1 = max_abs_f32(&metal_s2, &metal_s1s);
    let vs_port = max_abs_f32(&metal_s2, &portable);
    let s1_vs_port = max_abs_f32(&metal_s1s, &portable);
    eprintln!(
        "full-gate 27B [1,2,48,128] bf16: MetalS2 vs 2xMetalS1 max_abs={vs_s1} \
             MetalS2 vs portable max_abs={vs_port} 2xMetalS1 vs portable max_abs={s1_vs_port}"
    );

    let normed = rms_norm(&hidden, Some(&weight), 1e-6, None);
    let elem_s2 = rms_norm_gate_metal_impl(&normed, &gate, MlxDtype::Bfloat16)
        .expect("elementwise Metal S=2");
    let n0 = super::slice_seq_row_4d(&normed, 0);
    let n1 = super::slice_seq_row_4d(&normed, 1);
    let e0 = rms_norm_gate_metal_impl(&n0, &g0, MlxDtype::Bfloat16).expect("elem S=1 r0");
    let e1 = rms_norm_gate_metal_impl(&n1, &g1, MlxDtype::Bfloat16).expect("elem S=1 r1");
    let elem_s1s = concatenate(&[&e0, &e1], 1, None);
    let elem_port = super::portable_silu_mul_normed(&normed, &gate, MlxDtype::Bfloat16);
    mlx_sys::eval(&[&elem_s2, &elem_s1s, &elem_port]);
    let evs_s1 = max_abs_f32(&elem_s2, &elem_s1s);
    let evs_port = max_abs_f32(&elem_s2, &elem_port);
    eprintln!(
        "elementwise 27B [1,2,48,128] bf16: MetalS2 vs 2xMetalS1 max_abs={evs_s1} \
             MetalS2 vs portable max_abs={evs_port}"
    );

    assert_eq!(
        vs_s1, 0.0,
        "full-gate Metal S=2 must match two Metal S=1 rows, max_abs={vs_s1}"
    );
    assert_eq!(
        evs_s1, 0.0,
        "elementwise Metal S=2 must match two Metal S=1 rows, max_abs={evs_s1}"
    );
    assert_eq!(
        evs_port, 0.0,
        "elementwise Metal must match portable silu*mul on 27B S=2, max_abs={evs_port}"
    );
    assert!(
        vs_port > 0.0,
        "full-gate Metal RMS must stay observably off mx.fast.rms_norm on 27B S=2 \
             (repro of factory Metal-vs-portable); got max_abs={vs_port}"
    );

    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let later = rms_norm_gated_with_full_gate_policy(&hidden, &gate, &weight, 1e-6, false);
    let early = rms_norm_gated_with_full_gate_policy(&hidden, &gate, &weight, 1e-6, true);
    mlx_sys::eval(&[&later, &early, &portable]);
    assert_eq!(
        max_abs_f32(&later, &portable),
        0.0,
        "exact S=2 later-layer (allow=false) must stay portable"
    );
    assert_eq!(
        max_abs_f32(&early, &portable),
        0.0,
        "exact S=2 early-layer Metal unhooked after factory f4b5490d; stay portable"
    );
    let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let s1_hidden = super::slice_seq_row_4d(&hidden, 0);
    let s1_gate = super::slice_seq_row_4d(&gate, 0);
    let s1_exact = {
        let _ex = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        rms_norm_gated_with_full_gate_policy(&s1_hidden, &s1_gate, &weight, 1e-6, true)
    };
    let s1_port = super::portable_rms_norm_gated(&s1_hidden, &s1_gate, &weight, 1e-6);
    assert_eq!(
        max_abs_f32(&s1_exact, &s1_port),
        0.0,
        "exact S=1 must stay portable (MTP-off unchanged)"
    );
}

fn ones(shape: &[i32], dtype: MlxDtype) -> MlxArray {
    astype(
        &f32_array(&vec![1.0; shape.iter().product::<i32>() as usize], shape),
        dtype,
        None,
    )
}

#[test]
fn exact_profile_skips_rms_norm_gate_metal() {
    {
        let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        assert!(!super::skip_rms_norm_gate_metal_for_exact_verify());
    }
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    assert!(super::skip_rms_norm_gate_metal_for_exact_verify());
}

#[test]
fn exact_portable_gate_matches_silu_multiply_chain() {
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
    let hidden_data: Vec<f32> = (0..256)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let gate_data: Vec<f32> = (0..256)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.0625)
        .collect();
    let hidden = astype(
        &f32_array(&hidden_data, &[1, 2, 2, 64]),
        MlxDtype::Bfloat16,
        None,
    );
    let gate = astype(
        &f32_array(&gate_data, &[1, 2, 2, 64]),
        MlxDtype::Bfloat16,
        None,
    );
    let weight = astype(&f32_array(&vec![1.0; 64], &[64]), MlxDtype::Bfloat16, None);
    let shipped = rms_norm_gated_with_full_gate_policy(&hidden, &gate, &weight, 1e-6, false);
    let normed = rms_norm(&hidden, Some(&weight), 1e-6, None);
    let reference = astype(
        &multiply(
            &mlx_sys::ops::silu(&astype(&gate, MlxDtype::Float32, None), None),
            &astype(&normed, MlxDtype::Float32, None),
            None,
        ),
        MlxDtype::Bfloat16,
        None,
    );
    mlx_sys::eval(&[&shipped, &reference]);
    let a = astype(&shipped, MlxDtype::Float32, None);
    let b = astype(&reference, MlxDtype::Float32, None);
    mlx_sys::eval(&[&a, &b]);
    assert_eq!(
        a.data_f32(),
        b.data_f32(),
        "exact portable gate (one rms + silu_mul) must match silu*multiply"
    );
}

#[test]
fn compiled_silu_mul_normed_s2_matches_uncompiled_and_s1_row() {
    let hidden_data: Vec<f32> = (0..256)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let gate_data: Vec<f32> = (0..256)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.0625)
        .collect();
    let hidden = astype(
        &f32_array(&hidden_data, &[1, 2, 2, 64]),
        MlxDtype::Bfloat16,
        None,
    );
    let gate = astype(
        &f32_array(&gate_data, &[1, 2, 2, 64]),
        MlxDtype::Bfloat16,
        None,
    );
    let weight = astype(&f32_array(&vec![1.0; 64], &[64]), MlxDtype::Bfloat16, None);
    let normed = rms_norm(&hidden, Some(&weight), 1e-6, None);
    let uncompiled = super::portable_silu_mul_normed(&normed, &gate, MlxDtype::Bfloat16);
    let compiled = super::try_compiled_silu_mul_normed(&normed, &gate, MlxDtype::Bfloat16)
        .expect("exact S=2 pointwise silu*norm must compile");
    mlx_sys::eval(&[&uncompiled, &compiled]);
    let a = astype(&compiled, MlxDtype::Float32, None);
    let b = astype(&uncompiled, MlxDtype::Float32, None);
    mlx_sys::eval(&[&a, &b]);
    assert_eq!(
        a.data_f32(),
        b.data_f32(),
        "compiled pointwise silu*norm must be bit-exact vs uncompiled"
    );

    let h1 = slice(&hidden, &[0, 0, 0, 0], &[1, 1, 2, 64], &[1, 1, 1, 1], None);
    let g1 = slice(&gate, &[0, 0, 0, 0], &[1, 1, 2, 64], &[1, 1, 1, 1], None);
    let n1 = rms_norm(&h1, Some(&weight), 1e-6, None);
    let s1 = super::portable_silu_mul_normed(&n1, &g1, MlxDtype::Bfloat16);
    let s2_row0 = slice(
        &compiled,
        &[0, 0, 0, 0],
        &[1, 1, 2, 64],
        &[1, 1, 1, 1],
        None,
    );
    mlx_sys::eval(&[&s1, &s2_row0]);
    let r1 = astype(&s1, MlxDtype::Float32, None);
    let r2 = astype(&s2_row0, MlxDtype::Float32, None);
    mlx_sys::eval(&[&r1, &r2]);
    assert_eq!(
        r1.data_f32(),
        r2.data_f32(),
        "compiled S=2 row0 must match uncompiled S=1"
    );
    assert!(
        super::try_compiled_silu_mul_normed(&n1, &g1, MlxDtype::Bfloat16).is_none(),
        "S=1 must stay uncompiled so MTP-off decode fusion is unchanged"
    );
}
