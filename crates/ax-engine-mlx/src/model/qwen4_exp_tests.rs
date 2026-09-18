//! Shared-gate conservation through HC writes and the next recurrent layer.
//!
//! These generated tensors need not reproduce the captured M2 rounding drift.
//! The actual-weight replay establishes that cause; these tests protect scope,
//! singleton equivalence and immutable continuation with nonzero branch outputs.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use crate::model::LinearAttentionConfig;
use crate::model::shared::qwen4_exp_gdn::{Qwen4ExpGdn, Qwen4ExpGdnState, Qwen4ExpGdnWeights};
use crate::model::shared::qwen4_exp_moe::{
    Qwen4ExpExpertWeights, Qwen4ExpMoe, Qwen4ExpMoeWeights, Qwen4ExpResidentExperts,
};
use crate::model::shared::qwen4_exp_residual::{
    Qwen4ExpGatedResidual, Qwen4ExpGatedResidualWeights, Qwen4ExpStreamLayout,
    sigmoid_projection_dtype, silu_projection_dtype,
};
use crate::model::shared::utils::{ProjectionBatchPolicy, qw_with_policy};
use crate::weights::QuantizedWeight;
use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, add, astype, concatenate, contiguous, eval, multiply,
    quantize, slice,
};

// K=256 avoids MLX's separate K=64/128 quantized projection dispatch.
const HIDDEN: i32 = 256;
const SHARED_WIDTH: i32 = 64;
const PREFIX: i32 = 3;
const SHARED: ProjectionBatchPolicy = ProjectionBatchPolicy::Shared;
const ROW_EXACT: ProjectionBatchPolicy = ProjectionBatchPolicy::RowExact;

fn array(data: &[f32], shape: &[i32]) -> MlxArray {
    astype(
        &MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        ),
        MlxDtype::Bfloat16,
        None,
    )
}

fn wave(shape: &[i32], seed: i32) -> MlxArray {
    let data: Vec<f32> = (0..shape.iter().product())
        .map(|i| ((i * (seed * 2 + 3) + seed * 17) % 1009 - 504) as f32 / 2048.0)
        .collect();
    array(&data, shape)
}

fn pack(weight: MlxArray, mode: Option<MlxQuantizationMode>) -> QuantizedWeight {
    let Some(mode) = mode else {
        return QuantizedWeight::new(weight, None, None);
    };
    let packed = quantize(&weight, Some(32), Some(4), mode, None, None);
    let biases = (mode == MlxQuantizationMode::Affine).then(|| packed[2].clone());
    let mut result = QuantizedWeight::new(packed[0].clone(), Some(packed[1].clone()), biases);
    result.mode = if mode == MlxQuantizationMode::Mxfp4 {
        "mxfp4".into()
    } else {
        "affine".into()
    };
    result.group_size = 32;
    result.bits = 4;
    result
}

// Nonzero structured companions limit unrelated accumulation-order differences.
// Only the shared gate below uses a dense generated MXFP4 matrix.
fn sparse(output: i32, input: i32, seed: i32, scale: f32) -> MlxArray {
    let mut data = vec![0.0; (output * input) as usize];
    for row in 0..output {
        let column = (row * 13 + seed) % input;
        data[(row * input + column) as usize] = scale;
    }
    array(&data, &[output, input])
}

fn moe(
    gate_mode: Option<MlxQuantizationMode>,
    companion_mode: Option<MlxQuantizationMode>,
    routed: bool,
    shared: bool,
) -> Qwen4ExpMoe {
    let expert = |output, input, seed, scale| {
        let rows = [
            sparse(output, input, seed, scale),
            sparse(output, input, seed + 7, scale),
        ];
        QuantizedWeight::new(mlx_sys::stack(&[&rows[0], &rows[1]], 0, None), None, None)
    };
    Qwen4ExpMoe::new(
        HIDDEN as usize,
        32,
        SHARED_WIDTH as usize,
        2,
        2,
        true,
        Qwen4ExpMoeWeights {
            router: pack(sparse(2, HIDDEN, 9, 0.5), companion_mode),
            experts: Qwen4ExpExpertWeights::Resident(Box::new(Qwen4ExpResidentExperts {
                gate: expert(32, HIDDEN, 3, 0.5),
                up: expert(32, HIDDEN, 5, 0.5),
                down: expert(HIDDEN, 32, 7, if routed { 0.5 } else { 0.0 }),
            })),
            shared_gate: pack(wave(&[SHARED_WIDTH, HIDDEN], 11), gate_mode),
            shared_up: pack(sparse(SHARED_WIDTH, HIDDEN, 13, 0.5), companion_mode),
            shared_down: pack(
                sparse(HIDDEN, SHARED_WIDTH, 17, if shared { 0.5 } else { 0.0 }),
                companion_mode,
            ),
            shared_router: pack(sparse(1, HIDDEN, 19, 0.5), companion_mode),
        },
    )
    .unwrap()
}

fn hc(seed: i32) -> Qwen4ExpGatedResidual {
    Qwen4ExpGatedResidual::new(
        Qwen4ExpStreamLayout::new(2, HIDDEN as usize).unwrap(),
        32,
        1e-6,
        Qwen4ExpGatedResidualWeights {
            norm_gain: array(&vec![1.0; (2 * HIDDEN) as usize], &[2 * HIDDEN]),
            read_down: pack(sparse(32, 2 * HIDDEN, seed, 0.5), None),
            read_up: pack(sparse(2 * HIDDEN, 32, seed + 1, 0.5), None),
            write_inject: Some(pack(sparse(2, 2 * HIDDEN, seed + 2, 0.5), None)),
        },
    )
    .unwrap()
}

fn gdn() -> Qwen4ExpGdn {
    let config = LinearAttentionConfig {
        full_attention_interval: 4,
        num_key_heads: 1,
        num_value_heads: 1,
        key_head_dim: 4,
        value_head_dim: 4,
        conv_kernel_dim: 3,
        q_scale: 0.25,
        k_scale: 0.5,
    };
    Qwen4ExpGdn::new(
        config,
        HIDDEN as usize,
        1e-6,
        Qwen4ExpGdnWeights {
            qkv: pack(sparse(12, HIDDEN, 23, 0.5), None),
            gate: pack(sparse(4, HIDDEN, 29, 0.5), None),
            decay: pack(sparse(1, HIDDEN, 31, 0.5), None),
            beta: pack(sparse(1, HIDDEN, 37, 0.5), None),
            output: pack(sparse(HIDDEN, 4, 41, 0.5), None),
            conv: array(&[0.25, 0.5, 0.25].repeat(12), &[12, 3, 1]),
            a_log: array(&[0.0], &[1]),
            dt_bias: array(&[0.25], &[1]),
            norm_gain: array(&[1.0; 4], &[4]),
        },
    )
    .unwrap()
}

fn span(value: &MlxArray, start: i32, end: i32) -> MlxArray {
    contiguous(
        &slice(
            value,
            &[0, start, 0],
            &[1, end, value.shape()[2]],
            &[1, 1, 1],
            None,
        ),
        None,
    )
}

fn values(value: &MlxArray) -> Vec<f32> {
    let value = contiguous(&astype(value, MlxDtype::Float32, None), None);
    eval(&[&value]);
    let result = value.data_f32().to_vec();
    assert!(result.iter().all(|x| x.is_finite()));
    result
}

#[track_caller]
fn exact(actual: &MlxArray, expected: &MlxArray) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    let actual = values(actual);
    let expected = values(expected);
    let first = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b);
    let changed = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    let maximum = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(
        changed, 0,
        "first difference {first:?}; maximum absolute difference {maximum}"
    );
}

fn exact_state(actual: &Qwen4ExpGdnState, expected: &Qwen4ExpGdnState) {
    exact(&actual.conv, &expected.conv);
    exact(&actual.recurrent, &expected.recurrent);
}

fn singleton_moe(
    module: &Qwen4ExpMoe,
    input: &MlxArray,
    policy: ProjectionBatchPolicy,
) -> MlxArray {
    let outputs: Vec<_> = (0..input.shape()[1])
        .map(|row| {
            let output = module.forward(&span(input, row, row + 1), policy).unwrap();
            values(&output);
            output
        })
        .collect();
    concatenate(&outputs.iter().collect::<Vec<_>>(), 1, None)
}

fn singleton_gate_with_shared_companions(
    input: &MlxArray,
    companion_mode: Option<MlxQuantizationMode>,
) -> MlxArray {
    let gate = pack(
        wave(&[SHARED_WIDTH, HIDDEN], 11),
        Some(MlxQuantizationMode::Mxfp4),
    );
    let gate_rows: Vec<_> = (0..input.shape()[1])
        .map(|row| {
            let output = qw_with_policy(&span(input, row, row + 1), &gate, SHARED);
            values(&output);
            output
        })
        .collect();
    let gate_output = concatenate(&gate_rows.iter().collect::<Vec<_>>(), 1, None);
    let up = pack(sparse(SHARED_WIDTH, HIDDEN, 13, 0.5), companion_mode);
    let down = pack(sparse(HIDDEN, SHARED_WIDTH, 17, 0.5), companion_mode);
    let router = pack(sparse(1, HIDDEN, 19, 0.5), companion_mode);
    let activated = multiply(
        &silu_projection_dtype(&gate_output),
        &qw_with_policy(input, &up, SHARED),
        None,
    );
    let shared = multiply(
        &qw_with_policy(&activated, &down, SHARED),
        &sigmoid_projection_dtype(&qw_with_policy(input, &router, SHARED)),
        None,
    );
    // The untouched routed branch stays batched, as do all shared companions.
    // Affine companions need not equal a fully singleton MoE execution.
    let routed = moe(
        Some(MlxQuantizationMode::Mxfp4),
        companion_mode,
        true,
        false,
    )
    .forward(input, SHARED)
    .unwrap();
    add(&routed, &shared, None)
}

struct Chain {
    moe: Qwen4ExpMoe,
    mlp_hc: Qwen4ExpGatedResidual,
    next_hc: Qwen4ExpGatedResidual,
    next_gdn: Qwen4ExpGdn,
}

struct Output {
    moe_delta: MlxArray,
    mlp_residual: MlxArray,
    residual: MlxArray,
    state: Qwen4ExpGdnState,
}

impl Chain {
    fn forward(
        &self,
        input: &MlxArray,
        state: Option<&Qwen4ExpGdnState>,
        verifier: bool,
    ) -> Output {
        let policy = if verifier { ROW_EXACT } else { SHARED };
        let read = self.mlp_hc.read(input, policy).unwrap();
        let moe_delta = self
            .moe
            .forward_with_verifier_policy(read.branch_input(), SHARED, policy)
            .unwrap();
        let mlp_residual = read.write(&moe_delta).unwrap();
        let next_read = self.next_hc.read(&mlp_residual, policy).unwrap();
        let (delta, state) = self
            .next_gdn
            .forward(next_read.branch_input(), state, SHARED, policy)
            .unwrap();
        let residual = next_read.write(&delta).unwrap();
        Output {
            moe_delta,
            mlp_residual,
            residual,
            state,
        }
    }

    fn singletons(&self, input: &MlxArray, state: Option<&Qwen4ExpGdnState>) -> Output {
        let mut next = state.cloned();
        let mut outputs = Vec::new();
        for row in 0..input.shape()[1] {
            let output = self.forward(&span(input, row, row + 1), next.as_ref(), false);
            // Evaluate each complete call before scheduling the following singleton.
            values(&output.residual);
            values(&output.state.conv);
            values(&output.state.recurrent);
            next = Some(output.state.clone());
            outputs.push(output);
        }
        Output {
            moe_delta: concatenate(
                &outputs.iter().map(|x| &x.moe_delta).collect::<Vec<_>>(),
                1,
                None,
            ),
            mlp_residual: concatenate(
                &outputs.iter().map(|x| &x.mlp_residual).collect::<Vec<_>>(),
                1,
                None,
            ),
            residual: concatenate(
                &outputs.iter().map(|x| &x.residual).collect::<Vec<_>>(),
                1,
                None,
            ),
            state: next.unwrap(),
        }
    }
}

fn exact_output(actual: &Output, expected: &Output) {
    exact(&actual.moe_delta, &expected.moe_delta);
    exact(&actual.mlp_residual, &expected.mlp_residual);
    exact(&actual.residual, &expected.residual);
    exact_state(&actual.state, &expected.state);
}

#[test]
fn mxfp4_shared_gate_hc_and_following_gdn_preserve_nonempty_continuation() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let chain = Chain {
        moe: moe(Some(MlxQuantizationMode::Mxfp4), None, true, true),
        mlp_hc: hc(43),
        next_hc: hc(47),
        next_gdn: gdn(),
    };
    let input = wave(&[1, PREFIX + 3, HIDDEN * 2], 53);
    let prefix = chain.singletons(&span(&input, 0, PREFIX), None);
    let saved_input = values(&input);
    let saved_conv = values(&prefix.state.conv);
    let saved_recurrent = values(&prefix.state.recurrent);
    assert!(saved_conv.iter().any(|&x| x != 0.0));
    assert!(saved_recurrent.iter().any(|&x| x != 0.0));
    let suffix = span(&input, PREFIX, PREFIX + 2);
    let verifier = chain.forward(&suffix, Some(&prefix.state), true);
    let direct = chain.singletons(&suffix, Some(&prefix.state));
    exact_output(&verifier, &direct);
    assert!(values(&direct.moe_delta).iter().any(|&x| x != 0.0));
    assert_ne!(values(&direct.mlp_residual), values(&suffix));
    assert_ne!(values(&direct.state.conv), saved_conv);

    let next = span(&input, PREFIX + 2, PREFIX + 3);
    let committed = chain.forward(&next, Some(&verifier.state), false);
    let continued = chain.forward(&next, Some(&direct.state), false);
    exact_output(&committed, &continued);
    // Discard the staged pair and restart from the original populated checkpoint.
    let restarted = chain.singletons(&suffix, Some(&prefix.state));
    exact_output(&restarted, &direct);
    assert_eq!(values(&input), saved_input);
    assert_eq!(values(&prefix.state.conv), saved_conv);
    assert_eq!(values(&prefix.state.recurrent), saved_recurrent);
}

#[test]
fn shared_gate_policy_preserves_wrappers_non_mxfp4_and_legacy_row_exact() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let affine = Some(MlxQuantizationMode::Affine);
    let mxfp4 = Some(MlxQuantizationMode::Mxfp4);
    let input = wave(&[1, 2, HIDDEN], 59);
    for (gate_mode, companion_mode) in [
        (None, None),
        (affine, None),
        (None, mxfp4),
        (affine, mxfp4),
        (mxfp4, None),
        (mxfp4, affine),
        (mxfp4, mxfp4),
    ] {
        eprintln!("shared-gate control: gate={gate_mode:?}, companions={companion_mode:?}");
        let module = moe(gate_mode, companion_mode, true, true);
        let ordinary = module.forward(&input, SHARED).unwrap();
        let shared = module
            .forward_with_verifier_policy(&input, SHARED, SHARED)
            .unwrap();
        exact(&shared, &ordinary);
        let verifier = module
            .forward_with_verifier_policy(&input, SHARED, ROW_EXACT)
            .unwrap();
        if gate_mode != mxfp4 {
            // Other MXFP4 projections must not trigger a shared-gate override.
            exact(&verifier, &ordinary);
        } else {
            exact(
                &verifier,
                &singleton_gate_with_shared_companions(&input, companion_mode),
            );
            if companion_mode.is_none() {
                exact(&verifier, &singleton_moe(&module, &input, SHARED));
            }
        }
        let legacy = module.forward(&input, ROW_EXACT).unwrap();
        let explicit = module
            .forward_with_verifier_policy(&input, ROW_EXACT, ROW_EXACT)
            .unwrap();
        exact(&legacy, &explicit);
        exact(&legacy, &singleton_moe(&module, &input, ROW_EXACT));
    }
}

#[test]
fn shared_gate_verifier_retains_both_nonzero_moe_branches() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let mode = Some(MlxQuantizationMode::Mxfp4);
    let input = wave(&[1, 2, HIDDEN], 61);
    let run = |routed, shared| {
        moe(mode, None, routed, shared)
            .forward_with_verifier_policy(&input, SHARED, ROW_EXACT)
            .unwrap()
    };
    let routed = run(true, false);
    let shared = run(false, true);
    assert!(values(&routed).iter().any(|&x| x != 0.0));
    assert!(values(&shared).iter().any(|&x| x != 0.0));
    exact(&run(true, true), &add(&routed, &shared, None));
}
