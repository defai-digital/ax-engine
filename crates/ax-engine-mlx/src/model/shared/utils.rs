use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, add, astype,
    concatenate, contiguous, gather_mm, gather_qmm, matmul, multiply, quantized_matmul, reshape,
    slice, slice_last_dim, tanh, transpose,
};
use std::sync::OnceLock;

use super::super::config::ModelConfig;
use crate::fastpath;
use crate::weights::QuantizedWeight;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct QkvSlices {
    pub q: (i32, i32),
    pub gate: Option<(i32, i32)>,
    pub k: (i32, i32),
    pub v: (i32, i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProjectionBatchPolicy {
    Shared,
    /// Preserve the single-row reduction graph for every decode row.
    RowExact,
}

pub(crate) fn qkv_slices(cfg: &ModelConfig, head_dim: usize) -> QkvSlices {
    let q_size = (cfg.n_heads * head_dim) as i32;
    let kv_size = (cfg.n_kv_heads * head_dim) as i32;
    let gate = cfg.attn_output_gate.then_some((q_size, q_size * 2));
    let kv_start = if cfg.attn_output_gate {
        q_size * 2
    } else {
        q_size
    };
    QkvSlices {
        q: (0, q_size),
        gate,
        k: (kv_start, kv_start + kv_size),
        v: (kv_start + kv_size, kv_start + kv_size * 2),
    }
}

pub(crate) fn qw(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    qw_with_policy(x, qw, ProjectionBatchPolicy::Shared)
}

pub(crate) fn qw_with_policy(
    x: &MlxArray,
    qw: &QuantizedWeight,
    policy: ProjectionBatchPolicy,
) -> MlxArray {
    let shape = x.shape();
    if policy == ProjectionBatchPolicy::RowExact
        && shape.len() == 3
        && shape[0] > 1
        && shape[1] == 1
    {
        let rows: Vec<MlxArray> = (0..shape[0])
            .map(|row| {
                let row = slice(x, &[row, 0, 0], &[row + 1, 1, shape[2]], &[1, 1, 1], None);
                qw_direct(&contiguous(&row, None), qw)
            })
            .collect();
        let refs: Vec<&MlxArray> = rows.iter().collect();
        return concatenate(&refs, 0, None);
    }
    qw_direct(x, qw)
}

fn qw_direct(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
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
        let wt = transpose(&qw.weight, &[1, 0], None);
        matmul(x, &wt, None)
    }
}

pub(crate) fn mlx_slice_last_dim(x: &MlxArray, start: i32, end: i32) -> MlxArray {
    slice_last_dim(x, start, end, None)
}

pub(crate) fn scale_hidden_pub(hidden: &MlxArray, scale: f32) -> MlxArray {
    scale_hidden(hidden, scale)
}

pub(crate) fn scale_hidden(hidden: &MlxArray, scale: f32) -> MlxArray {
    // `cached_scalar` deduplicates the (value, dtype) pair across the process,
    // so steady-state decode pays one `multiply` op per call instead of
    // (astype + multiply). Saves ~4 ops/step on Gemma 4 E2B (one per scale
    // site: hidden_states_scale + 3 inside compute_per_layer_inputs_arr).
    let s_arr = mlx_sys::ops::cached_scalar(scale, hidden.dtype());
    multiply(hidden, &s_arr, None)
}

static ADD_MUL_SCALAR_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const ADD_MUL_SCALAR_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    float av = static_cast<float>(a[idx]);
    float bv = static_cast<float>(b[idx]);
    float scale_v = static_cast<float>(scale[0]);
    T rounded_sum = static_cast<T>(av + bv);
    out[idx] = static_cast<T>(static_cast<float>(rounded_sum) * scale_v);
"#;

pub(crate) fn add_then_multiply_scalar(a: &MlxArray, b: &MlxArray, scalar: &MlxArray) -> MlxArray {
    add_then_multiply_scalar_metal(a, b, scalar)
        .unwrap_or_else(|| multiply(&add(a, b, None), scalar, None))
}

fn add_then_multiply_scalar_metal(
    a: &MlxArray,
    b: &MlxArray,
    scalar: &MlxArray,
) -> Option<MlxArray> {
    if !fastpath::layer_scalar_fused_add_enabled() {
        return None;
    }
    add_then_multiply_scalar_metal_impl(a, b, scalar)
}

fn add_then_multiply_scalar_metal_impl(
    a: &MlxArray,
    b: &MlxArray,
    scalar: &MlxArray,
) -> Option<MlxArray> {
    if a.shape() != b.shape() || a.dtype() != b.dtype() || scalar.dtype() != a.dtype() {
        return None;
    }
    if !matches!(
        a.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let scalar_elements = scalar
        .shape()
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    if scalar_elements != 1 {
        return None;
    }
    let shape = a.shape();
    let element_count = shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = ADD_MUL_SCALAR_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_add_mul_scalar_v1",
            &["a", "b", "scale"],
            &["out"],
            ADD_MUL_SCALAR_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[a, b, scalar],
        &[KernelOutputSpec {
            shape,
            dtype: a.dtype(),
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: a.dtype(),
            },
            KernelTemplateArg::Int {
                name: "ElementCount",
                value: element_count,
            },
        ],
        (element_count, 1, 1),
        (256, 1, 1),
        None,
    );
    outputs.pop()
}

pub(crate) fn scalar_like(value: f32, dtype: MlxDtype) -> MlxArray {
    // Retained for callers outside the steady-state decode hot path
    // (e.g. MoE router masking, test fixtures) where the per-call astype is
    // not the bottleneck and value uniqueness is not guaranteed.
    let scalar = MlxArray::from_raw_data(
        &value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    astype(&scalar, dtype, None)
}

pub(crate) fn apply_final_logit_softcap(cfg: &ModelConfig, logits: &MlxArray) -> MlxArray {
    let Some(cap) = cfg.final_logit_softcapping.filter(|cap| *cap > 0.0) else {
        return logits.clone();
    };
    let inv_cap = 1.0_f32 / cap;
    let inv_cap_arr = mlx_sys::ops::cached_scalar(inv_cap, logits.dtype());
    let cap_arr = mlx_sys::ops::cached_scalar(cap, logits.dtype());
    let scaled = multiply(logits, &inv_cap_arr, None);
    multiply(&tanh(&scaled, None), &cap_arr, None)
}

pub(crate) fn shape_element_count(shape: &[i32]) -> usize {
    shape
        .iter()
        .map(|dim| usize::try_from(*dim).expect("MLX shape dims must be non-negative"))
        .product()
}

pub(crate) fn squeeze_switch_singleton(x: &MlxArray) -> MlxArray {
    let mut shape = x.shape();
    let ndim = shape.len();
    if ndim >= 2 && shape[ndim - 2] == 1 {
        shape.remove(ndim - 2);
        reshape(x, &shape, None)
    } else {
        x.clone()
    }
}

/// Gather-matmul for expert weights (quantized or dense).
///
/// `x`: [..., hidden], `qw.weight`: [num_experts, expert_size, hidden] (or packed).
/// `indices`: [..., top_k].  Returns [..., top_k, out_size].
pub(crate) fn qw_gather(
    x: &MlxArray,
    qw: &QuantizedWeight,
    indices: &MlxArray,
    sorted_indices: bool,
) -> MlxArray {
    if let Some(scales) = &qw.scales {
        gather_qmm(
            x,
            &qw.weight,
            scales,
            qw.biases.as_ref(),
            indices,
            true,
            Some(qw.group_size),
            Some(qw.bits),
            sorted_indices,
            None,
        )
    } else {
        // Dense experts: weight shape [N, out, in] → need [N, in, out] for gather_mm.
        let ndim = qw.weight.ndim();
        let mut axes: Vec<i32> = (0..ndim as i32).collect();
        let last = axes.len() - 1;
        axes.swap(last - 1, last);
        let wt = transpose(&qw.weight, &axes, None);
        gather_mm(x, &wt, indices, sorted_indices, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{MlxQuantizationMode, eval, quantize};

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
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
        let weight = array_f32(&weight_data, &[output_dim as i32, input_dim as i32]);
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
        };
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let input = array_f32(&input_data, &[2, 1, input_dim as i32]);
        let batched = qw_with_policy(&input, &weight, ProjectionBatchPolicy::RowExact);

        for row in 0..2 {
            let row_start = row * input_dim;
            let single_input = array_f32(
                &input_data[row_start..row_start + input_dim],
                &[1, 1, input_dim as i32],
            );
            let expected = qw(&single_input, &weight);
            let actual = contiguous(
                &slice(
                    &batched,
                    &[row as i32, 0, 0],
                    &[row as i32 + 1, 1, output_dim as i32],
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
}
