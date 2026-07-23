//! Diagnostic native standard-FA block-table attention over one fixed K/V slab.
//!
//! The default serving path gathers fixed-slab rows into MLX SDPA. This kernel
//! remains opt-in and accepts only decode (`query_len == 1`) whose rows all
//! reside in one slab, matching mlxcel's measured dispatch constraint.

use std::collections::HashMap;
use std::sync::OnceLock;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, reshape, slice, take,
    transpose,
};
use parking_lot::Mutex;

#[derive(Clone, Debug)]
pub(crate) struct PagedAttentionView {
    pub(crate) k_slab: MlxArray,
    pub(crate) v_slab: MlxArray,
    pub(crate) block_table: MlxArray,
    pub(crate) block_ids: Vec<u32>,
    pub(crate) key_len: usize,
    pub(crate) block_size: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
}

impl PagedAttentionView {
    /// Dense compatibility view used only when native dispatch declines.
    pub(crate) fn materialize(&self) -> (MlxArray, MlxArray) {
        let gathered = |slab: &MlxArray| {
            let rows = take(slab, &self.block_table, 0, None);
            let flat = reshape(
                &rows,
                &[
                    (self.block_ids.len() * self.block_size) as i32,
                    self.n_kv_heads as i32,
                    self.head_dim as i32,
                ],
                None,
            );
            let visible = slice(
                &flat,
                &[0, 0, 0],
                &[
                    self.key_len as i32,
                    self.n_kv_heads as i32,
                    self.head_dim as i32,
                ],
                &[1, 1, 1],
                None,
            );
            let batched = reshape(
                &visible,
                &[
                    1,
                    self.key_len as i32,
                    self.n_kv_heads as i32,
                    self.head_dim as i32,
                ],
                None,
            );
            transpose(&batched, &[0, 2, 1, 3], None)
        };
        (gathered(&self.k_slab), gathered(&self.v_slab))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct KernelKey {
    dtype_code: u8,
    n_q_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
}

static PAGED_DECODE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static PAGED_DECODE_CERTIFICATION: OnceLock<Mutex<HashMap<KernelKey, bool>>> = OnceLock::new();

fn dtype_code(dtype: MlxDtype) -> Option<u8> {
    match dtype {
        MlxDtype::Bfloat16 => Some(1),
        MlxDtype::Float16 => Some(2),
        MlxDtype::Float32 => Some(3),
        _ => None,
    }
}

/// Evaluate decode attention directly from a physical block table.
///
/// The first use of every dtype/head geometry is evaluated through
/// `try_eval`. A Metal compilation failure permanently disables that geometry
/// for the process and lets the caller materialize dense K/V instead.
pub(crate) fn paged_decode_attention(
    q: &MlxArray,
    view: &PagedAttentionView,
    scale: f32,
) -> Option<MlxArray> {
    let q_shape = q.shape();
    let k_shape = view.k_slab.shape();
    let v_shape = view.v_slab.shape();
    if q_shape.len() != 4
        || q_shape[0] != 1
        || q_shape[2] != 1
        || k_shape.len() != 4
        || v_shape != k_shape
        || view.key_len == 0
        || view.block_size == 0
        || view.n_kv_heads == 0
        || view.head_dim == 0
    {
        return None;
    }
    let n_q_heads = q_shape[1];
    let n_kv_heads = i32::try_from(view.n_kv_heads).ok()?;
    let head_dim = i32::try_from(view.head_dim).ok()?;
    let block_size = i32::try_from(view.block_size).ok()?;
    if q_shape[3] != head_dim
        || k_shape[1] != block_size
        || k_shape[2] != n_kv_heads
        || k_shape[3] != head_dim
        || n_q_heads <= 0
        || n_q_heads % n_kv_heads != 0
        || !view.head_dim.is_power_of_two()
        || view.head_dim > 256
        || q.dtype() != view.k_slab.dtype()
        || q.dtype() != view.v_slab.dtype()
        || view.block_table.dtype() != MlxDtype::Uint32
        || view.block_table.shape() != vec![view.key_len.div_ceil(view.block_size) as i32]
    {
        return None;
    }

    let key = KernelKey {
        dtype_code: dtype_code(q.dtype())?,
        n_q_heads,
        n_kv_heads,
        head_dim,
        block_size,
    };
    if PAGED_DECODE_CERTIFICATION
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .get(&key)
        .is_some_and(|certified| !certified)
    {
        return None;
    }

    let kernel = PAGED_DECODE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_fa_paged_decode_attention_v1",
            &["q", "k_slab", "v_slab", "block_table", "scale", "key_len"],
            &["out"],
            PAGED_DECODE_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let scale = MlxArray::from_f32(scale);
    // Key length changes after every decode token. It must be a runtime scalar,
    // not a template argument: specializing the Metal source by KeyLen creates
    // and compiles a distinct pipeline for every token position.
    let key_len = i32::try_from(view.key_len).ok()?;
    let key_len_array = MlxArray::from_raw_data(
        (&key_len as *const i32).cast(),
        std::mem::size_of::<i32>(),
        &[1],
        MlxDtype::Int32,
    );
    let output = kernel
        .try_apply_with_template(
            &[
                q,
                &view.k_slab,
                &view.v_slab,
                &view.block_table,
                &scale,
                &key_len_array,
            ],
            &[KernelOutputSpec {
                shape: vec![1, n_q_heads, 1, head_dim],
                dtype: q.dtype(),
            }],
            &[
                KernelTemplateArg::Dtype {
                    name: "T",
                    dtype: q.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "Hq",
                    value: n_q_heads,
                },
                KernelTemplateArg::Int {
                    name: "Hkv",
                    value: n_kv_heads,
                },
                KernelTemplateArg::Int {
                    name: "HeadDim",
                    value: head_dim,
                },
                KernelTemplateArg::Int {
                    name: "BlockSize",
                    value: block_size,
                },
            ],
            (head_dim, 1, n_q_heads),
            (head_dim, 1, 1),
            None,
        )
        .ok()?
        .pop()?;

    let already_certified = PAGED_DECODE_CERTIFICATION
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .get(&key)
        .copied();
    if already_certified.is_none() {
        let certified = mlx_sys::transforms::try_eval(&[&output]).is_ok();
        PAGED_DECODE_CERTIFICATION
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .insert(key, certified);
        if !certified {
            return None;
        }
    }
    Some(output)
}

const PAGED_DECODE_KERNEL_SOURCE: &str = r#"
    const int lane = thread_position_in_threadgroup.x;
    const int q_head = thread_position_in_grid.z;
    const int kv_head = q_head / (Hq / Hkv);
    const int key_length = key_len[0];

    threadgroup float reductions[256];
    threadgroup float coefficients[3]; // alpha, beta, new running max

    const int q_base = q_head * HeadDim;
    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    float value_acc = 0.0f;

    for (int token = 0; token < key_length; ++token) {
        const int logical_block = token / BlockSize;
        const int block_offset = token - logical_block * BlockSize;
        const uint physical_block = block_table[logical_block];
        const int kv_base =
            ((static_cast<int>(physical_block) * BlockSize + block_offset) * Hkv + kv_head)
            * HeadDim;

        reductions[lane] = static_cast<float>(q[q_base + lane])
            * static_cast<float>(k_slab[kv_base + lane]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int stride = HeadDim >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                reductions[lane] += reductions[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            const float score = reductions[0] * scale;
            const float new_max = max(running_max, score);
            coefficients[0] = exp(running_max - new_max);
            coefficients[1] = exp(score - new_max);
            coefficients[2] = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float alpha = coefficients[0];
        const float beta = coefficients[1];
        value_acc = value_acc * alpha
            + static_cast<float>(v_slab[kv_base + lane]) * beta;
        running_sum = running_sum * alpha + beta;
        running_max = coefficients[2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[q_base + lane] = static_cast<T>(value_acc / running_sum);
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{
        ScaledDotProductAttentionMask, eval, scaled_dot_product_attention_with_mask, slice,
    };

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn paged_decode_kernel_matches_dense_gqa_oracle() {
        let q = array_f32(&[0.2, -0.3, 0.5, 0.7, -0.4, 0.1, 0.8, -0.2], &[1, 2, 1, 4]);
        // Four physical blocks; the logical table deliberately uses 2 then 0.
        let k_slab = array_f32(
            &[
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // block 0
                9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, // block 1 (unused)
                -0.2, 0.4, 0.6, -0.8, 0.3, -0.1, 0.9, 0.2, // block 2
                8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, // block 3 (unused)
            ],
            &[4, 2, 1, 4],
        );
        let v_slab = array_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, // block 0
                9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, // unused
                -1.0, 0.5, 1.5, 2.5, 0.0, 1.0, 2.0, 3.0, // block 2
                8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, // unused
            ],
            &[4, 2, 1, 4],
        );
        let ids = [2_u32, 0_u32];
        let block_table = MlxArray::from_raw_data(
            ids.as_ptr().cast(),
            std::mem::size_of_val(&ids),
            &[2],
            MlxDtype::Uint32,
        );
        let view = PagedAttentionView {
            k_slab: k_slab.clone(),
            v_slab: v_slab.clone(),
            block_table,
            block_ids: ids.to_vec(),
            key_len: 3,
            block_size: 2,
            n_kv_heads: 1,
            head_dim: 4,
        };
        let actual = paged_decode_attention(&q, &view, 0.5).expect("eligible paged route");

        let block2_start = [2, 0, 0, 0];
        let block2_stop = [3, 2, 1, 4];
        let block0_start = [0, 0, 0, 0];
        let block0_stop = [1, 1, 1, 4];
        let strides = [1, 1, 1, 1];
        let k2 = slice(&k_slab, &block2_start, &block2_stop, &strides, None);
        let k0 = slice(&k_slab, &block0_start, &block0_stop, &strides, None);
        let v2 = slice(&v_slab, &block2_start, &block2_stop, &strides, None);
        let v0 = slice(&v_slab, &block0_start, &block0_stop, &strides, None);
        let dense_k = transpose(
            &mlx_sys::concatenate(&[&k2, &k0], 1, None),
            &[0, 2, 1, 3],
            None,
        );
        let dense_v = transpose(
            &mlx_sys::concatenate(&[&v2, &v0], 1, None),
            &[0, 2, 1, 3],
            None,
        );
        let expected = scaled_dot_product_attention_with_mask(
            &q,
            &dense_k,
            &dense_v,
            0.5,
            ScaledDotProductAttentionMask::None,
            None,
        );
        eval(&[&actual, &expected]);
        let actual_data = actual.data_f32();
        let expected_data = expected.data_f32();
        let cpu_oracle = [
            0.117_967_15,
            1.236_099_1,
            2.236_099_0,
            3.236_099_0,
            -0.101_205_83,
            1.088_636_0,
            2.088_636_0,
            3.088_636_0,
        ];
        let max_abs_diff = |left: &[f32], right: &[f32]| {
            left.iter()
                .zip(right)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max)
        };
        let native_vs_cpu = max_abs_diff(&actual_data, &cpu_oracle);
        assert!(native_vs_cpu <= 2.0e-5, "native_vs_cpu={native_vs_cpu}");

        // MLX's fused SDPA uses a lower-precision reduction on newer Metal
        // devices. Keep it as a layout oracle, but independently require the
        // native kernel to match the scalar f64 calculation above.
        let dense_vs_cpu = max_abs_diff(&expected_data, &cpu_oracle);
        assert!(dense_vs_cpu <= 2.0e-3, "dense_vs_cpu={dense_vs_cpu}");
        let native_vs_dense = max_abs_diff(&actual_data, &expected_data);
        assert!(
            native_vs_dense <= 2.0e-3,
            "native_vs_dense={native_vs_dense}"
        );
    }
}
