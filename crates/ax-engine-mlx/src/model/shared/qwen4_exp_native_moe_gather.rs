//! Sorted routed-expert projections of a Flash Next prefill on the tensor
//! units, with 64-row tiles.
//!
//! MLX's sorted gather sizes its row tiles by the pairs per expert of the
//! whole stack, which makes them 32 rows for a Flash Next prefill. Real
//! routing concentrates on a fraction of the experts (84 to 330 of 512 per
//! layer over a 2,047-row chunk of the benchmark prompt), and every tile of a
//! busy expert dequantizes that expert's weights again.
//!
//! `ax_moe_prefill_sorted_gather_nax_v1` is AX's port of that kernel (ADR-053,
//! `gather_qmm_mx_nax.metal`) with 64-row tiles and empty 16-row fragments
//! skipped. It accumulates in MLX's order, so its output is MLX's bit for bit.
//! Twelve layers of gate, up and down projections on the real weights and
//! routing, MacBook Pro M5 Max 128 GB: 109.5 ms and 7.9 J through MLX, 91.3 ms
//! and 5.9 J through the port at 2,047 rows; 61.1 ms and 2.3 J against 56.2 ms
//! and 1.9 J at 412 rows.
//!
//! The kernel's fragment layout is that of the tensor units, so it is admitted
//! on Neural Accelerator hosts only, and only after a self-check at the
//! caller's geometry reproduced MLX's gather bit for bit. A kernel that fails
//! to build, launch or match leaves the layer on MLX's gather.

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, MlxQuantizationMode,
    astype, reshape,
};
use std::sync::{Mutex, OnceLock};

use super::qwen4_exp_native_hc_prefill::{pattern, same_bits, single_output};
use crate::weights::QuantizedWeight;

/// Rows of one expert a threadgroup computes.
const TILE_ROWS: i32 = 64;
/// Simdgroups along the rows of a tile; each owns `TILE_ROWS / ROW_SIMDGROUPS`.
const ROW_SIMDGROUPS: i32 = 2;
/// Output columns a threadgroup computes, in two simdgroups of 32.
const TILE_COLUMNS: i32 = 64;
const COLUMN_SIMDGROUPS: i32 = 2;
/// K values dequantized per step; both dimensions must be multiples of it.
const K_STEP: i32 = 64;
const GROUP_SIZE: i32 = 32;
const BITS: i32 = 4;
/// Fewest experts of a stack. MLX hands a kernel an input of fewer than eight
/// elements in the constant address space, and the gather takes the offsets
/// as a device buffer.
const MIN_EXPERTS: i32 = 8;
/// Most pairs of one gather: far above any prefill chunk, and low enough for
/// the kernel's 32-bit row and tile arithmetic.
const MAX_PAIRS: i32 = 1 << 24;
/// Threads of the offsets kernel's threadgroups.
const OFFSET_THREADS: i32 = 32;

const GATHER_NAME: &str = "ax_moe_prefill_sorted_gather_nax_v1";
const OFFSETS_NAME: &str = "ax_moe_prefill_expert_offsets_v1";

/// `verify_qmm_mx_nax.metal` supplies the fragment coordinates, the tensor-unit
/// product and the MX dequantization the gather is written with.
const GATHER_HEADER: &str = concat!(
    include_str!("verify_qmm_mx_nax.metal"),
    "\n",
    include_str!("gather_qmm_mx_nax.metal")
);
/// The dequantized weight tile is 64 rows of `K_STEP + 8` values, MLX's
/// padded stride.
const GATHER_SOURCE: &str = r#"
    threadgroup bfloat ws[64 * (64 + 8)];
    ax_gq::gather_mxfp4_nax<InputT, BM, WM, KDim, NDim>(
        x, w, scales, offsets, out, params[0], params[1],
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup, ws);
"#;
/// First sorted row of each expert: the lower bound of its id.
const OFFSETS_SOURCE: &str = r#"
    const uint expert = thread_position_in_grid.x;
    int lo = 0;
    int hi = params[0];
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (indices[mid] < expert) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    out[expert] = lo;
"#;

static GATHER: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();
static OFFSETS: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();
/// Self-check verdict per geometry.
static VERDICTS: Mutex<Vec<(Geometry, bool)>> = Mutex::new(Vec::new());

/// Input width, output width and activation type of one projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    k: i32,
    n: i32,
    dtype: MlxDtype,
}

/// `(k, n, experts)` of an MXFP4 expert stack the kernel can read.
fn stack_dims(weight: &QuantizedWeight) -> Option<(i32, i32, i32)> {
    if weight.mlx_quantization_mode() != MlxQuantizationMode::Mxfp4
        || weight.bits != BITS
        || weight.group_size != GROUP_SIZE
        || weight.biases.is_some()
        || weight.linear_bias.is_some()
        || weight.weight.dtype() != MlxDtype::Uint32
    {
        return None;
    }
    let &[experts, n, packed] = weight.weight.shape().as_slice() else {
        return None;
    };
    let k = packed.checked_mul(32 / BITS)?;
    if experts < MIN_EXPERTS
        || n < TILE_COLUMNS
        || k < K_STEP
        || n % TILE_COLUMNS != 0
        || k % K_STEP != 0
    {
        return None;
    }
    let scales = weight.scales.as_ref()?;
    (scales.dtype() == MlxDtype::Uint8 && scales.shape() == [experts, n, k / GROUP_SIZE])
        .then_some((k, n, experts))
}

/// The 16-bit activation types the kernel is written and checked for.
fn admitted_activation(dtype: MlxDtype) -> bool {
    matches!(dtype, MlxDtype::Bfloat16 | MlxDtype::Float16)
}

/// Where each expert's rows start in a gather sorted by expert, shared by the
/// projections of one layer.
pub(super) struct SortedGather {
    offsets: MlxArray,
    pairs: i32,
    experts: i32,
}

impl SortedGather {
    /// Plans the three projections of one routed block, or `None` when any of
    /// them must stay on MLX's gather: a stack is not one the kernel reads, or
    /// a self-check failed. `indices` holds one expert id per pair in
    /// ascending order and `x` is `[pairs, 1, k]`. The caller decides whether
    /// the switch and the host admit the port.
    pub(super) fn plan(
        indices: &MlxArray,
        x: &MlxArray,
        projections: [&QuantizedWeight; 3],
    ) -> Option<Self> {
        if !admitted_activation(x.dtype()) {
            return None;
        }
        let &[pairs] = indices.shape().as_slice() else {
            return None;
        };
        let mut experts = None;
        for weight in projections {
            let (k, n, count) = stack_dims(weight)?;
            if *experts.get_or_insert(count) != count
                || !verified(Geometry {
                    k,
                    n,
                    dtype: x.dtype(),
                })
            {
                return None;
            }
        }
        let experts = experts?;
        Self::with_offsets(indices, pairs, experts)
    }

    fn with_offsets(indices: &MlxArray, pairs: i32, experts: i32) -> Option<Self> {
        if !(1..=MAX_PAIRS).contains(&pairs) || experts < 1 {
            return None;
        }
        let kernel = kernel(
            &OFFSETS,
            OFFSETS_NAME,
            &["indices", "params"],
            OFFSETS_SOURCE,
            "",
        )?;
        let indices = astype(indices, MlxDtype::Uint32, None);
        let offsets = single_output(kernel.try_apply_with_template(
            &[&indices, &int_params(&[pairs])],
            &[KernelOutputSpec {
                shape: vec![experts],
                dtype: MlxDtype::Int32,
            }],
            &[],
            (experts, 1, 1),
            (experts.min(OFFSET_THREADS), 1, 1),
            None,
        ))?;
        Some(Self {
            offsets,
            pairs,
            experts,
        })
    }

    /// `x[pair] @ weight[indices[pair]]^T` as `[pairs, 1, n]` in `x`'s type, or
    /// `None` for a projection whose geometry and activation type did not
    /// pass the self-check.
    pub(super) fn project(&self, x: &MlxArray, weight: &QuantizedWeight) -> Option<MlxArray> {
        let (k, n, _) = stack_dims(weight)?;
        if !admitted_activation(x.dtype())
            || !verified(Geometry {
                k,
                n,
                dtype: x.dtype(),
            })
        {
            return None;
        }
        self.launch(x, weight)
    }

    /// The kernel launch behind [`Self::project`], which the self-check runs
    /// before any verdict exists.
    fn launch(&self, x: &MlxArray, weight: &QuantizedWeight) -> Option<MlxArray> {
        let (k, n, experts) = stack_dims(weight)?;
        if experts != self.experts || x.shape() != [self.pairs, 1, k] {
            return None;
        }
        let scales = weight.scales.as_ref()?;
        let kernel = kernel(
            &GATHER,
            GATHER_NAME,
            &["x", "w", "scales", "offsets", "params"],
            GATHER_SOURCE,
            GATHER_HEADER,
        )?;
        // A tile never spans two experts, so every expert but one can add a
        // partial tile to the full ones.
        let tiles = self
            .pairs
            .min((self.pairs + TILE_ROWS - 1) / TILE_ROWS + self.experts - 1);
        let grid = (
            (n / TILE_COLUMNS).checked_mul(32)?,
            tiles.checked_mul(COLUMN_SIMDGROUPS)?,
            ROW_SIMDGROUPS,
        );
        let flat = reshape(x, &[self.pairs, k], None);
        let output = single_output(kernel.try_apply_with_template(
            &[
                &flat,
                &weight.weight,
                scales,
                &self.offsets,
                &int_params(&[self.pairs, self.experts]),
            ],
            &[KernelOutputSpec {
                shape: vec![self.pairs, n],
                dtype: x.dtype(),
            }],
            &[
                KernelTemplateArg::Dtype {
                    name: "InputT",
                    dtype: x.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "BM",
                    value: TILE_ROWS,
                },
                KernelTemplateArg::Int {
                    name: "WM",
                    value: ROW_SIMDGROUPS,
                },
                KernelTemplateArg::Int {
                    name: "KDim",
                    value: k,
                },
                KernelTemplateArg::Int {
                    name: "NDim",
                    value: n,
                },
            ],
            grid,
            (32, COLUMN_SIMDGROUPS, ROW_SIMDGROUPS),
            None,
        ))?;
        Some(reshape(&output, &[self.pairs, 1, n], None))
    }
}

/// Whether model load should compile the sorted expert gather port: Flash
/// Next packs only, with load-time warm-up on, where the kernel may run.
pub(crate) fn prefill_warmup_applies(qwen4_exp: bool, kernel_warmup: bool) -> bool {
    qwen4_exp && kernel_warmup && crate::fastpath::flash_next_moe_gather_nax_prefill_enabled()
}

/// Compile the gate/up and down specializations and run their self-checks at
/// load, so the first prefill pays neither. `Err` names the projection that
/// stays on MLX's gather; the request path then never launches the port.
pub(crate) fn warm_prefill_kernels(
    hidden: i32,
    intermediate: i32,
    dtype: MlxDtype,
) -> Result<(), String> {
    for (k, n) in [(hidden, intermediate), (intermediate, hidden)] {
        if !verified(Geometry { k, n, dtype }) {
            return Err(format!(
                "sorted expert gather port not verified for {k} -> {n} {dtype:?}"
            ));
        }
    }
    Ok(())
}

fn int_params(values: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        &[values.len() as i32],
        MlxDtype::Int32,
    )
}

fn kernel(
    cell: &'static OnceLock<Option<MlxMetalKernel>>,
    name: &str,
    inputs: &[&str],
    source: &str,
    header: &str,
) -> Option<&'static MlxMetalKernel> {
    cell.get_or_init(|| {
        match MlxMetalKernel::try_new(name, inputs, &["out"], source, header, true) {
            Ok(kernel) => Some(kernel),
            Err(error) => {
                tracing::warn!(
                    %error,
                    kernel = name,
                    "Flash Next sorted expert gather kernel failed to build; using MLX's gather"
                );
                None
            }
        }
    })
    .as_ref()
}

fn verified(geometry: Geometry) -> bool {
    let Ok(mut verdicts) = VERDICTS.lock() else {
        return false;
    };
    if let Some((_, verdict)) = verdicts.iter().find(|(seen, _)| *seen == geometry) {
        return *verdict;
    }
    let verdict = self_check(geometry).unwrap_or(false);
    if verdict {
        tracing::info!(
            ?geometry,
            "Flash Next sorted expert gather kernel matches MLX's gather"
        );
    } else {
        tracing::warn!(
            ?geometry,
            "Flash Next sorted expert gather kernel differs from MLX's gather; using MLX's gather"
        );
    }
    verdicts.push((geometry, verdict));
    verdict
}

/// Rows per expert of the self-check: a full tile and a part of a second,
/// experts with no rows, runs that end inside a fragment row and on one, and
/// a single row. The scheduler scans 32 experts at a time, so the runs go on
/// past the 32nd expert, as a Flash Next stack of 512 does.
const SELF_CHECK_RUNS: [i32; 42] = [
    70, 0, 17, 0, 1, 33, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 3, 65, 0, 9, 0, 0, 32, 0, 48, 0, 2,
];
/// Distinct matrices of the self-check stack, repeated to its 42 experts. It
/// does not divide the scheduler's 32, so a tile given to the expert one scan
/// away reads other weights.
const SELF_CHECK_DISTINCT: i32 = 7;

/// One expert id per row for runs of `runs[expert]` rows, in ascending order.
fn sorted_indices(runs: &[i32]) -> MlxArray {
    let ids: Vec<u32> = runs
        .iter()
        .enumerate()
        .flat_map(|(expert, rows)| std::iter::repeat_n(expert as u32, (*rows).max(0) as usize))
        .collect();
    MlxArray::from_raw_data(
        ids.as_ptr().cast(),
        std::mem::size_of_val(ids.as_slice()),
        &[ids.len() as i32],
        MlxDtype::Uint32,
    )
}

/// A fixed MXFP4 stack of `experts` matrices of `n` by `k`, the first
/// `distinct` of them repeated in order.
fn pattern_stack(experts: i32, distinct: i32, geometry: Geometry) -> Option<QuantizedWeight> {
    let distinct = distinct.clamp(1, experts.max(1));
    let dense = pattern(&[distinct, geometry.n, geometry.k], 37, 0.05);
    let quantized = mlx_sys::quantize(
        &dense,
        Some(GROUP_SIZE),
        Some(BITS),
        MlxQuantizationMode::Mxfp4,
        None,
        None,
    );
    let [packed, scales] = quantized.as_slice() else {
        return None;
    };
    let repeated = |array: &MlxArray| {
        let copies = vec![array; ((experts + distinct - 1) / distinct) as usize];
        let stacked = mlx_sys::concatenate(&copies, 0, None);
        let mut stop = stacked.shape();
        stop[0] = experts;
        mlx_sys::slice(&stacked, &[0, 0, 0], &stop, &[1, 1, 1], None)
    };
    Some(QuantizedWeight {
        weight: repeated(packed),
        scales: Some(repeated(scales)),
        biases: None,
        group_size: GROUP_SIZE,
        bits: BITS,
        mode: "mxfp4".to_owned(),
        linear_bias: None,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
        weight_t_view: Default::default(),
    })
}

/// MLX's sorted gather of the same pairs.
fn reference(x: &MlxArray, weight: &QuantizedWeight, indices: &MlxArray) -> Option<MlxArray> {
    Some(mlx_sys::gather_qmm_with_mode(
        x,
        &weight.weight,
        weight.scales.as_ref()?,
        None,
        indices,
        true,
        Some(GROUP_SIZE),
        Some(BITS),
        MlxQuantizationMode::Mxfp4,
        true,
        None,
    ))
}

/// The port against MLX's gather for `runs[expert]` rows per expert.
fn matches_reference(geometry: Geometry, runs: &[i32], distinct: i32) -> Option<bool> {
    let weight = pattern_stack(runs.len() as i32, distinct, geometry)?;
    let indices = sorted_indices(runs);
    let &[pairs] = indices.shape().as_slice() else {
        return None;
    };
    let x = astype(
        &pattern(&[pairs, 1, geometry.k], 41, 0.75),
        geometry.dtype,
        None,
    );
    let plan = SortedGather::with_offsets(&indices, pairs, runs.len() as i32)?;
    let actual = plan.launch(&x, &weight)?;
    let expected = reference(&x, &weight, &indices)?;
    same_bits(&actual, &expected)
}

fn self_check(geometry: Geometry) -> Option<bool> {
    matches_reference(geometry, &SELF_CHECK_RUNS, SELF_CHECK_DISTINCT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::eval;

    fn offsets_of(runs: &[i32]) -> Vec<u32> {
        let indices = sorted_indices(runs);
        let pairs = indices.shape()[0];
        let Some(plan) = SortedGather::with_offsets(&indices, pairs, runs.len() as i32) else {
            panic!("offsets kernel did not launch");
        };
        let offsets = astype(&plan.offsets, MlxDtype::Uint32, None);
        eval(&[&offsets]);
        offsets.data_u32().to_vec()
    }

    /// The offsets kernel is plain Metal and runs on every host.
    #[test]
    fn offsets_are_the_first_row_of_each_expert() {
        assert_eq!(offsets_of(&[3, 0, 2, 0, 0, 4]), [0, 3, 3, 5, 5, 5]);
        assert_eq!(offsets_of(&[0, 0, 5]), [0, 0, 0]);
        assert_eq!(offsets_of(&[7]), [0]);
        assert_eq!(offsets_of(&[2, 0, 0]), [0, 2, 2]);
        let runs: Vec<i32> = (0..100).map(|expert| (expert * 7) % 5).collect();
        let mut expected = Vec::new();
        let mut start = 0u32;
        for rows in &runs {
            expected.push(start);
            start += *rows as u32;
        }
        assert_eq!(offsets_of(&runs), expected);
    }

    #[test]
    fn stack_dims_admit_only_aligned_mxfp4_stacks() {
        let geometry = Geometry {
            k: 128,
            n: 64,
            dtype: MlxDtype::Bfloat16,
        };
        let Some(stack) = pattern_stack(8, 8, geometry) else {
            panic!("quantize failed");
        };
        assert_eq!(stack_dims(&stack), Some((128, 64, 8)));
        let mut other_mode = pattern_stack(8, 8, geometry).unwrap();
        other_mode.mode = "nvfp4".to_owned();
        assert_eq!(stack_dims(&other_mode), None);
        let mut biased = pattern_stack(8, 8, geometry).unwrap();
        biased.biases = biased.scales.clone();
        assert_eq!(stack_dims(&biased), None);
        // The offsets of fewer than eight experts are not a device buffer.
        assert_eq!(stack_dims(&pattern_stack(7, 7, geometry).unwrap()), None);
        // 96 output columns are not whole 64-column tiles.
        let ragged = pattern_stack(
            8,
            8,
            Geometry {
                k: 128,
                n: 96,
                dtype: MlxDtype::Bfloat16,
            },
        )
        .unwrap();
        assert_eq!(stack_dims(&ragged), None);
    }

    /// `ProvenExact` evidence for `AX_MLX_FLASH_NEXT_MOE_GATHER_NAX_PREFILL`:
    /// runs that fill a tile, end inside each fragment row, skip experts and
    /// leave a single row, at both Flash Next projection geometries and both
    /// 16-bit activation types. Every case holds at least four pairs per
    /// expert, where MLX runs the batched kernel the port stands in for. The
    /// comparison needs the tensor units.
    #[test]
    fn sorted_gather_port_matches_mlx_bit_for_bit() {
        if !crate::hardware::neural_accelerator_active() {
            return;
        }
        // 100 experts, four scans of the scheduler: runs on both sides of
        // the scan boundaries and a last expert with rows.
        let mut scans = vec![0; 100];
        for (expert, rows) in [
            (0, 5),
            (30, 64),
            (31, 1),
            (32, 130),
            (63, 17),
            (64, 2),
            (65, 96),
            (99, 85),
        ] {
            scans[expert] = rows;
        }
        let runs: [&[i32]; 5] = [
            &SELF_CHECK_RUNS,
            &[
                1, 0, 0, 16, 17, 31, 32, 33, 0, 48, 63, 64, 65, 127, 128, 129, 1,
            ],
            &[300, 2, 0, 0, 5, 90, 0, 3],
            &[0, 0, 0, 24, 0, 0, 0, 8],
            &scans,
        ];
        for dtype in [MlxDtype::Bfloat16, MlxDtype::Float16] {
            for (k, n) in [(2560, 640), (640, 2560), (128, 64)] {
                for runs in runs {
                    assert_eq!(
                        matches_reference(Geometry { k, n, dtype }, runs, 17),
                        Some(true),
                        "{dtype:?} k={k} n={n} runs={runs:?}"
                    );
                }
            }
        }
    }

    /// The self-check at the Flash Next projections, and the properties of
    /// its fixture it relies on: runs past the scheduler's first scan, a
    /// stack whose repeat does not line up with that scan, and enough pairs
    /// for MLX to run its batched kernel on the reference side.
    #[test]
    fn self_check_passes_at_the_flash_next_geometries() {
        let pairs: i32 = SELF_CHECK_RUNS.iter().sum();
        assert!(SELF_CHECK_RUNS[32..].iter().any(|rows| *rows > 0));
        assert_ne!(32 % SELF_CHECK_DISTINCT, 0);
        assert!(pairs >= 16 && pairs / SELF_CHECK_RUNS.len() as i32 >= 4);
        if !crate::hardware::neural_accelerator_active() {
            return;
        }
        for (k, n) in [(2560, 640), (640, 2560)] {
            let geometry = Geometry {
                k,
                n,
                dtype: MlxDtype::Bfloat16,
            };
            assert_eq!(self_check(geometry), Some(true), "{geometry:?}");
        }
    }

    #[test]
    fn a_plan_refuses_other_geometries() {
        if !crate::hardware::neural_accelerator_active() {
            return;
        }
        let geometry = Geometry {
            k: 128,
            n: 64,
            dtype: MlxDtype::Bfloat16,
        };
        let stack = pattern_stack(8, 8, geometry).unwrap();
        let indices = sorted_indices(&[4, 0, 6, 0, 0, 0, 0, 0]);
        let plan = SortedGather::with_offsets(&indices, 10, 8).unwrap();
        let x = pattern(&[10, 1, 128], 5, 0.5);
        assert!(plan.project(&x, &stack).is_some());
        // An activation type the kernel was not checked for.
        let wide_x = astype(&x, MlxDtype::Float32, None);
        assert!(plan.project(&wide_x, &stack).is_none());
        // Another row count, another input width and another stack size.
        assert!(
            plan.project(&pattern(&[9, 1, 128], 5, 0.5), &stack)
                .is_none()
        );
        assert!(
            plan.project(&pattern(&[10, 1, 64], 5, 0.5), &stack)
                .is_none()
        );
        let other = pattern_stack(9, 9, geometry).unwrap();
        assert!(plan.project(&x, &other).is_none());
    }

    #[test]
    fn a_plan_admits_verified_stacks_and_16_bit_activations_only() {
        if !crate::hardware::neural_accelerator_active() {
            return;
        }
        let wide = Geometry {
            k: 128,
            n: 64,
            dtype: MlxDtype::Bfloat16,
        };
        let narrow = Geometry {
            k: 64,
            n: 128,
            dtype: MlxDtype::Bfloat16,
        };
        let (gate, up, down) = (
            pattern_stack(8, 8, wide).unwrap(),
            pattern_stack(8, 8, wide).unwrap(),
            pattern_stack(8, 8, narrow).unwrap(),
        );
        let indices = sorted_indices(&[4, 0, 6, 0, 0, 0, 0, 0]);
        let x = pattern(&[10, 1, 128], 5, 0.5);
        assert!(SortedGather::plan(&indices, &x, [&gate, &up, &down]).is_some());
        let wide_x = astype(&x, MlxDtype::Float32, None);
        assert!(SortedGather::plan(&indices, &wide_x, [&gate, &up, &down]).is_none());
        // Stacks of different expert counts are not one routed block.
        let other = pattern_stack(9, 9, narrow).unwrap();
        assert!(SortedGather::plan(&indices, &x, [&gate, &up, &other]).is_none());
    }
}
