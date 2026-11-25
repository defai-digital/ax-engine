//! Routed-expert weighted sum of a sorted Flash Next prefill gather.
//!
//! A sorted prefill gather leaves the expert outputs in expert order. The MLX
//! chain then restores token order with a `take`, multiplies by the routing
//! weights and sums over the routed experts: three more passes over a
//! `[rows * top_k, hidden]` tensor per layer. `ax_moe_prefill_routed_sum_v1`
//! reads each routed row once, through the inverse sort order, and stores only
//! the `[rows, hidden]` sum.
//!
//! The kernel reproduces the chain bit for bit: every product rounds through
//! bf16 and the partial sums follow MLX's small strided reduction (rows
//! `j, j + Y, ...` per accumulator, `Y = min(8, top_k)`, accumulators folded in
//! order, bf16 at every step). A self-check against the MLX chain at the
//! caller's geometry runs before the first use; a kernel that fails to build,
//! launch or match leaves the layer on the MLX ops.

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, astype,
    expand_dims_axes, multiply, reshape, sum_axis, take,
};
use std::sync::{Mutex, OnceLock};

use super::qwen4_exp_native_hc_prefill::{pattern, same_bits, single_output};

/// Fewest rows the kernel runs at; the sorted prefill gather starts there.
pub(super) const MIN_ROWS: i32 = 64;

/// Accumulators of MLX's small strided reduction.
const REDUCTION_ACCUMULATORS: i32 = 8;
/// Largest reduction MLX runs through that kernel.
const SMALL_REDUCTION_LIMIT: i32 = 32;
const THREADS_PER_GROUP: i32 = 256;
const SELF_CHECK_ROWS: i32 = 3;
/// Rows a padded gather appends past the routed pairs in the self-check.
const SELF_CHECK_PADDING: i32 = 2;

const NAME: &str = "ax_moe_prefill_routed_sum_v1";

static KERNEL: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();
/// Self-check verdict per `(top_k, hidden)`.
static VERDICTS: Mutex<Vec<((i32, i32), bool)>> = Mutex::new(Vec::new());

/// `sum_k(routing[row, k] * down[inverse[row * top_k + k]])` as bf16
/// `[1, rows, hidden]`.
///
/// `down` holds one `hidden`-wide row per sorted pair and may carry padding
/// rows after them; `inverse` maps a pair in token order to its sorted row.
/// The kernel does not bounds-check `inverse`: it must be a permutation of the
/// sorted pairs, as the inverse of an argsort is.
pub(super) fn routed_sum(
    down: &MlxArray,
    inverse: &MlxArray,
    routing: &MlxArray,
    rows: i32,
    top_k: i32,
    hidden: i32,
) -> Option<MlxArray> {
    if rows < MIN_ROWS
        || !supported_geometry(top_k, hidden)
        || down.dtype() != MlxDtype::Bfloat16
        || routing.dtype() != MlxDtype::Bfloat16
        || !crate::fastpath::flash_next_moe_prefill_sum_enabled()
        || !verified(top_k, hidden)
    {
        return None;
    }
    launch(down, inverse, routing, rows, top_k, hidden)
}

/// A single routed expert is a reduction MLX skips (it would keep a `-0.0`
/// the kernel's `+0.0` start loses), and 32 or more leave its small kernel.
fn supported_geometry(top_k: i32, hidden: i32) -> bool {
    (2..SMALL_REDUCTION_LIMIT).contains(&top_k) && hidden > 0
}

fn launch(
    down: &MlxArray,
    inverse: &MlxArray,
    routing: &MlxArray,
    rows: i32,
    top_k: i32,
    hidden: i32,
) -> Option<MlxArray> {
    let pairs = rows.checked_mul(top_k)?;
    if down.shape().last() != Some(&hidden)
        || element_count(down)? / hidden < pairs
        || element_count(inverse)? != pairs
        || element_count(routing)? != pairs
    {
        return None;
    }
    let grid = rows.checked_mul(hidden)?;
    let kernel = KERNEL
        .get_or_init(|| {
            match MlxMetalKernel::try_new(
                NAME,
                &["down", "inverse", "routing"],
                &["out"],
                SOURCE,
                "",
                true,
            ) {
                Ok(kernel) => Some(kernel),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "Flash Next routed-sum prefill kernel failed to build; using MLX ops"
                    );
                    None
                }
            }
        })
        .as_ref()?;
    let inverse = astype(inverse, MlxDtype::Uint32, None);
    single_output(kernel.try_apply_with_template(
        &[down, &inverse, routing],
        &[KernelOutputSpec {
            shape: vec![1, rows, hidden],
            dtype: MlxDtype::Bfloat16,
        }],
        &[
            KernelTemplateArg::Int {
                name: "H",
                value: hidden,
            },
            KernelTemplateArg::Int {
                name: "K",
                value: top_k,
            },
            KernelTemplateArg::Int {
                name: "Y",
                value: top_k.min(REDUCTION_ACCUMULATORS),
            },
        ],
        (grid, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    ))
}

fn element_count(array: &MlxArray) -> Option<i32> {
    array
        .shape()
        .iter()
        .try_fold(1i32, |count, dim| count.checked_mul(*dim))
}

fn verified(top_k: i32, hidden: i32) -> bool {
    let key = (top_k, hidden);
    let Ok(mut verdicts) = VERDICTS.lock() else {
        return false;
    };
    if let Some((_, verdict)) = verdicts.iter().find(|(seen, _)| *seen == key) {
        return *verdict;
    }
    let verdict = self_check(top_k, hidden).unwrap_or(false);
    if !verdict {
        tracing::warn!(
            top_k,
            hidden,
            "Flash Next routed-sum prefill kernel differs from the MLX ops; using MLX ops"
        );
    }
    verdicts.push((key, verdict));
    verdict
}

/// The MLX chain the kernel replaces: unsort, weight, sum over the experts.
fn reference(
    down: &MlxArray,
    inverse: &MlxArray,
    routing: &MlxArray,
    rows: i32,
    top_k: i32,
    hidden: i32,
) -> MlxArray {
    let flat = reshape(down, &[-1, hidden], None);
    let restored = reshape(
        &take(&flat, inverse, 0, None),
        &[1, rows, top_k, hidden],
        None,
    );
    let routing = reshape(routing, &[1, rows, top_k], None);
    let weighted = multiply(&restored, &expand_dims_axes(&routing, &[-1], None), None);
    sum_axis(&weighted, -2, false, None)
}

/// Inverse order of a fixed permutation of `pairs` sorted rows.
fn self_check_inverse(pairs: i32) -> MlxArray {
    let count = pairs.max(0) as u32;
    // 7 is coprime to every pair count the check uses only when the count is
    // no multiple of 7, so step past it otherwise.
    let stride = if count.is_multiple_of(7) { 11 } else { 7 };
    let order: Vec<u32> = (0..count).map(|pair| (pair * stride + 3) % count).collect();
    MlxArray::from_raw_data(
        order.as_ptr().cast(),
        std::mem::size_of_val(order.as_slice()),
        &[pairs],
        MlxDtype::Uint32,
    )
}

fn self_check(top_k: i32, hidden: i32) -> Option<bool> {
    let rows = SELF_CHECK_ROWS;
    let pairs = rows.checked_mul(top_k)?;
    let down = pattern(&[pairs + SELF_CHECK_PADDING, 1, hidden], 29, 2.5);
    let inverse = self_check_inverse(pairs);
    let routing = pattern(&[1, rows, top_k], 31, 0.4);
    let actual = launch(&down, &inverse, &routing, rows, top_k, hidden)?;
    let expected = reference(&down, &inverse, &routing, rows, top_k, hidden);
    same_bits(&actual, &expected)
}

/// One thread per `(row, hidden column)`.
const SOURCE: &str = r#"
const uint index = thread_position_in_grid.x;
const uint row = index / H;
const uint column = index % H;
bfloat16_t partial[Y];
for (int j = 0; j < Y; j++) {
    partial[j] = static_cast<bfloat16_t>(0.0f);
    for (int k = j; k < K; k += Y) {
        const uint pair = row * K + k;
        const bfloat16_t product = static_cast<bfloat16_t>(
            float(down[size_t(inverse[pair]) * H + column]) * float(routing[pair]));
        partial[j] = static_cast<bfloat16_t>(float(product) + float(partial[j]));
    }
}
bfloat16_t total = partial[0];
for (int j = 1; j < Y; j++) {
    total = static_cast<bfloat16_t>(float(partial[j]) + float(total));
}
out[index] = total;
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_follows_the_small_reduction() {
        assert!(supported_geometry(10, 2560));
        assert!(supported_geometry(2, 64));
        assert!(!supported_geometry(1, 64));
        assert!(supported_geometry(31, 64));
        assert!(!supported_geometry(32, 64));
        assert!(!supported_geometry(0, 64));
        assert!(!supported_geometry(10, 0));
    }

    #[test]
    fn self_check_passes_at_the_flash_next_geometry() {
        assert_eq!(self_check(10, 2560), Some(true));
        assert_eq!(self_check(8, 2560), Some(true));
        assert_eq!(self_check(3, 128), Some(true));
        assert_eq!(self_check(17, 320), Some(true));
    }

    /// Prefill-sized gathers with a padded tail, bit for bit against the
    /// unsort, weight and sum chain.
    #[test]
    fn routed_sum_matches_the_mlx_ops_bit_for_bit_at_prefill_rows() {
        let (top_k, hidden) = (10, 2560);
        for (rows, padding, phase) in [(64, 0, 5u32), (97, 1078, 211), (209, 0, 1009)] {
            let pairs = rows * top_k;
            let down = pattern(&[pairs + padding, 1, hidden], phase, 3.0);
            let inverse = self_check_inverse(pairs);
            let routing = pattern(&[1, rows, top_k], phase + 3, 0.3);
            let actual = launch(&down, &inverse, &routing, rows, top_k, hidden).unwrap();
            let expected = reference(&down, &inverse, &routing, rows, top_k, hidden);
            assert_eq!(same_bits(&actual, &expected), Some(true), "rows {rows}");
        }
    }

    #[test]
    fn launch_rejects_mismatched_shapes() {
        let (rows, top_k, hidden) = (4, 10, 128);
        let pairs = rows * top_k;
        let down = pattern(&[pairs - 1, 1, hidden], 1, 1.0);
        let inverse = self_check_inverse(pairs);
        let routing = pattern(&[1, rows, top_k], 2, 0.3);
        assert!(launch(&down, &inverse, &routing, rows, top_k, hidden).is_none());
        let down = pattern(&[pairs, 1, hidden], 1, 1.0);
        let short = self_check_inverse(pairs - 1);
        assert!(launch(&down, &short, &routing, rows, top_k, hidden).is_none());
    }
}
