//! Native singleton recurrence. Q/K normalization, gates and QMM stay
//! on their existing MLX paths. MLX retains both F32 reductions; only the
//! elementwise products and state update are fused. The kernels read grouped
//! key heads directly, so decode no longer materialises the expanded layout.

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, reshape, sum_axis,
};
use std::sync::OnceLock;

static PREDICTION: OnceLock<MlxMetalKernel> = OnceLock::new();
static UPDATE: OnceLock<MlxMetalKernel> = OnceLock::new();

pub(super) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_MODE.with(std::cell::Cell::get) {
        return value;
    }
    // Unit tests keep the portable recurrence unless they opt in with
    // `with_mode`; production runs the native kernel unless it is switched off.
    if cfg!(test) {
        return false;
    }
    crate::fastpath::flash_next_gdn_metal_enabled()
}

/// Expanded entry point: `q`/`k` already carry one key head per value head,
/// so the kernel reads them with a unit group.
#[cfg_attr(not(test), allow(dead_code))] // covered by the component test
pub(super) fn singleton(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    decay: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
) -> Result<Option<(MlxArray, MlxArray)>, String> {
    launch(q, k, v, decay, beta, state, 1)
}

/// Grouped entry point: `q`/`k` carry `key_heads = heads / group` heads and the
/// kernels read key head `head / group` directly, so decode no longer gathers
/// the expanded layout with `flash_next_gdn::expand_heads`.
pub(super) fn singleton_grouped(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    decay: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
) -> Result<Option<(MlxArray, MlxArray)>, String> {
    let key_shape = q.shape();
    let state_shape = state.shape();
    if key_shape.len() != 4 || state_shape.len() != 4 {
        return Ok(None);
    }
    let (heads, key_heads) = (state_shape[1], key_shape[2]);
    if heads < 1 || key_heads < 1 || heads % key_heads != 0 {
        return Ok(None);
    }
    launch(q, k, v, decay, beta, state, heads / key_heads)
}

fn launch(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    decay: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
    group: i32,
) -> Result<Option<(MlxArray, MlxArray)>, String> {
    let shape = state.shape();
    if group < 1 || shape.len() != 4 || shape.iter().any(|&dim| dim <= 0) {
        return Ok(None);
    }
    let (batch, heads, values, keys) = (shape[0], shape[1], shape[2], shape[3]);
    let key_heads = heads / group;
    if !matches!(keys, 32 | 64 | 128 | 256)
        || key_heads < 1
        || heads % key_heads != 0
        || q.shape() != [batch, 1, key_heads, keys]
        || k.shape() != q.shape()
        || v.shape() != [batch, 1, heads, values]
        || decay.shape() != [batch, 1, heads]
        || beta.shape() != decay.shape()
        || [q, k, v, decay, beta, state]
            .iter()
            .any(|array| array.dtype() != MlxDtype::Float32)
    {
        return Ok(None);
    }
    let elements = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(values))
        .and_then(|n| n.checked_mul(keys))
        .ok_or("Flash Next GDN grid overflow")?;
    let prediction_kernel = PREDICTION.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_flash_next_gdn_prediction_v3",
            &["k", "decay", "state_in"],
            &["products"],
            PREDICTION_SOURCE,
            "",
            true,
        )
    });
    let update_kernel = UPDATE.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_flash_next_gdn_update_v3",
            &["q", "k", "v", "decay", "beta", "prediction", "state_in"],
            &["state_out", "products"],
            UPDATE_SOURCE,
            "",
            true,
        )
    });
    let template = [
        KernelTemplateArg::Int {
            name: "Keys",
            value: keys,
        },
        KernelTemplateArg::Int {
            name: "Values",
            value: values,
        },
        KernelTemplateArg::Int {
            name: "Elements",
            value: elements,
        },
        KernelTemplateArg::Int {
            name: "Heads",
            value: heads,
        },
        KernelTemplateArg::Int {
            name: "Group",
            value: group,
        },
    ];
    let spec = || KernelOutputSpec {
        shape: shape.clone(),
        dtype: MlxDtype::Float32,
    };
    let products = prediction_kernel
        .try_apply_with_template(
            &[k, decay, state],
            &[spec()],
            &template,
            (elements, 1, 1),
            (256, 1, 1),
            None,
        )?
        .into_iter()
        .next()
        .ok_or("Flash Next GDN omitted prediction products")?;
    // Preserve the portable reduction shape and MLX's reduction order.
    let prediction = sum_axis(&products, -1, true, None);
    let mut outputs = update_kernel
        .try_apply_with_template(
            &[q, k, v, decay, beta, &prediction, state],
            &[spec(), spec()],
            &template,
            (elements, 1, 1),
            (256, 1, 1),
            None,
        )?
        .into_iter();
    let state = outputs
        .next()
        .ok_or("Flash Next GDN omitted recurrent state")?;
    let products = outputs
        .next()
        .ok_or("Flash Next GDN omitted output products")?;
    let y = reshape(
        &sum_axis(&products, -1, false, None),
        &[batch, 1, heads, values],
        None,
    );
    #[cfg(test)]
    DISPATCHES.with(|counter| counter.set(counter.get() + 2));
    Ok(Some((y, state)))
}

#[cfg(test)]
thread_local! {
    static TEST_MODE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static DISPATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn with_mode<T>(enabled: bool, action: impl FnOnce() -> T) -> (T, u64) {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_MODE.with(|mode| mode.set(self.0));
        }
    }
    let restore = Restore(TEST_MODE.with(|mode| mode.replace(Some(enabled))));
    let before = DISPATCHES.with(std::cell::Cell::get);
    let result = action();
    let count = DISPATCHES.with(std::cell::Cell::get) - before;
    drop(restore);
    (result, count)
}

const PREDICTION_SOURCE: &str = r#"
#pragma clang fp contract(off)
const size_t index = thread_position_in_grid.x;
if (index >= Elements) return;
const size_t head = index / (Values * Keys);
// `head` folds batch and value head together, so rebase it per batch before
// mapping it onto the unexpanded key heads of `k`.
const size_t key_head = ((head % Heads) / Group) + (head / Heads) * (Heads / Group);
const uint key = index % Keys;
const float decayed = state_in[index] * decay[head];
products[index] = decayed * k[key_head * Keys + key];
"#;

const UPDATE_SOURCE: &str = r#"
#pragma clang fp contract(off)
const size_t index = thread_position_in_grid.x;
if (index >= Elements) return;
const size_t row = index / Keys;
const size_t head = row / Values;
// See the prediction kernel: `head` folds batch and value head together.
const size_t key_head = ((head % Heads) / Group) + (head / Heads) * (Heads / Group);
const uint key = index % Keys;
const float decayed = state_in[index] * decay[head];
const float residual = v[row] - prediction[row];
const float correction = residual * beta[head];
const float update = correction * k[key_head * Keys + key];
const float updated = decayed + update;
state_out[index] = updated;
products[index] = updated * q[key_head * Keys + key];
"#;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use mlx_sys::{add, eval, multiply, reshape, subtract, sum_axis};

    fn data(shape: &[i32], scale: f32, offset: f32) -> MlxArray {
        let values: Vec<f32> = (0..shape.iter().product::<i32>())
            .map(|i| ((i * 17 + 3) % 29 - 14) as f32 * scale + offset)
            .collect();
        MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values.as_slice()),
            shape,
            MlxDtype::Float32,
        )
    }

    fn close(actual: &MlxArray, expected: &MlxArray, label: &str) {
        eval(&[actual, expected]);
        assert_eq!(actual.shape(), expected.shape());
        let error = actual
            .data_f32()
            .iter()
            .zip(expected.data_f32())
            .map(|(&a, &b)| {
                assert!(a.is_finite() && b.is_finite());
                (a - b).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(error < 2e-6, "{label}: error {error}");
        assert_eq!(
            actual.data_f32(),
            expected.data_f32(),
            "{label}: exact arithmetic"
        );
    }

    #[test]
    fn singleton_repeated_state_matches_portable_recurrence() {
        for keys in [32, 64, 128, 256] {
            let (batch, heads, values) = (2, 3, 7);
            let q = data(&[batch, 1, heads, keys], 0.004, 0.0);
            let k = data(&[batch, 1, heads, keys], 0.007, 0.0);
            let v = data(&[batch, 1, heads, values], 0.04, 0.0);
            let decay = data(&[batch, 1, heads], 0.001, 0.95);
            let beta = data(&[batch, 1, heads], 0.001, 0.4);
            let mut native = data(&[batch, heads, values, keys], 0.005, 0.0);
            let mut portable = native.clone();
            for step in 0..32 {
                let (y, next) = singleton(&q, &k, &v, &decay, &beta, &native)
                    .unwrap()
                    .unwrap();
                let qr = reshape(&q, &[batch, heads, 1, keys], None);
                let kr = reshape(&k, &[batch, heads, 1, keys], None);
                let decayed = multiply(
                    &portable,
                    &reshape(&decay, &[batch, heads, 1, 1], None),
                    None,
                );
                let prediction = sum_axis(&multiply(&decayed, &kr, None), -1, true, None);
                let correction = multiply(
                    &subtract(
                        &reshape(&v, &[batch, heads, values, 1], None),
                        &prediction,
                        None,
                    ),
                    &reshape(&beta, &[batch, heads, 1, 1], None),
                    None,
                );
                portable = add(&decayed, &multiply(&correction, &kr, None), None);
                let reference = reshape(
                    &sum_axis(&multiply(&portable, &qr, None), -1, false, None),
                    &[batch, 1, heads, values],
                    None,
                );
                close(&y, &reference, &format!("output keys={keys} step={step}"));
                close(&next, &portable, &format!("state keys={keys} step={step}"));
                native = next;
            }
        }
    }

    #[test]
    fn unsupported_geometry_and_dtype_do_not_dispatch() {
        let q = data(&[1, 1, 2, 4], 0.1, 0.0);
        let v = data(&[1, 1, 2, 8], 0.1, 0.0);
        let gate = data(&[1, 1, 2], 0.0, 0.5);
        let state = data(&[1, 2, 8, 4], 0.0, 0.0);
        assert!(
            singleton(&q, &q, &v, &gate, &gate, &state)
                .unwrap()
                .is_none()
        );
        let q = data(&[1, 1, 2, 32], 0.1, 0.0);
        let state = data(&[1, 2, 8, 32], 0.0, 0.0);
        let wrong = mlx_sys::astype(&q, MlxDtype::Bfloat16, None);
        assert!(
            singleton(&wrong, &q, &v, &gate, &gate, &state)
                .unwrap()
                .is_none()
        );
        let wrong = data(&[1, 2, 2, 32], 0.1, 0.0);
        assert!(
            singleton(&wrong, &q, &v, &gate, &gate, &state)
                .unwrap()
                .is_none()
        );
    }

    /// Grouped key heads must read the same bits the `expand_heads` gather
    /// produces, so the expanded entry point stays the reference.
    #[test]
    fn grouped_singleton_matches_expanded_singleton() {
        // (value heads, key heads, keys, batch, q/k magnitude)
        let cases = [
            (48, 16, 64, 1, 1.0_f32),
            (48, 16, 128, 1, 1.0),
            (48, 16, 64, 2, 1.0),
            (48, 16, 128, 2, 1.0),
            (8, 4, 64, 1, 1.0),
            (8, 4, 128, 2, 1.0),
            (8, 8, 64, 1, 1.0),
            (8, 8, 128, 2, 1.0),
            // Subnormal-range q/k: the grouped read must copy the exact bits a
            // gather would have moved, so both routes stay bit-identical.
            (48, 16, 128, 2, 1e-40),
            (8, 4, 64, 1, 1e-40),
        ];
        let values = 5;
        for (heads, key_heads, keys, batch, magnitude) in cases {
            let q = data(&[batch, 1, key_heads, keys], 0.004 * magnitude, 0.0);
            let k = data(&[batch, 1, key_heads, keys], 0.007 * magnitude, 0.0);
            let v = data(&[batch, 1, heads, values], 0.04, 0.0);
            let decay = data(&[batch, 1, heads], 0.001, 0.95);
            let beta = data(&[batch, 1, heads], 0.001, 0.4);
            let mut state = data(&[batch, heads, values, keys], 0.005, 0.0);
            let (expanded_q, expanded_k) = crate::flash_next_gdn::expand_heads(&q, &k, heads);
            // The kernels keep one arithmetic order, so the portable route is
            // comparable bit-for-bit wherever no subnormal product appears.
            let portable_comparable = magnitude == 1.0;
            let mut portable = state.clone();
            for step in 0..8 {
                let label = format!(
                    "heads={heads} key_heads={key_heads} keys={keys} batch={batch} step={step}"
                );
                let (y, next) = singleton_grouped(&q, &k, &v, &decay, &beta, &state)
                    .unwrap()
                    .unwrap();
                let (expanded_y, expanded_next) =
                    singleton(&expanded_q, &expanded_k, &v, &decay, &beta, &state)
                        .unwrap()
                        .unwrap();
                close(&y, &expanded_y, &format!("{label} output vs expanded"));
                close(&next, &expanded_next, &format!("{label} state vs expanded"));
                if portable_comparable {
                    let (portable_y, portable_next) = crate::flash_next_gdn::portable_recurrence(
                        &expanded_q,
                        &expanded_k,
                        &v,
                        &decay,
                        &beta,
                        &portable,
                    )
                    .unwrap();
                    close(&y, &portable_y, &format!("{label} output vs portable"));
                    close(&next, &portable_next, &format!("{label} state vs portable"));
                    portable = portable_next;
                }
                state = next;
            }
        }
    }

    #[test]
    fn grouped_singleton_rejects_unsupported_grouping() {
        let (heads, values) = (8, 8);
        let v = data(&[1, 1, heads, values], 0.1, 0.0);
        let gate = data(&[1, 1, heads], 0.0, 0.5);
        let state = data(&[1, heads, values, 64], 0.0, 0.0);
        // Three key heads do not divide the eight value heads.
        let q = data(&[1, 1, 3, 64], 0.1, 0.0);
        assert!(
            singleton_grouped(&q, &q, &v, &gate, &gate, &state)
                .unwrap()
                .is_none()
        );
        // A grouped call keeps the expanded entry point's dtype restriction.
        let q = data(&[1, 1, 4, 64], 0.1, 0.0);
        let half = mlx_sys::astype(&q, MlxDtype::Bfloat16, None);
        assert!(
            singleton_grouped(&half, &half, &v, &gate, &gate, &state)
                .unwrap()
                .is_none()
        );
        // Unsupported key widths still fail closed after a valid grouping.
        let q = data(&[1, 1, 2, 48], 0.1, 0.0);
        let state = data(&[1, heads, values, 48], 0.0, 0.0);
        assert!(
            singleton_grouped(&q, &q, &v, &gate, &gate, &state)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn grouped_singleton_dispatches_two_kernels_per_update() {
        let (batch, heads, key_heads, values, keys) = (2, 8, 4, 5, 64);
        let q = data(&[batch, 1, key_heads, keys], 0.004, 0.0);
        let k = data(&[batch, 1, key_heads, keys], 0.007, 0.0);
        let v = data(&[batch, 1, heads, values], 0.04, 0.0);
        let decay = data(&[batch, 1, heads], 0.001, 0.95);
        let beta = data(&[batch, 1, heads], 0.001, 0.4);
        let state = data(&[batch, heads, values, keys], 0.005, 0.0);
        let ((_, _), grouped) = with_mode(true, || {
            singleton_grouped(&q, &k, &v, &decay, &beta, &state)
                .unwrap()
                .unwrap()
        });
        assert_eq!(
            grouped, 2,
            "prediction and update kernels dispatch once each"
        );
        let (expanded_q, expanded_k) = crate::flash_next_gdn::expand_heads(&q, &k, heads);
        let ((_, _), expanded) = with_mode(true, || {
            singleton(&expanded_q, &expanded_k, &v, &decay, &beta, &state)
                .unwrap()
                .unwrap()
        });
        assert_eq!(
            expanded, 2,
            "the expanded entry point keeps both dispatches"
        );
        // A mismatched key width still fails closed before any dispatch.
        let (_, rejected) = with_mode(true, || {
            singleton_grouped(&v, &v, &v, &decay, &beta, &state)
                .unwrap()
                .is_none()
        });
        assert_eq!(rejected, 0, "unsupported geometry dispatches nothing");
    }
}
