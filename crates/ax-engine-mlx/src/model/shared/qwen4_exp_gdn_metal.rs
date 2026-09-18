//! Experimental singleton recurrence. Q/K normalization, gates and QMM stay
//! on their existing MLX paths. MLX retains both F32 reductions; only the
//! elementwise products and state update are fused.

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
    std::env::var("AX_MLX_FLASH_NEXT_GDN_METAL")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
}

pub(super) fn singleton(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    decay: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
) -> Result<Option<(MlxArray, MlxArray)>, String> {
    let shape = state.shape();
    if shape.len() != 4 || shape.iter().any(|&dim| dim <= 0) {
        return Ok(None);
    }
    let (batch, heads, values, keys) = (shape[0], shape[1], shape[2], shape[3]);
    if !matches!(keys, 32 | 64 | 128 | 256)
        || q.shape() != [batch, 1, heads, keys]
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
            "ax_flash_next_gdn_prediction_v2",
            &["k", "decay", "state_in"],
            &["products"],
            PREDICTION_SOURCE,
            "",
            true,
        )
    });
    let update_kernel = UPDATE.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_flash_next_gdn_update_v2",
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
const uint key = index % Keys;
const float decayed = state_in[index] * decay[head];
products[index] = decayed * k[head * Keys + key];
"#;

const UPDATE_SOURCE: &str = r#"
#pragma clang fp contract(off)
const size_t index = thread_position_in_grid.x;
if (index >= Elements) return;
const size_t row = index / Keys;
const size_t head = row / Values;
const uint key = index % Keys;
const float decayed = state_in[index] * decay[head];
const float residual = v[row] - prediction[row];
const float correction = residual * beta[head];
const float update = correction * k[head * Keys + key];
const float updated = decayed + update;
state_out[index] = updated;
products[index] = updated * q[head * Keys + key];
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
}
