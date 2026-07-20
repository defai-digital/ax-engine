//! Regression coverage for the shim's row-contiguity guard on raw data
//! accessors: reading a transposed (strided) view as a dense slice used to
//! silently return re-ordered data.

use mlx_sys::{MlxArray, eval, ops};

#[test]
fn contiguous_data_read_round_trips() {
    let flat = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let arr = ops::reshape(&flat, &[2, 3], None);
    eval(&[&arr]);
    assert_eq!(arr.data_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn non_contiguous_view_read_fails_loudly() {
    let flat = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let arr = ops::reshape(&flat, &[2, 3], None);
    let transposed = ops::transpose(&arr, &[1, 0], None);
    eval(&[&transposed]);

    // The transposed view shares the dense buffer with `arr`; a raw read
    // would yield row-major order of the ORIGINAL layout, not the transpose.
    // The shim now rejects it instead of returning silently wrong data.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = transposed.data_f32();
    }));
    assert!(
        result.is_err(),
        "reading a non-row-contiguous view must fail, not reorder silently"
    );

    // Materializing a contiguous copy makes the same data readable.
    let contiguous = ops::contiguous(&transposed, None);
    eval(&[&contiguous]);
    assert_eq!(contiguous.data_f32(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
