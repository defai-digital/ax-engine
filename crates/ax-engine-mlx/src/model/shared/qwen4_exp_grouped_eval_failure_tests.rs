use std::cell::Cell;

use mlx_sys::{KernelOutputSpec, MlxArray, MlxMetalKernel, try_eval};

thread_local! {
    static PAIRS_BEFORE_FAILURE: Cell<Option<u8>> = const { Cell::new(None) };
}

struct ResetFault;

impl Drop for ResetFault {
    fn drop(&mut self) {
        PAIRS_BEFORE_FAILURE.with(|remaining| remaining.set(None));
    }
}

pub(crate) fn with_grouped_eval_compile_failure<T>(action: impl FnOnce() -> T) -> T {
    PAIRS_BEFORE_FAILURE.with(|remaining| assert_eq!(remaining.replace(Some(1)), None));
    let _guard = ResetFault;
    let result = action();
    PAIRS_BEFORE_FAILURE.with(|remaining| assert_eq!(remaining.get(), None, "fault not reached"));
    result
}

pub(super) fn evaluate(tiles: &[&MlxArray]) -> Result<(), String> {
    if let [first, second] = tiles
        && PAIRS_BEFORE_FAILURE.with(|remaining| match remaining.get() {
            Some(0) => {
                remaining.set(None);
                true
            }
            Some(count) => {
                remaining.set(Some(count - 1));
                false
            }
            None => false,
        })
    {
        // The first pair already completed. Materialize the first output
        // of this pair before the second output's kernel fails to compile.
        try_eval(&[first])?;
        let kernel = MlxMetalKernel::new(
            "ax_flash_next_grouped_eval_fault",
            &["input"],
            &["output"],
            "output[thread_position_in_grid.x] = ax_grouped_qsa_missing_symbol;",
            "",
            true,
        );
        let shape = second.shape();
        let count = shape.iter().product();
        let broken = kernel.try_apply_with_template(
            &[second],
            &[KernelOutputSpec {
                shape,
                dtype: second.dtype(),
            }],
            &[],
            (count, 1, 1),
            (32, 1, 1),
            None,
        )?;
        let [output] = broken.as_slice() else {
            return Err("fault kernel must have one output".into());
        };
        return try_eval(&[output]);
    }
    try_eval(tiles)
}
