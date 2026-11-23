//! Experimental copy into the existing Flash Next selected-attention layout.

use std::sync::OnceLock;

use mlx_sys::{KernelOutputSpec, LazyKernel, MlxArray, MlxDtype, try_eval, zeros};

use super::qwen4_exp_kv_history::KvHistory;

const KV_HEADS: i32 = 2;
const HEAD_DIM: i32 = 256;
const VECTORS_PER_HEAD: i32 = HEAD_DIM / 8;
const MAX_HISTORY_TOKENS: i32 = 8192;

static GATHER: LazyKernel = LazyKernel::new(
    "ax_flash_next_qsa_paired_gather_v1",
    &[
        "key_prefix",
        "value_prefix",
        "key_tail",
        "value_tail",
        "selection",
        "geometry",
    ],
    &["key_out", "value_out"],
    include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../metal/kernels/flash_next_qsa_paired_gather.metal"
    )),
    "",
    true,
);

pub(super) struct PairedGather<'a> {
    segments: [&'a MlxArray; 4],
    sealed_rows: i32,
    tokens: i32,
}

impl<'a> PairedGather<'a> {
    pub(super) fn new(keys: &'a KvHistory, values: &'a KvHistory) -> Option<Self> {
        let layout = |array: &MlxArray| match array.shape().as_slice() {
            [1, rows, KV_HEADS, HEAD_DIM] if *rows > 0 && array.dtype() == MlxDtype::Bfloat16 => {
                Some(*rows)
            }
            _ => None,
        };
        let (segments, sealed_rows, tokens) = match (keys.segments(), values.segments()) {
            ((Some(k), None), (Some(v), None)) | ((None, Some(k)), (None, Some(v))) => {
                let tokens = layout(k)?;
                if layout(v)? != tokens {
                    return None;
                }
                // A one-segment history only reads the tail arguments.
                ([k, v, k, v], 0, tokens)
            }
            ((Some(pk), Some(tk)), (Some(pv), Some(tv))) => {
                let sealed_rows = layout(pk)?;
                let tail_rows = layout(tk)?;
                if layout(pv)? != sealed_rows || layout(tv)? != tail_rows {
                    return None;
                }
                (
                    [pk, pv, tk, tv],
                    sealed_rows,
                    sealed_rows.checked_add(tail_rows)?,
                )
            }
            _ => return None,
        };
        (tokens <= MAX_HISTORY_TOKENS).then_some(Self {
            segments,
            sealed_rows,
            tokens,
        })
    }

    pub(super) fn gather(&self, selection: &MlxArray) -> Option<(MlxArray, MlxArray)> {
        if !pipeline_ready() {
            return None;
        }
        let gathered = self.launch(selection)?;
        #[cfg(test)]
        DISPATCHES.with(|count| count.set(count.get() + 1));
        Some(gathered)
    }

    fn launch(&self, selection: &MlxArray) -> Option<(MlxArray, MlxArray)> {
        let shape = selection.shape();
        let &[queries, width] = shape.as_slice() else {
            return None;
        };
        if queries <= 0 || width <= 0 || selection.dtype() != MlxDtype::Int32 {
            return None;
        }
        let packets_per_row = width.checked_mul(VECTORS_PER_HEAD)?;
        packets_per_row
            .checked_mul(queries)?
            .checked_mul(KV_HEADS)?;
        let output = || KernelOutputSpec {
            shape: vec![queries, KV_HEADS, width, HEAD_DIM],
            dtype: MlxDtype::Bfloat16,
        };
        let [pk, pv, tk, tv] = self.segments;
        // Geometry varies with the request. Pass values to one pipeline
        // instead of compiling a kernel per tile and segment length.
        let geometry = [width, self.sealed_rows, self.tokens];
        let geometry = MlxArray::from_raw_data(
            geometry.as_ptr().cast(),
            std::mem::size_of_val(&geometry),
            &[3],
            MlxDtype::Int32,
        );
        let result = GATHER.get()?.try_apply_with_template(
            &[pk, pv, tk, tv, selection, &geometry],
            &[output(), output()],
            &[],
            (packets_per_row, KV_HEADS, queries),
            (256, 1, 1),
            None,
        );
        let Ok(outputs) = result else {
            return None;
        };
        let [keys, values]: [MlxArray; 2] = outputs.try_into().ok()?;
        Some((keys, values))
    }
}

/// MLX compiles a custom Metal pipeline when its graph evaluates. Check
/// that pipeline once without synchronizing request tiles.
fn pipeline_ready() -> bool {
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| {
        let rows =
            KvHistory::from_flat(zeros(&[1, 1, KV_HEADS, HEAD_DIM], MlxDtype::Bfloat16, None));
        let indices = zeros(&[1, 1], MlxDtype::Int32, None);
        let ready = PairedGather::new(&rows, &rows)
            .and_then(|gather| gather.launch(&indices))
            .is_some_and(|(keys, values)| try_eval(&[&keys, &values]).is_ok());
        if !ready {
            tracing::warn!("Flash Next paired QSA gather failed to warm up; using stock gather");
        }
        ready
    })
}

pub(super) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    crate::fastpath::flash_next_qsa_paired_gather_enabled()
}

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static DISPATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn with_paired_gather<T>(enabled: bool, run: impl FnOnce() -> T) -> (T, usize) {
    struct Restore(Option<bool>, usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|slot| slot.set(self.0));
            DISPATCHES.with(|count| count.set(self.1));
        }
    }
    let _restore = Restore(
        OVERRIDE.with(|slot| slot.replace(Some(enabled))),
        DISPATCHES.with(|count| count.replace(0)),
    );
    (run(), DISPATCHES.with(std::cell::Cell::get))
}
