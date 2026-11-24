use super::{DUMP_CHUNK_LEN, DUMP_LAYER, DUMP_OFFSET};
use mlx_sys::{MlxArray, MlxDtype, astype, concatenate, contiguous, slice, try_eval};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cell::RefCell;

const PREFIX: usize = 2047;
const SPLIT: [usize; 4] = [512, 512, 512, 511];
const INPUT_NAMES: [&str; 5] = ["query", "key", "value", "decay", "beta"];
const DECAY_NAMES: [&str; 6] = [
    "projection",
    "biased_projection",
    "negative_abs_exp",
    "log1p",
    "softplus",
    "log_decay",
];
const DECAY_WITNESSES: [usize; 6] = [493, 494, 495, 1039, 1040, 1041];

struct Chunk {
    start: usize,
    len: usize,
    inputs: [MlxArray; 5],
    initial: MlxArray,
    result: Option<(MlxArray, MlxArray)>,
}

#[derive(Default)]
struct Capture {
    chunks: Vec<Chunk>,
    decay: Vec<(usize, [MlxArray; DECAY_NAMES.len()])>,
}

thread_local! {
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}

pub(crate) fn begin() {
    CAPTURE.with_borrow_mut(|capture| assert!(capture.replace(Capture::default()).is_none()));
}

fn in_prefix() -> bool {
    DUMP_LAYER.get() == 0
        && DUMP_CHUNK_LEN.get() > 0
        && DUMP_OFFSET.get() + DUMP_CHUNK_LEN.get() <= PREFIX
}

pub(crate) fn record_decay_stages(stages: [&MlxArray; DECAY_NAMES.len()]) {
    if in_prefix() {
        CAPTURE.with_borrow_mut(|capture| {
            if let Some(capture) = capture {
                capture
                    .decay
                    .push((DUMP_OFFSET.get(), stages.map(Clone::clone)));
            }
        });
    }
}

pub(crate) fn record_inputs([q, k, v, decay, beta, initial]: [&MlxArray; 6]) {
    if in_prefix() {
        CAPTURE.with_borrow_mut(|capture| {
            if let Some(capture) = capture {
                capture.chunks.push(Chunk {
                    start: DUMP_OFFSET.get(),
                    len: DUMP_CHUNK_LEN.get(),
                    inputs: [q, k, v, decay, beta].map(Clone::clone),
                    initial: initial.clone(),
                    result: None,
                });
            }
        });
    }
}

pub(crate) fn record_outputs(output: &MlxArray, state: &MlxArray, used_mlx_prefill: bool) {
    if in_prefix() {
        CAPTURE.with_borrow_mut(|capture| {
            if let Some(capture) = capture {
                assert!(used_mlx_prefill);
                let chunk = capture.chunks.last_mut().unwrap();
                assert_eq!(chunk.start, DUMP_OFFSET.get());
                assert!(
                    chunk
                        .result
                        .replace((output.clone(), state.clone()))
                        .is_none()
                );
            }
        });
    }
}

pub(crate) fn finish() -> Value {
    // Materialize the captured handles only after the production forward and
    // the output-ID control have completed.
    let Capture { chunks, decay } = CAPTURE.with_borrow_mut(|capture| capture.take().unwrap());
    assert_eq!(decay.len(), chunks.len());
    let partition: Vec<usize> = chunks.iter().map(|chunk| chunk.len).collect();
    assert!(partition == [PREFIX] || partition == SPLIT);
    let mut position = 0;
    let mut states: Vec<String> = Vec::new();
    for (chunk, (decay_start, _)) in chunks.iter().zip(&decay) {
        assert_eq!(chunk.start, position);
        assert_eq!(*decay_start, position);
        let initial = contiguous(&chunk.initial, None);
        try_eval(&[&initial]).unwrap();
        let initial_hash = hash(initial.data_f32());
        if let Some(previous) = states.last() {
            assert_eq!(previous, &initial_hash);
        } else {
            assert!(initial.data_f32().iter().all(|&value| value.to_bits() == 0));
            states.push(initial_hash);
        }
        states.push(state_hash(&chunk.result.as_ref().unwrap().1));
        position += chunk.len;
    }
    assert_eq!(position, PREFIX);

    let inputs: [MlxArray; 5] = std::array::from_fn(|index| {
        join_rows(chunks.iter().map(|chunk| &chunk.inputs[index]).collect())
    });
    let input_rows: serde_json::Map<String, Value> = INPUT_NAMES
        .iter()
        .zip(inputs.iter())
        .map(|(name, input)| ((*name).to_owned(), row_hashes(input)))
        .collect();
    let decay_stages: serde_json::Map<String, Value> = DECAY_NAMES
        .iter()
        .enumerate()
        .map(|(index, &name)| {
            let values = join_rows(decay.iter().map(|(_, stages)| &stages[index]).collect());
            let values = contiguous(&astype(&values, MlxDtype::Float32, None), None);
            (name.to_owned(), decay_witnesses(&values))
        })
        .chain([("decay".to_owned(), decay_witnesses(&inputs[3]))])
        .collect();
    let original = join_rows(
        chunks
            .iter()
            .map(|chunk| &chunk.result.as_ref().unwrap().0)
            .collect(),
    );
    let initial = contiguous(&chunks[0].initial, None);
    json!({
        "recurrence": "mlx_gated_delta_update",
        "layer": 0,
        "partition": partition,
        "input_rows": input_rows,
        "decay_stages": decay_stages,
        "original_output": row_hashes(&original),
        "original_boundary_states": states,
        "replays": {
            "whole": replay(&inputs, &initial, &[PREFIX], false),
            "split_lazy": replay(&inputs, &initial, &SPLIT, false),
            "split_fenced": replay(&inputs, &initial, &SPLIT, true),
        },
    })
}

fn decay_witnesses(array: &MlxArray) -> Value {
    let rows = row_hashes(array);
    let values = array.data_f32();
    let width = values.len() / PREFIX;
    let witnesses: Vec<Value> = DECAY_WITNESSES
        .iter()
        .map(|&position| {
            let bits: Vec<u32> = values[position * width..(position + 1) * width]
                .iter()
                .map(|value| value.to_bits())
                .collect();
            json!({"position": position, "bits": bits})
        })
        .collect();
    json!({"rows": rows, "witnesses": witnesses})
}

fn join_rows(rows: Vec<&MlxArray>) -> MlxArray {
    match rows.as_slice() {
        [one] => contiguous(one, None),
        _ => contiguous(&concatenate(&rows, 1, None), None),
    }
}

fn replay(inputs: &[MlxArray; 5], initial: &MlxArray, partition: &[usize], fence: bool) -> Value {
    try_eval(&[initial]).unwrap();
    let mut state = initial.clone();
    let mut outputs = Vec::with_capacity(partition.len());
    let mut offset = 0;
    for &len in partition {
        let [q, k, v, decay, beta] = if len == PREFIX {
            inputs.clone()
        } else {
            inputs.each_ref().map(|input| {
                let mut end = input.shape();
                let mut start = vec![0; end.len()];
                let stride = vec![1; end.len()];
                start[1] = offset as i32;
                end[1] = (offset + len) as i32;
                slice(input, &start, &end, &stride, None)
            })
        };
        let (output, next) =
            crate::flash_next_gdn::try_mlx_prefill(&q, &k, &v, &decay, &beta, &state).unwrap();
        if fence {
            try_eval(&[&output, &next]).unwrap();
        }
        outputs.push(output);
        state = next;
        offset += len;
    }
    assert_eq!(offset, PREFIX);
    let output = join_rows(outputs.iter().collect());
    try_eval(&[&output, &state]).unwrap();
    json!({"output": row_hashes(&output), "final_state": state_hash(&state)})
}

fn row_hashes(array: &MlxArray) -> Value {
    assert_eq!(array.dtype(), MlxDtype::Float32);
    let shape = array.shape();
    assert_eq!(shape[..2], [1, PREFIX as i32]);
    try_eval(&[array]).unwrap();
    let values = array.data_f32();
    let width = values.len() / PREFIX;
    assert_eq!(values.len(), width * PREFIX);
    let rows: Vec<String> = values.chunks_exact(width).map(hash).collect();
    json!({"shape": shape, "rows": rows})
}

fn state_hash(array: &MlxArray) -> String {
    let array = contiguous(array, None);
    try_eval(&[&array]).unwrap();
    hash(array.data_f32())
}

fn hash(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        assert!(value.is_finite());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    format!("{:x}", Sha256::digest(bytes))
}
