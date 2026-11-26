#![allow(clippy::expect_used, clippy::unwrap_used)]

//! Masked dense prefill attention against the per-query selected path.

use super::*;
use crate::qwen4_exp_qsa::{QsaConfig, QsaIndexerWeights, with_prefill_device_select};
use mlx_sys::eval;

const Q_HEADS: i32 = 4;
const KV_HEADS: i32 = 2;
/// A head dim MLX's multi-query attention kernel is compiled for.
const HEAD_DIM: i32 = 64;
/// Tokens in the history before the chunk's first query.
const PREFIX: i32 = 40;
/// Queries in the chunk, above [`MASKED_DENSE_MIN_QUERIES`].
const ROWS: i32 = 96;

fn cfg() -> Qwen4ExpAttentionConfig {
    Qwen4ExpAttentionConfig::new(
        (Q_HEADS * HEAD_DIM) as usize,
        Q_HEADS as usize,
        KV_HEADS as usize,
        HEAD_DIM as usize,
        4,
        10_000.0,
        1e-6,
    )
    .unwrap()
}

fn wave(shape: &[i32], seed: f32, dtype: MlxDtype) -> MlxArray {
    let count: i32 = shape.iter().product();
    let data: Vec<f32> = (0..count).map(|i| (i as f32 * 0.37 + seed).sin()).collect();
    astype(
        &MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            shape,
            MlxDtype::Float32,
        ),
        dtype,
        None,
    )
}

fn queries(rows: i32, dtype: MlxDtype) -> MlxArray {
    wave(&[1, Q_HEADS, rows, HEAD_DIM], 0.5, dtype)
}

fn history(tokens: i32, seed: f32, dtype: MlxDtype) -> MlxArray {
    wave(&[1, tokens, KV_HEADS, HEAD_DIM], seed, dtype)
}

/// History tokens no query of the chunk selects.
fn never_selected(token: i32) -> bool {
    token < PREFIX && token % 11 == 3
}

/// The tokens query `row` selects: a different subset of its visible history
/// for every row, of a different length, newest first, always with its own
/// position. Selection order differs from cache order on purpose.
fn selected_tokens(row: i32) -> Vec<i32> {
    let position = PREFIX + row;
    (0..=position)
        .rev()
        .filter(|&token| {
            token == position || (!never_selected(token) && (token * 7 + row * 3) % 5 != 0)
        })
        .collect()
}

fn selection(rows: std::ops::Range<i32>) -> QsaSelection {
    QsaSelection::from_host_tokens(vec![rows.map(selected_tokens).collect()])
}

fn bits(array: &MlxArray) -> Vec<u32> {
    let array = contiguous(&astype(array, MlxDtype::Float32, None), None);
    eval(&[&array]);
    array
        .data_f32()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn values(array: &MlxArray) -> Vec<f32> {
    bits(array).into_iter().map(f32::from_bits).collect()
}

/// Largest element difference. A value that is not finite on either side
/// fails here: `f32::max` would drop a NaN and report agreement.
fn max_abs_diff(left: &MlxArray, right: &MlxArray) -> f32 {
    let (left, right) = (values(left), values(right));
    assert_eq!(left.len(), right.len(), "output size");
    let mut largest = 0.0_f32;
    for (index, (a, b)) in left.iter().zip(&right).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "element {index} is not finite: {a} against {b}"
        );
        largest = largest.max((a - b).abs());
    }
    largest
}

fn query_rows(array: &MlxArray, rows: std::ops::Range<i32>) -> MlxArray {
    slice(
        array,
        &[0, 0, rows.start, 0],
        &[1, Q_HEADS, rows.end, HEAD_DIM],
        &[1, 1, 1, 1],
        None,
    )
}

fn masked(
    q: &MlxArray,
    k: &impl SelectedRows,
    v: &impl SelectedRows,
    selection: &QsaSelection,
) -> MlxArray {
    let before = masked_dense_calls_for_test();
    let out = with_masked_dense(true, || attend_selected(q, k, v, selection, cfg()).unwrap());
    assert_eq!(
        masked_dense_calls_for_test(),
        before + 1,
        "expected one masked dense call"
    );
    out
}

#[test]
fn masked_dense_attends_what_each_query_selected() {
    for (dtype, tolerance) in [(MlxDtype::Float32, 2e-5), (MlxDtype::Bfloat16, 2e-2)] {
        let tokens = PREFIX + ROWS;
        let q = queries(ROWS, dtype);
        let k = history(tokens, 0.75, dtype);
        let v = history(tokens, 1.25, dtype);
        let selection = selection(0..ROWS);
        let lengths: Vec<usize> = (0..ROWS as usize)
            .map(|row| selection.selected_len(0, row))
            .collect();
        assert!(
            lengths.iter().min() != lengths.iter().max(),
            "the chunk must mix selected lengths so padding is exercised"
        );

        let dense = masked(&q, &k, &v, &selection);
        assert_eq!(dense.shape(), vec![1, Q_HEADS, ROWS, HEAD_DIM]);
        // One exact-length gather and attention call per query.
        let per_query = with_masked_dense(false, || {
            with_batched_attention(false, || {
                attend_selected(&q, &k, &v, &selection, cfg()).unwrap()
            })
        });
        let padded = with_masked_dense(false, || {
            with_batched_attention(true, || {
                attend_selected(&q, &k, &v, &selection, cfg()).unwrap()
            })
        });
        let largest = values(&per_query)
            .into_iter()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()));
        assert!(largest > 0.1, "{dtype:?}: degenerate reference {largest}");
        for (name, reference) in [("per-query", &per_query), ("padded", &padded)] {
            let diff = max_abs_diff(&dense, reference);
            assert!(
                diff <= tolerance,
                "{dtype:?}: masked dense against {name} attention differs by {diff}"
            );
        }
    }
}

#[test]
fn a_query_row_does_not_depend_on_the_rows_around_it() {
    for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16] {
        let tokens = PREFIX + ROWS;
        let q = queries(ROWS, dtype);
        let k = history(tokens, 0.75, dtype);
        let v = history(tokens, 1.25, dtype);
        let whole = masked(&q, &k, &v, &selection(0..ROWS));

        // The last 70 rows as their own chunk over the same history.
        let tail = 26..ROWS;
        let alone = masked(
            &query_rows(&q, tail.clone()),
            &k,
            &v,
            &selection(tail.clone()),
        );
        assert_eq!(
            bits(&query_rows(&whole, tail)),
            bits(&alone),
            "{dtype:?}: a row changed with the rows around it"
        );
    }
}

#[test]
fn tokens_outside_the_selection_do_not_change_a_bit() {
    for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16] {
        let tokens = PREFIX + ROWS;
        let q = queries(ROWS, dtype);
        let k = history(tokens, 0.75, dtype);
        let v = history(tokens, 1.25, dtype);
        // Push every never-selected token's key and value far away. A large
        // key would win the softmax if the mask let it through.
        let bump: Vec<f32> = (0..tokens)
            .flat_map(|token| {
                let offset = if never_selected(token) { 64.0 } else { 0.0 };
                std::iter::repeat_n(offset, (KV_HEADS * HEAD_DIM) as usize)
            })
            .collect();
        let bump = astype(
            &MlxArray::from_raw_data(
                bump.as_ptr().cast(),
                std::mem::size_of_val(bump.as_slice()),
                &[1, tokens, KV_HEADS, HEAD_DIM],
                MlxDtype::Float32,
            ),
            dtype,
            None,
        );
        let selection = selection(0..ROWS);
        let plain = masked(&q, &k, &v, &selection);
        let bumped = masked(&q, &add(&k, &bump, None), &add(&v, &bump, None), &selection);
        assert_eq!(bits(&plain), bits(&bumped), "{dtype:?}");
    }
}

#[test]
fn device_and_host_selections_keep_the_same_tokens() {
    const INDEX_HIDDEN: i32 = 16;
    let weights = |seed: f32| {
        let data: Vec<f32> = (0..12 * INDEX_HIDDEN)
            .map(|i| (i as f32 * 0.017 + seed).sin())
            .collect();
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[12, INDEX_HIDDEN],
            MlxDtype::Float32,
        )
    };
    let ones = MlxArray::from_raw_data([1.0_f32; 4].as_ptr().cast(), 16, &[4], MlxDtype::Float32);
    let indexer = QsaIndexer::new(
        QsaConfig::new(2, 1, 4, 4, 8, 8, INDEX_HIDDEN as usize, 1e-6, 10_000.0).unwrap(),
        QsaIndexerWeights {
            qk_proj: QuantizedWeight::new(weights(0.0), None, None),
            q_norm: ones.clone(),
            k_norm: ones,
        },
    )
    .unwrap();
    let hidden = wave(&[1, ROWS, INDEX_HIDDEN], 0.2, MlxDtype::Float32);
    let select = |device: bool| {
        with_prefill_device_select(device, || {
            indexer
                .select(&hidden, &QsaIndexKeyCache::empty(), 0)
                .unwrap()
        })
    };
    let host = select(false);
    let device = select(true);
    assert!(
        host.device_tokens().is_none(),
        "host arm selected on device"
    );
    assert!(
        device.device_multi_query(),
        "device arm selected on the host"
    );

    for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16] {
        let q = queries(ROWS, dtype);
        let k = history(ROWS, 0.75, dtype);
        let v = history(ROWS, 1.25, dtype);
        assert_eq!(
            bits(&masked(&q, &k, &v, &host)),
            bits(&masked(&q, &k, &v, &device)),
            "{dtype:?}"
        );
    }
}

#[test]
fn chunks_below_the_row_floor_keep_the_padded_sub_batches() {
    let rows = (MASKED_DENSE_MIN_QUERIES - 1) as i32;
    let tokens = PREFIX + rows;
    let q = queries(rows, MlxDtype::Bfloat16);
    let k = history(tokens, 0.75, MlxDtype::Bfloat16);
    let v = history(tokens, 1.25, MlxDtype::Bfloat16);
    let selection = selection(0..rows);
    let before = masked_dense_calls_for_test();
    let on = with_masked_dense(true, || {
        attend_selected(&q, &k, &v, &selection, cfg()).unwrap()
    });
    assert_eq!(masked_dense_calls_for_test(), before);
    let off = with_masked_dense(false, || {
        attend_selected(&q, &k, &v, &selection, cfg()).unwrap()
    });
    assert_eq!(bits(&on), bits(&off));
}

#[test]
fn the_per_query_path_wins_over_the_switch_when_batching_is_off() {
    let tokens = PREFIX + ROWS;
    let q = queries(ROWS, MlxDtype::Bfloat16);
    let k = history(tokens, 0.75, MlxDtype::Bfloat16);
    let v = history(tokens, 1.25, MlxDtype::Bfloat16);
    let selection = selection(0..ROWS);
    let before = masked_dense_calls_for_test();
    let _ = with_masked_dense(true, || {
        with_batched_attention(false, || {
            attend_selected(&q, &k, &v, &selection, cfg()).unwrap()
        })
    });
    assert_eq!(masked_dense_calls_for_test(), before);
}

#[test]
fn the_masked_call_takes_a_chunk_only_when_batched_enabled_and_covered() {
    let tokens = i64::from(PREFIX + ROWS);
    let chunk = selection(0..ROWS);
    let takes = |masked: bool, batched: bool, rows: i32, tokens: i64| {
        with_masked_dense(masked, || {
            with_batched_attention(batched, || masked_dense_takes(&chunk, rows, tokens))
        })
    };
    assert!(takes(true, true, ROWS, tokens));
    assert!(!takes(false, true, ROWS, tokens), "switch off");
    assert!(!takes(true, false, ROWS, tokens), "per-query attention");
    assert!(!takes(true, true, 63, tokens), "below the row floor");
    assert!(
        !takes(true, true, ROWS, MASKED_DENSE_MAX_TOKENS + 1),
        "past the longest history"
    );
}

#[test]
fn masked_dense_covers_the_chunks_it_was_measured_faster_on() {
    // (queries, history tokens, expected)
    let cases: [(i64, i64, bool); 12] = [
        (2_048, 4_096, true),
        (913, 2_961, true),
        (64, 2_052, true),
        (63, 2_052, false),
        (16, 4_096, false),
        (64, 8_192, true),
        (64, 8_193, false),
        (511, 16_384, false),
        (512, 16_384, true),
        (2_048, 32_768, true),
        (2_048, 32_769, false),
        (2_048, 65_536, false),
    ];
    for (queries, tokens, expected) in cases {
        assert_eq!(
            masked_dense_covers(queries, tokens),
            expected,
            "{queries} queries over {tokens} tokens"
        );
    }
}

#[test]
fn a_long_history_keeps_small_chunks_on_the_padded_sub_batches() {
    // 64 queries at the end of a history one token past the short bound.
    let rows = MASKED_DENSE_MIN_QUERIES as i32;
    let tokens = MASKED_DENSE_SHORT_HISTORY_TOKENS as i32 + 1;
    let first = tokens - rows;
    let q = queries(rows, MlxDtype::Bfloat16);
    let k = history(tokens, 0.75, MlxDtype::Bfloat16);
    let v = history(tokens, 1.25, MlxDtype::Bfloat16);
    // Each query keeps its own position and 40 tokens before it.
    let selection = QsaSelection::from_host_tokens(vec![
        (0..rows)
            .map(|row| ((first + row - 40)..=(first + row)).rev().collect())
            .collect(),
    ]);
    let before = masked_dense_calls_for_test();
    let on = with_masked_dense(true, || {
        attend_selected(&q, &k, &v, &selection, cfg()).unwrap()
    });
    assert_eq!(masked_dense_calls_for_test(), before);
    let off = with_masked_dense(false, || {
        attend_selected(&q, &k, &v, &selection, cfg()).unwrap()
    });
    assert_eq!(bits(&on), bits(&off));
}

#[test]
fn every_batch_row_attends_its_own_selection_and_history() {
    let dtype = MlxDtype::Bfloat16;
    let tokens = PREFIX + ROWS;
    // Two batch rows that differ in queries, history and selection.
    let rows_of = |batch: i32| -> Vec<Vec<i32>> {
        (0..ROWS)
            .map(|row| {
                let mut chosen = selected_tokens(row);
                if batch == 1 {
                    chosen.retain(|token| token % 3 != 1 || *token == PREFIX + row);
                }
                chosen
            })
            .collect()
    };
    let parts: Vec<(MlxArray, MlxArray, MlxArray)> = (0..2)
        .map(|batch| {
            let seed = batch as f32 * 2.0;
            (
                wave(&[1, Q_HEADS, ROWS, HEAD_DIM], 0.5 + seed, dtype),
                history(tokens, 0.75 + seed, dtype),
                history(tokens, 1.25 + seed, dtype),
            )
        })
        .collect();
    let stack = |pick: fn(&(MlxArray, MlxArray, MlxArray)) -> &MlxArray| {
        concatenate(&[pick(&parts[0]), pick(&parts[1])], 0, None)
    };
    let both = QsaSelection::from_host_tokens(vec![rows_of(0), rows_of(1)]);
    let before = masked_dense_calls_for_test();
    let stacked = with_masked_dense(true, || {
        attend_selected(
            &stack(|part| &part.0),
            &stack(|part| &part.1),
            &stack(|part| &part.2),
            &both,
            cfg(),
        )
        .unwrap()
    });
    assert_eq!(masked_dense_calls_for_test(), before + 2);
    assert_eq!(stacked.shape(), vec![2, Q_HEADS, ROWS, HEAD_DIM]);
    for batch in 0..2 {
        let (q, k, v) = &parts[batch as usize];
        let alone = masked(
            q,
            k,
            v,
            &QsaSelection::from_host_tokens(vec![rows_of(batch)]),
        );
        let row = slice(
            &stacked,
            &[batch, 0, 0, 0],
            &[batch + 1, Q_HEADS, ROWS, HEAD_DIM],
            &[1, 1, 1, 1],
            None,
        );
        assert_eq!(bits(&row), bits(&alone), "batch row {batch}");
    }
}

#[test]
fn a_segmented_history_gives_the_bits_of_a_flat_one() {
    use crate::model::shared::qwen4_exp_kv_history::with_kv_segments;

    let dtype = MlxDtype::Bfloat16;
    let tokens = PREFIX + ROWS;
    let q = queries(ROWS, dtype);
    let k = history(tokens, 0.75, dtype);
    let v = history(tokens, 1.25, dtype);
    // A sealed prefix and a live tail of the chunk's last rows. The first
    // append reaches the seal length and seals; the second stays a tail.
    let sealed = tokens - 8;
    let split = |full: &MlxArray| -> KvHistory {
        let part = |from: i32, to: i32| {
            slice(
                full,
                &[0, from, 0, 0],
                &[1, to, KV_HEADS, HEAD_DIM],
                &[1, 1, 1, 1],
                None,
            )
        };
        with_kv_segments(Some((2, sealed as usize)), || {
            KvHistory::empty()
                .append(&part(0, sealed))
                .append(&part(sealed, tokens))
        })
    };
    let (split_k, split_v) = (split(&k), split(&v));
    assert!(
        split_k.segments().1.is_some() && split_v.segments().1.is_some(),
        "expected a live tail"
    );
    let selection = selection(0..ROWS);
    let flat = masked(&q, &k, &v, &selection);
    let segmented = masked(&q, &split_k, &split_v, &selection);
    assert_eq!(bits(&flat), bits(&segmented));
}

#[test]
fn a_host_token_outside_the_history_is_rejected() {
    let tokens = PREFIX + ROWS;
    let q = queries(ROWS, MlxDtype::Float32);
    let k = history(tokens, 0.75, MlxDtype::Float32);
    let v = history(tokens, 1.25, MlxDtype::Float32);
    let mut rows: Vec<Vec<i32>> = (0..ROWS).map(selected_tokens).collect();
    rows[5].push(tokens);
    let selection = QsaSelection::from_host_tokens(vec![rows]);
    let result = with_masked_dense(true, || attend_selected(&q, &k, &v, &selection, cfg()));
    assert!(result.is_err(), "a token past the history must be an error");
}
