#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::qwen4_exp_qsa::{QsaConfig, QsaIndexerWeights};
use mlx_sys::{MlxQuantizationMode, eval, quantize};

const HIDDEN: i32 = 256;
const PREFIX: i32 = 7;
const SHARED: ProjectionBatchPolicy = ProjectionBatchPolicy::Shared;
const ROW_EXACT: ProjectionBatchPolicy = ProjectionBatchPolicy::RowExact;

fn array(data: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Float32,
    )
}

fn wave(rows: i32, width: i32, seed: i32) -> MlxArray {
    let data: Vec<f32> = (0..rows * width)
        .map(|i| ((i * (seed * 2 + 3) + seed * 17) % 1009 - 504) as f32 / 1024.0)
        .collect();
    array(&data, &[rows, width])
}

fn projection(
    output: i32,
    input: i32,
    seed: i32,
    mode: Option<MlxQuantizationMode>,
) -> QuantizedWeight {
    let dense = wave(output, input, seed);
    let Some(mode) = mode else {
        return QuantizedWeight::new(astype(&dense, MlxDtype::Bfloat16, None), None, None);
    };
    let packed = quantize(
        &astype(&dense, MlxDtype::Bfloat16, None),
        Some(32),
        Some(4),
        mode,
        None,
        None,
    );
    let biases = if mode == MlxQuantizationMode::Affine {
        Some(packed[2].clone())
    } else {
        None
    };
    let mut weight = QuantizedWeight::new(packed[0].clone(), Some(packed[1].clone()), biases);
    weight.mode = if mode == MlxQuantizationMode::Mxfp4 {
        "mxfp4".into()
    } else {
        "affine".into()
    };
    weight.bits = 4;
    weight.group_size = 32;
    weight
}

fn gain(width: i32) -> MlxArray {
    astype(
        &array(&vec![1.0; width as usize], &[width]),
        MlxDtype::Bfloat16,
        None,
    )
}

fn module(
    key_mode: Option<MlxQuantizationMode>,
    index_mode: Option<MlxQuantizationMode>,
) -> Qwen4ExpAttention {
    module_with_modes(None, key_mode, None, None, index_mode)
}

fn module_with_modes(
    query_mode: Option<MlxQuantizationMode>,
    key_mode: Option<MlxQuantizationMode>,
    value_mode: Option<MlxQuantizationMode>,
    output_mode: Option<MlxQuantizationMode>,
    index_mode: Option<MlxQuantizationMode>,
) -> Qwen4ExpAttention {
    let indexer = QsaIndexer::new(
        QsaConfig::new(2, 1, 32, 32, 4, 8, HIDDEN as usize, 1e-6, 10_000.0).unwrap(),
        QsaIndexerWeights {
            qk_proj: projection(96, HIDDEN, 5, index_mode),
            q_norm: gain(32),
            k_norm: gain(32),
        },
    )
    .unwrap();
    Qwen4ExpAttention::new(
        Qwen4ExpAttentionConfig::new(HIDDEN as usize, 4, 1, 64, 32, 10_000.0, 1e-6).unwrap(),
        Qwen4ExpAttentionWeights {
            q_proj: projection(512, HIDDEN, 7, query_mode),
            k_proj: projection(64, HIDDEN, 11, key_mode),
            v_proj: projection(64, HIDDEN, 13, value_mode),
            // K=256 avoids MLX's separate K=64/128 quantized projection dispatch.
            o_proj: projection(HIDDEN, 256, 17, output_mode),
            q_norm: gain(64),
            k_norm: gain(64),
        },
        indexer,
    )
    .unwrap()
}

fn input() -> MlxArray {
    astype(
        &reshape(
            &wave(PREFIX + 3, HIDDEN, 19),
            &[1, PREFIX + 3, HIDDEN],
            None,
        ),
        MlxDtype::Bfloat16,
        None,
    )
}

fn span(value: &MlxArray, start: i32, end: i32) -> MlxArray {
    let mut starts = vec![0; value.ndim()];
    starts[1] = start;
    let mut ends = value.shape();
    ends[1] = end;
    contiguous(
        &slice(value, &starts, &ends, &vec![1; value.ndim()], None),
        None,
    )
}

fn values(value: &MlxArray) -> Vec<f32> {
    let value = contiguous(&astype(value, MlxDtype::Float32, None), None);
    eval(&[&value]);
    let values = value.data_f32().to_vec();
    assert!(values.iter().all(|value| value.is_finite()));
    values
}

fn exact(actual: &MlxArray, expected: &MlxArray) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(values(actual), values(expected));
}

fn cache_arrays(cache: &Qwen4ExpAttentionCache) -> [MlxArray; 3] {
    [
        cache.flat_keys().unwrap(),
        cache.flat_values().unwrap(),
        cache.index().keys().unwrap().clone(),
    ]
}

fn exact_cache(actual: &Qwen4ExpAttentionCache, expected: &Qwen4ExpAttentionCache) {
    assert_eq!(
        actual.token_count().unwrap(),
        expected.token_count().unwrap()
    );
    for (actual, expected) in cache_arrays(actual).into_iter().zip(cache_arrays(expected)) {
        exact(&actual, &expected);
    }
}

fn prefill(module: &Qwen4ExpAttention, input: &MlxArray) -> Qwen4ExpAttentionCache {
    module
        .forward(
            &span(input, 0, PREFIX),
            &Qwen4ExpAttentionCache::empty(),
            0,
            SHARED,
        )
        .unwrap()
        .into_next_state()
}

fn singletons(
    module: &Qwen4ExpAttention,
    input: &MlxArray,
    cache: &Qwen4ExpAttentionCache,
    policy: ProjectionBatchPolicy,
) -> Qwen4ExpAttentionCache {
    singleton_output(module, input, cache, policy).into_next_state()
}

fn evaluate(output: &Qwen4ExpAttentionOutput) {
    let owned = cache_arrays(output.next_state());
    let mut arrays = vec![output.delta()];
    arrays.extend(owned.iter());
    eval(&arrays);
}

fn singleton_output(
    module: &Qwen4ExpAttention,
    input: &MlxArray,
    cache: &Qwen4ExpAttentionCache,
    policy: ProjectionBatchPolicy,
) -> Qwen4ExpAttentionOutput {
    let first = module
        .forward(
            &span(input, PREFIX, PREFIX + 1),
            cache,
            PREFIX as usize,
            policy,
        )
        .unwrap();
    evaluate(&first);
    let second = module
        .forward(
            &span(input, PREFIX + 1, PREFIX + 2),
            first.next_state(),
            (PREFIX + 1) as usize,
            policy,
        )
        .unwrap();
    evaluate(&second);
    Qwen4ExpAttentionOutput {
        delta: concatenate(&[first.delta(), second.delta()], 1, None),
        next_state: second.into_next_state(),
    }
}

#[test]
fn mxfp4_verifier_key_and_index_histories_match_singletons_across_block_boundary() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let module = module(
        Some(MlxQuantizationMode::Mxfp4),
        Some(MlxQuantizationMode::Mxfp4),
    );
    let input = input();
    let cache = prefill(&module, &input);
    let snapshot = cache_arrays(&cache).map(|array| values(&array));
    let suffix = span(&input, PREFIX, PREFIX + 2);
    let verifier = module
        .forward_with_verifier_policy(&suffix, &cache, PREFIX as usize, SHARED, ROW_EXACT)
        .unwrap();
    let direct = singletons(&module, &input, &cache, SHARED);

    // Synthetic geometry checks the policy's state contract, not device-specific drift.
    assert_eq!(
        verifier.next_state().token_count().unwrap(),
        (PREFIX + 2) as usize
    );
    exact(
        &verifier.next_state().flat_keys().unwrap(),
        &direct.flat_keys().unwrap(),
    );
    exact(
        verifier.next_state().index().keys().unwrap(),
        direct.index().keys().unwrap(),
    );
    for ((updated, original), saved) in cache_arrays(verifier.next_state())
        .into_iter()
        .zip(cache_arrays(&cache))
        .zip(snapshot)
    {
        exact(&span(&updated, 0, PREFIX), &original);
        assert_eq!(values(&original), saved, "caller cache changed");
    }

    let selected = module
        .indexer
        .select_with_projection(&suffix, cache.index(), PREFIX as usize, |input, weight| {
            qw_with_policy(input, weight, ROW_EXACT)
        })
        .unwrap();
    let first = module
        .indexer
        .select(
            &span(&input, PREFIX, PREFIX + 1),
            cache.index(),
            PREFIX as usize,
        )
        .unwrap();
    let second = module
        .indexer
        .select(
            &span(&input, PREFIX + 1, PREFIX + 2),
            first.next_cache(),
            (PREFIX + 1) as usize,
        )
        .unwrap();
    assert_eq!(
        selected.tokens_for_query(0, 0),
        first.tokens_for_query(0, 0)
    );
    assert_eq!(
        selected.tokens_for_query(0, 1),
        second.tokens_for_query(0, 0)
    );
    assert_eq!(first.tokens_for_query(0, 0).len(), 8);
    assert_eq!(second.tokens_for_query(0, 0).len(), 9);
}

#[test]
fn verifier_policy_preserves_ordinary_dense_and_affine_qsa_paths() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    for mode in [
        None,
        Some(MlxQuantizationMode::Affine),
        Some(MlxQuantizationMode::Mxfp4),
    ] {
        let module = module(mode, mode);
        let input = input();
        let cache = prefill(&module, &input);
        let suffix = span(&input, PREFIX, PREFIX + 2);
        let ordinary = module
            .forward(&suffix, &cache, PREFIX as usize, SHARED)
            .unwrap();
        let shared = module
            .forward_with_verifier_policy(&suffix, &cache, PREFIX as usize, SHARED, SHARED)
            .unwrap();
        exact(ordinary.delta(), shared.delta());
        exact_cache(ordinary.next_state(), shared.next_state());
        let legacy_index = module
            .indexer
            .select(&suffix, cache.index(), PREFIX as usize)
            .unwrap();
        exact(
            ordinary.next_state().index().keys().unwrap(),
            legacy_index.next_cache().keys().unwrap(),
        );

        if mode != Some(MlxQuantizationMode::Mxfp4) {
            let verifier = module
                .forward_with_verifier_policy(&suffix, &cache, PREFIX as usize, SHARED, ROW_EXACT)
                .unwrap();
            exact(verifier.delta(), ordinary.delta());
            exact_cache(verifier.next_state(), ordinary.next_state());
        }

        // The MTP head's existing RowExact request applies across projection modes.
        let head = module
            .forward(&suffix, &cache, PREFIX as usize, ROW_EXACT)
            .unwrap();
        let direct = singletons(&module, &input, &cache, ROW_EXACT);
        exact_cache(head.next_state(), &direct);
    }
}

#[test]
fn verifier_policy_checks_key_and_index_quantization_independently() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    for (key_mode, index_mode) in [
        (MlxQuantizationMode::Mxfp4, MlxQuantizationMode::Affine),
        (MlxQuantizationMode::Affine, MlxQuantizationMode::Mxfp4),
    ] {
        let module = module(Some(key_mode), Some(index_mode));
        let input = input();
        let cache = prefill(&module, &input);
        let suffix = span(&input, PREFIX, PREFIX + 2);
        let ordinary = module
            .forward(&suffix, &cache, PREFIX as usize, SHARED)
            .unwrap();
        let verifier = module
            .forward_with_verifier_policy(&suffix, &cache, PREFIX as usize, SHARED, ROW_EXACT)
            .unwrap();
        let direct = singletons(&module, &input, &cache, SHARED);
        let expected_keys = if key_mode == MlxQuantizationMode::Mxfp4 {
            direct.flat_keys().unwrap()
        } else {
            ordinary.next_state().flat_keys().unwrap()
        };
        let expected_index = if index_mode == MlxQuantizationMode::Mxfp4 {
            direct.index().keys().unwrap()
        } else {
            ordinary.next_state().index().keys().unwrap()
        };
        exact(&verifier.next_state().flat_keys().unwrap(), &expected_keys);
        exact(
            verifier.next_state().index().keys().unwrap(),
            expected_index,
        );
        exact(
            &verifier.next_state().flat_values().unwrap(),
            &ordinary.next_state().flat_values().unwrap(),
        );
    }
}

#[test]
fn mxfp4_verifier_query_output_delta_and_continuation_match_ordinary_singletons() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let mxfp4 = Some(MlxQuantizationMode::Mxfp4);
    for (query_mode, output_mode) in [(mxfp4, None), (None, mxfp4), (mxfp4, mxfp4)] {
        let module = module_with_modes(query_mode, mxfp4, None, output_mode, mxfp4);
        let input = input();
        let cache = prefill(&module, &input);
        let original = cache.fork();
        let snapshot = cache_arrays(&cache).map(|array| values(&array));
        let verifier = module
            .forward_with_verifier_policy(
                &span(&input, PREFIX, PREFIX + 2),
                &cache,
                PREFIX as usize,
                SHARED,
                ROW_EXACT,
            )
            .unwrap();
        let direct = singleton_output(&module, &input, &original, SHARED);
        // Whole attention outputs, with independently evaluated ordinary singleton calls.
        // Generated weights protect the contract; actual-pack replay establishes device drift.
        exact(verifier.delta(), direct.delta());
        assert!(values(direct.delta()).iter().any(|&value| value != 0.0));
        exact_cache(verifier.next_state(), direct.next_state());
        for ((updated, original), saved) in cache_arrays(verifier.next_state())
            .into_iter()
            .zip(cache_arrays(&cache))
            .zip(snapshot)
        {
            exact(&span(&updated, 0, PREFIX), &original);
            assert_eq!(values(&original), saved, "caller cache changed");
        }

        let next = span(&input, PREFIX + 2, PREFIX + 3);
        let committed = module
            .forward(&next, verifier.next_state(), (PREFIX + 2) as usize, SHARED)
            .unwrap();
        let continued = module
            .forward(&next, direct.next_state(), (PREFIX + 2) as usize, SHARED)
            .unwrap();
        exact(committed.delta(), continued.delta());
        exact_cache(committed.next_state(), continued.next_state());
        assert_eq!(
            committed.next_state().token_count().unwrap(),
            (PREFIX + 3) as usize
        );

        // Discarding the staged pair must leave no future rows in the original checkpoint.
        let restarted = singleton_output(&module, &input, &cache, SHARED);
        exact(restarted.delta(), direct.delta());
        exact_cache(restarted.next_state(), direct.next_state());
    }
}

#[test]
fn query_output_override_preserves_shared_affine_dense_value_and_head_paths() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let affine = Some(MlxQuantizationMode::Affine);
    let mxfp4 = Some(MlxQuantizationMode::Mxfp4);
    for (query_mode, output_mode) in [
        (None, None),
        (affine, None),
        (None, affine),
        (affine, affine),
        (mxfp4, affine),
        (affine, mxfp4),
        (mxfp4, mxfp4),
    ] {
        let module = module_with_modes(query_mode, None, mxfp4, output_mode, None);
        let input = input();
        let cache = prefill(&module, &input);
        let suffix = span(&input, PREFIX, PREFIX + 2);
        let ordinary = module
            .forward(&suffix, &cache, PREFIX as usize, SHARED)
            .unwrap();
        let shared = module
            .forward_with_verifier_policy(&suffix, &cache, PREFIX as usize, SHARED, SHARED)
            .unwrap();
        exact(shared.delta(), ordinary.delta());
        exact_cache(shared.next_state(), ordinary.next_state());

        let verifier = module
            .forward_with_verifier_policy(&suffix, &cache, PREFIX as usize, SHARED, ROW_EXACT)
            .unwrap();
        // Q/O cannot change K, V or index caches; MXFP4 V keeps ordinary Shared arithmetic.
        exact_cache(verifier.next_state(), ordinary.next_state());
        if query_mode != mxfp4 && output_mode != mxfp4 {
            exact(verifier.delta(), ordinary.delta());
        }

        // Explicit RowExact for the MTP head still applies to every projection mode.
        let head = module
            .forward(&suffix, &cache, PREFIX as usize, ROW_EXACT)
            .unwrap();
        let direct = singleton_output(&module, &input, &cache, SHARED);
        exact(head.delta(), direct.delta());
        exact_cache(head.next_state(), direct.next_state());
    }
}

#[test]
fn device_selected_single_query_steps_are_bitwise_identical_to_host_selection() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    for mode in [
        None,
        Some(MlxQuantizationMode::Affine),
        Some(MlxQuantizationMode::Mxfp4),
    ] {
        let module = module(mode, mode);
        let input = input();
        let cache = prefill(&module, &input);
        let host = crate::qwen4_exp_qsa::with_device_select(false, || {
            singleton_output(&module, &input, &cache, SHARED)
        });
        let device = crate::qwen4_exp_qsa::with_device_select(true, || {
            singleton_output(&module, &input, &cache, SHARED)
        });
        exact(device.delta(), host.delta());
        exact_cache(device.next_state(), host.next_state());
    }
}

#[test]
fn batched_prefill_attention_matches_the_per_query_path() {
    use crate::qwen4_exp_qsa::{QsaIndexKeyCache, QsaIndexer};
    // 40 queries over a 4-token block / 2-block budget so many queries keep
    // different token counts (growing prefix, then a full budget plus tail).
    let queries_n = 40;
    let indexer: QsaIndexer = module(None, None).indexer;
    let hidden = astype(
        &reshape(&wave(queries_n, HIDDEN, 23), &[1, queries_n, HIDDEN], None),
        MlxDtype::Bfloat16,
        None,
    );
    // This comparison is the host token lists. Prefill device selection is
    // on by default; force it off so the lists stay on the host.
    let selection = crate::qwen4_exp_qsa::with_prefill_device_select(false, || {
        indexer
            .select(&hidden, &QsaIndexKeyCache::empty(), 0)
            .unwrap()
    });
    assert!(selection.device_tokens().is_none());
    let lengths: std::collections::BTreeSet<usize> = (0..queries_n as usize)
        .map(|q| selection.tokens_for_query(0, q).len())
        .collect();
    assert!(lengths.len() > 4, "need ragged selections, got {lengths:?}");
    let cfg = Qwen4ExpAttentionConfig::new(HIDDEN as usize, 24, 2, 64, 32, 10_000.0, 1e-6).unwrap();
    for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16] {
        let tensor = |rows: i32, seed: i32| {
            astype(
                &reshape(&wave(rows, 64, seed), &[1, rows, 64], None),
                dtype,
                None,
            )
        };
        let q = transpose(
            &reshape(
                &astype(&wave(queries_n * 24, 64, 29), dtype, None),
                &[1, queries_n, 24, 64],
                None,
            ),
            &[0, 2, 1, 3],
            None,
        );
        let keys = reshape(
            &concatenate(&[&tensor(queries_n, 31), &tensor(queries_n, 37)], 2, None),
            &[1, queries_n, 2, 64],
            None,
        );
        let values = reshape(
            &concatenate(&[&tensor(queries_n, 41), &tensor(queries_n, 43)], 2, None),
            &[1, queries_n, 2, 64],
            None,
        );
        let per_query = with_batched_attention(false, || {
            attend_selected(&q, &keys, &values, &selection, cfg).unwrap()
        });
        let batched = with_batched_attention(true, || {
            attend_selected(&q, &keys, &values, &selection, cfg).unwrap()
        });
        assert_eq!(batched.shape(), per_query.shape());
        let (a, b) = (values_of(&batched), values_of(&per_query));
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let bound = if dtype == MlxDtype::Float32 {
            2e-5
        } else {
            4e-3
        };
        assert!(
            max_abs <= bound,
            "{dtype:?}: batched attention drifted by {max_abs}"
        );
    }
}

/// Exact bits of an array, compared through f32 so bf16 payloads survive.
fn bit_pattern(array: &MlxArray) -> Vec<u32> {
    let widened = contiguous(&astype(array, MlxDtype::Float32, None), None);
    eval(&[&widened]);
    widened
        .data_f32()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

/// The fused selected-K/V gather must leave every batched prefill attention
/// bit unchanged: ragged host selections, a two-batch call and a segmented
/// (sealed prefix + live tail) history.
#[test]
fn fused_gather_batched_attention_matches_bit_for_bit() {
    use crate::model::shared::qwen4_exp_kv_history::{KvHistory, with_kv_segments};
    use crate::qwen4_exp_qsa::{QsaIndexKeyCache, QsaIndexer};

    let queries_n = 40;
    // A 4-token block / 2-block budget: many queries keep different token
    // counts, so each sub-batch is padded and the padding is masked.
    let indexer: QsaIndexer = module(None, None).indexer;
    let cfg = Qwen4ExpAttentionConfig::new(HIDDEN as usize, 24, 2, 64, 32, 10_000.0, 1e-6).unwrap();
    let hidden = |seed: i32| {
        astype(
            &reshape(
                &wave(queries_n, HIDDEN, seed),
                &[1, queries_n, HIDDEN],
                None,
            ),
            MlxDtype::Bfloat16,
            None,
        )
    };
    // Prefill device selection is on by default; force the host lists so this
    // test covers the host gather buffer as well.
    let select = |hidden: &MlxArray| {
        crate::qwen4_exp_qsa::with_prefill_device_select(false, || {
            indexer
                .select(hidden, &QsaIndexKeyCache::empty(), 0)
                .unwrap()
        })
    };
    let single = select(&hidden(23));
    let doubled = select(&concatenate(&[&hidden(23), &hidden(47)], 0, None));
    assert!(single.device_tokens().is_none(), "host selection expected");
    assert!(doubled.device_tokens().is_none(), "host selection expected");

    for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16] {
        let tensor = |rows: i32, seed: i32| {
            astype(
                &reshape(&wave(rows, 64, seed), &[1, rows, 64], None),
                dtype,
                None,
            )
        };
        let queries = transpose(
            &reshape(
                &astype(&wave(queries_n * 24, 64, 29), dtype, None),
                &[1, queries_n, 24, 64],
                None,
            ),
            &[0, 2, 1, 3],
            None,
        );
        // One batch row of K/V: two 64-wide halves become two KV heads.
        let row = |seed: i32| {
            reshape(
                &concatenate(
                    &[&tensor(queries_n, seed), &tensor(queries_n, seed + 6)],
                    2,
                    None,
                ),
                &[1, queries_n, 2, 64],
                None,
            )
        };
        let flat_keys = row(31);
        let flat_values = row(41);
        let split = |full: &MlxArray| {
            let sealed = slice(full, &[0, 0, 0, 0], &[1, 25, 2, 64], &[1, 1, 1, 1], None);
            let tail = slice(
                full,
                &[0, 25, 0, 0],
                &[1, queries_n, 2, 64],
                &[1, 1, 1, 1],
                None,
            );
            // `seal_tokens == 25` seals the prefix and keeps the rest as a
            // live tail, so the batched path has to flatten the history.
            with_kv_segments(Some((2, 25)), || {
                KvHistory::empty().append(&sealed).append(&tail)
            })
        };
        let split_keys = split(&flat_keys);
        let split_values = split(&flat_values);
        assert!(
            split_keys.segments().1.is_some() && split_values.segments().1.is_some(),
            "{dtype:?}: expected a live tail"
        );
        let two_keys = concatenate(&[&row(31), &row(53)], 0, None);
        let two_values = concatenate(&[&row(41), &row(67)], 0, None);
        let queries_two = concatenate(&[&queries, &queries], 0, None);

        let compare = |label: &str,
                       q: &MlxArray,
                       keys: &KvHistory,
                       values: &KvHistory,
                       selection: &QsaSelection| {
            let run = |fused: bool| {
                with_fused_gather(fused, || {
                    with_batched_attention(true, || {
                        attend_selected(q, keys, values, selection, cfg).unwrap()
                    })
                })
            };
            let baseline = run(false);
            let before = fused_gather_tiles_for_test();
            let fused = run(true);
            assert!(
                fused_gather_tiles_for_test() > before,
                "{dtype:?} {label}: the fused gather never ran"
            );
            assert_eq!(fused.shape(), baseline.shape(), "{dtype:?} {label}: shape");
            assert_eq!(
                bit_pattern(&fused),
                bit_pattern(&baseline),
                "{dtype:?} {label}: attention bits"
            );
        };
        compare(
            "one batch, one segment",
            &queries,
            &KvHistory::from_flat(flat_keys),
            &KvHistory::from_flat(flat_values),
            &single,
        );
        compare(
            "one batch, sealed plus tail",
            &queries,
            &split_keys,
            &split_values,
            &single,
        );
        compare(
            "two batch, one segment",
            &queries_two,
            &KvHistory::from_flat(two_keys),
            &KvHistory::from_flat(two_values),
            &doubled,
        );
    }
}

fn values_of(value: &MlxArray) -> Vec<f32> {
    values(value)
}

fn close(actual: &MlxArray, expected: &MlxArray, bound: f32, label: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{label}: shape");
    let (a, b) = (values(actual), values(expected));
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{label}: max abs difference {max_abs} (bound {bound})");
    assert!(
        max_abs <= bound,
        "{label}: max abs difference {max_abs} > {bound}"
    );
}

#[test]
fn dense_short_context_attention_matches_the_selected_path_and_keeps_caches_exact() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    for mode in [
        None,
        Some(MlxQuantizationMode::Affine),
        Some(MlxQuantizationMode::Mxfp4),
    ] {
        let module = module(mode, mode);
        // QSA geometry: ratio 4, budget 8 => every query keeps every token up to
        // 11 tokens of context.
        assert_eq!(module.indexer.config().dense_context_limit(), 11);
        let input = input();
        let steps = |dense: bool| {
            with_dense_attention(dense, || {
                let prefix = module
                    .forward(
                        &span(&input, 0, PREFIX),
                        &Qwen4ExpAttentionCache::empty(),
                        0,
                        SHARED,
                    )
                    .unwrap();
                evaluate(&prefix);
                // A three-token continuation reaches 10 tokens of context.
                let chunk = module
                    .forward(
                        &span(&input, PREFIX, PREFIX + 3),
                        prefix.next_state(),
                        PREFIX as usize,
                        SHARED,
                    )
                    .unwrap();
                evaluate(&chunk);
                // A singleton step from the same 7-token state: context 8.
                let single = module
                    .forward(
                        &span(&input, PREFIX, PREFIX + 1),
                        prefix.next_state(),
                        PREFIX as usize,
                        SHARED,
                    )
                    .unwrap();
                evaluate(&single);
                (prefix, chunk, single)
            })
        };
        let (dense_prefix, dense_chunk, dense_single) = steps(true);
        let (host_prefix, host_chunk, host_single) = steps(false);
        // The dense path advances the very same caches (keys, values, index keys).
        exact_cache(dense_prefix.next_state(), host_prefix.next_state());
        exact_cache(dense_chunk.next_state(), host_chunk.next_state());
        exact_cache(dense_single.next_state(), host_single.next_state());
        let bound = 5e-2;
        close(
            dense_prefix.delta(),
            host_prefix.delta(),
            bound,
            "prefix chunk",
        );
        close(
            dense_chunk.delta(),
            host_chunk.delta(),
            bound,
            "continued chunk",
        );
        close(
            dense_single.delta(),
            host_single.delta(),
            bound,
            "singleton",
        );
    }
}

#[test]
fn dense_attention_is_not_used_beyond_the_qsa_budget() {
    let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let module = module(None, None);
    let limit = module.indexer.config().dense_context_limit() as i32;
    let total = limit + 4;
    let long = astype(
        &reshape(&wave(total, HIDDEN, 19), &[1, total, HIDDEN], None),
        MlxDtype::Bfloat16,
        None,
    );
    let run = |dense: bool| {
        with_dense_attention(dense, || {
            let out = module
                .forward(&long, &Qwen4ExpAttentionCache::empty(), 0, SHARED)
                .unwrap();
            evaluate(&out);
            out
        })
    };
    // Past the budget selection is real, so the dense switch must not matter.
    exact(run(true).delta(), run(false).delta());
}
