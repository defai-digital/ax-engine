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

fn cache_arrays(cache: &Qwen4ExpAttentionCache) -> [&MlxArray; 3] {
    [
        cache.keys().unwrap(),
        cache.values().unwrap(),
        cache.index().keys().unwrap(),
    ]
}

fn exact_cache(actual: &Qwen4ExpAttentionCache, expected: &Qwen4ExpAttentionCache) {
    assert_eq!(
        actual.token_count().unwrap(),
        expected.token_count().unwrap()
    );
    for (actual, expected) in cache_arrays(actual).into_iter().zip(cache_arrays(expected)) {
        exact(actual, expected);
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
    let mut arrays = vec![output.delta()];
    arrays.extend(cache_arrays(output.next_state()));
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
    let snapshot = cache_arrays(&cache).map(values);
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
        verifier.next_state().keys().unwrap(),
        direct.keys().unwrap(),
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
        exact(&span(updated, 0, PREFIX), original);
        assert_eq!(values(original), saved, "caller cache changed");
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
            direct.keys().unwrap()
        } else {
            ordinary.next_state().keys().unwrap()
        };
        let expected_index = if index_mode == MlxQuantizationMode::Mxfp4 {
            direct.index().keys().unwrap()
        } else {
            ordinary.next_state().index().keys().unwrap()
        };
        exact(verifier.next_state().keys().unwrap(), expected_keys);
        exact(
            verifier.next_state().index().keys().unwrap(),
            expected_index,
        );
        exact(
            verifier.next_state().values().unwrap(),
            ordinary.next_state().values().unwrap(),
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
        let snapshot = cache_arrays(&cache).map(values);
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
            exact(&span(updated, 0, PREFIX), original);
            assert_eq!(values(original), saved, "caller cache changed");
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
