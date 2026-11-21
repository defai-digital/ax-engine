//! Compare the existing greedy prefill geometry with independently evaluated chunks.

use super::{
    MlxKVCache, MlxSamplingParams, MlxSamplingRequest, NativeModelArtifacts, PathBuf, Xorshift64,
};
use crate::generate::{
    CacheOnlyBarrier, CacheOnlyPrefillLayout, chunked_prefill, chunked_prefill_cache_only,
    decode_step,
};
use crate::model::ModelConfig;
use crate::weights::ModelWeights;

#[test]
#[ignore = "requires generated or pinned Flash Next weights and MLX"]
fn flash_next_fixed_prefill_chunks_preserve_state_with_sibling_decode() {
    let real_pack = std::env::var_os("AX_FLASH_NEXT_REAL_PACK");
    let artifacts = if let Some(root) = &real_pack {
        NativeModelArtifacts::from_dir(PathBuf::from(root)).unwrap()
    } else {
        let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").unwrap());
        let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
        manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
        // The synthetic expert fixture does not carry production pack metadata.
        manifest.runtime_status = ax_engine_core::NativeRuntimeStatus::default();
        NativeModelArtifacts::from_manifest_and_root(root, manifest).unwrap()
    };
    let (prompt, sibling_prompt, chunk_size) = if real_pack.is_some() {
        (
            serde_json::from_str::<Vec<u32>>(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap())
                .unwrap(),
            serde_json::from_str::<Vec<u32>>(
                &std::env::var("AX_FLASH_NEXT_SIBLING_PROMPT_IDS").unwrap(),
            )
            .unwrap(),
            std::env::var("AX_FLASH_NEXT_PREFILL_CHUNK")
                .unwrap()
                .parse::<usize>()
                .unwrap(),
        )
    } else {
        (
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            vec![9, 3, 7, 5, 2],
            4,
        )
    };
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    assert_eq!(cfg.model_family, "qwen4_exp");
    assert!(!crate::fastpath::skip_cache_only_split_for_family(
        &cfg.model_family,
        prompt.len()
    ));
    assert!(!crate::fastpath::cache_only_chunk_eval_enabled());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    assert!(weights.expert_stream.is_none());
    if real_pack.is_some() {
        let logical_len = match prompt.len() {
            2049 | 2050 => 2048,
            len => len,
        };
        let (logical_prompt, tail) = prompt.split_at(logical_len);
        assert_prefill_interleave(
            &cfg,
            &weights,
            logical_prompt,
            tail,
            &sibling_prompt,
            chunk_size,
        );
    } else {
        for tail in [&[][..], &[12][..], &[12, 13][..]] {
            assert_prefill_interleave(&cfg, &weights, &prompt, tail, &sibling_prompt, chunk_size);
        }
    }
}

fn assert_prefill_interleave(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    logical_prompt: &[u32],
    tail: &[u32],
    sibling_prompt: &[u32],
    chunk_size: usize,
) {
    let (&last, prefix) = logical_prompt.split_last().unwrap();
    assert!(chunk_size > 1 && prefix.len() > chunk_size);
    assert!(!sibling_prompt.is_empty());
    let prompt_len = logical_prompt.len() + tail.len();
    let sampling = MlxSamplingRequest::new(MlxSamplingParams::greedy(), &[]);
    let sibling_steps = prefix.chunks(chunk_size).len() + usize::from(!tail.is_empty());

    let decode = |cache: &mut MlxKVCache, rng: &mut Xorshift64, token| {
        decode_step(cfg, weights, token, cache, sampling, rng)
    };
    let prefill_sibling = |cache: &mut MlxKVCache, rng: &mut Xorshift64| {
        chunked_prefill(
            cfg,
            weights,
            sibling_prompt,
            cache,
            chunk_size,
            sampling,
            rng,
        )
    };

    let mut sibling_cache = MlxKVCache::new_contiguous(cfg.layer_count);
    let mut sibling_rng = Xorshift64::new(42);
    let mut expected_sibling = vec![prefill_sibling(&mut sibling_cache, &mut sibling_rng)];
    for _ in 0..sibling_steps {
        expected_sibling.push(decode(
            &mut sibling_cache,
            &mut sibling_rng,
            *expected_sibling.last().unwrap(),
        ));
    }
    let expected_sibling_state = sibling_cache.serialize_to_bytes();
    drop(sibling_cache);

    let mut baseline = MlxKVCache::new_contiguous(cfg.layer_count);
    let mut rng = Xorshift64::new(42);
    let first = if tail.is_empty() {
        chunked_prefill(
            cfg,
            weights,
            logical_prompt,
            &mut baseline,
            chunk_size,
            sampling,
            &mut rng,
        )
    } else {
        chunked_prefill_cache_only(
            cfg,
            weights,
            logical_prompt,
            &mut baseline,
            chunk_size,
            CacheOnlyPrefillLayout::PreserveFinalTokenStep,
            CacheOnlyBarrier::AsyncSubmit,
        );
        chunked_prefill(
            cfg,
            weights,
            tail,
            &mut baseline,
            chunk_size,
            sampling,
            &mut rng,
        )
    };
    assert_eq!(baseline.seq_len(), prompt_len);
    let expected_prefill_state = baseline.serialize_to_bytes();
    let mut expected = [first; 8];
    for index in 1..expected.len() {
        expected[index] = decode(&mut baseline, &mut rng, expected[index - 1]);
    }
    let expected_decode_state = baseline.serialize_to_bytes();
    drop(baseline);

    let mut arm_results = Vec::new();
    for with_sibling in [false, true] {
        let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
        let mut rng = Xorshift64::new(42);
        let mut sibling = MlxKVCache::new_contiguous(cfg.layer_count);
        let mut sibling_rng = Xorshift64::new(42);
        let mut sibling_token =
            with_sibling.then(|| prefill_sibling(&mut sibling, &mut sibling_rng));
        let mut sibling_matched = sibling_token.is_none_or(|token| token == expected_sibling[0]);
        let cache_only_singleton = (!tail.is_empty()).then(|| std::slice::from_ref(&last));
        for (index, chunk) in prefix
            .chunks(chunk_size)
            .chain(cache_only_singleton)
            .enumerate()
        {
            chunked_prefill_cache_only(
                cfg,
                weights,
                chunk,
                &mut cache,
                chunk_size,
                CacheOnlyPrefillLayout::Batched,
                CacheOnlyBarrier::Blocking,
            );
            if let Some(token) = sibling_token {
                let next = decode(&mut sibling, &mut sibling_rng, token);
                sibling_matched &= next == expected_sibling[index + 1];
                sibling_token = Some(next);
            }
        }
        let mut token = if tail.is_empty() {
            decode(&mut cache, &mut rng, last)
        } else {
            chunked_prefill(
                cfg, weights, tail, &mut cache, chunk_size, sampling, &mut rng,
            )
        };
        assert_eq!(cache.seq_len(), prompt_len);
        let prefill_state_matched = cache.serialize_to_bytes() == expected_prefill_state;
        let mut output_matched = token == first;
        for &next in &expected[1..] {
            token = decode(&mut cache, &mut rng, token);
            output_matched &= token == next;
        }
        let decode_state_matched = cache.serialize_to_bytes() == expected_decode_state;
        sibling_matched &= !with_sibling || sibling.serialize_to_bytes() == expected_sibling_state;
        eprintln!(
            "Flash Next fixed chunks {sibling_steps} tail={} with_sibling={with_sibling}: output={output_matched} prefill_state={prefill_state_matched} decode_state={decode_state_matched} sibling={sibling_matched}",
            tail.len(),
        );
        arm_results.push(
            output_matched && prefill_state_matched && decode_state_matched && sibling_matched,
        );
    }
    assert_eq!(arm_results, [true, true]);
}
