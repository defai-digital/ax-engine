//! Compare the existing greedy prefill geometry with independently evaluated chunks.

use super::{
    ExecutionMode, MlxKVCache, MlxRunner, MlxSamplingParams, MlxSamplingRequest,
    NativeModelArtifacts, PathBuf, RequestId, Xorshift64, context, execute_with_route_decisions,
    flash_runner_state_bytes, snapshot_trunk,
};
use crate::generate::{
    CacheOnlyBarrier, CacheOnlyPrefillLayout, chunked_prefill, chunked_prefill_cache_only,
    decode_step,
};
use crate::model::ModelConfig;
use crate::weights::ModelWeights;
use sha2::{Digest, Sha256};

fn synthetic_prefill_artifacts() -> NativeModelArtifacts {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").unwrap());
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
    // The synthetic expert fixture does not carry production pack metadata.
    manifest.runtime_status = ax_engine_core::NativeRuntimeStatus::default();
    NativeModelArtifacts::from_manifest_and_root(root, manifest).unwrap()
}

#[test]
#[ignore = "requires generated or pinned Flash Next weights and MLX"]
fn flash_next_fixed_prefill_chunks_preserve_state_with_sibling_decode() {
    let real_pack = std::env::var_os("AX_FLASH_NEXT_REAL_PACK");
    let artifacts = if let Some(root) = &real_pack {
        NativeModelArtifacts::from_dir(PathBuf::from(root)).unwrap()
    } else {
        synthetic_prefill_artifacts()
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

struct WidthObservation {
    boundary_hashes: Vec<String>,
    tokens: Vec<u32>,
    same_input_logits: Vec<Vec<f32>>,
}

#[test]
#[ignore = "requires generated or pinned resident Flash Next weights and MLX"]
fn flash_next_compare_prefill_widths_at_equal_logical_cuts() {
    let (artifacts, prompt, logical, baseline, candidate, output_tokens) = if let Some(root) =
        std::env::var_os("AX_FLASH_NEXT_REAL_PACK")
    {
        (
            NativeModelArtifacts::from_dir(PathBuf::from(root)).unwrap(),
            serde_json::from_str::<Vec<u32>>(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap())
                .unwrap(),
            2048,
            512,
            256,
            96,
        )
    } else {
        (
            synthetic_prefill_artifacts(),
            (1..=22).collect(),
            11,
            4,
            2,
            24,
        )
    };
    assert!(
        (logical..=4 * logical).contains(&prompt.len()) && prompt.len().is_multiple_of(logical)
    );
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    assert_eq!(cfg.model_family, "qwen4_exp");
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    assert!(weights.expert_stream.is_none());
    let trunk = weights.qwen4_exp.as_ref().unwrap();
    let sampling = MlxSamplingRequest::new(MlxSamplingParams::greedy(), &[]);

    let observe = |width: usize, forcing: Option<&[u32]>| {
        let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
        let mut rng = Xorshift64::new(42);
        let mut boundary_hashes = Vec::new();
        let mut tokens = Vec::with_capacity(output_tokens);
        for (index, item) in prompt.chunks_exact(logical).enumerate() {
            let (&last, prefix) = item.split_last().unwrap();
            for chunk in prefix.chunks(width) {
                chunked_prefill_cache_only(
                    &cfg,
                    &weights,
                    chunk,
                    &mut cache,
                    width,
                    CacheOnlyPrefillLayout::Batched,
                    CacheOnlyBarrier::Blocking,
                );
            }
            let boundary = (index + 1) * logical;
            if boundary == prompt.len() {
                tokens.push(decode_step(
                    &cfg, &weights, last, &mut cache, sampling, &mut rng,
                ));
            } else {
                chunked_prefill_cache_only(
                    &cfg,
                    &weights,
                    std::slice::from_ref(&last),
                    &mut cache,
                    1,
                    CacheOnlyPrefillLayout::Batched,
                    CacheOnlyBarrier::Blocking,
                );
            }
            assert_eq!(cache.seq_len(), boundary);
            assert_eq!(cache.qwen4_exp.as_ref().unwrap().position(), boundary);
            boundary_hashes.push(format!("{:x}", Sha256::digest(cache.serialize_to_bytes())));
        }
        let mut state = cache.qwen4_exp.clone().unwrap();
        for index in 1..output_tokens {
            tokens.push(decode_step(
                &cfg,
                &weights,
                tokens[index - 1],
                &mut cache,
                sampling,
                &mut rng,
            ));
        }
        let same_input_logits = forcing.unwrap_or(&tokens)[..2]
            .iter()
            .map(|&token| {
                let forward = crate::model::qwen4_exp::forward(
                    trunk,
                    &[token],
                    &state,
                    cfg.compile_cache_identity,
                    super::ProjectionBatchPolicy::Shared,
                )
                .unwrap();
                state = forward.state;
                let logits = mlx_sys::astype(&forward.logits, mlx_sys::MlxDtype::Float32, None);
                mlx_sys::eval(&[&logits]);
                let values = logits.data_f32();
                assert!(values.iter().all(|value| value.is_finite()));
                values.to_vec()
            })
            .collect();
        WidthObservation {
            boundary_hashes,
            tokens,
            same_input_logits,
        }
    };

    let reference = observe(baseline, None);
    let repeated = observe(baseline, Some(&reference.tokens));
    assert_eq!(reference.boundary_hashes, repeated.boundary_hashes);
    assert_eq!(reference.tokens, repeated.tokens);
    let compare_logits = |other: &WidthObservation| {
        reference
            .same_input_logits
            .iter()
            .zip(&other.same_input_logits)
            .map(|(a, b)| {
                assert_eq!(a.len(), b.len());
                let exact = a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
                let max_abs_diff = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                (exact, max_abs_diff)
            })
            .collect::<Vec<_>>()
    };
    assert!(compare_logits(&repeated).iter().all(|(exact, _)| *exact));
    let smaller = observe(candidate, Some(&reference.tokens));
    eprintln!(
        "{}",
        serde_json::json!({
            "prompt_tokens": prompt.len(),
            "logical_cut": logical,
            "baseline_width": baseline,
            "candidate_width": candidate,
            "baseline_reproduced": true,
            "baseline_state_sha256": reference.boundary_hashes,
            "candidate_state_sha256": smaller.boundary_hashes,
            "candidate_same_input_logits_exact_and_max_abs_diff": compare_logits(&smaller),
            "baseline_output_ids": reference.tokens,
            "candidate_output_ids": smaller.tokens,
        })
    );
}

#[test]
#[ignore = "requires a generated Flash Next fixture and MLX"]
fn flash_next_runner_routes_physical_prefill_without_early_sampling() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").unwrap());
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
    manifest.runtime_status = ax_engine_core::NativeRuntimeStatus::default();
    let artifacts = NativeModelArtifacts::from_manifest_and_root(root, manifest).unwrap();
    let mut runner = MlxRunner::from_artifacts(&artifacts, 4, true).unwrap();
    runner.set_mtp_requested(false);
    assert!(!runner.mtp_requested);

    let tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    let logical_end = 11;
    for tail in 0..=2 {
        let prompt = &tokens[..logical_end + tail];
        let run = |request, start, end, split_end| {
            let mut ctx = context(request, prompt.len(), 8);
            ctx.processed_prompt_tokens = start as u32;
            let route = if split_end {
                vec![(
                    format!("ax_scheduler_prefill_split_end.{request}"),
                    logical_end as u32,
                )]
            } else {
                Vec::new()
            };
            let output = execute_with_route_decisions(
                &runner,
                ctx,
                &prompt[start..end],
                ExecutionMode::Prefill,
                16,
                route,
            );
            let update = &output.request_updates[0];
            assert!(update.error.is_none(), "{:?}", update.error);
            assert_eq!(update.tokens_executed, (end - start) as u32);
            assert_eq!(update.output_token.is_some(), end == prompt.len());
            assert!(update.output_tokens.is_empty());
            update.output_token
        };

        let baseline_id = 2 * tail as u64 + 1;
        let split_id = baseline_id + 1;
        let mut baseline_token = run(baseline_id, 0, logical_end, false);
        let mut split_token = None;
        for (start, end) in [(0, 4), (4, 8), (8, 10), (10, 11)] {
            split_token = run(split_id, start, end, true);
        }
        if tail > 0 {
            baseline_token = run(baseline_id, logical_end, prompt.len(), false);
            split_token = run(split_id, logical_end, prompt.len(), false);
        }
        assert_eq!(split_token, baseline_token);
        let state = |id| {
            flash_runner_state_bytes(
                &snapshot_trunk(&runner, RequestId(id)).unwrap(),
                runner.cfg.layer_count,
            )
        };
        assert_eq!(state(split_id), state(baseline_id));
    }
}
