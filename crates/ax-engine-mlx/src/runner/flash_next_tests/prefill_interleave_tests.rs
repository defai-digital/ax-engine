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

#[test]
#[ignore = "requires the pinned resident Flash Next MXFP4 pack and prefix fork fixture"]
fn flash_next_serialized_prefix_fork_survives_peer_and_abort() {
    let ((), dispatches) =
        crate::model::shared::qwen4_exp_gdn_metal::with_mode(true, flash_next_prefix_fork_body);
    assert!(
        dispatches > 0,
        "the probe must exercise native singleton GDN"
    );
    eprintln!("FLASH_NEXT_PREFIX_FORK_NATIVE_GDN_DISPATCHES {dispatches}");
}

fn flash_next_prefix_fork_body() {
    use super::MlxPrefixSnapshot;

    const CUT: usize = 2048;
    const SUFFIX: usize = 64;
    const REPLY: usize = 16;
    const ABANDONED: usize = 4;
    const WIDTH: usize = 512;

    let pack = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let source = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_PREFIX_FORK_CASES").unwrap());
    let [seed, first_suffix, second_suffix, peer]: [Vec<u32>; 4] =
        serde_json::from_slice(&std::fs::read(source).unwrap()).unwrap();
    assert_eq!(
        [
            seed.len(),
            first_suffix.len(),
            second_suffix.len(),
            peer.len()
        ],
        [CUT, SUFFIX, SUFFIX, WIDTH]
    );
    assert_ne!(first_suffix, second_suffix);
    assert_ne!(seed[0], peer[0]);

    let artifacts = NativeModelArtifacts::from_dir(pack).unwrap();
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    assert_eq!(cfg.model_family, "qwen4_exp");
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let trunk = weights.qwen4_exp.as_ref().unwrap();
    assert!(trunk.expert_stream.is_none());
    let suffix_width = crate::fastpath::scale_prefill_chunk_for_remaining_in_family(
        1024,
        SUFFIX,
        &cfg.model_family,
    );
    assert_eq!(
        crate::fastpath::scale_prefill_chunk_for_remaining_in_family(1024, CUT, &cfg.model_family),
        WIDTH
    );

    let sampling = MlxSamplingRequest::new(MlxSamplingParams::greedy(), &[]);
    let cache = || MlxKVCache::new_contiguous(cfg.layer_count);
    let first = |tokens: &[u32], state: &mut MlxKVCache, width| {
        chunked_prefill(
            &cfg,
            &weights,
            tokens,
            state,
            width,
            sampling,
            &mut Xorshift64::new(42),
        )
    };
    let next = |state: &mut MlxKVCache, token| {
        decode_step(
            &cfg,
            &weights,
            token,
            state,
            sampling,
            &mut Xorshift64::new(42),
        )
    };
    let state_hash = |cache: &MlxKVCache| ax_engine_core::sha256_hex(&cache.serialize_to_bytes());

    let mut seed_cache = cache();
    let seed_token = first(&seed, &mut seed_cache, WIDTH);
    let wire = seed_cache.serialize_to_bytes();
    let seed_hash = ax_engine_core::sha256_hex(&wire);
    let snapshot =
        MlxPrefixSnapshot::from_shared_payload(wire.into(), seed, CUT, Some(seed_token), None);
    drop(seed_cache);
    let cold = |suffix: &[u32]| {
        let mut state = cache();
        chunked_prefill_cache_only(
            &cfg,
            &weights,
            &snapshot.tokens,
            &mut state,
            WIDTH,
            CacheOnlyPrefillLayout::PreserveFinalTokenStep,
            CacheOnlyBarrier::AsyncSubmit,
        );
        assert_eq!(state_hash(&state), seed_hash);
        let mut ids = vec![first(suffix, &mut state, suffix_width)];
        assert_eq!(state.seq_len(), CUT + SUFFIX);
        let suffix_hash = state_hash(&state);
        for _ in 1..REPLY {
            ids.push(next(&mut state, *ids.last().unwrap()));
        }
        (ids, suffix_hash, state_hash(&state))
    };
    let first_cold = cold(&first_suffix);
    let second_cold = cold(&second_suffix);
    assert_ne!(first_cold.1, second_cold.1);
    assert_ne!(first_cold.0, second_cold.0);
    assert!(
        [first_cold.0[0], second_cold.0[0]]
            .iter()
            .any(|token| *token != seed_token)
    );

    let mut peer_solo = cache();
    let mut peer_solo_ids = vec![first(&peer, &mut peer_solo, WIDTH)];
    for _ in 0..(ABANDONED + REPLY) {
        peer_solo_ids.push(next(&mut peer_solo, *peer_solo_ids.last().unwrap()));
    }
    let peer_hash = state_hash(&peer_solo);
    drop(peer_solo);
    let restore = || {
        let mut restored = snapshot.rehydrate_cache().unwrap();
        restored
            .verify_restored_snapshot(cfg.layer_count, CUT, None)
            .unwrap();
        restored
            .qwen4_exp
            .as_mut()
            .unwrap()
            .rebind_for_model(trunk, cfg.compile_cache_identity)
            .unwrap();
        assert_eq!(state_hash(&restored), seed_hash);
        restored
    };

    let mut peer_live = cache();
    let mut peer_live_ids = vec![first(&peer, &mut peer_live, WIDTH)];
    let mut advance_peer = || {
        peer_live_ids.push(next(&mut peer_live, *peer_live_ids.last().unwrap()));
    };
    let mut abandoned = restore();
    let mut abandoned_ids = vec![first(&first_suffix, &mut abandoned, suffix_width)];
    assert_eq!(state_hash(&abandoned), first_cold.1);
    advance_peer();
    for _ in 1..ABANDONED {
        abandoned_ids.push(next(&mut abandoned, *abandoned_ids.last().unwrap()));
        advance_peer();
    }
    assert_eq!(abandoned_ids, first_cold.0[..ABANDONED]);
    drop(abandoned);

    let mut replacement = restore();
    let mut second_ids = vec![first(&second_suffix, &mut replacement, suffix_width)];
    assert_eq!(state_hash(&replacement), second_cold.1);
    advance_peer();
    for _ in 1..REPLY {
        second_ids.push(next(&mut replacement, *second_ids.last().unwrap()));
        advance_peer();
    }
    assert_eq!(second_ids, second_cold.0);
    assert_eq!(state_hash(&replacement), second_cold.2);
    assert_eq!(peer_live_ids, peer_solo_ids);
    assert_eq!(state_hash(&peer_live), peer_hash);
    assert_eq!(
        ax_engine_core::sha256_hex(&snapshot.kv_cache_payload),
        seed_hash
    );
    eprintln!(
        "FLASH_NEXT_PREFIX_FORK {}",
        serde_json::json!({
            "qualification": false,
            "prefix_tokens": CUT,
            "suffix_tokens": SUFFIX,
            "physical_width": WIDTH,
            "suffix_width": suffix_width,
            "abandoned_outputs": abandoned_ids,
            "replacement_outputs": second_ids,
            "peer_outputs": peer_live_ids,
            "seed_state_sha256": seed_hash,
            "suffix_state_sha256": [first_cold.1, second_cold.1],
            "replacement_decode_state_sha256": second_cold.2,
            "peer_decode_state_sha256": peer_hash,
        })
    );
}

#[test]
#[ignore = "requires the pinned resident Flash Next MXFP4 pack and prefix suffix fixture"]
fn flash_next_serialized_prefix_matches_cold_across_suffix_boundaries() {
    let (rows, dispatches) = crate::model::shared::qwen4_exp_gdn_metal::with_mode(
        true,
        flash_next_prefix_suffix_matrix_body,
    );
    assert!(
        dispatches > 0,
        "the probe must exercise native singleton GDN"
    );
    eprintln!(
        "FLASH_NEXT_PREFIX_SUFFIX_MATRIX {}",
        serde_json::json!({
            "qualification": false,
            "native_gdn_dispatches": dispatches,
            "observations": rows,
        })
    );
}

fn flash_next_prefix_suffix_matrix_body() -> serde_json::Value {
    use super::MlxPrefixSnapshot;

    const PREFIX: usize = 2048;
    const PREFIX_WIDTH: usize = 512;
    const BASE_WIDTH: usize = 1024;
    const SUFFIX_LENGTHS: [usize; 10] = [1, 16, 63, 64, 65, 512, 1024, 1025, 2047, 2048];
    const DECODE_STEPS: usize = 4;

    let pack = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let fixture = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_PREFIX_SUFFIX_CASES").unwrap());
    let (prefix, suffixes): (Vec<u32>, Vec<Vec<u32>>) =
        serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    assert_eq!(prefix.len(), PREFIX);
    assert_eq!(
        suffixes.iter().map(Vec::len).collect::<Vec<_>>(),
        SUFFIX_LENGTHS
    );

    let artifacts = NativeModelArtifacts::from_dir(pack).unwrap();
    assert!(artifacts.audited_qwen4_exp_mxfp4());
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let trunk = weights.qwen4_exp.as_ref().unwrap();
    assert!(trunk.expert_stream.is_none());
    let chunk_width = |len| {
        crate::fastpath::scale_prefill_chunk_for_remaining_in_family(
            BASE_WIDTH,
            len,
            &cfg.model_family,
        )
    };
    assert_eq!(chunk_width(PREFIX), PREFIX_WIDTH);

    let sampling = MlxSamplingRequest::new(MlxSamplingParams::greedy(), &[]);
    let fresh = || MlxKVCache::new_contiguous(cfg.layer_count);
    let hash_state = |cache: &MlxKVCache| ax_engine_core::sha256_hex(&cache.serialize_to_bytes());
    let mut seed = fresh();
    let seed_token = chunked_prefill(
        &cfg,
        &weights,
        &prefix,
        &mut seed,
        PREFIX_WIDTH,
        sampling,
        &mut Xorshift64::new(42),
    );
    assert_eq!(seed.seq_len(), PREFIX);
    let payload = seed.serialize_to_bytes();
    let prefix_hash = ax_engine_core::sha256_hex(&payload);
    let snapshot = MlxPrefixSnapshot::from_shared_payload(
        payload.into(),
        prefix,
        PREFIX,
        Some(seed_token),
        None,
    );
    drop(seed);

    let finish = |mut cache: MlxKVCache, suffix: &[u32], width| {
        let mut rng = Xorshift64::new(42);
        let first = chunked_prefill(
            &cfg, &weights, suffix, &mut cache, width, sampling, &mut rng,
        );
        assert_eq!(cache.seq_len(), PREFIX + suffix.len());
        let after_prefill = hash_state(&cache);
        let mut ids = vec![first];
        for _ in 0..DECODE_STEPS {
            ids.push(decode_step(
                &cfg,
                &weights,
                *ids.last().unwrap(),
                &mut cache,
                sampling,
                &mut rng,
            ));
        }
        assert_eq!(cache.seq_len(), PREFIX + suffix.len() + DECODE_STEPS);
        (ids, after_prefill, hash_state(&cache))
    };

    let mut rows = Vec::with_capacity(suffixes.len());
    for suffix in suffixes {
        let width = chunk_width(suffix.len());
        assert_eq!(
            width,
            if suffix.len() == PREFIX {
                PREFIX_WIDTH
            } else {
                BASE_WIDTH
            }
        );

        let mut cold = fresh();
        chunked_prefill_cache_only(
            &cfg,
            &weights,
            &snapshot.tokens,
            &mut cold,
            PREFIX_WIDTH,
            CacheOnlyPrefillLayout::PreserveFinalTokenStep,
            CacheOnlyBarrier::AsyncSubmit,
        );
        assert_eq!(hash_state(&cold), prefix_hash);
        let cold = finish(cold, &suffix, width);

        let mut warm = snapshot.rehydrate_cache().unwrap();
        warm.verify_restored_snapshot(cfg.layer_count, PREFIX, None)
            .unwrap();
        warm.qwen4_exp
            .as_mut()
            .unwrap()
            .rebind_for_model(trunk, cfg.compile_cache_identity)
            .unwrap();
        assert_eq!(hash_state(&warm), prefix_hash);
        let warm = finish(warm, &suffix, width);
        assert_eq!(
            cold,
            warm,
            "suffix length {} changed the cold trajectory",
            suffix.len()
        );
        rows.push(serde_json::json!({
            "suffix_tokens": suffix.len(),
            "suffix_width": width,
            "cold": {"output_ids": cold.0, "post_prefill_sha256": cold.1, "post_decode_sha256": cold.2},
            "warm": {"output_ids": warm.0, "post_prefill_sha256": warm.1, "post_decode_sha256": warm.2},
        }));
    }
    assert_eq!(
        ax_engine_core::sha256_hex(&snapshot.kv_cache_payload),
        prefix_hash
    );
    serde_json::json!({"prefix_tokens": PREFIX, "prefix_width": PREFIX_WIDTH, "prefix_sha256": prefix_hash, "rows": rows})
}
