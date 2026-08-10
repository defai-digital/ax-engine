//! Compare sequential forward_argmax vs multi-token teacher-forced argmax.
//!
//! Usage: gemma_mt_identity_probe <model_dir> <id,id,...> [gen_steps]

use std::env;
use std::path::Path;
use std::process::ExitCode;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill},
    kv_cache::MlxKVCache,
    model::{ModelConfig, forward_all_positions_with_post_norm_greedy, forward_argmax},
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};
use mlx_sys::{MlxArray, MlxDtype, argmax, astype, eval, slice};

#[derive(Debug)]
struct CacheTailDiff {
    layer: usize,
    tensor: &'static str,
    position: usize,
    differing_elements: usize,
    max_abs_diff: f32,
}

fn array_diff(a: &MlxArray, b: &MlxArray) -> Option<(usize, f32)> {
    let a = astype(a, MlxDtype::Float32, None);
    let b = astype(b, MlxDtype::Float32, None);
    eval(&[&a, &b]);
    let a = a.data_f32();
    let b = b.data_f32();
    assert_eq!(a.len(), b.len(), "cache-tail arrays must have equal size");
    let mut differing_elements = 0usize;
    let mut max_abs_diff = 0.0f32;
    for (&lhs, &rhs) in a.iter().zip(b) {
        if lhs.to_bits() != rhs.to_bits() {
            differing_elements += 1;
            max_abs_diff = max_abs_diff.max((lhs - rhs).abs());
        }
    }
    (differing_elements > 0).then_some((differing_elements, max_abs_diff))
}

fn first_cache_tail_diff(
    lhs: &MlxKVCache,
    rhs: &MlxKVCache,
    layer_count: usize,
    start: usize,
    len: usize,
) -> Option<CacheTailDiff> {
    for layer in 0..layer_count {
        let Some((lhs_k, lhs_v)) = lhs.logical_layer_kv(layer) else {
            continue;
        };
        let Some((rhs_k, rhs_v)) = rhs.logical_layer_kv(layer) else {
            continue;
        };
        for position in start..start + len {
            for (tensor, lhs, rhs) in [("K", &lhs_k, &rhs_k), ("V", &lhs_v, &rhs_v)] {
                let lhs_shape = lhs.shape();
                let rhs_shape = rhs.shape();
                assert_eq!(lhs_shape, rhs_shape, "cache shapes differ at layer {layer}");
                let heads = lhs_shape[1];
                let head_dim = lhs_shape[3];
                let lhs_row = slice(
                    lhs,
                    &[0, 0, position as i32, 0],
                    &[1, heads, (position + 1) as i32, head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                let rhs_row = slice(
                    rhs,
                    &[0, 0, position as i32, 0],
                    &[1, heads, (position + 1) as i32, head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                if let Some((differing_elements, max_abs_diff)) = array_diff(&lhs_row, &rhs_row) {
                    return Some(CacheTailDiff {
                        layer,
                        tensor,
                        position,
                        differing_elements,
                        max_abs_diff,
                    });
                }
            }
        }
    }
    None
}

fn parse_ids(spec: &str) -> Result<Vec<u32>, String> {
    let ids: Result<Vec<_>, _> = spec
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse::<u32>().map_err(|_| format!("bad id {t}")))
        .collect();
    let ids = ids?;
    if ids.is_empty() {
        return Err("empty ids".into());
    }
    Ok(ids)
}

fn prefill(
    cfg: &ModelConfig,
    weights: &ax_engine_mlx::weights::ModelWeights,
    ids: &[u32],
) -> (MlxKVCache, u32) {
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let bootstrap = chunked_prefill(
        cfg,
        weights,
        ids,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), ids),
        &mut rng,
    );
    if let Some(raw) = env::var_os("AX_GEMMA_MT_PROBE_RING_SLACK") {
        let raw = raw
            .to_str()
            .expect("AX_GEMMA_MT_PROBE_RING_SLACK must be UTF-8");
        let slack = raw
            .parse::<usize>()
            .expect("AX_GEMMA_MT_PROBE_RING_SLACK must be an integer");
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(slack);
    }
    (cache, bootstrap)
}

fn step_argmax(
    cfg: &ModelConfig,
    weights: &ax_engine_mlx::weights::ModelWeights,
    cache: &mut MlxKVCache,
    last: u32,
) -> u32 {
    let off = cache.seq_len();
    let logits = forward_argmax(cfg, weights, &[last], cache, off);
    cache.advance(1);
    let pred = argmax(&logits, None);
    {
        let kv = cache.collect_eval_refs();
        let mut t: Vec<&_> = Vec::with_capacity(1 + kv.len());
        t.push(&pred);
        t.extend(kv);
        eval(&t);
    }
    pred.data_u32()[0]
}

fn step_mt_greedy(
    cfg: &ModelConfig,
    weights: &ax_engine_mlx::weights::ModelWeights,
    cache: &mut MlxKVCache,
    tokens: &[u32],
) -> Vec<u32> {
    let off = cache.seq_len();
    let (logits, _) = forward_all_positions_with_post_norm_greedy(cfg, weights, tokens, cache, off);
    cache.advance(tokens.len());
    let pred = argmax(&logits, None);
    {
        let kv = cache.collect_eval_refs();
        let mut t: Vec<&_> = Vec::with_capacity(1 + kv.len());
        t.push(&pred);
        t.extend(kv);
        eval(&t);
    }
    pred.data_u32().to_vec()
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: gemma_mt_identity_probe <model> <ids> [steps]")?;
    let ids = parse_ids(&args.next().ok_or("missing ids")?)?;
    let steps = args
        .next()
        .map(|s| s.parse().map_err(|_| format!("bad steps {s}")))
        .transpose()?
        .unwrap_or(16usize);
    let diagnose_cache = env::var_os("AX_GEMMA_MT_PROBE_CACHE_DIFF").is_some()
        && env::var_os("AX_GEMMA_MT_PROBE_RING_SLACK").is_none();

    let artifacts =
        NativeModelArtifacts::from_dir(Path::new(&model)).map_err(|e| format!("artifacts: {e}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).map_err(|e| format!("weights: {e}"))?;
    eprintln!(
        "moe_experts={} layers={} hidden={}",
        cfg.moe_expert_count,
        weights.layers.len(),
        cfg.hidden_size
    );

    // --- seq=1 path: forward_argmax vs multi-token greedy ---
    let (mut cache_seq, boot_seq) = prefill(&cfg, &weights, &ids);
    let (mut cache_mt, boot_mt) = prefill(&cfg, &weights, &ids);
    println!(
        "bootstrap_seq={boot_seq} bootstrap_mt={boot_mt} match={}",
        boot_seq == boot_mt
    );

    let mut last_seq = boot_seq;
    let mut last_mt = boot_mt;
    let mut seq_tokens = vec![boot_seq];
    let mut mt_tokens = vec![boot_mt];
    let mut seq1_diff: Option<usize> = None;
    if boot_seq != boot_mt {
        seq1_diff = Some(0);
    }
    for step in 1..steps {
        if seq1_diff.is_some() {
            break;
        }
        let s = step_argmax(&cfg, &weights, &mut cache_seq, last_seq);
        let m = step_mt_greedy(&cfg, &weights, &mut cache_mt, &[last_mt])[0];
        seq_tokens.push(s);
        mt_tokens.push(m);
        if s != m {
            seq1_diff = Some(step);
            eprintln!("seq1 DIFF step={step} seq={s} mt={m}");
            break;
        }
        last_seq = s;
        last_mt = m;
    }

    // --- multi-token batch seq=2 vs two sequential steps ---
    let (mut cache_s2, boot2) = prefill(&cfg, &weights, &ids);
    let (mut cache_m2, _) = prefill(&cfg, &weights, &ids);
    let t0 = step_argmax(&cfg, &weights, &mut cache_s2, boot2);
    let t1 = step_argmax(&cfg, &weights, &mut cache_s2, t0);
    let mt2 = step_mt_greedy(&cfg, &weights, &mut cache_m2, &[boot2, t0]);
    println!("seq1_first_diff={seq1_diff:?}");
    println!("seq1_tokens={seq_tokens:?}");
    println!("mt1_tokens={mt_tokens:?}");
    println!("seq2=[{boot2},{t0},{t1}] (boot + 2 steps)");
    println!("mt2_input=[{boot2},{t0}] mt2_pred={mt2:?}");
    println!(
        "mt2_match_pos0={} (expect {t0}) mt2_match_pos1={} (expect {t1})",
        mt2.first().copied() == Some(t0),
        mt2.get(1).copied() == Some(t1)
    );

    // Also seq=3 teacher force from prefill
    let (mut cache_s3, boot3) = prefill(&cfg, &weights, &ids);
    let (mut cache_m3, _) = prefill(&cfg, &weights, &ids);
    let a0 = step_argmax(&cfg, &weights, &mut cache_s3, boot3);
    let a1 = step_argmax(&cfg, &weights, &mut cache_s3, a0);
    let a2 = step_argmax(&cfg, &weights, &mut cache_s3, a1);
    let mt3 = step_mt_greedy(&cfg, &weights, &mut cache_m3, &[boot3, a0, a1]);
    println!("seq3_steps=[{boot3},{a0},{a1},{a2}]");
    println!("mt3_input=[{boot3},{a0},{a1}] mt3_pred={mt3:?}");
    println!(
        "mt3_match={} {} {}",
        mt3.first().copied() == Some(a0),
        mt3.get(1).copied() == Some(a1),
        mt3.get(2).copied() == Some(a2)
    );

    // Mid-decode multi-token: pure-direct N steps, then MT teacher-forced vs more PD.
    // Formal A/B first_diff often appears after a few decode tokens (not at prefill).
    let warm = steps.clamp(3, 8);
    let (mut cache_pd, boot_pd) = prefill(&cfg, &weights, &ids);
    let (mut cache_mt_mid, _) = prefill(&cfg, &weights, &ids);
    let mut hist = vec![boot_pd];
    let mut last = boot_pd;
    for _ in 0..warm {
        let prev = last;
        last = step_argmax(&cfg, &weights, &mut cache_pd, prev);
        hist.push(last);
        let _ = step_argmax(&cfg, &weights, &mut cache_mt_mid, prev);
    }
    // Both caches should be at same length; next pure-direct tokens
    let n0 = step_argmax(&cfg, &weights, &mut cache_pd, last);
    let n1 = step_argmax(&cfg, &weights, &mut cache_pd, n0);
    let n2 = step_argmax(&cfg, &weights, &mut cache_pd, n1);
    let mt_mid = step_mt_greedy(&cfg, &weights, &mut cache_mt_mid, &[last, n0, n1]);
    println!(
        "mid_warm={warm} hist_tail={:?}",
        &hist[hist.len().saturating_sub(4)..]
    );
    println!("mid_pd_next=[{n0},{n1},{n2}]");
    println!("mid_mt_input=[{last},{n0},{n1}] mid_mt_pred={mt_mid:?}");
    let mid_ok = mt_mid.first().copied() == Some(n0)
        && mt_mid.get(1).copied() == Some(n1)
        && mt_mid.get(2).copied() == Some(n2);
    println!("mid_mt_match={mid_ok}");
    if !mid_ok {
        eprintln!("MID-DECODE IDENTITY FAIL: expected [{n0},{n1},{n2}] got {mt_mid:?}");
        return Err("mid-decode multi-token identity failed".into());
    }

    // --- CONTAMINATION: multi-token with WRONG later tokens must keep pos0 == PD ---
    let (mut cache_pd_c, boot_c) = prefill(&cfg, &weights, &ids);
    let (mut cache_mt_c, _) = prefill(&cfg, &weights, &ids);
    let mut last_c = boot_c;
    for _ in 0..4 {
        let prev = last_c;
        last_c = step_argmax(&cfg, &weights, &mut cache_pd_c, prev);
        let _ = step_argmax(&cfg, &weights, &mut cache_mt_c, prev);
    }
    let (mut cache_pd2, boot2) = prefill(&cfg, &weights, &ids);
    let (mut cache_mt2, _) = prefill(&cfg, &weights, &ids);
    let mut last2 = boot2;
    for _ in 0..4 {
        let prev = last2;
        last2 = step_argmax(&cfg, &weights, &mut cache_pd2, prev);
        let _ = step_argmax(&cfg, &weights, &mut cache_mt2, prev);
    }
    let pd_next = step_argmax(&cfg, &weights, &mut cache_pd2, last2);
    // Wrong drafts at pos1/pos2
    let wrong = [last2, 1u32, 2u32];
    let mt_wrong = step_mt_greedy(&cfg, &weights, &mut cache_mt2, &wrong);
    let contam_pos0_ok = mt_wrong.first().copied() == Some(pd_next);
    println!("contam_wrong_drafts mt_pred={mt_wrong:?} pd_next={pd_next} pos0_ok={contam_pos0_ok}");
    if !contam_pos0_ok {
        eprintln!("CONTAMINATION FAIL: wrong drafts changed pos0 prediction");
        return Err("multi-token pos0 contaminated by later draft tokens".into());
    }

    // --- SUCCESSIVE_ALWAYS_ADOPT: N multi-token commits (S=3) vs pure-direct ---
    // Reproduces formal runner always-adopt + trim semantics after long prefill.
    {
        let (mut cache_pd, boot) = prefill(&cfg, &weights, &ids);
        let (mut cache_mt, boot2) = prefill(&cfg, &weights, &ids);
        assert_eq!(boot, boot2);
        let mut last_pd = boot;
        let mut last_mt = boot2;
        let mut succ_diff: Option<usize> = None;
        let mut first_cache_diff_step: Option<usize> = None;
        let mut pd_hist = vec![boot];
        let mut mt_hist = vec![boot2];
        for step in 0..steps {
            // pure-direct two steps for expected chain
            let e0 = step_argmax(&cfg, &weights, &mut cache_pd, last_pd);
            let e1 = step_argmax(&cfg, &weights, &mut cache_pd, e0);
            let e2 = step_argmax(&cfg, &weights, &mut cache_pd, e1);
            // MT teacher-forced with CORRECT drafts [last, e0, e1] — full accept path
            let off = cache_mt.seq_len();
            let input = [last_mt, e0, e1];
            let (logits, _) = forward_all_positions_with_post_norm_greedy(
                &cfg,
                &weights,
                &input,
                &mut cache_mt,
                off,
            );
            cache_mt.advance(input.len());
            let pred_arr = argmax(&logits, None);
            {
                let kv = cache_mt.collect_eval_refs();
                let mut trefs: Vec<&_> = Vec::with_capacity(1 + kv.len());
                trefs.push(&pred_arr);
                trefs.extend(kv);
                eval(&trefs);
            }
            let pred = pred_arr.data_u32().to_vec();
            if diagnose_cache {
                let cache_diff =
                    first_cache_tail_diff(&cache_pd, &cache_mt, cfg.layer_count, off, input.len());
                if step == 0 || (cache_diff.is_some() && first_cache_diff_step.is_none()) {
                    if let Some(diff) = &cache_diff {
                        println!(
                            "successive_cache_tail step={step} layer={} tensor={} position={} differing_elements={} max_abs_diff={}",
                            diff.layer,
                            diff.tensor,
                            diff.position,
                            diff.differing_elements,
                            diff.max_abs_diff
                        );
                    } else {
                        println!("successive_cache_tail step={step} diff=None");
                    }
                }
                if cache_diff.is_some() && first_cache_diff_step.is_none() {
                    first_cache_diff_step = Some(step);
                }
            }
            // accept count
            let mut ac = 0usize;
            let drafts = [e0, e1];
            for (i, &d) in drafts.iter().enumerate() {
                if pred.get(i).copied() == Some(d) {
                    ac += 1;
                } else {
                    break;
                }
            }
            let committed = off + 1 + ac;
            if !cache_mt.trim_to(committed) {
                eprintln!("trim refused committed={committed}");
            }
            let got0 = pred.first().copied().unwrap_or(0);
            pd_hist.push(e0);
            // Reset PD comparison: only take first token each outer step to avoid multi-step PD desync
            // Simpler successive: each step MT S=3 with correct teacher, take ac=2 full, last=e2
            if got0 != e0 || pred.get(1).copied() != Some(e1) || pred.get(2).copied() != Some(e2) {
                succ_diff = Some(step);
                eprintln!(
                    "SUCCESSIVE DIFF step={step} expect=[{e0},{e1},{e2}] pred={pred:?} ac={ac} seq_pd={} seq_mt={}",
                    cache_pd.seq_len(),
                    cache_mt.seq_len()
                );
                break;
            }
            if ac != 2 {
                succ_diff = Some(step);
                eprintln!("SUCCESSIVE partial ac={ac} step={step} pred={pred:?}");
                break;
            }
            // PD ran 3 steps; MT committed 3 positions (1 context write + 2 accepts? verify_len=3, committed=off+1+2=off+3)
            // last tokens
            last_pd = e2;
            last_mt = e2;
            mt_hist.push(e0);
            mt_hist.push(e1);
            mt_hist.push(e2);
            // Realign PD/MT lengths
            if cache_pd.seq_len() != cache_mt.seq_len() {
                eprintln!(
                    "len mismatch step={step} pd={} mt={}",
                    cache_pd.seq_len(),
                    cache_mt.seq_len()
                );
                succ_diff = Some(step);
                break;
            }
        }
        println!("successive_mt_first_diff={succ_diff:?}");

        // SUCCESSIVE S=2 (depth-1)
        {
            let (mut cache_pd, boot) = prefill(&cfg, &weights, &ids);
            let (mut cache_mt, boot2) = prefill(&cfg, &weights, &ids);
            let mut last_pd = boot;
            let mut last_mt = boot2;
            let mut succ_diff: Option<usize> = None;
            for step in 0..steps {
                let e0 = step_argmax(&cfg, &weights, &mut cache_pd, last_pd);
                let e1 = step_argmax(&cfg, &weights, &mut cache_pd, e0);
                let off = cache_mt.seq_len();
                let input = [last_mt, e0];
                let (logits, _) = forward_all_positions_with_post_norm_greedy(
                    &cfg,
                    &weights,
                    &input,
                    &mut cache_mt,
                    off,
                );
                cache_mt.advance(input.len());
                let pred_arr = argmax(&logits, None);
                {
                    let kv = cache_mt.collect_eval_refs();
                    let mut trefs: Vec<&_> = Vec::with_capacity(1 + kv.len());
                    trefs.push(&pred_arr);
                    trefs.extend(kv);
                    eval(&trefs);
                }
                let pred = pred_arr.data_u32().to_vec();
                if pred.first().copied() != Some(e0) || pred.get(1).copied() != Some(e1) {
                    succ_diff = Some(step);
                    eprintln!("SUCCESSIVE S2 DIFF step={step} expect=[{e0},{e1}] pred={pred:?}");
                    break;
                }
                let committed = off + 2; // full accept 1 draft
                let _ = cache_mt.trim_to(committed);
                last_pd = e1;
                last_mt = e1;
            }
            println!("successive_s2_first_diff={succ_diff:?}");
        }

        if succ_diff.is_some() {
            return Err("successive multi-token always-adopt drifted".into());
        }
    }

    // --- COMMIT SEAM: after MT teacher-forced accept of 2 correct tokens, next PD matches ---
    let (mut cache_pd3, boot3) = prefill(&cfg, &weights, &ids);
    let (mut cache_mt3, _) = prefill(&cfg, &weights, &ids);
    let mut last3 = boot3;
    for _ in 0..4 {
        let prev = last3;
        last3 = step_argmax(&cfg, &weights, &mut cache_pd3, prev);
        let _ = step_argmax(&cfg, &weights, &mut cache_mt3, prev);
    }
    let p0 = step_argmax(&cfg, &weights, &mut cache_pd3, last3);
    let p1 = step_argmax(&cfg, &weights, &mut cache_pd3, p0);
    let p2 = step_argmax(&cfg, &weights, &mut cache_pd3, p1);
    // MT advances with teacher-forced [last3, p0, p1] matching sequential
    let mt_commit = step_mt_greedy(&cfg, &weights, &mut cache_mt3, &[last3, p0, p1]);
    let commit_tok_ok = mt_commit.first().copied() == Some(p0)
        && mt_commit.get(1).copied() == Some(p1)
        && mt_commit.get(2).copied() == Some(p2);
    // After MT wrote 3 positions, next PD on both caches should match
    // MT cache is at last3+3 positions; PD cache continued to p2 (3 steps from last3)
    // Align: PD cache has processed last3,p0,p1,p2 as inputs... wait step_argmax(cache, token) processes token and predicts next.
    // After warm, last3 is last generated. PD: process last3->p0, p0->p1, p1->p2. Cache has KVs for last3,p0,p1.
    // MT: process [last3,p0,p1] in one go, predicts [p0,p1,p2], advance 3. Cache has KVs for last3,p0,p1.
    let pd_after = step_argmax(&cfg, &weights, &mut cache_pd3, p2);
    let mt_after = step_argmax(&cfg, &weights, &mut cache_mt3, p2);
    let commit_next_ok = pd_after == mt_after;
    println!(
        "commit_mt_pred={mt_commit:?} expect=[{p0},{p1},{p2}] tok_ok={commit_tok_ok} next_pd={pd_after} next_mt={mt_after} next_ok={commit_next_ok}"
    );
    if !commit_tok_ok || !commit_next_ok {
        eprintln!("COMMIT SEAM FAIL: tok_ok={commit_tok_ok} next_ok={commit_next_ok}");
        return Err("multi-token commit seam diverged from pure-direct".into());
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
