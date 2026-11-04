// SPDX-License-Identifier: Apache-2.0

//! Measurement-only entry points for microbenchmarks. Not a stable API.

use mlx_sys::{MlxArray, MlxDtype, argmax, astype, concatenate};

use crate::kv_cache::MlxKVCache;
use crate::model::{ModelConfig, layer_capture};
use crate::weights::ModelWeights;

/// Builds (lazily) the target verify forward of one DFlash2 round over `ids`
/// on `cache`, under the same scopes as `try_dflash2_decode`: relaxed Qwen
/// linear MTP session, target-verify fast path, exact arithmetic off, verify
/// QMM routing armed, linear prefix tape recording and hidden capture of
/// `capture_layers`. Returns the arrays the round evaluates (row argmax and
/// the bf16 captured hidden rows). The caller owns `cache` (pass a clone).
#[doc(hidden)]
pub fn dflash2_verify_forward_probe(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
    capture_layers: &[usize],
) -> Vec<MlxArray> {
    let _relaxed = crate::fastpath::scoped_qwen_linear_mtp_relaxed_session(true);
    let _fast = crate::fastpath::scoped_qwen_linear_mtp_target_verify(true);
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let _qmm = crate::model::shared::verify_qmm::QwenMtpVerifyQmmGuard::arm(true);
    let ids_arr = MlxArray::from_raw_data(
        ids.as_ptr() as *const u8,
        std::mem::size_of_val(ids),
        &[ids.len() as i32],
        MlxDtype::Uint32,
    );
    cache.begin_linear_prefix_capture(1);
    let capture = layer_capture::capture_layers(capture_layers);
    let (logits, _post_norm) = crate::model::forward_all_positions_with_post_norm_ids(
        cfg,
        weights,
        &ids_arr,
        ids.len(),
        cache,
        token_offset,
        false,
    );
    let captured = layer_capture::take();
    drop(capture);
    cache.advance(ids.len());
    let mut out = vec![argmax(&logits, None)];
    if let Some(captured) = captured {
        let refs: Vec<&MlxArray> = captured.iter().collect();
        out.push(astype(
            &concatenate(&refs, 2, None),
            MlxDtype::Bfloat16,
            None,
        ));
    }
    out
}

/// One lane for [`dflash2_packed_verify_probe`]: the lane's cache (a clone the
/// caller owns), the position of its first row and its verify ids.
#[doc(hidden)]
pub struct ProbeLane<'a> {
    pub cache: &'a mut MlxKVCache,
    pub token_offset: usize,
    pub ids: Vec<u32>,
}

/// Whether the row-packed multi-lane verify supports this model.
#[doc(hidden)]
pub fn dflash2_packed_verify_supported(cfg: &ModelConfig, weights: &ModelWeights) -> bool {
    crate::model::packed_verify::supported(cfg, weights)
}

/// Builds (lazily) one row-packed verify over every lane (see
/// `model::packed_verify`) under the same scopes as
/// [`dflash2_verify_forward_probe`]. Returns `[ΣW, vocab]` logits, then the
/// bf16 captured hidden rows `[1, ΣW, layers·hidden]` when `capture_layers` is
/// not empty. Every lane's cache is advanced by its rows.
#[doc(hidden)]
pub fn dflash2_packed_verify_probe(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    lanes: &mut [ProbeLane<'_>],
    capture_layers: &[usize],
) -> Vec<MlxArray> {
    let _relaxed = crate::fastpath::scoped_qwen_linear_mtp_relaxed_session(true);
    let _fast = crate::fastpath::scoped_qwen_linear_mtp_target_verify(true);
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let _qmm = crate::model::shared::verify_qmm::QwenMtpVerifyQmmGuard::arm(true);
    let ids: Vec<u32> = lanes.iter().flat_map(|l| l.ids.iter().copied()).collect();
    let ids_arr = MlxArray::from_raw_data(
        ids.as_ptr() as *const u8,
        std::mem::size_of_val(ids.as_slice()),
        &[ids.len() as i32],
        MlxDtype::Uint32,
    );
    for lane in lanes.iter_mut() {
        lane.cache.begin_linear_prefix_capture(1);
    }
    let capture = layer_capture::capture_layers(capture_layers);
    let mut packed: Vec<crate::model::packed_verify::PackedLane<'_>> = lanes
        .iter_mut()
        .map(|l| crate::model::packed_verify::PackedLane {
            rows: l.ids.len(),
            token_offset: l.token_offset,
            cache: &mut *l.cache,
        })
        .collect();
    let (logits, _normed) =
        crate::model::packed_verify::packed_verify_forward(cfg, weights, &ids_arr, &mut packed);
    drop(packed);
    let captured = layer_capture::take();
    drop(capture);
    for lane in lanes.iter_mut() {
        lane.cache.advance(lane.ids.len());
    }
    let mut out = vec![logits];
    if let Some(captured) = captured.filter(|c| !c.is_empty()) {
        let refs: Vec<&MlxArray> = captured.iter().collect();
        out.push(astype(
            &concatenate(&refs, 2, None),
            MlxDtype::Bfloat16,
            None,
        ));
    }
    out
}

/// Solo verify logits `[W, vocab]` of one lane, same scopes as the runner.
#[doc(hidden)]
pub fn dflash2_solo_verify_logits_probe(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    let _relaxed = crate::fastpath::scoped_qwen_linear_mtp_relaxed_session(true);
    let _fast = crate::fastpath::scoped_qwen_linear_mtp_target_verify(true);
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let _qmm = crate::model::shared::verify_qmm::QwenMtpVerifyQmmGuard::arm(true);
    let ids_arr = MlxArray::from_raw_data(
        ids.as_ptr() as *const u8,
        std::mem::size_of_val(ids),
        &[ids.len() as i32],
        MlxDtype::Uint32,
    );
    cache.begin_linear_prefix_capture(1);
    let (logits, _post_norm) = crate::model::forward_all_positions_with_post_norm_ids(
        cfg,
        weights,
        &ids_arr,
        ids.len(),
        cache,
        token_offset,
        false,
    );
    cache.advance(ids.len());
    logits
}

/// Median milliseconds of one DFlash2 drafter block per lane, drafted alone
/// (`packed == false`, one eval per lane) or in one packed graph, after the
/// drafter has absorbed `context_rows` rows. `None` when the drafter does not
/// load. Measurement only.
#[doc(hidden)]
pub fn dflash2_draft_probe(
    drafter_dir: &std::path::Path,
    target: &ModelWeights,
    lanes: usize,
    context_rows: i32,
    selector: bool,
    packed: bool,
    reps: usize,
) -> Option<f64> {
    use crate::dflash2::{Dflash2Drafter, Dflash2Weights, DrafterPrecision};
    let weights =
        std::sync::Arc::new(Dflash2Weights::load(drafter_dir, DrafterPrecision::Mxfp4).ok()?);
    weights.materialize();
    let width = weights.config().target_hidden_width();
    let block = weights.config().block_size;
    let hidden = |rows: i32| mlx_sys::zeros(&[1, rows, width], MlxDtype::Bfloat16, None);
    let mut drafters: Vec<Dflash2Drafter> = (0..lanes)
        .map(|_| Dflash2Drafter::new(weights.clone()).with_selector(selector))
        .collect();
    for drafter in &mut drafters {
        let ids = drafter
            .draft_block(1, &hidden(context_rows), target, block)
            .ok()?;
        mlx_sys::eval(&[&ids]);
    }
    if std::env::var_os("AX_COHORT_PROBE_DRAFT_STAGES").is_some() && lanes == 1 && !packed {
        let drafter = &mut drafters[0];
        drafter.set_stage_timing(true);
        let mut totals: Vec<(&'static str, f64)> = Vec::new();
        for _ in 0..reps {
            let ids = drafter.draft_block(7, &hidden(8), target, block).ok()?;
            mlx_sys::eval(&[&ids]);
            for (stage, ms) in drafter.take_stage_times() {
                match totals.iter_mut().find(|(name, _)| *name == stage) {
                    Some((_, total)) => *total += ms,
                    None => totals.push((stage, ms)),
                }
            }
        }
        drafter.set_stage_timing(false);
        for (stage, total) in totals {
            eprintln!("draft stage {stage}: {:.3} ms", total / reps as f64);
        }
    }
    let mut times = Vec::with_capacity(reps);
    for rep in 0..reps + 3 {
        let started = std::time::Instant::now();
        if packed && lanes > 1 {
            let mut lane_refs: Vec<(&mut Dflash2Drafter, u32, MlxArray)> =
                drafters.iter_mut().map(|d| (d, 7, hidden(8))).collect();
            let ids = Dflash2Drafter::draft_blocks_packed(&mut lane_refs, target, block)?;
            let refs: Vec<&MlxArray> = ids.iter().collect();
            mlx_sys::eval(&refs);
        } else {
            for drafter in &mut drafters {
                let ids = drafter.draft_block(7, &hidden(8), target, block).ok()?;
                mlx_sys::eval(&[&ids]);
            }
        }
        if rep >= 3 {
            times.push(started.elapsed().as_secs_f64() * 1e3);
        }
    }
    times.sort_by(f64::total_cmp);
    Some(times[times.len() / 2])
}
