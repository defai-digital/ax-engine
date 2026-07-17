use super::*;

pub(crate) fn direct_pipeline_clear_cache_due(emitted_tokens: u32, cadence: u32) -> bool {
    cadence != 0
        && emitted_tokens != 0
        && (emitted_tokens == 1 || emitted_tokens.saturating_sub(1).is_multiple_of(cadence))
}

/// The direct-pipeline dispatch, but carrying the pending array in the two
/// variants that need it, so the caller destructures infallibly instead of
/// re-reading the `Option` and defending against an unrepresentable
/// `action == FinishPending && pending == None` state.
///
/// Consumes the pending slot (via `Option::take`) exactly once, in lockstep
/// with the action choice — the same `(has_pending, final_by_max_output)`
/// truth table as [`direct_pipeline_action`], so behavior is identical to the
/// prior `action + match take()` pair on the happy path; the difference is
/// that the former `None`-arm `tracing::error!` fallbacks can no longer be
/// reached. (Decode-skeleton unification I1,
/// `.internal/specs/TECH-SPEC-DECODE-SKELETON-UNIFICATION.md`.)
pub(crate) enum DirectPipelineStep {
    FinishPending(MlxArray),
    ContinuePending(MlxArray),
    BootstrapFinal,
    Bootstrap,
}

pub(crate) fn next_direct_pipeline_step(
    pending_direct: &mut Option<MlxArray>,
    final_by_max_output: bool,
) -> DirectPipelineStep {
    match (pending_direct.take(), final_by_max_output) {
        (Some(pending), true) => DirectPipelineStep::FinishPending(pending),
        (Some(pending), false) => DirectPipelineStep::ContinuePending(pending),
        (None, true) => DirectPipelineStep::BootstrapFinal,
        (None, false) => DirectPipelineStep::Bootstrap,
    }
}

/// The direct-pipeline materialization point: block on the pending lazy token,
/// then read its scalar value off the GPU. Returns
/// `(token, eval_wall_us, read_wall_us)` so timing-sensitive callers can
/// record the split and timing-agnostic ones (the batched drain) can ignore
/// it.
///
/// This is the single barrier+readback the decode-skeleton plan centralizes
/// (I2); the zero-bubble reorder (I7) hooks exactly here once, instead of at
/// each open-coded `eval(&[&pending]) + first_u32_unchecked()` pair.
pub(crate) fn finish_pending_token(pending: &MlxArray) -> (u32, u32, u32) {
    let eval_started = Instant::now();
    eval(&[pending]);
    let eval_wall_us = elapsed_us(eval_started);
    let read_started = Instant::now();
    let tok = pending.first_u32_unchecked();
    let read_wall_us = elapsed_us(read_started);
    (tok, eval_wall_us, read_wall_us)
}

pub(crate) fn should_drain_pending_direct_before_ngram(
    is_greedy: bool,
    has_pending_direct: bool,
) -> bool {
    is_greedy && has_pending_direct
}

pub(crate) trait RouteDecisionSink {
    fn upsert_route_decision(&mut self, key: &str, value: u32);
}

impl RouteDecisionSink for Vec<(String, u32)> {
    fn upsert_route_decision(&mut self, key: &str, value: u32) {
        upsert_route_decision(self, key, value);
    }
}

pub(crate) struct IndexedRouteDecisions<'a> {
    pub(crate) decisions: &'a mut Vec<(String, u32)>,
    pub(crate) index: HashMap<String, usize>,
}

impl<'a> IndexedRouteDecisions<'a> {
    pub(crate) fn new(decisions: &'a mut Vec<(String, u32)>) -> Self {
        let mut compacted = Vec::with_capacity(decisions.len());
        let mut index = HashMap::with_capacity(decisions.len());
        for (key, value) in decisions.drain(..) {
            if index.contains_key(&key) {
                continue;
            }
            let position = compacted.len();
            index.insert(key.clone(), position);
            compacted.push((key, value));
        }
        *decisions = compacted;

        Self { decisions, index }
    }
}

impl RouteDecisionSink for IndexedRouteDecisions<'_> {
    fn upsert_route_decision(&mut self, key: &str, value: u32) {
        if let Some(position) = self.index.get(key).copied() {
            self.decisions[position].1 = value;
            return;
        }

        let position = self.decisions.len();
        self.decisions.push((key.to_string(), value));
        self.index.insert(key.to_string(), position);
    }
}

pub(crate) fn kv_layer_windows_from_config(cfg: &ModelConfig) -> Vec<Option<usize>> {
    let mut windows = vec![None; cfg.layer_count];
    for (idx, layer) in cfg.layer_configs.iter().enumerate().take(cfg.layer_count) {
        windows[idx] = layer.sliding_window;
    }
    windows
}
