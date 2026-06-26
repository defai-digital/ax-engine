use super::*;

pub(crate) fn direct_pipeline_clear_cache_due(emitted_tokens: u32, cadence: u32) -> bool {
    cadence != 0
        && emitted_tokens != 0
        && (emitted_tokens == 1 || emitted_tokens.saturating_sub(1).is_multiple_of(cadence))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DirectPipelineAction {
    FinishPending,
    ContinuePending,
    BootstrapFinal,
    Bootstrap,
}

pub(crate) fn direct_pipeline_action(
    has_pending_direct: bool,
    final_by_max_output: bool,
) -> DirectPipelineAction {
    if final_by_max_output && has_pending_direct {
        DirectPipelineAction::FinishPending
    } else if has_pending_direct {
        DirectPipelineAction::ContinuePending
    } else if final_by_max_output {
        DirectPipelineAction::BootstrapFinal
    } else {
        DirectPipelineAction::Bootstrap
    }
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
