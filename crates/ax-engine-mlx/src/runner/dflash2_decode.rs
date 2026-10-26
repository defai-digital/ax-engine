// SPDX-License-Identifier: Apache-2.0

//! DFlash2 speculative decode (ADR-045 increment (c)): the round loop around the
//! block drafter in `crate::dflash2`.
//!
//! One round drafts seven tokens from the target hidden states of the tokens
//! committed since the previous round, verifies `[anchor, drafts...]` in one
//! eight-row target forward on a clone of the cache while capturing the hidden
//! states of layers 5/19/33/47/61, accepts the longest matching prefix, commits
//! the rows that survive the EOS and output-budget cuts (adopting the clone,
//! restoring a checkpoint, or replaying the hybrid model's linear state from
//! the verify tape) and feeds the captured rows of those tokens to the next round.
//!
//! Greedy requests on the Qwen 3.8 hybrid dense model only. Anything else, and
//! any request whose captured hidden states stop matching its cache, takes the
//! normal decode routes: `try_dflash2_decode` returns `None` rather than failing.
//!
//! `AX_MLX_TREE_VERIFY` (default off) verifies a four-node tree of the first two
//! draft positions on separate cache clones instead of the single linear window.
//! Unset or `0` keeps the linear path.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use mlx_sys::{MlxArray, MlxDtype, argmax, astype, concatenate, contiguous, reshape, slice};

use super::*;
use crate::dflash2::{Dflash2Drafter, Dflash2Weights};
use crate::model::layer_capture;

/// RAM-only, detached BF16 prompt tail. Never retains a live model or MLX graph.
#[derive(Clone)]
pub(crate) struct Dflash2PrefixSnapshot {
    words: Arc<[u32]>,
    offset: usize,
    rows: usize,
    width: usize,
    target_identity: u64,
    drafter_identity: String,
}

impl Dflash2PrefixSnapshot {
    pub(crate) fn same_contract(&self, other: &Self) -> bool {
        self.offset == other.offset
            && self.target_identity == other.target_identity
            && self.drafter_identity == other.drafter_identity
    }
    pub(crate) fn bytes(&self) -> u64 {
        (self.words.len() as u64)
            .saturating_mul(4)
            .saturating_add(self.drafter_identity.len() as u64)
            .saturating_add(std::mem::size_of::<Self>() as u64)
    }

    fn capture(
        state: &Dflash2RequestState,
        weights: &Dflash2Weights,
        target_identity: u64,
    ) -> Option<Self> {
        if state.started || state.disabled {
            return None;
        }
        let hidden = state.pending_hidden.as_ref()?;
        let shape = hidden.shape();
        let width = weights.config().target_hidden_width() as usize;
        let rows = state
            .captured_rows
            .min((weights.config().sliding_window - 1) as usize);
        if rows == 0
            || !width.is_multiple_of(2)
            || shape != [1, rows as i32, width as i32]
            || hidden.dtype() != MlxDtype::Bfloat16
        {
            return None;
        }
        let packed = mlx_sys::view(&contiguous(hidden, None), MlxDtype::Uint32, None);
        mlx_sys::try_eval(&[&packed]).ok()?;
        Some(Self {
            words: packed.data_u32().to_vec().into(),
            offset: state.captured_rows,
            rows,
            width,
            target_identity,
            drafter_identity: weights.identity(),
        })
    }

    fn restore(
        &self,
        weights: Arc<Dflash2Weights>,
        target_identity: u64,
        offset: usize,
    ) -> Option<Dflash2RequestState> {
        let cfg = weights.config();
        if self.target_identity != target_identity
            || self.drafter_identity != weights.identity()
            || self.offset != offset
            || self.rows == 0
            || self.rows != offset.min((cfg.sliding_window - 1) as usize)
            || self.width != cfg.target_hidden_width() as usize
            || self.words.len().checked_mul(2)? != self.rows.checked_mul(self.width)?
            || offset > i32::MAX as usize
        {
            return None;
        }
        let packed = MlxArray::from_raw_data(
            self.words.as_ptr().cast(),
            self.words.len() * 4,
            &[1, self.rows as i32, (self.width / 2) as i32],
            MlxDtype::Uint32,
        );
        let hidden = mlx_sys::view(&packed, MlxDtype::Bfloat16, None);
        mlx_sys::try_eval(&[&hidden]).ok()?;
        let mut state = Dflash2RequestState::new(weights);
        state.pending_hidden = Some(hidden);
        state.captured_rows = offset;
        state.skipped_rows = offset - self.rows;
        Some(state)
    }
}

/// Per-request DFlash2 state.
pub(super) struct Dflash2RequestState {
    drafter: Dflash2Drafter,
    /// Target hidden states (`[1, n, width]`, bf16) of committed tokens the
    /// drafter has not consumed yet. Before the first round it holds the
    /// trailing prompt rows.
    pending_hidden: Option<MlxArray>,
    /// Tokens fed through the target (and captured), cumulatively.
    captured_rows: usize,
    /// Prompt rows dropped from the front of `pending_hidden` by the window cap
    /// before the drafter first saw them.
    skipped_rows: usize,
    /// The drafter has run at least once.
    started: bool,
    /// The captured rows stopped matching the cache; this request decodes normally.
    disabled: bool,
    window_stats: WindowStats,
    pub(super) rounds: u32,
    pub(super) drafted: u32,
    pub(super) accepted: u32,
    /// Last tree-verify round. Zero while [`crate::fastpath::tree_verify_enabled`]
    /// is off, which is the default.
    pub(super) tree_nodes_verified: u32,
    pub(super) tree_accepted_length: u32,
    /// `0` is the drafter's primary branch.
    pub(super) tree_branch_chosen: u32,
}

impl Dflash2RequestState {
    pub(super) fn new(weights: Arc<Dflash2Weights>) -> Self {
        Self {
            drafter: Dflash2Drafter::new(weights),
            pending_hidden: None,
            captured_rows: 0,
            skipped_rows: 0,
            started: false,
            disabled: false,
            window_stats: WindowStats::new(),
            rounds: 0,
            drafted: 0,
            accepted: 0,
            tree_nodes_verified: 0,
            tree_accepted_length: 0,
            tree_branch_chosen: 0,
        }
    }

    /// Test setup: hidden rows the next linear verify round will draft from.
    #[cfg(test)]
    pub(super) fn stage_pending_hidden_for_test(&mut self, hidden: MlxArray, captured_rows: usize) {
        self.pending_hidden = Some(hidden);
        self.captured_rows = captured_rows;
        self.disabled = false;
    }

    fn reset(&mut self) {
        self.drafter.reset();
        self.pending_hidden = None;
        self.captured_rows = 0;
        self.skipped_rows = 0;
        self.started = false;
        self.disabled = false;
        self.window_stats = WindowStats::new();
        self.tree_nodes_verified = 0;
        self.tree_accepted_length = 0;
        self.tree_branch_chosen = 0;
    }

    /// Appends captured prompt rows (`tail`: the trailing rows of an item whose
    /// `item_rows` tokens were fed through the target) and keeps the last
    /// `window` rows.
    fn append_prompt_rows(&mut self, tail: MlxArray, item_rows: usize, window: i32) {
        let combined = match self.pending_hidden.take() {
            Some(old) => concatenate(&[&old, &tail], 1, None),
            None => tail,
        };
        let shape = combined.shape();
        let rows = shape[1];
        self.pending_hidden = Some(if rows > window {
            contiguous(
                &slice(
                    &combined,
                    &[0, rows - window, 0],
                    &[shape[0], rows, shape[2]],
                    &[1, 1, 1],
                    None,
                ),
                None,
            )
        } else {
            combined
        });
        self.captured_rows += item_rows;
        // The accumulating capture may already have discarded leading rows.
        self.skipped_rows = self.captured_rows - rows.min(window) as usize;
    }
}

/// Smallest verify window: the anchor and two drafts.
const MIN_WINDOW: usize = 3;
/// Prior weight and rate for the per-position acceptance estimate, and the
/// per-observation decay that lets the estimate follow a change of content.
const PRIOR_WEIGHT: f64 = 2.0;
const PRIOR_RATE: f64 = 0.85;
const DECAY: f64 = 0.92;

/// Wall time (ms) of one round that verifies `window` rows: the drafter, the
/// target forward and the rollback. Measured on the M5 Max with the 27B MXFP4
/// pack (pinned windows, 7 prompts): 40.0 ms at four rows, 50.2 at six and 60.3
/// at eight. The fused matmul makes up to four rows cost the same; each row
/// after that adds about 5 ms.
fn round_cost_ms(window: usize) -> f64 {
    40.0 + 5.07 * window.saturating_sub(4) as f64
}

/// Per-position acceptance of one request's drafts, as decayed counts.
///
/// Position `j` is reached when the `j` drafts before it were accepted, so a
/// round that accepted `a` drafts of a window with `n` drafts observed
/// positions `0..=min(a, n - 1)`.
#[derive(Clone, Debug)]
struct WindowStats {
    seen: [f64; 8],
    hit: [f64; 8],
}

impl WindowStats {
    fn new() -> Self {
        Self {
            seen: [0.0; 8],
            hit: [0.0; 8],
        }
    }

    fn record(&mut self, verified_drafts: usize, accepted: usize) {
        for position in 0..verified_drafts.min(accepted + 1).min(self.seen.len()) {
            self.seen[position] = self.seen[position] * DECAY + 1.0;
            self.hit[position] =
                self.hit[position] * DECAY + if position < accepted { 1.0 } else { 0.0 };
        }
    }

    /// Smoothed acceptance rate per position. Each position is shrunk towards
    /// the rate of the position before it (the first towards `PRIOR_RATE`), so
    /// a position the request has not reached yet carries the last observed
    /// rate: drafts that keep being accepted earn the wider window, drafts that
    /// are not never do.
    fn rates(&self) -> [f64; 8] {
        let mut rates = [0.0; 8];
        let mut prior = PRIOR_RATE;
        for (position, rate) in rates.iter_mut().enumerate() {
            *rate =
                (self.hit[position] + prior * PRIOR_WEIGHT) / (self.seen[position] + PRIOR_WEIGHT);
            prior = *rate;
        }
        rates
    }

    /// Expected tokens committed by a round that verifies `window` rows: the
    /// anchor's successor plus each draft that survives every earlier one.
    fn expected_tokens(&self, window: usize) -> f64 {
        let rates = self.rates();
        let mut survival = 1.0;
        let mut tokens = 1.0;
        for rate in rates.iter().take(window.saturating_sub(1)) {
            survival *= rate;
            tokens += survival;
        }
        tokens
    }

    /// The window in `MIN_WINDOW..=max_window` with the best expected tokens
    /// per millisecond; the smaller window wins a tie.
    fn best_window(&self, max_window: usize) -> usize {
        let mut best = MIN_WINDOW.min(max_window);
        let mut best_rate = 0.0;
        for window in MIN_WINDOW.min(max_window)..=max_window {
            let rate = self.expected_tokens(window) / round_cost_ms(window);
            if rate > best_rate * 1.005 {
                best = window;
                best_rate = rate;
            }
        }
        best
    }
}

/// Logs why a step left the DFlash2 route and hands the step to the generic
/// routes.
fn decline(reason: &'static str) -> Option<Vec<u32>> {
    tracing::debug!(target: "ax_engine_mlx::dflash2", reason, "DFlash2 step declined");
    None
}

/// The linear verify fence already rebound the request cache. A failed read
/// cannot restore that cache, so this request fails instead of declining.
fn fail_linear_verify_fence(state: &mut RequestState, error: String) -> Option<Vec<u32>> {
    tracing::error!(
        target: "ax_engine_mlx::dflash2",
        %error,
        "DFlash2 linear verify fence failed; failing this request"
    );
    state.dflash2_verify_fence_error = Some(format!("DFlash2 linear verify fence failed: {error}"));
    None
}

/// Records the first MLX eval failure of a speculative step and logs it once.
fn note_mlx_eval_failure(slot: &RefCell<Option<String>>, error: &str) {
    if slot.borrow().is_some() {
        return;
    }
    tracing::warn!(
        target: "ax_engine_mlx::dflash2",
        error,
        "DFlash2 speculative evaluation failed; declining speculation"
    );
    *slot.borrow_mut() = Some(error.to_owned());
}

/// The greedy speculative walk: `drafts[i]` is checked against `predicted[i]`,
/// the target's choice after row `i` of the verify window. Returns how many
/// drafts were accepted and the committed tokens: the accepted drafts followed
/// by the target's own token at the first mismatch (or after the last draft).
pub(crate) fn greedy_walk(drafts: &[u32], predicted: &[u32]) -> (usize, Vec<u32>) {
    debug_assert_eq!(predicted.len(), drafts.len() + 1);
    let accepted = drafts
        .iter()
        .zip(predicted)
        .take_while(|(draft, target)| draft == target)
        .count();
    let mut tokens = drafts[..accepted].to_vec();
    tokens.push(predicted[accepted]);
    (accepted, tokens)
}

/// Cut a linear verify window the way the tree path does, before anything is
/// committed. `keep` is the anchor plus the emitted drafts, so the cache,
/// captured rows, and pending hidden stop at the tokens the caller emits.
fn linear_window_commit(
    new_tokens: &[u32],
    accepted: usize,
    budget: usize,
    eos: &[u32],
) -> (Vec<u32>, usize) {
    let (emitted, emitted_drafts) =
        super::tree_verify::limit_commit(new_tokens, accepted, budget, eos);
    let keep = 1 + emitted_drafts.min(accepted);
    (emitted, keep)
}

/// Draft rows that survive the EOS and output-budget cut. Acceptance telemetry
/// counts these, not the pre-cut greedy match, so it never credits a draft
/// past the cut. The emitted count is one higher (the target's bonus token).
fn committed_draft_count(pre_cut_accepted: usize, keep: usize) -> usize {
    keep.saturating_sub(1).min(pre_cut_accepted)
}

/// How the draft ids are read before verify.
#[derive(Clone, Copy)]
enum Dflash2DraftRead {
    /// Linear verify. The draft stays lazy so one later eval runs it with the
    /// target forward.
    Lazy,
    /// Tree verify. Runner-up ids are captured and the draft ids are read
    /// before any branch forward.
    Tree,
    /// Test-only: read the draft ids without runner-up capture.
    #[cfg(test)]
    EvalOnly,
}

/// Records the drafts that were actually committed, after the EOS and budget cut.
fn record_dflash2_committed_drafts(
    telemetry: &mut MtpTelemetry,
    drafted: usize,
    committed_drafts: usize,
) {
    telemetry.record_step(
        drafted,
        committed_drafts,
        &vec![MtpDraftSource::Mtp; drafted],
        None,
        committed_drafts,
    );
}

/// Drafts one window. `pending_hidden` stays in place until the caller commits
/// a verified prefix. A failed capture or draft read restores the drafter cursor.
fn dflash2_draft_window(
    dflash: &mut Dflash2RequestState,
    primary: u32,
    target: &ModelWeights,
    block: i32,
    window: usize,
    read: Dflash2DraftRead,
) -> Result<(MlxArray, crate::dflash2::DrafterCursor), &'static str> {
    let Some(hidden) = dflash.pending_hidden.clone() else {
        return Err("no pending hidden rows");
    };
    let cursor = dflash.drafter.snapshot_cursor();
    let capture_alternatives = matches!(read, Dflash2DraftRead::Tree);
    let drafted = if capture_alternatives {
        match dflash
            .drafter
            .draft_block_capturing_alternatives(primary, &hidden, target, block)
        {
            Ok(tokens) => tokens,
            // The drafter restored its own cursor. Restore again from the
            // pre-call snapshot so a partial runner-up write cannot stick.
            Err(_) => {
                dflash.drafter.restore_cursor(cursor);
                return Err("runner-up capture failed");
            }
        }
    } else {
        match dflash.drafter.draft_block(primary, &hidden, target, block) {
            Ok(tokens) => tokens,
            // Opt-in stage timing is the only read on this path. The drafter
            // restored its cursor; restore the pre-call snapshot too.
            // `pending_hidden` was cloned, not taken.
            Err(_) => {
                dflash.drafter.restore_cursor(cursor);
                return Err("linear draft evaluation failed");
            }
        }
    };
    let draft_arr = reshape(
        &slice(&drafted, &[0, 0], &[1, window as i32 - 1], &[1, 1], None),
        &[window as i32 - 1],
        None,
    );
    if !matches!(read, Dflash2DraftRead::Lazy) {
        // Fallible read of the draft ids. The request cache has not been written.
        // The drafter cursor has, so put it back when the read fails.
        if let Some(error) = crate::dflash2::take_dflash2_capture_verify_eval_error() {
            tracing::warn!(
                target: "ax_engine_mlx::dflash2",
                %error,
                "DFlash2 draft evaluation failed; declining speculation"
            );
            dflash.drafter.restore_cursor(cursor);
            return Err("tree verify evaluation failed");
        }
        if let Err(error) = mlx_sys::try_eval(&[&draft_arr]) {
            tracing::warn!(
                target: "ax_engine_mlx::dflash2",
                %error,
                "DFlash2 draft evaluation failed; declining speculation"
            );
            dflash.drafter.restore_cursor(cursor);
            return Err("tree verify evaluation failed");
        }
    }
    Ok((draft_arr, cursor))
}

/// Materialize a linear-verify cache clone. The request cache is not this clone.
fn try_linear_cache_eval(refs: &[&MlxArray]) -> Result<(), String> {
    if let Some(error) = crate::dflash2::take_dflash2_capture_verify_eval_error() {
        tracing::warn!(
            target: "ax_engine_mlx::dflash2",
            %error,
            "DFlash2 linear verify evaluation failed; declining speculation"
        );
        return Err(error);
    }
    if let Err(error) = mlx_sys::try_eval(refs) {
        tracing::warn!(
            target: "ax_engine_mlx::dflash2",
            %error,
            "DFlash2 linear verify evaluation failed; declining speculation"
        );
        return Err(error);
    }
    Ok(())
}

/// A draft was produced, then the verify outputs had the wrong shape.
/// Disable DFlash2 for the request and rewind the drafter cursor.
fn disable_dflash2_shape_mismatch(
    dflash: &mut Dflash2RequestState,
    cursor: crate::dflash2::DrafterCursor,
) {
    dflash.drafter.restore_cursor(cursor);
    dflash.disabled = true;
}

impl RequestState {
    /// False once DFlash2 gave up on this request.
    pub(super) fn dflash2_prefill_open(&self) -> bool {
        self.dflash2.as_ref().is_none_or(|state| !state.disabled)
    }
}

impl MlxRunner {
    pub(super) fn dflash2_prompt_ready(&self, state: &RequestState) -> bool {
        state.dflash2.as_ref().is_some_and(|dflash| {
            !dflash.disabled
                && !dflash.started
                && dflash.pending_hidden.is_some()
                && dflash.captured_rows == state.cache.seq_len()
        })
    }

    /// Capture at the exact trunk boundary, before later prompt forwards move it.
    pub(super) fn capture_dflash2_prefix(&self, state: &mut RequestState) {
        if !layer_capture::active() {
            return;
        }
        self.dflash2_finish_prefill_capture(state);
        if !self.prefix_cache.lock().enabled() {
            return;
        }
        let Some((dflash, weights)) = state.dflash2.as_ref().zip(self.dflash2.as_deref()) else {
            return;
        };
        if dflash.captured_rows != state.cache.seq_len() {
            return;
        }
        let hidden_bytes = dflash
            .pending_hidden
            .as_ref()
            .map_or(0, |hidden| hidden.nbytes() as u64);
        if hidden_bytes > self.prefix_cache.lock().policy.max_bytes {
            return;
        }
        if let Some(snapshot) =
            Dflash2PrefixSnapshot::capture(dflash, weights, self.cfg.compile_cache_identity)
        {
            state
                .dflash2_prefix_snapshots
                .insert(snapshot.offset, Arc::new(snapshot));
        }
    }

    pub(super) fn restore_dflash2_prefix(
        &self,
        state: &mut RequestState,
        snapshot: Option<&Dflash2PrefixSnapshot>,
    ) {
        state.dflash2 = snapshot
            .zip(self.dflash2.as_ref())
            .and_then(|(snapshot, weights)| {
                snapshot.restore(
                    weights.clone(),
                    self.cfg.compile_cache_identity,
                    state.cache.seq_len(),
                )
            });
    }

    /// Whether this request may use DFlash2: greedy, no logits processors, text
    /// only, and a drafter is loaded.
    pub(super) fn dflash2_request_eligible(
        &self,
        sampling: MlxSamplingParams,
        is_greedy: bool,
        has_media: bool,
    ) -> bool {
        self.dflash2.is_some()
            && !has_media
            && is_greedy
            && sampling.temperature <= 0.0
            && !sampling.uses_logits_processors()
            && !sampling.json_grammar_active
            && self.cfg.linear_attention.is_some()
    }

    /// Layers whose outputs the drafter conditions on.
    pub(super) fn dflash2_capture_layers(&self) -> Option<Vec<usize>> {
        self.dflash2
            .as_ref()
            .map(|weights| weights.config().target_layer_ids.clone())
    }

    /// Starts the accumulating capture for one prefill item.
    pub(super) fn dflash2_begin_prefill_capture(&self) -> Option<layer_capture::LayerCaptureGuard> {
        let layers = self.dflash2_capture_layers()?;
        Some(layer_capture::capture_layers_accumulating(
            &layers,
            self.dflash2_window_rows(),
        ))
    }

    pub(super) fn dflash2_window_rows(&self) -> i32 {
        self.dflash2
            .as_ref()
            .map_or(0, |weights| weights.config().sliding_window - 1)
    }

    /// Called at the end of a prefill item run under an accumulating capture
    /// scope: stores the captured rows, or disables DFlash2 for the request when
    /// they do not account for exactly the tokens in the cache.
    pub(super) fn dflash2_finish_prefill_capture(&self, state: &mut RequestState) {
        let Some(weights) = self.dflash2.as_ref() else {
            return;
        };
        let captured = layer_capture::take_accumulated();
        let cache_rows = state.cache.seq_len();
        let dflash = state
            .dflash2
            .get_or_insert_with(|| Dflash2RequestState::new(weights.clone()));
        let Some((tails, item_rows)) = captured else {
            // No forward ran under the scope: only a cache that did not move
            // (an empty quantum) leaves the request intact.
            if cache_rows != dflash.captured_rows {
                tracing::debug!(
                    target: "ax_engine_mlx::dflash2",
                    cache_rows,
                    captured = dflash.captured_rows,
                    "DFlash2 prefill item captured no rows for a cache that moved; disabled"
                );
                dflash.disabled = true;
            }
            return;
        };
        let item_rows = item_rows as usize;
        if item_rows == cache_rows {
            // The item covered the whole cache: a first item, or a recompute.
            dflash.reset();
        } else if dflash.disabled || dflash.captured_rows + item_rows != cache_rows {
            tracing::debug!(
                target: "ax_engine_mlx::dflash2",
                item_rows,
                cache_rows,
                captured = dflash.captured_rows,
                "DFlash2 prefill rows do not account for the cache; disabled"
            );
            dflash.disabled = true;
            return;
        }
        let refs: Vec<&MlxArray> = tails.iter().collect();
        let joined = astype(&concatenate(&refs, 2, None), MlxDtype::Bfloat16, None);
        dflash.append_prompt_rows(joined, item_rows, self.dflash2_window_rows());
    }

    /// One DFlash2 round, or `None` when this step must take another route.
    pub(super) fn try_dflash2_decode(
        &self,
        state: &mut RequestState,
        input_tokens: &[u32],
        sampling: MlxSamplingParams,
        is_greedy: bool,
        options: &DecodeOneOptions<'_>,
    ) -> Option<Vec<u32>> {
        let weights = self.dflash2.as_ref()?;
        let Some(ctx) = options.request_context else {
            return decline("no request context");
        };
        if !self.dflash2_request_eligible(sampling, is_greedy, false) {
            return decline("request not eligible");
        }
        if state.think_soft_close_armed || state.pending_direct.is_some() {
            return decline("think close armed or direct token pending");
        }
        if !(crate::fastpath::mtp_relaxed_target_verify_enabled()
            && crate::fastpath::mtp_linear_projected_replay_enabled())
        {
            return decline("relaxed projected-replay verify is off");
        }
        let cfg = weights.config();
        let block = cfg.block_size as usize;
        let remaining = ctx.max_output_tokens.saturating_sub(ctx.generated_len) as usize;
        // The last tokens of the budget decode directly: a round commits several.
        if remaining < 2 {
            return decline("output budget exhausted");
        }
        let think_budget_close_due = think_budget_close_decision(
            state.ngram_in_think,
            self.cfg.think_end_token_id,
            remaining as u32,
            state.think_emitted_tokens,
            ctx.max_think_tokens,
            ctx.answer_reserve_tokens,
            false,
        );
        if think_budget_close_due {
            return decline("think budget close due");
        }
        let Some(primary) = state
            .next_model_last_token
            .or_else(|| input_tokens.last().copied())
        else {
            return decline("no anchor token");
        };
        let token_offset = state.cache.seq_len();
        match state.dflash2.as_ref() {
            None => return decline("no captured prompt state"),
            Some(dflash) if dflash.disabled => return decline("disabled for this request"),
            Some(dflash) if dflash.pending_hidden.is_none() => {
                return decline("no pending hidden rows");
            }
            Some(dflash) if dflash.captured_rows != token_offset => {
                tracing::debug!(
                    target: "ax_engine_mlx::dflash2",
                    captured = dflash.captured_rows,
                    cache = token_offset,
                    "DFlash2 captured rows disagree with the cache"
                );
                return decline("captured rows disagree with the cache");
            }
            Some(_) => {}
        }

        // How many rows to verify: pinned, or the policy's pick for this request.
        let window = match crate::fastpath::dflash2_window_override() {
            Some(rows) => rows.clamp(MIN_WINDOW, block),
            None => state
                .dflash2
                .as_ref()
                .map_or(block, |dflash| dflash.window_stats.best_window(block)),
        };
        let verify_len = window;

        // --- Draft (lazy) ---------------------------------------------------
        // `pending_hidden` is cloned, not taken. A failed capture or verify
        // restores the drafter cursor and leaves the hidden rows for the next
        // round. Only a committed prefix replaces them.
        let draft_started = Instant::now();
        let tree = crate::fastpath::tree_verify_enabled();
        let (draft_arr, cursor) = {
            let dflash = state.dflash2.as_mut()?;
            if !dflash.started {
                dflash.drafter.seed_offset(dflash.skipped_rows as i32);
                dflash.started = true;
            }
            let read = if tree {
                Dflash2DraftRead::Tree
            } else {
                Dflash2DraftRead::Lazy
            };
            match dflash2_draft_window(dflash, primary, &self.weights, cfg.block_size, window, read)
            {
                Ok(drafted) => drafted,
                Err(reason) => return decline(reason),
            }
        };
        let draft_wall_us = elapsed_us(draft_started);

        // --- Verify (one target forward over [anchor, drafts] with hidden capture)
        // The verify graph chains on the lazy draft tokens, so building it
        // overlaps the drafter's GPU work and one eval batch runs both.
        let anchor_data = [primary];
        let anchor_arr = MlxArray::from_raw_data(
            anchor_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
        let ids_1d = concatenate(&[&anchor_arr, &draft_arr], 0, None);

        let _target_verify_fast_scope = crate::fastpath::scoped_qwen_linear_mtp_target_verify(true);
        let _target_verify_scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        let _verify_qmm_scope = crate::model::shared::verify_qmm::QwenMtpVerifyQmmGuard::arm(true);

        if crate::fastpath::tree_verify_enabled() {
            match self.try_dflash2_tree_verify(
                state,
                primary,
                &draft_arr,
                token_offset,
                options,
                draft_wall_us,
            ) {
                Ok(Some(tokens)) => return Some(tokens),
                Ok(None) => {}
                // Logged once at the eval site. Request KV was not written.
                // The drafter cursor and pending hidden go back to the pre-round
                // values so the next round can draft again.
                Err(()) => {
                    if let Some(dflash) = state.dflash2.as_mut() {
                        dflash.drafter.restore_cursor(cursor);
                    }
                    return decline("tree verify evaluation failed");
                }
            }
        }

        // Linear-attention state cannot be trimmed: verify on a clone that
        // records the prefix tape, then adopt, restore or replay it.
        let clone_started = Instant::now();
        let mut verify_cache = state.cache.clone();
        verify_cache.begin_linear_prefix_capture(1);
        let clone_wall_us = elapsed_us(clone_started);

        let verify_started = Instant::now();
        let capture = layer_capture::capture_layers(&cfg.target_layer_ids);
        let (logits_all, _post_norm) = crate::model::forward_all_positions_with_post_norm_ids(
            &self.cfg,
            &self.weights,
            &ids_1d,
            verify_len,
            &mut verify_cache,
            token_offset,
            false,
        );
        let captured = layer_capture::take();
        drop(capture);
        verify_cache.advance(verify_len);
        let Some(captured) = captured else {
            // The layers ran outside `layer_forward`: nothing was captured, so
            // the drafter cannot continue. Decode normally from here.
            if let Some(dflash) = state.dflash2.as_mut() {
                dflash.disabled = true;
            }
            return None;
        };
        if crate::fastpath::mtp_rebind_verify_fa_enabled() {
            // The verifier's K/V output backs every acceptance outcome; rebind
            // the rollback source so its obsolete handles cannot block MLX
            // donation at the evaluation fence below.
            let _ = state
                .cache
                .rebind_contiguous_attention_storage_from(&verify_cache);
        }
        let predicted_arr = argmax(&logits_all, None);
        let refs: Vec<&MlxArray> = captured.iter().collect();
        let hidden_all = astype(&concatenate(&refs, 2, None), MlxDtype::Bfloat16, None);
        // Rebind already published the verifier K/V into the request cache.
        // This read must not retain the pre-rebind handles: that blocks MLX
        // donation. A failure leaves the cache untrustworthy, so fail the
        // request instead of restoring it.
        if let Some(error) = crate::dflash2::take_dflash2_capture_verify_eval_error() {
            return fail_linear_verify_fence(state, error);
        }
        if let Err(error) = mlx_sys::try_eval(&[&predicted_arr, &hidden_all, &draft_arr]) {
            return fail_linear_verify_fence(state, error);
        }
        let verify_wall_us = elapsed_us(verify_started);
        let predicted: Vec<u32> = predicted_arr.data_u32().to_vec();
        let drafts_owned: Vec<u32> = draft_arr.data_u32().to_vec();
        let drafts: &[u32] = &drafts_owned;
        if predicted.len() != verify_len || drafts.len() + 1 != verify_len {
            // Cannot happen for a well-formed verify. Rewind the drafter and
            // leave DFlash2. Full-attention rebind, when it ran, stays.
            if let Some(dflash) = state.dflash2.as_mut() {
                disable_dflash2_shape_mismatch(dflash, cursor);
            }
            return None;
        }

        // --- Accept and commit ---------------------------------------------
        // Cut at EOS and the output budget before commit. Adopting the full
        // window would leave KV, GDN state, and captured rows for drafts the
        // caller never emits. `keep == verify_len` is the only full adopt.
        let (accepted, new_tokens) = greedy_walk(drafts, &predicted);
        let (emitted, keep) =
            linear_window_commit(&new_tokens, accepted, remaining, options.terminal_token_ids);
        let rollback_started = Instant::now();
        let mut cache_eval_failed = false;
        let adopted = if keep == verify_len {
            if !crate::fastpath::mtp_lazy_adopt_state_enabled() {
                let refs = verify_cache.collect_eval_refs();
                if try_linear_cache_eval(&refs).is_err() {
                    cache_eval_failed = true;
                    false
                } else {
                    verify_cache.trim_to(token_offset + keep)
                }
            } else {
                verify_cache.trim_to(token_offset + keep)
            }
        } else if keep == 1 {
            // The capture point is the committed anchor, so a complete miss
            // restores it; a tape-based verifier replays one row instead.
            (verify_cache.restore_linear_prefix_checkpoint()
                || self.dflash2_replay_prefix(&state.cache, &mut verify_cache, keep))
                && verify_cache.trim_to(token_offset + keep)
        } else {
            // The verifier produced every linear layer's inputs: rebuild the
            // gated-delta state of the emitted prefix, then trim the K/V.
            let replayed = self.dflash2_replay_prefix(&state.cache, &mut verify_cache, keep);
            if replayed && !crate::fastpath::mtp_lazy_adopt_state_enabled() {
                let refs = verify_cache.collect_eval_refs();
                if try_linear_cache_eval(&refs).is_err() {
                    cache_eval_failed = true;
                    false
                } else {
                    verify_cache.trim_to(token_offset + keep)
                }
            } else {
                replayed && verify_cache.trim_to(token_offset + keep)
            }
        };
        if cache_eval_failed {
            // Rebind already dropped the superseded K/V so the fence could
            // donate them. Logical length is still the pre-round length.
            if let Some(dflash) = state.dflash2.as_mut() {
                dflash.drafter.restore_cursor(cursor);
            }
            return decline("linear verify evaluation failed");
        }
        if adopted {
            verify_cache.clear_linear_prefix_checkpoint();
            state.cache = verify_cache;
        } else {
            // Rebuild the emitted prefix one token at a time on the
            // untouched cache. Drafts past the budget or EOS stay out.
            let kept = keep.saturating_sub(1).min(accepted).min(drafts.len());
            let after = recompute_committed_prefix_with_argmax(
                &self.cfg,
                &self.weights,
                &mut state.cache,
                primary,
                &drafts[..kept],
                token_offset,
            );
            debug_assert_eq!(after, predicted[kept]);
        }
        let rollback_wall_us = elapsed_us(rollback_started);

        // The rows of the committed tokens condition the next round.
        let width = hidden_all.shape()[2];
        let committed_hidden = slice(
            &hidden_all,
            &[0, 0, 0],
            &[1, keep as i32, width],
            &[1, 1, 1],
            None,
        );
        let committed_drafts = committed_draft_count(accepted, keep);
        if let Some(dflash) = state.dflash2.as_mut() {
            dflash.pending_hidden = Some(committed_hidden);
            dflash.captured_rows = token_offset + keep;
            dflash.rounds += 1;
            dflash.drafted += drafts.len() as u32;
            dflash.accepted += committed_drafts as u32;
            dflash.window_stats.record(drafts.len(), committed_drafts);
        }
        // A draft made before this route took over (the MTP head's first
        // proposal) no longer matches the position.
        state.mtp_pending_draft.clear();
        state.mtp_pending_draft_lazy = None;
        state.mtp_pending_draft_sources.clear();
        record_dflash2_committed_drafts(&mut state.mtp_telemetry, drafts.len(), committed_drafts);
        state.mtp_telemetry.record_timings(MtpStepTimings {
            cache_clone_wall_us: clone_wall_us,
            verify_forward_wall_us: verify_wall_us,
            rollback_wall_us,
            draft_wall_us,
            mtp_draft_wall_us: draft_wall_us,
            verify_tokens: saturating_u32(verify_len),
            emitted_tokens: saturating_u32(emitted.len()),
            ..MtpStepTimings::default()
        });
        // The budget and terminal cuts already happened. This latches the
        // last emitted token for the next step.
        Some(apply_decode_result(
            state,
            &emitted,
            options.terminal_token_ids,
        ))
    }

    /// Rebuilds every linear-attention layer's state for the first `keep` rows of
    /// the verify window from the tape the verify forward recorded.
    fn dflash2_replay_prefix(
        &self,
        source: &MlxKVCache,
        verify_cache: &mut MlxKVCache,
        keep: usize,
    ) -> bool {
        self.weights
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| layer.linear_attn.is_some())
            .all(|(layer_idx, layer)| {
                replay_linear_attention_mtp_prefix(
                    &self.cfg,
                    layer,
                    source,
                    verify_cache,
                    layer_idx,
                    keep,
                )
            })
    }

    /// Greedy tree verify for one request. Each branch clones `state.cache`
    /// (KV, linear/GDN, and QSA/PLE state) and runs the same singleton forward
    /// as the linear path. Only the winning branch is written back.
    fn try_dflash2_tree_verify(
        &self,
        state: &mut RequestState,
        anchor: u32,
        draft_arr: &MlxArray,
        token_offset: usize,
        options: &DecodeOneOptions<'_>,
        draft_wall_us: u32,
    ) -> Result<Option<Vec<u32>>, ()> {
        use super::tree_verify::{CommitLimits, DraftTree, verify_draft_tree};

        let remaining = options.request_context.map_or(0, |ctx| {
            ctx.max_output_tokens.saturating_sub(ctx.generated_len) as usize
        });
        let Some(capture_layers) = self
            .dflash2
            .as_ref()
            .map(|weights| weights.config().target_layer_ids.clone())
        else {
            return Ok(None);
        };
        // Draft ids were read in `dflash2_draft_window` before this call.
        // The request cache is still the pre-round snapshot.
        let drafted = draft_arr.data_u32();
        if drafted.is_empty() {
            return Ok(None);
        }
        let primary_len = drafted.len().min(2);
        let primary = drafted[..primary_len].to_vec();
        let Some(alternatives) = state
            .dflash2
            .as_ref()
            .map(|dflash| dflash.drafter.tree_alternatives)
        else {
            return Ok(None);
        };
        let Ok(tree) = DraftTree::from_top2(&primary, &alternatives) else {
            return Ok(None);
        };
        let hiddens: RefCell<Vec<Option<MlxArray>>> = RefCell::new(Vec::new());
        let eval_error: RefCell<Option<String>> = RefCell::new(None);
        let verify_started = Instant::now();
        let decision = {
            let base = &state.cache;
            verify_draft_tree(
                &tree,
                base,
                CommitLimits {
                    budget: remaining,
                    eos: options.terminal_token_ids,
                },
                |cache, branch| {
                    if eval_error.borrow().is_some() {
                        return super::tree_verify::BranchVerification {
                            accepted_drafts: 0,
                            committed: Vec::new(),
                            state: cache,
                            usable: false,
                        };
                    }
                    let (verification, hidden) = self.forward_tree_branch(
                        cache,
                        anchor,
                        branch,
                        token_offset,
                        &capture_layers,
                        CommitLimits {
                            budget: remaining,
                            eos: options.terminal_token_ids,
                        },
                        &eval_error,
                    );
                    hiddens.borrow_mut().push(hidden);
                    verification
                },
            )
        };
        // A branch clone failed to evaluate. Do not adopt any branch.
        if eval_error.borrow().is_some() {
            return Err(());
        }
        let Ok(decision) = decision else {
            return Ok(None);
        };
        if decision.tokens.is_empty() {
            return Ok(None);
        }
        let Some(hidden) = hiddens
            .into_inner()
            .get(decision.branch_index)
            .and_then(Clone::clone)
        else {
            return Ok(None);
        };
        let verify_wall_us = elapsed_us(verify_started);
        state.cache = decision.state;
        if let Some(dflash) = state.dflash2.as_mut() {
            dflash.pending_hidden = Some(hidden);
            dflash.captured_rows = token_offset + 1 + decision.cache_draft_rows;
            dflash.rounds = dflash.rounds.saturating_add(1);
            dflash.drafted = dflash
                .drafted
                .saturating_add(saturating_u32(decision.draft_len));
            // `cache_draft_rows` is the post-cut draft count. `accepted_drafts`
            // is the pre-cut greedy match and can include tokens never emitted.
            let committed_drafts = decision.cache_draft_rows;
            dflash.accepted = dflash
                .accepted
                .saturating_add(saturating_u32(committed_drafts));
            dflash
                .window_stats
                .record(decision.draft_len, committed_drafts);
            dflash.tree_nodes_verified = saturating_u32(decision.nodes_verified);
            dflash.tree_accepted_length = saturating_u32(committed_drafts);
            dflash.tree_branch_chosen = saturating_u32(decision.branch_index);
        }
        // `accepted_length` is the post-cut draft count, matching
        // `tree_accepted_length` and telemetry. `greedy_accepted` is the
        // pre-cut match on the chosen branch.
        tracing::debug!(
            target: "ax_engine_mlx::dflash2",
            tree_nodes_verified = decision.nodes_verified,
            accepted_length = decision.cache_draft_rows,
            greedy_accepted = decision.accepted_drafts,
            branch_chosen = decision.branch_index,
            "DFlash2 tree verify round"
        );
        state.mtp_pending_draft.clear();
        state.mtp_pending_draft_lazy = None;
        state.mtp_pending_draft_sources.clear();
        // `cache_draft_rows` is the post-cut draft count. `accepted_drafts` is
        // the pre-cut greedy match and can include tokens never emitted.
        record_dflash2_committed_drafts(
            &mut state.mtp_telemetry,
            decision.draft_len,
            decision.cache_draft_rows,
        );
        state.mtp_telemetry.record_timings(MtpStepTimings {
            verify_forward_wall_us: verify_wall_us,
            draft_wall_us,
            mtp_draft_wall_us: draft_wall_us,
            verify_tokens: saturating_u32(decision.nodes_verified),
            emitted_tokens: saturating_u32(decision.tokens.len()),
            ..MtpStepTimings::default()
        });
        Ok(Some(apply_decode_result(
            state,
            &decision.tokens,
            options.terminal_token_ids,
        )))
    }

    /// One linear greedy verify of `drafts` on a clone of `source`. The clone
    /// is the only cache this branch writes.
    #[allow(clippy::too_many_arguments)]
    fn forward_tree_branch(
        &self,
        source: MlxKVCache,
        anchor: u32,
        drafts: &[u32],
        token_offset: usize,
        capture_layers: &[usize],
        limits: super::tree_verify::CommitLimits<'_>,
        eval_error: &RefCell<Option<String>>,
    ) -> (
        super::tree_verify::BranchVerification<MlxKVCache>,
        Option<MlxArray>,
    ) {
        use super::tree_verify::{BranchVerification, limit_commit};

        let failed = |cache: MlxKVCache| {
            (
                BranchVerification {
                    accepted_drafts: 0,
                    committed: Vec::new(),
                    state: cache,
                    usable: false,
                },
                None,
            )
        };
        if drafts.is_empty() {
            return failed(source);
        }
        let mut ids = Vec::with_capacity(drafts.len() + 1);
        ids.push(anchor);
        ids.extend_from_slice(drafts);
        let verify_len = ids.len();
        let Ok(verify_len_i) = i32::try_from(verify_len) else {
            return failed(source);
        };
        let ids_1d = MlxArray::from_raw_data(
            ids.as_ptr().cast::<u8>(),
            ids.len() * std::mem::size_of::<u32>(),
            &[verify_len_i],
            MlxDtype::Uint32,
        );
        let mut verify_cache = source.clone();
        verify_cache.begin_linear_prefix_capture(1);
        let capture = layer_capture::capture_layers(capture_layers);
        let (logits_all, _post_norm) = crate::model::forward_all_positions_with_post_norm_ids(
            &self.cfg,
            &self.weights,
            &ids_1d,
            verify_len,
            &mut verify_cache,
            token_offset,
            false,
        );
        let captured = layer_capture::take();
        drop(capture);
        verify_cache.advance(verify_len);
        let Some(captured) = captured else {
            return failed(source);
        };
        let predicted_arr = argmax(&logits_all, None);
        let refs: Vec<&MlxArray> = captured.iter().collect();
        let hidden_all = astype(&concatenate(&refs, 2, None), MlxDtype::Bfloat16, None);
        // `verify_cache` is a clone. The request cache is written only if a
        // branch is adopted after every branch returns.
        if let Some(error) = crate::dflash2::take_dflash2_capture_verify_eval_error() {
            note_mlx_eval_failure(eval_error, &error);
            return failed(source);
        }
        if let Err(error) = mlx_sys::try_eval(&[&predicted_arr, &hidden_all]) {
            note_mlx_eval_failure(eval_error, &error);
            return failed(source);
        }
        let predicted = predicted_arr.data_u32().to_vec();
        if predicted.len() != verify_len {
            return failed(source);
        }
        let (accepted, committed) = greedy_walk(drafts, &predicted);
        let (_emitted, cache_draft_rows) =
            limit_commit(&committed, accepted, limits.budget, limits.eos);
        let retained = 1 + cache_draft_rows;
        let shape = hidden_all.shape();
        let (Some(&rows), Some(&width)) = (shape.get(1), shape.get(2)) else {
            return failed(source);
        };
        let Ok(retained_i) = i32::try_from(retained) else {
            return failed(source);
        };
        if width <= 0 || rows < retained_i {
            return failed(source);
        }
        let hidden = slice(
            &hidden_all,
            &[0, 0, 0],
            &[1, retained_i, width],
            &[1, 1, 1],
            None,
        );
        let (cache, eval_failed) = self.commit_tree_branch(
            source,
            verify_cache,
            anchor,
            drafts,
            cache_draft_rows,
            token_offset,
            eval_error,
        );
        if eval_failed {
            return failed(cache);
        }
        (
            BranchVerification {
                accepted_drafts: accepted,
                committed,
                state: cache,
                usable: true,
            },
            Some(hidden),
        )
    }

    /// Adopt, restore, or replay `source` so the cache keeps the anchor plus
    /// `cache_draft_rows` accepted drafts. `source` itself is not modified
    /// unless adoption fails and the committed prefix is recomputed on it.
    ///
    /// The bool is set when materializing the clone fails. The returned cache
    /// is then the untouched `source`, and the request cache is not this clone.
    #[allow(clippy::too_many_arguments)]
    fn commit_tree_branch(
        &self,
        source: MlxKVCache,
        mut verify_cache: MlxKVCache,
        anchor: u32,
        drafts: &[u32],
        cache_draft_rows: usize,
        token_offset: usize,
        eval_error: &RefCell<Option<String>>,
    ) -> (MlxKVCache, bool) {
        let retained = 1 + cache_draft_rows;
        let verify_len = drafts.len() + 1;
        let adopted = if retained == verify_len {
            if !crate::fastpath::mtp_lazy_adopt_state_enabled() {
                let refs = verify_cache.collect_eval_refs();
                // Full-window adopt still holds only the branch clone.
                if let Some(error) = crate::dflash2::take_dflash2_capture_verify_eval_error() {
                    note_mlx_eval_failure(eval_error, &error);
                    return (source, true);
                }
                if let Err(error) = mlx_sys::try_eval(&refs) {
                    note_mlx_eval_failure(eval_error, &error);
                    return (source, true);
                }
            }
            verify_cache.trim_to(token_offset + retained)
        } else if retained == 1 {
            (verify_cache.restore_linear_prefix_checkpoint()
                || self.dflash2_replay_prefix(&source, &mut verify_cache, 1))
                && verify_cache.trim_to(token_offset + 1)
        } else {
            let replayed = self.dflash2_replay_prefix(&source, &mut verify_cache, retained);
            if replayed && !crate::fastpath::mtp_lazy_adopt_state_enabled() {
                let refs = verify_cache.collect_eval_refs();
                // Replay wrote this clone, not the request cache. Drop it on failure.
                if let Some(error) = crate::dflash2::take_dflash2_capture_verify_eval_error() {
                    note_mlx_eval_failure(eval_error, &error);
                    return (source, true);
                }
                if let Err(error) = mlx_sys::try_eval(&refs) {
                    note_mlx_eval_failure(eval_error, &error);
                    return (source, true);
                }
            }
            replayed && verify_cache.trim_to(token_offset + retained)
        };
        if adopted {
            verify_cache.clear_linear_prefix_checkpoint();
            (verify_cache, false)
        } else {
            let mut rebuilt = source;
            let kept = cache_draft_rows.min(drafts.len());
            let _after = recompute_committed_prefix_with_argmax(
                &self.cfg,
                &self.weights,
                &mut rebuilt,
                anchor,
                &drafts[..kept],
                token_offset,
            );
            (rebuilt, false)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn prompt_sidecar_round_trips_a_truncated_tail_and_rejects_wrong_identity() {
        let (weights, _) = crate::dflash2::dflash2_tiny_fixture();
        let width = weights.config().target_hidden_width();
        let rows = weights.config().sliding_window - 1;
        let data: Vec<f32> = (0..rows * width).map(|n| n as f32 / 16.0).collect();
        let hidden = astype(
            &reshape(&MlxArray::from_f32_slice(&data), &[1, rows, width], None),
            MlxDtype::Bfloat16,
            None,
        );
        let mut cold = Dflash2RequestState::new(weights.clone());
        cold.append_prompt_rows(hidden.clone(), 19, rows);
        assert_eq!(cold.skipped_rows, 19 - rows as usize);
        let sidecar = Dflash2PrefixSnapshot::capture(&cold, &weights, 42).unwrap();
        assert_eq!(sidecar.words.len() * 4, hidden.nbytes());
        let mut warm = sidecar.restore(weights.clone(), 42, 19).unwrap();
        assert_eq!(warm.skipped_rows, cold.skipped_rows);
        let restored = astype(
            warm.pending_hidden.as_ref().unwrap(),
            MlxDtype::Float32,
            None,
        );
        mlx_sys::eval(&[&restored]);
        assert_eq!(restored.data_f32(), data.as_slice());
        // A warm suffix must advance the same tail and absolute offset.
        let suffix = mlx_sys::zeros(&[1, 3, width], MlxDtype::Bfloat16, None);
        cold.append_prompt_rows(suffix.clone(), 3, rows);
        warm.append_prompt_rows(suffix, 3, rows);
        let a = Dflash2PrefixSnapshot::capture(&cold, &weights, 42).unwrap();
        let b = Dflash2PrefixSnapshot::capture(&warm, &weights, 42).unwrap();
        assert_eq!(a.words, b.words);
        assert_eq!(warm.captured_rows, 22);
        assert!(sidecar.restore(weights.clone(), 43, 19).is_none());
        assert!(sidecar.restore(weights.clone(), 42, 18).is_none());
        let mut invalid = sidecar.clone();
        invalid.drafter_identity.push_str("-changed");
        assert!(invalid.restore(weights.clone(), 42, 19).is_none());
        invalid = sidecar.clone();
        invalid.words = Arc::from(&invalid.words[..invalid.words.len() - 1]);
        assert!(invalid.restore(weights.clone(), 42, 19).is_none());
        cold.started = true;
        assert!(Dflash2PrefixSnapshot::capture(&cold, &weights, 42).is_none());
    }

    #[test]
    fn prompt_sidecar_is_charged_and_released_by_cache_eviction() {
        let (weights, _) = crate::dflash2::dflash2_tiny_fixture();
        let mut state = Dflash2RequestState::new(weights.clone());
        state.append_prompt_rows(
            mlx_sys::zeros(
                &[1, 4, weights.config().target_hidden_width()],
                MlxDtype::Bfloat16,
                None,
            ),
            4,
            7,
        );
        let sidecar = Arc::new(Dflash2PrefixSnapshot::capture(&state, &weights, 42).unwrap());
        let weak = Arc::downgrade(&sidecar);
        let bytes = sidecar.bytes();
        let snapshot =
            MlxPrefixSnapshot::from_shared_payload(Arc::from([0u8; 8]), vec![1; 4], 4, None, None)
                .with_dflash2(Some(sidecar));
        assert_eq!(snapshot.bytes, 8 + 16 + bytes);
        let key = MlxPrefixCacheKey {
            model_id: "test".into(),
            route_policy: "test".into(),
            layer_layout: "test".into(),
            block_size_tokens: 4,
            token_count: 4,
            token_hash: 1,
            media_key: String::new(),
        };
        let mut too_small = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: snapshot.bytes - 1,
            max_entries: 1,
        });
        assert!(!too_small.insert(key.clone(), snapshot.clone()).stored);
        let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: snapshot.bytes,
            max_entries: 1,
        });
        assert!(cache.insert(key.clone(), snapshot).stored);
        assert_eq!(Arc::strong_count(&weak.upgrade().unwrap()), 2);
        let mut other = key;
        other.token_hash = 2;
        assert!(
            cache
                .insert(
                    other,
                    MlxPrefixSnapshot::from_shared_payload(
                        Arc::from([0u8; 8]),
                        vec![2; 4],
                        4,
                        None,
                        None
                    )
                )
                .stored
        );
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn window_follows_the_acceptance_of_the_request() {
        // Drafts that are always right earn the full window.
        let mut code = WindowStats::new();
        for _ in 0..12 {
            code.record(7, 7);
        }
        assert_eq!(code.best_window(8), 8);
        // Drafts that are always wrong shrink it to the four rows a round costs
        // the same as three.
        let mut noise = WindowStats::new();
        for _ in 0..12 {
            noise.record(3, 0);
        }
        assert!(noise.best_window(8) <= 4, "{}", noise.best_window(8));
        // Prose-like acceptance (one or two drafts per round) stays narrow.
        let mut prose = WindowStats::new();
        for round in 0..40 {
            prose.record(prose.best_window(8) - 1, 1 + round % 2);
        }
        assert!(prose.best_window(8) <= 5, "{}", prose.best_window(8));
        // A request that turns into code widens again.
        for _ in 0..40 {
            let window = prose.best_window(8);
            prose.record(window - 1, window - 1);
        }
        assert_eq!(prose.best_window(8), 8);
    }

    #[test]
    fn unreached_positions_inherit_the_last_observed_rate() {
        // Only four-row windows were verified and every draft was accepted: the
        // positions beyond them are assumed to behave the same way.
        let mut stats = WindowStats::new();
        for _ in 0..6 {
            stats.record(3, 3);
        }
        let rates = stats.rates();
        assert!(rates[2] > 0.9);
        assert!((rates[5] - rates[2]).abs() < 1e-9);
        assert_eq!(stats.best_window(8), 8);
        // Mostly rejected drafts do not extrapolate into a wide window.
        let mut weak = WindowStats::new();
        for _ in 0..6 {
            weak.record(3, 1);
        }
        assert!(weak.best_window(8) <= 4);
    }

    #[test]
    fn only_reached_positions_are_observed() {
        let mut stats = WindowStats::new();
        // Two drafts accepted in a window of seven drafts: positions 0, 1 hit,
        // position 2 reached and missed, positions 3.. never reached.
        stats.record(7, 2);
        assert_eq!(stats.seen[..4], [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(stats.hit[..4], [1.0, 1.0, 0.0, 0.0]);
        // A window that verifies two drafts never reaches a third position.
        let mut narrow = WindowStats::new();
        narrow.record(2, 2);
        assert_eq!(narrow.seen[..3], [1.0, 1.0, 0.0]);
    }

    #[test]
    fn expected_tokens_grow_with_the_window() {
        let stats = WindowStats::new();
        let mut previous = 0.0;
        for window in MIN_WINDOW..=8 {
            let tokens = stats.expected_tokens(window);
            assert!(tokens > previous);
            previous = tokens;
        }
        assert!(round_cost_ms(8) > round_cost_ms(4));
        assert!((round_cost_ms(3) - round_cost_ms(4)).abs() < 1e-9);
    }

    #[test]
    fn walk_accepts_the_longest_matching_prefix_and_adds_the_target_token() {
        // All seven drafts match: seven accepted plus the bonus token.
        let drafts = [1, 2, 3, 4, 5, 6, 7];
        let predicted = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            greedy_walk(&drafts, &predicted),
            (7, vec![1, 2, 3, 4, 5, 6, 7, 8])
        );
        // The third draft differs: two accepted, then the target's own token.
        let predicted = [1, 2, 9, 4, 5, 6, 7, 8];
        assert_eq!(greedy_walk(&drafts, &predicted), (2, vec![1, 2, 9]));
        // The first draft differs: nothing accepted, one token committed.
        let predicted = [9, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(greedy_walk(&drafts, &predicted), (0, vec![9]));
        // A later match does not count after a mismatch.
        let predicted = [1, 9, 3, 4, 5, 6, 7, 8];
        assert_eq!(greedy_walk(&drafts, &predicted), (1, vec![1, 9]));
    }

    #[test]
    fn linear_commit_cuts_mid_window_by_budget_and_eos() {
        let drafts = [10_u32, 20, 30, 40, 50, 60, 70];
        let predicted = [10, 20, 30, 40, 50, 60, 70, 80];
        let (accepted, new_tokens) = greedy_walk(&drafts, &predicted);
        assert_eq!(new_tokens, vec![10, 20, 30, 40, 50, 60, 70, 80]);
        let token_offset = 16usize;
        let verify_len = drafts.len() + 1;

        let (budget_tokens, budget_keep) = linear_window_commit(&new_tokens, accepted, 3, &[]);
        assert_eq!(budget_tokens, tokens_emitted_today(&new_tokens, 3, &[]));
        assert_eq!(budget_tokens, vec![10, 20, 30]);
        let (cache_len, captured_rows, hidden_rows) =
            committed_lengths(token_offset, verify_len, budget_keep);
        assert_eq!(cache_len, captured_rows);
        assert_eq!(hidden_rows, cache_len - token_offset);
        assert_eq!(budget_keep, 1 + budget_tokens.len().min(accepted));
        assert_eq!(cache_len - token_offset, 1 + budget_tokens.len());
        assert!(cache_len < token_offset + verify_len);

        let (eos_tokens, eos_keep) = linear_window_commit(&new_tokens, accepted, 8, &[40]);
        assert_eq!(eos_tokens, tokens_emitted_today(&new_tokens, 8, &[40]));
        assert_eq!(eos_tokens, vec![10, 20, 30, 40]);
        let (cache_len, captured_rows, hidden_rows) =
            committed_lengths(token_offset, verify_len, eos_keep);
        assert_eq!(cache_len, captured_rows);
        assert_eq!(hidden_rows, cache_len - token_offset);
        assert_eq!(eos_keep, 1 + eos_tokens.len().min(accepted));
        assert_eq!(cache_len - token_offset, 1 + eos_tokens.len());
        assert!(cache_len < token_offset + verify_len);

        let (full, full_keep) = linear_window_commit(&new_tokens, accepted, 8, &[]);
        assert_eq!(full, new_tokens);
        assert_eq!(full_keep, accepted + 1);
    }

    #[test]
    fn acceptance_telemetry_counts_post_cut_drafts() {
        let drafts = [10_u32, 20, 30, 40, 50, 60, 70];
        let predicted = [10, 20, 30, 40, 50, 60, 70, 80];
        let (accepted, new_tokens) = greedy_walk(&drafts, &predicted);
        assert_eq!(accepted, 7);

        let (budget_tokens, budget_keep) = linear_window_commit(&new_tokens, accepted, 3, &[]);
        let budget_counted = committed_draft_count(accepted, budget_keep);
        assert_eq!(budget_counted, 3);
        assert_eq!(budget_counted, budget_tokens.len().min(accepted));
        assert!(budget_counted < accepted);
        let mut stats = WindowStats::new();
        stats.record(drafts.len(), budget_counted);
        assert_eq!(stats.hit[2], 1.0);
        assert_eq!(stats.hit[3], 0.0, "a draft past the budget is not a hit");

        let (eos_tokens, eos_keep) = linear_window_commit(&new_tokens, accepted, 8, &[40]);
        assert_eq!(eos_tokens, vec![10, 20, 30, 40]);
        assert_eq!(committed_draft_count(accepted, eos_keep), 4);

        let (_, full_keep) = linear_window_commit(&new_tokens, accepted, 8, &[]);
        assert_eq!(committed_draft_count(accepted, full_keep), accepted);

        let mut telemetry = MtpTelemetry::default();
        record_dflash2_committed_drafts(&mut telemetry, drafts.len(), budget_counted);
        assert_eq!(telemetry.accepted_tokens, budget_counted as u32);
        assert_ne!(telemetry.accepted_tokens, accepted as u32);
        record_dflash2_committed_drafts(
            &mut telemetry,
            drafts.len(),
            committed_draft_count(accepted, eos_keep),
        );
        assert_eq!(telemetry.accepted_tokens, 3 + 4);
    }

    #[test]
    fn verify_eval_failure_keeps_the_request_ready_to_draft_again() {
        let (weights, target) = crate::dflash2::dflash2_tiny_fixture();
        let mut state = Dflash2RequestState::new(weights.clone());
        let width = weights.config().target_hidden_width();
        let hidden = mlx_sys::zeros(&[1, 2, width], MlxDtype::Bfloat16, None);
        mlx_sys::eval(&[&hidden]);
        let pending_ptr = hidden.data_raw();
        let pending_rows = hidden.shape()[1];
        state.pending_hidden = Some(hidden);
        state.captured_rows = 5;
        let block = weights.config().block_size;
        let window = MIN_WINDOW;

        crate::dflash2::fail_next_dflash2_capture_verify_eval_for_test(
            "injected capture/verify eval failure",
        );
        let failed = dflash2_draft_window(
            &mut state,
            3,
            &target,
            block,
            window,
            Dflash2DraftRead::EvalOnly,
        );
        assert!(
            matches!(failed, Err("tree verify evaluation failed")),
            "eval failure must decline the round"
        );
        let pending = state.pending_hidden.as_ref().expect("pending hidden");
        assert_eq!(pending.data_raw(), pending_ptr);
        assert_eq!(pending.shape()[1], pending_rows);
        assert_eq!(state.captured_rows, 5);
        assert_eq!(state.drafter.cache_offset(), 0);
        assert_eq!(state.drafter.cache_rows(), 0);

        let (drafts, _cursor) = dflash2_draft_window(
            &mut state,
            3,
            &target,
            block,
            window,
            Dflash2DraftRead::EvalOnly,
        )
        .expect("next round drafts");
        mlx_sys::eval(&[&drafts]);
        assert_eq!(drafts.data_u32().len(), window - 1);
        assert!(state.drafter.cache_offset() > 0);
        assert!(state.drafter.cache_rows() > 0);
        assert_eq!(state.captured_rows, 5);
        assert_eq!(
            state
                .pending_hidden
                .as_ref()
                .expect("pending hidden")
                .data_raw(),
            pending_ptr
        );
    }

    #[test]
    fn shape_mismatch_rewinds_the_drafter_after_a_successful_draft() {
        let (weights, target) = crate::dflash2::dflash2_tiny_fixture();
        let mut state = Dflash2RequestState::new(weights.clone());
        let width = weights.config().target_hidden_width();
        let hidden = mlx_sys::zeros(&[1, 2, width], MlxDtype::Bfloat16, None);
        mlx_sys::eval(&[&hidden]);
        let pending_ptr = hidden.data_raw();
        state.pending_hidden = Some(hidden);
        state.captured_rows = 5;
        let block = weights.config().block_size;
        let window = MIN_WINDOW;
        let (_draft_arr, cursor) = dflash2_draft_window(
            &mut state,
            3,
            &target,
            block,
            window,
            Dflash2DraftRead::Lazy,
        )
        .expect("draft");
        assert!(state.drafter.cache_offset() > 0);
        disable_dflash2_shape_mismatch(&mut state, cursor);
        assert!(state.disabled);
        assert_eq!(state.drafter.cache_offset(), 0);
        assert_eq!(state.drafter.cache_rows(), 0);
        assert_eq!(state.captured_rows, 5);
        assert_eq!(
            state
                .pending_hidden
                .as_ref()
                .expect("pending hidden")
                .data_raw(),
            pending_ptr
        );
    }

    /// EOS cut, then the output-budget cut, matching `apply_decode_result`
    /// followed by `truncate_sampled_tokens_for_stop`.
    fn tokens_emitted_today(tokens: &[u32], max_output: u32, eos: &[u32]) -> Vec<u32> {
        let mut output = Vec::new();
        for &token in tokens {
            output.push(token);
            if eos.contains(&token) {
                break;
            }
        }
        super::super::truncate_sampled_tokens_for_stop(output, 0, max_output, eos).0
    }

    /// Cache length, captured rows, and pending-hidden rows after a verify
    /// window is trimmed to `keep`.
    fn committed_lengths(
        token_offset: usize,
        verify_len: usize,
        keep: usize,
    ) -> (usize, usize, usize) {
        let mut cache = crate::kv_cache::MlxKVCache::new_contiguous(0);
        cache.set_seq_len(token_offset);
        cache.advance(verify_len);
        assert!(cache.trim_to(token_offset + keep));
        let captured_rows = token_offset + keep;
        let width = 2i32;
        let data = vec![0f32; verify_len * width as usize];
        let hidden = mlx_sys::MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, verify_len as i32, width],
            mlx_sys::MlxDtype::Float32,
        );
        let pending = mlx_sys::slice(
            &hidden,
            &[0, 0, 0],
            &[1, keep as i32, width],
            &[1, 1, 1],
            None,
        );
        mlx_sys::eval(&[&pending]);
        let hidden_rows = pending.shape()[1] as usize;
        (cache.seq_len(), captured_rows, hidden_rows)
    }
}
