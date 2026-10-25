//! Server-side client stop-sequence enforcement for the native MLX backend
//! (ADR-040 D2). The engine core has no string-stop concept — stop strings
//! are a text contract, and the server owns the text boundary (tokenizer,
//! channel filters, incremental decode). Delegated backends forward stops
//! upstream and never reach this module.

use ax_engine_sdk::{GenerateFinishReason, GenerateResponse};
use axum::Json;
use axum::http::StatusCode;

use crate::errors::{ErrorResponse, error_response};

/// OpenAI caps `stop` at 4 sequences; the byte cap bounds the streaming
/// holdback window (text withheld until it can no longer begin a match).
pub(crate) const MAX_CLIENT_STOP_SEQUENCES: usize = 4;
pub(crate) const MAX_CLIENT_STOP_SEQUENCE_BYTES: usize = 256;

pub(crate) fn validate_client_stop_sequences(
    stop: &[String],
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if stop.len() > MAX_CLIENT_STOP_SEQUENCES {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "stop accepts at most {MAX_CLIENT_STOP_SEQUENCES} sequences (received {})",
                stop.len()
            ),
        ));
    }
    for sequence in stop {
        if sequence.is_empty() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "stop sequences must not be empty".to_string(),
            ));
        }
        if sequence.len() > MAX_CLIENT_STOP_SEQUENCE_BYTES {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!(
                    "stop sequences must be at most {MAX_CLIENT_STOP_SEQUENCE_BYTES} bytes \
                     (received one of {} bytes)",
                    sequence.len()
                ),
            ));
        }
    }
    Ok(())
}

/// Byte offset of the earliest match of any stop sequence, with its length.
/// Ties at the same offset prefer the longest sequence so truncation removes
/// the whole matched stop text.
pub(crate) fn find_earliest_stop(text: &str, sequences: &[String]) -> Option<(usize, usize)> {
    let mut earliest: Option<(usize, usize)> = None;
    for sequence in sequences {
        if sequence.is_empty() {
            continue;
        }
        if let Some(index) = text.find(sequence.as_str()) {
            let better = match earliest {
                None => true,
                Some((at, len)) => index < at || (index == at && sequence.len() > len),
            };
            if better {
                earliest = Some((index, sequence.len()));
            }
        }
    }
    earliest
}

/// Truncate at the earliest stop match. Returns true when a match fired
/// (callers set `finish_reason: "stop"`).
pub(crate) fn truncate_at_stop(text: &mut String, sequences: &[String]) -> bool {
    if sequences.is_empty() {
        return false;
    }
    match find_earliest_stop(text, sequences) {
        Some((index, _)) => {
            text.truncate(index);
            true
        }
        None => false,
    }
}

/// Truncate a native `GenerateResponse`'s output text at the earliest client
/// stop match and mark the generation as stopped. Returns the matched stop
/// sequence (surfaces as Anthropic `stop_sequence`). For response surfaces
/// built directly from `GenerateResponse` rather than through the OpenAI
/// response builders.
pub(crate) fn apply_client_stops_to_generate_response(
    response: &mut GenerateResponse,
    client_stop_sequences: &[String],
) -> Option<String> {
    apply_client_stops(response, client_stop_sequences, false)
}

/// Like [`apply_client_stops_to_generate_response`], but a stop string inside
/// a separated reasoning span (`<think>…</think>`, a prefilled `</think>`
/// body, or a `<|channel>thought` channel) does not end the generation.
/// Native EOS is not a string stop and is left to the engine finish reason.
pub(crate) fn apply_client_stops_to_answer_only(
    response: &mut GenerateResponse,
    client_stop_sequences: &[String],
) -> Option<String> {
    apply_client_stops(response, client_stop_sequences, true)
}

fn apply_client_stops(
    response: &mut GenerateResponse,
    client_stop_sequences: &[String],
    answer_only: bool,
) -> Option<String> {
    if client_stop_sequences.is_empty() {
        return None;
    }
    let text = response.output_text.as_mut()?;
    let (index, len) = if answer_only {
        find_earliest_answer_stop(text, client_stop_sequences)?
    } else {
        find_earliest_stop(text, client_stop_sequences)?
    };
    let matched = text[index..index + len].to_string();
    text.truncate(index);
    response.finish_reason = Some(GenerateFinishReason::Stop);
    Some(matched)
}

/// Byte ranges of reasoning spans, including their tags, so a stop that
/// sits inside the reasoning text does not match.
fn reasoning_mask_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut cursor = 0usize;
    if let Some(close) = text.find("</think>")
        && !text[..close].contains("<think>")
    {
        let end = close + "</think>".len();
        spans.push((0, end));
        cursor = end;
    }
    while let Some(relative) = text[cursor..].find("<think>") {
        let start = cursor + relative;
        let after = start + "<think>".len();
        if let Some(end_rel) = text[after..].find("</think>") {
            let end = after + end_rel + "</think>".len();
            spans.push((start, end));
            cursor = end;
        } else {
            spans.push((start, text.len()));
            break;
        }
    }
    let mut scan = 0usize;
    const CHANNEL_OPEN: &str = "<|channel>thought\n";
    const CHANNEL_CLOSE: &str = "<channel|>";
    while let Some(relative) = text[scan..].find(CHANNEL_OPEN) {
        let start = scan + relative;
        let after = start + CHANNEL_OPEN.len();
        if let Some(end_rel) = text[after..].find(CHANNEL_CLOSE) {
            let end = after + end_rel + CHANNEL_CLOSE.len();
            spans.push((start, end));
            scan = end;
        } else {
            spans.push((start, text.len()));
            break;
        }
    }
    spans
}

fn find_earliest_answer_stop(text: &str, sequences: &[String]) -> Option<(usize, usize)> {
    let masked = reasoning_mask_spans(text);
    let mut earliest: Option<(usize, usize)> = None;
    for sequence in sequences {
        if sequence.is_empty() {
            continue;
        }
        let mut from = 0usize;
        while let Some(relative) = text[from..].find(sequence.as_str()) {
            let index = from + relative;
            let end = index + sequence.len();
            let overlaps_reasoning = masked
                .iter()
                .any(|(start, mask_end)| index < *mask_end && end > *start);
            if !overlaps_reasoning {
                let better = match earliest {
                    None => true,
                    Some((at, len)) => index < at || (index == at && sequence.len() > len),
                };
                if better {
                    earliest = Some((index, sequence.len()));
                }
                break;
            }
            from = index + sequence.len();
            if from > text.len() {
                break;
            }
        }
    }
    earliest
}

/// Incremental stop matcher for streamed text. Emits text as soon as it can
/// no longer begin a stop match; a match that spans chunk boundaries is
/// therefore caught before any of it is emitted.
pub(crate) struct StopSequenceScanner {
    sequences: Vec<String>,
    pending: String,
    matched: bool,
}

pub(crate) struct StopScanStep {
    pub(crate) emit: String,
    pub(crate) matched: bool,
}

impl StopSequenceScanner {
    pub(crate) fn new(sequences: Vec<String>) -> Option<Self> {
        if sequences.is_empty() {
            return None;
        }
        Some(Self {
            sequences,
            pending: String::new(),
            matched: false,
        })
    }

    pub(crate) fn push(&mut self, text: &str) -> StopScanStep {
        if self.matched {
            // A tool opener may defer termination. Retain later pre-call
            // content so a completed call can supersede the stop intact.
            self.pending.push_str(text);
            return StopScanStep {
                emit: String::new(),
                matched: true,
            };
        }
        self.pending.push_str(text);
        if let Some((index, _)) = find_earliest_stop(&self.pending, &self.sequences) {
            self.matched = true;
            let emit = self.pending.drain(..index).collect();
            return StopScanStep {
                emit,
                matched: true,
            };
        }
        let hold = self.holdback_len();
        let release = self.pending.len() - hold;
        let emit = self.pending[..release].to_string();
        self.pending.drain(..release);
        StopScanStep {
            emit,
            matched: false,
        }
    }

    /// End of stream with no match: release everything withheld.
    pub(crate) fn finish(&mut self) -> String {
        if self.matched {
            self.pending.clear();
        }
        std::mem::take(&mut self.pending)
    }

    /// A completed tool call supersedes client stops. Restore withheld text,
    /// including a deferred match, before emitting the call's content batch.
    pub(crate) fn finish_without_stop(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    #[cfg(test)]
    pub(crate) fn matched(&self) -> bool {
        self.matched
    }

    /// Length of the longest `pending` suffix that is a proper prefix of some
    /// stop sequence — the only bytes a future chunk could turn into a match.
    /// Always lands on a char boundary because it equals the length of a
    /// suffix of the (UTF-8) pending string that equals a prefix of a stop
    /// sequence.
    fn holdback_len(&self) -> usize {
        let pending = self.pending.as_bytes();
        let mut hold = 0usize;
        for sequence in &self.sequences {
            let sequence = sequence.as_bytes();
            let max = sequence.len().saturating_sub(1).min(pending.len());
            for len in (hold + 1..=max).rev() {
                if pending[pending.len() - len..] == sequence[..len] {
                    hold = len;
                    break;
                }
            }
        }
        hold
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn scanner(sequences: &[&str]) -> StopSequenceScanner {
        StopSequenceScanner::new(sequences.iter().map(|s| s.to_string()).collect())
            .expect("non-empty stop list")
    }

    #[test]
    fn validation_enforces_count_and_bounds() {
        let ok = vec!["a".to_string(); 4];
        validate_client_stop_sequences(&ok).expect("4 sequences allowed");
        let too_many = vec!["a".to_string(); 5];
        let _ = validate_client_stop_sequences(&too_many).expect_err("5 sequences rejected");
        let _ =
            validate_client_stop_sequences(&[String::new()]).expect_err("empty sequence rejected");
        let _ = validate_client_stop_sequences(&["x".repeat(257)]).expect_err("oversized rejected");
        validate_client_stop_sequences(&[]).expect("no stops is fine");
    }

    #[test]
    fn earliest_match_wins_and_ties_prefer_longest() {
        let stops = vec!["END".to_string(), "ENDING".to_string(), "STOP".to_string()];
        assert_eq!(find_earliest_stop("xxSTOPyyEND", &stops), Some((2, 4)));
        // Both END and ENDING match at 2; the longer one is removed whole.
        assert_eq!(find_earliest_stop("xxENDING", &stops), Some((2, 6)));
    }

    #[test]
    fn truncation_excludes_stop_text() {
        let mut text = "hello STOP world".to_string();
        assert!(truncate_at_stop(&mut text, &["STOP".to_string()]));
        assert_eq!(text, "hello ");
        let mut text = "no match".to_string();
        assert!(!truncate_at_stop(&mut text, &["STOP".to_string()]));
        assert_eq!(text, "no match");
    }

    #[test]
    fn match_within_one_chunk() {
        let mut scanner = scanner(&["STOP"]);
        let step = scanner.push("aaSTOPbb");
        assert_eq!(step.emit, "aa");
        assert!(step.matched);
        // Nothing further is ever emitted.
        assert_eq!(scanner.push("cc").emit, "");
        assert_eq!(scanner.finish(), "");
    }

    #[test]
    fn match_split_across_chunks_never_leaks() {
        let mut scanner = scanner(&["STOP"]);
        let step = scanner.push("hello ST");
        assert_eq!(step.emit, "hello ");
        assert!(!step.matched);
        let step = scanner.push("OP world");
        assert_eq!(step.emit, "");
        assert!(step.matched);
    }

    #[test]
    fn false_prefix_is_released_on_next_chunk() {
        let mut scanner = scanner(&["STOP"]);
        assert_eq!(scanner.push("aaST").emit, "aa");
        // "START" contains no live stop prefix suffix, so it all releases.
        let step = scanner.push("ART");
        assert!(!step.matched);
        assert_eq!(step.emit, "START");
        assert_eq!(scanner.finish(), "");
    }

    #[test]
    fn finish_releases_withheld_tail() {
        let mut scanner = scanner(&["\n\n"]);
        let step = scanner.push("line\n");
        assert_eq!(step.emit, "line");
        assert!(!step.matched);
        assert_eq!(scanner.finish(), "\n");
        assert!(!scanner.matched());
    }

    #[test]
    fn multibyte_text_near_holdback_stays_on_char_boundaries() {
        let mut scanner = scanner(&["終わり"]);
        let step = scanner.push("你好終");
        assert_eq!(step.emit, "你好");
        assert!(!step.matched);
        let step = scanner.push("わり後");
        assert!(step.matched);
        assert_eq!(step.emit, "");
    }

    #[test]
    fn multiple_sequences_hold_longest_candidate() {
        let mut scanner = scanner(&["ab", "xyz"]);
        // "x" could start xyz (hold 1), trailing "a" could start ab (hold 1);
        // the longer live candidate governs.
        let step = scanner.push("qqxy");
        assert_eq!(step.emit, "qq");
        let step = scanner.push("za");
        assert!(step.matched);
        assert_eq!(step.emit, "");
    }

    fn response_with_text(text: &str) -> GenerateResponse {
        GenerateResponse {
            request_id: 1,
            model_id: "qwen3".to_string(),
            prompt_tokens: Vec::new(),
            prompt_text: None,
            output_tokens: Vec::new(),
            output_token_logprobs: Vec::new(),
            output_text: Some(text.to_string()),
            prompt_token_count: None,
            output_token_count: None,
            status: ax_engine_sdk::GenerateStatus::Finished,
            finish_reason: Some(GenerateFinishReason::MaxOutputTokens),
            step_count: 1,
            ttft_step: None,
            route: ax_engine_sdk::GenerateRouteReport::default(),
            runtime: ax_engine_sdk::RuntimeReport {
                selected_backend: ax_engine_sdk::SelectedBackend::Mlx,
                support_tier: ax_engine_sdk::SupportTier::MlxPreview,
                resolution_policy: ax_engine_sdk::ResolutionPolicy::MlxOnly,
                capabilities: ax_engine_sdk::CapabilityReport::mlx_preview(),
                fallback_reason: None,
                host: Default::default(),
                metal_toolchain: Default::default(),
                mlx_runtime: None,
                mlx_model: None,
                delegated_runtime: None,
            },
            performance: Default::default(),
        }
    }

    #[test]
    fn answer_only_stop_ignores_a_match_inside_think_tags() {
        let mut response = response_with_text("<think>STOP now</think>hello STOP tail");
        let matched = apply_client_stops_to_answer_only(&mut response, &["STOP".to_string()]);
        assert_eq!(matched.as_deref(), Some("STOP"));
        assert_eq!(
            response.output_text.as_deref(),
            Some("<think>STOP now</think>hello ")
        );
        assert_eq!(response.finish_reason, Some(GenerateFinishReason::Stop));
    }

    #[test]
    fn answer_only_stop_keeps_generation_when_the_only_match_is_reasoning() {
        let mut response = response_with_text("<think>please STOP</think>the answer");
        let matched = apply_client_stops_to_answer_only(&mut response, &["STOP".to_string()]);
        assert!(matched.is_none());
        assert_eq!(
            response.output_text.as_deref(),
            Some("<think>please STOP</think>the answer")
        );
        assert_eq!(
            response.finish_reason,
            Some(GenerateFinishReason::MaxOutputTokens)
        );
    }

    #[test]
    fn answer_only_stop_ignores_a_prefilled_think_close() {
        let mut response = response_with_text("STOP inside</think>visible");
        let matched = apply_client_stops_to_answer_only(&mut response, &["STOP".to_string()]);
        assert!(matched.is_none());
        assert_eq!(
            response.output_text.as_deref(),
            Some("STOP inside</think>visible")
        );
    }

    #[test]
    fn full_text_stop_still_matches_inside_think_tags() {
        let mut response = response_with_text("<think>STOP</think>hello");
        let matched = apply_client_stops_to_generate_response(&mut response, &["STOP".to_string()]);
        assert_eq!(matched.as_deref(), Some("STOP"));
        assert_eq!(response.output_text.as_deref(), Some("<think>"));
    }
}
