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
    if client_stop_sequences.is_empty() {
        return None;
    }
    let text = response.output_text.as_mut()?;
    let (index, len) = find_earliest_stop(text, client_stop_sequences)?;
    let matched = text[index..index + len].to_string();
    text.truncate(index);
    response.finish_reason = Some(GenerateFinishReason::Stop);
    Some(matched)
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
            return StopScanStep {
                emit: String::new(),
                matched: true,
            };
        }
        self.pending.push_str(text);
        if let Some((index, _)) = find_earliest_stop(&self.pending, &self.sequences) {
            self.matched = true;
            let emit = self.pending[..index].to_string();
            self.pending.clear();
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
}
