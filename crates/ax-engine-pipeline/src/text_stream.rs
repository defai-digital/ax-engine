//! Streaming text-delta helpers shared by the gateway server and the
//! `ax-engine-pipeline-generate` CLI.

/// Decode text with trailing incomplete multi-byte codepoints stripped.
///
/// Byte-level BPE (Qwen/Gemma/etc.) can leave a partial UTF-8 sequence at the
/// end of a token window; HuggingFace-style decode renders that as U+FFFD. A
/// codepoint split across several tokens (byte-fallback) leaves a whole run of
/// them, so strip the run: leaving any behind leaks a bare U+FFFD and desyncs
/// the streaming cursor.
pub fn complete_decode_prefix(decoded: &str) -> &str {
    decoded.trim_end_matches('\u{FFFD}')
}

/// Diff consecutive full-sequence decodes for streaming deltas.
///
/// Holds back a trailing run of U+FFFD (incomplete multi-byte codepoints) and
/// never falls back to re-emitting the entire string when prefix strip fails —
/// that fallback was the source of CJK/emoji corruption and full-text re-sends.
pub fn stream_delta<'a>(already_emitted: &str, next_full_decode: &'a str) -> Option<&'a str> {
    let complete = complete_decode_prefix(next_full_decode);
    if complete.len() <= already_emitted.len() {
        return None;
    }
    if !complete.starts_with(already_emitted) {
        // Tokenizer non-monotonic decode or corrupted cursor: skip rather than
        // re-emit the whole string. The final non-stream decode remains correct.
        return None;
    }
    if !complete.is_char_boundary(already_emitted.len()) {
        return None;
    }
    let delta = &complete[already_emitted.len()..];
    if delta.is_empty() { None } else { Some(delta) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_decode_prefix_strips_trailing_replacement_run() {
        assert_eq!(complete_decode_prefix("hello"), "hello");
        assert_eq!(complete_decode_prefix("hello\u{FFFD}"), "hello");
        assert_eq!(complete_decode_prefix("\u{FFFD}"), "");
        // Consecutive replacements (a multi-byte codepoint split across tokens)
        // strip as a run, not just the last one.
        assert_eq!(complete_decode_prefix("\u{FFFD}\u{FFFD}"), "");
        assert_eq!(complete_decode_prefix("叫\u{FFFD}\u{FFFD}"), "叫");
        // Mid-string replacement (corrupt data) is left alone.
        assert_eq!(complete_decode_prefix("a\u{FFFD}b"), "a\u{FFFD}b");
    }

    #[test]
    fn stream_delta_emits_ascii_suffix_only() {
        assert_eq!(stream_delta("", "hello"), Some("hello"));
        assert_eq!(stream_delta("hel", "hello"), Some("lo"));
        assert_eq!(stream_delta("hello", "hello"), None);
    }

    #[test]
    fn stream_delta_holds_back_incomplete_multibyte_tail() {
        // Partial UTF-8 from byte-level BPE decodes as trailing U+FFFD.
        // Complete leading text is emitted; only the incomplete tail is held.
        assert_eq!(stream_delta("", "ab\u{FFFD}"), Some("ab"));
        assert_eq!(stream_delta("ab", "ab\u{FFFD}"), None);
        // Completing the codepoint emits only the new character.
        assert_eq!(stream_delta("ab", "ab你"), Some("你"));
        assert_eq!(stream_delta("", "🚀"), Some("🚀"));
        // Mixed complete CJK + incomplete tail: emit the complete codepoints.
        assert_eq!(stream_delta("你", "你好\u{FFFD}"), Some("好"));
        assert_eq!(stream_delta("你好", "你好世"), Some("世"));
    }

    #[test]
    fn stream_delta_never_reemits_full_string_on_prefix_mismatch() {
        // Historical bug: strip_prefix failure fell back to the whole decode,
        // re-sending prior content after a corrupted � prefix.
        assert_eq!(stream_delta("ab\u{FFFD}", "ab你"), None);
        assert_eq!(stream_delta("xy", "hello"), None);
    }

    /// Replay the streaming cursor protocol used by `generate` over a sequence
    /// of full-sequence decodes, returning the text a client would receive.
    fn replay_stream_decodes(decodes: &[&str]) -> String {
        let mut emitted = String::new();
        let mut streamed = String::new();
        for decode in decodes {
            if let Some(delta) = stream_delta(&emitted, decode) {
                // Mirrors the call site: the cursor advances only when a delta
                // is produced, and only over complete text.
                emitted = complete_decode_prefix(decode).to_string();
                streamed.push_str(delta);
            }
        }
        streamed
    }

    #[test]
    fn stream_delta_replays_byte_fallback_codepoint_split_across_tokens() {
        // Byte-fallback tokenizers split one CJK codepoint across tokens; the
        // decodes before the final byte end in a run of U+FFFD (one per pending
        // byte). Stripping only one replacement leaked a bare U+FFFD into the
        // stream, stored it as the cursor, and every later decode failed the
        // prefix check — the rest of the stream was dropped.
        assert_eq!(
            replay_stream_decodes(&["\u{FFFD}", "\u{FFFD}\u{FFFD}", "叫"]),
            "叫"
        );
        // The cursor stays usable for later codepoints, including a second
        // split one.
        assert_eq!(
            replay_stream_decodes(&[
                "\u{FFFD}",
                "\u{FFFD}\u{FFFD}",
                "叫",
                "叫\u{FFFD}",
                "叫\u{FFFD}\u{FFFD}",
                "叫好",
            ]),
            "叫好"
        );
    }

    #[test]
    fn stream_delta_replays_byte_fallback_codepoint_split_stepwise() {
        // Mirror of the gateway stream regression: a codepoint split across
        // byte-fallback tokens must print exactly once, without leaking U+FFFD
        // or dropping the rest of the stream.
        let mut emitted = String::new();
        let mut printed = String::new();
        for decode in ["\u{FFFD}", "\u{FFFD}\u{FFFD}", "叫", "叫\u{FFFD}", "叫好"] {
            if let Some(delta) = stream_delta(&emitted, decode) {
                printed.push_str(delta);
                emitted = complete_decode_prefix(decode).to_string();
            }
        }
        assert_eq!(printed, "叫好");
        assert_eq!(emitted, "叫好");
        assert_eq!(stream_delta("ab\u{FFFD}", "ab你"), None);
    }
}
