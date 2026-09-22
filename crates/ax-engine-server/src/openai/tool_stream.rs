//! Incremental tool-call scanning for native chat streams (ADR-040 D1).
//!
//! Content outside tool-call spans streams live; each completed call is
//! parsed by the same extractors the non-streaming path uses
//! (`openai/responses.rs`) and emitted as one spec-conformant
//! `delta.tool_calls` fragment. Marker text is withheld with a bounded
//! holdback so a marker split across token boundaries never leaks into
//! content and never stalls the stream.

use std::sync::Arc;

use crate::openai::dsml;
use crate::openai::requests::OpenAiToolContract;
use crate::openai::responses::{
    Gemma4ObjectScan, extract_bare_gemma4_tool_call_payload_at,
    extract_closed_xml_tool_call_payload_at, extract_gemma4_tool_call_payload_at,
    extract_xml_tool_call_payload_at, find_bare_gemma4_call, scan_gemma4_object_body,
};
use crate::openai::schema::{OpenAiFunctionCall, OpenAiToolCall};

const XML_OPEN: &str = "<tool_call>";
const XML_CLOSE: &str = "</tool_call>";
const GEMMA4_OPEN: &str = "<|tool_call>";
const GEMMA4_CLOSE: &str = "<tool_call|>";
const BARE_GEMMA4_LEAD: &str = "call:";

#[derive(Debug)]
pub(crate) enum ToolScanEvent {
    Content(String),
    Call(OpenAiToolCall),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolSpanKind {
    Xml,
    Gemma4,
    BareGemma4,
    /// DeepSeek `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` stanza; one
    /// stanza may carry several invokes.
    Dsml,
}

pub(crate) struct ToolCallStreamScanner {
    /// Withheld text: an open span plus at most one partial opener suffix.
    buffer: String,
    /// When set, the span starts at `buffer[0]`.
    span: Option<ToolSpanKind>,
    /// True once any non-whitespace content has been emitted; gates the
    /// bare-Gemma4 form, which is only valid as the leading output.
    emitted_visible: bool,
    /// True once a DSML stanza has emitted at least one call for this
    /// response. The non-streaming extractor hands the DSML leftover back as
    /// content without re-running the XML/Gemma4 extractors, so after a DSML
    /// stanza later non-DSML openers are content (further DSML stanzas are
    /// still parsed).
    dsml_emitted: bool,
    /// Per-span resume offset for the XML / Gemma4 closer search: bytes of the
    /// current span already known to hold no (or only already-failed) closer,
    /// so the next push scans only the appended tail instead of the whole
    /// growing buffer. Reset when the span changes.
    span_scan_offset: usize,
    /// Per-span resume state for the bare Gemma4 brace matcher.
    bare_scan_from: usize,
    bare_scan_depth: usize,
    calls_emitted: u32,
    contract: Option<Arc<OpenAiToolContract>>,
}

impl ToolCallStreamScanner {
    pub(crate) fn new(contract: Option<Arc<OpenAiToolContract>>) -> Self {
        Self {
            buffer: String::new(),
            span: None,
            emitted_visible: false,
            dsml_emitted: false,
            span_scan_offset: 0,
            bare_scan_from: 0,
            bare_scan_depth: 0,
            calls_emitted: 0,
            contract,
        }
    }

    #[cfg(test)]
    pub(crate) fn calls_emitted(&self) -> u32 {
        self.calls_emitted
    }

    pub(crate) fn push(&mut self, text: &str) -> Vec<ToolScanEvent> {
        self.buffer.push_str(text);
        self.drain_events(false)
    }

    /// End of stream: parse an unterminated span if the family grammar allows
    /// it (the XML extractor tolerates a missing closer), otherwise flush the
    /// withheld text as content. Never drops text, never leaves a call half
    /// emitted.
    pub(crate) fn finish(&mut self) -> Vec<ToolScanEvent> {
        let mut events = self.drain_events(true);
        if let Some(kind) = self.span.take()
            && !self.buffer.is_empty()
            && let Some((function, remaining)) = self.extract_span_at_start(kind, true)
        {
            events.push(ToolScanEvent::Call(self.build_call(function)));
            self.buffer = remaining;
        }
        if !self.buffer.is_empty() {
            let content = std::mem::take(&mut self.buffer);
            self.note_visible(&content);
            events.push(ToolScanEvent::Content(content));
        }
        events
    }

    fn drain_events(&mut self, at_end: bool) -> Vec<ToolScanEvent> {
        let mut events = Vec::new();
        loop {
            match self.span {
                None => {
                    if let Some((start, kind)) = self.find_earliest_opener() {
                        if start > 0 {
                            let content = self.buffer[..start].to_string();
                            self.buffer.drain(..start);
                            self.note_visible(&content);
                            events.push(ToolScanEvent::Content(content));
                        }
                        self.span = Some(kind);
                        self.span_scan_offset = 0;
                        self.bare_scan_from = 0;
                        self.bare_scan_depth = 0;
                        continue;
                    }
                    let hold = if at_end {
                        0
                    } else {
                        self.opener_holdback_len()
                    };
                    let release = self.buffer.len() - hold;
                    if release > 0 {
                        let content = self.buffer[..release].to_string();
                        self.buffer.drain(..release);
                        self.note_visible(&content);
                        events.push(ToolScanEvent::Content(content));
                    }
                    return events;
                }
                Some(ToolSpanKind::Dsml) => {
                    // Lenient DSML closer (filler tolerated), then the shared
                    // stanza parser, which yields every invoke in the stanza.
                    // An argument string can contain a literal `</tool_calls>`
                    // closer, so the first closer may not terminate the stanza;
                    // try each successive closer and emit the calls from the
                    // first slice that parses.
                    let mut parsed = false;
                    let mut from = 0usize;
                    while let Some((_, close_end)) =
                        dsml::find_dsml_tool_calls_close(&self.buffer, from)
                    {
                        from = close_end;
                        match dsml::parse_dsml_tool_calls(&self.buffer[..close_end]) {
                            Some((functions, _)) => {
                                for function in functions {
                                    events.push(ToolScanEvent::Call(self.build_call(function)));
                                }
                                self.buffer.drain(..close_end);
                                parsed = true;
                                break;
                            }
                            None => continue,
                        }
                    }
                    if parsed {
                        self.dsml_emitted = true;
                        self.span = None;
                        continue;
                    }
                    // No closer parsed a complete stanza. Release through the
                    // first closer as content (stopping at a later valid opener
                    // so a restarted call can still be rescanned, as the XML
                    // branch does at end of stream) instead of withholding the
                    // whole stream until EOS. With no closer at all we keep
                    // withholding and wait for more data.
                    let Some((_, first_close_end)) =
                        dsml::find_dsml_tool_calls_close(&self.buffer, 0)
                    else {
                        return events;
                    };
                    let end = self
                        .next_opener_after_start(first_close_end)
                        .map_or(first_close_end, |inner| inner.min(first_close_end));
                    let content = self.buffer[..end].to_string();
                    self.buffer.drain(..end);
                    self.note_visible(&content);
                    events.push(ToolScanEvent::Content(content));
                    self.span = None;
                    continue;
                }
                Some(kind) => {
                    let closer = match kind {
                        ToolSpanKind::Xml => Some(XML_CLOSE),
                        ToolSpanKind::Gemma4 => Some(GEMMA4_CLOSE),
                        ToolSpanKind::BareGemma4 | ToolSpanKind::Dsml => None,
                    };
                    if let Some(closer) = closer {
                        if at_end {
                            // End of stream: a closer must be present for the
                            // span to still be withheld here (no closer means
                            // `finish`'s unterminated-span handling owns it).
                            let Some(close_at) = self.buffer.find(closer) else {
                                return events;
                            };
                            match self.extract_span_at_start(kind, true) {
                                Some((function, remaining)) => {
                                    events.push(ToolScanEvent::Call(self.build_call(function)));
                                    self.buffer = remaining;
                                }
                                None => {
                                    // The withheld span may hold a later, valid
                                    // opener (the model restarted its call):
                                    // release only up to it so the rescan can
                                    // still extract that call, as the
                                    // non-streaming extractor does.
                                    let end = close_at + closer.len();
                                    let end = self
                                        .next_opener_after_start(end)
                                        .map_or(end, |inner| inner.min(end));
                                    let content = self.buffer[..end].to_string();
                                    self.buffer.drain(..end);
                                    self.note_visible(&content);
                                    events.push(ToolScanEvent::Content(content));
                                }
                            }
                            self.span = None;
                            self.span_scan_offset = 0;
                            continue;
                        }
                        // Resume the closer search from the last scanned
                        // offset instead of rescanning the whole buffer every
                        // push: a closer whose body already failed to parse
                        // can never succeed (the bytes before it are fixed).
                        let rest = &self.buffer[self.span_scan_offset..];
                        let Some(relative) = rest.find(closer) else {
                            self.span_scan_offset = self
                                .buffer
                                .len()
                                .saturating_sub(closer.len().saturating_sub(1));
                            return events;
                        };
                        let close_at = self.span_scan_offset + relative;
                        self.span_scan_offset = close_at + closer.len();
                        // The extractor re-tries every closer from the start,
                        // so call it only when a new closer arrived; a body
                        // that has not parsed yet stays withheld.
                        if let Some((function, remaining)) = self.extract_span_at_start(kind, false)
                        {
                            events.push(ToolScanEvent::Call(self.build_call(function)));
                            self.buffer = remaining;
                            self.span = None;
                            self.span_scan_offset = 0;
                        }
                        continue;
                    }
                    // Bare Gemma4: no closer marker; complete when the brace
                    // matcher inside the extractor succeeds. Resume the brace
                    // scan from where the previous push left off so a long
                    // bare body is not re-brace-matched from byte 0 every push.
                    if let Some(body_start) = self.bare_body_start() {
                        if self.bare_scan_from < body_start {
                            self.bare_scan_from = body_start;
                            self.bare_scan_depth = 1;
                        }
                        match scan_gemma4_object_body(
                            &self.buffer,
                            self.bare_scan_from,
                            self.bare_scan_depth,
                        ) {
                            Gemma4ObjectScan::Complete(_) => {
                                // The object is complete; re-run the shared
                                // extractor once to build the call.
                                if let Some((function, remaining)) =
                                    self.extract_span_at_start(ToolSpanKind::BareGemma4, at_end)
                                {
                                    events.push(ToolScanEvent::Call(self.build_call(function)));
                                    self.buffer = remaining;
                                    self.span = None;
                                    self.bare_scan_from = 0;
                                    self.bare_scan_depth = 0;
                                    continue;
                                }
                            }
                            Gemma4ObjectScan::Incomplete { from, depth } => {
                                self.bare_scan_from = from;
                                self.bare_scan_depth = depth;
                            }
                        }
                    }
                    if !self.bare_span_still_viable() {
                        // The span can no longer become a valid bare call
                        // (e.g. an illegal name character before `{`): stop
                        // withholding and stream it as ordinary content.
                        // Marking output as visible retires the bare form so
                        // the same `call:` text cannot re-enter a span.
                        self.span = None;
                        self.emitted_visible = true;
                        continue;
                    }
                    return events;
                }
            }
        }
    }

    /// Earliest marker opener strictly after `buffer[0]` and before `limit`.
    fn next_opener_after_start(&self, limit: usize) -> Option<usize> {
        let window = self.buffer.get(1..limit)?;
        let mut candidates: Vec<usize> = Vec::new();
        if !self.dsml_emitted {
            for opener in [XML_OPEN, GEMMA4_OPEN] {
                if let Some(index) = window.find(opener) {
                    candidates.push(index);
                }
            }
        }
        if let Some(index) = dsml::find_dsml_tool_calls_open(window) {
            candidates.push(index);
        }
        candidates.into_iter().min().map(|index| index + 1)
    }

    fn find_earliest_opener(&self) -> Option<(usize, ToolSpanKind)> {
        let mut earliest: Option<(usize, ToolSpanKind)> = None;
        let mut consider = |candidate: Option<usize>, kind: ToolSpanKind| {
            if let Some(index) = candidate
                && earliest.is_none_or(|(at, _)| index < at)
            {
                earliest = Some((index, kind));
            }
        };
        if self.dsml_emitted {
            // After a DSML stanza, later XML / Gemma4 / bare openers are
            // ordinary content (the non-streaming extractor does not re-run
            // them on the DSML leftover); only further DSML stanzas are
            // parsed.
            consider(
                dsml::find_dsml_tool_calls_open(&self.buffer),
                ToolSpanKind::Dsml,
            );
            return earliest;
        }
        consider(self.buffer.find(XML_OPEN), ToolSpanKind::Xml);
        consider(self.buffer.find(GEMMA4_OPEN), ToolSpanKind::Gemma4);
        consider(
            dsml::find_dsml_tool_calls_open(&self.buffer),
            ToolSpanKind::Dsml,
        );
        if !self.emitted_visible {
            consider(
                find_bare_gemma4_call(&self.buffer),
                ToolSpanKind::BareGemma4,
            );
        }
        earliest
    }

    fn extract_span_at_start(
        &self,
        kind: ToolSpanKind,
        at_end: bool,
    ) -> Option<(OpenAiFunctionCall, String)> {
        match kind {
            ToolSpanKind::Xml if at_end => extract_xml_tool_call_payload_at(&self.buffer, 0),
            ToolSpanKind::Xml => extract_closed_xml_tool_call_payload_at(&self.buffer, 0),
            ToolSpanKind::Gemma4 => extract_gemma4_tool_call_payload_at(&self.buffer, 0),
            ToolSpanKind::BareGemma4 => extract_bare_gemma4_tool_call_payload_at(&self.buffer, 0),
            // DSML stanzas are parsed whole in `drain_events` (several calls).
            ToolSpanKind::Dsml => None,
        }
    }

    /// Absolute byte offset just past the opening `{` of a bare
    /// `call:NAME{...}` body, once that brace has arrived.
    fn bare_body_start(&self) -> Option<usize> {
        let rest = self.buffer.strip_prefix(BARE_GEMMA4_LEAD)?;
        let brace = rest.find('{')?;
        Some(BARE_GEMMA4_LEAD.len() + brace + 1)
    }

    /// A bare span (`call:NAME{...}`) is abandoned as soon as the name region
    /// contains a character that can never appear in a tool name — otherwise
    /// prose that merely starts with `call:` would be withheld to end of
    /// stream.
    fn bare_span_still_viable(&self) -> bool {
        let Some(rest) = self.buffer.strip_prefix(BARE_GEMMA4_LEAD) else {
            return false;
        };
        let name_region = rest.split('{').next().unwrap_or(rest);
        name_region
            .trim()
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    }

    /// Longest buffer suffix that could still become a marker: a proper
    /// prefix of an opener, or — before any visible output — a leading
    /// whitespace run whose tail is a prefix of `call:`. Markers are ASCII,
    /// so the holdback always lands on a char boundary.
    fn opener_holdback_len(&self) -> usize {
        let bytes = self.buffer.as_bytes();
        let mut hold = 0usize;
        for opener in [XML_OPEN, GEMMA4_OPEN] {
            let opener = opener.as_bytes();
            let max = opener.len().saturating_sub(1).min(bytes.len());
            for len in (hold + 1..=max).rev() {
                if bytes[bytes.len() - len..] == opener[..len] {
                    hold = len;
                    break;
                }
            }
        }
        if !self.emitted_visible {
            let trimmed = self.buffer.trim_start();
            if !trimmed.is_empty()
                && trimmed.len() < BARE_GEMMA4_LEAD.len()
                && BARE_GEMMA4_LEAD.starts_with(trimmed)
            {
                hold = hold.max(trimmed.len());
            }
        }
        // DSML openers tolerate filler, so their holdback is computed by the
        // DSML matcher rather than by byte-prefix comparison. The `<` it
        // withholds is a char boundary; the rest of the suffix is whole chars.
        hold.max(dsml::partial_dsml_tool_calls_open_len(&self.buffer))
    }

    fn note_visible(&mut self, content: &str) {
        if !content.trim().is_empty() {
            self.emitted_visible = true;
        }
    }

    fn build_call(&mut self, mut function: OpenAiFunctionCall) -> OpenAiToolCall {
        if let Some(contract) = self.contract.as_deref() {
            function.name = contract.canonical_tool_name(&function.name);
            function.arguments = contract.canonical_arguments(&function.name, function.arguments);
        }
        let call = OpenAiToolCall {
            id: format!("call_{}", self.calls_emitted),
            tool_type: "function",
            function,
        };
        self.calls_emitted += 1;
        call
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    fn scanner() -> ToolCallStreamScanner {
        ToolCallStreamScanner::new(None)
    }

    fn content(events: &[ToolScanEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                ToolScanEvent::Content(text) => Some(text.as_str()),
                ToolScanEvent::Call(_) => None,
            })
            .collect()
    }

    fn calls(events: &[ToolScanEvent]) -> Vec<&OpenAiToolCall> {
        events
            .iter()
            .filter_map(|event| match event {
                ToolScanEvent::Call(call) => Some(call),
                ToolScanEvent::Content(_) => None,
            })
            .collect()
    }

    #[test]
    fn plain_content_streams_through() {
        let mut scanner = scanner();
        let events = scanner.push("hello world");
        assert_eq!(content(&events), "hello world");
        assert!(calls(&events).is_empty());
        assert!(scanner.finish().is_empty());
    }

    #[test]
    fn xml_call_with_surrounding_content() {
        let mut scanner = scanner();
        let mut events = scanner.push(
            "before <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Taipei\"}}</tool_call> after",
        );
        events.extend(scanner.finish());
        assert_eq!(content(&events), "before  after");
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].function.name, "get_weather");
        let arguments: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).expect("json arguments");
        assert_eq!(arguments, json!({"city": "Taipei"}));
    }

    #[test]
    fn opener_split_across_pushes_never_leaks_marker_text() {
        let mut scanner = scanner();
        let first = scanner.push("text <tool_");
        assert_eq!(content(&first), "text ");
        let second = scanner.push("call>{\"name\":\"f\",\"arguments\":{}}</tool_");
        assert_eq!(content(&second), "");
        let third = scanner.push("call>");
        assert_eq!(calls(&third).len(), 1);
        assert_eq!(content(&third), "");
        assert!(scanner.finish().is_empty());
    }

    #[test]
    fn two_calls_get_sequential_indices() {
        let mut scanner = scanner();
        let mut events = scanner.push(
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>",
        );
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(scanner.calls_emitted(), 2);
    }

    #[test]
    fn gemma4_dsl_call_parses() {
        let mut scanner = scanner();
        let mut events =
            scanner.push("<|tool_call>call:lookup{query:<|\"|>rust<|\"|>}<tool_call|>done");
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(content(&events), "done");
    }

    #[test]
    fn bare_gemma4_leading_call_parses() {
        let mut scanner = scanner();
        let mut events = scanner.push("call:ping{host:<|\"|>ax<|\"|>}");
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "ping");
    }

    #[test]
    fn bare_call_after_visible_content_is_plain_text() {
        let mut scanner = scanner();
        let mut events = scanner.push("I will call:ping{host:x}");
        events.extend(scanner.finish());
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "I will call:ping{host:x}");
    }

    #[test]
    fn prose_starting_with_call_is_not_withheld_forever() {
        let mut scanner = scanner();
        let mut events = scanner.push("call: me maybe, later today");
        events.extend(scanner.finish());
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "call: me maybe, later today");
    }

    #[test]
    fn tiel_missing_outer_json_brace_is_not_an_executable_call() {
        let malformed = "<tool_call>\n{\"name\":\"bash\",\"arguments\":{\"command\":\"cloc --include-ext=py\",\"description\":\"Count Python lines\"}\n</tool_call>";
        let mut scanner = scanner();
        let mut events = Vec::new();
        for chunk in malformed.as_bytes().chunks(7) {
            events.extend(scanner.push(std::str::from_utf8(chunk).expect("ASCII fixture")));
        }
        events.extend(scanner.finish());
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), malformed);
    }

    #[test]
    fn restarted_call_inside_invalid_span_is_extracted_at_end() {
        // Non-streaming extraction tries every opener; the stream scanner
        // must not swallow the valid restarted call when the outer span
        // never parses.
        let mut scanner = scanner();
        let mut events = scanner
            .push("<tool_call>oops <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>");
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "b");
        assert_eq!(content(&events), "<tool_call>oops ");
    }

    const DSML_CALL: &str = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Taipei</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";

    #[test]
    fn dsml_stanza_streams_as_one_call_with_surrounding_content() {
        let mut scanner = scanner();
        let mut events = scanner.push(&format!("before {DSML_CALL} after"));
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let arguments: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).expect("json arguments");
        assert_eq!(arguments, json!({"city": "Taipei"}));
        assert_eq!(content(&events), "before  after");
    }

    #[test]
    fn dsml_stanza_is_invariant_across_all_chunk_splits() {
        let text = format!("x {DSML_CALL} y");
        for split in 1..text.len() {
            if !text.is_char_boundary(split) {
                continue;
            }
            let mut scanner = scanner();
            let mut events = scanner.push(&text[..split]);
            events.extend(scanner.push(&text[split..]));
            events.extend(scanner.finish());
            assert_eq!(calls(&events).len(), 1, "split {split}");
            assert_eq!(content(&events), "x  y", "split {split}");
            assert!(!content(&events).contains("DSML"), "split {split}");
        }
    }

    #[test]
    fn dsml_stanza_with_two_invokes_emits_two_calls() {
        let two = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"a\"></｜DSML｜invoke><｜DSML｜invoke name=\"b\"></｜DSML｜invoke></｜DSML｜tool_calls>";
        let mut scanner = scanner();
        let mut events = scanner.push(two);
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn dsml_prose_with_a_lone_angle_bracket_is_not_withheld_forever() {
        let mut scanner = scanner();
        let mut events = scanner.push("a < b and <｜DSM");
        // The partial opener is withheld until it can be classified.
        assert_eq!(content(&events), "a < b and ");
        events.extend(scanner.push("X> is prose"));
        events.extend(scanner.finish());
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "a < b and <｜DSMX> is prose");
    }

    #[test]
    fn malformed_dsml_stanza_is_flushed_as_content_at_end() {
        let text = "<｜DSML｜tool_calls>garbage</｜DSML｜tool_calls>tail";
        let mut scanner = scanner();
        let mut events = scanner.push(text);
        events.extend(scanner.finish());
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), text);
    }

    #[test]
    fn dsml_empty_stanza_streams_prose_before_eos() {
        let bar = "\u{FF5C}";
        let empty_stanza = format!("<{bar}DSML{bar}tool_calls></{bar}DSML{bar}tool_calls>");
        let mut scanner = scanner();
        let first = scanner.push(&empty_stanza);
        // The empty stanza yields no call and is released immediately (not held
        // back to EOS), so following prose streams as content deltas before EOS.
        assert!(calls(&first).is_empty());
        let mut events = scanner.push("hello");
        events.extend(scanner.push(" world"));
        assert_eq!(content(&events), "hello world");
        assert!(scanner.finish().is_empty());
    }

    #[test]
    fn dsml_closer_inside_argument_string_does_not_prematurely_complete() {
        let bar = "\u{FF5C}";
        let stanza = format!(
            "<{bar}DSML{bar}tool_calls><{bar}DSML{bar}invoke name=\"echo\"><{bar}DSML{bar}parameter name=\"text\" string=\"true\">a</{bar}DSML{bar}tool_calls>b</{bar}DSML{bar}parameter></{bar}DSML{bar}invoke></{bar}DSML{bar}tool_calls>"
        );
        let mut scanner = scanner();
        let mut events = scanner.push(&stanza);
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1, "content: {:?}", content(&events));
        assert_eq!(calls[0].function.name, "echo");
        assert!(
            calls[0].function.arguments.contains("a</"),
            "arguments should preserve the embedded closer: {}",
            calls[0].function.arguments
        );
    }

    #[test]
    fn invalid_span_with_closer_is_flushed_as_content() {
        let mut scanner = scanner();
        let mut events = scanner.push("<|tool_call>not a call<tool_call|>tail");
        events.extend(scanner.finish());
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "<|tool_call>not a call<tool_call|>tail");
    }

    #[test]
    fn xml_call_with_closer_inside_argument_string_survives_chunking() {
        let mut scanner = scanner();
        let first =
            scanner.push("<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"a</tool_call>");
        assert!(calls(&first).is_empty());
        assert!(
            content(&first).is_empty(),
            "inner marker must stay withheld"
        );
        let mut events = scanner.push("b\"}}</tool_call>");
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert!(calls[0].function.arguments.contains("a</tool_call>b"));
    }

    #[test]
    fn xml_inner_closer_does_not_complete_a_streaming_call() {
        let body = r#"<tool_call>{"name":"echo","arguments":{"text":"a</tool_call>b"}}"#;
        let mut scanner = scanner();
        assert!(scanner.push(body).is_empty(), "wait for the actual closer");
        let mut events = scanner.push("</tool_call>tail");
        events.extend(scanner.finish());
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(content(&events), "tail");
    }

    #[test]
    fn xml_inner_closer_is_invariant_across_all_chunk_splits() {
        let text =
            r#"<tool_call>{"name":"echo","arguments":{"text":"a</tool_call>b"}}</tool_call>tail"#;
        for split in 0..=text.len() {
            let mut scanner = scanner();
            let mut events = scanner.push(&text[..split]);
            events.extend(scanner.push(&text[split..]));
            events.extend(scanner.finish());
            assert_eq!(calls(&events).len(), 1, "split {split}");
            assert_eq!(content(&events), "tail", "split {split}");
            assert_eq!(
                calls(&events)[0].function.arguments,
                r#"{"text":"a</tool_call>b"}"#,
                "split {split}"
            );
        }
    }

    #[test]
    fn unterminated_xml_call_parses_at_finish() {
        let mut scanner = scanner();
        let first = scanner.push("<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":1}}");
        assert!(first.is_empty());
        let events = scanner.finish();
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "f");
    }

    #[test]
    fn unterminated_gemma4_span_flushes_as_content_at_finish() {
        let mut scanner = scanner();
        let _ = scanner.push("<|tool_call>call:f{");
        let events = scanner.finish();
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "<|tool_call>call:f{");
    }

    #[test]
    fn qwen_function_xml_body_parses() {
        let mut scanner = scanner();
        let mut events = scanner.push(
            "<tool_call><function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function></tool_call>",
        );
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        let arguments: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).expect("json arguments");
        assert_eq!(arguments, json!({"path": "src/main.rs"}));
    }

    #[test]
    fn dsml_stanza_then_xml_opener_is_content_not_a_second_call() {
        // After a DSML stanza the non-streaming extractor hands the leftover
        // back as content and never re-runs the XML extractor on it, so a
        // later `<tool_call>` block is text. The stream scanner must agree:
        // one call (the DSML invoke), and the XML block survives as content.
        let bar = "\u{FF5C}";
        let stanza = format!(
            "<{bar}DSML{bar}tool_calls><{bar}DSML{bar}invoke name=\"a\"></{bar}DSML{bar}invoke></{bar}DSML{bar}tool_calls>"
        );
        let xml = "<tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let text = format!("{stanza}{xml}");

        let (non_streaming, leftover) =
            dsml::parse_dsml_tool_calls(&text).expect("DSML stanza parses");
        assert_eq!(non_streaming.len(), 1);
        assert_eq!(non_streaming[0].name.as_str(), "a");
        assert_eq!(leftover, xml);

        let mut scanner = scanner();
        let mut events = Vec::new();
        for (index, ch) in text.char_indices() {
            events.extend(scanner.push(&text[index..index + ch.len_utf8()]));
        }
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1, "content: {:?}", content(&events));
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(content(&events), xml);
    }

    #[test]
    fn long_xml_argument_streams_linearly_and_matches_non_streaming() {
        // A 64 KiB JSON argument streamed in 4-byte chunks must complete in
        // linear time (the per-span resume offsets avoid re-scanning the whole
        // growing buffer every push) and reproduce the whole-string extract.
        let argument = "x".repeat(64 * 1024);
        let body = format!("{{\"name\":\"f\",\"arguments\":{{\"payload\":\"{argument}\"}}}}");
        let text = format!("<tool_call>{body}</tool_call>");

        let (function, _) =
            extract_xml_tool_call_payload_at(&text, 0).expect("whole-string XML call extracts");

        let mut scanner = scanner();
        let mut events = Vec::new();
        let started = std::time::Instant::now();
        for chunk in text.as_bytes().chunks(4) {
            events.extend(scanner.push(std::str::from_utf8(chunk).expect("ASCII fixture")));
        }
        events.extend(scanner.finish());
        let elapsed = started.elapsed();

        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, function.name);
        assert_eq!(calls[0].function.arguments, function.arguments);
        assert!(content(&events).is_empty());
        assert!(
            elapsed.as_secs_f64() < 1.0,
            "64 KiB in 4-byte chunks must not rescan quadratically: {elapsed:?}"
        );
    }
}
