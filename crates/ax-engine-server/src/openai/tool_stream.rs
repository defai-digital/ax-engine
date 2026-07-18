//! Incremental tool-call scanning for native chat streams (ADR-040 D1).
//!
//! Content outside tool-call spans streams live; each completed call is
//! parsed by the same extractors the non-streaming path uses
//! (`openai/responses.rs`) and emitted as one spec-conformant
//! `delta.tool_calls` fragment. Marker text is withheld with a bounded
//! holdback so a marker split across token boundaries never leaks into
//! content and never stalls the stream.

use std::sync::Arc;

use crate::openai::requests::OpenAiToolContract;
use crate::openai::responses::{
    extract_bare_gemma4_tool_call_payload_at, extract_gemma4_tool_call_payload_at,
    extract_xml_tool_call_payload_at, find_bare_gemma4_call,
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
}

pub(crate) struct ToolCallStreamScanner {
    /// Withheld text: an open span plus at most one partial opener suffix.
    buffer: String,
    /// When set, the span starts at `buffer[0]`.
    span: Option<ToolSpanKind>,
    /// True once any non-whitespace content has been emitted; gates the
    /// bare-Gemma4 form, which is only valid as the leading output.
    emitted_visible: bool,
    calls_emitted: u32,
    contract: Option<Arc<OpenAiToolContract>>,
}

impl ToolCallStreamScanner {
    pub(crate) fn new(contract: Option<Arc<OpenAiToolContract>>) -> Self {
        Self {
            buffer: String::new(),
            span: None,
            emitted_visible: false,
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
            && let Some((function, remaining)) = self.extract_span_at_start(kind)
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
                Some(kind) => {
                    let closer = match kind {
                        ToolSpanKind::Xml => Some(XML_CLOSE),
                        ToolSpanKind::Gemma4 => Some(GEMMA4_CLOSE),
                        ToolSpanKind::BareGemma4 => None,
                    };
                    if let Some(closer) = closer {
                        let Some(close_at) = self.buffer.find(closer) else {
                            return events;
                        };
                        match self.extract_span_at_start(kind) {
                            Some((function, remaining)) => {
                                events.push(ToolScanEvent::Call(self.build_call(function)));
                                self.buffer = remaining;
                            }
                            None => {
                                // Closer present but not a valid call: flush
                                // the span through the closer and resume.
                                let end = close_at + closer.len();
                                let content = self.buffer[..end].to_string();
                                self.buffer.drain(..end);
                                self.note_visible(&content);
                                events.push(ToolScanEvent::Content(content));
                            }
                        }
                        self.span = None;
                        continue;
                    }
                    // Bare Gemma4: no closer marker; complete when the brace
                    // matcher inside the extractor succeeds.
                    if let Some((function, remaining)) =
                        self.extract_span_at_start(ToolSpanKind::BareGemma4)
                    {
                        events.push(ToolScanEvent::Call(self.build_call(function)));
                        self.buffer = remaining;
                        self.span = None;
                        continue;
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

    fn find_earliest_opener(&self) -> Option<(usize, ToolSpanKind)> {
        let mut earliest: Option<(usize, ToolSpanKind)> = None;
        let mut consider = |candidate: Option<usize>, kind: ToolSpanKind| {
            if let Some(index) = candidate
                && earliest.is_none_or(|(at, _)| index < at)
            {
                earliest = Some((index, kind));
            }
        };
        consider(self.buffer.find(XML_OPEN), ToolSpanKind::Xml);
        consider(self.buffer.find(GEMMA4_OPEN), ToolSpanKind::Gemma4);
        if !self.emitted_visible {
            consider(
                find_bare_gemma4_call(&self.buffer),
                ToolSpanKind::BareGemma4,
            );
        }
        earliest
    }

    fn extract_span_at_start(&self, kind: ToolSpanKind) -> Option<(OpenAiFunctionCall, String)> {
        match kind {
            ToolSpanKind::Xml => extract_xml_tool_call_payload_at(&self.buffer, 0),
            ToolSpanKind::Gemma4 => extract_gemma4_tool_call_payload_at(&self.buffer, 0),
            ToolSpanKind::BareGemma4 => extract_bare_gemma4_tool_call_payload_at(&self.buffer, 0),
        }
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
        hold
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
    fn invalid_span_with_closer_is_flushed_as_content() {
        let mut scanner = scanner();
        let mut events = scanner.push("<|tool_call>not a call<tool_call|>tail");
        events.extend(scanner.finish());
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "<|tool_call>not a call<tool_call|>tail");
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
}
