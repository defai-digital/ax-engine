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
    /// One piece of `function.arguments`. `id` and `name` are set only on
    /// the first piece for a call.
    Arguments {
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
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
    /// Calls that qualify a stream to finish `tool_calls`: an intact emitted
    /// call, or a canonical continuation applied onto a streamed header. A
    /// rejected continuation does not count.
    parsed_calls: u32,
    /// Times the canonical arguments were not a continuation of the text
    /// already streamed. The call closes on that streamed text.
    argument_continuations_rejected: u32,
    contract: Option<Arc<OpenAiToolContract>>,
    /// When true, a JSON `<tool_call>` whose arguments stay a prefix of the
    /// canonical arguments text is emitted as argument deltas. `new` leaves
    /// this off so callers keep a single final chunk. Production streaming
    /// turns it on only for `AX_API_INCREMENTAL_TOOL_ARGS`.
    incremental: bool,
    /// Header already sent for the open span. A later `Call` is suppressed.
    header_sent: bool,
    /// Client-visible argument text already emitted for the open span.
    streamed_arguments: String,
    /// This span cannot stream raw arguments (the parser cannot delimit a
    /// contiguous arguments substring, or canonicalization would rewrite it).
    incremental_blocked: bool,
}

impl ToolCallStreamScanner {
    /// One-shot scanner used by tests of the pre-polish path. Production
    /// streaming builds the scanner with `with_incremental`.
    #[cfg(test)]
    pub(crate) fn new(contract: Option<Arc<OpenAiToolContract>>) -> Self {
        Self::with_incremental(contract, false)
    }

    pub(crate) fn with_incremental(
        contract: Option<Arc<OpenAiToolContract>>,
        incremental: bool,
    ) -> Self {
        Self {
            buffer: String::new(),
            span: None,
            emitted_visible: false,
            dsml_emitted: false,
            span_scan_offset: 0,
            bare_scan_from: 0,
            bare_scan_depth: 0,
            calls_emitted: 0,
            parsed_calls: 0,
            argument_continuations_rejected: 0,
            contract,
            incremental,
            header_sent: false,
            streamed_arguments: String::new(),
            incremental_blocked: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn calls_emitted(&self) -> u32 {
        self.calls_emitted
    }

    /// Calls that qualify the stream to finish `tool_calls`. A header whose
    /// canonical continuation was rejected is excluded, as is a span that
    /// never parsed.
    pub(crate) fn parsed_calls(&self) -> u32 {
        self.parsed_calls
    }

    #[cfg(test)]
    pub(crate) fn argument_continuations_rejected(&self) -> u32 {
        self.argument_continuations_rejected
    }

    #[cfg(test)]
    fn testing_overwrite_streamed_arguments(&mut self, text: &str) {
        self.header_sent = true;
        self.streamed_arguments = text.to_string();
    }

    pub(crate) fn push(&mut self, text: &str) -> Vec<ToolScanEvent> {
        self.buffer.push_str(text);
        self.drain_events(false)
    }

    /// True while text is withheld (an open span or a partial opener), i.e.
    /// a later push may still turn the buffered text into a call. The stream
    /// stop scanner uses this to avoid terminating on a stop match that
    /// would strand the withheld call.
    pub(crate) fn has_withheld_text(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// End of stream: parse an unterminated span if the family grammar allows
    /// it (the XML extractor tolerates a missing closer), otherwise flush the
    /// withheld text as content. Never drops text, never leaves a call half
    /// emitted.
    ///
    /// Incremental deltas default off, so a span that fails here never sent a
    /// header and the caller does not finish `tool_calls`. With
    /// `AX_API_INCREMENTAL_TOOL_ARGS` on, a header may already have been
    /// streamed; a span that still fails to parse does not increment
    /// [`Self::parsed_calls`], and the stream must not claim `tool_calls`.
    pub(crate) fn finish(&mut self) -> Vec<ToolScanEvent> {
        let mut events = self.drain_events(true);
        if let Some(kind) = self.span.take()
            && !self.buffer.is_empty()
            && let Some((function, remaining)) = self.extract_span_at_start(kind, true)
        {
            self.push_parsed_call(function, &mut events);
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
                        self.header_sent = false;
                        self.streamed_arguments.clear();
                        self.incremental_blocked = false;
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
                                    self.push_parsed_call(function, &mut events);
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
                    // No closer parsed a complete stanza. A still-open invoke
                    // means the first closer sat inside the stanza (a literal
                    // closer in a parameter value), so more data can still
                    // complete it: keep withholding, like the XML branch, and
                    // let end of stream (`finish`) own the final disposition.
                    // Otherwise release through the first closer as content
                    // (stopping at a later valid opener so a restarted call can
                    // still be rescanned, as the XML branch does at end of
                    // stream) instead of withholding the whole stream until
                    // EOS. With no closer at all we keep withholding and wait
                    // for more data.
                    let Some((_, first_close_end)) =
                        dsml::find_dsml_tool_calls_close(&self.buffer, 0)
                    else {
                        return events;
                    };
                    let end = self
                        .next_opener_after_start(first_close_end)
                        .map_or(first_close_end, |inner| inner.min(first_close_end));
                    if !at_end && dsml::window_has_unclosed_invoke(&self.buffer[..end]) {
                        return events;
                    }
                    let content = self.buffer[..end].to_string();
                    self.buffer.drain(..end);
                    self.note_visible(&content);
                    events.push(ToolScanEvent::Content(content));
                    self.span = None;
                    continue;
                }
                Some(kind) => {
                    if self.incremental
                        && matches!(kind, ToolSpanKind::Xml)
                        && !self.incremental_blocked
                    {
                        self.poll_xml_argument_deltas(&mut events);
                    }
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
                                    self.push_parsed_call(function, &mut events);
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
                            // The remembered resume offset can land inside a
                            // multi-byte UTF-8 character; the next push slices
                            // `&buffer[offset..]`, so round it down to the
                            // nearest char boundary first.
                            let mut offset = self
                                .buffer
                                .len()
                                .saturating_sub(closer.len().saturating_sub(1));
                            while !self.buffer.is_char_boundary(offset) {
                                offset -= 1;
                            }
                            self.span_scan_offset = offset;
                            return events;
                        };
                        let close_at = self.span_scan_offset + relative;
                        self.span_scan_offset = close_at + closer.len();
                        // The extractor re-tries every closer from the start,
                        // so call it only when a new closer arrived; a body
                        // that has not parsed yet stays withheld.
                        if let Some((function, remaining)) = self.extract_span_at_start(kind, false)
                        {
                            self.push_parsed_call(function, &mut events);
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
                                    self.push_parsed_call(function, &mut events);
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

    fn push_parsed_call(
        &mut self,
        mut function: OpenAiFunctionCall,
        events: &mut Vec<ToolScanEvent>,
    ) {
        if let Some(contract) = self.contract.as_deref() {
            function.name = contract.canonical_tool_name(&function.name);
            function.arguments = contract.canonical_arguments(&function.name, function.arguments);
        }
        if self.header_sent {
            if let Some(rest) = function.arguments.strip_prefix(&self.streamed_arguments) {
                // Applied continuation: the client already has a prefix of
                // the canonical arguments, so this call qualifies.
                self.parsed_calls = self.parsed_calls.saturating_add(1);
                if !rest.is_empty() {
                    events.push(ToolScanEvent::Arguments {
                        id: None,
                        name: None,
                        arguments: rest.to_string(),
                    });
                }
            } else {
                // The canonical text is not a continuation of what the client
                // already has. Appending it would corrupt the JSON. Close on
                // the streamed prefix and record the rejection. Do not count
                // the call: the stream must not finish `tool_calls`.
                tracing::warn!(
                    streamed_bytes = self.streamed_arguments.len(),
                    canonical_bytes = function.arguments.len(),
                    "incremental tool arguments are not a prefix of the canonical arguments; closing the call on the streamed text"
                );
                self.argument_continuations_rejected =
                    self.argument_continuations_rejected.saturating_add(1);
            }
            self.header_sent = false;
            self.streamed_arguments.clear();
            return;
        }
        self.parsed_calls = self.parsed_calls.saturating_add(1);
        events.push(ToolScanEvent::Call(self.build_call(function)));
    }

    /// Emit argument deltas for an open JSON `<tool_call>` span. A span that
    /// is not JSON, or whose arguments would be rewritten, is left for the
    /// one-shot `Call` path.
    fn poll_xml_argument_deltas(&mut self, events: &mut Vec<ToolScanEvent>) {
        let Some(body) = self.buffer.strip_prefix(XML_OPEN) else {
            return;
        };
        let Some(progress) = xml_json_argument_progress(body) else {
            if !body.trim_start().is_empty() && !body.trim_start().starts_with('{') {
                self.incremental_blocked = true;
            }
            return;
        };
        let Some(name) = progress.name else {
            return;
        };
        if !progress.live {
            if progress.complete {
                self.incremental_blocked = true;
            }
            return;
        }
        if !progress.arguments_started {
            return;
        }
        let name = if let Some(contract) = self.contract.as_deref() {
            if contract.arguments_may_be_rewritten(&name) {
                self.incremental_blocked = true;
                return;
            }
            contract.canonical_tool_name(&name)
        } else {
            name
        };
        let visible = progress.visible;
        if !self.header_sent {
            let id = format!("call_{}", self.calls_emitted);
            self.calls_emitted += 1;
            let piece = if visible.starts_with(&self.streamed_arguments) {
                visible[self.streamed_arguments.len()..].to_string()
            } else {
                visible.clone()
            };
            events.push(ToolScanEvent::Arguments {
                id: Some(id),
                name: Some(name),
                arguments: piece,
            });
            self.header_sent = true;
            self.streamed_arguments = visible;
            return;
        }
        if visible.starts_with(&self.streamed_arguments)
            && visible.len() > self.streamed_arguments.len()
        {
            let piece = visible[self.streamed_arguments.len()..].to_string();
            events.push(ToolScanEvent::Arguments {
                id: None,
                name: None,
                arguments: piece,
            });
            self.streamed_arguments = visible;
        }
    }
}

struct XmlArgProgress {
    name: Option<String>,
    arguments_started: bool,
    visible: String,
    complete: bool,
    /// The visible text is the client-visible arguments string (raw when
    /// that string will not be rewritten).
    live: bool,
}

fn xml_json_argument_progress(body: &str) -> Option<XmlArgProgress> {
    let trimmed = skip_ws(body, 0);
    if trimmed >= body.len() {
        return Some(XmlArgProgress {
            name: None,
            arguments_started: false,
            visible: String::new(),
            complete: false,
            live: true,
        });
    }
    if body.as_bytes()[trimmed] != b'{' {
        return None;
    }
    let mut index = trimmed + 1;
    let mut name = None;
    let mut arguments_started = false;
    let mut visible = String::new();
    let mut complete = false;
    let mut live = true;
    loop {
        index = skip_ws(body, index);
        if index >= body.len() {
            break;
        }
        match body.as_bytes()[index] {
            b'}' => break,
            b',' => {
                index += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let (key, next) = parse_json_string(body, index)?;
        index = skip_ws(body, next);
        if index >= body.len() || body.as_bytes()[index] != b':' {
            break;
        }
        index = skip_ws(body, index + 1);
        if index >= body.len() {
            if key == "arguments" {
                arguments_started = true;
            }
            break;
        }
        if key == "name" && name.is_none() {
            let (value, next, string_complete) = partial_json_string(body, index)?;
            if !string_complete {
                break;
            }
            name = Some(value);
            index = next;
            continue;
        }
        if key == "arguments" && !arguments_started {
            arguments_started = true;
            let progress = partial_json_value(body, index)?;
            visible = progress.visible;
            complete = progress.complete;
            live = progress.live;
            break;
        }
        let progress = partial_json_value(body, index)?;
        if !progress.complete {
            break;
        }
        index = progress.end;
    }
    Some(XmlArgProgress {
        name,
        arguments_started,
        visible,
        complete,
        live,
    })
}

struct ValueProgress {
    visible: String,
    end: usize,
    complete: bool,
    live: bool,
}

fn partial_json_value(body: &str, start: usize) -> Option<ValueProgress> {
    if start >= body.len() {
        return None;
    }
    match body.as_bytes()[start] {
        b'"' => {
            let (decoded, end, complete) = partial_json_string(body, start)?;
            Some(ValueProgress {
                visible: decoded,
                end,
                complete,
                live: true,
            })
        }
        b'{' | b'[' => partial_json_container(body, start, 0),
        b't' | b'f' | b'n' | b'-' | b'0'..=b'9' => partial_json_literal(body, start),
        _ => None,
    }
}

fn partial_json_literal(body: &str, start: usize) -> Option<ValueProgress> {
    let rest = &body[start..];
    if rest.starts_with("true") || rest.starts_with("false") || rest.starts_with("null") {
        let word = if rest.starts_with("true") {
            "true"
        } else if rest.starts_with("false") {
            "false"
        } else {
            "null"
        };
        return Some(ValueProgress {
            visible: word.to_string(),
            end: start + word.len(),
            complete: true,
            live: true,
        });
    }
    if rest.starts_with('t') || rest.starts_with('f') || rest.starts_with('n') {
        return Some(ValueProgress {
            visible: String::new(),
            end: start,
            complete: false,
            live: true,
        });
    }
    let mut end = start;
    let bytes = body.as_bytes();
    if bytes.get(end) == Some(&b'-') {
        end += 1;
    }
    let mut saw_digit = false;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        saw_digit = true;
        end += 1;
    }
    if end < bytes.len() && bytes[end] == b'.' {
        end += 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            saw_digit = true;
            end += 1;
        }
    }
    if !saw_digit {
        return Some(ValueProgress {
            visible: String::new(),
            end: start,
            complete: false,
            live: true,
        });
    }
    let raw = body[start..end].to_string();
    let complete = end < bytes.len()
        && !bytes[end].is_ascii_digit()
        && bytes[end] != b'.'
        && bytes[end] != b'e'
        && bytes[end] != b'E';
    Some(ValueProgress {
        visible: raw,
        end,
        complete,
        live: true,
    })
}

/// JSON string as `serde_json` would emit it. An incomplete string omits the
/// closing quote so a later chunk can extend the same literal.
fn canonical_json_string(decoded: &str, complete: bool) -> String {
    let mut encoded = serde_json::to_string(decoded).unwrap_or_else(|_| "\"\"".to_string());
    if !complete {
        encoded.pop();
    }
    encoded
}

fn canonical_nested_value(body: &str, start: usize, depth: usize) -> Option<ValueProgress> {
    match body.as_bytes().get(start).copied()? {
        b'"' => {
            let (decoded, end, complete) = partial_json_string(body, start)?;
            Some(ValueProgress {
                visible: canonical_json_string(&decoded, complete),
                end,
                complete,
                live: true,
            })
        }
        b'{' | b'[' => partial_json_container(body, start, depth.saturating_add(1)),
        b't' | b'f' | b'n' | b'-' | b'0'..=b'9' => {
            let progress = partial_json_literal(body, start)?;
            if !progress.complete {
                return Some(ValueProgress {
                    visible: String::new(),
                    end: progress.end,
                    complete: false,
                    live: progress.live,
                });
            }
            let visible = serde_json::from_str::<serde_json::Value>(&progress.visible)
                .ok()
                .and_then(|value| serde_json::to_string(&value).ok())
                .unwrap_or(progress.visible);
            Some(ValueProgress {
                visible,
                end: progress.end,
                complete: true,
                live: true,
            })
        }
        _ => None,
    }
}

/// Canonical JSON for a container, matching `serde_json::to_string` on the
/// parsed value (sorted keys, compact, serde string escapes). Whitespace or
/// out-of-order keys set `live` false so the one-shot call path owns them.
fn partial_json_container(body: &str, start: usize, depth: usize) -> Option<ValueProgress> {
    if depth > 64 {
        return None;
    }
    let bytes = body.as_bytes();
    let opening = *bytes.get(start)?;
    let closing = match opening {
        b'{' => b'}',
        b'[' => b']',
        _ => return None,
    };
    let mut out = String::new();
    out.push(char::from(opening));
    let mut index = start + 1;
    let mut live = true;
    let mut complete = false;
    let mut need_comma = false;
    let mut last_key: Option<String> = None;
    loop {
        let skipped = skip_ws(body, index);
        if skipped != index {
            live = false;
        }
        index = skipped;
        if index >= body.len() {
            break;
        }
        let byte = bytes[index];
        if byte == closing {
            out.push(char::from(closing));
            index += 1;
            complete = true;
            break;
        }
        if byte == b',' {
            if !need_comma {
                return None;
            }
            index += 1;
            need_comma = false;
            continue;
        }
        if opening == b'{' {
            if byte != b'"' {
                return None;
            }
            let (key, next, key_complete) = partial_json_string(body, index)?;
            if !key_complete {
                break;
            }
            if last_key
                .as_ref()
                .is_some_and(|previous| key.as_str() < previous.as_str())
            {
                live = false;
            }
            last_key = Some(key.clone());
            index = next;
            let after_key = skip_ws(body, index);
            if after_key != index {
                live = false;
            }
            index = after_key;
            if index >= body.len() || bytes[index] != b':' {
                break;
            }
            index += 1;
            let after_colon = skip_ws(body, index);
            if after_colon != index {
                live = false;
            }
            index = after_colon;
            if index >= body.len() {
                break;
            }
            let value = canonical_nested_value(body, index, depth)?;
            if !value.live {
                live = false;
            }
            if value.visible.is_empty() && !value.complete {
                break;
            }
            if need_comma {
                out.push(',');
            }
            out.push_str(&canonical_json_string(&key, true));
            out.push(':');
            out.push_str(&value.visible);
            index = value.end;
            if !value.complete {
                break;
            }
            need_comma = true;
            continue;
        }
        let value = canonical_nested_value(body, index, depth)?;
        if !value.live {
            live = false;
        }
        if value.visible.is_empty() && !value.complete {
            break;
        }
        if need_comma {
            out.push(',');
        }
        out.push_str(&value.visible);
        index = value.end;
        if !value.complete {
            break;
        }
        need_comma = true;
    }
    if !live {
        return Some(ValueProgress {
            visible: String::new(),
            end: index,
            complete,
            live: false,
        });
    }
    Some(ValueProgress {
        visible: out,
        end: index,
        complete,
        live: true,
    })
}

fn parse_json_string(body: &str, start: usize) -> Option<(String, usize)> {
    let (decoded, end, complete) = partial_json_string(body, start)?;
    complete.then_some((decoded, end))
}

fn partial_json_string(body: &str, start: usize) -> Option<(String, usize, bool)> {
    if body.as_bytes().get(start) != Some(&b'"') {
        return None;
    }
    let mut decoded = String::new();
    let bytes = body.as_bytes();
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => return Some((decoded, index + 1, true)),
            b'\\' => {
                if index + 1 >= bytes.len() {
                    return Some((decoded, index, false));
                }
                let escaped = match bytes[index + 1] {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{0008}',
                    b'f' => '\u{000c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        // A short or still-hex tail is an unfinished escape.
                        // A non-hex byte makes the literal invalid.
                        let available = body.get(index + 2..).unwrap_or_default();
                        let hex_len = available
                            .bytes()
                            .take(4)
                            .take_while(|byte| byte.is_ascii_hexdigit())
                            .count();
                        if hex_len < 4 {
                            // The tail is still hex (or empty): hold the escape.
                            // A non-hex byte ends the literal.
                            if available.len() == hex_len {
                                return Some((decoded, index, false));
                            }
                            return None;
                        }
                        let hex = &available[..4];
                        let code = u32::from_str_radix(hex, 16).ok()?;
                        let ch = char::from_u32(code)?;
                        decoded.push(ch);
                        index += 6;
                        continue;
                    }
                    _ => return None,
                };
                decoded.push(escaped);
                index += 2;
            }
            _ => {
                let ch = body[index..].chars().next()?;
                decoded.push(ch);
                index += ch.len_utf8();
            }
        }
    }
    Some((decoded, index, false))
}

fn skip_ws(body: &str, mut index: usize) -> usize {
    while index < body.len() && body.as_bytes()[index].is_ascii_whitespace() {
        index += 1;
    }
    index
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::openai::requests::OpenAiToolContract;

    fn scanner() -> ToolCallStreamScanner {
        ToolCallStreamScanner::new(None)
    }

    fn content(events: &[ToolScanEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                ToolScanEvent::Content(text) => Some(text.as_str()),
                ToolScanEvent::Call(_) | ToolScanEvent::Arguments { .. } => None,
            })
            .collect()
    }

    fn calls(events: &[ToolScanEvent]) -> Vec<&OpenAiToolCall> {
        events
            .iter()
            .filter_map(|event| match event {
                ToolScanEvent::Call(call) => Some(call),
                ToolScanEvent::Content(_) | ToolScanEvent::Arguments { .. } => None,
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

    /// `echo` invoke whose `text` parameter value contains a literal
    /// `</｜DSML｜tool_calls>` closer, built from `U+FF5C` escapes so the
    /// stanza can only complete past that first closer.
    fn dsml_stanza_with_embedded_closer() -> String {
        let bar = "\u{FF5C}";
        format!(
            "<{bar}DSML{bar}tool_calls><{bar}DSML{bar}invoke name=\"echo\"><{bar}DSML{bar}parameter name=\"text\" string=\"true\">a</{bar}DSML{bar}tool_calls>b</{bar}DSML{bar}parameter></{bar}DSML{bar}invoke></{bar}DSML{bar}tool_calls>"
        )
    }

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
        let stanza = dsml_stanza_with_embedded_closer();
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
    fn dsml_embedded_closer_split_across_pushes_keeps_the_call() {
        // Per-token decode pushes the literal closer inside the parameter
        // value long before the stanza-closing closer. The failed parse must
        // keep withholding: releasing the prefix would leak the raw DSML
        // markers as content and lose the call the non-streaming path
        // extracts from the same tokens.
        let text = dsml_stanza_with_embedded_closer();
        let closer = "</\u{FF5C}DSML\u{FF5C}tool_calls>";
        let split = text.find(closer).expect("embedded closer") + closer.len();

        let mut scanner = scanner();
        let first = scanner.push(&text[..split]);
        assert!(calls(&first).is_empty());
        assert!(
            content(&first).is_empty(),
            "the still-open stanza must stay withheld: {:?}",
            content(&first)
        );

        let mut events = scanner.push(&text[split..]);
        events.extend(scanner.finish());
        let calls = calls(&events);
        assert_eq!(calls.len(), 1, "content: {:?}", content(&events));
        assert_eq!(calls[0].function.name, "echo");
        let arguments: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).expect("json arguments");
        assert_eq!(
            arguments,
            json!({"text": "a</\u{FF5C}DSML\u{FF5C}tool_calls>b"}),
        );
        assert!(
            content(&events).is_empty(),
            "content: {:?}",
            content(&events)
        );
        assert_eq!(scanner.calls_emitted(), 1);
    }

    #[test]
    fn dsml_embedded_closer_is_invariant_across_all_chunk_splits() {
        let text = dsml_stanza_with_embedded_closer();
        for split in 0..=text.len() {
            if !text.is_char_boundary(split) {
                continue;
            }
            let mut scanner = scanner();
            let mut events = scanner.push(&text[..split]);
            events.extend(scanner.push(&text[split..]));
            events.extend(scanner.finish());
            let calls = calls(&events);
            assert_eq!(calls.len(), 1, "split {split}: {:?}", content(&events));
            assert_eq!(calls[0].function.name, "echo", "split {split}");
            let arguments: serde_json::Value =
                serde_json::from_str(&calls[0].function.arguments).expect("json arguments");
            assert_eq!(
                arguments,
                json!({"text": "a</\u{FF5C}DSML\u{FF5C}tool_calls>b"}),
                "split {split}"
            );
            assert!(
                content(&events).is_empty(),
                "split {split}: {:?}",
                content(&events)
            );
        }
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
    fn xml_call_with_multibyte_arguments_survives_byte_chunk_streaming() {
        // The JSON argument carries accented letters and an ellipsis. Streamed
        // one char at a time (1-3 bytes per push), the remembered resume
        // offset (`len - (closer.len() - 1)`) can land inside a multi-byte
        // character; it must be rounded to a char boundary instead of panicking
        // on the next push's `&buffer[offset..]` slice.
        let text = "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"caf\u{e9} \u{2026} na\u{ef}ve\"}}</tool_call>";
        let (function, _) =
            extract_xml_tool_call_payload_at(text, 0).expect("whole-string XML call extracts");

        let mut scanner = scanner();
        let mut events = Vec::new();
        for (index, ch) in text.char_indices() {
            events.extend(scanner.push(&text[index..index + ch.len_utf8()]));
        }
        events.extend(scanner.finish());

        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, function.name);
        assert_eq!(calls[0].function.arguments, function.arguments);
        assert!(content(&events).is_empty());
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

    fn argument_pieces(events: &[ToolScanEvent]) -> (Option<String>, Option<String>, String) {
        let mut id = None;
        let mut name = None;
        let mut arguments = String::new();
        for event in events {
            if let ToolScanEvent::Arguments {
                id: event_id,
                name: event_name,
                arguments: piece,
            } = event
            {
                if event_id.is_some() {
                    id = event_id.clone();
                    name = event_name.clone();
                }
                arguments.push_str(piece);
            }
        }
        (id, name, arguments)
    }

    #[test]
    fn incremental_json_arguments_stream_in_pieces_and_name_is_sent_once() {
        let mut scanner = ToolCallStreamScanner::with_incremental(None, true);
        let first = scanner.push("<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"");
        let (id, name, arguments) = argument_pieces(&first);
        assert_eq!(id.as_deref(), Some("call_0"));
        assert_eq!(name.as_deref(), Some("lookup"));
        assert_eq!(arguments, "{\"city\":\"");
        assert!(calls(&first).is_empty());
        let mut second = scanner.push("Taipei\"}}</tool_call> tail");
        second.extend(scanner.finish());
        let (id, name, arguments) = argument_pieces(&second);
        assert!(id.is_none(), "id is sent only on the first piece");
        assert!(name.is_none(), "name is sent only on the first piece");
        assert_eq!(arguments, "Taipei\"}");
        assert!(calls(&second).is_empty(), "no second full call chunk");
        assert_eq!(content(&second), " tail");
        assert_eq!(scanner.calls_emitted(), 1);
    }

    #[test]
    fn whitespace_json_arguments_stay_one_final_call() {
        let mut scanner = ToolCallStreamScanner::with_incremental(None, true);
        let mut events = scanner.push(
            "<tool_call>{\"name\": \"lookup\", \"arguments\": {\"city\": \"Taipei\"}}</tool_call>",
        );
        events.extend(scanner.finish());
        assert!(
            argument_pieces(&events).2.is_empty(),
            "rewritten arguments must not stream raw"
        );
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].id, "call_0");
    }

    #[test]
    fn qwen_function_calls_stay_one_final_chunk_when_incremental() {
        let mut scanner = ToolCallStreamScanner::with_incremental(None, true);
        let mut events = scanner.push(
            "<tool_call><function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function></tool_call>",
        );
        events.extend(scanner.finish());
        assert!(argument_pieces(&events).2.is_empty());
        assert_eq!(calls(&events).len(), 1);
    }

    fn streamed_argument_text(events: &[ToolScanEvent]) -> Vec<String> {
        let mut calls = Vec::new();
        for event in events {
            match event {
                ToolScanEvent::Call(call) => calls.push(call.function.arguments.clone()),
                ToolScanEvent::Arguments { id, arguments, .. } => {
                    if id.is_some() || calls.is_empty() {
                        calls.push(arguments.clone());
                    } else if let Some(current) = calls.last_mut() {
                        current.push_str(arguments);
                    }
                }
                ToolScanEvent::Content(_) => {}
            }
        }
        calls
    }

    fn non_streaming_arguments(text: &str, contract: Option<&OpenAiToolContract>) -> Vec<String> {
        let mut rest = text.to_string();
        let mut arguments = Vec::new();
        while let Some(start) = rest.find("<tool_call>") {
            let Some((mut function, remaining)) =
                crate::openai::responses::extract_xml_tool_call_payload_at(&rest, start)
            else {
                break;
            };
            if let Some(contract) = contract {
                function.name = contract.canonical_tool_name(&function.name);
                function.arguments =
                    contract.canonical_arguments(&function.name, function.arguments);
            }
            arguments.push(function.arguments);
            if remaining == rest {
                break;
            }
            rest = remaining;
        }
        arguments
    }

    fn assert_chunked_matches_non_streaming(text: &str, contract: Option<OpenAiToolContract>) {
        let expected = non_streaming_arguments(text, contract.as_ref());
        let mut scanner = ToolCallStreamScanner::with_incremental(contract.map(Arc::new), true);
        let mut events = Vec::new();
        let mut pending = String::new();
        for ch in text.chars() {
            pending.push(ch);
            if pending.chars().count() == 3 {
                events.extend(scanner.push(&pending));
                pending.clear();
            }
        }
        if !pending.is_empty() {
            events.extend(scanner.push(&pending));
        }
        events.extend(scanner.finish());
        assert_eq!(
            streamed_argument_text(&events),
            expected,
            "streamed deltas diverged for {text}"
        );
    }

    #[test]
    fn incremental_deltas_match_non_streaming_arguments() {
        let edit = OpenAiToolContract::from_tools(Some(&json!([{
            "type": "function",
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}}
                }
            }
        }])))
        .expect("edit contract");
        assert_chunked_matches_non_streaming(
            "<tool_call>{\"name\":\"edit\",\"arguments\":{\"file_path\":\"notes.rs\"}}</tool_call>",
            Some(edit),
        );

        let echo = OpenAiToolContract::from_tools(Some(&json!([{
            "type": "function",
            "function": {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}}
                }
            }
        }])))
        .expect("echo contract");
        assert_chunked_matches_non_streaming(
            "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"a\\nb\"}}</tool_call>",
            Some(echo.clone()),
        );
        assert_chunked_matches_non_streaming(
            "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"caf\\u00e9\"}}</tool_call>",
            Some(echo.clone()),
        );
        assert_chunked_matches_non_streaming(
            "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"a\\nb\"}}</tool_call><tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"caf\\u00e9\"}}</tool_call>",
            Some(echo),
        );
    }

    #[test]
    fn incremental_header_uses_the_canonical_tool_name() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        }]);
        let contract = OpenAiToolContract::from_tools(Some(&tools)).expect("contract");
        let mut scanner = ToolCallStreamScanner::with_incremental(Some(Arc::new(contract)), true);
        let events = scanner.push(
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"query\":\"ports\"}}</tool_call>",
        );
        let (_id, name, arguments) = argument_pieces(&events);
        assert_eq!(name.as_deref(), Some("read"));
        assert_eq!(arguments, "{\"query\":\"ports\"}");
    }

    #[test]
    fn canonical_arguments_that_do_not_continue_the_prefix_are_not_appended() {
        let mut scanner = ToolCallStreamScanner::with_incremental(None, true);
        let opened = scanner.push("<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"ab");
        assert!(!argument_pieces(&opened).2.is_empty());
        scanner.testing_overwrite_streamed_arguments("XXXX");
        let mut closed = scanner.push("c\"}}</tool_call>");
        closed.extend(scanner.finish());
        assert_eq!(scanner.argument_continuations_rejected(), 1);
        let concatenated = format!(
            "{}{}",
            argument_pieces(&opened).2,
            argument_pieces(&closed).2
        );
        assert!(
            !concatenated.contains("XXXX"),
            "the overwritten prefix is not re-sent"
        );
        assert_ne!(
            concatenated, "{\"text\":\"abc\"}",
            "a non-continuation must not be spliced into the canonical object"
        );
        assert_eq!(
            scanner.parsed_calls(),
            0,
            "a rejected continuation must not qualify for tool_calls"
        );
    }

    #[test]
    fn rejected_canonical_continuation_does_not_qualify_for_tool_calls() {
        // `parsed_calls` is what the stream finish consults. A header whose
        // canonical arguments are not a continuation must not qualify, while
        // an applied continuation and an intact one-shot call still do.
        let mut scanner = ToolCallStreamScanner::with_incremental(None, true);
        let _opened = scanner.push("<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"ab");
        scanner.testing_overwrite_streamed_arguments("XXXX");
        let mut events = scanner.push("c\"}}</tool_call>");
        assert_eq!(scanner.argument_continuations_rejected(), 1);
        assert_eq!(
            scanner.parsed_calls(),
            0,
            "a rejected continuation must not qualify for tool_calls"
        );

        events
            .extend(scanner.push(
                "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"ok\"}}</tool_call>",
            ));
        events.extend(scanner.finish());
        assert_eq!(scanner.argument_continuations_rejected(), 1);
        assert_eq!(
            scanner.parsed_calls(),
            1,
            "a later valid streamed call still qualifies for tool_calls"
        );
        assert!(
            events.iter().any(|event| matches!(
                event,
                ToolScanEvent::Arguments { arguments, .. } if arguments.contains("ok")
            )) || calls(&events)
                .iter()
                .any(|call| call.function.arguments.contains("ok")),
            "the valid call is still emitted: {events:?}"
        );

        let mut intact = ToolCallStreamScanner::with_incremental(None, false);
        let mut intact_events = intact
            .push("<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"ab\"}}</tool_call>");
        intact_events.extend(intact.finish());
        assert_eq!(intact.argument_continuations_rejected(), 0);
        assert_eq!(
            intact.parsed_calls(),
            1,
            "an intact emitted call still qualifies for tool_calls"
        );
        assert_eq!(calls(&intact_events).len(), 1);
    }
}
