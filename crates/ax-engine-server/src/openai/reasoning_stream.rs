//! Streaming reasoning extraction for Qwen `<think>…</think>` output.
//!
//! The think tags survive decode as plain text, so this is a text-level
//! scanner with the same holdback discipline as the stop/tool scanners: tag
//! text is withheld until it can no longer become a tag, think-body text is
//! routed to `delta.reasoning_content`, and everything else streams as
//! ordinary content. Gemma 4 thinking is channel-token-framed and handled by
//! `Gemma4ChannelStreamFilter`'s reasoning capture instead.

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ThinkScanState {
    /// Before any visible content: decide between a leading `<think>` block
    /// and ordinary pass-through.
    Lead,
    /// Inside `<think>`: route text to reasoning until the closing tag.
    InThink,
    /// After the think block (or when output never opened one).
    Passed,
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct ThinkScanStep {
    pub(crate) reasoning: String,
    pub(crate) content: String,
}

pub(crate) struct ThinkTagScanner {
    state: ThinkScanState,
    buffer: String,
}

impl ThinkTagScanner {
    pub(crate) fn new() -> Self {
        Self {
            state: ThinkScanState::Lead,
            buffer: String::new(),
        }
    }

    pub(crate) fn push(&mut self, text: &str) -> ThinkScanStep {
        let mut step = ThinkScanStep::default();
        if self.state == ThinkScanState::Passed {
            step.content.push_str(text);
            return step;
        }
        self.buffer.push_str(text);
        loop {
            match self.state {
                ThinkScanState::Lead => {
                    let lead_ws = self.buffer.len() - self.buffer.trim_start().len();
                    let rest = &self.buffer[lead_ws..];
                    if let Some(after) = rest.strip_prefix(THINK_OPEN) {
                        step.content.push_str(&self.buffer[..lead_ws]);
                        self.buffer = after.to_string();
                        self.state = ThinkScanState::InThink;
                        continue;
                    }
                    if !rest.is_empty() && THINK_OPEN.starts_with(rest) {
                        // Could still become `<think>`: hold.
                        return step;
                    }
                    if rest.is_empty() {
                        // Only whitespace so far; hold (it may precede a tag).
                        return step;
                    }
                    self.state = ThinkScanState::Passed;
                    step.content.push_str(&self.buffer);
                    self.buffer.clear();
                    return step;
                }
                ThinkScanState::InThink => {
                    if let Some(close_at) = self.buffer.find(THINK_CLOSE) {
                        step.reasoning.push_str(&self.buffer[..close_at]);
                        let after = self.buffer[close_at + THINK_CLOSE.len()..].to_string();
                        self.buffer.clear();
                        self.state = ThinkScanState::Passed;
                        step.content.push_str(&after);
                        return step;
                    }
                    // Release all but the longest suffix that could still
                    // begin `</think>` (ASCII tag: char-boundary-safe).
                    let hold = longest_suffix_prefix_of(&self.buffer, THINK_CLOSE);
                    let release = self.buffer.len() - hold;
                    if release > 0 {
                        step.reasoning.push_str(&self.buffer[..release]);
                        self.buffer.drain(..release);
                    }
                    return step;
                }
                ThinkScanState::Passed => {
                    step.content.push_str(&self.buffer);
                    self.buffer.clear();
                    return step;
                }
            }
        }
    }

    /// End of stream: whatever is withheld flushes to the stage it belongs
    /// to — an unclosed think body continues as reasoning (its start already
    /// streamed as reasoning and cannot be retracted).
    pub(crate) fn finish(&mut self) -> ThinkScanStep {
        let mut step = ThinkScanStep::default();
        let leftover = std::mem::take(&mut self.buffer);
        match self.state {
            ThinkScanState::InThink => step.reasoning.push_str(&leftover),
            ThinkScanState::Lead | ThinkScanState::Passed => step.content.push_str(&leftover),
        }
        self.state = ThinkScanState::Passed;
        step
    }
}

fn longest_suffix_prefix_of(text: &str, tag: &str) -> usize {
    let text = text.as_bytes();
    let tag = tag.as_bytes();
    let max = tag.len().saturating_sub(1).min(text.len());
    for len in (1..=max).rev() {
        if text[text.len() - len..] == tag[..len] {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_think_block_splits_reasoning_from_content() {
        let mut scanner = ThinkTagScanner::new();
        let step = scanner.push("<think>plan the fix</think>the answer");
        assert_eq!(step.reasoning, "plan the fix");
        assert_eq!(step.content, "the answer");
        assert_eq!(scanner.push(" continues").content, " continues");
    }

    #[test]
    fn open_tag_split_across_pushes() {
        let mut scanner = ThinkTagScanner::new();
        let step = scanner.push("<th");
        assert_eq!(step, ThinkScanStep::default());
        let step = scanner.push("ink>reasoning ");
        assert_eq!(step.reasoning, "reasoning ");
        assert_eq!(step.content, "");
        let step = scanner.push("done</think>answer");
        assert_eq!(step.reasoning, "done");
        assert_eq!(step.content, "answer");
    }

    #[test]
    fn close_tag_split_across_pushes_never_leaks_into_reasoning() {
        let mut scanner = ThinkTagScanner::new();
        let _ = scanner.push("<think>abc</th");
        let step = scanner.push("ink>xyz");
        assert_eq!(step.reasoning, "");
        assert_eq!(step.content, "xyz");
    }

    #[test]
    fn output_without_think_streams_as_content() {
        let mut scanner = ThinkTagScanner::new();
        let step = scanner.push("plain answer");
        assert_eq!(step.reasoning, "");
        assert_eq!(step.content, "plain answer");
    }

    #[test]
    fn false_tag_prefix_is_released_as_content() {
        let mut scanner = ThinkTagScanner::new();
        assert_eq!(scanner.push("<t"), ThinkScanStep::default());
        let step = scanner.push("able>");
        assert_eq!(step.content, "<table>");
    }

    #[test]
    fn unclosed_think_flushes_as_reasoning_at_finish() {
        let mut scanner = ThinkTagScanner::new();
        let step = scanner.push("<think>half a thought <");
        // The trailing "<" could begin "</think>", so it is withheld.
        assert_eq!(step.reasoning, "half a thought ");
        let step = scanner.finish();
        assert_eq!(step.reasoning, "<");
        assert_eq!(step.content, "");
    }

    #[test]
    fn leading_whitespace_before_think_is_content() {
        let mut scanner = ThinkTagScanner::new();
        let step = scanner.push("\n<think>x</think>y");
        assert_eq!(step.content, "\ny");
        assert_eq!(step.reasoning, "x");
    }
}
