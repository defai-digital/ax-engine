//! Minimal markdown → styled lines for the chat transcript.
//!
//! Covers the subset models actually emit: headings, emphasis, inline code,
//! fenced code blocks, lists, blockquotes, rules, and links. Anything else
//! degrades to plain text instead of vanishing. Wrapping is left to the
//! transcript's `Paragraph`; this only produces structure and styles.

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use super::theme;

/// Render markdown into transcript lines (assistant replies only).
pub(super) fn render_markdown(text: &str) -> Vec<Line<'static>> {
    Renderer::default().run(text)
}

#[derive(Default)]
struct Renderer {
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,
    bold: bool,
    italic: bool,
    strike: bool,
    link: Option<String>,
    heading: bool,
    in_code_block: bool,
    /// List stack; `Some(n)` = ordered counter, `None` = bullet.
    lists: Vec<Option<u64>>,
    /// Marker for the current item ("• " / "3. ") plus first-line tracking.
    item_marker: Option<String>,
    item_first_line: bool,
    quote_depth: usize,
}

impl Renderer {
    fn run(mut self, text: &str) -> Vec<Line<'static>> {
        let options = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
        for event in Parser::new_ext(text, options) {
            self.handle(event);
        }
        self.flush_line();
        while self.lines.last().is_some_and(|l| l.spans.is_empty()) {
            self.lines.pop();
        }
        if self.lines.is_empty() {
            self.lines.push(Line::raw(""));
        }
        self.lines
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Start(tag) => self.start_tag(tag),
            Event::End(tag) => self.end_tag(tag),
            Event::Text(text) => {
                if self.in_code_block {
                    for line in text.lines() {
                        self.push_code_line(line);
                    }
                } else {
                    let style = self.current_style();
                    self.spans.push(Span::styled(text.into_string(), style));
                }
            }
            Event::Code(code) => {
                self.spans.push(Span::styled(
                    code.into_string(),
                    Style::default().fg(theme::FEATURE),
                ));
            }
            Event::SoftBreak => self.spans.push(Span::raw(" ")),
            Event::HardBreak => self.flush_line(),
            Event::Rule => {
                self.flush_line();
                self.lines
                    .push(Line::from(Span::styled("─".repeat(24), theme::label())));
                self.end_block();
            }
            Event::Html(_) | Event::InlineHtml(_) => {}
            Event::TaskListMarker(done) => {
                let mark = if done { "[x] " } else { "[ ] " };
                self.spans.push(Span::styled(mark, theme::body_dim()));
            }
            Event::FootnoteReference(_) | Event::InlineMath(_) | Event::DisplayMath(_) => {}
        }
    }

    fn start_tag(&mut self, tag: Tag) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { .. } => {
                self.flush_line();
                self.heading = true;
            }
            Tag::Emphasis => self.italic = true,
            Tag::Strong => self.bold = true,
            Tag::Strikethrough => self.strike = true,
            Tag::CodeBlock(kind) => {
                self.flush_line();
                self.in_code_block = true;
                if let CodeBlockKind::Fenced(lang) = kind
                    && !lang.is_empty()
                {
                    self.lines.push(Line::from(Span::styled(
                        format!("── {lang} "),
                        theme::label(),
                    )));
                }
            }
            Tag::List(start) => {
                self.flush_line();
                self.lists.push(start);
            }
            Tag::Item => {
                self.flush_line();
                self.item_first_line = true;
                self.item_marker = Some(match self.lists.last_mut() {
                    Some(Some(n)) => {
                        let marker = format!("{n}. ");
                        *n += 1;
                        marker
                    }
                    _ => "• ".to_string(),
                });
            }
            Tag::BlockQuote(_) => {
                self.flush_line();
                self.quote_depth += 1;
            }
            Tag::Link { dest_url, .. } => self.link = Some(dest_url.to_string()),
            // Images: alt text arrives as plain Text events — leave as-is.
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush_line();
                // Inside a list item, blank lines come from Item ends instead.
                if self.lists.is_empty() {
                    self.end_block();
                }
            }
            TagEnd::Heading(_) => {
                self.heading = false;
                self.end_block();
            }
            TagEnd::Emphasis => self.italic = false,
            TagEnd::Strong => self.bold = false,
            TagEnd::Strikethrough => self.strike = false,
            TagEnd::CodeBlock => {
                self.in_code_block = false;
                self.end_block();
            }
            TagEnd::List(_) => {
                self.lists.pop();
                self.item_marker = None;
                if self.lists.is_empty() {
                    self.end_block();
                }
            }
            TagEnd::Item => self.flush_line(),
            TagEnd::BlockQuote(_) => {
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.end_block();
            }
            TagEnd::Link => {
                // Terminals can't click styled text; show the target.
                if let Some(url) = self.link.take()
                    && !url.is_empty()
                {
                    self.spans
                        .push(Span::styled(format!(" ({url})"), theme::label()));
                }
            }
            _ => {}
        }
    }

    fn current_style(&self) -> Style {
        let mut style = if self.heading {
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD)
        } else if self.quote_depth > 0 {
            theme::body_dim()
        } else {
            theme::body()
        };
        if self.bold {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.italic {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if self.strike {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        if self.link.is_some() {
            style = style.add_modifier(Modifier::UNDERLINED);
        }
        style
    }

    fn push_code_line(&mut self, text: &str) {
        let style = Style::default().fg(theme::TEXT).bg(theme::CODE_BG);
        self.lines
            .push(Line::from(Span::styled(format!("  {text}"), style)));
    }

    /// Flush pending spans as one line with the structural prefix (quote bars,
    /// list indent, item marker / continuation padding).
    fn flush_line(&mut self) {
        if self.spans.is_empty() {
            return;
        }
        let mut prefix = String::new();
        for _ in 0..self.quote_depth {
            prefix.push_str("│ ");
        }
        if !self.lists.is_empty() {
            prefix.push_str(&"  ".repeat(self.lists.len().saturating_sub(1)));
            if let Some(marker) = &self.item_marker {
                if self.item_first_line {
                    prefix.push_str(marker);
                } else {
                    prefix.push_str(&" ".repeat(marker.chars().count()));
                }
            }
        }
        let mut spans = Vec::new();
        if !prefix.is_empty() {
            spans.push(Span::styled(prefix, theme::body_dim()));
        }
        spans.append(&mut self.spans);
        self.lines.push(Line::from(spans));
        self.item_first_line = false;
    }

    /// End a block: flush and separate from the next block with one blank line.
    fn end_block(&mut self) {
        self.flush_line();
        if self.lines.last().is_some_and(|l| !l.spans.is_empty()) {
            self.lines.push(Line::raw(""));
        }
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

    use super::*;
    use ratatui::style::Color;

    fn plain(lines: &[Line]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn heading_is_bold_accent_without_hashes() {
        let lines = render_markdown("## Title\n");
        assert_eq!(plain(&lines), "Title");
        assert_eq!(lines[0].spans[0].style.fg, Some(theme::ACCENT));
        assert!(
            lines[0].spans[0]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
    }

    #[test]
    fn bold_italic_inline_code_styles() {
        let lines = render_markdown("a **b** *c* `d`\n");
        let spans = &lines[0].spans;
        assert_eq!(spans[1].content.as_ref(), "b");
        assert!(spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(spans[3].content.as_ref(), "c");
        assert!(spans[3].style.add_modifier.contains(Modifier::ITALIC));
        assert_eq!(spans[5].content.as_ref(), "d");
        assert_eq!(spans[5].style.fg, Some(theme::FEATURE));
    }

    #[test]
    fn fenced_code_block_gets_bg_and_lang_header() {
        let lines = render_markdown("```rust\nfn main() {}\n```\n");
        assert_eq!(plain(&lines), "── rust \n  fn main() {}");
        let code = &lines[1].spans[0];
        assert_eq!(code.style.bg, Some(theme::CODE_BG));
        assert_eq!(code.style.fg, Some(Color::White));
    }

    #[test]
    fn bullets_and_numbered_lists_get_markers() {
        let lines = render_markdown("- one\n- two\n\n1. a\n2. b\n");
        assert_eq!(plain(&lines), "• one\n• two\n\n1. a\n2. b");
    }

    #[test]
    fn blockquote_gets_bar_prefix_and_dim() {
        let lines = render_markdown("> quoted\n");
        assert_eq!(plain(&lines), "│ quoted");
        assert_eq!(lines[0].spans[1].style.fg, Some(theme::DIM));
    }

    #[test]
    fn link_appends_url() {
        let lines = render_markdown("[docs](https://example.com)\n");
        assert_eq!(plain(&lines), "docs (https://example.com)");
    }

    #[test]
    fn paragraphs_separate_with_blank_line() {
        let lines = render_markdown("one\n\ntwo\n");
        assert_eq!(plain(&lines), "one\n\ntwo");
    }

    #[test]
    fn strikethrough_applies_modifier() {
        let lines = render_markdown("~~gone~~\n");
        assert!(
            lines[0].spans[0]
                .style
                .add_modifier
                .contains(Modifier::CROSSED_OUT)
        );
    }

    #[test]
    fn empty_input_yields_single_blank_line() {
        let lines = render_markdown("");
        assert_eq!(lines.len(), 1);
    }
}
