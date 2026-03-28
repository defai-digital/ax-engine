use std::io::{self, Write};

use ax_engine_core::tokenizer::Tokenizer;

const FLUSH_TOKEN_THRESHOLD: usize = 8;
const FLUSH_BYTE_THRESHOLD: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamAction {
    Continue,
    Stop,
}

/// Buffered token streamer for terminal output.
///
/// This keeps streaming responsive while avoiding a full stdout flush on every
/// generated token.
pub struct StreamPrinter<W: Write> {
    writer: W,
    pending: String,
    pending_tokens: usize,
    stop_strings: Vec<String>,
    stop_token_ids: Vec<u32>,
    stopped: bool,
}

impl<W: Write> StreamPrinter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            pending: String::new(),
            pending_tokens: 0,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            stopped: false,
        }
    }

    pub fn with_stops(writer: W, mut stop_strings: Vec<String>, stop_token_ids: Vec<u32>) -> Self {
        stop_strings.retain(|stop| !stop.is_empty());
        Self {
            writer,
            pending: String::new(),
            pending_tokens: 0,
            stop_strings,
            stop_token_ids,
            stopped: false,
        }
    }

    pub fn push_token(&mut self, tokenizer: &Tokenizer, tok: u32) -> io::Result<StreamAction> {
        if self.stopped {
            return Ok(StreamAction::Stop);
        }
        if self.stop_token_ids.contains(&tok) {
            self.stopped = true;
            self.flush()?;
            return Ok(StreamAction::Stop);
        }
        if let Some(text) = tokenizer.render_token(tok) {
            return self.push_text(&text);
        }
        Ok(StreamAction::Continue)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.writer.write_all(self.pending.as_bytes())?;
        self.writer.flush()?;
        self.pending.clear();
        self.pending_tokens = 0;
        Ok(())
    }

    fn push_text(&mut self, text: &str) -> io::Result<StreamAction> {
        self.pending.push_str(text);
        self.pending_tokens += 1;

        if let Some(stop_offset) = self.first_stop_match() {
            self.flush_prefix(stop_offset)?;
            self.pending.clear();
            self.pending_tokens = 0;
            self.stopped = true;
            return Ok(StreamAction::Stop);
        }

        let held_back = self.longest_partial_stop_suffix();
        let safe_len = self.pending.len().saturating_sub(held_back);
        let should_flush = self.pending_tokens >= FLUSH_TOKEN_THRESHOLD
            || safe_len >= FLUSH_BYTE_THRESHOLD
            || self.pending[..safe_len].contains('\n');
        if should_flush && safe_len > 0 {
            self.flush_prefix(safe_len)?;
        }

        Ok(StreamAction::Continue)
    }

    fn flush_prefix(&mut self, len: usize) -> io::Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.writer.write_all(&self.pending.as_bytes()[..len])?;
        self.writer.flush()?;
        self.pending.drain(..len);
        self.pending_tokens = 0;
        Ok(())
    }

    fn first_stop_match(&self) -> Option<usize> {
        self.stop_strings
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
    }

    fn longest_partial_stop_suffix(&self) -> usize {
        let mut longest = 0;

        for stop in &self.stop_strings {
            for (prefix_len, _) in stop.char_indices().skip(1) {
                if self.pending.ends_with(&stop[..prefix_len]) {
                    longest = longest.max(prefix_len);
                }
            }
        }

        longest
    }

    #[cfg(test)]
    fn into_inner(self) -> W {
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Cursor;

    use ax_engine_core::tokenizer::vocab::TokenType;
    use ax_engine_core::tokenizer::{Tokenizer, Vocab};

    use super::*;

    fn make_test_tokenizer() -> Tokenizer {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "hello".to_string(),
            " world".to_string(),
            "stop".to_string(),
            "tail".to_string(),
            "é".to_string(),
            "clair".to_string(),
        ];
        let scores = vec![0.0; tokens.len()];
        let types = vec![
            TokenType::Unknown,
            TokenType::Control,
            TokenType::Control,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
        ];
        let mut token_to_id = HashMap::new();
        for (idx, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), idx as u32);
        }

        Tokenizer::from_vocab(Vocab {
            tokens,
            scores,
            types,
            token_to_id,
            merge_ranks: None,
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            add_bos: false,
            add_eos: false,
            add_space_prefix: false,
            model_type: "llama".to_string(),
            eot_id: None,
        })
    }

    #[test]
    fn test_stream_printer_holds_partial_stop_suffix() {
        let mut stream = StreamPrinter::with_stops(
            Cursor::new(Vec::new()),
            vec!["stop".to_string()],
            Vec::new(),
        );

        assert_eq!(stream.push_text("hello s").unwrap(), StreamAction::Continue);
        stream.flush().unwrap();

        let writer = stream.into_inner();
        assert_eq!(String::from_utf8(writer.into_inner()).unwrap(), "hello s");
    }

    #[test]
    fn test_stream_printer_suppresses_stop_string_across_chunks() {
        let mut stream = StreamPrinter::with_stops(
            Cursor::new(Vec::new()),
            vec!["stop".to_string()],
            Vec::new(),
        );

        assert_eq!(stream.push_text("hello s").unwrap(), StreamAction::Continue);
        assert_eq!(stream.push_text("top world").unwrap(), StreamAction::Stop);

        let writer = stream.into_inner();
        assert_eq!(String::from_utf8(writer.into_inner()).unwrap(), "hello ");
    }

    #[test]
    fn test_stream_printer_stops_on_stop_token_id_before_emitting_text() {
        let tokenizer = make_test_tokenizer();
        let mut stream = StreamPrinter::with_stops(Cursor::new(Vec::new()), Vec::new(), vec![5]);

        assert_eq!(
            stream.push_token(&tokenizer, 3).unwrap(),
            StreamAction::Continue
        );
        assert_eq!(
            stream.push_token(&tokenizer, 5).unwrap(),
            StreamAction::Stop
        );

        let writer = stream.into_inner();
        assert_eq!(String::from_utf8(writer.into_inner()).unwrap(), "hello");
    }

    #[test]
    fn test_stream_printer_handles_overlapping_stop_strings() {
        let mut stream = StreamPrinter::with_stops(
            Cursor::new(Vec::new()),
            vec!["###".to_string(), "##".to_string()],
            Vec::new(),
        );

        assert_eq!(stream.push_text("hello #").unwrap(), StreamAction::Continue);
        assert_eq!(stream.push_text("## world").unwrap(), StreamAction::Stop);

        let writer = stream.into_inner();
        assert_eq!(String::from_utf8(writer.into_inner()).unwrap(), "hello ");
    }

    #[test]
    fn test_stream_printer_handles_utf8_stop_strings() {
        let mut stream = StreamPrinter::with_stops(
            Cursor::new(Vec::new()),
            vec!["éclair".to_string()],
            Vec::new(),
        );

        assert_eq!(stream.push_text("caf").unwrap(), StreamAction::Continue);
        assert_eq!(stream.push_text("éclair noir").unwrap(), StreamAction::Stop);

        let writer = stream.into_inner();
        assert_eq!(String::from_utf8(writer.into_inner()).unwrap(), "caf");
    }

    #[test]
    fn test_stream_printer_flush_boundary_preserves_partial_stop() {
        let mut stream = StreamPrinter::with_stops(
            Cursor::new(Vec::new()),
            vec!["stop".to_string()],
            Vec::new(),
        );
        let near_boundary = format!("{}s", "a".repeat(64));

        assert_eq!(
            stream.push_text(&near_boundary).unwrap(),
            StreamAction::Continue
        );
        assert_eq!(
            stream.push_text("top trailing").unwrap(),
            StreamAction::Stop
        );

        let writer = stream.into_inner();
        assert_eq!(
            String::from_utf8(writer.into_inner()).unwrap(),
            "a".repeat(64)
        );
    }
}
