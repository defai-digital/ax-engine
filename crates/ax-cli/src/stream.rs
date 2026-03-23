use std::io::{self, Write};

use ax_core::tokenizer::Tokenizer;

const FLUSH_TOKEN_THRESHOLD: usize = 8;
const FLUSH_BYTE_THRESHOLD: usize = 64;

/// Buffered token streamer for terminal output.
///
/// This keeps streaming responsive while avoiding a full stdout flush on every
/// generated token.
pub struct StreamPrinter<W: Write> {
    writer: W,
    pending: String,
    pending_tokens: usize,
}

impl<W: Write> StreamPrinter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            pending: String::new(),
            pending_tokens: 0,
        }
    }

    pub fn push_token(&mut self, tokenizer: &Tokenizer, tok: u32) -> io::Result<()> {
        if let Some(text) = tokenizer.render_token(tok) {
            self.pending.push_str(&text);
            self.pending_tokens += 1;
            if self.pending_tokens >= FLUSH_TOKEN_THRESHOLD
                || self.pending.len() >= FLUSH_BYTE_THRESHOLD
                || self.pending.ends_with('\n')
            {
                self.flush()?;
            }
        }
        Ok(())
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
}
