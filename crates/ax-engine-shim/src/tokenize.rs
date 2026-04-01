//! Tokenization: text to tokens and tokens to text.

use std::ffi::CStr;
use std::os::raw::c_char;

use crate::types::*;

fn has_valid_token_output_capacity(tokens: *mut LlamaToken, n_tokens_max: i32) -> bool {
    tokens.is_null() || n_tokens_max >= 0
}

/// Tokenize text into token IDs.
///
/// Returns the number of tokens written.
/// If `tokens` is null, returns the number of tokens that would be produced.
/// Returns negative (-n_needed) if the output buffer is too small.
#[unsafe(no_mangle)]
pub extern "C" fn llama_tokenize(
    model: *const LlamaModel,
    text: *const c_char,
    text_len: i32,
    tokens: *mut LlamaToken,
    n_tokens_max: i32,
    add_special: bool,
    _parse_special: bool,
) -> i32 {
    if model.is_null() || text.is_null() {
        return -1;
    }
    if !has_valid_token_output_capacity(tokens, n_tokens_max) {
        tracing::error!("llama_tokenize: negative output capacity: {n_tokens_max}");
        return -1;
    }

    let model_ref = unsafe { &*model };

    // Extract text: if text_len >= 0, use as byte length; if negative, null-terminated
    let text_str = if text_len >= 0 {
        let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, text_len as usize) };
        match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("llama_tokenize: invalid UTF-8: {e}");
                return -1;
            }
        }
    } else {
        match unsafe { CStr::from_ptr(text) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("llama_tokenize: invalid UTF-8: {e}");
                return -1;
            }
        }
    };

    let result = model_ref
        .tokenizer
        .encode_with_options(text_str, add_special, _parse_special);
    let n_tokens = result.len() as i32;

    // If output buffer is null, just return the count needed
    if tokens.is_null() {
        return n_tokens;
    }

    // Check buffer size
    if n_tokens > n_tokens_max {
        return -n_tokens; // negative = buffer too small, abs value = needed size
    }

    // Copy tokens to output buffer
    let out = unsafe { std::slice::from_raw_parts_mut(tokens, n_tokens as usize) };
    for (i, &tok) in result.iter().enumerate() {
        out[i] = tok as LlamaToken;
    }

    n_tokens
}

/// Convert a token ID to text.
///
/// Returns the number of bytes written. Returns negative if buffer too small.
#[unsafe(no_mangle)]
pub extern "C" fn llama_token_to_piece(
    model: *const LlamaModel,
    token: LlamaToken,
    buf: *mut c_char,
    length: i32,
    _lstrip: i32,
    special: bool,
) -> i32 {
    if model.is_null() || buf.is_null() || length <= 0 {
        return 0;
    }

    let model_ref = unsafe { &*model };
    if token < 0 {
        return 0;
    }

    let id = token as u32;

    // Use render_token for proper SentencePiece/byte-token decoding.
    // render_token returns None for special tokens; when the `special` flag
    // is set, fall back to the raw vocab string so callers can see BOS/EOS.
    let rendered: String;
    let bytes: &[u8] = match model_ref.tokenizer.render_token(id) {
        Some(s) => {
            rendered = s;
            rendered.as_bytes()
        }
        None => {
            if special {
                match model_ref.tokenizer.decode_token(id) {
                    Some(t) => t.as_bytes(),
                    None => return 0,
                }
            } else {
                return 0;
            }
        }
    };

    let n = bytes.len();

    if n as i32 > length {
        return -(n as i32); // buffer too small
    }

    let out = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, n) };
    out.copy_from_slice(bytes);

    n as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_valid_token_output_capacity_allows_count_only_calls() {
        assert!(has_valid_token_output_capacity(std::ptr::null_mut(), -1));
    }

    #[test]
    fn test_has_valid_token_output_capacity_rejects_negative_lengths_for_output_buffers() {
        let mut token = 0;
        assert!(!has_valid_token_output_capacity(
            std::ptr::from_mut(&mut token),
            -1
        ));
    }

    #[test]
    fn test_has_valid_token_output_capacity_accepts_non_negative_lengths() {
        let mut token = 0;
        assert!(has_valid_token_output_capacity(
            std::ptr::from_mut(&mut token),
            0
        ));
    }
}
