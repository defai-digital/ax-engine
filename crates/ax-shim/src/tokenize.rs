//! Tokenization: text to tokens and tokens to text.

use std::ffi::CStr;
use std::os::raw::c_char;

use crate::types::*;

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

    let model_ref = unsafe { &*model };

    // Extract text: if text_len > 0, use as byte length; otherwise null-terminated
    let text_str = if text_len > 0 {
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

    let result = model_ref.tokenizer.encode(text_str, add_special);
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
    _special: bool,
) -> i32 {
    if model.is_null() || buf.is_null() || length <= 0 {
        return 0;
    }

    let model_ref = unsafe { &*model };
    if token < 0 {
        return 0;
    }

    let text = match model_ref.tokenizer.decode_token(token as u32) {
        Some(t) => t,
        None => return 0,
    };

    let bytes = text.as_bytes();
    let n = bytes.len();

    if n as i32 > length {
        return -(n as i32); // buffer too small
    }

    let out = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, n) };
    out.copy_from_slice(bytes);

    n as i32
}
