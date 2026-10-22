//! Grammar-constrained decoding: a JSON grammar compiled from the server's
//! schema subset, and per-step allowed-token masks over a model vocabulary.

pub mod json;
pub mod token_mask;

pub use json::{JsonGrammar, JsonGrammarError, JsonMatcher};
pub use token_mask::{MaskWorkspace, TokenVocab, apply_token_mask, mask_allows};
