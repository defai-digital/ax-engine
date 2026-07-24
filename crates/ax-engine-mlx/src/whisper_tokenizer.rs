//! OpenAI Whisper's native multilingual token contract.
//!
//! The canonical `mlx-community/whisper-large-v3-turbo` checkpoint intentionally
//! contains only `config.json` and `weights.safetensors`; mlx-whisper ships the
//! tokenizer vocabulary as a package asset. Keeping the same pinned asset here
//! lets AX load that checkpoint unchanged instead of requiring an unrelated
//! Hugging Face `tokenizer.json`.

use std::collections::BTreeSet;

use base64::Engine as _;

pub(crate) const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

// Exact output of OpenAI Whisper's `Tokenizer.non_speech_tokens` for the
// multilingual.tiktoken vocabulary. These are mergeable-token ids, so the set
// is identical for 99- and 100-language checkpoints.
const NON_SPEECH_TOKENS: &[u32] = &[
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359,
    503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268,
    3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331,
    12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520,
    26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254,
];

const MERGEABLE_VOCAB_SIZE: usize = 50_257;
const MULTILINGUAL_VOCAB: &str = include_str!("assets/multilingual.tiktoken");

#[derive(Clone)]
pub(crate) struct WhisperTokenizer {
    mergeable_tokens: Vec<Vec<u8>>,
    pub(crate) eot: u32,
    pub(crate) sot: u32,
    pub(crate) translate: u32,
    pub(crate) transcribe: u32,
    pub(crate) no_timestamps: u32,
    pub(crate) timestamp_begin: u32,
    pub(crate) blank: u32,
    pub(crate) language_ids: Vec<(&'static str, u32)>,
    pub(crate) suppress: Vec<u32>,
}

impl WhisperTokenizer {
    pub(crate) fn new(n_vocab: usize) -> Result<Self, String> {
        if !matches!(n_vocab, 51_865 | 51_866) {
            return Err(format!(
                "supported Whisper multilingual vocabularies are 51865 and 51866, got {n_vocab}"
            ));
        }
        let num_languages = n_vocab
            .checked_sub(51_766)
            .ok_or_else(|| format!("invalid Whisper vocabulary size {n_vocab}"))?;
        if num_languages == 0 || num_languages > LANGUAGES.len() {
            return Err(format!(
                "Whisper vocabulary resolves to unsupported language count {num_languages}"
            ));
        }

        let mergeable_tokens = parse_mergeable_tokens()?;
        let eot = MERGEABLE_VOCAB_SIZE as u32;
        let sot = eot + 1;
        let language_start = sot + 1;
        let language_ids = LANGUAGES
            .iter()
            .take(num_languages)
            .enumerate()
            .map(|(index, code)| (*code, language_start + index as u32))
            .collect::<Vec<_>>();
        let translate = language_start + num_languages as u32;
        let transcribe = translate + 1;
        let sot_lm = transcribe + 1;
        let sot_prev = sot_lm + 1;
        let no_speech = sot_prev + 1;
        let no_timestamps = no_speech + 1;
        let timestamp_begin = no_timestamps + 1;
        if timestamp_begin as usize >= n_vocab {
            return Err(format!(
                "Whisper special token range exceeds vocabulary {n_vocab}"
            ));
        }

        let mut suppress = BTreeSet::from_iter(NON_SPEECH_TOKENS.iter().copied());
        suppress.extend([translate, transcribe, sot, sot_prev, sot_lm, no_speech]);

        Ok(Self {
            mergeable_tokens,
            eot,
            sot,
            translate,
            transcribe,
            no_timestamps,
            timestamp_begin,
            blank: 220,
            language_ids,
            suppress: suppress.into_iter().collect(),
        })
    }

    pub(crate) fn language_token(&self, code: &str) -> Option<u32> {
        let normalized = code.trim().to_ascii_lowercase();
        self.language_ids
            .iter()
            .find(|(candidate, _)| *candidate == normalized)
            .map(|(_, id)| *id)
    }

    pub(crate) fn language_for_token(&self, token: u32) -> Option<&'static str> {
        self.language_ids
            .iter()
            .find(|(_, id)| *id == token)
            .map(|(code, _)| *code)
    }

    pub(crate) fn initial_tokens(
        &self,
        language: Option<&str>,
        translate: bool,
    ) -> Result<Vec<u32>, String> {
        let language = language.unwrap_or("en");
        let language_token = self.language_token(language).ok_or_else(|| {
            format!("unsupported Whisper language {language:?}; use an ISO-639-1 language code")
        })?;
        Ok(vec![
            self.sot,
            language_token,
            if translate {
                self.translate
            } else {
                self.transcribe
            },
            self.no_timestamps,
        ])
    }

    pub(crate) fn decode_text(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for token in tokens.iter().copied().filter(|token| *token < self.eot) {
            if let Some(piece) = self.mergeable_tokens.get(token as usize) {
                bytes.extend_from_slice(piece);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn parse_mergeable_tokens() -> Result<Vec<Vec<u8>>, String> {
    let mut tokens = vec![None; MERGEABLE_VOCAB_SIZE];
    for (line_index, line) in MULTILINGUAL_VOCAB.lines().enumerate() {
        let (encoded, rank) = line.split_once(' ').ok_or_else(|| {
            format!(
                "invalid multilingual.tiktoken line {}",
                line_index.saturating_add(1)
            )
        })?;
        let rank = rank.parse::<usize>().map_err(|error| {
            format!(
                "invalid multilingual.tiktoken rank on line {}: {error}",
                line_index.saturating_add(1)
            )
        })?;
        if rank >= tokens.len() || tokens[rank].is_some() {
            return Err(format!(
                "invalid or duplicate multilingual.tiktoken rank {rank}"
            ));
        }
        // The canonical asset's final mergeable rank is written as a single
        // "=". Python's `base64.b64decode` (used by Whisper/tiktoken) accepts
        // it as an empty byte sequence; mirror that one lenient edge case.
        let bytes = if encoded == "=" {
            Vec::new()
        } else {
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|error| {
                    format!(
                        "invalid multilingual.tiktoken base64 on line {}: {error}",
                        line_index.saturating_add(1)
                    )
                })?
        };
        tokens[rank] = Some(bytes);
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(rank, token)| {
            token.ok_or_else(|| format!("multilingual.tiktoken is missing rank {rank}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_v3_turbo_special_ids_match_openai_contract() {
        let tokenizer = WhisperTokenizer::new(51_866).expect("tokenizer should load");
        assert_eq!(tokenizer.eot, 50_257);
        assert_eq!(tokenizer.sot, 50_258);
        assert_eq!(tokenizer.language_ids.len(), 100);
        assert_eq!(tokenizer.translate, 50_359);
        assert_eq!(tokenizer.transcribe, 50_360);
        assert_eq!(tokenizer.no_timestamps, 50_364);
        assert_eq!(tokenizer.timestamp_begin, 50_365);
        assert_eq!(tokenizer.blank, 220);
    }

    #[test]
    fn native_vocabulary_decodes_ascii_and_unicode() {
        let tokenizer = WhisperTokenizer::new(51_866).expect("tokenizer should load");
        assert_eq!(
            tokenizer.decode_text(&[400, 370, 11, 452, 7177, 6280]),
            " And so, my fellow Americans"
        );
        assert_eq!(tokenizer.decode_text(&[38_088]), "こんにちは");
    }

    #[test]
    fn language_prompt_rejects_unknown_codes() {
        let tokenizer = WhisperTokenizer::new(51_866).expect("tokenizer should load");
        assert_eq!(
            tokenizer
                .initial_tokens(Some("en"), false)
                .expect("English is supported"),
            vec![50_258, 50_259, 50_360, 50_364]
        );
        assert!(tokenizer.initial_tokens(Some("zz"), false).is_err());
    }
}
