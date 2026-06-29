pub(crate) mod microbatch;
pub(crate) mod records;

use ax_engine_sdk::EmbeddingPooling;

pub(crate) fn parse_embedding_max_tokens(value: Option<String>, default_tokens: usize) -> usize {
    value
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&tokens| tokens > 0)
        .unwrap_or(default_tokens)
}

pub(crate) fn parse_embedding_pooling(pooling: Option<&str>) -> Result<EmbeddingPooling, String> {
    match pooling.unwrap_or("last") {
        "last" => Ok(EmbeddingPooling::Last),
        "mean" => Ok(EmbeddingPooling::Mean),
        "cls" => Ok(EmbeddingPooling::Cls),
        other => Err(format!(
            "unknown pooling strategy {other:?}; expected \"last\", \"mean\", or \"cls\""
        )),
    }
}

pub(crate) fn parse_embedding_timeout_ms(value: Option<String>, default_ms: u64) -> u64 {
    value
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|&millis| millis > 0)
        .unwrap_or(default_ms)
}

#[cfg(test)]
mod tests {
    use super::{parse_embedding_max_tokens, parse_embedding_timeout_ms};

    #[test]
    fn embedding_max_tokens_accepts_positive_trimmed_value() {
        assert_eq!(
            parse_embedding_max_tokens(Some(" 16384 ".into()), 8192),
            16384
        );
    }

    #[test]
    fn embedding_max_tokens_falls_back_for_disabled_or_invalid_values() {
        for value in [
            None,
            Some("".into()),
            Some("0".into()),
            Some("-1".into()),
            Some("many".into()),
        ] {
            assert_eq!(parse_embedding_max_tokens(value, 8192), 8192);
        }
    }

    #[test]
    fn embedding_timeout_accepts_positive_trimmed_value() {
        assert_eq!(
            parse_embedding_timeout_ms(Some(" 45000 ".into()), 30_000),
            45_000
        );
    }

    #[test]
    fn embedding_timeout_falls_back_for_disabled_or_invalid_values() {
        for value in [
            None,
            Some("".into()),
            Some("0".into()),
            Some("-1".into()),
            Some("fast".into()),
        ] {
            assert_eq!(parse_embedding_timeout_ms(value, 30_000), 30_000);
        }
    }
}
