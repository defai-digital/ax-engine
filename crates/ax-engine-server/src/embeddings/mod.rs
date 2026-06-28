pub(crate) mod microbatch;
pub(crate) mod records;

use ax_engine_sdk::EmbeddingPooling;

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
