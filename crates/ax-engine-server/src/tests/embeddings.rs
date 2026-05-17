use crate::app_state::{EmbeddingBatchKey, EmbeddingBatchRequestOptions};
use crate::embeddings::microbatch::{collect_embedding_batch_groups, pooling_code};
use ax_engine_sdk::EmbeddingPooling;

#[test]
fn microbatch_groups_requests_by_options() {
    let groups = collect_embedding_batch_groups(&[
        EmbeddingBatchRequestOptions {
            pooling: EmbeddingPooling::Last,
            normalize: true,
        },
        EmbeddingBatchRequestOptions {
            pooling: EmbeddingPooling::Last,
            normalize: true,
        },
        EmbeddingBatchRequestOptions {
            pooling: EmbeddingPooling::Mean,
            normalize: true,
        },
        EmbeddingBatchRequestOptions {
            pooling: EmbeddingPooling::Last,
            normalize: false,
        },
    ]);

    assert_eq!(groups.len(), 3);
    assert_eq!(
        groups[0],
        (
            EmbeddingBatchKey {
                pooling_code: pooling_code(EmbeddingPooling::Mean),
                normalize: true
            },
            vec![2]
        )
    );
    assert_eq!(
        groups[1],
        (
            EmbeddingBatchKey {
                pooling_code: pooling_code(EmbeddingPooling::Last),
                normalize: false
            },
            vec![3]
        )
    );
    assert_eq!(
        groups[2],
        (
            EmbeddingBatchKey {
                pooling_code: pooling_code(EmbeddingPooling::Last),
                normalize: true
            },
            vec![0, 1]
        )
    );
}
