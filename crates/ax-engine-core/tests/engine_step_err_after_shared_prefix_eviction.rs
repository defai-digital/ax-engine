// End-to-end repro through EngineCore::step():
// retained-cache prefix share + KV exhaustion -> eviction strips the cache
// entries backing the live share -> InsufficientCapacity rollback hits an
// InvariantViolation and step() returns Err.
use ax_engine_core::{
    CacheGroupId, EngineCore, ModelId, RequestId, SequenceNo,
    kv::KvManagerConfig,
    request::{RequestMultimodalInputs, RequestSubmission},
};

fn submission(id: u64, seq: u64, tokens: Vec<u32>) -> RequestSubmission {
    RequestSubmission {
        request_id: RequestId(id),
        model_id: ModelId("testmodel".into()),
        input_tokens: tokens,
        multimodal_inputs: RequestMultimodalInputs::default(),
        sampling_params: Default::default(),
        max_output_tokens: 1,
        arrival_sequence: SequenceNo(seq),
        metadata: None,
    }
}

#[test]
fn engine_step_survives_shared_prefix_cache_eviction_under_pressure() {
    // 3 blocks of 4 tokens.
    let mut engine = EngineCore::with_kv_config(KvManagerConfig::validated(CacheGroupId(0), 4, 3));

    // req1: 8-token prompt fills two blocks; decode of its single output token
    // takes the third. max_output 1 -> finishes fast.
    engine
        .submit(submission(1, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]))
        .unwrap();
    for _ in 0..6 {
        engine.step(16, true).unwrap();
    }
    // req1 freed; its 2 full prompt blocks retained in prefix cache, 1 free.
    assert_eq!(engine.kv_manager().available_block_count(), 1);

    // req2: same 8-token prefix + 5 extra tokens (tail needs 2 blocks, 1 free).
    engine
        .submit(submission(
            2,
            2,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ))
        .unwrap();

    let mut last_err = None;
    for _ in 0..6 {
        match engine.step(16, true) {
            Ok(_) => {}
            Err(e) => {
                last_err = Some(e);
                break;
            }
        }
    }
    assert!(
        last_err.is_none(),
        "EngineCore::step() returned Err: {last_err:?}"
    );
}
