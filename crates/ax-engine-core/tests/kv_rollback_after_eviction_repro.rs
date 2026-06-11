// Repro: eviction of retained-cache entries that back a live shared prefix
// makes the sharer the sole block owner; the engine's InsufficientCapacity
// rollback path (engine.rs resolve_kv_schedule_plan) then hits an
// InvariantViolation and the whole step errors.
use ax_engine_core::kv::{AllocationStatus, KvManager, KvManagerConfig};
use ax_engine_core::{CacheGroupId, RequestId};

#[test]
fn rollback_after_capacity_eviction_of_shared_prefix_cache() {
    let mut manager = KvManager::new(KvManagerConfig::validated(CacheGroupId(0), 4, 2));

    // req1 fills both blocks, then frees -> both blocks promoted to retained cache.
    manager
        .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap();
    manager.allocate(RequestId(1), 8).unwrap();
    manager.free(RequestId(1)).unwrap();
    assert_eq!(manager.available_block_count(), 0);

    // req2 shares the retained 8-token prefix (refcounts -> 2).
    manager
        .register_request(RequestId(2), vec![1, 2, 3, 4, 5, 6, 7, 8, 9])
        .unwrap();
    let lookup = manager
        .lookup_prefix(RequestId(2), &[1, 2, 3, 4, 5, 6, 7, 8, 9])
        .unwrap();
    assert!(lookup.hit && lookup.uses_retained_cache());
    manager.share_prefix(RequestId(2), &lookup).unwrap();

    // req2 needs one more block for token 9; no free blocks. allocate() evicts
    // the cache entries backing req2's own shared prefix (gaining nothing) and
    // still reports InsufficientCapacity.
    let plan = manager.allocate(RequestId(2), 1).unwrap();
    assert_eq!(
        plan.allocation_status,
        AllocationStatus::InsufficientCapacity
    );

    // Engine then rolls back the prefix share (engine.rs ~line 431).
    let result = manager.rollback_prefix_share(RequestId(2), &lookup);
    eprintln!("rollback result: {result:?}");
    assert!(
        result.is_ok(),
        "rollback after eviction fails: {result:?} -> EngineCore::step() returns Err"
    );
}
