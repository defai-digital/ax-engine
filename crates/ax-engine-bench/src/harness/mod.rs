//! Serving-shaped stress test harness scaffolding.
//!
//! This module is the home of latency collection, claim-gating, and pressure
//! observation utilities used by the workload fixtures under
//! `crates/ax-engine-bench/src/workloads/`. See
//! `.internal/prd/engine-serving-invariants.md` (Phase 1 for metrics,
//! Phase 4 for `pressure_observer`, Phase 6 for `ngram_claim_gate`).

pub(crate) mod metrics;

// Only `WorkloadReport` is currently consumed outside this module (by workload
// fixtures). The remaining metrics types — `LatencyChannel`, `LatencySamples`,
// `PostRestartCacheCounts` — stay reachable via `crate::harness::metrics::*`
// until Phase 4/Phase 5 wiring needs them at the harness boundary.
pub(crate) use metrics::WorkloadReport;
