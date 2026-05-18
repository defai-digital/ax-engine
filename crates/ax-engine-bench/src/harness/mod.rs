//! Serving-shaped stress test harness scaffolding.
//!
//! This module is the home of latency collection, claim-gating, and pressure
//! observation utilities used by the workload fixtures under
//! `crates/ax-engine-bench/src/workloads/`. See
//! `.internal/prd/engine-serving-invariants.md` (Phase 1 for metrics,
//! Phase 4 for `pressure_observer`, Phase 6 for `ngram_claim_gate`).

pub(crate) mod metrics;
pub(crate) mod ngram_claim_gate;
pub(crate) mod pressure_observer;

// `WorkloadReport` is consumed by workload fixtures; the remaining metrics
// types stay reachable via `crate::harness::metrics::*`. Phase 4 added
// `pressure_observer` as the harness adapter for `ax_engine_core::mempressure`.
pub(crate) use metrics::WorkloadReport;
