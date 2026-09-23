//! Single source of truth for the Flash Next MTP direct-fallback
//! attribution series: one `/metrics` counter per block or error reason.
//!
//! Included by `app_state` (accumulation), `metrics` (rendering) and the
//! `tests/metrics.rs` contract test, so the engine route keys and the
//! published metric names cannot drift apart.

/// `ax_mlx_flash_next_mtp_direct_fallback_*` route keys, in reason order.
/// Kept in sync with `FlashNextMtpFallbackReason` in
/// `crates/ax-engine-mlx/src/runner/mod.rs`.
pub(crate) const ROUTE_KEYS: [&str; 7] = [
    "ax_mlx_flash_next_mtp_direct_fallback_not_strict_greedy",
    "ax_mlx_flash_next_mtp_direct_fallback_think_control",
    "ax_mlx_flash_next_mtp_direct_fallback_pending_direct",
    "ax_mlx_flash_next_mtp_direct_fallback_no_budget",
    "ax_mlx_flash_next_mtp_direct_fallback_cursor_unavailable",
    "ax_mlx_flash_next_mtp_direct_fallback_components_unavailable",
    "ax_mlx_flash_next_mtp_direct_fallback_step_error",
];

/// Route-key prefix shared by every reason above.
pub(crate) const ROUTE_KEY_PREFIX: &str = "ax_mlx_flash_next_mtp_direct_fallback_";

/// `/metrics` counter name published for a reason suffix.
pub(crate) fn metric_name(suffix: &str) -> String {
    format!("ax_engine_flash_next_mtp_direct_fallback_{suffix}_total")
}
