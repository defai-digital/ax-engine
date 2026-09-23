//! Contract test for the Flash Next MTP direct-fallback attribution series.
//!
//! `ax-engine-server` is a binary-only crate, so this target includes the
//! shipped key/name table by path instead of importing it. The end-to-end
//! wiring (that `/metrics` actually renders these names) is asserted by the
//! in-crate `src/tests/metrics.rs` unit test over the rendered response body.

#[path = "../src/flash_next_fallback_keys.rs"]
mod flash_next_fallback_keys;

use flash_next_fallback_keys::{ROUTE_KEY_PREFIX, ROUTE_KEYS, metric_name};

#[test]
fn flash_next_fallback_reason_metrics_published() {
    assert_eq!(ROUTE_KEYS.len(), 7, "one route key per fallback reason");
    for key in ROUTE_KEYS {
        assert!(
            key.starts_with(ROUTE_KEY_PREFIX),
            "route key {key} lacks the documented prefix"
        );
        let suffix = key.strip_prefix(ROUTE_KEY_PREFIX).unwrap_or_default();
        assert!(
            !suffix.is_empty(),
            "route key {key} has an empty reason suffix"
        );
        let name = metric_name(suffix);
        assert_eq!(
            name,
            format!("ax_engine_flash_next_mtp_direct_fallback_{suffix}_total"),
        );
        assert!(
            name.ends_with("_total"),
            "counter {name} must be a _total series"
        );
    }
    assert!(ROUTE_KEYS.contains(&"ax_mlx_flash_next_mtp_direct_fallback_step_error"));
    assert!(ROUTE_KEYS.contains(&"ax_mlx_flash_next_mtp_direct_fallback_cursor_unavailable"));
}
