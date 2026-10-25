//! Platform probes for invariant I-4 memory-pressure observation.
//!
//! These helpers feed `ax_engine_core::mempressure` via the bench-harness
//! adapter at `crates/ax-engine-bench/src/harness/pressure_observer.rs`.
//! They are intentionally narrow:
//!
//! - [`host_resident_bytes`] returns the current process's resident-set
//!   size, sampled via `getrusage(RUSAGE_SELF)` on Unix.
//! - [`process_memory_footprint`] returns this process's physical footprint
//!   from macOS `proc_pid_rusage(RUSAGE_INFO_V4)`.
//! - [`device_active_bytes`] returns MLX's currently-allocated GPU memory
//!   in bytes via `mlx_get_active_memory`.
//! - [`device_recommended_working_set_bytes`] returns Metal's
//!   `recommendedMaxWorkingSetSize` as the device budget.
//!
//! Numeric probes return `Option<u64>`; [`process_memory_footprint`] returns
//! `Option<ProcessFootprint>`. Absent or failing probes return `None`
//! so the downstream classifier can distinguish "probe unavailable" from
//! a probe that legitimately sampled zero (PRD §7.2).
//!
//! Linux behavior is documented inline at each probe; the kernel
//! returns `ru_maxrss` in kilobytes there, not bytes (BSD-style on
//! macOS), so the conversion is platform-conditional.

use crate::ffi;

/// Current process resident-set bytes.
///
/// On macOS, `getrusage` reports `ru_maxrss` in bytes; this is the *peak*
/// RSS observed so far for the process, which is the most defensible
/// "host pressure" signal available without paying a mach `task_info`
/// syscall on every call. On Linux, `ru_maxrss` is in kilobytes (so we
/// multiply by 1024). On other Unixes the same conversion is assumed;
/// adjust per platform if a target's `ru_maxrss` unit differs.
///
/// Returns `None` on:
/// - Non-Unix targets (no Unix `getrusage`).
/// - `getrusage` returning non-zero (very rare in practice).
pub fn host_resident_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc != 0 {
            return None;
        }
        // Saturating cast: ru_maxrss is `c_long` (i64 on macOS and 64-bit
        // Linux); negative values are not expected but we treat them as
        // unavailable rather than wrapping into u64.
        if usage.ru_maxrss < 0 {
            return None;
        }
        let raw = usage.ru_maxrss as u64;
        // macOS reports bytes; Linux reports kilobytes. Other targets are
        // assumed to follow Linux per POSIX.1-2001.
        if cfg!(target_os = "macos") {
            Some(raw)
        } else {
            Some(raw.saturating_mul(1024))
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Physical footprint of this process from macOS `proc_pid_rusage`.
///
/// `lifetime_max_phys_footprint_bytes` is at least `phys_footprint_bytes`.
/// The kernel does not sample the live footprint and the lifetime peak
/// atomically, so a raw peak a page under the live value is raised to the
/// live sample before it is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessFootprint {
    pub phys_footprint_bytes: u64,
    pub lifetime_max_phys_footprint_bytes: u64,
}

/// Lifetime peak published next to a live footprint sample.
///
/// `proc_pid_rusage` fills `ri_phys_footprint` and
/// `ri_lifetime_max_phys_footprint` in one struct, but the kernel does not
/// sample the two counters atomically. The live value can therefore exceed
/// the recorded lifetime peak by about a page. Raising the peak to the live
/// sample keeps the published invariant `lifetime_max >= live`.
fn lifetime_max_at_least_live(live_bytes: u64, lifetime_max_bytes: u64) -> u64 {
    lifetime_max_bytes.max(live_bytes)
}

/// Current and lifetime-max physical footprint of this process.
///
/// On macOS this calls `proc_pid_rusage` with `RUSAGE_INFO_V4` and reads
/// `ri_phys_footprint` and `ri_lifetime_max_phys_footprint`. The lifetime
/// peak is then raised to the live sample when the two counters disagree,
/// so a caller always observes
/// `lifetime_max_phys_footprint_bytes >= phys_footprint_bytes`.
/// Returns `None` off macOS and when the call fails.
pub fn process_memory_footprint() -> Option<ProcessFootprint> {
    #[cfg(target_os = "macos")]
    {
        // `rusage_info_v4` is a POD of integers and a byte array, so the
        // all-zero bit pattern is valid. `proc_pid_rusage` writes that
        // struct through `*mut rusage_info_t` (`*mut *mut c_void`).
        let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
        // SAFETY: `info` is a zeroed `rusage_info_v4`, large enough for
        // `RUSAGE_INFO_V4`. The pid is this process. The kernel writes the
        // struct and does not retain the pointer.
        let rc = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::pid_t,
                libc::RUSAGE_INFO_V4,
                info.as_mut_ptr() as *mut libc::rusage_info_t,
            )
        };
        if rc != 0 {
            return None;
        }
        // SAFETY: a zero return means `proc_pid_rusage` initialized the
        // `RUSAGE_INFO_V4` struct.
        let info = unsafe { info.assume_init() };
        let live = info.ri_phys_footprint;
        Some(ProcessFootprint {
            phys_footprint_bytes: live,
            lifetime_max_phys_footprint_bytes: lifetime_max_at_least_live(
                live,
                info.ri_lifetime_max_phys_footprint,
            ),
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Active bytes currently held by MLX on the GPU.
///
/// Calls `mlx_get_active_memory`. Returns `None` if the FFI call returns
/// non-zero (e.g., MLX not initialized) or if the runtime is CPU-only.
pub fn device_active_bytes() -> Option<u64> {
    let mut bytes: usize = 0;
    // Expected-failure probe (e.g., CPU-only runtime); polled on a cadence,
    // so capture quietly instead of printing `mlx error:` on every tick.
    let _quiet = crate::error::QuietErrorCapture::new();
    let rc = unsafe { ffi::mlx_get_active_memory(&mut bytes as *mut usize) };
    if rc != 0 {
        // Swallowed probe failure: drain the handler slot so the message is
        // not misattributed to a later failing call on this thread.
        crate::error::clear_stale_error();
        return None;
    }
    Some(bytes as u64)
}

/// Peak bytes used by MLX since program start or last reset.
///
/// Calls `mlx_get_peak_memory`. Returns `None` if the FFI call fails.
pub fn device_peak_bytes() -> Option<u64> {
    let mut bytes: usize = 0;
    // Expected-failure probe; see device_active_bytes.
    let _quiet = crate::error::QuietErrorCapture::new();
    let rc = unsafe { ffi::mlx_get_peak_memory(&mut bytes as *mut usize) };
    if rc != 0 {
        // Swallowed probe failure: drain the handler slot so the message is
        // not misattributed to a later failing call on this thread.
        crate::error::clear_stale_error();
        return None;
    }
    Some(bytes as u64)
}

/// Cache bytes held by MLX's allocator (not actively used, not returned to OS).
///
/// Calls `mlx_get_cache_memory`. Returns `None` if the FFI call fails.
pub fn device_cache_bytes() -> Option<u64> {
    let mut bytes: usize = 0;
    // Expected-failure probe; see device_active_bytes.
    let _quiet = crate::error::QuietErrorCapture::new();
    let rc = unsafe { ffi::mlx_get_cache_memory(&mut bytes as *mut usize) };
    if rc != 0 {
        // Swallowed probe failure: drain the handler slot so the message is
        // not misattributed to a later failing call on this thread.
        crate::error::clear_stale_error();
        return None;
    }
    Some(bytes as u64)
}

/// Recommended Metal working-set bytes for the active device.
///
/// This is the conservative budget callers should compare `device_active_bytes`
/// against. Returns `None` on CPU-only hardware.
pub fn device_recommended_working_set_bytes() -> Option<u64> {
    let size = crate::transforms::max_recommended_working_set_size();
    if size == 0 { None } else { Some(size as u64) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::lock_process_global;

    #[test]
    fn lifetime_max_at_least_live_clamps_a_stale_peak() {
        // The kernel race reports a live footprint about a page above the
        // lifetime counter. The published peak must rise to the live sample.
        assert_eq!(
            lifetime_max_at_least_live(37_274_392, 37_241_624),
            37_274_392
        );
        assert_eq!(lifetime_max_at_least_live(4_096 + 100, 4_096), 4_196);
        // A peak that is already ahead of the live sample stays put.
        assert_eq!(lifetime_max_at_least_live(100, 4_096), 4_096);
        assert_eq!(lifetime_max_at_least_live(7, 7), 7);
        assert_eq!(lifetime_max_at_least_live(0, 0), 0);
        assert_eq!(lifetime_max_at_least_live(u64::MAX, u64::MAX - 1), u64::MAX);
    }

    #[test]
    fn process_memory_footprint_is_positive_on_macos() {
        let value = process_memory_footprint();
        #[cfg(target_os = "macos")]
        {
            assert!(
                value.is_some_and(|footprint| {
                    footprint.phys_footprint_bytes > 0
                        && footprint.lifetime_max_phys_footprint_bytes
                            >= footprint.phys_footprint_bytes
                }),
                "process_memory_footprint should return a positive footprint on macOS, got {value:?}"
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert!(value.is_none());
        }
    }

    #[test]
    fn host_resident_bytes_is_nonzero_on_unix() {
        // We are obviously *some* RSS — the test process exists. On Unix
        // the probe must return a positive number; on a non-Unix target
        // the probe returns None (the test still passes because there is
        // nothing to assert positively).
        let value = host_resident_bytes();
        #[cfg(unix)]
        {
            assert!(
                value.is_some_and(|v| v > 0),
                "host_resident_bytes should return positive bytes on Unix, got {value:?}"
            );
        }
        #[cfg(not(unix))]
        {
            assert!(value.is_none());
        }
    }

    #[test]
    fn device_active_bytes_reports_or_skips_gracefully() {
        // We do not assert a particular value — MLX may or may not have
        // active allocations at this point — only that the probe does not
        // panic and returns Some/None deterministically. The counter is
        // process-global, so take the shared guard.
        let _guard = lock_process_global();
        let _ = device_active_bytes();
    }

    #[test]
    fn device_recommended_working_set_bytes_is_some_on_metal_hosts() {
        // On Apple Silicon CI hosts the value must be Some(>0); on a
        // CPU-only host it is acceptable to be None. We assert only that
        // when Some, the value is positive (a zero budget would break
        // pressure classification).
        if let Some(v) = device_recommended_working_set_bytes() {
            assert!(v > 0, "expected positive working-set budget, got 0");
        }
    }

    #[test]
    fn device_peak_bytes_reports_or_skips_gracefully() {
        // The peak counter is process-global, so take the shared guard.
        let _guard = lock_process_global();
        let _ = device_peak_bytes();
    }

    #[test]
    fn device_cache_bytes_reports_or_skips_gracefully() {
        // The cache counter is process-global, so take the shared guard.
        let _guard = lock_process_global();
        let _ = device_cache_bytes();
    }
}
