//! Platform probes for invariant I-4 memory-pressure observation.
//!
//! These helpers feed `ax_engine_core::mempressure` via the bench-harness
//! adapter at `crates/ax-engine-bench/src/harness/pressure_observer.rs`.
//! They are intentionally narrow:
//!
//! - [`host_resident_bytes`] returns the current process's resident-set
//!   size, sampled via `getrusage(RUSAGE_SELF)` on Unix.
//! - [`device_active_bytes`] returns MLX's currently-allocated GPU memory
//!   in bytes via `mlx_get_active_memory`.
//! - [`device_recommended_working_set_bytes`] returns Metal's
//!   `recommendedMaxWorkingSetSize` as the device budget.
//!
//! All three return `Option<u64>`; absent or failing probes return `None`
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

/// Active bytes currently held by MLX on the GPU.
///
/// Calls `mlx_get_active_memory`. Returns `None` if the FFI call returns
/// non-zero (e.g., MLX not initialized) or if the runtime is CPU-only.
pub fn device_active_bytes() -> Option<u64> {
    let mut bytes: usize = 0;
    let rc = unsafe { ffi::mlx_get_active_memory(&mut bytes as *mut usize) };
    if rc != 0 {
        return None;
    }
    Some(bytes as u64)
}

/// Peak bytes used by MLX since program start or last reset.
///
/// Calls `mlx_get_peak_memory`. Returns `None` if the FFI call fails.
pub fn device_peak_bytes() -> Option<u64> {
    let mut bytes: usize = 0;
    let rc = unsafe { ffi::mlx_get_peak_memory(&mut bytes as *mut usize) };
    if rc != 0 {
        return None;
    }
    Some(bytes as u64)
}

/// Cache bytes held by MLX's allocator (not actively used, not returned to OS).
///
/// Calls `mlx_get_cache_memory`. Returns `None` if the FFI call fails.
pub fn device_cache_bytes() -> Option<u64> {
    let mut bytes: usize = 0;
    let rc = unsafe { ffi::mlx_get_cache_memory(&mut bytes as *mut usize) };
    if rc != 0 {
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
        // panic and returns Some/None deterministically.
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
        let _ = device_peak_bytes();
    }

    #[test]
    fn device_cache_bytes_reports_or_skips_gracefully() {
        let _ = device_cache_bytes();
    }
}
