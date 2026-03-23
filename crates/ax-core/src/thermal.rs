//! Thermal pressure monitoring for macOS.
//!
//! Reads `NSProcessInfo.thermalState` via Objective-C runtime FFI to detect
//! system thermal pressure. Provides throttle recommendations:
//!
//! | Thermal State | Action                                  |
//! |---------------|-----------------------------------------|
//! | Nominal/Fair  | None — full performance                 |
//! | Serious       | Reduce batch size and concurrency       |
//! | Critical      | Fall back to CPU-only (disable Metal)   |

use std::ffi::{c_char, c_void};

/// macOS thermal pressure state (mirrors `NSProcessInfoThermalState`).
///
/// Ordered by severity: `Nominal < Fair < Serious < Critical`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i64)]
pub enum ThermalState {
    /// No thermal pressure. Full performance available.
    Nominal = 0,
    /// Mild thermal pressure. System may reduce performance slightly.
    Fair = 1,
    /// Significant thermal pressure. Active throttling recommended.
    Serious = 2,
    /// Critical thermal pressure. Maximum throttling, CPU-only fallback.
    Critical = 3,
}

impl ThermalState {
    /// Parse from the raw `NSProcessInfoThermalState` integer value.
    pub fn from_raw(raw: i64) -> Self {
        match raw {
            0 => Self::Nominal,
            1 => Self::Fair,
            2 => Self::Serious,
            3 => Self::Critical,
            _ => Self::Critical, // unknown → treat as critical (safe default)
        }
    }

    /// Whether this state requires any throttling.
    pub fn requires_throttle(&self) -> bool {
        *self >= Self::Serious
    }
}

impl std::fmt::Display for ThermalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nominal => write!(f, "nominal"),
            Self::Fair => write!(f, "fair"),
            Self::Serious => write!(f, "serious"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Throttle action recommended based on thermal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleAction {
    /// No throttling needed. Full performance.
    None,
    /// Reduce batch size and/or concurrency.
    ReduceBatch,
    /// Fall back to CPU-only (disable Metal GPU).
    CpuOnly,
}

/// Determine the appropriate throttle action for the given thermal state.
pub fn throttle_action(state: ThermalState) -> ThrottleAction {
    match state {
        ThermalState::Nominal | ThermalState::Fair => ThrottleAction::None,
        ThermalState::Serious => ThrottleAction::ReduceBatch,
        ThermalState::Critical => ThrottleAction::CpuOnly,
    }
}

/// Read the current macOS thermal state via `NSProcessInfo`.
///
/// Uses Objective-C runtime FFI: `[[NSProcessInfo processInfo] thermalState]`.
/// Returns `ThermalState::Nominal` if the API call fails.
pub fn current_thermal_state() -> ThermalState {
    // SAFETY: calling Objective-C runtime functions — standard macOS system calls.
    let raw = unsafe { thermal_state_raw() };
    ThermalState::from_raw(raw as i64)
}

/// Thermal monitor that tracks state transitions and provides throttle guidance.
///
/// Polls `NSProcessInfo.thermalState` and logs when the thermal state changes.
/// Use `recommend()` to get the current throttle action.
pub struct ThermalMonitor {
    last_state: ThermalState,
}

impl ThermalMonitor {
    /// Create a new monitor, reading the initial thermal state.
    pub fn new() -> Self {
        let state = current_thermal_state();
        tracing::info!("ThermalMonitor initialized: {state}");
        Self { last_state: state }
    }

    /// Poll the current thermal state. Logs on state transitions.
    pub fn poll(&mut self) -> ThermalState {
        let state = current_thermal_state();
        if state != self.last_state {
            if state > self.last_state {
                tracing::warn!(
                    "thermal pressure increased: {} → {}",
                    self.last_state,
                    state
                );
            } else {
                tracing::info!(
                    "thermal pressure decreased: {} → {}",
                    self.last_state,
                    state
                );
            }
            self.last_state = state;
        }
        state
    }

    /// Get the last observed thermal state (without polling).
    pub fn last_state(&self) -> ThermalState {
        self.last_state
    }

    /// Poll and return the recommended throttle action.
    pub fn recommend(&mut self) -> ThrottleAction {
        throttle_action(self.poll())
    }
}

impl Default for ThermalMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// --- Objective-C runtime FFI ---
//
// We call NSProcessInfo.thermalState via the ObjC runtime rather than
// pulling in an objc crate. This matches the pattern in metrics/counters.rs
// (raw mach kernel FFI for RSS tracking).

// Ensure Foundation framework is loaded (NSProcessInfo lives here).
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> *mut c_void;
    fn sel_registerName(name: *const c_char) -> *mut c_void;
}

// Single objc_msgSend declaration — both pointer and integer returns use register x0
// on aarch64, so a single signature works for all our calls.
unsafe extern "C" {
    fn objc_msgSend(obj: *mut c_void, sel: *mut c_void) -> *mut c_void;
}

/// Read the raw `NSProcessInfoThermalState` integer (0–3).
///
/// # Safety
/// Calls Objective-C runtime functions. Safe on macOS with Foundation loaded.
unsafe fn thermal_state_raw() -> isize {
    unsafe {
        let cls = objc_getClass(c"NSProcessInfo".as_ptr());
        if cls.is_null() {
            return 0; // fallback: nominal
        }

        let sel_process_info = sel_registerName(c"processInfo".as_ptr());
        let process_info = objc_msgSend(cls, sel_process_info);
        if process_info.is_null() {
            return 0; // fallback: nominal
        }

        let sel_thermal = sel_registerName(c"thermalState".as_ptr());
        // thermalState returns NSInteger; on aarch64 both ptr and int use x0.
        objc_msgSend(process_info, sel_thermal) as isize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_state_from_raw() {
        assert_eq!(ThermalState::from_raw(0), ThermalState::Nominal);
        assert_eq!(ThermalState::from_raw(1), ThermalState::Fair);
        assert_eq!(ThermalState::from_raw(2), ThermalState::Serious);
        assert_eq!(ThermalState::from_raw(3), ThermalState::Critical);
    }

    #[test]
    fn test_thermal_state_from_raw_unknown() {
        // Unknown values default to Critical (safe default)
        assert_eq!(ThermalState::from_raw(42), ThermalState::Critical);
        assert_eq!(ThermalState::from_raw(-1), ThermalState::Critical);
    }

    #[test]
    fn test_thermal_state_ordering() {
        assert!(ThermalState::Nominal < ThermalState::Fair);
        assert!(ThermalState::Fair < ThermalState::Serious);
        assert!(ThermalState::Serious < ThermalState::Critical);
    }

    #[test]
    fn test_thermal_state_display() {
        assert_eq!(ThermalState::Nominal.to_string(), "nominal");
        assert_eq!(ThermalState::Fair.to_string(), "fair");
        assert_eq!(ThermalState::Serious.to_string(), "serious");
        assert_eq!(ThermalState::Critical.to_string(), "critical");
    }

    #[test]
    fn test_requires_throttle() {
        assert!(!ThermalState::Nominal.requires_throttle());
        assert!(!ThermalState::Fair.requires_throttle());
        assert!(ThermalState::Serious.requires_throttle());
        assert!(ThermalState::Critical.requires_throttle());
    }

    #[test]
    fn test_throttle_action_mapping() {
        assert_eq!(throttle_action(ThermalState::Nominal), ThrottleAction::None);
        assert_eq!(throttle_action(ThermalState::Fair), ThrottleAction::None);
        assert_eq!(
            throttle_action(ThermalState::Serious),
            ThrottleAction::ReduceBatch
        );
        assert_eq!(
            throttle_action(ThermalState::Critical),
            ThrottleAction::CpuOnly
        );
    }

    #[test]
    fn test_current_thermal_state_returns_valid() {
        // On macOS, this should return a valid state
        let state = current_thermal_state();
        let raw = state as i64;
        assert!(
            (0..=3).contains(&raw),
            "thermal state should be 0–3, got {raw}"
        );
    }

    #[test]
    fn test_thermal_monitor_creation() {
        let monitor = ThermalMonitor::new();
        let state = monitor.last_state();
        assert!((state as i64) <= 3);
    }

    #[test]
    fn test_thermal_monitor_poll() {
        let mut monitor = ThermalMonitor::new();
        let state = monitor.poll();
        assert!((state as i64) <= 3);
        // last_state should match poll result
        assert_eq!(monitor.last_state(), state);
    }

    #[test]
    fn test_thermal_monitor_recommend() {
        let mut monitor = ThermalMonitor::new();
        let action = monitor.recommend();
        // On a non-thermally-stressed test machine, expect None
        // But any valid action is acceptable
        assert!(
            action == ThrottleAction::None
                || action == ThrottleAction::ReduceBatch
                || action == ThrottleAction::CpuOnly
        );
    }
}
