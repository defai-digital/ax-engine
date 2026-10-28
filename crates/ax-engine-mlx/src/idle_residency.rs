// SPDX-License-Identifier: Apache-2.0

//! Opt-in idle GPU residency keep-warm.
//!
//! macOS can un-wire an idle model's buffers. A later cached prompt then pays
//! a multi-second first token. When `AX_MLX_IDLE_TICK_MS` is a positive
//! interval, the model worker submits one tiny GPU copy on that cadence after
//! the last request, for at most `AX_MLX_IDLE_TICK_MAX_S` seconds (default
//! 900). Unset or `0` leaves the worker on its blocking receive.
//!
//! The tick never changes the wired limit, the cache limit, or the memory
//! limit. It runs only on the caller-supplied stream, which the model worker
//! owns.

use std::time::{Duration, Instant};

use mlx_sys::{MlxDtype, MlxStream, add, try_eval, zeros};

/// Default keep-warm window after the last request: 15 minutes.
pub const DEFAULT_MAX_WINDOW: Duration = Duration::from_secs(900);

/// `AX_MLX_IDLE_TICK_MS`: tick interval. Unset, empty, non-numeric, or `0` disables.
pub const IDLE_TICK_MS_ENV: &str = "AX_MLX_IDLE_TICK_MS";

/// `AX_MLX_IDLE_TICK_MAX_S`: seconds to keep ticking after the last request.
pub const IDLE_TICK_MAX_S_ENV: &str = "AX_MLX_IDLE_TICK_MAX_S";

/// Resolved keep-warm intervals. A zero [`Self::tick_interval`] is off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdleResidencyConfig {
    tick_interval: Duration,
    max_window: Duration,
}

impl IdleResidencyConfig {
    /// Kill switch engaged: no ticks.
    pub fn disabled() -> Self {
        Self {
            tick_interval: Duration::ZERO,
            max_window: DEFAULT_MAX_WINDOW,
        }
    }

    /// Build from the two environment values. Invalid tick text disables.
    /// Invalid window text keeps [`DEFAULT_MAX_WINDOW`].
    pub fn from_raw(tick_ms: Option<&str>, max_s: Option<&str>) -> Self {
        let tick_ms = parse_u64(tick_ms).unwrap_or(0);
        let max_s = parse_u64(max_s).unwrap_or(DEFAULT_MAX_WINDOW.as_secs());
        Self::from_durations(Duration::from_millis(tick_ms), Duration::from_secs(max_s))
    }

    pub fn from_durations(tick_interval: Duration, max_window: Duration) -> Self {
        Self {
            tick_interval,
            max_window,
        }
    }

    pub fn enabled(self) -> bool {
        !self.tick_interval.is_zero()
    }

    pub fn tick_interval(self) -> Duration {
        self.tick_interval
    }

    pub fn max_window(self) -> Duration {
        self.max_window
    }
}

fn parse_u64(raw: Option<&str>) -> Option<u64> {
    let text = raw?.trim();
    if text.is_empty() {
        return None;
    }
    text.parse().ok()
}

/// Why a keep-warm episode ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdleStopReason {
    /// `now - last_activity` reached the configured window.
    MaxWindow,
    /// The worker is leaving its loop.
    Shutdown,
}

impl IdleStopReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxWindow => "max_window",
            Self::Shutdown => "shutdown",
        }
    }
}

/// What the worker should do on this idle poll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdleTickDecision {
    /// Kill switch is off. Block until a command arrives.
    Disabled,
    /// A request is in flight. Do not submit GPU work.
    SkipBusy,
    /// OS or device memory is over budget. Do not submit GPU work.
    SkipPressure,
    /// Interval has not elapsed. `wait` is until the next tick or the window end.
    Wait { wait: Duration },
    /// The window already elapsed. Block until the next request.
    Parked,
    /// Submit one tiny GPU copy. `started` is the episode's first tick.
    Tick { started: bool },
    /// Episode ended. `log` is true only on the active-to-idle transition.
    Stopped { reason: IdleStopReason, log: bool },
}

/// Clock and flags for one poll. `now` is monotonic time since the worker
/// created the policy, so tests can advance it without sleeping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdleTickContext {
    pub now: Duration,
    pub busy: bool,
    pub under_memory_pressure: bool,
    pub shutting_down: bool,
}

/// Pure keep-warm state machine. GPU work stays outside this type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdleResidencyPolicy {
    config: IdleResidencyConfig,
    last_activity: Duration,
    last_tick: Duration,
    active: bool,
    parked: bool,
    shut_down: bool,
    tick_count: u64,
}

impl IdleResidencyPolicy {
    /// `last_activity` starts at zero, which the worker treats as "just loaded".
    pub fn new(config: IdleResidencyConfig) -> Self {
        Self {
            config,
            last_activity: Duration::ZERO,
            last_tick: Duration::ZERO,
            active: false,
            parked: false,
            shut_down: false,
            tick_count: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled()
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// A request, command, or successful session recycle just used the device.
    /// Restarts a parked window. Does not clear an in-progress episode, so
    /// ticks between requests do not log a fresh start.
    pub fn note_activity(&mut self, now: Duration) {
        if self.shut_down {
            return;
        }
        self.last_activity = now;
        self.last_tick = now;
        self.parked = false;
    }

    pub fn poll(&mut self, ctx: IdleTickContext) -> IdleTickDecision {
        if !self.config.enabled() {
            return IdleTickDecision::Disabled;
        }
        if self.shut_down || ctx.shutting_down {
            return self.stop(IdleStopReason::Shutdown);
        }
        if ctx.busy {
            return IdleTickDecision::SkipBusy;
        }
        if ctx.under_memory_pressure {
            return IdleTickDecision::SkipPressure;
        }
        if self.parked {
            return IdleTickDecision::Parked;
        }
        let idle_for = ctx.now.saturating_sub(self.last_activity);
        if idle_for >= self.config.max_window {
            return self.stop(IdleStopReason::MaxWindow);
        }
        let since_tick = ctx.now.saturating_sub(self.last_tick);
        if since_tick >= self.config.tick_interval {
            let started = !self.active;
            self.active = true;
            self.last_tick = ctx.now;
            self.tick_count = self.tick_count.saturating_add(1);
            return IdleTickDecision::Tick { started };
        }
        IdleTickDecision::Wait {
            wait: self
                .config
                .tick_interval
                .saturating_sub(since_tick)
                .min(self.config.max_window.saturating_sub(idle_for)),
        }
    }

    fn stop(&mut self, reason: IdleStopReason) -> IdleTickDecision {
        let log = self.active;
        self.active = false;
        match reason {
            IdleStopReason::MaxWindow => self.parked = true,
            IdleStopReason::Shutdown => self.shut_down = true,
        }
        IdleTickDecision::Stopped { reason, log }
    }
}

/// `kern.memorystatus_vm_pressure_level`: 1 is normal. Warning and above are
/// 2, 4, and 8. `None` or 0 means the probe was unavailable.
///
/// Device pressure is active MLX bytes strictly above Metal's recommended
/// working set. A resident model near that set is not pressure: the wired
/// cap already sits just under it, and skipping there would disable keep-warm
/// for the models that need it.
pub fn probes_under_memory_pressure(
    pressure_level: Option<u32>,
    active_bytes: Option<u64>,
    working_set_bytes: Option<u64>,
) -> bool {
    if matches!(pressure_level, Some(level) if level > 1) {
        return true;
    }
    matches!(
        (active_bytes, working_set_bytes),
        (Some(active), Some(working)) if working > 0 && active > working
    )
}

/// Sample host vm pressure (cached) and MLX active bytes.
///
/// Unavailable probes are not pressure. The sysctl read is cached so a
/// one-second tick does not fork on every wake.
pub fn sample_under_memory_pressure() -> bool {
    probes_under_memory_pressure(
        cached_vm_pressure_level(),
        mlx_sys::device_active_bytes(),
        mlx_sys::device_recommended_working_set_bytes(),
    )
}

pub(crate) fn cached_vm_pressure_level() -> Option<u32> {
    const TTL: Duration = Duration::from_secs(5);
    static CACHE: std::sync::Mutex<Option<(Instant, Option<u32>)>> = std::sync::Mutex::new(None);
    {
        let guard = match CACHE.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some((sampled_at, level)) = *guard
            && sampled_at.elapsed() < TTL
        {
            return level;
        }
    }
    let level = crate::hardware::sysctl_string(&["-n", "kern.memorystatus_vm_pressure_level"])
        .and_then(|value| value.trim().parse::<u32>().ok());
    let mut guard = match CACHE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    *guard = Some((Instant::now(), level));
    level
}

/// One small add on `stream`, then eval. Does not change residency limits.
///
/// `mlx_eval` follows the calling thread's default stream, so this pins
/// `stream` as that default first. Call only from the model worker thread.
/// Returns false when the runtime rejects the op. A panic from the MLX
/// wrappers is contained and reported as false so a keep-warm failure cannot
/// retire the worker.
pub fn touch_gpu_residency(stream: &MlxStream) -> bool {
    let touched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stream.set_as_default();
        let seed = zeros(&[4], MlxDtype::Float32, Some(stream));
        let copy = add(&seed, &seed, Some(stream));
        try_eval(&[&copy]).is_ok()
    }));
    touched.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: u64, busy: bool, pressure: bool, shutting_down: bool) -> IdleTickContext {
        IdleTickContext {
            now: Duration::from_secs(secs),
            busy,
            under_memory_pressure: pressure,
            shutting_down,
        }
    }

    fn every_second(max_s: u64) -> IdleResidencyPolicy {
        IdleResidencyPolicy::new(IdleResidencyConfig::from_raw(
            Some("1000"),
            Some(&max_s.to_string()),
        ))
    }

    #[test]
    fn idle_tick_disabled_by_default() {
        let config = IdleResidencyConfig::from_raw(None, None);
        assert!(!config.enabled());
        assert_eq!(config.tick_interval(), Duration::ZERO);
        assert_eq!(config.max_window(), DEFAULT_MAX_WINDOW);
        assert_eq!(DEFAULT_MAX_WINDOW, Duration::from_secs(900));

        for raw in [None, Some("0"), Some(" 0 "), Some(""), Some("no")] {
            let mut policy = IdleResidencyPolicy::new(IdleResidencyConfig::from_raw(raw, None));
            assert!(
                !policy.enabled(),
                "raw {raw:?} must leave the kill switch off"
            );
            assert_eq!(
                policy.poll(at(3_600, false, false, false)),
                IdleTickDecision::Disabled
            );
            assert_eq!(policy.tick_count(), 0);
        }

        let invalid_window = IdleResidencyConfig::from_raw(Some("1000"), Some("later"));
        assert!(invalid_window.enabled());
        assert_eq!(invalid_window.max_window(), DEFAULT_MAX_WINDOW);
    }

    #[test]
    fn idle_tick_skips_when_busy() {
        let mut policy = every_second(900);
        assert_eq!(
            policy.poll(at(1, true, false, false)),
            IdleTickDecision::SkipBusy
        );
        assert_eq!(policy.tick_count(), 0);
        assert_eq!(
            policy.poll(at(1, false, false, false)),
            IdleTickDecision::Tick { started: true }
        );
        assert_eq!(policy.tick_count(), 1);
    }

    #[test]
    fn idle_tick_skips_under_pressure() {
        assert!(!probes_under_memory_pressure(None, None, None));
        assert!(!probes_under_memory_pressure(Some(0), None, None));
        assert!(!probes_under_memory_pressure(Some(1), Some(10), Some(100)));
        assert!(!probes_under_memory_pressure(Some(1), Some(100), Some(100)));
        assert!(probes_under_memory_pressure(Some(2), None, None));
        assert!(probes_under_memory_pressure(Some(4), None, None));
        assert!(probes_under_memory_pressure(Some(8), None, None));
        assert!(probes_under_memory_pressure(Some(1), Some(101), Some(100)));

        let mut policy = every_second(900);
        assert_eq!(
            policy.poll(at(1, false, true, false)),
            IdleTickDecision::SkipPressure
        );
        assert_eq!(policy.tick_count(), 0);
        assert_eq!(
            policy.poll(at(1, false, false, false)),
            IdleTickDecision::Tick { started: true }
        );
        assert_eq!(policy.tick_count(), 1);
    }

    #[test]
    fn idle_tick_stops_after_max_window() {
        let mut policy = every_second(3);
        assert_eq!(
            policy.poll(at(1, false, false, false)),
            IdleTickDecision::Tick { started: true }
        );
        assert_eq!(
            policy.poll(at(2, false, false, false)),
            IdleTickDecision::Tick { started: false }
        );
        assert_eq!(
            policy.poll(at(3, false, false, false)),
            IdleTickDecision::Stopped {
                reason: IdleStopReason::MaxWindow,
                log: true,
            }
        );
        assert_eq!(policy.tick_count(), 2);
        assert_eq!(
            policy.poll(at(4, false, false, false)),
            IdleTickDecision::Parked
        );
        assert_eq!(policy.tick_count(), 2);

        policy.note_activity(Duration::from_secs(4));
        assert_eq!(
            policy.poll(at(5, false, false, false)),
            IdleTickDecision::Tick { started: true }
        );
        assert_eq!(policy.tick_count(), 3);
    }

    #[test]
    fn idle_tick_stops_on_shutdown() {
        let mut policy = every_second(900);
        assert_eq!(
            policy.poll(at(1, false, false, false)),
            IdleTickDecision::Tick { started: true }
        );
        assert_eq!(
            policy.poll(at(2, false, false, true)),
            IdleTickDecision::Stopped {
                reason: IdleStopReason::Shutdown,
                log: true,
            }
        );
        assert_eq!(policy.tick_count(), 1);
        assert_eq!(
            policy.poll(at(3, false, false, false)),
            IdleTickDecision::Stopped {
                reason: IdleStopReason::Shutdown,
                log: false,
            }
        );
        assert_eq!(policy.tick_count(), 1);

        let mut never_ticked = every_second(900);
        assert_eq!(
            never_ticked.poll(at(1, false, false, true)),
            IdleTickDecision::Stopped {
                reason: IdleStopReason::Shutdown,
                log: false,
            }
        );
        assert_eq!(never_ticked.tick_count(), 0);
    }
}
