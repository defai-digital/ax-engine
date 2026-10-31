//! Active-serving host pressure policy. Unavailable probes cannot clear a
//! previous pressure episode. Existing requests retain their owned state.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum HostPressure {
    #[default]
    Normal,
    Warning,
    Critical,
}

const RECOVERY: Duration = Duration::from_secs(10);

#[derive(Debug, Default)]
pub struct PressureLatch {
    level: HostPressure,
    recovery: Option<(HostPressure, Duration)>,
    /// A run strictly below the recovery candidate, so a two-level drop
    /// does not restart its interval when the first level is reached.
    lower: Option<(HostPressure, Duration)>,
    last_observed: Option<Duration>,
}

impl PressureLatch {
    pub fn observe(&mut self, sample: Option<u32>, now: Duration) -> HostPressure {
        if self.last_observed.is_some_and(|last| now < last) {
            self.recovery = None;
            self.lower = None;
            // Keep the high-water mark: a later jump back to the old time
            // must not count an interval that the clock never observed.
            self.level = self.level.max(match sample {
                Some(2) => HostPressure::Warning,
                Some(4 | 8) => HostPressure::Critical,
                _ => self.level,
            });
            return self.level;
        }
        self.last_observed = Some(now);
        let observed = match sample {
            Some(1) => HostPressure::Normal,
            Some(2) => HostPressure::Warning,
            Some(4 | 8) => HostPressure::Critical,
            _ => {
                self.recovery = None;
                self.lower = None;
                return self.level;
            }
        };
        if observed >= self.level {
            self.level = observed;
            self.recovery = None;
            self.lower = None;
            return self.level;
        }
        match self.recovery {
            Some((candidate, _)) if observed <= candidate => match self.lower {
                Some((lower, _)) if observed <= lower => {}
                _ if observed < candidate => self.lower = Some((observed, now)),
                _ => self.lower = None,
            },
            _ => {
                self.recovery = Some((observed, now));
                self.lower = None;
            }
        }
        while let Some((candidate, since)) = self.recovery {
            if now.saturating_sub(since) < RECOVERY {
                break;
            }
            self.level = candidate;
            self.recovery = self.lower.take();
        }
        self.level
    }
}

/// Shared by native admission and worker cache reclamation. The platform
/// probe has a five-second TTL; escalation applies on the first observed
/// warning/critical sample, recovery requires a sustained lower level.
pub fn sample() -> HostPressure {
    static STATE: OnceLock<(Instant, Mutex<PressureLatch>)> = OnceLock::new();
    let (epoch, latch) = STATE.get_or_init(|| (Instant::now(), Mutex::default()));
    let observed = crate::idle_residency::cached_vm_pressure_level();
    let mut latch = latch.lock().unwrap_or_else(|poison| poison.into_inner());
    latch.observe(observed, epoch.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escalation_is_immediate_and_recovery_requires_sustained_samples() {
        let mut latch = PressureLatch::default();
        let mut at = |level, time| latch.observe(level, Duration::from_secs(time));
        assert_eq!(at(None, 0), HostPressure::Normal);
        assert_eq!(at(Some(2), 1), HostPressure::Warning);
        assert_eq!(at(Some(4), 2), HostPressure::Critical);
        assert_eq!(at(Some(1), 3), HostPressure::Critical);
        assert_eq!(at(Some(1), 12), HostPressure::Critical);
        assert_eq!(at(Some(1), 13), HostPressure::Normal);
    }

    #[test]
    fn unknown_and_worse_samples_interrupt_recovery() {
        for unknown in [None, Some(0), Some(3), Some(16)] {
            let mut latch = PressureLatch::default();
            let mut at = |level, time| latch.observe(level, Duration::from_secs(time));
            assert_eq!(at(Some(8), 0), HostPressure::Critical);
            at(Some(1), 1);
            assert_eq!(at(unknown, 11), HostPressure::Critical);
            at(Some(1), 12);
            assert_eq!(at(Some(2), 21), HostPressure::Critical);
            assert_eq!(at(Some(2), 30), HostPressure::Critical);
            assert_eq!(at(Some(2), 31), HostPressure::Warning);
            at(Some(1), 32);
            assert_eq!(at(Some(1), 42), HostPressure::Normal);
        }
    }

    #[test]
    fn two_level_recovery_counts_the_lower_run_from_its_start() {
        let mut latch = PressureLatch::default();
        let mut at = |level, time| latch.observe(level, Duration::from_secs(time));
        assert_eq!(at(Some(4), 0), HostPressure::Critical);
        assert_eq!(at(Some(2), 1), HostPressure::Critical);
        for time in 2..11 {
            assert_eq!(at(Some(1), time), HostPressure::Critical);
        }
        assert_eq!(at(Some(1), 11), HostPressure::Warning);
        assert_eq!(at(Some(1), 12), HostPressure::Normal);
    }

    #[test]
    fn clock_regression_restarts_recovery() {
        let mut latch = PressureLatch::default();
        latch.observe(Some(4), Duration::from_secs(100));
        latch.observe(Some(1), Duration::from_secs(101));
        assert_eq!(
            latch.observe(Some(1), Duration::from_secs(1)),
            HostPressure::Critical
        );
        assert_eq!(
            latch.observe(Some(1), Duration::from_secs(10)),
            HostPressure::Critical
        );
        assert_eq!(
            latch.observe(Some(1), Duration::from_secs(101)),
            HostPressure::Critical
        );
        assert_eq!(
            latch.observe(Some(1), Duration::from_secs(110)),
            HostPressure::Critical
        );
        assert_eq!(
            latch.observe(Some(1), Duration::from_secs(111)),
            HostPressure::Normal
        );
    }
}
