// SPDX-License-Identifier: Apache-2.0

//! Runtime Apple Silicon Neural Accelerator detection.
//!
//! This is a **policy gate**, not an AX-owned NAX kernel dispatcher. MLX
//! selects NAX GEMM/SDPA internally from the admitted wheel. AX uses the
//! cached capability to:
//! - leave Metal command-buffer caps at MLX defaults on M5+ except families
//!   with measured M5 wins (`qwen3_next`);
//! - prefer native `mask="causal"` SDPA on Qwen full-attn prefill chunks
//!   (`seq >= 1024`) so MLX can take NAX fused attention.
//!
//! Detection is `silicon_generation >= 5` and running macOS 26.2+. It does
//! not inspect the linked `libmlx` for NAX symbols. Results are process-
//! cached. Tests may override the snapshot on the current thread.

use std::process::Command;
use std::sync::OnceLock;

#[cfg(test)]
use std::cell::Cell;

/// Cached hardware snapshot used by NAX policy gates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HardwareCapabilities {
    /// Apple M-series generation (`Some(5)` for M5 / M5 Max / M5 Ultra).
    pub silicon_generation: Option<u32>,
    pub macos_major: Option<u32>,
    pub macos_minor: Option<u32>,
    /// True when the SoC generation is M5 or newer.
    pub has_neural_accelerator: bool,
    /// True when the running OS is macOS 26.2 or newer.
    pub macos_supports_na: bool,
}

impl HardwareCapabilities {
    /// Both the SoC and the running OS can drive NAX kernels.
    #[inline]
    pub fn neural_accelerator_active(self) -> bool {
        self.has_neural_accelerator && self.macos_supports_na
    }

    #[cfg(test)]
    pub fn m5_na() -> Self {
        Self {
            silicon_generation: Some(5),
            macos_major: Some(26),
            macos_minor: Some(6),
            has_neural_accelerator: true,
            macos_supports_na: true,
        }
    }

    #[cfg(test)]
    pub fn m4() -> Self {
        Self {
            silicon_generation: Some(4),
            macos_major: Some(26),
            macos_minor: Some(6),
            has_neural_accelerator: false,
            macos_supports_na: true,
        }
    }

    #[cfg(test)]
    pub fn m5_old_macos() -> Self {
        Self {
            silicon_generation: Some(5),
            macos_major: Some(26),
            macos_minor: Some(0),
            has_neural_accelerator: true,
            macos_supports_na: false,
        }
    }
}

static DETECTED: OnceLock<HardwareCapabilities> = OnceLock::new();

#[cfg(test)]
thread_local! {
    static OVERRIDE: Cell<Option<HardwareCapabilities>> = const { Cell::new(None) };
}

/// Process-cached hardware snapshot, or the current thread's test override.
pub fn current() -> HardwareCapabilities {
    #[cfg(test)]
    if let Some(hw) = OVERRIDE.with(|slot| slot.get()) {
        return hw;
    }
    *DETECTED.get_or_init(detect_hardware)
}

/// True on M5+ running macOS 26.2+.
#[inline]
pub fn neural_accelerator_active() -> bool {
    current().neural_accelerator_active()
}

/// Restores the previous thread-local hardware override on drop.
#[cfg(test)]
pub struct HardwareOverrideGuard {
    previous: Option<HardwareCapabilities>,
}

#[cfg(test)]
impl Drop for HardwareOverrideGuard {
    fn drop(&mut self) {
        OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

/// Override detected hardware for the rest of this thread (tests).
#[cfg(test)]
pub fn override_hardware(hw: HardwareCapabilities) -> HardwareOverrideGuard {
    let previous = OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(hw));
        previous
    });
    HardwareOverrideGuard { previous }
}

fn detect_hardware() -> HardwareCapabilities {
    let silicon_generation = detect_silicon_generation();
    let (macos_major, macos_minor) = detect_macos_version();
    let caps = HardwareCapabilities {
        silicon_generation,
        macos_major,
        macos_minor,
        has_neural_accelerator: silicon_generation.is_some_and(|generation| generation >= 5),
        macos_supports_na: macos_supports_neural_accelerator(macos_major, macos_minor),
    };
    tracing::info!(
        target = "ax_engine_mlx",
        silicon_generation = ?caps.silicon_generation,
        macos_major = ?caps.macos_major,
        macos_minor = ?caps.macos_minor,
        has_neural_accelerator = caps.has_neural_accelerator,
        macos_supports_na = caps.macos_supports_na,
        neural_accelerator_active = caps.neural_accelerator_active(),
        "detected Apple Silicon Neural Accelerator policy snapshot"
    );
    caps
}

pub(crate) fn macos_supports_neural_accelerator(major: Option<u32>, minor: Option<u32>) -> bool {
    match (major, minor) {
        (Some(major), _) if major > 26 => true,
        (Some(26), Some(minor)) if minor >= 2 => true,
        _ => false,
    }
}

pub(crate) fn parse_apple_m_series_generation(soc: &str) -> Option<u32> {
    let digits = soc
        .trim()
        .strip_prefix("Apple M")?
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

pub(crate) fn parse_macos_major_minor(version: &str) -> (Option<u32>, Option<u32>) {
    let mut parts = version.trim().split('.');
    let major = parts.next().and_then(|p| p.parse().ok());
    let minor = parts.next().and_then(|p| p.parse().ok());
    (major, minor)
}

fn detect_silicon_generation() -> Option<u32> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    sysctl_string(&["-n", "machdep.cpu.brand_string"])
        .or_else(|| command_stdout("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"]))
        .and_then(|brand| parse_apple_m_series_generation(&brand))
}

fn detect_macos_version() -> (Option<u32>, Option<u32>) {
    if !cfg!(target_os = "macos") {
        return (None, None);
    }
    let raw = sysctl_string(&["-n", "kern.osproductversion"])
        .or_else(|| command_stdout("/usr/sbin/sysctl", &["-n", "kern.osproductversion"]))
        .or_else(|| command_stdout("/usr/bin/sw_vers", &["-productVersion"]));
    raw.map(|version| parse_macos_major_minor(&version))
        .unwrap_or((None, None))
}

fn sysctl_string(args: &[&str]) -> Option<String> {
    command_stdout("sysctl", args).or_else(|| command_stdout("/usr/sbin/sysctl", args))
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_m_series_generation() {
        assert_eq!(parse_apple_m_series_generation("Apple M4"), Some(4));
        assert_eq!(parse_apple_m_series_generation("Apple M5 Max"), Some(5));
        assert_eq!(parse_apple_m_series_generation("Apple M5 Ultra"), Some(5));
        assert_eq!(parse_apple_m_series_generation("Apple M10 Ultra"), Some(10));
        assert_eq!(parse_apple_m_series_generation("Intel Core i9"), None);
    }

    #[test]
    fn parses_macos_major_minor() {
        assert_eq!(parse_macos_major_minor("26.6.2"), (Some(26), Some(6)));
        assert_eq!(parse_macos_major_minor("26.2"), (Some(26), Some(2)));
        assert_eq!(parse_macos_major_minor("27.0"), (Some(27), Some(0)));
        assert_eq!(parse_macos_major_minor("Tahoe"), (None, None));
    }

    #[test]
    fn macos_na_gate_requires_26_2() {
        assert!(!macos_supports_neural_accelerator(Some(26), Some(0)));
        assert!(!macos_supports_neural_accelerator(Some(26), Some(1)));
        assert!(macos_supports_neural_accelerator(Some(26), Some(2)));
        assert!(macos_supports_neural_accelerator(Some(27), Some(0)));
        assert!(!macos_supports_neural_accelerator(None, Some(2)));
    }

    #[test]
    fn m5_plus_on_26_2_is_active() {
        assert!(HardwareCapabilities::m5_na().neural_accelerator_active());
        assert!(!HardwareCapabilities::m4().neural_accelerator_active());
        assert!(!HardwareCapabilities::m5_old_macos().neural_accelerator_active());
        assert!(
            HardwareCapabilities {
                silicon_generation: Some(10),
                macos_major: Some(26),
                macos_minor: Some(2),
                has_neural_accelerator: true,
                macos_supports_na: true,
            }
            .neural_accelerator_active()
        );
    }

    #[test]
    fn thread_override_restores_previous() {
        let _outer = override_hardware(HardwareCapabilities::m4());
        assert!(!neural_accelerator_active());
        {
            let _inner = override_hardware(HardwareCapabilities::m5_na());
            assert!(neural_accelerator_active());
        }
        assert!(!neural_accelerator_active());
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn local_host_reports_m_series_and_macos() {
        let hw = current();
        eprintln!(
            "live NA policy snapshot: silicon_generation={:?} macos={:?}.{:?} \
             has_neural_accelerator={} macos_supports_na={} active={}",
            hw.silicon_generation,
            hw.macos_major,
            hw.macos_minor,
            hw.has_neural_accelerator,
            hw.macos_supports_na,
            hw.neural_accelerator_active()
        );
        assert!(
            hw.silicon_generation.is_some(),
            "expected an Apple M-series generation, got {hw:?}"
        );
        assert!(
            hw.macos_major.is_some(),
            "expected a parsed macOS version, got {hw:?}"
        );
        assert_eq!(
            hw.has_neural_accelerator,
            hw.silicon_generation
                .is_some_and(|generation| generation >= 5)
        );
        assert_eq!(
            hw.macos_supports_na,
            macos_supports_neural_accelerator(hw.macos_major, hw.macos_minor)
        );
    }
}
