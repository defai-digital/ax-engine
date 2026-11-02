// SPDX-License-Identifier: Apache-2.0

//! Runtime Apple Silicon Neural Accelerator detection.
//!
//! This is a **policy gate**, not an AX-owned NAX kernel dispatcher. MLX
//! selects NAX GEMM/SDPA internally from the admitted wheel. AX uses the
//! cached capability to:
//! - leave Metal command-buffer caps at MLX defaults on M5+ except families
//!   with measured M5 wins (`qwen3_next`);
//! - prefer native `mask="causal"` SDPA on allowlisted full-attn prefill
//!   chunks (`seq >= 1024`) so MLX can take NAX fused attention.
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
    pub(crate) fn neural_accelerator_active(self) -> bool {
        self.has_neural_accelerator && self.macos_supports_na
    }

    #[cfg(test)]
    pub(crate) fn m5_na() -> Self {
        Self {
            silicon_generation: Some(5),
            macos_major: Some(26),
            macos_minor: Some(6),
            has_neural_accelerator: true,
            macos_supports_na: true,
        }
    }

    #[cfg(test)]
    pub(crate) fn m4() -> Self {
        Self {
            silicon_generation: Some(4),
            macos_major: Some(26),
            macos_minor: Some(6),
            has_neural_accelerator: false,
            macos_supports_na: true,
        }
    }

    #[cfg(test)]
    pub(crate) fn m5_old_macos() -> Self {
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
pub(crate) fn current() -> HardwareCapabilities {
    #[cfg(test)]
    if let Some(hw) = OVERRIDE.with(|slot| slot.get()) {
        return hw;
    }
    *DETECTED.get_or_init(detect_hardware)
}

/// True on M5+ running macOS 26.2+.
#[inline]
pub(crate) fn neural_accelerator_active() -> bool {
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
pub(crate) fn override_hardware(hw: HardwareCapabilities) -> HardwareOverrideGuard {
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
        has_neural_accelerator: generation_has_neural_accelerator(silicon_generation),
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

/// Every Apple M-series generation from M5 on carries the GPU neural
/// accelerators. NAX policy is "M5 or later", never an M5-only match, so new
/// generations (M6, ...) take the NAX paths without code changes.
pub(crate) fn generation_has_neural_accelerator(generation: Option<u32>) -> bool {
    generation.is_some_and(|generation| generation >= 5)
}

/// First MLX release line carrying `affine_qmv_nax` (skinny 6-16 row quantized
/// GEMV on the neural accelerators; MLX 914732419, reported as 0.32.5).
const MLX_SKINNY_QMV_NAX_VERSION: (u32, u32, u32) = (0, 32, 5);

/// True on M5+ / macOS 26.2+ when the linked MLX also has the skinny-row NAX
/// quantized GEMV. AX's own verify QMM kernels were admitted against the
/// older stock curve; this gate lets them hand wider verify windows back to
/// stock MLX without touching M3/M4 or older MLX builds.
pub(crate) fn skinny_qmv_nax_active() -> bool {
    #[cfg(test)]
    if let Some(value) = SKINNY_OVERRIDE.with(|slot| slot.get()) {
        return value;
    }
    nax_verify_kernel_status().active
}

/// What `skinny_qmv_nax_active` decided and why; surfaced by `ax-engine doctor`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NaxVerifyKernelStatus {
    /// M5 or later on macOS 26.2+.
    pub neural_accelerator: bool,
    /// Version string the linked MLX runtime reports.
    pub mlx_version: Option<String>,
    /// The linked MLX is 0.32.5 or newer.
    pub version_ok: bool,
    /// Whether the linked `mlx.metallib` carries `qmv_nax`; `None` when it
    /// could not be located.
    pub metallib_has_qmv_nax: Option<bool>,
    /// The verify QMM / SDPA hand-offs to stock NAX kernels are enabled.
    pub active: bool,
}

/// Resolved once per process from the host and the loaded MLX runtime.
pub fn nax_verify_kernel_status() -> NaxVerifyKernelStatus {
    static LINKED: OnceLock<(Option<String>, bool, Option<bool>)> = OnceLock::new();
    let neural_accelerator = neural_accelerator_active();
    if !neural_accelerator {
        return NaxVerifyKernelStatus::default();
    }
    let (mlx_version, version_ok, kernels) = LINKED
        .get_or_init(|| {
            let raw = mlx_sys::runtime_version().ok();
            let version_ok = raw
                .as_deref()
                .and_then(parse_semver)
                .is_some_and(|version| version >= MLX_SKINNY_QMV_NAX_VERSION);
            // A runtime built below macOS 26.2 reports the same version with
            // the NAX kernels compiled out; stock QMM would then be the slow
            // pre-NAX curve. Fail open only when no metallib can be found.
            let kernels = linked_metallib_has(NAX_QMV_KERNEL_MARKER);
            tracing::info!(
                target = "ax_engine_mlx",
                version_ok,
                metallib_has_qmv_nax = ?kernels,
                active = version_ok && kernels != Some(false),
                "MLX skinny NAX verify kernels"
            );
            (raw, version_ok, kernels)
        })
        .clone();
    NaxVerifyKernelStatus {
        neural_accelerator,
        active: version_ok && kernels != Some(false),
        mlx_version,
        version_ok,
        metallib_has_qmv_nax: kernels,
    }
}

/// Kernel-name fragment every MLX `affine_qmv_nax` instantiation carries.
const NAX_QMV_KERNEL_MARKER: &[u8] = b"qmv_nax";

/// Whether the `mlx.metallib` next to the loaded `libmlx.dylib` contains
/// `needle`; `None` when that metallib cannot be located or read.
fn linked_metallib_has(needle: &[u8]) -> Option<bool> {
    let lib = loaded_image_path("libmlx.dylib")?;
    let metallib = std::path::Path::new(&lib).parent()?.join("mlx.metallib");
    file_contains(&metallib, needle)
}

#[cfg(target_os = "macos")]
fn loaded_image_path(file_name: &str) -> Option<String> {
    unsafe extern "C" {
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_name(index: u32) -> *const std::ffi::c_char;
    }
    // SAFETY: libSystem's dyld image list; names are NUL-terminated paths that
    // stay valid while the image is loaded, and libmlx is never unloaded.
    let count = unsafe { _dyld_image_count() };
    (0..count).find_map(|index| {
        let name = unsafe { _dyld_get_image_name(index) };
        if name.is_null() {
            return None;
        }
        let path = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
        path.ends_with(&format!("/{file_name}"))
            .then(|| path.into_owned())
    })
}

#[cfg(not(target_os = "macos"))]
fn loaded_image_path(_file_name: &str) -> Option<String> {
    None
}

/// Streams `path` in chunks looking for `needle` (chunks overlap by
/// `needle.len() - 1` bytes so a match across a boundary is found).
fn file_contains(path: &std::path::Path, needle: &[u8]) -> Option<bool> {
    use std::io::Read;
    if needle.is_empty() {
        return Some(true);
    }
    let mut file = std::fs::File::open(path).ok()?;
    let mut buffer = vec![0_u8; 8 << 20];
    let mut carried = 0;
    loop {
        let read = file.read(&mut buffer[carried..]).ok()?;
        if read == 0 {
            return Some(false);
        }
        let filled = carried + read;
        let window = &buffer[..filled];
        let first = needle[0];
        let mut start = 0;
        while let Some(offset) = window[start..].iter().position(|&byte| byte == first) {
            let at = start + offset;
            if window.len() - at < needle.len() {
                break;
            }
            if &window[at..at + needle.len()] == needle {
                return Some(true);
            }
            start = at + 1;
        }
        carried = (needle.len() - 1).min(filled);
        buffer.copy_within(filled - carried..filled, 0);
    }
}

#[cfg(test)]
thread_local! {
    static SKINNY_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Restores the previous skinny-NAX override on drop (tests).
#[cfg(test)]
pub struct SkinnyNaxOverrideGuard {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for SkinnyNaxOverrideGuard {
    fn drop(&mut self) {
        SKINNY_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
pub(crate) fn override_skinny_qmv_nax(value: bool) -> SkinnyNaxOverrideGuard {
    let previous = SKINNY_OVERRIDE.with(|slot| slot.replace(Some(value)));
    SkinnyNaxOverrideGuard { previous }
}

pub(crate) fn parse_semver(raw: &str) -> Option<(u32, u32, u32)> {
    let mut parts = raw.trim().split(|ch: char| !ch.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    Some((major, minor, patch))
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
        .and_then(|brand| parse_apple_m_series_generation(&brand))
}

fn detect_macos_version() -> (Option<u32>, Option<u32>) {
    if !cfg!(target_os = "macos") {
        return (None, None);
    }
    let raw = sysctl_string(&["-n", "kern.osproductversion"])
        .or_else(|| command_stdout("/usr/bin/sw_vers", &["-productVersion"]));
    raw.map(|version| parse_macos_major_minor(&version))
        .unwrap_or((None, None))
}

/// Read a sysctl value even when a service's PATH omits /usr/sbin.
pub(crate) fn sysctl_string(args: &[&str]) -> Option<String> {
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
    fn neural_accelerator_policy_covers_m5_and_later() {
        for (brand, expected) in [
            ("Apple M3 Max", false),
            ("Apple M4 Pro", false),
            ("Apple M5", true),
            ("Apple M5 Max", true),
            ("Apple M6", true),
            ("Apple M6 Max", true),
            ("Apple M6 Ultra", true),
            ("Apple M7 Pro", true),
        ] {
            assert_eq!(
                generation_has_neural_accelerator(parse_apple_m_series_generation(brand)),
                expected,
                "{brand}"
            );
        }
        assert!(!generation_has_neural_accelerator(None));
    }

    #[test]
    fn parses_mlx_versions_for_the_skinny_nax_gate() {
        assert_eq!(parse_semver("0.32.5"), Some((0, 32, 5)));
        assert_eq!(parse_semver("0.32.4"), Some((0, 32, 4)));
        assert_eq!(parse_semver("0.33.0.dev20261102"), Some((0, 33, 0)));
        assert_eq!(parse_semver("1.0"), Some((1, 0, 0)));
        assert_eq!(parse_semver("dev"), None);
        assert!((0, 32, 4) < MLX_SKINNY_QMV_NAX_VERSION);
        assert!((0, 33, 0) >= MLX_SKINNY_QMV_NAX_VERSION);
    }

    #[test]
    fn metallib_probe_finds_markers_across_chunk_boundaries() {
        let dir = std::env::temp_dir().join(format!("ax-metallib-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("probe.bin");
        // Marker straddling the 8 MiB chunk boundary.
        let mut data = vec![b'x'; (8 << 20) + 64];
        let at = (8 << 20) - 3;
        data[at..at + NAX_QMV_KERNEL_MARKER.len()].copy_from_slice(NAX_QMV_KERNEL_MARKER);
        std::fs::write(&path, &data).unwrap();
        assert_eq!(file_contains(&path, NAX_QMV_KERNEL_MARKER), Some(true));
        std::fs::write(&path, b"affine_qmv_fast qmm_t steel").unwrap();
        assert_eq!(file_contains(&path, NAX_QMV_KERNEL_MARKER), Some(false));
        assert_eq!(
            file_contains(&dir.join("missing"), NAX_QMV_KERNEL_MARKER),
            None
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn locates_the_loaded_libmlx_and_its_metallib() {
        let _ = mlx_sys::runtime_version();
        let lib = loaded_image_path("libmlx.dylib").expect("libmlx is loaded");
        let metallib = std::path::Path::new(&lib)
            .parent()
            .unwrap()
            .join("mlx.metallib");
        assert!(metallib.exists(), "{}", metallib.display());
        let has = linked_metallib_has(NAX_QMV_KERNEL_MARKER);
        eprintln!(
            "linked metallib {} has qmv_nax: {has:?}",
            metallib.display()
        );
        assert!(has.is_some());
    }

    #[test]
    fn skinny_nax_override_restores_previous() {
        {
            let _on = override_skinny_qmv_nax(true);
            assert!(skinny_qmv_nax_active());
            let _off = override_skinny_qmv_nax(false);
            assert!(!skinny_qmv_nax_active());
        }
        let _m4 = override_hardware(HardwareCapabilities::m4());
        assert!(!skinny_qmv_nax_active());
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
