// SPDX-License-Identifier: Apache-2.0

//! Bounded residency policies for the two audited Tiel MXFP4 exports.
//!
//! M4 Pro 64 GiB sessions may retain optional experts when their known load
//! budget fits. M5 Max has a separate post-load wired-residency policy.
//!
//! Releasing wired residency reduced the idle-to-first-submit wait in the
//! M5 Max campaign without changing token output. This policy runs after
//! load; it does not change buffer-cache or allocation limits. Evidence:
//! `benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-wired/`.
//!
//! Metadata fingerprints identify the tested export configuration, not the
//! authenticity of every weight byte. Unknown exports or hardware retain
//! existing wiring. Numeric operator overrides and expert streaming take
//! precedence. Unwired buffers remain subject to OS eviction under pressure.

use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Fixed stable identifier for the automatic no-wire policy event.
const POLICY_ID: &str = "tiel-auto-no-wire-v1";

/// SHA256 of `config.json` for the tested export configuration.
const CONFIG_SHA256: &str = "0d647a1c2b2083419243bb7183baf35ed097fa129f41e7bd3b60437f3b2c8fb4";

/// SHA256 of `axquant_manifest.json` for the Tiel pack.
const TIEL_MANIFEST_SHA256: &str =
    "b8a532700e394c55a0d9895bd80969a4bf4df0f69e5b3942a33a50f962c1d776";

/// SHA256 of `axquant_manifest.json` for the Cyber pack.
const CYBER_MANIFEST_SHA256: &str =
    "ad320320e419eaaeacab5b8e14145bca88a7afe63aa7e16f85347f9074593573";

/// Minimum unified memory (128 GiB) below which the policy never applies.
const MIN_UNIFIED_MEMORY_BYTES: u64 = 128 * 1024 * 1024 * 1024;

/// The only trimmed `machdep.cpu.brand_string` the policy accepts.
const REQUIRED_BRAND: &str = "Apple M5 Max";

/// Per-file cap on metadata reads. Hashes cover small metadata, not weights.
const MAX_METADATA_BYTES: u64 = 1024 * 1024;

/// Resolved inputs for the pure residency decision. All I/O (env, file
/// reads, sysctl) is resolved before this is evaluated so each guard can be
/// varied independently in tests.
#[derive(Clone, Debug, Default, PartialEq)]
struct ResidencyDecision {
    /// `Some(scale)` when a valid numeric `AX_MLX_WIRED_LIMIT_SCALE` is set.
    wired_limit_scale_override: Option<f64>,
    /// True when expert streaming is active for this load.
    expert_streaming_active: bool,
    /// Number of *other* live runners currently holding wired residency.
    /// Clearing the process-wide wired limit would unwire those sibling models,
    /// so a non-zero count blocks the clear.
    sibling_wired_holders: usize,
    /// Hex SHA256 of `config.json` (`None` when unreadable/oversized).
    config_sha256: Option<String>,
    /// Hex SHA256 of `axquant_manifest.json` (`None` when unreadable/oversized).
    manifest_sha256: Option<String>,
    /// `hw.memsize` in bytes (`None` when unknown).
    unified_memory_bytes: Option<u64>,
    /// Trimmed `machdep.cpu.brand_string` (`None` when unknown).
    cpu_brand_string: Option<String>,
}

/// Pure residency decision: clear wired residency only when every guard
/// passes. Guards are ordered so the I/O path can read hardware last (after
/// the metadata digests already matched); here every input is pre-resolved.
fn decide_clear_wired_residency(inputs: &ResidencyDecision) -> bool {
    // Guard 1: preserve operator numeric override semantics.
    if inputs.wired_limit_scale_override.is_some() {
        return false;
    }
    // Guard 2: never clear while experts are paged.
    if inputs.expert_streaming_active {
        return false;
    }
    // Guard 3: never clear while a sibling runner holds wired residency.
    if inputs.sibling_wired_holders > 0 {
        return false;
    }
    // Guard 4: config digest must match the tested export configuration.
    if inputs.config_sha256.as_deref() != Some(CONFIG_SHA256) {
        return false;
    }
    // Guard 5: manifest digest must match the Tiel or Cyber pack.
    if !matches!(
        inputs.manifest_sha256.as_deref(),
        Some(TIEL_MANIFEST_SHA256) | Some(CYBER_MANIFEST_SHA256)
    ) {
        return false;
    }
    // Guard 6: hardware (>= 128 GiB and exactly `Apple M5 Max`).
    let enough_memory = inputs
        .unified_memory_bytes
        .is_some_and(|bytes| bytes >= MIN_UNIFIED_MEMORY_BYTES);
    if !enough_memory {
        return false;
    }
    if inputs.cpu_brand_string.as_deref() != Some(REQUIRED_BRAND) {
        return false;
    }
    true
}

/// Process-wide count of live runners currently holding wired residency.
static WIRED_RESIDENCY_HOLDERS: AtomicUsize = AtomicUsize::new(0);

/// RAII guard: marks one runner as holding wired residency and releases the
/// hold on drop, keeping [`WIRED_RESIDENCY_HOLDERS`] accurate so a sibling
/// model's wired residency is never silently cleared by a Tiel export loading
/// second.
pub(crate) struct WiredResidencyGuard {
    held: bool,
}

impl WiredResidencyGuard {
    /// Record that this runner wired residency (call after `set_wired_limit`).
    pub(crate) fn acquire() -> Self {
        WIRED_RESIDENCY_HOLDERS.fetch_add(1, Ordering::AcqRel);
        Self { held: true }
    }

    /// A guard that holds no residency (the `wired_cap == 0` path).
    pub(crate) fn unheld() -> Self {
        Self { held: false }
    }

    /// Whether this runner actually holds wired residency. The no-wire policy
    /// must subtract one from the process-wide holder count only when this is
    /// true; an unheld runner's own wired cap is zero.
    pub(crate) fn is_held(&self) -> bool {
        self.held
    }
}

impl Drop for WiredResidencyGuard {
    fn drop(&mut self) {
        if self.held {
            WIRED_RESIDENCY_HOLDERS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Number of live runners currently holding wired residency.
pub(crate) fn wired_residency_holders() -> usize {
    WIRED_RESIDENCY_HOLDERS.load(Ordering::Acquire)
}

/// Number of *sibling* runners holding wired residency: the process-wide count
/// minus one for this runner when it itself holds a guard. An unheld runner
/// (its own wired cap was zero) must not subtract, or a single sibling holder
/// would be miscounted as zero and its residency silently cleared.
fn sibling_holder_count(total_holders: usize, current_held: bool) -> usize {
    total_holders.saturating_sub(usize::from(current_held))
}

/// Exact metadata identity shared by the two independent residency policies.
fn audited_export(root: &Path) -> bool {
    let config = read_metadata_bounded(&root.join("config.json"))
        .map(|bytes| ax_engine_core::sha256_hex(&bytes));
    let manifest = read_metadata_bounded(&root.join("axquant_manifest.json"))
        .map(|bytes| ax_engine_core::sha256_hex(&bytes));
    config.as_deref() == Some(CONFIG_SHA256)
        && matches!(
            manifest.as_deref(),
            Some(TIEL_MANIFEST_SHA256) | Some(CYBER_MANIFEST_SHA256)
        )
}

pub(crate) fn session_auto_resident_fits(
    artifacts: &ax_engine_core::NativeModelArtifacts,
    budget: crate::expert_stream::SessionResidencyBudget,
    manifest: &crate::expert_stream::ExpertStreamManifest,
) -> bool {
    use crate::tiel_resident_budget::{ResidentBudgetInputs, permits_resident_load};
    use ax_engine_core::memory_budget::{estimated_footprint_bytes, estimated_kv_pool_bytes};

    // Keep unsupported models and required packs out before hardware/MLX probes.
    if manifest.required || !audited_export(artifacts.root_dir()) {
        return false;
    }
    let brand = cpu_brand_string();
    let physical = crate::expert_stream::unified_memory_bytes();
    if brand.as_deref() != Some("Apple M4 Pro") || physical != Some(64 * 1024 * 1024 * 1024) {
        return false;
    }
    let geometry = serde_json::to_value(artifacts.manifest())
        .ok()
        .and_then(|value| serde_json::from_value(value).ok());
    let kv = geometry
        .as_ref()
        .and_then(|geometry| estimated_kv_pool_bytes(geometry, budget.kv_pool_tokens));
    // Do not trust a smaller sidecar estimate over the actual tensor inventory.
    let tensor_bytes = artifacts
        .tensor_specs()
        .iter()
        .fold(0u64, |sum, tensor| sum.saturating_add(tensor.length_bytes));
    let weights = tensor_bytes.max(manifest.estimated_full_resident_bytes);
    let inputs = ResidentBudgetInputs {
        audited_export: true,
        required: manifest.required,
        cpu_brand: brand,
        physical_bytes: physical,
        pressure_level: crate::hardware::sysctl_string(&[
            "-n",
            "kern.memorystatus_vm_pressure_level",
        ])
        .and_then(|value| value.parse().ok()),
        working_set_bytes: mlx_sys::device_recommended_working_set_bytes(),
        active_bytes: mlx_sys::device_active_bytes(),
        kv_pool_tokens: budget.kv_pool_tokens,
        prefill_chunk: budget.prefill_chunk,
        footprint_bytes: kv.map(|bytes| estimated_footprint_bytes(weights, Some(bytes))),
    };
    let permitted = permits_resident_load(&inputs);
    tracing::info!(
        target: "ax_engine_mlx::runner", policy = "tiel-session-resident-v1",
        permitted, kv_pool_tokens = budget.kv_pool_tokens,
        prefill_chunk = budget.prefill_chunk,
        footprint_bytes = ?inputs.footprint_bytes, active_bytes = ?inputs.active_bytes,
        working_set_bytes = ?inputs.working_set_bytes, pressure_level = ?inputs.pressure_level,
        "evaluated bounded Tiel session residency"
    );
    permitted
}

/// Evaluate and apply the automatic no-wire residency policy after weights
/// are loaded.
///
/// Reads the metadata digests first and only probes hardware (spawning
/// `sysctl`) once those digests match, so non-matching models never fork a
/// process. On a pass, clears wired residency and traces an info event with
/// the stable [`POLICY_ID`] and no private paths.
pub(crate) fn maybe_clear_wired_residency(
    root: &Path,
    expert_streaming_active: bool,
    held_wired_residency: bool,
) {
    let wired_limit_scale_override = wired_limit_scale_override();
    if wired_limit_scale_override.is_some() || expert_streaming_active {
        tracing::debug!(
            target: "ax_engine_mlx::runner",
            policy = POLICY_ID,
            operator_override = wired_limit_scale_override.is_some(),
            expert_streaming_active,
            "automatic no-wire residency skipped: operator override or expert streaming"
        );
        return;
    }
    // Metadata digests gate the hardware probe: an unknown or modified export
    // configuration retains the existing policy and never spawns `sysctl`.
    let config_sha256 = read_metadata_bounded(&root.join("config.json"))
        .map(|bytes| ax_engine_core::sha256_hex(&bytes));
    let manifest_sha256 = read_metadata_bounded(&root.join("axquant_manifest.json"))
        .map(|bytes| ax_engine_core::sha256_hex(&bytes));
    if config_sha256.as_deref() != Some(CONFIG_SHA256)
        || !matches!(
            manifest_sha256.as_deref(),
            Some(TIEL_MANIFEST_SHA256) | Some(CYBER_MANIFEST_SHA256)
        )
    {
        tracing::debug!(
            target: "ax_engine_mlx::runner",
            policy = POLICY_ID,
            "automatic no-wire residency skipped: export metadata is unknown or unreadable"
        );
        return;
    }

    // Metadata matched; resolve the remaining guards and apply the pure
    // decision (which re-checks every guard as the single source of truth).
    // Subtract this runner only when it actually wired residency; an unheld
    // runner (its own wired cap was zero) must not subtract, or a single
    // sibling holder would be miscounted as zero and its residency cleared.
    let sibling_wired_holders =
        sibling_holder_count(wired_residency_holders(), held_wired_residency);
    let inputs = ResidencyDecision {
        wired_limit_scale_override,
        expert_streaming_active,
        sibling_wired_holders,
        config_sha256,
        manifest_sha256,
        unified_memory_bytes: crate::expert_stream::unified_memory_bytes(),
        cpu_brand_string: cpu_brand_string(),
    };
    if !decide_clear_wired_residency(&inputs) {
        if sibling_wired_holders > 0 {
            // A sibling model already wired residency; clearing the
            // process-wide limit would silently unwire it too. Log once per
            // process rather than on every recycle/load.
            static SIBLING_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if SIBLING_WARNED.swap(true, std::sync::atomic::Ordering::AcqRel) {
                return;
            }
            tracing::warn!(
                target: "ax_engine_mlx::runner",
                policy = POLICY_ID,
                sibling_wired_holders,
                "automatic no-wire residency skipped: a sibling model holds wired residency"
            );
            return;
        }
        // Metadata already matched an audited Tiel/Cyber export. The no-wire
        // policy itself stays M5 Max / >= 128 GiB. M4 Pro 64 GiB uses
        // tiel-session-resident-v1 and keeps this wiring decision. Warn once
        // per process: worker recycles and repeated loads would otherwise
        // repeat the same host-level message on every load.
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if WARNED.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        tracing::warn!(
            target: "ax_engine_mlx::runner",
            policy = POLICY_ID,
            unified_memory_bytes = ?inputs.unified_memory_bytes,
            cpu_brand = ?inputs.cpu_brand_string,
            "automatic no-wire residency kept the existing wiring policy: \
             tiel-auto-no-wire-v1 applies to Apple M5 Max hosts with at least \
             128 GiB. On Apple M4 Pro with exactly 64 GiB, expert residency is \
             decided by tiel-session-resident-v1"
        );
        return;
    }

    mlx_sys::set_wired_limit(0);
    tracing::info!(
        target: "ax_engine_mlx::runner",
        policy = POLICY_ID,
        "automatic no-wire residency policy applied; cleared process-wide wired residency after load"
    );
}

/// A valid numeric `AX_MLX_WIRED_LIMIT_SCALE` value, when present. Absent or
/// non-numeric values are not an operator override.
fn wired_limit_scale_override() -> Option<f64> {
    std::env::var("AX_MLX_WIRED_LIMIT_SCALE")
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Read a small metadata file, bounded to [`MAX_METADATA_BYTES`]. Returns
/// `None` for missing/unreadable/oversized files so the guard conservatively
/// retains the existing policy.
fn read_metadata_bounded(path: &Path) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::with_capacity(256);
    file.take(MAX_METADATA_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_METADATA_BYTES {
        return None;
    }
    Some(bytes)
}

/// Trimmed `machdep.cpu.brand_string`, or `None` when unknown.
fn cpu_brand_string() -> Option<String> {
    crate::hardware::sysctl_string(&["-n", "machdep.cpu.brand_string"])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matching_decision() -> ResidencyDecision {
        ResidencyDecision {
            wired_limit_scale_override: None,
            expert_streaming_active: false,
            sibling_wired_holders: 0,
            config_sha256: Some(CONFIG_SHA256.to_string()),
            manifest_sha256: Some(TIEL_MANIFEST_SHA256.to_string()),
            unified_memory_bytes: Some(MIN_UNIFIED_MEMORY_BYTES),
            cpu_brand_string: Some(REQUIRED_BRAND.to_string()),
        }
    }

    #[test]
    fn tiel_pack_full_match_clears() {
        assert!(decide_clear_wired_residency(&matching_decision()));
    }

    #[test]
    fn sibling_wired_holder_skips_clear() {
        // With a sibling runner holding wired residency the clear is skipped.
        let mut inputs = matching_decision();
        inputs.sibling_wired_holders = 1;
        assert!(!decide_clear_wired_residency(&inputs));
        // At zero (no sibling) the same inputs clear.
        let mut inputs = matching_decision();
        inputs.sibling_wired_holders = 0;
        assert!(decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn unheld_current_runner_does_not_miscount_sibling_holders() {
        // One other runner holds residency while the current guard is unheld
        // (its own wired cap was zero). Subtracting 1 unconditionally would
        // miscount zero siblings and clear the process-wide limit, unwiring
        // runner A; the sibling count must stay 1 and the clear is skipped.
        assert_eq!(sibling_holder_count(1, false), 1);
        assert!(!decide_clear_wired_residency(&ResidencyDecision {
            sibling_wired_holders: sibling_holder_count(1, false),
            ..matching_decision()
        }));
        // A held current runner still subtracts itself from the count.
        assert_eq!(sibling_holder_count(1, true), 0);
        // No holders plus an unheld current runner stays zero.
        assert_eq!(sibling_holder_count(0, false), 0);
    }

    #[test]
    fn wired_residency_guard_tracks_holders() {
        assert_eq!(wired_residency_holders(), 0);
        {
            let _guard = WiredResidencyGuard::acquire();
            assert_eq!(wired_residency_holders(), 1);
            {
                let _second = WiredResidencyGuard::acquire();
                assert_eq!(wired_residency_holders(), 2);
            }
            assert_eq!(wired_residency_holders(), 1);
        }
        assert_eq!(wired_residency_holders(), 0);
        // An unheld guard does not touch the count.
        {
            let _unheld = WiredResidencyGuard::unheld();
            assert_eq!(wired_residency_holders(), 0);
        }
        assert_eq!(wired_residency_holders(), 0);
    }

    #[test]
    fn cyber_pack_full_match_clears() {
        let mut inputs = matching_decision();
        inputs.manifest_sha256 = Some(CYBER_MANIFEST_SHA256.to_string());
        assert!(decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn other_config_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.config_sha256 = Some("0".repeat(64));
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn other_manifest_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.manifest_sha256 = Some("1".repeat(64));
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn unknown_metadata_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.config_sha256 = None;
        assert!(!decide_clear_wired_residency(&inputs));

        let mut inputs = matching_decision();
        inputs.manifest_sha256 = None;
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn unknown_memory_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.unified_memory_bytes = None;
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn smaller_host_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.unified_memory_bytes = Some(MIN_UNIFIED_MEMORY_BYTES - 1);
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn unknown_brand_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.cpu_brand_string = None;
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn wrong_brand_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.cpu_brand_string = Some("Apple M4 Max".to_string());
        assert!(!decide_clear_wired_residency(&inputs));

        let mut inputs = matching_decision();
        inputs.cpu_brand_string = Some("Apple M5 Ultra".to_string());
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn streaming_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.expert_streaming_active = true;
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn numeric_override_does_not_clear() {
        let mut inputs = matching_decision();
        inputs.wired_limit_scale_override = Some(0.0);
        assert!(!decide_clear_wired_residency(&inputs));

        let mut inputs = matching_decision();
        inputs.wired_limit_scale_override = Some(0.9);
        assert!(!decide_clear_wired_residency(&inputs));
    }

    #[test]
    fn no_override_is_not_a_blocker() {
        let mut inputs = matching_decision();
        inputs.wired_limit_scale_override = None;
        assert!(decide_clear_wired_residency(&inputs));
    }
    #[test]
    fn metadata_reader_rejects_missing_and_oversized_files() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("ax-tiel-metadata-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        assert!(read_metadata_bounded(&root.join("absent.json")).is_none());
        let path = root.join("metadata.json");
        let bytes = vec![b'x'; MAX_METADATA_BYTES as usize];
        std::fs::write(&path, &bytes)?;
        assert_eq!(
            read_metadata_bounded(&path).as_deref(),
            Some(bytes.as_slice())
        );
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        file.set_len(MAX_METADATA_BYTES + 1)?;
        assert!(read_metadata_bounded(&path).is_none());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
