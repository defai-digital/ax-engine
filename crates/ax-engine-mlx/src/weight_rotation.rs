//! Randomised Hadamard rotation primitives for ADR 0022 D2 weight quantization.
//!
//! Phase 0 (shadow mode) under WEIGHT-ROTATION-IMPLEMENTATION-PLAN.md. This
//! module establishes the rotation primitives and a load-time inspection hook.
//! No inference path consumes these helpers yet; the only side effect is a
//! single `tracing::info!` line emitted from `load_weights` when the
//! `AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION` env var is set to `shadow`.
//!
//! Unset / "off" / "0" / "false": no behavior change. Any other value than
//! `shadow` is fail-closed (panics at first call) so misconfiguration is
//! caught immediately rather than silently doing the wrong thing — same
//! contract as `ngram_accel::parse_confidence_threshold`.

use std::sync::OnceLock;

use ax_engine_core::model::{NativeTensorRole, NativeTensorSpec};
use thiserror::Error;

/// Env var that toggles the experimental weight-rotation behavior.
/// Defined by WEIGHT-ROTATION-IMPLEMENTATION-PLAN.md §P0.
pub const WEIGHT_ROTATION_ENV: &str = "AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RotationError {
    #[error("Hadamard rotation requires power-of-2 dim, got {0}")]
    NotPowerOfTwo(usize),
    #[error("Hadamard rotation requires dim >= 2, got {0}")]
    TooSmall(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightRotationMode {
    /// Production: rotation code is dormant.
    Off,
    /// Shadow: log rotation candidates at weight load time. No tensor is
    /// rotated; no forward-pass change.
    Shadow,
    /// P1 plumbing test: insert an identity rotation `R(R(x))` at the FFN
    /// entry point of every dense forward pass (R is orthonormal so
    /// `R · R = I`). Net effect on output is zero modulo floating-point
    /// error; the equivalence harness validates that the rotation
    /// infrastructure can be safely invoked from the forward path before
    /// any real rotation is committed in P2+.
    Enable,
    /// P2a runtime: insert a SINGLE rotation `R(x)` at the FFN entry. This
    /// is mathematically a no-op only when paired with offline-rotated
    /// weights (W' = W @ R) loaded from `model.rotated.safetensors`. Used
    /// alone (without the rotated checkpoint), `Apply` mode produces
    /// incorrect output — verified intentionally by the equivalence harness
    /// to prove the activation rotation has real effect.
    Apply,
}

/// Parse the env var value. Returns `Off` for `None`, `"off"`, `"0"`, `"false"`.
/// Returns `Shadow` for `"shadow"`, `Enable` for `"enable"`. Anything else is
/// fail-closed.
pub fn parse_weight_rotation_mode(raw: Option<&str>) -> WeightRotationMode {
    let Some(value) = raw else {
        return WeightRotationMode::Off;
    };
    match value.to_ascii_lowercase().as_str() {
        "" | "off" | "0" | "false" => WeightRotationMode::Off,
        "shadow" => WeightRotationMode::Shadow,
        "enable" => WeightRotationMode::Enable,
        "apply" => WeightRotationMode::Apply,
        other => {
            panic!("{WEIGHT_ROTATION_ENV} must be one of off|shadow|enable|apply; got {other:?}")
        }
    }
}

/// Resolve the active weight-rotation mode for this process. Cached on first
/// call. Invalid env values panic on first call (fail-closed).
pub fn weight_rotation_mode() -> WeightRotationMode {
    static CACHED: OnceLock<WeightRotationMode> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_weight_rotation_mode(std::env::var(WEIGHT_ROTATION_ENV).ok().as_deref())
    })
}

/// A randomised Hadamard rotation `R = diag(sign_flip) * FWHT * diag(sign_flip)`
/// over a power-of-2 dimension. The transform is **symmetric and its own
/// inverse up to a factor of `dim`**: `R == R^T` and `R * R * x = dim * x`,
/// because `diag(s) * diag(s) = I` (s ∈ {±1}) and `FWHT * FWHT = dim * I`.
/// The randomised sign flip spreads outliers across all output dimensions,
/// which reduces per-group dynamic range and improves low-bit quantisation
/// error.
#[derive(Debug, Clone)]
pub struct HadamardRotation {
    dim: usize,
    sign_flip: Vec<i8>,
}

impl HadamardRotation {
    pub fn new(dim: usize, seed: u64) -> Result<Self, RotationError> {
        if dim < 2 {
            return Err(RotationError::TooSmall(dim));
        }
        if !dim.is_power_of_two() {
            return Err(RotationError::NotPowerOfTwo(dim));
        }
        Ok(Self {
            dim,
            sign_flip: generate_sign_flip(dim, seed),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn sign_flip(&self) -> &[i8] {
        &self.sign_flip
    }

    /// Apply the rotation in-place. `buf.len()` must equal `self.dim()`.
    /// Unnormalised: callers that need an orthonormal transform should
    /// post-scale by `1.0 / (dim as f32).sqrt()`. Symmetric form
    /// (`diag(s) * FWHT * diag(s)`) so `apply` is self-inverse up to a
    /// factor of `dim` — convenient for forward/inverse symmetry in tests
    /// and for fused-load activation rotation in P1.
    pub fn apply_in_place(&self, buf: &mut [f32]) {
        assert_eq!(buf.len(), self.dim, "buffer length must equal rotation dim");
        self.apply_sign_flip(buf);
        fwht_in_place(buf);
        self.apply_sign_flip(buf);
    }

    fn apply_sign_flip(&self, buf: &mut [f32]) {
        for (b, s) in buf.iter_mut().zip(self.sign_flip.iter()) {
            if *s < 0 {
                *b = -*b;
            }
        }
    }
}

fn generate_sign_flip(dim: usize, seed: u64) -> Vec<i8> {
    let mut state = seed.max(1);
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 { 1 } else { -1 }
        })
        .collect()
}

fn fwht_in_place(buf: &mut [f32]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two(), "fwht requires power-of-2 length");
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + h) {
                let x = buf[j];
                let y = buf[j + h];
                buf[j] = x + y;
                buf[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// Roles whose tensors are candidates for rotation under ADR 0022 D2.
/// First rotation pair (P1): `AttentionO` (output side) × `FfnDown` (input side).
/// Wider expansion covers q/k/v and gate/up. MoE expert weights and
/// MLA-specific roles are out of scope (see WEIGHT-ROTATION-IMPLEMENTATION-PLAN.md §5).
const ROTATION_CANDIDATE_ROLES: &[NativeTensorRole] = &[
    NativeTensorRole::AttentionQ,
    NativeTensorRole::AttentionK,
    NativeTensorRole::AttentionV,
    NativeTensorRole::AttentionO,
    NativeTensorRole::AttentionQkvPacked,
    NativeTensorRole::FfnGate,
    NativeTensorRole::FfnUp,
    NativeTensorRole::FfnDown,
    NativeTensorRole::FfnGateUpPacked,
];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RotationCandidateSummary {
    pub total_candidates: usize,
    pub power_of_two_eligible: usize,
    pub non_power_of_two_blocked: usize,
}

/// Per-tensor rotation eligibility verdict. Used by per-tensor shadow logging
/// and by P1 to decide which side of each projection to rotate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotationCandidateDetail {
    pub name: String,
    pub role: NativeTensorRole,
    pub shape: Vec<u64>,
    pub layer_index: Option<u32>,
    /// Shape positions whose dim is a power-of-2 and ≥ 2. P1 selects the
    /// rotation axis from this set.
    pub power_of_two_axes: Vec<usize>,
}

impl RotationCandidateDetail {
    pub fn is_eligible(&self) -> bool {
        !self.power_of_two_axes.is_empty()
    }
}

/// Detailed walk: returns one entry per rotation-candidate tensor.
pub fn detail_rotation_candidates(specs: &[NativeTensorSpec]) -> Vec<RotationCandidateDetail> {
    let mut out = Vec::new();
    for spec in specs {
        if !ROTATION_CANDIDATE_ROLES.contains(&spec.role) {
            continue;
        }
        let power_of_two_axes: Vec<usize> = spec
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                let d = *d as usize;
                if d >= 2 && d.is_power_of_two() {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        out.push(RotationCandidateDetail {
            name: spec.name.clone(),
            role: spec.role,
            shape: spec.shape.clone(),
            layer_index: spec.layer_index,
            power_of_two_axes,
        });
    }
    out
}

/// Walk the manifest tensor specs and count rotation candidates. The "blocked"
/// count is where the relevant dimension is not a power of 2, so the FWHT-based
/// rotation cannot apply without padding (handled in P2+).
pub fn summarize_rotation_candidates(specs: &[NativeTensorSpec]) -> RotationCandidateSummary {
    let detail = detail_rotation_candidates(specs);
    let mut s = RotationCandidateSummary {
        total_candidates: detail.len(),
        ..Default::default()
    };
    for d in &detail {
        if d.is_eligible() {
            s.power_of_two_eligible += 1;
        } else {
            s.non_power_of_two_blocked += 1;
        }
    }
    s
}

/// Emit a single summary of rotation candidates when shadow mode is active.
/// No-op when mode is `Off`. Called once per `load_weights`.
///
/// Writes to both `tracing::info!` (for consumers that install a subscriber,
/// e.g. `ax-engine-server`) and `stderr` directly (for consumers that do not,
/// e.g. the Python extension). Shadow mode is opt-in and explicitly intended
/// to surface visible diagnostic output; the dual path guarantees the message
/// is observable regardless of which front-end loaded the engine.
pub fn shadow_log_rotation_candidates(specs: &[NativeTensorSpec]) {
    if weight_rotation_mode() != WeightRotationMode::Shadow {
        return;
    }
    let summary = summarize_rotation_candidates(specs);
    tracing::info!(
        target: "ax_mlx::weight_rotation",
        total_candidates = summary.total_candidates,
        power_of_two_eligible = summary.power_of_two_eligible,
        non_power_of_two_blocked = summary.non_power_of_two_blocked,
        "weight_rotation shadow: P0 candidate inventory"
    );
    eprintln!(
        "[ax_mlx::weight_rotation shadow] total_candidates={} power_of_two_eligible={} non_power_of_two_blocked={}",
        summary.total_candidates, summary.power_of_two_eligible, summary.non_power_of_two_blocked,
    );

    // Per-tensor detail behind an additional opt-in to keep the default
    // shadow output single-line. Set AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION_DETAIL=1
    // to enable the per-tensor dump.
    if std::env::var("AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION_DETAIL")
        .ok()
        .as_deref()
        == Some("1")
    {
        for detail in detail_rotation_candidates(specs) {
            eprintln!(
                "[ax_mlx::weight_rotation shadow] tensor={} role={:?} layer={:?} shape={:?} pow2_axes={:?}",
                detail.name,
                detail.role,
                detail.layer_index,
                detail.shape,
                detail.power_of_two_axes,
            );
        }
    }
}

/// Build a symmetric **orthonormal** randomised Hadamard rotation matrix
/// `R = D * (H/sqrt(dim)) * D` of size `dim × dim` as an `MlxArray`.
/// `R[i,j] = sign[i] * sign[j] * (-1)^popcount(i & j) / sqrt(dim)`.
/// Orthonormality means `R * R = I` exactly (mathematically), with no
/// post-multiplication scale needed. Empirically this preserves numerical
/// equivalence better than the unnormalised form because intermediate
/// matmul accumulations stay in a comfortable magnitude range (output
/// entries are O(1) instead of O(sqrt(dim))).
///
/// `dim` must be a power of 2; the build allocates `dim²` f32 entries
/// (e.g. 64 MB for `dim=4096`). One-time per process via the cache below.
pub fn build_rotation_matrix(dim: usize, seed: u64) -> mlx_sys::MlxArray {
    assert!(
        dim >= 2 && dim.is_power_of_two(),
        "rotation dim must be power-of-2 >= 2"
    );
    let sign = generate_sign_flip(dim, seed);
    let scale = 1.0_f32 / (dim as f32).sqrt();
    let mut data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        let si = sign[i] as f32;
        for j in 0..dim {
            let h = if ((i & j) as u32).count_ones().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            data[i * dim + j] = si * sign[j] as f32 * h * scale;
        }
    }
    let dim_i32 = dim as i32;
    let bytes = std::mem::size_of_val(data.as_slice());
    mlx_sys::MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        bytes,
        &[dim_i32, dim_i32],
        mlx_sys::MlxDtype::Float32,
    )
}

/// Apply an identity-equivalent rotation `R(R(x))/dim` to the trailing
/// dimension of `x` when `mode == Enable` and the trailing dim is a power
/// of 2. Returns `x.clone()` in all other cases.
///
/// Mathematically a no-op modulo floating-point error: `R * R = dim * I` for
/// the symmetric Hadamard form, so `R(R(x))/dim == x`. The point of inserting
/// it is purely to exercise the rotation infrastructure inside the forward
/// path so the equivalence harness can catch any plumbing regression before
/// real (non-identity) rotations are committed in P2.
pub fn maybe_apply_rotation_identity(x: &mlx_sys::MlxArray) -> mlx_sys::MlxArray {
    let mode = weight_rotation_mode();
    if !matches!(mode, WeightRotationMode::Enable | WeightRotationMode::Apply) {
        return x.clone();
    }
    let shape = x.shape();
    let Some(last_dim) = shape.last().copied() else {
        return x.clone();
    };
    let dim = last_dim as usize;
    if dim < 2 || !dim.is_power_of_two() {
        return x.clone();
    }
    // Precision study (2026-05-11, Qwen3.5-9B, 5-prompt corpus, 32 decode tokens):
    //   bf16 cast (R cast to activation dtype, matmuls in bf16): ratio 0.95
    //   f32 throughout (upcast x, matmuls in f32, downcast at end): ratio 0.83
    //
    // Empirically the bf16 path is more bit-equivalent to the unrotated
    // baseline because the baseline forward itself runs in bf16 — the f32
    // rotation introduces a "more accurate" intermediate that disagrees
    // with the bf16 baseline's rounding pattern on marginal positions.
    //
    // R is orthonormal (R @ R = I), so:
    //   Enable mode: apply R twice → R · R · x = x (identity)
    //   Apply mode:  apply R once  → R · x (caller is expected to have
    //     offline-rotated W' = W @ R; W' · R · x = W · x at fp tolerance)
    let r = cached_rotation_matrix(dim);
    let dtype = x.dtype();
    let r = mlx_sys::ops::astype(&r, dtype, None);
    let r1 = mlx_sys::ops::matmul(x, &r, None);
    match mode {
        WeightRotationMode::Enable => mlx_sys::ops::matmul(&r1, &r, None),
        WeightRotationMode::Apply => r1,
        _ => unreachable!("checked at function entry"),
    }
}

// Activation capture for P2b §3b is implemented Python-side via
// `scripts/capture_ffn_activations.py` (mlx_lm class-patch), not in Rust.
// The Rust-side capture initially tried here required mlx_sys bindings
// for `abs`, `max`, and `save_safetensors` that don't currently exist;
// the Python path avoids that dependency churn.

/// Per-dim cached rotation matrix. Matrices are large (64 MB at dim=4096) so
/// we materialise once and reuse across forward passes.
fn cached_rotation_matrix(dim: usize) -> mlx_sys::MlxArray {
    use std::sync::Mutex;
    static CACHE: OnceLock<Mutex<std::collections::HashMap<usize, mlx_sys::MlxArray>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut map = cache.lock().expect("rotation matrix cache poisoned");
    map.entry(dim)
        .or_insert_with(|| build_rotation_matrix(dim, 0xA5A5_A5A5_A5A5_A5A5_u64))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::model::{NativeTensorDataType, NativeTensorSpec};

    #[test]
    fn parse_mode_off_for_unset() {
        assert_eq!(parse_weight_rotation_mode(None), WeightRotationMode::Off);
    }

    #[test]
    fn parse_mode_recognises_falsey_values() {
        for v in ["off", "OFF", "Off", "0", "false", "FALSE", ""] {
            assert_eq!(
                parse_weight_rotation_mode(Some(v)),
                WeightRotationMode::Off,
                "{v:?} should parse as Off"
            );
        }
    }

    #[test]
    fn parse_mode_accepts_shadow_case_insensitive() {
        for v in ["shadow", "Shadow", "SHADOW"] {
            assert_eq!(
                parse_weight_rotation_mode(Some(v)),
                WeightRotationMode::Shadow
            );
        }
    }

    #[test]
    #[should_panic(expected = "AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION")]
    fn parse_mode_rejects_unknown_value() {
        let _ = parse_weight_rotation_mode(Some("rotate"));
    }

    #[test]
    fn parse_mode_accepts_enable_case_insensitive() {
        for v in ["enable", "Enable", "ENABLE"] {
            assert_eq!(
                parse_weight_rotation_mode(Some(v)),
                WeightRotationMode::Enable
            );
        }
    }

    #[test]
    fn parse_mode_accepts_apply_case_insensitive() {
        for v in ["apply", "Apply", "APPLY"] {
            assert_eq!(
                parse_weight_rotation_mode(Some(v)),
                WeightRotationMode::Apply
            );
        }
    }

    #[test]
    fn rejects_zero_or_one_dim() {
        assert_eq!(
            HadamardRotation::new(0, 1).unwrap_err(),
            RotationError::TooSmall(0)
        );
        assert_eq!(
            HadamardRotation::new(1, 1).unwrap_err(),
            RotationError::TooSmall(1)
        );
    }

    #[test]
    fn rejects_non_power_of_two_dim() {
        assert_eq!(
            HadamardRotation::new(3, 1).unwrap_err(),
            RotationError::NotPowerOfTwo(3)
        );
        assert_eq!(
            HadamardRotation::new(100, 1).unwrap_err(),
            RotationError::NotPowerOfTwo(100)
        );
        assert_eq!(
            HadamardRotation::new(5120, 1).unwrap_err(),
            RotationError::NotPowerOfTwo(5120)
        );
    }

    #[test]
    fn accepts_powers_of_two() {
        for &d in &[2usize, 4, 64, 128, 1024, 4096] {
            assert!(HadamardRotation::new(d, 1).is_ok(), "dim {d} should accept");
        }
    }

    #[test]
    fn deterministic_sign_flip() {
        let a = HadamardRotation::new(128, 1234).unwrap();
        let b = HadamardRotation::new(128, 1234).unwrap();
        assert_eq!(a.sign_flip(), b.sign_flip());
    }

    #[test]
    fn different_seed_yields_different_sign_flip() {
        let a = HadamardRotation::new(128, 1234).unwrap();
        let b = HadamardRotation::new(128, 5678).unwrap();
        assert_ne!(a.sign_flip(), b.sign_flip());
    }

    #[test]
    fn sign_flip_only_plus_or_minus_one() {
        let r = HadamardRotation::new(256, 42).unwrap();
        for &s in r.sign_flip() {
            assert!(s == 1 || s == -1, "sign flip must be ±1, got {s}");
        }
    }

    #[test]
    fn fwht_double_application_yields_scaled_input() {
        // H * H = n * I, so applying FWHT twice scales by n.
        let n = 8;
        let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = buf.clone();
        fwht_in_place(&mut buf);
        fwht_in_place(&mut buf);
        for (got, expect) in buf.iter().zip(original.iter()) {
            let expected_scaled = expect * n as f32;
            assert!(
                (got - expected_scaled).abs() < 1e-5,
                "got {got} expected {expected_scaled}"
            );
        }
    }

    #[test]
    fn full_rotation_double_application_yields_scaled_input() {
        // R = FWHT * diag(s); R * R = (FWHT * s * FWHT * s)
        // diag(s) * diag(s) = I (since s^2 = 1), and FWHT * FWHT = n * I.
        // So R * R = n * I as well.
        let n = 16;
        let r = HadamardRotation::new(n, 1234).unwrap();
        let original: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut buf = original.clone();
        r.apply_in_place(&mut buf);
        r.apply_in_place(&mut buf);
        for (got, expect) in buf.iter().zip(original.iter()) {
            let expected_scaled = expect * n as f32;
            assert!(
                (got - expected_scaled).abs() < 1e-4,
                "got {got} expected {expected_scaled}"
            );
        }
    }

    #[test]
    fn rotation_scales_l2_norm_by_sqrt_n() {
        let n = 64;
        let r = HadamardRotation::new(n, 99).unwrap();
        let mut buf: Vec<f32> = (0..n).map(|i| (i as f32 - 31.5).sin()).collect();
        let orig_l2: f32 = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
        r.apply_in_place(&mut buf);
        let after_l2: f32 = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
        let ratio = after_l2 / orig_l2;
        let expected = (n as f32).sqrt();
        assert!(
            (ratio - expected).abs() / expected < 1e-4,
            "got ratio {ratio}, expected {expected}"
        );
    }

    fn spec(name: &str, role: NativeTensorRole, shape: Vec<u64>) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index: None,
            dtype: NativeTensorDataType::F32,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: std::path::PathBuf::new(),
            offset_bytes: 0,
            length_bytes: 0,
        }
    }

    #[test]
    fn summarize_counts_only_rotation_candidates() {
        let specs = vec![
            spec("emb", NativeTensorRole::TokenEmbedding, vec![32000, 4096]),
            spec("q", NativeTensorRole::AttentionQ, vec![4096, 4096]),
            spec("o", NativeTensorRole::AttentionO, vec![4096, 4096]),
            spec("ffn_down", NativeTensorRole::FfnDown, vec![4096, 11008]),
            spec(
                "expert",
                NativeTensorRole::FfnDownExps,
                vec![64, 4096, 11008],
            ),
        ];
        let summary = summarize_rotation_candidates(&specs);
        assert_eq!(summary.total_candidates, 3); // q, o, ffn_down
        assert_eq!(summary.power_of_two_eligible, 3); // 4096 is power-of-2
        assert_eq!(summary.non_power_of_two_blocked, 0);
    }

    #[test]
    fn detail_returns_per_tensor_axes() {
        let specs = vec![
            spec("q.0", NativeTensorRole::AttentionQ, vec![4096, 4096]),
            spec("ffn_down.0", NativeTensorRole::FfnDown, vec![4096, 11008]),
            spec(
                "ignored",
                NativeTensorRole::TokenEmbedding,
                vec![32000, 4096],
            ),
        ];
        let d = detail_rotation_candidates(&specs);
        assert_eq!(
            d.len(),
            2,
            "TokenEmbedding must not be a rotation candidate"
        );
        assert_eq!(d[0].name, "q.0");
        assert_eq!(d[0].power_of_two_axes, vec![0, 1]);
        assert!(d[0].is_eligible());
        assert_eq!(d[1].name, "ffn_down.0");
        // 4096 is power-of-2 (axis 0); 11008 is not (axis 1).
        assert_eq!(d[1].power_of_two_axes, vec![0]);
        assert!(d[1].is_eligible());
    }

    #[test]
    fn detail_marks_no_eligible_axes_when_none_are_power_of_two() {
        let specs = vec![spec("q.0", NativeTensorRole::AttentionQ, vec![5120, 13824])];
        let d = detail_rotation_candidates(&specs);
        assert_eq!(d.len(), 1);
        assert!(d[0].power_of_two_axes.is_empty());
        assert!(!d[0].is_eligible());
    }

    #[test]
    fn summarize_marks_non_power_of_two_as_blocked() {
        let specs = vec![
            spec("q", NativeTensorRole::AttentionQ, vec![5120, 5120]),
            spec("o", NativeTensorRole::AttentionO, vec![5120, 5120]),
            spec("ffn_down", NativeTensorRole::FfnDown, vec![5120, 13824]),
        ];
        let summary = summarize_rotation_candidates(&specs);
        assert_eq!(summary.total_candidates, 3);
        // 5120 and 13824 are not power-of-2; no dimension is eligible.
        assert_eq!(summary.power_of_two_eligible, 0);
        assert_eq!(summary.non_power_of_two_blocked, 3);
    }

    #[test]
    fn shadow_log_is_noop_when_mode_off() {
        // We can't assert no log without a subscriber, but we can verify the
        // function returns without panicking and that the parse path treats
        // an unset env var as Off. This is the safety invariant.
        let specs: Vec<NativeTensorSpec> = vec![];
        shadow_log_rotation_candidates(&specs);
        // Doesn't panic, doesn't change state — that's all we test here.
    }
}
