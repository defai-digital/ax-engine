//! Shared Qwen3.5 benchmark/profiler configuration types.
//!
//! These settings are bench-only knobs and audit records. Keeping them in one
//! module avoids duplicating arch-specific enums across the CLI entrypoint and
//! the profiling/reporting helpers.

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35RecurrentStateMode {
    #[default]
    Auto,
    CpuAlias,
    SlotBuffer,
    BackendOwned,
}

impl Qwen35RecurrentStateMode {
    pub fn as_env_value(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::CpuAlias => Some("cpu_alias"),
            Self::SlotBuffer => Some("slot_buffer"),
            Self::BackendOwned => Some("backend_owned"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CpuAlias => "cpu_alias",
            Self::SlotBuffer => "slot_buffer",
            Self::BackendOwned => "backend_owned",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35AlphaBetaStorageMode {
    #[default]
    Auto,
    F32,
    F16,
}

impl Qwen35AlphaBetaStorageMode {
    pub fn as_env_value(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::F32 => Some("f32"),
            Self::F16 => Some("f16"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Qwen35PrefillDTypeAudit {
    pub requested_recurrent_state_mode: Qwen35RecurrentStateMode,
    pub effective_recurrent_state_mode: String,
    pub requested_alpha_beta_storage_mode: Qwen35AlphaBetaStorageMode,
    pub effective_alpha_beta_storage_dtype: String,
    pub requested_slot_buffer_priming: bool,
    pub effective_slot_buffer_priming: bool,
    pub requested_same_kv_prewarm: bool,
    pub effective_same_kv_prewarm: bool,
    pub requested_force_backend_state_batch: bool,
    pub effective_force_backend_state_batch: bool,
    pub runtime_batch_prefill_prefers_f16_io: bool,
    pub dense_batch_projection_wrong_type_suspected: bool,
    pub recurrent_state_logical_dtype: String,
    pub recurrent_state_storage: String,
    pub recurrent_snapshot_dtype: String,
    pub recurrent_slot_mut_api_dtype: String,
    pub recurrent_batch_scratch_dtype: String,
    pub recurrent_handoff_alpha_beta_dtype: String,
    pub recurrent_handoff_observed_state_path: String,
    pub recurrent_handoff_observed_state_owner: String,
    pub recurrent_handoff_cpu_alias_layers: u64,
    pub recurrent_handoff_slot_buffer_layers: u64,
    pub recurrent_handoff_backend_carryover_layers: u64,
    pub recurrent_handoff_backend_zero_init_layers: u64,
    pub recurrent_handoff_cpu_materialization_layers: u64,
    pub recurrent_handoff_fused_tail_layers: u64,
    pub recurrent_state_batch_kind: String,
    pub recurrent_state_batch_backend_native_layers: u64,
    pub recurrent_state_batch_cpu_direct_layers: u64,
    pub recurrent_state_batch_cpu_direct_materialized_from_backend_layers: u64,
    pub recurrent_state_batch_cpu_gathered_layers: u64,
    pub recurrent_state_batch_cpu_gathered_materialized_from_backend_layers: u64,
    pub recurrent_f32_contract_ceiling_suspected: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
pub enum Qwen35SpecVerifyBranchArg {
    Auto,
    On,
    Off,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
pub enum Qwen35PrefillRecurrentStateModeArg {
    Auto,
    CpuAlias,
    SlotBuffer,
    BackendOwned,
}

impl Qwen35PrefillRecurrentStateModeArg {
    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CpuAlias => "cpu-alias",
            Self::SlotBuffer => "slot-buffer",
            Self::BackendOwned => "backend-owned",
        }
    }
}

impl From<Qwen35PrefillRecurrentStateModeArg> for Qwen35RecurrentStateMode {
    fn from(value: Qwen35PrefillRecurrentStateModeArg) -> Self {
        match value {
            Qwen35PrefillRecurrentStateModeArg::Auto => Qwen35RecurrentStateMode::Auto,
            Qwen35PrefillRecurrentStateModeArg::CpuAlias => Qwen35RecurrentStateMode::CpuAlias,
            Qwen35PrefillRecurrentStateModeArg::SlotBuffer => Qwen35RecurrentStateMode::SlotBuffer,
            Qwen35PrefillRecurrentStateModeArg::BackendOwned => {
                Qwen35RecurrentStateMode::BackendOwned
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
pub enum Qwen35PrefillAlphaBetaStorageModeArg {
    Auto,
    F32,
    F16,
}

impl Qwen35PrefillAlphaBetaStorageModeArg {
    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

impl From<Qwen35PrefillAlphaBetaStorageModeArg> for Qwen35AlphaBetaStorageMode {
    fn from(value: Qwen35PrefillAlphaBetaStorageModeArg) -> Self {
        match value {
            Qwen35PrefillAlphaBetaStorageModeArg::Auto => Qwen35AlphaBetaStorageMode::Auto,
            Qwen35PrefillAlphaBetaStorageModeArg::F32 => Qwen35AlphaBetaStorageMode::F32,
            Qwen35PrefillAlphaBetaStorageModeArg::F16 => Qwen35AlphaBetaStorageMode::F16,
        }
    }
}

pub fn qwen35_spec_verify_branch_env(value: Qwen35SpecVerifyBranchArg) -> Option<&'static str> {
    match value {
        Qwen35SpecVerifyBranchArg::Auto => None,
        Qwen35SpecVerifyBranchArg::On => Some("on"),
        Qwen35SpecVerifyBranchArg::Off => Some("off"),
    }
}

pub fn qwen35_prefill_recurrent_state_mode_env(
    value: Qwen35PrefillRecurrentStateModeArg,
) -> Option<&'static str> {
    match value {
        Qwen35PrefillRecurrentStateModeArg::Auto => None,
        Qwen35PrefillRecurrentStateModeArg::CpuAlias => Some("cpu_alias"),
        Qwen35PrefillRecurrentStateModeArg::SlotBuffer => Some("slot_buffer"),
        Qwen35PrefillRecurrentStateModeArg::BackendOwned => Some("backend_owned"),
    }
}

pub fn qwen35_prefill_alpha_beta_storage_mode_env(
    value: Qwen35PrefillAlphaBetaStorageModeArg,
) -> Option<&'static str> {
    match value {
        Qwen35PrefillAlphaBetaStorageModeArg::Auto => None,
        Qwen35PrefillAlphaBetaStorageModeArg::F32 => Some("f32"),
        Qwen35PrefillAlphaBetaStorageModeArg::F16 => Some("f16"),
    }
}

pub fn qwen35_prefill_force_backend_state_batch_env(enabled: bool) -> Option<&'static str> {
    enabled.then_some("1")
}
