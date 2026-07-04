use std::collections::VecDeque;

use ax_engine_core::TurboQuantPreset;
use thiserror::Error;

use crate::model::ModelConfig;

pub type FullPrecisionKvTokenVectors = (Vec<f32>, Vec<f32>);

pub const TURBOQUANT_SLOT_ALIGNMENT_BYTES: usize = 16;
pub const TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM: usize = 128;
pub const TURBOQUANT_EXTENDED_FUSED_DECODE_HEAD_DIM: usize = 256;
pub const TURBOQUANT_GEMMA4_FULL_ATTENTION_FUSED_DECODE_HEAD_DIM: usize = 512;
pub const TURBOQUANT_ROUTE_METADATA_SCHEMA_VERSION: u32 = 3;
pub const TURBOQUANT_CODEC_VERSION_UNIFORM_HADAMARD: u32 = 1;
pub const TURBOQUANT_CODEC_VERSION_RHT_LLOYD_MAX: u32 = 2;

mod layer_support;
mod production;
mod quality;
#[cfg(test)]
mod tests;

mod codec;
mod open_tq_metal;

pub use codec::*;
pub use layer_support::*;
pub use open_tq_metal::*;
pub use production::*;
pub use quality::*;
