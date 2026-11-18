//! Runtime quantization bound to native descriptors, not guessed from bit width.

use mlx_sys::MlxDtype;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExpertQuantizationMode {
    Affine,
    Mxfp4,
    Mxfp8,
}

impl ExpertQuantizationMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Mxfp4 => "mxfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }

    pub(super) fn packed_mx_bits(self) -> Option<u32> {
        match self {
            Self::Affine => None,
            Self::Mxfp4 => Some(4),
            Self::Mxfp8 => Some(8),
        }
    }

    pub(super) fn validate_mx_pair(
        self,
        weight_shape: &[i32],
        weight_dtype: MlxDtype,
        scale_shape: &[i32],
        scale_dtype: MlxDtype,
        experts: u32,
    ) -> Result<(), String> {
        let packed_per_group = self.packed_mx_bits().ok_or_else(|| {
            "MX expert weight/scale validation requires an explicit MX mode".to_string()
        })?;
        // The group width and the U32 word width are both 32.
        if weight_dtype != MlxDtype::Uint32
            || scale_dtype != MlxDtype::Uint8
            || weight_shape.len() != 3
            || scale_shape.len() != 3
            || weight_shape.iter().any(|&d| d <= 0)
            || scale_shape.iter().any(|&d| d <= 0)
            || weight_shape[..2] != scale_shape[..2]
            || weight_shape[0] as u32 != experts
            || u64::from(weight_shape[2] as u32)
                != u64::from(scale_shape[2] as u32) * u64::from(packed_per_group)
        {
            return Err(format!(
                "incompatible {} expert weight/scale pair: {weight_dtype:?} {weight_shape:?}, {scale_dtype:?} {scale_shape:?}",
                self.as_str().to_ascii_uppercase()
            ));
        }
        Ok(())
    }
}

pub(crate) type ExpertQuantizationModes = HashMap<String, ExpertQuantizationMode>;

pub(super) fn mode_for(
    modes: Option<&ExpertQuantizationModes>,
    name: &str,
) -> Result<ExpertQuantizationMode, String> {
    match modes {
        Some(modes) => modes
            .get(name)
            .copied()
            .ok_or_else(|| format!("expert quantization is not bound for {name}")),
        // Existing callers have an affine contract. Mixed-format callers must
        // bind every projection explicitly; their map cannot fall back here.
        None => Ok(ExpertQuantizationMode::Affine),
    }
}
