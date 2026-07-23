//! Shared pure geometry for VL towers (WS-V1 / WS-V2).
//!
//! These helpers are unit-tested without Apple Silicon models so convert,
//! server budget checks, and golden fixture generators share one contract.

/// Soft-token count for a ViT-style tower: `(h/p)*(w/p) / merge²` capped.
pub fn vit_soft_token_count(
    height: u32,
    width: u32,
    patch_size: u32,
    merge_size: u32,
    max_soft_tokens: u32,
) -> Option<u32> {
    if height == 0 || width == 0 || patch_size == 0 || merge_size == 0 || max_soft_tokens == 0 {
        return None;
    }
    let ph = height / patch_size;
    let pw = width / patch_size;
    if ph == 0 || pw == 0 {
        return None;
    }
    let merge_area = merge_size.saturating_mul(merge_size);
    if merge_area == 0 {
        return None;
    }
    let patches = ph.saturating_mul(pw);
    Some((patches / merge_area).clamp(1, max_soft_tokens))
}

/// Qwen3-VL MRoPE section lengths for a single image: `[T, H, W]` with T=1 for
/// pure image (no video). Used to build position id grids.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MropeSections {
    pub temporal: u32,
    pub height: u32,
    pub width: u32,
}

impl MropeSections {
    pub fn for_image(grid_h: u32, grid_w: u32) -> Self {
        Self {
            temporal: 1,
            height: grid_h.max(1),
            width: grid_w.max(1),
        }
    }

    pub fn total_positions(self) -> u32 {
        self.temporal
            .saturating_mul(self.height)
            .saturating_mul(self.width)
    }
}

/// Expand MRoPE sections into interleaved position ids of length `3 * n` for
/// the three axes (t,h,w) packed as `[t0,h0,w0, t1,h1,w1, ...]`.
pub fn mrope_position_ids(sections: MropeSections) -> Vec<u32> {
    let mut out = Vec::with_capacity(sections.total_positions() as usize * 3);
    for t in 0..sections.temporal {
        for h in 0..sections.height {
            for w in 0..sections.width {
                out.push(t);
                out.push(h);
                out.push(w);
            }
        }
    }
    out
}

/// LLaVA-style scatter indices: for each soft token, the absolute prompt
/// position it overwrites. `placeholder_positions` must be sorted ascending.
pub fn scatter_merge_indices(
    placeholder_positions: &[usize],
    soft_tokens_per_placeholder: &[u32],
) -> Result<Vec<usize>, String> {
    if placeholder_positions.len() != soft_tokens_per_placeholder.len() {
        return Err(format!(
            "placeholder count {} != soft-token groups {}",
            placeholder_positions.len(),
            soft_tokens_per_placeholder.len()
        ));
    }
    let mut out = Vec::new();
    let mut cursor_shift = 0usize;
    for (i, &base) in placeholder_positions.iter().enumerate() {
        let n = soft_tokens_per_placeholder[i] as usize;
        if n == 0 {
            return Err("soft_tokens_per_placeholder must be > 0".into());
        }
        let start = base.saturating_add(cursor_shift);
        for j in 0..n {
            out.push(start.saturating_add(j));
        }
        // Placeholder is one token expanded to n soft tokens → shift later
        // placeholders by n-1.
        cursor_shift = cursor_shift.saturating_add(n.saturating_sub(1));
    }
    Ok(out)
}

/// DeepStack layer injection indices for Qwen3-VL: feature layers are injected
/// at fixed language-layer depths. Default schedule mirrors common mlx-vlm
/// configs: inject after language layers 0, 1, 2 when three feature maps exist.
pub fn deepstack_injection_layers(num_feature_maps: usize, language_layers: u32) -> Vec<u32> {
    if num_feature_maps == 0 || language_layers == 0 {
        return Vec::new();
    }
    let n = num_feature_maps.min(language_layers as usize);
    (0..n as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vit_soft_tokens_basic() {
        // 224×224, patch 14 → 16×16=256 patches, merge 2 → 64 soft tokens
        assert_eq!(vit_soft_token_count(224, 224, 14, 2, 256), Some(64));
        assert_eq!(vit_soft_token_count(0, 224, 14, 2, 256), None);
        assert_eq!(vit_soft_token_count(224, 224, 14, 2, 32), Some(32)); // clamp
    }

    #[test]
    fn mrope_image_grid() {
        let s = MropeSections::for_image(2, 3);
        assert_eq!(s.total_positions(), 6);
        let ids = mrope_position_ids(s);
        assert_eq!(ids.len(), 18);
        // first (t,h,w)=(0,0,0)
        assert_eq!(&ids[0..3], &[0, 0, 0]);
        // last (0,1,2)
        assert_eq!(&ids[15..18], &[0, 1, 2]);
    }

    #[test]
    fn scatter_indices_expand_placeholders() {
        // two images at positions 5 and 10, each 2 soft tokens
        let idx = scatter_merge_indices(&[5, 10], &[2, 2]).unwrap();
        assert_eq!(idx, vec![5, 6, 11, 12]);
    }

    #[test]
    fn deepstack_schedule() {
        assert_eq!(deepstack_injection_layers(3, 28), vec![0, 1, 2]);
        assert!(deepstack_injection_layers(0, 28).is_empty());
    }
}
