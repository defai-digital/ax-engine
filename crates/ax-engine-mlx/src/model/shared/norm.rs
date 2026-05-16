use mlx_sys::{MlxArray, reshape, rms_norm};
use std::sync::OnceLock;

pub(crate) fn rms_norm_no_scale_bshd(
    x: MlxArray,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> MlxArray {
    if use_flat_qk_norm_path() {
        let batch = x.shape()[0] as usize;
        let flat = reshape(&x, &[(batch * n_heads * seq) as i32, head_dim as i32], None);
        let normed = rms_norm(&flat, None, eps, None);
        return reshape(
            &normed,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
    }
    rms_norm(&x, None, eps, None)
}

pub(crate) fn use_flat_qk_norm_path() -> bool {
    static USE_FLAT: OnceLock<bool> = OnceLock::new();
    *USE_FLAT.get_or_init(|| std::env::var("AX_MLX_QK_NORM_FLAT").as_deref() == Ok("1"))
}

pub(crate) fn rms_norm_opt(x: &MlxArray, norm: Option<&MlxArray>, eps: f32) -> MlxArray {
    if let Some(n) = norm {
        rms_norm(x, Some(n), eps, None)
    } else {
        x.clone()
    }
}
