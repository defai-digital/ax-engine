use mlx_sys::{MlxArray, MlxDtype, add, arange, expand_dims, greater_equal, less, logical_and};

/// Build the boolean causal mask used by mlx-lm's `create_causal_mask`.
///
/// Shape is `[seq_len, offset + seq_len]`. `window_size` applies the same
/// sliding-window rule as mlx-lm: `linds >= rinds && linds < rinds + window`.
pub fn create_causal_mask(seq_len: usize, offset: usize, window_size: Option<usize>) -> MlxArray {
    let key_len = offset + seq_len;
    let rinds = arange(0.0, key_len as f64, 1.0, MlxDtype::Int32, None);
    let linds = if offset == 0 {
        rinds.clone()
    } else {
        arange(offset as f64, key_len as f64, 1.0, MlxDtype::Int32, None)
    };
    let linds = expand_dims(&linds, 1, None);
    let rinds = expand_dims(&rinds, 0, None);

    let causal = greater_equal(&linds, &rinds, None);
    if let Some(window_size) = window_size {
        let window = scalar_i32(window_size as i32);
        let upper = add(&rinds, &window, None);
        let sliding = less(&linds, &upper, None);
        logical_and(&causal, &sliding, None)
    } else {
        causal
    }
}

fn scalar_i32(value: i32) -> MlxArray {
    MlxArray::from_raw_data(
        &value as *const i32 as *const u8,
        std::mem::size_of_val(&value),
        &[],
        MlxDtype::Int32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::eval;

    fn mask_data(mask: &MlxArray) -> Vec<u8> {
        eval(&[mask]);
        let len = mask.nbytes();
        let ptr = mask.data_raw();
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    #[test]
    fn causal_mask_matches_mlx_lm_lower_triangle() {
        let mask = create_causal_mask(4, 0, None);

        assert_eq!(mask.shape(), vec![4, 4]);
        assert_eq!(
            mask_data(&mask),
            vec![1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
        );
    }

    #[test]
    fn causal_mask_applies_sliding_window_rule() {
        let mask = create_causal_mask(4, 0, Some(2));

        assert_eq!(mask.shape(), vec![4, 4]);
        assert_eq!(
            mask_data(&mask),
            vec![1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
        );
    }

    #[test]
    fn causal_mask_uses_cache_offset_for_key_length() {
        let mask = create_causal_mask(2, 3, None);

        assert_eq!(mask.shape(), vec![2, 5]);
        assert_eq!(mask_data(&mask), vec![1, 1, 1, 1, 0, 1, 1, 1, 1, 1]);
    }
}
