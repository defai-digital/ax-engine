//! Duplicate row planning for batched n-gram staging.
//!
//! Reads each unique row once into its first-occurrence slot, then replicates
//! bytes to duplicate slots. Full caller-ordered planes feed the same single
//! dequantize as before, so arithmetic is unchanged while OS reads drop.
//! No reference code is reused here.

use std::collections::HashMap;

/// First-occurrence plan for one gather.
#[derive(Clone, Debug)]
pub(crate) struct DedupPlan {
    /// Unique rows in first-occurrence order.
    pub unique_rows: Vec<u64>,
    /// Caller position of each unique row's first occurrence.
    pub representative: Vec<usize>,
    /// Caller position to unique index.
    pub index_map: Vec<usize>,
}

/// Plan unique reads for `row_ids`. Fails closed on allocation failure
/// instead of aborting.
pub(crate) fn plan(row_ids: &[u64]) -> Result<DedupPlan, String> {
    let mut seen: HashMap<u64, usize> = HashMap::new();
    if seen.try_reserve(row_ids.len()).is_err() {
        return Err("dedup map reserve failed".to_string());
    }
    let mut unique_rows: Vec<u64> = Vec::new();
    if unique_rows.try_reserve(row_ids.len()).is_err() {
        return Err("dedup rows reserve failed".to_string());
    }
    let mut representative: Vec<usize> = Vec::new();
    if representative.try_reserve(row_ids.len()).is_err() {
        return Err("dedup representative reserve failed".to_string());
    }
    let mut index_map: Vec<usize> = Vec::new();
    if index_map.try_reserve(row_ids.len()).is_err() {
        return Err("dedup index reserve failed".to_string());
    }
    for (pos, row) in row_ids.iter().enumerate() {
        if let Some(found) = seen.get(row) {
            index_map.push(*found);
        } else {
            let uniq = unique_rows.len();
            seen.insert(*row, uniq);
            unique_rows.push(*row);
            representative.push(pos);
            index_map.push(uniq);
        }
    }
    Ok(DedupPlan {
        unique_rows,
        representative,
        index_map,
    })
}

/// Replicate representative slots to duplicate positions for every plane.
/// `staging` holds full caller-ordered planes; `row_bytes` holds each plane's
/// row length in the same order. Fails closed on any layout mismatch.
pub(crate) fn expand(
    staging: &mut [Vec<u8>],
    plan: &DedupPlan,
    row_bytes: &[usize],
) -> Result<(), String> {
    if staging.len() != row_bytes.len() {
        return Err("dedup plane count mismatch".to_string());
    }
    let count = plan.index_map.len();
    for (plane, row_len) in staging.iter_mut().zip(row_bytes.iter()) {
        let row_len = *row_len;
        let expect = count
            .checked_mul(row_len)
            .ok_or_else(|| "dedup plane bytes overflow".to_string())?;
        if plane.len() != expect {
            return Err("dedup plane length mismatch".to_string());
        }
        if row_len == 0 {
            continue;
        }
        for (pos, uniq) in plan.index_map.iter().enumerate() {
            let rep = match plan.representative.get(*uniq) {
                Some(rep) => *rep,
                None => return Err("dedup representative missing".to_string()),
            };
            if rep == pos {
                continue;
            }
            let src_start = match rep.checked_mul(row_len) {
                Some(start) => start,
                None => return Err("dedup source offset overflow".to_string()),
            };
            let src_end = match src_start.checked_add(row_len) {
                Some(end) => end,
                None => return Err("dedup source end overflow".to_string()),
            };
            let dst_start = match pos.checked_mul(row_len) {
                Some(start) => start,
                None => return Err("dedup destination overflow".to_string()),
            };
            let dst_end = match dst_start.checked_add(row_len) {
                Some(end) => end,
                None => return Err("dedup destination end overflow".to_string()),
            };
            if src_end > plane.len() || dst_end > plane.len() {
                return Err("dedup copy out of bounds".to_string());
            }
            plane.copy_within(src_start..src_end, dst_start);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn plan_keeps_first_occurrence_order_and_maps_duplicates() {
        let plan = super::plan(&[3, 0, 3, 1, 2]).unwrap();
        assert_eq!(plan.unique_rows, vec![3, 0, 1, 2]);
        assert_eq!(plan.representative, vec![0, 1, 3, 4]);
        assert_eq!(plan.index_map, vec![0, 1, 0, 2, 3]);
        let single = super::plan(&[7, 7, 7]).unwrap();
        assert_eq!(single.unique_rows, vec![7]);
        assert_eq!(single.representative, vec![0]);
        assert_eq!(single.index_map, vec![0, 0, 0]);
        let empty = super::plan(&[]).unwrap();
        assert!(empty.unique_rows.is_empty());
        assert!(empty.index_map.is_empty());
    }

    #[test]
    fn expand_replicates_representative_bytes_in_order() {
        let plan = super::plan(&[3, 0, 3, 1, 2]).unwrap();
        let mut staging: Vec<Vec<u8>> = vec![vec![0u8; 5 * 2], vec![0u8; 5]];
        staging[0][0..2].copy_from_slice(&[10, 11]);
        staging[0][2..4].copy_from_slice(&[20, 21]);
        staging[0][6..8].copy_from_slice(&[30, 31]);
        staging[0][8..10].copy_from_slice(&[40, 41]);
        staging[1][0] = 1;
        staging[1][1] = 2;
        staging[1][3] = 3;
        staging[1][4] = 4;
        expand(&mut staging, &plan, &[2, 1]).unwrap();
        assert_eq!(staging[0], vec![10, 11, 20, 21, 10, 11, 30, 31, 40, 41]);
        assert_eq!(staging[1], vec![1, 2, 1, 3, 4]);
    }
}
