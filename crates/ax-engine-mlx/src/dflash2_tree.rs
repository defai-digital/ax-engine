// SPDX-License-Identifier: Apache-2.0

//! DFlash2 draft tree for one-forward tree verify.
//!
//! The drafter scores `K` candidates at each of its `P` positions, and its
//! selector scores every candidate against the token before it. A draft chain
//! spends all verify rows on one path; a tree spends them on the most likely
//! continuations, so a round whose first choice misses can still accept a
//! runner-up and what follows it.
//!
//! [`build_tree_layout`] grows the tree on the device, best first by the
//! product of each step's softmax share (the rule
//! `scripts/simulate_dflash2_tree.py` measures), and emits it in pre-order
//! with the best child first. The verify graph can therefore be built from
//! lazy arrays while the drafter still runs, with no host round trip between
//! draft and verify. [`reference_tree`] is the same rule on the host, and
//! [`accepted_path`] is the greedy walk over the verified rows.

use std::sync::OnceLock;

use mlx_sys::{KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, reshape};

/// Children one node may add to the frontier.
pub(crate) const TREE_MAX_CHILDREN: i32 = 4;
/// Convolution-tail entries per row: the gated-delta layers' kernel size
/// minus one.
pub(crate) const TREE_TAIL_LEN: i32 = 3;

/// Rows of the draft tree a host verifies when `AX_MLX_DFLASH2_TREE_ROWS` is
/// unset and a NAX kernel serves its verify projections. MacBook Pro M5 Max
/// 128 GB, Qwen 3.8 27B MXFP4, decode tok/s over 3 prompts x 64 tokens and
/// over 7 prompts x 256 tokens: chain 100.0 and 95.4, 8 rows 111.4 and 100.8,
/// 10 rows 98.4 on the second, 16 rows 108.4 and 102.8. Past eight rows the
/// verify costs about 3 ms more (the LM head and the attention leave their
/// eight-row kernels, the projections read a second row tile), which 16 rows
/// repay only on the longer outputs.
pub(crate) const DEFAULT_TREE_ROWS: i32 = 8;

/// Rows of the draft tree this process verifies, or `None` for the chain.
/// Call it after Metal is initialized: the host default probes the NAX
/// hand-off.
pub(crate) fn tree_rows() -> Option<i32> {
    static CACHED: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        tree_rows_for(
            crate::fastpath::dflash2_tree_rows_setting(),
            crate::model::shared::verify_qmm::nax_verify_handoff_active,
        )
    })
}

fn tree_rows_for(setting: Option<i32>, nax_handoff: impl FnOnce() -> bool) -> Option<i32> {
    match setting {
        Some(0) => None,
        Some(rows) => Some(rows),
        None => nax_handoff().then_some(DEFAULT_TREE_ROWS),
    }
}

/// One draft tree as lazy device arrays, rows in pre-order.
#[derive(Clone)]
pub(crate) struct TreeLayout {
    pub rows: i32,
    /// `[rows]` u32: row 0 is the anchor, the others draft tokens.
    pub ids: MlxArray,
    /// `[rows]` i32: 0 for the anchor.
    pub depth: MlxArray,
    /// `[rows]` i32: the parent row, `-1` for the anchor and for filler rows.
    pub parent: MlxArray,
    /// `[rows, TREE_TAIL_LEN]` i32: see
    /// `gated_delta_fused_verify_tree_from_qkv`.
    pub tail_src: MlxArray,
    /// `[rows, rows]` bool: row `i` may attend row `j` (itself or an
    /// ancestor).
    pub visible: MlxArray,
}

#[cfg(test)]
impl TreeLayout {
    /// The layout of a pre-order tree given on the host by `parents`.
    pub(crate) fn from_parents(ids: &[u32], parents: &[i32]) -> Self {
        let rows = ids.len();
        let mut paths: Vec<Vec<usize>> = Vec::with_capacity(rows);
        for (row, &parent) in parents.iter().enumerate() {
            let mut path = if parent < 0 {
                Vec::new()
            } else {
                paths[parent as usize].clone()
            };
            path.push(row);
            paths.push(path);
        }
        let tail_len = TREE_TAIL_LEN as usize;
        let mut depth = Vec::with_capacity(rows);
        let mut tails = Vec::with_capacity(rows * tail_len);
        let mut visible = vec![0u8; rows * rows];
        for (row, path) in paths.iter().enumerate() {
            let d = path.len() - 1;
            depth.push(d as i32);
            for slot in 0..tail_len {
                let back = tail_len - slot;
                tails.push(if d >= back {
                    path[d - back] as i32
                } else {
                    -((slot + d) as i32 + 1)
                });
            }
            for &other in path {
                visible[row * rows + other] = 1;
            }
        }
        let raw = |ptr: *const u8, bytes: usize, shape: &[i32], dtype: MlxDtype| {
            MlxArray::from_raw_data(ptr, bytes, shape, dtype)
        };
        let n = rows as i32;
        Self {
            rows: n,
            ids: raw(ids.as_ptr().cast(), rows * 4, &[n], MlxDtype::Uint32),
            depth: raw(depth.as_ptr().cast(), rows * 4, &[n], MlxDtype::Int32),
            parent: raw(parents.as_ptr().cast(), rows * 4, &[n], MlxDtype::Int32),
            tail_src: raw(
                tails.as_ptr().cast(),
                tails.len() * 4,
                &[n, TREE_TAIL_LEN],
                MlxDtype::Int32,
            ),
            visible: raw(visible.as_ptr(), rows * rows, &[n, n], MlxDtype::Bool),
        }
    }
}

static TREE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// Grows the best-first tree of `rows` rows from one draft's candidate table.
///
/// `candidates` and `unary` are `[1, P, K]` (token ids and drafter logits),
/// `edge0` is `[1, K]` (selector score of each first-position candidate after
/// the anchor) and `edges` is `[1, P - 1, K, K]` (`[p - 1][i][j]`: candidate
/// `j` at position `p` after candidate `i` at `p - 1`). `anchor` is `[1]`.
/// Rows the table cannot fill repeat the anchor with no parent; the walk
/// never reaches them.
pub(crate) fn build_tree_layout(
    candidates: &MlxArray,
    unary: &MlxArray,
    edge0: &MlxArray,
    edges: &MlxArray,
    anchor: &MlxArray,
    rows: i32,
) -> Option<TreeLayout> {
    let shape = candidates.shape();
    if shape.len() != 3 || shape[0] != 1 {
        return None;
    }
    let (positions, top_k) = (shape[1], shape[2]);
    if !(2..=crate::linear_attention_ops::GATED_DELTA_TREE_MAX_ROWS).contains(&rows)
        || !(2..crate::linear_attention_ops::GATED_DELTA_TREE_MAX_DEPTH).contains(&positions)
        || !(TREE_MAX_CHILDREN..=32).contains(&top_k)
        || candidates.dtype() != MlxDtype::Uint32
        || anchor.dtype() != MlxDtype::Uint32
        || unary.dtype() != MlxDtype::Float32
        || edge0.dtype() != MlxDtype::Float32
        || edges.dtype() != MlxDtype::Float32
        || unary.shape() != shape
        || edge0.shape() != vec![1, top_k]
        || edges.shape() != vec![1, positions - 1, top_k, top_k]
        || anchor.shape() != vec![1]
    {
        return None;
    }
    let kernel = TREE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_dflash2_draft_tree_v1",
            &["candidates", "unary", "edge0", "edges", "anchor"],
            &["ids", "node_depth", "parent", "tail_src", "visible"],
            TREE_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel
        .try_apply_with_template(
            &[
                &reshape(candidates, &[positions, top_k], None),
                &reshape(unary, &[positions, top_k], None),
                &reshape(edge0, &[top_k], None),
                &reshape(edges, &[positions - 1, top_k, top_k], None),
                anchor,
            ],
            &[
                KernelOutputSpec {
                    shape: vec![rows],
                    dtype: MlxDtype::Uint32,
                },
                KernelOutputSpec {
                    shape: vec![rows],
                    dtype: MlxDtype::Int32,
                },
                KernelOutputSpec {
                    shape: vec![rows],
                    dtype: MlxDtype::Int32,
                },
                KernelOutputSpec {
                    shape: vec![rows, TREE_TAIL_LEN],
                    dtype: MlxDtype::Int32,
                },
                KernelOutputSpec {
                    shape: vec![rows, rows],
                    dtype: MlxDtype::Bool,
                },
            ],
            &[
                KernelTemplateArg::Int {
                    name: "P",
                    value: positions,
                },
                KernelTemplateArg::Int {
                    name: "K",
                    value: top_k,
                },
                KernelTemplateArg::Int {
                    name: "N",
                    value: rows,
                },
                KernelTemplateArg::Int {
                    name: "MaxChildren",
                    value: TREE_MAX_CHILDREN,
                },
                KernelTemplateArg::Int {
                    name: "TailLen",
                    value: TREE_TAIL_LEN,
                },
            ],
            (1, 1, 1),
            (1, 1, 1),
            None,
        )
        .ok()?;
    if outputs.len() != 5 {
        return None;
    }
    let visible = outputs.pop()?;
    let tail_src = outputs.pop()?;
    let parent = outputs.pop()?;
    let depth = outputs.pop()?;
    let ids = outputs.pop()?;
    Some(TreeLayout {
        rows,
        ids,
        depth,
        parent,
        tail_src,
        visible,
    })
}

// One thread grows the tree: at most `N` pops from a frontier of at most
// `N * MaxChildren` entries over a `K`-wide table is a few thousand steps.
const TREE_KERNEL_SOURCE: &str = r#"
    if (thread_position_in_grid.x != 0) {
        return;
    }
    constexpr int Frontier = N * MaxChildren;
    int n_depth[N];
    int n_cand[N];
    int n_parent[N];
    float n_score[N];
    float f_score[Frontier];
    int f_depth[Frontier];
    int f_cand[Frontier];
    int f_parent[Frontier];
    bool f_used[Frontier];
    int f_count = 0;
    n_depth[0] = 0;
    n_cand[0] = -1;
    n_parent[0] = -1;
    n_score[0] = 1.0f;
    int count = 1;
    int node = 0;
    while (true) {
        const int d = n_depth[node];
        if (d < P) {
            float share[K];
            bool taken[K];
            float top = -INFINITY;
            for (int j = 0; j < K; ++j) {
                const float edge = d == 0
                    ? edge0[j]
                    : edges[((d - 1) * K + n_cand[node]) * K + j];
                share[j] = unary[d * K + j] + edge;
                top = max(top, share[j]);
                taken[j] = false;
            }
            float total = 0.0f;
            for (int j = 0; j < K; ++j) {
                share[j] = metal::precise::exp(share[j] - top);
                total += share[j];
            }
            for (int c = 0; c < MaxChildren; ++c) {
                int best = -1;
                for (int j = 0; j < K; ++j) {
                    if (!taken[j] && (best < 0 || share[j] > share[best])) {
                        best = j;
                    }
                }
                taken[best] = true;
                f_score[f_count] = n_score[node] * (share[best] / total);
                f_depth[f_count] = d + 1;
                f_cand[f_count] = best;
                f_parent[f_count] = node;
                f_used[f_count] = false;
                f_count += 1;
            }
        }
        if (count >= N) {
            break;
        }
        int best = -1;
        for (int f = 0; f < f_count; ++f) {
            if (!f_used[f] && (best < 0 || f_score[f] > f_score[best])) {
                best = f;
            }
        }
        if (best < 0) {
            break;
        }
        f_used[best] = true;
        n_depth[count] = f_depth[best];
        n_cand[count] = f_cand[best];
        n_parent[count] = f_parent[best];
        n_score[count] = f_score[best];
        node = count;
        count += 1;
    }

    // Pre-order, best child first: siblings were appended best first.
    int order[N];
    int place[N];
    int stack[N];
    int sp = 0;
    int placed = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const int at = stack[--sp];
        place[at] = placed;
        order[placed++] = at;
        for (int c = count - 1; c > at; --c) {
            if (n_parent[c] == at) {
                stack[sp++] = c;
            }
        }
    }

    for (int row = 0; row < N; ++row) {
        for (int other = 0; other < N; ++other) {
            visible[row * N + other] = other == row;
        }
        if (row >= placed) {
            ids[row] = anchor[0];
            node_depth[row] = 0;
            parent[row] = -1;
            for (int t = 0; t < TailLen; ++t) {
                tail_src[row * TailLen + t] = -(t + 1);
            }
            continue;
        }
        const int at = order[row];
        const int d = n_depth[at];
        ids[row] = at == 0 ? anchor[0] : candidates[(d - 1) * K + n_cand[at]];
        node_depth[row] = d;
        parent[row] = at == 0 ? -1 : place[n_parent[at]];
        for (int t = 0; t < TailLen; ++t) {
            tail_src[row * TailLen + t] = -(t + d + 1);
        }
        int up = at;
        for (int back = 1; back <= d; ++back) {
            up = n_parent[up];
            visible[row * N + place[up]] = true;
            if (back <= TailLen) {
                tail_src[row * TailLen + (TailLen - back)] = place[up];
            }
        }
    }
"#;

/// The rows of the verified path and the tokens it commits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeWalk {
    /// Rows from the anchor down the accepted path.
    pub path: Vec<usize>,
    /// The accepted draft tokens, then the target's token after the last row.
    pub tokens: Vec<u32>,
}

/// Greedy walk: from the anchor, follow the child whose token is the target's
/// choice at the current row (`predicted[row]`) until no child matches.
/// `None` when the arrays disagree in length or are empty.
pub(crate) fn accepted_path(ids: &[u32], parent: &[i32], predicted: &[u32]) -> Option<TreeWalk> {
    if ids.is_empty() || ids.len() != parent.len() || ids.len() != predicted.len() {
        return None;
    }
    let mut path = vec![0usize];
    let mut tokens = Vec::new();
    let mut row = 0usize;
    loop {
        let target = predicted[row];
        let next =
            (row + 1..ids.len()).find(|&child| parent[child] == row as i32 && ids[child] == target);
        tokens.push(target);
        match next {
            Some(child) => {
                path.push(child);
                row = child;
            }
            None => break,
        }
    }
    Some(TreeWalk { path, tokens })
}

/// Host form of the device tree: `(token, depth, parent)` per row, pre-order.
/// `unary[p][j]`, `edge0[j]` and `edges[p - 1][i][j]` are as in
/// [`build_tree_layout`].
#[cfg(test)]
pub(crate) fn reference_tree(
    anchor: u32,
    candidates: &[Vec<u32>],
    unary: &[Vec<f32>],
    edge0: &[f32],
    edges: &[Vec<Vec<f32>>],
    rows: usize,
) -> Vec<(u32, i32, i32)> {
    let positions = candidates.len();
    let top_k = edge0.len();
    // (depth, candidate, parent, score), best-first order.
    let mut nodes: Vec<(usize, usize, i32, f32)> = vec![(0, usize::MAX, -1, 1.0)];
    let mut frontier: Vec<(f32, usize, usize, usize, bool)> = Vec::new();
    let mut node = 0usize;
    loop {
        let (depth, candidate, _, score) = nodes[node];
        if depth < positions {
            let mut share: Vec<f32> = (0..top_k)
                .map(|j| {
                    unary[depth][j]
                        + if depth == 0 {
                            edge0[j]
                        } else {
                            edges[depth - 1][candidate][j]
                        }
                })
                .collect();
            let top = share.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for value in &mut share {
                *value = (*value - top).exp();
            }
            let total: f32 = share.iter().sum();
            let mut taken = vec![false; top_k];
            for _ in 0..TREE_MAX_CHILDREN {
                let mut best: Option<usize> = None;
                for j in 0..top_k {
                    if !taken[j] && best.is_none_or(|b| share[j] > share[b]) {
                        best = Some(j);
                    }
                }
                let Some(best) = best else { break };
                taken[best] = true;
                frontier.push((score * (share[best] / total), depth + 1, best, node, false));
            }
        }
        if nodes.len() >= rows {
            break;
        }
        let mut best: Option<usize> = None;
        for (index, entry) in frontier.iter().enumerate() {
            if !entry.4 && best.is_none_or(|b| entry.0 > frontier[b].0) {
                best = Some(index);
            }
        }
        let Some(best) = best else { break };
        frontier[best].4 = true;
        let (score, depth, candidate, parent, _) = frontier[best];
        nodes.push((depth, candidate, parent as i32, score));
        node = nodes.len() - 1;
    }
    let mut order = Vec::with_capacity(nodes.len());
    let mut stack = vec![0usize];
    while let Some(at) = stack.pop() {
        order.push(at);
        for child in (at + 1..nodes.len()).rev() {
            if nodes[child].2 == at as i32 {
                stack.push(child);
            }
        }
    }
    let mut place = vec![0usize; nodes.len()];
    for (row, &at) in order.iter().enumerate() {
        place[at] = row;
    }
    let mut out: Vec<(u32, i32, i32)> = order
        .iter()
        .map(|&at| {
            let (depth, candidate, parent, _) = nodes[at];
            if at == 0 {
                (anchor, 0, -1)
            } else {
                (
                    candidates[depth - 1][candidate],
                    depth as i32,
                    place[parent as usize] as i32,
                )
            }
        })
        .collect();
    out.resize(rows, (anchor, 0, -1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_rows_default_follows_the_nax_hand_off() {
        assert_eq!(tree_rows_for(None, || true), Some(DEFAULT_TREE_ROWS));
        assert_eq!(tree_rows_for(None, || false), None);
        assert_eq!(tree_rows_for(Some(0), || true), None);
        assert_eq!(tree_rows_for(Some(16), || false), Some(16));
    }

    fn pattern(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503))
                    .wrapping_add(0x9E37_79B9);
                lo + ((x >> 7) % 4096) as f32 / 4096.0 * (hi - lo)
            })
            .collect()
    }

    fn f32_array(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr().cast::<u8>(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn host_i32(array: &MlxArray) -> Vec<i32> {
        let wide = mlx_sys::astype(array, MlxDtype::Float32, None);
        mlx_sys::eval(&[&wide]);
        wide.data_f32().iter().map(|value| *value as i32).collect()
    }

    fn u32_array(data: &[u32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr().cast::<u8>(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Uint32,
        )
    }

    /// The device tree is the host reference's, and its tail and visibility
    /// tables are the ones the rows' root paths imply.
    #[test]
    fn device_tree_matches_host_reference() {
        const P: usize = 7;
        const K: usize = 16;
        for (seed, rows, spread) in [
            (1u64, 16usize, 6.0f32),
            (2, 16, 1.5),
            (3, 12, 12.0),
            (4, 8, 3.0),
        ] {
            let cand_data: Vec<u32> = (0..P * K)
                .map(|i| 1000 + (i as u32 * 37 + seed as u32) % 5000)
                .collect();
            let unary_data = pattern(P * K, seed, -spread, spread);
            let edge0_data = pattern(K, seed + 100, -1.0, 1.0);
            let edges_data = pattern((P - 1) * K * K, seed + 200, -2.0, 2.0);
            let anchor = 42u32;
            let layout = build_tree_layout(
                &u32_array(&cand_data, &[1, P as i32, K as i32]),
                &f32_array(&unary_data, &[1, P as i32, K as i32]),
                &f32_array(&edge0_data, &[1, K as i32]),
                &f32_array(&edges_data, &[1, P as i32 - 1, K as i32, K as i32]),
                &u32_array(&[anchor], &[1]),
                rows as i32,
            )
            .expect("tree kernel");
            mlx_sys::eval(&[
                &layout.ids,
                &layout.depth,
                &layout.parent,
                &layout.tail_src,
                &layout.visible,
            ]);
            let candidates: Vec<Vec<u32>> = cand_data.chunks(K).map(<[u32]>::to_vec).collect();
            let unary: Vec<Vec<f32>> = unary_data.chunks(K).map(<[f32]>::to_vec).collect();
            let edges: Vec<Vec<Vec<f32>>> = edges_data
                .chunks(K * K)
                .map(|block| block.chunks(K).map(<[f32]>::to_vec).collect())
                .collect();
            let expected = reference_tree(anchor, &candidates, &unary, &edge0_data, &edges, rows);
            let ids = layout.ids.data_u32().to_vec();
            let depth = host_i32(&layout.depth);
            let parent = host_i32(&layout.parent);
            let got: Vec<(u32, i32, i32)> = (0..rows)
                .map(|row| (ids[row], depth[row], parent[row]))
                .collect();
            assert_eq!(got, expected, "seed {seed}");
            // Pre-order with the best child first: a row's parent is the
            // nearest earlier row one level up, and the first child of the
            // anchor chain is always the next row.
            assert_eq!((depth[0], parent[0]), (0, -1));
            assert_eq!(parent[1], 0);
            let tails = host_i32(&layout.tail_src);
            let visible: Vec<bool> = host_i32(&layout.visible).iter().map(|v| *v != 0).collect();
            for row in 0..rows {
                let mut path = vec![row];
                while parent[*path.last().unwrap()] >= 0 {
                    path.push(parent[*path.last().unwrap()] as usize);
                }
                path.reverse();
                assert_eq!(path.len() as i32 - 1, depth[row], "seed {seed} row {row}");
                if row > 0 {
                    let nearest = (0..row).rev().find(|&r| depth[r] == depth[row] - 1);
                    assert_eq!(nearest, Some(parent[row] as usize), "seed {seed} row {row}");
                }
                let d = path.len() - 1;
                for slot in 0..TREE_TAIL_LEN as usize {
                    let back = TREE_TAIL_LEN as usize - slot;
                    let want = if d >= back {
                        path[d - back] as i32
                    } else {
                        -((slot + d) as i32 + 1)
                    };
                    assert_eq!(
                        tails[row * TREE_TAIL_LEN as usize + slot],
                        want,
                        "seed {seed} row {row}"
                    );
                }
                for other in 0..rows {
                    assert_eq!(
                        visible[row * rows + other],
                        path.contains(&other),
                        "seed {seed} row {row} other {other}"
                    );
                }
            }
        }
    }

    #[test]
    fn walk_follows_matching_children_and_adds_the_bonus_token() {
        // 0 -> 1 -> 2; 1 -> 3; 0 -> 4 -> 5.
        let ids = [9, 10, 11, 12, 20, 21];
        let parent = [-1, 0, 1, 1, 0, 4];
        // The target says 10 after the anchor, 12 after row 1, 77 after row 3.
        let walk = accepted_path(&ids, &parent, &[10, 12, 0, 77, 0, 0]).unwrap();
        assert_eq!(walk.path, vec![0, 1, 3]);
        assert_eq!(walk.tokens, vec![10, 12, 77]);
        // A runner-up at the first position.
        let walk = accepted_path(&ids, &parent, &[20, 0, 0, 0, 21, 5]).unwrap();
        assert_eq!(walk.path, vec![0, 4, 5]);
        assert_eq!(walk.tokens, vec![20, 21, 5]);
        // Nothing matches: only the target's own token.
        let walk = accepted_path(&ids, &parent, &[99, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(walk.path, vec![0]);
        assert_eq!(walk.tokens, vec![99]);
        // A filler row (no parent) is never entered.
        let walk = accepted_path(&[9, 10, 9], &[-1, 0, -1], &[9, 3, 3]).unwrap();
        assert_eq!(walk.path, vec![0]);
        assert!(accepted_path(&ids, &parent, &[1, 2]).is_none());
    }
}

#[cfg(test)]
mod rope_tests {
    use mlx_sys::{MlxArray, MlxDtype, astype, eval, reshape, rope, rope_dynamic, slice};

    /// MLX RoPE with one offset per batch row rotates row `i` to
    /// `offsets[i]`, bit for bit what a scalar-offset call per row gives.
    #[test]
    fn per_row_rope_offsets_match_scalar_offset_calls() {
        let (rows, heads, dim) = (6i32, 3i32, 64i32);
        let data: Vec<f32> = (0..rows * heads * dim)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let x = MlxArray::from_raw_data(
            data.as_ptr().cast::<u8>(),
            data.len() * 4,
            &[rows, heads, 1, dim],
            MlxDtype::Float32,
        );
        let positions: [i32; 6] = [1000, 1001, 1002, 1002, 1001, 1003];
        let offsets = MlxArray::from_raw_data(
            positions.as_ptr().cast::<u8>(),
            positions.len() * 4,
            &[rows],
            MlxDtype::Int32,
        );
        for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16] {
            let typed = astype(&x, dtype, None);
            let batched = astype(
                &rope_dynamic(
                    &typed,
                    dim,
                    false,
                    Some(10_000_000.0),
                    1.0,
                    &offsets,
                    None,
                    None,
                ),
                MlxDtype::Float32,
                None,
            );
            eval(&[&batched]);
            let got = batched.data_f32().to_vec();
            let width = (heads * dim) as usize;
            for (row, &position) in positions.iter().enumerate() {
                let one = reshape(
                    &slice(
                        &typed,
                        &[row as i32, 0, 0, 0],
                        &[row as i32 + 1, heads, 1, dim],
                        &[1, 1, 1, 1],
                        None,
                    ),
                    &[1, heads, 1, dim],
                    None,
                );
                let want = astype(
                    &rope(
                        &one,
                        dim,
                        false,
                        Some(10_000_000.0),
                        1.0,
                        position,
                        None,
                        None,
                    ),
                    MlxDtype::Float32,
                    None,
                );
                eval(&[&want]);
                let want = want.data_f32();
                let got_bits: Vec<u32> = got[row * width..(row + 1) * width]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
                assert_eq!(got_bits, want_bits, "{dtype:?} row {row}");
            }
        }
    }
}
