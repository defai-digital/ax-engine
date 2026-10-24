// SPDX-License-Identifier: Apache-2.0

//! Small DFlash2 draft tree (first correctness increment).
//!
//! The drafter's linear chain is the primary branch. Up to two draft positions
//! may each carry one extra top-2 token, for at most four nodes. Each
//! root-to-leaf branch is verified on its own clone of the caller-supplied
//! state (KV plus linear/GDN and QSA/PLE state, whatever the clone contains).
//! The longest accepted prefix wins; a tie keeps the primary branch. Only
//! tokens the target's greedy decision accepted are emitted, then cut at EOS
//! and the output budget.
//!
//! `AX_MLX_TREE_VERIFY` is off unless set to a truthy value. Unset or `0`
//! leaves the linear DFlash2 verify path in place.

/// Most nodes a tree in this increment may hold.
pub(crate) const MAX_TREE_NODES: usize = 4;
/// Draft positions that may branch. Position 0 is the first draft token.
const MAX_BRANCH_POSITIONS: usize = 2;

/// One speculative draft token. `position` is `0` at a root and `parent + 1`
/// everywhere else.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeNode {
    pub parent: Option<usize>,
    pub token: u32,
    pub position: u32,
}

/// Why [`DraftTree::try_from_nodes`] refused a tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeInvalid {
    Empty,
    TooManyNodes,
    Cycle,
    Position,
    DanglingParent,
    Primary,
}

/// A validated draft tree. The primary leaf is the drafter's linear chain.
#[derive(Clone, Debug)]
pub(crate) struct DraftTree {
    nodes: Vec<TreeNode>,
    primary_leaf: usize,
}

#[derive(Clone, Debug)]
struct Branch {
    is_primary: bool,
    tokens: Vec<u32>,
    node_indices: Vec<usize>,
}

impl DraftTree {
    /// `primary` is the drafter's linear chain. Only the first two tokens are
    /// expanded. `alternatives[i]` is the other top-2 token at that position
    /// when it differs from the primary token.
    pub(crate) fn from_top2(
        primary: &[u32],
        alternatives: &[Option<u32>],
    ) -> Result<Self, TreeInvalid> {
        let depth = primary.len().min(MAX_BRANCH_POSITIONS);
        if depth == 0 {
            return Err(TreeInvalid::Empty);
        }
        let mut nodes = Vec::with_capacity(MAX_TREE_NODES);
        let mut parent = None;
        for (position, token) in primary.iter().take(depth).enumerate() {
            nodes.push(TreeNode {
                parent,
                token: *token,
                position: position as u32,
            });
            parent = Some(nodes.len() - 1);
        }
        let primary_leaf = depth - 1;
        if let Some(Some(token)) = alternatives.first()
            && *token != primary[0]
        {
            nodes.push(TreeNode {
                parent: None,
                token: *token,
                position: 0,
            });
        }
        if depth >= 2
            && let Some(Some(token)) = alternatives.get(1)
            && *token != primary[1]
        {
            nodes.push(TreeNode {
                parent: Some(0),
                token: *token,
                position: 1,
            });
        }
        Self::try_from_nodes(nodes, primary_leaf)
    }

    pub(crate) fn try_from_nodes(
        nodes: Vec<TreeNode>,
        primary_leaf: usize,
    ) -> Result<Self, TreeInvalid> {
        validate(&nodes, primary_leaf)?;
        Ok(Self {
            nodes,
            primary_leaf,
        })
    }

    fn branches(&self) -> Vec<Branch> {
        let mut is_parent = vec![false; self.nodes.len()];
        for node in &self.nodes {
            if let Some(parent) = node.parent {
                is_parent[parent] = true;
            }
        }
        let mut leaves: Vec<usize> = (0..self.nodes.len())
            .filter(|index| !is_parent[*index])
            .collect();
        if let Some(position) = leaves.iter().position(|leaf| *leaf == self.primary_leaf) {
            let primary = leaves.remove(position);
            leaves.insert(0, primary);
        }
        leaves
            .into_iter()
            .map(|leaf| {
                let (tokens, node_indices) = self.path_from_leaf(leaf);
                Branch {
                    is_primary: leaf == self.primary_leaf,
                    tokens,
                    node_indices,
                }
            })
            .collect()
    }

    fn path_from_leaf(&self, leaf: usize) -> (Vec<u32>, Vec<usize>) {
        let mut tokens = Vec::new();
        let mut indices = Vec::new();
        let mut cursor = Some(leaf);
        let mut guard = 0;
        while let Some(index) = cursor {
            guard += 1;
            if guard > MAX_TREE_NODES {
                break;
            }
            tokens.push(self.nodes[index].token);
            indices.push(index);
            cursor = self.nodes[index].parent;
        }
        tokens.reverse();
        indices.reverse();
        (tokens, indices)
    }
}

fn validate(nodes: &[TreeNode], primary_leaf: usize) -> Result<(), TreeInvalid> {
    if nodes.is_empty() {
        return Err(TreeInvalid::Empty);
    }
    if nodes.len() > MAX_TREE_NODES {
        return Err(TreeInvalid::TooManyNodes);
    }
    for (index, node) in nodes.iter().enumerate() {
        if let Some(parent) = node.parent {
            if parent == index {
                return Err(TreeInvalid::Cycle);
            }
            if parent >= nodes.len() {
                return Err(TreeInvalid::DanglingParent);
            }
        }
    }
    if has_cycle(nodes) {
        return Err(TreeInvalid::Cycle);
    }
    for node in nodes {
        match node.parent {
            None => {
                if node.position != 0 {
                    return Err(TreeInvalid::Position);
                }
            }
            Some(parent) => {
                let expected = nodes[parent].position.checked_add(1);
                if expected != Some(node.position) {
                    return Err(TreeInvalid::Position);
                }
            }
        }
    }
    if primary_leaf >= nodes.len() || !is_leaf(nodes, primary_leaf) {
        return Err(TreeInvalid::Primary);
    }
    Ok(())
}

fn has_cycle(nodes: &[TreeNode]) -> bool {
    for start in 0..nodes.len() {
        let mut seen = vec![false; nodes.len()];
        let mut cursor = Some(start);
        while let Some(index) = cursor {
            if seen[index] {
                return true;
            }
            seen[index] = true;
            cursor = nodes[index].parent;
        }
    }
    false
}

fn is_leaf(nodes: &[TreeNode], index: usize) -> bool {
    !nodes.iter().any(|node| node.parent == Some(index))
}

/// Output budget and terminal tokens applied after the target accepts a branch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CommitLimits<'a> {
    pub budget: usize,
    pub eos: &'a [u32],
}

/// Greedy result of verifying one branch on an isolated state clone.
#[derive(Clone, Debug)]
pub(crate) struct BranchVerification<S> {
    /// Draft tokens that matched the target, not counting the bonus token.
    pub accepted_drafts: usize,
    /// Accepted drafts followed by the target's own next token.
    pub committed: Vec<u32>,
    pub state: S,
    /// False when the forward did not produce a usable greedy decision.
    pub usable: bool,
}

/// The branch that was committed, after EOS and budget cuts.
#[derive(Clone, Debug)]
pub(crate) struct TreeCommit<S> {
    pub tokens: Vec<u32>,
    /// Model-accepted draft count of the chosen branch, before the budget cut.
    pub accepted_drafts: usize,
    /// `0` is the drafter's primary branch.
    pub branch_index: usize,
    /// Draft tokens on the chosen branch, before EOS and budget cuts.
    pub draft_len: usize,
    pub nodes_verified: usize,
    /// Draft rows of the chosen branch that stay in the cache (not the anchor,
    /// not the bonus token).
    pub cache_draft_rows: usize,
    pub state: S,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeVerifyError {
    Invalid(TreeInvalid),
    NoBranch,
}

/// How many committed tokens survive the budget and EOS, and how many of those
/// are draft rows the cache must retain.
pub(crate) fn limit_commit(
    tokens: &[u32],
    accepted_drafts: usize,
    budget: usize,
    eos: &[u32],
) -> (Vec<u32>, usize) {
    let mut emitted = Vec::new();
    for &token in tokens {
        if emitted.len() >= budget {
            break;
        }
        emitted.push(token);
        if eos.contains(&token) {
            break;
        }
    }
    let cache_draft_rows = emitted.len().min(accepted_drafts);
    (emitted, cache_draft_rows)
}

fn accepted_prefix_ok(drafts: &[u32], accepted: usize, committed: &[u32]) -> bool {
    accepted <= drafts.len()
        && committed.len() == accepted + 1
        && committed[..accepted] == drafts[..accepted]
}

struct Ranked<S> {
    branch_index: usize,
    is_primary: bool,
    accepted_drafts: usize,
    draft_len: usize,
    committed: Vec<u32>,
    state: S,
}

/// Verify every root-to-leaf branch on a fresh clone of `base`, then keep the
/// longest accepted prefix. Ties keep the primary branch.
///
/// `verify` receives that clone and the branch's draft tokens. It must score
/// the branch with the canonical greedy walk and must not observe another
/// branch's writes. Greedy only: the callback is the singleton linear verify.
pub(crate) fn verify_draft_tree<S: Clone>(
    tree: &DraftTree,
    base: &S,
    limits: CommitLimits<'_>,
    mut verify: impl FnMut(S, &[u32]) -> BranchVerification<S>,
) -> Result<TreeCommit<S>, TreeVerifyError> {
    validate(&tree.nodes, tree.primary_leaf).map_err(TreeVerifyError::Invalid)?;
    let branches = tree.branches();
    let mut seen = vec![false; tree.nodes.len()];
    let mut best: Option<Ranked<S>> = None;
    for (branch_index, branch) in branches.iter().enumerate() {
        let outcome = verify(base.clone(), &branch.tokens);
        if !outcome.usable {
            continue;
        }
        for index in &branch.node_indices {
            seen[*index] = true;
        }
        if !accepted_prefix_ok(&branch.tokens, outcome.accepted_drafts, &outcome.committed) {
            continue;
        }
        let replace = match &best {
            None => true,
            Some(current) => {
                outcome.accepted_drafts > current.accepted_drafts
                    || (outcome.accepted_drafts == current.accepted_drafts
                        && branch.is_primary
                        && !current.is_primary)
            }
        };
        if replace {
            best = Some(Ranked {
                branch_index,
                is_primary: branch.is_primary,
                accepted_drafts: outcome.accepted_drafts,
                draft_len: branch.tokens.len(),
                committed: outcome.committed,
                state: outcome.state,
            });
        }
    }
    let best = best.ok_or(TreeVerifyError::NoBranch)?;
    let (tokens, cache_draft_rows) = limit_commit(
        &best.committed,
        best.accepted_drafts,
        limits.budget,
        limits.eos,
    );
    Ok(TreeCommit {
        tokens,
        accepted_drafts: best.accepted_drafts,
        branch_index: best.branch_index,
        draft_len: best.draft_len,
        nodes_verified: seen.iter().filter(|node| **node).count(),
        cache_draft_rows,
        state: best.state,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::runner::dflash2_decode::greedy_walk;

    fn node(parent: Option<usize>, token: u32, position: u32) -> TreeNode {
        TreeNode {
            parent,
            token,
            position,
        }
    }

    fn unlimited() -> CommitLimits<'static> {
        CommitLimits {
            budget: usize::MAX,
            eos: &[],
        }
    }

    /// Target that follows `primary` and then emits `bonus`. A diverging prefix
    /// is corrected toward `primary` at the first position that still has one.
    fn linear_target(primary: &[u32], bonus: u32, drafts: &[u32]) -> Vec<u32> {
        (0..=drafts.len())
            .map(|index| {
                let fed = &drafts[..index];
                if fed == primary {
                    bonus
                } else if index < primary.len() {
                    primary[index]
                } else {
                    bonus
                }
            })
            .collect()
    }

    fn verify_linear<S: Clone>(
        state: S,
        drafts: &[u32],
        predicted: Vec<u32>,
    ) -> BranchVerification<S> {
        let (accepted, committed) = greedy_walk(drafts, &predicted);
        BranchVerification {
            accepted_drafts: accepted,
            committed,
            state,
            usable: true,
        }
    }

    #[test]
    fn tree_validation_rejects_bad_trees() {
        let cycle = DraftTree::try_from_nodes(vec![node(Some(1), 1, 1), node(Some(0), 2, 0)], 0);
        assert_eq!(cycle.err(), Some(TreeInvalid::Cycle));

        let skipped_position =
            DraftTree::try_from_nodes(vec![node(None, 1, 0), node(Some(0), 2, 2)], 1);
        assert_eq!(skipped_position.err(), Some(TreeInvalid::Position));

        let root_not_zero = DraftTree::try_from_nodes(vec![node(None, 1, 3)], 0);
        assert_eq!(root_not_zero.err(), Some(TreeInvalid::Position));

        let mut chain = Vec::new();
        for position in 0..5 {
            chain.push(node(
                if position == 0 {
                    None
                } else {
                    Some(position - 1)
                },
                position as u32,
                position as u32,
            ));
        }
        assert_eq!(
            DraftTree::try_from_nodes(chain, 4).err(),
            Some(TreeInvalid::TooManyNodes)
        );

        assert_eq!(
            DraftTree::try_from_nodes(vec![node(Some(3), 1, 1)], 0).err(),
            Some(TreeInvalid::DanglingParent)
        );
        assert_eq!(
            DraftTree::from_top2(&[], &[None, None]).err(),
            Some(TreeInvalid::Empty)
        );
    }

    #[test]
    fn tree_branch_state_isolated() {
        #[derive(Clone, Debug)]
        struct Stamp {
            generation: u64,
            label: u32,
        }
        let primary = [10, 20];
        let tree = DraftTree::from_top2(&primary, &[Some(11), Some(21)]).unwrap();
        let base = Stamp {
            generation: 1,
            label: 0,
        };
        let mut seen = Vec::new();
        let decision = verify_draft_tree(&tree, &base, unlimited(), |mut state, drafts| {
            seen.push((state.generation, state.label, drafts.to_vec()));
            state.generation = state.generation.saturating_add(drafts.len() as u64);
            state.label = drafts[0];
            verify_linear(state, drafts, linear_target(&primary, 30, drafts))
        })
        .unwrap();
        assert!(
            seen.len() >= 2,
            "expected the primary branch and an alternative"
        );
        assert!(
            seen.iter()
                .all(|(generation, label, _)| *generation == 1 && *label == 0),
            "a later branch observed an earlier branch's writes: {seen:?}"
        );
        assert_eq!(base.generation, 1);
        assert_eq!(base.label, 0);
        assert_eq!(decision.tokens, vec![10, 20, 30]);
    }

    #[test]
    fn tree_commit_matches_linear() {
        let primary = [10, 20];
        let tree = DraftTree::from_top2(&primary, &[Some(11), Some(21)]).unwrap();
        let predicted = linear_target(&primary, 30, &primary);
        let (linear_accepted, linear_tokens) = greedy_walk(&primary, &predicted);
        let decision = verify_draft_tree(&tree, &primary.to_vec(), unlimited(), |state, drafts| {
            verify_linear(state, drafts, linear_target(&primary, 30, drafts))
        })
        .unwrap();
        assert_eq!(decision.branch_index, 0);
        assert_eq!(decision.accepted_drafts, linear_accepted);
        assert_eq!(decision.tokens, linear_tokens);
        assert!(!decision.tokens.contains(&11));
        assert!(!decision.tokens.contains(&21));
        assert_eq!(decision.state, primary.to_vec());
    }

    #[test]
    fn tree_prefers_longest_accept() {
        let primary = [10, 20];
        let tree = DraftTree::from_top2(&primary, &[Some(11), Some(21)]).unwrap();
        let predict = |drafts: &[u32]| {
            if drafts == [10, 21] {
                vec![10, 21, 40]
            } else if drafts.first() == Some(&10) {
                // Primary mismatches at the second draft.
                let mut predicted = vec![10, 21];
                predicted.extend(std::iter::repeat_n(0, drafts.len().saturating_sub(1)));
                predicted.truncate(drafts.len() + 1);
                if predicted.len() < drafts.len() + 1 {
                    predicted.resize(drafts.len() + 1, 0);
                }
                predicted
            } else {
                let mut predicted = vec![10];
                predicted.resize(drafts.len() + 1, 0);
                predicted
            }
        };
        let decision = verify_draft_tree(&tree, &(), unlimited(), |state, drafts| {
            verify_linear(state, drafts, predict(drafts))
        })
        .unwrap();
        assert_ne!(decision.branch_index, 0);
        assert_eq!(decision.accepted_drafts, 2);
        assert_eq!(decision.tokens, vec![10, 21, 40]);
        assert!(!decision.tokens.contains(&20));

        // Equal acceptance keeps the primary branch.
        let tie = verify_draft_tree(&tree, &(), unlimited(), |state, drafts| {
            let mut predicted = vec![10, 99];
            predicted.resize(drafts.len() + 1, 0);
            verify_linear(state, drafts, predicted)
        })
        .unwrap();
        assert_eq!(tie.branch_index, 0);
        assert_eq!(tie.accepted_drafts, 1);
        assert_eq!(tie.tokens, vec![10, 99]);
    }

    #[test]
    fn tree_respects_eos_and_budget() {
        let primary = [10, 20];
        let tree = DraftTree::from_top2(&primary, &[Some(11), None]).unwrap();
        let run = |budget: usize, eos: &'static [u32]| {
            verify_draft_tree(&tree, &(), CommitLimits { budget, eos }, |state, drafts| {
                verify_linear(state, drafts, linear_target(&primary, 30, drafts))
            })
            .unwrap()
        };
        let at_eos = run(8, &[20]);
        assert_eq!(at_eos.tokens, vec![10, 20]);
        assert_eq!(at_eos.cache_draft_rows, 2);
        assert!(!at_eos.tokens.contains(&30));

        let bonus_eos = run(8, &[30]);
        assert_eq!(bonus_eos.tokens, vec![10, 20, 30]);
        assert_eq!(bonus_eos.cache_draft_rows, 2);

        let budget_one = run(1, &[]);
        assert_eq!(budget_one.tokens, vec![10]);
        assert_eq!(budget_one.cache_draft_rows, 1);
        assert!(!budget_one.tokens.contains(&20));

        let budget_two = run(2, &[]);
        assert_eq!(budget_two.tokens, vec![10, 20]);
        assert!(!budget_two.tokens.contains(&30));

        let budget_zero = run(0, &[]);
        assert!(budget_zero.tokens.is_empty());
        assert_eq!(budget_zero.cache_draft_rows, 0);

        // The rejected alternative never becomes an output token.
        assert!(!at_eos.tokens.contains(&11));
    }
}
