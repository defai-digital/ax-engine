#!/usr/bin/env python3
"""Offline tree-verify simulator for DFlash2 candidate traces.

Reads `AX_MLX_DFLASH2_CANDIDATE_TRACE` JSONL captures and prices how many
tokens a DFlash2 round would commit if the target verified a tree of draft
alternatives instead of one chain, before any tree-verify kernel is built.

Trace lines (one JSON object each; `request` is optional and separates
requests whose lines interleave):

* `{"kind":"request_start"}` opens a request (or a new stretch of one after
  tokens were decoded on another route);
* `{"kind":"draft","anchor":t,"top_k":K,"positions":P,"candidates":[P][K],
  "unary":[P][K],"edge0":[K],"edges":[P-1][K][K],"chain":[P]}` is one draft:
  each position's top-k tokens, their drafter logits, the selector's edge score
  of each first-position candidate after the anchor, and of candidate `j` at
  position `p` after candidate `i` at position `p - 1` (`edges[p-1][i][j]`);
* `{"kind":"commit","tokens":[...],"accepted_drafts":n,"window":w}` is what
  the round emitted. The commits of a request, concatenated, are its output
  after the first token, which is the ground truth every policy is scored
  against (greedy decoding: the target's continuation of an anchor does not
  depend on what was drafted).

Policies (a round emits its accepted drafts plus the target's own token):

* `chain(w)`: the engine's chain cut to `w` verify rows (anchor + `w - 1`
  drafts). `chain(P + 1)` is the engine with a full window.
* `tree(rows, children, est)`: a best-first tree of `rows - 1` draft rows
  under the anchor. A node's children are the next position's candidates
  scored by `unary + edge` given the node's own candidate; `q` is their
  softmax; a child's path score is its parent's score times `est(q, depth)`;
  only a node's top `children` children by `q` are eligible. `est` is `q`
  itself or `calibrated`, a logistic recalibration of `q` per depth fitted on
  the trace and cross-fitted by request.
* `oracle`: accepted = leading positions whose true token is among that
  position's candidates, whatever the row budget (an upper bound).

Modes: `paired` scores every recorded round at its own anchor. `exact`
replays whole requests round by round and needs a draft at every output
position, i.e. a trace recorded with `AX_MLX_DFLASH2_TRACE_ACCEPT_CAP=0`; with
`--cost rows=ms` points it also projects tokens per second.

Deterministic: no randomness, stable ordering, stdlib only.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

NEG_INF = float("-inf")
DEFAULT_TREE_ROWS = (8, 12, 16)
DEFAULT_CHILDREN = (2, 3, 4)
ESTIMATORS = ("q", "calibrated")
MIN_CHAIN_ROWS = 4
# Calibration: logit(q) is clipped, binned, and fitted with a ridge pull
# towards the identity (slope 1, intercepts 0).
LOGIT_LIMIT = 14.0
BIN_WIDTH = 0.25
RIDGE = 1.0
MIN_SLOPE = 0.05

Estimate = Callable[[Sequence[float], int], Sequence[float]]


class TraceError(ValueError):
    """A trace that cannot be reconstructed or does not support the mode."""


@dataclass(frozen=True)
class Draft:
    anchor: int
    top_k: int
    positions: int
    candidates: tuple[tuple[int, ...], ...]
    unary: tuple[tuple[float, ...], ...]
    edge0: tuple[float, ...]
    edges: tuple[tuple[tuple[float, ...], ...], ...]
    chain: tuple[int, ...]


@dataclass
class Round:
    draft: Draft
    # Tokens the request had committed before this round.
    index: int
    window: int | None = None
    matched_drafts: int | None = None
    _children: dict[tuple[int, int], tuple[list[float], list[int]]] = field(
        default_factory=dict, repr=False
    )

    def children(self, depth: int, parent: int) -> tuple[list[float], list[int]]:
        """Softmax `q` of the candidates under a node and their order by `q`.

        `depth` is the node's depth (0 for the anchor) and `parent` its
        candidate index (ignored for the anchor).
        """
        key = (depth, parent if depth else 0)
        cached = self._children.get(key)
        if cached is None:
            q = softmax(child_logits(self.draft, depth, parent))
            order = sorted(range(len(q)), key=lambda j: (-q[j], j))
            cached = (q, order)
            self._children[key] = cached
        return cached


@dataclass
class Request:
    name: str
    tokens: list[int] = field(default_factory=list)
    rounds: list[Round] = field(default_factory=list)
    # Commits with no draft record (cohort rounds, a failed trace read).
    undrafted_commits: int = 0

    def truth(self, index: int) -> list[int]:
        return self.tokens[index:]

    def is_dense(self) -> bool:
        return [rnd.index for rnd in self.rounds] == list(range(len(self.tokens)))


def _score(value: object) -> float:
    # The runtime writes non-finite scores as null.
    return NEG_INF if value is None else float(value)  # type: ignore[arg-type]


def parse_draft(obj: dict, where: str) -> Draft:
    try:
        top_k = int(obj["top_k"])
        positions = int(obj["positions"])
        draft = Draft(
            anchor=int(obj["anchor"]),
            top_k=top_k,
            positions=positions,
            candidates=tuple(tuple(int(t) for t in row) for row in obj["candidates"]),
            unary=tuple(tuple(_score(v) for v in row) for row in obj["unary"]),
            edge0=tuple(_score(v) for v in obj["edge0"]),
            edges=tuple(
                tuple(tuple(_score(v) for v in row) for row in table) for table in obj["edges"]
            ),
            chain=tuple(int(t) for t in obj["chain"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TraceError(f"{where}: malformed draft record: {error!r}") from error
    well_formed = (
        top_k >= 1
        and positions >= 1
        and len(draft.candidates) == positions
        and len(draft.unary) == positions
        and all(len(row) == top_k for row in draft.candidates)
        and all(len(row) == top_k for row in draft.unary)
        and len(draft.edge0) == top_k
        and len(draft.edges) == positions - 1
        and all(len(table) == top_k for table in draft.edges)
        and all(len(row) == top_k for table in draft.edges for row in table)
        and len(draft.chain) == positions
    )
    if not well_formed:
        raise TraceError(f"{where}: draft record shapes disagree with top_k/positions")
    return draft


def load_traces(paths: Sequence[Path]) -> list[Request]:
    """Rebuild every request: its emitted tokens and its drafted rounds.

    A draft's anchor must be the last token its request committed (the first
    round of a request has nothing to check against).
    """
    requests: list[Request] = []
    for path in paths:
        requests.extend(load_trace(path))
    return [request for request in requests if request.tokens]


def load_trace(path: Path) -> list[Request]:
    requests: list[Request] = []
    # Per `request` id (None when the trace has none): the open request and
    # its draft that still waits for a commit.
    current: dict[object, Request] = {}
    pending: dict[object, Draft] = {}

    def open_request(key: object) -> Request:
        request = Request(name=f"{path.name}#{len(requests) + 1}")
        current[key] = request
        requests.append(request)
        return request

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            where = f"{path}:{line_no}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceError(f"{where}: invalid JSON: {error}") from error
            if not isinstance(obj, dict):
                raise TraceError(f"{where}: expected a JSON object")
            kind = obj.get("kind")
            key = obj.get("request")
            if kind == "request_start":
                if key in pending:
                    raise TraceError(f"{where}: request_start after a draft with no commit")
                open_request(key)
            elif kind == "draft":
                if key in pending:
                    raise TraceError(f"{where}: draft after a draft with no commit")
                request = current.get(key) or open_request(key)
                draft = parse_draft(obj, where)
                if request.tokens and draft.anchor != request.tokens[-1]:
                    raise TraceError(
                        f"{where}: draft anchor {draft.anchor} is not the last committed "
                        f"token {request.tokens[-1]}"
                    )
                pending[key] = draft
            elif kind == "commit":
                request = current.get(key) or open_request(key)
                try:
                    tokens = [int(t) for t in obj["tokens"]]
                    window = None if obj.get("window") is None else int(obj["window"])
                    matched = obj.get("matched_drafts")
                    matched = None if matched is None else int(matched)
                except (KeyError, TypeError, ValueError) as error:
                    raise TraceError(f"{where}: malformed commit record: {error!r}") from error
                drafted = pending.pop(key, None)
                if drafted is None:
                    request.undrafted_commits += 1
                else:
                    request.rounds.append(Round(drafted, len(request.tokens), window, matched))
                request.tokens.extend(tokens)
            else:
                raise TraceError(f"{where}: unknown record kind {kind!r}")
    return requests


def softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    if top == NEG_INF:
        return [1.0 / len(logits)] * len(logits)
    weights = [math.exp(value - top) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def child_logits(draft: Draft, depth: int, parent: int) -> list[float]:
    """Selector scores of the candidates at position `depth` under a node at
    that depth (the anchor for depth 0) holding candidate `parent`."""
    unary = draft.unary[depth]
    edges = draft.edge0 if depth == 0 else draft.edges[depth - 1][parent]
    return [u + e for u, e in zip(unary, edges, strict=True)]


def matching_prefix(drafts: Sequence[int], truth: Sequence[int]) -> int:
    accepted = 0
    for draft, target in zip(drafts, truth, strict=False):
        if draft != target:
            break
        accepted += 1
    return accepted


def chain_accepted(draft: Draft, rows: int, truth: Sequence[int]) -> int:
    return matching_prefix(draft.chain[: max(rows - 1, 0)], truth)


def oracle_accepted(draft: Draft, truth: Sequence[int]) -> int:
    accepted = 0
    for candidates, target in zip(draft.candidates, truth, strict=False):
        if target not in candidates:
            break
        accepted += 1
    return accepted


def identity_estimate(q: Sequence[float], depth: int) -> Sequence[float]:
    del depth
    return q


def logit(prob: float) -> float:
    if prob <= 0.0:
        return -LOGIT_LIMIT
    if prob >= 1.0:
        return LOGIT_LIMIT
    return max(-LOGIT_LIMIT, min(LOGIT_LIMIT, math.log(prob / (1.0 - prob))))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    expo = math.exp(value)
    return expo / (1.0 + expo)


@dataclass(frozen=True)
class Calibration:
    """P(candidate is the target's token | its parent was) by depth (1-based):
    `sigmoid(slope * logit(q) + intercepts[depth - 1])`."""

    slope: float = 1.0
    intercepts: tuple[float, ...] = ()

    def intercept(self, depth: int) -> float:
        if not self.intercepts:
            return 0.0
        return self.intercepts[min(max(depth, 1), len(self.intercepts)) - 1]

    def probability(self, q: float, depth: int) -> float:
        return sigmoid(self.slope * logit(q) + self.intercept(depth))

    def estimate(self, q: Sequence[float], depth: int) -> Sequence[float]:
        """Calibrated estimates of all siblings. At most one sibling is the
        target's token, so estimates that sum past 1 are scaled down."""
        estimates = [self.probability(value, depth) for value in q]
        total = sum(estimates)
        if total > 1.0:
            estimates = [value / total for value in estimates]
        return estimates


# (depth, logit bin) -> [samples, sum of logit(q), samples that were the target's token]
Samples = dict[tuple[int, int], list[float]]


def calibration_samples(request: Request) -> Samples:
    """Every candidate under a node on the true path, labelled by whether it
    is the target's next token."""
    samples: Samples = {}
    for rnd in request.rounds:
        truth = request.truth(rnd.index)
        parent = 0
        for depth in range(min(rnd.draft.positions, len(truth))):
            q, _ = rnd.children(depth, parent)
            hit = None
            for j, token in enumerate(rnd.draft.candidates[depth]):
                x = logit(q[j])
                cell = samples.setdefault((depth + 1, math.floor(x / BIN_WIDTH)), [0.0, 0.0, 0.0])
                cell[0] += 1.0
                cell[1] += x
                if token == truth[depth]:
                    cell[2] += 1.0
                    hit = j
            if hit is None:
                break
            parent = hit
    return samples


def merge_samples(parts: Sequence[Samples], sign: float = 1.0, into: Samples | None = None):
    merged: Samples = {} if into is None else {key: list(cell) for key, cell in into.items()}
    for part in parts:
        for key, cell in part.items():
            target = merged.setdefault(key, [0.0, 0.0, 0.0])
            for slot in range(3):
                target[slot] += sign * cell[slot]
    return {key: cell for key, cell in merged.items() if cell[0] > 0.5}


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; `None` when singular."""
    size = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(column + 1, size):
            factor = rows[r][column] / rows[column][column]
            if factor:
                for c in range(column, size + 1):
                    rows[r][c] -= factor * rows[column][c]
    solution = [0.0] * size
    for r in range(size - 1, -1, -1):
        tail = sum(rows[r][c] * solution[c] for c in range(r + 1, size))
        solution[r] = (rows[r][size] - tail) / rows[r][r]
    return solution


def _log_sigmoid(value: float) -> float:
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def fit_calibration(samples: Samples, positions: int, iterations: int = 50) -> Calibration:
    """Ridge logistic regression of the label on `logit(q)` with one intercept
    per depth, by damped Newton steps. No samples leaves the identity."""
    points = [
        (depth, cell[1] / cell[0], cell[0], cell[2])
        for (depth, _), cell in sorted(samples.items())
        if 1 <= depth <= positions
    ]
    theta = [1.0] + [0.0] * positions
    if not points:
        return Calibration(slope=1.0, intercepts=tuple(theta[1:]))

    def loss(params: Sequence[float]) -> float:
        total = 0.5 * RIDGE * ((params[0] - 1.0) ** 2 + sum(b * b for b in params[1:]))
        for depth, x, count, positives in points:
            z = params[0] * x + params[depth]
            total -= positives * _log_sigmoid(z) + (count - positives) * _log_sigmoid(-z)
        return total

    current = loss(theta)
    size = positions + 1
    for _ in range(iterations):
        gradient = [RIDGE * (theta[0] - 1.0)] + [RIDGE * b for b in theta[1:]]
        hessian = [[RIDGE if r == c else 0.0 for c in range(size)] for r in range(size)]
        for depth, x, count, positives in points:
            p = sigmoid(theta[0] * x + theta[depth])
            residual = count * p - positives
            weight = count * p * (1.0 - p)
            gradient[0] += residual * x
            gradient[depth] += residual
            hessian[0][0] += weight * x * x
            hessian[0][depth] += weight * x
            hessian[depth][0] += weight * x
            hessian[depth][depth] += weight
        step = solve_linear(hessian, gradient)
        if step is None:
            break
        scale = 1.0
        improved = False
        while scale > 1e-4:
            trial = [value - scale * delta for value, delta in zip(theta, step, strict=True)]
            trial_loss = loss(trial)
            if trial_loss <= current:
                improved = current - trial_loss > 1e-10
                theta, current = trial, trial_loss
                break
            scale *= 0.5
        if not improved:
            break
    # A positive slope keeps the estimate increasing in q.
    return Calibration(slope=max(theta[0], MIN_SLOPE), intercepts=tuple(theta[1:]))


def cross_fit(requests: Sequence[Request], positions: int) -> tuple[list[Calibration], Calibration]:
    """One calibration per request, fitted on the other requests, and the
    pooled fit. A single request can only be scored by its own fit."""
    parts = [calibration_samples(request) for request in requests]
    total = merge_samples(parts)
    pooled = fit_calibration(total, positions)
    if len(requests) < 2:
        return [pooled] * len(requests), pooled
    fits = [
        fit_calibration(merge_samples([part], sign=-1.0, into=total), positions) for part in parts
    ]
    return fits, pooled


# A placed tree node: (depth, candidate index, parent node index, path score).
Node = tuple[int, int, int, float]


def grow_tree(rnd: Round, rows: int, children: int, estimate: Estimate) -> list[Node]:
    """Best-first tree of at most `rows` rows. Node 0 is the anchor; every
    other node is one draft row, placed after its parent."""
    nodes: list[Node] = [(0, -1, -1, 1.0)]
    heap: list[tuple[float, int, int, int, int]] = []
    pushed = 0

    def expand(index: int) -> None:
        nonlocal pushed
        depth, candidate, _, score = nodes[index]
        if depth >= rnd.draft.positions:
            return
        q, order = rnd.children(depth, candidate)
        estimates = estimate(q, depth + 1)
        for j in order[: max(children, 0)]:
            heapq.heappush(heap, (-score * estimates[j], pushed, depth + 1, j, index))
            pushed += 1

    expand(0)
    while heap and len(nodes) < rows:
        negative, _, depth, candidate, parent = heapq.heappop(heap)
        nodes.append((depth, candidate, parent, -negative))
        expand(len(nodes) - 1)
    return nodes


def tree_accepted(rnd: Round, nodes: Sequence[Node], truth: Sequence[int]) -> int:
    """Depth of the deepest node whose whole path is the target's tokens."""
    by_parent: dict[int, dict[int, int]] = {}
    for index, (depth, candidate, parent, _) in enumerate(nodes):
        if index:
            token = rnd.draft.candidates[depth - 1][candidate]
            by_parent.setdefault(parent, {})[token] = index
    node = 0
    accepted = 0
    while accepted < len(truth):
        following = by_parent.get(node, {}).get(truth[accepted])
        if following is None:
            break
        node = following
        accepted += 1
    return accepted


@dataclass(frozen=True)
class Policy:
    kind: str
    rows: int
    children: int = 0
    est: str = ""

    @property
    def name(self) -> str:
        if self.kind == "chain":
            return f"chain({self.rows})"
        if self.kind == "tree":
            est = "cal" if self.est == "calibrated" else self.est
            return f"tree({self.rows},c{self.children},{est})"
        return "oracle"


def policy_accepted(
    policy: Policy, rnd: Round, truth: Sequence[int], calibration: Calibration
) -> int:
    if policy.kind == "chain":
        return chain_accepted(rnd.draft, policy.rows, truth)
    if policy.kind == "oracle":
        return oracle_accepted(rnd.draft, truth)
    estimate = calibration.estimate if policy.est == "calibrated" else identity_estimate
    return tree_accepted(rnd, grow_tree(rnd, policy.rows, policy.children, estimate), truth)


def policy_emitted(
    policy: Policy, rnd: Round, truth: Sequence[int], calibration: Calibration
) -> int:
    """Tokens the round emits: accepted drafts plus the target's own token,
    never more than the request still had to emit."""
    return min(policy_accepted(policy, rnd, truth, calibration) + 1, len(truth))


def default_grid(
    positions: int,
    tree_rows: Sequence[int] = DEFAULT_TREE_ROWS,
    children: Sequence[int] = DEFAULT_CHILDREN,
    estimators: Sequence[str] = ESTIMATORS,
) -> list[Policy]:
    full = positions + 1
    grid = [Policy("chain", rows) for rows in range(min(MIN_CHAIN_ROWS, full), full + 1)]
    grid += [
        Policy("tree", rows, fanout, est)
        for rows in tree_rows
        for fanout in children
        for est in estimators
    ]
    grid.append(Policy("oracle", max([full, *tree_rows])))
    return grid


def evaluate_paired(
    requests: Sequence[Request], policies: Sequence[Policy], calibrations: Sequence[Calibration]
) -> list[list[tuple[int, int]]]:
    """Per policy, per request: (rounds, tokens emitted) with every recorded
    round scored at its own anchor."""
    results = []
    for policy in policies:
        per_request = []
        for request, calibration in zip(requests, calibrations, strict=True):
            emitted = sum(
                policy_emitted(policy, rnd, request.truth(rnd.index), calibration)
                for rnd in request.rounds
            )
            per_request.append((len(request.rounds), emitted))
        results.append(per_request)
    return results


def require_dense(requests: Sequence[Request]) -> None:
    sparse = [request.name for request in requests if not request.is_dense()]
    if sparse:
        shown = ", ".join(sparse[:3]) + (", ..." if len(sparse) > 3 else "")
        raise TraceError(
            "exact mode needs a draft record at every output position; record the trace with "
            f"AX_MLX_DFLASH2_TRACE_ACCEPT_CAP=0 ({len(sparse)} sparse request(s): {shown})"
        )


def evaluate_exact(
    requests: Sequence[Request], policies: Sequence[Policy], calibrations: Sequence[Calibration]
) -> list[list[tuple[int, int]]]:
    """Per policy, per request: (rounds, tokens) of the request replayed from
    its first position, each round starting where the previous one stopped."""
    require_dense(requests)
    results = []
    for policy in policies:
        per_request = []
        for request, calibration in zip(requests, calibrations, strict=True):
            position = 0
            rounds = 0
            while position < len(request.tokens):
                rnd = request.rounds[position]
                emitted = policy_emitted(policy, rnd, request.truth(position), calibration)
                position += max(emitted, 1)
                rounds += 1
            per_request.append((rounds, len(request.tokens)))
        results.append(per_request)
    return results


def engine_check(requests: Sequence[Request]) -> tuple[int, int]:
    """(rounds compared, rounds where the chain cut to the engine's window
    matches the drafts the engine's verify matched). Only rounds whose whole
    window has ground truth are compared."""
    compared = 0
    agreed = 0
    for request in requests:
        for rnd in request.rounds:
            if rnd.window is None or rnd.matched_drafts is None:
                continue
            truth = request.truth(rnd.index)
            if len(truth) < rnd.window - 1:
                continue
            compared += 1
            agreed += chain_accepted(rnd.draft, rnd.window, truth) == rnd.matched_drafts
    return compared, agreed


def parse_cost(value: str) -> tuple[int, float]:
    try:
        rows_text, ms_text = value.split("=", 1)
        rows, ms = int(rows_text), float(ms_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected rows=ms, got {value!r}") from error
    if rows < 1 or not ms > 0 or not math.isfinite(ms):
        raise argparse.ArgumentTypeError(f"rows must be >= 1 and ms > 0, got {value!r}")
    return rows, ms


def round_ms(costs: dict[int, float], rows: int) -> tuple[float, bool]:
    """Round cost at `rows` by linear interpolation between the given points,
    and whether `rows` lies outside them (then the nearest segment is
    extended; a single point prices every row count the same)."""
    points = sorted(costs.items())
    if not points:
        raise ValueError("no cost points")
    if rows in costs:
        return costs[rows], False
    outside = rows < points[0][0] or rows > points[-1][0]
    if len(points) == 1:
        return points[0][1], True
    upper = next((i for i, (at, _) in enumerate(points) if at > rows), len(points) - 1)
    upper = max(upper, 1)
    (x0, y0), (x1, y1) = points[upper - 1], points[upper]
    return y0 + (y1 - y0) * (rows - x0) / (x1 - x0), outside


def summarize(
    policies: Sequence[Policy],
    totals: Sequence[tuple[int, int]],
    baseline: Policy,
    costs: dict[int, float] | None,
) -> list[dict]:
    """Report rows of one evaluation: `totals[i]` is (rounds, tokens) of
    `policies[i]`; ratios are against `baseline`."""
    rows = []
    for policy, (rounds, tokens) in zip(policies, totals, strict=True):
        row: dict = {
            "policy": policy.name,
            "kind": policy.kind,
            "rows": policy.rows,
            "rounds": rounds,
            "tokens": tokens,
            "tokens_per_round": tokens / rounds if rounds else None,
        }
        if policy.kind == "tree":
            row["children"] = policy.children
            row["est"] = policy.est
        if costs:
            ms, extrapolated = round_ms(costs, policy.rows)
            row["round_ms"] = ms
            row["cost_extrapolated"] = extrapolated
            row["tok_per_s"] = 1000.0 * tokens / (rounds * ms) if rounds else None
        rows.append(row)
    reference = next((row for row in rows if row["policy"] == baseline.name), None)
    for row in rows:
        for key, ratio in (("tokens_per_round", "vs_chain"), ("tok_per_s", "tok_per_s_vs_chain")):
            if key in row:
                base = reference.get(key) if reference else None
                row[ratio] = row[key] / base if base and row[key] is not None else None
    return rows


def sum_requests(per_request: Sequence[tuple[int, int]]) -> tuple[int, int]:
    return sum(r for r, _ in per_request), sum(t for _, t in per_request)


def build_report(
    requests: Sequence[Request],
    policies: Sequence[Policy],
    mode: str,
    costs: dict[int, float] | None = None,
    per_request: bool = False,
) -> dict:
    """Evaluate `policies` in `mode` (`paired`, `exact` or `both`)."""
    if not requests or not any(request.rounds for request in requests):
        raise TraceError("the trace holds no drafted round")
    positions = max(rnd.draft.positions for request in requests for rnd in request.rounds)
    baseline = Policy("chain", positions + 1)
    if baseline not in policies:
        policies = [baseline, *policies]
    if any(policy.est == "calibrated" for policy in policies):
        calibrations, pooled = cross_fit(requests, positions)
    else:
        calibrations, pooled = [Calibration()] * len(requests), None
    compared, agreed = engine_check(requests)
    report: dict = {
        "requests": len(requests),
        "rounds": sum(len(request.rounds) for request in requests),
        "tokens": sum(len(request.tokens) for request in requests),
        "undrafted_commits": sum(request.undrafted_commits for request in requests),
        "positions": positions,
        "top_k": max(rnd.draft.top_k for request in requests for rnd in request.rounds),
        "dense": all(request.is_dense() for request in requests),
        "baseline": baseline.name,
        "engine_check": {"rounds_compared": compared, "rounds_agreed": agreed},
        "cost_ms": {str(rows): ms for rows, ms in sorted(costs.items())} if costs else None,
    }
    if pooled is not None:
        report["calibration"] = {
            "fit": "cross-fit by request" if len(requests) > 1 else "in-sample (one request)",
            "pooled_slope": pooled.slope,
            "pooled_intercepts": list(pooled.intercepts),
        }
    evaluations = []
    if mode in ("paired", "both"):
        # A paired round count is not a replay: it carries no throughput.
        evaluations.append(("paired", evaluate_paired(requests, policies, calibrations), None))
    if mode in ("exact", "both"):
        evaluations.append(("exact", evaluate_exact(requests, policies, calibrations), costs))
    for name, results, priced in evaluations:
        totals = [sum_requests(per) for per in results]
        report[name] = {"policies": summarize(policies, totals, baseline, priced)}
        if per_request:
            report[name]["per_request"] = [
                {
                    "request": request.name,
                    "policies": summarize(
                        policies, [per[index] for per in results], baseline, priced
                    ),
                }
                for index, request in enumerate(requests)
            ]
    return report


def _cell(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_table(rows: Sequence[dict], baseline: str) -> list[str]:
    priced = any("tok_per_s" in row for row in rows)
    header = ["policy", "rows", "rounds", "tokens", "tok/round", f"vs {baseline}"]
    if priced:
        header += ["round_ms", "tok/s", f"tok/s vs {baseline}"]
    table = [header]
    for row in rows:
        line = [
            row["policy"],
            _cell(row["rows"]),
            _cell(row["rounds"]),
            _cell(row["tokens"]),
            _cell(row["tokens_per_round"]),
            _cell(row["vs_chain"]),
        ]
        if priced:
            mark = "*" if row.get("cost_extrapolated") else ""
            line += [
                _cell(row.get("round_ms"), 2) + mark,
                _cell(row.get("tok_per_s"), 2),
                _cell(row.get("tok_per_s_vs_chain")),
            ]
        table.append(line)
    widths = [max(len(line[column]) for line in table) for column in range(len(header))]
    return [
        "  ".join(
            cell.ljust(width) if column == 0 else cell.rjust(width)
            for column, (cell, width) in enumerate(zip(line, widths, strict=True))
        ).rstrip()
        for line in table
    ]


def format_report(report: dict) -> str:
    lines = [
        "DFlash2 tree-verify simulation",
        f"requests {report['requests']}  rounds {report['rounds']}  tokens {report['tokens']}  "
        f"positions {report['positions']}  top_k {report['top_k']}  "
        f"dense {'yes' if report['dense'] else 'no'}",
    ]
    if report["undrafted_commits"]:
        lines.append(f"commits with no draft record: {report['undrafted_commits']}")
    check = report["engine_check"]
    if check["rounds_compared"]:
        lines.append(
            "engine check: the chain cut to the engine's window matched the engine's verify in "
            f"{check['rounds_agreed']} of {check['rounds_compared']} rounds"
        )
    calibration = report.get("calibration")
    if calibration:
        intercepts = ", ".join(f"{value:.3f}" for value in calibration["pooled_intercepts"])
        lines.append(
            f"calibration ({calibration['fit']}): pooled slope "
            f"{calibration['pooled_slope']:.3f}, intercepts by depth [{intercepts}]"
        )
    titles = {
        "paired": "paired: every recorded round scored at its own anchor",
        "exact": "exact: requests replayed round by round",
    }
    for name, title in titles.items():
        section = report.get(name)
        if not section:
            continue
        lines += ["", title, *format_table(section["policies"], report["baseline"])]
        if any(row.get("cost_extrapolated") for row in section["policies"]):
            lines.append("* round cost extended beyond the --cost points")
        for entry in section.get("per_request", []):
            lines += ["", f"{name} {entry['request']}"]
            lines += format_table(entry["policies"], report["baseline"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path, nargs="+", help="Candidate trace JSONL file(s).")
    parser.add_argument(
        "--mode",
        choices=("paired", "exact", "both"),
        default="paired",
        help="paired: score recorded rounds; exact: replay requests (needs a trace recorded "
        "with AX_MLX_DFLASH2_TRACE_ACCEPT_CAP=0); both: the two tables.",
    )
    parser.add_argument(
        "--cost",
        type=parse_cost,
        action="append",
        default=[],
        metavar="ROWS=MS",
        help="Measured round wall (ms) at a verify row count; repeat. Exact mode projects "
        "tokens per second with linear interpolation between the points.",
    )
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_TREE_ROWS))
    parser.add_argument("--children", type=int, nargs="+", default=list(DEFAULT_CHILDREN))
    parser.add_argument("--est", choices=ESTIMATORS, nargs="+", default=list(ESTIMATORS))
    parser.add_argument("--per-request", action="store_true", help="Add per-request tables.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args(argv)

    for path in args.traces:
        if not path.is_file():
            parser.error(f"trace not found: {path}")
    if any(rows < 1 for rows in args.rows) or any(fanout < 1 for fanout in args.children):
        parser.error("--rows and --children must be >= 1")
    costs = dict(args.cost) or None
    try:
        requests = load_traces(args.traces)
        if not any(request.rounds for request in requests):
            raise TraceError("the trace holds no drafted round")
        positions = max(rnd.draft.positions for request in requests for rnd in request.rounds)
        policies = default_grid(positions, args.rows, args.children, args.est)
        report = build_report(requests, policies, args.mode, costs, args.per_request)
    except TraceError as error:
        print(f"simulate_dflash2_tree: {error}", file=sys.stderr)
        return 2
    report = {"traces": [str(path) for path in args.traces], "mode": args.mode, **report}
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
