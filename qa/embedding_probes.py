"""Embedding model QA probes against a running ``ax-engine-server``.

Tiers (market-aligned, not full MTEB download)
----------------------------------------------
* **smoke** — engine health + small STS triple set
* **standard** (default) — smoke + MTEB-shaped pair classification + retrieval

Full public MTEB remains external. Family oracle remains
``scripts/verify_embedding_models.py``.

AX ``POST /v1/embeddings`` expects pre-tokenized token IDs; provide
``tokenizer.json`` from the model artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from embedding_bank import (
    CORPUS,
    PAIR_CLASSIFICATION,
    RETRIEVAL_CASES,
    SEMANTIC_TRIPLES,
)


@dataclass
class EmbedProbeResult:
    name: str
    passed: bool
    detail: str = ""
    hard: bool = True
    elapsed_ms: float = 0.0
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbedReport:
    base_url: str
    model: str
    pooling: str
    tier: str = "standard"
    results: list[EmbedProbeResult] = field(default_factory=list)
    dim: int = 0
    # Market-comparable mini-metrics (not full MTEB scores).
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def hard_passed(self) -> bool:
        return all(r.passed or r.skipped for r in self.results if r.hard)

    @property
    def summary_line(self) -> str:
        hard = [r for r in self.results if r.hard and not r.skipped]
        ok = sum(1 for r in hard if r.passed)
        skip = sum(1 for r in self.results if r.skipped)
        bits = [f"embed[{self.tier}] hard {ok}/{len(hard)}", f"dim={self.dim}"]
        if "retrieval_hit_at_1" in self.metrics:
            bits.append(f"hit@1={self.metrics['retrieval_hit_at_1']:.2f}")
        if "pair_classification_ap" in self.metrics:
            bits.append(f"pair_ap={self.metrics['pair_classification_ap']:.2f}")
        elif "pair_classification_accuracy" in self.metrics:
            bits.append(f"pair_acc={self.metrics['pair_classification_accuracy']:.2f}")
        if "sts_triple_accuracy" in self.metrics:
            bits.append(f"sts_acc={self.metrics['sts_triple_accuracy']:.2f}")
        return " ".join(bits) + f" ({skip} skipped)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "kind": "embedding_probes",
            "tier": self.tier,
            "base_url": self.base_url,
            "model": self.model,
            "pooling": self.pooling,
            "dim": self.dim,
            "hard_passed": self.hard_passed,
            "summary": self.summary_line,
            "metrics": self.metrics,
            "market_alignment": {
                "mteb_full": False,
                "mteb_shaped_tasks": self.tier == "standard",
                "tasks": [
                    "engine_smoke",
                    "sts_triple_ranking",
                    *(
                        ["pair_classification", "retrieval_hit_at_1_mrr"]
                        if self.tier == "standard"
                        else []
                    ),
                ],
                "oracle": "scripts/verify_embedding_models.py",
            },
            "results": [r.as_dict() for r in self.results],
        }


def hit_at_k(ranked_indices: Sequence[int], gold: int, k: int = 1) -> float:
    return 1.0 if gold in list(ranked_indices)[:k] else 0.0


def mrr_at_k(ranked_indices: Sequence[int], gold: int, k: int = 10) -> float:
    for rank, idx in enumerate(list(ranked_indices)[:k], start=1):
        if idx == gold:
            return 1.0 / rank
    return 0.0


def rank_by_cosine(query: Sequence[float], docs: Sequence[Sequence[float]]) -> list[int]:
    scored = [(i, cosine(query, d)) for i, d in enumerate(docs)]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in scored]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def is_finite_vector(v: Sequence[float]) -> bool:
    return all(math.isfinite(float(x)) for x in v)


def infer_pooling(model_id: str, artifacts_hint: str = "") -> str:
    blob = f"{model_id} {artifacts_hint}".lower()
    if "embeddinggemma" in blob or "gemma-embedding" in blob:
        return "mean"
    return "last"


def _eos_token_string_from_config(tokenizer_path: str | Path) -> Optional[str]:
    """Read ``eos_token`` from tokenizer_config.json next to tokenizer.json."""
    path = Path(tokenizer_path)
    cfg_path = path.with_name("tokenizer_config.json") if path.name == "tokenizer.json" else path / "tokenizer_config.json"
    if not cfg_path.is_file():
        # also try parent when path is the artifact root
        alt = path.parent / "tokenizer_config.json"
        cfg_path = alt if alt.is_file() else cfg_path
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    eos = cfg.get("eos_token")
    if isinstance(eos, dict):
        content = eos.get("content")
        return str(content) if content is not None else None
    if isinstance(eos, str) and eos:
        return eos
    return None


def resolve_eos_id(tok: Any, tokenizer_path: Optional[str] = None) -> Optional[int]:
    """Resolve EOS id matching verify_embedding_models / HF family contracts.

    Bug fixed: earlier code preferred ``</s>`` (often present but wrong for Qwen
    chat/embedding templates). Prefer ``tokenizer_config.json`` eos, then
    ``<|im_end|>``, ``<|endoftext|>``, and only then legacy ``</s>``.
    """
    # transformers objects expose eos_token_id directly
    eos_attr = getattr(tok, "eos_token_id", None)
    if eos_attr is not None:
        try:
            return int(eos_attr)
        except (TypeError, ValueError):
            pass

    candidates: list[str] = []
    if tokenizer_path:
        cfg_eos = _eos_token_string_from_config(tokenizer_path)
        if cfg_eos:
            candidates.append(cfg_eos)
    # Qwen chat/embedding; pad-as-eot; legacy
    for cand in ("<|im_end|>", "<|endoftext|>", "</s>", "<eos>"):
        if cand not in candidates:
            candidates.append(cand)

    token_to_id = getattr(tok, "token_to_id", None)
    if not callable(token_to_id):
        convert = getattr(tok, "convert_tokens_to_ids", None)
        if callable(convert):
            for cand in candidates:
                try:
                    tid = convert(cand)
                except Exception:
                    continue
                if tid is not None and int(tid) >= 0:
                    # HF uses unk id for missing tokens sometimes
                    unk = getattr(tok, "unk_token_id", None)
                    if unk is not None and int(tid) == int(unk):
                        continue
                    return int(tid)
        return None

    for cand in candidates:
        tid = token_to_id(cand)
        if tid is not None:
            return int(tid)
    return None


def load_tokenizer(tokenizer_path: str) -> Any:
    """Load a tokenizer from ``tokenizer.json`` (or its parent artifact dir).

    Returns ``(backend, tokenizer_obj, eos_id)``. Prefer the lightweight
    ``tokenizers`` package; fall back to ``transformers.AutoTokenizer``.
    """
    path = Path(tokenizer_path)
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(path))
        return ("tokenizers", tok, resolve_eos_id(tok, str(path)))
    except ImportError:
        pass
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "embedding QA needs either the `tokenizers` or `transformers` package"
        ) from exc
    root = path.parent if path.name == "tokenizer.json" else path
    tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    return ("transformers", tok, resolve_eos_id(tok, str(path)))


def encode_texts(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    model_kind: str = "qwen3",
) -> list[list[int]]:
    """Encode texts to token-id rows matching AX embedding family conventions.

    - Qwen3-Embedding: no special tokens, append EOS if present.
    - EmbeddingGemma: add special tokens (CLS/BOS style path).

    ``tokenizer`` may be a raw HF object or ``(backend, obj[, eos_id])`` from
    ``load_tokenizer``.
    """
    backend = "auto"
    tok = tokenizer
    eos_id: Optional[int] = None
    if isinstance(tokenizer, tuple):
        if len(tokenizer) == 3:
            backend, tok, eos_id = tokenizer
        elif len(tokenizer) == 2:
            backend, tok = tokenizer

    rows: list[list[int]] = []
    for text in texts:
        if backend == "transformers" or hasattr(tok, "eos_token_id"):
            # transformers.PreTrainedTokenizerFast / AutoTokenizer
            if model_kind == "embeddinggemma":
                ids = list(tok.encode(text, add_special_tokens=True))
            else:
                ids = list(tok.encode(text, add_special_tokens=False))
                use_eos = eos_id if eos_id is not None else getattr(tok, "eos_token_id", None)
                if use_eos is not None and (not ids or ids[-1] != int(use_eos)):
                    ids.append(int(use_eos))
        else:
            # tokenizers.Tokenizer
            if model_kind == "embeddinggemma":
                encoded = tok.encode(text, add_special_tokens=True)
                ids = list(encoded.ids)
            else:
                encoded = tok.encode(text, add_special_tokens=False)
                ids = list(encoded.ids)
                use_eos = eos_id if eos_id is not None else resolve_eos_id(tok)
                if use_eos is not None and (not ids or ids[-1] != int(use_eos)):
                    ids.append(int(use_eos))
        if not ids:
            raise ValueError(f"empty tokenization for {text!r}")
        rows.append([int(i) for i in ids])
    return rows


def infer_model_kind(model_id: str, artifacts_hint: str = "") -> str:
    blob = f"{model_id} {artifacts_hint}".lower()
    if "embeddinggemma" in blob or ("gemma" in blob and "embed" in blob):
        return "embeddinggemma"
    return "qwen3"


def default_batch_threshold(model_id: str, artifacts_hint: str = "") -> float:
    """Match verify_embedding_models: large Qwen 4-bit/DWQ needs a looser gate."""
    blob = f"{model_id} {artifacts_hint}".lower()
    if "8b" in blob and ("4bit" in blob or "dwq" in blob):
        return 0.996
    return 0.999


def _post_embeddings(
    base_url: str,
    model: str,
    input_ids: list[int] | list[list[int]],
    *,
    pooling: str,
    normalize: bool = True,
    timeout: float = 60.0,
) -> tuple[int, dict[str, Any] | str]:
    url = f"{base_url.rstrip('/')}/v1/embeddings"
    payload = {
        "model": model,
        "input": input_ids,
        "pooling": pooling,
        "normalize": normalize,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = int(getattr(resp, "status", 200) or 200)
            try:
                return status, json.loads(body)
            except json.JSONDecodeError:
                return status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        try:
            return int(exc.code), json.loads(raw)
        except json.JSONDecodeError:
            return int(exc.code), raw
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f"connection_error: {exc}"


def extract_vectors(body: dict[str, Any] | str) -> list[list[float]]:
    if not isinstance(body, dict):
        return []
    data = body.get("data") or []
    # Sort by index if present
    try:
        data = sorted(data, key=lambda d: int(d.get("index", 0)))
    except Exception:
        pass
    out: list[list[float]] = []
    for item in data:
        emb = item.get("embedding")
        if isinstance(emb, list) and emb:
            out.append([float(x) for x in emb])
    return out


def probe_api_shape(
    base_url: str,
    model: str,
    token_rows: list[list[int]],
    *,
    pooling: str,
    timeout: float,
) -> tuple[EmbedProbeResult, list[list[float]]]:
    name = "api_shape"
    start = time.monotonic()
    status, body = _post_embeddings(
        base_url, model, token_rows, pooling=pooling, timeout=timeout
    )
    elapsed = (time.monotonic() - start) * 1000
    if status != 200 or not isinstance(body, dict):
        return (
            EmbedProbeResult(
                name, False, f"HTTP {status}: {str(body)[:200]}", elapsed_ms=elapsed
            ),
            [],
        )
    if body.get("object") != "list":
        return (
            EmbedProbeResult(
                name, False, f"object={body.get('object')!r} expected list", elapsed_ms=elapsed
            ),
            [],
        )
    vecs = extract_vectors(body)
    if len(vecs) != len(token_rows):
        return (
            EmbedProbeResult(
                name,
                False,
                f"data len {len(vecs)} != input batch {len(token_rows)}",
                elapsed_ms=elapsed,
            ),
            [],
        )
    dim = len(vecs[0])
    if dim < 8:
        return (
            EmbedProbeResult(name, False, f"dim={dim} too small", elapsed_ms=elapsed),
            [],
        )
    for i, v in enumerate(vecs):
        if len(v) != dim:
            return (
                EmbedProbeResult(
                    name, False, f"row {i} dim {len(v)} != {dim}", elapsed_ms=elapsed
                ),
                [],
            )
        if not is_finite_vector(v):
            return (
                EmbedProbeResult(
                    name, False, f"row {i} has non-finite values", elapsed_ms=elapsed
                ),
                [],
            )
        if all(abs(x) < 1e-12 for x in v):
            return (
                EmbedProbeResult(name, False, f"row {i} is all zeros", elapsed_ms=elapsed),
                [],
            )
    return (
        EmbedProbeResult(
            name,
            True,
            f"batch={len(vecs)} dim={dim} object=list",
            elapsed_ms=elapsed,
        ),
        vecs,
    )


def probe_l2_normalized(
    vectors: list[list[float]],
    *,
    tol: float = 1e-3,
) -> EmbedProbeResult:
    name = "l2_normalized"
    if not vectors:
        return EmbedProbeResult(name, False, "no vectors")
    errs = [abs(l2_norm(v) - 1.0) for v in vectors]
    max_err = max(errs)
    ok = max_err <= tol
    return EmbedProbeResult(
        name,
        ok,
        f"max |||v||-1| = {max_err:.6g} (tol {tol})",
    )


def probe_batch_vs_single(
    base_url: str,
    model: str,
    token_rows: list[list[int]],
    batch_vecs: list[list[float]],
    *,
    pooling: str,
    timeout: float,
    threshold: float = 0.999,
) -> EmbedProbeResult:
    name = "batch_vs_single"
    start = time.monotonic()
    single_vecs: list[list[float]] = []
    for row in token_rows:
        status, body = _post_embeddings(
            base_url, model, row, pooling=pooling, timeout=timeout
        )
        if status != 200:
            return EmbedProbeResult(
                name,
                False,
                f"single HTTP {status}: {str(body)[:160]}",
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        vecs = extract_vectors(body)
        if len(vecs) != 1:
            return EmbedProbeResult(
                name,
                False,
                f"single returned {len(vecs)} rows",
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        single_vecs.append(vecs[0])
    if len(single_vecs) != len(batch_vecs):
        return EmbedProbeResult(
            name,
            False,
            "length mismatch",
            elapsed_ms=(time.monotonic() - start) * 1000,
        )
    cosines = [cosine(a, b) for a, b in zip(single_vecs, batch_vecs)]
    min_c = min(cosines) if cosines else 0.0
    ok = min_c >= threshold
    return EmbedProbeResult(
        name,
        ok,
        f"min cosine(single, batch)={min_c:.6f} (threshold {threshold})",
        elapsed_ms=(time.monotonic() - start) * 1000,
    )


def probe_determinism(
    base_url: str,
    model: str,
    token_row: list[int],
    *,
    pooling: str,
    timeout: float,
    threshold: float = 0.9999,
) -> EmbedProbeResult:
    name = "determinism"
    start = time.monotonic()
    status1, body1 = _post_embeddings(
        base_url, model, token_row, pooling=pooling, timeout=timeout
    )
    status2, body2 = _post_embeddings(
        base_url, model, token_row, pooling=pooling, timeout=timeout
    )
    elapsed = (time.monotonic() - start) * 1000
    if status1 != 200 or status2 != 200:
        return EmbedProbeResult(
            name, False, f"HTTP {status1}/{status2}", elapsed_ms=elapsed
        )
    v1 = extract_vectors(body1)
    v2 = extract_vectors(body2)
    if len(v1) != 1 or len(v2) != 1:
        return EmbedProbeResult(name, False, "bad response shape", elapsed_ms=elapsed)
    c = cosine(v1[0], v2[0])
    return EmbedProbeResult(
        name,
        c >= threshold,
        f"cosine(run1, run2)={c:.8f} (threshold {threshold})",
        elapsed_ms=elapsed,
    )


def probe_semantic_order(
    base_url: str,
    model: str,
    tokenizer: Any,
    *,
    model_kind: str,
    pooling: str,
    timeout: float,
    margin: float = 0.02,
    triples: Optional[Sequence[tuple[str, str, str]]] = None,
) -> tuple[EmbedProbeResult, dict[str, float]]:
    """STS-style ranking: similar pair must outrank a random negative."""
    name = "sts_triple_ranking"
    items = list(triples) if triples is not None else list(SEMANTIC_TRIPLES)
    start = time.monotonic()
    wins = 0
    margins: list[float] = []
    details: list[str] = []
    for i, (anchor, positive, negative) in enumerate(items):
        rows = encode_texts(tokenizer, [anchor, positive, negative], model_kind=model_kind)
        status, body = _post_embeddings(
            base_url, model, rows, pooling=pooling, timeout=timeout
        )
        if status != 200:
            return (
                EmbedProbeResult(
                    name,
                    False,
                    f"triple {i} HTTP {status}",
                    elapsed_ms=(time.monotonic() - start) * 1000,
                ),
                {},
            )
        vecs = extract_vectors(body)
        if len(vecs) != 3:
            return (
                EmbedProbeResult(
                    name,
                    False,
                    f"triple {i} got {len(vecs)} vectors",
                    elapsed_ms=(time.monotonic() - start) * 1000,
                ),
                {},
            )
        c_pos = cosine(vecs[0], vecs[1])
        c_neg = cosine(vecs[0], vecs[2])
        margins.append(c_pos - c_neg)
        ok = c_pos > c_neg + margin
        details.append(f"t{i}: pos={c_pos:.3f} neg={c_neg:.3f}")
        if ok:
            wins += 1
    need = max(1, (len(items) * 2 + 2) // 3)  # ~2/3
    passed = wins >= need
    acc = wins / max(1, len(items))
    metrics = {
        "sts_triple_accuracy": acc,
        "sts_triple_mean_margin": sum(margins) / max(1, len(margins)),
        "sts_triple_n": float(len(items)),
    }
    return (
        EmbedProbeResult(
            name,
            passed,
            f"wins {wins}/{len(items)} (need {need}); " + "; ".join(details[:4]),
            elapsed_ms=(time.monotonic() - start) * 1000,
        ),
        metrics,
    )


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Binary average precision (MTEB pair-classification style ranking metric)."""
    if not scores or not any(labels):
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = 0
    prec_sum = 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i]:
            hits += 1
            prec_sum += hits / rank
    return prec_sum / sum(1 for lab in labels if lab)


def probe_pair_classification(
    base_url: str,
    model: str,
    tokenizer: Any,
    *,
    model_kind: str,
    pooling: str,
    timeout: float,
    min_ap: float = 0.80,
    min_pos_neg_margin: float = 0.02,
) -> tuple[EmbedProbeResult, dict[str, float]]:
    """MTEB PairClassification shape: rank positives above negatives (AP + margin)."""
    name = "pair_classification"
    start = time.monotonic()
    scores: list[float] = []
    labels: list[bool] = []
    for a, b, label in PAIR_CLASSIFICATION:
        rows = encode_texts(tokenizer, [a, b], model_kind=model_kind)
        status, body = _post_embeddings(
            base_url, model, rows, pooling=pooling, timeout=timeout
        )
        if status != 200:
            return (
                EmbedProbeResult(
                    name,
                    False,
                    f"HTTP {status}",
                    elapsed_ms=(time.monotonic() - start) * 1000,
                ),
                {},
            )
        vecs = extract_vectors(body)
        if len(vecs) != 2:
            return (
                EmbedProbeResult(
                    name,
                    False,
                    f"expected 2 vectors got {len(vecs)}",
                    elapsed_ms=(time.monotonic() - start) * 1000,
                ),
                {},
            )
        scores.append(cosine(vecs[0], vecs[1]))
        labels.append(bool(label))
    n = len(PAIR_CLASSIFICATION)
    ap = average_precision(scores, labels)
    pos = [s for s, lab in zip(scores, labels) if lab]
    neg = [s for s, lab in zip(scores, labels) if not lab]
    mean_pos = sum(pos) / max(1, len(pos))
    mean_neg = sum(neg) / max(1, len(neg))
    margin = mean_pos - mean_neg
    # Best fixed-threshold accuracy (diagnostic only; AP is the gate).
    best_acc = 0.0
    for thr in sorted(set(scores)):
        correct = sum((s >= thr) == lab for s, lab in zip(scores, labels))
        best_acc = max(best_acc, correct / max(1, n))
    metrics = {
        "pair_classification_ap": ap,
        "pair_classification_accuracy": best_acc,
        "pair_classification_mean_pos": mean_pos,
        "pair_classification_mean_neg": mean_neg,
        "pair_classification_margin": margin,
        "pair_classification_n": float(n),
    }
    passed = ap >= min_ap and margin >= min_pos_neg_margin
    return (
        EmbedProbeResult(
            name,
            passed,
            f"AP={ap:.3f} (min {min_ap}) margin={margin:.3f} "
            f"(pos={mean_pos:.3f} neg={mean_neg:.3f}) best_acc={best_acc:.3f}",
            hard=True,
            elapsed_ms=(time.monotonic() - start) * 1000,
        ),
        metrics,
    )


def probe_retrieval(
    base_url: str,
    model: str,
    tokenizer: Any,
    *,
    model_kind: str,
    pooling: str,
    timeout: float,
    min_hit_at_1: float = 0.80,
) -> tuple[EmbedProbeResult, dict[str, float]]:
    """MTEB Retrieval shape: Hit@1 and MRR on a tiny original corpus."""
    name = "retrieval"
    start = time.monotonic()
    hits: list[float] = []
    mrrs: list[float] = []
    details: list[str] = []
    for i, case in enumerate(RETRIEVAL_CASES):
        texts = [case["query"], *case["docs"]]
        rows = encode_texts(tokenizer, texts, model_kind=model_kind)
        status, body = _post_embeddings(
            base_url, model, rows, pooling=pooling, timeout=timeout
        )
        if status != 200:
            return (
                EmbedProbeResult(
                    name,
                    False,
                    f"case {i} HTTP {status}",
                    elapsed_ms=(time.monotonic() - start) * 1000,
                ),
                {},
            )
        vecs = extract_vectors(body)
        if len(vecs) != len(texts):
            return (
                EmbedProbeResult(
                    name,
                    False,
                    f"case {i} vector count {len(vecs)}",
                    elapsed_ms=(time.monotonic() - start) * 1000,
                ),
                {},
            )
        ranked = rank_by_cosine(vecs[0], vecs[1:])
        gold = int(case["gold"])
        h = hit_at_k(ranked, gold, k=1)
        m = mrr_at_k(ranked, gold, k=4)
        hits.append(h)
        mrrs.append(m)
        details.append(f"q{i}: rank0={ranked[0]} gold={gold}")
    hit1 = sum(hits) / max(1, len(hits))
    mrr = sum(mrrs) / max(1, len(mrrs))
    metrics = {
        "retrieval_hit_at_1": hit1,
        "retrieval_mrr": mrr,
        "retrieval_n": float(len(RETRIEVAL_CASES)),
    }
    return (
        EmbedProbeResult(
            name,
            hit1 >= min_hit_at_1,
            f"hit@1={hit1:.3f} mrr={mrr:.3f} min_hit@1={min_hit_at_1}; "
            + "; ".join(details),
            hard=True,
            elapsed_ms=(time.monotonic() - start) * 1000,
        ),
        metrics,
    )


def probe_empty_rejected(
    base_url: str,
    model: str,
    *,
    pooling: str,
    timeout: float,
) -> EmbedProbeResult:
    name = "empty_rejected"
    start = time.monotonic()
    status, body = _post_embeddings(
        base_url, model, [], pooling=pooling, timeout=timeout
    )
    elapsed = (time.monotonic() - start) * 1000
    ok = status in (400, 422)
    return EmbedProbeResult(
        name,
        ok,
        f"HTTP {status} (expect 400/422) body={str(body)[:120]}",
        elapsed_ms=elapsed,
    )


def run_embedding_probes(
    base_url: str,
    model: str,
    tokenizer_path: str,
    *,
    pooling: Optional[str] = None,
    model_kind: Optional[str] = None,
    artifacts_hint: str = "",
    timeout: float = 90.0,
    batch_threshold: Optional[float] = None,
    semantic_margin: float = 0.02,
    tier: str = "standard",
) -> EmbedReport:
    """Run embedding QA.

    ``tier``:
      * ``smoke`` — engine health + 3 STS triples
      * ``standard`` — smoke + full STS bank + pair classification + retrieval
    """
    tier = (tier or "standard").lower().strip()
    if tier not in ("smoke", "standard"):
        raise ValueError(f"unknown embed tier: {tier}")

    kind = model_kind or infer_model_kind(model, artifacts_hint)
    pool = pooling or infer_pooling(model, artifacts_hint)
    bt = (
        batch_threshold
        if batch_threshold is not None
        else default_batch_threshold(model, artifacts_hint)
    )
    report = EmbedReport(base_url=base_url, model=model, pooling=pool, tier=tier)

    tokenizer = load_tokenizer(tokenizer_path)
    token_rows = encode_texts(tokenizer, CORPUS, model_kind=kind)

    shape, batch_vecs = probe_api_shape(
        base_url, model, token_rows, pooling=pool, timeout=timeout
    )
    report.results.append(shape)
    if batch_vecs:
        report.dim = len(batch_vecs[0])

    if not shape.passed:
        return report

    report.results.append(probe_l2_normalized(batch_vecs))
    report.results.append(
        probe_batch_vs_single(
            base_url,
            model,
            token_rows[:4],  # keep smoke fast
            batch_vecs[:4],
            pooling=pool,
            timeout=timeout,
            threshold=bt,
        )
    )
    report.results.append(
        probe_determinism(
            base_url,
            model,
            token_rows[0],
            pooling=pool,
            timeout=timeout,
        )
    )
    report.results.append(
        probe_empty_rejected(base_url, model, pooling=pool, timeout=timeout)
    )

    # STS triples: smoke uses first 3; standard uses full bank.
    triples = SEMANTIC_TRIPLES[:3] if tier == "smoke" else SEMANTIC_TRIPLES
    sts_result, sts_metrics = probe_semantic_order(
        base_url,
        model,
        tokenizer,
        model_kind=kind,
        pooling=pool,
        timeout=timeout,
        margin=semantic_margin,
        triples=triples,
    )
    report.results.append(sts_result)
    report.metrics.update(sts_metrics)

    if tier == "standard":
        pair_result, pair_metrics = probe_pair_classification(
            base_url,
            model,
            tokenizer,
            model_kind=kind,
            pooling=pool,
            timeout=timeout,
        )
        report.results.append(pair_result)
        report.metrics.update(pair_metrics)

        ret_result, ret_metrics = probe_retrieval(
            base_url,
            model,
            tokenizer,
            model_kind=kind,
            pooling=pool,
            timeout=timeout,
        )
        report.results.append(ret_result)
        report.metrics.update(ret_metrics)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AX Engine embedding QA probes")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer.json (from model artifacts)",
    )
    parser.add_argument(
        "--pooling",
        default=None,
        choices=["last", "mean", "cls"],
        help="Pooling strategy (default: inferred from model id)",
    )
    parser.add_argument(
        "--model-kind",
        default=None,
        choices=["qwen3", "embeddinggemma"],
        help="Tokenization family (default: inferred)",
    )
    parser.add_argument(
        "--tier",
        default="standard",
        choices=["smoke", "standard"],
        help="smoke = engine health; standard = + pair classification + retrieval",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--json-output", default=None)
    parser.add_argument(
        "--batch-threshold",
        type=float,
        default=None,
        help="Override batch-vs-single cosine threshold (default: model-aware)",
    )
    parser.add_argument("--semantic-margin", type=float, default=0.02)
    args = parser.parse_args(argv)

    if not Path(args.tokenizer).is_file():
        print(f"tokenizer not found: {args.tokenizer}", flush=True)
        return 2

    report = run_embedding_probes(
        args.base_url,
        args.model,
        args.tokenizer,
        pooling=args.pooling,
        model_kind=args.model_kind,
        timeout=args.timeout,
        batch_threshold=args.batch_threshold,
        semantic_margin=args.semantic_margin,
        tier=args.tier,
    )
    for r in report.results:
        flag = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        print(f"  [{flag}] {r.name}: {r.detail} ({r.elapsed_ms:.0f}ms)")
    print(report.summary_line)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.as_dict(), indent=2))
        print(f"JSON: {path}")
    return 0 if report.hard_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
