#!/usr/bin/env python3
"""Compare n-gram drafting algorithms in the same Python decode loop.

Four columns, all sharing the same mlx_lm Python decode loop with greedy
argmax acceptance (T=0):

  - suite baseline:   no speculation, pure greedy decode
  - lightning:        trigram lookup, longest-match-first (adaptive_k)
  - rapid-mlx:        suffix decoding, multi-order voting with confidence floor
  - ax-ngram:         bigram+trigram+fourgram, majority-recency, confidence=0.4
                      (Python mirror of crates/ax-engine-mlx/src/ngram_accel.rs)

This is a fair algorithmic comparison: same model, same decode loop, same
acceptance criterion (argmax-only). The only variable is the drafting strategy.

For AX Engine end-to-end throughput (Rust/Metal decode, threshold acceptance),
see the "AX Engine N-gram Effective Throughput" table in the README, measured
by scripts/bench_ngram_vs_lightning.py.

Default mode follows the Rapid-MLX eligibility bench shape: ordinary MLX
community models, four real chat/tool/code workloads, greedy decoding, and
measurement gates that reject too-short or physically implausible runs.
For a fair Rapid-MLX-style comparison, pass a normal MLX community model
snapshot (for example ``mlx-community/Qwen3-4B-4bit``), not a
``Youssofal/*MTPLX*`` bundle. Hybrid Qwen3.5/Qwen3.6 caches are reported as
``non_trimmable_cache`` instead of emitting fake speculative numbers.

The older synthetic random-token prompt shape is still available with
``--prompt-mode random`` when side-by-side comparison with AX random-token
artifacts is needed.

All three speculative algorithms are implemented inline — no external package
install.

Lightning algorithm:   lookup_drafts_with_pending + adaptive_k
                       (mirrors lightning-mlx/vllm_mlx/speculative/ngram_drafter.py)
Rapid-MLX algorithm:   SuffixDecodingDrafter
                       (mirrors Rapid-MLX/vllm_mlx/speculative/suffix_decoding.py)
AX n-gram algorithm:   NgramTable (bigram+trigram+fourgram, majority-recency)
                       (mirrors crates/ax-engine-mlx/src/ngram_accel.rs)

Usage:
  python3 scripts/bench_speculative_suite.py \\
      --model-dir /path/to/model \\
      --output-dir benchmarks/results/speculative-suite/$(date +%F)-qwen3-4b \\
      --repetitions 3 --warmup 1

  # Random-token shape for AX benchmark parity:
      --prompt-mode random --prompt-tokens 128,512,2048

  # Skip one path:
      --skip-lightning
      --skip-rapid-mlx
      --skip-ax-ngram
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_LM_RANDOM_SEED = 0
DEFAULT_PROMPT_TOKENS = "128,512,2048"
DEFAULT_GEN_TOKENS = 256
DEFAULT_REPS = 3
DEFAULT_WARMUP = 1
DEFAULT_COOLDOWN = 8.0
NUM_DRAFT_TOKENS = 4
NGRAM_SIZE = 3
RAPID_SUFFIX_MAX_DRAFT_TOKENS = 8
RAPID_SUFFIX_MAX_SUFFIX_LEN = 4
RAPID_SUFFIX_MIN_CONFIDENCE = 0.3
# AX Engine n-gram constants (mirrors crates/ax-engine-mlx/src/ngram_accel.rs)
AX_NGRAM_MAX_DRAFT_LEN = 4
AX_NGRAM_CONFIDENCE_THRESHOLD = 0.4
AX_NGRAM_MIN_SUPPORT = 1
AX_NGRAM_TAIL_SIZE = 4
MIN_DECODE_TIME = 0.5
TPS_CEILING = 500.0


RAPID_WORKLOADS: list[tuple[str, str, dict]] = [
    (
        "chat",
        "low_repeat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a short, friendly explanation of why the sky appears "
                        "blue. Two paragraphs, no bullet points."
                    ),
                }
            ],
        },
    ),
    (
        "json_array",
        "structured",
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Return a JSON array of exactly 12 objects, each with the "
                        "fields `id` (integer 1..12), `name` (a short fruit "
                        "name) and `color` (a CSS color name). Output JSON only, "
                        "no commentary."
                    ),
                }
            ],
        },
    ),
    (
        "tool_loop",
        "agentic",
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a data-pipeline assistant. Use the tools exactly "
                        "when needed. Always end with a tool call to `submit_summary` "
                        "once data has been fetched, parsed, validated and written."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Build a fresh export of the `invoices` table for Q1 2026. "
                        "Read the latest snapshot, parse it, validate row counts, "
                        "write the result to S3, and finalize."
                    ),
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_snapshot",
                        "description": "Read latest data snapshot",
                        "parameters": {
                            "type": "object",
                            "properties": {"table": {"type": "string"}},
                            "required": ["table"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "parse_rows",
                        "description": "Parse rows",
                        "parameters": {
                            "type": "object",
                            "properties": {"format": {"type": "string"}},
                            "required": ["format"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "validate_counts",
                        "description": "Validate row counts vs expected",
                        "parameters": {
                            "type": "object",
                            "properties": {"expected": {"type": "integer"}},
                            "required": ["expected"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_s3",
                        "description": "Write export to S3",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                            "required": ["key"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "submit_summary",
                        "description": "Final summary, terminates the loop",
                        "parameters": {
                            "type": "object",
                            "properties": {"status": {"type": "string"}},
                            "required": ["status"],
                        },
                    },
                },
            ],
            "tool_choice": "auto",
        },
    ),
    (
        "code_edit",
        "structured",
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Below is a Python function. Replace the bare `except:` "
                        "with `except Exception as e:` and log the error using "
                        "`logger.exception`. Return the entire updated function "
                        "as a fenced code block.\n\n"
                        "```python\n"
                        "def fetch(url):\n"
                        "    try:\n"
                        "        return httpx.get(url, timeout=5).json()\n"
                        "    except:\n"
                        "        return None\n"
                        "```"
                    ),
                }
            ],
        },
    ),
]


# ---------------------------------------------------------------------------
# Lightning n-gram drafter (self-contained)
# Implements lookup_drafts_with_pending + adaptive_k from lightning-mlx.
# ---------------------------------------------------------------------------


class _LightningDrafter:
    def __init__(self, ngram_size: int = 3, num_draft_tokens: int = 4) -> None:
        self.ngram_size = ngram_size
        self.num_draft_tokens = num_draft_tokens
        self._history: list[int] = []
        self._index: dict[tuple, list[int]] = defaultdict(list)
        self.total_drafted = 0
        self.total_accepted = 0

    def feed(self, token: int) -> None:
        n = self.ngram_size
        self._history.append(token)
        pos = len(self._history) - 1
        for size in range(1, n + 1):
            if pos + 1 >= size:
                key = tuple(self._history[pos + 1 - size : pos + 1])
                self._index[key].append(pos + 1 - size)

    def feed_many(self, tokens: list[int]) -> None:
        for t in tokens:
            self.feed(t)

    def predict_next(self, pending_token: int) -> list[int]:
        """Return draft list (adaptive_k: longest match wins)."""
        history = self._history
        n = self.ngram_size
        K = self.num_draft_tokens
        if len(history) + 1 < n:
            return []
        query = (
            tuple(history[-(n - 1) :]) + (int(pending_token),)
            if n > 1
            else (int(pending_token),)
        )
        positions = self._index.get(query, [])
        if not positions:
            return []
        current_start = len(history) + 1 - n
        best: list[int] = []
        for start in positions:
            if start == current_start:
                continue
            cont_begin = start + n
            cont_end = min(cont_begin + K, len(history))
            cont = history[cont_begin:cont_end]
            if len(cont) > len(best):
                best = cont
        return list(best[:K])


# ---------------------------------------------------------------------------
# Rapid-MLX SuffixDecodingDrafter (self-contained)
# Mirrors Rapid-MLX/vllm_mlx/speculative/suffix_decoding.py
# ---------------------------------------------------------------------------


class _RapidMLXSuffixDrafter:
    def __init__(
        self,
        max_draft_tokens: int = RAPID_SUFFIX_MAX_DRAFT_TOKENS,
        max_suffix_len: int = RAPID_SUFFIX_MAX_SUFFIX_LEN,
        min_confidence: float = RAPID_SUFFIX_MIN_CONFIDENCE,
    ) -> None:
        if max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be >= 1")
        if max_suffix_len < 1:
            raise ValueError("max_suffix_len must be >= 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.max_draft_tokens = max_draft_tokens
        self.max_suffix_len = max_suffix_len
        self.min_confidence = min_confidence
        self._history: list[int] = []
        # _suffix_index[k][kgram] = start positions where the kgram appeared.
        self._suffix_index: list[dict[tuple, list[int]]] = [
            defaultdict(list) for _ in range(max_suffix_len + 1)
        ]
        self.total_drafted = 0
        self.total_accepted = 0

    def feed(self, token: int) -> None:
        self._history.append(token)
        pos = len(self._history) - 1
        for n in range(1, min(self.max_suffix_len + 1, pos + 2)):
            start = pos - n + 1
            if start >= 0:
                ngram = tuple(self._history[start : pos + 1])
                self._suffix_index[n][ngram].append(start)

    def feed_many(self, tokens: list[int]) -> None:
        for t in tokens:
            self.feed(t)

    def predict_next(self, pending_token: int) -> list[int]:
        """Return a confidence-truncated draft for history + pending_token."""
        effective_len = len(self._history) + 1
        max_k = min(self.max_suffix_len, effective_len)
        for k in range(max_k, 0, -1):
            if k == 1:
                query = (int(pending_token),)
            else:
                if len(self._history) < k - 1:
                    continue
                query = tuple(self._history[-(k - 1) :]) + (int(pending_token),)
            starts = self._suffix_index[k].get(query, [])
            if not starts:
                continue
            draft = self._build_draft_from_starts(starts, k)
            if draft:
                return draft
        return []

    def _build_draft_from_starts(self, starts: list[int], suffix_len: int) -> list[int]:
        draft: list[int] = []
        for offset in range(self.max_draft_tokens):
            counter: Counter[int] = Counter()
            for start in starts:
                cont_pos = start + suffix_len + offset
                if 0 <= cont_pos < len(self._history):
                    counter[self._history[cont_pos]] += 1
            if not counter:
                break
            token, count = counter.most_common(1)[0]
            confidence = count / sum(counter.values())
            if confidence < self.min_confidence:
                break
            draft.append(token)
            starts = [
                start
                for start in starts
                if start + suffix_len + offset < len(self._history)
                and self._history[start + suffix_len + offset] == token
            ]
            if not starts:
                break
        return draft


# ---------------------------------------------------------------------------
# AX Engine n-gram drafter (self-contained)
# Mirrors crates/ax-engine-mlx/src/ngram_accel.rs NgramTable:
#   bigram + trigram + fourgram lookup, majority-recency policy,
#   confidence threshold filtering, chained multi-step prediction.
# ---------------------------------------------------------------------------


class _AXNgramDrafter:
    """Python mirror of AX Engine's NgramTable for fair same-loop comparison.

    Differences from the Rust implementation that are intentional for this
    benchmark context:
      - No LRU eviction (benchmark runs are short: 128–256 tokens)
      - No verifier feedback tracking (same-loop uses argmax-only acceptance)
      - No prompt/output source distinction (min_support=1 uniformly)

    The algorithmic core is identical: multi-order lookup (4→3→2 gram),
    majority-recency continuation selection, confidence threshold gate,
    and chained prediction where each draft token extends the context.
    """

    def __init__(
        self,
        max_len: int = AX_NGRAM_MAX_DRAFT_LEN,
        min_support: int = AX_NGRAM_MIN_SUPPORT,
        confidence_threshold: float = AX_NGRAM_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.max_len = max_len
        self.min_support = min_support
        self.confidence_threshold = confidence_threshold
        # Each map: context_tuple → {token: [count, last_seen_index]}
        self._bigrams: dict[tuple, dict[int, list]] = defaultdict(dict)
        self._trigrams: dict[tuple, dict[int, list]] = defaultdict(dict)
        self._fourgrams: dict[tuple, dict[int, list]] = defaultdict(dict)
        self._tail: list[int] = []
        self._obs_index: int = 0
        self.total_drafted = 0
        self.total_accepted = 0

    def _record(self, table: dict, key: tuple, token: int) -> None:
        entry = table[key]
        if token in entry:
            entry[token][0] += 1
            entry[token][1] = self._obs_index
        else:
            entry[token] = [1, self._obs_index]

    def feed(self, token: int) -> None:
        self._obs_index += 1
        tail = self._tail
        n = len(tail)
        if n >= 2:
            self._record(self._bigrams, (tail[-2], tail[-1]), token)
        if n >= 3:
            self._record(self._trigrams, (tail[-3], tail[-2], tail[-1]), token)
        if n >= 4:
            self._record(
                self._fourgrams,
                (tail[-4], tail[-3], tail[-2], tail[-1]),
                token,
            )
        tail.append(token)
        if len(tail) > AX_NGRAM_TAIL_SIZE:
            tail.pop(0)

    def feed_many(self, tokens: list[int]) -> None:
        for t in tokens:
            self.feed(t)

    def _select(self, table: dict, key: tuple) -> tuple[int, int] | None:
        """Majority-recency: pick continuation with highest count; ties broken
        by most recent observation.  Returns (token, support) or None if the
        best candidate fails min_support or confidence_threshold."""
        entry = table.get(key)
        if not entry:
            return None
        best_token = None
        best_count = 0
        best_seen = 0
        total = 0
        for tok, (count, seen) in entry.items():
            total += count
            if count > best_count or (count == best_count and seen > best_seen):
                best_token = tok
                best_count = count
                best_seen = seen
        if best_token is None:
            return None
        if best_count < self.min_support:
            return None
        confidence = best_count / total if total > 0 else 0.0
        if confidence < self.confidence_threshold:
            return None
        return (best_token, best_count)

    def predict_next(self, pending_token: int) -> list[int]:
        """Chain multi-order lookups to produce up to max_len draft tokens.

        Mirrors NgramTable::predict_with_policy — try fourgram, then trigram,
        then bigram at each step; each accepted draft token extends the tail
        context for the next step.
        """
        draft: list[int] = []
        # Working buffer: last AX_NGRAM_TAIL_SIZE tokens + pending_token
        buf = list(self._tail)
        if len(buf) < 2:
            return []
        # The pending_token is the last committed token; it's already in the
        # tail from the caller's feed(curr) before predict_next(curr).
        # We use the current tail state directly.

        for _ in range(self.max_len):
            n = len(buf)
            selected = None
            if n >= 4:
                selected = self._select(
                    self._fourgrams, (buf[-4], buf[-3], buf[-2], buf[-1])
                )
                if selected is None:
                    selected = self._select(self._trigrams, (buf[-3], buf[-2], buf[-1]))
                if selected is None:
                    selected = self._select(self._bigrams, (buf[-2], buf[-1]))
            elif n == 3:
                selected = self._select(self._trigrams, (buf[-3], buf[-2], buf[-1]))
                if selected is None:
                    selected = self._select(self._bigrams, (buf[-2], buf[-1]))
            elif n == 2:
                selected = self._select(self._bigrams, (buf[-2], buf[-1]))

            if selected is None:
                break
            token, _support = selected
            draft.append(token)
            buf.append(token)
            if len(buf) > AX_NGRAM_TAIL_SIZE:
                buf.pop(0)

        return draft


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def random_token_prompt(vocab_size: int, n_tokens: int) -> list[int]:
    rng = random.Random(MLX_LM_RANDOM_SEED)
    return [rng.randrange(vocab_size) for _ in range(n_tokens)]


def model_vocab_size(model_dir: Path) -> int:
    cfg = json.loads((model_dir / "config.json").read_text())
    for key in ("vocab_size", "text_config"):
        if key == "vocab_size" and key in cfg:
            return int(cfg["vocab_size"])
        if key == "text_config" and isinstance(cfg.get(key), dict):
            if "vocab_size" in cfg[key]:
                return int(cfg[key]["vocab_size"])
    tokenizer_cfg_path = model_dir / "tokenizer_config.json"
    if tokenizer_cfg_path.exists():
        tokenizer_cfg = json.loads(tokenizer_cfg_path.read_text())
        if "vocab_size" in tokenizer_cfg:
            return int(tokenizer_cfg["vocab_size"])
    raise RuntimeError(f"could not determine vocab_size from {model_dir}/config.json")


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------


def _load_model(model_dir: Path):
    from mlx_lm.utils import load_model as _lm_load_model, load_tokenizer

    print(f"  [load] {model_dir}", file=sys.stderr, flush=True)
    model, _ = _lm_load_model(model_dir, strict=False)
    tokenizer = load_tokenizer(model_dir, {})
    return model, tokenizer


def _prefill(model, kv_cache, prompt_ids: list[int], chunk: int = 512):
    import mlx.core as mx

    y = mx.array(prompt_ids, mx.uint32)
    while y.size > 1:
        n = min(chunk, y.size - 1)
        model(y[:n][None], cache=kv_cache)
        mx.eval([c.state for c in kv_cache])
        mx.clear_cache()
        y = y[n:]
    logits = model(y[None], cache=kv_cache)
    logits = logits[:, -1, :]
    mx.eval(logits)
    return logits


def _argmax_token(logits_1d) -> int:
    import mlx.core as mx

    return int(mx.argmax(logits_1d).item())


def _classify_measurement(token_count: int, decode_time: float) -> dict:
    if token_count <= 0:
        return {
            "tps": None,
            "token_count": token_count,
            "decode_time": decode_time,
            "rejected_reason": "zero_generated_tokens",
        }
    if decode_time < MIN_DECODE_TIME:
        return {
            "tps": None,
            "token_count": token_count,
            "decode_time": decode_time,
            "rejected_reason": f"decode_time<{MIN_DECODE_TIME}s",
        }
    tps = token_count / decode_time
    if tps > TPS_CEILING:
        return {
            "tps": None,
            "token_count": token_count,
            "decode_time": decode_time,
            "rejected_reason": f"tps>{TPS_CEILING}",
        }
    return {
        "tps": tps,
        "token_count": token_count,
        "decode_time": decode_time,
        "rejected_reason": None,
    }


def _median_valid_tps(raw_runs: list[dict]) -> float | None:
    values = [run["tps"] for run in raw_runs if run.get("tps") is not None]
    return statistics.median(values) if values else None


# ---------------------------------------------------------------------------
# Baseline run (greedy, no speculation)
# ---------------------------------------------------------------------------


def run_baseline(
    model,
    prompt_ids: list[int],
    gen_tokens: int,
    reps: int,
    warmup: int,
    cooldown: float,
    eos_id: int,
    ignore_eos: bool = False,
) -> dict:
    import mlx.core as mx
    from mlx_lm.models import cache as mlx_cache

    raw_runs: list[dict] = []
    all_runs = warmup + reps
    for rep in range(all_runs):
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass
        kv = mlx_cache.make_prompt_cache(model)
        logits = _prefill(model, kv, prompt_ids)
        curr = _argmax_token(logits.squeeze(0))
        t0 = time.perf_counter()
        output: list[int] = [curr]
        for _ in range(gen_tokens - 1):
            if not ignore_eos and curr == eos_id:
                break
            logits = model(mx.array([curr], mx.uint32)[None], cache=kv)
            logits = logits[:, -1, :]
            mx.eval(logits)
            curr = _argmax_token(logits.squeeze(0))
            output.append(curr)
        elapsed = time.perf_counter() - t0
        if rep >= warmup:
            raw_runs.append(_classify_measurement(len(output), elapsed))
        if rep < all_runs - 1 and cooldown > 0:
            time.sleep(cooldown)

    return {
        "tok_s_median": _median_valid_tps(raw_runs),
        "raw_runs": raw_runs,
    }


# ---------------------------------------------------------------------------
# Speculative run — shared logic for both lightning and Rapid-MLX
# ---------------------------------------------------------------------------


def run_speculative(
    model,
    prompt_ids: list[int],
    gen_tokens: int,
    reps: int,
    warmup: int,
    cooldown: float,
    eos_id: int,
    drafter_cls,
    drafter_kwargs: dict,
    ignore_eos: bool = False,
) -> dict:
    import mlx.core as mx
    from mlx_lm.models import cache as mlx_cache

    # Check KV cache trimmability
    kv_test = mlx_cache.make_prompt_cache(model)
    try:
        trimmable = bool(mlx_cache.can_trim_prompt_cache(kv_test))
    except AttributeError:
        trimmable = all(getattr(c, "is_trimmable", lambda: False)() for c in kv_test)
    del kv_test

    if not trimmable:
        return {
            "tok_s_median": None,
            "accept_rate": None,
            "total_drafted": 0,
            "total_accepted": 0,
            "trimmable": False,
            "skipped_reason": "non_trimmable_cache",
            "raw_runs": [],
        }

    raw_runs: list[dict] = []
    total_drafted = 0
    total_accepted = 0
    all_runs = warmup + reps

    for rep in range(all_runs):
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

        kv = mlx_cache.make_prompt_cache(model)
        logits = _prefill(model, kv, prompt_ids)
        curr = _argmax_token(logits.squeeze(0))
        t0 = time.perf_counter()
        output: list[int] = [curr]

        drafter = drafter_cls(**drafter_kwargs)
        drafter.feed_many(prompt_ids)

        rep_drafted = 0
        rep_accepted = 0

        for _ in range(gen_tokens - 1):
            if (not ignore_eos and curr == eos_id) or len(output) >= gen_tokens:
                break

            drafts = drafter.predict_next(curr)

            if not drafts:
                committed = curr
                logits = model(mx.array([committed], mx.uint32)[None], cache=kv)
                logits = logits[:, -1, :]
                mx.eval(logits)
                curr = _argmax_token(logits.squeeze(0))
                drafter.feed(committed)
                output.append(curr)
            else:
                k = len(drafts)
                verify_tokens = mx.array([curr] + list(drafts), mx.uint32)
                logits = model(verify_tokens[None], cache=kv)
                logits = logits[0, : k + 1, :]
                mx.eval(logits)

                n_accept = 0
                for i in range(k):
                    predicted = _argmax_token(logits[i])
                    if predicted == drafts[i]:
                        n_accept += 1
                    else:
                        break

                next_tok = _argmax_token(logits[n_accept])
                trim_amount = k - n_accept
                if trim_amount > 0:
                    mlx_cache.trim_prompt_cache(kv, trim_amount)

                rep_drafted += k
                rep_accepted += n_accept
                drafter.feed(curr)
                for i in range(n_accept):
                    drafter.feed(drafts[i])
                    output.append(drafts[i])
                output.append(next_tok)
                curr = next_tok

                if not ignore_eos and eos_id in output[-(n_accept + 1) :]:
                    break

        elapsed = time.perf_counter() - t0
        if rep >= warmup:
            raw_runs.append(_classify_measurement(len(output), elapsed))
            total_drafted += rep_drafted
            total_accepted += rep_accepted
        if rep < all_runs - 1 and cooldown > 0:
            time.sleep(cooldown)

    accept_rate = total_accepted / total_drafted if total_drafted > 0 else None
    return {
        "tok_s_median": _median_valid_tps(raw_runs),
        "accept_rate": accept_rate,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "trimmable": trimmable,
        "raw_runs": raw_runs,
    }


# ---------------------------------------------------------------------------
# Benchmark case construction and runner
# ---------------------------------------------------------------------------


def _workload_prompt_text(tokenizer, body: dict) -> str:
    messages = body["messages"]
    try:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "tools" in body:
            kwargs["tools"] = body["tools"]
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        lines = []
        for message in messages:
            lines.append(
                f"{message.get('role', 'user').title()}: {message.get('content', '')}"
            )
        return "\n".join(lines) + "\nAssistant:"


def build_cases(
    *,
    prompt_mode: str,
    tokenizer,
    vocab_size: int,
    prompt_lengths: list[int],
) -> list[dict]:
    if prompt_mode == "random":
        return [
            {
                "id": f"random_p{tokens}",
                "category": "random",
                "label": f"random {tokens}",
                "prompt_tokens": tokens,
                "prompt_ids": random_token_prompt(vocab_size, tokens),
            }
            for tokens in prompt_lengths
        ]

    cases = []
    for workload_id, category, body in RAPID_WORKLOADS:
        prompt_text = _workload_prompt_text(tokenizer, body)
        prompt_ids = tokenizer.encode(prompt_text)
        cases.append(
            {
                "id": workload_id,
                "category": category,
                "label": workload_id,
                "prompt_tokens": len(prompt_ids),
                "prompt_ids": prompt_ids,
            }
        )
    return cases


def _format_tps(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "—"


def _format_accept(result: dict | None) -> str:
    if not result:
        return "skipped"
    if result.get("skipped_reason"):
        return result["skipped_reason"]
    accept_rate = result.get("accept_rate")
    if accept_rate is None:
        return "no match"
    return f"{accept_rate:.1%}"


def run_case(
    model,
    tokenizer,
    case: dict,
    gen_tokens: int,
    reps: int,
    warmup: int,
    cooldown: float,
    skip_lightning: bool,
    skip_rapid_mlx: bool,
    skip_ax_ngram: bool = False,
    ignore_eos: bool = False,
) -> dict:
    import mlx.core as mx

    try:
        mx.clear_cache()
    except Exception:
        pass

    prompt_ids = case["prompt_ids"]
    eos_id = tokenizer.eos_token_id or 0

    print(
        f"\n  case={case['id']}  category={case['category']}  "
        f"prompt_tokens={case['prompt_tokens']}  gen_tokens={gen_tokens}",
        flush=True,
    )

    # Baseline
    print("    [baseline] ...", end="", flush=True)
    baseline = run_baseline(
        model,
        prompt_ids,
        gen_tokens,
        reps,
        warmup,
        cooldown,
        eos_id,
        ignore_eos=ignore_eos,
    )
    print(f" {_format_tps(baseline['tok_s_median'])} tok/s", flush=True)

    # Lightning
    lightning_result: dict | None = None
    if not skip_lightning:
        print("    [lightning] ...", end="", flush=True)
        try:
            lightning_result = run_speculative(
                model,
                prompt_ids,
                gen_tokens,
                reps,
                warmup,
                cooldown,
                eos_id,
                _LightningDrafter,
                {"ngram_size": NGRAM_SIZE, "num_draft_tokens": NUM_DRAFT_TOKENS},
                ignore_eos=ignore_eos,
            )
            ar = _format_accept(lightning_result)
            print(
                f" {_format_tps(lightning_result['tok_s_median'])} tok/s  accept={ar}",
                flush=True,
            )
        except Exception as e:
            print(f" ERROR: {e}", flush=True)

    # Rapid-MLX SuffixDecoding
    rapid_result: dict | None = None
    if not skip_rapid_mlx:
        print("    [rapid-mlx] ...", end="", flush=True)
        try:
            rapid_result = run_speculative(
                model,
                prompt_ids,
                gen_tokens,
                reps,
                warmup,
                cooldown,
                eos_id,
                _RapidMLXSuffixDrafter,
                {
                    "max_draft_tokens": RAPID_SUFFIX_MAX_DRAFT_TOKENS,
                    "max_suffix_len": RAPID_SUFFIX_MAX_SUFFIX_LEN,
                    "min_confidence": RAPID_SUFFIX_MIN_CONFIDENCE,
                },
                ignore_eos=ignore_eos,
            )
            ar = _format_accept(rapid_result)
            print(
                f" {_format_tps(rapid_result['tok_s_median'])} tok/s  accept={ar}",
                flush=True,
            )
        except Exception as e:
            print(f" ERROR: {e}", flush=True)

    # AX Engine n-gram (same-loop, argmax-only acceptance)
    ax_ngram_result: dict | None = None
    if not skip_ax_ngram:
        print("    [ax-ngram] ...", end="", flush=True)
        try:
            ax_ngram_result = run_speculative(
                model,
                prompt_ids,
                gen_tokens,
                reps,
                warmup,
                cooldown,
                eos_id,
                _AXNgramDrafter,
                {
                    "max_len": AX_NGRAM_MAX_DRAFT_LEN,
                    "min_support": AX_NGRAM_MIN_SUPPORT,
                    "confidence_threshold": AX_NGRAM_CONFIDENCE_THRESHOLD,
                },
                ignore_eos=ignore_eos,
            )
            ar = _format_accept(ax_ngram_result)
            print(
                f" {_format_tps(ax_ngram_result['tok_s_median'])} tok/s  accept={ar}",
                flush=True,
            )
        except Exception as e:
            print(f" ERROR: {e}", flush=True)

    return {
        "case_id": case["id"],
        "category": case["category"],
        "prompt_tokens": case["prompt_tokens"],
        "gen_tokens": gen_tokens,
        "ignore_eos": ignore_eos,
        "baseline": baseline,
        "lightning": lightning_result,
        "rapid_mlx": rapid_result,
        "ax_ngram_same_loop": ax_ngram_result,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=["rapid", "random"],
        default="rapid",
        help="rapid uses Rapid-MLX chat/json/tool/code workloads; random uses synthetic random-token prompts.",
    )
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--generation-tokens", type=int, default=DEFAULT_GEN_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    parser.add_argument("--skip-lightning", action="store_true")
    parser.add_argument("--skip-rapid-mlx", action="store_true")
    parser.add_argument("--skip-ax-ngram", action="store_true")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Always generate exactly generation-tokens tokens (ignore EOS). "
        "Use for models where random-token prompts trigger EOS at step 0.",
    )
    args = parser.parse_args()

    prompt_lengths = [int(x.strip()) for x in args.prompt_tokens.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_dir}", flush=True)
    print(
        f"Prompt mode: {args.prompt_mode}  gen={args.generation_tokens}  "
        f"reps={args.repetitions}+{args.warmup}w",
        flush=True,
    )

    vocab_size = model_vocab_size(args.model_dir)
    model, tokenizer = _load_model(args.model_dir)
    cases = build_cases(
        prompt_mode=args.prompt_mode,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        prompt_lengths=prompt_lengths,
    )

    rows: list[dict] = []
    for case in cases:
        row = run_case(
            model,
            tokenizer,
            case,
            args.generation_tokens,
            args.repetitions,
            args.warmup,
            args.cooldown,
            args.skip_lightning,
            args.skip_rapid_mlx,
            args.skip_ax_ngram,
            ignore_eos=args.ignore_eos,
        )
        rows.append(row)

    artifact = {
        "schema": "ax.bench.speculative_suite.v1",
        "date": str(date.today()),
        "model_dir": str(args.model_dir),
        "prompt_mode": args.prompt_mode,
        "sampling": "greedy_argmax_t0",
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "measurement_gates": {
            "min_decode_time_s": MIN_DECODE_TIME,
            "tps_ceiling": TPS_CEILING,
        },
        "fairness_contract": {
            "reference": "Rapid-MLX SuffixDecoding eligibility bench",
            "model_requirement": "ordinary MLX community snapshot; do not use Youssofal MTPLX bundles for this comparison",
            "hybrid_policy": "skip speculative paths when the prompt cache is not trimmable",
        },
        "lightning": {
            "ngram_size": NGRAM_SIZE,
            "num_draft_tokens": NUM_DRAFT_TOKENS,
        },
        "rapid_mlx": {
            "algorithm": "SuffixDecodingDrafter",
            "max_draft_tokens": RAPID_SUFFIX_MAX_DRAFT_TOKENS,
            "max_suffix_len": RAPID_SUFFIX_MAX_SUFFIX_LEN,
            "min_confidence": RAPID_SUFFIX_MIN_CONFIDENCE,
        },
        "ax_ngram_same_loop": {
            "algorithm": "NgramTable (Python mirror)",
            "source": "crates/ax-engine-mlx/src/ngram_accel.rs",
            "max_draft_len": AX_NGRAM_MAX_DRAFT_LEN,
            "min_support": AX_NGRAM_MIN_SUPPORT,
            "confidence_threshold": AX_NGRAM_CONFIDENCE_THRESHOLD,
            "tail_size": AX_NGRAM_TAIL_SIZE,
            "orders": ["bigram", "trigram", "fourgram"],
            "policy": "majority_recency",
            "acceptance": "argmax_only (same as lightning/rapid-mlx)",
        },
        "results": rows,
    }

    artifact_path = args.output_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nArtifact: {artifact_path}", flush=True)

    # Summary table
    print(f"\n=== SUMMARY (tok/s, T=0 greedy, {args.prompt_mode} prompts) ===")
    header = (
        f"{'case':>14}  {'pt':>6}  {'baseline':>10}  "
        f"{'lightning':>20}  {'rapid-mlx':>20}  {'ax-ngram':>20}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        case_id = row["case_id"]
        pt = row["prompt_tokens"]
        base = _format_tps(row["baseline"]["tok_s_median"])
        lt = row["lightning"]
        rm = row["rapid_mlx"]
        ax = row["ax_ngram_same_loop"]
        lt_str = (
            f"{_format_tps(lt['tok_s_median'])} ({_format_accept(lt)})"
            if lt
            else "skipped"
        )
        rm_str = (
            f"{_format_tps(rm['tok_s_median'])} ({_format_accept(rm)})"
            if rm
            else "skipped"
        )
        ax_str = (
            f"{_format_tps(ax['tok_s_median'])} ({_format_accept(ax)})"
            if ax
            else "skipped"
        )
        print(
            f"{case_id:>14}  {pt:>6}  {base:>10}  "
            f"{lt_str:>20}  {rm_str:>20}  {ax_str:>20}"
        )

    # Write summary markdown
    summary_lines = [
        "# Speculative Suite Benchmark\n",
        f"Date: {date.today()}  \n",
        f"Model: `{args.model_dir}`  \n",
        f"Prompt mode: `{args.prompt_mode}`  \n",
        f"Sampling: greedy (T=0)  \n",
        f"Gen tokens: {args.generation_tokens}, Reps: {args.repetitions}+{args.warmup}w\n\n",
        f"Measurement gates: decode_time >= {MIN_DECODE_TIME}s, tok/s <= {TPS_CEILING}.  \n\n",
        "All four columns share the same Python mlx_lm decode loop with argmax-only "
        "acceptance. This is a fair algorithmic comparison of drafting strategies.\n\n",
        "| Case | Category | Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx suffix | ax-ngram (same-loop) |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        case_id = row["case_id"]
        category = row["category"]
        pt = row["prompt_tokens"]
        base = _format_tps(row["baseline"]["tok_s_median"])
        lt = row["lightning"]
        rm = row["rapid_mlx"]
        ax = row["ax_ngram_same_loop"]
        lt_str = (
            f"{_format_tps(lt['tok_s_median'])} (ar={_format_accept(lt)})"
            if lt
            else "—"
        )
        rm_str = (
            f"{_format_tps(rm['tok_s_median'])} (ar={_format_accept(rm)})"
            if rm
            else "—"
        )
        ax_str = (
            f"{_format_tps(ax['tok_s_median'])} (ar={_format_accept(ax)})"
            if ax
            else "—"
        )
        summary_lines.append(
            f"| {case_id} | {category} | {pt} | {base} | {lt_str} | {rm_str} | {ax_str} |\n"
        )

    summary_path = args.output_dir / "summary.md"
    summary_path.write_text("".join(summary_lines))
    print(f"Summary: {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
