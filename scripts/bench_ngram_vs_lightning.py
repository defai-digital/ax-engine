#!/usr/bin/env python3
"""Compare n-gram acceleration: ax-engine (server) vs mlx_lm paths.

Three measurement paths run on the same model:

  AX ENGINE PATH
    Start ax-engine-server twice (direct + ngram_accel), send the prompt
    suite over HTTP, record actual tok/s and n-gram accept rate from
    server telemetry.

  LIGHTNING N-GRAM PATH
    Load the model via mlx_lm, run actual n-gram speculative decoding:
    at each decode step, get draft tokens from the same n-gram algorithm
    as lightning-mlx (lookup_drafts_with_pending + adaptive_k), verify
    them in one model forward pass (temperature=0.6 acceptance, same
    sampling as ax-engine), and trim rejected tokens from the KV cache.
    Reports real tok/s with speculation overhead and real acceptance rate.
    Non-trimmable caches fall back to oracle path.

    This is the primary n-gram-to-n-gram comparison: if lightning accept
    rate is also low, the model itself has no n-gram opportunity.

  MLX_LM ORACLE PATH
    Load the model via mlx_lm, run standard generate to get the full
    output token sequence (no speculation), then simulate the n-gram
    algorithm post-hoc.  The oracle rate is the theoretical upper-bound
    on what ANY prompt-lookup n-gram implementation can achieve.

VERDICT
    A model is flagged "no n-gram opportunity" when:
      - lightning n-gram accept rate on high-repeat prompts < NO_OPPORTUNITY_LIGHTNING_THRESHOLD
        (confirmed by oracle hit rate < NO_OPPORTUNITY_ORACLE_THRESHOLD)
    A model is flagged "ax under-performing" when:
      - lightning accept rate is high but ax accept rate is materially lower

Usage:
    # Full run (ax-engine + lightning + oracle)
    python3 scripts/bench_ngram_vs_lightning.py \\
        --model-dir /path/to/model \\
        --output-dir benchmarks/results/ngram-compare/$(date +%F)-<model>

    # ax-engine path only
    python3 scripts/bench_ngram_vs_lightning.py \\
        --model-dir /path/to/model \\
        --skip-oracle --skip-lightning \\
        --output-dir ...

    # Smoke run (1 warmup + 2 reps, low cooldown)
    python3 scripts/bench_ngram_vs_lightning.py \\
        --model-dir /path/to/model \\
        --repetitions 2 --warmup 1 --cooldown 5 \\
        --output-dir ...

Prerequisites:
    cargo build -p ax-engine-server --release
    pip install mlx-lm  # for --lightning and --oracle paths
"""
from __future__ import annotations

import argparse
import gc
import http.client
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AX_ENGINE_SERVER = REPO_ROOT / "target" / "release" / "ax-engine-server"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Below this lightning real accept rate on high-repeat prompts → "no n-gram opportunity"
NO_OPPORTUNITY_LIGHTNING_THRESHOLD = 0.10

# Below this oracle first-hit rate on high-repeat prompts → "no n-gram opportunity" (fallback)
NO_OPPORTUNITY_ORACLE_THRESHOLD = 0.10

# Below this ax accept rate relative to lightning → "ax under-performing"
AX_UNDERPERFORM_RELATIVE_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Prompts — high/med/low repeat matching lightning-mlx benchmark categories
# ---------------------------------------------------------------------------

PROMPTS: list[tuple[str, str, str]] = [
    # (id, category, prompt)
    (
        "getter_setter",
        "high_repeat",
        "Generate a Python class called UserProfile with 10 getter/setter method pairs "
        "for fields: name, age, email, phone, address, city, state, zip_code, country, timezone. "
        "Each getter returns self._field, each setter validates and assigns. Include type hints. "
        "Output ONLY Python code.",
    ),
    (
        "json_array",
        "high_repeat",
        "Generate a JSON array with 20 user objects. Each object has: "
        "id (int), name (string), email (string), role (admin/user/editor/viewer), "
        "active (boolean), created_at (ISO date string). Output ONLY valid JSON.",
    ),
    (
        "sql_inserts",
        "high_repeat",
        "Generate 20 SQL INSERT INTO products (id, name, price, category, stock, sku) VALUES "
        "statements with realistic product data. Output ONLY SQL.",
    ),
    (
        "markdown_table",
        "med_repeat",
        "Write a markdown table of 15 programming languages with columns: "
        "Name, Year Created, Paradigm, Typing, Primary Use, Speed Rating. "
        "Output ONLY the markdown table.",
    ),
    (
        "csv_data",
        "med_repeat",
        "Generate CSV with header and 20 rows. Columns: employee_id, name, department, "
        "salary, hire_date, manager, location. Use realistic data. Output ONLY CSV.",
    ),
    (
        "creative_story",
        "low_repeat",
        "Write a short creative story (about 200 words) about a robot discovering "
        "music for the first time. Be vivid and original.",
    ),
    (
        "explain_tcp",
        "low_repeat",
        "Explain how TCP/IP congestion control works in 200 words. "
        "Be technical but clear. Cover slow start, AIMD, and fast recovery.",
    ),
]

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
MAX_TOKENS = 512
DEFAULT_WARMUP = 2
DEFAULT_REPS = 5
DEFAULT_COOLDOWN = 10.0
SERVER_PORT_BASE = 58200

# ---------------------------------------------------------------------------
# Inline n-gram drafter — lightning-mlx algorithm, self-contained
#
# Implements lookup_drafts_with_pending + adaptive_k from:
#   lightning-mlx/vllm_mlx/speculative/ngram_drafter.py
# without any external dependency.
# ---------------------------------------------------------------------------

class _NgramOracle:
    """Post-hoc n-gram oracle using lightning-mlx matching semantics.

    Feed it a token sequence, then call accept_rate() to get the fraction
    of output tokens that the n-gram drafter predicted correctly.  Used only
    for analysis — no actual model inference.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        num_draft_tokens: int = 4,
        min_occurrences: int = 1,
        adaptive_k: bool = True,
    ) -> None:
        self.ngram_size = ngram_size
        self.num_draft_tokens = num_draft_tokens
        self.min_occurrences = min_occurrences
        self.adaptive_k = adaptive_k
        self._history: list[int] = []
        self._index: dict[tuple[int, ...], list[int]] = defaultdict(list)

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
        """Return draft list given the next token is `pending_token`."""
        history = self._history
        n = self.ngram_size
        K = self.num_draft_tokens
        if len(history) + 1 < n:
            return []
        query = tuple(history[-(n - 1):]) + (int(pending_token),) if n > 1 else (int(pending_token),)
        positions = self._index.get(query, [])
        if not positions:
            return []
        current_start = len(history) + 1 - n
        continuations = []
        for start in positions:
            if start == current_start:
                continue
            cont_begin = start + n
            cont_end = min(cont_begin + K, len(history))
            cont = tuple(history[cont_begin:cont_end])
            if cont:
                continuations.append(cont)
        if not continuations:
            return []
        counter = Counter(continuations)
        most_common, freq = counter.most_common(1)[0]
        if freq < self.min_occurrences:
            return []
        K_max = K
        if self.adaptive_k:
            K_max = max(1, min(K, freq + 1))
        return list(most_common[:K_max])

    def simulate(self, prompt_tokens: list[int], output_tokens: list[int]) -> dict:
        """Simulate the n-gram drafter on a known output sequence.

        Mirrors the lightning-mlx call convention: predict_next(t) is called
        with t NOT yet in history, then t is added.  This means the n-gram
        query is (history[-n+1:], t), which is the pattern "given context,
        t was just emitted — what comes next?".

        first_token_hit_rate: fraction of output positions where the drafter's
          depth-0 prediction matches the actual next token.
        draft_coverage: fraction of positions where the drafter proposed any draft.
        token_accept_rate: accepted draft tokens / total drafted tokens (greedy).
        """
        self.feed_many(prompt_tokens)
        drafted = 0
        accepted = 0
        first_hit = 0
        steps_with_draft = 0

        for i, actual in enumerate(output_tokens):
            # Call predict_next(actual) BEFORE feeding it — actual is the
            # "pending" token that hasn't been added to history yet, so
            # the query is (history[-n+1:], actual) without duplication.
            if i < len(output_tokens) - 1:
                drafts = self.predict_next(actual)
                if drafts:
                    steps_with_draft += 1
                    drafted += len(drafts)
                    remaining = output_tokens[i + 1:]
                    for d_idx, d_tok in enumerate(drafts):
                        if d_idx < len(remaining) and d_tok == remaining[d_idx]:
                            accepted += 1
                        else:
                            break
                    if drafts[0] == output_tokens[i + 1]:
                        first_hit += 1
            self.feed(actual)

        total_steps = max(len(output_tokens) - 1, 1)
        return {
            "drafted": drafted,
            "accepted": accepted,
            "steps_with_draft": steps_with_draft,
            "first_token_hit_rate": first_hit / total_steps,
            "token_accept_rate": accepted / drafted if drafted > 0 else 0.0,
            "draft_coverage": steps_with_draft / total_steps,
        }


# ---------------------------------------------------------------------------
# MLX_LM oracle path
# ---------------------------------------------------------------------------

def _load_mlx_model(model_dir: Path):
    """Load model via mlx_lm, return (model, tokenizer)."""
    from mlx_lm.utils import load_model as _load_model, load_tokenizer
    model, _ = _load_model(model_dir, strict=False)
    tokenizer = load_tokenizer(model_dir, {})
    return model, tokenizer


def _make_prompt_str(tokenizer, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"User: {prompt_text}\nAssistant:"


def _run_oracle_prompt(
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int,
    reps: int,
    warmup: int,
    cooldown: float,
) -> dict:
    """Run one prompt N times via mlx_lm, compute oracle stats each time.

    Returns median tok/s, ttft, and oracle accept metrics.
    """
    import mlx.core as mx
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.generate import generate_step as mlx_generate_step

    prompt_str = _make_prompt_str(tokenizer, prompt_text)
    prompt_ids = tokenizer.encode(prompt_str)

    measured_tps: list[float] = []
    measured_ttft: list[float] = []
    oracle_first_hit: list[float] = []
    oracle_coverage: list[float] = []
    oracle_token_ar: list[float] = []
    trimmable: bool | None = None

    all_runs = warmup + reps
    for rep in range(all_runs):
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

        prompt_arr = mx.array(prompt_ids, mx.uint32)
        kv = mlx_cache.make_prompt_cache(model)

        if trimmable is None:
            trimmable = bool(kv[0].is_trimmable()) if kv else False

        t_start = time.perf_counter()
        t_first = None
        output_ids: list[int] = []

        import mlx.core as _mx
        _T = SAMPLING["temperature"]
        _tp = SAMPLING["top_p"]
        _tk = SAMPLING["top_k"]
        try:
            from mlx_lm.sample_utils import make_sampler as _ms
            bench_sampler = _ms(_T, top_p=_tp, top_k=_tk)
        except (ImportError, TypeError):
            bench_sampler = lambda logits: _mx.random.categorical(logits / _T)
        for (tid, _), _ in zip(
            mlx_generate_step(prompt_arr, model, prompt_cache=kv, sampler=bench_sampler),
            range(max_tokens),
        ):
            if t_first is None:
                t_first = time.perf_counter()
            t = int(tid)
            output_ids.append(t)
            if t == tokenizer.eos_token_id:
                break

        t_end = time.perf_counter()
        ttft = (t_first - t_start) if t_first else 0.0
        decode_s = t_end - (t_first or t_start)
        tps = len(output_ids) / decode_s if decode_s > 0 and output_ids else 0.0

        if rep >= warmup:
            measured_tps.append(tps)
            measured_ttft.append(ttft)

            # Oracle simulation — independent of trimmable flag
            oracle = _NgramOracle(ngram_size=3, num_draft_tokens=4, min_occurrences=1, adaptive_k=True)
            sim = oracle.simulate(prompt_ids, output_ids)
            oracle_first_hit.append(sim["first_token_hit_rate"])
            oracle_coverage.append(sim["draft_coverage"])
            oracle_token_ar.append(sim["token_accept_rate"])

        if rep < all_runs - 1 and cooldown > 0:
            time.sleep(cooldown)

    def _med(xs):
        return statistics.median(xs) if xs else 0.0

    return {
        "tok_s_median": _med(measured_tps),
        "ttft_median": _med(measured_ttft),
        "trimmable": trimmable,
        "oracle_first_token_hit_rate": _med(oracle_first_hit),
        "oracle_draft_coverage": _med(oracle_coverage),
        "oracle_token_accept_rate": _med(oracle_token_ar),
    }


def run_oracle_path(
    model_dir: Path,
    reps: int,
    warmup: int,
    cooldown: float,
) -> list[dict]:
    """Load model once, run all prompts, return per-prompt oracle results."""
    print("\n[mlx_lm oracle path]", flush=True)
    try:
        model, tokenizer = _load_mlx_model(model_dir)
    except Exception as e:
        print(f"  ERROR loading model: {e}", flush=True)
        return []

    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    results = []
    for pid, category, prompt in PROMPTS:
        print(f"  {pid} [{category}] ...", end="", flush=True)
        try:
            r = _run_oracle_prompt(model, tokenizer, prompt, MAX_TOKENS, reps, warmup, cooldown)
            r.update({"prompt_id": pid, "category": category})
            results.append(r)
            hit_str = f" oracle_hit={r['oracle_first_token_hit_rate']:.1%}"
            cov_str = f" cov={r['oracle_draft_coverage']:.1%}"
            trim_str = " (non-trimmable)" if not r["trimmable"] else ""
            print(f" {r['tok_s_median']:.1f} tok/s{hit_str}{cov_str}{trim_str}", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            results.append({
                "prompt_id": pid,
                "category": category,
                "error": str(e),
            })

    del model, tokenizer
    try:
        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Lightning n-gram speculative decode path
# ---------------------------------------------------------------------------

def _run_lightning_speculative_prompt(
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int,
    reps: int,
    warmup: int,
    cooldown: float,
    num_draft_tokens: int = 4,
    ngram_size: int = 3,
) -> dict:
    """Real n-gram speculative decode with lightning-mlx algorithm (temperature=0.6 acceptance).

    Implements lookup_drafts_with_pending + adaptive_k (same as _NgramOracle.predict_next)
    as an actual live speculative decode loop:
      1. Get draft tokens from the n-gram index.
      2. Pass [curr_token, *drafts] to the model in one call.
      3. Temperature-sample each position (temp=0.6/top_p=0.95/top_k=20, same as ax-engine).
         Accept draft token if the sampled token matches; use sampled token as correction otherwise.
      4. Trim rejected tokens from the KV cache via trim_prompt_cache.

    Falls back to non-speculative decode when KV cache is not trimmable, in which
    case tok/s reflects standard generation without any n-gram benefit.
    """
    import mlx.core as mx
    from mlx_lm.models import cache as mlx_cache

    _T = SAMPLING["temperature"]
    _tp = SAMPLING["top_p"]
    _tk = SAMPLING["top_k"]
    try:
        from mlx_lm.sample_utils import make_sampler as _ms
        _sampler = _ms(_T, top_p=_tp, top_k=_tk)
    except (ImportError, TypeError):
        _sampler = lambda logits: mx.random.categorical(logits / _T)

    def _sample_token(logits_1d):
        return int(_sampler(logits_1d).item())

    prompt_str = _make_prompt_str(tokenizer, prompt_text)
    prompt_ids = tokenizer.encode(prompt_str)
    eos_id = tokenizer.eos_token_id

    kv_test = mlx_cache.make_prompt_cache(model)
    try:
        trimmable = bool(mlx_cache.can_trim_prompt_cache(kv_test))
    except AttributeError:
        trimmable = all(getattr(c, "is_trimmable", lambda: False)() for c in kv_test)
    del kv_test

    measured_tps: list[float] = []
    ngram_drafted_total = 0
    ngram_accepted_total = 0

    all_runs = warmup + reps
    for rep in range(all_runs):
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

        kv = mlx_cache.make_prompt_cache(model)

        # Prefill: process prompt[:-1] into KV cache, leave last token as input
        y = mx.array(prompt_ids, mx.uint32)
        while y.size > 1:
            n_proc = min(512, y.size - 1)
            model(y[:n_proc][None], cache=kv)
            mx.eval([c.state for c in kv])
            mx.clear_cache()
            y = y[n_proc:]

        # Process last prompt token → logit for first output token
        logits = model(y[None], cache=kv)
        logits = logits[:, -1, :]
        mx.eval(logits)

        curr_token = _sample_token(logits.squeeze(0))
        t_first = time.perf_counter()
        output_ids = [curr_token]

        drafter = _NgramOracle(
            ngram_size=ngram_size,
            num_draft_tokens=num_draft_tokens,
            min_occurrences=1,
            adaptive_k=True,
        )
        drafter.feed_many(prompt_ids)
        # Invariant: curr_token is NOT yet in drafter history at the top of each
        # loop iteration, so predict_next(curr_token) sees it as the pending token
        # (query = history[-n+1:] + [curr_token]) without duplication.

        rep_drafted = 0
        rep_accepted = 0

        for _ in range(max_tokens - 1):
            if curr_token == eos_id or len(output_ids) >= max_tokens:
                break

            drafts = drafter.predict_next(curr_token) if trimmable else []

            if not drafts:
                pending = curr_token
                logits = model(mx.array([curr_token], mx.uint32)[None], cache=kv)
                logits = logits[:, -1, :]
                mx.eval(logits)
                curr_token = _sample_token(logits.squeeze(0))
                drafter.feed(pending)
                output_ids.append(curr_token)
            else:
                k = len(drafts)
                verify_tokens = mx.array([curr_token] + list(drafts), mx.uint32)
                logits = model(verify_tokens[None], cache=kv)
                logits = logits[0, :k + 1, :]
                mx.eval(logits)

                n_accept = 0
                for i in range(k):
                    sampled = _sample_token(logits[i])
                    if sampled == drafts[i]:
                        n_accept += 1
                    else:
                        break

                next_token = _sample_token(logits[n_accept])
                trim_amount = k - n_accept
                if trim_amount > 0:
                    mlx_cache.trim_prompt_cache(kv, trim_amount)

                rep_drafted += k
                rep_accepted += n_accept

                drafter.feed(curr_token)
                for i in range(n_accept):
                    drafter.feed(drafts[i])
                    output_ids.append(drafts[i])
                # next_token becomes the new curr_token; feed deferred to next iteration
                output_ids.append(next_token)
                curr_token = next_token

                # Stop if EOS landed in the accepted window
                if eos_id in output_ids[-(n_accept + 1):]:
                    break

        t_end = time.perf_counter()
        decode_s = t_end - t_first
        tps = len(output_ids) / decode_s if decode_s > 0 and output_ids else 0.0

        if rep >= warmup:
            measured_tps.append(tps)
            ngram_drafted_total += rep_drafted
            ngram_accepted_total += rep_accepted

        if rep < all_runs - 1 and cooldown > 0:
            time.sleep(cooldown)

    return {
        "tok_s_median": statistics.median(measured_tps) if measured_tps else 0.0,
        "lightning_accept_rate": ngram_accepted_total / ngram_drafted_total if ngram_drafted_total > 0 else None,
        "lightning_drafted": ngram_drafted_total,
        "lightning_accepted": ngram_accepted_total,
        "trimmable": trimmable,
    }


def run_lightning_path(
    model_dir: Path,
    reps: int,
    warmup: int,
    cooldown: float,
) -> list[dict]:
    """Load model once, run all prompts with real lightning n-gram speculative decode."""
    print("\n[lightning n-gram path]", flush=True)
    try:
        model, tokenizer = _load_mlx_model(model_dir)
    except Exception as e:
        print(f"  ERROR loading model: {e}", flush=True)
        return []

    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    results = []
    for pid, category, prompt in PROMPTS:
        print(f"  {pid} [{category}] ...", end="", flush=True)
        try:
            r = _run_lightning_speculative_prompt(model, tokenizer, prompt, MAX_TOKENS, reps, warmup, cooldown)
            r.update({"prompt_id": pid, "category": category})
            results.append(r)
            ar_str = (
                f" lightning_ar={r['lightning_accept_rate']:.1%}"
                if r["lightning_accept_rate"] is not None
                else " (no drafts)"
            )
            trim_str = " (non-trimmable, spec disabled)" if not r.get("trimmable", True) else ""
            print(f" {r['tok_s_median']:.1f} tok/s{ar_str}{trim_str}", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            results.append({"prompt_id": pid, "category": category, "error": str(e)})

    del model, tokenizer
    try:
        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# ax-engine server path
# ---------------------------------------------------------------------------

def _free_port(base: int) -> int:
    for port in range(base, base + 100):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        try:
            s.connect(("127.0.0.1", port))
        except OSError:
            return port
        finally:
            s.close()
    raise RuntimeError("No free port found")


def _wait_for_server(port: int, timeout: int = 180) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/v1/models")
            if conn.getresponse().status == 200:
                return True
        except OSError:
            pass
        time.sleep(1)
    return False


def _start_server(model_dir: Path, port: int, env_extra: dict) -> subprocess.Popen:
    env = {**os.environ, **env_extra}
    cmd = [
        str(AX_ENGINE_SERVER),
        "--mlx-model-artifacts-dir", str(model_dir),
        "--port", str(port),
        "--mlx",
        "--support-tier", "mlx-certified",
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        proc.kill()


_ax_tokenizer_cache: dict = {}


def _get_ax_tokenizer(model_dir: Path):
    """Load tokenizer for ax-engine path (cached per model dir)."""
    key = str(model_dir)
    if key not in _ax_tokenizer_cache:
        from mlx_lm.utils import load_tokenizer
        _ax_tokenizer_cache[key] = load_tokenizer(str(model_dir), {})
    return _ax_tokenizer_cache[key]


def _generate_stream_ax(
    port: int,
    token_ids: list[int],
    max_tokens: int,
) -> tuple[float, int, dict]:
    """POST to /v1/generate/stream; returns (decode_s, output_tokens, ngram_telem).

    decode_s is wall time from first output token to last output token.
    ngram_telem keys: ax_ngram_draft_tokens, ax_ngram_accepted_tokens (cumulative
    from the last step's crossover_decisions).
    """
    body = json.dumps({
        "input_tokens": token_ids,
        "max_output_tokens": max_tokens,
        "sampling": {
            "temperature": SAMPLING["temperature"],
            "top_p": SAMPLING["top_p"],
            "top_k": SAMPLING["top_k"],
        },
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request(
        "POST", "/v1/generate/stream",
        body=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        resp.read()
        return 0.0, 0, {}

    t_first: float | None = None
    t_last: float | None = None
    output_len = 0
    ngram_telem: dict = {}

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
        if not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:])
        except Exception:
            continue

        req = obj.get("request", {})
        curr_len = req.get("output_len", 0)
        if curr_len > 0:
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
        output_len = max(output_len, curr_len)

        # Extract cumulative n-gram telemetry from step crossover_decisions
        step_cross = obj.get("step", {}).get("route", {}).get("crossover_decisions", {})
        for k in ("ax_ngram_draft_tokens", "ax_ngram_accepted_tokens"):
            if k in step_cross:
                ngram_telem[k] = step_cross[k]

    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.001
    return decode_s, output_len, ngram_telem


def _run_ax_prompt(
    port: int,
    token_ids: list[int],
    reps: int,
    warmup: int,
    cooldown: float,
) -> dict:
    all_runs = warmup + reps
    measured_tps: list[float] = []
    ngram_drafted = 0
    ngram_accepted = 0

    for rep in range(all_runs):
        decode_s, n_tok, telem = _generate_stream_ax(port, token_ids, MAX_TOKENS)

        if rep >= warmup:
            tps = n_tok / decode_s if decode_s > 0 else 0.0
            measured_tps.append(tps)
            ngram_drafted += telem.get("ax_ngram_draft_tokens", 0)
            ngram_accepted += telem.get("ax_ngram_accepted_tokens", 0)

        if rep < all_runs - 1 and cooldown > 0:
            time.sleep(cooldown)

    return {
        "tok_s_median": statistics.median(measured_tps) if measured_tps else 0.0,
        "ngram_accept_rate": ngram_accepted / ngram_drafted if ngram_drafted > 0 else None,
        "ngram_drafted": ngram_drafted,
        "ngram_accepted": ngram_accepted,
    }


def run_ax_path(
    model_dir: Path,
    reps: int,
    warmup: int,
    cooldown: float,
) -> tuple[list[dict], list[dict]]:
    """Run ax-engine direct + ngram_accel. Returns (direct_results, ngram_results).

    Uses /v1/generate/stream (the repo-owned MLX native endpoint) with pre-tokenized
    token IDs.  Tokenization uses mlx_lm which is also needed by the lightning/oracle
    paths, so this path shares the same dependency.
    """
    tokenizer = _get_ax_tokenizer(model_dir)
    tokenized = {
        pid: tokenizer.encode(_make_prompt_str(tokenizer, prompt))
        for pid, _, prompt in PROMPTS
    }

    port = _free_port(SERVER_PORT_BASE)
    direct_results: list[dict] = []
    ngram_results: list[dict] = []

    for mode_label, env_extra, results_list in [
        ("direct", {"AX_NO_SPEC": "1"}, direct_results),
        ("ngram_accel", {}, ngram_results),
    ]:
        print(f"\n[ax-engine {mode_label}]", flush=True)
        proc = _start_server(model_dir, port, env_extra)
        try:
            if not _wait_for_server(port):
                print(f"  ERROR: server did not start", flush=True)
                continue
            for pid, category, _ in PROMPTS:
                print(f"  {pid} [{category}] ...", end="", flush=True)
                try:
                    r = _run_ax_prompt(port, tokenized[pid], reps, warmup, cooldown)
                    r.update({"prompt_id": pid, "category": category})
                    results_list.append(r)
                    ar_str = (
                        f" ngram_ar={r['ngram_accept_rate']:.1%}"
                        if r["ngram_accept_rate"] is not None
                        else ""
                    )
                    print(f" {r['tok_s_median']:.1f} tok/s{ar_str}", flush=True)
                except Exception as e:
                    print(f" ERROR: {e}", flush=True)
                    results_list.append({"prompt_id": pid, "category": category, "error": str(e)})
        finally:
            _stop_server(proc)
            time.sleep(5)

    return direct_results, ngram_results


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def _cat_med(results: list[dict], category: str, key: str = "tok_s_median") -> float | None:
    vals = [r[key] for r in results if r.get("category") == category and key in r and not r.get("error")]
    return statistics.median(vals) if vals else None


def _overall_med(results: list[dict], key: str = "tok_s_median") -> float | None:
    vals = [r[key] for r in results if key in r and not r.get("error")]
    return statistics.median(vals) if vals else None


def _mean_rate(results: list[dict], key: str) -> float | None:
    vals = [r[key] for r in results if r.get(key) is not None and not r.get("error")]
    return sum(vals) / len(vals) if vals else None


def _verdict(
    ax_direct: list[dict],
    ax_ngram: list[dict],
    oracle: list[dict],
    lightning: list[dict],
) -> list[str]:
    lines = []
    high_oracle = _cat_med(oracle, "high_repeat", "oracle_first_token_hit_rate")
    high_ax_ar = _mean_rate(
        [r for r in ax_ngram if r.get("category") == "high_repeat"],
        "ngram_accept_rate",
    )
    high_lightning_ar = _mean_rate(
        [r for r in lightning if r.get("category") == "high_repeat"],
        "lightning_accept_rate",
    )
    high_direct_tps = _cat_med(ax_direct, "high_repeat")
    high_ngram_tps = _cat_med(ax_ngram, "high_repeat")
    high_lightning_tps = _cat_med(lightning, "high_repeat")
    ax_speedup = high_ngram_tps / high_direct_tps if (high_ngram_tps and high_direct_tps) else None
    lightning_speedup = high_lightning_tps / high_direct_tps if (high_lightning_tps and high_direct_tps) else None

    # Primary signal: lightning real accept rate (n-gram vs n-gram comparison)
    if high_lightning_ar is not None:
        if high_lightning_ar < NO_OPPORTUNITY_LIGHTNING_THRESHOLD:
            oracle_confirm = (
                f" Oracle first-hit also confirms: {high_oracle:.1%}."
                if high_oracle is not None else ""
            )
            lines.append(
                f"NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is "
                f"{high_lightning_ar:.1%} (< {NO_OPPORTUNITY_LIGHTNING_THRESHOLD:.0%} threshold). "
                f"The model's output lacks sufficient repeated token patterns — n-gram cannot help "
                f"regardless of implementation.{oracle_confirm}"
            )
        elif high_ax_ar is not None:
            gap = high_lightning_ar - high_ax_ar
            if high_lightning_ar > 0 and gap > AX_UNDERPERFORM_RELATIVE_THRESHOLD * high_lightning_ar:
                lines.append(
                    f"AX UNDER-PERFORMING vs lightning: lightning accept={high_lightning_ar:.1%} "
                    f"but ax accept={high_ax_ar:.1%} (gap={gap:.1%}). "
                    f"The model has n-gram potential but ax-engine is not capturing it fully."
                )
            else:
                lines.append(
                    f"AX N-GRAM ON PAR WITH LIGHTNING: lightning accept={high_lightning_ar:.1%}, "
                    f"ax accept={high_ax_ar:.1%} (gap={gap:.1%})."
                )
        else:
            # lightning finds opportunity but ax made zero draft attempts on high-repeat prompts.
            # Most common cause: only_in_think gate blocks structured output (code/SQL/JSON)
            # that arrives after </think> on thinking models.
            lines.append(
                f"AX NO DRAFT ATTEMPTS on high-repeat (lightning accept={high_lightning_ar:.1%}): "
                f"ax-engine made no n-gram draft attempts despite model having potential. "
                f"Likely cause: only_in_think gate is blocking structured output after </think>."
            )
    elif high_oracle is not None:
        # Fallback to oracle-only verdict when lightning path was skipped
        if high_oracle < NO_OPPORTUNITY_ORACLE_THRESHOLD:
            lines.append(
                f"NO N-GRAM OPPORTUNITY (oracle): first-token hit rate on high-repeat prompts is "
                f"{high_oracle:.1%} (< {NO_OPPORTUNITY_ORACLE_THRESHOLD:.0%}). "
                f"The model does not produce sufficient repeated token patterns."
            )
        elif high_ax_ar is not None:
            gap = high_oracle - high_ax_ar
            if gap > AX_UNDERPERFORM_RELATIVE_THRESHOLD * high_oracle:
                lines.append(
                    f"AX UNDER-PERFORMING (oracle baseline): oracle hit={high_oracle:.1%} "
                    f"but ax accept={high_ax_ar:.1%} (gap={gap:.1%})."
                )
            else:
                lines.append(
                    f"AX PERFORMING WELL: oracle hit={high_oracle:.1%}, ax accept={high_ax_ar:.1%} "
                    f"(gap={gap:.1%})."
                )

    if ax_speedup is not None:
        lines.append(f"ax-engine n-gram speedup on high-repeat: {ax_speedup:.2f}×")
    if lightning_speedup is not None:
        lines.append(f"lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): {lightning_speedup:.2f}×")
    return lines


def write_summary(
    model_dir: Path,
    ax_direct: list[dict],
    ax_ngram: list[dict],
    oracle: list[dict],
    lightning: list[dict],
    output_dir: Path,
) -> None:
    def fmt(v):
        return f"{v:.1f}" if v is not None else "—"
    def fmtp(v):
        return f"{v:.1%}" if v is not None else "—"
    def fmtx(a, b):
        if a and b and b > 0:
            return f"{a / b:.2f}×"
        return "—"

    lines = []
    lines.append("# N-gram Opportunity Benchmark")
    lines.append("")
    lines.append(f"Date: {date.today().isoformat()}  ")
    lines.append(f"Model: `{model_dir}`  ")
    lines.append(f"Prompts: {len(PROMPTS)} ({sum(1 for _,c,_ in PROMPTS if c=='high_repeat')} high, "
                 f"{sum(1 for _,c,_ in PROMPTS if c=='med_repeat')} med, "
                 f"{sum(1 for _,c,_ in PROMPTS if c=='low_repeat')} low repeat)  ")
    lines.append(f"Max tokens: {MAX_TOKENS}  ")
    lines.append("")
    lines.append("> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.")
    lines.append("> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.")
    lines.append("> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for v in _verdict(ax_direct, ax_ngram, oracle, lightning):
        lines.append(f"> {v}")
    lines.append("")
    lines.append("## Per-prompt results")
    lines.append("")
    lines.append("| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    oracle_by_id = {r["prompt_id"]: r for r in oracle if "prompt_id" in r}
    direct_by_id = {r["prompt_id"]: r for r in ax_direct if "prompt_id" in r}
    ngram_by_id = {r["prompt_id"]: r for r in ax_ngram if "prompt_id" in r}
    lightning_by_id = {r["prompt_id"]: r for r in lightning if "prompt_id" in r}

    for pid, cat, _ in PROMPTS:
        dr = direct_by_id.get(pid, {})
        nr = ngram_by_id.get(pid, {})
        lr = lightning_by_id.get(pid, {})
        orr = oracle_by_id.get(pid, {})
        lines.append(
            f"| {pid} | {cat[:3]} "
            f"| {fmt(dr.get('tok_s_median'))} "
            f"| {fmt(nr.get('tok_s_median'))} "
            f"| {fmtx(nr.get('tok_s_median'), dr.get('tok_s_median'))} "
            f"| {fmtp(nr.get('ngram_accept_rate'))} "
            f"| {fmt(lr.get('tok_s_median'))} "
            f"| {fmtp(lr.get('lightning_accept_rate'))} "
            f"| {fmtp(orr.get('oracle_first_token_hit_rate'))} "
            f"| {fmtp(orr.get('oracle_draft_coverage'))} |"
        )

    lines.append("")
    lines.append("## By category (medians)")
    lines.append("")
    lines.append("| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat in ("high_repeat", "med_repeat", "low_repeat"):
        d_tps = _cat_med(ax_direct, cat)
        n_tps = _cat_med(ax_ngram, cat)
        l_tps = _cat_med(lightning, cat)
        n_ar = _mean_rate([r for r in ax_ngram if r.get("category") == cat], "ngram_accept_rate")
        l_ar = _mean_rate([r for r in lightning if r.get("category") == cat], "lightning_accept_rate")
        o_hit = _cat_med([r for r in oracle if r.get("category") == cat], "oracle_first_token_hit_rate")
        lines.append(
            f"| {cat} | {fmt(d_tps)} | {fmt(n_tps)} | {fmtx(n_tps, d_tps)} | {fmtp(n_ar)}"
            f" | {fmt(l_tps)} | {fmtx(l_tps, d_tps)} | {fmtp(l_ar)} | {fmtp(o_hit)} |"
        )

    lines.append("")
    lines.append("## Column definitions")
    lines.append("")
    lines.append("- **ax direct**: ax-engine with n-gram disabled, baseline throughput (temperature=0.6)")
    lines.append("- **ax ngram**: ax-engine with n-gram acceleration enabled (temperature=0.6)")
    lines.append("- **ax accept**: fraction of ax-engine draft tokens accepted by the verifier")
    lines.append("- **lightning**: mlx_lm model with real n-gram speculative decode (temperature=0.6 acceptance)")
    lines.append("- **lightning accept**: fraction of n-gram draft tokens accepted (temperature=0.6; comparable to ax accept)")
    lines.append("- **lightning speedup**: lightning n-gram tok/s ÷ ax direct tok/s (different sampling, indicative only)")
    lines.append("- **oracle hit**: fraction of output tokens the n-gram drafter would predict correctly")
    lines.append("  (post-hoc temperature-sampled simulation; theoretical upper bound for any n-gram implementation at temp=0.6)")
    lines.append("- **oracle cov**: fraction of decode steps where the n-gram drafter proposed any draft")
    lines.append("")
    lines.append("**Interpretation:** If lightning accept < 10% on high-repeat prompts, the model itself")
    lines.append("has no n-gram opportunity (oracle hit will also be low). If lightning accept is high but")
    lines.append("ax accept is low, ax-engine is under-performing on n-gram speculation for this model.")

    out = output_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {out}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="Model directory (used by both ax-engine and mlx_lm)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for JSON artifact and summary.md")
    parser.add_argument("--skip-oracle", action="store_true",
                        help="Skip the mlx_lm post-hoc oracle path")
    parser.add_argument("--skip-lightning", action="store_true",
                        help="Skip the real lightning n-gram speculative decode path")
    parser.add_argument("--skip-ax", action="store_true",
                        help="Skip the ax-engine path")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    args = parser.parse_args()

    need_mlx = not args.skip_oracle or not args.skip_lightning
    if not args.skip_ax and not AX_ENGINE_SERVER.exists():
        sys.exit(
            f"ERROR: ax-engine-server not found at {AX_ENGINE_SERVER}\n"
            f"Run: cargo build -p ax-engine-server --release"
        )
    if need_mlx:
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            sys.exit(
                "ERROR: mlx_lm not installed. Run: pip install mlx-lm\n"
                "       (or use --skip-oracle --skip-lightning to run ax-engine only)"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ax_direct: list[dict] = []
    ax_ngram: list[dict] = []
    oracle: list[dict] = []
    lightning: list[dict] = []

    if not args.skip_ax:
        ax_direct, ax_ngram = run_ax_path(
            args.model_dir, args.repetitions, args.warmup, args.cooldown
        )

    if not args.skip_lightning:
        lightning = run_lightning_path(
            args.model_dir, args.repetitions, args.warmup, args.cooldown
        )

    if not args.skip_oracle:
        oracle = run_oracle_path(
            args.model_dir, args.repetitions, args.warmup, args.cooldown
        )

    artifact = {
        "schema": "ax.bench.ngram_vs_lightning.v2",
        "date": date.today().isoformat(),
        "model_dir": str(args.model_dir),
        "sampling": SAMPLING,
        "max_tokens": MAX_TOKENS,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "cooldown": args.cooldown,
        "prompts": [{"id": p, "category": c} for p, c, _ in PROMPTS],
        "ax_direct": ax_direct,
        "ax_ngram": ax_ngram,
        "lightning": lightning,
        "oracle": oracle,
    }
    artifact_path = args.output_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"Artifact: {artifact_path}", flush=True)

    write_summary(args.model_dir, ax_direct, ax_ngram, oracle, lightning, args.output_dir)

    # Print verdict to stdout
    print("\n=== VERDICT ===", flush=True)
    for v in _verdict(ax_direct, ax_ngram, oracle, lightning):
        print(f"  {v}", flush=True)


if __name__ == "__main__":
    main()
