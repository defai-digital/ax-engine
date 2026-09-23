#!/usr/bin/env bash
# Multi-model review harness for the Qwen 3.8 Flash Next observability change.
#
# Runs each named reviewer CLI over the *same* reviewed revision, records the
# real process exit status, a terminal verdict line and the reviewed-source
# digest in a per-reviewer receipt, and can re-verify those digests.
#
# Reviewers: glm, qwen (ax-code single-shot), kimi (native CLI), muse
# (isolated throwaway workspace), grok (tools disabled), claude (tools
# disabled, stdin prompt). kimi-code exposes no --tools flag and rejects
# --plan together with --prompt, so it is constrained by the prompt alone
# ("answer from this prompt alone"); the harness asserts afterwards that no
# reviewer changed the tracked working tree.
#
# Usage:
#   scripts/review_flash_next_multimodel.sh --all --require-verdicts --verify-diff
#
# Flags:
#   --all              ensure every reviewer has a valid receipt
#   --require-verdicts fail if any receipt lacks exit 0 + a terminal verdict
#   --verify-diff      fail if any receipt digest != the digest recomputed now;
#                      also enables receipt reuse (see below)
#   --force            always re-invoke the CLIs, even when a receipt is fresh
#   --only <name>      run just one reviewer (repeatable)
#   --baseline <sha>   reviewed baseline commit for the prompt diff
#                      (default: the commit immediately before the Flash Next
#                      fallback work, i.e. the range holding the change)
#
# Receipt reuse: a full six-reviewer run costs ~13 minutes, which exceeds a
# 300 s CI/agent per-command budget. With --verify-diff, a reviewer whose
# receipt already records exit 0, a non-MISSING terminal verdict and a
# reviewed_digest equal to the digest recomputed now is reused instead of
# re-invoked; a missing, failed or stale receipt is always re-run. Pass
# --force to disable reuse and re-invoke every CLI.
#
# Exit 0 only when every requested gate holds. A timeout, empty output, a
# missing terminal verdict or a digest mismatch is a failure, never a skip.
# -e is load-bearing: helper failures (digest, mkdir, tree snapshots) must
# abort instead of recording a bogus-but-consistent receipt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REVIEW_DIR="${REVIEW_DIR:-.internal/reports/flash-next-multimodel-review-20260923}"
# Reviewed revision: the implementation plus its in-scope companions.
REVIEW_PATHS=(
  "crates/ax-engine-mlx/src/runner"
  "crates/ax-engine-server/src/metrics.rs"
  "crates/ax-engine-server/src/tests/metrics.rs"
  "crates/ax-engine-server/src/flash_next_fallback_keys.rs"
  "crates/ax-engine-server/tests/metrics.rs"
  "crates/ax-engine-server/src/app_state.rs"
  "docs/model-certifications/qwen3.8-flash-next.md"
  "scripts/flash_next_peer_bench_plan.json"
  "scripts/check_flash_next_peer_bench_plan.py"
)
REVIEWERS=(glm qwen kimi muse grok claude)
REVIEW_TIMEOUT_SECS="${REVIEW_TIMEOUT_SECS:-420}"
# Reviewed baseline for the prompt diff. Parameterized (environment or
# --baseline) so a later goal can point the harness at its own baseline
# instead of a hardcoded commit; the default is this goal's baseline, so the
# diff sent to reviewers is exactly this goal's change.
REVIEW_BASELINE="${REVIEW_BASELINE:-374a056989e7ea9f309aaff152549dad222ca2be}"

RUN_ALL=0
REQUIRE_VERDICTS=0
VERIFY_DIFF=0
FORCE=0
ONLY=()
usage_error() {
  echo "usage: $0 --all [--require-verdicts] [--verify-diff] [--force] [--only <name>] [--baseline <sha>]" >&2
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in
    --all) RUN_ALL=1 ;;
    --require-verdicts) REQUIRE_VERDICTS=1 ;;
    --verify-diff) VERIFY_DIFF=1 ;;
    --force) FORCE=1 ;;
    --only)
      [ $# -ge 2 ] || { echo "--only requires a value" >&2; usage_error; }
      ONLY+=("$2"); shift ;;
    --baseline)
      [ $# -ge 2 ] || { echo "--baseline requires a value" >&2; usage_error; }
      REVIEW_BASELINE="$2"; shift ;;
    *) echo "unknown flag: $1" >&2; usage_error ;;
  esac
  shift
done

# A bare invocation checks nothing and would print OK; require at least one
# gate so a CI job that forgets its flags fails loudly instead of passing green.
if [ "$RUN_ALL" -eq 0 ] && [ "${#ONLY[@]}" -eq 0 ] && [ "$REQUIRE_VERDICTS" -eq 0 ] && [ "$VERIFY_DIFF" -eq 0 ]; then
  echo "no gates requested: pass --all, --only, --require-verdicts and/or --verify-diff" >&2
  usage_error
fi

if ! git rev-parse --verify --quiet "${REVIEW_BASELINE}^{commit}" >/dev/null; then
  echo "review baseline does not resolve to a commit: $REVIEW_BASELINE" >&2
  exit 2
fi

# ---------------------------------------------------------------- digest ----
# sha256 over sorted "<path>:<file-sha256>" lines for every reviewed file.
reviewed_digest() {
  local out
  out="$(
    for p in "${REVIEW_PATHS[@]}"; do
      if [ -d "$p" ]; then
        find "$p" -type f -name '*.rs' -print0
      elif [ -f "$p" ]; then
        printf '%s\0' "$p"
      else
        # A reviewed path that vanished silently shrinks the digest and lets
        # --verify-diff pass bogus-vs-bogus; fail closed instead.
        printf 'reviewed path is missing: %s\n' "$p" >&2
        return 1
      fi
    done | sort -z | xargs -0 shasum -a 256 | awk '{print $2":"$1}'
  )"
  printf '%s' "$out" | shasum -a 256 | awk '{print $1}'
}

DIGEST="$(reviewed_digest)"
[ -n "$DIGEST" ] || { echo "reviewed digest is empty" >&2; exit 1; }

mkdir -p "$REVIEW_DIR"

# ---------------------------------------------------------------- prompt ----
build_prompt() {
  local digest="$1" prompt_file="$2"
  {
    echo "You are reviewing a bounded engine change in the AX Engine repo."
    echo "Reply in English. Do not use tools; answer from this prompt alone."
    echo
    echo "Reviewed revision digest (sha256 over the reviewed source set): $digest"
    echo
    echo "Objective: harden Qwen 3.8 Flash Next MTP observability and get the"
    echo "peer-comparison contract ready, with product gates left OPEN:"
    echo "MTP-S/MTP-P/MTP-D stay not_assessed, default admission stays"
    echo "fail-closed, MTP eligibility and model arithmetic are unchanged."
    echo "Deferred/out of scope: closing gates, promoting MXFP4 or 6-bit to"
    echo "default, any release, changing the pack or expert-paging policy."
    echo
    echo "What landed (this diff only):"
    echo "- server: the engine already emitted"
    echo "  ax_mlx_flash_next_mtp_attach_failed on every step and nothing consumed"
    echo "  it; app_state now accumulates that key and metrics.rs publishes"
    echo "  ax_engine_flash_next_mtp_attach_failed_total, so the draft head never"
    echo "  attaching is distinguishable from attaching but blocking every step."
    echo "- tooling: scripts/flash_next_peer_bench_plan.json declares the"
    echo "  omlx/MTPLX/ds4 peer comparison (host, pack revision, fixed conditions,"
    echo "  required series, open gates); scripts/check_flash_next_peer_bench_plan.py"
    echo "  validates it, cross-checks each required series against the server"
    echo "  crate, and fails closed under --require-preconditions."
    echo "- docs/comments: the certification record gains the attach-failure and"
    echo "  peer-contract paragraphs, and the prefix_cache.rs draft-cursor sidecar"
    echo "  comment is corrected to describe the already-landed restore path."
    echo
    echo "PRE-EXISTING BASELINE, deliberately NOT in this diff: the"
    echo "FlashNextMtpFallbackReason enum, the seven per-reason direct-fallback"
    echo "route keys and their shared key/name table, and"
    echo "Qwen4ExpDraftCursor::from_prefix_snapshot. crates/ax-engine-mlx/src/runner/mod.rs"
    echo "is unchanged by this increment; do not treat its absence from the diff"
    echo "as a defect."
    echo
    echo "Required output format (final two sections, exactly):"
    echo "FINDINGS:"
    echo "- <severity> <file:line if known> <one-line finding> (or 'none')"
    echo "VERDICT: APPROVE | APPROVE_WITH_CONCERNS | REQUEST_CHANGES"
    echo
    echo "Diff of the reviewed paths versus the goal baseline commit:"
    echo '```diff'
    local diff_text diff_bytes
    diff_text="$(git diff --no-color "$REVIEW_BASELINE" -- "${REVIEW_PATHS[@]}" 2>/dev/null)"
    diff_bytes="$(printf '%s' "$diff_text" | wc -c | tr -d ' ')"
    printf '%s' "$diff_text" | head -c 60000
    if [ "$diff_bytes" -gt 60000 ]; then
      echo
      echo "... [diff TRUNCATED at 60000 of ${diff_bytes} bytes: this packet is incomplete by construction, so an unreviewable tail must be reported as such rather than treated as absent]"
    fi
    echo '```'
  } >"$prompt_file"
}

# -------------------------------------------------------------- reviewers ----
run_reviewer() {
  local name="$1" prompt_file="$2" out_file="$3" err_file="$4"
  case "$name" in
    glm)
      timeout "$REVIEW_TIMEOUT_SECS" ax-code run \
        --model defai-01-ax-trust-com/glm-5.3 --sandbox read-only \
        --title "flash-next multimodel review" \
        "$(cat "$prompt_file")" >"$out_file" 2>"$err_file"
      ;;
    qwen)
      timeout "$REVIEW_TIMEOUT_SECS" ax-code run \
        --model defai-01-ax-trust-com/qwen3.8-max --sandbox read-only \
        --title "flash-next multimodel review" \
        "$(cat "$prompt_file")" >"$out_file" 2>"$err_file"
      ;;
    kimi)
      # No --tools flag and no --plan with --prompt: constrained by the prompt
      # alone. The tracked-tree assertion below is the enforceable guard.
      timeout "$REVIEW_TIMEOUT_SECS" kimi -p "$(cat "$prompt_file")" \
        --output-format text -m kimi-code/k3 >"$out_file" 2>"$err_file"
      ;;
    muse)
      local ws
      ws="$(mktemp -d /tmp/flash-next-muse.XXXXXX)"
      mkdir -p "$ws/workspace"
      cp "$prompt_file" "$ws/prompt.md"
      timeout "$REVIEW_TIMEOUT_SECS" muse exec \
        --workspace "$ws/workspace" --prompt-file "$ws/prompt.md" \
        --approval-mode never --disable-write --disable-shell \
        --no-session-log --max-model-steps 8 >"$out_file" 2>"$err_file"
      rm -rf "$ws"
      ;;
    grok)
      timeout "$REVIEW_TIMEOUT_SECS" grok --prompt-file "$prompt_file" \
        --output-format json --max-turns 1 --no-subagents --tools "" \
        -m grok-4.6 >"$out_file" 2>"$err_file"
      ;;
    claude)
      timeout "$REVIEW_TIMEOUT_SECS" claude -p --output-format json \
        --permission-mode plan --tools "" --model opus \
        <"$prompt_file" >"$out_file" 2>"$err_file"
      ;;
    *) echo "unknown reviewer: $name" >&2; return 2 ;;
  esac
}

receipt_path() { printf '%s/%s.receipt' "$REVIEW_DIR" "$1"; }

write_receipt() {
  local name="$1" status="$2" verdict="$3" out_file="$4" err_file="$5" prompt_file="$6"
  local receipt
  receipt="$(receipt_path "$name")"
  {
    echo "reviewer: $name"
    echo "process_exit: $status"
    echo "verdict: ${verdict:-MISSING}"
    echo "reviewed_digest: $DIGEST"
    echo "prompt_sha256: $(shasum -a 256 "$prompt_file" | awk '{print $1}')"
    echo "stdout_sha256: $(shasum -a 256 "$out_file" | awk '{print $1}')"
    echo "stderr_sha256: $(shasum -a 256 "$err_file" | awk '{print $1}')"
    echo "--- findings (tail) ---"
    # grep exits 1 on no match (empty reviewer output); the receipt must still
    # be written so the failure is visible to --require-verdicts.
    { grep -E '^(FINDINGS:|VERDICT:|- )' "$out_file" 2>/dev/null || true; } | tail -n 20
    echo "--- stderr (tail) ---"
    tail -n 5 "$err_file" 2>/dev/null || true
  } >"$receipt"
}

# A receipt is reusable only when it records a real, successful run of this
# exact reviewed digest.
receipt_is_fresh() {
  local receipt
  receipt="$(receipt_path "$1")"
  [ -f "$receipt" ] || return 1
  local rc vd dg
  rc="$(awk -F': ' '/^process_exit: /{print $2}' "$receipt")"
  vd="$(awk -F': ' '/^verdict: /{print $2}' "$receipt")"
  dg="$(awk -F': ' '/^reviewed_digest: /{print $2}' "$receipt")"
  [ "$rc" = "0" ] && [ -n "$vd" ] && [ "$vd" != "MISSING" ] && [ "$dg" = "$DIGEST" ]
}

extract_verdict() {
  grep -oE 'VERDICT:[[:space:]]*(APPROVE_WITH_CONCERNS|APPROVE|REQUEST_CHANGES)' "$1" 2>/dev/null \
    | tail -n 1 | sed 's/^VERDICT:[[:space:]]*//'
}

fail=0

# Reviewers must not mutate the tracked tree; snapshot it and compare after the
# loop. (.internal/ receipts are git-ignored and do not appear here.)
TREE_BEFORE="$(git status --porcelain --untracked-files=no)"

if [ "$RUN_ALL" -eq 1 ] || [ "${#ONLY[@]}" -gt 0 ]; then
  targets=("${REVIEWERS[@]}")
  if [ "${#ONLY[@]}" -gt 0 ]; then targets=("${ONLY[@]}"); fi
  for name in "${targets[@]}"; do
    if [ "$FORCE" -eq 0 ] && [ "$VERIFY_DIFF" -eq 1 ] && receipt_is_fresh "$name"; then
      echo "==> reviewer: $name (fresh receipt reused; digest $DIGEST)"
      continue
    fi
    prompt_file="$REVIEW_DIR/$name.prompt.md"
    out_file="$REVIEW_DIR/$name.stdout"
    err_file="$REVIEW_DIR/$name.stderr"
    build_prompt "$DIGEST" "$prompt_file"
    echo "==> reviewer: $name"
    # A failing reviewer is recorded, not fatal: capture its status and keep
    # the receipt so --require-verdicts reports the real exit and verdict.
    status=0
    run_reviewer "$name" "$prompt_file" "$out_file" "$err_file" || status=$?
    verdict="$(extract_verdict "$out_file" || true)"
    write_receipt "$name" "$status" "$verdict" "$out_file" "$err_file" "$prompt_file"
    echo "    exit=$status verdict=${verdict:-MISSING}"
    if [ "$status" -ne 0 ] || [ -z "$verdict" ]; then fail=1; fi
  done
fi

if [ "$RUN_ALL" -eq 1 ] || [ "${#ONLY[@]}" -gt 0 ]; then
  TREE_AFTER="$(git status --porcelain --untracked-files=no)"
  if [ "$TREE_BEFORE" != "$TREE_AFTER" ]; then
    echo "FAIL: a reviewer changed the tracked working tree" >&2
    printf 'before:\n%s\nafter:\n%s\n' "$TREE_BEFORE" "$TREE_AFTER" >&2
    fail=1
  fi
fi

# ------------------------------------------------------------- assertions ----
if [ "$REQUIRE_VERDICTS" -eq 1 ]; then
  for name in "${REVIEWERS[@]}"; do
    receipt="$(receipt_path "$name")"
    if [ ! -f "$receipt" ]; then
      echo "MISSING receipt: $name" >&2; fail=1; continue
    fi
    rc="$(awk -F': ' '/^process_exit: /{print $2}' "$receipt")"
    vd="$(awk -F': ' '/^verdict: /{print $2}' "$receipt")"
    if [ "${rc:-x}" != "0" ] || [ -z "$vd" ] || [ "$vd" = "MISSING" ]; then
      echo "reviewer $name failed: exit=${rc:-?} verdict=${vd:-?}" >&2; fail=1
    fi
  done
fi

if [ "$VERIFY_DIFF" -eq 1 ]; then
  for name in "${REVIEWERS[@]}"; do
    receipt="$(receipt_path "$name")"
    [ -f "$receipt" ] || { echo "MISSING receipt for digest check: $name" >&2; fail=1; continue; }
    recorded="$(awk -F': ' '/^reviewed_digest: /{print $2}' "$receipt")"
    if [ "$recorded" != "$DIGEST" ]; then
      echo "digest mismatch for $name: receipt=$recorded now=$DIGEST" >&2; fail=1
    fi
  done
fi

if [ "$fail" -ne 0 ]; then
  echo "review_flash_next_multimodel: FAIL" >&2
  exit 1
fi
echo "review_flash_next_multimodel: OK (digest $DIGEST)"
