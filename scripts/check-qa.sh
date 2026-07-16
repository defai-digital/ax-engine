#!/usr/bin/env bash
# Offline QA harness gate (no model weights, no GPU server).
#
# Always-required CI/process check for the qa/ package: bank integrity,
# unit tests, and py_compile of harness modules. GPU matrix / surface
# probes against a live server are separate (see qa/README.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

cd "$ROOT_DIR"

echo "==> QA: py_compile harness modules"
"$PYTHON_BIN" -m py_compile \
  qa/__init__.py \
  qa/prompt_def.py \
  qa/prompts.py \
  qa/question_bank.py \
  qa/checkers.py \
  qa/client.py \
  qa/reporter.py \
  qa/run_qa.py \
  qa/generate_summary.py \
  qa/surface_probes.py \
  qa/embedding_probes.py \
  qa/embedding_bank.py \
  scripts/run_qa_matrix.py \
  scripts/test_qa_checkers.py \
  scripts/test_qa_sampling.py \
  scripts/test_run_qa_matrix.py \
  scripts/test_qa_surface_probes.py \
  scripts/test_qa_embedding_probes.py

echo "==> QA: validate question bank"
"$PYTHON_BIN" qa/run_qa.py --validate-bank

echo "==> QA: unit tests (sampling, checkers, matrix, surface, embedding)"
"$PYTHON_BIN" -m unittest \
  scripts/test_qa_sampling.py \
  scripts/test_qa_checkers.py \
  scripts/test_run_qa_matrix.py \
  scripts/test_qa_surface_probes.py \
  scripts/test_qa_embedding_probes.py \
  -v

echo "==> QA: offline gate OK"
