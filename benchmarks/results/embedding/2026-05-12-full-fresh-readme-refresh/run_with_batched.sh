#!/usr/bin/env bash
# Re-run all 3 embedding models with --include-batched, then update README.
set -uo pipefail

REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

source .venv/bin/activate

EMB_OUTDIR="benchmarks/results/embedding/2026-05-12-full-fresh-readme-refresh"
EMB_BENCH="scripts/bench_embedding_models.py"
EMB_README_UPDATER="scripts/update_readme_embedding.py"

cleanup_servers() {
    pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
    sleep 2
}

# Clean previous JSONs so updater picks the new ones
rm -rf "$EMB_OUTDIR"/2026-05-12-* "$EMB_OUTDIR"/logs-*.log

run_embedding() {
    local label="$1"
    local model_dir="$2"
    echo ""
    echo "  [embedding] $label  ($(date '+%H:%M:%S'))"
    cleanup_servers
    PYTHONUNBUFFERED=1 python -u "$EMB_BENCH" \
        --model-dir "$model_dir" \
        --model-label "$label" \
        --trials 5 \
        --cooldown 10 \
        --skip-swift \
        --skip-ax-http \
        --include-batched \
        --output-dir "$EMB_OUTDIR" \
        2>&1 | tee "$EMB_OUTDIR/logs-${label}.log"
    sleep 30
}

run_embedding qwen3-embedding-0.6b-8bit   .internal/models/qwen3-embedding-0.6b-8bit
run_embedding qwen3-embedding-4b-4bit     .internal/models/qwen3-embedding-4b-4bit
run_embedding qwen3-embedding-8b-4bit-dwq .internal/models/qwen3-embedding-8b-4bit-dwq

echo ""
echo "  [readme] applying embedding rows (single + batched)"
python "$EMB_README_UPDATER" --results-dir "$EMB_OUTDIR"

echo ""
echo "Done at $(date '+%H:%M:%S')."
