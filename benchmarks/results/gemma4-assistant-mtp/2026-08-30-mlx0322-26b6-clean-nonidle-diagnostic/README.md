# Gemma 4 26B 6-bit clean non-idle diagnostic

This directory is diagnostic evidence, not a Tier 2 certificate or publication
benchmark. The run used a clean AX Engine tree but intentionally omitted the
idle-host admission gates because periodic host synchronization and macOS
indexing prevented an idle measurement window.

## Scope

- Model: `AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP`
- Local snapshot revision: `a279773d3eecc75d317ec7049bc80bd4a1ec4da2`
- AX Engine: 7.2.0 at `0d779e3ad77337404e0effe908d2e71ca536473d`
- Build state: `git_tracked_dirty=false`
- MLX: 0.32.2 from the pinned Python 3.12 wheel environment
- Workload: four `long_code` prompts, 1,000 generated tokens, two warmups and
  five measured repetitions per prompt
- Controls: greedy sampling, recurrent assistant depth 2, n-gram stacking off,
  15-second repetition cooldown, and 10-second inter-case cooldown

The Hub download metadata and a separate AXQuant integration smoke bind the
local snapshot to the revision above. The benchmark wrapper artifacts record
the repository ID and local path, but not the immutable revision; they are not
standalone revision-binding evidence.

## Result

- Direct aggregate decode: 66.776 tok/s
- Assistant-MTP aggregate decode: 90.399 tok/s
- Aggregate speedup: 1.354x
- Paired exactness: 20/20 token sequences equal, with 1,000 tokens in every arm
- Per-pair speedup: 1.290x minimum, 1.353x median, 1.391x maximum
- Assistant drafts: 14,260 proposed, 12,865 accepted (90.22%)
- Assistant validation/depth: validated at depth 2 in all 20 measured trials
- N-gram proposals and accepts: zero

The raw artifacts retain the observed non-idle conditions. The direct window
started at one-minute load 25.479 and ended at 31.857; the MTP window started at
34.191 and ended at 25.310. These conditions disqualify the throughput numbers
from certification even though all exactness and acceleration checks passed.

An idle-gated rerun with one-minute load at most 4.0 and top-process CPU at most
50% remains required. The wider Gemma Tier 2 matrix also remains open.
