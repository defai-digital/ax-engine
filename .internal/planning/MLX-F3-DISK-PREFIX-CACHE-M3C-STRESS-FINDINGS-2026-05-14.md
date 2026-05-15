# F3 M3C — Disk Prefix Cache Stress + Eviction Pressure (2026-05-14)

PRD: `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`
Prior findings:
- `MLX-F3-DISK-PREFIX-CACHE-M3B-LOCKING-FINDINGS-2026-05-14.md`
- `MLX-F3-DISK-PREFIX-CACHE-M4-FINDINGS-2026-05-14.md`

## 1. Status: **M3C LANDED.** Short cache-primitive stress and tight-budget eviction evidence pass.

New reusable harness:

```bash
cargo run -p ax-engine-mlx --bin disk-prefix-cache-stress -- \
  --output benchmarks/results/disk-prefix-cache-stress/2026-05-14-m3b-stress.json
```

Artifact:
`benchmarks/results/disk-prefix-cache-stress/2026-05-14-m3b-stress.json`.

## 2. Stress shape

The harness exercises the real `DiskPrefixCache` implementation, not
a mock:

- 4 worker processes;
- 24 insert + get iterations per worker;
- 8 overlapping keys, so writers race on the same canonical entries;
- zero corruption load failures;
- zero read misses;
- zero worker failures.

The same artifact also includes a tight-budget eviction-pressure
section:

- `max_entries = 2`;
- 5 sequential inserts;
- 3 evictions observed;
- newest two entries survived;
- oldest entry was evicted.

This is intentionally a short cache-primitive stress. It validates
the advisory-lock, atomic-rename, checksum, and eviction paths without
loading four full model-serving processes. A longer AX-serving soak is
still a promotion decision, not a missing runtime primitive.

## 3. Tests

- `cargo test -p ax-engine-mlx --lib disk_prefix_cache --quiet`
  passed: 16 / 16 disk-prefix-cache tests.
- `python3 -m json.tool benchmarks/results/disk-prefix-cache-stress/2026-05-14-m3b-stress.json`
  passed.
- `cargo run -p ax-engine-mlx --bin disk-prefix-cache-stress -- --output ...`
  passed and wrote a PASS artifact.
- `cargo clippy -p ax-engine-mlx --all-targets -- -D warnings`
  passed.

## 4. Remaining blockers

- full AX-serving soak / promotion decision;
- M5 runtime/docs promotion review.
