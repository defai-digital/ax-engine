# Tiel co-residency and bounded competing-memory checks

Both exact Tiel MXFP4 MTP packs passed native-session co-residency in both load orders on MacBook Pro M5 Max 128 GiB. Additional resident allocations of 16 and 32 GiB passed the full concurrent request stage. The 48 GiB attempt triggered the compressor-growth guard and was stopped; it is **not a passed sustained-load result**. No AX Engine runtime defect was reproduced and no runtime tuning was applied.

## Contract

- Runtime source: `730ae4dd79a25465f05398b69e388710429fd7ee`; extension SHA-256 and source hashes are bound in `results.json`. Optimized `release-pyext`, Rust 1.97.1, MLX 0.32.2.
- Exact packs and revisions: [preceding pack audit](../2026-09-19-path/README.md). No model inference on M3.
- One process per load-order campaign. Each model is constructed, used and destroyed on its own fixed worker thread, honoring the Python session/iterator thread-affinity contract. Models load sequentially before overlapping requests begin.
- Same recorded coding input token arrays, greedy sampling, seed zero, 128 output tokens, ignore EOS, prefix reuse disabled, native throughput MTP requested. Two warmups followed by three measured requests per ordinary phase. Three-second GPU-idle interval before each serial request or synchronized pair.
- Both load orders cover isolated baseline, co-resident alternation, synchronized concurrent API requests, canceled-stream recovery and survivor-after-close. API start/end overlap is recorded; this does not prove simultaneous GPU kernel execution.
- Tiel-first additionally covers the memory ladder. Its child commits anonymous deterministic random data in 1 GiB increments and proves residency through RSS. It observes `vm_stat`, `vm.swapusage` and `kern.memorystatus_vm_pressure_level` during requests.
- Guards: at least 16 GiB estimated free+inactive+speculative headroom after the next increment; release for >128 MiB swap growth, >1 GiB physical compressor growth, critical OS pressure, parent EOF or a 180-second child deadline. The headroom sum is an estimate, not a guarantee of reclaimability.
- Wired zero is asserted idempotently between phases. Memory/cache usage is recorded; their caps are not changed to simulate unavailable getters in the admitted MLX wheel. No system memory limit or unrelated process was modified.

## Results

- **72 completed requests**, including warmups and recovery probes, matched each model's isolated token baseline exactly. MTP submissions and depth-three draft activity were observed.
- MLX active allocation stayed near **21.25 GiB** for one model and **42.50 GiB** for two; closing one owner returned it to approximately 21.25 GiB while its peer remained usable. All observed quiescent wired limits were zero.
- **16 / 32 GiB:** six overlapping requests per stage completed while the load was retained; the child then released normally.
- **48 GiB:** the child released after compressor growth exceeded the 1 GiB threshold (observed maximum about 1.82 GiB). One in-flight pair finished correctly; later pairs were not issued. The recovery pair and survivor checks passed. These rows are retained as interrupted-load evidence and excluded from clean-load timing comparisons.
- Recorded OS pressure remained at normal level **1**, and swap stayed **519.75 MiB**. Therefore this is bounded competing-memory evidence, not warning/critical-pressure, OOM or endurance qualification.

Concurrent API requests and the memory-sampling helper change the workload. Raw TTFT/completion/decode observations are included, but three repetitions and differing memory history are insufficient for latency percentiles or a new speed claim. In particular, Tiel isolated/survivor TTFT varied across preliminary runs. Do not substitute these observations for the existing [matched single-model MTPLX comparison](../2026-09-19-wired/README.md).

## Harness regression found and fixed

The initial pressure helper wrote a large final telemetry record into an undrained stdout pipe. After the compressor guard released memory, the child could remain alive while blocked writing that record. Process liveness incorrectly suggested that the pressure allocation was still held. That initial 48 GiB result was excluded.

The final harness writes child telemetry to a regular file, records when release starts, and checks the structured release reason even when the process exits successfully. A clean stage requires every request to end before normal release. The corrected campaign stops issuing pairs when the child exits and never advances after a guard stop. Seven unit checks cover allocation reserves, thresholds, normal exit after a guard, and requests after memory release.

## Validation and limits

The final source-bound verifier passes. Post-run pack file hashes and MTP norms match the preceding campaign. Rust: 3,747 tests passed. Python: 209 passed, 26 skipped, 142 subtests passed. Formatting, focused MLX library Clippy, maturin, scripts, both qualification dry runs and canonical claims passed. Full-workspace Clippy remains failing in unchanged core/SDK tests; this run also reached 239 SDK lint diagnostics that were not printed in the preceding parallel check. No Rust source changed in this campaign.

This does not qualify mixed families, model loading/unloading during active generation, server multi-model routing, reversed-order pressure, arbitrary prompts or long-context/endurance behavior. It changes no defaults or certification status.

```bash
python3 benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-coexistence/test_probe.py
python3 benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-coexistence/verify.py --check-source
```

For hardware reproduction, provide a directory containing both exact pack names and a cases directory with `tiel-cases.json` / `cyber-cases.json`, each a one-element array containing its recorded `token_ids`. Use the matching extension and MLX library, clear inherited `AX_*` / `MTPLX_*` flags, and run one process at a time:

```bash
python probe.py --models MODELS_DIR --cases CASES_DIR --order cyber --output cyber-core.json
python probe.py --models MODELS_DIR --cases CASES_DIR --order tiel --pressure --output tiel-pressure.json
```

The script fails before model load on any host other than M5 Max 128 GiB. Keep child JSONL files alongside the result. Do not disable its guards to reproduce the 48 GiB stop.
