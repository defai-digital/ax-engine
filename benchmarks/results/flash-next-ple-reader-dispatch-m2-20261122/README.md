# Flash Next native PLE reader dispatch on M2 Ultra

The test uses the pinned Flash Next MXFP4 publisher pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` on a Mac Studio M2 Ultra
with 192 GB, external SSD, macOS 27.2.1, Rust 1.99.1 and MLX 0.32.4.
All 49 publisher member hashes and the matching native manifest were
verified. An isolated Rust test invokes the production row staging
method with the actual checkpoint hash metadata. The loader opens the
128 table shards and reads only the requested rows.

TensorFold keeps parked host workers for cold gathers and uses the
caller for already requested pages. MLX's positional loader also
reuses workers for independent destinations. The AX baseline creates
scoped workers per gather. With 48 reads and a 32-thread setting, it
partitions a singleton decode stage into 24 parts, 23 of them spawned.
The test measures the existing one- and 32-thread settings in
`1, 32, 32, 1` process order.

Each process performs 288 singleton decode stages from three frozen
greedy responses and four 512-token prefill stages from a frozen long
prompt. It then repeats each row sequence in the same process. In all
four processes, the returned staging bytes and read volumes match
exactly, covering 4,608 decode rows and 32,768 prefill rows per pass.
Pressure stays normal before and after every arm.

| Repeated native staging | 1 configured thread, p50 | 32 configured threads, p50 |
| --- | ---: | ---: |
| One decode token | 0.0282 ms | 0.3422–0.3508 ms |
| 512-token prefill | 17.2703–17.4144 ms | 30.5559–31.2629 ms |

The timer includes staging allocation, row assignment, dispatch, file
reads and duplicate scatter. Hashing occurs outside the timer. The
initial pass follows publisher hashing, and the repeated pass reuses
the same input pages. The test supplies an M2 functional and staging
signal. The [raw samples](observed-m2.json.gz) and [replay](comparison.json)
report both passes and both orders.

The corrected M5 run is queued after the owned ordinary-reference
recovery. A parked-worker comparison needs the same read plan and
cache conditioning, a cold control with measured page occupancy,
and concurrent demand versus speculative prefetch checks. Demand
progress independent of an unrelated prefetch is an existing contract.

The [probe patch](probe.patch.gz), [frozen workload](workload.json.gz),
and [driver](run_probe.py) bind the experiment to its exact build.
Twenty focused native row-reader tests passed. Run
`shasum -a 256 -c SHA256SUMS` and `python3 -B replay.py` from this
directory to verify the result.
