# Flash Next sidecar descriptor reader on M5 Max

The ordinary 32K, one-request resident MXFP4 profile ran on a MacBook
Pro M5 Max 128 GB with an internal SSD, macOS 27.2.1, Rust 1.99.1 and
MLX 0.32.4. All four arms used the same isolated release-server build
and published pack revision `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
The production trunk reader was held fixed, and MTP was attached and
inactive. Each APFS clone began with one of the selected sidecar pages
resident. All 49 published files were verified before and after the
campaign.

| Arm | Ready | Trunk load | Sidecar load | 512-token TTFT | 8,192-token TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Guarded mmap 1 | 72.193 s | 58.244 s | 3.380 s | 0.569 s | 9.032 s |
| Descriptor lazy 1 | 42.106 s | 30.803 s | 0.440 s | 1.583 s | 11.864 s |
| Descriptor lazy 2 | 38.191 s | 28.771 s | 0.433 s | 0.821 s | 9.707 s |
| Guarded mmap 2 | 42.187 s | 28.513 s | 3.670 s | 0.571 s | 9.515 s |

The isolated positional reader removed roughly three seconds from
loading the same 31 sidecar tensors in both arms. Trunk timing varied
within the same reader, and storage-controller state could vary across
boots. Each startup contributed one request per length. First-token
latency for the 512- and 8,192-token cases was higher in both descriptor
arms. Keep the production reader pending a repeat that separates the
first request from subsequent serving.

All four runs matched 128 greedy IDs over 512-, 2,048-, 8,192- and
32,736-token prompts. Each also matched the 96 retained IDs from the
prior native matrix. Health passed before and after requests, and
overflow returned the expected 400. All 89 resource samples showed
normal pressure. The maximum observed lifetime footprint was 80.25 GiB.

## Startup failure controls

Truncation before startup failed manifest validation before listening
with either MTP policy. With an intact payload and an invalid
head-normalization layout, required MTP exited before listening and
disabled MTP served all 32 expected direct IDs. All four controls
released their ports. The isolated FFI checks separately cover a
recoverable short read during evaluation, independent reader cursor
and descriptor ownership.

## Replay

The [archive](evidence.tar.xz) contains the exact source patch, build
receipt, model manifest, publisher inventory, launch scripts, both
result receipts and all four memory sample logs.
[Checksums](sources.json) bind each byte stream.

```sh
python3 benchmarks/results/flash-next-native-descriptor-reader-m5-20261121/replay.py
```

For a device rerun, extract the archive to a scratch directory and
apply `m5-sidecar.patch` to the exact `build.json` source base. Build
`ax-engine-server` under the release-server profile with its pinned
Rust and MLX versions. Build `loader_footprint.dylib` from
`loader_footprint.c` using the host C compiler. Run
`run_m5_native_sidecar.py`, then `check_fault_policy.py`, with the same
server and pinned pack, the archived inputs, an empty output path for
each and APFS clone storage on the pack's volume.
