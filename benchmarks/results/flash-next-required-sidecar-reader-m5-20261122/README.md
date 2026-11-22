# Flash Next Required MTP sidecar reader on M5 Max

A MacBook Pro M5 Max 128 GB with internal SSD ran both readers on
macOS 27.2.1 and MLX 0.32.4. One Rust 1.99.1 release-server executable
opened the pinned MXFP4 pack, revision
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`, in four fresh processes.
The experiment holds the resident trunk reader fixed and selects the
sidecar reader for each arm. The canonical Required MTP verifier served
one request at a time with an 8,224-token limit. Each arm used a new
APFS clone and verified 14 requests, including the 8,192 plus 32 token
boundary. All 49 publisher members were hashed before and after.

| Arm | Ready | Resident trunk | Sidecar | First request TTFT |
| --- | ---: | ---: | ---: | ---: |
| Mmap, short first | 70.176 s | 56.468 s | 4.168 s | 0.607 s |
| Positional, short first | 42.108 s | 31.053 s | 0.528 s | 1.626 s |
| Positional, long first | 40.159 s | 30.284 s | 0.530 s | 10.185 s |
| Mmap, long first | 46.167 s | 32.319 s | 4.950 s | 10.193 s |

The positional reader reduced the isolated sidecar phase by 3.64 s
and 4.42 s. Resident-trunk timing varied independently by up to
25.41 s within a pair, so the ready-time difference cannot be
attributed to the sidecar alone. All arms passed the 90% CPU-idle
floor before and after requests. All 100 memory-pressure samples were
normal. Health and overflow matched, all 448 IDs per arm were
identical, and each arm recorded 279 canonical verification steps
and zero route errors.

The short-first pair misses the predeclared 3% warmed TTFT band at
512 and 8,192 prompt tokens, with increases of 7.44% and 4.90%.
The long-first pair fits the band at every measured length. Keep
mapped loading. The scoped experiment provides no promotion of
Required MTP, direct 32K readiness, or release qualification.

## Replay

Run `python3 -B replay.py` from this directory to recheck the raw
startup traces, sampler readings, request IDs, verifier counts and
timing bands. The archive omits the executable and footprint dylib.
Their tested digests, source patch, launch driver and inputs are
included. For a device repeat, apply `reader.patch` to
`c456d9ae3`, build `ax-engine-server` under the release-server profile
with Rust 1.99.1 and MLX 0.32.4, and compile the footprint probe
from `loader_footprint.c`. Use the archived driver with a fresh
output path and scratch directory on the pack's APFS volume.
