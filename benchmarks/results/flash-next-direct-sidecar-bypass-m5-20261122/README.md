# Flash Next direct sidecar bypass on M5 Max

Four fresh processes ran a pinned MXFP4 pack, revision
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`, on a MacBook Pro
M5 Max 128 GB, internal APFS SSD, macOS 27.2.1 and MLX 0.32.4.
Each used release-server build `88e2598cd` with Rust 1.99.1.
The standard resident profile admitted one request and 32,768
total context tokens. Only explicit MTP sidecar loading differed.
All 49 publisher files were hashed before and after the campaign.

| Arm | Ready | Sidecar phase | MLX active at ready |
| --- | ---: | ---: | ---: |
| Mapped, short first | 72.187 s | 4.630 s | 68.673 GiB |
| Bypassed, short first | 40.102 s | 0.000003 s | 63.816 GiB |
| Bypassed, long first | 40.193 s | 0.000001 s | 63.816 GiB |
| Mapped, long first | 44.208 s | 5.064 s | 68.673 GiB |

Each startup passed 14 HTTP cases and supplied 448 exact generated
IDs. Both loaders kept MTP off with zero verifier steps and zero
attachment errors. Health, the 32K completion boundary and requested
context overflow matched. Both orders met the 90% CPU-idle floor
before and after testing, and pressure remained normal in all
observed samples.

Explicit bypass removes the 4.857 GiB of live MLX storage and the
measured sidecar phase. Trunk loading varies by 25.48 s in the first
pair. The complete readiness difference includes that variance.
Both bypassed arms finish under 120 seconds with these cloned inputs.

The predeclared 3% warmed TTFT band fails. With a short prompt first,
TTFT rises by 8.26% at 2,048 tokens and 4.04% at 8,192 tokens.
Both lengths improve in the reverse order. Keep mapped sidecar
loading for ordinary startup. The explicit bypass has a measured
memory benefit under the pinned direct envelope. The campaign
does not supply the fresh-delivery qualification receipt.

Run `python3 -B replay.py` for the full receipt, exact output IDs,
loading traces, memory-pressure samples and timing verdict. The
archive includes the tested driver, scrubbed logs, publisher
inventory and source/build receipt. To rerun on the device, build
the pinned release-server source and `loader_footprint.c` against the
admitted MLX toolchain, then follow the archived launcher with a
fresh output and scratch location on the pack's APFS volume.
