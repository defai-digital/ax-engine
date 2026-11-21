# Native Flash Next MXFP4 descriptor-reader comparison

Four single-request native server launches on Mac Studio M2 Ultra 192 GB
compared the guarded, descriptor-bound name-order mmap reader with the
experimental descriptor-owned MLX lazy CPU reader. MLX was 0.32.4,
macOS was 27.2.1 and the pack lived on an external APFS SSD. All launches
used the same experimental lazy CPU trunk (15 shards, 1,942 tensors).
Each started from its own APFS clone with one of 318,256 selected sidecar
OS pages resident. The pack checks covered all 49 published files before
and after the comparison. Reader order reversed between MTP policies.

| MTP policy | Sidecar reader | Spawn to ready | Trunk load | Sidecar load | Peak sampled startup RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Disabled | Guarded mmap | 151.410 s | 80.208 s | 45.190 s | 70.935 GiB |
| Disabled | Descriptor lazy | 46.539 s | 28.496 s | 2.172 s | 68.981 GiB |
| Required | Descriptor lazy | 101.220 s | 81.617 s | 2.170 s | 65.907 GiB |
| Required | Guarded mmap | 131.535 s | 87.616 s | 26.439 s | 73.053 GiB |

Both reader pairs matched all 96 greedy output IDs over 512-, 2,048- and
8,192-token prompts. The sidecar was available under both policies.
Disabled MTP ran zero verifier steps. Each required arm ran 56 verifier
steps and accepted 37 draft tokens. All four arms recorded zero route
errors. Startup samples showed normal host pressure.

The sidecar timer covers materialization of the same 31 selected BF16
tensors. Each arm supplies a single timing. Trunk timing also varied,
especially in the disabled pair. Storage-controller cache state was uncontrolled. The launch and
sampled RSS differences can include effects from both load phases.
These are M2 startup diagnostics. The M5 internal SSD tests and the
separate native M5 serving comparison inform product adoption.

## Startup failure controls

The [fault receipt](fault.json) distinguishes the two startup boundaries.
Truncating a clone's sidecar by two bytes failed manifest validation
before either MTP policy could open a listener. Setting an unsupported
MTP normalization layout on a separate clone preserved the sidecar
payload hash and made attachment fail. Required MTP exited before
listening. Disabled MTP returned the same 32 direct tokens as the
intact pack for the 512-token control, reported the head unavailable
and passed the health check. Each process released its port.

The Rust release-profile FFI tests separately covered lazy-array
evaluation after an in-place short read, descriptor ownership across
rename and a subsequent intact read in the same process.

## Reproduction

[Source and artifact hashes](source.json) bind the exact Rust and MLX
versions, server, [scratch patch](native-reader-lab.patch.gz), scripts,
startup samples and [publisher inventory](publisher-inventory.json).
Apply the decompressed patch to the recorded source base in a scratch
checkout and build package `ax-engine-server` with the `release-server`
profile. The patch adds an Apple-only MLX positional reader and
benchmark-only loader selectors for the scratch server.

Run [the positive harness](run.py) with a pinned local pack, an empty
output path, an APFS scratch directory on the pack's volume and the
[sidecar selection](sidecar-selection.json). Use the Python support
files from the commits in the source receipt. Put the cache helper
`run_load_order.py` in a staging directory. Put the three Python files
from `scripts/` listed in the source receipt and the archived
`watch_flash_next_reference.py` into its `scripts/` subdirectory.
Set `PYTHONPATH` to the staging directory. Pass a build receipt with
the source and binary hashes and run the positive harness followed by
[the fault harness](check_fault_policy.py) with the same binary and
positive receipt. [The verifier](verify_receipts.py) reads
`build.json`, `server.patch`, both fault runner revisions,
`run.py`, `positive/` and `fault/` from the experiment directory.
