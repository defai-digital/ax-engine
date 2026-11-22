# Flash Next paired positional reader functional control

Apple MLX's pinned file reader issues concurrent positional batches
into allocator-owned target arrays. This diagnostic holds the AX
resident selection, name-order allocations and resulting array type
fixed. For a selected tensor of at least 64 MiB, two simultaneous
positional reads fill disjoint ranges of the same private allocation.
Smaller tensors follow the serial path. Both reads must finish before
array construction, and both modes reject the partial-read fixture.

On Mac Studio M2 Ultra 192 GB, external APFS SSD and MLX 0.32.4,
all 18 arms reproduce their reference weights, Metal checksum, and
allocator size. This covers 266 tensors and 3,220,919,680 unique
published payload bytes. Each of the two shards exercises precisely
three divided tensor reads in the paired arm. The mapped comparison
and independent source checksum also agree. Pressure is normal in
every captured arm. This is functional evidence. The same executable
is staged on the M5 primary hardware for a separate timing decision.

From the repository root or this directory, run `replay.py` with
Python. It verifies the test against the earlier independent
[owned-storage checksums](../flash-next-owned-positional-shards-m2-20261122/README.md)
and binds the probe code, compiler, MLX and workload to the receipt.
For a device rerun, build `direct_weights.cpp` as C++20 against the
pinned MLX 0.32.4 system headers and library. Run the Python driver
with the archived selection, pinned inventory, an exact local pack
and scratch storage on the same APFS volume.
