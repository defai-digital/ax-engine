# Flash Next affine8 PLE row parity

The published MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`
keeps all 128 PLE table shards in affine 8-bit group 32. Its packed
weights use U32 and the scales and biases use BF16. Five shards place
the three tensor planes in more than one safetensors file.

The [probe](../../../scripts/probe_flash_next_affine8_ple.py) compared
bounded byte reads and MLX dequantization with the ordinary MLX-load,
gather and dequantize route. It ran on the CPU on an M2 Ultra 192 GB,
macOS 27.2.1, with MLX 0.32.4 and the pack on an external SSD.
Each shard contributed first, last and interior rows with a repeated
out-of-order lookup. The bounded path read 92,160 payload bytes for
those selected rows. Pack hash verification and the comparison route
use their own reads.

- All 102,400 compared BF16 values match bit for bit across the
  128 shards.
- Computing the affine formula in FP32 and casting to BF16 changed
  62,092 values. The maximum absolute difference was 0.000244140625.
- A bounded file control rejected negative and out-of-range indices,
  truncated rows, a wrong file size and a mismatching publisher hash.

The [receipt](rows.json.gz) retains per-shard row selections, output
digests, numerical differences and every consumed publisher member
hash. All 14 consumed files passed the publisher hash check again
after the probe. The pinned MLX-VLM reference offers external PLE
storage for affine Q4 and NVFP4. The row results establish the
dequantization primitive for a separate affine8 diagnostic.
Full-graph token parity, process footprint, the frozen M5 comparison
and release qualification require their own evidence.

## Metal comparison

The probe repeated the same rows on Metal with the workspace MLX
0.32.4 environment and with isolated MLX and MLX Metal 0.32.2, the
versions used by the pinned reference. Both matched the ordinary
loaded route across all 128 shards. The two Metal environments
produced identical output digests for all sampled shards. The CPU
digest differed from Metal on all sampled shards. The FP32 formula
matched Metal on these rows and differed from the CPU calculation.
Storage-only comparisons need to use the graph's original MLX
backend for dequantization.

The [MLX 0.32.4](metal-0.32.4.json.gz) and
[MLX 0.32.2](metal-0.32.2.json.gz) receipts retain the sampled
rows, both output digests and the verified pack members. All 14
consumed files matched the publisher hashes after both runs.
