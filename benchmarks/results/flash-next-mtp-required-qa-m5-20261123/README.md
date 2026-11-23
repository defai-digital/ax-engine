# Flash Next Required MTP runner cohort on M5 Max

MacBook Pro M5 Max 128 GiB, internal NVMe SSD, macOS 27.2.1,
MLX and MLX Metal 0.32.4, Rust 1.99.1. This run uses the same signed
runner test and frozen 105-item input cohort as the
[M2 Ultra audit](../flash-next-mtp-required-qa-m2-20261123/README.md).
All 49 publisher members and the derived manifest for
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` verify before and after.

- Disabled and Required emit the same 948 token IDs across all 105
  cases. Both routes stop at first EOS in every case.
- The verifier records 480 decisions, with 363 acceptances and 117
  rejections. The host argmax agrees with every accepted draft and
  the replay reconciles all 843 positions after the first token.
- The 29,774-token lookup accepts both drafts. Route counters report
  zero errors, cost parks, and restarts. All 152 sampled pressure levels
  are normal, with peak sampled runner RSS of 73.36 GiB.

M5 and M2 direct outputs agree in 102 of 105 cases. The same three
cases also differ across hardware under Required. Within each hardware
run, Required matches its own Disabled control in all 105 cases.
The replay lists the affected cases and verifies exact trace accounting.

This is primary-tier greedy text functional evidence for the explicit
Required route. MTP performance and default promotion stay open.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
