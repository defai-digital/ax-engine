# Flash Next 32K MTP safety control on M5 Max

MacBook Pro M5 Max 128 GB, macOS 27.2.1, internal NVMe SSD,
Rust 1.99.1, and MLX 0.32.4. The test uses the same signed `9df17f26a`
canonical singleton verifier, exact prompts, and pinned resident MXFP4
pack as the [M2 boundary control](../flash-next-mtp-context-boundary-m2-20261123/README.md).
The launcher verifies all 49 publisher members and the derived
manifest before and after the test. All 62 pressure samples are normal.

| Workload | Input tokens | Output tokens | Verifier decisions | Accepted drafts | Rejected drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen lookup | 29,774 | 64 | 36 | 27 | 9 |
| Fixed workload near the context cap | 32,640 | 128 | 77 | 50 | 27 |

The second case reaches 32,768 combined tokens. Every accepted draft
matches the host argmax on the verifier's own state. The audit checks
all 190 positions after the two first tokens. Both M5 paths emit the
same IDs for all 192 generated positions.

Between M2 and M5, 47 of 64 lookup IDs and 116 of 128 fixed-window IDs
differ. The differences begin at zero-based output positions 17 and 1,
respectively. Replay retains every differing position. This capture
contains token decisions and does not measure the logit margins.

This is a fixed-budget greedy safety control. It disables the MTP cost
controller and ignores EOS. The independent MTP performance, natural
termination and default-promotion gates remain open.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
