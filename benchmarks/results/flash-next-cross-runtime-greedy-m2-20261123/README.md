# Flash Next greedy divergence across hardware and reference paths

The [M5 natural-load archive](../flash-next-current-natural-load-m5-20261123/README.md)
freezes the signed AX server, source patch, token inputs, and single-lane
output oracle. The M2 replay uses the same signed binary, mapped resident
profile, library digest, and one-slot scheduler settings, with MTP,
generic n-gram acceleration and prefix reuse disabled. It runs on Mac
Studio M2 Ultra 192 GiB, macOS 27.2.1, with MLX and MLX Metal 0.32.4.
Two repetitions of each native case produce identical IDs. All 87 AX
pressure samples are normal and all five counters drain.

Both peer probes use the same local MLX 0.32.4 runtime on M2. oMLX
`5eeec7757` uses its declared MLX-VLM dependency `138d7617` and original
mmap PLE path. The bounded diagnostic uses the upstream trunk at
`6102cb4a` and the existing AX affine8 positional reader. The bounded
probe uses MLX 0.32.4 for this comparison. The approved ordinary
reference has its own frozen runtime and profile.

Each value is the number of common initial IDs in a 16-token greedy
trace. The peers compare against native AX on the same M2.

| Comparison | 512 input tokens | 2,048 input tokens | 8,192 input tokens |
| --- | ---: | ---: | ---: |
| AX M2 and AX M5 | 16 | 6 | 1 |
| oMLX, 1,024-token chunks | 2 | 6 | 0 |
| Bounded MLX-VLM, 1,024-token chunks | 2 | 6 | 1 |

For the 8,192-token oMLX case, increasing the chunk from 1,024 to
2,048 changes the first choice from ID 70330 to 5525. The latter matches
the first AX token on both hardware tiers. The 1,024-token run puts
AX's first choice second, by 0.3125 logit units. Both oMLX processes
reproduce the observed token paths, rankings and first-split margins.
At 512 and 2,048 tokens the two peers diverge from native AX M2 at the
same position, with peer-specific margins. The receipts include each
peer's scores at the first differing decision, when the preceding
generated IDs still agree.

Changing forward geometry can affect greedy output within one runtime,
and the same AX binary can differ across hardware tiers. Keep the
hardware, MLX release, and prefill cuts explicit in comparisons.
The scope covers three frozen prompts and provides no release verdict.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B compare.py | cmp comparison.json -
```

For live replay, extract the linked natural-load archive into a scratch
directory and supply the signed binary named by its build receipt as
`bundle/ax-engine-server`. Run `run_native.py BUNDLE PACK OUT` from the
M2 workspace using the recorded MLX library. From the repository root,
run `trace_peers.py omlx PACK NATIVE_RESULT MLX_VLM_SOURCE` with the two
oMLX source trees on `PYTHONPATH`. For the bounded path use
`trace_peers.py bounded PACK NATIVE_RESULT MLX_VLM_SOURCE` with the
older source tree on `PYTHONPATH`. Both Python replays use the dependencies
in the [oMLX trunk smoke](../flash-next-omlx-trunk-m2-20261123/README.md)
and an offline model environment.
