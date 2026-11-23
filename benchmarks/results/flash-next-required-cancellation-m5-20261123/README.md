# Flash Next MTP cancellation and reuse on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX 0.32.4.
The signed executable contains the cancellation counter fix
in `776fe54ce`. It uses the 49 verified MXFP4 publisher files at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. Both processes use the
one-slot 32,768-token resident envelope and select `AlignedBuffered`
automatically. The default arm omits the MTP flag. The Required arm
passes an explicit flag and disables the experimental cost controller
to expose the verifier.

Each mode completes five native actions and three OpenAI completions
actions. Native SSE disconnect and explicit stepwise cancel both
release the request and permit a same-prompt replacement. OpenAI SSE
disconnect also releases the request and preserves usage and text on
replacement. All completed native outputs match the same 64 token IDs,
and all completed OpenAI texts match across both modes. The bounded
prompt is 512 tokens.

| Counter observation | Default verified / accepted | Required verified / accepted |
| --- | ---: | ---: |
| Complete native request | 0 / 0 | 37 / 26 |
| Native SSE disconnect, after drain | 0 / 0 | 5 / 4 |
| Native stepwise cancel, after drain | 0 / 0 | 2 / 1 |
| OpenAI SSE disconnect, while open | 0 / 0 | 2 / 1 |
| OpenAI SSE disconnect, after drain | 0 / 0 | 4 / 3 |
| Complete OpenAI replacement | 0 / 0 | 37 / 26 |

All actions drain the five retained resource gauges. All 88 memory
pressure observations are normal. This is a bounded lifecycle and
same-prompt output control, with broader quality, timing and
MTP-S/P/D tracked separately.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
