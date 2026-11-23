# Flash Next fixed-prefix extension with a content-addressed model view

Mac Studio M2 Ultra 192 GB ran the pinned Flash Next MXFP4 pack on
external SSD with macOS 27.2.1, Rust 1.99.1, and MLX and MLX Metal
0.32.4. The content-addressed view uses hardlinks for the original
publisher payloads and SHA-256-named tensor symlinks. The runner rechecks
all 49 publisher members before and after the comparison. Both arms
use the same signed `release-server` executable and source patch, with
the experimental fixed-prefix switch as the only changed launch input.

The experiment seeds a standalone 2,048-token prefix at physical width
512. The target then supplies a distinct 64-token suffix while another
request is decoding. Each of the three eligible pairs records one hit
and 2,048 physically reused tokens. This includes the interrupted
request and its replacement. The unseeded request and 65-token suffix
control record zero reuse, as do all five switch-off pairs. The original
two-step scheduler geometry remains visible, with the target's first
output on its suffix step and the independent peer present on both
steps. The queue and request counters drain after every pair.

Both processes match 1,346 output IDs, including the interrupted
stream's first token and all peer replies. All resource samples
report normal memory pressure across 1,329 samples. Peak physical
footprint is 88.62 GiB. The replay reconciles the paired output arrays,
seed, resource samples, and raw scheduler observations with the original
[mixed-serving corpus](../flash-next-mxfp4-mixed-serving-m5-20261121/README.md).
The receipt covers functional activation
with this diagnostic model view. The ordinary installed layout and M5
performance use separate runs.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
