# Flash Next post-load startup trace on M2 Ultra

The diagnostic reused the [native MXFP4 pair's experimental binary](../flash-next-native-loader-m2-20261121/source.json)
and the same pinned pack on the external-SSD M2 Ultra 192 GB host. It
ran the lazy CPU resident loader with a single slot, MTP disabled and
a 32K context. The launch reproduced all 96 greedy output token IDs
from the original pair.

After the resident loader marker, six one-second samples at offsets
0 through 84 seconds placed the native generation thread in the
filtered mmap reader, BF16 array initialization and memory copy. An
open-file observation in that window included the 5.21 GB
`mtp.safetensors` member. The native startup path still attaches the
eligible MTP sidecar when the request policy disables drafting. The
first ordinary kernel warm-up log appeared 102.40 seconds after the
resident loader marker. The sample at 104 seconds showed normal
Flash Next forward work. The HTTP listener appeared after 121.12 seconds.

The [diagnostic receipt](result.json) records the actual sample
offsets, selected frame counts, the launch phases and the exact
binary and preceding parity receipt. Sampling windows identify
where the worker spent the observed time. They do not attribute the
entire post-load interval to any one call. The next loader control
compares all 31 tensors in the same pinned sidecar through three
readers.
