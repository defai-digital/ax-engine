# Flash Next measured endurance pilot on M5 Max

The pinned Flash Next MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` served 37 measured requests
on MacBook Pro M5 Max 128 GB with macOS 27.2.1 and internal SSD. The
`b36b7b287` release-server build used Rust 1.99.1 and MLX 0.32.4. The
profile used resident experts under Auto, one active request, MTP
disabled, an effective 32,768-token context limit, and a 2,048-token
scheduler budget.

Both cache-reclamation probes passed during the 103.829-second
preflight. The measured window lasted 1,289.973 seconds on one server
process. All 37 requests succeeded and drained the lifecycle gauges.
There were 26 short, six medium, three shared-prefix, and two long
requests. The server terminated normally with exit code zero.

Across 86 resource samples, process RSS ranged from 71.04 to 75.99 GiB
and host swap use ranged from 6.63 to 6.76 GiB. Swap finished 100 MiB
below the baseline median. The runner reported zero health failures,
drain timeouts, missing KV reports, sample-continuity concerns, or
memory and performance alerts.

The recorded status is `completed`, with verdict `watch`. The
six-minute baseline contained eight short, two medium, one shared-prefix,
and zero long requests. The performance comparison needs at least eight
samples per shape. The pilot establishes the preflight, measurement,
drain and observability path. Longer endurance and same-shape
performance stability remain open.

The [compressed receipt](pilot.json.gz) retains all request and capacity
observations, sampled memory and lifecycle readings, the final verdict,
and the raw artifact hashes.
