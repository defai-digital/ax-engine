"""Rotate the same isolated runner between the two distinct sidecar readers."""

import run_load_order

run_load_order.MODES = ("name-order", "lazy-cpu")
run_load_order.main()
