"""Run counterbalanced MLX path and descriptor load probes."""

import run_load_order

run_load_order.MODES = ("path-lazy", "descriptor-lazy")
run_load_order.main()
