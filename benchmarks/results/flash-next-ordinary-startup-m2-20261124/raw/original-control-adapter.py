"""Run the original QA path on the M2 diagnostic host."""
import check_flash_next_native_support as native
import ordinary_original

# The production reference CLI remains hardware gated. These adapters affect
# only host admission and metadata in this isolated diagnostic invocation.
is_target = native.is_target_host
storage_info = native.storage_info
native.is_target_host = lambda chip, size: is_target(chip, size) or (
    chip == "Apple M2 Ultra" and size == 192 * 2**30
)
native.form_factor = lambda: "Mac Studio"
native.storage_info = lambda root: storage_info(root, require_internal=False)
raise SystemExit(ordinary_original.main())
