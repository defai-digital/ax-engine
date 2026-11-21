# Flash Next native startup after a selected weight read failure

The test ran the [same Flash Next MXFP4 comparison binary](../flash-next-native-loader-m2-20261121/source.json)
on Mac Studio M2 Ultra 192 GB with MLX 0.32.4 and an isolated clone
of the pinned pack. The injector reported a selected payload read in
`model-00001-of-00025.safetensors` before returning `EIO` from the
MLX positional reader for `model-00012-of-00025.safetensors`. It did
so in the owned process and made no changes to the files.

The server surfaced the typed MLX evaluation error and exited with
status 1. The control observed no listener or completed loader event.
It reaped the process and rebound the original port. This checks
initial startup through the experimental reader.

The same injector passed the existing mmap arm and reached the
expected read error in the isolated lazy reader control. Separate
harness controls fail when the read error occurs before the first
shard, when a listener is reported, or when the loader completes.

- [Native receipt](receipt.json), [error events](error-events.txt) and [startup samples](startup-samples.jsonl)
- [Isolated loader control](isolated-loader-control.json) and [harness controls](harness-decision-control.json)
- [Injector source](shard_read_fault.c) and [file digests](source.json)

Compile the injector as a test-only macOS dynamic library with
`xcrun clang -std=c11 -dynamiclib`. Run the experimental lazy CPU
server against a scratch pack clone with the library in
`DYLD_INSERT_LIBRARIES`. Set `AX_TEST_EARLIER_SHARD` and
`AX_TEST_FAILED_SHARD` to the canonical paths of the two clone members
named in the receipt. Use the resident launch and direct policy from
the [MXFP4 comparison](../flash-next-native-loader-m2-20261121/README.md).
