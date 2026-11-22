# Flash Next ordinary reference headroom on M5 Max

The signed ordinary MLX-VLM run reached 121.557 GiB of tracked MLX
allocation on a MacBook Pro M5 Max with 128 GiB of unified memory,
leaving a 6.443 GiB arithmetic margin. Its first three QA rows reached
about 117.7 GiB. The latest ordinary 27.2.1 attempts stopped before
completing the matrix. The pinned MLX-VLM file-backed PLE path
supports affine Q4/group32 and NVFP4/group16. The publisher pack has
128 affine 8-bit/group32 PLE shards. AX's bounded affine8 reference
diagnostic has matching tokens and grades under a separate execution
profile.

The recovery queue uses the original MLX-VLM 0.7.0rc0 source, MLX
0.32.2, tokenizer, frozen prompts, and ordinary execution settings. M5
bootstrap verified the pinned language module and all 13 staged source
and workload fingerprints. During the ongoing native workload the
device reported normal system memory pressure. A separate passive
sample recorded 12.969 GiB visible RSS for the largest virtualization
process and 13.007 GiB across both virtualization-related processes. These are process RSS and
allocator measurements with different accounting scopes. They inform
recovery scheduling without establishing how much physical memory a
later reference run will require.

The queued recovery waits for the owned native and reader runs to
finish. It uses normal pressure, the signed allocator margin, the
largest visible background RSS and the existing 90% idle floor to
select a retry window for up to 30 minutes. It checks capacity again
after creating the verified local pack view and after the four-case
heartbeat. The heartbeat checks original output IDs, text, stopping
and progress under normal sampled pressure. If admitted, the complete
ordinary runner uses the current native matrix's prompt digests and
its own unchanged before-and-after source and inventory checks. The
monitor stops only the owned reference process at critical pressure
or after a prolonged lack of progress.

The [bootstrap](boot-preflight.json) and [decision replay](comparison.json)
cover staging and the capacity observation. The complete reference QA
and timing evidence is pending. The launcher archive retains frozen
inputs and local orchestration, with the signed reference anchored in
its original qualification directory.

Run `shasum -a 256 -c SHA256SUMS` and `python3 -B replay.py` from this
directory to verify the archived inputs and capacity decision.
