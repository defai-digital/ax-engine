# Flash Next MXFP4 installed target QA

Collection is complete; **acceptance failed**. This is the installed
`85a2bab0` opt-in runtime on MacBook Pro M5 Max 128 GiB, using default Auto
expert paging and the pinned MXFP4 MTP pack on a NAS over SMB. The installed
runtime and original checkers were not changed during collection.

| Check | Result |
| --- | --- |
| Coverage | 105 original inputs in each of direct and required MTP |
| Quality | 102 passes and three retained failures in each mode |
| Stops and shutdown | 210 normal stops; both servers exit 0 |
| Direct/MTP text identity | 103/105 pairs match; two differences remain |
| Direct/MTP checker identity | All 105 pairs match |
| Long lookup | Both modes return `1734` for the 29,774-token input |
| Integrity | Frozen supervisor verifies 68 installed/pack files and six QA files before and after collection |
| Release | Not qualified; no default-MTP promotion |

The three quality failures are `instruction_alphabet_first`,
`science_gravity_earth`, and `knowledge_water_formula`. Their outputs match
the same-pack functional reference failures. Neither those answers nor their
scores were normalized. The reference separately fails `format_csv_pair`;
neither AX mode has an AX-only checker failure in this cohort. Each AX mode
differs from the reference text on four inputs. See the
[reference evidence](../2026-09-18-flash-next-mxfp4-references/README.md).

Direct/MTP differences occur in `reasoning_cause_effect` and
`reasoning_syllogism_roses`. The first differing character offsets are 339
and 55, respectively. These are decoded-text offsets, **not generated token
positions**. The OpenAI response records do not contain output token IDs or
logit margins, so they cannot establish a near-tie explanation. First-token
and state diagnostics remain required; checker agreement does not close
the exact-output gate.

The raw records retain elapsed times for bounded request completion. They
are not a controlled throughput comparison or a cold-storage measurement:
the integrity scans warm storage caches. The observed 326/437 accepted draft
tokens are QA telemetry, not the independent trained/permuted-head gate.
The frozen runtime health still contains the former Studio SKU label; that
historical metadata is preserved and does not identify the tested hardware.

- `summary.json`: identities, verdicts, raw text differences, scope and hashes.
- `installed-qa.json.gz`: all 210 responses, original checks, metrics and exits.
- `reference-comparison.json.gz`: complete functional comparison with MLX-VLM.
- `integrity.json.gz`: original installed and QA digest inventories.

Private locations are replaced by stable hash labels. Outputs, numerical
values and verdicts are unchanged; gzip timestamps are zero. Original source
hashes are separate from the sanitized artifact hashes. M5 numerical, MTP
state/rollback, lifecycle, fresh delivery, performance and memory gates remain
open. MLX-VLM is not a replacement for the missing `mlx_lm.benchmark` primary
baseline.
