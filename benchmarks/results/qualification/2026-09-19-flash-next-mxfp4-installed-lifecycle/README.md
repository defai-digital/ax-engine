# Current-source Flash Next installed lifecycle

Result: **PASS for the fourteen-action lifecycle contract only. Product qualification remains open.**

Source `5f583018` and wheel `06c65e80` ran on the MacBook Pro M5 Max 128 GiB
MXFP4/NAS target. Default and required-MTP modes each completed baseline, full
SSE, one/two-token budgets, stop, an unfinished disconnect with an active
producer, and identical recovery. Original paging policy remained in effect;
Flash Next family admission still required the explicit experimental opt-in.
Both servers and the worker exited cleanly. All 47 model payloads and 52
installed runtime files passed the original pre/post integrity checks.

The bundle retains responses, stream events, metrics, logs, execution contract,
preflight, ownership receipts and the independent root verdict. The public
reader reconstructs all fourteen actions from the sanitized records. Private
paths use stable hash labels; private storage-console output has a hash receipt.
Numeric, boolean and null values are preserved, and gzip timestamp is zero.
Original private-record hashes identify the source evidence; they are not
checksums of its sanitized representation. Public artifact hashes are separate
in [summary.json](summary.json).

Run `python verify_evidence.py` and `python -m unittest test_verify_evidence.py`
from this directory. Neither command loads model weights.

The original QA failures and earlier lifecycle failures remain unchanged in
their dated evidence. Current-source full QA, independent numerical coverage,
memory/NAS behavior, throughput and fresh delivery remain separate gates.
MTP-S, MTP-P and MTP-D are all `not_assessed`. This evidence authorizes no release,
acceleration claim or default-MTP promotion.
