# Flash Next Required MTP runner cohort on M2 Ultra

Mac Studio, Apple M2 Ultra, 192 GiB unified memory, internal SSD, macOS
27.2.1, MLX and MLX Metal 0.32.4. The signed test executable builds from
`1d49fd2ea` and [the recorded test patch](source.patch.gz) with Rust 1.99.1.
Pack revision `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` matches all 49
publisher files and the derived manifest before and after the run.

The native runner uses the production `Disabled` and `Required` session
policies, the default cost-controller settings, a 2,048-token runner
prefill quantum and the frozen 105-item installed-QA token inputs. It
runs greedy text up to each original budget. This exercise includes an
experimental 29,774-token lookup under Required and measures functional
behavior of the runner.

- All 105 cases stop at their first EOS and release request state.
- Disabled and Required emit the same 953 token IDs. The audit reconciles
  all 848 positions after the first token of each answer.
- The host argmax over the verifier's actual correction row agrees with
  all 366 accepted drafts. The runner records 482 decisions and 116
  rejected drafts. The 29,774-token lookup accepts both of its drafts.
- Every case exercises the verifier. The route counters record zero step
  errors, cost parks and park resumes.

This is M2 functional evidence for the pinned pack. The primary-tier
replay and MTP-S/P/D gates remain open. The record supplies no latency
or throughput comparison.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
