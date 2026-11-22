# Flash Next sidecar admission and Required MTP startup

Mac Studio M2 Ultra 192 GB observations use MLX 0.32.4, Rust
1.99.1, and MXFP4 revision `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
This is functional and local CLI evidence. M5 release qualification
and throughput remain separate.

The `88e2598cd` build served seven HTTP requests with the Flash Next
sidecar attached and seven with it explicitly bypassed. The direct
responses match after dropping the API creation timestamp and the
prefill and decode timing fields. The remaining request and usage
fields agree. Both arms report zero verifier steps and zero attachment
failures. Model-based MTP is inactive at initialization in both arms.
Required MTP with the bypass enabled rejected activation in 58.852 s
after loading the trunk.

The `9b5349f62` build rejects the same conflict before native startup
when opening a fresh session without shared weights. The native CLI
returned `MlxMtpRequiredButUnavailable` in 0.538 s. The existing
`AX_NO_SPEC` preflight returned that error in 0.298 s. Neither log
shows the runner construction, wire-up, or payload markers. The
preflight samples were taken on the software-gate Mac Studio of the
same hardware class. These durations are observations rather than
a same-machine startup timing comparison. The preflight verified all
49 file sizes and the selected runtime metadata hashes against the
pinned publisher inventory.

Run `python3 -B replay.py` for the saved receipt checks. To rerun the
negative controls, use `check_preflight.py --help` and supply the
exact release-server build, native pack, and publisher inventory.
