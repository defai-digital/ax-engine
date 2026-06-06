# ADR: Local Serve CLI and MTPLX Conversion Workflow

## Status

Proposed.

## Context

AX Engine exposes strong low-level surfaces but weak product-level composition
for common local serving workflows. Users currently assemble model acquisition,
manifest generation, server launch, and MTP sidecar conversion from separate
commands. That is precise but too expensive for day-to-day local use.

The desired workflow is a small command vocabulary:

```text
ax-engine serve qwen3.5-9b
ax-engine status
ax-engine convert-mtplx /path/to/Qwen3.6-35B-A3B-4bit --mtp-source /path/to/Qwen3.6-35B-A3B
```

The architectural risk is that a convenience CLI could blur AX's evidence
boundaries: direct repo-owned MLX runtime, delegated compatibility routes, MTP
sidecar provenance, and benchmark/publication claims must remain explicit.

## Decision

AX Engine will add a new top-level orchestration CLI named `ax-engine`.

The CLI will:

- wrap `ax-engine-server` for local serving;
- keep `ax-engine-server` as the actual HTTP server process;
- introduce a data-driven alias registry for common models;
- expose `convert-mtplx` as a stable wrapper around the existing MTP sidecar
  packaging contract;
- ship stable JSON output for automation;
- defer daemon persistence, boot services, and TUI monitoring until foreground
  serve and conversion are stable.

## Alternatives Considered

| Alternative | Pros | Cons |
|---|---|---|
| **A1: New `ax-engine` orchestration CLI** (chosen) | Clean user-facing vocabulary; does not overload benchmark CLI; can wrap existing binaries | Adds another installed binary |
| A2: Add subcommands to `ax-engine-server` | Keeps one server binary | Mixes process orchestration with server runtime args; harder to preserve current server CLI |
| A3: Add subcommands to `ax-engine-bench` | Reuses existing installed CLI | `bench` becomes a product server launcher, weakening its workload-contract role |
| A4: Keep scripts only | No Rust packaging work | Installed users still lack a stable UX; scripts are not a clean automation contract |
| A5: Implement daemon/status first | Matches external tools faster | Higher OS-specific risk before the core serve plan is stable |

## Rationale

`ax-engine-server` should remain boring: it parses runtime flags and starts the
HTTP/gRPC server. A local product CLI has a different job: resolve aliases,
download or locate models, check manifests, prepare sidecars, print plans, and
then launch the server. Splitting those responsibilities keeps runtime behavior
auditable and makes the convenience layer easier to evolve.

`ax-engine-bench` should remain the workload-contract and readiness CLI. It can
continue to provide `generate-manifest` and `doctor`, but using it as the main
serving command would make the CLI name misleading for end users.

Daemon persistence is intentionally not part of the first decision. A daemon
manager introduces OS-specific launch behavior, stale process metadata, log
rotation, cleanup semantics, and upgrade concerns. Those are valid features,
but they should follow a foreground command that already proves alias
resolution and server startup.

## Consequences

### Positive

- Users get a short command for common serve workflows.
- Existing server flags and runtime contracts remain intact.
- `convert-mtplx` becomes discoverable and automatable.
- JSON plans make automation and UI wrappers possible without parsing shell
  output.
- Alias additions can be reviewed as product/runtime support decisions.

### Negative

- Release packaging must include one more binary.
- Alias registry drift becomes a maintenance concern.
- The CLI must discover source-checkout versus installed-tool mode reliably.
- A partial implementation may temporarily duplicate some server preset data.

### Risks

- Friendly aliases may be mistaken for support or performance claims.
- `--download` can hide network and cache behavior unless output is explicit.
- `convert-mtplx` can package a sidecar that is technically valid but not
  benchmark-worthy for public README claims.
- Daemon follow-up work can become platform-specific and harder to test.

## Guardrails

- Alias output must include the resolved repo/path and selected backend.
- Runtime metadata remains authoritative for backend and support tier.
- `convert-mtplx` must write and validate `ax_mtp_sidecar_manifest.json`.
- Local `--mtp-source` paths must use real local shard discovery or fail closed;
  they must not be forwarded to the helper's `--hf-repo` argument as repo ids.
- README performance claims must not be updated from conversion output alone.
- `serve --dry-run --json` must exist before foreground launch is promoted.
- Daemon mode must start as non-persistent; boot persistence requires a follow-up
  ADR.

## Validation

The decision is implemented when:

- `ax-engine serve <alias> --dry-run --json` emits an
  `ax.local_serve_plan.v1` document;
- `ax-engine serve <local-dir> --dry-run --json` works without alias lookup;
- foreground `ax-engine serve` launches `ax-engine-server` with equivalent
  explicit flags;
- `ax-engine convert-mtplx ... --json` emits an `ax.convert_mtplx.v1` document;
- existing server, doctor, and script gates continue to pass;
- documentation explains direct runtime, delegated routes, and sidecar
  provenance without adding new performance claims.

## Follow-ups

- Decide whether alias profiles should live in Rust code, generated data, or a
  checked-in TOML file.
- Add non-persistent daemon/status/kill with a process registry.
- Add boot-persistent service support through a separate ADR.
- Add TUI monitoring only after status metadata is stable.
- Consider a Python API wrapper once the CLI JSON contracts are stable.
