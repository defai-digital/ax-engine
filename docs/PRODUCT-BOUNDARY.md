# Product Boundary

AX Engine should be treated as the runtime foundation of the AutomatosX stack,
not as the full serving and orchestration product.

## AX Engine Owns

- Apple-Silicon-native model loading and inference
- Metal/CPU/hybrid execution backends
- local CLI, Rust SDK, Python bindings, benchmarking tooling, and a lightweight
  single-node HTTP surface for local and edge use

## AX Engine Does Not Own

- multi-node request routing
- fleet orchestration and rollout control
- tenancy, authentication, quotas, or billing
- policy enforcement, audit planes, or sovereign deployment control planes

Those concerns belong in AX Serving and the broader AutomatosX governance
stack.

## Recommended Packaging

Best practice is to keep the runtime and the control plane separate:

- `ax-engine`: permissive-licensed runtime and developer entry point
- `ax-serving`: production serving, routing, orchestration, and enterprise controls

This separation should hold even if the codebases are later moved into a
shared monorepo.

## Recommended HTTP Strategy

AX Engine now ships `ax-engine-server` as a lightweight surface for:

- local development
- edge deployments
- single-node OpenAI-compatible integrations

That HTTP surface should stay intentionally thin. It should expose the runtime
without turning AX Engine into the cluster control plane.

## Rollout Order

1. Keep AX Engine focused on runtime quality, SDKs, and local ergonomics.
2. Keep `ax-engine-server` thin and compatibility-focused.
3. Keep multi-node and governed serving in AX Serving.
