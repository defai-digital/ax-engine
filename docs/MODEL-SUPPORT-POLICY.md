# Model Support Policy

AX Engine is a focused Apple Silicon inference runtime for repo-owned model
families. The default product contract is direct MLX execution: AX owns the
model graph, token/KV runtime, scheduling, acceleration policy, server behavior,
and benchmark attribution.

Compatibility adapters for upstream `mlx_lm.server` and `llama.cpp` are
explicit migration and validation paths. They do not widen the AX-owned runtime
surface and must not be described as AX inference performance.

## Support Tiers

| Tier | Meaning | Requirements |
| --- | --- | --- |
| Certified direct | Publicly recommended direct runtime family | Repo-owned graph, checked download or manifest path, server/SDK smoke coverage, benchmark artifacts, documented limitations |
| Preview direct | Direct runtime family suitable for active testing | Repo-owned graph, manifest path, smoke coverage, clear gaps before certification |
| Incubating | Active family under evaluation | Recent upstream release, local artifacts available, architecture probe, implementation plan |
| Legacy freeze | Previously supported family with weak upstream activity | Security and severe correctness fixes only; no new acceleration or benchmark promises |
| End of life | Family removed from active support planning | No direct support work unless a new release or business requirement reopens the family |

## Promotion Gates

A model family should not move into public direct support until all of these are
true:

- upstream family has had a meaningful release or active artifact refresh within
  the last six months
- MLX safetensors or a reproducible conversion path is available
- AX has a repo-owned graph implementation instead of a delegated backend
- server or SDK smoke coverage exercises the route
- public performance claims have benchmark artifacts with route identity,
  prompt provenance, repetition policy, dirty-state provenance, and peer rows
  where applicable
- known limitations are documented before the model appears in public results tables

## Six-Month Activity Rule

Six months without a meaningful upstream model-family release is a default stop
signal for new AX support work. Exceptions require an explicit owner decision
and one of these reasons:

- a current customer or internal deployment depends on the family
- the family is a stable compatibility baseline for a directly supported model
- the family provides unique architecture coverage needed for AX runtime work
- a new upstream release is imminent and already available for validation

When none of those exceptions apply, move the family to legacy freeze or end of
life instead of adding new graph, benchmark, or release work.

## Delegated Adapter Boundary

`mlx_lm_delegated` and `llama_cpp` are explicit compatibility adapters. They
may be used for migration, route-contract checks, or external reference rows,
but they are not deployment defaults and are not evidence of AX-owned token/KV
runtime behavior.

Do not add a delegated adapter to make an unsupported model appear supported.
The preferred path is either direct implementation with evidence or a closed
unsupported response.
