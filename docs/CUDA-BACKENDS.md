# CUDA Backends

AX Engine is Mac-first, but its API and control plane can also front NVIDIA
inference workers on Linux. CUDA support is deliberately **delegated**: the
portable Rust server owns routing, policy, security, identity, and
observability, while a separately supervised GPU worker owns model execution.

> [!IMPORTANT]
> The vLLM provider foundation and Thor candidate are implemented, but CUDA
> profiles remain release candidates until their hardware-specific quality,
> performance, security, soak, and rollback gates pass. CUDA support does not
> change the default Mac installation or imply GA support for every NVIDIA GPU.

AX OCR's product deployment priority is explicit: Apple Silicon Mac and NVIDIA
Thor are co-primary targets, while certified Linux x86_64 CUDA PCs are the
secondary support platform. That roadmap priority is separate from release
maturity: a candidate profile remains a candidate until every applicable gate
below passes.

## Platform Strategy

| Product priority | Platform and workload | Default path | Optional optimized path |
| --- | --- | --- | --- |
| Primary | macOS 26+ on Apple Silicon | Repo-owned AX MLX/Metal runtime | None; `mlx_lm.server` remains an explicit compatibility adapter |
| Primary | Linux aarch64 NVIDIA Thor, broad OCR/VLM coverage | AX Engine → vLLM | AX Engine → TensorRT Edge-LLM after per-model certification |
| Secondary | Linux x86_64 CUDA, broad OCR/VLM coverage | AX Engine → vLLM | AX Engine → TensorRT-LLM after per-model certification |

This is not a universal speed ranking. vLLM is the compatibility and model
coverage path. TensorRT-LLM and TensorRT Edge-LLM are separate optimization
lanes that may become the preferred route for a particular model only after
like-for-like quality, latency, throughput, memory, power, and operational
validation.

```text
AX product / AX OCR
        |
        v
portable AX Engine server
        |
        +-- selected_backend=vllm
        |      +-- cuda-linux-x86_64-<certified-sku>
        |      `-- cuda-linux-aarch64-thor-sm110
        |
        +-- selected_backend=tensor_rt_llm       (x86 optimized lane)
        `-- selected_backend=tensor_rt_edge_llm  (Thor optimized lane)
```

## One vLLM Provider, Two Runtime Profiles

AX Engine has one logical `vllm` provider. Linux x86_64 and Thor do **not**
have separate request/response implementations. Both use the same:

- OpenAI-compatible chat/completions and SSE contract;
- ordered text/image content parts;
- readiness and exact upstream model identity checks;
- authentication, TLS, timeout, error, usage, and finish-reason mapping;
- no-retry policy for generation POST requests;
- runtime and capability reporting.

Architecture-specific behavior belongs in the independently packaged
`ax-engine-vllm-runtime` profiles:

- `cuda-linux-x86_64-a6000-sm86`
- `cuda-linux-aarch64-thor-sm110`

Profiles validate the OS, CPU architecture, GPU/compute capability, Python,
PyTorch, CUDA, vLLM version, and architecture-specific dependency-lock digest
before starting the worker. A profile mismatch fails closed; it never selects
another provider.

## Ownership Boundary

### AX Engine

AX Engine owns:

- provider selection and public backend identity;
- OpenAI request/response and stream translation;
- bounded data-URI validation and multimodal forwarding;
- upstream readiness, authentication, TLS, timeouts, and error mapping;
- admission, observability, and operator-facing runtime metadata;
- the optional Python/OCI `ax-engine-vllm-runtime` distribution.

### GPU worker

The selected worker owns:

- model loading and GPU execution;
- scheduler, KV cache, kernels, and engine-specific tuning;
- the model/runtime compatibility matrix.

vLLM, TensorRT-LLM, and TensorRT Edge-LLM remain separate processes and crash
domains. AX Engine does not import Python, PyTorch, or vLLM into the Rust
server, and it does not silently fall back between workers.

### AX OCR and other products

Product clients retain workflow and acceptance responsibilities. For AX OCR,
that includes document/page processing, output formatting, accuracy corpora,
model artifact policy, quantization, and release evidence. Generic vLLM HTTP,
SSE, launcher, and compatibility behavior belongs in AX Engine. A direct-vLLM
client may remain only during a measured migration/rollback window.

## Building the Portable Control Plane

The normal Mac server keeps the default MLX feature:

```bash
cargo build -p ax-engine-server --profile release-server
```

Build a Linux control plane without MLX linkage:

```bash
cargo build -p ax-engine-server \
  --profile release-server \
  --no-default-features \
  --features delegated-server
```

The delegated-only binary must not link `mlx-sys` or advertise native MLX
capability. Conversely, a build without `mlx-native-server` must reject an MLX
selection instead of reporting a false capability.

## Running vLLM Through AX Engine

Install or run the separately released
[`ax-engine-vllm-runtime`](../packages/ax-engine-vllm-runtime/README.md) on the
CUDA host. Inspect and preflight the selected profile before serving:

```bash
ax-engine-vllm-runtime --list-profiles

ax-engine-vllm-runtime \
  --profile cuda-linux-aarch64-thor-sm110 \
  --model baidu/Unlimited-OCR \
  --served-model-name baidu/Unlimited-OCR \
  --check-only \
  --json
```

Start the worker on loopback:

```bash
ax-engine-vllm-runtime \
  --profile cuda-linux-aarch64-thor-sm110 \
  --model baidu/Unlimited-OCR \
  --served-model-name baidu/Unlimited-OCR \
  --host 127.0.0.1 \
  --port 8000
```

Then start the delegated-only AX Engine server:

```bash
ax-engine-server \
  --model-id ax-ocr \
  --support-tier vllm \
  --vllm-server-url http://127.0.0.1:8000/v1 \
  --vllm-upstream-model-id baidu/Unlimited-OCR \
  --vllm-model-profile unlimited-ocr \
  --vllm-runtime-profile cuda-linux-aarch64-thor-sm110 \
  --vllm-max-in-flight 2 \
  --port 31418
```

For a certified x86_64 A6000 host, select
`cuda-linux-x86_64-a6000-sm86` in both commands. The AX-facing model id and the
upstream model id are intentionally separate. `/v1/models` and `/v1/runtime`
expose both identities so clients can pin the route they evaluated.

## TensorRT Optimization Lanes

TensorRT routes stay first-class and explicit:

- x86 CUDA: `--support-tier tensor-rt-llm` with
  `--tensorrt-llm-server-url`, `--tensorrt-llm-upstream-version`, and
  `--tensorrt-llm-execution-backend`;
- Thor: `--support-tier tensor-rt-edge-llm` with
  `--edge-llm-server-url`, `--edge-llm-upstream-version`, and
  `--edge-llm-execution-backend`.

AX fails session construction when either TensorRT route lacks its exact
upstream version or execution-path identity. The optional provider-specific
`--*-upstream-model-id` defaults to the AX-facing model id; a certified
deployment should also set `--*-runtime-profile` to its hardware/artifact
profile. Identity flags are valid only with their matching support tier and
never select or switch a provider implicitly.

For example, an x86 route backed by `trtllm-serve` can be configured as:

```text
ax-engine-server \
  --model-id ax-ocr \
  --support-tier tensor-rt-llm \
  --tensorrt-llm-server-url http://127.0.0.1:8000 \
  --tensorrt-llm-upstream-model-id <loaded-model-id> \
  --tensorrt-llm-upstream-version <exact-tensorrt-llm-version> \
  --tensorrt-llm-execution-backend pytorch \
  --tensorrt-llm-runtime-profile cuda-linux-x86_64-a6000-sm86
```

A Thor Edge-LLM route uses the corresponding contract:

```text
ax-engine-server \
  --model-id ax-ocr \
  --support-tier tensor-rt-edge-llm \
  --edge-llm-server-url http://127.0.0.1:8090 \
  --edge-llm-upstream-model-id <loaded-model-id> \
  --edge-llm-upstream-version <exact-edge-llm-version> \
  --edge-llm-execution-backend cpp \
  --edge-llm-runtime-profile cuda-linux-aarch64-thor-sm110
```

`GET /v1/runtime` exposes the selected provider, upstream model id, runtime
profile, upstream version, execution backend, readiness, and a redacted
endpoint authority. These values are the configured control-plane assertion;
release automation must compare them with the independently captured worker
package/image manifest. AX does not infer a worker version from a URL, and
`readiness=configured` is not proof that the declared binary is running.

They are not implementations of the vLLM provider and must retain distinct
backend identities, model matrices, tuning, error metadata, and release
evidence. Multimodal transport details are documented in
[TensorRT L2 Image Forwarding](TENSORRT-L2-MULTIMODAL.md).

The TensorRT adapters reuse AX's provider-neutral delegated HTTP policy and the
same strict OpenAI SSE framing reader as vLLM. The shared reader bounds each
frame to 1 MiB while reading, so an unterminated upstream line cannot first
grow memory without bound. It also validates UTF-8/JSON and provider error
events, and rejects EOF before `[DONE]`. Request content DTOs remain
provider-specific: Edge-LLM's experimental staged-image contract is
intentionally not treated as equivalent to vLLM's validated ordered data-URI
contract. Sampler behavior, capabilities, runtime identity, and release
evidence likewise remain separate.

For Thor, use NVIDIA's
[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM), not the
desktop/datacenter TensorRT-LLM path. NVIDIA's current support matrix lists
Jetson Thor with JetPack 7.x and describes the C++ runtime as the
production-oriented deployment layer. Its OpenAI-compatible Python server is
still explicitly experimental, however. AX must therefore pin and certify the
exact server contract while this lane is a candidate; a GA deployment should
prefer an AX-owned, versioned sidecar over the stable C++ runtime if the
upstream server remains experimental.

The current Edge-LLM model matrix includes Qwen 3/3.5/3.6 and the full Gemma 4
family, but does not list `baidu/Unlimited-OCR`. Edge-LLM is consequently an
optimization lane for models it explicitly supports, not a drop-in replacement
for AX OCR's vLLM compatibility route.

For x86 CUDA, TensorRT-LLM's documented
[`trtllm-serve`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/quick-start-guide.md)
provides the OpenAI-compatible process boundary consumed by AX.
Do not infer its execution backend from the project name: TensorRT-LLM 1.2
[removed the legacy TensorRT engine backend](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)
and made PyTorch the sole execution backend. AX profiles must therefore pin and
report both the TensorRT-LLM release and its execution path, and every major
upgrade must repeat quality, latency, throughput, memory, and rollback gates.
Existing engine-build recipes or performance evidence cannot be carried across
that breaking change. Until those identities are machine-readable in
`/v1/runtime`, the TensorRT-LLM lane remains non-GA even when route-level
compatibility tests pass.

Do not switch a model from vLLM to a TensorRT lane merely because the engine
starts. Promotion requires the same model/checkpoint semantics, product
quality, streaming behavior, concurrency policy, and rollback contract.

## Security and Reliability Rules

- Bind the worker to loopback by default.
- A remote vLLM endpoint requires explicit `--vllm-allow-remote`, verified
  HTTPS, and an optional pinned CA.
- Pass credentials through an environment variable or a regular secret file,
  never a raw CLI argument or image layer.
- Remote image URLs and bare client filesystem paths are rejected. The vLLM
  OCR profile accepts bounded inline PNG/JPEG data URIs.
- Readiness GET requests may use bounded retries. Generation POST requests are
  never retried automatically because replay is not safely idempotent.
- Delegated JSON and SSE requests use separate keep-alive pools keyed by their
  `Accept` contract. A stream that ends at the OpenAI `[DONE]` sentinel cannot
  evict the reusable JSON connection for the next non-stream request.
- AX Engine enables `TCP_NODELAY` on accepted HTTP sockets. This keeps the SSE
  response headers and first small event from waiting on Nagle/delayed-ACK
  interaction; it does not change worker scheduling or model latency.
- Delegated control-plane startup performs readiness only. Native MLX warm-up
  generation is explicitly gated off for vLLM and every other delegated
  backend.
- Backend failure returns an error with the selected provider identity. There
  is no silent cross-provider fallback.
- TensorRT routes fail closed without machine-readable upstream-version and
  execution-backend identity. Promotion evidence cross-checks that configured
  identity against the worker package/image rather than trusting it as a
  runtime probe.
- Run worker containers as non-root with a read-only root filesystem and
  explicit writable cache/tmpfs mounts.
- Generate an SPDX SBOM and scan the exact image digest, not only a detached
  filesystem SBOM. Keep the raw result even when a VEX document is applied.
- A VEX statement is deployment-scoped evidence, not a package fix. If its
  ingress, network, model, media, or digest assumptions change, treat the VEX
  as invalid and rescan/review before deployment.

## Release Gates

A runtime profile is releasable only when evidence covers:

1. exact source, wheel, lock, image digest, SBOM, and native build provenance;
2. native control-plane build/tests with no MLX linkage;
3. non-root/read-only GPU preflight of the exact OCI image;
4. real non-stream and stream requests through AX Engine;
5. direct-worker versus AX-mediated product quality parity;
6. latency, throughput, memory, power, admission, cancel, and failure tests;
7. a full 24-hour soak with no hidden retries or restarts;
8. rollback rehearsal and an observation window before deleting a product's
   direct provider;
9. reviewed vulnerability disposition and immutable build provenance for the
   release digest.

Thor release validation must reproduce the exact candidate on two independent
Thor/SM110 hosts. One machine may run the formal soak while the second performs
clean-image reproduction and failure tests. An x86 profile needs the same
evidence on its certified GPU SKU; a native Python environment does not replace
the OCI gate.

## Current Candidate Boundary

As of 2026-07-24:

- the shared vLLM provider, delegated-only server, runtime package, and
  Unlimited-OCR multimodal contract are implemented;
- the hardened Thor arm64 OCI candidate
  `sha256:c1bc3b4f3c5870de2290604bc5e3dbd37066b36dbad2137f9b145634cc1bec3b`
  has passed 11/11 native preflight plus real direct/AX-mediated stream and
  non-stream OCR on `df-thor-02`;
- the second Thor also passed client-disconnect cancellation, a bounded read
  timeout, worker-kill/error/recovery, malformed-SSE, single-generation-POST,
  no-fallback tests, and an old/new image rollback rehearsal;
- the hardened image has an SPDX 2.3 SBOM and an exact-image Grype scan. The
  raw scan has zero Critical and one High finding:
  `CVE-2026-8461` in FFmpeg 8.1.1's MagicYUV video decoder. An exact-digest
  OpenVEX statement marks it not affected only for AX's text/inline-image OCR
  ingress with a private worker; AX rejects video before upstream. Direct
  generic vLLM or video use is outside that statement, and human security
  approval remains required;
- the original digest's 24-hour run remains in progress on `df-thor-01`.
  Because the hardened digest changed, a separate 24-hour clock started on
  `df-thor-02` at `2026-07-24T09:15:46Z`; neither partial run is a pass and
  their durations are never combined;
- the x86_64 image and runtime closure exist, but native OCI execution and
  A6000 Unlimited-OCR/soak evidence remain release gates.
- the active secondary x86 CUDA target is **RTX A6000 / SM86**
  (`cuda-linux-x86_64-a6000-sm86`). Formal A100-SXM4-80GB / SM80 WNA16 remains
  an optional legacy profile; A6000 is not required to impersonate A100, and
  dual free A6000 hosts may split TensorRT-LLM vs Unlimited-OCR lanes instead
  of concurrent double occupancy of one 48 GiB card.
- same-worker TensorRT-LLM checks on `tnr-0` validate the portable control
  plane without restarting the preserved worker. Three 30-pair runs and one
  100-pair diagnostic meet the non-stream p50/p95 proxy-overhead gate after
  separating JSON/SSE pools and enabling `TCP_NODELAY`; deterministic
  loopback tests isolate sub-millisecond proxy overhead. These results are
  control-plane evidence only, not an engine speed ranking or an
  Unlimited-OCR/OCI release pass.
- TensorRT-LLM smoke and operator scripts must pass machine-readable
  `--tensorrt-llm-upstream-version` and `--tensorrt-llm-execution-backend`
  (see `scripts/tensorrt_llm_a600_smoke.sh`). Missing identity must fail closed
  before worker I/O (`scripts/validate_tensorrt_identity_fail_closed.sh`).

These statements describe candidate evidence, not a GA promise. Keep
production defaults and public support claims unchanged until every applicable
gate above is complete.

## Related Documentation

- [Architecture](ARCHITECTURE.md)
- [Server](SERVER.md)
- [Getting Started](GETTING-STARTED.md)
- [TensorRT L2 Image Forwarding](TENSORRT-L2-MULTIMODAL.md)
- [vLLM runtime package](../packages/ax-engine-vllm-runtime/README.md)
- [vLLM OCI build](../containers/vllm/README.md)
