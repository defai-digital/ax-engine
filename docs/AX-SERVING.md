# AX Serving

AX Serving is a commercial product for fleet routing and orchestration.
It handles request admission and routing across inference systems, including
Apple Silicon and NVIDIA/CUDA deployments.
[Contact us](mailto:enquiry@defai.digital) for access, deployment options, or
migration guidance.

For Linux CUDA inference, use
[AX Engine for SGLang](https://github.com/defai-digital/ax-engine-sglang#quick-start).

**Ports:** Engine defaults to **`127.0.0.1:31418`**. AX Serving’s public
gateway defaults to **`127.0.0.1:18080`**. Do not mix them. Full map:
[Ports](PORTS.md).

AX Engine no longer exposes the former `vllm`, `tensor_rt_llm`, or
`tensor_rt_edge_llm` support tiers, their provider-specific CLI flags, the
`ax-engine-vllm-runtime` package, or NVIDIA container and qualification
scripts. Generic delegated compatibility for `mlx_lm.server` and llama.cpp
remains in AX Engine.

## Migration

| Former AX Engine surface | AX Serving destination |
| --- | --- |
| CUDA provider selection and routing | `ax-dynamo-adapter` plus AX Serving backend configuration |
| vLLM runtime profiles and launcher | `integrations/nvidia/vllm-runtime/` |
| Worker compatibility identity | `integrations/nvidia/compatibility-manifest.schema.json` |
| Readiness, smoke, and soak gates | AX Serving NVIDIA qualification scripts and CI |
| CUDA OCI image | AX Serving NVIDIA runtime container target |

Requests that still send former CUDA-only OpenAI extensions to AX Engine fail
with an explicit migration error instead of being silently ignored or routed
to another backend.
