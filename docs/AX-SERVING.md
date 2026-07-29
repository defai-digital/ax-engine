# NVIDIA Serving Moved To AX Serving

AX Engine owns local Apple Silicon inference. NVIDIA/CUDA worker integration,
heterogeneous routing, runtime qualification, and Dynamo connectivity now live
in [AX Serving](https://github.com/defai-digital/ax-serving).

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

Use the
[full migration guide](https://github.com/defai-digital/ax-serving/blob/main/docs/migrations/ax-engine-cuda-to-ax-serving.md)
for configuration mapping, rollout order, and rollback guidance. The
[Dynamo integration guide](https://github.com/defai-digital/ax-serving/blob/main/docs/integrations/nvidia/DYNAMO.md)
documents adapter configuration and operational contracts.

Requests that still send former CUDA-only OpenAI extensions to AX Engine fail
with an explicit migration error instead of being silently ignored or routed
to another backend.
