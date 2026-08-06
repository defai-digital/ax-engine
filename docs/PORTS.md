# Network ports and listen settings

This page is the source of truth for **where AX Engine listens** and how that
differs from **AX Serving** and common delegated peers. Defaults match
`crates/ax-engine-server` (`DEFAULT_INFERENCE_PORT`, `--host`).

## AX Engine (this repo)

| Surface | Default | How to change | Who connects |
| --- | --- | --- | --- |
| **OpenAI-compatible HTTP** | `127.0.0.1:31418` | `--host` / `--port` (CLI: `ax-engine serve …`, binary: `ax-engine-server`) | Apps, curl, SDKs, OpenClaw, local agents |
| **Optional gRPC adapter** | **off** | `--grpc-bind-address host:port` (e.g. `127.0.0.1:50051`) | gRPC clients only; not required for REST |
| **LAN DNS-SD advertise** | off | `--advertise-lan` (and related LAN flags) | Publishes the **same HTTP port** the server bound; see [LAN Discovery](LAN-DISCOVERY.md) |

Constants in code:

- Host default: `127.0.0.1`
- Port default: **`31418`** (`DEFAULT_INFERENCE_PORT`)

### Minimal local serve

```bash
# Same as omitting --port: default is 31418
ax-engine serve ax-gemma4-12b
# or explicit:
ax-engine serve ax-gemma4-12b --host 127.0.0.1 --port 31418

curl -sS http://127.0.0.1:31418/v1/runtime
curl -sS http://127.0.0.1:31418/v1/models
```

### Bind on all interfaces (LAN)

```bash
ax-engine serve /path/to/model --host 0.0.0.0 --port 31418
# Optional: advertise for AX Serving agents
ax-engine-server --host 0.0.0.0 --port 31418 --advertise-lan …
```

Clients then use the machine’s LAN IP (or the advertised URL), still on
**31418** unless you changed `--port`.

### Client base URLs

| Client | Typical base URL |
| --- | --- |
| OpenAI-style HTTP | `http://127.0.0.1:31418` or `http://127.0.0.1:31418/v1` (product-dependent; many SDKs want the `/v1` suffix) |
| Runtime probe | `GET http://127.0.0.1:31418/v1/runtime` |

Do **not** point AX Engine clients at **18080** unless you intentionally front
Engine with [AX Serving](https://github.com/defai-digital/ax-serving).

## AX Serving (separate product)

[AX Serving](https://github.com/defai-digital/ax-serving) is the fleet /
control-plane product. Its **defaults are not Engine’s**.

| Process (AX Serving) | Default | Role |
| --- | --- | --- |
| `ax-serving-api` **public** gateway | `127.0.0.1:18080` | OpenAI clients / AX Serving SDKs |
| `ax-serving-api` **internal** plane | `127.0.0.1:19090` | Worker register / heartbeat (not public chat) |
| Node / Thor **agent** | listen `…:18081` (typical) | Agent data plane |
| Upstream **runtime** (example) | often Engine `:31418` or another worker `:8000` / `:8080` | What the agent proxies to |

```text
App / SDK  ──►  ax-serving-api :18080     (Serving public API)
                   │
                   │ internal :19090
                   ▼
              node agent :18081
                   │
                   ▼
           ax-engine-server :31418        (Engine default)
           or other worker URL
```

| You are talking to… | Port |
| --- | ---: |
| AX Engine alone | **31418** |
| AX Serving public gateway | **18080** |
| AX Serving internal control plane | **19090** |

Migration and CUDA ownership: [AX Serving](AX-SERVING.md) ·
[ax-serving repo](https://github.com/defai-digital/ax-serving).

## Common delegated peers (Engine adapters)

These are **not** AX Engine listen ports. They are separate processes Engine
may call when you select an explicit support tier.

| Peer | Example listen | Engine flag / notes |
| --- | --- | --- |
| `mlx_lm.server` | `127.0.0.1:8090` | `--support-tier mlx_lm_delegated` + `--mlx-lm-server-url` |
| llama.cpp server | `127.0.0.1:8081` (example) | `--support-tier llama_cpp` + server URL |
| Custom worker behind Serving | operator-chosen (e.g. 8000, 8080) | Configured in AX Serving / agent, not Engine’s default |

## Auth and related settings (not ports)

| Setting | Typical env / flag | Notes |
| --- | --- | --- |
| Inference API key | `--api-key` / server auth config | Covers HTTP and optional gRPC when enabled |
| Request timeouts | `--request-timeout-secs`, `--grpc-request-timeout-secs` | See [Server](SERVER.md) |
| Multi-model load | `POST /v1/model/load` on the **same** HTTP port | [Multi-model](SERVER.md#multi-model-serving) |

## Checklist

1. Local Engine smoke: hit **`http://127.0.0.1:31418`**.
2. Fleet via Serving: clients hit **Serving `:18080`**; Engine may still listen on **31418** as a worker.
3. Never assume Engine defaults to 18080, 8080, or 8000.
4. After changing `--port`, update every client `base_url` / `baseURL` and any Serving agent `runtime` URL.

## Related docs

- [Server](SERVER.md) — routes, multi-model, flags
- [Getting Started](GETTING-STARTED.md) — first serve
- [CLI](CLI.md) — `ax-engine serve`
- [LAN Discovery](LAN-DISCOVERY.md) — mDNS advertise
- [Local Engine Clients](LOCAL-ENGINE-CLIENTS.md) — in-process vs sidecar
- [AX Serving](AX-SERVING.md) — product split and CUDA migration
