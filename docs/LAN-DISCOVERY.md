# LAN discovery (mDNS / DNS-SD)

Status: Active  
Scope: opt-in advertise so AX Serving agents and operators can find local
`ax-engine-server` instances without hard-coding IPs  
Related: AX Serving design
`docs/designs/ax-engine-integration-and-lan-discovery-2026-07-14.md`

## Why

On a home or lab LAN, several Macs may run AX Engine. Operators should not need
to copy `192.168.x.y` into agent env files by hand. Discovery is **opt-in** and
**unauthenticated** — it is not a substitute for worker tokens or API keys.

This is **not** exo-style model sharding. AX Engine still serves whole models;
AX Serving routes requests to workers. Discovery only answers “where is an
engine HTTP endpoint?”

## Enable advertise

```bash
ax-engine-server \
  --host 0.0.0.0 \
  --port 31418 \
  --mlx \
  --mlx-model-artifacts-dir /absolute/path/to/artifacts \
  --api-key "$AX_ENGINE_API_KEY" \
  --advertise-lan \
  --lan-cluster home-lab
```

Environment equivalents:

| Flag | Env |
| --- | --- |
| `--advertise-lan` | `AX_ENGINE_ADVERTISE_LAN=1` |
| `--allow-open-lan` | `AX_ENGINE_ALLOW_OPEN_LAN=1` |
| `--lan-cluster` | `AX_ENGINE_LAN_CLUSTER` |
| `--lan-instance-name` | `AX_ENGINE_LAN_INSTANCE_NAME` |
| `--lan-advertise-host` | `AX_ENGINE_LAN_ADVERTISE_HOST` |

### Rules

- Bind host must not be loopback-only when advertising (use `0.0.0.0` or a LAN IP).
- Advertised address prefers private IPv4 (RFC1918). Link-local is avoided so
  AX Serving worker registration can accept the URL.
- **API key required by default.** Without `--api-key` / `AX_ENGINE_API_KEY`,
  advertise is refused so a keyless instance is not discoverable on the LAN
  with `auth=open`. To opt in deliberately, pass `--allow-open-lan` or set
  `AX_ENGINE_ALLOW_OPEN_LAN=1`.

## Service type

```text
_ax-engine._tcp.local.
```

TXT keys: `proto`, `kind`, `version`, `model`, `auth`, `scheme`, `path`,
`cluster`, `instance`, `platform`.

`model` tracks the current default model. Registry operations that change the
default re-announce the service so browsers do not retain the startup model.

## HTTP verify

After browse, clients should call:

```http
GET /v1/discovery
```

This endpoint is unauthenticated by design (same class as `/healthz`). It never
returns API keys or local filesystem secrets. When any loaded model's generation
worker is down it returns **503** (same readiness bar as `/health`) so agents do
not register a partially unavailable peer after mDNS browse.

## Browse from AX Serving

```bash
ax-servingctl discover --timeout-secs 3 --cluster home-lab --json
```

Or set on the runtime agent:

```bash
AXS_DISCOVER_LAN=1
AXS_DISCOVER_LAN_CLUSTER=home-lab
# optional: AXS_DISCOVER_LAN_INSTANCE=<instance name>
# leave AXS_NODE_RUNTIME_URL unset to resolve via mDNS
```

## Security notes

- mDNS can be spoofed on the LAN. Always verify `/v1/discovery` and still use
  `AXS_RUNTIME_API_KEY` / gateway worker tokens for production.
- Do not enable `--advertise-lan` on hostile networks with `auth=open`.
