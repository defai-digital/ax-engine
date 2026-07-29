# AX Engine static Mac pipeline runtime

This internal crate runs one dense Llama 3 model as a static pipeline across two or more Apple
Silicon Macs. AX Serving owns placement, admission, the immutable generation manifest, gang
readiness, and the public API. AX Engine owns stage-local weights and KV, activation transfer,
ordered request state, and cleanup.

The current path is an experimental bring-up surface, not a production-qualified 405B claim. It
supports greedy generation only and has source/mock/numerical coverage; it has not completed a
retained real-weight two-Mac fault, performance, or soak campaign.

## Bootstrap contract

Obtain these authenticated documents from `ax-mac-cluster-adapter`:

```text
GET /internal/cluster/engine-topology
GET /internal/cluster/ranks/{rank}/plan
x-ax-cluster-control-token: <rank-control-token>
```

The rank process requires `--bootstrap-plan` whenever `--coordinator-url` is enabled. Before loading
MLX weights it verifies the plan's cluster, generation, manifest, and rank; hashes every declared
artifact; rejects unsafe or duplicate paths; and confirms that `model-manifest.json` and every
stage-selected safetensor file are covered. With `--artifact-base-url`, the rank fetches only the
files in its rank-specific plan, enforces the certified byte size and SHA-256 while streaming, and
atomically installs each file. Existing verified files are reused without a network request.

## Processes

Run one rank service on every Mac:

```bash
ax-engine-pipeline-rank \
  --topology ./engine-topology.json \
  --bootstrap-plan ./rank-0-plan.json \
  --model-dir ./model \
  --artifact-base-url https://artifact-host.example/models/immutable-revision/ \
  --artifact-token '<rank-artifact-token>' \
  --rank 0 \
  --worker-token '<worker-data-plane-token>' \
  --coordinator-url http://adapter:9200 \
  --control-token '<rank-control-token>' \
  --peer-bandwidth-bytes-per-second 1000000000 \
  --peer-latency-micros 1000 \
  --listen 0.0.0.0:9300
```

Then run the OpenAI-compatible frontend beside rank 0:

```bash
ax-engine-pipeline-gateway \
  --topology ./engine-topology.json \
  --model-dir ./model \
  --endpoints http://mac-a:9300,http://mac-b:9300 \
  --worker-token '<worker-data-plane-token>' \
  --api-key '<rank0-runtime-token>' \
  --model-id '<manifest model.runtime_model_id>' \
  --listen 0.0.0.0:9400
```

The gateway refuses to start unless every endpoint is ready and reports the expected rank order,
cluster, generation, manifest digest, and model artifact digest. Keep the artifact token, worker
data-plane token, rank control token, gateway API key, AX Serving dispatch token, and AX Serving
control token separate. The example `http://` URLs assume a private encrypted overlay such as
WireGuard; do not send bearer credentials or activation payloads over an untrusted plaintext LAN.

The topology's `micro_batch_limit` bounds independent request-sequence steps that may be in flight
at once. Async HTTP transfers let those stable micro-batch units overlap on different stages;
callers commit each request in order, and every rank retains cancellation tombstones.

## Deliberate limits

- dense `llama3` AX-native artifacts only;
- static pipeline parallelism only (no tensor/hybrid parallelism);
- greedy sampling (`temperature=0`) only;
- no live generation replacement (artifact preparation is startup-only);
- no tool calls, structured output, prefix-aware routing, or tensor-shaped batch fusion;
- HTTP activation transport, without a certified encrypted mesh profile in this crate.
