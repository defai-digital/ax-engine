# Unsupported peer lanes

The Gemma4 peer lanes were attempted on the same host and prompt contract.

MTPLX 2.9.0 reported:

```text
[MTP] AX-gemma-4 declares MTP layer(s) but ships no MTP weights; serving autoregressive (no speculative draft head).
ERROR: generate_mtpk requires an MTP-enabled runtime
```

The same result occurred for Gemma4 31B and Gemma4 26B-A4B. The exact model
paths and full AXQ metadata are retained in the successful AX raw artifacts.

OMLX 0.6.4 reported that its VLM loader received 356 parameters not present in
the model, including `model.vision_tower.encoder.layers.0.*`. OMLX's regular
text engine can start the Gemma pack only with MTP inactive; it therefore does
not qualify as a Gemma assistant-MTP peer result. No direct-mode number is
substituted for either unsupported lane.
