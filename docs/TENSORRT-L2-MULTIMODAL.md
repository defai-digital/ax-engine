# TensorRT L2 Image Forwarding

AX Engine can forward inline images from OpenAI-compatible
`/v1/chat/completions` requests to the delegated NVIDIA backends.

This is a transport feature, not a claim that a model has been ported or that
OCR quality is production-ready. The configured vendor server must already be
running a compatible VLM with its visual runtime/engine enabled.

## Request contract

Send images as base64 `data:` URIs:

```json
{
  "model": "Qwen2.5-VL-3B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this document."},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,<BASE64>",
            "detail": "high"
          }
        }
      ]
    }
  ],
  "max_tokens": 512
}
```

Remote `http(s)` media URLs and bare client filesystem paths are rejected.
Audio and video remain disabled on the TensorRT L2 path. A decoded image is
limited to 64 MiB and must have a recognized raster image format.

## Provider behavior

### TensorRT-LLM on x86 CUDA

AX preserves the OpenAI `image_url` content shape and forwards it to
`trtllm-serve`.

### TensorRT Edge-LLM on Thor

The current experimental Edge-LLM server consumes a local `image` path. AX
therefore decodes the data URI into a request-scoped file with owner-only
permissions, translates the content part to Edge-LLM's `image` form, and
deletes the file when the request is released.

This path is enabled only when the Edge-LLM base URL uses `localhost`, a
127.0.0.0/8 address, or `::1`. AX and Edge-LLM must also share the same
temporary filesystem and run with compatible file permissions. A container
with a private `/tmp` does not satisfy that requirement even when its endpoint
is exposed on loopback.

## Validation boundary

Passing an image through AX proves only that the provider received the expected
payload. Production OCR validation must separately cover:

- the model checkpoint and visual engine pair;
- OCR correctness (for example CER/WER/ANLS);
- pages/s, TTFT, p95 latency, memory, and power;
- streaming and multi-image requests;
- pinned NVIDIA runtime versions and compatibility probes.

Provider references:

- [TensorRT-LLM multimodal serving](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html#multimodal-serving)
- [TensorRT Edge-LLM VLM inference](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/examples/vlm.html)
- [TensorRT Edge-LLM experimental server](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/examples/experimental-server.html)
