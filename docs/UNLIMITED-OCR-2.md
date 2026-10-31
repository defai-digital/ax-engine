# Unlimited-OCR-2 native preview

AX Engine has an experimental native Rust/MLX path for
[`AutomatosX/AX-Unlimited-OCR-2-AXQ-MXFP8`](https://huggingface.co/AutomatosX/AX-Unlimited-OCR-2-AXQ-MXFP8)
at revision `462c40bb4f2c7eeb3e1230e912940efcbdefc8a3`.
Use a build containing this implementation and an explicit local model
directory. The published pack card still describes its original reference
storage release; these runtime results belong to AX Engine's subsequent
implementation. The model repository's Python code is not executed.

The native graph uses protected DASH routing weights, page-aware RoPE,
the source-layout MXFP8 expert matrices, and the existing dual vision tower.
The helper accepts 1..32 ordered page images and requires one
`<PAGE><image>` pair per image. It expands placeholders and attaches RGB
pixels for native processing. Single-page requests use dynamic tiles by
default (up to 16); multi-page requests default to global views.

```python
from tokenizers import Tokenizer
import ax_engine

model_dir = "/path/to/AX-Unlimited-OCR-2-AXQ-MXFP8"
images = ["page-1.png", "page-2.png"]
tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
prompt = "<PAGE><image>" * len(images) + "Multi page parsing."
request = ax_engine.prepare_unlimited_ocr_image_request(
    model_dir,
    [0, *tokenizer.encode(prompt, add_special_tokens=False).ids],
    images,
)
with ax_engine.Session(
    model_id="unlimited_ocr2", mlx=True, mlx_model_artifacts_dir=model_dir
) as session:
    result = session.generate(
        request.input_tokens,
        multimodal_inputs=request.multimodal_inputs,
        max_output_tokens=8192,
        no_repeat_ngram_size=35,
        ngram_window=1024,
    )
if result.status == "failed":
    raise RuntimeError("OCR inference failed")
print(tokenizer.decode(result.output_tokens, skip_special_tokens=False))
```

Outputs retain `<PAGE>`, `<BLOCK>` and `<FORK>` layout markers. As in the
released generation, a page-count guard keeps the output to one page per input
image: EOS is blocked until every page has started, only EOS may follow an extra
`<PAGE>` (so a complete output can end with one empty `<PAGE>` marker), and a
page reaching 20,000 tokens is closed. `ignore_eos` disables it, and
`AX_MLX_UNLIMITED_OCR_PAGE_GUARD=0` is the kill switch. A model
directory without a native manifest is mapped automatically; a stale v1
manifest must be regenerated. Unsupported sparse configurations, incomplete
experts, protected router quantization and conflicting marker ids fail closed.
Inline chat image parts are not accepted; use the prepared native request.

The [test report](UNLIMITED-OCR-2-TEST-REPORT.md) records the synthetic controls
and official-sample quality probes, including scoring methods and limits.
The [functional evidence](../benchmarks/unlimited-ocr2-mxfp8-20261031/README.md)
covers the exact pack on MacBook Pro M5 Max 128 GB with MLX 0.32.4. The
controls check complete synthetic OCR text, page count and identical output
after another request. These are bounded functional tests, not document
quality or performance qualification. Long documents, broad languages and
layouts, cancellation/interleaving stress, and BF16 quality comparisons need
separate evidence. MTP is not supported. Full per-request KV and routing
keys are retained, so total memory grows with document length; this preview
does not provide the upstream system's constant-memory document scheduling.
