# Unlimited-OCR-2 MXFP8 test report

Test date: 2026-10-29. Report published: 2026-10-30.
Implementation: [`ff1ecb0a8`](https://github.com/defai-digital/ax-engine/commit/ff1ecb0a8ab78c4a96b43b70caaa17601f4ce85e).
Status: experimental native functionality with bounded English quality evidence.

> **Superseded encoder numerics.** These results were produced before the
> shared Unlimited-OCR vision encoder was aligned with the released
> `deepencoder.py`: CLIP MLP QuickGELU, CLIP LayerNorm epsilon `1e-5`, and
> tile-path position resizes (antialiased bicubic CLIP positions, half-pixel
> SAM `rel_pos`). The functional and quality numbers below describe the
> earlier encoder and must be re-run before they are quoted for current code.

## Environment and artifact identity

| Item | Tested value |
| --- | --- |
| Hardware | MacBook Pro M5 Max 128 GB, internal SSD |
| OS / MLX | macOS 27.0.1 / published MLX 0.32.4 wheel |
| Toolchain | Homebrew Cargo/Rust 1.99.1 |
| Build | `maturin develop --profile release-pyext` |
| Pack | `AutomatosX/AX-Unlimited-OCR-2-AXQ-MXFP8` |
| Pack revision | `462c40bb4f2c7eeb3e1230e912940efcbdefc8a3` |
| Native family | `unlimited_ocr2` |

The model repository's Python code is not executed during inference.
Both weight shards, config, tokenizer and tensor index were SHA-256 checked.
All 5,015 indexed tensors enter the manifest, with zero dropped tensors,
including 108 protected router/indexer tensors and 2,112 source expert matrices.
The build binds 509 source files, rechecked before and after compilation.
The official-sample run used the same native extension hash as the functional
controls. Hashes and build details are retained in the
[functional evidence](../benchmarks/unlimited-ocr2-mxfp8-20261029/README.md)
and [quality metrics receipt](../benchmarks/unlimited-ocr2-official-samples-20261029/metrics.json).

## Native functional controls

| Control | Result | Request seconds |
| --- | --- | ---: |
| Single page, global view | Complete normalized body and page count match | 0.353 |
| Single page, dynamic tiles | Complete normalized body and page count match | 0.556 |
| Ordered two-page input | Both bodies and output page order/count match | 0.626 |
| Repeat the first request after the other controls | Output tokens identical to the first request | 0.287 |

All four controls passed. A separate text-only negative control returned
`status=failed`, `finish_reason=error` and no output tokens. Callers should
inspect result status. Load time in this run was 0.919 seconds; process peak
RSS was 4,478,468,096 bytes (4.17 GiB). These are single-run observations,
not cold-start, throughput or long-document memory benchmarks.

## Official inputs and reference answers

The [official demo image](https://huggingface.co/spaces/baidu/Unlimited-OCR-2/blob/78c27d176793811cf8b5e74b62ceb04dd773c29a/example_images/demo_doc.png)
is a synthetic quarterly report with paragraphs, a table and a footer.
It has no separate published reference answer; its comparison uses an
independent manual visual transcription. It was tested with global-only
and dynamic-tile preprocessing.

[MarathonDocBench](https://huggingface.co/datasets/baidu/MarathonDocBench)
provides page images and ground-truth annotations. The tested revision is
`fa05fccbe82ee6ec628006a1d58bb153954c3ee3`. Four English pages from different
documents were selected before inference, taking the first eligible page
per document in lexicographic document/page order. Eligibility required
at least 300 letters, at least 99.5% ASCII letters, 700..6000 ground-truth
characters, a `text_block`, and no code, table or formula annotation categories.
The subset includes a table of contents and degraded scans. All four pages
used dynamic tiles.

Input hashes and page identities are in the metrics receipt. Original page
images and transcriptions remain local research inputs; the dataset card's
source-document redistribution restriction is preserved.

## Quality method and results

Generation used temperature zero, a 4,096-token output cap,
`no_repeat_ngram_size=35` and `ngram_window=128`. All six calls finished
normally with `finish_reason=stop`, one output page and no truncation.

The independent scorer extracts `<FORK>` bodies, converts HTML table
content to visible text, removes Markdown heading markers, applies Unicode
NFKC, case folding and whitespace normalization, then calculates Levenshtein
character error rate (CER) and word error rate (WER) using RapidFuzz 3.14.6.
Punctuation remains in the primary scores. Unordered token F1 uses
word/number multisets and does not assess reading order. These custom
per-page metrics are separate from the official eight-metric evaluator.

| Input / mode | CER, lower is better | WER | Unordered token F1 | Request seconds |
| --- | ---: | ---: | ---: | ---: |
| Official demo, global | 0.00% | 0.00% | 100.00% | 2.77 |
| Official demo, tiles | 0.00% | 0.00% | 100.00% | 4.65 |
| `mp3lan_0116`, page 3 | 12.66% | 17.14% | 99.64% | 7.33 |
| `mp3lan_0117`, page 1 | 7.72% | 4.95% | 99.01% | 12.47 |
| `mp3lan_0118`, page 1 | 0.41% | 1.81% | 99.65% | 9.59 |
| `mp3lan_0128`, page 6 | 0.00% | 0.00% | 100.00% | 15.73 |

Across the four dataset pages, micro CER is **4.07%** (294 edits / 7,224
normalized reference characters), and micro WER is **4.21%** (44 edits /
1,044 reference words). The two demo runs each match all 12 table-cell
strings in order. That cell check is not TableTEDS.

The larger CER values include formatting and annotation effects:

- `mp3lan_0116`, page 3: 92 deleted punctuation characters, primarily
  table-of-contents leader dots, three deleted letters/digits and no substitutions.
- `mp3lan_0117`, page 1: 136 deleted punctuation characters, 33 substitutions
  and two insertions.

Primary scores retain these differences; references were not adjusted after
seeing predictions. A separately labeled post-hoc alphanumeric-only
sensitivity check is retained in the metrics receipt. It does not replace
primary CER or establish error-free scan recognition.

## Regression checks and evidence limits

The implementation's local software gates passed: 4,525 Rust tests
(92 ignored), 333 Python tests (7 skipped), formatting, Clippy with
`--all-targets --all-features -- -D warnings`, `maturin develop`, Ruff,
mypy, script checks, both Qwen 3.8 qualification dry runs and primary-claim
checks. Focused M5 checks passed 15 core tests and two native MLX tests.
Independent 1.5-entmax math also matches 16 pinned upstream CPU oracle cases.
These software checks are separate from document quality measurements.

The quality probe covers five distinct English images and six executions.
It does not qualify Chinese OCR, full documents, the broad table/formula/layout
metrics, official leaderboard scores, or full-model BF16 parity. Cancellation
and interleaving stress remain unqualified. MTP is unsupported. Full K/V and
router keys are retained, so total memory grows with document length.

For operator usage, see [Unlimited-OCR-2 native preview](UNLIMITED-OCR-2.md).
