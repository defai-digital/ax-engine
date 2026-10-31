# Unlimited-OCR-2 MXFP8 test report

Test date: 2026-10-31. Report published: 2026-10-31.
Implementation: [`b4a283e9e`](https://github.com/defai-digital/ax-engine/commit/b4a283e9eface0aea509df5a5a946aa57d822ba9).
Status: experimental native functionality with bounded English quality evidence.

## What changed since the 2026-10-29 report

Two changes since the [2026-10-29 run](../benchmarks/unlimited-ocr2-official-samples-20261029/metrics.json):

- **Encoder numerics.** The shared vision encoder now follows the released
  `deepencoder.py`: CLIP MLP QuickGELU, CLIP LayerNorm epsilon `1e-5`, and
  tile-path position resizes (antialiased bicubic CLIP positions, half-pixel
  SAM `rel_pos`).
- **Page-count guard.** Generation now has the guard that the released
  generation runs by default: one output page per input image.

Same inputs and scorer as before. Dataset micro CER moved from 4.07% to 4.49%
and micro WER from 4.21% to 4.89%:

- `mp3lan_0117` improved.
- `mp3lan_0118` worsened. Most of the change is the "ERIC Full Text Provided
  by ERIC" logo text: it is printed on the page but absent from the ground
  truth, and the new encoder transcribes it.
- `mp3lan_0128` differs only in curly versus straight quotes.

The same encoder before/after comparison gave identical scores on
MacBook Pro M5 Max 128 GB, Mac mini M4 Pro 64 GB and Mac Studio M2 Ultra 192 GB.
Token-level differences between those chips were confined to layout
coordinates and block labels.

Without the guard, global-view multi-page requests could keep emitting
repeated pages: four pages failed with the earlier encoder, and two pages
failed once the encoder matched the reference. With the guard, two- and
four-page requests return every page body in order on all three chips. Single-page
outputs are token-identical with and without it.

## Environment and artifact identity

| Item | Tested value |
| --- | --- |
| Hardware | MacBook Pro M5 Max 128 GB, internal SSD |
| OS / MLX | macOS 27.0.1 / published MLX 0.32.4 wheel |
| Toolchain | Homebrew Cargo/Rust 1.99.1 |
| Build | `maturin build --profile release-pyext` from a clean `git archive` of the implementation commit |
| Pack | `AutomatosX/AX-Unlimited-OCR-2-AXQ-MXFP8` |
| Pack revision | `462c40bb4f2c7eeb3e1230e912940efcbdefc8a3` |
| Native family | `unlimited_ocr2` |

The model repository's Python code is not executed during inference.
Both weight shards, config, tokenizer and tensor index were SHA-256 checked.
All 5,015 indexed tensors enter the manifest, with zero dropped tensors,
including 108 protected router/indexer tensors and 2,112 source expert matrices
(manifest regenerated at this commit; tensor entries identical to 2026-10-29).
The build binds 527 source files, rechecked before and after compilation.
The official-sample run used the same native extension hash as the functional
controls. Hashes and build details are retained in the
[functional evidence](../benchmarks/unlimited-ocr2-mxfp8-20261031/README.md)
and [quality metrics receipt](../benchmarks/unlimited-ocr2-official-samples-20261031/metrics.json).

## Native functional controls

| Control | Result | Request seconds |
| --- | --- | ---: |
| Single page, global view | Complete normalized body and page count match | 0.369 |
| Single page, dynamic tiles | Complete normalized body and page count match | 0.583 |
| Ordered two-page input | Both bodies and output page order/count match | 0.640 |
| Repeat the first request after the other controls | Output tokens identical to the first request | 0.284 |

All four controls passed. A complete multi-page output can end with one empty
`<PAGE>` marker before EOS. The released guard leaves the same marker, so the
witness does not count it as a page. A separate text-only negative control
returned `status=failed`, `finish_reason=error` and no output tokens. Callers
should inspect result status. Load time in this run was 2.803 seconds; process
peak RSS was 4,486,086,656 bytes (4.18 GiB). These are single-run
observations, not cold-start, throughput or long-document memory benchmarks.

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
| Official demo, global | 0.00% | 0.00% | 100.00% | 2.79 |
| Official demo, tiles | 0.00% | 0.00% | 100.00% | 4.44 |
| `mp3lan_0116`, page 3 | 12.66% | 17.14% | 99.64% | 7.33 |
| `mp3lan_0117`, page 1 | 7.45% | 3.96% | 99.01% | 13.19 |
| `mp3lan_0118`, page 1 | 2.13% | 4.71% | 98.25% | 10.02 |
| `mp3lan_0128`, page 6 | 0.09% | 0.62% | 100.00% | 15.51 |

Across the four dataset pages, micro CER is **4.49%** (324 edits / 7,224
normalized reference characters), and micro WER is **4.89%** (51 edits /
1,044 reference words). The two demo runs each match all 12 table-cell
strings in order. That cell check is not TableTEDS.

The larger CER values include formatting and annotation effects:

- `mp3lan_0116`, page 3: 92 deleted punctuation characters, primarily
  table-of-contents leader dots, three deleted letters/digits and no substitutions.
- `mp3lan_0117`, page 1: 133 deleted punctuation characters, 32 substitutions
  and two insertions.
- `mp3lan_0118`, page 1: 39 insertions, mostly the logo line printed in the
  page corner that the annotation omits, plus `MF01` read as `MFO1` in
  typewriter type.
- `mp3lan_0128`, page 6: two substitutions, curly quotes where the
  annotation has straight quotes.

Primary scores retain these differences; references were not adjusted after
seeing predictions. A separately labeled post-hoc alphanumeric-only
sensitivity check is retained in the metrics receipt. It does not replace
primary CER or establish error-free scan recognition.

## Regression checks and evidence limits

The implementation commit passed all 27 CI checks, including formatting,
Clippy, the Rust, Python and SDK suites and script smoke tests. Local runs
also passed the 2,167 `ax-engine-mlx` tests, Clippy with
`--all-targets --all-features -- -D warnings`, and the script checks.
Independent 1.5-entmax math matches 16 pinned upstream CPU oracle cases.
These software checks are separate from document quality measurements.

The quality probe covers five distinct English images and six executions.
It does not qualify Chinese OCR, full documents, the broad table/formula/layout
metrics, official leaderboard scores, or full-model BF16 parity. Cancellation
and interleaving stress remain unqualified. MTP is unsupported. Full K/V and
router keys are retained, so total memory grows with document length.
Four-page dynamic-tile prompts (3,581 tokens) exceed the default
`max_batch_tokens` of 2,048.

For operator usage, see [Unlimited-OCR-2 native preview](UNLIMITED-OCR-2.md).
