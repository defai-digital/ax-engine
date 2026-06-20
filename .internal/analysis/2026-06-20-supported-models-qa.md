# Supported Models QA

Date: 2026-06-20

## Scope

Ran live server QA for the remaining direct-support models that did not yet have current QA evidence:

- Gemma 4 E2B 4-bit
- Gemma 4 E4B 4-bit
- Gemma 4 12B 4-bit text and inline-image multimodal chat
- Qwen3-4B 4-bit
- Qwen3.5-9B 4-bit
- Qwen3-Coder-Next 4-bit

This completes the current direct-support model QA set when combined with the existing 2026-06-20 evidence for:

- Gemma 4 26B
- Gemma 4 31B
- Qwen 3.6 27B
- Qwen 3.6 35B
- Qwen 3.6 35B coding/tool-call behavior

## Live QA Results

| Model | Result | Notes |
| --- | --- | --- |
| Gemma 4 E2B 4-bit | Pass after fix | Arithmetic and executable Python passed. Tool calls initially emitted bare Gemma DSL (`call:read_file{path:README.md}`), then passed after parser fix with `content: null` and structured `tool_calls`. |
| Gemma 4 E4B 4-bit | Pass after fix | Arithmetic passed. A short code probe was truncated by the token cap, but the parser/runtime issue was the same bare Gemma DSL tool-call form and passed after fix. |
| Gemma 4 12B 4-bit text | Pass after fix | Arithmetic and executable Python passed. Tool calls passed after the same bare Gemma DSL parser fix. |
| Gemma 4 12B 4-bit multimodal | Pass | Inline PNG OpenAI chat request returned a valid answer after replacing the initial bad-CRC QA fixture with a generated valid PNG. |
| Qwen3-4B 4-bit | Pass with model-discretion caveat | Arithmetic and executable Python passed. `tool_choice: auto` produced prose instead of a tool call; explicit `tool_choice` produced a structured `read_file` call with `content: null`. |
| Qwen3.5-9B 4-bit | Pass | Arithmetic, executable Python, and `tool_choice: auto` structured tool call passed. |
| Qwen3-Coder-Next 4-bit | Pass with model-quality caveat | Arithmetic, executable Python, and structured tool call passed. One `merge_intervals` prompt returned code that solved the output but mutated the input list; classified as model output quality, not a server/runtime bug. |

## Bug Fixed

Gemma 4 native tool-call parsing only accepted the wrapped Ollama/Gemma DSL form:

```text
<|tool_call>call:read_file{path:README.md}<tool_call|>
```

Live Gemma 4 E2B/E4B/12B output emitted the bare form:

```text
call:read_file{path:README.md}
```

The response parser now accepts that bare form when it starts the assistant content, reuses the existing Gemma DSL argument parser, and returns OpenAI-compatible `tool_calls` with `content: null`.

## Verification

- `cargo fmt --check`
- `cargo test -p ax-engine-server openai_responses --quiet`
- `cargo build -p ax-engine-server`
- Live all-model QA harness over ports `18130`-`18135`
- Live Gemma tool-call recheck over ports `18140`-`18142`
- Live Gemma 4 12B valid-PNG multimodal recheck on port `18143`
- Live Qwen3-4B explicit-tool-choice recheck on port `18144`

Live QA required unsandboxed launches because MLX could not access Metal from the sandboxed session.
