# Qwen 3.6 35B Coding QA

Date: 2026-06-20

## Scope

Checked Qwen 3.6 35B coding-agent behavior after user reports that coding tasks failed.

Artifacts tested:

- `mlx-community/Qwen3.6-35B-A3B-4bit`
- `mlx-community/Qwen3.6-35B-A3B-6bit`
- `ax-local/Qwen3.6-35B-MTP`

## Live QA Results

| Variant | Result | Notes |
| --- | --- | --- |
| 4-bit | Pass | Executed generated Python for interval merging and bug-fix tasks; strict coding JSON passed; non-streaming tool call parsed; tool-result follow-up produced corrected code; streaming tool call emitted structured tool-call deltas. |
| 6-bit | Reproduced and fixed one client-facing issue | Executable `chunked` implementation passed. Tool-call parsing worked, but the response initially included assistant prose in `message.content` alongside `tool_calls`, which can break strict coding-agent clients. |
| MTP | Pass after fix | Executed generated palindrome implementation; tool-call response returned `content: null` with a structured `read_file` call. |

## Bug Fixed

Parsed OpenAI tool-call responses could preserve model preamble text in `message.content` when a tool call was also present. Some Qwen 3.6 35B coding outputs do this even though the prompt asks the model to reply only with the tool call. This creates a mixed assistant-text-plus-tool-call turn and can fail stricter coding clients.

Fix:

- Clear assistant `content` whenever parsed tool calls are present.
- Keep `finish_reason: "tool_calls"`.
- Rely on the existing serializer to emit empty tool-call content as JSON `null`.
- Updated OpenAI response parser regression tests to enforce tool-call-only content after extraction, including recovered/truncated Qwen tool-call forms.

## Verification

- `cargo fmt --check`
- `cargo test -p ax-engine-server openai_responses --quiet`
- `cargo build -p ax-engine-server`
- Live 4-bit coding QA on port `18120`
- Live 6-bit coding QA and post-fix tool-call normalization on port `18121`
- Live MTP coding QA and post-fix tool-call normalization on port `18122`

Live QA required unsandboxed launches because MLX could not access Metal from the sandboxed session.
