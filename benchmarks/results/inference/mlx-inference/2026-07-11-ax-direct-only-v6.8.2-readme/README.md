# README-only AX direct benchmark source projection

This directory is the source projection used by the README performance-artifact checker.

- The 2026-07-11 clean AX-only sweep supplies the current 4/6-bit Gemma 4 and Qwen 3.6 rows.
- Gemma 4 E2B 6-bit uses the condition-checked `clean-r6` rerun.
- Gemma 4 E4B 4-bit uses the condition-checked 2026-07-07 record refresh because the 2026-07-11 host-load gate was not publishable for that row.
- Qwen 3.5 9B, Qwen3-Coder-Next, and GLM 4.7 Flash retain their existing condition-checked full-fresh artifacts.

The files are symlinks to the raw benchmark artifacts; no benchmark values are copied or hand-edited here.
