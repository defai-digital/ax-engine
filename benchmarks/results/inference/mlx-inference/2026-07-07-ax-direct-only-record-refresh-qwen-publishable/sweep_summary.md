# AX-only Qwen publishable overlay summary

- publication_candidate: true
- failed_row_count: 0
- status_counts: ok=4
- completed_row_count: 4/4

This directory excludes the original `qwen3_6-35b-a3b-6bit.json` from the continuation sweep because its recorded `host.performance_conditions.load_average.one_minute` was `2.206`, above the README publication limit of `2.000`. The replacement row comes from the single-row rerun, which recorded `host.performance_conditions.load_average.one_minute=1.395`.

The 4/6-bit rows are retained here as condition-checked rerun evidence, but the README composite high-water merge only publishes cells that match or improve the prior record; lower rerun cells keep the earlier faster artifact.

| slug | status | notes |
|---|---|---|
| qwen3_6-27b-4bit | ok | copied from continuation sweep |
| qwen3_6-27b-6bit | ok | copied from continuation sweep |
| qwen3_6-35b-a3b-4bit | ok | copied from continuation sweep |
| qwen3_6-35b-a3b-6bit | ok | copied from single-row rerun |
