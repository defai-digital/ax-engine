"""HTML + JSON report generators for AX Engine QA."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _pass_badge(passed: bool, *, soft: bool = False) -> str:
    if soft and passed:
        return (
            '<span style="background:#64748b;color:#fff;padding:2px 8px;'
            'border-radius:4px;font-size:12px">SOFT OK</span>'
        )
    if soft and not passed:
        return (
            '<span style="background:#a16207;color:#fff;padding:2px 8px;'
            'border-radius:4px;font-size:12px">SOFT</span>'
        )
    if passed:
        return (
            '<span style="background:#16a34a;color:#fff;padding:2px 8px;'
            'border-radius:4px;font-size:12px">PASS</span>'
        )
    return (
        '<span style="background:#dc2626;color:#fff;padding:2px 8px;'
        'border-radius:4px;font-size:12px">FAIL</span>'
    )


def _response_as_dict(resp: Any) -> dict[str, Any]:
    return {
        "text": getattr(resp, "text", "") or "",
        "finish_reason": getattr(resp, "finish_reason", None),
        "prompt_tokens": getattr(resp, "prompt_tokens", 0),
        "completion_tokens": getattr(resp, "completion_tokens", 0),
        "total_tokens": getattr(resp, "total_tokens", 0),
        "ttft_ms": getattr(resp, "ttft_ms", 0.0),
        "elapsed_ms": getattr(resp, "elapsed_ms", 0.0),
        "stream": getattr(resp, "stream", False),
        "error": getattr(resp, "error", None),
    }


def build_results_payload(results: list[dict], metadata: dict) -> dict[str, Any]:
    """Machine-readable suite contract for gates and matrix runners."""
    total = len(results)
    hard_passed = sum(1 for r in results if r["report"].auto_pass)
    hard_failed = total - hard_passed
    soft_failed = 0
    for r in results:
        soft_failed += sum(
            1 for c in r["report"].checks if (not c.hard) and (not c.passed)
        )

    items = []
    for r in results:
        rpt = r["report"]
        resp = r["response"]
        items.append(
            {
                "prompt_id": r["prompt_id"],
                "category": r.get("category", ""),
                "mode_label": r.get("mode", ""),
                "stream": bool(r.get("stream")),
                "hard_pass": rpt.auto_pass,
                "summary": rpt.summary,
                "response": _response_as_dict(resp),
                "report": rpt.as_dict(),
            }
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": 1,
        "generated_at": now,
        "title": metadata.get("title", "AX Engine QA Report"),
        "version": metadata.get("version", "unknown"),
        "commit": metadata.get("commit", "unknown"),
        "model": metadata.get("model", "unknown"),
        "base_url": metadata.get("base_url", ""),
        "seed": metadata.get("seed"),
        "bank_size": metadata.get("bank_size"),
        "sample_size": metadata.get("sample_size", total),
        "sampled_ids": metadata.get("sampled_ids") or [],
        "mode_label": metadata.get("mode_label", "direct"),
        "mode_note": metadata.get(
            "mode_note",
            "mode_label is a report tag; decode path is controlled by server flags",
        ),
        "totals": {
            "items": total,
            "hard_passed": hard_passed,
            "hard_failed": hard_failed,
            "soft_failed_checks": soft_failed,
            "pass_rate": (hard_passed / total * 100.0) if total else 0.0,
        },
        "ok": hard_failed == 0 and total > 0,
        "results": items,
    }


def generate_json_report(results: list[dict], metadata: dict) -> str:
    return json.dumps(build_results_payload(results, metadata), indent=2, ensure_ascii=False)


def generate_html_report(results: list[dict], metadata: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = metadata.get("title", "AX Engine QA Report")
    version = metadata.get("version", "unknown")
    commit = metadata.get("commit", "unknown")
    seed = metadata.get("seed", "n/a")
    bank_size = metadata.get("bank_size", "n/a")
    sample_size = metadata.get("sample_size", len(results))
    sampled_ids = metadata.get("sampled_ids") or []
    mode_label = metadata.get("mode_label", "n/a")
    mode_note = metadata.get(
        "mode_note",
        "Mode is a report label; server flags control direct/ngram/MTP.",
    )

    total = len(results)
    passed = sum(1 for r in results if r["report"].auto_pass)
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    rows = []
    for r in results:
        rpt = r["report"]
        resp = r["response"]
        mode = r["mode"]
        stream = r["stream"]
        category = r.get("category", "")
        preview = _esc(rpt.output_preview[:200])
        checks_html = " | ".join(
            f"{_pass_badge(c.passed, soft=not c.hard)} "
            f"{_esc(c.name)}: {_esc(c.detail)}"
            for c in rpt.checks
        )
        rows.append(
            f"""<tr>
<td>{_esc(r["prompt_id"])}</td>
<td>{_esc(category)}</td>
<td>{_esc(mode)}</td>
<td>{"Yes" if stream else "No"}</td>
<td>{_pass_badge(rpt.auto_pass)}</td>
<td>{_esc(rpt.summary)}</td>
<td>{resp.elapsed_ms:.0f}ms</td>
<td>{_esc(resp.finish_reason or "N/A")}</td>
<td><details><summary>Preview</summary><pre style="max-width:600px;white-space:pre-wrap;font-size:11px">{preview}</pre></details></td>
<td style="font-size:11px">{checks_html}</td>
</tr>"""
        )

    table_rows = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; }}
.summary {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card h3 {{ margin: 0 0 8px; color: #666; font-size: 14px; }}
.card .value {{ font-size: 28px; font-weight: 700; }}
.pass {{ color: #16a34a; }}
.fail {{ color: #dc2626; }}
.note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 10px 14px; margin: 12px 0; border-radius: 4px; font-size: 13px; color: #1e3a5f; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #1a1a2e; color: #fff; padding: 10px 12px; text-align: left; font-size: 13px; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 13px; vertical-align: top; }}
tr:hover {{ background: #f0f4ff; }}
pre {{ margin: 0; }}
details summary {{ cursor: pointer; color: #2563eb; }}
.meta {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>{_esc(title)}</h1>
<div class="meta">Generated: {now} | Version: {_esc(version)} | Commit: {_esc(commit)} | Seed: {_esc(seed)} | Sample: {_esc(sample_size)} / bank {_esc(bank_size)} | Mode label: {_esc(mode_label)} | IDs: {_esc(', '.join(sampled_ids) if sampled_ids else 'n/a')}</div>
<div class="note">{_esc(mode_note)}</div>
<div class="summary">
<div class="card"><h3>Total Tests</h3><div class="value">{total}</div></div>
<div class="card"><h3>Hard Passed</h3><div class="value pass">{passed}</div></div>
<div class="card"><h3>Hard Failed</h3><div class="value fail">{failed}</div></div>
<div class="card"><h3>Hard Pass Rate</h3><div class="value {"pass" if pass_rate >= 80 else "fail"}">{pass_rate:.1f}%</div></div>
</div>
<table>
<thead><tr>
<th>Prompt</th><th>Category</th><th>Mode label</th><th>Stream</th><th>Hard result</th><th>Checks</th><th>Latency</th><th>Finish</th><th>Output</th><th>Details</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>"""
