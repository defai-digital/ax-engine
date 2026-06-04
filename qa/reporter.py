"""HTML report generator for AX Engine QA."""

import html
from datetime import datetime


def _esc(s):
    return html.escape(str(s))


def _pass_badge(passed):
    if passed:
        return '<span style="background:#16a34a;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">PASS</span>'
    return '<span style="background:#dc2626;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">FAIL</span>'


def generate_html_report(results, metadata):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = metadata.get("title", "AX Engine QA Report")
    version = metadata.get("version", "unknown")
    commit = metadata.get("commit", "unknown")

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
        preview = _esc(rpt.output_preview[:200])
        checks_html = " | ".join(
            f"{_pass_badge(c.passed)} {_esc(c.name)}: {_esc(c.detail)}"
            for c in rpt.checks
        )
        rows.append(f"""<tr>
<td>{_esc(r["prompt_id"])}</td>
<td>{_esc(mode)}</td>
<td>{"Yes" if stream else "No"}</td>
<td>{_pass_badge(rpt.auto_pass)}</td>
<td>{rpt.summary}</td>
<td>{resp.elapsed_ms:.0f}ms</td>
<td>{_esc(resp.finish_reason or "N/A")}</td>
<td><details><summary>Preview</summary><pre style="max-width:600px;white-space:pre-wrap;font-size:11px">{preview}</pre></details></td>
<td style="font-size:11px">{checks_html}</td>
</tr>""")

    table_rows = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; }}
.summary {{ display: flex; gap: 20px; margin: 20px 0; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card h3 {{ margin: 0 0 8px; color: #666; font-size: 14px; }}
.card .value {{ font-size: 28px; font-weight: 700; }}
.pass {{ color: #16a34a; }}
.fail {{ color: #dc2626; }}
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
<div class="meta">Generated: {now} | Version: {_esc(version)} | Commit: {_esc(commit)}</div>
<div class="summary">
<div class="card"><h3>Total Tests</h3><div class="value">{total}</div></div>
<div class="card"><h3>Passed</h3><div class="value pass">{passed}</div></div>
<div class="card"><h3>Failed</h3><div class="value fail">{failed}</div></div>
<div class="card"><h3>Pass Rate</h3><div class="value {"pass" if pass_rate >= 80 else "fail"}">{pass_rate:.1f}%</div></div>
</div>
<table>
<thead><tr>
<th>Prompt</th><th>Mode</th><th>Stream</th><th>Result</th><th>Checks</th><th>Latency</th><th>Finish</th><th>Output</th><th>Details</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>"""
