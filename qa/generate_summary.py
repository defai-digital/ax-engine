#!/usr/bin/env python3
"""Generate QA summary page linking all reports."""
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_report_info(report_path: Path) -> dict:
    """Extract key info from report HTML."""
    content = report_path.read_text()
    
    # Extract model name from filename
    # Format: qa-{model_id}-{mode}-{timestamp}.html
    match = re.match(r"qa-(.+?)-(direct|ngram)-(\d{8}-\d{6})\.html", report_path.name)
    if not match:
        return None
    
    model_id = match.group(1)
    mode = match.group(2)
    timestamp = match.group(3)
    
    # Extract pass rate from HTML
    pass_rate_match = re.search(r'<div class="value [^"]*">(\d+\.?\d*)%</div>', content)
    pass_rate = pass_rate_match.group(1) + "%" if pass_rate_match else "N/A"
    
    # Extract passed/failed counts
    passed_match = re.search(r'<div class="value pass">(\d+)</div>', content)
    failed_match = re.search(r'<div class="value fail">(\d+)</div>', content)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    total = passed + failed
    
    return {
        "model_id": model_id,
        "mode": mode,
        "timestamp": timestamp,
        "pass_rate": pass_rate,
        "passed": passed,
        "failed": failed,
        "total": total,
        "filename": report_path.name,
    }


def generate_summary_html(reports: list[dict]) -> str:
    """Generate summary HTML page."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Group by model. The input list is already de-duplicated to latest
    # model/mode reports, so totals reflect the current run set.
    models = {}
    for r in reports:
        mid = r["model_id"]
        if mid not in models:
            models[mid] = {"direct": None, "ngram": None}
        models[mid][r["mode"]] = r
    
    # Build table rows
    rows = []
    for model_id in sorted(models.keys()):
        info = models[model_id]
        direct = info.get("direct")
        ngram = info.get("ngram")
        
        direct_link = f'<a href="{direct["filename"]}">{direct["pass_rate"]}</a>' if direct else "N/A"
        ngram_link = f'<a href="{ngram["filename"]}">{ngram["pass_rate"]}</a>' if ngram else "N/A"
        
        direct_detail = f'{direct["passed"]}/{direct["total"]}' if direct else "-"
        ngram_detail = f'{ngram["passed"]}/{ngram["total"]}' if ngram else "-"
        
        # Check parity
        if direct and ngram:
            parity = "✅ Identical" if direct["passed"] == ngram["passed"] else "⚠️ Differs"
        else:
            parity = "-"
        
        rows.append(f"""<tr>
<td><strong>{model_id}</strong></td>
<td>{direct_link}</td>
<td>{direct_detail}</td>
<td>{ngram_link}</td>
<td>{ngram_detail}</td>
<td>{parity}</td>
</tr>""")
    
    table_rows = "\n".join(rows)
    
    # Calculate overall stats
    total_tests = sum(r["total"] for r in reports)
    total_passed = sum(r["passed"] for r in reports)
    overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AX Engine QA Summary</title>
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
th {{ background: #1a1a2e; color: #fff; padding: 12px 16px; text-align: left; font-size: 14px; }}
td {{ padding: 12px 16px; border-bottom: 1px solid #eee; font-size: 14px; }}
tr:hover {{ background: #f0f4ff; }}
a {{ color: #2563eb; text-decoration: none; font-weight: 600; }}
a:hover {{ text-decoration: underline; }}
.meta {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
.note {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; margin: 20px 0; border-radius: 4px; }}
</style>
</head>
<body>
<h1>AX Engine QA Summary</h1>
<div class="meta">Generated: {now} | Reports: {len(reports)} | Models: {len(models)}</div>

<div class="summary">
<div class="card"><h3>Total Tests</h3><div class="value">{total_tests}</div></div>
<div class="card"><h3>Total Passed</h3><div class="value pass">{total_passed}</div></div>
<div class="card"><h3>Total Failed</h3><div class="value fail">{total_tests - total_passed}</div></div>
<div class="card"><h3>Overall Pass Rate</h3><div class="value {'pass' if overall_rate >= 70 else 'fail'}">{overall_rate:.1f}%</div></div>
</div>

<table>
<thead>
<tr>
<th>Model</th>
<th>Direct Pass Rate</th>
<th>Direct Detail</th>
<th>N-gram Pass Rate</th>
<th>N-gram Detail</th>
<th>Direct vs N-gram</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2 style="margin-top: 40px;">All Reports</h2>
<ul>
{"".join(f'<li><a href="{r["filename"]}">{r["filename"]}</a> — {r["model_id"]} ({r["mode"]}) — {r["pass_rate"]}</li>' for r in sorted(reports, key=lambda x: (x["model_id"], x["mode"])))}
</ul>

</body>
</html>"""


def main():
    reports_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports")
    
    if not reports_dir.exists():
        print(f"Reports directory not found: {reports_dir}")
        sys.exit(1)
    
    # Find latest valid QA report per model/mode.
    reports_by_key = {}
    for f in reports_dir.glob("qa-*.html"):
        if f.stat().st_size < 15000:  # Skip old/broken reports
            continue
        info = parse_report_info(f)
        if info:
            key = (info["model_id"], info["mode"])
            current = reports_by_key.get(key)
            if current is None or info["timestamp"] > current["timestamp"]:
                reports_by_key[key] = info
    reports = list(reports_by_key.values())
    
    if not reports:
        print("No valid QA reports found")
        sys.exit(1)
    
    print(f"Found {len(reports)} QA reports")
    
    html = generate_summary_html(reports)
    output_path = reports_dir / "summary.html"
    output_path.write_text(html)
    print(f"Summary page generated: {output_path}")


if __name__ == "__main__":
    main()
