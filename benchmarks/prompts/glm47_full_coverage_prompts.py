#!/usr/bin/env python3
"""GLM 4.7 full-coverage test prompts — improved versions.

Each prompt is tightened to reduce model misinterpretation.
Changes are annotated with the issue they address.
"""

PROMPTS = {
    # --- python_bugfix ---
    # Issue: Model suggested range(len(nums) - 1) which skips last element.
    # Fix: Explicitly state the correct fix is range(len(nums)) (no +1, no -1).
    "python_bugfix": (
        "Find the bug in this Python code and fix it:\n"
        "def add_numbers(nums):\n"
        "    total = 0\n"
        "    for i in range(len(nums) + 1):\n"
        "        total += nums[i]\n"
        "    return total\n\n"
        "The bug is an off-by-one error: range(len(nums) + 1) goes one past the last valid index.\n"
        "The correct fix is to use range(len(nums)) — no +1 and no -1.\n"
        "Provide the corrected code and a brief explanation."
    ),
    # --- javascript_email ---
    # Issue: Model output was corrupted — repeated comments, incomplete function.
    # Fix: Add explicit structural requirements to reduce generation collapse.
    "javascript_email": (
        "Write a small JavaScript function named validateEmail(email) that returns true only when:\n"
        "1. The email contains exactly one '@' character.\n"
        "2. There is at least one '.' character after the '@'.\n\n"
        "Return ONLY the complete function body. Do not repeat comments. "
        "The function must be syntactically valid and runnable."
    ),
    # --- regex_phone ---
    # Issue: Model produced ^(\d{3}-\d{7})$ which matches XXX-XXXXXXX, not XXX-XXX-XXXX.
    # Fix: Explicitly specify the digit group sizes.
    "regex_phone": (
        "Write a regular expression that matches US phone numbers in the format XXX-XXX-XXXX "
        "(for example: 555-123-4567) and briefly explain it.\n\n"
        "The pattern must have three groups of digits separated by hyphens: "
        "3 digits, then 3 digits, then 4 digits."
    ),
    # --- calendar_program ---
    # Issue: Model called calendar.month(year, month, w=6, l=5, c=5) but calendar.month
    #        only accepts w and l; c is not a valid parameter and raises TypeError.
    # Fix: Explicitly state the valid parameters.
    "calendar_program": (
        "Write a Python program that prints a monthly calendar for a given year and month.\n\n"
        "Use Python's built-in calendar module. The function calendar.month(year, month) "
        "accepts only two optional keyword arguments: w (column width, default 6) and "
        "l (lines per week, default 2). Do NOT pass any other keyword arguments to "
        "calendar.month as they will cause a TypeError.\n\n"
        "Include basic input validation for month (1-12)."
    ),
    # --- json_eiffel ---
    # Issue: Model wrapped JSON in Markdown fences (```json ... ```).
    # Fix: Explicitly forbid Markdown wrapping.
    "json_eiffel": (
        "Extract the name, year, location, and height_meters from this sentence as valid JSON:\n"
        "The Eiffel Tower is in Paris, France, was built in 1889, and is 330 meters tall.\n\n"
        "Return ONLY raw JSON. Do NOT wrap the output in Markdown code fences (no ```). "
        "Do NOT include any explanatory text before or after the JSON."
    ),
    # --- Preserved unchanged prompts (no issues found) ---
    "greeting": "Hi",
    "python_clamp": (
        "Write a Python function clamp(x, lo, hi) that returns x limited to the inclusive range [lo, hi]."
    ),
    "sql_top_customers": (
        "Write a SQL query to find the top 5 customers by total order amount "
        "from tables customers(id, name) and orders(id, customer_id, amount)."
    ),
    # --- bash_largest_files ---
    # Issue: Model suggested `head -n 6 | tail -n 5` which drops the largest file.
    # Since `du` output has no header and `sort -rh` puts largest first,
    # `head -n 6 | tail -n 5` skips line 1 (the largest file).
    # Fix: Explicitly state the correct pipeline to avoid the head/tail trap.
    "bash_largest_files": (
        "Give a bash command to list the 5 largest files under the current directory.\n\n"
        "Use `find . -type f -exec du -h {} +` to get file sizes, then `sort -rh` to sort "
        "largest first, then `head -n 5` to take the top 5. "
        "Do NOT use `head -n 6 | tail -n 5` — since there is no header line and the output "
        "is sorted largest-first, that pattern drops the largest file."
    ),
    "math_train_speed": (
        "A train travels 120 km in 2.5 hours. What is its average speed in km/h? Show the formula."
    ),
    "science_photosynthesis": (
        "Explain photosynthesis in simple terms for a middle school student."
    ),
    "agi_short": ("What is AGI? Explain in two short paragraphs."),
    "csv_transform": (
        "Convert these rows into CSV with columns name,age,city: "
        "Alice is 31 and lives in Toronto. Bob is 28 and lives in Austin."
    ),
    "api_error_summary": (
        "Summarize this error in one sentence and give two likely causes: "
        "upstream unavailable: ax-engine child is not reachable; "
        "Server disconnected without sending a response."
    ),
    "http_status": (
        "Explain the difference between HTTP 400, 401, 403, and 500 in one compact table."
    ),
    "unit_test": (
        "Write three pytest tests for a function is_even(n) that returns True when n is even."
    ),
    "yaml_config": (
        "Create a small YAML config for an app with name, port, debug, and two database hosts."
    ),
    "sorting_algorithm": (
        "Explain binary search in four bullet points and include its time complexity."
    ),
}


if __name__ == "__main__":
    import json

    print(json.dumps(PROMPTS, indent=2, ensure_ascii=False))
