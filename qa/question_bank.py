"""Large QA question bank for AX Engine quality tests.

Inspired by common public eval *dimensions* (not copied item-for-item from
copyrighted benchmarks): instruction following, math word problems, coding
tasks, logical reasoning, short knowledge/common-sense checks, JSON format
discipline, reading comprehension, translation, creative writing, and unit-test
generation.

The bank is intentionally larger than a single run. Use `prompts.sample_prompts`
so each QA session draws a stratified subset — this reduces overfitting to a
fixed dozen prompts and makes “memorized golden answers” less useful.
"""

from __future__ import annotations

from prompts import QaPrompt

# ---------------------------------------------------------------------------
# Bank — keep items independent, short, and auto-checkable where possible.
# Prefer exact_answer / regex / structural checks over soft keyword matching.
# ---------------------------------------------------------------------------

QUESTION_BANK: list[QaPrompt] = [
    # ----- reasoning -----
    QaPrompt(
        id="reasoning_syllogism_roses",
        category="reasoning",
        system=None,
        user=(
            "If all roses are flowers and some flowers fade quickly, can we conclude "
            "that some roses fade quickly? Answer yes or no first, then explain."
        ),
        keywords=["no", "rose", "flower"],
        exact_answer="no",
        min_length=40,
        description="Classic invalid syllogism (answer is no)",
    ),
    QaPrompt(
        id="reasoning_syllogism_cats",
        category="reasoning",
        system=None,
        user=(
            "All cats are mammals. All mammals are animals. Are all cats animals? "
            "Answer yes or no, then one sentence of reasoning."
        ),
        keywords=["yes", "cat", "animal"],
        exact_answer="yes",
        min_length=20,
        description="Valid syllogism",
    ),
    QaPrompt(
        id="reasoning_analogy_teacher",
        category="reasoning",
        system=None,
        user="Complete the analogy: Doctor is to Hospital as Teacher is to ___. Explain briefly.",
        keywords=["school", "teacher"],
        min_length=25,
        description="Workplace analogy",
    ),
    QaPrompt(
        id="reasoning_cause_effect",
        category="reasoning",
        system=None,
        user=(
            "A river floods after heavy rain. Which is more likely the cause: "
            "(A) the rain (B) people buying umbrellas? Answer A or B and explain."
        ),
        keywords=["rain", "flood"],
        exact_answer="a",
        min_length=30,
        description="Cause vs correlated event",
    ),
    QaPrompt(
        id="reasoning_set_membership",
        category="reasoning",
        system=None,
        user=(
            "Set A = {1, 2, 3}. Set B = {2, 3, 4}. List the intersection A ∩ B as numbers "
            "separated by commas only."
        ),
        keywords=["2", "3"],
        regex_patterns=[r"2\s*,\s*3|3\s*,\s*2"],
        min_length=3,
        description="Set intersection",
    ),
    QaPrompt(
        id="reasoning_false_premise",
        category="reasoning",
        system=None,
        user=(
            "Premise: Every bird can fly. Penguin is a bird. Conclusion: Penguin can fly. "
            "Is the reasoning valid given the premises, and is the conclusion true in reality? "
            "Answer in two short sentences."
        ),
        keywords=["valid", "true", "false", "penguin", "fly"],
        min_length=40,
        description="Validity vs truth",
    ),
    # ----- math -----
    QaPrompt(
        id="math_apples_store",
        category="math",
        system=None,
        user=(
            "A store has 47 apples. They sell 18 in the morning and 12 in the afternoon. "
            "Then they receive a delivery of 30 more. How many apples do they have now? "
            "Show work, then put the final integer on its own last line."
        ),
        keywords=["47", "18", "12", "30"],
        exact_answer="47",
        min_length=20,
        description="47-18-12+30 = 47",
    ),
    QaPrompt(
        id="math_percent_discount",
        category="math",
        system=None,
        user=(
            "A jacket costs $80. It is discounted by 25%. What is the sale price in dollars? "
            "Reply with only the number."
        ),
        exact_answer="60",
        min_length=1,
        description="80 * 0.75 = 60",
    ),
    QaPrompt(
        id="math_average_scores",
        category="math",
        system=None,
        user=(
            "Scores: 70, 80, 90. What is the arithmetic mean? Reply with only the number."
        ),
        exact_answer="80",
        min_length=1,
        description="Simple mean",
    ),
    QaPrompt(
        id="math_train_speed",
        category="math",
        system=None,
        user=(
            "A train travels 120 km in 2 hours at constant speed. How many km does it travel "
            "in 5 hours at the same speed? Final answer as an integer."
        ),
        exact_answer="300",
        min_length=1,
        description="Speed × time",
    ),
    QaPrompt(
        id="math_lcm_small",
        category="math",
        system=None,
        user="What is the least common multiple of 4 and 6? Reply with only the number.",
        exact_answer="12",
        min_length=1,
        description="LCM(4,6)=12",
    ),
    QaPrompt(
        id="math_fraction_half",
        category="math",
        system=None,
        user="What is 1/2 + 1/4? Reply as a simplified fraction like a/b only.",
        exact_answer="3/4",
        min_length=3,
        description="Fraction addition",
    ),
    # ----- code -----
    QaPrompt(
        id="code_python_email",
        category="code",
        system="You are a helpful coding assistant.",
        user=(
            "Write a Python function that checks if a string is a valid email address. "
            "Include a docstring and type hints."
        ),
        keywords=["def", "return", "str"],
        regex_patterns=[r"def\s+\w+\s*\("],
        min_length=80,
        description="Python email validation",
    ),
    QaPrompt(
        id="code_python_palindrome",
        category="code",
        system="You are a helpful coding assistant.",
        user=(
            "Write a Python function is_palindrome(s: str) -> bool that ignores spaces "
            "and letter case."
        ),
        keywords=["def", "return"],
        regex_patterns=[r"def\s+is_palindrome\s*\("],
        min_length=60,
        description="Python palindrome checker",
    ),
    QaPrompt(
        id="code_javascript_flatten",
        category="code",
        system="You are a helpful coding assistant.",
        user=(
            "Write a JavaScript function that flattens a nested array to any depth. "
            "Use modern ES6+ syntax."
        ),
        keywords=["function", "return", "array"],
        regex_patterns=[r"(function|const|=>)"],
        min_length=60,
        description="JS flatten array",
    ),
    QaPrompt(
        id="code_sql_top_customers",
        category="code",
        system="You are a database expert.",
        user=(
            "Write a SQL query to find the top 3 customers by total order amount from "
            "tables customers (id, name) and orders (id, customer_id, amount). "
            "Use GROUP BY and ORDER BY."
        ),
        keywords=["SELECT", "GROUP BY", "ORDER BY"],
        regex_patterns=[r"SELECT", r"GROUP\s+BY", r"ORDER\s+BY"],
        min_length=50,
        description="SQL aggregation top-N",
    ),
    QaPrompt(
        id="code_python_fizzbuzz",
        category="code",
        system="You are a helpful coding assistant.",
        user=(
            "Write a Python function fizzbuzz(n: int) -> list[str] returning strings for "
            "1..n with Fizz/Buzz/FizzBuzz rules."
        ),
        keywords=["def", "fizz", "buzz"],
        regex_patterns=[r"def\s+fizzbuzz\s*\("],
        min_length=80,
        description="Classic FizzBuzz",
    ),
    QaPrompt(
        id="code_bash_count_lines",
        category="code",
        system=None,
        user=(
            "Write a one-line bash command that counts non-empty lines in file notes.txt. "
            "Output only the command."
        ),
        keywords=["notes.txt"],
        regex_patterns=[r"grep|awk|sed|wc"],
        min_length=10,
        description="Bash one-liner",
    ),
    # ----- instruction following -----
    QaPrompt(
        id="instruction_numbered_countries",
        category="instruction",
        system=None,
        user=(
            "List exactly 5 countries in South America. Format as a numbered list. "
            "Do not add any extra text before or after the list."
        ),
        keywords=["1.", "2.", "3.", "4.", "5."],
        regex_patterns=[r"1\.\s+\w+", r"5\.\s+\w+"],
        min_length=30,
        description="Numbered list constraint",
    ),
    QaPrompt(
        id="instruction_three_colors",
        category="instruction",
        system=None,
        user=(
            "Output exactly these three color words in this order, lowercase, "
            "separated by single spaces, and nothing else: red green blue"
        ),
        exact_answer="red green blue",
        min_length=13,
        max_repetition_ratio=0.5,
        description="Exact three-token instruction following",
    ),
    QaPrompt(
        id="instruction_json_only_true",
        category="instruction",
        system=None,
        user='Reply with only the JSON value true (lowercase), with no markdown and no other text.',
        exact_answer="true",
        min_length=4,
        description="Strict JSON literal",
    ),
    QaPrompt(
        id="instruction_bullet_three",
        category="instruction",
        system=None,
        user=(
            "Give exactly three bullet points about water conservation. "
            "Each bullet must start with '- '."
        ),
        regex_patterns=[r"(?m)^-\s+\S+"],
        min_length=40,
        description="Bullet format",
    ),
    QaPrompt(
        id="instruction_no_digit",
        category="instruction",
        system=None,
        user=(
            "Describe the color of a clear daytime sky in one sentence. "
            "Do not use any digit characters 0-9."
        ),
        keywords=["blue", "sky"],
        regex_patterns=[r"^[^0-9]+$"],
        min_length=15,
        description="Constraint: no digits",
    ),
    QaPrompt(
        id="instruction_uppercase_yes",
        category="instruction",
        system=None,
        user='Answer the question "Is ice usually cold?" with exactly YES or NO in uppercase only.',
        exact_answer="YES",
        min_length=2,
        description="Case-constrained yes/no",
    ),
    # ----- creative -----
    QaPrompt(
        id="creative_robot_music",
        category="creative",
        system=None,
        user="Write a short paragraph (3-4 sentences) about a robot discovering music for the first time.",
        keywords=["robot", "music", "sound"],
        min_length=80,
        max_repetition_ratio=0.15,
        description="Creative paragraph",
    ),
    QaPrompt(
        id="creative_haiku_rain",
        category="creative",
        system=None,
        user="Write one English haiku about rain. Three lines only.",
        keywords=["rain"],
        min_length=20,
        max_repetition_ratio=0.2,
        description="Haiku form",
    ),
    QaPrompt(
        id="creative_product_tagline",
        category="creative",
        system=None,
        user="Write a 6-word product tagline for a reusable water bottle.",
        min_length=10,
        max_repetition_ratio=0.3,
        description="Short marketing line",
    ),
    QaPrompt(
        id="creative_dialogue",
        category="creative",
        system=None,
        user=(
            "Write a 4-line dialogue between a lighthouse keeper and a sailor. "
            "Prefix each line with Keeper: or Sailor:."
        ),
        regex_patterns=[r"Keeper:", r"Sailor:"],
        min_length=40,
        description="Formatted dialogue",
    ),
    # ----- knowledge / common sense -----
    QaPrompt(
        id="knowledge_capital_france",
        category="knowledge",
        system=None,
        user="What is the capital city of France? Reply with only the city name.",
        exact_answer="Paris",
        min_length=4,
        description="Basic geography",
    ),
    QaPrompt(
        id="knowledge_water_formula",
        category="knowledge",
        system=None,
        user="What is the chemical formula for water? Reply with only the formula.",
        exact_answer="H2O",
        min_length=3,
        description="Basic chemistry",
    ),
    QaPrompt(
        id="knowledge_planet_count",
        category="knowledge",
        system=None,
        user=(
            "How many planets are in the Solar System under the IAU definition after 2006? "
            "Reply with only the number."
        ),
        exact_answer="8",
        min_length=1,
        description="Astronomy fact",
    ),
    QaPrompt(
        id="common_sense_umbrella",
        category="common_sense",
        system=None,
        user=(
            "You are going outside and it is raining heavily. Which is more useful: "
            "an umbrella or a swimming pool? Answer with one word: umbrella or pool."
        ),
        exact_answer="umbrella",
        min_length=6,
        description="Common-sense tool choice",
    ),
    QaPrompt(
        id="common_sense_ice_melt",
        category="common_sense",
        system=None,
        user=(
            "Ice cubes are left on a warm kitchen counter. What usually happens after an hour? "
            "One short sentence."
        ),
        keywords=["melt", "water", "liquid"],
        min_length=15,
        description="Physical common sense",
    ),
    QaPrompt(
        id="knowledge_http_get",
        category="knowledge",
        system=None,
        user=(
            "In HTTP, which method is typically used to retrieve a resource without a body? "
            "Reply with only the method name in uppercase."
        ),
        exact_answer="GET",
        min_length=3,
        description="Web basics",
    ),
    # ----- science / reading -----
    QaPrompt(
        id="science_photosynthesis_one",
        category="science",
        system=None,
        user=(
            "In one sentence, what do plants produce in photosynthesis that animals breathe?"
        ),
        keywords=["oxygen", "o2"],
        min_length=15,
        description="Photosynthesis product",
    ),
    QaPrompt(
        id="science_boiling_celsius",
        category="science",
        system=None,
        user=(
            "At standard atmospheric pressure, water boils at what temperature in Celsius? "
            "Reply with only the number."
        ),
        exact_answer="100",
        min_length=2,
        description="Boiling point",
    ),
    QaPrompt(
        id="reading_comprehension_short",
        category="reading",
        system=None,
        user=(
            "Passage: Maya left the library at noon with three books about oceans. "
            "She rode the bus home and started reading the thickest book first.\n"
            "Question: How many books did Maya take from the library? Reply with only the number."
        ),
        exact_answer="3",
        min_length=1,
        description="Short reading comprehension",
    ),
    QaPrompt(
        id="reading_who_arrived",
        category="reading",
        system=None,
        user=(
            "Passage: Sam arrived before Taylor. Taylor arrived before Uma.\n"
            "Who arrived first? Reply with only the name."
        ),
        exact_answer="Sam",
        min_length=3,
        description="Ordering comprehension",
    ),
    # ----- translation -----
    QaPrompt(
        id="translation_fr_weather",
        category="translation",
        system=None,
        user=(
            "Translate to French: 'The weather is beautiful today and I would like to go "
            "for a walk in the park.'"
        ),
        keywords=["aujourd", "parc", "beau", "temps", "promenade", "marche"],
        min_length=20,
        description="EN→FR everyday sentence",
    ),
    QaPrompt(
        id="translation_es_hello",
        category="translation",
        system=None,
        user="Translate to Spanish: 'Good morning, how are you?' Reply with only the translation.",
        keywords=["buenos", "días", "como", "estás", "estas"],
        min_length=8,
        description="EN→ES greeting",
    ),
    QaPrompt(
        id="translation_de_thanks",
        category="translation",
        system=None,
        user="Translate to German: 'Thank you very much.' Reply with only the translation.",
        keywords=["danke"],
        min_length=5,
        description="EN→DE courtesy",
    ),
    # ----- json / structured -----
    QaPrompt(
        id="json_invoice_nested",
        category="json",
        system=None,
        user=(
            "Return only valid JSON (no markdown fences, no prose) for this invoice: "
            "invoice AX-1042, customer Mina, currency USD, items: cable quantity 2 "
            "unit_price 4.25, dock quantity 1 unit_price 31.00. "
            "Include invoice_id, customer, currency, items, and total "
            "(total must equal sum of quantity*unit_price)."
        ),
        keywords=["AX-1042", "Mina"],
        regex_patterns=[r'"invoice_id"', r'"items"'],
        min_length=60,
        json_expected_total=39.5,
        description="Nested invoice JSON with total 39.50",
    ),
    QaPrompt(
        id="json_user_record",
        category="json",
        system=None,
        user=(
            'Return only JSON: {"name":"Alex","age":30,"active":true} but change age to 31. '
            "No markdown fences."
        ),
        keywords=["Alex", "31"],
        regex_patterns=[r'"age"\s*:\s*31'],
        min_length=20,
        description="JSON edit task",
    ),
    QaPrompt(
        id="json_array_primes",
        category="json",
        system=None,
        user=(
            "Return only a JSON array of the first five prime numbers as integers, "
            "no markdown."
        ),
        regex_patterns=[r"\[\s*2\s*,\s*3\s*,\s*5\s*,\s*7\s*,\s*11\s*\]"],
        min_length=10,
        description="JSON array of primes",
    ),
    # ----- testing / summarization / format stress -----
    QaPrompt(
        id="unit_test_is_even",
        category="testing",
        system=None,
        user=(
            "Write three pytest tests for a function is_even(n) that returns True when n is even."
        ),
        keywords=["assert", "is_even"],
        regex_patterns=[r"def test_\w+\("],
        min_length=100,
        min_test_count=3,
        description="Three pytest tests",
    ),
    QaPrompt(
        id="summarization_ml_two",
        category="summarization",
        system=None,
        user="Summarize the concept of machine learning in exactly 2 sentences.",
        keywords=["machine learning", "data", "model", "learn"],
        min_length=40,
        max_repetition_ratio=0.1,
        description="Two-sentence summary",
    ),
    QaPrompt(
        id="summarization_http_one",
        category="summarization",
        system=None,
        user="In one sentence, summarize what HTTP is used for on the web.",
        keywords=["http", "web", "request", "protocol", "browser", "server"],
        min_length=25,
        description="One-sentence summary",
    ),
    QaPrompt(
        id="format_csv_three",
        category="format",
        system=None,
        user=(
            "Output exactly three CSV lines with header name,score then two data rows: "
            "Ada,95 and Lin,88. No other text."
        ),
        regex_patterns=[r"name\s*,\s*score", r"Ada\s*,\s*95", r"Lin\s*,\s*88"],
        min_length=20,
        description="Strict CSV format",
    ),
    QaPrompt(
        id="format_xml_tag",
        category="format",
        system=None,
        user='Return only this XML: <ok status="1"/> with no other characters.',
        exact_answer='<ok status="1"/>',
        min_length=10,
        description="Exact XML fragment",
    ),
    # ----- multi-step / agentic-style (still single-turn) -----
    QaPrompt(
        id="reasoning_multi_step_ages",
        category="reasoning",
        system=None,
        user=(
            "Ana is 3 years older than Bo. Bo is 10. How old is Ana? "
            "Reply with only the number."
        ),
        exact_answer="13",
        min_length=1,
        description="Simple multi-step ages",
    ),
    QaPrompt(
        id="math_work_rate",
        category="math",
        system=None,
        user=(
            "A machine makes 12 widgets per hour. How many widgets in 3.5 hours? "
            "Reply with only the number."
        ),
        exact_answer="42",
        min_length=1,
        description="Rate × time",
    ),
    QaPrompt(
        id="code_python_sum_list",
        category="code",
        system="You are a helpful coding assistant.",
        user=(
            "Write a Python function sum_positive(nums: list[int]) -> int that sums only "
            "positive integers in nums."
        ),
        regex_patterns=[r"def\s+sum_positive\s*\("],
        keywords=["return"],
        min_length=50,
        description="Filter-and-sum function",
    ),
    QaPrompt(
        id="instruction_alphabet_first",
        category="instruction",
        system=None,
        user="List the first five letters of the English alphabet separated by commas only.",
        exact_answer="a,b,c,d,e",
        min_length=9,
        description="Ordered list constraint",
    ),
    QaPrompt(
        id="knowledge_binary_bit",
        category="knowledge",
        system=None,
        user="How many bits are in one byte by standard convention? Reply with only the number.",
        exact_answer="8",
        min_length=1,
        description="CS fundamentals",
    ),
    QaPrompt(
        id="common_sense_refrigerator",
        category="common_sense",
        system=None,
        user=(
            "You need to keep milk cold overnight without electricity. Which is better: "
            "a cooler with ice packs, or an open window in summer? Answer cooler or window."
        ),
        exact_answer="cooler",
        min_length=5,
        description="Practical common sense",
    ),
    QaPrompt(
        id="reading_color_ball",
        category="reading",
        system=None,
        user=(
            "Passage: The red ball is under the table. The blue ball is on the shelf.\n"
            "Where is the red ball? One short phrase."
        ),
        keywords=["under", "table"],
        min_length=8,
        description="Locate detail in passage",
    ),
    QaPrompt(
        id="json_boolean_flags",
        category="json",
        system=None,
        user=(
            'Return only JSON object {"ready":true,"retries":0} with those exact keys and values.'
        ),
        regex_patterns=[r'"ready"\s*:\s*true', r'"retries"\s*:\s*0'],
        min_length=15,
        description="Boolean/int JSON object",
    ),
    QaPrompt(
        id="instruction_repeat_token",
        category="instruction",
        system=None,
        user='Output the word "ping" exactly four times separated by single spaces, nothing else.',
        exact_answer="ping ping ping ping",
        min_length=19,
        description="Controlled repetition (not garbage)",
    ),
    QaPrompt(
        id="science_gravity_earth",
        category="science",
        system=None,
        user=(
            "Approximate surface gravity on Earth in m/s^2 is often taught as what integer? "
            "Reply with only the number."
        ),
        exact_answer="10",
        exact_answer_aliases=["9.8", "9.81", "10"],
        min_length=1,
        description="g ≈ 9.8/10",
    ),
    QaPrompt(
        id="translation_fr_thanks",
        category="translation",
        system=None,
        user="Translate to French: 'Thank you.' Reply with only the translation.",
        keywords=["merci"],
        min_length=4,
        description="EN→FR short",
    ),
    QaPrompt(
        id="creative_two_sentence_myth",
        category="creative",
        system=None,
        user="In exactly two sentences, invent a tiny myth about why the moon has craters.",
        min_length=40,
        max_repetition_ratio=0.15,
        description="Two-sentence myth",
    ),
    QaPrompt(
        id="math_modulo",
        category="math",
        system=None,
        user="What is 17 mod 5? Reply with only the number.",
        exact_answer="2",
        min_length=1,
        description="Modulo",
    ),
    QaPrompt(
        id="code_sql_join",
        category="code",
        system="You are a database expert.",
        user=(
            "Write SQL that lists employee names and department names from employees "
            "(id, name, department_id) and departments (id, name) using an INNER JOIN."
        ),
        keywords=["JOIN", "SELECT"],
        regex_patterns=[r"JOIN", r"SELECT"],
        min_length=40,
        description="SQL join",
    ),
    QaPrompt(
        id="reasoning_odd_one_out",
        category="reasoning",
        system=None,
        user=(
            "Which does not belong: apple, banana, carrot, grape? "
            "Answer with the odd item only."
        ),
        exact_answer="carrot",
        min_length=5,
        description="Category odd-one-out",
    ),
    QaPrompt(
        id="unit_test_clamp",
        category="testing",
        system=None,
        user=(
            "Write at least two pytest tests for clamp(x, lo, hi) that returns x limited to [lo, hi]."
        ),
        regex_patterns=[r"def test_\w+\("],
        min_test_count=2,
        keywords=["assert", "clamp"],
        min_length=60,
        description="Two clamp unit tests",
    ),
    QaPrompt(
        id="instruction_sort_numbers",
        category="instruction",
        system=None,
        user="Sort these numbers ascending and reply as comma-separated values only: 9, 2, 7, 2",
        exact_answer="2,2,7,9",
        exact_answer_aliases=["2, 2, 7, 9"],
        min_length=7,
        description="Sort with duplicates",
    ),
    QaPrompt(
        id="knowledge_python_list",
        category="knowledge",
        system=None,
        user=(
            "In Python 3, what built-in type is created by the literal [1, 2, 3]? "
            "Reply with only the type name in lowercase."
        ),
        exact_answer="list",
        min_length=4,
        description="Python type name",
    ),
    QaPrompt(
        id="common_sense_sleep",
        category="common_sense",
        system=None,
        user=(
            "A person has not slept in 48 hours. Are they more likely to be alert or tired? "
            "One word: alert or tired."
        ),
        exact_answer="tired",
        min_length=4,
        description="Fatigue common sense",
    ),
    QaPrompt(
        id="reading_count_cats",
        category="reading",
        system=None,
        user=(
            "Passage: Two cats sat on the fence. A third cat joined them later.\n"
            "How many cats are on the fence at the end? Number only."
        ),
        exact_answer="3",
        min_length=1,
        description="Count entities",
    ),
    QaPrompt(
        id="format_markdown_heading",
        category="format",
        system=None,
        user=(
            "Output only a markdown H1 heading with the text Release Notes "
            "(the line must start with #)."
        ),
        regex_patterns=[r"(?m)^#\s+Release Notes\s*$"],
        min_length=5,
        description="Markdown H1 only",
    ),
    QaPrompt(
        id="summarization_tcp_one",
        category="summarization",
        system=None,
        user="In one sentence, summarize what TCP provides compared to UDP at a high level.",
        keywords=["reliable", "connection", "order", "tcp", "udp", "packet"],
        min_length=30,
        description="Networking summary",
    ),
    QaPrompt(
        id="math_negative_sum",
        category="math",
        system=None,
        user="Compute -3 + 11. Reply with only the number.",
        exact_answer="8",
        min_length=1,
        description="Signed arithmetic",
    ),
    QaPrompt(
        id="code_python_dict_get",
        category="code",
        system="You are a helpful coding assistant.",
        user=(
            "Write a Python one-liner expression that safely gets key 'id' from dict d "
            "with default 0 using .get."
        ),
        regex_patterns=[r"\.get\s*\(\s*['\"]id['\"]"],
        min_length=8,
        description="dict.get usage",
    ),
    QaPrompt(
        id="instruction_exclude_word",
        category="instruction",
        system=None,
        user=(
            "Describe a bicycle in one sentence without using the words wheel or wheels."
        ),
        keywords=["bike", "bicycle", "ride", "pedal", "transport"],
        regex_patterns=[r"(?i)^(?!.*\bwheels?\b).+$"],
        min_length=20,
        description="Forbidden-word constraint",
    ),
    QaPrompt(
        id="json_nested_address",
        category="json",
        system=None,
        user=(
            'Return only JSON: {"user":{"name":"Bo","city":"Oslo"}} with those exact values.'
        ),
        regex_patterns=[r'"name"\s*:\s*"Bo"', r'"city"\s*:\s*"Oslo"'],
        min_length=20,
        description="Nested JSON object",
    ),
    QaPrompt(
        id="reasoning_transitivity",
        category="reasoning",
        system=None,
        user=(
            "If A is taller than B and B is taller than C, is A taller than C? "
            "Yes or no only."
        ),
        exact_answer="yes",
        min_length=2,
        description="Transitive relation",
    ),
    QaPrompt(
        id="science_states_of_matter",
        category="science",
        system=None,
        user=(
            "Name the three classical states of matter in English, comma-separated, "
            "in any order."
        ),
        keywords=["solid", "liquid", "gas"],
        min_length=15,
        description="States of matter",
    ),
    QaPrompt(
        id="creative_alliteration",
        category="creative",
        system=None,
        user="Write one alliterative sentence about seven silver seals.",
        keywords=["silver", "seal"],
        min_length=15,
        max_repetition_ratio=0.25,
        description="Alliteration prompt",
    ),
    QaPrompt(
        id="knowledge_html_anchor",
        category="knowledge",
        system=None,
        user=(
            "In HTML, which tag name is used for a hyperlink? "
            "Reply with only the tag name without angle brackets."
        ),
        exact_answer="a",
        min_length=1,
        description="HTML basics",
    ),
]


def bank_size() -> int:
    return len(QUESTION_BANK)


def all_bank_categories() -> list[str]:
    return sorted({p.category for p in QUESTION_BANK})
