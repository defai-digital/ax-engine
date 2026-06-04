"""Test prompt suite for AX Engine QA — covers reasoning, code, creative, instruction-following, math."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QaPrompt:
    id: str
    category: str
    system: Optional[str]
    user: str
    keywords: list[str] = field(default_factory=list)
    regex_patterns: list[str] = field(default_factory=list)
    min_length: int = 20
    max_repetition_ratio: float = 0.3
    description: str = ""


PROMPTS: list[QaPrompt] = [
    QaPrompt(
        id="reasoning_logic",
        category="reasoning",
        system=None,
        user="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning step by step.",
        keywords=["rose", "flower", "fade", "conclude", "some"],
        min_length=50,
        description="Logical syllogism reasoning",
    ),
    QaPrompt(
        id="code_python",
        category="code",
        system="You are a helpful coding assistant.",
        user="Write a Python function that checks if a string is a valid email address. Include docstring and type hints.",
        keywords=["def", "return", "str", "@"],
        regex_patterns=[r"def\s+\w+\s*\("],
        min_length=80,
        description="Python email validation function",
    ),
    QaPrompt(
        id="code_javascript",
        category="code",
        system="You are a helpful coding assistant.",
        user="Write a JavaScript function that flattens a nested array to any depth. Use modern ES6+ syntax.",
        keywords=["function", "return", "array", "flat"],
        regex_patterns=[r"(function|const|=>)"],
        min_length=60,
        description="JavaScript array flattening",
    ),
    QaPrompt(
        id="creative_story",
        category="creative",
        system=None,
        user="Write a short paragraph (3-4 sentences) about a robot discovering music for the first time.",
        keywords=["robot", "music", "sound"],
        min_length=80,
        max_repetition_ratio=0.15,
        description="Creative writing — robot discovers music",
    ),
    QaPrompt(
        id="instruction_format",
        category="instruction",
        system=None,
        user="List exactly 5 countries in South America. Format as a numbered list. Do not add any extra text.",
        keywords=["1.", "2.", "3.", "4.", "5."],
        regex_patterns=[r"1\.\s+\w+", r"5\.\s+\w+"],
        min_length=30,
        description="Exact instruction following — numbered list",
    ),
    QaPrompt(
        id="math_arithmetic",
        category="math",
        system=None,
        user="A store has 47 apples. They sell 18 in the morning and 12 in the afternoon. Then they receive a delivery of 30 more. How many apples do they have now? Show your work.",
        keywords=["47", "18", "12", "30"],
        min_length=30,
        description="Multi-step arithmetic word problem",
    ),
    QaPrompt(
        id="summarization",
        category="reasoning",
        system=None,
        user="Summarize the concept of 'machine learning' in exactly 2 sentences.",
        keywords=["machine learning", "data", "model", "learn"],
        min_length=40,
        max_repetition_ratio=0.1,
        description="Concise summarization ability",
    ),
    QaPrompt(
        id="translation",
        category="instruction",
        system=None,
        user="Translate the following sentence to French: 'The weather is beautiful today and I would like to go for a walk in the park.'",
        keywords=["le", "la", "est", "aujourd"],
        min_length=20,
        description="English to French translation",
    ),
    QaPrompt(
        id="code_sql",
        category="code",
        system="You are a database expert.",
        user="Write a SQL query to find the top 3 customers by total order amount from tables 'customers' (id, name) and 'orders' (id, customer_id, amount). Use GROUP BY and ORDER BY.",
        keywords=["SELECT", "FROM", "GROUP BY", "ORDER BY", "LIMIT"],
        regex_patterns=[r"SELECT", r"GROUP\s+BY", r"ORDER\s+BY"],
        min_length=50,
        description="SQL query with aggregation and sorting",
    ),
    QaPrompt(
        id="reasoning_analogy",
        category="reasoning",
        system=None,
        user="Complete the analogy: Doctor is to Hospital as Teacher is to ___. Explain why.",
        keywords=["school", "teacher", "hospital", "doctor"],
        min_length=30,
        description="Analogy completion with explanation",
    ),
]


def get_prompt_by_id(prompt_id: str) -> Optional[QaPrompt]:
    for p in PROMPTS:
        if p.id == prompt_id:
            return p
    return None


def get_prompts_by_category(category: str) -> list[QaPrompt]:
    return [p for p in PROMPTS if p.category == category]


def all_categories() -> list[str]:
    return sorted(set(p.category for p in PROMPTS))
