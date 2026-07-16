"""Original embedding QA fixtures (MTEB-*inspired* task shapes, not MTEB items).

MTEB is the public market suite (STS, retrieval, classification, …). We do not
ship copyrighted MTEB datasets. These fixtures use the *same task shapes* with
original short English items suitable for automated gates.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# STS-style triples: (anchor, positive paraphrase, hard-ish negative)
# Expect cosine(a,p) > cosine(a,n) + margin
# ---------------------------------------------------------------------------

SEMANTIC_TRIPLES: list[tuple[str, str, str]] = [
    (
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "The stock market rose sharply today.",
    ),
    (
        "A recipe for chocolate chip cookies.",
        "How to bake cookies with chocolate chips.",
        "Installing Kubernetes on bare metal.",
    ),
    (
        "Vector databases store embeddings for search.",
        "Embedding vectors enable semantic retrieval.",
        "The weather will be sunny tomorrow.",
    ),
    (
        "Photosynthesis converts light into chemical energy in plants.",
        "Plants use sunlight to produce energy-rich molecules.",
        "A jazz quartet played at the downtown club.",
    ),
    (
        "Python is a popular programming language for data science.",
        "Data scientists often write code in Python.",
        "The train arrived twenty minutes late.",
    ),
    (
        "Mount Everest is the highest mountain above sea level.",
        "Everest is Earth's tallest peak above sea level.",
        "A quiet library encourages deep reading.",
    ),
]

# ---------------------------------------------------------------------------
# Pair classification (MTEB PairClassification shape): (text_a, text_b, is_paraphrase)
# ---------------------------------------------------------------------------

PAIR_CLASSIFICATION: list[tuple[str, str, bool]] = [
    ("How do I reset my password?", "Steps to change a forgotten password.", True),
    ("How do I reset my password?", "Best espresso machines under two hundred dollars.", False),
    ("The cat sat on the mat.", "A cat was sitting on a mat.", True),
    ("The cat sat on the mat.", "Quantum error correction codes.", False),
    ("Ship order number 4421 to Austin.", "Deliver order 4421 to Austin Texas.", True),
    ("Ship order number 4421 to Austin.", "Cancel the subscription immediately.", False),
    ("Summarize this research paper.", "Give a short summary of the paper.", True),
    ("Summarize this research paper.", "Tune a guitar to standard pitch.", False),
    ("Apple released a new phone model.", "A new smartphone was launched by Apple.", True),
    ("Apple released a new phone model.", "River levels rose after heavy rain.", False),
]

# ---------------------------------------------------------------------------
# Retrieval (MTEB Retrieval shape): query + corpus; gold index of relevant doc
# ---------------------------------------------------------------------------

RETRIEVAL_CASES: list[dict] = [
    {
        "query": "symptoms of dehydration",
        "docs": [
            "Dehydration often causes thirst, dry mouth, and dark urine.",
            "A compiler translates source code into machine code.",
            "Basketball teams score by shooting the ball through a hoop.",
            "Soil pH affects which crops grow well in a garden.",
        ],
        "gold": 0,
    },
    {
        "query": "how to make sourdough bread",
        "docs": [
            "Network latency is the delay before a transfer of data begins.",
            "Sourdough bread is made with a fermented flour-and-water starter.",
            "Glaciers carve U-shaped valleys over long periods.",
            "A mutex protects shared state in concurrent programs.",
        ],
        "gold": 1,
    },
    {
        "query": "what is a binary search tree",
        "docs": [
            "Olive oil is a staple of Mediterranean cooking.",
            "Tides are caused mainly by the moon's gravity.",
            "A binary search tree keeps keys ordered for efficient lookup.",
            "Coral reefs support highly diverse marine life.",
        ],
        "gold": 2,
    },
    {
        "query": "benefits of regular exercise",
        "docs": [
            "Regular exercise improves cardiovascular health and mood.",
            "JSON is a lightweight data-interchange format.",
            "The Amazon is the largest rainforest by area.",
            "A sonnet is a fourteen-line poem.",
        ],
        "gold": 0,
    },
    {
        "query": "causes of climate change",
        "docs": [
            "Piano tuning adjusts string tension to target pitches.",
            "Greenhouse gas emissions are a major driver of climate change.",
            "HTML structures the content of web pages.",
            "Bees pollinate many flowering plants.",
        ],
        "gold": 1,
    },
]

# Shape / batch / determinism corpus
CORPUS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "AX Engine runs local embeddings on Apple Silicon.",
    "What is the capital of France?",
    "A short query",
    "Vector databases and retrieval augmented generation.",
    "Hello world",
    "Embeddings map text into continuous vector spaces.",
    "Local inference avoids sending private data to the cloud.",
]
