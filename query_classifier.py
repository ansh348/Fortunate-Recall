"""
query_classifier.py — Classify user queries into behavioral categories.

Uses XAI Grok to map natural-language questions to 2-3 BehavioralCategory
values with confidence scores.  This drives category-aware retrieval routing
(Paper §5.2.9).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env()

from openai import AsyncOpenAI  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — canonical 11 categories (must match decay_engine.CATEGORIES)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "OBLIGATIONS",
    "RELATIONAL_BONDS",
    "HEALTH_WELLBEING",
    "IDENTITY_SELF_CONCEPT",
    "HOBBIES_RECREATION",
    "PREFERENCES_HABITS",
    "INTELLECTUAL_INTERESTS",
    "LOGISTICAL_CONTEXT",
    "PROJECTS_ENDEAVORS",
    "FINANCIAL_MATERIAL",
    "OTHER",
]
CATEGORY_SET = frozenset(CATEGORIES)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CategoryMatch:
    """A single category match with confidence score."""
    category: str   # one of CATEGORIES
    score: float    # 0.0–1.0


@dataclass
class QueryClassification:
    """Result of classifying a query into behavioral categories."""
    query: str
    matches: list[CategoryMatch]
    low_confidence: bool = False


# ---------------------------------------------------------------------------
# Session-level cache
# ---------------------------------------------------------------------------
_classifier_cache: dict[str, QueryClassification] = {}


def clear_classifier_cache():
    """Clear the session-level classifier cache. Call at evaluation start."""
    _classifier_cache.clear()


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------
QUERY_CLASSIFIER_SYSTEM = """\
You are a query routing system for a personal memory assistant.  Given a user's
question, determine which 2-3 behavioral memory categories are most likely to
contain the ANSWER.

You are NOT classifying the question itself — you are predicting which MEMORY
CATEGORIES in the user's knowledge graph would store facts relevant to answering
this question.

## Categories

1. OBLIGATIONS — Tasks, deadlines, appointments, promises, duties.
   Questions about "what do I need to do", "when is my appointment", "did I promise…"
2. RELATIONAL_BONDS — Family, friends, colleagues, relationship dynamics.
   Questions about "who is my…", "how is [person]", "what did [person] say…"
3. HEALTH_WELLBEING — Health conditions, medications, diagnoses, fitness, therapy.
   Questions about "what condition", "what medication", "how is my health…"
4. IDENTITY_SELF_CONCEPT — Core traits: name, age, ethnicity, occupation, beliefs,
   personality, learning style, self-descriptions.
   Questions about "what is my name", "where am I from", "what do I do for work…"
5. HOBBIES_RECREATION — Leisure activities, sports, games, collections.
   Questions about "what do I do for fun", "fishing", "cycling", "gaming…"
6. PREFERENCES_HABITS — Tastes, media, food, routines, daily patterns.
   Questions about "favorite", "what show", "what do I like", "morning routine…"
7. INTELLECTUAL_INTERESTS — Curiosities, learning goals, academic interests, reading.
   Questions about "what am I studying", "interested in", "reading…"
8. LOGISTICAL_CONTEXT — Schedules, locations, travel, one-time logistics.
   Questions about "where was", "what time", "when did I go", "travel plans…"
9. PROJECTS_ENDEAVORS — Ongoing work, startups, research, creative projects.
   Questions about "my project", "deadline", "milestone", "working on…"
10. FINANCIAL_MATERIAL — Budget, purchases, expenses, owned items, salary.
    Questions about "how much", "what did I buy", "my budget", "salary…"
11. OTHER — Does not fit above categories.

## Instructions

- Return the top 2-3 categories whose stored facts would answer this query.
- Assign a confidence score (0.0–1.0) to each.  Scores need not sum to 1.
- If the query is vague or spans many categories, give all scores < 0.4.
- Think about WHERE the answer lives, not what the question is about.
  Example: "What did my mom say about my fishing trip?"
    → RELATIONAL_BONDS (conversation with mom) + HOBBIES_RECREATION (fishing trip)

## Output (strict JSON, no commentary)

{"categories": [{"category": "CATEGORY_NAME", "score": 0.8}, {"category": "CATEGORY_NAME_2", "score": 0.5}]}"""


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

async def classify_query(
    query: str,
    xai_client: AsyncOpenAI,
    use_cache: bool = True,
) -> QueryClassification:
    """Classify a user query into 2-3 behavioral categories.

    Args:
        query: The user's question text.
        xai_client: AsyncOpenAI client configured for XAI Grok.
        use_cache: If True, return cached result for identical queries.

    Returns:
        QueryClassification with sorted matches (highest score first).
        On low confidence or error, returns all categories as fallback.
    """
    if use_cache and query in _classifier_cache:
        return _classifier_cache[query]

    try:
        resp = await xai_client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": QUERY_CLASSIFIER_SYSTEM},
                {"role": "user", "content": f"Classify this query:\n\n{query}"},
            ],
            max_tokens=400,
        )
        raw = json.loads(resp.choices[0].message.content)
        raw_categories = raw.get("categories", [])

        matches = []
        for entry in raw_categories:
            cat = entry.get("category", "").upper().replace(" ", "_")
            score = float(entry.get("score", 0.0))
            if cat in CATEGORY_SET and score > 0.0:
                matches.append(CategoryMatch(category=cat, score=score))

        matches.sort(key=lambda m: -m.score)
        matches = matches[:3]

        low_confidence = (not matches) or all(m.score < 0.4 for m in matches)

        if low_confidence:
            result = _fallback_classification(query)
        else:
            result = QueryClassification(query=query, matches=matches, low_confidence=False)

    except Exception:
        result = _fallback_classification(query)

    if use_cache:
        _classifier_cache[query] = result

    return result


def _fallback_classification(query: str) -> QueryClassification:
    """Return all non-OTHER categories with uniform low scores."""
    matches = [
        CategoryMatch(category=cat, score=0.3)
        for cat in CATEGORIES if cat != "OTHER"
    ]
    return QueryClassification(query=query, matches=matches, low_confidence=True)
