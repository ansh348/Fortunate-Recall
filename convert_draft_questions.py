#!/usr/bin/env python3
"""
Convert draft_questions from persona YAML files (personas 21-40) into the
lifemembench_questions.json format and append them to the existing file.
"""

import json
import re
import yaml
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIFEMEMEVAL_DIR = os.path.join(BASE_DIR, "LifeMemEval")
QUESTIONS_JSON = os.path.join(LIFEMEMEVAL_DIR, "lifemembench_questions.json")

# Persona definitions: (folder, yaml_filename, short_name)
PERSONAS = [
    ("21_thanh",    "persona_021_thanh.yaml",     "thanh"),
    ("22_alex",     "persona_022_alex.yaml",      "alex"),
    ("23_mirri",    "persona_023_mirri.yaml",     "mirri"),
    ("24_jerome",   "persona_024_jerome.yaml",    "jerome"),
    ("25_ingrid",   "persona_025_ingrid.yaml",    "ingrid"),
    ("26_dmitri",   "persona_026_dmitri.yaml",    "dmitri"),
    ("27_yoli",     "persona_027_yoli.yaml",      "yoli"),
    ("28_dariush",  "persona_028_dariush.yaml",   "dariush"),
    ("29_aroha",    "persona_029_aroha.yaml",     "aroha"),
    ("30_mehmet",   "persona_030_mehmet.yaml",    "mehmet"),
    ("31_saga",     "persona_031_saga.yaml",      "saga"),
    ("32_kofi",     "persona_032_kofi.yaml",      "kofi"),
    ("33_valentina","persona_033_valentina.yaml",  "valentina"),
    ("34_billy",    "persona_034_billy.yaml",     "billy"),
    ("35_pan",      "persona_035_pan.yaml",       "pan"),
    ("36_marley",   "persona_036_marley.yaml",    "marley"),
    ("37_leila",    "persona_037_leila.yaml",     "leila"),
    ("38_chenoa",   "persona_038_chenoa.yaml",    "chenoa"),
    ("39_joonho",   "persona_039_joonho.yaml",    "joonho"),
    ("40_zara",     "persona_040_zara.yaml",      "zara"),
]

# Attack vector mapping
AV_MAP = {
    1: "AV1_superseded_preference",
    2: "AV2_expired_logistics",
    3: "AV3_stable_identity",
    4: "AV4_multi_version_fact",
    5: "AV5_broad_query",
    6: "AV6_cross_session_contradiction",
    7: "AV7_selective_forgetting",
    8: "AV8_numeric_preservation",
    9: "AV9_soft_supersession",
}

# Attack vector description templates
AV_DESC_TEMPLATES = {
    1: "Superseded preference - system must surface CURRENT state not OLD",
    2: "Expired logistics - system must correctly identify temporal status of events",
    3: "Stable identity - system must accurately recall core identity facts",
    4: "Multi-version fact - system must surface LATEST version not earlier one",
    5: "Broad query - system must synthesize information across multiple sessions",
    6: "Cross-session contradiction - system must acknowledge both positions",
    7: "Selective forgetting - system must recognize retracted/scrapped plans",
    8: "Numeric preservation - system must recall specific numbers accurately",
    9: "Soft supersession - system must reflect undecided/tentative state accurately",
}

# Category keyword heuristics for inferring relevant_categories
CATEGORY_KEYWORDS = {
    "OBLIGATIONS": ["deadline", "inspection", "exam", "meeting", "event", "wedding",
                     "surgery", "appointment", "ceremony", "celebration", "birthday",
                     "presentation", "review", "scheduled", "upcoming", "guild",
                     "camp", "accreditation"],
    "RELATIONAL_BONDS": ["friend", "wife", "husband", "mother", "father", "brother",
                          "sister", "son", "daughter", "grandmother", "grandfather",
                          "family", "boyfriend", "girlfriend", "partner", "pet", "cat",
                          "dog", "best friend", "apprentice", "children", "kids",
                          "nephew", "niece", "aunt", "uncle", "cousin"],
    "HEALTH_WELLBEING": ["health", "diabetes", "medication", "medicine", "doctor",
                          "surgery", "disease", "kidney", "pain", "vitamin",
                          "adhd", "therapy", "therapist", "supplement", "ibuprofen",
                          "condition", "diagnosis", "blood", "chronic", "cancer",
                          "mental", "anxiety", "depression", "ritalin", "methylphenidate",
                          "cataract", "deficiency"],
    "IDENTITY_SELF_CONCEPT": ["transition", "identity", "education", "background",
                               "heritage", "culture", "who", "self", "graduated",
                               "qualification", "role", "career", "craft",
                               "belonging", "impostor", "born", "raised", "origin"],
    "HOBBIES_RECREATION": ["hobby", "fitness", "sport", "pilates", "muay thai",
                            "yoga", "exercise", "game", "music", "art", "techno",
                            "party", "keyboard", "recreation", "leisure", "fishing",
                            "surfing", "climbing", "hiking", "cooking", "garden",
                            "photography", "calligraphy", "döner", "football"],
    "PREFERENCES_HABITS": ["preference", "feed", "brand", "pomade", "product",
                            "diet", "routine", "language", "programming", "tool",
                            "uses", "switched", "favorite", "clippers", "chemical",
                            "method", "style", "equipment"],
    "INTELLECTUAL_INTERESTS": ["interest", "book", "research", "study", "learn",
                                "reading", "side project", "framework", "theory",
                                "architecture", "design", "sustainable", "science",
                                "philosophy", "creative"],
    "LOGISTICAL_CONTEXT": ["live", "apartment", "house", "rent", "lease", "vehicle",
                            "car", "truck", "boat", "location", "move", "moving",
                            "address", "shop", "studio", "office", "commute",
                            "transport", "drive"],
    "PROJECTS_ENDEAVORS": ["project", "plan", "startup", "business", "shop",
                            "work", "job", "firm", "company", "museum", "platform",
                            "pivot", "second shop", "expansion", "build", "open",
                            "venture", "manage", "pond", "farm"],
    "FINANCIAL_MATERIAL": ["cost", "price", "salary", "money", "budget", "pay",
                            "lira", "dong", "dirham", "euro", "dollar", "afford",
                            "expense", "saving", "equity", "income"],
}


def infer_categories(question_text, correct_text, attack_vector_num):
    """Infer relevant categories from the question and answer text."""
    combined = (question_text + " " + correct_text).lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[category] = score

    # Sort by score descending, take top categories (at least 1, at most 3)
    sorted_cats = sorted(scores.items(), key=lambda x: -x[1])

    if not sorted_cats:
        # Fallback based on attack vector
        av_defaults = {
            1: ["PREFERENCES_HABITS"],
            2: ["OBLIGATIONS"],
            3: ["IDENTITY_SELF_CONCEPT"],
            4: ["PROJECTS_ENDEAVORS"],
            5: ["IDENTITY_SELF_CONCEPT"],
            6: ["PREFERENCES_HABITS"],
            7: ["PROJECTS_ENDEAVORS"],
            8: ["FINANCIAL_MATERIAL"],
            9: ["LOGISTICAL_CONTEXT"],
        }
        return av_defaults.get(attack_vector_num, ["IDENTITY_SELF_CONCEPT"])

    # Take top 1-3 categories that actually scored
    result = []
    for cat, score in sorted_cats:
        if score >= 1 and len(result) < 3:
            result.append(cat)
    return result if result else [sorted_cats[0][0]]


def normalize_wrong_if_surfaced(value):
    """Normalize wrong_if_surfaced to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    # It's a string — might contain parenthetical notes; keep as single element
    return [str(value)]


def generate_av_description(av_num, question_text):
    """Generate a contextual attack vector description."""
    template = AV_DESC_TEMPLATES.get(av_num, "Unknown attack vector")
    # Make it slightly more specific by referencing the question topic
    return f"{template} — question: {question_text}"


def fix_yaml_content(raw_text):
    """
    Fix invalid YAML where wrong_if_surfaced has a parenthetical note
    outside the quoted string, e.g.:
      wrong_if_surfaced: "Never uses chemicals" (without noting the later treatment)
    becomes:
      wrong_if_surfaced: "Never uses chemicals (without noting the later treatment)"
    """
    # Pattern: wrong_if_surfaced: "..." (...)
    # Captures: the key+indent, the quoted string content, and the parenthetical
    pattern = re.compile(
        r'^(\s*wrong_if_surfaced:\s*)"((?:[^"\\]|\\.)*?)"\s+(\(.*\))\s*$',
        re.MULTILINE
    )
    return pattern.sub(r'\1"\2 \3"', raw_text)


def process_persona(folder, yaml_file, short_name):
    """Process a single persona YAML file and return converted questions."""
    yaml_path = os.path.join(LIFEMEMEVAL_DIR, folder, yaml_file)

    if not os.path.exists(yaml_path):
        print(f"  WARNING: File not found: {yaml_path}")
        return []

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Fix the common YAML issue with parenthetical notes outside quotes
    fixed_content = fix_yaml_content(raw_content)
    data = yaml.safe_load(fixed_content)

    draft_questions = data.get("draft_questions")
    if not draft_questions:
        print(f"  WARNING: No draft_questions in {yaml_file}")
        return []

    converted = []
    for idx, dq in enumerate(draft_questions, start=1):
        question_text = dq["question"]
        correct_text = dq["correct"]
        av_num = dq["attack_vector"]
        wrong_raw = dq.get("wrong_if_surfaced")

        q_id = f"{short_name}_q{idx:02d}"
        group_id = f"lifemembench_{short_name}"
        attack_vector = AV_MAP.get(av_num, f"AV{av_num}_unknown")
        wrong_indicators = normalize_wrong_if_surfaced(wrong_raw)
        categories = infer_categories(question_text, correct_text, av_num)
        av_description = generate_av_description(av_num, question_text)

        entry = {
            "id": q_id,
            "persona": short_name,
            "group_id": group_id,
            "question": question_text,
            "correct_answer": correct_text,
            "wrong_answer_indicators": wrong_indicators,
            "attack_vector": attack_vector,
            "attack_vector_description": av_description,
            "relevant_categories": categories,
            "temporal_notes": "",
        }
        converted.append(entry)

    return converted


def main():
    print("=" * 60)
    print("LIFEMEMBENCH — Draft Questions Converter (Personas 21-40)")
    print("=" * 60)

    # Load existing questions
    existing_questions = []
    if os.path.exists(QUESTIONS_JSON):
        with open(QUESTIONS_JSON, "r", encoding="utf-8") as f:
            existing_questions = json.load(f)
        print(f"\nLoaded {len(existing_questions)} existing questions from lifemembench_questions.json")
    else:
        print(f"\nNo existing questions file found — will create new one.")

    # Process all 20 personas
    all_new_questions = []
    summary = []

    print("\nProcessing personas:\n")
    for folder, yaml_file, short_name in PERSONAS:
        questions = process_persona(folder, yaml_file, short_name)
        count = len(questions)
        all_new_questions.extend(questions)
        summary.append((short_name, count))
        print(f"  {short_name:12s}  {count:3d} questions")

    # Combine and write
    combined = existing_questions + all_new_questions
    with open(QUESTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # Print summary
    total_new = len(all_new_questions)
    total_combined = len(combined)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Personas processed:    {len(summary)}")
    print(f"  New questions added:   {total_new}")
    print(f"  Previous questions:    {len(existing_questions)}")
    print(f"  Total questions now:   {total_combined}")
    print(f"\n  Output file: {QUESTIONS_JSON}")

    # Per-persona breakdown
    print("\n  Per-persona breakdown:")
    for name, count in summary:
        print(f"    {name:12s}  {count:3d}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
