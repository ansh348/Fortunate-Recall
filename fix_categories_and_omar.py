"""
fix_categories_and_omar.py — Batch Cypher fixes for Changes 4b and 5.

Change 4b: Fix 6 misclassified edge categories.
Change 5:  Add superseded_by metadata to omar's stale Lyft edge.

Run once, then re-evaluate.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

load_env()

from neo4j import AsyncGraphDatabase


# ── Change 4b: Edge category fixes ──────────────────────────────────────
# Each entry: (group_id, fact_substring, wrong_category, correct_category)
CATEGORY_FIXES = [
    # priya_q05: pet/dog edge miscategorised
    ("lifemembench_priya", "RELATIONAL_BONDS", "PREFERENCES_HABITS",
     ["dog", "pet", "golden retriever"]),
    # marcus_q13: employee count edge
    ("lifemembench_marcus", "IDENTITY_SELF_CONCEPT", "PROJECTS_ENDEAVORS",
     ["carlos", "fourth employee", "hired"]),
    # elena_q12: stress relief / knitting / anxiety
    ("lifemembench_elena", "PREFERENCES_HABITS", "HEALTH_WELLBEING",
     ["knit", "anxiety", "stress"]),
    # amara_q02: phone switch
    ("lifemembench_amara", "FINANCIAL_MATERIAL", "PREFERENCES_HABITS",
     ["samsung", "galaxy", "iphone", "android", "phone"]),
    # amara_q07: alcohol / wine
    ("lifemembench_amara", "LOGISTICAL_CONTEXT", "HEALTH_WELLBEING",
     ["alcohol", "drink", "wine", "drunk"]),
    # omar_q07: moving / Dallas / staying
    ("lifemembench_omar", "IDENTITY_SELF_CONCEPT", "LOGISTICAL_CONTEXT",
     ["dallas", "moving", "houston", "staying"]),
]


async def fix_categories(driver):
    """Change 4b: Fix misclassified edge categories."""
    print("\n=== Change 4b: Edge Category Fixes ===\n")
    total_fixed = 0

    for group_id, wrong_cat, correct_cat, keywords in CATEGORY_FIXES:
        # Find candidate edges: wrong category + matching keyword(s)
        result = await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.group_id = $gid
              AND e.fr_primary_category = $wrong_cat
              AND e.fr_enriched = true
            RETURN e.uuid AS uuid, e.fact AS fact, e.fr_primary_category AS cat,
                   e.fr_membership_weights AS weights
            """,
            gid=group_id,
            wrong_cat=wrong_cat,
        )
        records = result.records if hasattr(result, "records") else result

        matched = []
        for rec in records:
            d = rec.data() if hasattr(rec, "data") else dict(rec)
            fact_low = (d.get("fact") or "").lower()
            if any(kw in fact_low for kw in keywords):
                matched.append(d)

        if not matched:
            print(f"  [{group_id}] {wrong_cat} -> {correct_cat}: "
                  f"NO matching edges found (keywords: {keywords})")
            continue

        for d in matched:
            uuid = d["uuid"]
            # Update primary category
            old_weights = d.get("weights", "{}")
            if isinstance(old_weights, str):
                weights = json.loads(old_weights)
            else:
                weights = old_weights or {}

            # Swap weights: give correct_cat the old wrong_cat weight
            old_wrong_w = weights.get(wrong_cat, 0.3)
            old_correct_w = weights.get(correct_cat, 0.0)
            weights[correct_cat] = max(old_wrong_w, old_correct_w + 0.1)
            weights[wrong_cat] = min(0.05, old_correct_w)

            # Renormalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: round(v / total, 4) for k, v in weights.items()}

            await driver.execute_query(
                """
                MATCH ()-[e:RELATES_TO]->()
                WHERE e.uuid = $uuid
                SET e.fr_primary_category = $new_cat,
                    e.fr_membership_weights = $weights
                """,
                uuid=uuid,
                new_cat=correct_cat,
                weights=json.dumps(weights),
            )
            print(f"  FIXED [{group_id}] {wrong_cat} -> {correct_cat}: "
                  f"{d['fact'][:80]}")
            total_fixed += 1

    print(f"\n  Total edges recategorised: {total_fixed}")
    return total_fixed


# ── Change 5: Fix omar_q01 (Lyft superseded by Uber) ────────────────────

async def fix_omar_lyft(driver):
    """Change 5: Mark omar's stale Lyft edge as superseded by the Uber edge."""
    print("\n=== Change 5: omar_q01 Lyft -> Uber supersession ===\n")
    group_id = "lifemembench_omar"

    # Find the Lyft edge
    lyft_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
          AND toLower(e.fact) CONTAINS 'lyft'
          AND (toLower(e.fact) CONTAINS 'driv' OR toLower(e.fact) CONTAINS 'rideshare'
               OR toLower(e.fact) CONTAINS 'gig')
        RETURN e.uuid AS uuid, e.fact AS fact,
               e.fr_superseded_by AS superseded_by,
               e.fr_supersession_confidence AS sup_conf
        """,
        gid=group_id,
    )
    lyft_records = lyft_result.records if hasattr(lyft_result, "records") else lyft_result

    # Find the Uber edge
    uber_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
          AND toLower(e.fact) CONTAINS 'uber'
          AND (toLower(e.fact) CONTAINS 'driv' OR toLower(e.fact) CONTAINS 'rideshare'
               OR toLower(e.fact) CONTAINS 'pays' OR toLower(e.fact) CONTAINS 'income'
               OR toLower(e.fact) CONTAINS 'bills')
        RETURN e.uuid AS uuid, e.fact AS fact
        """,
        gid=group_id,
    )
    uber_records = uber_result.records if hasattr(uber_result, "records") else uber_result

    if not lyft_records:
        print("  WARNING: No Lyft edge found!")
        return 0
    if not uber_records:
        print("  WARNING: No Uber edge found!")
        return 0

    # Print all matches for verification
    print("  Lyft edges found:")
    for rec in lyft_records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        sup = d.get("superseded_by")
        print(f"    [{d['uuid'][:12]}] {d['fact'][:80]}")
        if sup:
            print(f"      Already superseded_by: {sup}")

    print("  Uber edges found:")
    for rec in uber_records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        print(f"    [{d['uuid'][:12]}] {d['fact'][:80]}")

    # Use the first Uber edge as the superseder
    uber_d = uber_records[0].data() if hasattr(uber_records[0], "data") else dict(uber_records[0])
    uber_uuid = uber_d["uuid"]

    fixed = 0
    for rec in lyft_records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        lyft_uuid = d["uuid"]

        if d.get("superseded_by"):
            print(f"  SKIP: Lyft edge {lyft_uuid[:12]} already superseded")
            continue

        await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.uuid = $uuid
            SET e.fr_superseded_by = $uber_uuid,
                e.fr_supersession_confidence = 0.95
            """,
            uuid=lyft_uuid,
            uber_uuid=uber_uuid,
        )
        print(f"  FIXED: Lyft edge {lyft_uuid[:12]} now superseded_by Uber edge {uber_uuid[:12]}")
        fixed += 1

    return fixed


async def main():
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "testpassword123")

    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        cat_fixed = await fix_categories(driver)
        omar_fixed = await fix_omar_lyft(driver)
        print(f"\n{'='*50}")
        print(f"DONE: {cat_fixed} categories fixed, {omar_fixed} supersession(s) added")
        print(f"{'='*50}")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
