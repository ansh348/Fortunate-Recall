r"""
reenrich_edges_v8.py - Re-classify edges with entity-aware v8 prompt.

FIXES over v7:
  1. Pulls source/target entity names into the classifier prompt so the LLM
     knows WHO the fact is about (fixes "Bass weighed 15 pounds" ambiguity).
  2. World-knowledge detection as Step 0 in the LLM prompt. Edges about
     geography, public facts, etc. get tagged fr_is_world_knowledge=true
     and should be excluded from behavioral decay in evaluate_v4.
  3. Removes the 500-char fact truncation that could corrupt long edge facts.
  4. Distribution report shows personal vs world-knowledge breakdown.
  5. --dry-run flag classifies 20 edges without writing, for sanity checking.

WHY THIS MATTERS:
  v7 produced 1,420 OTHER edges (24%) at edge level vs 1 OTHER fact (0.1%)
  at fact level. Root cause: Graphiti extracts ALL relational triples from
  conversations — including world knowledge (Tokyo geography, Nice bike-sharing).
  These are structurally identical to personal memory edges but should NOT get
  behavioral decay treatment. The classifier also couldn't distinguish
  "Bass weighed 15 pounds" (HOBBIES) from ambiguous trivia because it had
  no entity context.

Usage:
    python reenrich_edges_v8.py --dry-run     # classify 20 edges, print results, don't write
    python reenrich_edges_v8.py               # full reclassification
    python evaluate_v4.py --evaluate
"""

import asyncio
import json
import os
import sys
import time as time_module
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

from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase

# ===========================================================================
# V8 Classifier Prompt (entity-aware, world-knowledge detection)
# ===========================================================================

V7_CATEGORIES = [
    "OBLIGATIONS", "RELATIONAL_BONDS", "HEALTH_WELLBEING", "IDENTITY_SELF_CONCEPT",
    "HOBBIES_RECREATION", "PREFERENCES_HABITS", "INTELLECTUAL_INTERESTS",
    "LOGISTICAL_CONTEXT", "PROJECTS_ENDEAVORS", "FINANCIAL_MATERIAL", "OTHER",
]

# Identity mapping — v7/v8 category names ARE the decay engine canonical names.
V8_TO_DECAY = {cat: cat for cat in V7_CATEGORIES}

SYSTEM_PROMPT = """You are a memory classification system for a conversational AI. You classify extracted facts from a personal knowledge graph into behavioral categories that determine how the memory should age over time.

IMPORTANT CONTEXT: Every fact you see was extracted from a real conversation between a user and an AI assistant. Even facts that LOOK like general knowledge were discussed because they are relevant to the user's life, projects, hobbies, or decisions. Default to treating facts as personal memories unless they are PURELY encyclopedic with zero connection to the user.

You will receive:
- **Source entity** and **Target entity**: the two nodes this fact connects in the graph.
- **Fact**: the extracted claim stored on the edge between those entities.
- **Is speaker edge**: whether the user or assistant is one of the entities. If true, this is ALWAYS a personal memory — skip Step 0.

## Step 0: Personal vs World Knowledge

BEFORE classifying into behavioral categories, determine whether this fact is a PERSONAL MEMORY or GENERAL WORLD KNOWLEDGE.

**CRITICAL RULES:**
- If `is_speaker_edge` is true, this is ALWAYS personal. Skip to Step 1.
- If the fact describes something the user DOES, OWNS, BOUGHT, PLANS, VISITED, LIKES, or IS — it is personal, even if the entities are objects/places.
- Only tag as world knowledge if the fact is a standalone encyclopedic statement that would be true regardless of who the user is AND has no actionable relevance to the user's life.
- **When in doubt, classify as personal.** False negatives (missing a WK tag) are harmless. False positives (tagging personal facts as WK) destroy memory.

Examples of world knowledge (→ OTHER):
- "Tokyo Tower was built in 1958" (pure historical trivia, no user connection)
- "Python is a programming language" (encyclopedic definition)
- "LeBron James played for the LA Lakers" (celebrity fact)
- "Nice has a bike-sharing system called Vélo Bleu" (city infrastructure fact)

Examples that LOOK like world knowledge but are PERSONAL (→ classify normally):
- "humidifier maintains humidity for plants" → user is growing plants, this is their setup (HOBBIES_RECREATION)
- "Humpback Rocks is a challenging hike in the Blue Ridge Mountains" → user hikes there (HOBBIES_RECREATION)
- "50-inch 4K TV was picked up for $350" → user's purchase (FINANCIAL_MATERIAL)
- "shade cloth protects plants from excessive heat" → user's greenhouse project (PROJECTS_ENDEAVORS or HOBBIES_RECREATION)
- "Call of Duty game was discounted by 20%" → user's purchase context (FINANCIAL_MATERIAL)
- "Saunders-Monticello Trail offers beautiful views" → user's local trail (HOBBIES_RECREATION or PREFERENCES_HABITS)

Key signal: Could this fact help the assistant remember something useful about the user's life? → personal memory. Is it a random encyclopedia entry with no user connection? → world knowledge.

## Categories (for personal memories only)

1. OBLIGATIONS — Tasks, deadlines, appointments, promises, action items. Time-bound, lose relevance after completion.
2. RELATIONAL_BONDS — Relationships with people: family, partners, friends, colleagues. Status, dynamics, emotional quality.
3. HEALTH_WELLBEING — Physical/mental health, medications, diagnoses, chronic conditions, fitness metrics, mental health, diet/nutrition, explicitly health-motivated behaviors.
4. IDENTITY_SELF_CONCEPT — Core stable traits: ethnicity, name, occupation, heritage, values, beliefs, personality. Things that wouldn't change if you woke up tomorrow with amnesia about your preferences. Stable across years. Near-zero decay.
5. HOBBIES_RECREATION — Active leisure involving accumulated skill or equipment investment: fishing, coin collecting, cycling, painting, gardening, cooking, photography. Can go dormant but reactivate. Slow decay (months to years).
6. PREFERENCES_HABITS — Current tastes and consumption patterns: media preferences (TV shows, movies, documentaries), food choices, subscription services, lifestyle routines, favorite restaurants. What you LIKE right now. Moderate decay (weeks to months), easily superseded.
7. INTELLECTUAL_INTERESTS — Curiosities, academic fascinations, active learning goals. Exploration WITHOUT a concrete deliverable. If actively producing/building/delivering something -> PROJECTS_ENDEAVORS instead.
8. LOGISTICAL_CONTEXT — Transient scheduling details: appointment times, one-time locations, errands, travel logistics. Recurring activities are NOT logistical even if they have dates attached.
9. PROJECTS_ENDEAVORS — Ongoing works: startups, research papers, creative projects, long-term goals with milestones and timelines.
10. FINANCIAL_MATERIAL — Budget, income, expenses, purchases, debts, assets, owned devices, hardware specs, tech upgrades.
11. OTHER — World knowledge, or genuinely does not fit categories 1-10.

## Classification Rules (for personal memories)

- **CLASSIFY THE FACT, NOT THE CONTEXT.** Determine the category based on the nature of the information being stored (what this fact reveals about someone's life), NOT the conversational context surrounding it.
- When the fact is a price, cost, or financial amount -> FINANCIAL_MATERIAL.
- When the fact is a transient scheduling detail (appointment time, location visited once, errand) -> LOGISTICAL_CONTEXT. Recurring activities, hobbies, and events are NOT logistical even if they have dates attached.
- When the fact is about who someone IS at their core (name, role, ethnicity, heritage, core values) -> IDENTITY_SELF_CONCEPT.
- When the fact is about an active leisure activity involving skill or equipment (fishing, photography, cycling, cooking, gardening) -> HOBBIES_RECREATION.
- When the fact is about current tastes, media consumption, food preferences, subscriptions, or lifestyle routines -> PREFERENCES_HABITS.
- **Key boundary test:** Does it involve accumulated SKILL? -> HOBBIES_RECREATION. Is it a consumption CHOICE that could flip tomorrow? -> PREFERENCES_HABITS. Is it a core trait unchanged across years? -> IDENTITY_SELF_CONCEPT.
- **PREFERENCES_HABITS GUARDRAILS:** PREFERENCES_HABITS is ONLY for consumption patterns that could change next week with no loss of identity, skill, or health impact.
  - If someone is LEARNING a language, skill, or subject -> INTELLECTUAL_INTERESTS.
  - If someone declares "I am a [role/aspiration]" -> IDENTITY_SELF_CONCEPT.
  - If a behavior is health/wellbeing-motivated -> HEALTH_WELLBEING.
  - If the fact is about device/hardware ownership, cost, or purchase -> FINANCIAL_MATERIAL.
  - If the fact is about usage frequency or wearing pattern -> PREFERENCES_HABITS.
  - If the fact is about activity the item supports (art supplies for art hobby) -> HOBBIES_RECREATION.
- **FINANCIAL_MATERIAL GUARDRAIL:** A physical item being mentioned does NOT automatically make it FINANCIAL_MATERIAL. Classify by what the fact IS ABOUT: item value/cost/ownership -> FINANCIAL_MATERIAL. Usage frequency -> PREFERENCES_HABITS. Activity it supports -> HOBBIES_RECREATION.
- **DELIVERABLE TEST (MANDATORY OVERRIDE):** If the fact involves creating, editing, or revising a NAMED artifact -> primary category MUST be PROJECTS_ENDEAVORS.
- **NAMED PERSON DISAMBIGUATION:** When a named person appears, ask: "Is this fact ABOUT the relationship itself, or does the person merely provide context?" Timing/counts/events involving a person -> classify by WHAT is measured. Relationship origin or dynamics -> RELATIONAL_BONDS.
- Assign membership weights across ALL relevant categories. Weights must sum to 1.0.
- Primary category gets the STRICTLY highest weight (minimum 0.3). No ties.
- OTHER must NEVER exceed 0.5 weight for personal memories. If uncertain, distribute across top 2-3 plausible categories.
- For world knowledge, OTHER should be 1.0.

## Output Format (strict JSON, no commentary)

{"primary_category": "CATEGORY_NAME", "is_world_knowledge": false, "weights": {"OBLIGATIONS": 0.0, "RELATIONAL_BONDS": 0.0, "HEALTH_WELLBEING": 0.0, "IDENTITY_SELF_CONCEPT": 0.0, "HOBBIES_RECREATION": 0.0, "PREFERENCES_HABITS": 0.0, "INTELLECTUAL_INTERESTS": 0.0, "LOGISTICAL_CONTEXT": 0.0, "PROJECTS_ENDEAVORS": 0.0, "FINANCIAL_MATERIAL": 0.0, "OTHER": 0.0}, "emotional_loading": {"detected": false, "type": null, "intensity": null}, "confidence": "high"}"""


# ===========================================================================
# Validation (ported from classify_facts.py, extended for v8)
# ===========================================================================

def validate_and_normalize(raw: dict) -> dict:
    """Normalize classifier output. Fixes casing, missing keys, renormalizes weights."""

    is_world_knowledge = raw.get("is_world_knowledge", False)

    primary = raw.get("primary_category", "OTHER")
    primary_upper = primary.upper().replace(" ", "_")

    best_match = None
    for canon in V7_CATEGORIES:
        if primary_upper == canon.upper():
            best_match = canon
            break
    if best_match is None:
        for canon in V7_CATEGORIES:
            if canon.upper() in primary_upper or primary_upper in canon.upper():
                best_match = canon
                break
    if best_match is None:
        best_match = "OTHER"
    primary = best_match

    raw_weights = raw.get("weights", {})
    weights = {}
    for canon in V7_CATEGORIES:
        found = False
        for k, v in raw_weights.items():
            if k.upper().replace(" ", "_") == canon.upper():
                weights[canon] = float(v)
                found = True
                break
        if not found:
            weights[canon] = 0.0

    # World knowledge: force OTHER=1.0, skip all guardrails
    if is_world_knowledge:
        weights = {k: 0.0 for k in V7_CATEGORIES}
        weights["OTHER"] = 1.0
        primary = "OTHER"
        emo = raw.get("emotional_loading", {})
        if not isinstance(emo, dict):
            emo = {"detected": False, "type": None, "intensity": None}
        return {
            "primary_category": primary,
            "is_world_knowledge": True,
            "weights": weights,
            "emotional_loading": emo,
            "confidence": raw.get("confidence", "low"),
        }

    # OTHER suppression (personal memories only)
    if primary == "OTHER" and weights.get("OTHER", 0) < 0.4:
        non_other = {k: v for k, v in weights.items() if k != "OTHER" and v > 0}
        if non_other:
            runner_up = max(non_other, key=non_other.get)
            if weights["OTHER"] - non_other[runner_up] < 0.1:
                primary = runner_up

    # Ensure primary has strictly highest weight
    max_weight = max(weights.values()) if weights else 0.3
    if weights[primary] < max_weight:
        weights[primary] = max_weight + 0.05
    for k in weights:
        if k != primary and weights[k] >= weights[primary]:
            weights[primary] = weights[k] + 0.05

    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    else:
        weights[primary] = 1.0

    emo = raw.get("emotional_loading", {})
    if not isinstance(emo, dict):
        emo = {"detected": False, "type": None, "intensity": None}

    return {
        "primary_category": primary,
        "is_world_knowledge": False,
        "weights": weights,
        "emotional_loading": emo,
        "confidence": raw.get("confidence", "unknown"),
    }


def map_to_decay_engine(v8_result: dict) -> dict:
    """Map v8 category names to decay engine category names."""
    v8_primary = v8_result["primary_category"]
    decay_primary = V8_TO_DECAY.get(v8_primary, "OTHER")

    decay_weights = {}
    for cat, weight in v8_result["weights"].items():
        decay_cat = V8_TO_DECAY.get(cat, "OTHER")
        decay_weights[decay_cat] = decay_weights.get(decay_cat, 0.0) + weight

    emo = v8_result.get("emotional_loading", {})
    return {
        "primary_category": decay_primary,
        "membership_weights": decay_weights,
        "is_world_knowledge": v8_result.get("is_world_knowledge", False),
        "confidence": 0.9 if v8_result.get("confidence") == "high" else 0.7,
        "emotional_loading": emo.get("detected", False) if isinstance(emo, dict) else False,
    }


# ===========================================================================
# Classification (entity-aware)
# ===========================================================================

def _is_speaker_entity(name: str) -> bool:
    """Check if an entity name represents a speaker (user or assistant)."""
    lower = name.lower().strip()
    return lower in ('user', 'assistant') or lower.startswith('user ') or lower.startswith('assistant ')


async def classify_edge_fact(
    fact_text: str,
    src_name: str,
    tgt_name: str,
    xai_client: AsyncOpenAI,
) -> dict:
    """Classify a single edge fact using the v8 entity-aware prompt."""

    fact_display = fact_text if fact_text else "(empty)"
    is_speaker = _is_speaker_entity(src_name or '') or _is_speaker_entity(tgt_name or '')

    prompt = f"""Classify this knowledge graph edge into behavioral categories.

**Source entity:** {src_name or 'unknown'}
**Target entity:** {tgt_name or 'unknown'}
**Fact:** {fact_display}
**Is speaker edge:** {is_speaker}

{"This edge directly involves the user/assistant — it is ALWAYS a personal memory. Skip Step 0 and go straight to classification." if is_speaker else "Step 0: Is this a personal memory about the user, or general world knowledge? Remember: these facts come from personal conversations, so lean toward personal unless purely encyclopedic."}
Step 1: Classify WHAT this fact IS (relationship? hobby? schedule? identity? etc.)

Respond with JSON only."""

    for attempt in range(3):
        try:
            resp = await xai_client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
            )
            raw = json.loads(resp.choices[0].message.content)
            # Hard override: speaker edges are NEVER world knowledge
            if is_speaker and raw.get("is_world_knowledge"):
                raw["is_world_knowledge"] = False
            validated = validate_and_normalize(raw)
            mapped = map_to_decay_engine(validated)
            return mapped
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 * (2 ** attempt))
            else:
                return {
                    "primary_category": "OTHER",
                    "membership_weights": {"OTHER": 1.0},
                    "is_world_knowledge": False,
                    "confidence": 0.0,
                    "emotional_loading": False,
                    "error": str(e),
                }


# ===========================================================================
# Main
# ===========================================================================

async def main():
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    group_id = "full_234"

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )

    if dry_run:
        print("=" * 60)
        print("DRY RUN — classifying 20 edges, printing results, no writes")
        print("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Clear old edge classifications (only with --force)
    # ---------------------------------------------------------------
    if not dry_run and force:
        print("Step 1: Clearing old edge classifications (--force)...")
        await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.group_id = $gid AND e.fr_enriched = true
            SET e.fr_enriched = false,
                e.fr_primary_category = null,
                e.fr_membership_weights = null,
                e.fr_confidence = null,
                e.fr_classified_ts = null,
                e.fr_is_world_knowledge = null,
                e.fr_emotional_loading = null
            """,
            gid=group_id,
        )
    elif not dry_run:
        # Resume mode: count already-enriched edges
        already = await driver.execute_query(
            "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid AND e.fr_enriched = true RETURN count(e) AS cnt",
            gid=group_id,
        )
        already_count = already.records[0]["cnt"]
        if already_count > 0:
            print(f"Step 1: Resuming — {already_count} edges already enriched, skipping clear. Use --force to redo all.")

    # ---------------------------------------------------------------
    # Step 1b: Show sample entity names (informational)
    # ---------------------------------------------------------------
    print("\nStep 1b: Scanning entity names in the graph...")
    entity_sample = await driver.execute_query(
        """
        MATCH (n:Entity)
        WHERE n.group_id = $gid
        RETURN DISTINCT n.name AS name
        ORDER BY name
        LIMIT 50
        """,
        gid=group_id,
    )
    entity_names = [r.data()["name"] for r in entity_sample.records if r.data().get("name")]
    print(f"  Sample entity names (first 20): {entity_names[:20]}")
    print(f"  NOTE: Graphiti does not create a 'User' entity. Entity context helps")
    print(f"  the LLM infer personal vs world knowledge from the triple structure.")

    # Count
    count_result = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid RETURN count(e) AS total",
        gid=group_id,
    )
    total = count_result.records[0]["total"]
    effective_total = 20 if dry_run else total
    print(f"\nTotal edges in graph: {total}" + (f" (dry run: classifying {effective_total})" if dry_run else ""))

    # ---------------------------------------------------------------
    # Step 2: Classify in batches (with entity context)
    # ---------------------------------------------------------------
    print(f"\nStep 2: Classifying edges with v8 entity-aware prompt...")
    batch_size = 20 if dry_run else 200
    classified = 0
    errors = 0
    world_knowledge_count = 0
    personal_count = 0
    t0 = time_module.time()

    while classified < effective_total:
        # KEY FIX: dry run reads ALL edges (ignores fr_enriched since we
        # didn't clear them). Full run only reads unclassified edges.
        if dry_run:
            batch_result = await driver.execute_query(
                """
                MATCH (src)-[e:RELATES_TO]->(tgt)
                WHERE e.group_id = $gid
                RETURN e.uuid AS uuid, e.fact AS fact,
                       src.name AS src_name, tgt.name AS tgt_name
                SKIP $skip LIMIT $limit
                """,
                gid=group_id,
                skip=classified,
                limit=batch_size,
            )
        else:
            batch_result = await driver.execute_query(
                """
                MATCH (src)-[e:RELATES_TO]->(tgt)
                WHERE e.group_id = $gid AND (e.fr_enriched IS NULL OR e.fr_enriched = false)
                RETURN e.uuid AS uuid, e.fact AS fact,
                       src.name AS src_name, tgt.name AS tgt_name
                LIMIT $limit
                """,
                gid=group_id,
                limit=batch_size,
            )

        records = batch_result.records if hasattr(batch_result, "records") else batch_result
        if not records:
            break

        sem = asyncio.Semaphore(40)

        async def classify_one(rec):
            nonlocal errors, world_knowledge_count, personal_count
            d = rec.data() if hasattr(rec, "data") else dict(rec)
            uuid = d["uuid"]
            fact = d.get("fact", "") or ""
            src_name = d.get("src_name", "") or ""
            tgt_name = d.get("tgt_name", "") or ""

            async with sem:
                result = await classify_edge_fact(fact, src_name, tgt_name, xai_client)

            if "error" in result:
                errors += 1

            if result.get("is_world_knowledge"):
                world_knowledge_count += 1
            else:
                personal_count += 1

            if dry_run:
                wk_tag = " [WORLD KNOWLEDGE]" if result.get("is_world_knowledge") else ""
                print(f"  ({src_name}) --[{fact[:70]}]--> ({tgt_name})")
                print(f"    -> {result['primary_category']}{wk_tag}  conf={result['confidence']}")
                return

            weights_json = json.dumps(result["membership_weights"])
            try:
                await driver.execute_query(
                    """
                    MATCH ()-[e:RELATES_TO]->()
                    WHERE e.uuid = $uuid
                    SET e.fr_enriched = true,
                        e.fr_primary_category = $cat,
                        e.fr_membership_weights = $weights,
                        e.fr_confidence = $conf,
                        e.fr_emotional_loading = $emo,
                        e.fr_is_world_knowledge = $wk,
                        e.fr_classified_ts = $ts
                    """,
                    uuid=uuid,
                    cat=result["primary_category"],
                    weights=weights_json,
                    conf=result["confidence"],
                    emo=result["emotional_loading"],
                    wk=result.get("is_world_knowledge", False),
                    ts=time_module.time(),
                )
            except Exception as write_err:
                errors += 1
                print(f"    Write error for {uuid}: {write_err}")

        tasks = [classify_one(rec) for rec in records]
        await asyncio.gather(*tasks, return_exceptions=True)

        classified += len(records)
        elapsed = time_module.time() - t0
        remaining = (elapsed / max(1, classified)) * (effective_total - classified)
        if not dry_run:
            print(f"  Classified {classified}/{effective_total} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, {errors} errors, {world_knowledge_count} WK, {personal_count} personal)")

    elapsed = time_module.time() - t0
    print(f"\nClassification complete: {classified} edges in {elapsed:.0f}s ({errors} errors)")
    print(f"  Personal memories: {personal_count}")
    print(f"  World knowledge:   {world_knowledge_count}")

    if dry_run:
        print(f"\n{'='*60}")
        print(f"Dry run complete. Review above, then run without --dry-run.")
        print(f"{'='*60}")
        await driver.close()
        return

    # ---------------------------------------------------------------
    # Step 3: Distribution report
    # ---------------------------------------------------------------
    dist_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid AND e.fr_enriched = true
        RETURN e.fr_primary_category AS cat,
               e.fr_is_world_knowledge AS wk,
               count(*) AS cnt
        ORDER BY cnt DESC
        """,
        gid=group_id,
    )
    print("\nEdge classification distribution:")
    print(f"  {'Category':30s} | {'Count':>6s} | {'Type'}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*20}")
    total_wk = 0
    total_personal = 0
    for rec in dist_result.records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        cat = d["cat"] or "NULL"
        wk = d.get("wk", False)
        cnt = d["cnt"]
        tag = "world-knowledge" if wk else "personal"
        if wk:
            total_wk += cnt
        else:
            total_personal += cnt
        print(f"  {cat:30s} | {cnt:6d} | {tag}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*20}")
    print(f"  {'TOTAL PERSONAL':30s} | {total_personal:6d} |")
    print(f"  {'TOTAL WORLD KNOWLEDGE':30s} | {total_wk:6d} |")

    # Personal-only distribution (this is what matters for the paper)
    print("\nPersonal memory distribution (behavioral ontology):")
    personal_dist = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid AND e.fr_enriched = true
              AND (e.fr_is_world_knowledge IS NULL OR e.fr_is_world_knowledge = false)
        RETURN e.fr_primary_category AS cat, count(*) AS cnt
        ORDER BY cnt DESC
        """,
        gid=group_id,
    )
    for rec in personal_dist.records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        print(f"  {d['cat']:30s}: {d['cnt']}")

    # ---------------------------------------------------------------
    # Step 4: Sanity check on known-tricky facts
    # ---------------------------------------------------------------
    print("\nSanity check — sample edge classifications:")
    sample_result = await driver.execute_query(
        """
        MATCH (src)-[e:RELATES_TO]->(tgt)
        WHERE e.group_id = $gid AND e.fr_enriched = true
          AND (toLower(e.fact) CONTAINS 'bass' OR toLower(e.fact) CONTAINS 'fish'
               OR toLower(e.fact) CONTAINS 'comedy' OR toLower(e.fact) CONTAINS 'comedian'
               OR toLower(e.fact) CONTAINS 'cocktail' OR toLower(e.fact) CONTAINS 'friday'
               OR toLower(e.fact) CONTAINS 'wednesday' OR toLower(e.fact) CONTAINS 'bike'
               OR toLower(e.fact) CONTAINS 'temple' OR toLower(e.fact) CONTAINS 'nice')
        RETURN src.name AS src, e.fact AS fact, tgt.name AS tgt,
               e.fr_primary_category AS cat, e.fr_is_world_knowledge AS wk
        LIMIT 30
        """,
        gid=group_id,
    )
    for rec in sample_result.records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        wk_tag = " [WK]" if d.get("wk") else ""
        print(f"  [{d['cat']:25s}]{wk_tag} ({d['src']}) --[{d['fact'][:60]}]--> ({d['tgt']})")

    # ---------------------------------------------------------------
    # Step 5: OTHER audit — what personal edges are still OTHER?
    # ---------------------------------------------------------------
    print("\nOTHER audit — personal edges still classified as OTHER:")
    other_result = await driver.execute_query(
        """
        MATCH (src)-[e:RELATES_TO]->(tgt)
        WHERE e.group_id = $gid AND e.fr_enriched = true
              AND e.fr_primary_category = 'OTHER'
              AND (e.fr_is_world_knowledge IS NULL OR e.fr_is_world_knowledge = false)
        RETURN src.name AS src, e.fact AS fact, tgt.name AS tgt
        LIMIT 15
        """,
        gid=group_id,
    )
    other_personal = other_result.records
    if other_personal:
        for rec in other_personal:
            d = rec.data() if hasattr(rec, "data") else dict(rec)
            print(f"  ({d['src']}) --[{d['fact'][:70]}]--> ({d['tgt']})")
    else:
        print("  (none — clean!)")

    print(f"\nDone. Now run: python evaluate_v4.py --evaluate")
    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())