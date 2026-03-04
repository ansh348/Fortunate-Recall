"""
cosine_diagnostic.py -- Measure actual cosine similarity of Graphiti top-1 results.

For each of 112 questions, embeds the query, calls graphiti.search() for top-1,
fetches the fact_embedding from Neo4j, and computes cosine similarity.

Groups results into ROUTING_GAINS, ROUTING_REGRESSIONS, and UNAFFECTED to
determine if cosine similarity can separate gains from regressions for
confidence gating.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Env setup (mirrors evaluate_lifemembench.py)
# ---------------------------------------------------------------------------

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

for var in ["OPENAI_API_KEY"]:
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTIONS_PATH = PROJECT_ROOT / "LifeMemEval" / "lifemembench_questions.json"

ROUTING_GAINS = {
    "amara_q02", "amara_q03", "marcus_q02", "omar_q02",
    "tom_q03", "david_q11", "jake_q09", "omar_q07",
}

ROUTING_REGRESSIONS = {
    "david_q09", "david_q12", "elena_q07", "elena_q09",
    "jake_q08", "marcus_q01", "marcus_q04", "priya_q08",
    "priya_q11", "priya_q13", "priya_q14",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_graphiti_client() -> Graphiti:
    xai_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    llm_client = OpenAIClient(
        client=xai_client,
        config=LLMConfig(
            model="grok-4-1-fast-reasoning",
            small_model="grok-4-1-fast-reasoning",
        ),
    )
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig())
    return Graphiti(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        llm_client=llm_client,
        embedder=embedder,
    )


async def get_fact_embedding(driver, uuid: str):
    """Fetch fact_embedding for a single edge UUID from Neo4j."""
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.uuid = $uuid
        RETURN e.fact_embedding AS emb
        LIMIT 1
        """,
        uuid=uuid,
    )
    records = result.records if hasattr(result, "records") else result
    if records:
        d = records[0].data() if hasattr(records[0], "data") else dict(records[0])
        raw = d.get("emb")
        if raw is not None:
            return list(raw)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")

    graphiti = get_graphiti_client()
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig())
    driver = graphiti.driver

    results = []  # (qid, av, cosine_top1, rank, fact_preview)

    for i, q in enumerate(questions):
        qid = q["id"]
        group_id = q["group_id"]
        question = q["question"]
        av = q["attack_vector"]

        # Embed the query
        query_vec = await embedder.create(input_data=question.replace('\n', ' '))

        # Search
        try:
            search_results = await graphiti.search(
                question, group_ids=[group_id], num_results=50,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] {qid}: SEARCH ERROR: {e}")
            results.append((qid, av, None, -1, "ERROR"))
            continue

        if not search_results:
            print(f"  [{i+1}/{len(questions)}] {qid}: NO RESULTS")
            results.append((qid, av, 0.0, -1, "NO RESULTS"))
            continue

        top1 = search_results[0]
        uuid = str(getattr(top1, "uuid", ""))
        fact_text = str(getattr(top1, "fact", ""))

        # Fetch embedding from Neo4j
        fact_emb = await get_fact_embedding(driver, uuid)

        if fact_emb is None:
            print(f"  [{i+1}/{len(questions)}] {qid}: NO EMBEDDING in Neo4j for {uuid[:8]}")
            results.append((qid, av, None, 0, fact_text[:60]))
            continue

        cos = cosine_sim(query_vec, fact_emb)

        # Also compute cosine for top-5 to see the spread
        results.append((qid, av, cos, 0, fact_text[:80]))
        print(f"  [{i+1}/{len(questions)}] {qid}: cosine={cos:.4f}  {av}")

    await graphiti.close()

    # ---------------------------------------------------------------------------
    # Group and report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY DIAGNOSTIC")
    print("=" * 80)

    gains, regressions, unaffected = [], [], []
    for qid, av, cos, rank, preview in results:
        if cos is None:
            continue
        if qid in ROUTING_GAINS:
            gains.append((qid, av, cos, preview))
        elif qid in ROUTING_REGRESSIONS:
            regressions.append((qid, av, cos, preview))
        else:
            unaffected.append((qid, av, cos, preview))

    def print_group(name, items):
        print(f"\n{'-' * 80}")
        print(f"{name} ({len(items)} questions)")
        print(f"{'-' * 80}")
        if not items:
            print("  (none)")
            return
        cosines = [c for _, _, c, _ in items]
        avg = sum(cosines) / len(cosines)
        print(f"  AVG COSINE: {avg:.4f}")
        print(f"  MIN: {min(cosines):.4f}  MAX: {max(cosines):.4f}")
        print()
        for qid, av, cos, preview in sorted(items, key=lambda x: x[2]):
            print(f"  {cos:.4f}  {qid:<16s} {av:<35s} {preview}")

    print_group("ROUTING GAINS (routing helped these)", gains)
    print_group("ROUTING REGRESSIONS (routing hurt these)", regressions)
    print_group("UNAFFECTED (93 questions)", unaffected)

    # Summary comparison
    if gains and regressions:
        avg_g = sum(c for _, _, c, _ in gains) / len(gains)
        avg_r = sum(c for _, _, c, _ in regressions) / len(regressions)
        avg_u = sum(c for _, _, c, _ in unaffected) / len(unaffected) if unaffected else 0
        print(f"\n{'=' * 80}")
        print("VERDICT")
        print(f"{'=' * 80}")
        print(f"  Gains avg cosine:       {avg_g:.4f}")
        print(f"  Regressions avg cosine: {avg_r:.4f}")
        print(f"  Unaffected avg cosine:  {avg_u:.4f}")
        print(f"  Gap (regressions - gains): {avg_r - avg_g:.4f}")
        if avg_r - avg_g > 0.05:
            print("  -> Regressions have HIGHER cosine -- confidence gating CAN separate them")
            print("     (suppress routing when Graphiti cosine is high)")
        elif avg_g - avg_r > 0.05:
            print("  -> Gains have HIGHER cosine -- confidence gating direction is REVERSED")
            print("     (suppress routing when Graphiti cosine is LOW)")
        else:
            print("  -> Gap too small -- cosine similarity CANNOT separate gains from regressions")
            print("     (confidence gating won't work)")


if __name__ == "__main__":
    asyncio.run(main())
