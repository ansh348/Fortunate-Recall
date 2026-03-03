"""
diagnose_retrieval.py -- Figure out WHERE the pipeline is failing.

For each of the 25 PoC questions, checks:
    1. Does the expected answer exist ANYWHERE in Neo4j? (entity names, edge facts)
    2. If yes, at what rank does Graphiti search find it? (top 10? top 50?)
    3. What does Graphiti actually return in top 5?

This separates "fact never extracted" from "fact exists but search misses it"
from "fact found but wrong rank".

Usage:
    python diagnose_retrieval.py
"""

import asyncio
import json
import os
import re
import sys
import time as time_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'LongMemEval' / 'data'
POC_ARTIFACTS_DIR = DATA_DIR / 'poc_artifacts'

def load_env():
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

load_env()

from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig


def get_graphiti_client() -> Graphiti:
    xai_client = AsyncOpenAI(
        api_key=os.environ['XAI_API_KEY'],
        base_url="https://api.x.ai/v1",
    )
    llm_client = OpenAIClient(
        client=xai_client,
        config=LLMConfig(
            model="grok-4-1-fast-non-reasoning",
            small_model="grok-4-1-fast-non-reasoning",
        ),
    )
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig())
    return Graphiti(
        os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        os.environ.get('NEO4J_USER', 'neo4j'),
        os.environ.get('NEO4J_PASSWORD', 'testpassword123'),
        llm_client=llm_client,
        embedder=embedder,
    )


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[,$%]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def _answer_in_text(answer: str, text: str) -> bool:
    """Check if answer appears in text (flexible matching)."""
    ans = _normalize(answer)
    txt = _normalize(text)

    if ans in txt:
        return True

    # Single word / short answers
    words = ans.split()
    if len(words) == 1:
        if re.search(r'\b' + re.escape(words[0]) + r'\b', txt):
            return True
    elif len(words) <= 3:
        # Check all words present
        if all(w in txt for w in words if len(w) > 2):
            return True
    else:
        # Token overlap
        tokens = [w for w in words if len(w) > 2]
        if tokens:
            matches = sum(1 for w in tokens if w in txt)
            if matches / len(tokens) >= 0.5:
                return True

    return False


def _extract_fact_text(result) -> str:
    if hasattr(result, 'fact'):
        return str(result.fact)
    if hasattr(result, 'name'):
        return str(result.name)
    if hasattr(result, 'summary'):
        return str(result.summary)
    return str(result)


async def main():
    questions = json.load(open(POC_ARTIFACTS_DIR / 'poc_questions.json', encoding='utf-8'))
    graphiti = get_graphiti_client()
    group_id = "poc_kill_gate"

    print(f"{'='*70}")
    print(f"RETRIEVAL DIAGNOSIS: Where is the pipeline failing?")
    print(f"{'='*70}\n")

    diagnosis = []

    for qi, q in enumerate(questions):
        question = q['question']
        answer = q['answer']
        # For long answers (preferences), extract first meaningful phrase
        short_answer = answer[:60]

        print(f"\n[Q{qi+1}] {question[:65]}...")
        print(f"  Answer: {short_answer}")

        # ---------------------------------------------------------------
        # CHECK 1: Does the answer exist in entity EDGES (facts)?
        # ---------------------------------------------------------------
        # Search edges for the answer text
        edge_hits = []
        search_terms = []

        # Build search terms from the answer
        ans_norm = _normalize(answer)
        ans_words = [w for w in ans_norm.split() if len(w) > 2]

        # For short answers, also search with the question keywords
        if len(ans_words) <= 3:
            # Use answer + question keywords
            q_words = [w for w in _normalize(question).split() if len(w) > 3]
            search_terms = ans_words + q_words[:3]
        else:
            search_terms = ans_words[:5]

        # Neo4j full-text search on edge facts
        search_str = ' OR '.join(f'"{w}"' for w in search_terms[:5] if w)

        try:
            edge_result = await graphiti.driver.execute_query(
                """
                MATCH ()-[e:RELATES_TO]-()
                WHERE e.group_id = $group_id
                  AND toLower(e.fact) CONTAINS $answer_lower
                RETURN e.fact AS fact, e.uuid AS uuid
                LIMIT 10
                """,
                group_id=group_id,
                answer_lower=_normalize(answer) if len(answer) < 30 else _normalize(answer.split()[0]),
            )
            records = edge_result.records if hasattr(edge_result, 'records') else edge_result
            for rec in records:
                data = rec.data() if hasattr(rec, 'data') else dict(rec)
                edge_hits.append(data['fact'])
        except Exception as e:
            pass  # some queries may fail

        # Also try a broader search with individual answer words
        if not edge_hits and ans_words:
            for search_word in ans_words[:3]:
                if len(search_word) < 3:
                    continue
                try:
                    r2 = await graphiti.driver.execute_query(
                        """
                        MATCH ()-[e:RELATES_TO]-()
                        WHERE e.group_id = $group_id
                          AND toLower(e.fact) CONTAINS $word
                        RETURN e.fact AS fact
                        LIMIT 5
                        """,
                        group_id=group_id,
                        word=search_word,
                    )
                    records = r2.records if hasattr(r2, 'records') else r2
                    for rec in records:
                        data = rec.data() if hasattr(rec, 'data') else dict(rec)
                        f = data['fact']
                        if _answer_in_text(answer, f):
                            edge_hits.append(f)
                except:
                    pass

        # Deduplicate
        edge_hits = list(dict.fromkeys(edge_hits))

        # ---------------------------------------------------------------
        # CHECK 2: Does the answer exist in entity NODES?
        # ---------------------------------------------------------------
        node_hits = []
        for search_word in ans_words[:3]:
            if len(search_word) < 3:
                continue
            try:
                nr = await graphiti.driver.execute_query(
                    """
                    MATCH (n:Entity)
                    WHERE n.group_id = $group_id
                      AND (toLower(n.name) CONTAINS $word
                           OR toLower(n.summary) CONTAINS $word)
                    RETURN n.name AS name, n.summary AS summary
                    LIMIT 5
                    """,
                    group_id=group_id,
                    word=search_word,
                )
                records = nr.records if hasattr(nr, 'records') else nr
                for rec in records:
                    data = rec.data() if hasattr(rec, 'data') else dict(rec)
                    text = f"{data['name']}: {data.get('summary', '')}"
                    if _answer_in_text(answer, text):
                        node_hits.append(text[:100])
            except:
                pass

        node_hits = list(dict.fromkeys(node_hits))

        # ---------------------------------------------------------------
        # CHECK 3: Graphiti search results at different depths
        # ---------------------------------------------------------------
        search_rank = None  # rank where answer first appears
        top5_facts = []

        try:
            results = await graphiti.search(
                question,
                group_ids=[group_id],
                num_results=50,  # wider net
            )

            for ri, r in enumerate(results):
                fact_text = _extract_fact_text(r)
                if ri < 5:
                    top5_facts.append(fact_text[:80])
                if _answer_in_text(answer, fact_text):
                    if search_rank is None:
                        search_rank = ri + 1
        except Exception as e:
            top5_facts = [f"Search error: {e}"]

        # ---------------------------------------------------------------
        # Diagnosis
        # ---------------------------------------------------------------
        in_edges = len(edge_hits) > 0
        in_nodes = len(node_hits) > 0
        in_graph = in_edges or in_nodes

        if not in_graph:
            status = "🔴 NOT IN GRAPH"
            explanation = "Answer was never extracted as entity/edge"
        elif search_rank is not None and search_rank <= 5:
            status = f"🟢 FOUND @ rank {search_rank}"
            explanation = "In graph AND in top 5 search"
        elif search_rank is not None and search_rank <= 10:
            status = f"🟡 FOUND @ rank {search_rank}"
            explanation = "In graph, in top 10 but not top 5"
        elif search_rank is not None:
            status = f"🟠 FOUND @ rank {search_rank}"
            explanation = "In graph but buried deep in results"
        else:
            status = "🔵 IN GRAPH, SEARCH MISSES"
            explanation = "Exists in Neo4j but Graphiti search doesn't find it"

        print(f"  Status: {status}")
        print(f"  Explanation: {explanation}")
        if edge_hits:
            print(f"  Edge matches: {edge_hits[0][:80]}...")
        if node_hits:
            print(f"  Node matches: {node_hits[0][:80]}...")
        if search_rank:
            print(f"  Search rank: {search_rank}/50")
        print(f"  Top 5 Graphiti results:")
        for i, f in enumerate(top5_facts[:5]):
            print(f"    {i+1}. {f}")

        diagnosis.append({
            'question_id': q['question_id'],
            'question': question,
            'answer': answer[:100],
            'question_type': q.get('question_type'),
            'in_edges': in_edges,
            'in_nodes': in_nodes,
            'in_graph': in_graph,
            'search_rank': search_rank,
            'status': status,
            'edge_hits': edge_hits[:3],
            'node_hits': node_hits[:3],
            'top5_facts': top5_facts,
        })

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"DIAGNOSIS SUMMARY")
    print(f"{'='*70}\n")

    not_in_graph = sum(1 for d in diagnosis if not d['in_graph'])
    in_graph_search_miss = sum(1 for d in diagnosis if d['in_graph'] and d['search_rank'] is None)
    found_deep = sum(1 for d in diagnosis if d['search_rank'] and d['search_rank'] > 10)
    found_top10 = sum(1 for d in diagnosis if d['search_rank'] and d['search_rank'] <= 10)
    found_top5 = sum(1 for d in diagnosis if d['search_rank'] and d['search_rank'] <= 5)

    print(f"  🔴 Not in graph (never extracted):     {not_in_graph}/25")
    print(f"  🔵 In graph but search misses:          {in_graph_search_miss}/25")
    print(f"  🟠 Found but deep (rank 11-50):         {found_deep}/25")
    print(f"  🟡 Found in top 10:                     {found_top10 - found_top5}/25")
    print(f"  🟢 Found in top 5:                      {found_top5}/25")

    print(f"\n  Total in graph: {25 - not_in_graph}/25")
    print(f"  Graphiti search ceiling (top 50): {found_top5 + (found_top10 - found_top5) + found_deep}/25")

    if not_in_graph > 15:
        print(f"\n  ⚠️  VERDICT: Graphiti isn't extracting most answer facts.")
        print(f"     The knowledge graph is too abstract for these specific queries.")
        print(f"     Consider: episode-level retrieval, or enriching edge extraction.")
    elif in_graph_search_miss > 10:
        print(f"\n  ⚠️  VERDICT: Facts exist but Graphiti search can't find them.")
        print(f"     Consider: BM25 tuning, embedding quality, or direct Cypher search.")
    elif found_deep > 5:
        print(f"\n  ⚠️  VERDICT: Facts exist and search finds them, but too deep.")
        print(f"     The decay reranker CAN help here — increase num_results or blend_weight.")
    else:
        print(f"\n  ✅ VERDICT: Facts are mostly findable. Focus on answer matching + reranking.")

    # Save
    out_path = POC_ARTIFACTS_DIR / 'retrieval_diagnosis.json'
    with open(out_path, 'w') as f:
        json.dump(diagnosis, f, indent=2, default=str)
    print(f"\n  Full diagnosis saved to {out_path}")

    await graphiti.close()


if __name__ == '__main__':
    asyncio.run(main())