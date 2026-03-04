"""Quick edge verification for retraction/contradiction edges."""
import asyncio, json, os
from pathlib import Path

env_path = Path('.env')
if env_path.exists():
    for line in open(env_path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key, val = line.split('=', 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

from neo4j import AsyncGraphDatabase

CHECKS = [
    ("omar_q12: career plan", "lifemembench_omar",
     ["century 21", "real estate license", "passed exam", "brokerage"]),
    ("priya_q09: migraine detail", "lifemembench_priya",
     ["migraine", "omega-3", "omega"]),
    ("tom_q07: social media contradiction", "lifemembench_tom",
     ["instagram", "social media", "bee photo"]),
    ("elena_q03: NP dropped", "lifemembench_elena",
     ["dropped", "afford", "student loan", "debt", "gave up np"]),
    ("jake_q05: EV dead", "lifemembench_jake",
     ["insurance", "dead", "ev charger", "not ready", "can not do"]),
    ("marcus_q03: Germantown scrapped", "lifemembench_marcus",
     ["scrapped", "too much risk", "decided against", "not going to"]),
    ("amara_q06: LLM dropped", "lifemembench_amara",
     ["dropped", "not worth", "decided against", "gave up", "stopped pursuing"]),
    ("amara_q11: non-boxing hobbies", "lifemembench_amara",
     ["cook", "nigerian", "podcast", "reading about justice"]),
    ("david_q04: abandoned book", "lifemembench_david",
     ["abandoned", "gave up", "nobody would read", "writing a book", "nonfiction"]),
    ("amara_q14: 18000 cost", "lifemembench_amara",
     ["18,000", "18000", "eighteen thousand", "tuition"]),
    ("tom_q06: barn scrapped", "lifemembench_tom",
     ["scrapped", "refused", "denied", "too expensive", "barn conversion"]),
    ("jake_q02: dating status", "lifemembench_jake",
     ["dating kayla", "broke up", "girlfriend", "seeing kayla"]),
    ("omar_q06: PS5 contradiction", "lifemembench_omar",
     ["never spends", "ps5", "playstation", "stress"]),
]

async def main():
    driver = AsyncGraphDatabase.driver(
        os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.environ.get('NEO4J_USER', 'neo4j'),
              os.environ.get('NEO4J_PASSWORD', 'testpassword123'))
    )
    for label, gid, keywords in CHECKS:
        print(f"\n=== {label} ===")
        conditions = " OR ".join(f'toLower(e.fact) CONTAINS "{kw}"' for kw in keywords)
        query = f"""
            MATCH (s)-[e:RELATES_TO]->(t)
            WHERE e.group_id = $gid AND ({conditions})
            RETURN e.fact AS fact, e.fr_primary_category AS cat, e.expired_at AS expired
        """
        r = await driver.execute_query(query, gid=gid)
        recs = r.records if hasattr(r, 'records') else r
        if not recs:
            print("  (no matches)")
        for rec in recs:
            d = rec.data() if hasattr(rec, 'data') else dict(rec)
            exp = ' [EXPIRED]' if d.get('expired') else ''
            print(f"  [{d.get('cat','?'):<25s}]{exp} {d['fact'][:130]}")
    await driver.close()

asyncio.run(main())
