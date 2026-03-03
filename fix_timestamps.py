r"""
fix_timestamps.py - Assign edge timestamps by session order.

Problem (v1): All 5,861 edges were bulk-ingested in ~3.5 hours, so created_at
timestamps are nearly identical — decay computes exp(-lambda * ~0) = ~1.0.

Problem (v2): The previous fix linearly remapped by original creation time, but
that ordering has no correlation with session chronology. Session 3000's edges
could end up with timestamps from month 1 and session 100's from month 5.

Fix (v3): Use Episodic nodes to recover the true session order.
    1. Fetch all edges with their episode UUIDs
    2. Fetch all Episodic nodes and parse session index from ep.name ("full_s{N}")
    3. Map each edge to its highest session index (most recent session it appeared in)
    4. Assign timestamps by session order: session 0 → 180 days ago, max session → 1 hour ago
    5. Edges within the same session get spread across a 30-minute window
    6. Batch update in Neo4j

This means edges are timestamped in the order they were *discussed*, not ingested.

Usage:
    python fix_timestamps.py
    python evaluate_v4.py --evaluate
"""

import asyncio
import math
import os
import re
import sys
import time as time_module
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta

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

# Regex to parse session index from Episodic node name like "full_s123"
SESSION_RE = re.compile(r"full_s(\d+)")


async def main():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    group_id = "full_234"

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    # -------------------------------------------------------------------
    # Step 1: Fetch edge → episode mapping
    # -------------------------------------------------------------------
    print("Fetching edge → episode mapping...")
    result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
        UNWIND e.episodes AS ep_uuid
        WITH e.uuid AS edge_uuid, ep_uuid
        MATCH (ep:Episodic {uuid: ep_uuid})
        RETURN edge_uuid, ep.name AS ep_name
        """,
        gid=group_id,
    )
    records = result.records if hasattr(result, "records") else result

    # Build edge_uuid → set of session indices
    edge_sessions: dict[str, set[int]] = defaultdict(set)
    unknown_ep_names = set()
    for rec in records:
        d = rec.data() if hasattr(rec, "data") else dict(rec)
        edge_uuid = d["edge_uuid"]
        ep_name = d["ep_name"]
        m = SESSION_RE.search(ep_name or "")
        if m:
            edge_sessions[edge_uuid].add(int(m.group(1)))
        else:
            unknown_ep_names.add(ep_name)

    print(f"  Mapped {len(edge_sessions)} edges to sessions via episodes")
    if unknown_ep_names:
        print(f"  (Skipped {len(unknown_ep_names)} unrecognised episode names: {list(unknown_ep_names)[:5]})")

    # -------------------------------------------------------------------
    # Step 2: Fetch ALL edges (some may have no episode match)
    # -------------------------------------------------------------------
    print("Fetching all edges...")
    all_edges_result = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
        RETURN e.uuid AS uuid
        """,
        gid=group_id,
    )
    all_records = all_edges_result.records if hasattr(all_edges_result, "records") else all_edges_result
    all_edge_uuids = [
        (rec.data() if hasattr(rec, "data") else dict(rec))["uuid"]
        for rec in all_records
    ]
    print(f"  Total edges: {len(all_edge_uuids)}")

    if not all_edge_uuids:
        print("No edges found. Exiting.")
        await driver.close()
        return

    # -------------------------------------------------------------------
    # Step 3: Assign each edge its session index
    # -------------------------------------------------------------------
    # For edges with multiple episodes, use the HIGHEST session index (most recent)
    # For edges with no episode match, assign to the median session index
    all_session_indices = set()
    for s in edge_sessions.values():
        all_session_indices |= s

    if all_session_indices:
        sorted_indices = sorted(all_session_indices)
        median_session = sorted_indices[len(sorted_indices) // 2]
    else:
        median_session = 0

    edge_to_session: dict[str, int] = {}
    no_session_count = 0
    for uuid in all_edge_uuids:
        if uuid in edge_sessions:
            edge_to_session[uuid] = max(edge_sessions[uuid])
        else:
            edge_to_session[uuid] = median_session
            no_session_count += 1

    if no_session_count:
        print(f"  {no_session_count} edges had no episode match -> assigned to median session {median_session}")

    # -------------------------------------------------------------------
    # Step 4: Group edges by session, sort sessions chronologically
    # -------------------------------------------------------------------
    session_to_edges: dict[int, list[str]] = defaultdict(list)
    for uuid, sess_idx in edge_to_session.items():
        session_to_edges[sess_idx].append(uuid)

    sorted_sessions = sorted(session_to_edges.keys())
    num_sessions = len(sorted_sessions)
    total_edges = len(all_edge_uuids)

    print(f"  {num_sessions} distinct sessions (range: {sorted_sessions[0]} .. {sorted_sessions[-1]})")

    # -------------------------------------------------------------------
    # Step 5: Compute timestamps by session order
    # -------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    span_days = 180  # 6 months
    oldest_target = now - timedelta(days=span_days)
    newest_target = now - timedelta(hours=1)
    oldest_ts = oldest_target.timestamp()
    newest_ts = newest_target.timestamp()
    total_span = newest_ts - oldest_ts

    # Each session gets a base time, linearly interpolated by session rank
    # Edges within a session spread across a 30-minute window
    intra_session_window = 30 * 60  # 30 minutes in seconds

    print(f"\nTarget timeline:")
    print(f"  Oldest session -> {oldest_target.strftime('%Y-%m-%d %H:%M')} ({span_days} days ago)")
    print(f"  Newest session -> {newest_target.strftime('%Y-%m-%d %H:%M')} (1 hour ago)")
    print(f"  Span: {total_span/3600:.0f} hours ({total_span/86400:.0f} days)")

    updates: list[tuple[str, str]] = []  # (uuid, iso_timestamp)

    for rank, sess_idx in enumerate(sorted_sessions):
        # Session base time: linear interpolation by rank
        if num_sessions > 1:
            frac = rank / (num_sessions - 1)
        else:
            frac = 1.0
        session_base_ts = oldest_ts + frac * total_span

        edges_in_session = session_to_edges[sess_idx]
        n_edges = len(edges_in_session)

        for j, uuid in enumerate(edges_in_session):
            # Spread within the 30-minute window
            if n_edges > 1:
                intra_offset = (j / (n_edges - 1)) * intra_session_window
            else:
                intra_offset = 0.0
            edge_ts = session_base_ts + intra_offset
            edge_dt = datetime.fromtimestamp(edge_ts, tz=timezone.utc)
            updates.append((uuid, edge_dt.isoformat()))

    # Show samples
    print(f"\nSample session → timestamp mapping:")
    sample_ranks = [0, num_sessions // 4, num_sessions // 2, 3 * num_sessions // 4, num_sessions - 1]
    running_edge_idx = 0
    for rank, sess_idx in enumerate(sorted_sessions):
        n_in = len(session_to_edges[sess_idx])
        if rank in sample_ranks:
            # Show the first edge of this session
            uuid_short = updates[running_edge_idx][0][:12]
            ts_short = updates[running_edge_idx][1][:19]
            print(f"  Session {sess_idx:>6d} (rank {rank:>5d}, {n_in:>3d} edges) -> {ts_short}  ({uuid_short}...)")
        running_edge_idx += n_in

    # -------------------------------------------------------------------
    # Step 6: Batch update in Neo4j
    # -------------------------------------------------------------------
    print(f"\nUpdating {total_edges} edges in Neo4j...")

    batch_size = 200
    updated = 0
    t0 = time_module.time()

    for start in range(0, total_edges, batch_size):
        batch = updates[start : start + batch_size]
        params = [{"uuid": uuid, "ts": ts} for uuid, ts in batch]

        await driver.execute_query(
            """
            UNWIND $params AS p
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.uuid = p.uuid
            SET e.created_at = datetime(p.ts)
            """,
            params=params,
        )

        updated += len(batch)
        if updated % 1000 == 0 or updated == total_edges:
            elapsed = time_module.time() - t0
            print(f"  Updated {updated}/{total_edges} ({elapsed:.0f}s)")

    elapsed = time_module.time() - t0
    print(f"\nDone: {updated} edges updated in {elapsed:.0f}s")

    # -------------------------------------------------------------------
    # Step 7: Verify
    # -------------------------------------------------------------------
    verify = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.group_id = $gid
        RETURN min(e.created_at) AS oldest, max(e.created_at) AS newest
        """,
        gid=group_id,
    )
    vrec = verify.records[0]
    v_oldest = vrec["oldest"]
    v_newest = vrec["newest"]

    if hasattr(v_oldest, "to_native"):
        new_spread_hours = (v_newest.to_native().timestamp() - v_oldest.to_native().timestamp()) / 3600
    elif hasattr(v_oldest, "timestamp"):
        new_spread_hours = (v_newest.timestamp() - v_oldest.timestamp()) / 3600
    else:
        new_spread_hours = 0

    print(f"\nVerification:")
    print(f"  Oldest edge: {v_oldest}")
    print(f"  Newest edge: {v_newest}")
    print(f"  New spread: {new_spread_hours:.0f} hours ({new_spread_hours/24:.0f} days)")

    # -------------------------------------------------------------------
    # Step 8: Show expected decay differentiation
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Expected decay differentiation at key time points:")
    print(f"{'='*60}")
    print(f"{'Age':>12s} {'IDENTITY':>10s} {'RELATION':>10s} {'HEALTH':>10s} {'PREFS':>10s} {'HOBBIES':>10s} {'LOGISTIC':>10s}")

    rates = {
        "IDENTITY": 0.0005,
        "RELATION": 0.0015,
        "HEALTH": 0.0025,
        "PREFS": 0.0050,
        "HOBBIES": 0.0035,
        "LOGISTIC": 0.0080,
    }

    for hours_ago, label in [(1, "1 hour"), (24, "1 day"), (168, "1 week"),
                              (720, "1 month"), (2160, "3 months"), (4320, "6 months")]:
        vals = {k: math.exp(-r * hours_ago) for k, r in rates.items()}
        row = f"{label:>12s}"
        for k in rates:
            row += f" {vals[k]:>10.4f}"
        print(row)

    print(f"\nNow run: python evaluate_v4.py --evaluate --sweep")
    print(f"With the flipped alpha formula:")
    print(f"  alpha=0.0 -> pure semantic (engines should tie)")
    print(f"  alpha=1.0 -> pure activation (engines should diverge)")

    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
