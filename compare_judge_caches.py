"""Compare old vs new judge cache verdicts after prompt change."""
import json
from pathlib import Path

ARTIFACTS = Path("LifeMemEval/artifacts")
OLD_CACHE = ARTIFACTS / "lifemembench_judge_cache.json.bak4"
NEW_CACHE = ARTIFACTS / "lifemembench_judge_cache.json"


def main():
    old = json.load(open(OLD_CACHE, encoding="utf-8"))
    new = json.load(open(NEW_CACHE, encoding="utf-8"))

    shared_keys = set(old) & set(new)
    print(f"Old cache entries: {len(old)}")
    print(f"New cache entries: {len(new)}")
    print(f"Shared keys:       {len(shared_keys)}")
    print()

    false_to_true = []
    true_to_false = []
    wrong_changed = []

    for key in shared_keys:
        o_sc = old[key].get("supports_correct", False)
        n_sc = new[key].get("supports_correct", False)
        o_wi = old[key].get("contains_wrong_indicator", False)
        n_wi = new[key].get("contains_wrong_indicator", False)

        if not o_sc and n_sc:
            false_to_true.append(key)
        if o_sc and not n_sc:
            true_to_false.append(key)
        if o_wi != n_wi:
            wrong_changed.append(key)

    print("=== supports_correct changes ===")
    print(f"  false -> true:  {len(false_to_true)}")
    print(f"  true -> false:  {len(true_to_false)}")
    print(f"  net new HITs:   +{len(false_to_true) - len(true_to_false)}")
    print()
    print(f"=== contains_wrong_indicator changes: {len(wrong_changed)} ===")
    print()

    # --- Top-5 specific analysis (only priya questions) ---
    priya_old = {k: v for k, v in old.items() if v.get("question_id", "").startswith("priya_")}
    priya_new = {k: v for k, v in new.items() if v.get("question_id", "").startswith("priya_")}
    priya_shared = set(priya_old) & set(priya_new)

    p_ft = [k for k in priya_shared
            if not priya_old[k].get("supports_correct") and priya_new[k].get("supports_correct")]
    p_tf = [k for k in priya_shared
            if priya_old[k].get("supports_correct") and not priya_new[k].get("supports_correct")]

    print(f"=== Priya-only (top-5 edges): {len(priya_shared)} shared ===")
    print(f"  false -> true:  {len(p_ft)}")
    print(f"  true -> false:  {len(p_tf)}")
    print()

    # --- 5 example flips (false -> true) ---
    examples = p_ft[:5] if p_ft else false_to_true[:5]
    print("=" * 80)
    print("5 EXAMPLE VERDICT FLIPS (supports_correct: false -> true)")
    print("=" * 80)
    for i, key in enumerate(examples, 1):
        o = old.get(key) or priya_old.get(key, {})
        n = new.get(key) or priya_new.get(key, {})
        print(f"\n--- Example {i} ---")
        print(f"  question_id: {n.get('question_id', o.get('question_id', '?'))}")
        print(f"  edge_uuid:   {n.get('edge_uuid', o.get('edge_uuid', '?'))}")
        print(f"  OLD reasoning: {o.get('reasoning', '?')}")
        print(f"  NEW reasoning: {n.get('reasoning', '?')}")

    if not examples:
        print("\n  (No flips found in shared keys)")

    # --- wrong_indicator changes detail ---
    if wrong_changed:
        print()
        print("=" * 80)
        print("WRONG INDICATOR CHANGES")
        print("=" * 80)
        for key in wrong_changed[:5]:
            o = old[key]
            n = new[key]
            print(f"\n  question_id: {n.get('question_id')}")
            print(f"  old contains_wrong: {o.get('contains_wrong_indicator')} -> new: {n.get('contains_wrong_indicator')}")
            print(f"  OLD reasoning: {o.get('reasoning')}")
            print(f"  NEW reasoning: {n.get('reasoning')}")


if __name__ == "__main__":
    main()
