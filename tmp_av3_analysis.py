import json
import sys

# Load all three files
with open("LifeMemEval/artifacts/lifemembench_results.json", "r", encoding="utf-8") as f:
    fr_results = json.load(f)

with open("LifeMemEval/artifacts/mem0_results.json", "r", encoding="utf-8") as f:
    mem0_results = json.load(f)

with open("LifeMemEval/lifemembench_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# Build lookup dicts
# Questions by question_id
q_lookup = {}
for q in questions:
    q_lookup[q["id"]] = q

# FR results by config -> question_id
fr_lookup = {}
for config in ["full", "uniform", "baseline"]:
    fr_lookup[config] = {}
    for entry in fr_results["per_question"][config]:
        fr_lookup[config][entry["question_id"]] = entry

# Mem0 results by question_id (check structure)
mem0_lookup = {}
if "per_question" in mem0_results:
    # Try different possible structures
    if isinstance(mem0_results["per_question"], dict):
        for config_name, entries in mem0_results["per_question"].items():
            if isinstance(entries, list):
                for entry in entries:
                    mem0_lookup[entry["question_id"]] = entry
                break  # just use the first/only config
    elif isinstance(mem0_results["per_question"], list):
        for entry in mem0_results["per_question"]:
            mem0_lookup[entry["question_id"]] = entry

# If mem0_lookup is still empty, try to find the right key
if not mem0_lookup:
    print("DEBUG: mem0_results keys:", list(mem0_results.keys()))
    if "per_question" in mem0_results:
        pq = mem0_results["per_question"]
        if isinstance(pq, dict):
            print("DEBUG: per_question sub-keys:", list(pq.keys()))
            for k, v in pq.items():
                if isinstance(v, list) and len(v) > 0:
                    print(f"DEBUG: '{k}' has {len(v)} entries, first entry keys: {list(v[0].keys())}")
                    for entry in v:
                        mem0_lookup[entry["question_id"]] = entry

# Identify AV3 failures under "full" config
av3_failures = []
for entry in fr_results["per_question"]["full"]:
    if entry.get("attack_vector") == "AV3_stable_identity" and entry.get("av_pass") == False:
        av3_failures.append(entry)

print(f"=" * 120)
print(f"AV3 (Stable Identity) FAILURE ANALYSIS")
print(f"Total AV3 failures under 'full' config: {len(av3_failures)}")
print(f"=" * 120)

# Classify each failure
decay_killed = []
retrieval_miss = []
partial = []

for entry in av3_failures:
    qid = entry["question_id"]
    uniform_entry = fr_lookup["uniform"].get(qid)
    baseline_entry = fr_lookup["baseline"].get(qid)

    uniform_pass = uniform_entry.get("av_pass", False) if uniform_entry else False
    baseline_pass = baseline_entry.get("av_pass", False) if baseline_entry else False

    if uniform_pass and baseline_pass:
        decay_killed.append(qid)
    elif uniform_pass or baseline_pass:
        # passes under exactly one
        if uniform_pass and not baseline_pass:
            partial.append((qid, "uniform"))
        elif baseline_pass and not uniform_pass:
            partial.append((qid, "baseline"))
    else:
        retrieval_miss.append(qid)

# Also handle case: both pass = DECAY_KILLED, one passes = PARTIAL
# Re-check: DECAY_KILLED means passes under uniform OR baseline (either or both)
# Let me re-read the requirement more carefully:
# A) DECAY_KILLED: av_pass=True under "uniform" OR "baseline", but False under "full"
# C) PARTIAL: av_pass=False under full, but passes under exactly one of uniform/baseline (but not both)
# So DECAY_KILLED = passes under BOTH, PARTIAL = passes under exactly one

# Actually re-reading: "DECAY_KILLED: av_pass=True under 'uniform' OR 'baseline'"
# This could mean either. But then PARTIAL says "exactly one (but not both)"
# So DECAY_KILLED = both pass, PARTIAL = exactly one passes
# This is consistent with my classification above.

# Wait, re-reading again more carefully:
# DECAY_KILLED: passes under uniform OR baseline (i.e., at least one)
# PARTIAL: passes under exactly one
# These overlap. Let me re-interpret:
# DECAY_KILLED = passes under BOTH uniform AND baseline
# PARTIAL = passes under exactly one
# RETRIEVAL_MISS = fails under all

# That's what I have. Let's proceed.

print(f"\n{'=' * 120}")
print(f"CLASSIFICATION SUMMARY")
print(f"{'=' * 120}")
print(f"  DECAY_KILLED (passes both uniform & baseline, fails full): {len(decay_killed)}")
print(f"  PARTIAL (passes exactly one of uniform/baseline):          {len(partial)}")
print(f"  RETRIEVAL_MISS (fails all configs):                        {len(retrieval_miss)}")
print(f"  TOTAL:                                                     {len(decay_killed) + len(partial) + len(retrieval_miss)}")

# ===== SECTION A: DECAY_KILLED =====
print(f"\n{'=' * 120}")
print(f"A) DECAY_KILLED FAILURES ({len(decay_killed)} questions)")
print(f"{'=' * 120}")
header = f"{'QID':<12} {'Rank(uni)':<10} {'Rank(full)':<11} {'Act(full)':<10} {'Sem(full)':<10} {'Act(uni)':<10} {'Sem(uni)':<10} {'Blend(f)':<10} {'Blend(u)':<10}"
print(header)
print("-" * len(header))

for qid in sorted(decay_killed):
    full_entry = fr_lookup["full"][qid]
    uni_entry = fr_lookup["uniform"][qid]

    # Find correct edge in full top5
    correct_full = None
    for fact in full_entry.get("top5_facts", []):
        if fact.get("supports_correct"):
            correct_full = fact
            break

    # Find correct edge in uniform top5
    correct_uni = None
    for fact in uni_entry.get("top5_facts", []):
        if fact.get("supports_correct"):
            correct_uni = fact
            break

    act_f = f"{correct_full['activation']:.4f}" if correct_full and correct_full.get('activation') is not None else "N/A"
    sem_f = f"{correct_full['semantic']:.4f}" if correct_full and correct_full.get('semantic') is not None else "N/A"
    blend_f = f"{correct_full['blended']:.4f}" if correct_full and correct_full.get('blended') is not None else "N/A"

    act_u = f"{correct_uni['activation']:.4f}" if correct_uni and correct_uni.get('activation') is not None else "N/A"
    sem_u = f"{correct_uni['semantic']:.4f}" if correct_uni and correct_uni.get('semantic') is not None else "N/A"
    blend_u = f"{correct_uni['blended']:.4f}" if correct_uni and correct_uni.get('blended') is not None else "N/A"

    rank_uni = uni_entry.get("answer_rank", "N/A")
    rank_full = full_entry.get("answer_rank", "N/A")

    print(f"{qid:<12} {str(rank_uni):<10} {str(rank_full):<11} {act_f:<10} {sem_f:<10} {act_u:<10} {sem_u:<10} {blend_f:<10} {blend_u:<10}")

# ===== SECTION B: RETRIEVAL_MISS =====
print(f"\n{'=' * 120}")
print(f"B) RETRIEVAL_MISS FAILURES ({len(retrieval_miss)} questions)")
print(f"{'=' * 120}")
header = f"{'QID':<12} {'Mem0 Pass':<10} {'Mem0 Rank':<10} {'FR Rank(f)':<11} {'FR Rank(u)':<11} {'FR Rank(b)':<11}"
print(header)
print("-" * len(header))

for qid in sorted(retrieval_miss):
    mem0_entry = mem0_lookup.get(qid, {})
    mem0_pass = mem0_entry.get("av_pass", "N/A")
    mem0_rank = mem0_entry.get("answer_rank", "N/A")

    fr_rank_f = fr_lookup["full"][qid].get("answer_rank", "N/A")
    fr_rank_u = fr_lookup["uniform"].get(qid, {}).get("answer_rank", "N/A")
    fr_rank_b = fr_lookup["baseline"].get(qid, {}).get("answer_rank", "N/A")

    print(f"{qid:<12} {str(mem0_pass):<10} {str(mem0_rank):<10} {str(fr_rank_f):<11} {str(fr_rank_u):<11} {str(fr_rank_b):<11}")

# ===== SECTION C: PARTIAL =====
print(f"\n{'=' * 120}")
print(f"C) PARTIAL FAILURES ({len(partial)} questions)")
print(f"{'=' * 120}")
header = f"{'QID':<12} {'Passes Under':<13} {'Rank(full)':<11} {'Rank(uni)':<10} {'Rank(base)':<11}"
print(header)
print("-" * len(header))

for qid, passes_config in sorted(partial):
    fr_rank_f = fr_lookup["full"][qid].get("answer_rank", "N/A")
    fr_rank_u = fr_lookup["uniform"].get(qid, {}).get("answer_rank", "N/A")
    fr_rank_b = fr_lookup["baseline"].get(qid, {}).get("answer_rank", "N/A")

    print(f"{qid:<12} {passes_config:<13} {str(fr_rank_f):<11} {str(fr_rank_u):<10} {str(fr_rank_b):<11}")

# ===== SECTION 3: Correct edge in top5 for ALL 26 failures =====
print(f"\n{'=' * 120}")
print(f"3) CORRECT EDGE IN TOP-5 FACTS (full config) FOR ALL {len(av3_failures)} FAILURES")
print(f"{'=' * 120}")
header = f"{'QID':<12} {'Category':<15} {'In Top5?':<9} {'Rank':<6} {'Activation':<11} {'Semantic':<10} {'Blended':<10} {'Source':<20}"
print(header)
print("-" * len(header))

for entry in sorted(av3_failures, key=lambda x: x["question_id"]):
    qid = entry["question_id"]

    # Determine category
    if qid in decay_killed:
        cat = "DECAY_KILLED"
    elif qid in retrieval_miss:
        cat = "RETRIEVAL_MISS"
    else:
        cat = "PARTIAL"

    correct_fact = None
    for fact in entry.get("top5_facts", []):
        if fact.get("supports_correct"):
            correct_fact = fact
            break

    if correct_fact:
        rank = correct_fact.get("rank", "N/A")
        act = f"{correct_fact['activation']:.4f}" if correct_fact.get('activation') is not None else "N/A"
        sem = f"{correct_fact['semantic']:.4f}" if correct_fact.get('semantic') is not None else "N/A"
        blend = f"{correct_fact['blended']:.4f}" if correct_fact.get('blended') is not None else "N/A"
        source = correct_fact.get("source", "N/A")
        in_top5 = "YES"
    else:
        rank = "-"
        act = "-"
        sem = "-"
        blend = "-"
        source = "-"
        in_top5 = "NO"

    print(f"{qid:<12} {cat:<15} {in_top5:<9} {str(rank):<6} {act:<11} {sem:<10} {blend:<10} {source:<20}")

# Count how many have correct in top5
has_correct = sum(1 for e in av3_failures if any(f.get("supports_correct") for f in e.get("top5_facts", [])))
print(f"\nCorrect edge found in top-5: {has_correct}/{len(av3_failures)}")

# ===== SECTION 4: Mem0 Gap Decomposition =====
print(f"\n{'=' * 120}")
print(f"4) MEM0 GAP DECOMPOSITION")
print(f"{'=' * 120}")

mem0_also_fails = 0
mem0_passes_fr_fails = 0
mem0_pass_categories = {"DECAY_KILLED": 0, "RETRIEVAL_MISS": 0, "PARTIAL": 0}
mem0_fail_categories = {"DECAY_KILLED": 0, "RETRIEVAL_MISS": 0, "PARTIAL": 0}

for entry in av3_failures:
    qid = entry["question_id"]
    mem0_entry = mem0_lookup.get(qid, {})
    mem0_pass = mem0_entry.get("av_pass", False)

    # Determine category
    if qid in decay_killed:
        cat = "DECAY_KILLED"
    elif qid in retrieval_miss:
        cat = "RETRIEVAL_MISS"
    else:
        cat = "PARTIAL"

    if mem0_pass:
        mem0_passes_fr_fails += 1
        mem0_pass_categories[cat] += 1
    else:
        mem0_also_fails += 1
        mem0_fail_categories[cat] += 1

print(f"  Mem0 ALSO fails:        {mem0_also_fails}/{len(av3_failures)}")
print(f"  Mem0 passes, FR fails:  {mem0_passes_fr_fails}/{len(av3_failures)}")
print()
print(f"  Mem0-passes/FR-fails by category:")
for cat in ["DECAY_KILLED", "RETRIEVAL_MISS", "PARTIAL"]:
    print(f"    {cat}: {mem0_pass_categories[cat]}")
print()
print(f"  Mem0-also-fails by category:")
for cat in ["DECAY_KILLED", "RETRIEVAL_MISS", "PARTIAL"]:
    print(f"    {cat}: {mem0_fail_categories[cat]}")

# Detailed table
print(f"\n  Detailed Mem0 vs FR comparison for all {len(av3_failures)} AV3 failures:")
header = f"  {'QID':<12} {'FR Category':<15} {'Mem0 Pass':<10} {'Mem0 Rank':<10}"
print(header)
print("  " + "-" * (len(header) - 2))
for entry in sorted(av3_failures, key=lambda x: x["question_id"]):
    qid = entry["question_id"]
    mem0_entry = mem0_lookup.get(qid, {})
    mem0_pass = mem0_entry.get("av_pass", "N/A")
    mem0_rank = mem0_entry.get("answer_rank", "N/A")

    if qid in decay_killed:
        cat = "DECAY_KILLED"
    elif qid in retrieval_miss:
        cat = "RETRIEVAL_MISS"
    else:
        cat = "PARTIAL"

    print(f"  {qid:<12} {cat:<15} {str(mem0_pass):<10} {str(mem0_rank):<10}")

# ===== SECTION 5: DECAY_KILLED activation comparison =====
print(f"\n{'=' * 120}")
print(f"5) DECAY_KILLED: ACTIVATION SCORE COMPARISON (full vs uniform)")
print(f"{'=' * 120}")
header = f"{'QID':<12} {'Act(full)':<11} {'Act(uni)':<11} {'Delta':<10} {'Sem(full)':<11} {'Sem(uni)':<11} {'Blend(f)':<11} {'Blend(u)':<11}"
print(header)
print("-" * len(header))

for qid in sorted(decay_killed):
    full_entry = fr_lookup["full"][qid]
    uni_entry = fr_lookup["uniform"][qid]

    correct_full = None
    for fact in full_entry.get("top5_facts", []):
        if fact.get("supports_correct"):
            correct_full = fact
            break

    correct_uni = None
    for fact in uni_entry.get("top5_facts", []):
        if fact.get("supports_correct"):
            correct_uni = fact
            break

    act_f = correct_full['activation'] if correct_full and correct_full.get('activation') is not None else None
    act_u = correct_uni['activation'] if correct_uni and correct_uni.get('activation') is not None else None
    sem_f = correct_full['semantic'] if correct_full and correct_full.get('semantic') is not None else None
    sem_u = correct_uni['semantic'] if correct_uni and correct_uni.get('semantic') is not None else None
    blend_f = correct_full['blended'] if correct_full and correct_full.get('blended') is not None else None
    blend_u = correct_uni['blended'] if correct_uni and correct_uni.get('blended') is not None else None

    delta = f"{act_f - act_u:.4f}" if act_f is not None and act_u is not None else "N/A"

    print(f"{qid:<12} {(f'{act_f:.4f}' if act_f is not None else 'N/A'):<11} {(f'{act_u:.4f}' if act_u is not None else 'N/A'):<11} {delta:<10} {(f'{sem_f:.4f}' if sem_f is not None else 'N/A'):<11} {(f'{sem_u:.4f}' if sem_u is not None else 'N/A'):<11} {(f'{blend_f:.4f}' if blend_f is not None else 'N/A'):<11} {(f'{blend_u:.4f}' if blend_u is not None else 'N/A'):<11}")

# Averages
act_deltas = []
for qid in decay_killed:
    full_entry = fr_lookup["full"][qid]
    uni_entry = fr_lookup["uniform"][qid]
    cf = next((f for f in full_entry.get("top5_facts", []) if f.get("supports_correct")), None)
    cu = next((f for f in uni_entry.get("top5_facts", []) if f.get("supports_correct")), None)
    if cf and cu and cf.get("activation") is not None and cu.get("activation") is not None:
        act_deltas.append(cf["activation"] - cu["activation"])

if act_deltas:
    print(f"\nAverage activation delta (full - uniform): {sum(act_deltas)/len(act_deltas):.4f} (across {len(act_deltas)} questions with data)")

# ===== SECTION 6: Correct answers for all 26 failures =====
print(f"\n{'=' * 120}")
print(f"6) CORRECT ANSWERS FOR ALL {len(av3_failures)} AV3 FAILURES")
print(f"{'=' * 120}")
header = f"{'QID':<12} {'Category':<15} {'Correct Answer':<60} {'Question (truncated)':<50}"
print(header)
print("-" * len(header))

for entry in sorted(av3_failures, key=lambda x: x["question_id"]):
    qid = entry["question_id"]
    q_data = q_lookup.get(qid, {})
    correct_ans = q_data.get("correct_answer", q_data.get("answer", "N/A"))
    question_text = q_data.get("question", "N/A")[:47] + "..." if len(q_data.get("question", "")) > 47 else q_data.get("question", "N/A")

    if qid in decay_killed:
        cat = "DECAY_KILLED"
    elif qid in retrieval_miss:
        cat = "RETRIEVAL_MISS"
    else:
        cat = "PARTIAL"

    # Truncate correct answer if too long
    ans_display = str(correct_ans)[:57] + "..." if len(str(correct_ans)) > 57 else str(correct_ans)

    print(f"{qid:<12} {cat:<15} {ans_display:<60} {question_text:<50}")

# ===== EXTRA: Check all field names in top5_facts =====
print(f"\n{'=' * 120}")
print(f"APPENDIX: Sample top5_facts entry field names")
print(f"{'=' * 120}")
sample = av3_failures[0]
if sample.get("top5_facts") and len(sample["top5_facts"]) > 0:
    print(f"Fields: {list(sample['top5_facts'][0].keys())}")
    print(f"Sample entry: {json.dumps(sample['top5_facts'][0], indent=2)}")

# Also show a sample question entry
print(f"\nSample question entry fields: {list(list(q_lookup.values())[0].keys())}")

# Check all AV3 questions total (pass + fail)
av3_total = sum(1 for e in fr_results["per_question"]["full"] if e.get("attack_vector") == "AV3_stable_identity")
av3_pass = sum(1 for e in fr_results["per_question"]["full"] if e.get("attack_vector") == "AV3_stable_identity" and e.get("av_pass") == True)
print(f"\nAV3 overall: {av3_pass}/{av3_total} pass ({av3_pass/av3_total*100:.1f}%), {av3_total - av3_pass} fail")
