import json, hashlib
oracle = json.load(open('LongMemEval/data/longmemeval_oracle.json', encoding='utf-8'))
s_data = json.load(open('LongMemEval/data/longmemeval_s_cleaned.json', encoding='utf-8'))
s_lookup = {q['question_id']: q for q in s_data}
types = {'single-session-user','single-session-assistant','single-session-preference','knowledge-update'}
filtered = [q for q in oracle if q['question_type'] in types]
hashes = set()
total_raw = 0
for q in filtered:
    s = s_lookup.get(q['question_id'], {})
    for sess in s.get('haystack_sessions', []):
        total_raw += 1
        h = hashlib.sha256(json.dumps(sess, sort_keys=True).encode()).hexdigest()[:16]
        hashes.add(h)
print(f'Questions: {len(filtered)}')
print(f'Raw sessions: {total_raw}')
print(f'Unique sessions: {len(hashes)}')
print(f'Est cost: {len(hashes)*0.20}')