import json
oracle = json.load(open('LongMemEval/data/longmemeval_oracle.json', encoding='utf-8'))
s_data = json.load(open('LongMemEval/data/longmemeval_s_cleaned.json', encoding='utf-8'))
s_lookup = {q['question_id']: q for q in s_data}
poc = json.load(open('LongMemEval/data/poc_artifacts/poc_questions.json', encoding='utf-8'))
poc_ids = {q['question_id'] for q in poc}
total = sum(len(s_lookup.get(qid, {}).get('haystack_sessions', [])) for qid in poc_ids)
print(f'PoC questions: {len(poc_ids)}')
print(f'PoC raw sessions: {total}')
print(f'Cost per session: {17/total:.4f}')
print(f'Full run estimate: {9977 * 17/total:.0f}')