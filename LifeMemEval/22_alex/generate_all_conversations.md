# ============================================================
# LIFEMEMBENCH — CONVERSATION GENERATOR
# ============================================================
# Usage: Place this file in a persona directory alongside the persona YAML, then run:
#   claude -p "$(cat generate_conversations.md)"
#
# Example:
#   cd LifeMemEval/5_amara
#   copy ..\..\generate_conversations.md .
#   claude -p "$(cat generate_conversations.md)"
# ============================================================

## Your Task

Read the persona YAML file in this directory (the file matching `persona_*.yaml`). Generate 35 naturalistic multi-session conversations between a simulated user and an AI assistant. Save each as a JSON file in a `sessions/` subdirectory.

Use 7 parallel agents to generate sessions in batches for speed:
- Agent 1: Sessions 1-5
- Agent 2: Sessions 6-10
- Agent 3: Sessions 11-15
- Agent 4: Sessions 16-20
- Agent 5: Sessions 21-25
- Agent 6: Sessions 26-30
- Agent 7: Sessions 31-35

## Architecture

You simulate BOTH sides of the conversation:

**THE USER:** Their full persona, voice, and timeline are in the YAML file. They reveal facts incidentally through real conversations — never as announcements. Match their exact voice: sentence length, slang, verbal tics, formality, emoji habits, typing style. A 68-year-old British beekeeper sounds NOTHING like a 24-year-old Boston electrician.

**THE ASSISTANT:** A helpful AI assistant. Each session is completely independent — the assistant has ZERO memory of any other session. The assistant has never spoken to this user before.

## Output Format

Each session saved as `sessions/session_XX.json`:

```json
{
  "session_id": 1,
  "persona": "firstname_lastname",
  "date": "YYYY-MM-DD",
  "type": "critical",
  "topic": "topic from session_plan",
  "facts_disclosed": ["fact 1", "fact 2"],
  "turns": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

- `facts_disclosed`: For critical sessions, list the ground truth facts embedded. For filler sessions, use an empty array `[]`.
- `date`: Space sessions across 18 months (Nov 2024 — May 2026), roughly 2-3 sessions per month.
- `type`: "critical" or "filler" as specified in session_plan.

## Critical Rules for the User Agent

1. **Facts emerge naturally through conversation, NEVER as announcements.** The user is asking for help, venting, planning, or chatting. Facts come out because they're relevant to what they're doing, not because they're informing the AI.
   - BAD: "I have diabetes and I drive a Ford F-150."
   - GOOD: "my blood sugar was wild this morning, had to sit in the truck for 10 minutes before I could drive to the shop"

2. **Maintain the persona's distinct voice EXACTLY as described in the personality field.** Read it carefully. Match:
   - Sentence length
   - Capitalization and punctuation habits
   - Slang and verbal tics (the specific words/phrases listed)
   - Emoji usage (or lack thereof)
   - Formality level
   - Typing style (phone typos, abbreviations, etc.)

3. **Filler sessions contain ZERO critical facts.** No facts from stable_identity, supersession_events, expiring_logistics, contradictions, retractions, or ambiguous sections. Just a person living their life asking an AI for help with normal stuff. Use the filler_topics list in the YAML.

4. **Do NOT re-explain established facts in later sessions.** If a health condition was disclosed in session 4, a later session should NOT have the user explain it again from scratch. They might reference the TOPIC naturally but should not re-state the fact as if telling someone new.

5. **Each session is 8-15 turns** (turn = one user message + one assistant response). Critical sessions that embed multiple facts should be on the longer side (12-15). Simple filler can be shorter (8-10).

6. **Supersession new-facts must feel natural, not forced.** When someone mentions they switched jobs or moved, it comes up organically — not "I want to inform you that I have changed employers."

7. **Retractions must be explicit and definitive.** The user must clearly kill the plan: "forget about that", "that's dead", "off the table", "not happening."

8. **Contradictions must be implicit.** The user does NOT say "I know I said X before but now Y." They just say the contradicting thing naturally, with no awareness of the conflict.

9. **The how_disclosed fields in the YAML are INSPIRATION, not scripts.** Embed the same fact in the same conversational context, but use DIFFERENT wording. The YAML shows the scenario and the vibe — you write fresh dialogue that captures the same information naturally. Do NOT copy how_disclosed text verbatim into the conversation. If the YAML says the user mentions their brother is a GP while asking a medical question, write a NEW version of that scene with different words. Same fact, same context, different dialogue.

10. **Vary conversation openers.** Don't start every session with "hey can you help me with X." Real people sometimes:
    - Dive straight in: "what's the best way to..."
    - Vent first: "man I'm so tired of..."
    - Ask casually: "quick question about..."
    - Open with context: "so I'm dealing with this thing at work..."
    - No greeting at all, just the question
    - Random opener: "settle something for me"
    Match the persona's style. Jake says "yo" not "Hello." Tom says "I wonder if you might help me with something" not "hey."

## Critical Rules for the Assistant

1. **ABSOLUTE ZERO cross-session memory.** The assistant knows NOTHING about this user unless the user says it in THIS session. This is the single most important rule.

2. **The assistant must NEVER introduce specifics the user hasn't mentioned in THIS session.** If the user says "that plan fell through," the assistant can say "Sorry to hear that — what happened?" but CANNOT say "Oh no, the Germantown location didn't work out?"

3. **Be genuinely helpful.** Give real advice, real recommendations, real information. Not scripted placeholder responses.

4. **Match the user's register.** If they're casual, be casual. If they're formal, be professional. Average 3-5 sentences per response. Longer only for substantive help (drafts, technical guidance, research, etc.).

5. **Don't be excessively verbose.** Don't over-explain. Don't give 5 paragraphs when 2 sentences suffice. Match the energy of the conversation.

## Quality Checks (Run After Generation)

After generating all 35 sessions, run these checks:

```
For each session:
  1. [ ] facts_disclosed matches what's actually in the conversation
  2. [ ] User voice matches persona personality (check verbal tics, sentence length, slang)
  3. [ ] No facts from LATER sessions appear in EARLIER sessions
  4. [ ] Assistant NEVER references information from other sessions
  5. [ ] Assistant NEVER introduces specifics the user hasn't mentioned in THIS session
  6. [ ] Filler sessions contain ZERO critical facts from the YAML timeline
  7. [ ] Retraction session has explicit, definitive language killing the plan
  8. [ ] Contradiction session states the conflicting fact naturally, no self-awareness
  9. [ ] Session length is 8-15 turns
  10. [ ] Dates are chronologically ordered across all 35 sessions
  11. [ ] how_disclosed text from the YAML is NOT copied verbatim into conversations
  12. [ ] Conversation openers are varied (not all "hey can you help me with")
  13. [ ] Every session starts with a user message, not an assistant message
```

Print results as:
```
QUALITY CHECK RESULTS
Sessions with voice issues: [list]
Sessions with cross-session leaks: [list]
Sessions with fact timing errors: [list]
Filler sessions with fact contamination: [list]
Sessions with verbatim how_disclosed copying: [list]
Sessions with identical openers: [list]
ALL CHECKS PASSED: YES/NO
```

If any check fails, fix the session and re-verify.

## GO

1. Read the persona YAML in this directory
2. Create `sessions/` directory
3. Generate all 35 sessions using 7 parallel agents
4. Run quality checks
5. Fix any failures
6. Report results