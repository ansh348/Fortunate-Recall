# ============================================================
# LIFEMEMBENCH — CONVERSATION GENERATION: MARCUS THOMPSON
# Run in Claude Code: claude -p "$(cat generate_marcus.md)"
# ============================================================

## Your Task

Generate 35 naturalistic multi-session conversations between a simulated user ("Marcus") and an AI assistant. Save each as a JSON file in `sessions/`.

## Architecture

You simulate BOTH sides:

**MARCUS (the user):** A 43-year-old tire shop owner in Memphis. Read his full persona in `persona_002_marcus.yaml`. He reveals facts incidentally through real conversations — never announces them.

**THE ASSISTANT:** A helpful AI assistant. Each session is independent — the assistant has ZERO memory of previous sessions.

## Critical Rules for Marcus (User Agent)

1. **NEVER state facts as announcements.**
   - BAD: "I should mention that I'm a former Marine and I have Type 2 diabetes."
   - GOOD: "man my blood sugar was all over the place this morning. some days the metformin just doesn't cut it"

2. **Facts emerge from context, not from info dumps.**
   - BAD: "I drive a 2018 Ford F-150 and I play poker every Thursday."
   - GOOD: "my F-150 is making this grinding noise when I brake, it's an 18 so it's got some miles on it"

3. **Marcus has a consistent voice:**
   - Short, direct sentences. Working man's vocabulary.
   - Says "man", "brother", "appreciate it", "I tell you what"
   - Dry humor, understated
   - Not tech-savvy — asks basic questions about software, websites, marketing
   - Doesn't use emojis or "lol" — he's not that guy
   - Types from his phone sometimes — occasional typos, shorter messages
   - Opens up slowly about personal stuff
   - Practical — always thinking about costs and time

4. **Filler sessions are pure noise.** No critical facts from the timeline. Just a man running a tire shop asking an AI for help with normal stuff.

5. **Each session is 8-15 turns.**

6. **Marcus asks for practical things:** help writing emails, QuickBooks questions, business advice, marketing help. He's not philosophical — he wants solutions.

## Critical Rules for the Assistant

1. **ABSOLUTE ZERO cross-session memory.** The assistant has never spoken to Marcus before in each session. It does NOT know his name, his shop, his daughter, his truck, ANYTHING — unless Marcus says it in THIS session.

2. **The assistant must NEVER introduce specifics the user hasn't mentioned in THIS session.** If Marcus says "the expansion plan fell through," the assistant CANNOT say "the Germantown location?" unless Marcus said Germantown first. The assistant can only respond to what's in front of it.

3. **Be genuinely helpful.** Give real QuickBooks advice, real email drafts, real business guidance.

4. **Match Marcus's energy.** He's direct — don't overexplain. Keep responses practical and concise. Average 3-5 sentences. Longer only for substantive technical help (email drafts, accounting guidance, etc.).

## Critical Rule for Fact Continuity

**Do NOT re-explain established facts in later sessions.** If Marcus's diabetes was established in session 4, a later session should NOT have him explain "I have Type 2 diabetes and I take metformin" again. He might reference the TOPIC ("blood sugar's been off") but should not re-state the full fact as if informing someone for the first time. The benchmark tests whether memory systems remember early disclosures — re-stating them defeats the purpose.

## Session Plan

Read `persona_002_marcus.yaml` for the full session plan. Generate all 35 sessions sequentially.

**Date mapping (approximate — 2-3 sessions per month):**
- Sessions 1-4: November 2024
- Sessions 5-7: December 2024
- Sessions 8-10: January 2025
- Sessions 11-13: February 2025
- Sessions 14-16: March 2025
- Sessions 17-19: April 2025
- Sessions 20-22: May 2025
- Sessions 23-25: June-July 2025
- Sessions 26-28: August-October 2025
- Sessions 29-31: November 2025 - January 2026
- Sessions 32-35: February-May 2026

## Output Format

```json
{
  "session_id": 1,
  "persona": "marcus_thompson",
  "date": "2024-11-05",
  "type": "critical",
  "topic": "business costs / ordering inventory",
  "facts_disclosed": ["Uses Discount Tire Direct as supplier"],
  "turns": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Save to: `sessions/session_XX.json`

## Quality Checks (Do These After Generation)

- [ ] Every fact disclosed in its designated session — not before, not after
- [ ] No fact from a LATER session appears in an EARLIER session
- [ ] Retraction in session 21 explicitly negates the Germantown expansion
- [ ] Filler sessions contain ZERO critical facts from the timeline
- [ ] Marcus's voice is consistent: short sentences, practical, "man", "brother", "appreciate it"
- [ ] The assistant NEVER references information from other sessions
- [ ] The assistant NEVER introduces specifics Marcus hasn't said in the current session
- [ ] Established facts are NOT re-explained in later sessions
- [ ] Conversations are practical and grounded — Marcus wants solutions, not therapy

## Start

Read the persona file, create `sessions/` directory, generate all 35 sessions. Quality over speed.
