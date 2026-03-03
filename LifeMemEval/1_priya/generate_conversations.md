# ============================================================
# LIFEMEMBENCH — CONVERSATION GENERATION ORCHESTRATOR
# Run this in Claude Code with: claude -p "$(cat generate_conversations.md)"
# ============================================================

## Your Task

You are an orchestrator that generates naturalistic multi-session conversations between a simulated user ("Priya") and an AI assistant. You will produce 35 separate conversation sessions saved as individual JSON files.

## Architecture

You will simulate BOTH sides of the conversation:

**PRIYA (the user):** A 34-year-old ML engineer. You have her full persona timeline in `persona_001_priya.yaml`. For each session, you know exactly which facts she should naturally disclose and what the conversation topic is. She does NOT dump facts — she reveals them incidentally, the way a real person would in conversation with an AI assistant.

**THE ASSISTANT:** A helpful, warm AI assistant. Responds naturally. Asks follow-up questions. Gives advice. Does NOT have access to the persona timeline — it only knows what Priya has told it in previous sessions. It does NOT carry memory between sessions (each session is independent, like a fresh chat).

## Critical Rules for Priya (User Agent)

1. **NEVER state facts directly as evidence.** 
   - BAD: "I should mention that I have ADHD which was diagnosed during my college years."
   - GOOD: "ugh sorry I keep jumping between topics, my brain just does that. Been like this since college when I finally got the ADHD diagnosis and everything clicked"

2. **Facts emerge from context, not from announcements.**
   - BAD: "I am vegetarian and have been for 5 years."
   - GOOD: "can you suggest some restaurants near South Congress? I'm vegetarian btw so steakhouses are out lol, been that way for like 5 years now"

3. **Priya has a consistent voice:**
   - Uses lowercase casually, occasional "lol", "lmao", "tbh", "ngl"
   - Gets excited about technical topics (ML, research)
   - Slightly neurotic about health stuff
   - Vents about practical frustrations (traffic, landlord, scheduling)
   - Warm but direct
   - Types fast, sometimes makes small typos
   - Uses "..." when trailing off in thought

4. **Filler sessions must feel completely natural.** No forced fact disclosure. Just a person chatting with an AI about cooking, movies, code, life. The conversation should be interesting on its own.

5. **Each session is 8-15 turns** (a turn = one user message + one assistant response). Critical sessions can be longer. Filler sessions can be shorter.

6. **Priya asks for real things:** coding help, recipe suggestions, advice on decisions, recommendations. She's not monologuing — she wants the AI to be useful.

## Critical Rules for the Assistant

1. **No memory between sessions.** Each session starts fresh. The assistant does NOT reference previous sessions. If Priya mentions something from a past session, the assistant treats it as new information.

2. **Be genuinely helpful.** Give real coding advice, real recipe suggestions, real recommendations. Don't be a passive listener.

3. **Ask follow-up questions naturally** but don't interrogate.

4. **Match Priya's energy.** When she's excited, be engaged. When she's venting, be supportive. When she needs help, actually help.

## Session Plan

Read the session_plan from persona_001_priya.yaml. For each session:

1. Note the session number, type (critical/filler), facts to disclose (if critical), and topic
2. Note the approximate date (sessions span Jan 2025 — June 2026, roughly 2-3 sessions per month)
3. Generate the full conversation
4. Save to `sessions/session_XX.json`

## Output Format

Each session file should be:

```json
{
  "session_id": 1,
  "persona": "priya_sharma",
  "date": "2025-01-15",
  "type": "critical",
  "topic": "work help",
  "facts_disclosed": ["Works at Google/Gemini memory team", "Lives in Austin"],
  "turns": [
    {
      "role": "user",
      "content": "hey! so I'm working on this retrieval pipeline for our memory system at work and I'm stuck on something..."
    },
    {
      "role": "assistant", 
      "content": "Hey! I'd be happy to help with your retrieval pipeline. What specifically are you stuck on?"
    },
    ...
  ]
}
```

## Execution Steps

1. Read `persona_001_priya.yaml` completely
2. Create `sessions/` directory
3. Generate sessions 1 through 35 sequentially
4. For each session, write the full conversation to `sessions/session_XX.json`
5. After all sessions, create `sessions/manifest.json` listing all sessions with metadata

## Quality Checks (Do These)

After generating all 35 sessions, verify:
- [ ] Every fact in the persona timeline was disclosed in its designated session
- [ ] No fact was disclosed BEFORE its designated session
- [ ] Superseded facts: the OLD fact appears in the OLD session, the NEW fact appears in the NEW session
- [ ] The retraction (dog/landlord) explicitly negates the original in session 16
- [ ] Filler sessions contain NO critical facts (they're pure noise)
- [ ] Priya's voice is consistent across all sessions
- [ ] The assistant never references previous sessions
- [ ] Conversations feel natural — you'd believe a real person wrote the user messages

## Start

Read the persona file, create the directory structure, and begin generating session 1. Work through all 35 sessions sequentially. Take your time on each one — quality over speed.
