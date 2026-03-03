# ============================================================
# LIFEMEMBENCH — BATCH PERSONA YAML GENERATOR
# ============================================================
# Run this in Claude Code:
#   claude -p "$(cat generate_all_personas.md)"
# ============================================================

## Your Task

Generate 14 persona YAML files (personas 5-18) for the LifeMemBench benchmark. Each persona must follow the EXACT structure shown in the reference examples below, with all 7 attack vectors covered.

## Output

Create each file at:
```
personas/persona_XXX_firstname.yaml
```

## The 14 Personas to Generate

Use this diversity matrix. Each entry gives you the skeleton — you design the full timeline.

```
5:  Amara Okafor | 37F | Nigerian-British | Barrister (criminal defense) | LLB | London, UK
    Voice: Sharp, precise, no wasted words, British English, occasionally intense about justice, formal with strangers but warm underneath
    Supersessions: changes chambers, drops running -> boxing, switches iPhone -> Android
    Identity: parents Igbo from Lagos, has endometriosis, brother Chidi is a doctor

6:  Jake Brennan | 24M | White American (Irish) | Apprentice electrician | Trade school | Boston, MA
    Voice: Heavy Boston accent in text ('wicked','kid'), super casual, drops g's, no punctuation sometimes, young energy
    Supersessions: quits vaping -> starts again -> quits again, girlfriend breaks up -> dating someone new, switches from Bud Light to craft beer
    Identity: dad is union electrician, has dyslexia, plays rec hockey

7:  Fatima Al-Rashidi | 45F | Saudi Arabian | Interior designer (own firm) | BFA London | Riyadh, Saudi Arabia
    Voice: Elegant, design-focused vocabulary, bilingual English/Arabic hints, references aesthetics constantly, ambitious
    Supersessions: main fabric supplier change, residential -> commercial projects, office moves from Al Olaya to KAFD
    Identity: divorced raising two sons alone, chronic back pain from car accident, father was architect

8:  Tom Whitfield | 68M | White British | Retired civil engineer / part-time beekeeper | BEng | Cotswolds, UK
    Voice: Gentle, methodical, British understatement, talks about bees like people, 'rather' and 'quite', proper punctuation
    Supersessions: switches honey jar supplier, sells Land Rover -> electric car, stops pub quiz -> joins walking group
    Identity: wife Margaret died 2 years ago, atrial fibrillation (blood thinners), daughter in New Zealand

9:  Kenji Nakamura | 31M | Japanese | Freelance graphic designer | BFA Tama Art Uni | Osaka, Japan
    Voice: Creative, visual thinker, mixes English design jargon with casual speech, night owl, anxious about income
    Supersessions: Figma -> Framer, drops major client -> picks up better one, shared studio -> home office
    Identity: chronic insomnia, mother runs izakaya, red-green colorblind

10: Rosa Gutierrez | 55F | Peruvian-American | Family court judge | JD Georgetown | Miami, FL
    Voice: Authoritative but compassionate, legal precision even casually, matriarch energy, bilingual, proud immigrant story
    Supersessions: yoga -> swimming, BMW -> Lexus, changes housekeeper
    Identity: immigrated from Lima at 12, husband Carlos retired firefighter, high blood pressure

11: Callum Fraser | 29M | Scottish | Farm veterinarian | BVetMed Edinburgh | Inverness, Scotland
    Voice: Deadpan Scottish humor, 'aye' and 'wee', pragmatic, weather-obsessed, types with shortcuts
    Supersessions: NHS rural -> private practice, old Defender -> Hilux, rugby -> fell running
    Identity: grew up on sheep farm, Crohn's disease, sister Fiona midwife in Glasgow

12: Diane Holloway | 61F | White American | Hospice chaplain | MDiv Yale Divinity | Asheville, NC
    Voice: Deeply reflective, comfortable with heavy topics, literary references, 'I wonder' a lot, never rushed
    Supersessions: switches hospice orgs, gardening -> watercolor (back problems), Episcopal -> UCC denomination
    Identity: lost son Michael to overdose 8 years ago, osteoarthritis both knees, husband Greg woodworker

13: Raj Malhotra | 39M | Indian (Punjabi) | Plumber (own business) | ITI diploma India, moved UK at 22 | Birmingham, UK
    Voice: Cheerful, hardworking, proud, 'innit' and 'sorted', talks about kids constantly, practical
    Supersessions: Ford -> Vauxhall vans, Sky Sports -> DAZN, wife Simran teacher -> school admin
    Identity: wife Simran + kids Arjun 8 Preet 5, asthma (inhaler), parents in Jalandhar visits yearly

14: Nadia Petrov | 33F | Russian-Canadian | Sous chef (fine dining) | Culinary diploma | Montreal, Canada
    Voice: Intense about food, perfectionist, swears when stressed, French/English/Russian mix, passionate, night owl
    Supersessions: restaurant changes head chef, switches apartment, ballet -> CrossFit
    Identity: immigrated from St. Petersburg at 14, lactose intolerant, grandmother Babushka taught her to cook

15: Samuel Osei | 47M | Ghanaian-American | Neurosurgeon | MD Johns Hopkins | Minneapolis, MN
    Voice: Precise, calm, measured paragraphs, occasional Ghanaian expressions, deeply religious Methodist, formal but kind
    Supersessions: open surgery -> minimally invasive focus, YMCA -> climbing gym, condo -> house
    Identity: immigrated from Accra at 18 on scholarship, wife Abena pharmacist, sleep apnea (CPAP)

16: Lily Chen | 22F | Chinese-Australian | Uni student (env science) + barista | Completing BSc | Melbourne, Australia
    Voice: Gen Z, abbreviations, 'literally','genuinely','lowkey', Australian 'reckon','arvo', climate-anxious
    Supersessions: changes thesis topic, breaks up with girlfriend -> dates someone new, switches oat milk brand
    Identity: parents run Chinese restaurant, eczema (stress flares), grandmother speaks only Cantonese

17: Omar Hassan | 36M | Sudanese-American | Uber driver + studying real estate license | Some college | Houston, TX
    Voice: Hustler mentality, 'bro', pragmatic about money, tired but driven, texts between rides
    Supersessions: Uber -> Lyft -> back to Uber, moves apartments (cheaper), changes RE study program
    Identity: Darfur refugee arrived US at 19, son Ismail 7 co-parents, sends money to mother in Khartoum

18: Bruna Costa | 41F | Brazilian | Physical therapist (sports clinic) | DPT from USP | Dubai, UAE
    Voice: High energy, Portuguese expressions when excited, warm, fitness-focused, misses Brazil
    Supersessions: clinic changes ownership, running -> cycling (knee), changes apartment (closer to clinic)
    Identity: moved Sao Paulo to Dubai 5 years ago, hypothyroidism (levothyroxine), mother is best friend
```

## CRITICAL STRUCTURE RULES

Every persona YAML MUST have ALL of these sections, no exceptions:

### 1. persona (header)
```yaml
persona:
  name: "Full Name"
  age: XX
  gender: male/female
  location_current: "City, Country"
  occupation_current: "Description"
  personality: >
    3-4 sentences with verbal tics, typing style,
    emotional patterns, sentence length tendencies
```

### 2. stable_identity (3-4 facts, sessions 1-7)
Each with: fact, session, category, how_disclosed
Categories: IDENTITY_SELF_CONCEPT, HEALTH_WELLBEING, RELATIONAL_BONDS

### 3. supersession_events (2-3 events)
Each with: slot, old_fact, old_session, old_how, new_fact, new_session, new_how, category, attack_vector
- old_session and new_session must be 10+ sessions apart
- Include how_disclosed for BOTH old and new

### 4. expiring_logistics (2-3 events)
Each with: fact, session, event_date (YYYY-MM-DD), status (expired/upcoming), how_disclosed
- Mix of expired and upcoming relative to mid-2026

### 5. contradictions (1-2 events)
Each with: old_fact, old_session, old_how, new_fact, new_session, new_how
- The contradiction must be IMPLICIT — user doesn't realize they're contradicting themselves

### 6. retractions (1 event)
Each with: original_fact, original_session, original_how, retraction, retraction_session, retraction_how
- Retraction must be EXPLICIT: "that's dead", "forget about it", "off the table"

### 7. ambiguous (1 event)
Each with: existing_fact, existing_session, tentative_fact, tentative_session, tentative_how, resolution
- Must be genuinely undecided — not a soft supersession

### 8. filler_topics (8-10 topics)
Specific to this persona's life. NOT generic.

### 9. session_plan (35 sessions)
- ~18 critical, ~17 filler
- Format: { type: critical/filler, facts: [...], topic: "..." }
- Critical facts spread across sessions 1-30
- Filler sessions contain ZERO critical facts
- Supersession new-facts appear 10+ sessions after old-facts

### 10. draft_questions (10-12 questions)
Each with: question, attack_vector, correct, wrong_if_surfaced
- MUST cover all 7 attack vectors
- At least 1 question per attack vector

## REFERENCE EXAMPLES

Look at the completed files in the `personas/` directory:
- persona_003_elena.yaml
- persona_004_david.yaml

Also reference the two completed personas in the parent directories:
- ../priya_1/persona_001_priya.yaml
- ../marcus_2/persona_002_marcus.yaml

These show the exact format and level of detail expected.

## QUALITY RULES

1. **how_disclosed fields are CRITICAL.** They must describe a natural conversational context, not just the fact. "Comes up when discussing X" with example dialogue.

2. **Contradictions must be implicit.** The user never says "I changed my mind about X." They just say something that conflicts with earlier statements.

3. **Retractions must be explicit.** The user clearly kills a plan: "that's off the table", "forget about the X thing", "not happening."

4. **Voice must be distinct.** A 68-year-old British beekeeper does NOT sound like a 24-year-old Boston electrician. Verbal tics, sentence length, formality, slang — all different.

5. **Filler topics must be persona-specific.** A chef asks about ingredients. A lawyer asks about client communication. A student asks about study techniques. No generic filler.

6. **Session dates should span 18 months** from roughly Nov 2024 to May 2026.

7. **Every persona needs at least one question for each attack vector (1-7)** plus 3-5 additional questions. Total: 10-12 per persona.

## GO

Generate all 14 persona YAML files. Work through them one at a time. Quality over speed — each persona should feel like a real person with a real life.
