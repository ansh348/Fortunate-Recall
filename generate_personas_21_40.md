# ============================================================
# LIFEMEMBENCH — BATCH PERSONA YAML GENERATOR (21-40)
# ============================================================
# Run this in Claude Code:
#   claude -p "$(cat generate_personas_21_40.md)"
# ============================================================

## Your Task

Generate 20 persona YAML files (personas 21-40) for the LifeMemBench benchmark. Each persona must follow the EXACT structure shown in the reference examples below, with all 9 attack vectors covered.

## Output

Create each file at:
```
personas/persona_XXX_firstname.yaml
```

## The 20 Personas to Generate

Use this diversity matrix. Each entry gives you the skeleton — you design the full timeline.

```
21: Thanh Nguyen | 63M | Vietnamese | Shrimp farmer (family operation, 3 ponds) | Primary school only | Mekong Delta, Vietnam
    Voice: Speaks through actions, agricultural metaphors, broken English with Vietnamese syntax, articles dropped, tenses mixed, proud of ponds, weather-obsessed for real economic reasons, early riser, says 'you know' as filler, calls AI 'friend', numbers matter (yield, price per kilo, feed cost)
    Supersessions: switches feed supplier (cheaper Thai brand), sells old boat -> builds new one, eldest son takes over pond 3
    Identity: lost two brothers in the war, wife Lan runs roadside shrimp stall, has diabetes (poorly managed, avoids doctors)

22: Alex Richter | 30M (trans man) | German | Backend software engineer (fintech startup) | BSc Informatik TU Munich | Berlin, Germany
    Voice: Dry, technical, precise, mixes German and English tech jargon, sarcastic warmly, types lowercase, straightforward about being trans without centering it
    Supersessions: startup pivots (payments -> lending), switches Kotlin -> Rust, changes therapist
    Identity: transitioned at 25, has ADHD (methylphenidate), cat named Kernel

23: Mirri Yunupingu | 42F | Yolngu (Indigenous Australian) | Park ranger and cultural liaison | Cert IV Conservation + traditional knowledge | Kakadu, NT, Australia
    Voice: Calm, observational, Yolngu Matha words alongside English, notices things others miss, patient, fierce about land rights
    Supersessions: new park management changes patrol routes, switches 4WD Hilux -> Ranger, stops tour groups -> school education programs
    Identity: elder-in-training (cultural knowledge keeper), chronic kidney disease, raising sister's two kids after sister passed

24: Jerome Baptiste | 35M | Haitian-American | Long-haul truck driver | CDL + some community college | Atlanta, GA (home base)
    Voice: Road philosopher, Haitian Creole phrases, listens to podcasts constantly, lonely but independent, voice-to-text so long rambly messages no punctuation
    Supersessions: switches trucking company (better routes), Freightliner -> Peterbilt, flatbed -> refrigerated loads
    Identity: came from Port-au-Prince at 12 after earthquake, high cholesterol (supposed to take statins), daughter Naomi 9 lives with mother in New Jersey

25: Ingrid Solberg | 48F | Norwegian | Marine biologist (research station) | PhD U of Bergen | Tromsø, Norway
    Voice: Precise scientific language, dry Nordic humor, comfortable with silence, writes like mini-papers, worried about ocean acidification, says 'interesting' when she means 'terrifying'
    Supersessions: research focus cod -> kelp farming, changes ski route (avalanche risk), partner moves in -> partner moves out
    Identity: intentionally childfree, Raynaud's syndrome (circulation in fingers), father was fisherman in Lofoten

26: Dmitri Volkov | 44M | Russian-Israeli | Cybersecurity consultant (independent) | MSc Technion + IDF intelligence | Tel Aviv, Israel
    Voice: Paranoid by profession, dark humor, Russian bluntness + Israeli directness, short declarative sentences, never uses real name online, sentimental about mother's cooking
    Supersessions: drops major client (ethical concerns), Signal -> Briar for comms, moves apartments (security upgrade)
    Identity: immigrated to Israel from Moscow at 15, wife Daria is cellist, has tinnitus (military service)

27: Yolanda 'Yoli' Chambers | 27F | Jamaican-Canadian | Music producer + part-time DJ | Music production diploma Metalworks | Toronto, Canada
    Voice: High energy, musical metaphors everywhere, Jamaican patois + Toronto slang, 'ting' and 'still', night owl, texts in bursts, lots of exclamation marks
    Supersessions: DAW Ableton -> Logic -> back to Ableton, drops manager -> self-manages, changes DJ residency venue
    Identity: moved from Kingston at 16, sickle cell trait (affects energy), grandmother in Jamaica raised her until 10

28: Dariush Farhadi | 51M | Iranian-American | Periodontist (own practice) | DDS UCLA + residency | San Diego, CA
    Voice: Meticulous, perfectionist, clinical precision + warm Persian hospitality, calls everyone 'jan', loves hosting dinners, nostalgic about Tehran, formal emails casual texts
    Supersessions: upgrades laser equipment (LANAP -> new system), switches investment advisor, changes country club
    Identity: family fled Iran 1979 revolution (he was 4), wife Shirin teaches Farsi at community center, has gout (flares, loves red meat anyway)

29: Aroha Tāne | 38F | Māori (Ngāi Tahu) | Midwife (community practice) | Bachelor of Midwifery Otago | Christchurch, New Zealand
    Voice: Warm, grounding, te reo Māori naturally (whānau, kōrero, aroha), tikanga-informed, fiercely protective, calm in emergencies, mentions moana as reset
    Supersessions: hospital births -> home birth focus, Mazda -> Toyota Aqua hybrid, drops netball -> waka ama (outrigger canoe)
    Identity: has moko kauae (chin tattoo), divorced co-parents daughter Maia 11, gestational diabetes history (monitors blood sugar)

30: Mehmet Yılmaz | 56M | Turkish | Traditional barber (owns shop since 1995) | Apprenticed age 14 | Kadıköy, Istanbul, Turkey
    Voice: Storyteller, Turkish proverbs in conversation, old-school but surprisingly progressive, some German (worked in Berlin 5 years), types slowly, uses '...' a lot
    Supersessions: apprentice leaves -> trains new one, switches pomade brand (supplier stopped), landlord raises rent -> renegotiates
    Identity: wife Ayşe is seamstress, lower back pain (30 years standing), lost son in traffic accident 12 years ago

31: Saga Lindqvist | 34F | Swedish | Data analyst at Spotify | MSc Statistics Stockholm U | Stockholm, Sweden
    Voice: Analytical even casually, quantifies everything, dry Swedish humor, feminist, loves spreadsheets, Swedish word order leaks, 'fika' is personality trait
    Supersessions: team restructure (podcast analytics -> music recommendations), switches climbing gym, ends long-term relationship -> starts dating women
    Identity: recently came out bisexual at 33, celiac disease (strict gluten-free), twin brother Erik architect in Gothenburg

32: Kofi Mensah | 40M | Ghanaian | Cocoa farmer + village cooperative chairman | Secondary school + ag extension | Ashanti Region, Ghana
    Voice: Community-minded, proverbs and parables, Akan expressions, talks about rain like markets, dignified, texts concisely (data expensive)
    Supersessions: cooperative switches buying agent, changes fertilizer brand (subsidized), eldest son drops out -> returns to school
    Identity: wife Akosua four children, malaria 2-3x per year (endemic), father was previous cooperative chairman

33: Valentina Herrera | 32F | Colombian | Investigative journalist (national newspaper) | BA Journalism U Javeriana | Bogotá, Colombia
    Voice: Intense, cautious (sources to protect), writes beautifully even casually, fierce and vulnerable, Colombian slang ('parcero','bacano'), texts at 2am, trying to quit smoking
    Supersessions: investigative focus corruption -> environmental crime, changes apartment (security), quits smoking -> relapses -> quits
    Identity: father was journalist (family relocated when she was 8 due to threats), has PTSD from covering armed conflict, partner Lucía is human rights lawyer

34: William 'Billy' Kootook | 49M | Inuit (Inuvialuit) | Wildlife researcher + hunting guide | BSc Env Science U Alberta + traditional knowledge | Tuktoyaktuk, NWT, Canada
    Voice: Quiet authority, fewer words say more, traditional + Western science coexist, dark humor about cold, worried about permafrost thaw, 'out on the land' is default
    Supersessions: research caribou -> muskox, Ski-Doo -> Polaris snowmobile, stops polar bear tours -> cultural education trips
    Identity: wife Annie teaches community school, frostbite damage three fingers (partial feeling lost), grandfather taught him to read ice

35: Pornpan 'Pan' Srisai | 29F | Thai | Street food vendor (khao man gai) + Instagram food creator | Vocational culinary cert | Bangkok, Thailand
    Voice: Bubbly, entrepreneurial, food is love language, Tinglish, emoji natural, hustler with warmth, talks about cart like baby, early morning market runs sacred
    Supersessions: switches rice supplier (quality dropped), charcoal -> gas stove -> back to charcoal (taste), moves cart location (new BTS station)
    Identity: mother taught khao man gai recipe (3 generations), has acid reflux (ironic), younger brother in army (conscription)

36: Marcus 'Marley' Williams | 25M | Black British (Afro-Caribbean) | Physio student + youth football coach | Completing BSc Physio | Bristol, UK
    Voice: Bristol meets Caribbean family culture, UK slang, passionate about accessible physio, code-switches (uni/coaching/home), types fast with abbreviations, 'you know' filler
    Supersessions: coaching U-14s -> U-17s, changes placement hospital, roommate leaves -> studio flat
    Identity: raised by grandmother after mother passed age 7, torn ACL repaired (drives physio interest), grandmother from Barbados

37: Leila Handal | 46F | Palestinian-Chilean | Architect (heritage restoration) | MArch PUC Chile | Santiago, Chile
    Voice: Poetic about buildings, sees history in walls, Spanish/English with Arabic from grandparents, passionate about Palestinian diaspora architecture, references Neruda and Darwish
    Supersessions: firm shifts residential -> heritage restoration, changes yoga studio, stops university teaching -> more fieldwork
    Identity: grandparents fled Bethlehem 1948, husband Diego civil engineer, chronic migraines (fluorescent lighting trigger)

38: Chenoa Begay | 53F | Diné (Navajo) | Family nurse practitioner (IHS clinic) | MSN U of New Mexico | Chinle, Arizona (Navajo Nation)
    Voice: Warm but no-nonsense, walks between Western medicine and traditional healing, Diné Bizaad phrases, frustrated with IHS underfunding, talks about sheep like family, dry humor
    Supersessions: clinic gets new telehealth system, F-150 -> Tacoma (reservation roads), stops weaving competitively -> teaches granddaughter
    Identity: born to Tódích'íi'nii (Bitter Water) clan, type 2 diabetes (manages carefully), husband Raymond silversmith

39: Park Joon-ho | 71M | Korean | Retired math professor / teaches go (baduk) | PhD Math Seoul National U | Busan, South Korea
    Voice: Precise, mathematical metaphors, gentle teacher, old-fashioned Korean formality, go positions as life philosophy, misses late wife, carefully constructed messages, occasionally melancholy
    Supersessions: teaching beginner -> advanced go students, changes morning walk route (construction), physical newspapers -> reluctantly uses tablet
    Identity: wife passed 3 years ago (cancer), glaucoma (progressive, eye drops), son Samsung engineer in Seoul visits monthly

40: Zara Ibrahim | 23F | Sudanese-Emirati (born UAE) | Junior architect (sustainable design firm) | BArch American U of Sharjah | Abu Dhabi, UAE
    Voice: Young professional, passionate about sustainable desert architecture, trilingual Arabic/English/French, caught between Sudanese heritage and Gulf upbringing, 'wallah' and 'yalla', impostor syndrome
    Supersessions: firm switches her residential -> museum project, pilates -> muay thai, best friend moves abroad
    Identity: parents Sudanese but born/raised UAE (identity tension), vitamin D deficiency (ironic in desert), grandmother in Khartoum never met in person
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

### 2. stable_identity (3-4 facts, sessions 1-8)
Each with: fact, session, category, how_disclosed
Categories: IDENTITY_SELF_CONCEPT, HEALTH_WELLBEING, RELATIONAL_BONDS
- how_disclosed must include example dialogue in the persona's EXACT voice

### 3. supersession_events (2-3 events)
Each with: slot, old_fact, old_session, old_how, new_fact, new_session, new_how, category, attack_vector
- old_session and new_session must be 10+ sessions apart
- attack_vector: use 1 (simple supersession) or 4 (multi-version)
- Include how_disclosed for BOTH old and new in the persona's voice

### 4. expiring_logistics (2-3 events)
Each with: fact, session, event_date (YYYY-MM-DD), status (expired/upcoming), how_disclosed, category
- At least one expired, at least one upcoming relative to mid-2026
- Dates must be specific (not vague)
- category: OBLIGATIONS_COMMITMENTS or LOGISTICAL_CONTEXT

### 5. contradictions (1-2 events)
Each with: old_fact, old_session, old_how, new_fact, new_session, new_how, category
- The contradiction must be IMPLICIT — user doesn't realize they're contradicting themselves

### 6. retractions (1 event)
Each with: original_fact, original_session, original_how, retraction, retraction_session, retraction_how, category
- Retraction must be EXPLICIT and DEFINITIVE: "that's dead", "forget about it", "off the table", "not happening", "scrapped"

### 7. ambiguous (1 event)
Each with: existing_fact, existing_session, tentative_fact, tentative_session, tentative_how, resolution, category
- Must be genuinely undecided — not a soft supersession
- Resolution should be "Undecided" or similar

### 8. filler_topics (8-10 topics)
Specific to this persona's life, profession, culture. NOT generic.

### 9. session_plan (35 sessions)
- ~18 critical, ~17 filler
- Format: `{ type: critical/filler, facts: [...], topic: "..." }`
- Critical facts spread across sessions 1-30
- Sessions 31-35 should be mostly filler (wind-down)
- Filler sessions contain ZERO critical facts
- Supersession new-facts appear 10+ sessions after old-facts
- Session dates span Nov 2024 — May 2026 (18 months)

### 10. draft_questions (10-14 questions)
Each with: question, attack_vector, correct, wrong_if_surfaced
- MUST cover all 9 attack vectors:
  - AV1: Superseded preferences (old fact should not surface)
  - AV2: Expired logistics (past events should not appear as upcoming)
  - AV3: Stable identity (should always surface despite age/noise)
  - AV4: Multi-version facts (must return the LATEST version)
  - AV5: Broad aggregation (collect facts across multiple sessions)
  - AV6: Cross-session contradictions (must surface the conflict or the latest version)
  - AV7: Selective forgetting (retracted plans must not surface as active)
  - AV8: Numeric preservation (specific numbers must be preserved through extraction)
  - AV9: Soft supersession (ambiguous/tentative updates must not override established facts)
- At least 1 question per attack vector, 10-14 questions total

## ATTACK VECTOR DESIGN GUIDELINES

**AV1 (Superseded preference):** Design a preference that cleanly changes. The question asks for the CURRENT value. Surfacing the old value is a failure.

**AV2 (Expired logistics):** Include specific dates that will have passed. The question asks about current/upcoming events. Surfacing a past appointment as upcoming is a failure.

**AV3 (Stable identity):** Core traits mentioned once early on. The question asks about them much later. Not finding them (buried under noise) is a failure.

**AV4 (Multi-version fact):** A fact that changes through multiple versions (e.g., job title changes twice, or a preference changes and changes back). The question asks for the latest.

**AV5 (Broad aggregation):** A question that requires collecting facts from multiple sessions. E.g., "What are Thanh's family members?" requires info from sessions 2, 5, 8.

**AV6 (Cross-session contradiction):** The user says one thing early, contradicts it later without awareness. The system should surface the contradiction or the latest version.

**AV7 (Selective forgetting):** A retracted plan. The question asks about current plans. Surfacing the retracted plan as active is a failure.

**AV8 (Numeric preservation):** Facts with specific numbers ($, kg, %, counts). The question asks for the number. Returning the fact without the number is a failure.

**AV9 (Soft supersession):** A tentative update that doesn't fully replace. E.g., "thinking about moving to London" shouldn't override "lives in Manchester."

## VOICE QUALITY CHECKLIST

Before finalizing each persona, verify the how_disclosed dialogue:

1. [ ] Matches the persona's sentence length pattern
2. [ ] Uses the persona's specific verbal tics and slang
3. [ ] Appropriate formality level
4. [ ] Appropriate punctuation/capitalization style
5. [ ] Cultural references are authentic
6. [ ] A 68-year-old beekeeper sounds NOTHING like a 24-year-old electrician
7. [ ] Numeric facts include SPECIFIC numbers (not "some money" but "$300-400")
8. [ ] Retraction uses EXPLICIT kill language
9. [ ] Contradiction is IMPLICIT (no self-awareness)

## REFERENCE FORMAT

Look at the completed persona files in the `personas/` directory for the exact YAML structure, especially:
- persona_003_elena.yaml
- persona_004_david.yaml
- persona_020_aisha.yaml

Match that format exactly.

## GO

Generate all 20 persona YAML files (021-040). Work through them one at a time. Each persona should feel like a real person with a real life — not a template with names swapped.
