# LifeMemBench Extraction Gap Audit Report

*Generated: 2026-03-03 17:31*

## Executive Summary

- **Total gaps:** 51 out of 112 questions (av_pass=False in 'full' config)
- **E1 (Benchmark design):** 2
- **E2 (Extraction miss):** 4 (8%)
- **E3 (Vague/routing gap):** 34
- **E4 (Wrong category):** 6
- **E5 (Composite/multi-session):** 4
- **RANKING (not extraction):** 1

**Key finding:** 78% of gaps are E3+E4 (retrieval/routing). The correct facts ARE extracted into edges but not retrieved for the right questions. Only 8% are true extraction misses (E2). The bottleneck is retrieval and category routing, not the extraction model.

## Full Gap Table

| # | question_id | persona | attack_vector | expected_fact | classification | conversation_file | nearest_edge | reason |
|---|------------|---------|---------------|---------------|---------------|-------------------|-------------|--------|
| 1 | priya_q01 | priya | AV1_superseded_preference | Started eating fish -- pescatarian (was vegetarian) | **E3** | session_22.json | user started eating fish again a few weeks ago | Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap |
| 2 | priya_q03 | priya | AV3_stable_identity | Has ADHD, diagnosed in college | **E3** | session_02.json | user received ADHD diagnosis since college | Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap |
| 3 | priya_q05 | priya | AV7_selective_forgetting | Wanted dog but landlord said no -- retraction | **E4** | session_12.json | user has a rental relationship with landlord | Answer edge exists but category {'RELATIONAL_BONDS', 'FINANCIAL_MATERIAL'} doesn |
| 4 | priya_q07 | priya | AV1_superseded_preference | Switched to rock climbing, quit hot yoga | **E2** | session_25.json | user got into climbing three weeks ago | Fact in session 25 -- only topic-level edges exist (no correct-answer edge extra |
| 5 | priya_q09 | priya | AV3_stable_identity | Chronic migraines triggered by stress/screens; needs omega-3 | **E3** | session_04.json | user has chronic migraines that occur pretty regularly | Answer edge exists (9 enriched) but not retrieved -- routing/retrieval gap |
| 6 | marcus_q02 | marcus | AV1_superseded_preference | Bought 2024 Ram 1500, traded in Ford F-150 | **E3** | session_24.json | user bought a 2024 Ram 1500 Big Horn trim in granite crystal | Answer edge exists (6 enriched) but not retrieved -- routing/retrieval gap |
| 7 | marcus_q03 | marcus | AV7_selective_forgetting | Scrapped Germantown second location -- too much financial ri | **E3** | session_14.json | user has been looking at a spot in Germantown for a second s | Answer edge exists (7 enriched) but not retrieved -- routing/retrieval gap |
| 8 | marcus_q06 | marcus | AV1_superseded_preference | Stopped poker, started fishing on weekends | **E3** | session_23.json | user has been going to Sardis Lake mostly for fishing, which | Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap |
| 9 | marcus_q13 | marcus | AV8_numeric_preservation | Hired Carlos as 4th employee -- business growing | **E4** | session_26.json | user just brought on a new guy named Carlos because business | Answer edge exists but category {'IDENTITY_SELF_CONCEPT', 'RELATIONAL_BONDS', 'O |
| 10 | elena_q01 | elena | AV4_multi_version | Transferred from Lurie to Rush University Medical Center | **E3** | session_21.json | user is employed at Rush University Medical Center | Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap |
| 11 | elena_q02 | elena | AV1_superseded_preference | Quit keto, switched to Mediterranean diet | **E3** | session_22.json | user's doctor told user to switch to Mediterranean diet | Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap |
| 12 | elena_q03 | elena | AV7_selective_forgetting | Dropped NP plan -- can't afford it with student loans | **E3** | session_09.json | user is looking at DePaul's DNP program with a pediatric NP  | Answer edge exists (9 enriched) but not retrieved -- routing/retrieval gap |
| 13 | elena_q04 | elena | AV3_stable_identity | Generalized anxiety disorder, takes sertraline | **E3** | session_03.json | user is on sertraline for anxiety and it usually helps but l | Answer edge exists (3 enriched) but not retrieved -- routing/retrieval gap |
| 14 | elena_q08 | elena | AV3_stable_identity | Parents from Guadalajara, first-gen college grad, sister Sof | **E5** | session_02.json | Sofia is user's sister | Composite fact across 3 sessions -- partial answer edges exist but incomplete |
| 15 | elena_q12 | elena | AV5_broad_query | Knitting, Mediterranean diet, anxiety management for stress  | **E4** | session_22.json | user plans to watch Very Pink Knits on YouTube | Answer edge exists but category {'PREFERENCES_HABITS'} doesn't match relevant {' |
| 16 | elena_q13 | elena | AV8_numeric_preservation | Student loans -- question says $40,000, conversation may say | **E1** | session_17.json | user has about 42k left on student loans | Edge has 'about 42k' but question expects exact '$40,000' -- benchmark mismatch |
| 17 | david_q01 | david | AV4_multi_version | Switching from AP Euro to AP US History | **E3** | session_23.json | department head pulled user aside last week to ask if user w | Answer edge exists (5 enriched) but not retrieved -- routing/retrieval gap |
| 18 | david_q03 | david | AV4_multi_version | Bought Subaru Outback, replacing old Camry | **E3** | session_24.json | user went to the dealership on Saturday and drove home in a  | Answer edge exists (10 enriched) but not retrieved -- routing/retrieval gap |
| 19 | david_q04 | david | AV7_selective_forgetting | Abandoned book about teaching through primary sources | **E2** | session_09.json | user is about a fifteen-minute walk from Powell's Books sinc | Fact in session 9 -- only topic-level edges exist (no correct-answer edge extrac |
| 20 | david_q05 | david | AV3_stable_identity | Mild hearing loss in left ear, wears hearing aid | **E3** | session_05.json | user wears a hearing aid due to mild hearing loss in left ea | Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap |
| 21 | david_q11 | david | AV2_expired_logistics | Parent-teacher conferences March 12 (expired) | **E3** | session_10.json | David's principal is supportive and has offered to sit in on | Answer edge exists (4 enriched) but not retrieved -- routing/retrieval gap |
| 22 | amara_q01 | amara | AV4_multi_version | Moved from 4 Brick Court to 7 Bedford Row as senior tenant | **E3** | session_19.json | Amara Okafor joins 7 Bedford Row as a senior tenant | Answer edge exists (8 enriched) but not retrieved -- routing/retrieval gap |
| 23 | amara_q02 | amara | AV1_superseded_preference | Switched from iPhone 14 Pro to Samsung Galaxy S25 | **E4** | session_22.json | user has just set up a new Samsung Galaxy S25 | Answer edge exists but category {'FINANCIAL_MATERIAL'} doesn't match relevant {' |
| 24 | amara_q03 | amara | AV1_superseded_preference | Dropped running, switched to boxing in Bethnal Green | **E3** | session_25.json | user has started boxing at a gym in Bethnal Green with class | Answer edge exists (7 enriched) but not retrieved -- routing/retrieval gap |
| 25 | amara_q06 | amara | AV7_selective_forgetting | Dropped UCL human rights LLM -- not worth time/money | **E2** | session_11.json | user has been looking at doing an LLM in human rights law at | Fact in session 11 -- only topic-level edges exist (no correct-answer edge extra |
| 26 | amara_q07 | amara | AV6_cross_session | Said no alcohol 2 years, but had wine at chambers dinner | **E4** | session_09.json | Charles hosted the chambers dinner at his house in Hampstead | Answer edge exists but category {'LOGISTICAL_CONTEXT'} doesn't match relevant {' |
| 27 | amara_q08 | amara | AV9_soft_supersession | Lives in Hackney, considering move to Islington | **E3** | session_01.json | user has an early train from Hackney Central into the Temple | Answer edge exists (10 enriched) but not retrieved -- routing/retrieval gap |
| 28 | amara_q11 | amara | AV5_broad_query | Boxing, Nigerian cooking, podcasts, reading about justice | **E3** | session_25.json | user has started boxing at a gym in Bethnal Green with class | Answer edge exists (4 enriched) but not retrieved -- routing/retrieval gap |
| 29 | amara_q12 | amara | AV3_stable_identity | LLB from King's College London | **E3** | session_02.json | user did LLB at King's | Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap |
| 30 | amara_q14 | amara | AV8_numeric_preservation | UCL LLM would have cost GBP 18,000 | **E2** | session_11.json (expected) | user plans to apply for wasted costs against CPS under secti | Entity node references fact but no answer edge extracted |
| 31 | jake_q02 | jake | AV4_multi_version | Megan broke up, now dating Kayla (nurse at Mass General) | **E3** | session_24.json | Kayla is a nurse at Mass General | Answer edge exists (3 enriched) but not retrieved -- routing/retrieval gap |
| 32 | jake_q03 | jake | AV3_stable_identity | Mom waitresses at Eire Pub in Dorchester, 20 years | **E3** | session_03.json | Megan lives in an apartment in Dorchester | Answer edge exists (8 enriched) but not retrieved -- routing/retrieval gap |
| 33 | jake_q05 | jake | AV7_selective_forgetting | EV charger side gig dead -- dad said not ready, insurance to | **E3** | session_11.json | user is thinking worst case he could do EV charger installat | Answer edge exists (7 enriched) but not retrieved -- routing/retrieval gap |
| 34 | jake_q07 | jake | AV1_superseded_preference | Got into craft beer -- favorites from Trillium Brewing | **E3** | session_26.json | user was dragged to Trillium by Marcus like a month ago. | Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap |
| 35 | jake_q09 | jake | AV3_stable_identity | Irish-American, grandparents County Cork, dad Brennan & Sons | **E5** | session_01.json | user's mom has been working at the Eire Pub in Dorchester fo | Composite fact across 3 sessions -- partial answer edges exist but incomplete |
| 36 | jake_q10 | jake | AV5_broad_query | Rec hockey Wednesdays, gaming PC, craft beer, sports with th | **E5** | session_04.json | user plays in a rec league every Wednesday night at Bajko Ri | Composite fact across 3 sessions -- partial answer edges exist but incomplete |
| 37 | tom_q02 | tom | AV1_superseded_preference | Sold Land Rover Defender, bought Hyundai Ioniq 5 | **E3** | session_24.json | user has just bought a Hyundai Ioniq 5 | Answer edge exists (3 enriched) but not retrieved -- routing/retrieval gap |
| 38 | tom_q03 | tom | AV1_superseded_preference | Stopped pub quiz at The Fox, joined walking group | **E3** | session_23.json | user has recently joined a walking group | Answer edge exists (11 enriched) but not retrieved -- routing/retrieval gap |
| 39 | tom_q04 | tom | AV3_stable_identity | Atrial fibrillation, takes warfarin | **E3** | session_04.json | user has atrial fibrillation | Answer edge exists (3 enriched) but not retrieved -- routing/retrieval gap |
| 40 | tom_q06 | tom | AV7_selective_forgetting | Barn conversion scrapped -- planning permission denied, too  | **E3** | session_11.json | user needs to check with Cotswold District Council for plann | Answer edge exists (4 enriched) but not retrieved -- routing/retrieval gap |
| 41 | tom_q07 | tom | AV6_cross_session | Said no social media, but posted bee photo on Instagram (200 | **E3** | session_09.json | user has no interest in Facebook or Instagram whatsoever | Answer edge exists (9 enriched) but not retrieved -- routing/retrieval gap |
| 42 | tom_q11 | tom | AV3_stable_identity | BEng civil engineering, worked at Arup 35 years | **E3** | session_05.json | user spent thirty-five years at Arup in civil engineering | Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap |
| 43 | omar_q01 | omar | AV4_multi_version | Uber -> Lyft -> back to Uber (triple version) | **RANKING** | session_01.json | user has been doing uber full time for 3 years | Edge exists and is answerable -- stale Lyft edge outranks correct Uber edge |
| 44 | omar_q02 | omar | AV1_superseded_preference | Moved from Alief 1BR to Gulfton studio, saves $200/mo | **E3** | session_22.json | user just moved to Gulfton | Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap |
| 45 | omar_q05 | omar | AV7_selective_forgetting | Camry plan dead -- car sold before financing secured | **E3** | session_10.json | the whole Camry plan is dead for now | Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap |
| 46 | omar_q06 | omar | AV6_cross_session | Said never spends on himself but bought PS5 for stress | **E3** | session_25.json | user got PS5 off Facebook Marketplace with 2 controllers and | Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap |
| 47 | omar_q07 | omar | AV9_soft_supersession | Staying in Houston but cousin says Dallas RE market is bette | **E4** | session_04.json | Khalid lives in Dallas | Answer edge exists but category {'IDENTITY_SELF_CONCEPT'} doesn't match relevant |
| 48 | omar_q08 | omar | AV5_broad_query | Uber driver + starting at Century 21 in April 2026 | **E5** | session_26.json | user starts at Century 21 on April 1st | Composite fact across 2 sessions -- partial answer edges exist but incomplete |
| 49 | omar_q11 | omar | AV3_stable_identity | No explicit health conditions -- tired/stressed from overwor | **E1** | -- | -- | Correct answer is 'None explicitly mentioned' -- no extractable fact |
| 50 | omar_q12 | omar | AV3_stable_identity | Real estate career -- passed exam, joining Century 21 | **E3** | session_09.json | user is studying for real estate license between rides | Answer edge exists (21 enriched) but not retrieved -- routing/retrieval gap |
| 51 | omar_q14 | omar | AV8_numeric_preservation | Sends $300-400/month to mother in Khartoum | **E3** | session_03.json | user sends user's mom 300 400 every month depending on what  | Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap |

## Summary Counts

| Category | Count | % |
|----------|-------|---|
| E1 | 2 | 3.9% |
| E2 | 4 | 7.8% |
| E3 | 34 | 66.7% |
| E4 | 6 | 11.8% |
| E5 | 4 | 7.8% |
| RANKING | 1 | 2.0% |
| **Total** | **51** | **100%** |

## Per-Persona Breakdown

| Persona | Total | E1 | E2 | E3 | E4 | E5 | RANKING |
|---------|-------|----|----|----|----|----|----|
| amara | 9 | 0 | 2 | 5 | 2 | 0 | 0 |
| david | 5 | 0 | 1 | 4 | 0 | 0 | 0 |
| elena | 7 | 1 | 0 | 4 | 1 | 1 | 0 |
| jake | 6 | 0 | 0 | 4 | 0 | 2 | 0 |
| marcus | 4 | 0 | 0 | 3 | 1 | 0 | 0 |
| omar | 9 | 1 | 0 | 5 | 1 | 1 | 1 |
| priya | 5 | 0 | 1 | 3 | 1 | 0 | 0 |
| tom | 6 | 0 | 0 | 6 | 0 | 0 | 0 |

## Per-Attack-Vector Breakdown

| Attack Vector | Total | E1 | E2 | E3 | E4 | E5 | RANKING |
|--------------|-------|----|----|----|----|----|----|
| AV1_superseded_preference | 11 | 0 | 1 | 9 | 1 | 0 | 0 |
| AV2_expired_logistics | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| AV3_stable_identity | 12 | 1 | 0 | 9 | 0 | 2 | 0 |
| AV4_multi_version_fact | 6 | 0 | 0 | 5 | 0 | 0 | 1 |
| AV5_broad_query | 4 | 0 | 0 | 1 | 1 | 2 | 0 |
| AV6_cross_session_contradiction | 3 | 0 | 0 | 2 | 1 | 0 | 0 |
| AV7_selective_forgetting | 8 | 0 | 2 | 5 | 1 | 0 | 0 |
| AV8_numeric_preservation | 4 | 1 | 1 | 1 | 1 | 0 | 0 |
| AV9_soft_supersession | 2 | 0 | 0 | 1 | 1 | 0 | 0 |

## Detailed Analysis

### priya_q01 — **E3**

- **Question:** What dietary restrictions does Priya have?
- **Correct answer:** Pescatarian (was vegetarian, now eats fish)
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Started eating fish -- pescatarian (was vegetarian)
- **Classification:** E3
- **Reason:** Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 22 — `and I haven't eaten. my doctor has been on me about this actually. oh and I actually started eating fish again a few wee...`
- **Matching edges:** 2
  - `[89944e6a-fb2]` (eating fish again) "user started eating fish again a few weeks ago"
  - `[678a7f58-bb5]` (pescatarian) "user is pescatarian in context of travel to Japan"
- **Matching entity nodes:** 30

### priya_q03 — **E3**

- **Question:** What should I know about Priya's learning or work style?
- **Correct answer:** She has ADHD (diagnosed in college)
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Has ADHD, diagnosed in college
- **Classification:** E3
- **Reason:** Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 2 — `w I'm describing this. my brain just does that. been like this since college when I finally got the ADHD diagnosis, I've...`
- **Matching edges:** 2
  - `[2d4598c4-760]` (adhd) "user received ADHD diagnosis since college"
  - `[1af18e0c-afe]` (adhd) "user has ADHD"
- **Matching entity nodes:** 10

### priya_q05 — **E4**

- **Question:** Is Priya planning to get a pet?
- **Correct answer:** No — she wanted a dog but landlord said no
- **Attack vector:** AV7_selective_forgetting
- **Expected fact:** Wanted dog but landlord said no -- retraction
- **Classification:** E4
- **Reason:** Answer edge exists but category {'RELATIONAL_BONDS', 'FINANCIAL_MATERIAL'} doesn't match relevant {'PREFERENCES_HABITS', 'LOGISTICAL_CONTEXT'}
- **Conversation evidence:** session 12 — `out something is not impulsive, that's just a slow decision. What's the thing? I've been looking at golden retriever pup...`
- **Matching edges:** 2
  - `[b48aebd3-ce5]` (landlord) "user has a rental relationship with landlord"
  - `[288b26df-816]` (landlord) "lease has clause allowing landlord to deny pets at discretion"
- **Matching entity nodes:** 18

### priya_q07 — **E2**

- **Question:** What exercise does Priya do?
- **Correct answer:** Rock climbing
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Switched to rock climbing, quit hot yoga
- **Classification:** E2
- **Reason:** Fact in session 25 -- only topic-level edges exist (no correct-answer edge extracted)
- **Conversation evidence:** session 25 — `I completely dropped yoga btw. got into climbing and it's so much better for my brain — the problem solving aspect is li...`
- **Matching entity nodes:** 17

### priya_q09 — **E3**

- **Question:** What health conditions should Priya's meal plan account for?
- **Correct answer:** Chronic migraines (triggered by stress/screens). Also was told to get more omega-3s.
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Chronic migraines triggered by stress/screens; needs omega-3
- **Classification:** E3
- **Reason:** Answer edge exists (9 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 4 — `p thinking through a technical design decision? fair warning I might be a bit scattered, I've got a migraine today and s...`
- **Matching edges:** 9
  - `[1ef0982c-feb]` (migraine) "user has chronic migraines that occur pretty regularly"
  - `[6fbf0ba4-e64]` (migraine) "user's migraines are triggered by screens and stress"
  - `[d4d29ab7-acf]` (migraine) "user has been on a preventive medication for migraines for about a year which has reduced frequency "
- **Matching entity nodes:** 17

### marcus_q02 — **E3**

- **Question:** What vehicle does Marcus drive?
- **Correct answer:** 2024 Ram 1500
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Bought 2024 Ram 1500, traded in Ford F-150
- **Classification:** E3
- **Reason:** Answer edge exists (6 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 24 — `happen at the shop, or is this a personal win? finally pulled the trigger on a new truck. got a 24 Ram 1500. traded in t...`
- **Matching edges:** 6
  - `[5595fa76-301]` (ram 1500, 2024 ram) "user bought a 2024 Ram 1500 Big Horn trim in granite crystal color with 5.7 Hemi"
  - `[b712e7f5-3da]` (ram 1500) "24 Ram 1500 has Big Horn trim"
  - `[8d5d755b-208]` (ram 1500) "24 Ram 1500 is granite crystal the dark gray"
- **Matching entity nodes:** 21

### marcus_q03 — **E3**

- **Question:** Is Marcus planning to open a second shop location?
- **Correct answer:** No — he considered Germantown but scrapped it (too much financial risk)
- **Attack vector:** AV7_selective_forgetting
- **Expected fact:** Scrapped Germantown second location -- too much financial risk
- **Classification:** E3
- **Reason:** Answer edge exists (7 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 14 — `pots? And do you have a specific area in mind, or are you still scouting? been looking at a spot in Germantown for a sec...`
- **Matching edges:** 7
  - `[aad7d3d6-622]` (germantown) "user has been looking at a spot in Germantown for a second shop"
  - `[530324b8-d1e]` (germantown) "second shop is a spot in Germantown"
  - `[79118edf-6db]` (germantown) "second shop has rent of 3200 a month in Germantown"
- **Matching entity nodes:** 5

### marcus_q06 — **E3**

- **Question:** What does Marcus do for fun?
- **Correct answer:** Fishing on weekends
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Stopped poker, started fishing on weekends
- **Classification:** E3
- **Reason:** Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 23 — `ave you stepped back from it? I actually quit the poker game. been getting out on the lake instead. fishing is better fo...`
- **Matching edges:** 1
  - `[30e1f910-3a9]` (fishing) "user has been going to Sardis Lake mostly for fishing, which is about an hour south"
- **Matching entity nodes:** 4

### marcus_q13 — **E4**

- **Question:** How many employees does Marcus currently have at the shop?
- **Correct answer:** 4 (recently hired Carlos as the fourth employee)
- **Attack vector:** AV8_numeric_preservation
- **Expected fact:** Hired Carlos as 4th employee -- business growing
- **Classification:** E4
- **Reason:** Answer edge exists but category {'IDENTITY_SELF_CONCEPT', 'RELATIONAL_BONDS', 'OBLIGATIONS'} doesn't match relevant {'PROJECTS_ENDEAVORS', 'FINANCIAL_MATERIAL'}
- **Conversation evidence:** session 26 — `the work independently. What's going on — did you bring someone on? yeah just brought on a new guy, Carlos. business has...`
- **Matching edges:** 4
  - `[b5672ceb-f7c]` (carlos) "user just brought on a new guy named Carlos because business has been good enough that user couldn't"
  - `[435841c4-438]` (carlos) "Carlos worked at a Firestone for a couple years"
  - `[892e5b82-afd]` (carlos) "user employs Carlos as one of four guys who can handle the volume without everybody burning out"
- **Matching entity nodes:** 11

### elena_q01 — **E3**

- **Question:** Where does Elena work?
- **Correct answer:** Rush University Medical Center
- **Attack vector:** AV4_multi_version_fact
- **Expected fact:** Transferred from Lurie to Rush University Medical Center
- **Classification:** E3
- **Reason:** Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 21 — `I have some news and I'm like still processing it honestly I'm all ears! What's going on? so I'm at Rush now. like it ac...`
- **Matching edges:** 1
  - `[a894530c-7de]` (rush university) "user is employed at Rush University Medical Center"
- **Matching entity nodes:** 19

### elena_q02 — **E3**

- **Question:** What diet is Elena following?
- **Correct answer:** Mediterranean diet
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Quit keto, switched to Mediterranean diet
- **Classification:** E3
- **Reason:** Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 22 — `ok real talk I need to change how I'm eating, do you know much about the Mediterranean diet I do! It's one of the most w...`
- **Matching edges:** 2
  - `[768bb866-5a3]` (mediterranean) "user's doctor told user to switch to Mediterranean diet"
  - `[2889f6ba-f48]` (mediterranean) "user made the switch to Mediterranean diet about two weeks ago"
- **Matching entity nodes:** 5

### elena_q03 — **E3**

- **Question:** Is Elena planning to get a nurse practitioner degree?
- **Correct answer:** No — she dropped that plan due to student loan debt
- **Attack vector:** AV7_selective_forgetting
- **Expected fact:** Dropped NP plan -- can't afford it with student loans
- **Classification:** E3
- **Reason:** Answer edge exists (9 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 9 — `I'm an RN, I have my BSN, and I've been working pediatric nursing for about 4 years. I'm looking at nurse practitioner p...`
- **Matching edges:** 9
  - `[a309a85e-c67]` (np) "user is looking at DePaul's DNP program with a pediatric NP track"
  - `[d4327e68-d30]` (nurse practitioner) "user is looking at nurse practitioner programs long-term"
  - `[24e6a732-9a9]` (np) "user was thinking maybe fall 2025 for DePaul DNP program"
- **Matching entity nodes:** 28

### elena_q04 — **E3**

- **Question:** What mental health conditions does Elena have?
- **Correct answer:** Generalized anxiety disorder (takes sertraline)
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Generalized anxiety disorder, takes sertraline
- **Classification:** E3
- **Reason:** Answer edge exists (3 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 3 — `ything, my patients from the shift, stuff I might have missed, things I need to do tomorrow, random anxiety spirals abou...`
- **Matching edges:** 3
  - `[4fa8da2b-01f]` (sertraline) "user is on sertraline for anxiety and it usually helps but lately can't keep up especially with nigh"
  - `[ca7cf89d-15f]` (sertraline) "user drafted message to send about sertraline regarding sleep fragmentation for about a week and wor"
  - `[87a82c6a-67e]` (sertraline) "user plans to send MyChart message to doctor right now about sertraline"
- **Matching entity nodes:** 6

### elena_q08 — **E5**

- **Question:** What's Elena's family background?
- **Correct answer:** Mexican-American, parents from Guadalajara, first-gen college grad. Sister Sofia has Down syndrome.
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Parents from Guadalajara, first-gen college grad, sister Sofia has Down syndrome
- **Classification:** E5
- **Reason:** Composite fact across 3 sessions -- partial answer edges exist but incomplete
- **Conversation evidence:** session 2 — `t and we'll figure out what's realistic. so Saturday I'm off which is rare and I promised my sister Sofia I'd come spend...`
- **Matching edges:** 29
  - `[693b5b27-d51]` (sofia) "Sofia is user's sister"
  - `[0701462f-97d]` (sofia) "user's parents are the main caregivers for Sofia"
  - `[412b9e4b-60c]` (sofia) "user has not visited Sofia in like 3 weeks because of work"
- **Matching entity nodes:** 16

### elena_q12 — **E4**

- **Question:** What does Elena do for stress relief?
- **Correct answer:** Learning to knit, Mediterranean diet, trying anxiety management techniques
- **Attack vector:** AV5_broad_query
- **Expected fact:** Knitting, Mediterranean diet, anxiety management for stress relief
- **Classification:** E4
- **Reason:** Answer edge exists but category {'PREFERENCES_HABITS'} doesn't match relevant {'HEALTH_WELLBEING', 'HOBBIES_RECREATION'}
- **Conversation evidence:** session 22 — `ok real talk I need to change how I'm eating, do you know much about the Mediterranean diet I do! It's one of the most w...`
- **Matching edges:** 1
  - `[f3a98fd8-635]` (knit) "user plans to watch Very Pink Knits on YouTube"
- **Matching entity nodes:** 20

### elena_q13 — **E1**

- **Question:** How much does Elena owe in student loans?
- **Correct answer:** $40,000
- **Attack vector:** AV8_numeric_preservation
- **Expected fact:** Student loans -- question says $40,000, conversation may say 42k
- **Classification:** E1
- **Reason:** Edge has 'about 42k' but question expects exact '$40,000' -- benchmark mismatch
- **Conversation evidence:** session 17 — `almost 60k for the program. and here's the thing that nobody talks about — I still have like 40k in student loans from m...`
- **Matching edges:** 1
  - `[17e42d73-5d3]` (42k) "user has about 42k left on student loans"
- **Matching entity nodes:** 6

### david_q01 — **E3**

- **Question:** What subject does David teach?
- **Correct answer:** AP US History (switched from AP European History)
- **Attack vector:** AV4_multi_version_fact
- **Expected fact:** Switching from AP Euro to AP US History
- **Classification:** E3
- **Reason:** Answer edge exists (5 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 23 — `g about next year. Our department head pulled me aside last week and asked if I would take over the AP US History sectio...`
- **Matching edges:** 5
  - `[347aa38e-c17]` (ap us history, us history) "department head pulled user aside last week to ask if user would take over the AP US History section"
  - `[934d2194-c35]` (ap us history, us history) "user was asked by department head if user would take over the AP US History sections"
  - `[d83925fe-60c]` (ap us history, us history) "retiring teacher has been handling AP US History and is retiring at the end of this year"
- **Matching entity nodes:** 39

### david_q03 — **E3**

- **Question:** What car does David drive?
- **Correct answer:** Subaru Outback
- **Attack vector:** AV4_multi_version_fact
- **Expected fact:** Bought Subaru Outback, replacing old Camry
- **Classification:** E3
- **Reason:** Answer edge exists (10 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 24 — `We finally did it. Went to the dealership on Saturday and drove home in a Subaru Outback. Congratulations! The Outback i...`
- **Matching edges:** 10
  - `[6412b709-25b]` (subaru, outback) "user went to the dealership on Saturday and drove home in a Subaru Outback"
  - `[620a59a4-8f2]` (subaru, outback) "user purchased the Subaru Outback in Premium trim with heated seats and better infotainment system"
  - `[cf56c791-d12]` (subaru, outback) "user's Subaru Outback is in Autumn Green Metallic color chosen by Sarah"
- **Matching entity nodes:** 17

### david_q04 — **E2**

- **Question:** Is David writing a book?
- **Correct answer:** No — he abandoned the primary sources book project
- **Attack vector:** AV7_selective_forgetting
- **Expected fact:** Abandoned book about teaching through primary sources
- **Classification:** E2
- **Reason:** Fact in session 9 -- only topic-level edges exist (no correct-answer edge extracted)
- **Conversation evidence:** session 9 — `some outside perspective. Of course -- I'd be glad to help. What's the project? I've been writing a book. Or rather, I'v...`
- **Matching entity nodes:** 63

### david_q05 — **E3**

- **Question:** What health conditions does David have?
- **Correct answer:** Mild hearing loss in left ear (wears hearing aid)
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Mild hearing loss in left ear, wears hearing aid
- **Classification:** E3
- **Reason:** Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 5 — `I want to do. Before I get into the specifics, I should mention a practical constraint: I have mild hearing loss in my l...`
- **Matching edges:** 1
  - `[aa481679-d7b]` (hearing loss, hearing aid) "user wears a hearing aid due to mild hearing loss in left ear"
- **Matching entity nodes:** 6

### david_q11 — **E3**

- **Question:** Does David have any parent events coming up?
- **Correct answer:** No (parent-teacher conferences were March 12, already passed)
- **Attack vector:** AV2_expired_logistics
- **Expected fact:** Parent-teacher conferences March 12 (expired)
- **Classification:** E3
- **Reason:** Answer edge exists (4 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 10 — `Parent-teacher conferences are next Wednesday, March 12th, and I am not looking forward to them. I realize that's...`
- **Matching edges:** 4
  - `[aeabd287-ff6]` (parent-teacher conference) "David's principal is supportive and has offered to sit in on the parent-teacher conference for Stude"
  - `[d09ac5dd-251]` (parent-teacher conference) "David has a parent-teacher conference scheduled with Student C's father who is on the school board n"
  - `[01db1e07-ce1]` (parent-teacher conference) "David has a parent-teacher conference scheduled with Student A's parents next Wednesday March 12th"
- **Matching entity nodes:** 5

### amara_q01 — **E3**

- **Question:** Which chambers does Amara practise at?
- **Correct answer:** 7 Bedford Row (moved from 4 Brick Court as a senior tenant)
- **Attack vector:** AV4_multi_version_fact
- **Expected fact:** Moved from 4 Brick Court to 7 Bedford Row as senior tenant
- **Classification:** E3
- **Reason:** Answer edge exists (8 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 19 — `ious violence, drugs conspiracies, that sort of thing. Fifteen years at the Bar. I've just joined 7 Bedford Row as a sen...`
- **Matching edges:** 8
  - `[58c23be9-582]` (bedford row, 7 bedford) "Amara Okafor joins 7 Bedford Row as a senior tenant"
  - `[ce009f0c-56a]` (bedford row, 7 bedford) "Prior to joining 7 Bedford Row, Amara Okafor practised at 4 Brick Court"
  - `[829ce940-85f]` (bedford row) "user works at chambers in Bedford Row"
- **Matching entity nodes:** 12

### amara_q02 — **E4**

- **Question:** What phone does Amara use?
- **Correct answer:** Samsung Galaxy S25 (switched from iPhone to Android)
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Switched from iPhone 14 Pro to Samsung Galaxy S25
- **Classification:** E4
- **Reason:** Answer edge exists but category {'FINANCIAL_MATERIAL'} doesn't match relevant {'PREFERENCES_HABITS'}
- **Conversation evidence:** session 22 — `Quick question — I've just set up a new Samsung Galaxy S25 and I'm trying to get my work email configured. The IT depart...`
- **Matching edges:** 2
  - `[fe371188-f81]` (samsung, galaxy s25, galaxy) "user has just set up a new Samsung Galaxy S25"
  - `[9b7e7ba7-10a]` (samsung, galaxy s25, galaxy) "user is getting the hang of navigating the Samsung Galaxy S25 which is rather different"
- **Matching entity nodes:** 26

### amara_q03 — **E3**

- **Question:** What exercise does Amara do?
- **Correct answer:** Boxing at a gym in Bethnal Green
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Dropped running, switched to boxing in Bethnal Green
- **Classification:** E3
- **Reason:** Answer edge exists (7 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 25 — `I need to sort out a weekly schedule that accommodates a new gym routine. I've started boxing at a gym in Bethnal Green ...`
- **Matching edges:** 7
  - `[67498be6-80e]` (boxing, bethnal green) "user has started boxing at a gym in Bethnal Green with classes on Monday, Wednesday, and Friday even"
  - `[8941b4f2-09e]` (boxing, bethnal green) "gym in Bethnal Green offers boxing classes Monday, Wednesday, and Friday evenings at half six"
  - `[e93ac55a-356]` (bethnal green) "gym in Bethnal Green has open gym time until nine where user could use the bags outside of class hou"
- **Matching entity nodes:** 13

### amara_q06 — **E2**

- **Question:** Is Amara pursuing any further education?
- **Correct answer:** No — she considered a human rights LLM at UCL but dropped the idea
- **Attack vector:** AV7_selective_forgetting
- **Expected fact:** Dropped UCL human rights LLM -- not worth time/money
- **Classification:** E2
- **Reason:** Fact in session 11 -- only topic-level edges exist (no correct-answer edge extracted)
- **Conversation evidence:** session 11 — `r more about targeted self-directed learning? Formal study, actually. I've been looking at doing an LLM in human rights ...`
- **Matching entity nodes:** 18

### amara_q07 — **E4**

- **Question:** Does Amara drink alcohol?
- **Correct answer:** Unclear — she said she hadn't drunk in two years, but later mentioned having wine at a chambers dinner
- **Attack vector:** AV6_cross_session_contradiction
- **Expected fact:** Said no alcohol 2 years, but had wine at chambers dinner
- **Classification:** E4
- **Reason:** Answer edge exists but category {'LOGISTICAL_CONTEXT'} doesn't match relevant {'HEALTH_WELLBEING', 'PREFERENCES_HABITS'}
- **Conversation evidence:** session 9 — `. Both have the atmosphere and menu depth to make it feel special. Berber & Q also has an excellent wine and cocktail li...`
- **Matching edges:** 1
  - `[4a9b4143-e7f]` (chambers dinner) "Charles hosted the chambers dinner at his house in Hampstead on Friday night."
- **Matching entity nodes:** 11

### amara_q08 — **E3**

- **Question:** Where does Amara live?
- **Correct answer:** Hackney, East London — though she's been considering a move to Islington
- **Attack vector:** AV9_soft_supersession
- **Expected fact:** Lives in Hackney, considering move to Islington
- **Classification:** E3
- **Reason:** Answer edge exists (10 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 1 — `round in circles on this before. I need to finish this tonight because I've got an early train from Hackney Central into...`
- **Matching edges:** 10
  - `[c04065b6-83d]` (hackney) "user has an early train from Hackney Central into the Temple in the morning"
  - `[5fa72abc-5bb]` (hackney) "user lives in Hackney and has been there for years"
  - `[d6140214-05d]` (hackney) "user wants to stay in Hackney due to park, transport links, and neighbourhood"
- **Matching entity nodes:** 30

### amara_q11 — **E3**

- **Question:** What does Amara do outside of work?
- **Correct answer:** Boxing, Nigerian cooking experiments, podcasts, reading about justice
- **Attack vector:** AV5_broad_query
- **Expected fact:** Boxing, Nigerian cooking, podcasts, reading about justice
- **Classification:** E3
- **Reason:** Answer edge exists (4 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 25 — `I need to sort out a weekly schedule that accommodates a new gym routine. I've started boxing at a gym in Bethnal Green ...`
- **Matching edges:** 4
  - `[67498be6-80e]` (boxing) "user has started boxing at a gym in Bethnal Green with classes on Monday, Wednesday, and Friday even"
  - `[8941b4f2-09e]` (boxing) "gym in Bethnal Green offers boxing classes Monday, Wednesday, and Friday evenings at half six"
  - `[9da0e809-e3d]` (boxing) "Kofi is user's boxing trainer at the gym and is excellent, demanding but encouraging, and notices wh"
- **Matching entity nodes:** 33

### amara_q12 — **E3**

- **Question:** What is Amara's educational background?
- **Correct answer:** LLB from King's College London
- **Attack vector:** AV3_stable_identity
- **Expected fact:** LLB from King's College London
- **Classification:** E3
- **Reason:** Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 2 — `I think I've got enough to work with for now. Though it does make me reflect — when I was doing my LLB at King's, nobody...`
- **Matching edges:** 1
  - `[270471c2-2d4]` (llb) "user did LLB at King's"
- **Matching entity nodes:** 1

### amara_q14 — **E2**

- **Question:** How much would the UCL master's programme have cost Amara?
- **Correct answer:** £18,000
- **Attack vector:** AV8_numeric_preservation
- **Expected fact:** UCL LLM would have cost GBP 18,000
- **Classification:** E2
- **Reason:** Entity node references fact but no answer edge extracted
- **Conversation evidence:** session None — `...`
- **Matching entity nodes:** 6

### jake_q02 — **E3**

- **Question:** Who is Jake dating?
- **Correct answer:** Kayla, a nurse at Mass General (broke up with Megan)
- **Attack vector:** AV4_multi_version_fact
- **Expected fact:** Megan broke up, now dating Kayla (nurse at Mass General)
- **Classification:** E3
- **Reason:** Answer edge exists (3 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 24 — `been together a while? its still kinda new so like maybe our 4th or 5th time hangin out. this girl kayla I been seein sh...`
- **Matching edges:** 3
  - `[48f78f04-876]` (kayla) "Kayla is a nurse at Mass General"
  - `[3081ed16-f34]` (kayla) "user has been seeing Kayla for their 4th or 5th time hanging out"
  - `[cb9e7931-223]` (kayla) "Kayla is from the area of Boston"
- **Matching entity nodes:** 11

### jake_q03 — **E3**

- **Question:** What does Jake's mom do?
- **Correct answer:** Waitresses at the Eire Pub in Dorchester — been there 20 years
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Mom waitresses at Eire Pub in Dorchester, 20 years
- **Classification:** E3
- **Reason:** Answer edge exists (8 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 3 — `dorchester actually her apartment is like right near where my mom works. my moms been workin at the eire pub in dorchest...`
- **Matching edges:** 8
  - `[c9c50487-2ce]` (dorchester) "Megan lives in an apartment in Dorchester"
  - `[b35df5fc-21e]` (eire pub, dorchester) "user's mom has been working at the Eire Pub in Dorchester for 20 years"
  - `[72a13d65-f69]` (eire pub, dorchester) "Eire Pub is located in Dorchester"
- **Matching entity nodes:** 10

### jake_q05 — **E3**

- **Question:** Is Jake doing any side work?
- **Correct answer:** No — he wanted to start installing EV home chargers on the side but his dad said he needs his journeyman license first and the insurance is too expensive
- **Attack vector:** AV7_selective_forgetting
- **Expected fact:** EV charger side gig dead -- dad said not ready, insurance too expensive
- **Classification:** E3
- **Reason:** Answer edge exists (7 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 11 — `electricians around here arent even doin that work yet Yeah, that's a growing market for sure. Home EV charger installat...`
- **Matching edges:** 7
  - `[17d41347-bfa]` (ev charger) "user is thinking worst case he could do EV charger installations through dads company on the side to"
  - `[59988836-29b]` (ev charger) "user is considering certifications or permits in Massachusetts to do EV charger installations indepe"
  - `[5739f627-828]` (ev charger) "user had a plan to start doing EV charger installs on the side at people's houses on weekends"
- **Matching entity nodes:** 13

### jake_q07 — **E3**

- **Question:** What does Jake drink?
- **Correct answer:** Craft beer, especially Trillium IPAs
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Got into craft beer -- favorites from Trillium Brewing
- **Classification:** E3
- **Reason:** Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 26 — `dude have you ever been to trillium brewin before Trillium is a great brewery! They're based in the Boston area and they...`
- **Matching edges:** 2
  - `[e49a6235-792]` (trillium) "user was dragged to Trillium by Marcus like a month ago."
  - `[257cbe5a-9ae]` (trillium) "user is buying Trillium cans every week."
- **Matching entity nodes:** 9

### jake_q09 — **E5**

- **Question:** What's Jake's family background?
- **Correct answer:** Irish-American, grandparents from County Cork. Dad runs Brennan & Sons Electric. Mom waitresses at Eire Pub in Dorchester. Sister Bridget.
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Irish-American, grandparents County Cork, dad Brennan & Sons, mom Eire Pub, sister Bridget
- **Classification:** E5
- **Reason:** Composite fact across 3 sessions -- partial answer edges exist but incomplete
- **Conversation evidence:** session 1 — `ought and wants to gut the whole thing new panel new circuits the works. I work for my dads company brennan and sons ele...`
- **Matching edges:** 3
  - `[b35df5fc-21e]` (eire pub) "user's mom has been working at the Eire Pub in Dorchester for 20 years"
  - `[72a13d65-f69]` (eire pub) "Eire Pub is located in Dorchester"
  - `[f211c184-3d0]` (eire pub) "Megan's apartment is right near where user's mom works at the Eire Pub in Dorchester"
- **Matching entity nodes:** 20

### jake_q10 — **E5**

- **Question:** What does Jake do outside of work?
- **Correct answer:** Rec hockey on Wednesdays, gaming (new PC), craft beer, sports with the boys
- **Attack vector:** AV5_broad_query
- **Expected fact:** Rec hockey Wednesdays, gaming PC, craft beer, sports with the boys
- **Classification:** E5
- **Reason:** Composite fact across 3 sessions -- partial answer edges exist but incomplete
- **Conversation evidence:** session 4 — `r something intense. Whatever it was, your muscles are probably screaming at you right now. nah its hockey I play in a r...`
- **Matching edges:** 5
  - `[0a486058-a72]` (bajko) "user plays in a rec league every Wednesday night at Bajko Rink"
  - `[45d6a0d2-d74]` (bajko) "user plays right wing usually at Bajko Rink but covered for center last night after injury in second"
  - `[117ddc91-610]` (bajko) "user's hockey league at Bajko Rink goes all the way through March"
- **Matching entity nodes:** 14

### tom_q02 — **E3**

- **Question:** What car does Tom drive?
- **Correct answer:** Hyundai Ioniq 5 (electric — sold his Land Rover Defender)
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Sold Land Rover Defender, bought Hyundai Ioniq 5
- **Classification:** E3
- **Reason:** Answer edge exists (3 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 24 — `nd nature.  What make and model have you got, and what's been confusing you so far? It is a Hyundai Ioniq 5. My daughter...`
- **Matching edges:** 3
  - `[8a4f0b67-f1a]` (ioniq, hyundai) "user has just bought a Hyundai Ioniq 5"
  - `[dcbb8da8-1e8]` (ioniq, hyundai) "user drove past the hives the other day in the Hyundai Ioniq 5 and the bees did not even look up"
  - `[3b011018-0dd]` (ioniq, hyundai) "user noticed the bees did not even look up when driving past the hives in the Hyundai Ioniq 5"
- **Matching entity nodes:** 9

### tom_q03 — **E3**

- **Question:** What does Tom do socially?
- **Correct answer:** Walking group (stopped the pub quiz after Gerald couldn't make it)
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Stopped pub quiz at The Fox, joined walking group
- **Classification:** E3
- **Reason:** Answer edge exists (11 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 23 — `in the Cotswolds — circular routes, ideally, between five and eight miles. I have recently joined a walking group and I ...`
- **Matching edges:** 11
  - `[d31a7bb5-d91]` (walking group) "user has recently joined a walking group"
  - `[12be49b2-d98]` (walking group) "walking group has about twelve members, though not everyone comes every week"
  - `[1e1af1bc-7ec]` (walking group) "walking group is fairly mixed with some quite fit members and others more leisurely"
- **Matching entity nodes:** 10

### tom_q04 — **E3**

- **Question:** Does Tom have any health conditions?
- **Correct answer:** Atrial fibrillation — takes warfarin (blood thinners)
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Atrial fibrillation, takes warfarin
- **Classification:** E3
- **Reason:** Answer edge exists (3 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 4 — `e never had an allergic reaction.  As for medication — yes, that is rather the complication. I have atrial fibrillation,...`
- **Matching edges:** 3
  - `[ddaa58cf-a17]` (atrial fibrillation) "user has atrial fibrillation"
  - `[e77f4ffc-3e8]` (atrial fibrillation, warfarin) "user takes warfarin for atrial fibrillation"
  - `[2a560c8d-1a4]` (warfarin) "user takes warfarin as blood thinner causing excessive bruising from bee sting"
- **Matching entity nodes:** 6

### tom_q06 — **E3**

- **Question:** Is Tom planning any building projects?
- **Correct answer:** No — he planned to convert a barn into a honey processing room but it was scrapped (planning permission denied, too expensive)
- **Attack vector:** AV7_selective_forgetting
- **Expected fact:** Barn conversion scrapped -- planning permission denied, too expensive
- **Classification:** E3
- **Reason:** Answer edge exists (4 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 11 — `happy to help you think through an idea. What have you been considering? Well, I have an old stone barn at the back of t...`
- **Matching edges:** 4
  - `[7baaa617-8b1]` (planning permission) "user needs to check with Cotswold District Council for planning permission to convert the stone barn"
  - `[6b3b5f97-9ba]` (barn conversion) "user submitted a planning application for the barn conversion which was refused"
  - `[e6796199-6f8]` (barn conversion) "user received a builder's estimate of forty thousand pounds for the barn conversion"
- **Matching entity nodes:** 20

### tom_q07 — **E3**

- **Question:** Does Tom use social media?
- **Correct answer:** Inconsistent — he said he'd never use social media, but later posted a bee photo on Instagram that got 200 likes
- **Attack vector:** AV6_cross_session_contradiction
- **Expected fact:** Said no social media, but posted bee photo on Instagram (200 likes)
- **Classification:** E3
- **Reason:** Answer edge exists (9 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 9 — `s to deal with.  As for what Sophie means -- I believe she is thinking of social media. Facebook or Instagram, that sort...`
- **Matching edges:** 9
  - `[6d9274a6-333]` (instagram) "user has no interest in Facebook or Instagram whatsoever"
  - `[8ae005cf-b02]` (instagram) "user has no interest in Facebook or Instagram whatsoever"
  - `[8bb0b2cc-6e2]` (instagram) "Sophie set up an Instagram account for the user when she was here in the summer"
- **Matching entity nodes:** 7

### tom_q11 — **E3**

- **Question:** What was Tom's career?
- **Correct answer:** Civil engineer — BEng, worked at Arup for 35 years
- **Attack vector:** AV3_stable_identity
- **Expected fact:** BEng civil engineering, worked at Arup 35 years
- **Classification:** E3
- **Reason:** Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 5 — `and I can answer them because, as it happens, this is rather my area. I spent thirty-five years at Arup — civil engineer...`
- **Matching edges:** 1
  - `[83c2443d-d96]` (arup) "user spent thirty-five years at Arup in civil engineering"
- **Matching entity nodes:** 2

### omar_q01 — **RANKING**

- **Question:** What rideshare platform does Omar drive for?
- **Correct answer:** Uber (was on Uber, switched to Lyft, then back to Uber because Lyft had too few rides in Houston)
- **Attack vector:** AV4_multi_version_fact
- **Expected fact:** Uber -> Lyft -> back to Uber (triple version)
- **Classification:** RANKING
- **Reason:** Edge exists and is answerable -- stale Lyft edge outranks correct Uber edge
- **Conversation evidence:** session 1 — `whats good. long day out here driving man Hey! Sounds like a grind. What kind of driving do you do? uber bro. been doing...`
- **Matching edges:** 9
  - `[49b1bde9-ad0]` (uber) "user has been doing uber full time for 3 years"
  - `[22c8232e-ed1]` (uber) "user switched from Uber due to issues with surge pricing disappearing and smaller fares for same dis"
  - `[26900567-63b]` (uber) "user toggles between both Uber and Lyft apps, running both at the same time"
- **Matching entity nodes:** 15

### omar_q02 — **E3**

- **Question:** Where does Omar live?
- **Correct answer:** Studio in Gulfton (moved from 1-bedroom in Alief to save $200/month)
- **Attack vector:** AV1_superseded_preference
- **Expected fact:** Moved from Alief 1BR to Gulfton studio, saves $200/mo
- **Classification:** E3
- **Reason:** Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 22 — `Moving is brutal. Where'd you end up? Hopefully not too far from where you were. nah just moved to gulfton. not far at a...`
- **Matching edges:** 2
  - `[9352112d-631]` (gulfton) "user just moved to Gulfton"
  - `[e6a44e42-ce2]` (gulfton) "Gulfton spot is closer to a library for user"
- **Matching entity nodes:** 9

### omar_q05 — **E3**

- **Question:** Is Omar planning to buy a car?
- **Correct answer:** No — he wanted a used Toyota Camry but the car got sold before he could secure financing
- **Attack vector:** AV7_selective_forgetting
- **Expected fact:** Camry plan dead -- car sold before financing secured
- **Classification:** E3
- **Reason:** Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 10 — `how many mpg does a 2019 toyota camry get The 2019 Toyota Camry gets around 29 city / 41 highway for the base L and LE t...`
- **Matching edges:** 1
  - `[9ac071f2-c57]` (whole camry plan) "the whole Camry plan is dead for now"
- **Matching entity nodes:** 30

### omar_q06 — **E3**

- **Question:** Does Omar spend money on himself?
- **Correct answer:** He said he never spends on himself, but he bought a PS5 for stress relief
- **Attack vector:** AV6_cross_session_contradiction
- **Expected fact:** Said never spends on himself but bought PS5 for stress
- **Classification:** E3
- **Reason:** Answer edge exists (1 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 25 — `you play fifa at all I don't play myself, but I know a lot about it. Are you into FIFA? just got a ps5 bro. been playing...`
- **Matching edges:** 1
  - `[4513bf8f-e64]` (ps5) "user got PS5 off Facebook Marketplace with 2 controllers and FIFA for 350"
- **Matching entity nodes:** 3

### omar_q07 — **E4**

- **Question:** Is Omar planning to move?
- **Correct answer:** He's staying in Houston for now (Ismail is here), but his cousin says the Dallas RE market is better — he's considering it for the future
- **Attack vector:** AV9_soft_supersession
- **Expected fact:** Staying in Houston but cousin says Dallas RE market is better
- **Classification:** E4
- **Reason:** Answer edge exists but category {'IDENTITY_SELF_CONCEPT'} doesn't match relevant {'RELATIONAL_BONDS', 'LOGISTICAL_CONTEXT'}
- **Conversation evidence:** session 4 — `someone told me i should move to dallas for real estate. like the market is better there or whatever Dallas does have a ...`
- **Matching edges:** 1
  - `[8c61b5f8-d4c]` (dallas) "Khalid lives in Dallas"
- **Matching entity nodes:** 49

### omar_q08 — **E5**

- **Question:** What does Omar do for money?
- **Correct answer:** Uber driver (full-time) + starting at Century 21 real estate brokerage in April 2026
- **Attack vector:** AV5_broad_query
- **Expected fact:** Uber driver + starting at Century 21 in April 2026
- **Classification:** E5
- **Reason:** Composite fact across 2 sessions -- partial answer edges exist but incomplete
- **Conversation evidence:** session 26 — `roup chat said same thing. lyft just doesnt have the riders in houston like that. so i went back to uber Back to Uber — ...`
- **Matching edges:** 2
  - `[4e083636-940]` (century 21) "user starts at Century 21 on April 1st"
  - `[8edfa00f-f15]` (century 21) "Century 21 office is in Energy Corridor"
- **Matching entity nodes:** 26

### omar_q11 — **E1**

- **Question:** What are Omar's health conditions?
- **Correct answer:** None explicitly mentioned (but he's tired and stressed from overwork)
- **Attack vector:** AV3_stable_identity
- **Expected fact:** No explicit health conditions -- tired/stressed from overwork
- **Classification:** E1
- **Reason:** Correct answer is 'None explicitly mentioned' -- no extractable fact
- **Conversation evidence:** session None — `...`

### omar_q12 — **E3**

- **Question:** What's Omar's career plan?
- **Correct answer:** Real estate — passed the exam, joining Century 21 in April 2026, still driving Uber on the side
- **Attack vector:** AV3_stable_identity
- **Expected fact:** Real estate career -- passed exam, joining Century 21
- **Classification:** E3
- **Reason:** Answer edge exists (21 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 9 — `s. What kind of licensing exam are we talking about, and how much time do you have before the test? real estate exam. ap...`
- **Matching edges:** 21
  - `[098475cc-50e]` (real estate) "user is studying for real estate license between rides"
  - `[fbd254c7-4ba]` (real estate) "user knows Houston, its neighborhoods, and areas where people are buying as advantage for real estat"
  - `[b385a171-97a]` (real estate) "user is studying for the real estate exam between rides and at night when Ismail is asleep"
- **Matching entity nodes:** 30

### omar_q14 — **E3**

- **Question:** How much does Omar send his mother in Khartoum each month?
- **Correct answer:** $300-400 per month
- **Attack vector:** AV8_numeric_preservation
- **Expected fact:** Sends $300-400/month to mother in Khartoum
- **Classification:** E3
- **Reason:** Answer edge exists (2 enriched) but not retrieved -- routing/retrieval gap
- **Conversation evidence:** session 3 — `ting at once. Where does your mom live? khartoum. i send her money every month no matter what. like 300 400 depending on...`
- **Matching edges:** 2
  - `[f7b4096d-186]` (300, 400) "user sends user's mom 300 400 every month depending on what user makes"
  - `[f5b098b3-14b]` (400) "user putting $2400 a year toward real estate fund from studio savings"
- **Matching entity nodes:** 16

## Key Patterns & Insights

### Fact Types Grok Consistently Misses (E2 by AV)

- AV7_selective_forgetting: 2 misses
- AV1_superseded_preference: 1 misses
- AV8_numeric_preservation: 1 misses

### E2 Breakdown by Temporal Pattern

- Stable identity facts (AV3): 0 — single-mention facts buried under noise
- Superseded preferences (AV1): 1 — both old and new versions missing
- Retractions (AV7): 2 — retracted plans not extracted at all

### Retrieval Failures by AV (E3 -- dominant category)

- AV1_superseded_preference: 9 retrieval failures
- AV3_stable_identity: 9 retrieval failures
- AV7_selective_forgetting: 5 retrieval failures
- AV4_multi_version_fact: 5 retrieval failures
- AV6_cross_session_contradiction: 2 retrieval failures
- AV2_expired_logistics: 1 retrieval failures
- AV9_soft_supersession: 1 retrieval failures
- AV5_broad_query: 1 retrieval failures
- AV8_numeric_preservation: 1 retrieval failures

### Entity Node Summaries vs Edge Facts

- 4 gaps where entity node has relevant info but no answer edge was created
- This suggests entity summarization captures some facts that edge extraction misses

## Recommendations

### For E3+E4 (Retrieval/Routing -- Primary Bottleneck, ~78% of gaps)
- **This is the #1 issue:** correct facts ARE in the graph but not retrieved
- Improve semantic retrieval: current embedding-based search misses relevant edges
- Category routing: E4 gaps show edges exist with wrong fr_primary_category
- Consider query expansion: rephrase user questions to match edge phrasing
- Top-k increase: some facts exist but rank below the top-5 cutoff
- Cross-category retrieval: allow queries to pull from multiple categories

### For E1 (Benchmark Design Issues)
- elena_q13: conversation says '42k' but question expects '$40,000' -- fix question
- omar_q11: correct answer is 'None explicitly mentioned' -- consider rewording

### For E2 (Extraction Misses -- 4 gaps)
- priya_q07: 'rock climbing' not extracted (only vague 'climbing' edges)
- david_q04: 'abandoned the book' not captured as retraction
- amara_q06: 'dropped the LLM' retraction not captured
- amara_q14: GBP 18,000 tuition cost not extracted
- Add extraction prompt instructions for retractions and specific numeric details

### For E5 (Composite Facts -- 4 gaps)
- Cross-session fact synthesis: family background, hobbies, career summaries
- Consider composite edge creation for identity/family questions
- Multi-edge retrieval: allow broad queries to pull from multiple edges
