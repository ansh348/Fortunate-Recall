# LifeMemBench Failure Audit Report

*Generated: 2026-03-04 21:54*

**Results file:** `lifemembench_results.json` (alpha=0.3, 112 questions)

## Executive Summary

- **Total questions:** 112
- **Passing (av_pass=True):** 71 (63.4%)
- **Failing (av_pass=False):** 41 (36.6%)
- **Answerable but failing:** 3

| Classification | Count | % of Failures | Description |
|---|---|---|---|
| **E1** | 2 | 4.9% | Benchmark design bug (wrong ground truth) |
| **E2** | 2 | 4.9% | Fact never extracted (no edge exists) |
| **E3-TRUE** | 27 | 65.9% | Correct edge exists but retrieval didn't surface it |
| **E3-WEAK** | 2 | 4.9% | Edge exists but too vague/partial to answer |
| **E4** | 4 | 9.8% | Edge in wrong behavioral category |
| **AV7-CROSS** | 1 | 2.4% | Cross-category supersession failure |
| **RETRACTION** | 2 | 4.9% | Retraction event not extracted |
| **RANKING** | 1 | 2.4% | Correct edge in pool but outranked |
| **Total** | **41** | **100%** | |

**Key insight:** 33/41 failures (80%) are fixable by code changes alone (retrieval/routing/ranking). Only 4 require re-ingestion.

## Table 1: Per-Question Failure Detail

| # | Question ID | AV | Persona | Question | Expected Answer | Top Retrieved Edges | Classification | Fix |
|---|---|---|---|---|---|---|---|---|
| 1 | `amara_q02` | AV1 | amara | What phone does Amara use? | Samsung Galaxy S25 (switched from iPhone to Android) | #1 (bl=0.950) Amara works at chambers.<br>#2 (bl=0.912) Amara is a member of the Bar.<br>#3 (bl=0.876) user does not need phone buzzing for round-robin emails about the chambers Chris<br>#4 (bl=0.858) Amara has a warm but professional relationship with Charles.<br>#5 (bl=0.847) Amara has been at chambers for years | **E4** | (a) Code change only |
| 2 | `amara_q03` | AV1 | amara | What exercise does Amara do? | Boxing at a gym in Bethnal Green | #1 (bl=0.950) Amara has endometriosis<br>#2 (bl=0.930) Amara is a member of the Bar.<br>#3 (bl=0.911) Amara works at chambers.<br>#4 (bl=0.893) Amara has a warm but professional relationship with Charles.<br>#5 (bl=0.864) Amara has been at chambers for years | **E3-TRUE** | (a) Code change only |
| 3 | `amara_q06` | AV7 | amara | Is Amara pursuing any further education? | No — she considered a human rights LLM at UCL but dropped the idea | #1 (bl=0.950) Amara is a member of the Bar.<br>#2 (bl=0.931) Amara works at chambers.<br>#3 (bl=0.916) Amara ended up chasing CPS for materials in previous case with instructing solic<br>#4 (bl=0.911) Amara has endometriosis<br>#5 (bl=0.865) Amara has been at chambers for years | **RETRACTION** | (b) Re-ingestion only |
| 4 | `amara_q07` | AV6 | amara | Does Amara drink alcohol? | Unclear — she said she hadn't drunk in two years, but later mentioned having win | #1 (bl=0.950) Amara is a member of the Bar.<br>#2 (bl=0.923) Amara has endometriosis<br>#3 (bl=0.896) Amara works at chambers.<br>#4 (bl=0.865) Amara has a warm but professional relationship with Charles.<br>#5 (bl=0.843) Amara has been at chambers for years | **E4** | (a) Code change only |
| 5 | `amara_q08` | AV9 | amara | Where does Amara live? | Hackney, East London — though she's been considering a move to Islington | #1 (bl=0.950) Amara is a member of the Bar.<br>#2 (bl=0.931) Amara has endometriosis<br>#3 (bl=0.912) Amara works at chambers.<br>#4 (bl=0.895) Amara has a warm but professional relationship with Charles.<br>#5 (bl=0.866) Amara has been at chambers for years | **E3-TRUE** | (a) Code change only |
| 6 | `amara_q11` | AV5 | amara | What does Amara do outside of work? | Boxing, Nigerian cooking experiments, podcasts, reading about justice | #1 (bl=0.950) Amara is a member of the Bar.<br>#2 (bl=0.931) Amara works at chambers.<br>#3 (bl=0.911) Amara has endometriosis<br>#4 (bl=0.865) Amara has been at chambers for years<br>#5 (bl=0.856) Amara has a warm but professional relationship with Charles. | **E3-TRUE** | (a) Code change only |
| 7 | `amara_q12` | AV3 | amara | What is Amara's educational background? | LLB from King's College London | #1 (bl=0.950) Amara's senior clerk is quite good<br>#2 (bl=0.931) Amara is a member of the Bar.<br>#3 (bl=0.912) Amara's brother is Chidi who is a GP<br>#4 (bl=0.876) Amara works at chambers.<br>#5 (bl=0.868) most of Amara's work is Crown Court, so hearings are in person | **E3-TRUE** | (a) Code change only |
| 8 | `amara_q14` | AV8 | amara | How much would the UCL master's programme have cost Amara? | £18,000 | #1 (bl=0.950) Amara has endometriosis<br>#2 (bl=0.912) Amara is a member of the Bar.<br>#3 (bl=0.874) Amara works at chambers.<br>#4 (bl=0.803) two barristers actually did the LLM in human rights law at UCL<br>#5 (bl=0.770) user had been looking at the LLM in human rights law at UCL | **E2** | (b) Re-ingestion only |
| 9 | `david_q01` | AV4 | david | What subject does David teach? | AP US History (switched from AP European History) | #1 (bl=0.974) user is David<br>#2 (bl=0.938) David lives in the Pacific Northwest<br>#3 (bl=0.933) David's student is a junior with strong reading skills.<br>#4 (bl=0.931) David has twenty-four years of working with primary sources in a real classroom<br>#5 (bl=0.913) David teaches user's student who is a junior with strong reading skills. | **E3-TRUE** | (a) Code change only |
| 10 | `david_q03` | AV4 | david | What car does David drive? | Subaru Outback | #1 (bl=0.974) user is David<br>#2 (bl=0.956) David lives in the Pacific Northwest<br>#3 (bl=0.924) David is considering entering the March invitational.<br>#4 (bl=0.912) user had been driving a fourteen-year-old Camry *[STALE]*<br>#5 (bl=0.893) user mentioned getting new car to one of my students who reacted to driving four *[STALE]* | **E3-TRUE** | (a) Code change only |
| 11 | `david_q04` | AV7 | david | Is David writing a book? | No — he abandoned the primary sources book project | #1 (bl=0.974) user is David<br>#2 (bl=0.956) David lives in the Pacific Northwest<br>#3 (bl=0.951) David's student is a junior with strong reading skills.<br>#4 (bl=0.901) David occasionally gets a handwritten note from a student, and those are genuine<br>#5 (bl=0.898) David is considering entering the March invitational. | **RETRACTION** | (b) Re-ingestion only |
| 12 | `david_q05` | AV3 | david | What health conditions does David have? | Mild hearing loss in left ear (wears hearing aid) | #1 (bl=0.974) David lives in the Pacific Northwest<br>#2 (bl=0.938) user is David<br>#3 (bl=0.911) David is considering entering the March invitational.<br>#4 (bl=0.905) David knows a few coaches at Portland-area schools.<br>#5 (bl=0.895) David has survived more than twenty Teacher Appreciation Weeks at this point | **E3-TRUE** | (a) Code change only |
| 13 | `david_q11` | AV2 | david | Does David have any parent events coming up? | No (parent-teacher conferences were March 12, already passed) | #1 (bl=0.969) David is already dreading Teacher Appreciation Week coming up at his school<br>#2 (bl=0.897) David is considering entering the March invitational.<br>#3 (bl=0.881) user is David<br>#4 (bl=0.862) David lives in the Pacific Northwest<br>#5 (bl=0.835) David has a parent-teacher conference scheduled with Student A's parents next We *[STALE]* | **E3-TRUE** | (a) Code change only |
| 14 | `elena_q01` | AV4 | elena | Where does Elena work? | Rush University Medical Center | #1 (bl=0.985) Elena works night shifts with her coworkers at the hospital and does 12-hour shi<br>#2 (bl=0.951) Elena is little Elena from the west side.<br>#3 (bl=0.910) Elena is the user<br>#4 (bl=0.898) Chicago is home for Elena.<br>#5 (bl=0.889) Elena is engaged to Marco | **E3-TRUE** | (a) Code change only |
| 15 | `elena_q02` | AV1 | elena | What diet is Elena following? | Mediterranean diet | #1 (bl=0.950) Elena is the user<br>#2 (bl=0.934) Elena is little Elena from the west side.<br>#3 (bl=0.931) user is referred to as Elena<br>#4 (bl=0.919) Chicago is home for Elena.<br>#5 (bl=0.911) Elena is engaged to Marco | **E3-TRUE** | (a) Code change only |
| 16 | `elena_q03` | AV7 | elena | Is Elena planning to get a nurse practitioner degree? | No — she dropped that plan due to student loan debt | #1 (bl=0.931) Jasmine stayed in nursing school because of watching nurses like Elena<br>#2 (bl=0.913) Elena works night shifts with her coworkers at the hospital and does 12-hour shi<br>#3 (bl=0.912) user is referred to as Elena<br>#4 (bl=0.875) Elena is engaged to Marco<br>#5 (bl=0.865) Jessica is user's charge nurse | **E3-TRUE** | (a) Code change only |
| 17 | `elena_q04` | AV3 | elena | What mental health conditions does Elena have? | Generalized anxiety disorder (takes sertraline) | #1 (bl=0.985) Elena works night shifts with her coworkers at the hospital and does 12-hour shi<br>#2 (bl=0.933) Elena is little Elena from the west side.<br>#3 (bl=0.931) user is referred to as Elena<br>#4 (bl=0.901) Chicago is home for Elena.<br>#5 (bl=0.893) Elena is engaged to Marco | **E3-TRUE** | (a) Code change only |
| 18 | `elena_q08` | AV3 | elena | What's Elena's family background? | Mexican-American, parents from Guadalajara, first-gen college grad. Sister Sofia | #1 (bl=0.973) Elena's mom screamed and cried upon learning of Elena's engagement.<br>#2 (bl=0.938) Elena's mom already has opinions on the wedding.<br>#3 (bl=0.931) user is referred to as Elena<br>#4 (bl=0.916) Elena is little Elena from the west side.<br>#5 (bl=0.901) Elena's sister yelled WEDDING WEDDING WEDDING upon learning of her engagement. | **E3-WEAK** | (c) Both code + re-ingestion |
| 19 | `elena_q12` | AV5 | elena | What does Elena do for stress relief? | Learning to knit, Mediterranean diet, trying anxiety management techniques | #1 (bl=0.985) Elena works night shifts with her coworkers at the hospital and does 12-hour shi<br>#2 (bl=0.962) Diana got a couples massage gift certificate for Elena and Marco<br>#3 (bl=0.937) Chicago is home for Elena.<br>#4 (bl=0.898) Elena is little Elena from the west side.<br>#5 (bl=0.877) Elena is the user | **E4** | (a) Code change only |
| 20 | `elena_q13` | AV8 | elena | How much does Elena owe in student loans? | $40,000 | #1 (bl=0.975) user's student loans are all federal loans<br>#2 (bl=0.955) user has about 42k left on student loans<br>#3 (bl=0.916) user's student loans have interest rate around 5.5% or 6%<br>#4 (bl=0.905) user has been paying on student loans for like 4 years<br>#5 (bl=0.851) Elena spent $340 on DoorDash last month | **E1** | (d) Benchmark fix |
| 21 | `jake_q03` | AV3 | jake | What does Jake's mom do? | Waitresses at the Eire Pub in Dorchester — been there 20 years | #1 (bl=0.982) user is identified as Jake on the suggested birthday card for user's mom<br>#2 (bl=0.937) Bridget is the daughter of user's mom<br>#3 (bl=0.933) this dude asked user what he does and said 'oh so you didnt go to college huh' w<br>#4 (bl=0.910) user plans to send user's mom a birthday text tomorrow<br>#5 (bl=0.907) user's mom is on user's case about the suit fitting every single day | **E3-TRUE** | (a) Code change only |
| 22 | `jake_q05` | AV7 | jake | Is Jake doing any side work? | No — he wanted to start installing EV home chargers on the side but his dad said | #1 (bl=0.963) user is identified as Jake on the suggested birthday card for user's mom<br>#2 (bl=0.963) user got out of high school and started doing electrical work<br>#3 (bl=0.950) user introduced himself as Jake in the text message to Mr. Patterson<br>#4 (bl=0.907) other guys on the crew listen to user now due to journeyman status<br>#5 (bl=0.871) user works on job sites all day | **E3-TRUE** | (a) Code change only |
| 23 | `jake_q07` | AV1 | jake | What does Jake drink? | Craft beer, especially Trillium IPAs | #1 (bl=0.963) user is identified as Jake on the suggested birthday card for user's mom<br>#2 (bl=0.950) user introduced himself as Jake in the text message to Mr. Patterson<br>#3 (bl=0.915) this dude asked user what he does and said 'oh so you didnt go to college huh' w<br>#4 (bl=0.891) Danny is assigned to bring breakfast supplies (eggs, bacon, bread, coffee, butte<br>#5 (bl=0.882) Sully has a place where user brought beers last weekend. | **E3-TRUE** | (a) Code change only |
| 24 | `jake_q10` | AV5 | jake | What does Jake do outside of work? | Rec hockey on Wednesdays, gaming (new PC), craft beer, sports with the boys | #1 (bl=0.963) user is identified as Jake on the suggested birthday card for user's mom<br>#2 (bl=0.957) Danny is one of the boys with user planning the lake trip<br>#3 (bl=0.950) user introduced himself as Jake in the text message to Mr. Patterson<br>#4 (bl=0.895) this dude asked user what he does and said 'oh so you didnt go to college huh' w<br>#5 (bl=0.850) user got out of high school and started doing electrical work | **E3-WEAK** | (c) Both code + re-ingestion |
| 25 | `marcus_q03` | AV7 | marcus | Is Marcus planning to open a second shop location? | No — he considered Germantown but scrapped it (too much financial risk) | #1 (bl=0.876) second shop already has 3 bays and a lift<br>#2 (bl=0.847) shop owner has good people working with him<br>#3 (bl=0.821) user plans to put up security cameras at the shop<br>#4 (bl=0.793) shop owner expects to be busy with summer coming up<br>#5 (bl=0.742) user owns independent shop | **E3-TRUE** | (a) Code change only |
| 26 | `marcus_q06` | AV1 | marcus | What does Marcus do for fun? | Fishing on weekends | #1 (bl=0.950) Robert Chandler mentioned getting a lawyer in the voicemail to Marcus Thompson<br>#2 (bl=0.913) user should have a brief written record of today's incident with the customer in<br>#3 (bl=0.870) Reggie is one of user's guys and is pretty organized<br>#4 (bl=0.838) Marcus Thompson's shop used quality name-brand parts for Robert Chandler's brake<br>#5 (bl=0.806) user wants to post on the facebook page but does not know what to put | **E3-TRUE** | (a) Code change only |
| 27 | `marcus_q13` | AV8 | marcus | How many employees does Marcus currently have at the shop? | 4 (recently hired Carlos as the fourth employee) | #1 (bl=0.931) Marcus Thompson's shop used quality name-brand parts for Robert Chandler's brake<br>#2 (bl=0.926) shop owner earned business success with how he treats customers<br>#3 (bl=0.911) user works at the shop<br>#4 (bl=0.908) shop owner has good people working with him<br>#5 (bl=0.880) user plans to put up security cameras at the shop | **E4** | (a) Code change only |
| 28 | `omar_q01` | AV4 | omar | What rideshare platform does Omar drive for? | Uber (was on Uber, switched to Lyft, then back to Uber because Lyft had too few  | #1 (bl=0.950) user is Omar<br>#2 (bl=0.934) user has been driving for Lyft *[STALE]*<br>#3 (bl=0.911) Omar is the user<br>#4 (bl=0.900) Uber currently pays the bills for user **[CORRECT]**<br>#5 (bl=0.880) assistant refers to user as Omar | **RANKING** | (a) Code change only |
| 29 | `omar_q02` | AV1 | omar | Where does Omar live? | Studio in Gulfton (moved from 1-bedroom in Alief to save $200/month) | #1 (bl=0.950) user is Omar<br>#2 (bl=0.931) Omar is the user<br>#3 (bl=0.912) assistant refers to user as Omar<br>#4 (bl=0.890) Khalid lives in Dallas<br>#5 (bl=0.865) user thinks Spring Branch is where the money is real estate wise | **E3-TRUE** | (a) Code change only |
| 30 | `omar_q05` | AV7 | omar | Is Omar planning to buy a car? | No — he wanted a used Toyota Camry but the car got sold before he could secure f | #1 (bl=0.950) user is Omar<br>#2 (bl=0.930) Omar is the user<br>#3 (bl=0.910) assistant refers to user as Omar<br>#4 (bl=0.903) user plans to hit AutoZone tomorrow<br>#5 (bl=0.849) Spring Branch is a good area to buy in Houston | **E3-TRUE** | (a) Code change only |
| 31 | `omar_q06` | AV6 | omar | Does Omar spend money on himself? | He said he never spends on himself, but he bought a PS5 for stress relief | #1 (bl=0.950) user is Omar<br>#2 (bl=0.931) user is sending money to Sudan, specifically Khartoum<br>#3 (bl=0.930) Omar is the user<br>#4 (bl=0.910) assistant refers to user as Omar<br>#5 (bl=0.909) user is sending money to user's mom | **E3-TRUE** | (a) Code change only |
| 32 | `omar_q11` | AV3 | omar | What are Omar's health conditions? | None explicitly mentioned (but he's tired and stressed from overwork) | #1 (bl=0.930) user is Omar<br>#2 (bl=0.911) couple of user's siblings send what they can to user's mom<br>#3 (bl=0.901) user and Amira are co-parents of Ismail<br>#4 (bl=0.870) Omar is the user<br>#5 (bl=0.866) user sends user's mom 300 400 every month depending on what user makes | **E1** | (d) Benchmark fix |
| 33 | `priya_q03` | AV3 | priya | What should I know about Priya's learning or work style? | She has ADHD (diagnosed in college) | #1 (bl=0.981) Priya works at Anthropic as referenced in conversation about her good year<br>#2 (bl=0.950) user's paper work is technically aligned with what they do on the memory team<br>#3 (bl=0.911) user is an ML engineer with mix of self-directed deep work and meetings<br>#4 (bl=0.865) mom would definitely know Hawkins<br>#5 (bl=0.830) user works with user's team on AI systems where philosophical topics like consci | **E3-TRUE** | (a) Code change only |
| 34 | `priya_q05` | AV7 | priya | Is Priya planning to get a pet? | No — she wanted a dog but landlord said no | #1 (bl=0.982) Priya works at Anthropic as referenced in conversation about her good year<br>#2 (bl=0.843) user advised to get a $15 alarm clock from Amazon<br>#3 (bl=0.828) user has been looking at golden retriever puppies seriously *[STALE]*<br>#4 (bl=0.767) user is planning to buy Hawkins Futura Hard Anodised kadai for mom's birthday<br>#5 (bl=0.765) user plans to get a rug from Ruggable as highest impact item with budget $150-40 | **AV7-CROSS** | (a) Code change only |
| 35 | `priya_q07` | AV1 | priya | What exercise does Priya do? | Rock climbing | #1 (bl=1.000) Priya works at Anthropic as referenced in conversation about her good year<br>#2 (bl=0.891) user's paper work is technically aligned with what they do on the memory team<br>#3 (bl=0.844) breeder does early neurological stimulation stuff on the puppies<br>#4 (bl=0.831) user was going to do shrimp in Japanese curry since user does not eat meat<br>#5 (bl=0.808) user describes hot yoga as the main thing keeping them sane and their lifeline *[STALE]* | **E2** | (b) Re-ingestion only |
| 36 | `priya_q09` | AV3 | priya | What health conditions should Priya's meal plan account for? | Chronic migraines (triggered by stress/screens). Also was told to get more omega | #1 (bl=0.911) user's doctor has been on user about skipping meals<br>#2 (bl=0.855) user plans to roast cauliflower in meal prep<br>#3 (bl=0.843) user plans to try meal prep this sunday<br>#4 (bl=0.835) user's paper work is technically aligned with what they do on the memory team<br>#5 (bl=0.822) user plans to roast sweet potato in meal prep | **E3-TRUE** | (a) Code change only |
| 37 | `tom_q02` | AV1 | tom | What car does Tom drive? | Hyundai Ioniq 5 (electric — sold his Land Rover Defender) | #1 (bl=1.000) user is addressed as Tom<br>#2 (bl=0.931) assistant addresses user as Tom<br>#3 (bl=0.925) Tom is living alone in the four-bedroom detached house<br>#4 (bl=0.902) Tom has accumulated rather a lot of things over forty years in the four-bedroom <br>#5 (bl=0.887) Tom lives in the Cotswolds pottering with bees | **E3-TRUE** | (a) Code change only |
| 38 | `tom_q03` | AV1 | tom | What does Tom do socially? | Walking group (stopped the pub quiz after Gerald couldn't make it) | #1 (bl=0.964) user is addressed as Tom<br>#2 (bl=0.950) assistant addresses user as Tom<br>#3 (bl=0.931) the garden was always user's wife's domain and she knew exactly what to do with <br>#4 (bl=0.908) Tom is living alone in the four-bedroom detached house<br>#5 (bl=0.900) Tom worked with David Hargreaves for years sharing offices site visits and terri | **E3-TRUE** | (a) Code change only |
| 39 | `tom_q04` | AV3 | tom | Does Tom have any health conditions? | Atrial fibrillation — takes warfarin (blood thinners) | #1 (bl=0.963) user is addressed as Tom<br>#2 (bl=0.953) Tom has not seen most colleagues including David Hargreaves since retiring three<br>#3 (bl=0.950) assistant addresses user as Tom<br>#4 (bl=0.899) Tom's hives are all at the out-apiaries<br>#5 (bl=0.889) Tom is living alone in the four-bedroom detached house | **E3-TRUE** | (a) Code change only |
| 40 | `tom_q06` | AV7 | tom | Is Tom planning any building projects? | No — he planned to convert a barn into a honey processing room but it was scrapp | #1 (bl=1.000) user is addressed as Tom<br>#2 (bl=0.960) Tom has accumulated rather a lot of things over forty years in the four-bedroom <br>#3 (bl=0.934) Tom worked with David Hargreaves for years sharing offices site visits and terri<br>#4 (bl=0.907) Tom is living alone in the four-bedroom detached house<br>#5 (bl=0.863) Tom worked with Patricia Okonkwo for years sharing offices site visits and terri | **E3-TRUE** | (a) Code change only |
| 41 | `tom_q07` | AV6 | tom | Does Tom use social media? | Inconsistent — he said he'd never use social media, but later posted a bee photo | #1 (bl=1.000) user is addressed as Tom<br>#2 (bl=0.930) Sophie encouraged user to set up online presence but Carrd suggested instead of <br>#3 (bl=0.915) Tom worked with David Hargreaves for years sharing offices site visits and terri<br>#4 (bl=0.910) assistant addresses user as Tom<br>#5 (bl=0.896) Tom has not seen most colleagues including David Hargreaves since retiring three | **E3-TRUE** | (a) Code change only |

## Table 2: Failures by Attack Vector

| Attack Vector | Total Qs | Failing | E1 | E2 | E3-TRUE | E3-WEAK | E4 | AV7-CROSS | RETRACTION | RANKING |
|---|---|---|---|---|---|---|---|---|---|---|
| AV1 superseded-preference | 14 | 9 | · | 1 | 7 | · | 1 | · | · | · |
| AV2 expired-logistics | 19 | 1 | · | · | 1 | · | · | · | · | · |
| AV3 stable-identity | 22 | 9 | 1 | · | 7 | 1 | · | · | · | · |
| AV4 multi-version-fact | 9 | 4 | · | · | 3 | · | · | · | · | 1 |
| AV5 broad-query | 8 | 3 | · | · | 1 | 1 | 1 | · | · | · |
| AV6 cross-session-contradiction | 8 | 3 | · | · | 2 | · | 1 | · | · | · |
| AV7 selective-forgetting | 8 | 8 | · | · | 5 | · | · | 1 | 2 | · |
| AV8 numeric-preservation | 16 | 3 | 1 | 1 | · | · | 1 | · | · | · |
| AV9 soft-supersession | 8 | 1 | · | · | 1 | · | · | · | · | · |
| **Total** | **112** | **41** | **2** | **2** | **27** | **2** | **4** | **1** | **2** | **1** |

## Table 3: Fixability Summary

| Fix Type | Count | % | Categories | Action |
|---|---|---|---|---|
| (a) Code change only | 33 | 80% | E4, E3-TRUE, RANKING, AV7-CROSS | Fix retrieval routing, ranking, or category logic |
| (b) Re-ingestion only | 4 | 10% | RETRACTION, E2 | Re-run extraction/ingestion pipeline to capture missing facts |
| (c) Both code + re-ingestion | 2 | 5% | E3-WEAK | Improve extraction for composite facts AND fix retrieval for multi-session answers |
| (d) Benchmark fix | 2 | 5% | E1 | Update benchmark ground truth to match actual conversation data |
| **Total** | **41** | **100%** | | |

## Table 4: Detailed Failure Reasons

### E1

**elena_q13** — elena | AV8_numeric_preservation
- **Q:** How much does Elena owe in student loans?
- **Expected:** $40,000
- **Reason:** Edge has 'about 42k' but question expects exact '$40,000' — benchmark/conversation mismatch
- **Expected fact:** Student loans -- question says $40,000, conversation may say 42k
- **Answer rank:** not found (pool size: 88)
- **Top retrieved:**
  - #1 (bl=0.975) user's student loans are all federal loans
  - #2 (bl=0.955) user has about 42k left on student loans
  - #3 (bl=0.916) user's student loans have interest rate around 5.5% or 6%

**omar_q11** — omar | AV3_stable_identity
- **Q:** What are Omar's health conditions?
- **Expected:** None explicitly mentioned (but he's tired and stressed from overwork)
- **Reason:** Correct answer is 'None explicitly mentioned' — no extractable fact possible
- **Expected fact:** No explicit health conditions -- tired/stressed from overwork
- **Answer rank:** not found (pool size: 69)
- **Top retrieved:**
  - #1 (bl=0.930) user is Omar
  - #2 (bl=0.911) couple of user's siblings send what they can to user's mom
  - #3 (bl=0.901) user and Amira are co-parents of Ismail

### E2

**amara_q14** — amara | AV8_numeric_preservation
- **Q:** How much would the UCL master's programme have cost Amara?
- **Expected:** £18,000
- **Reason:** UCL LLM cost £18,000 in session 11 — entity node references it but no answer edge extracted
- **Expected fact:** UCL LLM would have cost GBP 18,000
- **Answer rank:** not found (pool size: 134)
- **Top retrieved:**
  - #1 (bl=0.950) Amara has endometriosis
  - #2 (bl=0.912) Amara is a member of the Bar.
  - #3 (bl=0.874) Amara works at chambers.

**priya_q07** — priya | AV1_superseded_preference
- **Q:** What exercise does Priya do?
- **Expected:** Rock climbing
- **Reason:** Rock climbing fact in session 25 but no correct-answer edge extracted (only topic-level edges)
- **Expected fact:** Switched to rock climbing, quit hot yoga
- **Answer rank:** 10 (pool size: 128)
- **Top retrieved:**
  - #1 (bl=1.000) Priya works at Anthropic as referenced in conversation about her good year
  - #2 (bl=0.891) user's paper work is technically aligned with what they do on the memory team
  - #3 (bl=0.844) breeder does early neurological stimulation stuff on the puppies

### E3-TRUE

**amara_q03** — amara | AV1_superseded_preference
- **Q:** What exercise does Amara do?
- **Expected:** Boxing at a gym in Bethnal Green
- **Reason:** Edge 'user has started boxing at a gym in Bethnal Green' exists (7 enriched) — superseded preference not surfaced
- **Expected fact:** Dropped running, switched to boxing in Bethnal Green
- **Answer rank:** not found (pool size: 101)
- **Top retrieved:**
  - #1 (bl=0.950) Amara has endometriosis
  - #2 (bl=0.930) Amara is a member of the Bar.
  - #3 (bl=0.911) Amara works at chambers.

**amara_q08** — amara | AV9_soft_supersession
- **Q:** Where does Amara live?
- **Expected:** Hackney, East London — though she's been considering a move to Islington
- **Reason:** Edge referencing Hackney exists (10 enriched) — soft supersession fact about considering Islington buried
- **Expected fact:** Lives in Hackney, considering move to Islington
- **Answer rank:** not found (pool size: 128)
- **Top retrieved:**
  - #1 (bl=0.950) Amara is a member of the Bar.
  - #2 (bl=0.931) Amara has endometriosis
  - #3 (bl=0.912) Amara works at chambers.

**amara_q11** — amara | AV5_broad_query
- **Q:** What does Amara do outside of work?
- **Expected:** Boxing, Nigerian cooking experiments, podcasts, reading about justice
- **Reason:** Edge about boxing in Bethnal Green exists (4 enriched) — broad query fails to aggregate hobbies
- **Expected fact:** Boxing, Nigerian cooking, podcasts, reading about justice
- **Answer rank:** not found (pool size: 139)
- **Top retrieved:**
  - #1 (bl=0.950) Amara is a member of the Bar.
  - #2 (bl=0.931) Amara works at chambers.
  - #3 (bl=0.911) Amara has endometriosis

**amara_q12** — amara | AV3_stable_identity
- **Q:** What is Amara's educational background?
- **Expected:** LLB from King's College London
- **Reason:** Edge 'user did LLB at King's' exists (1 enriched) — stable identity fact not retrieved
- **Expected fact:** LLB from King's College London
- **Answer rank:** not found (pool size: 107)
- **Top retrieved:**
  - #1 (bl=0.950) Amara's senior clerk is quite good
  - #2 (bl=0.931) Amara is a member of the Bar.
  - #3 (bl=0.912) Amara's brother is Chidi who is a GP

**david_q01** — david | AV4_multi_version_fact
- **Q:** What subject does David teach?
- **Expected:** AP US History (switched from AP European History)
- **Reason:** Edge about switching to AP US History exists (5 enriched) — multi-version fact not surfaced
- **Expected fact:** Switching from AP Euro to AP US History
- **Answer rank:** not found (pool size: 138)
- **Top retrieved:**
  - #1 (bl=0.974) user is David
  - #2 (bl=0.938) David lives in the Pacific Northwest
  - #3 (bl=0.933) David's student is a junior with strong reading skills.

**david_q03** — david | AV4_multi_version_fact
- **Q:** What car does David drive?
- **Expected:** Subaru Outback
- **Reason:** Edge 'drove home in a Subaru Outback' exists (10 enriched) — vehicle supersession not surfaced
- **Expected fact:** Bought Subaru Outback, replacing old Camry
- **Answer rank:** not found (pool size: 119)
- **Top retrieved:**
  - #1 (bl=0.974) user is David
  - #2 (bl=0.956) David lives in the Pacific Northwest
  - #3 (bl=0.924) David is considering entering the March invitational.

**david_q05** — david | AV3_stable_identity
- **Q:** What health conditions does David have?
- **Expected:** Mild hearing loss in left ear (wears hearing aid)
- **Reason:** Edge 'user wears a hearing aid, mild hearing loss in left ear' exists (1 enriched) — stable fact buried
- **Expected fact:** Mild hearing loss in left ear, wears hearing aid
- **Answer rank:** not found (pool size: 87)
- **Top retrieved:**
  - #1 (bl=0.974) David lives in the Pacific Northwest
  - #2 (bl=0.938) user is David
  - #3 (bl=0.911) David is considering entering the March invitational.

**david_q11** — david | AV2_expired_logistics
- **Q:** Does David have any parent events coming up?
- **Expected:** No (parent-teacher conferences were March 12, already passed)
- **Reason:** Edge about parent-teacher conferences exists (4 enriched) — expired logistics edge not retrieved
- **Expected fact:** Parent-teacher conferences March 12 (expired)
- **Answer rank:** not found (pool size: 121)
- **Top retrieved:**
  - #1 (bl=0.969) David is already dreading Teacher Appreciation Week coming up at his school
  - #2 (bl=0.897) David is considering entering the March invitational.
  - #3 (bl=0.881) user is David

**elena_q01** — elena | AV4_multi_version_fact
- **Q:** Where does Elena work?
- **Expected:** Rush University Medical Center
- **Reason:** Edge 'user is employed at Rush University Medical Center' exists (1 enriched) — not retrieved for workplace query
- **Expected fact:** Transferred from Lurie to Rush University Medical Center
- **Answer rank:** not found (pool size: 85)
- **Top retrieved:**
  - #1 (bl=0.985) Elena works night shifts with her coworkers at the hospital and does 12-hour shifts
  - #2 (bl=0.951) Elena is little Elena from the west side.
  - #3 (bl=0.910) Elena is the user

**elena_q02** — elena | AV1_superseded_preference
- **Q:** What diet is Elena following?
- **Expected:** Mediterranean diet
- **Reason:** Edge 'doctor told user to switch to Mediterranean diet' exists (2 enriched) — retrieval gap
- **Expected fact:** Quit keto, switched to Mediterranean diet
- **Answer rank:** not found (pool size: 100)
- **Top retrieved:**
  - #1 (bl=0.950) Elena is the user
  - #2 (bl=0.934) Elena is little Elena from the west side.
  - #3 (bl=0.931) user is referred to as Elena

**elena_q03** — elena | AV7_selective_forgetting
- **Q:** Is Elena planning to get a nurse practitioner degree?
- **Expected:** No — she dropped that plan due to student loan debt
- **Reason:** Edge about dropping NP plan exists (9 enriched) — retraction edge not surfaced despite keyword match
- **Expected fact:** Dropped NP plan -- can't afford it with student loans
- **Answer rank:** not found (pool size: 88)
- **Top retrieved:**
  - #1 (bl=0.931) Jasmine stayed in nursing school because of watching nurses like Elena
  - #2 (bl=0.913) Elena works night shifts with her coworkers at the hospital and does 12-hour shifts
  - #3 (bl=0.912) user is referred to as Elena

**elena_q04** — elena | AV3_stable_identity
- **Q:** What mental health conditions does Elena have?
- **Expected:** Generalized anxiety disorder (takes sertraline)
- **Reason:** Edge 'user is on sertraline for anxiety' exists (3 enriched) — stable identity fact buried
- **Expected fact:** Generalized anxiety disorder, takes sertraline
- **Answer rank:** not found (pool size: 83)
- **Top retrieved:**
  - #1 (bl=0.985) Elena works night shifts with her coworkers at the hospital and does 12-hour shifts
  - #2 (bl=0.933) Elena is little Elena from the west side.
  - #3 (bl=0.931) user is referred to as Elena

**jake_q03** — jake | AV3_stable_identity
- **Q:** What does Jake's mom do?
- **Expected:** Waitresses at the Eire Pub in Dorchester — been there 20 years
- **Reason:** Edge about mom at Eire Pub in Dorchester exists (8 enriched) — family background fact buried
- **Expected fact:** Mom waitresses at Eire Pub in Dorchester, 20 years
- **Answer rank:** not found (pool size: 82)
- **Top retrieved:**
  - #1 (bl=0.982) user is identified as Jake on the suggested birthday card for user's mom
  - #2 (bl=0.937) Bridget is the daughter of user's mom
  - #3 (bl=0.933) this dude asked user what he does and said 'oh so you didnt go to college huh' when user said he is 

**jake_q05** — jake | AV7_selective_forgetting
- **Q:** Is Jake doing any side work?
- **Expected:** No — he wanted to start installing EV home chargers on the side but his dad said he needs his journeyman license first and the insurance is too expensive
- **Reason:** Edge about EV charger side gig exists (7 enriched) — retraction not surfaced by retrieval
- **Expected fact:** EV charger side gig dead -- dad said not ready, insurance too expensive
- **Answer rank:** not found (pool size: 125)
- **Top retrieved:**
  - #1 (bl=0.963) user is identified as Jake on the suggested birthday card for user's mom
  - #2 (bl=0.963) user got out of high school and started doing electrical work
  - #3 (bl=0.950) user introduced himself as Jake in the text message to Mr. Patterson

**jake_q07** — jake | AV1_superseded_preference
- **Q:** What does Jake drink?
- **Expected:** Craft beer, especially Trillium IPAs
- **Reason:** Edge 'user was dragged to Trillium' exists (2 enriched) — craft beer preference not surfaced
- **Expected fact:** Got into craft beer -- favorites from Trillium Brewing
- **Answer rank:** not found (pool size: 116)
- **Top retrieved:**
  - #1 (bl=0.963) user is identified as Jake on the suggested birthday card for user's mom
  - #2 (bl=0.950) user introduced himself as Jake in the text message to Mr. Patterson
  - #3 (bl=0.915) this dude asked user what he does and said 'oh so you didnt go to college huh' when user said he is 

**marcus_q03** — marcus | AV7_selective_forgetting
- **Q:** Is Marcus planning to open a second shop location?
- **Expected:** No — he considered Germantown but scrapped it (too much financial risk)
- **Reason:** Edge about scrapping Germantown location exists (7 enriched) — not surfaced by any retrieval strategy
- **Expected fact:** Scrapped Germantown second location -- too much financial risk
- **Answer rank:** not found (pool size: 105)
- **Top retrieved:**
  - #1 (bl=0.876) second shop already has 3 bays and a lift
  - #2 (bl=0.847) shop owner has good people working with him
  - #3 (bl=0.821) user plans to put up security cameras at the shop

**marcus_q06** — marcus | AV1_superseded_preference
- **Q:** What does Marcus do for fun?
- **Expected:** Fishing on weekends
- **Reason:** Edge 'user has been going to Sardis Lake for fishing' exists (1 enriched) — category routing misses it
- **Expected fact:** Stopped poker, started fishing on weekends
- **Answer rank:** not found (pool size: 79)
- **Top retrieved:**
  - #1 (bl=0.950) Robert Chandler mentioned getting a lawyer in the voicemail to Marcus Thompson
  - #2 (bl=0.913) user should have a brief written record of today's incident with the customer including date, time, 
  - #3 (bl=0.870) Reggie is one of user's guys and is pretty organized

**omar_q02** — omar | AV1_superseded_preference
- **Q:** Where does Omar live?
- **Expected:** Studio in Gulfton (moved from 1-bedroom in Alief to save $200/month)
- **Reason:** Edge 'user just moved to Gulfton' exists (2 enriched) — address supersession not surfaced
- **Expected fact:** Moved from Alief 1BR to Gulfton studio, saves $200/mo
- **Answer rank:** not found (pool size: 89)
- **Top retrieved:**
  - #1 (bl=0.950) user is Omar
  - #2 (bl=0.931) Omar is the user
  - #3 (bl=0.912) assistant refers to user as Omar

**omar_q05** — omar | AV7_selective_forgetting
- **Q:** Is Omar planning to buy a car?
- **Expected:** No — he wanted a used Toyota Camry but the car got sold before he could secure financing
- **Reason:** Edge 'the whole Camry plan is dead' exists (1 enriched) — retraction edge not surfaced
- **Expected fact:** Camry plan dead -- car sold before financing secured
- **Answer rank:** not found (pool size: 104)
- **Top retrieved:**
  - #1 (bl=0.950) user is Omar
  - #2 (bl=0.930) Omar is the user
  - #3 (bl=0.910) assistant refers to user as Omar

**omar_q06** — omar | AV6_cross_session_contradiction
- **Q:** Does Omar spend money on himself?
- **Expected:** He said he never spends on himself, but he bought a PS5 for stress relief
- **Reason:** Edge 'user got PS5 off Facebook Marketplace' exists (1 enriched) — cross-session contradiction buried
- **Expected fact:** Said never spends on himself but bought PS5 for stress
- **Answer rank:** not found (pool size: 108)
- **Top retrieved:**
  - #1 (bl=0.950) user is Omar
  - #2 (bl=0.931) user is sending money to Sudan, specifically Khartoum
  - #3 (bl=0.930) Omar is the user

**priya_q03** — priya | AV3_stable_identity
- **Q:** What should I know about Priya's learning or work style?
- **Expected:** She has ADHD (diagnosed in college)
- **Reason:** Edge 'user received ADHD diagnosis since college' exists (2 enriched) — semantic search misses it
- **Expected fact:** Has ADHD, diagnosed in college
- **Answer rank:** not found (pool size: 127)
- **Top retrieved:**
  - #1 (bl=0.981) Priya works at Anthropic as referenced in conversation about her good year
  - #2 (bl=0.950) user's paper work is technically aligned with what they do on the memory team
  - #3 (bl=0.911) user is an ML engineer with mix of self-directed deep work and meetings

**priya_q09** — priya | AV3_stable_identity
- **Q:** What health conditions should Priya's meal plan account for?
- **Expected:** Chronic migraines (triggered by stress/screens). Also was told to get more omega-3s.
- **Reason:** Edge 'user has chronic migraines' exists (9 enriched) — retrieval misses HEALTH_WELLBEING edges
- **Expected fact:** Chronic migraines triggered by stress/screens; needs omega-3
- **Answer rank:** not found (pool size: 153)
- **Top retrieved:**
  - #1 (bl=0.911) user's doctor has been on user about skipping meals
  - #2 (bl=0.855) user plans to roast cauliflower in meal prep
  - #3 (bl=0.843) user plans to try meal prep this sunday

**tom_q02** — tom | AV1_superseded_preference
- **Q:** What car does Tom drive?
- **Expected:** Hyundai Ioniq 5 (electric — sold his Land Rover Defender)
- **Reason:** Edge 'user has just bought a Hyundai Ioniq 5' exists (3 enriched) — vehicle supersession not surfaced
- **Expected fact:** Sold Land Rover Defender, bought Hyundai Ioniq 5
- **Answer rank:** not found (pool size: 141)
- **Top retrieved:**
  - #1 (bl=1.000) user is addressed as Tom
  - #2 (bl=0.931) assistant addresses user as Tom
  - #3 (bl=0.925) Tom is living alone in the four-bedroom detached house

**tom_q03** — tom | AV1_superseded_preference
- **Q:** What does Tom do socially?
- **Expected:** Walking group (stopped the pub quiz after Gerald couldn't make it)
- **Reason:** Edge 'user has recently joined a walking group' exists (11 enriched) — hobby supersession not surfaced
- **Expected fact:** Stopped pub quiz at The Fox, joined walking group
- **Answer rank:** not found (pool size: 153)
- **Top retrieved:**
  - #1 (bl=0.964) user is addressed as Tom
  - #2 (bl=0.950) assistant addresses user as Tom
  - #3 (bl=0.931) the garden was always user's wife's domain and she knew exactly what to do with each rose

**tom_q04** — tom | AV3_stable_identity
- **Q:** Does Tom have any health conditions?
- **Expected:** Atrial fibrillation — takes warfarin (blood thinners)
- **Reason:** Edge 'user has atrial fibrillation' exists (3 enriched) — stable health fact buried in noise
- **Expected fact:** Atrial fibrillation, takes warfarin
- **Answer rank:** not found (pool size: 89)
- **Top retrieved:**
  - #1 (bl=0.963) user is addressed as Tom
  - #2 (bl=0.953) Tom has not seen most colleagues including David Hargreaves since retiring three years ago
  - #3 (bl=0.950) assistant addresses user as Tom

**tom_q06** — tom | AV7_selective_forgetting
- **Q:** Is Tom planning any building projects?
- **Expected:** No — he planned to convert a barn into a honey processing room but it was scrapped (planning permission denied, too expensive)
- **Reason:** Edge about barn conversion planning permission exists (4 enriched) — retraction not surfaced
- **Expected fact:** Barn conversion scrapped -- planning permission denied, too expensive
- **Answer rank:** not found (pool size: 95)
- **Top retrieved:**
  - #1 (bl=1.000) user is addressed as Tom
  - #2 (bl=0.960) Tom has accumulated rather a lot of things over forty years in the four-bedroom detached house
  - #3 (bl=0.934) Tom worked with David Hargreaves for years sharing offices site visits and terrible canteen coffee

**tom_q07** — tom | AV6_cross_session_contradiction
- **Q:** Does Tom use social media?
- **Expected:** Inconsistent — he said he'd never use social media, but later posted a bee photo on Instagram that got 200 likes
- **Reason:** Edges about 'no social media' AND 'Instagram bee photo' exist (9 enriched) — contradiction not surfaced
- **Expected fact:** Said no social media, but posted bee photo on Instagram (200 likes)
- **Answer rank:** not found (pool size: 133)
- **Top retrieved:**
  - #1 (bl=1.000) user is addressed as Tom
  - #2 (bl=0.930) Sophie encouraged user to set up online presence but Carrd suggested instead of social media
  - #3 (bl=0.915) Tom worked with David Hargreaves for years sharing offices site visits and terrible canteen coffee

### E3-WEAK

**elena_q08** — elena | AV3_stable_identity
- **Q:** What's Elena's family background?
- **Expected:** Mexican-American, parents from Guadalajara, first-gen college grad. Sister Sofia has Down syndrome.
- **Reason:** Composite fact across 3 sessions (Guadalajara, first-gen, Sofia/Down syndrome) — partial edges exist but incomplete
- **Expected fact:** Parents from Guadalajara, first-gen college grad, sister Sofia has Down syndrome
- **Answer rank:** not found (pool size: 83)
- **Top retrieved:**
  - #1 (bl=0.973) Elena's mom screamed and cried upon learning of Elena's engagement.
  - #2 (bl=0.938) Elena's mom already has opinions on the wedding.
  - #3 (bl=0.931) user is referred to as Elena

**jake_q10** — jake | AV5_broad_query
- **Q:** What does Jake do outside of work?
- **Expected:** Rec hockey on Wednesdays, gaming (new PC), craft beer, sports with the boys
- **Reason:** Composite fact across 3 sessions (rec hockey, gaming PC, craft beer) — partial edges exist but incomplete
- **Expected fact:** Rec hockey Wednesdays, gaming PC, craft beer, sports with the boys
- **Answer rank:** not found (pool size: 152)
- **Top retrieved:**
  - #1 (bl=0.963) user is identified as Jake on the suggested birthday card for user's mom
  - #2 (bl=0.957) Danny is one of the boys with user planning the lake trip
  - #3 (bl=0.950) user introduced himself as Jake in the text message to Mr. Patterson

### E4

**amara_q02** — amara | AV1_superseded_preference
- **Q:** What phone does Amara use?
- **Expected:** Samsung Galaxy S25 (switched from iPhone to Android)
- **Reason:** Samsung Galaxy S25 edge in {FINANCIAL_MATERIAL} but question routes to {PREFERENCES_HABITS}
- **Expected fact:** Switched from iPhone 14 Pro to Samsung Galaxy S25
- **Answer rank:** not found (pool size: 166)
- **Top retrieved:**
  - #1 (bl=0.950) Amara works at chambers.
  - #2 (bl=0.912) Amara is a member of the Bar.
  - #3 (bl=0.876) user does not need phone buzzing for round-robin emails about the chambers Christmas party

**amara_q07** — amara | AV6_cross_session_contradiction
- **Q:** Does Amara drink alcohol?
- **Expected:** Unclear — she said she hadn't drunk in two years, but later mentioned having wine at a chambers dinner
- **Reason:** Chambers dinner wine edge in {LOGISTICAL_CONTEXT} but contradiction question needs {PREFERENCES_HABITS, HEALTH_WELLBEING}
- **Expected fact:** Said no alcohol 2 years, but had wine at chambers dinner
- **Answer rank:** not found (pool size: 107)
- **Top retrieved:**
  - #1 (bl=0.950) Amara is a member of the Bar.
  - #2 (bl=0.923) Amara has endometriosis
  - #3 (bl=0.896) Amara works at chambers.

**elena_q12** — elena | AV5_broad_query
- **Q:** What does Elena do for stress relief?
- **Expected:** Learning to knit, Mediterranean diet, trying anxiety management techniques
- **Reason:** Knitting edge in {PREFERENCES_HABITS} but broad query routes to {HOBBIES_RECREATION, HEALTH_WELLBEING}
- **Expected fact:** Knitting, Mediterranean diet, anxiety management for stress relief
- **Answer rank:** not found (pool size: 108)
- **Top retrieved:**
  - #1 (bl=0.985) Elena works night shifts with her coworkers at the hospital and does 12-hour shifts
  - #2 (bl=0.962) Diana got a couples massage gift certificate for Elena and Marco
  - #3 (bl=0.937) Chicago is home for Elena.

**marcus_q13** — marcus | AV8_numeric_preservation
- **Q:** How many employees does Marcus currently have at the shop?
- **Expected:** 4 (recently hired Carlos as the fourth employee)
- **Reason:** Carlos hiring edge in {IDENTITY_SELF_CONCEPT, RELATIONAL_BONDS} but question expects {PROJECTS_ENDEAVORS}
- **Expected fact:** Hired Carlos as 4th employee -- business growing
- **Answer rank:** not found (pool size: 138)
- **Top retrieved:**
  - #1 (bl=0.931) Marcus Thompson's shop used quality name-brand parts for Robert Chandler's brake job
  - #2 (bl=0.926) shop owner earned business success with how he treats customers
  - #3 (bl=0.911) user works at the shop

### AV7-CROSS

**priya_q05** — priya | AV7_selective_forgetting
- **Q:** Is Priya planning to get a pet?
- **Expected:** No — she wanted a dog but landlord said no
- **Reason:** Retraction edges in {RELATIONAL_BONDS, FINANCIAL_MATERIAL} but query routes to {PREFERENCES_HABITS, LOGISTICAL_CONTEXT}
- **Expected fact:** Wanted dog but landlord said no -- retraction
- **Answer rank:** 6 (pool size: 157)
- **Top retrieved:**
  - #1 (bl=0.982) Priya works at Anthropic as referenced in conversation about her good year
  - #2 (bl=0.843) user advised to get a $15 alarm clock from Amazon
  - #3 (bl=0.828) user has been looking at golden retriever puppies seriously *[STALE]*

### RETRACTION

**amara_q06** — amara | AV7_selective_forgetting
- **Q:** Is Amara pursuing any further education?
- **Expected:** No — she considered a human rights LLM at UCL but dropped the idea
- **Reason:** Dropped UCL human rights LLM in sessions 11/17 — retraction event never extracted
- **Expected fact:** Dropped UCL human rights LLM -- not worth time/money
- **Answer rank:** not found (pool size: 127)
- **Top retrieved:**
  - #1 (bl=0.950) Amara is a member of the Bar.
  - #2 (bl=0.931) Amara works at chambers.
  - #3 (bl=0.916) Amara ended up chasing CPS for materials in previous case with instructing solicitor

**david_q04** — david | AV7_selective_forgetting
- **Q:** Is David writing a book?
- **Expected:** No — he abandoned the primary sources book project
- **Reason:** Book abandonment ('nobody would read it') in sessions 9/18 — retraction never extracted as edge
- **Expected fact:** Abandoned book about teaching through primary sources
- **Answer rank:** not found (pool size: 124)
- **Top retrieved:**
  - #1 (bl=0.974) user is David
  - #2 (bl=0.956) David lives in the Pacific Northwest
  - #3 (bl=0.951) David's student is a junior with strong reading skills.

### RANKING

**omar_q01** — omar | AV4_multi_version_fact
- **Q:** What rideshare platform does Omar drive for?
- **Expected:** Uber (was on Uber, switched to Lyft, then back to Uber because Lyft had too few rides in Houston)
- **Reason:** Correct 'Uber pays the bills' edge at rank 4; stale 'driving for Lyft' edge at rank 2 outranks it
- **Expected fact:** Uber -> Lyft -> back to Uber (triple version)
- **Answer rank:** 4 (pool size: 109)
- **Top retrieved:**
  - #1 (bl=0.950) user is Omar
  - #2 (bl=0.934) user has been driving for Lyft *[STALE]*
  - #3 (bl=0.911) Omar is the user

## Per-Persona Summary

| Persona | Failing | E1 | E2 | E3-TRUE | E3-WEAK | E4 | AV7-CROSS | RETRACTION | RANKING |
|---|---|---|---|---|---|---|---|---|---|
| amara | 8 | · | 1 | 4 | · | 2 | · | 1 | · |
| david | 5 | · | · | 4 | · | · | · | 1 | · |
| elena | 7 | 1 | · | 4 | 1 | 1 | · | · | · |
| jake | 4 | · | · | 3 | 1 | · | · | · | · |
| marcus | 3 | · | · | 2 | · | 1 | · | · | · |
| omar | 5 | 1 | · | 3 | · | · | · | · | 1 |
| priya | 4 | · | 1 | 2 | · | · | 1 | · | · |
| tom | 5 | · | · | 5 | · | · | · | · | · |
