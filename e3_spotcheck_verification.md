# E3 Spot-Check Verification

## Summary

- Total E3 reviewed: **34**
- TRUE retrieval failures (keep E3): **18**
- Related but insufficient (E3-WEAK): **15**
- Misclassified extraction gaps (change to E2): **1**
- Non-TRUE E3 total (E3-WEAK + E2): **16**

## Verification Table

| question_id | classification | nearest_edge_summary | would_edge_answer_question | revised_classification | notes |
|---|---|---|---|---|---|
| amara_q01 | E3 | Amara Okafor joins 7 Bedford Row as a senior tenant | YES | keep E3 | Edge directly states she joined 7 Bedford Row as senior tenant. |
| amara_q03 | E3 | user has started boxing at a gym in Bethnal Green with classes on Monday, Wednesday, and Friday even | YES | keep E3 | Edge directly states boxing at a gym in Bethnal Green. |
| amara_q08 | E3 | user has an early train from Hackney Central into the Temple in the morning | NO | E3-WEAK | Edge implies Hackney context but does not cleanly encode full answer state including consideration of Islington move. |
| amara_q11 | E3 | user has started boxing at a gym in Bethnal Green with classes on Monday, Wednesday, and Friday even | NO | E3-WEAK | Edge covers only boxing; question expects a broader multi-activity profile. |
| amara_q12 | E3 | user did LLB at King's | YES | keep E3 | Edge directly states LLB at King's. |
| david_q01 | E3 | department head pulled user aside last week to ask if user would take over the AP US History section | NO | E3-WEAK | Edge says he was asked to take over AP US History, not clearly that this is the settled current assignment. |
| david_q03 | E3 | user went to the dealership on Saturday and drove home in a Subaru Outback | YES | keep E3 | Edge explicitly states he drove home in a Subaru Outback. |
| david_q05 | E3 | user wears a hearing aid due to mild hearing loss in left ear | YES | keep E3 | Edge directly states mild left-ear hearing loss and hearing aid use. |
| david_q11 | E3 | David's principal is supportive and has offered to sit in on the parent-teacher conference for Stude | NO | E3-WEAK | Edge references conferences but does not encode the key temporal conclusion (already passed, so no upcoming events). |
| elena_q01 | E3 | user is employed at Rush University Medical Center | YES | keep E3 | Edge directly states current employer (Rush University Medical Center). |
| elena_q02 | E3 | user's doctor told user to switch to Mediterranean diet | YES | keep E3 | Edge directly states switch to Mediterranean diet. |
| elena_q03 | E3 | user is looking at DePaul's DNP program with a pediatric NP track | NO | E3-WEAK | Edge states NP/DNP interest only; missing dropped-plan and debt-based cancellation. |
| elena_q04 | E3 | user is on sertraline for anxiety and it usually helps but lately can't keep up especially with nigh | YES | keep E3 | Edge contains anxiety + sertraline, which answers the question intent. |
| jake_q02 | E3 | Kayla is a nurse at Mass General | NO | E3-WEAK | Edge identifies Kayla and her job but does not explicitly assert dating status or breakup transition. |
| jake_q03 | E3 | Megan lives in an apartment in Dorchester | NO | change to E2 | Nearest edge is about Megan in Dorchester and does not answer mother occupation at all (keyword-only location overlap). |
| jake_q05 | E3 | user is thinking worst case he could do EV charger installations through dads company on the side to | NO | E3-WEAK | Edge captures tentative EV side-work idea, not the final "no" state with license/insurance blockers. |
| jake_q07 | E3 | user was dragged to Trillium by Marcus like a month ago. | NO | E3-WEAK | Edge references Trillium visit but not explicit drink preference (craft beer/IPAs) as answer statement. |
| marcus_q02 | E3 | user bought a 2024 Ram 1500 Big Horn trim in granite crystal color with 5.7 Hemi | YES | keep E3 | Edge explicitly contains the exact vehicle answer (2024 Ram 1500). |
| marcus_q03 | E3 | user has been looking at a spot in Germantown for a second shop | NO | E3-WEAK | Edge reflects exploratory intent (looking at Germantown) but not the final retraction/scrapped decision. |
| marcus_q06 | E3 | user has been going to Sardis Lake mostly for fishing, which is about an hour south | YES | keep E3 | Edge directly states fishing as current leisure activity. |
| omar_q02 | E3 | user just moved to Gulfton | YES | keep E3 | Edge directly states moved to Gulfton, which answers the location question core. |
| omar_q05 | E3 | the whole Camry plan is dead for now | YES | keep E3 | Edge directly encodes the key conclusion that Camry purchase plan is dead. |
| omar_q06 | E3 | user got PS5 off Facebook Marketplace with 2 controllers and FIFA for 350 | NO | E3-WEAK | Edge captures PS5 spending event but misses the cross-session inconsistency framing from the benchmark answer. |
| omar_q12 | E3 | user is studying for real estate license between rides | NO | E3-WEAK | Edge is pre-outcome (studying), missing passed exam + Century 21 start details. |
| omar_q14 | E3 | user sends user's mom 300 400 every month depending on what user makes | YES | keep E3 | Edge directly gives $300-400 monthly remittance amount. |
| priya_q01 | E3 | user started eating fish again a few weeks ago | NO | E3-WEAK | Edge captures fish reintroduction but not the explicit dietary label/constraint framing (pescatarian). |
| priya_q03 | E3 | user received ADHD diagnosis since college | YES | keep E3 | Edge directly states ADHD diagnosis since college. |
| priya_q09 | E3 | user has chronic migraines that occur pretty regularly | NO | E3-WEAK | Edge has chronic migraines but omits key qualifiers (stress/screens triggers, omega-3 guidance). |
| tom_q02 | E3 | user has just bought a Hyundai Ioniq 5 | YES | keep E3 | Edge directly states Hyundai Ioniq 5 purchase/current vehicle. |
| tom_q03 | E3 | user has recently joined a walking group | YES | keep E3 | Edge directly states current walking-group social activity. |
| tom_q04 | E3 | user has atrial fibrillation | YES | keep E3 | Edge directly states atrial fibrillation; this answers the health-condition question. |
| tom_q06 | E3 | user needs to check with Cotswold District Council for planning permission to convert the stone barn | NO | E3-WEAK | Edge describes planning-permission step, not the final scrapped/denied/too-expensive outcome. |
| tom_q07 | E3 | user has no interest in Facebook or Instagram whatsoever | NO | E3-WEAK | Edge contains only "no social media" side, missing later Instagram contradiction and therefore not fully correct. |
| tom_q11 | E3 | user spent thirty-five years at Arup in civil engineering | YES | keep E3 | Edge directly states career at Arup in civil engineering for 35 years. |
