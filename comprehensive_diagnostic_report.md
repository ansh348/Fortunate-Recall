# Comprehensive Dry-Run Diagnostic Report

*Generated: 2026-03-04 19:32 UTC | Runtime: 507s*

**Current:** 71/112 passing (63.4%), 41 failing

## Phase 1: Baseline Mapping

- **Total failing questions:** 41
- **Correct edge IN pool:** 31
- **Correct edge NOT in pool:** 10

### Classification Breakdown

| Classification | Count | In Pool | Not In Pool |
|---|---|---|---|
| AV7-CROSS | 1 | 1 | 0 |
| E1 | 2 | 1 | 1 |
| E2 | 2 | 0 | 2 |
| E3-TRUE | 27 | 23 | 4 |
| E3-WEAK | 2 | 2 | 0 |
| E4 | 4 | 3 | 1 |
| RANKING | 1 | 1 | 0 |
| RETRACTION | 2 | 0 | 2 |

### Per-Question Baseline

| # | QID | AV | Class | Correct? | Category | Cosine | Act | Sem | Blended | Rank | Pool | Top1 bl | Gap |
|---|-----|----|----|---------|----------|--------|-----|-----|---------|------|------|---------|-----|
| 1 | amara_q02 | AV1 | E4 | Y | PREFEREN | 0.354 | 0.31 | 0.350 | 0.343 | 64 | 166 | 0.950 | 0.607 |
| 2 | amara_q03 | AV1 | E3-TRUE | Y | HOBBIES_ | 0.311 | 0.69 | 0.118 | 0.406 | 29 | 101 | 0.950 | 0.544 |
| 3 | amara_q06 | AV7 | RETRACTION | N | — | — | — | — | — | — | 127 | 0.950 | — |
| 4 | amara_q07 | AV6 | E4 | Y | LOGISTIC | 0.175 | — | — | — | — | 112 | 0.950 | — |
| 5 | amara_q08 | AV9 | E3-TRUE | Y | LOGISTIC | 0.287 | 0.54 | 0.406 | 0.419 | 39 | 128 | 0.950 | 0.531 |
| 6 | amara_q11 | AV5 | E3-TRUE | Y | HOBBIES_ | 0.208 | 0.64 | 0.191 | 0.236 | 62 | 144 | 0.950 | 0.714 |
| 7 | amara_q12 | AV3 | E3-TRUE | Y | IDENTITY | 0.209 | — | — | — | — | 107 | 0.950 | — |
| 8 | amara_q14 | AV8 | E2 | N | — | — | — | — | — | — | 129 | 0.950 | — |
| 9 | david_q01 | AV4 | E3-TRUE | Y | OBLIGATI | 0.300 | 0.60 | 0.286 | 0.317 | 65 | 138 | 0.974 | 0.658 |
| 10 | david_q03 | AV4 | E3-TRUE | Y | FINANCIA | 0.382 | 0.54 | 0.666 | 0.660 | 17 | 119 | 0.974 | 0.314 |
| 11 | david_q04 | AV7 | RETRACTION | N | — | — | — | — | — | — | 124 | 0.974 | — |
| 12 | david_q05 | AV3 | E3-TRUE | Y | HEALTH_W | 0.151 | 0.00 | 0.134 | 0.121 | 76 | 87 | 0.974 | 0.854 |
| 13 | david_q11 | AV2 | E3-TRUE | Y | RELATION | 0.613 | 0.53 | 1.000 | 0.835 | 5 | 121 | 0.969 | 0.134 |
| 14 | elena_q01 | AV4 | E3-TRUE | Y | IDENTITY | 0.286 | — | — | — | — | 85 | 0.985 | — |
| 15 | elena_q02 | AV1 | E3-TRUE | Y | HEALTH_W | 0.431 | 0.28 | 0.801 | 0.749 | 11 | 100 | 0.950 | 0.201 |
| 16 | elena_q03 | AV7 | E3-TRUE | Y | INTELLEC | 0.544 | 0.04 | 0.800 | 0.686 | 17 | 88 | 0.931 | 0.245 |
| 17 | elena_q04 | AV3 | E3-TRUE | Y | HEALTH_W | 0.285 | 0.12 | 0.284 | 0.267 | 54 | 84 | 0.985 | 0.718 |
| 18 | elena_q08 | AV3 | E3-WEAK | Y | RELATION | 0.425 | 0.24 | 0.653 | 0.612 | 20 | 67 | 0.973 | 0.361 |
| 19 | elena_q12 | AV5 | E4 | Y | HEALTH_W | 0.184 | 0.08 | 0.189 | 0.179 | 90 | 108 | 0.985 | 0.807 |
| 20 | elena_q13 | AV8 | E1 | Y | FINANCIA | 0.546 | 0.49 | 0.980 | 0.956 | 2 | 125 | 0.975 | 0.019 |
| 21 | jake_q03 | AV3 | E3-TRUE | Y | RELATION | 0.373 | 0.23 | 0.857 | 0.794 | 11 | 82 | 0.982 | 0.188 |
| 22 | jake_q05 | AV7 | E3-TRUE | Y | PROJECTS | 0.311 | 0.36 | 0.571 | 0.550 | 26 | 125 | 0.963 | 0.413 |
| 23 | jake_q07 | AV1 | E3-TRUE | Y | LOGISTIC | 0.284 | 0.53 | 0.653 | 0.629 | 17 | 116 | 0.963 | 0.334 |
| 24 | jake_q10 | AV5 | E3-WEAK | Y | HOBBIES_ | 0.287 | 0.27 | 0.257 | 0.259 | 80 | 152 | 0.963 | 0.704 |
| 25 | marcus_q03 | AV7 | E3-TRUE | Y | PROJECTS | 0.520 | — | — | — | — | 106 | 0.876 | — |
| 26 | marcus_q06 | AV1 | E3-TRUE | Y | HOBBIES_ | 0.200 | 0.43 | 0.183 | 0.234 | 39 | 79 | 0.950 | 0.716 |
| 27 | marcus_q13 | AV8 | E4 | Y | RELATION | 0.407 | 0.40 | 0.397 | 0.397 | 34 | 139 | 0.931 | 0.534 |
| 28 | omar_q01 | AV4 | RANKING | Y | IDENTITY | 0.523 | 0.56 | 0.917 | 0.899 | 3 | 123 | 0.950 | 0.051 |
| 29 | omar_q02 | AV1 | E3-TRUE | Y | LOGISTIC | 0.344 | 0.63 | 0.474 | 0.552 | 22 | 91 | 0.950 | 0.398 |
| 30 | omar_q05 | AV7 | E3-TRUE | Y | FINANCIA | 0.284 | 0.25 | 0.255 | 0.254 | 51 | 103 | 0.950 | 0.696 |
| 31 | omar_q06 | AV6 | E3-TRUE | Y | FINANCIA | 0.185 | 0.53 | 0.151 | 0.170 | 86 | 108 | 0.950 | 0.780 |
| 32 | omar_q11 | AV3 | E1 | N | — | — | — | — | — | — | 69 | 0.930 | — |
| 33 | priya_q03 | AV3 | E3-TRUE | Y | HEALTH_W | 0.201 | — | — | — | — | 127 | 0.981 | — |
| 34 | priya_q05 | AV7 | AV7-CROSS | Y | RELATION | 0.321 | 0.37 | 0.778 | 0.758 | 6 | 108 | 0.982 | 0.223 |
| 35 | priya_q07 | AV1 | E2 | N | — | — | — | — | — | — | 128 | 1.000 | — |
| 36 | priya_q09 | AV3 | E3-TRUE | Y | HEALTH_W | 0.285 | 0.00 | 0.609 | 0.548 | 19 | 153 | 0.911 | 0.363 |
| 37 | tom_q02 | AV1 | E3-TRUE | Y | FINANCIA | 0.307 | 0.50 | 0.292 | 0.302 | 50 | 122 | 1.000 | 0.697 |
| 38 | tom_q03 | AV1 | E3-TRUE | Y | HOBBIES_ | 0.251 | 0.63 | 0.247 | 0.324 | 44 | 153 | 0.964 | 0.640 |
| 39 | tom_q04 | AV3 | E3-TRUE | Y | HEALTH_W | 0.292 | 0.00 | 0.449 | 0.404 | 33 | 89 | 0.963 | 0.559 |
| 40 | tom_q06 | AV7 | E3-TRUE | Y | PROJECTS | 0.329 | 0.05 | 0.315 | 0.236 | 42 | 95 | 1.000 | 0.763 |
| 41 | tom_q07 | AV6 | E3-TRUE | Y | PREFEREN | 0.390 | 0.77 | 0.646 | 0.658 | 18 | 133 | 1.000 | 0.341 |

## Phase 2: Fix Simulation Results

### Fix A: Alpha halving (reduce category decay weights by 50%)

- **Recovered:** 6 — ['david_q11', 'elena_q02', 'elena_q13', 'jake_q03', 'omar_q01', 'priya_q05']
- **Regressions:** 0 — (none)
- **Net gain:** 6
- **Implementation effort:** Low — Tune constants in CATEGORY_DECAY dict

### Fix B: Pool expansion (Graphiti top-50 → top-200)

- **Recovered:** 5 — ['amara_q07', 'amara_q12', 'elena_q01', 'marcus_q03', 'priya_q03']
- **Regressions:** 0 — (none)
- **Net gain:** 5
- **Implementation effort:** Medium — Change num_results parameter in build_candidate_pool + retest

### Fix C: Cross-category retrieval (all 11 categories for AV7)

- **Recovered:** 3 — ['elena_q03', 'marcus_q03', 'tom_q06']
- **Regressions:** 0 — (none)
- **Net gain:** 3
- **Implementation effort:** Medium — Add cross-category fallback in route_and_retrieve

### Fix D: Routing inertness verification (diagnostic only)

- **Recovered:** 0 — (none)
- **Regressions:** 0 — (none)
- **Net gain:** 0
- **Implementation effort:** N/A — Diagnostic only — no code change

### Fix E: Semantic-only ranking (disable activation decay)

- **Recovered:** 3 — ['elena_q02', 'elena_q03', 'jake_q03']
- **Regressions:** 1 — ['jake_q11']
- **Net gain:** 2
- **Implementation effort:** Low — Add config flag to disable activation in rerank_candidates

### Fix F: Retraction recovery (re-ingestion to extract retraction events)

- **Recovered:** 2 — ['amara_q06', 'david_q04']
- **Regressions:** 0 — (none)
- **Net gain:** 2
- **Implementation effort:** High — Re-ingest all personas with improved extraction prompts

### Fix G: Aggressive semantic floor (threshold 0.85, floors +5%)

- **Recovered:** 1 — ['jake_q03']
- **Regressions:** 0 — (none)
- **Net gain:** 1
- **Implementation effort:** Low — Tune SEMANTIC_FLOOR constants

### Fix H: Benchmark correction (fix ground truth for E1 questions)

- **Recovered:** 2 — ['elena_q13', 'omar_q11']
- **Regressions:** 0 — (none)
- **Net gain:** 2
- **Implementation effort:** Low — Edit lifemembench_questions.json ground truth

## Phase 3: Interaction Analysis

- **Recoverable:** 16 / 41
- **Unrecoverable:** 25 — ['amara_q02', 'amara_q03', 'amara_q08', 'amara_q11', 'amara_q14', 'david_q01', 'david_q03', 'david_q05', 'elena_q04', 'elena_q08', 'elena_q12', 'jake_q05', 'jake_q07', 'jake_q10', 'marcus_q06', 'marcus_q13', 'omar_q02', 'omar_q05', 'omar_q06', 'priya_q07', 'priya_q09', 'tom_q02', 'tom_q03', 'tom_q04', 'tom_q07']
- **Minimal fix set:** ['A', 'B', 'C', 'F', 'H']

### Recovery Matrix

| QID | Fixes | Count |
|-----|-------|-------|
| amara_q02 | — | 0 |
| amara_q03 | — | 0 |
| amara_q06 | F | 1 |
| amara_q07 | B | 1 |
| amara_q08 | — | 0 |
| amara_q11 | — | 0 |
| amara_q12 | B | 1 |
| amara_q14 | — | 0 |
| david_q01 | — | 0 |
| david_q03 | — | 0 |
| david_q04 | F | 1 |
| david_q05 | — | 0 |
| david_q11 | A | 1 |
| elena_q01 | B | 1 |
| elena_q02 | A, E | 2 |
| elena_q03 | C, E | 2 |
| elena_q04 | — | 0 |
| elena_q08 | — | 0 |
| elena_q12 | — | 0 |
| elena_q13 | A, H | 2 |
| jake_q03 | A, E, G | 3 |
| jake_q05 | — | 0 |
| jake_q07 | — | 0 |
| jake_q10 | — | 0 |
| marcus_q03 | B, C | 2 |
| marcus_q06 | — | 0 |
| marcus_q13 | — | 0 |
| omar_q01 | A | 1 |
| omar_q02 | — | 0 |
| omar_q05 | — | 0 |
| omar_q06 | — | 0 |
| omar_q11 | H | 1 |
| priya_q03 | B | 1 |
| priya_q05 | A | 1 |
| priya_q07 | — | 0 |
| priya_q09 | — | 0 |
| tom_q02 | — | 0 |
| tom_q03 | — | 0 |
| tom_q04 | — | 0 |
| tom_q06 | C | 1 |
| tom_q07 | — | 0 |

## Phase 4: Projected Score Table

| Fix | Recovered | Regressions | Net | New Pass | New Rate |
|-----|-----------|-------------|-----|----------|----------|
| A | 6 | 0 | +6 | 77/112 | 68.8% |
| B | 5 | 0 | +5 | 76/112 | 67.9% |
| C | 3 | 0 | +3 | 74/112 | 66.1% |
| D | 0 | 0 | +0 | 71/112 | 63.4% |
| E | 3 | 1 | +2 | 73/112 | 65.2% |
| F | 2 | 0 | +2 | 73/112 | 65.2% |
| G | 1 | 0 | +1 | 72/112 | 64.3% |
| H | 2 | 0 | +2 | 73/112 | 65.2% |
| ALL | 16 | 1 | +15 | 86/112 | 76.8% |

## Phase 5: Recommendation

### Fix Ranking (by ROI = net_gain / effort)

| Priority | Fix | Net Gain | Effort | ROI |
|----------|-----|----------|--------|-----|
| 1 | A | +6 | Low | 6.0 |
| 2 | B | +5 | Medium | 2.5 |
| 3 | E | +2 | Low | 2.0 |
| 4 | H | +2 | Low | 2.0 |
| 5 | C | +3 | Medium | 1.5 |
| 6 | G | +1 | Low | 1.0 |
| 7 | F | +2 | High | 0.7 |

### Path to 70% (78/112)

- **Currently passing:** 71
- **Need:** +7 net recoveries
- Fix A (+6) → 77/112 (68.8%)
- Fix B (+5) → 82/112 (73.2%)

**Recommended combination:** ['A', 'B']
**Projected: 82/112 (73.2%) — target achieved**
