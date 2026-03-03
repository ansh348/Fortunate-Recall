# Graphiti Patches — Numeric Preservation in Edge Pipeline

These files are patched copies of Graphiti source files that fix numeric detail
loss during edge extraction, deduplication, and resolution.

## Problem

Graphiti's LLM extraction pipeline was losing specific numbers during edge
processing: "$950/month" became "pays rent", "$300-400" became "sends money",
"7 bass" became "goes fishing". This caused 58 of 63 AV benchmark failures.

## Root Causes Found

1. **resolve_extracted_edge** (edge_operations.py:843-846): When dedup LLM
   marks a new edge as duplicate of an existing one, the code unconditionally
   kept the OLD edge and discarded the new one. If the old edge lacked numbers
   (from before extraction prompt fix), numbers were lost.

2. **dedupe_edges_bulk** (bulk_utils.py:489-500): In bulk ingestion, when
   duplicate edges were found across episodes, the edge with the
   lexicographically smallest UUID was kept — arbitrary and could discard
   the numeric-rich version.

## Changes Made

### edge_operations.py
- **Added** `_count_numeric_tokens(text)` helper (counts $amounts, percentages,
  bare numbers via regex)
- **Modified** `resolve_extracted_edge` duplicate resolution: when a duplicate
  is found, compares numeric token counts and transplants the richer fact text
  onto the existing edge (preserving UUID/metadata)

### bulk_utils.py
- **Added** post-dedup numeric preference: after UUID-based dedup compression,
  iterates duplicate groups and ensures the canonical edge gets the most
  numeric-rich fact text from its group
- **Imported** `_count_numeric_tokens` from edge_operations

### dedupe_edges.py
- **Expanded** dedup prompt Guidelines with 6 domain-specific examples of
  facts that are NOT duplicates due to numeric specificity differences

### extract_edges.py (previously patched — included for completeness)
- Already has CRITICAL numeric preservation instructions in extraction rules

## How to Apply

Copy these files over the Graphiti source:

```bash
cp graphiti_patches/edge_operations.py graphiti/graphiti_core/utils/maintenance/edge_operations.py
cp graphiti_patches/bulk_utils.py graphiti/graphiti_core/utils/bulk_utils.py
cp graphiti_patches/dedupe_edges.py graphiti/graphiti_core/prompts/dedupe_edges.py
cp graphiti_patches/extract_edges.py graphiti/graphiti_core/prompts/extract_edges.py
```
