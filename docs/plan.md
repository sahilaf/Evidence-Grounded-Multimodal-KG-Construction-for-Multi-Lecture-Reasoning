# Plan to Improve Concept Extraction and Knowledge Graph Quality

## Current Diagnosis

The pipeline is producing a weak knowledge graph because quality is being lost at several stages, then hidden by permissive evaluation.

The strongest observed symptoms are:

- The final KG has only 14 concepts and 6 relationships across three neural network lectures.
- All 14 concepts appear in all 3 lectures, which is suspiciously uniform and suggests template-like extraction or over-merging.
- QA retrieves irrelevant first concepts: `What is a neural network?` answers with `Layer`, and an activation-function query answers with a gradient-descent definition.
- The notebook reports 100% answer/citation/reasoning success even when answers are semantically wrong.

## Root Causes

### 1. Inconsistent paths and repeated `Config` classes

The notebook redefines `Config` multiple times with different `DRIVE_ROOT` values:

- Preprocessing uses `/content/drive/MyDrive/Data_mining1`
- Anchor detection and later stages use `/content/drive/MyDrive/Data_mining`

This can silently mix stale frames, transcripts, anchors, and extracted knowledge from different runs.

### 2. Anchor detection optimizes frame importance, not concept coverage

Anchor selection keeps about 9% of frames with a hard cap of 160 anchors. This can be fine for compression, but there is no check that selected anchors cover:

- every transcript segment containing definitions,
- OCR-visible slide text,
- first mentions of key terms,
- examples and relationship explanations,
- prerequisite chains.

The CLIP alignment is computed only every 5th segment and truncates transcript text to 77 tokens, so many frames receive no multimodal score. The final score is also normalized per lecture, which makes weak signals look strong if an entire lecture has weak evidence.

### 3. The extractor prompt forces synthetic output

The prompt asks the model to extract `5-10 SPECIFIC technical concepts` from every anchor. Many anchors will not contain 5-10 real concepts, so the model is incentivized to invent or repeat a fixed list like `neural network`, `gradient descent`, `activation function`, `matrix multiplication`, `derivative`, `loss function`, etc.

This explains why each lecture ends up with the same 14 concepts.

### 4. Extraction has no grounding fields

Extracted concepts only keep `name` and `definition`. They do not preserve:

- source anchor id,
- timestamp,
- transcript span,
- quoted evidence,
- visual/OCR evidence,
- confidence,
- extraction mode,
- whether the concept came from transcript, image, or both.

Without grounding, bad concepts cannot be audited, and the graph cannot distinguish lecture evidence from model-generated summaries.

### 5. Filtering is too shallow

The concept filter only checks short names, short definitions, a small bad-word list, and a few misspellings. It does not reject:

- concepts absent from transcript/OCR evidence,
- generic educational terms,
- repeated canned definitions,
- visually inferred but unsupported terms,
- concepts with no source span,
- low-confidence extractions.

### 6. Deduplication breaks relationship consistency

Concepts are deduplicated after extraction, but relationships are not canonicalized against the deduped concept list. The graph then drops relationships when source/target nodes do not exist. This likely contributes to only 6 final edges.

### 7. Entity resolution over-merges and under-tracks provenance

The graph merges concepts using MiniLM embeddings of concept names only with a 0.90 threshold. This is risky because short technical labels can be semantically close even when distinct. Also, when a concept repeats within the same lecture, the lecture id is appended repeatedly rather than maintained as a set.

### 8. Retrieval and QA mask upstream problems

QA always returns top 5 concepts even if similarity is weak. It then uses the first node from the subgraph, whose order is not guaranteed to match retrieval rank. This is why the neural-network question can answer with `Layer`.

The evaluation only checks whether an answer exists, citations exist, and a reasoning path exists. It does not check correctness, grounding, or whether the top retrieved concept matches the question.

## Improvement Plan

### Phase 1: Make the pipeline auditable

1. Consolidate configuration into one object or one notebook cell.
2. Use one `DRIVE_ROOT` for all stages.
3. Add run ids and save every output under a run-specific directory.
4. Add a manifest file with:
   - input videos,
   - transcript files,
   - frame directory,
   - anchor file,
   - model names,
   - extraction prompt version,
   - timestamps.
5. Add stage-level quality reports:
   - transcript word count and segment count,
   - anchors per lecture and temporal coverage,
   - extraction success rate,
   - concepts per anchor distribution,
   - relationships kept/dropped,
   - graph merge decisions.

### Phase 2: Improve anchor selection for concept recall

1. Add OCR over selected frames and nearby frames.
2. Select anchors by coverage categories, not only global top score:
   - definition-like transcript segments,
   - slide/title transitions,
   - equations,
   - examples,
   - first mention of candidate terms,
   - relationship phrases such as `depends on`, `used to`, `minimizes`, `computed by`.
3. Expand each anchor into a local window:
   - transcript from `t-15s` to `t+20s`,
   - neighboring frames around slide transitions,
   - OCR text from the selected frame.
4. Keep recall high initially, then dedupe downstream. Target 15-25% of semantically relevant segments during extraction experiments.
5. Add an anchor recall test using a small gold list of expected concepts per lecture.

### Phase 3: Replace forced concept generation with grounded extraction

1. Change the prompt from `Extract 5-10 concepts` to `Extract only concepts explicitly supported by the transcript or visible frame. Return an empty list if none are present.`
2. Require this schema:

```json
{
  "concepts": [
    {
      "name": "string",
      "definition": "string",
      "evidence_quote": "short transcript or OCR quote",
      "source": "transcript|ocr|both|visual",
      "confidence": 0.0
    }
  ],
  "relationships": [
    {
      "source": "string",
      "target": "string",
      "type": "prerequisite_of|component_of|causes|optimizes|uses|example_of|related_to",
      "evidence_quote": "short quote",
      "confidence": 0.0
    }
  ]
}
```

3. Run extraction in two passes:
   - Pass A: transcript/OCR-only candidate concept extraction.
   - Pass B: vision-language validation and relationship extraction for candidates.
4. Use deterministic generation first: `do_sample=False`, low max tokens, and strict JSON repair.
5. Store raw model response, parsed JSON, and filtered JSON for every anchor.

### Phase 4: Add real validation and canonicalization

1. Normalize concept names with a canonicalizer:
   - lowercase,
   - trim punctuation,
   - singularize common plurals,
   - map aliases such as `cost function` -> `loss function` only when evidence supports it.
2. Reject concepts unless their name or alias appears in transcript/OCR, or the model provides high-confidence visual evidence.
3. Deduplicate using multiple signals:
   - exact/alias match first,
   - embedding similarity second,
   - definition similarity third,
   - reject merges for known distinct pairs.
4. Preserve all mentions:
   - `mentions`: list of lecture, anchor, timestamp, evidence quote.
   - `definitions`: unique definitions with source references.
   - `lectures`: set, not append-only list.
5. Canonicalize relationships after concept dedupe.
6. Drop or flag relationships whose endpoints cannot be mapped to canonical concepts.
7. Keep relationship evidence and confidence.

### Phase 5: Rebuild the KG with stricter semantics

1. Build graph nodes from canonical concepts only.
2. Add edges only from validated relationships.
3. Use a typed relationship ontology:
   - `prerequisite_of`
   - `component_of`
   - `optimizes`
   - `uses`
   - `computed_by`
   - `example_of`
   - `contrasts_with`
4. Track provenance on every node and edge.
5. Add graph sanity checks:
   - no all-lecture concepts unless evidence exists in all lectures,
   - no duplicate lecture ids per node,
   - minimum edge density threshold,
   - relationship endpoint coverage,
   - top concepts by evidence count.

### Phase 6: Fix retrieval and QA evaluation

1. Precompute node embeddings once; do not re-embed all nodes per query.
2. Score retrieval with a hybrid method:
   - exact concept-name match,
   - alias match,
   - embedding similarity,
   - definition/evidence match.
3. Add a minimum similarity threshold and return `not found` when below threshold.
4. Preserve retrieval order in the subgraph and answer from the top matched concept, not arbitrary graph node order.
5. Classify evolution questions before generic `how/explain` questions.
6. Evaluate with correctness checks:
   - top-1 concept accuracy,
   - top-3 concept recall,
   - answer faithfulness to evidence,
   - relationship accuracy,
   - citation evidence validity.

## Suggested Implementation Order

1. Fix `Config` path drift and add run manifests.
2. Modify extraction schema to include evidence and confidence.
3. Remove the `5-10 concepts` requirement.
4. Add OCR and transcript-window context.
5. Canonicalize concepts before saving relationships.
6. Rebuild graph with provenance.
7. Fix QA retrieval ordering and thresholds.
8. Add a small gold evaluation set for the three lectures.

## Minimum Acceptance Criteria

- Each extracted concept has at least one evidence quote and timestamp.
- Empty or low-information anchors can produce zero concepts.
- At least 80% of relationships have endpoints mapped to canonical concepts.
- The graph has plausible lecture-specific concepts, not the same fixed list in every lecture.
- `What is a neural network?` retrieves `neural network` as top-1.
- `How does activation function appear across lectures?` does not answer with a gradient-descent definition.
- Evaluation flags wrong answers instead of counting any non-empty answer as success.

