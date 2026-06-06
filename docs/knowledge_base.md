# Knowledge Base for Paper: Evidence-Grounded Multimodal KG Construction for Multi-Lecture Reasoning

## Working Title

Evidence-Grounded Multimodal Knowledge Graph Construction for Multi-Lecture Educational Reasoning

## Core Research Problem

Lecture videos contain knowledge across multiple modalities: spoken explanations, slide text, diagrams, equations, and temporal sequencing. Standard transcript-only RAG can retrieve relevant text, but it struggles with:

- cross-lecture concept continuity,
- visual concepts shown on slides or diagrams,
- explicit prerequisite and component relationships,
- grounded citation to exact lecture moments,
- faithful reasoning over concept dependencies.

The proposed system converts multi-lecture video content into an evidence-grounded multimodal knowledge graph where every concept and relationship is linked to transcript, OCR, visual, timestamp, and lecture provenance.

## Main Contribution

The main contribution should be framed as:

> A multimodal, evidence-grounded pipeline for constructing educational knowledge graphs from lecture videos, enabling cross-lecture reasoning and grounded question answering.

Avoid claiming generic SOTA unless you run strong baselines. A safer and stronger claim is that the pipeline improves grounding, auditability, and retrieval quality compared with transcript-only RAG and unguided VLM extraction.

## Proposed Method

The pipeline has seven stages.

1. **Lecture Preprocessing**
   - Download lecture videos.
   - Extract audio.
   - Extract frames at 1 FPS.
   - Save all artifacts to a run-specific Drive directory.

2. **ASR Transcription**
   - Use Faster-Whisper `large-v3`.
   - Generate timestamped transcript segments.
   - Align transcript segments to frame ids using segment midpoint timestamps.

3. **High-Recall Semantic Anchor Selection**
   - Select candidate anchors using visual changes, transcript keywords, relation cues, and first mentions.
   - Use a high-recall setting rather than aggressive compression.
   - Expand each anchor with a transcript context window around the timestamp.

4. **OCR Extraction**
   - Run EasyOCR on selected anchor frames.
   - Extract visible slide text, equations, labels, and diagram annotations.
   - OCR is saved per anchor.

5. **Evidence-Grounded VLM Extraction**
   - Use Qwen2.5-VL-7B-Instruct.
   - Input: image frame, transcript window, OCR text, and candidate terms.
   - Output: strict JSON containing concepts and typed relationships.
   - Empty outputs are allowed.
   - Every concept and relationship must include an evidence quote, source type, confidence score, timestamp, and anchor id.

6. **Canonicalization and KG Construction**
   - Normalize concept names.
   - Resolve aliases.
   - Deduplicate concepts using exact match, alias match, embedding similarity, and token overlap.
   - Canonicalize relationship endpoints after concept deduplication.
   - Drop relationships whose endpoints cannot be mapped.
   - Build a NetworkX `MultiDiGraph` with provenance-rich nodes and edges.

7. **Graph-Augmented QA**
   - Retrieve relevant concepts using hybrid retrieval:
     - exact concept match,
     - alias match,
     - semantic embedding match,
     - evidence-count prior.
   - Extract local subgraph around retrieved concepts.
   - Generate answers only from KG context.
   - Cite lecture ids and timestamps.

## Why This Is Better Than the Earlier Pipeline

The previous notebook had several weaknesses:

- It forced the VLM to extract 5-10 concepts per anchor, causing hallucinated or repeated concepts.
- It did not require evidence quotes.
- It deduplicated concepts but did not canonicalize relationships afterward.
- The KG collapsed into only 14 concepts and 6 edges.
- All concepts appeared in all lectures, which suggested over-merging or template-like extraction.
- QA evaluation counted any non-empty answer as successful.

The new pipeline fixes this by requiring evidence, allowing empty anchors, validating concepts, preserving provenance, canonicalizing edges, and adding retrieval/evaluation checks.

## Key Terms

**Semantic Anchor**

A selected lecture frame and timestamp that is likely to contain educationally important content, such as a definition, diagram, equation, example, or relation cue.

**Evidence-Grounded Concept**

A concept whose extracted definition is supported by a transcript quote, OCR quote, or high-confidence visual evidence.

**Evidence-Grounded Relationship**

A typed edge between two concepts that includes a supporting quote and provenance.

**Canonical Concept**

A normalized graph node created by merging aliases and repeated mentions of the same concept.

**Multi-Lecture Reasoning**

Reasoning that uses knowledge distributed across more than one lecture, including recurrence, prerequisite chains, and concept evolution.

## Node Schema

Each KG node represents a canonical concept.

Important fields:

- `id`
- `display_name`
- `canonical_name`
- `aliases`
- `primary_definition`
- `definitions`
- `mentions`
- `lectures`
- `evidence_count`
- `avg_confidence`

Each mention stores:

- `video_id`
- `anchor_id`
- `frame_id`
- `timestamp`
- `frame_path`
- `evidence_quote`
- `source`
- `confidence`

## Edge Schema

Each KG edge represents a validated relationship.

Important fields:

- `source`
- `target`
- `type`
- `video_id`
- `anchor_id`
- `timestamp`
- `evidence_quote`
- `confidence`

Allowed edge types:

- `prerequisite_of`
- `component_of`
- `uses`
- `optimizes`
- `computed_by`
- `example_of`
- `contrasts_with`
- `related_to`

For publication, `related_to` should be used sparingly because it is less semantically informative.

## Recommended Baselines

Use simple but credible baselines:

1. **Transcript-only RAG**
   - Embed transcript chunks.
   - Retrieve top chunks.
   - Generate QA answer.

2. **OCR + Transcript RAG**
   - Combine transcript chunks with OCR text.
   - Retrieve using vector search.

3. **Ungrounded VLM Extraction**
   - Ask the VLM to extract concepts from anchors without evidence validation.

4. **Proposed Full Pipeline**
   - Transcript + OCR + visual anchor + evidence validation + KG + GraphRAG.

## Evaluation Metrics

### Concept Extraction

Use a manually labeled gold set.

Metrics:

- precision,
- recall,
- F1,
- evidence validity rate.

Evidence validity rate:

> Percentage of extracted concepts whose evidence quote truly supports the concept.

### Relationship Extraction

Metrics:

- relation precision,
- relation recall,
- relation F1,
- endpoint mapping rate,
- evidence-supported edge rate.

Endpoint mapping rate:

> Percentage of extracted relationships whose source and target map to valid canonical concepts.

### Retrieval

Metrics:

- top-1 concept accuracy,
- top-3 concept accuracy,
- top-5 concept recall,
- mean reciprocal rank.

### QA

Metrics:

- answer correctness,
- faithfulness to KG evidence,
- citation accuracy,
- unsupported-answer rate.

### Graph Quality

Metrics:

- number of canonical concepts,
- number of validated relationships,
- edge density,
- isolated node count,
- average evidence per concept,
- cross-lecture concept count,
- relationship endpoint coverage.

## Suggested Ablation Study

Run these variants:

| Variant | Description |
|---|---|
| Transcript-only | Uses only ASR transcript for extraction/retrieval |
| OCR-only | Uses only frame OCR |
| Transcript + OCR | No visual frame reasoning |
| VLM without grounding | Extracts from image/transcript but no evidence validation |
| Full proposed pipeline | Transcript + OCR + image + evidence validation + KG |
| Plain RAG QA | Vector retrieval without KG |
| GraphRAG QA | Proposed KG-based retrieval |

The most important ablation is:

> Full evidence-grounded KG vs transcript-only RAG.

## Expected Findings

Likely paper findings:

- Evidence validation reduces hallucinated concepts.
- OCR improves extraction of slide-specific terms and equations.
- Canonicalization increases graph coherence.
- KG retrieval improves multi-hop and cross-lecture questions.
- Plain RAG may answer definitions well, but KG-RAG should do better on prerequisites, relationships, and concept evolution.

## Claims to Make Carefully

Safe claims:

- The pipeline improves auditability through evidence-grounded nodes and edges.
- The method supports cross-lecture reasoning by linking recurring concepts.
- Evidence validation reduces unsupported concepts.
- Graph-based retrieval improves relationship-oriented QA compared with flat retrieval.

Avoid unless fully proven:

- “SOTA performance”
- “eliminates hallucination”
- “fully automatic concept understanding”
- “complete lecture knowledge graph”
- “human-level reasoning”

## Paper Structure

### Introduction

Discuss the challenge of reasoning over lecture videos. Emphasize that educational content is multimodal and temporally distributed. Motivate the need for evidence-grounded KG construction.

### Related Work

Cover:

- educational video understanding,
- multimodal RAG,
- knowledge graph construction with LLMs,
- GraphRAG,
- lecture QA systems.

### Method

Describe:

- preprocessing,
- anchor selection,
- OCR,
- grounded extraction,
- canonicalization,
- KG construction,
- GraphRAG QA.

### Experiments

Describe:

- dataset,
- baselines,
- metrics,
- annotation process,
- implementation details.

### Results

Report:

- extraction performance,
- relationship performance,
- graph quality,
- QA performance,
- ablation results.

### Discussion

Discuss:

- grounding benefits,
- failure cases,
- noisy ASR/OCR,
- relation extraction limitations,
- scalability.

### Limitations

Important limitations:

- OCR can fail on low-resolution slides.
- VLM extraction may still miss implicit concepts.
- Gold annotation is labor-intensive.
- Relationship extraction is harder than concept extraction.
- Frame sampling may miss fast slide transitions.

## Implementation Artifacts

Notebook:

- `Evidence_Grounded_Multimodal_KG_Construction.ipynb`

Planning file:

- `plan.md`

Generated outputs:

- `manifest.json`
- `lecture_records.json`
- transcripts
- anchors
- OCR files
- raw extractions
- validated concept mentions
- canonical concepts
- canonical relationships
- `knowledge_graph.pkl`
- `knowledge_graph.graphml`
- `concepts_summary.csv`
- `relationships_summary.csv`
- `kg_sanity_report.json`
- `qa_results_grounded.json`
- `retrieval_eval.json`
- `ablation_table.csv`

## Minimum Publication-Quality Checklist

Before writing final results, ensure:

- Every concept has timestamped evidence.
- Every edge has mapped endpoints and evidence.
- The KG does not contain the same fixed concept list for every lecture.
- Retrieval top-1 for obvious questions is correct.
- QA answers cite actual lecture timestamps.
- Evaluation includes at least one manually annotated gold set.
- Ablations compare against transcript-only RAG.

## One-Sentence Pitch

This work presents an evidence-grounded multimodal pipeline that converts lecture videos into provenance-rich knowledge graphs for faithful cross-lecture educational reasoning.

