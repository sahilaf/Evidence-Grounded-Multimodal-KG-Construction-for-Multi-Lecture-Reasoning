# Evidence-Grounded Multimodal Knowledge Graph Construction for Multi-Lecture Reasoning

## Abstract

Lecture videos contain educational knowledge distributed across speech, slide text, diagrams, equations, and temporal presentation order. Standard transcript-only retrieval-augmented generation (RAG) can answer local factual questions, but it often lacks explicit concept structure, provenance, and support for reasoning across lectures. We present an evidence-grounded multimodal pipeline for constructing knowledge graphs from lecture videos. The system extracts timestamped frames and transcripts, selects high-recall semantic anchors, performs OCR over anchor frames, and uses a vision-language model to extract only concepts and relationships supported by transcript, OCR, or visual evidence. Extracted mentions are validated, canonicalized into graph nodes, and connected by typed relationships with preserved evidence quotes and timestamps. On a three-lecture neural networks series, the pipeline processed 3,118 video frames, 756 transcript segments, and 559 semantic anchors. It produced 1,022 validated concept mentions, 172 canonical concepts, and 282 validated graph relationships, with 90.38% relationship endpoint coverage. In a starter retrieval evaluation, GraphRAG retrieval achieved 100% top-1 accuracy, 100% top-3 accuracy, and 100% mean top-5 recall over three seed questions. Qualitative QA results show improved grounding for definitions, relationships, prerequisites, and cross-lecture concept queries. These results suggest that evidence-grounded multimodal KG construction is a promising direction for faithful educational video reasoning, while also revealing remaining challenges in entity canonicalization, isolated node reduction, and larger-scale human evaluation.

## 1. Introduction

Educational lectures are inherently multimodal. A single concept may be introduced verbally, labeled on a slide, illustrated by a diagram, expanded through equations, and revisited in later lectures. This creates a challenge for question answering systems: the relevant information is not always contained in a single transcript chunk, and flat retrieval may miss relationships such as prerequisites, components, examples, and optimization dependencies.

Retrieval-augmented generation (RAG) addresses some limitations of parametric language models by grounding generation in retrieved external context. However, conventional RAG over lecture transcripts treats educational content as a sequence of text chunks rather than as a structured concept network. GraphRAG-style approaches argue that graph-structured indexes can better support global and relational reasoning over complex corpora. For educational videos, this motivates a multimodal graph construction pipeline that links concepts across lectures while retaining evidence for every extraction.

This work proposes an evidence-grounded multimodal knowledge graph construction pipeline for multi-lecture reasoning. The central design principle is auditability: every node and edge in the graph must be traceable to lecture evidence, including transcript windows, OCR text, frame timestamps, and confidence scores. Unlike unguided extraction prompts that force a fixed number of concepts per frame, our pipeline allows empty outputs and filters unsupported concepts and relationships.

The contributions are:

1. An end-to-end multimodal lecture-to-KG pipeline combining ASR, visual anchor selection, OCR, vision-language extraction, canonicalization, and graph-based retrieval.
2. An evidence-grounded extraction schema requiring support quotes, source modality, confidence, timestamp, anchor id, and lecture id for concepts and relationships.
3. A provenance-rich educational KG with 172 canonical concepts and 282 relationships extracted from a three-lecture neural networks series.
4. A GraphRAG-style QA layer using hybrid concept retrieval and local subgraph grounding.
5. A publication-oriented evaluation scaffold for concept extraction, relationship extraction, retrieval, QA faithfulness, and ablations.

## 2. Related Work

RAG was introduced as a way to combine parametric sequence-to-sequence models with retrieved non-parametric memory for knowledge-intensive NLP tasks. Lewis et al. showed that retrieval can improve open-domain question answering by conditioning generation on external documents [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

GraphRAG extends this idea by constructing graph-based indexes over source corpora. Edge et al. propose graph-based retrieval for query-focused summarization over large private text collections, arguing that flat chunk retrieval can fail for global or relational questions [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). Our work adapts this motivation to educational videos, where relationships among concepts are central to reasoning.

For speech transcription, we build on Whisper-style large-scale weakly supervised ASR. Radford et al. demonstrated that speech recognition models trained on 680,000 hours of multilingual and multitask supervision generalize well across settings [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356). We use Faster-Whisper with the `large-v3` model for efficient lecture transcription.

For multimodal extraction, we use Qwen2.5-VL-7B-Instruct, a recent vision-language model family designed for visual-textual understanding tasks including document and chart interpretation [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923). For semantic retrieval and canonicalization, we use BGE-large English embeddings [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).

## 3. Method

### 3.1 Pipeline Overview

The system converts a lecture series into a provenance-rich knowledge graph through ten stages:

1. Download lecture videos.
2. Extract audio and frames.
3. Transcribe audio with timestamped ASR.
4. Select high-recall semantic anchors.
5. Run OCR over anchor frames.
6. Extract grounded concepts and relationships with a VLM.
7. Validate extracted items using evidence and confidence thresholds.
8. Canonicalize concepts and relationship endpoints.
9. Build a typed knowledge graph.
10. Perform GraphRAG-style retrieval and QA.

All intermediate artifacts are saved under a run-specific directory in Google Drive, enabling reproducibility and resumable execution.

### 3.2 Preprocessing and Transcription

Videos are downloaded from YouTube and processed into synchronized audio, frame, and transcript streams. Frames are extracted at 1 FPS. Audio is converted to 16 kHz mono WAV and transcribed using Faster-Whisper `large-v3`. Each transcript segment stores start time, end time, text, and a primary frame id computed from the segment midpoint.

### 3.3 Semantic Anchor Selection

The anchor selector is designed for recall rather than aggressive compression. It combines three signals:

- visual frame-change score,
- transcript keyword and relationship-cue score,
- first-mention score based on TF-IDF candidate terms.

Anchors are selected with a temporal spacing constraint and expanded with a transcript window around the timestamp. This gives the VLM enough local context to avoid interpreting a frame in isolation.

### 3.4 OCR

EasyOCR is applied to selected anchor frames. OCR text is stored per anchor and later used as extraction evidence. This step is important for educational videos because slide labels, equations, and diagram annotations often do not appear verbatim in the spoken transcript.

### 3.5 Evidence-Grounded VLM Extraction

For each anchor, Qwen2.5-VL receives:

- the selected frame,
- the transcript window,
- OCR text,
- candidate terms extracted from local text.

The model is instructed to return strict JSON. Unlike the earlier pipeline, it is not asked to extract a fixed number of concepts. Empty outputs are valid. Each concept must include:

- `name`,
- `definition`,
- `evidence_quote`,
- `source`,
- `confidence`.

Each relationship must include:

- `source`,
- `target`,
- `type`,
- `evidence_quote`,
- `source_type`,
- `confidence`.

The allowed relationship types are `prerequisite_of`, `component_of`, `uses`, `optimizes`, `computed_by`, `example_of`, `contrasts_with`, and `related_to`.

### 3.6 Validation and Canonicalization

Extracted concepts are filtered if they lack a name, definition, evidence quote, sufficient confidence, or textual evidence support. Visual-only concepts require higher confidence. Relationship extraction is filtered by endpoint validity, relation type, evidence quote, and confidence.

Concept names are normalized with alias handling, then deduplicated using embedding similarity, token overlap, and fuzzy string matching. Relationships are canonicalized after concept deduplication. Edges whose endpoints cannot be mapped to canonical nodes are dropped and logged.

### 3.7 Knowledge Graph Construction

The final KG is a NetworkX `MultiDiGraph`. Nodes represent canonical concepts and store aliases, definitions, mentions, lecture coverage, evidence count, and confidence. Edges represent typed relationships and store source concept, target concept, relation type, lecture id, timestamp, anchor id, evidence quote, and confidence.

### 3.8 GraphRAG Question Answering

QA uses hybrid retrieval over canonical concepts. The retrieval score combines:

- semantic similarity,
- exact concept-name match,
- alias match,
- fuzzy match,
- evidence-count prior.

The system extracts a one-hop local subgraph around retrieved concepts and formats concept definitions, evidence quotes, and relationships as context for grounded answer generation. The answer prompt instructs the model to cite lecture ids and timestamps and to avoid unsupported claims.

## 4. Experimental Setup

### 4.1 Dataset

We evaluate on a three-lecture neural networks series:

| Lecture ID | Topic | Frames | Transcript Segments | Words |
|---|---:|---:|---:|---:|
| `neural_net_01` | Neural Networks | 1,120 | 287 | 3,277 |
| `neural_net_02` | Gradient Descent | 1,232 | 283 | 3,698 |
| `neural_net_03` | Backpropagation | 766 | 186 | 2,222 |

Total:

| Quantity | Value |
|---|---:|
| Lectures | 3 |
| Frames | 3,118 |
| Transcript segments | 756 |
| Transcript words | 9,197 |

### 4.2 Models

| Component | Model / Tool |
|---|---|
| ASR | Faster-Whisper `large-v3` |
| OCR | EasyOCR |
| VLM extraction | Qwen2.5-VL-7B-Instruct |
| Embeddings | BAAI `bge-large-en-v1.5` |
| Graph | NetworkX `MultiDiGraph` |

### 4.3 Metrics

The current run reports pipeline and retrieval metrics:

- number of selected anchors,
- OCR non-empty rate,
- raw concepts and relationships,
- validated concept mentions,
- canonical concepts,
- validated relationships,
- relationship endpoint coverage,
- graph size,
- retrieval top-1 accuracy,
- retrieval top-3 accuracy,
- mean top-5 recall.

A larger manually annotated gold set is required for final concept F1, relation F1, and QA faithfulness.

## 5. Results

### 5.1 Anchor Selection and OCR

The pipeline selected 559 semantic anchors from 3,118 extracted frames.

| Lecture ID | Frames | Anchors | Anchor Rate |
|---|---:|---:|---:|
| `neural_net_01` | 1,120 | 201 | 17.95% |
| `neural_net_02` | 1,232 | 221 | 17.94% |
| `neural_net_03` | 766 | 137 | 17.89% |
| **Total** | **3,118** | **559** | **17.93%** |

OCR returned non-empty text for most anchors.

| Lecture ID | Anchors | Non-Empty OCR | OCR Non-Empty Rate |
|---|---:|---:|---:|
| `neural_net_01` | 201 | 195 | 97.01% |
| `neural_net_02` | 221 | 203 | 91.86% |
| `neural_net_03` | 137 | 131 | 95.62% |
| **Total** | **559** | **529** | **94.63%** |

### 5.2 Extraction Yield

The VLM produced 1,155 raw concepts and 400 raw relationships. After evidence and confidence validation, 1,022 concept mentions and 312 relationship mentions remained.

| Lecture ID | Anchors | Raw Concepts | Raw Relationships |
|---|---:|---:|---:|
| `neural_net_01` | 201 | 409 | 145 |
| `neural_net_02` | 221 | 445 | 136 |
| `neural_net_03` | 137 | 301 | 119 |
| **Total** | **559** | **1,155** | **400** |

| Stage | Count |
|---|---:|
| Raw concept mentions | 1,155 |
| Validated concept mentions | 1,022 |
| Canonical concepts | 172 |
| Raw relationships | 400 |
| Validated relationship mentions | 312 |
| Final graph relationships | 282 |

### 5.3 Knowledge Graph Statistics

The final graph contains 172 canonical concepts and 282 relationships. Relationship endpoint coverage is 90.38%, meaning most validated relationship mentions could be mapped to canonical concept nodes.

| Metric | Value |
|---|---:|
| Canonical concepts | 172 |
| Final relationships | 282 |
| Relationship endpoint coverage | 90.38% |
| Average evidence per concept | 5.94 |

The top concepts by evidence count include weights, loss function, neuron, bias, gradient descent, neural network, gradient, sigmoid, activation function, weighted sum, hidden layers, and backpropagation. This suggests that the graph captures central neural-network training concepts rather than collapsing to the small fixed list produced by the previous pipeline.

### 5.4 Retrieval Results

A starter retrieval evaluation was run over three seed questions.

| Question | Gold Concept(s) | Top Retrieved Concepts | Top-1 Hit |
|---|---|---|---|
| What is a neural network? | neural network | neural network, network, neural networks, deep neural network, convolutional neural network | Yes |
| What is gradient descent? | gradient descent | gradient descent, gradient, gradient descent step, negative gradient, gradient direction | Yes |
| Explain the relationship between gradient descent and loss function. | gradient descent, loss function | gradient descent, loss function, function, gradient, negative gradient | Yes |

Aggregate retrieval metrics:

| Metric | Value |
|---|---:|
| Top-1 accuracy | 100% |
| Top-3 accuracy | 100% |
| Mean top-5 recall | 100% |

These retrieval results are encouraging but preliminary because the gold set currently contains only three questions.

### 5.5 Qualitative QA Results

For the question “What is a neural network?”, the system retrieved `neural network` as the top concept with score 0.871. The generated answer described neural networks as layered computational models with neurons, weights, biases, activation functions, and learning through cost minimization.

For “Explain the relationship between gradient descent and loss function,” the system retrieved `gradient descent` and `cost function/loss function` as the top two concepts. The answer correctly described gradient descent as adjusting weights and biases along the negative gradient to minimize the cost function.

For “What are prerequisites for understanding backpropagation?”, the system retrieved `backpropagation`, `propagation`, `backpropagation algorithm`, `idea propagating backwards`, `function`, and `weights`. The generated answer identified cost functions, weights, biases, propagation, and gradient descent as necessary background.

For “How does activation function appear across the lectures?”, the system retrieved `activation`, `function`, `activations`, `cost function`, `activation values`, and `cost`. The answer linked activation functions and activation values to how neural networks process information across layers.

## 6. Discussion

The evidence-grounded approach substantially improves over the earlier pipeline. The previous KG contained only 14 concepts and 6 relationships, with all concepts appearing across all three lectures. In contrast, the new pipeline produced 172 canonical concepts and 282 relationships, while retaining timestamped provenance. This demonstrates the value of allowing variable extraction per anchor, incorporating OCR, validating evidence, and canonicalizing relationships after concept deduplication.

The results also reveal important limitations. First, canonicalization is not yet perfect: singular and plural variants such as `weight` and `weights`, or `bias` and `biases`, remain separate concepts. Second, the graph contains many isolated nodes, including some noisy or overly specific concepts such as visual artifacts, numeric labels, and vague terms. Third, some generated QA answers may include general neural-network knowledge beyond the strict evidence context, so final evaluation should include human faithfulness judgments. Fourth, the retrieval evaluation is currently too small for publication-level claims.

Despite these limitations, the system’s outputs are much more auditable than the earlier version. Each concept mention stores a source quote, timestamp, lecture id, anchor id, frame id, and confidence. This enables manual inspection, error analysis, and future human-in-the-loop refinement.

## 7. Limitations

The current study has several limitations:

- The dataset contains only three lectures from one neural-network series.
- The retrieval evaluation uses only three seed questions.
- Concept and relationship F1 are not yet measured against a manually annotated gold standard.
- Entity canonicalization still leaves duplicates and near-duplicates.
- Many isolated nodes suggest that graph pruning and relation enrichment are needed.
- OCR quality depends on slide resolution and frame selection.
- VLM extraction may still introduce unsupported or overly generic concepts despite evidence constraints.
- QA generation should be evaluated for faithfulness, not only retrieval success.

## 8. Future Work

Future work should focus on:

1. Building a manually annotated gold set for concept and relationship extraction.
2. Adding stronger canonicalization for singular/plural variants and known neural-network aliases.
3. Pruning low-evidence isolated nodes.
4. Comparing against transcript-only RAG, OCR+transcript RAG, and ungrounded VLM extraction.
5. Evaluating answer faithfulness with human raters or LLM-as-judge calibrated against human judgments.
6. Extending the dataset to multiple lecture series and domains.
7. Adding explicit concept evolution tracking across lecture order.

## 9. Conclusion

We presented an evidence-grounded multimodal pipeline for constructing knowledge graphs from lecture videos. By combining ASR, semantic anchor selection, OCR, VLM extraction, evidence validation, canonicalization, and GraphRAG retrieval, the system transforms lecture videos into a provenance-rich graph suitable for multi-lecture reasoning. On a three-lecture neural-network series, the pipeline produced 172 canonical concepts and 282 relationships with 90.38% relationship endpoint coverage. Preliminary retrieval and QA results suggest that the graph supports grounded definitions, relationship reasoning, prerequisite queries, and cross-lecture concept tracing. The approach is promising for educational video understanding, but final publication-quality validation requires larger gold annotations, stronger canonicalization, and broader ablation studies.

## References

- Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130).
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Qwen Team. Qwen2.5-VL Technical Report. [arXiv:2502.13923](https://arxiv.org/abs/2502.13923).
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. Robust Speech Recognition via Large-Scale Weak Supervision. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).
- BAAI. BGE-large English embedding model. [Hugging Face: BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).

