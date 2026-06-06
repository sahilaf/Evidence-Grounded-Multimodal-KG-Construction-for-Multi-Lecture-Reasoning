# Evidence-Grounded Multimodal KG Construction for Multi-Lecture Reasoning

This repository contains a Colab-ready pipeline for constructing an evidence-grounded multimodal knowledge graph from lecture videos and using it for multi-lecture reasoning.

The project focuses on **grounded educational knowledge extraction**: every concept and relationship is tied back to lecture evidence such as transcript windows, OCR text, frame timestamps, anchor ids, and confidence scores.

## Dataset

The processed dataset is hosted on Hugging Face:

[sahilfarib/evidence-grounded-multimodal-kg-multi-lecture-reasoning](https://huggingface.co/datasets/sahilfarib/evidence-grounded-multimodal-kg-multi-lecture-reasoning)

The dataset includes raw videos/audio/frames plus processed transcripts, anchors, OCR, VLM extractions, validated concept mentions, canonical concepts, canonical relationships, graph exports, QA outputs, and evaluation scaffolds.

## Results Snapshot

The current run uses three 3Blue1Brown neural-network lectures.

| Metric | Value |
|---|---:|
| Lectures | 3 |
| Extracted frames | 3,118 |
| Transcript segments | 756 |
| Transcript words | 9,197 |
| Semantic anchors | 559 |
| Anchors with non-empty OCR | 529 |
| Raw concept mentions | 1,155 |
| Validated concept mentions | 1,022 |
| Canonical concepts | 172 |
| Raw relationships | 400 |
| Validated relationship mentions | 312 |
| Final graph relationships | 282 |
| Relationship endpoint coverage | 90.38% |
| Starter retrieval top-1 accuracy | 100% |
| Starter retrieval top-3 accuracy | 100% |
| Starter retrieval mean top-5 recall | 100% |

The retrieval numbers are from a small starter gold set and should not be interpreted as final benchmark performance. The paper draft discusses this limitation.

## Repository Structure

```text
.
├── notebooks/
│   ├── Evidence_Grounded_Multimodal_KG_Construction.ipynb
│   └── Full_Pipeline_legacy.ipynb
├── docs/
│   ├── draft.md
│   ├── knowledge_base.md
│   └── plan.md
├── make_evidence_notebook.py
├── requirements.txt
├── config.example.json
├── CITATION.cff
├── LICENSE
└── README.md
```

Large generated artifacts are not committed to Git. They are available on Hugging Face.

## Pipeline Overview

1. **Video preprocessing**
   - Download lecture videos.
   - Extract audio and 1 FPS frames.
   - Save all run artifacts in a reproducible directory.

2. **ASR transcription**
   - Transcribe lectures with Faster-Whisper `large-v3`.
   - Align transcript segments to frame timestamps.

3. **Semantic anchor selection**
   - Select high-recall anchors using visual frame changes, transcript cues, relationship cues, and first mentions.
   - Attach local transcript windows to each anchor.

4. **OCR**
   - Run EasyOCR on selected anchor frames.
   - Preserve OCR text per anchor.

5. **Evidence-grounded VLM extraction**
   - Use Qwen2.5-VL-7B-Instruct.
   - Extract only concepts and relationships supported by transcript, OCR, or visual evidence.
   - Empty anchor outputs are allowed.

6. **Validation and canonicalization**
   - Filter unsupported or low-confidence concepts.
   - Normalize concept names.
   - Deduplicate canonical concepts.
   - Canonicalize relationship endpoints after concept deduplication.

7. **Knowledge graph construction**
   - Build a NetworkX `MultiDiGraph`.
   - Store provenance on every node and edge.

8. **GraphRAG QA**
   - Retrieve concepts using hybrid exact, alias, fuzzy, semantic, and evidence-count scoring.
   - Extract a local subgraph.
   - Generate answers grounded in graph context.

## Main Notebook

Open this notebook in Colab Pro/A100:

[notebooks/Evidence_Grounded_Multimodal_KG_Construction.ipynb](notebooks/Evidence_Grounded_Multimodal_KG_Construction.ipynb)

Recommended first test before running the full VLM extraction:

```python
run_extraction('neural_net_01', limit_anchors=5)
```

Then run the full extraction once the prompt behavior looks reasonable.

## Models and Tools

| Component | Model / Tool |
|---|---|
| ASR | Faster-Whisper `large-v3` |
| OCR | EasyOCR |
| VLM extraction | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Embeddings | `BAAI/bge-large-en-v1.5` |
| Graph | NetworkX `MultiDiGraph` |

## Paper Draft

The current paper draft is available at:

[docs/draft.md](docs/draft.md)

Supporting notes:

- [docs/knowledge_base.md](docs/knowledge_base.md)
- [docs/plan.md](docs/plan.md)

## Limitations

- The current evaluation uses only three lectures.
- The starter retrieval evaluation is too small for strong benchmark claims.
- Concept canonicalization still leaves singular/plural duplicates such as `weight` and `weights`.
- Some isolated nodes remain and should be pruned or linked.
- Final publication claims require a manually annotated gold set for concept F1, relation F1, QA faithfulness, and ablations.

## Citation

If you use this code or dataset, cite the repository metadata in `CITATION.cff`.

## License

Code is released under the MIT License. Check the original lecture/video source terms before redistributing derived media in other contexts.

