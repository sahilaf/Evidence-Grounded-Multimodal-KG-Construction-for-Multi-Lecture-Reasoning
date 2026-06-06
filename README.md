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

The retrieval numbers are from a small starter gold set and should not be interpreted as final benchmark performance.

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

1. Download lecture videos and extract audio/frames.
2. Transcribe audio with Faster-Whisper `large-v3`.
3. Select high-recall semantic anchors.
4. Run EasyOCR on anchor frames.
5. Extract evidence-grounded concepts and relationships with Qwen2.5-VL.
6. Validate, canonicalize, and build a provenance-rich NetworkX knowledge graph.
7. Use hybrid retrieval and local graph context for GraphRAG-style QA.

## Main Notebook

Open this notebook in Colab Pro/A100:

[notebooks/Evidence_Grounded_Multimodal_KG_Construction.ipynb](notebooks/Evidence_Grounded_Multimodal_KG_Construction.ipynb)

Recommended first test before running the full VLM extraction:

```python
run_extraction('neural_net_01', limit_anchors=5)
```

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

