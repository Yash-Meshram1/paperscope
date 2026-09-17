# PaperScope

A local research-paper explorer for asking questions and comparing papers. Answers are grounded in retrieved passages, with citations that link back to the evidence shown in the app.

PaperScope combines dense retrieval, BM25, reciprocal rank fusion, and a cross-encoder reranker. Questions that name a paper are routed to that paper; comparison questions retrieve evidence from each named paper within a shared result budget.

## How it works

```text
PDFs → Docling → sections → chunks → embeddings + Qdrant

Question → paper-name routing → dense search + BM25
         → rank fusion → reranking → Ollama → answer + source cards
```

- **Embedding:** Qwen3-Embedding-0.6B, 1024 dimensions
- **Search:** local Qdrant and BM25, combined with reciprocal rank fusion
- **Reranking:** Qwen3-Reranker-0.6B
- **Generation:** Qwen3.5-9B through Ollama
- **Interface:** Streamlit, with cached search resources and expandable source passages

Example questions:

- How does DPR retrieve passages?
- Compare Self-RAG and CRAG.
- What does Lost in the Middle say about long contexts?

## Setup

Use Python 3.12 and install Ollama separately. From the repository root:

```bash
python -m venv .venv
# Activate the environment: .venv\Scripts\Activate.ps1 on Windows
# or source .venv/bin/activate on Linux/macOS
python -m pip install -r requirements-lock-cpu.txt
ollama pull qwen3.5:9b
```



Download the retrieval models once:

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
CrossEncoder("Qwen/Qwen3-Reranker-0.6B")
```

### Local data

PDFs, extracted text, vectors, and database files are not included. The app expects:

```text
data/processed/chunks.json
data/processed/embeddings.npy
storage/qdrant/                 # collection: research_papers
```

The notebooks document parsing, section modeling, chunking, and indexing. To work through them, install `jupyterlab` and `docling`, place the PDFs below in `data/papers/`, and start Jupyter from `notebooks/`. They contain exploratory cells, so review them before running. The embedding notebook uses CUDA and recreates its collection; use it for preparing a fresh index, with the app closed. Change its model device to `cpu` if needed.

The development corpus contains these ten papers:

| Paper | Local PDF filename |
|---|---|
| Retrieval-Augmented Generation — Lewis et al. | `01_rag_lewis_2020.pdf` |
| Dense Passage Retrieval | `02_dpr_karpukhin_2020.pdf` |
| ColBERT | `03_colbert_khattab_2020.pdf` |
| HyDE | `04_hyde_gao_2023.pdf` |
| Retrieval-Augmented Generation for Large Language Models: A Survey | `05_rag_survey_gao_2023.pdf` |
| Self-RAG | `06_self_rag_asai_2023.pdf` |
| Lost in the Middle | `07_lost_in_middle_liu_2023.pdf` |
| RAPTOR | `08_raptor_sarthi_2024.pdf` |
| Corrective Retrieval Augmented Generation | `09_crag_yan_2024.pdf` |
| Qwen3 Embedding | `10_qwen3_embedding_2025.pdf` |

### Run the app

```bash
python -m streamlit run streamlit_app.py
```

Startup loads existing artifacts and cached models; it does not index papers. Close notebooks using the same Qdrant directory before starting the app.

| Environment variable | Default |
|---|---|
| `PAPERSCOPE_ROOT` | Repository root |
| `PAPERSCOPE_DEVICE` | `cpu` |
| `PAPERSCOPE_OLLAMA_MODEL` | `qwen3.5:9b` |
| `PAPERSCOPE_OLLAMA_URL` | `http://localhost:11434/api/chat` |

## Retrieval behavior

`top_k` is the maximum number of passages across the whole response. A single named paper can fill that budget. Comparisons reserve one passage per available named paper, then fill remaining slots by ranking score. If the budget cannot cover every paper, the highest-scoring paper winners are selected.

Generic questions keep a two-passage-per-paper limit and may return fewer results. Unrecognized names fall back to full-corpus retrieval; recognized names in a mixed question still apply a filter. References and acknowledgments are excluded. Source numbers follow the exact evidence order sent to generation.

## Evaluation

The benchmark contains 30 single-paper questions and 10 comparisons. Relevance labels identify expected papers, not individual supporting passages.

| Earlier pipeline comparison: 30 questions | MRR@5 |
|---|---:|
| Dense | 0.858 |
| BM25 | 0.844 |
| Hybrid RRF | 0.953 |
| Reranked | 1.000 |

The application pipeline run on 14 September 2026 achieved **1.000 MRR@5** on the 30 single-paper questions and **1.000 paper Recall@5** on the 10 comparisons. Query-level results and runtime details are saved in `data/evaluation/benchmark_run.json`. The evaluator independently scores only the first five passages.

These are small development-set results. Paper-name routing directly affects paper coverage, and the earlier comparison used different candidate budgets. The scores do not measure answer correctness or citation support. `retrieval_metrics.json` preserves the earlier saved metrics separately; its multi-paper value has no matching historical query-level run.

Run tests without the paper corpus:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

On Linux/macOS, use `PYTHONPATH=src python -m unittest discover -s tests -v`. To evaluate a prepared local index:

```bash
python scripts/evaluate.py --k 5
```

New evaluation runs are saved under `data/evaluation/runs/` without replacing previous results.

## Project structure

```text
streamlit_app.py       Web interface
src/paperscope/        Parsing, chunking, retrieval, generation, and resource loading
notebooks/            Data preparation and retrieval experiments
scripts/evaluate.py   Application retrieval benchmark
tests/                Retrieval, artifact validation, and Streamlit tests
data/evaluation/      Questions and recorded results
```

Notebook outputs are omitted from this repository. Source page ranges are inherited from sections, aliases come from a fixed registry, and citation support is not automatically verified. The app uses embedded Qdrant with one resource owner per process.
