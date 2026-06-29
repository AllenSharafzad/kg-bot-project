# Transparent Policy GraphRAG

> A Streamlit application that ingests university policy documents (PDF or XML), maps them into a **Neo4j knowledge graph**, and answers natural-language questions with **grounded citations, deterministic conflict detection, and a visual reasoning trace**.

**Author:** Alireza (Allen) Sharafzad · MSc Data Science & AI, Bournemouth University
**Corpus:** BU 8A *Code of Practice for Research Degrees 2024-25*

---

## Why this exists

Flat, chunk-and-embed RAG retrieves text that is *semantically similar* to a question — but on dense compliance documents "similar" is not "correct." A PhD word-limit clause and an MPhil word-limit clause are near-identical in embedding space yet **mutually exclusive** in regulation. This platform replaces flat chunking with a **schema-injected knowledge graph**: every rule, condition, and outcome is a typed node with a provenance-bearing ID, so retrieval traverses *structure*, not just *proximity* — and every answer can cite the exact section it came from.

---

## Features

- **PDF → Knowledge Graph pipeline** — upload a policy PDF; it is extracted, converted to strict schema-conformant XML, and mapped into Neo4j with idempotent, provenance-anchored node IDs (e.g. `C_13_1_LawThesisWordLimit`).
- **Modular-RAG query path** — Adaptive routing (`DIRECT_LOOKUP` vs `COMPLEX_REASONING`) → structured Cypher first → hybrid fallback → CRAG validation → grounded answer.
- **Hybrid retrieval** — dual-stream **Reciprocal Rank Fusion (RRF, k=60)** combining dense vector similarity (OpenAI embeddings) with strict keyword indices, so rare statutory terms are never dropped by embedding dilution.
- **CRAG verification layer** — an automated relevance gate (`evaluate_context_relevance`) that refuses to synthesise an answer from weak context rather than hallucinating.
- **Visual reasoning trace** — a "the AI looked here" subgraph showing exactly which nodes grounded each answer, with risk nodes flagged red.
- **Live trust dashboard + RAGAS benchmarking** — transparency metrics and a GraphRAG-vs-SimpleRAG gold-dataset comparison.
- **Resilient by design** — OS-truststore SSL for corporate/university proxies, an offline local-embedding fallback, self-healing cached connections, and a consistent fail-loud / fail-open error contract.

---

## Architecture overview — the 5 macro layers

`app.py` is a single-file application organised into five macro-layers (see the banner comments in the source):

| Layer | Responsibility | Key functions / constants |
|---|---|---|
| **1 · Config & Environment** | Page setup, credential resolution, embedding-backend selection, LLM prompt contracts | `get_cred`, `_init_embedder`, `EMBEDDING_DIMS`, `SEMANTIC_THRESHOLD`, `CYPHER_GENERATION_TEMPLATE`, `QA_SYSTEM_TEMPLATE` |
| **2 · Infrastructure** | Cached, self-healing connections to external services | `init_neo4j`, `init_chain` |
| **3 · Ingestion** | Write path: PDF → strict XML → namespaced graph | `process_pdf_to_xml`, `_merge_xml_fragments`, `ingest_xml_to_neo4j`, `generate_human_label` |
| **4 · Query & Retrieval** | Read path: route → Cypher → Fusion/RRF → CRAG → grounded answer, plus reasoning-view rendering & benchmarking | `run_query`, `_semantic_search`, `keyword_fallback_search`, `reciprocal_rank_fusion`, `fetch_reasoning_subgraph`, `benchmark_question` |
| **5 · UI / Streamlit** | Top-level script: session state, sidebar controls, main trust/analytics panel | `_init_state`, sidebar block, main panel |

```
PDF ──► process_pdf_to_xml ──► <Policy> XML ──► ingest_xml_to_neo4j ──► Neo4j graph
                                                                            │
 question ──► run_query ──► route ──► Cypher chain ─┐                       │
                                                    ├──► CRAG gate ──► grounded answer + Reasoning View
                              hybrid RRF fallback ──┘                       ▲
                                                              vector + keyword indices
```

**Companion module:** `graphrag_policy_bot.py` is the CLI prototype that also hosts the shared Fusion-RAG / CRAG / Adaptive-routing helpers (`generate_multiple_queries`, `reciprocal_rank_fusion`, `evaluate_context_relevance`, `route_query`) imported by `app.py`.

---

## Tech stack

- **Python 3.11**, **Streamlit 1.35+**
- **Neo4j AuraDB** (cloud knowledge graph)
- **LangChain**: `langchain-neo4j`, `langchain-openai`, `langchain-core`
- **Models (OpenAI):** GPT-4o (Cypher generation @ T=0, QA synthesis @ T=0.1, PDF→XML extraction @ T=0), GPT-4o-mini (semantic node labelling), `text-embedding-3-small` (1536-dim cosine vector index)
- **Offline fallback:** `all-MiniLM-L6-v2` (384-dim) via HuggingFace, auto-selected when the OpenAI endpoint is unreachable
- **PyMuPDF** (PDF text extraction) · **streamlit-agraph + Altair + pandas** (visualisation)

---

## Installation & Setup

### 1. Clone and create an environment
```bash
git clone https://github.com/AllenSharafzad/kg-bot-project.git
cd kg-bot-project
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r Requirements.txt
```

### 3. Provide credentials
Create a `.env` file in the project root (never commit it — it is git-ignored):
```dotenv
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j
```
> Credentials resolve from the environment first, then from `st.secrets`, so the same code runs locally (`.env`) and on Streamlit Cloud unchanged. Semantic search needs `OPENAI_API_KEY`; without it the app still answers via structured Cypher and the local embedding fallback.

---

## How to run
```bash
streamlit run app.py
```
The app opens in your browser. The sidebar shows live connection status and a **🔄 Reconnect** button (useful on an AuraDB free-tier cold start).

**Quick health check (headless):**
```bash
python -m py_compile app.py            # syntax/compile check
streamlit run app.py --server.headless true
```

---

## How to ingest a new policy

The platform is intentionally **single-corpus**: ingesting a new policy **replaces** the current graph (the ingester wipes existing nodes for a clean rebuild).

1. Launch the app and open the **sidebar ingest panel**.
2. **Upload a policy PDF.** The pipeline runs automatically:
   `process_pdf_to_xml` → PyMuPDF text → 15k-char overlapping chunks → GPT-4o structured XML (T=0) → dedup/merge → well-formedness gate.
3. The validated XML is passed to `ingest_xml_to_neo4j`, which MERGEs a `(:Policy)` root plus `(:Rule)`/`(:Condition)`/`(:Outcome)` nodes and their relationships, adds human-readable labels, risk flags, and vector embeddings, and rebuilds the vector indices at the live embedding dimension.
4. Watch the **live process log** for per-step status and the final stats summary (rules / conditions / outcomes / errors).

> Image-only scans with no OCR layer fail loudly (`RuntimeError`) rather than ingesting a blank graph — that is by design.

---

## Evaluation

```bash
# Quick mode (5 questions)
python ragas_evaluation.py --quick

# Full gold-standard set (20 questions, see evaluation_dataset.py)
python ragas_evaluation.py
```
The in-app **RAGAS Benchmark** panel (§7.6) compares GraphRAG against a flat SimpleRAG baseline on context precision, context recall, path accuracy, and faithfulness.

---

## How to extend the system (maintainability notes)

The bottom of `app.py` carries a detailed **`# EXTENSIBILITY & FUTURE WORK`** block. In brief:

- **Add a new edge type** (e.g. `SUPERSEDES`): register it in `CYPHER_GENERATION_TEMPLATE` (definition + `OPTIONAL MATCH` + a worked example), MERGE it in `ingest_xml_to_neo4j`, and — if it must be cited — add a grounding rule to `QA_SYSTEM_TEMPLATE`. Keep the two `CYPHER_GENERATION_TEMPLATE` copies (`app.py` and `graphrag_policy_bot.py`) in sync.
- **Swap the embedding backend:** edit `_init_embedder` (model + dimension); **re-ingest** afterwards so the vector indices rebuild at the new width.
- **Tune retrieval:** `SEMANTIC_THRESHOLD` (0.70, COMPLEX mode only) and the RRF constant `k=60` — change them alongside a RAGAS re-run so every tuning change is measured.
- **Go multi-policy (future work):** remove the clean-slate wipe in `ingest_xml_to_neo4j` and rely on the existing `policy_id` namespace; `get_ingested_policies` / `delete_policy` already operate per policy.

### Project layout
| Path | What it is |
|---|---|
| `app.py` | The whole product — single-file Streamlit application |
| `graphrag_policy_bot.py` | CLI prototype + shared Fusion-RAG / CRAG / routing helpers |
| `evaluation_dataset.py` | 20-question gold-standard test set |
| `ragas_evaluation.py` | Standalone evaluation pipeline |
| `Requirements.txt` | Python dependencies |
| `.env` | Local credentials — **never committed** |
| `CLAUDE.md` | Cross-laptop / git workflow notes |

---

## License & data note

Policy source PDFs are copyrighted institutional material and are **not** committed to this repository. The code is provided for academic and research purposes.
