# Recent Advances in Retrieval-Augmented Generation: A Critical Comparison

**Author:** Alireza (Allen) Sharafzad
**Programme:** MSc Data Science & AI, Bournemouth University
**Supervisor:** Sofia
**Document type:** Literature positioning · prepared in response to supervisor feedback
**Date:** May 2026

---

## 1. Introduction

The original Retrieval-Augmented Generation framework [1] paired a dense-vector retriever with a generative LLM and quickly became the dominant pattern for grounding language models in external knowledge. Since 2024, however, multiple research groups have argued that pure embedding-similarity retrieval is structurally inadequate for tasks that require multi-hop reasoning, corpus-level abstraction, or the detection of relationships *between* retrieved units. This document surveys the most relevant recent approaches — Microsoft GraphRAG [2], LightRAG [3], HippoRAG [4], RAPTOR [5], and Self-RAG [6] — and positions the present project (*Transparent Policy GraphRAG*) against them.

The goal is **critical comparison**, not summary. Where our approach offers a genuine advantage we name it; where the state of the art surpasses what we have built we acknowledge it as an explicit gap.

---

## 2. Naive RAG and Its Structural Ceiling

The Lewis et al. baseline [1] retrieves the top-*k* text chunks by cosine similarity and conditions the generator on the concatenation. Two limitations recur in the recent literature:

- **No relational primitives.** Embeddings collapse the relationships *between* chunks into a single similarity scalar. A passage stating *"At least one external examiner is required"* and a passage stating *"No supervisor may act as an internal examiner"* may both be retrieved, but the retriever has no mechanism to encode that they jointly *constrain* a third clause about examiner appointment.
- **No global view.** Top-*k* retrieval cannot answer corpus-level questions ("What are the main themes of this document?") because no single chunk contains the answer.

These two failure modes motivate the structural and hierarchical extensions surveyed below.

---

## 3. Recent RAG Approaches

### 3.1 Microsoft GraphRAG (Edge et al., 2024) [2]

**Core idea.** An LLM extracts entities and relations from every chunk of a corpus, building a knowledge graph. The Leiden algorithm partitions the graph into a hierarchy of communities; each community receives an LLM-generated summary at multiple granularities. At query time, two modes are available:

- **Local search** retrieves entities near matched terms (closest to traditional RAG).
- **Global search** aggregates community summaries to answer corpus-wide thematic questions — *"What are the principal causes of X across the document set?"*

**Strengths.** Strong performance on **query-focused summarization** over large unstructured corpora. The community hierarchy gives the system a "view from altitude" that no chunk-level retriever has.

**Limitations.** The graph is built by LLM extraction over raw text, which is **expensive** (one LLM call per chunk) and **noisy** (the schema emerges bottom-up, varies between documents, and contains hallucinated relations). Microsoft's own benchmarking notes a substantial token cost compared to baseline RAG. There is no explicit primitive for **policy contradiction** or domain-specific reasoning — communities are detected by structural cohesion, not regulatory logic.

### 3.2 LightRAG (Guo et al., 2024) [3]

**Core idea.** A lighter-weight graph-RAG that retains the dual-level retrieval idea (low-level entities + high-level themes) but replaces the full Leiden hierarchy with a faster construction pipeline and supports **incremental updates** — new documents can be added without rebuilding the entire graph.

**Strengths.** Significant speed and cost improvements over Microsoft GraphRAG, with comparable answer quality on the published benchmarks. Incremental updates are practically important for any deployed system that ingests new content periodically.

**Limitations.** Like Microsoft GraphRAG, the graph schema is induced by LLM extraction — generic entities and relations rather than a domain-engineered ontology. Conflict detection, contradiction resolution, and policy-style first-class relationships are not addressed.

### 3.3 HippoRAG (Gutiérrez et al., 2024) [4]

**Core idea.** Inspired by hippocampal memory indexing theory: builds a knowledge graph and performs retrieval by running **Personalized PageRank** seeded from query-matched nodes. The PageRank distribution surfaces nodes that are multi-hop relevant rather than only one-hop similar.

**Strengths.** Explicit multi-hop retrieval without iterative re-prompting. Strong on benchmarks requiring entity association across distant passages.

**Limitations.** PageRank is structure-agnostic — it does not distinguish between, for example, a `HAS_CONDITION` edge and a `CONFLICTS_WITH` edge. Edge semantics are flattened into a single graph-walk probability.

### 3.4 RAPTOR (Sarthi et al., 2024) [5]

**Core idea.** Hierarchical clustering of text chunks combined with recursive LLM summarization at each cluster level. At query time the retriever can match against summaries at any depth, giving it a multi-resolution view of the document.

**Strengths.** Captures hierarchy without requiring graph construction. Useful when the corpus has a tree-like organisation (e.g., textbooks).

**Limitations.** Still operates on flat text — no explicit relations, no conflict primitive, no ability to answer relational questions like *"which of these clauses contradict each other?"*

### 3.5 Self-RAG (Asai et al., 2023) [6]

**Core idea.** Trains the LLM to emit special **reflection tokens** that decide *when* to retrieve, *what* to retrieve, and *whether the retrieved context supports the response*. The retrieval policy is learned, not heuristic.

**Strengths.** Strong calibration: reduces unnecessary retrievals and self-checks faithfulness.

**Limitations.** Does not change the underlying retrieval substrate — it is orthogonal to the question of how knowledge is structured.

---

## 4. Our Approach in Context

Our system, *Transparent Policy GraphRAG*, makes three commitments that distinguish it from the systems above. They are commitments, not improvements — each carries a tradeoff.

### 4.1 Schema-driven rather than extraction-driven

MS GraphRAG and LightRAG **construct** the graph from raw text via LLM extraction. The schema emerges bottom-up: nodes are *Person*, *Organization*, *Concept*, etc., and edges are whatever the LLM proposes.

We instead consume an **explicit XML schema**: `Rule → HAS_CONDITION → Condition → HAS_OUTCOME → Outcome`, with a first-class `CONFLICTS_WITH` edge carrying `scenario` and `resolution` properties. The schema is authored once (or generated by an LLM PDF→XML pipeline that is itself part of the system) and validated against the regulatory document.

**Tradeoff.** The schema does not generalise to arbitrary corpora; we cannot ingest a news archive without redesigning the ontology. In return we get **deterministic, queryable regulatory primitives** that no extraction-based system surfaces.

### 4.2 Conflict detection as a first-class primitive

Of the systems surveyed, **none** treat contradiction as a stored, queryable graph object. MS GraphRAG can describe conflicts in a community summary if the LLM happens to surface them; HippoRAG cannot encode them at all; LightRAG inherits MS GraphRAG's blind spot.

In our system, when the user asks *"can a staff member who supervised the candidate also serve as an internal examiner?"*, the Cypher query traverses a `CONFLICTS_WITH` edge between two stored Conditions and the QA prompt is locked to emit the deterministic phrase

> *"A policy conflict has been detected between Section 5.1.2 and Section 5.1.6"*

followed by the stored resolution verbatim. The signal is **independent** of embedding similarity, LLM temperature, and prompt phrasing.

This is the most significant differentiator. It is achievable specifically *because* we have committed to a domain-engineered schema; it is not a feature MS GraphRAG could easily add without abandoning its bottom-up extraction philosophy.

### 4.3 Closed-world abstention

When the graph contains no matching subgraph for a query, our system returns a calibrated refusal:

> *"This information is not available in the current policy graph. Please consult the BU Doctoral College directly."*

MS GraphRAG and LightRAG operate under an open-world assumption: any vector retriever returns *something*, and the LLM will synthesise an answer from whatever was retrieved. For governance applications this is unacceptable — a confident but unfounded answer about, say, re-enrolment policy carries administrative and legal risk. Self-RAG [6] addresses this through learned reflection tokens, which is a complementary but more complex solution.

---

## 5. Comparison Table

| Property | Naive RAG [1] | RAPTOR [5] | MS GraphRAG [2] | LightRAG [3] | HippoRAG [4] | **Ours** |
|---|---|---|---|---|---|---|
| Knowledge structure | Flat chunks | Hierarchical summaries | Extracted KG | Extracted KG | Extracted KG | **Schema-driven KG** |
| Schema authorship | None | None | LLM extraction | LLM extraction | LLM extraction | **Human + LLM PDF→XML** |
| Multi-hop retrieval | ✗ | ✗ (hierarchical only) | ✓ (community) | ✓ (dual-level) | ✓ (PageRank) | **✓ (Cypher traversal)** |
| Corpus-level summarisation | ✗ | ✓ | ✓ (global mode) | ✓ | ✗ | ✗ |
| Conflict / contradiction | ✗ | ✗ | implicit | implicit | ✗ | **✓ (explicit edge)** |
| Closed-world refusal | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Source citation | per chunk | per cluster | per community | per node | per node | **per `(BU CoP §x.y.z)`** |
| Incremental updates | ✓ | ✗ | ✗ | **✓** | partial | ✓ (per-policy MERGE) |
| Indexing cost | low | medium | **high** | medium | medium | medium |
| Domain generality | high | high | high | high | high | **low (regulatory)** |

---

## 6. Limitations and Gaps in Our Approach

A credible positioning requires honest acknowledgement of where the state of the art exceeds what we have built.

1. **No corpus-level summarisation.** We do not implement community detection or hierarchical summarisation. We cannot answer *"What are the central themes of the BU Code of Practice?"* — only structured questions about specific clauses. MS GraphRAG and RAPTOR address this and we do not.

2. **Domain-bound ontology.** Our `Rule / Condition / Outcome / Conflict` schema is regulatory-document-specific. It does not transfer to medical literature, news archives, or scientific papers without redesign. MS GraphRAG and LightRAG are domain-agnostic by construction.

3. **Manual or semi-manual schema authorship.** We provide an LLM-assisted PDF→XML pipeline, but the schema choices (what counts as a Rule? when is a sentence an Outcome?) still require human judgement. Pure extraction-based systems avoid this at the cost of schema noise.

4. **No incremental updates beyond per-policy MERGE.** LightRAG is more sophisticated here — it can incorporate new documents into an existing graph without rebuilding. We support per-policy ingestion and deletion, but not finer-grained streaming updates.

5. **Vector indices were retrofitted.** Our semantic search layer (Neo4j vector index + LLM query variants + 0.70 threshold) was added after the structural pipeline was built. A from-scratch design starting with hybrid retrieval might be cleaner.

---

## 7. Conclusion

The recent generation of graph-augmented RAG systems — Microsoft GraphRAG, LightRAG, HippoRAG — share a common belief that flat embedding retrieval is insufficient for reasoning-intensive tasks. They differ in *how* they construct and traverse the graph, but they all rely on **bottom-up LLM extraction over generic entity types**.

Our system takes a different position: for regulatory documents specifically, the regulatory logic is already a graph in the author's mind, and encoding it explicitly as a domain ontology yields capabilities that emergent extraction cannot replicate — most notably, **deterministic conflict detection** and **closed-world abstention**. These are not engineering improvements over MS GraphRAG; they are different commitments suited to a different problem.

The honest positioning of this work is therefore:

> *We do not improve on Microsoft GraphRAG or LightRAG at corpus-wide summarisation. We address a problem they cannot — clause-level regulatory interpretation with first-class contradiction primitives — and we do so at the price of domain generality.*

This framing both honours the state of the art and stakes a defensible claim for a niche the surveyed systems do not occupy.

---

## References

[1] P. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 33, pp. 9459–9474.

[2] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, C. Mody, S. Truitt, J. Larson (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130. Open-source implementation: github.com/microsoft/graphrag.

[3] Z. Guo, L. Xia, Y. Yu, T. Ao, C. Huang (2024). *LightRAG: Simple and Fast Retrieval-Augmented Generation.* arXiv:2410.05779. Code: github.com/HKUDS/LightRAG.

[4] B. J. Gutiérrez, Y. Shu, Y. Gu, M. Yasunaga, Y. Su (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* arXiv:2405.14831.

[5] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, C. D. Manning (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* ICLR 2024. arXiv:2401.18059.

[6] A. Asai, Z. Wu, Y. Wang, A. Sil, H. Hajishirzi (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* arXiv:2310.11511.

[7] S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, X. Wu (2024). *Unifying Large Language Models and Knowledge Graphs: A Roadmap.* IEEE Transactions on Knowledge and Data Engineering 36(7), pp. 3580–3599.

[8] S. Es, J. James, L. Espinosa-Anke, S. Schockaert (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
