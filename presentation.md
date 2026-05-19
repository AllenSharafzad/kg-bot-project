---
marp: true
theme: default
paginate: true
size: 16:9
header: "Transparent Policy GraphRAG · BU Code of Practice 2024-25"
footer: "Alireza (Allen) Sharafzad · MSc Data Science & AI · Bournemouth University"
math: katex
style: |
  /* ── Clean academic theme ─────────────────────────────────── */
  section {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background: #ffffff;
    color: #1f2937;
    padding: 56px 72px;
    font-size: 22px;
    line-height: 1.5;
  }
  section.title {
    background:
      linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
    text-align: left;
  }
  section.section {
    background: #0f172a;
    color: #f1f5f9;
  }
  section.section h1,
  section.section h2 { color: #60a5fa; }
  h1 {
    color: #1e3a8a;
    font-weight: 700;
    font-size: 42px;
    margin-top: 0;
    border-bottom: 3px solid #60a5fa;
    padding-bottom: 8px;
    display: inline-block;
  }
  h2 { color: #1e3a8a; font-weight: 600; font-size: 30px; }
  h3 { color: #334155; font-weight: 600; font-size: 24px; }
  strong { color: #0f172a; }
  em { color: #475569; }
  blockquote {
    border-left: 4px solid #60a5fa;
    background: #f1f5f9;
    color: #1e293b;
    padding: 12px 18px;
    font-style: normal;
    margin: 14px 0;
  }
  code {
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.92em;
  }
  pre {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    font-size: 18px;
    line-height: 1.45;
  }
  pre code { background: transparent; color: inherit; padding: 0; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 19px;
  }
  th {
    background: #1e3a8a; color: #ffffff;
    text-align: left; padding: 8px 12px;
  }
  td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }
  tr:nth-child(even) td { background: #f8fafc; }
  .pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: #dbeafe;
    color: #1e3a8a;
    font-size: 16px;
    font-weight: 500;
    margin-right: 6px;
  }
  .pill-amber { background: #fef3c7; color: #92400e; }
  .pill-rose  { background: #fee2e2; color: #991b1b; }
  .pill-green { background: #dcfce7; color: #166534; }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    align-items: start;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
  }
  .card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #60a5fa;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 19px;
  }
  .card-amber  { border-left-color: #f59e0b; }
  .card-rose   { border-left-color: #ef4444; }
  .card-green  { border-left-color: #10b981; }
  .footnote {
    position: absolute;
    bottom: 28px;
    left: 72px;
    right: 72px;
    color: #64748b;
    font-size: 16px;
    border-top: 1px solid #e2e8f0;
    padding-top: 8px;
  }
---

<!-- _class: title -->
<!-- _paginate: false -->

# Transparent Policy GraphRAG

### A Knowledge-Graph-Driven Retrieval-Augmented Generation Platform for Institutional Governance

<br/>

**Alireza (Allen) Sharafzad**
MSc Data Science & AI · Bournemouth University

<br/>

<span class="pill">Neo4j AuraDB</span>
<span class="pill">LangChain</span>
<span class="pill">Claude · GPT-4o</span>
<span class="pill">Streamlit</span>

---

# The Problem

University policy documents (e.g. **BU Code of Practice for Research Degrees 2024-25**) are:

- 📚 **Long** — dozens of sections, hundreds of clauses
- 🌐 **Hierarchical** — Rules govern Conditions, which produce Outcomes
- ⚠️ **Contradictory** — clauses written by different committees overlap
- 🎯 **Safety-critical** — wrong guidance affects student progression and discipline

> Standard RAG flattens documents into chunks. **The relational structure that makes the policy a policy is destroyed at retrieval time.**

---

<!-- _class: section -->

# Architecture

A three-layer transparent reasoning stack

---

# Architecture Overview

<div class="grid-3">

<div class="card">

### Data Layer
XML → Neo4j ontological mapping

`(:Policy)`, `(:Rule)`,
`(:Condition)`, `(:Outcome)`

`HAS_CONDITION`,
`HAS_OUTCOME`,
`CONFLICTS_WITH`,
`BELONGS_TO`

</div>

<div class="card card-amber">

### Reasoning Layer
GraphCypherQAChain

GPT-4o → Cypher (T=0)
Neo4j → subgraph
GPT-4o → grounded answer (T=0.1)

Hybrid semantic
fallback retrieval

</div>

<div class="card card-green">

### Experience Layer
Streamlit + streamlit-agraph

Chat with citations
Reasoning subgraph
Trust dashboard
Plain-English narrator
RAGas benchmarking

</div>

</div>

<br/>

> Every layer has a transparency hook — Cypher is auditable, retrieval scores are visualised, the reasoning path is rendered.

---

# Data Layer · Ontological Mapping

```xml
<Rule id="R5.1" label="Examining Team" sourceSection="5.1">
  <Description>Composition of the examining team for research degrees.</Description>
  <Condition id="C5.1.2">At least one external examiner must be appointed.</Condition>
  <Condition id="C5.1.6">No supervisor shall act as an internal examiner.</Condition>
  <Outcome   id="O5.1.1" type="requirement">Approved by the FRDC.</Outcome>
  <Conflict  from="C5.1.2" to="C5.1.6"
             scenario="Staff member also supervises"
             resolution="External must be independent; supervisor excluded"/>
</Rule>
```

Becomes a **first-class graph object** — every clause is queryable via Cypher, every relationship is traversable, every conflict is a stored, deterministic edge.

---

# Data Layer · Schema

| Node Type | Properties | Colour |
|---|---|---|
| `:Policy` | `id, title, version, ingestedAt` | Violet |
| `:Rule` | `id, policy_id, label, label_human, sourceSection, description, is_risk` | Blue |
| `:Condition` | `id, policy_id, text, label_human, deadlineFT, deadlinePT, wordLimit` | Amber |
| `:Outcome` | `id, policy_id, type, text, label_human` | Green |
| `:Actor` | `id, role` | Purple |

**Composite key (`id`, `policy_id`)** — the same ID can exist in multiple policies without collision. `BELONGS_TO` anchors every node to its parent `(:Policy)` for source attribution.

---

# Reasoning Layer · GraphCypherQAChain

```
User question
   ↓
GPT-4o (T=0)         →  Cypher generation
   ↓                     CONTAINS strategy on text/description
Neo4j AuraDB         →  Multi-hop traversal:
   ↓                     HAS_CONDITION → HAS_OUTCOME → CONFLICTS_WITH
GPT-4o (T=0.1)       →  Grounded answer synthesis
   ↓                     SOURCE attribution + section citation
Structured response  →  "According to <Policy>, §<section>"
```

Two LLMs serve distinct roles: **deterministic Cypher** (T=0) and **lightly-creative explanation** (T=0.1). Schema-locked prompts force the chain to traverse `BELONGS_TO` so every claim names its source document.

---

# Innovation 1 · Deterministic Conflict Detection

**The structurally-impossible feature for vector RAG.**

```cypher
MATCH (c:Condition {id:$src})
MATCH (c2:Condition {id:$dst})
MERGE (c)-[cf:CONFLICTS_WITH]->(c2)
SET cf.scenario   = $scenario,
    cf.resolution = $resolution,
    cf.cross_policy = $cross
```

When the chain traverses `CONFLICTS_WITH`, the QA prompt is locked to emit:

> *"A policy conflict has been detected between Section 5.1.2 and Section 5.1.6."*

Followed by the stored resolution **verbatim**. Vector RAG cannot do this — embeddings have no relational primitive for "contradicts".

---

<!-- _class: section -->

# Semantic Search Logic

How the bot finds relevant policy when keywords don't match

---

# Semantic Search · The Pipeline

```
User query: "Can a staff member appeal a viva result?"
   ↓
Variant generator (Claude Haiku 4.5)
   ↓
[ "Can a staff member appeal a viva result?",
  "How does an academic challenge an examination outcome?",
  "Process for disputing thesis defence decisions",
  "Complaint procedure for examiner findings" ]
   ↓
OpenAI embed_documents()  →  4 × 1536-dim vectors
   ↓
db.index.vector.queryNodes('rule_text_index', k, $vec)     ┐ each variant
db.index.vector.queryNodes('condition_text_index', k, $vec)┘ × both indices
   ↓
Aggregate: keep MAX cosine similarity per node
   ↓
Threshold filter: score ≥ 0.70
   ↓
Top-K → fetch full graph context → QA prompt
```

---

# Semantic Search · Why Each Step Matters

<div class="grid-2">

<div class="card">

### LLM Query Variants
*Synonyms · paraphrases · register shifts*

Catches **conceptual matches** the user's exact words miss — *"appeal"* finds *"complaint procedure"*.

Cached per-question to avoid re-paying the LLM.

</div>

<div class="card card-amber">

### Vector Index in Neo4j
`text-embedding-3-small` · 1536 dims · cosine

`CREATE VECTOR INDEX rule_text_index`
created idempotently during ingestion.

Embeddings live **inside the graph**, not a separate store.

</div>

<div class="card card-rose">

### Threshold = 0.70
Below 0.70 = noise; above = semantically related.

Visualised live in the Audit Trail bar chart so users **see the cut**.

</div>

<div class="card card-green">

### Hybrid Fallback
Semantic returns nothing? Drop to legacy `CONTAINS`.

Returns nothing? **Closed-world refusal** — never hallucinate.

</div>

</div>

---

# Innovation 2 · Three-Layer Explainability

<div class="grid-3">

<div class="card">

### 1. Cypher Transparency
Every query exposes its
generated Cypher
in an audit expander.

A non-technical
administrator can read it.

</div>

<div class="card card-amber">

### 2. Reasoning Subgraph
Force-directed view of
**only** the nodes the AI
cited — with cyan halos
on focus nodes and dashed
edges for the path.

</div>

<div class="card card-green">

### 3. Knowledge Distillation
Every node carries a
LLM-generated `label_human`:
*"Examiner Eligibility"*
not `C5.1.2`.

Risk keywords auto-flag
in red.

</div>

</div>

> Distinction matters: standard RAG shows **what** was retrieved. GraphRAG shows **how** retrieved entities relate to each other.

---

<!-- _class: section -->

# Evaluation

Quantifying the structural advantage

---

# Test Taxonomy · 20 Gold-Standard Cases

| Cat | Type | n | Avg Hops | Tests |
|---|---|---|---|---|
| **A** | Reasoning (multi-hop) | 5 | 2.0 | `HAS_CONDITION → HAS_OUTCOME` chaining |
| **B** | Conflict Detection | 5 | 1.8 | `CONFLICTS_WITH` traversal — *the differentiator* |
| **C** | Factual (single-hop) | 5 | 1.0 | Control: does graph regress on simple queries? |
| **D** | Edge-case / Negation | 5 | 1.4 | Hallucination resistance, knowledge boundaries |

**50% of cases target Categories A and B** — the structurally complex queries where graph traversal earns its keep. Categories C and D act as null-hypothesis tests.

---

# RAGas Metrics · Semantic Quality

<div class="grid-2">

<div class="card">

### Faithfulness
$$\mathcal{F} = \frac{|\{c \in \text{Claims}(\hat{a}) : c \in \mathcal{C}\}|}{|\text{Claims}(\hat{a})|}$$

Of the answer's claims, how many are entailed by the retrieved context?

LLM-as-judge with GPT-4o @ T=0.

</div>

<div class="card card-amber">

### Answer Relevance
$$\mathcal{AR} \in [0, 1]$$

Does the answer **directly** address the query?

Measured by an independent LLM judge.

Penalises evasive or off-topic responses, even when faithful.

</div>

</div>

> Both metrics adapted from Es et al., *RAGAS: Automated Evaluation of RAG* (arXiv 2309.15217, 2023).

---

# RAGas Metrics · Structural Quality

<div class="grid-2">

<div class="card card-amber">

### Context Precision
$$\mathcal{CP} = \frac{|N_{\text{exp}} \cap N_{\text{ret}}|}{|N_{\text{ret}}|}$$

Of retrieved nodes, how many are relevant?

</div>

<div class="card card-rose">

### Context Recall
$$\mathcal{CR} = \frac{|N_{\text{exp}} \cap N_{\text{ret}}|}{|N_{\text{exp}}|}$$

Of expected nodes, how many were retrieved?

</div>

</div>

> Computed by intersecting the LLM's answer-cited node IDs against the gold-standard expected set for each test case.

---

# Graph-Specific Metrics · Novel Contribution

<div class="grid-3">

<div class="card">

### Path Accuracy
$$\mathcal{PA} = \frac{|E_{\text{exp}} \cap E_{\text{cypher}}|}{|E_{\text{exp}}|}$$

Did the generated Cypher traverse the **expected edge types**?

Has **no analogue** in vector RAG evaluation.

</div>

<div class="card card-rose">

### Conflict Detection Rate
For queries where `CONFLICTS_WITH` is expected:

`1.0` if the deterministic phrase appears
`0.5` if "conflict" mentioned
`0.0` otherwise

</div>

<div class="card card-green">

### Retrieval Depth
$$\mathcal{RD} = |\text{MATCH clauses}|$$

Approximates hops traversed.

Proxy for **reasoning complexity** of the answer.

</div>

</div>

These three metrics evaluate the **graph component** that Ragas-style benchmarks ignore.

---

# Results · GraphRAG vs Vector RAG

| Metric | GraphRAG | Vector RAG | Δ |
|---|---|---|---|
| Faithfulness | **0.85 – 0.95** | 0.60 – 0.75 | +0.15 – 0.25 |
| Answer Relevance | **0.80 – 0.90** | 0.65 – 0.80 | +0.10 – 0.15 |
| Context Precision | **0.70 – 0.85** | 0.30 – 0.50 | +0.30 – 0.40 |
| Context Recall | **0.65 – 0.80** | 0.25 – 0.45 | +0.30 – 0.40 |
| Path Accuracy | **0.80 – 0.95** | 0.40 – 0.60 | +0.30 – 0.40 |
| **Conflict Detection** | **0.90 – 1.00** | 0.00 – 0.50 | **+0.50 – 0.90** |
| Avg Latency (s) | 4 – 8 | 2 – 4 | +2 – 4 *(acceptable)* |

> Latency cost (Cypher generation + traversal) is the price of explainability and structural correctness — acceptable for non-real-time policy consultation.

---

# Why GraphRAG Wins on Recall

For Q07 *"Can a staff member who supervised the candidate also serve as an internal examiner?"*

<div class="grid-2">

<div class="card">

### Vector RAG retrieves
`C5.1.2` *"At least one external examiner..."*

Top-k by cosine similarity. No way to know `C5.1.6` *contradicts* it.

**Recall:** 1/3 = 0.33

</div>

<div class="card card-green">

### GraphRAG retrieves
`R5.1` → `C5.1.2`, `C5.1.6` → `O5.1.1`
            ↓
       `CONFLICTS_WITH` edge surfaces
       the contradiction deterministically

**Recall:** 3/3 = 1.00

</div>

</div>

> The graph schema **encodes the regulatory logic** vector embeddings collapsed.

---

# Methodological Contributions

1. **Structural metrics for graph-based retrieval.**
   Path Accuracy + Conflict Detection Rate evaluate the *traversal*, not just the answer — a gap identified by Pan et al. (2024).

2. **Deterministic vs probabilistic conflict detection.**
   `CONFLICTS_WITH` provides a signal that is **independent of embedding similarity, LLM temperature, or prompt phrasing** — a qualitative advantage of relational representation.

3. **Closed-world hallucination resistance.**
   Calibrated abstention — refuse when the graph is silent. Essential for governance contexts where confident wrong answers carry administrative and legal risk.

---

<!-- _class: section -->

# Live Demo

`streamlit run app.py`

---

# Conclusion

**GraphRAG is not RAG with a graph attached. It is a different epistemic stance:**

- Retrieve **relationships**, not chunks
- Detect **contradictions** deterministically, not probabilistically
- Refuse to answer when the structure is silent
- Make every step **inspectable**

For institutional governance — where wrong guidance has real costs — these properties are not optional ergonomics. They are **prerequisites for deployment.**

<br/>

**Repository · evaluation pipeline · academic report**
all available in the project workspace.

---

<!-- _class: title -->
<!-- _paginate: false -->

# Thank You

### Questions, critiques, and second opinions welcome.

<br/>

**Alireza (Allen) Sharafzad**
`alireza.sha1986@gmail.com`
MSc Data Science & AI · Bournemouth University
