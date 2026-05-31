# State-of-the-Project & Architecture Review
### Transparent Policy GraphRAG — BU Policy Knowledge Platform

**Author:** Alireza (Allen) Sharafzad · MSc Data Science & AI, Bournemouth University
**Document compiled:** 2026-05-31
**Scope:** Full technical state of `app.py` and `graphrag_policy_bot.py` after the OpenAI migration, the connection/packaging fixes, the policy-listing Cypher fix, and the `streamlit-agraph` render fix.

> **Verification note (read first).** Every count, model name, parameter, and code location in this document was read directly out of the source files and queried against the live Neo4j AuraDB instance on 2026-05-31 — not taken from memory or from the figures in the original request. Where the requested figures disagreed with reality, the discrepancy is documented in **Section 6** rather than reproduced. Treat this document as the source of truth over any earlier verbal numbers.

---

## 1. Executive Pipeline Overview

The platform is a **Modular RAG** system: retrieval is decomposed into independently swappable layers (routing → primary structured retrieval → semantic/keyword fallback → corrective validation → grounded synthesis), rather than a single monolithic "embed-and-stuff-the-prompt" chain. There are two distinct macro-flows.

### 1a. Ingestion flow (document → knowledge graph)

```
PDF upload (Streamlit sidebar)
   │
   ▼
_extract_pdf_text()              PyMuPDF (fitz) — raw text extraction
   │
   ▼
_chunk_text()                    fixed 15,000-char windows, 1,500-char overlap
   │
   ▼
process_pdf_to_xml()             per-chunk GPT-4o (T=0) with PDF_TO_XML_SYSTEM
   │   → raw XML fragments (one per chunk)
   ▼
_merge_xml_fragments()           dedup by tag::id, reject generic IDs, merge children
   │
   ▼
_wrap_xml_fragment() + ET.fromstring()   sanitise + validate well-formedness
   │
   ▼
ingest_xml_to_neo4j()
   │   1. MATCH (n) DETACH DELETE n          ← mandatory clean-slate wipe
   │   2. MERGE (:Policy)                     ← namespace root
   │   3. DROP+CREATE vector indices          ← rule_text_index, condition_text_index
   │   4. per Rule/Condition/Outcome:
   │        - MERGE node (scoped by policy_id)
   │        - generate_human_label()  (gpt-4o-mini)
   │        - embed text (text-embedding-3-small)
   │        - MERGE relationships
   ▼
Neo4j AuraDB  (Policy → Rule → Condition / Outcome, plus CONFLICTS_WITH)
```

### 1b. Query flow (question → grounded, cited answer)

```
User question
   │
   ▼
_clean_user_query()              strip whitespace / UI leakage
   │
   ▼
_route_query()  (Adaptive RAG)   GPT-4o (T=0) → DIRECT_LOOKUP | COMPLEX_REASONING
   │                             sets use_fusion = (route == COMPLEX_REASONING)
   ▼
GraphCypherQAChain.invoke()      PRIMARY path:
   │   - GPT-4o (T=0)  generates Cypher from live schema
   │   - Neo4j executes traversal
   │   - GPT-4o (T=0.1) synthesises answer from returned rows
   ▼
Was the primary result empty / "not available"?
   │                         │
   │ no                      │ yes
   ▼                         ▼
post-hoc CRAG gate     keyword_fallback_search() → _semantic_search()
on primary rows           - COMPLEX: multi-query Fusion RAG + RRF
   │                       - DIRECT : dual-stream (vector + keyword) RRF, top-5
   │                          │
   │                          ▼
   │                       CRAG gate: evaluate_context_relevance()  (GPT-4o T=0)
   │                          RELEVANT → _ground_answer_from_rows()
   │                          else     → _CRAG_REFUSAL
   ▼                          ▼
   └──────────► answer + audit metadata ◄────────┘
                   │
                   ▼
extract_ids_from_text() + resolve_section_refs_to_ids()
                   │
                   ▼
fetch_reasoning_subgraph() → streamlit-agraph Reasoning View
```

### 1c. Model inventory (all OpenAI as of this build)

| Role | Model | Temperature | Where |
|---|---|---|---|
| Cypher generation | `gpt-4o` | 0.0 | `init_chain()` (cypher_llm) |
| QA synthesis | `gpt-4o` | 0.1 | `init_chain()` (qa_llm) |
| PDF → XML extraction | `gpt-4o` | 0.0 (max_tokens 8000) | `process_pdf_to_xml()` |
| Adaptive routing | `gpt-4o` | 0.0 (max_tokens 50) | `route_query()` |
| Query expansion (Fusion) | `gpt-4o` | 0.0 (max_tokens 400) | `generate_multiple_queries()` |
| CRAG validation | `gpt-4o` | 0.0 (max_tokens 80) | `evaluate_context_relevance()` |
| Semantic node labels | `gpt-4o-mini` | 0.0 (max_tokens 20) | `generate_human_label()` |
| Plain-English narrator + search variants | `gpt-4o-mini` | 0.2 (max_tokens 120) | `_init_narrator()` |
| Embeddings | `text-embedding-3-small` | — | `_init_embedder()` (1536-dim cosine) |

> Offline/proxy fallback for embeddings: if the OpenAI embedder fails its live `embed_query("probe")` test, the code falls back to `all-MiniLM-L6-v2` (384-dim) via `langchain_huggingface` / `langchain_community`. The vector indices are created at whichever dimension is active (`EMBEDDING_DIMS`).

---

## 2. The PDF-to-XML Transformation Layer

This layer turns an unstructured, multi-page policy PDF into strict, schema-conformant XML that the Neo4j ingester can map deterministically. It is the most failure-prone part of any document-to-KG pipeline, so it is built defensively.

### 2a. `process_pdf_to_xml(pdf_file, log_fn)`

Responsibilities, in order:

1. **Text extraction.** Reads the uploaded bytes and calls `_extract_pdf_text()` (PyMuPDF / `fitz`). Raises `RuntimeError` if no extractable text is found (e.g. an image-only scan with no OCR layer).
2. **Chunking.** Calls `_chunk_text(text, chunk_size=15_000, overlap=1_500)`. The 15k window is deliberately small: GPT-4o produces *valid, complete* XML far more reliably on bounded inputs than on a whole document, and the 1.5k overlap guarantees that a section header split across a chunk boundary still appears intact in at least one chunk.
3. **LLM extraction loop.** A single `ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=8000)` instance is constructed **once** (OpenAI-only — no Anthropic branch remains). Each chunk is sent with the `PDF_TO_XML_SYSTEM` prompt plus a per-chunk instruction: *"Derive all element IDs from section numbers visible in THIS chunk's text."* Empty responses are skipped with a warning; an LLM exception is logged with a truncated traceback and re-raised (fail-loud — the user must know extraction failed).
4. **Merge.** If only one chunk was produced, its fragment is used directly. Otherwise `_merge_xml_fragments()` deduplicates and stitches the fragments.
5. **Wrap + validate.** `_wrap_xml_fragment()` produces a single `<Policy>…</Policy>` document; `ET.fromstring()` is the final gate. A parse failure logs the exact offending line and re-raises, so malformed XML never reaches Neo4j.

### 2b. `_merge_xml_fragments(raw_fragments, log_fn)`

Because chunks overlap, the **same** `Rule` can legitimately appear in two adjacent fragments. Naive concatenation would create duplicate nodes; dropping later copies would lose conditions that only appeared in the second chunk. The merger resolves both:

- **Sanitisation** (`_sanitise`): strips Markdown code fences (` ```xml `), strips an `<?xml …?>` prolog, escapes bare `&` that are not already valid entities, and removes non-XML control characters (`\x00–\x08`, `\x0b`, `\x0c`, `\x0e–\x1f`).
- **Parsing:** each fragment is wrapped in `<root>…</root>` unless it already starts with `<Policy …>`, then parsed with `ElementTree`. Parse failures are counted and skipped (one bad chunk never aborts the whole merge).
- **Top-level walk:** only direct-child `<Rule>` elements are collected; nested sub-rules ride along inside their parent and are not double-counted.
- **Deduplication key:** `tag + "::" + id`. Compound section-anchored IDs (see 2c) make collisions between genuinely different elements effectively impossible.
- **Child re-merge:** when a `Rule` id is seen again, its **new, unique** `<Condition>`/`<Outcome>` children are appended to the first registered copy instead of being discarded.
- **Generic rejection** (`_is_generic`): any element whose id or label is in `{"untitled","unknown","unnamed","n/a","none",""}` or matches the bare-ID regex `^[rco]\d{1,3}$` (e.g. `R1`, `C2`, `O12`) is rejected and counted in the `skipped` tally that is surfaced to the UI log.

### 2c. Solving the "Untitled / generic node" problem — Compound Section-Anchored IDs

The original failure mode was a graph full of `Untitled`, `Unknown`, `C1`, `O1`-style nodes: when the extractor could not name an element, it emitted a placeholder, and those placeholders collided and tangled in Neo4j. The fix is enforced **at three layers**:

1. **Prompt contract (`PDF_TO_XML_SYSTEM`).** The naming rule is mandatory and absolute:

   | Element | ID format | Example |
   |---|---|---|
   | Rule | `R_{SectionNumber}` | `R_13`, `R_7_2` |
   | Condition | `C_{SectionNumber}_{DescriptiveSlug}` | `C_13_1_LawThesisWordLimit` |
   | Outcome | `O_{SectionNumber}_{DescriptiveSlug}` | `O_7_2_WithdrawalSanction` |

   - `SectionNumber` = the literal section/subsection digits from the text, with dots replaced by underscores (`7.2 → 7_2`).
   - `DescriptiveSlug` = a 2–5-word PascalCase phrase derived **only** from the actual policy text.
   - `Untitled`, `Unknown`, bare single-letter+digit IDs, and empty strings are explicitly **FORBIDDEN**.
   - The decisive instruction: *"If you cannot derive a descriptive name from the text DO NOT emit the element at all."* — i.e. **omission over fabrication**. A missing node is recoverable; a tangled placeholder node is corrupting.

2. **Merge-time rejection.** `_is_generic()` is a second line of defence: even if the model violates the contract, the bare-ID regex and the forbidden-value set strip the offending elements before they reach the database.

3. **Section-anchored uniqueness.** Because the ID embeds the section number, two different conditions can never accidentally share an ID, and a re-seen condition in an overlapping chunk is correctly recognised as the *same* node and merged rather than duplicated.

The net effect: node identity is **stable, human-traceable, and provenance-bearing** — a node ID alone tells you which clause of the source document it came from.

---

## 3. Graph Schema & Neo4j Mapping

### 3a. The 4-node statutory hierarchy

```
(:Policy)
   ▲  ▲  ▲
   │  │  │  [:BELONGS_TO]      (every member node is scoped to its policy)
   │  │  │
(:Rule) ──[:HAS_CONDITION]──► (:Condition)
   │
   └──────[:HAS_OUTCOME]────► (:Outcome)

(:Condition) ──[:CONFLICTS_WITH]── (:Condition)     (deterministic conflict edges)
```

**Node properties (as stored in the live graph):**

| Label | Properties |
|---|---|
| `Policy` | `id`, `title`, `version`, `ingestedAt` |
| `Rule` | `id`, `label`, `sourceSection`, `description`, `label_human`, `is_risk`, `embedding`, `policy_id` |
| `Condition` | `id`, `text`, `label_human`, `is_risk`, `embedding`, `policy_id` |
| `Outcome` | `id`, `type`, `text`, `label_human`, `is_risk`, `policy_id` |

**Relationship types:**

| Relationship | Direction | Notes |
|---|---|---|
| `BELONGS_TO` | `(Rule\|Condition\|Outcome) → (Policy)` | the namespace anchor for multi-policy isolation |
| `HAS_CONDITION` | `(Rule) → (Condition)` | a rule's qualifying clauses |
| `HAS_OUTCOME` | `(Rule) → (Outcome)` | the consequence/result of a rule |
| `CONFLICTS_WITH` | `(Condition) ↔ (Condition)` | props: `scenario`, `resolution`, `sections`, `cross_policy`; usually intra-policy |

### 3b. The `policy_id` namespace

Every member node carries a `policy_id` equal to its Policy's `id` (`{slugified-title}-{slugified-version}`, e.g. `8a-code-of-practice-for-research-degrees-2024-25`). This namespacing means:
- queries can be scoped to one policy or compared across policies;
- `delete_policy()` can detach-delete an entire policy sub-graph (`MATCH (n)-[:BELONGS_TO]->(p) DETACH DELETE p, n`) without touching others;
- cross-policy `CONFLICTS_WITH` edges (flagged `cross_policy = true`) are explicitly distinguishable from intra-policy ones.

### 3c. Semantic human-readable labels (`gpt-4o-mini`)

Raw policy IDs (`C_13_1_LawThesisWordLimit`) are precise but not friendly in a graph visualisation. `generate_human_label(text)`:
- uses `ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=20)`;
- prompts for a concise **2–3 word** title capturing the clause's core meaning, returning only the title;
- is decorated with `@st.cache_data(max_entries=2048)` so identical text is never re-billed to OpenAI;
- hard-caps output to 4 words to protect UI layout;
- falls back to a 40-char truncation of the source text if `OPENAI_API_KEY` is absent or the call fails.

The resulting `label_human` is what the Reasoning View renders on each node, while the precise `id` and `sourceSection` remain available in the audit trail.

### 3d. Risk flagging and vector indices

- **`is_risk`** is computed at ingest by `is_risk_text()` against `RISK_KEYWORDS` (`withdraw`, `sanction`, `penalty`, `failure`, `terminate`, `expel`, `revoke`, `reject`, …). Risk nodes are coloured red in the graph.
- **Vector indices** `rule_text_index` (on `Rule.embedding`) and `condition_text_index` (on `Condition.embedding`) are `DROP`-then-`CREATE`d on every ingest so a dimension change (1536 ↔ 384) never leaves a stale index. Both use cosine similarity at `EMBEDDING_DIMS`.

---

## 4. Hybrid Search & Routing Architecture

Retrieval is **Adaptive RAG**: the cost/precision profile is chosen per-query *before* any embedding work, by `_route_query()`.

### 4a. The router (`route_query`, GPT-4o, T=0)

Classifies the cleaned question into exactly one of two routes via a strict two-line response (`ROUTE:` / `REASON:`), `max_tokens=50`:

- **`DIRECT_LOOKUP`** — simple, single-hop, factual questions that map to one section/rule (e.g. *"What is the draft submission deadline?"*).
- **`COMPLEX_REASONING`** — multi-hop, thematic, comparative, or cross-policy questions.

**Fail-open contract:** any empty query, missing key, or exception returns `COMPLEX_REASONING`, so the most thorough pipeline is always the safe default. `run_query()` sets `use_fusion = (route == "COMPLEX_REASONING")` and threads that flag through the entire fallback path.

### 4b. Primary retrieval — always structured first

Regardless of route, `GraphCypherQAChain.invoke()` runs first: GPT-4o (T=0) generates Cypher from the **live, enhanced** schema, Neo4j executes it, and GPT-4o (T=0.1) synthesises an answer from the returned rows. The fallback semantic/keyword machinery only engages when the primary result is empty **or** the answer contains the sentinel *"not available in the current policy graph."*

### 4c. `COMPLEX_REASONING` → Multi-query Fusion RAG (`use_fusion=True`)

1. `generate_multiple_queries()` (GPT-4o, T=0) produces 3 alternative phrasings, each targeting a different retrieval angle (formal/regulatory, plain student-facing, synonym/adjacent-concept). With the original, that yields up to 4 query strings. Determinism (T=0) is intentional: the variant diversity comes from the prompt's three explicit angles, and reproducibility matters for evaluation.
2. Each query is embedded and run against **both** `rule_text_index` and `condition_text_index` (`score_window=12` candidates per index).
3. The per-query ranked lists are fused with **Reciprocal Rank Fusion** (`reciprocal_rank_fusion`, `k=60`): `RRF(d) = Σ 1/(k + rank_i(d))`. Items that rank highly across multiple variants accumulate the highest fused score.
4. **Kept set:** candidates with cosine `max_sim ≥ threshold` (`SEMANTIC_THRESHOLD`), capped at `top_k`. The cosine threshold *is* applied here — in complex mode we trust embedding similarity.

### 4d. `DIRECT_LOOKUP` → dual-stream RRF with keyword boosting (`use_fusion=False`)

No query expansion (it would add cost with no benefit on a single-section lookup). Instead, two retrieval **streams** are fused:

- **Stream A — vector:** the single original query embedded and run over both indices.
- **Stream B — keyword:** keywords are extracted from the isolated question:
  - **hard keywords** intersect a curated `_DOMAIN_KWS` set (`word`, `limit`, `thesis`, `submission`, `deadline`, `resubmit`, `withdrawal`, `viva`, `award`, `penalty`, `fail`, …) — these bias retrieval toward the specific chapters that answer factual policy lookups;
  - **soft keywords** are 4+-char alphabetic tokens minus stopwords;
  - up to 8 keywords total. A Cypher query counts, per `Rule`/`Condition`, how many keywords appear in `description + label + text + label_human`, ranking by match count.

Both streams are fused by RRF (`k=60`) — hence "dual-stream" (the audit log reports it as *hybrid N-stream RRF*). **Critically, the cosine threshold is NOT applied in direct mode:** the kept set is the **strict top-5 by RRF rank**. This is deliberate — a keyword-only domain hit (cosine 0.0 because the embedder never surfaced it) would otherwise be dropped, losing exactly the precise factual node a direct lookup needs.

### 4e. The CRAG (Corrective RAG) Quality Gate

`evaluate_context_relevance(rows, question)` (GPT-4o, T=0, `max_tokens=80`) is an **independent judge** of retrieval quality, classifying the context as `RELEVANT` / `IRRELEVANT` / `AMBIGUOUS` via a strict `STATUS:` / `REASON:` response. It is invoked at **two points**:

1. **Fallback path (pre-generation gate):** after the semantic/keyword fallback returns rows, CRAG runs *before* answer synthesis. `RELEVANT` → `_ground_answer_from_rows()` builds the answer; otherwise the user receives `_CRAG_REFUSAL` (an honest "the retrieved context does not contain this") instead of a hallucinated answer. If the fallback returns nothing at all, CRAG is short-circuited to `IRRELEVANT` without an LLM call.
2. **Primary path (post-hoc gate):** when the primary chain *did* return rows, CRAG runs as an independent confirmation signal surfaced in the audit trail; a non-`RELEVANT` verdict downgrades the answer to the refusal template.

**Fail-open contract:** empty/uncalled/error cases default to `RELEVANT`, so the validation layer can never itself block a legitimate answer. Every verdict (`crag_status`, `crag_reason`) is returned in the result dict and rendered in the transparency dashboard.

### 4f. From answer to Reasoning View

`run_query()` then extracts only the node IDs the answer *actually cites* (`extract_ids_from_text`) plus any section references resolved to IDs (`resolve_section_refs_to_ids`), unions them, and hands them to `fetch_reasoning_subgraph()` for the `streamlit-agraph` visualisation — so the picture shows the grounding for *this* answer, not the whole graph.

---

## 5. Major Engineering Bugs Resolved

### 5.1 `CypherSyntaxError` — post-`WITH` aggregation in `get_ingested_policies` (silent, high-impact)

**Symptom:** the sidebar showed *"No policies ingested yet"* even though Neo4j held a fully populated policy (142 Rules, etc.).

**Root cause:** the policy-listing query (then named `fetch_policies`) ended with an aggregating `RETURN` followed by `ORDER BY p.ingestedAt DESC, p.title`. Newer Neo4j / AuraDB rejects accessing a pre-`WITH` variable (`p`) after a `DISTINCT`/aggregating projection:

```
Neo.ClientError.Statement.SyntaxError — In a WITH/RETURN with DISTINCT or an
aggregation, it is not possible to access variables declared before the
WITH/RETURN: p   →   "ORDER BY p.ingestedAt DESC, p.title"
```

The function wrapped the query in a bare `try/except: return []`, so the syntax error was **silently swallowed** and an empty list was rendered — a populated graph looked empty. (Notably, the bug was *not* what it first appeared to be: the sidebar already queried Neo4j directly and did **not** rely on `st.session_state`.)

**Fix:**
- Project the final aggregation through an explicit `WITH (… count(DISTINCT o) AS outcomes)` and `ORDER BY` the **aliases** (`ingestedAt`, `title`) — both legal post-aggregation.
- Renamed to `get_ingested_policies(graph)` and **removed the error-swallowing** — it now raises on failure.
- The sidebar wraps the call in `try/except` and renders `st.error("Could not list policies: …")`, so a query fault can never again masquerade as "no data."

### 5.2 `streamlit-agraph` render crash — `.target` vs `.to`

**Symptom:** `AttributeError: 'Edge' object has no attribute 'target'` while rendering the Reasoning View (`app.py` ~line 3251).

**Root cause:** `streamlit-agraph`'s `Edge` object exposes the destination node as `.to`, not `.target`. The kept-edge filter referenced the non-existent attribute.

**Fix:** `e.target` → `e.to` in the comprehension `[e for e in r_edges if e.source in kept_ids and e.to in kept_ids]`. It was the only occurrence in the file (`.source` was already correct).

### 5.3 Removal of the Anthropic / `langchain-anthropic` "ghost" dependency

**Symptom / root cause:** the codebase guarded several features behind `ChatAnthropic` (`try: from langchain_anthropic import ChatAnthropic / except: ChatAnthropic = None`) and an `ANTHROPIC_API_KEY`. The package was **declared in `Requirements.txt` but never installed**, and **no `ANTHROPIC_API_KEY` existed in `.env`**, so every Claude-guarded path silently fell through — the project was effectively OpenAI-only by accident, with dead conditional branches obscuring that.

**Decision:** the project is **strictly OpenAI** (GPT-4o + OpenAI embeddings). All Anthropic code paths were removed, not merely disabled.

**Fix (both files):**
- Deleted the `ChatAnthropic` import guards and the `ANTHROPIC_API_KEY` variables.
- `route_query`, `generate_multiple_queries`, `evaluate_context_relevance` → `ChatOpenAI(model="gpt-4o", temperature=0)` only (`graphrag_policy_bot.py`).
- `process_pdf_to_xml` → GPT-4o only; raises clearly if `OPENAI_API_KEY` is missing.
- `_init_narrator` (and therefore `generate_search_variants`) → `gpt-4o-mini` only.
- The PDF-ingest button gate changed from `not (ANTHROPIC_API_KEY or OPENAI_API_KEY)` to `not OPENAI_API_KEY`.
- Removed `langchain-anthropic` from `Requirements.txt`. Docstrings that said "Claude Sonnet"/"Haiku" were corrected to "GPT-4o"/"GPT-4o-mini".

### 5.4 Neo4j stale-"offline" connection cache (earlier in the session)

**Symptom:** the app showed `🔴 Offline` while the database was reachable (a direct connection returned the full graph).

**Root cause:** `init_neo4j()`/`init_chain()` are `@st.cache_resource`-cached. A connection that failed once (e.g. AuraDB free-tier cold-start) had its `(None, error)` tuple **pinned for the whole server lifetime**, so the app never retried even after the DB recovered.

**Fix:** on failure, call `init_neo4j.clear()` / `init_chain.clear()` so the next rerun retries live; added a sidebar **🔄 Reconnect** button that clears the caches and reruns on demand.

### 5.5 Packaging gaps in `Requirements.txt`

Several hard imports were undeclared. Added `streamlit-agraph`, `altair`, `pandas`, `httpx`, and `truststore` so a fresh `pip install -r Requirements.txt` reproduces a working environment (relevant to the two-laptop / OneDrive workflow).

---

## 6. Current Status & Verification Metrics

> **Correction of the requested figures.** The original request cited *"156 Rules, 167 Conditions, 72 Outcomes"* and *"PhD 80,000/40,000/95,000 word limits."* A live query on 2026-05-31 shows those numbers conflate **relationship counts with node counts**, and the 95,000 figure does not exist in the data. The verified figures below supersede them.

### 6a. Live ingestion counts (queried 2026-05-31; `Policy.ingestedAt = 2026-05-29T22:13:46Z`)

| Metric | Verified value | Note on the requested figure |
|---|---|---|
| Total nodes | **373** | — |
| Total relationships | **611** | — |
| `:Policy` nodes | **1** | `8a-code-of-practice-for-research-degrees`, v2024-25 |
| `:Rule` nodes | **142** | requested "156" — not matched by any count in the graph |
| `:Condition` nodes | **162** | requested "167" = the `HAS_CONDITION` **edge** count, not nodes |
| `:Outcome` nodes | **68** | requested "72" = the `HAS_OUTCOME` **edge** count, not nodes |
| `BELONGS_TO` edges | **372** | every member node → its Policy |
| `HAS_CONDITION` edges | **167** | 5 more than Condition nodes → a few conditions attach to >1 rule |
| `HAS_OUTCOME` edges | **72** | 4 more than Outcome nodes → a few outcomes shared across rules |
| `CONFLICTS_WITH` edges | **0** | schema-supported; none present in the current single-policy corpus |

The slight excess of `HAS_CONDITION`/`HAS_OUTCOME` edges over their node counts is expected and healthy: the section-anchored merge correctly lets a shared clause attach to multiple rules rather than duplicating the node.

### 6b. Factual grounding — thesis word limits

A content scan of `Condition`/`Rule`/`Outcome` text confirms the corpus contains:

- **Doctoral/PhD thesis:** *"The thesis would normally be c. **40–80,000 words** depending on the discipline and nature of thesis format."* ✓ (80,000 present)
- **MPhil thesis:** *"An MPhil thesis would normally be c. **20–40,000 words** …"* ✓ (40,000 present)
- **95,000:** **not present** (0 matches for `95000` / `95,000`). If a 95k figure is expected, it is not in the currently ingested document and should not be cited as grounded.

These are stored as ranges in `Condition` nodes, which is faithful to the source ("c. 40–80,000"), not as single hard caps — worth noting for any evaluation question phrased as an exact maximum.

### 6c. Build verification

- `python -m py_compile app.py graphrag_policy_bot.py` → **clean**.
- Full module import of `app.py` executes top-to-bottom without exception; `_narrator` initialises (OpenAI), the Neo4j graph connects, and `get_ingested_policies(graph)` returns the policy with correct counts.
- No residual `anthropic` / `ChatAnthropic` / `claude` references remain in either file (verified by grep).
- Streamlit app boots headless and serves `HTTP 200` on `/_stcore/health`.
- OpenAI embedder passes its live `embed_query("probe")` connectivity test → `.env` `OPENAI_API_KEY` and network confirmed working (1536-dim active).

### 6d. Known caveats / outstanding items

- **`.env` requires `OPENAI_API_KEY` only.** No Anthropic key is needed or used.
- **Evaluation pipeline** (`ragas_evaluation.py`) still needs `ragas` + `datasets` installed (declared in `Requirements.txt`, not yet in the environment).
- **Single policy ingested.** The schema and `policy_id` namespacing support multiple policies and cross-policy conflicts, but the current graph holds exactly one policy. Note that `ingest_xml_to_neo4j()` performs a **full `MATCH (n) DETACH DELETE n` wipe at the start of every ingest**, so the platform currently operates as a single-corpus system: re-ingesting replaces the entire graph rather than adding alongside existing policies.
- **`langchain` / `langchain-community` version skew** remains in the environment (stale 0.3.0 leftovers vs the 1.x stack). They are not hard dependencies of this code (only a guarded optional HuggingFace fallback touches `langchain_community`), so they are harmless, but `pip uninstall`-ing them would silence the resolver warnings.
