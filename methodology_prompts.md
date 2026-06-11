# Methodology — Reverse-Engineered Prompt Sequence

### Transparent Policy GraphRAG — Reconstructing the Build from First Principles

**Author:** Alireza (Allen) Sharafzad · MSc Data Science & AI, Bournemouth University
**Purpose:** A faithful, engineer-level reconstruction of the prompt sequence that would guide an AI coding agent to build the *exact* pipeline implemented in [`app.py`](app.py) and [`graphrag_policy_bot.py`](graphrag_policy_bot.py), as verified in [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md).

---

## How to read this document

This is a **methodology artefact**, not marketing copy. Each phase contains a numbered sequence of prompts written as they would realistically be issued to a coding agent (e.g. Claude Code) by a Lead Technical Architect. Every prompt is anchored to the **actual function names, constants, and control flow** present in the codebase — not idealised placeholders.

> **A note on honesty (read before citing this in the thesis).** Two retrieval modules exist. The **production Streamlit app** ([`app.py`](app.py)) ingests a 4-node hierarchy with the relationship set `BELONGS_TO`, `HAS_CONDITION`, `HAS_OUTCOME`, `CONFLICTS_WITH`. The **CLI prototype** ([`graphrag_policy_bot.py`](graphrag_policy_bot.py)) carries a *richer* Cypher-generation contract in `CYPHER_GENERATION_TEMPLATE` that additionally names `ESCALATES_TO`, `OVERRIDES`, `GOVERNED_BY`, `SATISFIES`, `TRIGGERS`, and `RESTRICTS`. The prompts below reflect that split accurately rather than pretending the production graph instantiates every edge type. The live graph (queried 2026-05-31) holds **373 nodes / 611 relationships** with **0** `CONFLICTS_WITH` edges materialised in the current single-policy corpus.

Each prompt block is followed by **Acceptance criteria** — the observable conditions that tell the agent (and the supervisor) the step is genuinely complete.

---

## Phase 1 — XML Grounding & Ontology Mapping

> **Goal:** Transform an unstructured policy PDF into strict, schema-conformant XML in which every element carries a *section-anchored, provenance-bearing* ID, and in which `Untitled`/`Unknown`/bare-ID placeholder nodes are structurally impossible.

### Prompt 1.1 — PDF text extraction with fail-loud semantics

```
Implement `_extract_pdf_text(pdf_bytes, log_fn)` using PyMuPDF (`fitz`).
Open the document from bytes, concatenate the text of every page, and return
the joined string. If the concatenated text is empty (image-only scan with no
OCR layer), the *caller* — `process_pdf_to_xml` — must raise
`RuntimeError("No extractable text found in the PDF.")`. Do not silently
return an empty string downstream. Route every status line through
`log_fn(level, message)` with level ∈ {info, success, warn, err} so the
Streamlit process log can render extraction progress in real time.
```

**Acceptance criteria:** a text-bearing PDF yields a non-empty string; an image-only PDF surfaces a `RuntimeError` to the UI, not a blank graph.

### Prompt 1.2 — Sliding-window chunking (bounded inputs for valid XML)

```
Implement `_chunk_text(text, chunk_size, overlap)` returning a list of
overlapping character windows. In `process_pdf_to_xml`, call it with
CHUNK_SIZE = 15_000 and CHUNK_OVERLAP = 1_500.

Rationale to encode in the docstring: GPT-4o produces *valid, complete* XML far
more reliably on bounded inputs than on a whole document; the 1,500-char
overlap guarantees that a section header split across a chunk boundary still
appears intact in at least one chunk. Log the total char count, the chunk
count, and the window/overlap sizes.
```

**Acceptance criteria:** a 60k-char document splits into ~5 overlapping windows; no section heading is severed such that it appears in zero chunks.

### Prompt 1.3 — The `PDF_TO_XML_SYSTEM` prompt contract (the anti-`Untitled` covenant)

```
Author the `PDF_TO_XML_SYSTEM` constant as a Knowledge-Engineering-Agent system
prompt. State explicitly that every emitted element is MERGEd directly into a
live production Neo4j database, so any generic placeholder or malformed tag
corrupts it permanently. Mandate the compound section-anchored ID grammar:

    Rule       id = R_{SectionNumber}                       e.g. R_13, R_7_2
    Condition  id = C_{SectionNumber}_{DescriptiveSlug}      e.g. C_13_1_LawThesisWordLimit
    Outcome    id = O_{SectionNumber}_{DescriptiveSlug}      e.g. O_7_2_WithdrawalSanction

Rules for the grammar:
  - SectionNumber = the literal section/subsection digits from the text, with
    dots replaced by underscores (7.2 -> 7_2).
  - DescriptiveSlug = a 2-5 word PascalCase phrase derived ONLY from the actual
    policy text — never invented.
  - FORBID `Untitled`, `Unknown`, empty strings, and bare single-letter+digit
    IDs (R1, C2, O12).
  - The decisive instruction: "If you cannot derive a descriptive name from the
    text, DO NOT emit the element at all." Omission over fabrication — a missing
    node is recoverable; a tangled placeholder node is corrupting.
```

**Acceptance criteria:** prompting against a sample section yields IDs of the form `C_13_1_LawThesisWordLimit`; no element ever carries `Untitled` or `R1`.

### Prompt 1.4 — The per-chunk extraction loop

```
In `process_pdf_to_xml(pdf_file, log_fn)`, construct ONE
`ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=8000, api_key=OPENAI_API_KEY,
**_llm_kwargs())` instance (OpenAI-only; no Anthropic branch). For each chunk,
send `PDF_TO_XML_SYSTEM` plus the per-chunk instruction verbatim:
"CHUNK i OF N: Derive all element IDs from section numbers visible in THIS
chunk's text." followed by the chunk body. Skip empty responses with a warning.
On an LLM exception, log `type(err).__name__`, the last 600 chars of the
traceback, and RE-RAISE (fail-loud — the user must know extraction failed).
Raise if OPENAI_API_KEY is absent.
```

**Acceptance criteria:** N chunks produce ≤ N raw XML fragments; an API failure aborts loudly rather than yielding partial silent output.

### Prompt 1.5 — `_merge_xml_fragments` (dedup, sanitise, reject generics)

```
Implement `_merge_xml_fragments(raw_fragments, log_fn)`:
  - `_sanitise`: strip Markdown ```xml fences, strip any <?xml ...?> prolog,
    escape bare `&` not already part of a valid entity, and remove non-XML
    control chars (\x00-\x08, \x0b, \x0c, \x0e-\x1f).
  - Wrap each fragment in <root>...</root> unless it already starts with
    <Policy ...>, then parse with ElementTree. A parse failure is COUNTED and
    SKIPPED — one bad chunk never aborts the whole merge.
  - Collect only direct-child <Rule> elements at top level (nested sub-rules
    ride inside their parent; never double-counted).
  - Deduplication key = tag + "::" + id. When a Rule id recurs, append its NEW,
    unique <Condition>/<Outcome> children to the first registered copy instead
    of discarding them.
  - `_is_generic(el)`: reject any element whose id or label is in
    {"untitled","unknown","unnamed","n/a","none",""} OR matches the bare-ID
    regex ^[rco]\d{1,3}$. Increment a `skipped` tally surfaced to the UI log.
```

**Acceptance criteria:** the same `Rule` appearing in two overlapping chunks resolves to ONE node whose conditions are the union of both copies; any `R1`/`Untitled` leakage from the model is stripped before it reaches the DB.

### Prompt 1.6 — Wrap + validate well-formedness (the final gate)

```
Implement `_wrap_xml_fragment(xml_fragment)` to produce a single
<Policy>...</Policy> byte string. In `process_pdf_to_xml`, pass the result
through `ET.fromstring()` as the final gate BEFORE returning. On
`ET.ParseError`, decode, locate `e.position[0]`, log the offending line
(truncated to 200 chars), and re-raise — malformed XML must never reach Neo4j.
```

**Acceptance criteria:** `process_pdf_to_xml` returns parseable `bytes`; a deliberately corrupted fragment is reported with its exact offending line number and blocks ingestion.

---

## Phase 2 — Neo4j Relational Schema Construction

> **Goal:** Map validated XML into a `policy_id`-namespaced graph using explicit, idempotent `MERGE` Cypher, with semantic human labels, risk flagging, and per-dimension vector indices — all scoped so a single policy can be detached without touching others.

### Prompt 2.1 — Policy namespace root + clean-slate wipe

```
Implement `ingest_xml_to_neo4j(xml_content, graph, log_fn, policy_title=None,
policy_version=None)`. Resolve metadata with precedence: explicit args > XML
root attributes (if <Policy>) > "Untitled Policy"/"unspecified". Build the
namespace key `pid = f"{_slugify(title)}-{_slugify(version)}"` (e.g.
`8a-code-of-practice-for-research-degrees-2024-25`).

Begin with a mandatory clean-slate wipe: `MATCH (n) DETACH DELETE n` (wrapped so
a wipe failure warns but does not crash). Then:
    MERGE (p:Policy {id:$pid})
    SET p.title=$title, p.version=$version, p.ingestedAt=datetime()

Document explicitly that this makes the platform a single-corpus system:
re-ingesting REPLACES the entire graph.
```

**Acceptance criteria:** ingest produces exactly one `:Policy` node whose `id` is the slugified title-version; a prior graph is fully cleared first.

### Prompt 2.2 — Dimension-safe vector indices (DROP + CREATE)

```
Before creating member nodes, DROP-then-CREATE both vector indices so a
dimension change (1536 OpenAI <-> 384 HuggingFace fallback) never leaves a
stale index:

    DROP INDEX rule_text_index IF EXISTS
    CREATE VECTOR INDEX rule_text_index FOR (r:Rule) ON (r.embedding)
      OPTIONS {indexConfig: {`vector.dimensions`: EMBEDDING_DIMS,
                             `vector.similarity_function`: 'cosine'}}

Repeat for `condition_text_index` on (c:Condition). Use the module-level
`EMBEDDING_DIMS` (1536 by default, overridden to 384 if the HuggingFace
fallback activates). Skip gracefully if `_embedder is None`.
```

**Acceptance criteria:** both indices exist at the active embedding dimension; switching embedder dimension and re-ingesting does not error on a stale index.

### Prompt 2.3 — `MERGE` Rules / Conditions / Outcomes scoped by `policy_id`

```
Iterate `root.iter("Rule")`. For each Rule with a non-empty id, MERGE it scoped
to the policy and attach BELONGS_TO in one statement:

    MERGE (r:Rule {id:$id, policy_id:$pid})
    SET r.label=$label, r.sourceSection=$sec, r.description=$desc,
        r.label_human=$label_human, r.is_risk=$is_risk
    WITH r MATCH (p:Policy {id:$pid}) MERGE (r)-[:BELONGS_TO]->(p)

Then MERGE each child Condition/Outcome (scoped by the SAME policy_id) and
connect with `(r)-[:HAS_CONDITION]->(c)` and `(r)-[:HAS_OUTCOME]->(o)`.
Maintain the stats dict: rules, conditions, outcomes, has_condition,
has_outcome, conflicts, errors. The compound section-anchored id ($id) is the
MERGE key — this is what makes re-ingestion idempotent and lets a shared clause
attach to multiple rules rather than duplicating the node.
```

**Acceptance criteria:** every member node carries `policy_id == pid` and a `BELONGS_TO` edge; counts match the XML; a condition referenced by two rules yields one node with two `HAS_CONDITION` edges.

### Prompt 2.4 — Semantic human labels (`generate_human_label`, gpt-4o-mini, cached)

```
Implement `generate_human_label(text)` decorated with
`@st.cache_data(max_entries=2048)` so identical text is never re-billed.
Use `ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=20,
**_llm_kwargs())`. Prompt for a concise 2-3 word title capturing the clause's
core meaning; return only the title; hard-cap to 4 words to protect UI layout.
Fall back to a 40-char truncation of the source text if OPENAI_API_KEY is
absent or the call fails. Store the result as `label_human` on the node; keep
the precise `id` and `sourceSection` for the audit trail.
```

**Acceptance criteria:** `C_13_1_LawThesisWordLimit` renders as a friendly label like "Thesis Word Limit"; repeated identical text triggers no second LLM call.

### Prompt 2.5 — Risk flagging + embeddings

```
Implement `is_risk_text(*parts)` matching any token in `RISK_KEYWORDS`
(withdraw, sanction, penalty, failure, terminate, expel, revoke, reject, ...).
Set `is_risk` at ingest so risk nodes colour red in the Reasoning View.

For each Rule/Condition, build `embed_text` from the human label + label +
description (joined by " · "), embed via `_embed_with_retry(embed_text)`
(text-embedding-3-small), and SET it onto `r.embedding` / `c.embedding`.
```

**Acceptance criteria:** nodes containing sanction language carry `is_risk=true`; both vector indices populate and answer `db.index.vector.queryNodes` calls.

### Prompt 2.6 — Conflict edges + the prototype's richer edge contract

```
Support deterministic `CONFLICTS_WITH` edges between Conditions, with edge
properties `scenario`, `resolution`, `sections`, `cross_policy`. (The current
single-policy corpus materialises zero of these, but the schema must accept
them.)

Separately, in the CLI prototype `graphrag_policy_bot.py`, author
`CYPHER_GENERATION_TEMPLATE` to teach the Cypher LLM the FULL relational
vocabulary so multi-hop reasoning queries can be generated:
  [:HAS_CONDITION]  Rule -> Condition
  [:HAS_OUTCOME]    Rule -> Outcome
  [:GOVERNED_BY]    Actor -> Rule
  [:SATISFIES]      Actor -> Condition
  [:TRIGGERS]       Rule -> Outcome
  [:ESCALATES_TO]   Rule/Outcome -> Rule
  [:OVERRIDES]      Rule -> Rule  (mitigating rule takes precedence)
  [:CONFLICTS]      Condition <-> Condition  (props: scenario, resolution)
  [:RESTRICTS]      Rule/Outcome -> Actor
  [:BELONGS_TO]     Rule -> PolicyDomain
Mandate: ALWAYS RETURN r.sourceSection so answers can cite the section; ALWAYS
traverse OVERRIDES and CONFLICTS for any eligibility query; NEVER invent IDs or
property values absent from the live schema.
```

**Acceptance criteria:** the schema accepts a `CONFLICTS_WITH` edge with all four properties; the prototype's Cypher LLM emits traversals over `ESCALATES_TO`/`OVERRIDES` for an escalation question.

### Prompt 2.7 — Policy listing + deletion (post-aggregation Cypher correctness)

```
Implement `get_ingested_policies(graph)`. CRITICAL: newer Neo4j/AuraDB forbids
accessing a pre-WITH variable after a DISTINCT/aggregation. Project the final
aggregation through an explicit `WITH (... count(DISTINCT o) AS outcomes)` and
`ORDER BY` the ALIASES (`ingestedAt`, `title`), never the bare `p`. Do NOT
swallow errors — raise on failure; the sidebar wraps the call and renders
`st.error("Could not list policies: ...")` so a query fault can never
masquerade as "no data".

Implement `delete_policy(graph, policy_id)` using
`MATCH (n)-[:BELONGS_TO]->(p) ... DETACH DELETE p, n` scoped by policy_id, so a
single policy sub-graph detaches without touching others.
```

**Acceptance criteria:** a populated graph lists its policy with correct rule/condition/outcome counts; a Cypher fault is shown as an explicit error, not an empty list.

---

## Phase 3 — Hybrid Search, Adaptive Routing & CRAG Architecture

> **Goal:** A Modular RAG query path: clean → route (Adaptive RAG) → structured-first Cypher → conditional fallback (Fusion RAG **or** dual-stream RRF) → CRAG validation → grounded answer → reasoning subgraph.

### Prompt 3.1 — Query isolation + adaptive router

```
Implement `_clean_user_query(raw)` to strip whitespace and any UI leakage.

Implement `route_query(user_query, graph=None, chain=None)` using
`ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=50, **_llm_kwargs())`.
Classify into exactly one route via a strict two-line response:
    ROUTE: <DIRECT_LOOKUP|COMPLEX_REASONING>
    REASON: <one phrase, max 12 words>
DIRECT_LOOKUP = simple, single-hop, single-section factual lookups.
COMPLEX_REASONING = multi-hop, thematic, comparative, or cross-policy.
FAIL-OPEN CONTRACT: empty query, missing key, or any exception returns
COMPLEX_REASONING so the most thorough pipeline is always the safe default.
In `run_query`, set `use_fusion = (route == "COMPLEX_REASONING")` and thread
that flag through the entire fallback path.
```

**Acceptance criteria:** "What is the draft submission deadline?" → `DIRECT_LOOKUP`; "Compare examiner independence rules across all policies" → `COMPLEX_REASONING`; an LLM outage still routes (to `COMPLEX_REASONING`).

### Prompt 3.2 — Structured-first primary retrieval

```
In `run_query(question, chain)`, ALWAYS run `chain.invoke({"query": question})`
first (a `GraphCypherQAChain`: gpt-4o T=0 generates Cypher from the live
enhanced schema; Neo4j executes; gpt-4o T=0.1 synthesises). Pull
`intermediate_steps[0]["query"]` as the cypher and `[1]["context"]` as raw rows.
Only engage the fallback machinery when the result is empty OR the answer
contains the sentinel "not available in the current policy graph".
```

**Acceptance criteria:** an answerable factual question is served by the primary chain with no fallback; an out-of-corpus question trips the sentinel and enters fallback.

### Prompt 3.3 — Multi-query expansion (Fusion RAG, COMPLEX only)

```
Implement `generate_multiple_queries(user_query, num_queries=3)` using
`ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=400, **_llm_kwargs())`.
Each variant targets a DIFFERENT retrieval angle:
  1. formal/regulatory phrasing  2. plain student-facing  3. synonym/adjacent.
Output only the questions, one per line, no numbering/preamble. T=0 is
intentional: variant diversity comes from the three explicit angles, and
determinism matters for evaluation reproducibility. Fall back to [user_query]
on any failure.
```

**Acceptance criteria:** a complex question yields up to 3 distinct rephrasings plus the original (≤ 4 query strings); a failure degrades to the single original query.

### Prompt 3.4 — Reciprocal Rank Fusion primitive

```
Implement `reciprocal_rank_fusion(ranked_lists, k=60)` returning
[(node_id, score)] sorted descending, where
    RRF(d) = Σ_i  1 / (k + rank_i(d))   (rank 1-indexed)
Items ranking highly across multiple lists accumulate the highest fused score.
k=60 per the original RRF paper.
```

**Acceptance criteria:** a node appearing at rank 1 in two streams outranks a node appearing at rank 1 in only one; output is deterministic for fixed inputs.

### Prompt 3.5 — `_semantic_search`: the two-mode engine

```
Implement `_semantic_search(graph, question, top_k=8, threshold=SEMANTIC_THRESHOLD,
score_window=12, use_fusion=True)`. SEMANTIC_THRESHOLD = 0.70.

COMPLEX_REASONING (use_fusion=True):
  - queries = [original] + generate_multiple_queries(...).
  - Embed all; for each vector, query BOTH `rule_text_index` and
    `condition_text_index` via `CALL db.index.vector.queryNodes($idx, $k, $vec)`
    with k=score_window=12; track max cosine per (label,id).
  - Fuse per-query rankings with reciprocal_rank_fusion(k=60).
  - KEPT = candidates with max cosine >= threshold (0.70), capped at top_k.
    The cosine threshold IS applied here — in complex mode we trust embedding
    similarity.

DIRECT_LOOKUP (use_fusion=False): dual-stream RRF with keyword boosting.
  - Stream A (vector): the single original query embedded over both indices.
  - Stream B (keyword): extract keywords from the ISOLATED question —
      * hard kws = intersection with `_DOMAIN_KWS` {word, limit, words, maximum,
        minimum, thesis, chapter, submit, submission, deadline, resubmit,
        withdrawal, examination, viva, award, penalty, fail, failure, sanction,
        appendix, abstract};
      * soft kws = 4+-char alphabetic tokens minus `_STOPWORDS`;
      * `direct_kws = (hard + soft) deduped, capped at 8`.
    Run a Cypher CONTAINS query counting, per Rule/Condition, how many keywords
    appear in coalesce(description)+label+text+label_human; ORDER BY match count.
  - Fuse Stream A + Stream B with reciprocal_rank_fusion(k=60).
  - CRITICAL: do NOT apply the cosine threshold in direct mode. KEPT = strict
    top-5 by RRF rank. A keyword-only domain hit (cosine 0.0 because the
    embedder never surfaced it) must survive — it is exactly the precise factual
    node a direct lookup needs. Preserve cosine=0.0 for keyword-only nodes so
    they still appear in the transparency chart with their RRF rank.

Emit an annotated cypher header logging mode, "RRF k=60", threshold, the query
list, and each hit's (label, id) → cosine | rrf with a ✓ kept / ✗ dropped flag.
```

**Acceptance criteria:** in direct mode, a node found only by keyword (cosine 0.0) appears in the kept top-5; in complex mode, only nodes with cosine ≥ 0.70 are kept; the audit header lists per-node cosine and RRF scores.

### Prompt 3.6 — Fallback wrapper

```
Implement `keyword_fallback_search(graph, question, use_fusion=True)` that calls
`_semantic_search(..., use_fusion=use_fusion)` first and falls back to a legacy
CONTAINS query only if vector search yields nothing. Return
(annotated_cypher, rows, scores).
```

**Acceptance criteria:** with embeddings present, the vector/RRF path is used; with embeddings unavailable, a CONTAINS query still returns candidate rows.

### Prompt 3.7 — CRAG validator + the two gates

```
Implement `evaluate_context_relevance(retrieved_rows, user_query)` using
`ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=80, **_llm_kwargs())`.
Build a compact snapshot (top 6 rows; keys policyTitle, ruleLabel, ruleSection,
ruleDescription, conditionText, outcomes; each value truncated to ~120 chars).
Require a strict two-line verdict:
    STATUS: <RELEVANT|IRRELEVANT|AMBIGUOUS>
    REASON: <one sentence, max 25 words>
Short-circuit to IRRELEVANT (no LLM call) when there are no rows. FAIL-OPEN:
empty query / missing key / any exception defaults to RELEVANT so the validator
can never itself block a legitimate answer.

Wire BOTH gates in `run_query`:
  (1) Fallback path — PRE-GENERATION gate: after keyword_fallback_search returns
      rows, run CRAG BEFORE synthesis. RELEVANT -> `_ground_answer_from_rows`;
      otherwise return `_CRAG_REFUSAL.format(status=...)`.
  (2) Primary path — POST-HOC gate: when the primary chain returned rows, run
      CRAG as an independent confirmation; a non-RELEVANT verdict downgrades the
      answer to the refusal template.
Return `crag_status` and `crag_reason` in the result dict for the transparency
dashboard.
```

**Acceptance criteria:** an off-topic fallback context yields `_CRAG_REFUSAL` instead of a hallucinated answer; a CRAG outage never blocks a valid answer (defaults RELEVANT); both verdicts surface in the audit trail.

### Prompt 3.8 — Answer → Reasoning View grounding

```
After synthesis, extract ONLY the node IDs the answer actually cites with
`extract_ids_from_text(answer)`, plus section references resolved via
`extract_sections_from_text` + `resolve_section_refs_to_ids(graph, sections,
policy_ids=active_policy_ids)`. Union them and pass to
`fetch_reasoning_subgraph(graph, node_ids, ...)` for the streamlit-agraph
visualisation — the picture must show the grounding for THIS answer, not the
whole graph. Scope to `extract_active_policy_ids(raw)`.
```

**Acceptance criteria:** the rendered subgraph contains only nodes cited in the answer (plus resolved sections), never the full 373-node graph.

---

## Phase 4 — Network Resilience & Windows UTF-8 Fixes

> **Goal:** Make the app run unattended behind a corporate/university TLS-intercepting proxy on Windows, with deterministic UTF-8 stdout, an explicit truststore-backed HTTP client threaded into every LLM, and self-healing connection caches.

### Prompt 4.1 — Windows UTF-8 stdout

```
At the very top of `app.py`, before any other import that may print, force
UTF-8 stdout so emoji/log glyphs never raise UnicodeEncodeError under the
Windows cp1252 console:

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
```

**Acceptance criteria:** the process log (which uses ✓, 🤖, 🏛, etc.) prints without `UnicodeEncodeError` in a stock Windows terminal.

### Prompt 4.2 — Truststore SSL injection (proxy bypass)

```
Immediately after the UTF-8 block, inject the OS trust store so a corporate/
university proxy's TLS interception certificate is honoured, then build an
EXPLICIT httpx client carrying that context:

    try:
        import ssl as _ssl
        import httpx as _httpx
        import truststore as _truststore
        _truststore.inject_into_ssl()
        _ssl_ctx = _truststore.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        _http_client = _httpx.Client(verify=_ssl_ctx)
    except Exception:
        _http_client = None

`inject_into_ssl()` patches `ssl.SSLContext` globally; the explicit
`_http_client` is what we pass to every LLM constructor so Streamlit's module
cache cannot hold a stale SSL context.
```

**Acceptance criteria:** behind a TLS-intercepting proxy, OpenAI/Neo4j HTTPS calls succeed without `SSLCertVerificationError`; with no proxy, `_http_client` is still valid (or `None` and gracefully ignored).

### Prompt 4.3 — `_llm_kwargs()` threaded into every client

```
Implement `_llm_kwargs()` returning {"http_client": _http_client} when a
truststore client is available, else {}. Spread `**_llm_kwargs()` into EVERY
`ChatOpenAI(...)` and the OpenAI embedder constructor across both `app.py` and
`graphrag_policy_bot.py`: cypher_llm, qa_llm, process_pdf_to_xml,
route_query, generate_multiple_queries, evaluate_context_relevance,
generate_human_label, _init_narrator, and `_init_embedder`.
```

**Acceptance criteria:** grepping for `ChatOpenAI(` shows `**_llm_kwargs()` on every instantiation; removing the proxy still works (kwargs empty).

### Prompt 4.4 — Embedder probe + offline fallback

```
In `_init_embedder`, after constructing OpenAIEmbeddings (text-embedding-3-small,
1536-dim, `**_llm_kwargs()`), run a LIVE `embed_query("probe")` connectivity
test. If it raises, fall back to `all-MiniLM-L6-v2` (384-dim) via
langchain_huggingface / langchain_community, and set the module-level
`EMBEDDING_DIMS` to whichever dimension is active so the vector indices
(Prompt 2.2) are created at the correct size. Implement `_embed_with_retry(
embed_text, max_attempts=3)` for transient network blips during ingest.
```

**Acceptance criteria:** with a working key the probe passes and `EMBEDDING_DIMS == 1536`; with the OpenAI endpoint blocked, the app degrades to 384-dim local embeddings and still builds indices that match.

### Prompt 4.5 — Self-healing connection caches

```
`init_neo4j()` and `init_chain(_graph)` are `@st.cache_resource`-cached. A
connection that fails once (AuraDB free-tier cold-start) must NOT pin a
(None, error) tuple for the server lifetime. On failure, call
`init_neo4j.clear()` / `init_chain.clear()` so the next rerun retries live, and
add a sidebar "🔄 Reconnect" button that clears both caches and reruns on
demand. Use `Neo4jGraph(url, username, password, enhanced_schema=True,
sanitize=True)` and call `graph.refresh_schema()` so the live schema is injected
into Cypher generation.
```

**Acceptance criteria:** an AuraDB cold-start that first shows offline recovers on the next rerun or via the Reconnect button — no server restart required.

### Prompt 4.6 — Packaging closure

```
Ensure `Requirements.txt` declares every hard import so a fresh
`pip install -r Requirements.txt` reproduces a working environment on either
laptop: add `streamlit-agraph`, `altair`, `pandas`, `httpx`, and `truststore`.
Remove `langchain-anthropic` — the project is strictly OpenAI (GPT-4o +
text-embedding-3-small); delete the dead `ChatAnthropic` import guards and the
`ANTHROPIC_API_KEY` references, and change the ingest-button gate from
`not (ANTHROPIC_API_KEY or OPENAI_API_KEY)` to `not OPENAI_API_KEY`.
```

**Acceptance criteria:** `python -m py_compile app.py graphrag_policy_bot.py` is clean; a fresh venv install boots the app headless to `HTTP 200` on `/_stcore/health`; no residual `anthropic`/`ChatAnthropic`/`claude` references remain.

---

## Appendix — Verified anchor inventory

| Phase | Key functions / constants (exact names) |
|---|---|
| 1 | `_extract_pdf_text`, `_chunk_text` (15_000 / 1_500), `PDF_TO_XML_SYSTEM`, `process_pdf_to_xml`, `_merge_xml_fragments`, `_is_generic` (`^[rco]\d{1,3}$`), `_sanitise`, `_wrap_xml_fragment` |
| 2 | `ingest_xml_to_neo4j`, `_slugify`, `MATCH (n) DETACH DELETE n`, `rule_text_index`/`condition_text_index`, `EMBEDDING_DIMS`, `generate_human_label` (gpt-4o-mini), `is_risk_text`/`RISK_KEYWORDS`, `_embed_with_retry`, `BELONGS_TO`/`HAS_CONDITION`/`HAS_OUTCOME`/`CONFLICTS_WITH`, `CYPHER_GENERATION_TEMPLATE` (`ESCALATES_TO`/`OVERRIDES`/`GOVERNED_BY`), `get_ingested_policies`, `delete_policy` |
| 3 | `_clean_user_query`, `route_query`/`_route_query`, `use_fusion`, `generate_multiple_queries`, `reciprocal_rank_fusion` (k=60), `_semantic_search` (`SEMANTIC_THRESHOLD=0.70`, `score_window=12`, `_DOMAIN_KWS`), `keyword_fallback_search`, `evaluate_context_relevance`, `_CRAG_REFUSAL`, `_ground_answer_from_rows`, `extract_ids_from_text`, `resolve_section_refs_to_ids`, `fetch_reasoning_subgraph` |
| 4 | `sys.stdout.reconfigure`, `truststore.inject_into_ssl`, `_ssl_ctx`, `_http_client`, `_llm_kwargs`, `_init_embedder` (probe + `all-MiniLM-L6-v2` 384-dim), `init_neo4j.clear()`/`init_chain.clear()`, `Neo4jGraph(enhanced_schema=True, sanitize=True)` |

*All identifiers above were read directly from [`app.py`](app.py) and [`graphrag_policy_bot.py`](graphrag_policy_bot.py) and cross-checked against [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md).*
