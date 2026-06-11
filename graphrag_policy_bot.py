"""
graphrag_policy_bot.py
═══════════════════════════════════════════════════════════════════════════════
GraphRAG Policy Chatbot — Phase 3 Integration Layer
Project: Extending Policy Chatbots with Knowledge Graphs and GraphRAG
Source:  BU 8A Code of Practice for Research Degrees 2024-25

Architecture:
  User Query
      │
      ▼
  [1] Neo4jGraph   — establishes connection to AuraDB, refreshes schema
      │
      ▼
  [2] Cypher Generation LLM  — converts natural language → Cypher query
      │
      ▼
  [3] Neo4j AuraDB traversal — returns subgraph (nodes + edges + properties)
      │
      ▼
  [4] Grounding & Answer LLM — generates response ONLY from graph context
      │
      ▼
  [5] Cited Answer + traversal path audit trail

Dependencies:
  pip install langchain-neo4j langchain-openai python-dotenv

Environment variables (.env):
  NEO4J_URI       = neo4j+s://<your-instance>.databases.neo4j.io
  NEO4J_USERNAME  = neo4j
  NEO4J_PASSWORD  = <your-password>
  OPENAI_API_KEY  = sk-...

Author:  Alireza (Allen) Sharafzad
Date:    May 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import textwrap

# Ensure Unicode output works on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Windows SSL: inject OS trust store so corporate/university proxies work
try:
    import ssl as _ssl
    import httpx as _httpx
    import truststore as _truststore
    _truststore.inject_into_ssl()
    _ssl_ctx = _truststore.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
    _http_client = _httpx.Client(verify=_ssl_ctx)
except Exception:
    _http_client = None


def _llm_kwargs() -> dict:
    """Return the keyword arguments every ChatOpenAI/embedder constructor needs.

    Returns:
        dict: ``{"http_client": <httpx.Client>}`` when a truststore-backed HTTP
        client was successfully built at import time (see the SSL injection block
        above), otherwise an empty dict ``{}``.

    Logical Rationale:
        Spreading ``**_llm_kwargs()`` into *every* LLM constructor is what lets
        the whole module operate behind a corporate/university TLS-intercepting
        proxy. We pass an *explicit* ``httpx.Client`` (rather than relying solely
        on the global ``truststore.inject_into_ssl()`` patch) so that no cached
        OpenAI client can silently bind to a stale SSL context. The empty-dict
        fallback keeps the call sites identical whether or not a proxy is present.
    """
    return {"http_client": _http_client} if _http_client is not None else {}


from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()  # reads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY

NEO4J_URI         = os.getenv("NEO4J_URI")
NEO4J_USERNAME    = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

# Validate — fail fast with a clear message rather than a cryptic Neo4j error
for var, val in [
    ("NEO4J_URI",      NEO4J_URI),
    ("NEO4J_PASSWORD", NEO4J_PASSWORD),
    ("OPENAI_API_KEY", OPENAI_API_KEY),
]:
    if not val:
        raise EnvironmentError(
            f"Missing environment variable: {var}\n"
            f"Add it to your .env file or export it before running."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 0.5  FUSION RAG — QUERY GENERATION + RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — Fusion RAG layer
#   This section implements the two primitives that turn a single user question
#   into a *fused, multi-angle* ranking:
#       generate_multiple_queries() — query expansion (recall ↑)
#       reciprocal_rank_fusion()    — rank aggregation (precision ↑)
#   Both are pure/stateless helpers imported by app.py's `_semantic_search`.
#
# EXTENDING THIS SECTION
#   • To add a 4th retrieval *angle* (e.g. a "legal-citation" rewrite), raise the
#     `num_queries` default and append a 4th bullet to the prompt in
#     generate_multiple_queries(); keep T=0 so the variant set stays reproducible
#     for evaluation runs.
#   • To add a new *stream* to the fusion (e.g. a BM25 full-text stream), do NOT
#     change reciprocal_rank_fusion() — it already accepts an arbitrary number of
#     ranked lists. Just append the new stream's ranked node-ID list to the
#     `ranked_lists` argument at the call site in app.py._semantic_search.
#   • To tune fusion aggressiveness, change `k` (see its justification below).
# ─────────────────────────────────────────────────────────────────────────────

def generate_multiple_queries(user_query: str, num_queries: int = 3) -> list[str]:
    """Generate alternative phrasings of a policy question for Fusion RAG.

    Uses GPT-4o (T=0) to rewrite the question from several retrieval angles so
    that downstream vector search casts a wider net than a single embedding can.

    Args:
        user_query (str): The cleaned user question. Whitespace-only input is
            returned unchanged (wrapped in a single-element list).
        num_queries (int): How many alternative phrasings to request. Default 3;
            with the original question this yields up to 4 query strings.

    Returns:
        list[str]: Plain-string questions with numbering/bullets/preamble
        stripped. Always non-empty — falls back to ``[user_query]`` on a missing
        API key or any exception, so the pipeline can never stall here.

    Mathematical/Logical Rationale:
        Query expansion increases *recall* by sampling several points in the
        embedding space around the user's intent. Each prompt angle targets a
        different lexical register:
          1. formal/regulatory  → matches the policy's own statutory wording;
          2. plain student-facing → matches colloquial Condition-node text;
          3. synonym/adjacent    → matches semantically neighbouring nodes.
        Temperature is fixed at 0 deliberately: variant *diversity* is supplied
        by the three explicit angles, not by sampling noise, which keeps the
        expanded query set reproducible across evaluation runs (a requirement for
        comparable RAGAS scores).
    """
    if not user_query.strip():
        return [user_query]

    if not OPENAI_API_KEY:
        return [user_query]

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=400,
        api_key=OPENAI_API_KEY,
        **_llm_kwargs(),
    )

    prompt = (
        f"You are helping to retrieve university policy information. "
        f"Generate exactly {num_queries} alternative phrasings of the question below. "
        f"Each variant should approach the question differently:\n"
        f"  1. Use formal policy / regulatory language\n"
        f"  2. Use plain student-facing language\n"
        f"  3. Focus on related synonyms or adjacent concepts\n\n"
        f"Output ONLY the {num_queries} questions — one per line, "
        f"no numbering, no bullets, no preamble.\n\n"
        f"Question: {user_query.strip()}"
    )

    try:
        resp = llm.invoke(prompt)
        text = (getattr(resp, "content", "") or str(resp)).strip()
        variants = []
        for line in text.splitlines():
            line = line.strip().lstrip("-•123456789. )").strip().strip('"').strip()
            if line and line.lower() != user_query.strip().lower():
                variants.append(line)
        return variants[:num_queries] if variants else [user_query]
    except Exception:
        return [user_query]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse several ranked node-ID lists into one consensus ranking via RRF.

    Args:
        ranked_lists (list[list[str]]): Each inner list is a ranking of node IDs,
            best first. Lists may overlap and need not be the same length. In
            this system the lists are the per-query vector rankings (Fusion mode)
            or the vector + keyword streams (Direct-Lookup mode).
        k (int): Smoothing constant. Default 60 (see rationale below).

    Returns:
        list[tuple[str, float]]: ``(node_id, rrf_score)`` pairs sorted by
        descending fused score.

    Mathematical/Logical Rationale:
        Reciprocal Rank Fusion scores each document by

            RRF(d) = Σ_i  1 / (k + rank_i(d))          (rank 1-indexed)

        summed over every list ``i`` in which ``d`` appears. Two properties make
        this the right choice for heterogeneous streams:
          • **Scale-invariance.** RRF consumes *ranks*, never raw scores, so a
            cosine similarity (0–1) and a keyword match-count (0–N) can be fused
            without normalising two incomparable distributions.
          • **Consensus reward.** A node ranked highly across multiple lists
            accumulates contributions from each, so agreement between streams
            dominates any single stream's idiosyncratic ordering.
        The constant ``k`` damps the influence of top ranks: as k grows, the gap
        between rank 1 and rank 2 shrinks, making the fusion more forgiving of a
        node's absence from one list. ``k = 60`` is the value validated in the
        original Cormack et al. RRF work and is retained here unchanged — it is
        empirically robust across sparse/dense heterogeneous corpora, which is
        exactly the vector-plus-keyword regime this system fuses.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, node_id in enumerate(ranked, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ─────────────────────────────────────────────────────────────────────────────
# 0.6  CORRECTIVE RAG (CRAG) — CONTEXT RELEVANCE VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — Corrective RAG gate
#   evaluate_context_relevance() is an *independent judge* of retrieval quality,
#   sitting between retrieval and answer synthesis. app.py invokes it at two
#   points (pre-generation on the fallback path; post-hoc on the primary path);
#   a non-RELEVANT verdict downgrades the answer to an honest refusal rather than
#   letting the QA LLM synthesise over weak context.
#
# EXTENDING THIS SECTION
#   • To add a new verdict class (e.g. "PARTIAL"), extend the prompt's
#     classification rules AND the accepted-token set in the parse loop below;
#     then handle the new status at the call sites in app.py.run_query.
#   • To make the gate stricter, lower the bar at the call site (treat AMBIGUOUS
#     as a refusal) rather than editing this judge — keep the judge's vocabulary
#     stable so audit-trail verdicts remain comparable across versions.
#   • To swap the judge model, change only the ChatOpenAI(model=...) line; the
#     strict two-line STATUS/REASON contract is model-agnostic.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_context_relevance(retrieved_rows: list, user_query: str) -> dict:
    """Judge whether retrieved policy context can reliably answer the query (CRAG).

    Args:
        retrieved_rows (list): Neo4j result dicts (from ``graph.query()``). Only
            the top 6 rows and a fixed key subset are sent to the judge to keep
            the prompt small and the verdict cheap.
        user_query (str): The original user question.

    Returns:
        dict: ``{"status": "RELEVANT"|"IRRELEVANT"|"AMBIGUOUS", "reason": str}``.

    Mathematical/Logical Rationale:
        This is the "evaluate" step of Corrective RAG: rather than trusting that
        high retrieval scores imply *answerability*, an LLM judge inspects the
        actual content and returns a discrete verdict. The design encodes two
        deliberate asymmetries:
          • **Short-circuit on emptiness.** No rows ⇒ ``IRRELEVANT`` with no LLM
            call — a degenerate input needs no model to classify, saving latency
            and tokens.
          • **Fail-open contract.** A missing key, empty query, or any exception
            defaults to ``RELEVANT``. The validation layer is a *guard*, not a
            *gate that can lock out a legitimate answer*: a flaky judge must never
            be more harmful than no judge. The cost of a rare false-RELEVANT is
            bounded by the QA prompt's own grounding rules, whereas a
            false-IRRELEVANT would silently suppress a correct answer.
        Temperature is 0 so the same context yields the same verdict, which keeps
        the ``crag_status`` field in the audit trail reproducible.
    """
    if not retrieved_rows:
        return {
            "status": "IRRELEVANT",
            "reason": "No policy context was retrieved.",
        }
    if not (user_query or "").strip():
        return {
            "status": "RELEVANT",
            "reason": "Empty query — validation skipped.",
        }

    # Build a compact context snapshot: top 6 rows, key fields only.
    # We deliberately avoid sending full raw rows to keep the prompt small.
    def _snippet(row: dict) -> str:
        """Compress one result row to a short key:value string (≤120 chars/field).

        Keeps the CRAG judge prompt small by sending only the salient fields of
        the top rows rather than full raw dicts.
        """
        parts = []
        for key in ("policyTitle", "ruleLabel", "ruleSection",
                    "ruleDescription", "conditionText", "outcomes"):
            val = row.get(key)
            if val:
                text = (str(val[0]) if isinstance(val, list) and val else str(val))
                parts.append(f"{key}: {text[:120]}")
        return " | ".join(parts) if parts else str(row)[:200]

    snapshot = "\n".join(
        f"[{i + 1}] {_snippet(r)}"
        for i, r in enumerate(retrieved_rows[:6])
    )

    if not OPENAI_API_KEY:
        return {
            "status": "RELEVANT",
            "reason": "CRAG LLM unavailable — validation skipped.",
        }

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=80,
        api_key=OPENAI_API_KEY,
        **_llm_kwargs(),
    )

    prompt = (
        "You are a retrieval quality judge for a university policy knowledge base.\n\n"
        f"USER QUERY:\n{user_query.strip()}\n\n"
        f"RETRIEVED POLICY CONTEXT:\n{snapshot}\n\n"
        "Assess whether the retrieved context contains sufficient and relevant "
        "policy information to answer the query accurately.\n\n"
        "Classification rules:\n"
        "  RELEVANT   — context directly addresses the query with specific policy details\n"
        "  IRRELEVANT — context is about a different topic or contains no useful information\n"
        "  AMBIGUOUS  — context partially matches but is missing critical details\n\n"
        "Reply with EXACTLY this format (two lines, nothing else):\n"
        "STATUS: <RELEVANT|IRRELEVANT|AMBIGUOUS>\n"
        "REASON: <one sentence explanation, max 25 words>"
    )

    try:
        resp = llm.invoke(prompt)
        text = (getattr(resp, "content", "") or str(resp)).strip()

        status = "RELEVANT"
        reason = "Context evaluated."
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("STATUS:"):
                raw_val = line.split(":", 1)[1].strip().upper()
                if raw_val in ("RELEVANT", "IRRELEVANT", "AMBIGUOUS"):
                    status = raw_val
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return {"status": status, "reason": reason}
    except Exception:
        return {
            "status": "RELEVANT",
            "reason": "CRAG validation error — proceeding normally.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 0.7  ADAPTIVE RAG — QUERY COMPLEXITY ROUTER
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — Adaptive RAG router
#   route_query() chooses the retrieval *profile* before any embedding or graph
#   work begins, so the expensive Fusion pipeline only runs when a question
#   actually needs it. app.py maps the verdict to a single boolean
#   (`use_fusion = route == "COMPLEX_REASONING"`) threaded through the fallback.
#
# EXTENDING THIS SECTION
#   • To add a third route (e.g. "AGGREGATION" for count/statistics questions),
#     extend the prompt's option list and the ROUTE token parser below, then
#     branch on the new route in app.py.run_query.
#   • The router is intentionally LLM-based, not keyword-based, so it generalises
#     to unseen phrasings; if cost becomes a concern, a few-shot distilled
#     classifier can replace the ChatOpenAI call without changing the contract.
# ─────────────────────────────────────────────────────────────────────────────

def route_query(user_query: str, graph=None, chain=None) -> dict:
    """Classify query complexity to select the optimal retrieval path (Adaptive RAG).

    Args:
        user_query (str): The cleaned user question.
        graph: Unused here; accepted so the router shares a call signature with
            other entry points and can later consult the live schema if needed.
        chain: Unused here; reserved for the same forward-compatibility reason.

    Returns:
        dict: ``{"route": "DIRECT_LOOKUP"|"COMPLEX_REASONING", "reason": str}``.
            DIRECT_LOOKUP — simple, single-hop, single-section factual questions
            (e.g. "What is the draft submission deadline?"). COMPLEX_REASONING —
            multi-hop, thematic, comparative, or cross-policy questions.

    Mathematical/Logical Rationale:
        This realises the Adaptive RAG principle of matching retrieval *cost* to
        query *complexity*. A direct lookup maps to one section, so multi-query
        expansion would spend tokens and latency for no recall gain; a complex
        question benefits from the diverse-angle Fusion + RRF + CRAG path. Uses
        GPT-4o at T=0, max_tokens=50 for a fast, deterministic two-line verdict.
        **Fail-open contract:** empty query, missing key, or any exception
        returns COMPLEX_REASONING, so the most thorough pipeline is always the
        safe default — misrouting a hard question to the cheap path would harm
        answer quality, whereas the reverse only costs a little extra compute.
    """
    if not (user_query or "").strip():
        return {"route": "COMPLEX_REASONING", "reason": "Empty query — defaulting to full pipeline."}

    if not OPENAI_API_KEY:
        return {"route": "COMPLEX_REASONING", "reason": "Router LLM unavailable — defaulting to full pipeline."}

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=50,
        api_key=OPENAI_API_KEY,
        **_llm_kwargs(),
    )

    prompt = (
        "You are a query complexity classifier for a university policy knowledge base.\n\n"
        f"QUERY: {user_query.strip()}\n\n"
        "Classify this query into exactly one routing path:\n"
        "  DIRECT_LOOKUP     — simple, factual, single-section question\n"
        "  COMPLEX_REASONING — multi-hop, comparative, or cross-policy question\n\n"
        "Reply with EXACTLY this format (two lines, nothing else):\n"
        "ROUTE: <DIRECT_LOOKUP|COMPLEX_REASONING>\n"
        "REASON: <one phrase, max 12 words>"
    )

    try:
        resp = llm.invoke(prompt)
        text = (getattr(resp, "content", "") or str(resp)).strip()

        route  = "COMPLEX_REASONING"
        reason = "Defaulting to full pipeline."
        for line in text.splitlines():
            line = line.strip().lstrip("-•*># ")
            upper = line.upper()
            if upper.startswith("ROUTE:"):
                raw_val = line.split(":", 1)[1].strip().upper()
                # Accept either exact token or a line that contains one of them
                if "DIRECT_LOOKUP" in raw_val:
                    route = "DIRECT_LOOKUP"
                elif "COMPLEX_REASONING" in raw_val:
                    route = "COMPLEX_REASONING"
            elif upper.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return {"route": route, "reason": reason}
    except Exception as _e:
        return {"route": "COMPLEX_REASONING", "reason": f"Router fallback ({type(_e).__name__}) — full pipeline."}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NEO4J CONNECTION  (Part 1 of the brief)
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — graph connection & live schema
#   The schema returned by refresh_schema() is injected verbatim into the Cypher
#   prompt, so the LLM only ever sees node labels / relationship types / property
#   keys that actually exist. This is the single most important hallucination
#   guard in the pipeline: it makes inventing a non-existent edge type unlikely.
#
# EXTENDING THIS SECTION
#   • To point at a different database, change only the .env variables — no code
#     change is needed.
#   • `enhanced_schema=True` adds sampled property *values* to the schema text,
#     which materially improves Cypher quality; turn it off only if the schema
#     string grows large enough to crowd the prompt's token budget.
#   • `sanitize=True` strips harmful characters from LLM-generated Cypher; keep
#     it on whenever `allow_dangerous_requests=True` is used downstream.
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> Neo4jGraph:
    """Connect to Neo4j AuraDB and load the live schema for prompt injection.

    Returns:
        Neo4jGraph: A connected graph handle whose ``.schema`` has been refreshed.

    Logical Rationale:
        ``Neo4jGraph`` maintains the Bolt/Bolt+S connection pool and, via
        ``refresh_schema()``, reads every node label, relationship type, and
        property key from the live database. That schema string is later fed into
        ``CYPHER_GENERATION_PROMPT``, so the Cypher LLM is grounded in what the
        graph *actually* contains rather than what it might assume — the primary
        defence against generating traversals over edges that do not exist.
        ``enhanced_schema=True`` enriches that text with sampled property values;
        ``sanitize=True`` hardens the graph against malicious generated Cypher.
    """
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        enhanced_schema=True,   # includes property value samples in schema
        sanitize=True,          # strips harmful characters from LLM-generated Cypher
    )
    graph.refresh_schema()
    print("[Neo4j] Connected to AuraDB.")
    print("[Neo4j] Schema loaded:")
    print(textwrap.indent(graph.schema, "    "))
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LLM SETUP — TWO SEPARATE MODELS
# ─────────────────────────────────────────────────────────────────────────────
# We use two LLM instances deliberately:
#
#   cypher_llm  — converts natural language to Cypher.
#                 Temperature = 0: we want deterministic, syntactically
#                 correct Cypher, not creative variation.
#
#   qa_llm      — generates the final grounded answer from the graph result.
#                 Temperature = 0.1: allows slightly more natural phrasing
#                 while still being tightly constrained by the system prompt.
# ─────────────────────────────────────────────────────────────────────────────

def build_llms():
    """Construct the two purpose-separated GPT-4o instances used by the chain.

    Returns:
        tuple[ChatOpenAI, ChatOpenAI]: ``(cypher_llm, qa_llm)``.

    Logical Rationale:
        Generation and synthesis have opposite tolerance for creativity, so they
        get different temperatures:
          • ``cypher_llm`` (T=0) — Cypher must be deterministic and syntactically
            exact; any sampling noise risks an invalid or subtly wrong query.
          • ``qa_llm`` (T=0.1) — a touch of warmth yields more natural prose while
            the QA system prompt still pins the answer to the graph context.
        Both receive ``**_llm_kwargs()`` so they share the truststore HTTP client.
    """
    cypher_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=OPENAI_API_KEY,
        **_llm_kwargs(),
    )
    qa_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=OPENAI_API_KEY,
        **_llm_kwargs(),
    )
    return cypher_llm, qa_llm


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — Cypher-generation & grounded-answer contracts
#   CYPHER_GENERATION_TEMPLATE teaches the LLM the full relational vocabulary of
#   the graph (node types, edge types, mandatory query rules, worked examples).
#   QA_SYSTEM_PROMPT constrains the answer LLM to the returned rows only.
#
# EXTENDING THIS SECTION — adding a NEW EDGE TYPE to the schema
#   1. Add the edge to the "RELATIONSHIP TYPES" block below with its direction
#      and any edge properties (mirror the existing [:CONFLICTS] entry).
#   2. If the edge participates in eligibility/escalation reasoning, add a line
#      to "MANDATORY QUERY RULES" telling the LLM when to OPTIONAL MATCH it.
#   3. Add at least one worked EXAMPLE that traverses the new edge — few-shot
#      examples raise Cypher accuracy far more than prose rules alone.
#   4. Mirror the ingestion side in app.py.ingest_xml_to_neo4j so the edge is
#      actually materialised, and update QA_SYSTEM_PROMPT if the answer must
#      surface the new relationship explicitly (cf. RULE 2/3/4 below).
# ─────────────────────────────────────────────────────────────────────────────

# ── 3a. CYPHER GENERATION PROMPT ─────────────────────────────────────────────
# This instructs the first LLM how to convert a question into Cypher.
# Critical rules:
#   - ALWAYS return sourceSection so the answer LLM can cite it
#   - ALWAYS traverse OVERRIDES and CONFLICTS edges for any eligibility query
#   - NEVER invent node IDs or property values not in the schema
# ─────────────────────────────────────────────────────────────────────────────

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher query generator for the Bournemouth University
Knowledge Graph (BU 8A Code of Practice for Research Degrees 2024-25).

Graph Schema:
{schema}

═══════════════════════ NODE TYPES ════════════════════════════════════════════
  (:Rule)        — id, label, sourceSection, description, policyDocument
  (:Condition)   — id, text, plus domain-specific properties:
                   deadlineFT, deadlinePT, wordLimit, independencePeriodYears,
                   minimumExaminers, specialCase
  (:Outcome)     — id, type, text, timescaleFT, timescalePT, validityMonths
  (:Actor)       — id, role  [PGR, Supervisor, FRDC, Examiner, ...]
  (:PolicyDomain)— id, name

═══════════════════════ RELATIONSHIP TYPES ════════════════════════════════════
  [:HAS_CONDITION]   Rule      → Condition
  [:HAS_OUTCOME]     Rule      → Outcome
  [:GOVERNED_BY]     Actor     → Rule
  [:SATISFIES]       Actor     → Condition
  [:TRIGGERS]        Rule      → Outcome
  [:ESCALATES_TO]    Rule/Outcome → Rule
  [:OVERRIDES]       Rule      → Rule  (mitigating rule takes precedence)
  [:CONFLICTS]       Condition ↔ Condition  (props: scenario, resolution)
  [:RESTRICTS]       Rule/Outcome → Actor
  [:BELONGS_TO]      Rule      → PolicyDomain

═══════════════════════ MANDATORY QUERY RULES ═════════════════════════════════
1. ALWAYS include r.sourceSection in RETURN so answers can cite policy sections.
2. For ANY query involving eligibility, deadlines, or approvals:
     - OPTIONAL MATCH mitigating Rules via [:OVERRIDES]
     - OPTIONAL MATCH [:ESCALATES_TO] chains
3. For ANY query involving examiner appointment or BU staff PGRs:
     - MATCH the [:CONFLICTS] edge between Conditions
     - RETURN cf.scenario AND cf.resolution from the CONFLICTS relationship
4. NEVER use LIMIT unless the user explicitly asks for a count or sample.
5. Use OPTIONAL MATCH (not MATCH) for edges that may not exist — this prevents
   empty result sets when a mitigating rule has not yet been encoded.
6. Return node IDs alongside text properties so answers are traceable.

═══════════════════════ EXAMPLES ══════════════════════════════════════════════
Question: What happens if a PGR misses their Probationary Review deadline?
Cypher:
MATCH (r:Rule {{id: "R4.1"}})-[:HAS_CONDITION]->(c:Condition {{id: "C4.1.2"}})
MATCH (r)-[:HAS_OUTCOME]->(o:Outcome)
OPTIONAL MATCH (mit:Rule)-[:OVERRIDES]->(r)
OPTIONAL MATCH (o {{type: "Withdrawal"}})-[:ESCALATES_TO]->(esc:Rule)
RETURN r.id, r.label, r.sourceSection,
       c.id, c.text,
       collect(o.type)         AS outcomeTypes,
       collect(o.text)         AS outcomeTexts,
       mit.label               AS mitigatingRule,
       mit.sourceSection       AS mitigatingSection,
       esc.id                  AS escalatesTo,
       esc.label               AS escalationLabel

Question: Can Professor Walsh examine Dr Chen who is a BU staff PGR?
Cypher:
MATCH (r:Rule {{id: "R5.1"}})-[:HAS_CONDITION]->(c1:Condition {{id: "C5.1.2"}})
MATCH (r)-[:HAS_CONDITION]->(c2:Condition {{id: "C5.1.6"}})
OPTIONAL MATCH (c1)-[cf:CONFLICTS]->(c2)
MATCH (r)-[:HAS_OUTCOME]->(o:Outcome)
RETURN r.id, r.label, r.sourceSection,
       c1.id, c1.text,
       c2.id, c2.text, c2.independencePeriodYears,
       cf.scenario      AS conflictScenario,
       cf.resolution    AS conflictResolution,
       collect(o.id)    AS outcomeIds,
       collect(o.type)  AS outcomeTypes,
       collect(o.text)  AS outcomeTexts

═══════════════════════════════════════════════════════════════════════════════
Now write ONLY the Cypher query for the following question. Output raw Cypher
with no explanation, no markdown fences, no preamble.

Question: {question}
Cypher:
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)


# ── 3b. GROUNDED ANSWER PROMPT  (Part 3 of the brief) ────────────────────────
# This instructs the answer LLM to stay strictly within the graph context.
# It must:
#   - Cite sourceSection for every factual claim
#   - Surface OVERRIDES resolutions before stating a negative outcome
#   - Never invent rules or sections not present in the graph context
# ─────────────────────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """
You are a policy advisor for Bournemouth University Postgraduate Research students.
Your ONLY source of knowledge is the graph query result provided below.
You MUST follow every rule in this system prompt without exception.

════════════════════════ GROUNDING RULES ══════════════════════════════════════

RULE 1 — CITE EVERY FACTUAL CLAIM
  After every policy statement, cite the sourceSection in parentheses.
  Format: (BU CoP §<section>)
  Example: "The Probationary Review must be submitted within 3 months of
            full-time enrolment (BU CoP §10.4.2)."

RULE 2 — OVERRIDES TAKE PRIORITY
  If the graph context contains a mitigatingRule or an OVERRIDES relationship,
  state the mitigating rule BEFORE stating any negative outcome.
  Never lead with withdrawal or failure if a mitigating path exists.

RULE 3 — SURFACE CONFLICTS EXPLICITLY
  If the graph context contains a conflictScenario and conflictResolution,
  state both clearly. Label the conflict: "Conflicting requirements detected."
  Then state the resolution.

RULE 4 — ESCALATION CHAINS
  If escalatesTo is present in the graph context, explain the full escalation
  sequence with its time windows (e.g. "10 working days per step").

RULE 5 — NEVER INVENT
  If information needed to answer the question is NOT in the graph context,
  say exactly: "This information is not available in the current policy graph.
  Please consult the BU Doctoral College directly."
  Do NOT guess, approximate, or use general knowledge.

RULE 6 — STRUCTURE
  Always structure the answer as:
    1. Direct answer (1–2 sentences)
    2. Policy basis (cited rules and conditions)
    3. Applicable outcomes (with timescales where present)
    4. Mitigating or conflicting rules (if any)
    5. Recommended action for the student/actor

════════════════════════════════════════════════════════════════════════════════

Graph Query Result:
{context}

Question: {question}

Answer:
"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CHAIN ASSEMBLY  (Part 2 of the brief)
# ─────────────────────────────────────────────────────────────────────────────

def build_chain(graph: Neo4jGraph, cypher_llm, qa_llm) -> GraphCypherQAChain:
    """Assemble the two-LLM GraphCypherQAChain (generation → execution → synthesis).

    Args:
        graph (Neo4jGraph): Connected graph whose live schema feeds the Cypher prompt.
        cypher_llm: GPT-4o (T=0) instance that turns NL → Cypher.
        qa_llm: GPT-4o (T=0.1) instance that turns graph rows → grounded answer.

    Returns:
        GraphCypherQAChain: A chain configured to return intermediate steps so the
        generated Cypher and raw rows are inspectable for the audit trail.

    Logical Rationale:
        The chain runs three internal steps:
          A. cypher_llm + CYPHER_GENERATION_PROMPT → Cypher string
          B. execute Cypher on Neo4j → raw graph result (list of dicts)
          C. qa_llm + QA_PROMPT + graph result → grounded natural-language answer
        ``top_k=20`` caps rows returned per query: large enough to capture a
        rule plus its conditions/outcomes/conflicts for a policy KG of this size,
        small enough to keep the synthesis prompt well within budget.
        ``return_intermediate_steps=True`` is what makes the system *transparent*
        — the exact Cypher and rows are surfaced rather than hidden.

    Key parameters:
      allow_dangerous_requests  — must be True; LangChain requires explicit opt-in
                                  because the chain executes LLM-generated code on
                                  a live database. Safe here because:
                                    (a) we use sanitize=True on the graph
                                    (b) the Neo4j user has read-only access (see docs)
      return_intermediate_steps — set True so we can inspect the generated Cypher
                                  in the response, which gives full traceability
      verbose                   — logs the Cypher query to stdout during development
      top_k                     — max nodes returned per query; 20 is appropriate
                                  for policy KGs of this size
    """
    chain = GraphCypherQAChain.from_llm(
        llm=qa_llm,
        graph=graph,
        cypher_llm=cypher_llm,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        qa_prompt=QA_PROMPT,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        verbose=True,
        top_k=20,
        validate_cypher=True,   # LangChain validates syntax before execution
    )
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SCENARIO-BASED RETRIEVAL HELPERS  (Part 4 of the brief)
# ─────────────────────────────────────────────────────────────────────────────
# For scenarios that require specific graph traversal patterns (OVERRIDES,
# CONFLICTS, ESCALATES_TO), we provide pre-built direct Cypher queries that
# bypass the LLM-to-Cypher step. This guarantees the correct edges are always
# traversed, regardless of how the user phrases the question.
#
# Pattern: scenario functions run raw Cypher via graph.query(), then format
# the result into the grounding context that the QA LLM consumes.
# ─────────────────────────────────────────────────────────────────────────────

def scenario_a_sick_leave_override(graph: Neo4jGraph) -> dict:
    """Run the pre-built Scenario A traversal: illness override of a missed deadline.

    Args:
        graph (Neo4jGraph): Connected graph handle.

    Returns:
        dict: ``{"scenario", "description", "graph_result"}`` where
        ``graph_result`` is the raw Neo4j rows for the QA LLM to ground on.

    Logical Rationale:
        Eligibility questions hinge on whether a *mitigating* rule suppresses a
        negative outcome. A free-form LLM-to-Cypher attempt may omit the
        ``OVERRIDES`` / ``ESCALATES_TO`` traversal and wrongly report withdrawal.
        Hard-coding the traversal (R4.1 ─HAS_CONDITION→ C4.1.2, then OPTIONAL
        MATCH the OVERRIDES and ESCALATES_TO edges) *guarantees* the mitigating
        path is always inspected, regardless of phrasing.
    """
    cypher = """
    MATCH (r:Rule {id: "R4.1"})-[:HAS_CONDITION]->(c:Condition {id: "C4.1.2"})
    MATCH (r)-[:HAS_OUTCOME]->(o:Outcome)

    // OVERRIDES: find any Rule that mitigates the consequences of R4.1
    OPTIONAL MATCH (mit:Rule)-[:OVERRIDES]->(r)

    // ESCALATES_TO: follow withdrawal consequence to next active rule
    OPTIONAL MATCH (o_wd:Outcome {type: "Withdrawal"})<-[:HAS_OUTCOME]-(r)
    OPTIONAL MATCH (o_wd)-[:ESCALATES_TO]->(esc:Rule)

    RETURN
        r.id                   AS ruleId,
        r.label                AS ruleLabel,
        r.sourceSection        AS ruleSection,
        c.id                   AS conditionId,
        c.text                 AS conditionText,
        c.deadlineFT           AS deadlineFT,
        c.deadlinePT           AS deadlinePT,
        collect(DISTINCT o.type + ": " + o.text) AS outcomes,
        mit.id                 AS mitigatingRuleId,
        mit.label              AS mitigatingRuleLabel,
        mit.sourceSection      AS mitigatingSection,
        esc.id                 AS escalationRuleId,
        esc.label              AS escalationLabel,
        esc.sourceSection      AS escalationSection
    """
    result = graph.query(cypher)
    return {
        "scenario": "A — Borderline Eligibility (Illness Override)",
        "description": (
            "PGR missed Probationary Review deadline due to illness. "
            "Query checks for OVERRIDES edges that suppress the withdrawal outcome."
        ),
        "graph_result": result,
    }


def scenario_b_conflicts_edge(graph: Neo4jGraph) -> dict:
    """Run the pre-built Scenario B traversal: BU-staff-PGR examiner-independence conflict.

    Args:
        graph (Neo4jGraph): Connected graph handle.

    Returns:
        dict: ``{"scenario", "description", "graph_result"}``; the rows expose
        ``conflictScenario`` and ``conflictResolution`` from the CONFLICTS edge.

    Logical Rationale:
        The resolution of a conflicting-requirements case is stored *on the edge*
        (``CONFLICTS {scenario, resolution}``) between the two competing
        Conditions, not on either node. The query therefore OPTIONAL MATCHes the
        edge and returns its properties so the QA LLM can surface the encoded
        resolution verbatim instead of attempting to reason it out.
    """
    cypher = """
    MATCH (r:Rule {id: "R5.1"})
    MATCH (r)-[:HAS_CONDITION]->(c1:Condition {id: "C5.1.2"})
    MATCH (r)-[:HAS_CONDITION]->(c2:Condition {id: "C5.1.6"})

    // CONFLICTS edge — carries scenario and resolution as properties
    OPTIONAL MATCH (c1)-[cf:CONFLICTS]->(c2)

    MATCH (r)-[:HAS_OUTCOME]->(o:Outcome)

    RETURN
        r.id                        AS ruleId,
        r.label                     AS ruleLabel,
        r.sourceSection             AS ruleSection,
        c1.id                       AS condition1Id,
        c1.text                     AS condition1Text,
        c2.id                       AS condition2Id,
        c2.text                     AS condition2Text,
        c2.independencePeriodYears  AS independenceWindowYears,
        cf.scenario                 AS conflictScenario,
        cf.resolution               AS conflictResolution,
        collect(DISTINCT o.type + ": " + o.text) AS outcomes
    """
    result = graph.query(cypher)
    return {
        "scenario": "B — Conflicting Rules (BU Staff PGR + Examiner Association)",
        "description": (
            "BU staff PGR proposed external examiner has recent association. "
            "Query surfaces the CONFLICTS edge and its encoded resolution."
        ),
        "graph_result": result,
    }


def scenario_c_escalation_chain(graph: Neo4jGraph) -> dict:
    """Run the pre-built Scenario C traversal: two simultaneous escalation chains.

    Args:
        graph (Neo4jGraph): Connected graph handle.

    Returns:
        dict: ``{"scenario", "description", "graph_result"}`` containing both
        escalation chains so the LLM can explain every active process.

    Logical Rationale:
        A single student may be subject to more than one escalation process at
        once (here: probationary-review failure *and* an engagement lapse). The
        query deliberately traverses *both* ``ESCALATES_TO`` chains in one shot —
        Chain 1 from the R4.1 Withdrawal outcome, Chain 2 R3.1→R3.2→outcome — so
        the answer cannot accidentally report only one of two concurrent risks.
    """
    cypher = """
    // Chain 1: Probationary Review failure escalation
    MATCH (r_prob:Rule {id: "R4.1"})-[:HAS_OUTCOME]->(o_prob:Outcome {type: "Withdrawal"})
    OPTIONAL MATCH (o_prob)-[:ESCALATES_TO]->(esc1:Rule)

    // Chain 2: Lack of engagement escalation
    MATCH (r_eng:Rule {id: "R3.1"})
    OPTIONAL MATCH (r_eng)-[:ESCALATES_TO]->(r_lack:Rule {id: "R3.2"})
    OPTIONAL MATCH (r_lack)-[:HAS_OUTCOME]->(o_lack:Outcome)

    RETURN
        r_prob.id              AS probRuleId,
        r_prob.label           AS probRuleLabel,
        r_prob.sourceSection   AS probSection,
        o_prob.type            AS probOutcomeType,
        o_prob.text            AS probOutcomeText,
        esc1.id                AS escalation1Id,
        esc1.label             AS escalation1Label,
        esc1.sourceSection     AS escalation1Section,

        r_eng.id               AS engRuleId,
        r_eng.label            AS engRuleLabel,
        r_eng.sourceSection    AS engSection,
        r_lack.id              AS lackRuleId,
        r_lack.label           AS lackRuleLabel,
        r_lack.sourceSection   AS lackSection,
        collect(DISTINCT o_lack.type + ": " + o_lack.text) AS lackOutcomes
    """
    result = graph.query(cypher)
    return {
        "scenario": "C — Escalation Chain (Failed Review + Engagement Lapse)",
        "description": (
            "Two simultaneous escalation chains are active for the same PGR. "
            "Query traverses both ESCALATES_TO paths and returns all active consequences."
        ),
        "graph_result": result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QUERY ROUTER
# ─────────────────────────────────────────────────────────────────────────────
# The router detects whether a question should bypass LLM-to-Cypher generation
# and go straight to a pre-built scenario query. This ensures the most
# important graph edges (OVERRIDES, CONFLICTS, ESCALATES_TO) are always
# traversed correctly.
#
# Detection is intentionally keyword-based here. In a production system,
# replace this with an intent classifier (fine-tuned or few-shot prompted).
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_KEYWORDS = {
    "scenario_a": [
        "sick", "illness", "ill", "hospital", "medical",
        "missed", "missed deadline", "probationary review", "extension",
    ],
    "scenario_b": [
        "staff", "examiner", "independence", "association",
        "co-author", "external examiner", "conflict", "bu staff",
    ],
    "scenario_c": [
        "engagement", "withdrawal", "escalat", "lack of engagement",
        "no response", "failing review", "withdrawn",
    ],
}

def cli_route_query(question: str, graph: Neo4jGraph, chain: GraphCypherQAChain) -> dict:
    """Route a question to a guaranteed-coverage scenario query or the open LLM chain.

    Args:
        question (str): The raw user question.
        graph (Neo4jGraph): Connected graph handle for scenario traversals.
        chain (GraphCypherQAChain): The LLM-to-Cypher chain for open-domain queries.

    Returns:
        dict: ``{"answer", "cypher_used", "graph_result", "route"}`` where
        ``route`` ∈ {"scenario_A", "scenario_B", "scenario_C", "llm_chain"}.

    Logical Rationale:
        Detection is keyword-based and ordered B → A → C, not alphabetical: the
        Scenario-B examiner/conflict vocabulary is the most specific, so it is
        tested first to avoid a generic word (e.g. "missed") capturing a question
        that actually concerns examiner independence. Any question matching no
        scenario falls through to the LLM chain. In production this keyword gate
        would be replaced by an intent classifier (see the section comment).
    """
    q_lower = question.lower()

    # Check for scenario-specific keywords
    if any(kw in q_lower for kw in SCENARIO_KEYWORDS["scenario_b"]):
        raw = scenario_b_conflicts_edge(graph)
        context = str(raw["graph_result"])
        answer = chain.qa_chain.invoke({"context": context, "question": question})
        return {
            "answer": answer,
            "cypher_used": "Pre-built: scenario_b_conflicts_edge()",
            "graph_result": raw["graph_result"],
            "route": "scenario_B",
        }

    if any(kw in q_lower for kw in SCENARIO_KEYWORDS["scenario_a"]):
        raw = scenario_a_sick_leave_override(graph)
        context = str(raw["graph_result"])
        answer = chain.qa_chain.invoke({"context": context, "question": question})
        return {
            "answer": answer,
            "cypher_used": "Pre-built: scenario_a_sick_leave_override()",
            "graph_result": raw["graph_result"],
            "route": "scenario_A",
        }

    if any(kw in q_lower for kw in SCENARIO_KEYWORDS["scenario_c"]):
        raw = scenario_c_escalation_chain(graph)
        context = str(raw["graph_result"])
        answer = chain.qa_chain.invoke({"context": context, "question": question})
        return {
            "answer": answer,
            "cypher_used": "Pre-built: scenario_c_escalation_chain()",
            "graph_result": raw["graph_result"],
            "route": "scenario_C",
        }

    # General query — use LLM-to-Cypher chain
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "cypher_used": result["intermediate_steps"][0]["query"]
                       if result.get("intermediate_steps") else "N/A",
        "graph_result": result["intermediate_steps"][1]["context"]
                        if result.get("intermediate_steps") else "N/A",
        "route": "llm_chain",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DISPLAY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    """Pretty-print a query result (route, Cypher, answer, raw rows) to stdout.

    Args:
        result (dict): A result dict as returned by ``cli_route_query``.

    Returns:
        None. Output is written to stdout, word-wrapped to 70 columns for
        terminal readability; the raw graph result is printed as an audit trail.
    """
    separator = "═" * 72
    print(f"\n{separator}")
    print(f"  ROUTE    : {result['route']}")
    print(f"  CYPHER   : {result['cypher_used']}")
    print(separator)
    print("\n  ANSWER:\n")
    # Wrap long lines for terminal readability
    for line in result["answer"].split("\n"):
        print(textwrap.fill(line, width=70, subsequent_indent="    "))
    print(f"\n{separator}")
    print(f"  RAW GRAPH RESULT (audit trail):\n  {result['graph_result']}")
    print(f"{separator}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  DEMO QUERIES
# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [

    # ── Scenario A — OVERRIDES path (illness override of withdrawal)
    "I am a full-time PhD student and I missed my Probationary Review deadline "
    "because I was ill and in hospital. What are my options now?",

    # ── Scenario B — CONFLICTS path (BU staff PGR, examiner independence)
    "A BU staff member PhD student has proposed an external examiner who "
    "co-authored a paper with the supervisor 14 months ago. Is this allowed?",

    # ── Scenario C — ESCALATES_TO path (dual escalation chains)
    "A PhD student failed to resubmit after their Probationary Review extension "
    "and has also not recorded any engagement for two months. "
    "What withdrawal process applies?",

    # ── General open-domain query (uses LLM-to-Cypher chain)
    "What is the maximum number of PGRs a first supervisor can supervise "
    "simultaneously at BU?",
]


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Build and wire the CLI bot's components, returning them ready for the REPL.

    Returns:
        tuple[Neo4jGraph, GraphCypherQAChain]: ``(graph, chain)`` — a connected
        graph and the assembled two-LLM chain, used by the ``__main__`` loop.

    Logical Rationale:
        Construction order is fixed by dependency: the graph must exist (and its
        schema be loaded) before the chain can be assembled, because the chain
        injects that schema into its Cypher prompt. Returning both objects lets
        the interactive loop reuse a single warm connection across questions.
    """
    print("\n" + "═" * 72)
    print("  GraphRAG Policy Bot — BU Code of Practice for Research Degrees")
    print("  Phase 3: Python Integration Layer")
    print("═" * 72 + "\n")

    # Build components
    graph = build_graph()
    cypher_llm, qa_llm = build_llms()
    chain = build_chain(graph, cypher_llm, qa_llm)

    print("\n[System] Chain assembled. Ready for questions.\n")
    return graph, chain


if __name__ == "__main__":
    graph, chain = main()
    print("--- BU Policy GraphRAG Bot is Ready ---")
    while True:
        user_input = input("Your Question (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        result = cli_route_query(user_input, graph, chain)
        print_result(result)