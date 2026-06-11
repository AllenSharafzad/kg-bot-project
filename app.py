"""
app.py — BU Policy GraphRAG Platform
═══════════════════════════════════════════════════════════════════════════════
Transparent Knowledge Management & Reasoning Platform
Author : Alireza (Allen) Sharafzad — MSc Data Science & AI, BU

Run    : streamlit run app.py

Requirements:
    pip install streamlit langchain-neo4j langchain-openai python-dotenv \
                neo4j streamlit-agraph
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Windows UTF-8 ────────────────────────────────────────────────────────────
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Windows SSL: use the OS trust store so corporate/university proxies work ─
# inject_into_ssl() patches ssl.SSLContext globally; _http_client passes an
# explicit httpx.Client with the truststore context to every LLM constructor
# so Streamlit's module cache cannot hold on to a stale SSL context.
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
    """Return {'http_client': <client>} when a truststore client is available."""
    return {"http_client": _http_client} if _http_client is not None else {}

import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import streamlit as st
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from streamlit_agraph import agraph, Node, Edge, Config

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None
from langchain_core.prompts import PromptTemplate

# Fusion RAG + CRAG + Adaptive RAG helpers live in the CLI prototype module.
try:
    from graphrag_policy_bot import (
        generate_multiple_queries,
        reciprocal_rank_fusion,
        evaluate_context_relevance,
        route_query as _route_query,
    )
    _FUSION_RAG_AVAILABLE = True
except Exception:
    _FUSION_RAG_AVAILABLE = False

    # ── Fallback shims ────────────────────────────────────────────────────────
    # These no-op/degraded stand-ins are defined ONLY when graphrag_policy_bot
    # could not be imported, so the app still runs (with Fusion/CRAG/Adaptive RAG
    # disabled) rather than crashing at import. They mirror the real signatures.
    def generate_multiple_queries(q, num_queries=3):  # noqa: F811
        """Fallback: no expansion — return the single query unchanged."""
        return [q]

    def reciprocal_rank_fusion(ranked_lists, k=60):   # noqa: F811
        """Fallback RRF, identical to the real one: Σ 1/(k + rank) per node id."""
        scores = {}
        for lst in ranked_lists:
            for rank, nid in enumerate(lst, 1):
                scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    def evaluate_context_relevance(rows, query):      # noqa: F811
        """Fallback CRAG: fail-open to RELEVANT so the pipeline is never blocked."""
        return {"status": "RELEVANT", "reason": "CRAG module unavailable."}

    def _route_query(query):                          # noqa: F811
        """Fallback router: always COMPLEX_REASONING (the safe, thorough default)."""
        return {"route": "COMPLEX_REASONING", "reason": "Adaptive RAG module unavailable."}


# ─────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BU Policy GraphRAG Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Node colours (as specified)
COLOR_RULE       = "#60a5fa"    # blue
COLOR_CONDITION  = "#f59e0b"    # amber
COLOR_CONFLICT   = "#ef4444"    # red — used for CONFLICTS edges and condition-in-conflict
COLOR_OUTCOME    = "#10b981"    # green (for completeness)
COLOR_ACTOR      = "#a78bfa"    # purple
COLOR_DEFAULT    = "#94a3b8"    # slate


# ─────────────────────────────────────────────────────────────────────────────
# 1.  (no global CSS — UI built with native Streamlit components)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CREDENTIALS
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — credentials & embedder selection
#   Secrets resolve from the environment first, then Streamlit secrets, so the
#   same code runs locally (.env) and on Streamlit Cloud (st.secrets) unchanged.
#   This section also chooses the embedding backend and pins EMBEDDING_DIMS.
#
# EXTENDING THIS SECTION — swapping the embedding fallback model
#   The offline fallback model is named in _init_embedder() below: change the
#   ``"all-MiniLM-L6-v2"`` string (and, if its width differs from 384, the
#   ``return candidate, 384`` dimension) to adopt a different local encoder.
#   Because every vector index is (re)created at the live EMBEDDING_DIMS during
#   ingest, switching dimension is safe ONLY if you re-ingest afterwards so the
#   indices are rebuilt at the new width. To add a third backend, append another
#   (package, class) pair to the probe loop — order = priority.
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

def get_cred(key: str) -> str:
    """Resolve a secret from the environment, falling back to Streamlit secrets.

    Args:
        key (str): The credential name (e.g. ``"OPENAI_API_KEY"``).

    Returns:
        str: The resolved value, or ``""`` if set in neither source.

    Logical Rationale:
        Environment variables take precedence so a developer's local ``.env``
        always wins; ``st.secrets`` is consulted only as a fallback and wrapped
        in try/except because accessing it raises when no secrets file exists
        (the common local-dev case). The empty-string default keeps every call
        site branch-free — callers test truthiness, never catch.
    """
    val = os.getenv(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            val = ""
    return val

NEO4J_URI      = get_cred("NEO4J_URI")
NEO4J_USERNAME = get_cred("NEO4J_USERNAME") or "neo4j"
NEO4J_PASSWORD = get_cred("NEO4J_PASSWORD")
OPENAI_API_KEY    = get_cred("OPENAI_API_KEY")

# ── Embeddings (semantic search) ─────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS  = 1536          # overridden below if HuggingFace fallback is used
# CRITICAL PARAMETER — SEMANTIC_THRESHOLD = 0.70
#   Minimum Neo4j cosine similarity (range [0, 1]) for a vector hit to be KEPT in
#   COMPLEX_REASONING / Fusion mode. Empirical justification (per ARCHITECTURE_REVIEW.md
#   §4c): on the BU 8A corpus, scores below ~0.70 were dominated by topically
#   adjacent but non-answering clauses, so 0.70 is the precision/recall knee that
#   trims drift without discarding genuine paraphrase matches. NOTE: this cut is
#   *intentionally NOT applied* in DIRECT_LOOKUP mode (see _semantic_search §4d),
#   where a keyword-only hit may legitimately score 0.0 cosine yet be the exact
#   factual node required — there the strict top-5-by-RRF rule governs instead.
SEMANTIC_THRESHOLD = 0.70


@st.cache_resource(show_spinner=False)
def _init_embedder():
    """Select the best available embedding backend and report its dimension.

    Returns:
        tuple[object | None, int]: ``(embedder_instance, embedding_dimension)``.
        ``(None, 1536)`` if no backend could be initialised (semantic search is
        then disabled but the app still runs on structured Cypher retrieval).

    Logical Rationale:
        Selection is a priority cascade, each tier gated by a *live* probe call:
          1. OpenAI ``text-embedding-3-small`` (1536-dim) via the truststore HTTP
             client — preferred for quality. A ``embed_query("probe")`` call
             verifies the endpoint is actually reachable *before* the instance is
             trusted, so a present-but-blocked key fails over rather than erroring
             at first real use.
          2. Local ``all-MiniLM-L6-v2`` (384-dim) via langchain-huggingface, then
             langchain-community — fully offline, for restrictive proxies.
        The returned dimension is propagated to ``EMBEDDING_DIMS`` so the vector
        indices are created at whichever width is live (see §5 ingester). The
        function is ``@st.cache_resource`` so the probe runs once per session.
    """
    if OpenAIEmbeddings is not None and OPENAI_API_KEY:
        try:
            candidate = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=OPENAI_API_KEY,
                **_llm_kwargs(),
            )
            candidate.embed_query("probe")          # live connectivity test
            return candidate, 1536
        except Exception:
            pass

    # Local fallback — works fully offline / behind restrictive proxies
    for _pkg, _cls in [
        ("langchain_huggingface",          "HuggingFaceEmbeddings"),
        ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
    ]:
        try:
            _mod = __import__(_pkg, fromlist=[_cls])
            _HFE = getattr(_mod, _cls)
            candidate = _HFE(model_name="all-MiniLM-L6-v2")
            candidate.embed_query("probe")
            return candidate, 384
        except Exception:
            continue

    return None, 1536


_result       = _init_embedder()
_embedder     = _result[0]
EMBEDDING_DIMS = _result[1]     # 1536 (OpenAI) or 384 (HuggingFace local)


def _embed_with_retry(embed_text: str, max_attempts: int = 3) -> list | None:
    """Embed text with exponential-backoff retry to survive transient API blips.

    Args:
        embed_text (str): The text to embed. Empty/falsy input returns ``None``.
        max_attempts (int): Maximum tries before giving up. Default 3.

    Returns:
        list | None: The embedding vector on success, or ``None`` once every
        attempt is exhausted (the caller then skips embedding that node).

    Mathematical/Logical Rationale:
        Delays double each attempt (1 s → 2 s → 4 s). Exponential backoff spaces
        retries so a brief rate-limit or network hiccup during a bulk ingest is
        absorbed without hammering the endpoint, while the hard attempt cap
        guarantees ingestion cannot hang indefinitely on a persistent outage —
        a missing embedding degrades search for one node, not the whole run.
    """
    if _embedder is None or not embed_text:
        return None
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            return _embedder.embed_query(embed_text)
        except Exception:
            if attempt == max_attempts:
                return None
            time.sleep(delay)
            delay *= 2
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — Cypher-generation & grounded-answer contracts (production graph)
#   CYPHER_GENERATION_TEMPLATE is the LIVE app's schema contract; it lists only
#   the relationship types the Streamlit ingester actually materialises
#   (BELONGS_TO, HAS_CONDITION, HAS_OUTCOME, CONFLICTS_WITH, plus OVERRIDES /
#   ESCALATES_TO which the prompt tolerates if present). QA_SYSTEM_TEMPLATE pins
#   the answer to the returned rows and mandates the conflict-detection phrasing.
#
# EXTENDING THIS SECTION — adding a NEW EDGE TYPE
#   1. Add the edge under "RELATIONSHIP TYPES" with direction + properties.
#   2. Add it to MANDATORY RULE 4's OPTIONAL MATCH list so missing edges never
#      wipe the result set.
#   3. Add a worked EXAMPLE that traverses it (few-shot > prose for Cypher).
#   4. Materialise it in ingest_xml_to_neo4j (§5) and, if it must be surfaced to
#      the reader, add a GROUNDING RULE to QA_SYSTEM_TEMPLATE.
#   Keep this template in sync with graphrag_policy_bot.CYPHER_GENERATION_TEMPLATE.
# ─────────────────────────────────────────────────────────────────────────────
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher query generator for the Bournemouth University
Knowledge Graph (BU 8A Code of Practice for Research Degrees 2024-25).

Graph Schema:
{schema}

NODE TYPES:
  (:Policy)      — id, title, version, ingestedAt
  (:Rule)        — id, policy_id, label, label_human, sourceSection, description
  (:Condition)   — id, policy_id, text, label_human, deadlineFT, deadlinePT, wordLimit, ...
  (:Outcome)     — id, policy_id, type, text, label_human
  (:Actor)       — id, role

RELATIONSHIP TYPES:
  [:BELONGS_TO]       Rule/Condition/Outcome → Policy
  [:HAS_CONDITION]    Rule      → Condition
  [:HAS_OUTCOME]      Rule      → Outcome
  [:CONFLICTS_WITH]   Condition ↔ Condition  (props: scenario, resolution, sections, cross_policy)
  [:OVERRIDES]        Rule      → Rule
  [:ESCALATES_TO]     Rule      → Rule

POLICY NAMESPACE — CRITICAL:
• Every Rule/Condition/Outcome carries a policy_id. The SAME node id
  (e.g. R5.1) may exist in multiple policies — NEVER match only on id.
• ALWAYS traverse (:Rule)-[:BELONGS_TO]->(p:Policy) and return p.title AS
  policyTitle and p.id AS policyId so the answer can cite the source document.
• CONFLICTS_WITH edges are almost always intra-policy; a property
  cf.cross_policy=true marks the rare cross-document contradiction.

CRITICAL STRATEGY — READ CAREFULLY:
• Most questions are CONCEPTUAL (e.g. "Can a staff member be an examiner?",
  "What happens if I miss my deadline?") — the user's words will NOT appear as
  node IDs or relationship types. You MUST search the TEXT of nodes.
• Your Cypher MUST use case-insensitive CONTAINS on Condition.text,
  Rule.description, Rule.label, Outcome.text, and label_human.
• Extract the 2-4 most important content keywords from the question, lowercase
  them, and match them with toLower(<prop>) CONTAINS 'keyword'.
• Return a broad result — include the Rule, its Conditions, its Outcomes,
  and any CONFLICTS_WITH relationships.
• Do NOT invent node IDs. Do NOT filter by a specific id unless the user
  explicitly mentions a policy section.

MANDATORY RULES:
1. ALWAYS include r.sourceSection in RETURN for citations.
2. Always return r.id, r.label, c.id, c.text so answers are traceable.
3. ALWAYS join (r)-[:BELONGS_TO]->(p:Policy) and RETURN p.title AS policyTitle
   and p.id AS policyId so the answer can name the source document.
4. Use OPTIONAL MATCH for [:HAS_OUTCOME], [:CONFLICTS_WITH], [:OVERRIDES],
   [:ESCALATES_TO] edges so missing edges don't wipe the result.
5. OUTPUT ONLY raw Cypher — no markdown, no explanation.

EXAMPLE
Question: Can a staff member be an examiner?
Cypher:
MATCH (r:Rule)-[:BELONGS_TO]->(p:Policy)
MATCH (r)-[:HAS_CONDITION]->(c:Condition)
WHERE toLower(c.text) CONTAINS 'staff'
   OR toLower(c.text) CONTAINS 'examiner'
   OR toLower(r.description) CONTAINS 'examiner'
OPTIONAL MATCH (r)-[:HAS_OUTCOME]->(o:Outcome)
OPTIONAL MATCH (c)-[cf:CONFLICTS_WITH]-(c2:Condition)
RETURN p.title AS policyTitle, p.id AS policyId,
       r.id, r.label, r.sourceSection, r.description,
       c.id, c.text, c.label_human,
       collect(DISTINCT o.type + ": " + o.text) AS outcomes,
       cf.scenario    AS conflictScenario,
       cf.resolution  AS conflictResolution,
       cf.cross_policy AS conflictCrossPolicy,
       c2.id          AS conflictingConditionId,
       c2.text        AS conflictingConditionText

EXAMPLE
Question: What happens if I miss my Probationary Review deadline?
Cypher:
MATCH (r:Rule)-[:BELONGS_TO]->(p:Policy)
MATCH (r)-[:HAS_CONDITION]->(c:Condition)
WHERE toLower(r.description) CONTAINS 'probationary'
   OR toLower(c.text) CONTAINS 'probationary'
   OR toLower(c.text) CONTAINS 'deadline'
OPTIONAL MATCH (r)-[:HAS_OUTCOME]->(o:Outcome)
OPTIONAL MATCH (mit:Rule)-[:OVERRIDES]->(r)
RETURN p.title AS policyTitle, p.id AS policyId,
       r.id, r.label, r.sourceSection, r.description,
       c.id, c.text, c.deadlineFT, c.deadlinePT,
       collect(DISTINCT o.type + ": " + o.text) AS outcomes,
       mit.id AS mitigatingRuleId, mit.label AS mitigatingRuleLabel

Question: {question}
Cypher:
"""

QA_SYSTEM_TEMPLATE = """
You are a policy advisor for Bournemouth University Postgraduate Research students.
Your ONLY source of knowledge is the graph query result below.

GROUNDING RULES:
1. SOURCE ATTRIBUTION — every factual paragraph MUST begin with
   "According to the <policyTitle>, …" naming the policyTitle from the
   graph result. If multiple policies appear, attribute each claim to its
   source; never mix claims from different policies in one sentence.
2. CITE every factual claim using the node's sourceSection → "(<policyTitle> §<section>)".
3. CONFLICT DETECTION — if the graph result contains a CONFLICTS_WITH relationship
   (keys such as conflictScenario / conflictResolution / conflictSections, or
   a row with two Conditions joined by CONFLICTS_WITH), you MUST:
     • Begin that part of the answer with the exact phrase:
       "A policy conflict has been detected between Section X and Section Y"
       (substitute the two sourceSection values).
     • If conflictCrossPolicy is true, add "(across <policyTitleA> and <policyTitleB>)".
     • Then state the stored resolution verbatim.
4. If information is absent, reply EXACTLY:
   "This information is not available in the current policy graph.
    Please consult the BU Doctoral College directly."
5. Do NOT use general knowledge outside the graph context.

STRUCTURE THE ANSWER:
1. Direct answer (1–2 sentences)
2. Policy basis (cited rules / conditions)
3. Applicable outcomes
4. Conflicts detected (if any) — use the exact phrase above
5. Recommended action

Graph Query Result:
{context}

Question: {question}

Answer:
"""

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_SYSTEM_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RESOURCE INITIALISATION  (cached)
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — cached, self-healing connections
#   init_neo4j() and init_chain() are @st.cache_resource so the Bolt pool and the
#   GraphCypherQAChain are built once and reused across reruns. Both return a
#   (resource, error) tuple instead of raising, so the UI can render status.
#
# EXTENDING THIS SECTION
#   • Self-healing: a failed connection must NOT pin a (None, error) tuple for
#     the server lifetime. The CALLER clears the cache on failure
#     (init_neo4j.clear() / init_chain.clear()) and the sidebar exposes a
#     "🔄 Reconnect" button — preserve that pattern for any new cached resource.
#   • To change Cypher/QA models or top_k, edit init_chain() only; the prompts
#     live in §3 and the chain wiring is centralised here.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_neo4j():
    """Open (and cache) the Neo4j AuraDB connection with its live schema loaded.

    Returns:
        tuple[Neo4jGraph | None, str | None]: ``(graph, None)`` on success or
        ``(None, error_message)`` on failure — never raises, so the sidebar can
        display the fault and offer a reconnect.

    Logical Rationale:
        Returning an error tuple rather than throwing keeps the cached resource
        contract uniform and lets a cold-start AuraDB failure surface as a status
        line. ``enhanced_schema``/``sanitize`` mirror build_graph() in the CLI
        module. Because the result is cached, the caller must call
        ``init_neo4j.clear()`` after a failure so the next rerun retries live.
    """
    if not (NEO4J_URI and NEO4J_PASSWORD):
        return None, "Missing NEO4J_URI or NEO4J_PASSWORD"
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            enhanced_schema=True, sanitize=True,
        )
        graph.refresh_schema()
        return graph, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def init_chain(_graph):
    """Assemble (and cache) the two-LLM GraphCypherQAChain over the given graph.

    Args:
        _graph (Neo4jGraph): The connected graph. The leading underscore tells
            Streamlit NOT to hash this argument (a Neo4jGraph is unhashable);
            the chain is therefore cached once for the session.

    Returns:
        tuple[GraphCypherQAChain | None, str | None]: ``(chain, None)`` on
        success or ``(None, error_message)`` on failure.

    Logical Rationale:
        Two GPT-4o instances are wired in: cypher_llm (T=0, exact Cypher) and
        qa_llm (T=0.1, fluent grounded prose) — the same generation/synthesis
        split as the CLI module. ``top_k=25`` bounds rows per query; the slightly
        higher value than the prototype's 20 reflects this graph's denser
        Condition fan-out. ``return_intermediate_steps=True`` exposes the Cypher
        and rows that drive the transparency dashboard and Reasoning View.
    """
    if _graph is None or not OPENAI_API_KEY:
        return None, "Graph or API key unavailable"
    try:
        cypher_llm = ChatOpenAI(model="gpt-4o", temperature=0,   api_key=OPENAI_API_KEY, **_llm_kwargs())
        qa_llm     = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY, **_llm_kwargs())
        chain = GraphCypherQAChain.from_llm(
            llm=qa_llm, graph=_graph,
            cypher_llm=cypher_llm,
            cypher_prompt=CYPHER_PROMPT, qa_prompt=QA_PROMPT,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
            verbose=False, top_k=25, validate_cypher=True,
        )
        return chain, None
    except Exception as e:
        return None, str(e)


def count_nodes(graph) -> int:
    """Return the total node count in the graph, or 0 if unavailable.

    Args:
        graph (Neo4jGraph | None): The connected graph, or None.

    Returns:
        int: ``count(n)`` over all nodes; 0 on a missing graph or any query
        error (used only for a sidebar metric, so a soft 0 is preferable to a
        crash).
    """
    if graph is None:
        return 0
    try:
        res = graph.query("MATCH (n) RETURN count(n) AS n")
        return int(res[0]["n"]) if res else 0
    except Exception:
        return 0


def get_ingested_policies(graph) -> list[dict]:
    """
    Read the (:Policy) nodes straight from Neo4j (never from st.session_state)
    with their Rule/Condition/Outcome child counts, newest first.
    Each item: {id, title, version, ingestedAt, rules, conditions, outcomes}.

    Raises on query failure so the caller can surface the real Neo4j error
    instead of silently rendering an empty list — a previous version swallowed
    a CypherSyntaxError here, which made populated graphs read as "no policies".
    """
    if graph is None:
        return []
    # NOTE: the final aggregation is projected through a WITH and the ORDER BY
    # references the *aliases* (ingestedAt, title), not p.* — newer Neo4j/AuraDB
    # rejects accessing pre-WITH variables after a DISTINCT/aggregating RETURN.
    q = """
    MATCH (p:Policy)
    OPTIONAL MATCH (r:Rule      {policy_id: p.id})
    WITH p, count(DISTINCT r) AS rules
    OPTIONAL MATCH (c:Condition {policy_id: p.id})
    WITH p, rules, count(DISTINCT c) AS conditions
    OPTIONAL MATCH (o:Outcome   {policy_id: p.id})
    WITH p, rules, conditions, count(DISTINCT o) AS outcomes
    RETURN p.id              AS id,
           p.title           AS title,
           p.version         AS version,
           toString(p.ingestedAt) AS ingestedAt,
           rules,
           conditions,
           outcomes
    ORDER BY ingestedAt DESC, title
    """
    return list(graph.query(q))


def delete_policy(graph, policy_id: str) -> dict:
    """
    Detach-delete a Policy node AND every Rule/Condition/Outcome attached
    to it via BELONGS_TO. Returns a stats dict with counts of what was
    removed so the UI can report back.

    Cross-policy CONFLICTS_WITH edges pointing at the deleted conditions
    are also removed (that is what DETACH does) — the conditions on the
    other side of such edges are untouched.
    """
    if graph is None or not policy_id:
        return {"deleted": 0, "error": "no graph / no policy_id"}
    try:
        # Count first so we can report what we removed
        stats = graph.query(
            """
            MATCH (p:Policy {id:$pid})
            OPTIONAL MATCH (n)-[:BELONGS_TO]->(p)
            RETURN count(DISTINCT p) AS policies,
                   count(DISTINCT n) AS members
            """,
            {"pid": policy_id},
        )
        row = stats[0] if stats else {"policies": 0, "members": 0}

        graph.query(
            """
            MATCH (p:Policy {id:$pid})
            OPTIONAL MATCH (n)-[:BELONGS_TO]->(p)
            DETACH DELETE p, n
            """,
            {"pid": policy_id},
        )
        graph.refresh_schema()
        return {
            "policies": int(row.get("policies") or 0),
            "members":  int(row.get("members")  or 0),
            "error":    None,
        }
    except Exception as e:
        return {"policies": 0, "members": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  XML → NEO4J INGESTER  (with live process log + semantic labelling)
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — XML → knowledge graph mapping
#   ingest_xml_to_neo4j() (below) is the single writer to the graph. It wipes the
#   graph, MERGEs the Policy root, (re)creates the vector indices at the live
#   EMBEDDING_DIMS, then MERGEs each Rule/Condition/Outcome — every node scoped by
#   a `policy_id` namespace and embedded for semantic search.
#
# EXTENDING THIS SECTION
#   • New node label: add an `iter("<Tag>")` loop mirroring the Rule loop, MERGE
#     it with its `policy_id`, and attach it with the appropriate edge.
#   • New edge type: add a MERGE after the node loops; if it should be searchable
#     by the Cypher LLM, also declare it in §3's CYPHER_GENERATION_TEMPLATE.
#   • New risk vocabulary: extend RISK_KEYWORDS — is_risk_text() and the red
#     node colouring pick it up automatically.
#   • Idempotency depends on the compound section-anchored id being the MERGE
#     key; never MERGE a member node on a generated/ephemeral id.
# ─────────────────────────────────────────────────────────────────────────────
def _coerce(value: str):
    """Coerce a numeric-looking string attribute to int/float for Cypher typing.

    Args:
        value (str): A raw XML attribute value.

    Returns:
        int | float | str: An ``int`` for whole numbers, ``float`` for decimals,
        else the original trimmed string. Typed properties let Cypher do numeric
        comparisons (e.g. ``c.wordLimit > 40000``) instead of string compares.
    """
    v = value.strip()
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    return v


# Keywords that mark a node as a "risk" → coloured red in the graph
RISK_KEYWORDS = (
    "withdraw", "withdrawal", "withdrawn",
    "sanction", "penalty", "penalis", "penaliz",
    "failure", "fail", "terminated", "terminate",
    "dismissal", "expel", "expelled", "revoke",
    "rejection", "rejected",
)

def is_risk_text(*parts) -> bool:
    """Flag whether any text fragment contains risk/sanction vocabulary.

    Args:
        *parts: Arbitrary text fragments (label, description, text, …); falsy
            fragments are ignored.

    Returns:
        bool: True if any ``RISK_KEYWORDS`` token appears in the concatenated,
        lower-cased text. The result is stored as ``is_risk`` on the node and
        drives its red colouring in the Reasoning View.

    Logical Rationale:
        A cheap substring scan (not an LLM call) keeps risk-flagging deterministic
        and free at ingest scale; matching on stems like "penalis"/"penaliz"
        catches British/American spellings and inflections without a stemmer.
    """
    blob = " ".join(p for p in parts if p).lower()
    return any(kw in blob for kw in RISK_KEYWORDS)


@st.cache_data(show_spinner=False, max_entries=2048)
def generate_human_label(text: str) -> str:
    """Produce a concise 2-3 word human-readable title for a node's text (cached).

    Args:
        text (str): The node's source text (description / condition / outcome).

    Returns:
        str: A 2-4 word title (hard-capped). ``"Untitled"`` for empty input; a
        40-char truncation if ``OPENAI_API_KEY`` is absent or the call fails.

    Logical Rationale:
        Precise IDs like ``C_13_1_LawThesisWordLimit`` are provenance-bearing but
        unreadable on a graph node, so a friendly ``label_human`` is generated
        with gpt-4o-mini (T=0, max_tokens=20) — the cheapest deterministic model
        adequate for a 2-3 word title. ``@st.cache_data(max_entries=2048)`` means
        identical text is never re-billed, and the 4-word hard cap protects the
        visualisation layout from a verbose model response.
    """
    text = (text or "").strip()
    if not text:
        return "Untitled"
    if not OPENAI_API_KEY:
        return text[:40] + ("…" if len(text) > 40 else "")
    try:
        labeler = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY,
            max_tokens=20,
            **_llm_kwargs(),
        )
        prompt = (
            "Given the following policy text, generate a concise, 2-3 word "
            "title that captures its core meaning for a general audience. "
            "Return ONLY the title — no quotes, no punctuation, no prefix.\n\n"
            f"Text: {text}"
        )
        out = labeler.invoke(prompt).content.strip().strip('"').strip("'")
        # Hard cap to protect the UI
        words = out.split()
        return " ".join(words[:4]) if words else text[:40]
    except Exception:
        return text[:40] + ("…" if len(text) > 40 else "")


# ─────────────────────────────────────────────────────────────────────────────
# 5.5  PDF → XML AUTOMATED PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — PDF → strict XML extraction
#   The most failure-prone layer: an unstructured PDF becomes schema-conformant
#   XML the ingester can map deterministically. Pipeline:
#     _extract_pdf_text → _chunk_text → per-chunk gpt-4o (PDF_TO_XML_SYSTEM)
#     → _merge_xml_fragments → _wrap_xml_fragment → ET.fromstring (final gate).
#   Defence-in-depth against placeholder nodes lives at THREE layers: the prompt
#   contract (PDF_TO_XML_SYSTEM), merge-time rejection (_is_generic), and the
#   section-anchored id grammar that makes collisions impossible.
#
# EXTENDING THIS SECTION
#   • New element type in the XML: add it to PDF_TO_XML_SYSTEM's OUTPUT SCHEMA,
#     collect it in _merge_xml_fragments, and map it in ingest_xml_to_neo4j.
#   • Swap the extraction model: change the ChatOpenAI(model=...) line in
#     process_pdf_to_xml; the strict-XML contract is model-agnostic.
#   • Tune chunking via CHUNK_SIZE / CHUNK_OVERLAP in process_pdf_to_xml
#     (justified inline there).
# ─────────────────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


PDF_TO_XML_SYSTEM = """You are a Knowledge Engineering Agent. Your sole task is to convert Bournemouth University policy text into strict hierarchical XML that feeds a Neo4j Knowledge Graph. Every element you emit is merged directly into a live production database — any generic placeholder or malformed tag will corrupt it permanently.

════════════════════════════════════════════
ABSOLUTE NAMING RULES  (no exceptions)
════════════════════════════════════════════
All id attributes MUST follow this compound format:

  Rule      →  R_{SectionNumber}
               e.g. R_13   R_7_2
  Condition →  C_{SectionNumber}_{DescriptiveSlug}
               e.g. C_13_1_LawThesisWordLimit   C_7_2_MaxResubmissions
  Outcome   →  O_{SectionNumber}_{DescriptiveSlug}
               e.g. O_13_1_ExaminationBoardAward   O_7_2_WithdrawalSanction

Rules:
• SectionNumber = the exact section/subsection digits from the text (e.g. 13, 7.2, 4.1.3).
  Replace dots with underscores: section 7.2 → "7_2".
• DescriptiveSlug = PascalCase phrase (2–5 words) derived ONLY from the actual policy text.
• FORBIDDEN id values: "Untitled", "Unknown", "C1", "O1", any single-letter+digit without
  a section prefix, any empty string.
• FORBIDDEN label values: "Untitled", "Unknown", any single word that does not name the
  specific policy clause, any empty string.
• Every label attribute MUST be a readable 2–7 word phrase a policy officer would recognise.
• If you cannot derive a descriptive name from the text DO NOT emit the element at all.

════════════════════════════════════════════
OUTPUT SCHEMA  (follow exactly)
════════════════════════════════════════════
<Rule id="R_{section}" label="{Descriptive Title 2-7 words}" sourceSection="{e.g. 13.1}">
  <Description>{One sentence: what this rule governs.}</Description>

  <Condition id="C_{section}_{Slug}" label="{Descriptive Condition Name}" parentRule="R_{section}"
             deadlineFT="{value or omit attr}" deadlinePT="{value or omit attr}"
             wordLimit="{value or omit attr}">
    {Exact or closely paraphrased condition text from the document.}
  </Condition>

  <Outcome id="O_{section}_{Slug}" label="{Descriptive Outcome Name}" parentRule="R_{section}"
           type="{requirement|sanction|award|exception}">
    {Exact or closely paraphrased outcome text from the document.}
  </Outcome>

  <Conflict from="C_{id}" to="C_{id}"
            scenario="{exact conflicting situation}"
            resolution="{which rule governs and why}"/>
</Rule>

Nesting rules:
• <Condition> and <Outcome> MUST be direct children of their parent <Rule>.
• Sub-sections each become their own <Rule> nested inside the parent <Rule>.
• <Conflict> uses self-closing form; all other elements must be explicitly closed.

════════════════════════════════════════════
RISK KEYWORD DETECTION
════════════════════════════════════════════
If the text contains: withdrawal, failure, terminated, excluded, penalty, sanction,
lapsed, void, rejected, resubmit — the enclosing <Outcome> MUST carry type="sanction".

════════════════════════════════════════════
OUTPUT RULES
════════════════════════════════════════════
1. Output ONLY raw XML — zero markdown, zero ```xml fences, zero <?xml?> prolog.
2. Every opened tag must be explicitly closed. No truncation mid-element.
3. Do NOT invent information absent from the text.
4. Do NOT emit any element you cannot name specifically — omit it entirely.
5. Anchor every id to the section numbers visible in THIS chunk's text.
"""


def _extract_pdf_text(pdf_bytes: bytes, log_fn) -> str:
    """Extract all text from an uploaded PDF using PyMuPDF (fitz).

    Args:
        pdf_bytes (bytes): Raw bytes of the uploaded PDF.
        log_fn (callable): ``log_fn(level, message)`` progress sink.

    Returns:
        str: All page text joined by blank lines, stripped. May be empty for an
        image-only scan with no OCR layer — the CALLER (process_pdf_to_xml)
        treats an empty result as a fatal RuntimeError.

    Raises:
        RuntimeError: If PyMuPDF is not installed.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")

    log_fn("info", "📄 Opening PDF with PyMuPDF...")
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        log_fn("info", f"📑 PDF has {len(doc)} pages — extracting text...")
        for i, page in enumerate(doc, 1):
            pages.append(page.get_text("text"))
    text = "\n\n".join(pages).strip()
    log_fn("success", f"✓ Extracted {len(text):,} characters from {len(pages)} pages.")
    return text


def _wrap_xml_fragment(xml_fragment: str) -> bytes:
    """Sanitise an LLM XML fragment and wrap it in a single <Policy> root.

    Args:
        xml_fragment (str): Raw model output — possibly fenced, prologued, or
            containing unescaped ``&`` and stray control characters.

    Returns:
        bytes: A UTF-8 ``<Policy>…</Policy>`` document ready for ``ET.fromstring``.

    Logical Rationale:
        The model emits sibling ``<Rule>`` elements with no single root and
        occasionally wraps them in ```` ```xml ```` fences or an ``<?xml?>``
        prolog. Well-formed XML requires exactly one root, so we strip those
        artefacts, escape bare ``&`` that are not already valid entities, and
        delete non-XML control chars (which would otherwise make the parser
        reject the whole document). The wrap is the last transform before the
        ``ET.fromstring`` validation gate in process_pdf_to_xml.
    """
    s = xml_fragment.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("xml"):
            s = s[3:]
        s = s.strip("`").strip()
    if s.lower().startswith("<?xml"):
        s = s.split("?>", 1)[-1].strip()

    # Sanitise common LLM XML mistakes:
    # 1. Unescaped & that aren't already part of a valid entity reference.
    s = re.sub(r"&(?!(?:amp|lt|gt|apos|quot|#\d+|#x[\da-fA-F]+);)", "&amp;", s)
    # 2. Strip non-XML control characters (U+0000–U+0008, U+000B–U+000C, U+000E–U+001F)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)

    return f"<Policy>\n{s}\n</Policy>".encode("utf-8")


def _chunk_text(text: str, chunk_size: int = 80_000, overlap: int = 2_000) -> list[str]:
    """Split text into overlapping fixed-size character windows for LLM processing.

    Args:
        text (str): The full extracted document text.
        chunk_size (int): Window size in characters. The signature default
            (80_000) is generous; process_pdf_to_xml deliberately overrides it
            to the smaller, empirically-tuned 15_000 (see its body).
        overlap (int): Characters shared between consecutive windows.

    Returns:
        list[str]: One element (the whole text) if it fits in a single window,
        otherwise the ordered overlapping windows.

    Mathematical/Logical Rationale:
        Consecutive windows advance by ``chunk_size - overlap`` characters, so
        the trailing ``overlap`` chars of window *n* reappear at the head of
        window *n+1*. That overlap guarantees a section header straddling a
        boundary still appears intact in at least one chunk, which is what keeps
        the section-anchored id extraction reliable across boundaries.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _merge_xml_fragments(raw_fragments: list[str], log_fn) -> str:
    """Merge per-chunk XML fragments into one deduplicated <Rule> body.

    Args:
        raw_fragments (list[str]): One raw XML string per processed chunk.
        log_fn (callable): ``log_fn(level, message)`` progress/skip-tally sink.

    Returns:
        str: A single merged XML body (sibling ``<Rule>`` elements) ready for
        ``_wrap_xml_fragment``.

    Mathematical/Logical Rationale:
        Because windows overlap (see _chunk_text), the SAME Rule can legitimately
        appear in two adjacent fragments. Naive concatenation would duplicate
        nodes; dropping the later copy would lose conditions that only appeared
        in the second window. The merger resolves both:
          • Dedup key = ``tag + "::" + id``. Compound section-anchored ids make a
            collision between genuinely different elements effectively impossible,
            so an id match is a true duplicate.
          • On a repeat id, only the NEW unique ``<Condition>``/``<Outcome>``
            children are appended to the first registered copy (set-union of
            children, not replacement).
          • ``_is_generic`` rejects any element whose id/label is in _FORBIDDEN or
            matches the bare-id regex ``^[rco]\\d{1,3}$`` — a second line of
            defence so a placeholder that slipped past the prompt never reaches
            Neo4j. Rejections are counted and surfaced in the UI log.
    """
    import copy

    _FORBIDDEN = {"untitled", "unknown", "unnamed", "n/a", "none", ""}
    _BARE_ID   = re.compile(r"^[rco]\d{1,3}$", re.IGNORECASE)  # R1, C2, O12 …

    def _is_generic(val: str) -> bool:
        """True if an id/label is a forbidden placeholder or a bare ID (R1/C2/O12).

        The merge-time guard that rejects elements the prompt should never have
        emitted — second line of defence behind the PDF_TO_XML_SYSTEM contract.
        """
        v = (val or "").strip().lower()
        return v in _FORBIDDEN or bool(_BARE_ID.match(v))

    def _sanitise(s: str) -> str:
        """Strip code fences / XML prolog, escape bare ``&``, drop control chars.

        Prepares a single raw fragment so ElementTree can parse it; mirrors the
        cleaning in _wrap_xml_fragment but applied per-fragment before merging.
        """
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("xml"):
                s = s[3:]
            s = s.strip("`").strip()
        if s.lower().startswith("<?xml"):
            s = s.split("?>", 1)[-1].strip()
        s = re.sub(r"&(?!(?:amp|lt|gt|apos|quot|#\d+|#x[\da-fA-F]+);)", "&amp;", s)
        s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
        return s

    # rule_id → deepcopy of the winning ET.Element
    rule_registry: dict[str, ET.Element] = {}
    # rule_id → set of "ChildTag::child_id" keys already inside that Rule
    child_seen: dict[str, set[str]] = {}
    skipped = 0
    parse_errors = 0

    for i, frag in enumerate(raw_fragments, 1):
        s = _sanitise(frag.strip())
        try:
            wrapper = s if re.match(r"\s*<Policy[\s>]", s) else f"<root>{s}</root>"
            frag_root = ET.fromstring(wrapper.encode("utf-8"))
        except ET.ParseError as e:
            parse_errors += 1
            log_fn("warn", f"⚠ Fragment {i} failed to parse ({e}) — skipped.")
            continue

        # Walk only direct children (top-level Rules)
        for rule_el in list(frag_root):
            if rule_el.tag != "Rule":
                continue
            rid    = (rule_el.get("id") or "").strip()
            rlabel = (rule_el.get("label") or "").strip()

            if not rid or _is_generic(rid) or _is_generic(rlabel):
                skipped += 1
                continue

            if rid not in rule_registry:
                # First time: deepcopy the entire Rule subtree
                cloned = copy.deepcopy(rule_el)
                rule_registry[rid] = cloned
                child_seen[rid] = set()
                for child in cloned:
                    if child.tag in ("Condition", "Outcome"):
                        cid = (child.get("id") or "").strip()
                        if cid and not _is_generic(cid):
                            child_seen[rid].add(f"{child.tag}::{cid}")
                        else:
                            skipped += 1
            else:
                # Rule already registered: merge any new unique children
                existing = rule_registry[rid]
                for child in rule_el:
                    if child.tag not in ("Condition", "Outcome"):
                        continue
                    cid    = (child.get("id") or "").strip()
                    clabel = (child.get("label") or "").strip()
                    if not cid or _is_generic(cid) or _is_generic(clabel):
                        skipped += 1
                        continue
                    key = f"{child.tag}::{cid}"
                    if key not in child_seen[rid]:
                        child_seen[rid].add(key)
                        existing.append(copy.deepcopy(child))

    if parse_errors:
        log_fn("warn", f"⚠ {parse_errors} fragment(s) could not be parsed and were skipped.")
    if skipped:
        log_fn("warn", f"⚠ Rejected {skipped} generic/Untitled element(s) during merge.")

    log_fn("info",
           f"✓ Merge complete: {len(rule_registry)} unique Rule(s) "
           f"from {len(raw_fragments)} chunk(s).")

    return "\n".join(
        ET.tostring(rule_el, encoding="unicode")
        for rule_el in rule_registry.values()
    )


def process_pdf_to_xml(pdf_file, log_fn) -> bytes:
    """
    End-to-end: uploaded PDF → extracted text → GPT-4o structured XML → bytes
    ready for ingest_xml_to_neo4j. Uses GPT-4o (temperature=0) exclusively.
    """
    pdf_bytes = pdf_file.read() if hasattr(pdf_file, "read") else pdf_file
    text = _extract_pdf_text(pdf_bytes, log_fn)

    if not text:
        raise RuntimeError("No extractable text found in the PDF.")

    # CRITICAL PARAMETERS — CHUNK_SIZE = 15_000, CHUNK_OVERLAP = 1_500
    #   Empirical justification (per ARCHITECTURE_REVIEW.md §2a): GPT-4o emits
    #   *valid, complete* XML far more reliably on bounded inputs than on a whole
    #   document, so the 15k window is deliberately small — well below the model's
    #   context limit — to keep each extraction self-contained and parseable. The
    #   1.5k (=10%) overlap guarantees a section header split across a boundary
    #   still appears intact in at least one chunk, preserving the section-anchored
    #   id grammar. Raising CHUNK_SIZE risks truncated/invalid XML; lowering it
    #   raises cost and the chance of severing a clause from its section number.
    CHUNK_SIZE = 15_000
    CHUNK_OVERLAP = 1_500
    chunks = _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    log_fn("info",
           f"📄 PDF text is {len(text):,} chars — split into {len(chunks)} chunk(s) "
           f"of ~{CHUNK_SIZE:,} chars (overlap {CHUNK_OVERLAP:,}).")

    # GPT-4o structured-extraction LLM (deterministic, OpenAI-only).
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for PDF → XML extraction.")
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=8000,
        api_key=OPENAI_API_KEY,
        **_llm_kwargs(),
    )
    model_name = "gpt-4o"

    raw_fragments: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        chunk_label = f"chunk {i}/{len(chunks)}" if len(chunks) > 1 else "full document"
        log_fn("info", f"🤖 Sending {chunk_label} ({len(chunk):,} chars) to {model_name}...")
        prompt = (
            PDF_TO_XML_SYSTEM
            + f"\n\nCHUNK {i} OF {len(chunks)}: "
            + "Derive all element IDs from section numbers visible in THIS chunk's text.\n\n"
            + "Input Text to Process:\n"
            + chunk
        )
        try:
            resp = llm.invoke(prompt)
        except Exception as llm_err:
            import traceback as _tb
            log_fn("err", f"LLM call failed on {chunk_label}: {type(llm_err).__name__}: {llm_err}")
            log_fn("err", _tb.format_exc()[-600:])
            raise
        fragment = getattr(resp, "content", str(resp)) or ""
        if not fragment.strip():
            log_fn("warn", f"⚠ {chunk_label} returned empty XML — skipping.")
            continue
        log_fn("success", f"✓ {chunk_label}: received {len(fragment):,} chars of XML.")
        raw_fragments.append(fragment)

    if not raw_fragments:
        raise RuntimeError("LLM returned empty XML for all chunks.")

    if len(raw_fragments) == 1:
        xml_fragment = raw_fragments[0]
    else:
        log_fn("info", f"🔗 Merging {len(raw_fragments)} XML fragments (deduplicating by element id)...")
        xml_fragment = _merge_xml_fragments(raw_fragments, log_fn)
        log_fn("success", f"✓ Merged XML: {len(xml_fragment):,} chars total.")

    if not xml_fragment.strip():
        raise RuntimeError("LLM returned an empty XML fragment.")

    log_fn("success",
           f"✓ Received {len(xml_fragment):,} chars of XML from {model_name}.")

    xml_bytes = _wrap_xml_fragment(xml_fragment)
    # Sanity-check: can ET parse it?
    try:
        ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        log_fn("err", f"LLM XML did not parse: {e}")
        # Show the offending line to aid debugging
        lines = xml_bytes.decode("utf-8", errors="replace").splitlines()
        if hasattr(e, "position"):
            bad_line = e.position[0]  # 1-based
            snippet = lines[bad_line - 1] if bad_line <= len(lines) else "(out of range)"
            log_fn("err", f"Offending line {bad_line}: {snippet[:200]}")
        raise
    log_fn("success", "✓ XML parsed cleanly — ready for Neo4j ingestion.")
    return xml_bytes


def _slugify(s: str) -> str:
    """Turn a free-text title/version into a safe, stable id fragment.

    Args:
        s (str): Arbitrary free text (e.g. a policy title or version).

    Returns:
        str: Lower-cased, hyphen-collapsed slug; ``"policy"`` for empty input.

    Logical Rationale:
        Used to build the ``policy_id`` namespace key
        (``f"{_slugify(title)}-{_slugify(version)}"``). Determinism is the whole
        point: the same title/version must always yield the same slug so re-ingest
        MERGEs onto the existing Policy node rather than spawning a duplicate.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "policy"


def ingest_xml_to_neo4j(xml_content: bytes, graph, log_fn,
                         policy_title: str | None = None,
                         policy_version: str | None = None):
    """
    Parse XML and MERGE Rules / Conditions / Outcomes plus relationships,
    all scoped to a (:Policy) root node via a policy_id namespace.

    `log_fn(level, message)` is called for every step so the UI can render
    a real-time process log (level ∈ {info, success, warn, err}).
    Returns a stats dict.

    Policy metadata precedence:
      1. Explicit policy_title / policy_version arguments
      2. Attributes on the XML root element if it is <Policy>
      3. Fallback: "Untitled Policy" / "unspecified"
    """
    log_fn("info", "📄 Parsing XML document...")
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        log_fn("err", f"XML parse error: {e}")
        raise

    log_fn("success", f"XML parsed — root element: <{root.tag}>")

    # ── Resolve policy metadata ─────────────────────────────────────────────
    title   = (policy_title   or root.get("title")   or "Untitled Policy").strip()
    version = (policy_version or root.get("version") or "unspecified").strip()
    pid     = f"{_slugify(title)}-{_slugify(version)}"
    log_fn("info", f"🏛  Policy: “{title}” (version {version}) — id={pid}")

    stats = {"rules": 0, "conditions": 0, "outcomes": 0,
             "has_condition": 0, "has_outcome": 0,
             "conflicts": 0, "errors": 0,
             "policy_id": pid, "policy_title": title}

    def run(cypher, params):
        """Execute one ingest Cypher write, logging + counting failures softly.

        Returns True on success, False on error (incrementing stats['errors']).
        A single failed MERGE must not abort the whole ingest, so errors are
        tallied and surfaced rather than raised.
        """
        try:
            graph.query(cypher, params)
            return True
        except Exception as e:
            log_fn("err", f"Cypher failed: {e}")
            stats["errors"] += 1
            return False

    # ── Wipe all existing graph data before fresh ingestion ─────────────────
    log_fn("info", "🗑  Wiping existing graph data for clean ingestion...")
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        log_fn("success", "✓ All existing nodes and relationships deleted.")
    except Exception as _wipe_err:
        log_fn("warn", f"⚠ Could not wipe graph: {_wipe_err}")

    # ── Policy root node ─────────────────────────────────────────────────────
    run(
        """
        MERGE (p:Policy {id:$pid})
        SET p.title=$title, p.version=$version, p.ingestedAt=datetime()
        """,
        {"pid": pid, "title": title, "version": version},
    )
    log_fn("success", f"✓ Policy node created / updated: {pid}")

    # ── Vector index bootstrap (DROP + CREATE to handle dimension changes) ────
    if _embedder is not None:
        try:
            graph.query("DROP INDEX rule_text_index IF EXISTS")
            graph.query(f"""
            CREATE VECTOR INDEX rule_text_index
            FOR (r:Rule) ON (r.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {EMBEDDING_DIMS},
              `vector.similarity_function`: 'cosine'
            }}}}
            """)
            graph.query("DROP INDEX condition_text_index IF EXISTS")
            graph.query(f"""
            CREATE VECTOR INDEX condition_text_index
            FOR (c:Condition) ON (c.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {EMBEDDING_DIMS},
              `vector.similarity_function`: 'cosine'
            }}}}
            """)
            log_fn("success",
                   f"✓ Vector indices created ({EMBEDDING_DIMS}-dim cosine): "
                   "rule_text_index, condition_text_index.")
        except Exception as e:
            log_fn("warn", f"Vector index creation failed (will skip embedding): {e}")

    # ── Rules, Conditions, Outcomes ──────────────────────────────────────────
    for rule_el in root.iter("Rule"):
        rid    = (rule_el.get("id") or "").strip()
        if not rid:
            log_fn("warn", "Skipping <Rule> with no id")
            continue
        rlabel = (rule_el.get("label") or rid).strip()
        rsec   = (rule_el.get("sourceSection") or "").strip()
        rdesc  = ""
        d = rule_el.find("Description")
        if d is not None and d.text:
            rdesc = d.text.strip()

        # Semantic label via LLM (based on description, fall back to label/id)
        rule_basis = rdesc or rlabel or rid
        rlabel_human = generate_human_label(rule_basis)
        log_fn("info", f"🧠 Labelled Rule {rid} → “{rlabel_human}”")
        log_fn("info", f"📋 Extracting Rule {rid} — “{rlabel}”")
        if run(
            """
            MERGE (r:Rule {id:$id, policy_id:$pid})
            SET r.label=$label, r.sourceSection=$sec, r.description=$desc,
                r.label_human=$label_human, r.is_risk=$is_risk
            WITH r
            MATCH (p:Policy {id:$pid})
            MERGE (r)-[:BELONGS_TO]->(p)
            """,
            {"id": rid, "pid": pid, "label": rlabel, "sec": rsec, "desc": rdesc,
             "label_human": rlabel_human,
             "is_risk": is_risk_text(rdesc, rlabel)},
        ):
            stats["rules"] += 1

        # Embedding for semantic search — uses description + label + human label
        if _embedder is not None:
            embed_text = " · ".join(t for t in (rlabel_human, rlabel, rdesc) if t)
            if embed_text:
                vec = _embed_with_retry(embed_text)
                if vec is not None:
                    run(
                        "MATCH (r:Rule {id:$id, policy_id:$pid}) "
                        "SET r.embedding = $vec",
                        {"id": rid, "pid": pid, "vec": vec},
                    )
                else:
                    log_fn("err", f"⚠ Embedding failed after 3 attempts for Rule {rid} — node stored without vector")

        # Conditions
        for cond_el in rule_el.iter("Condition"):
            cid   = (cond_el.get("id") or "").strip()
            if not cid:
                log_fn("warn", f"Rule {rid}: Condition with no id")
                continue
            ctext = (cond_el.text or "").strip()
            extras = {k: _coerce(v) for k, v in cond_el.attrib.items() if k != "id"}

            clabel_human = generate_human_label(ctext)
            log_fn("info", f"🧠 Labelled Condition {cid} → “{clabel_human}”")
            log_fn("info", f"  ↳ Linking Condition {cid} to Rule {rid}")

            # MERGE condition node scoped to policy_id
            run(
                """
                MERGE (c:Condition {id:$id, policy_id:$pid})
                SET c.text=$text, c += $extras,
                    c.label_human=$label_human, c.is_risk=$is_risk
                WITH c
                MATCH (p:Policy {id:$pid})
                MERGE (c)-[:BELONGS_TO]->(p)
                """,
                {"id": cid, "pid": pid, "text": ctext, "extras": extras,
                 "label_human": clabel_human,
                 "is_risk": is_risk_text(ctext)},
            )

            # Embedding on the condition text for semantic fallback search
            if _embedder is not None:
                embed_text = " · ".join(t for t in (clabel_human, ctext) if t)
                if embed_text:
                    vec = _embed_with_retry(embed_text)
                    if vec is not None:
                        run(
                            "MATCH (c:Condition {id:$id, policy_id:$pid}) "
                            "SET c.embedding = $vec",
                            {"id": cid, "pid": pid, "vec": vec},
                        )
                    else:
                        log_fn("err", f"⚠ Embedding failed after 3 attempts for Condition {cid} — node stored without vector")

            if run(
                """
                MATCH (r:Rule {id:$rid, policy_id:$pid})
                MATCH (c:Condition {id:$cid, policy_id:$pid})
                MERGE (r)-[:HAS_CONDITION]->(c)
                """,
                {"rid": rid, "cid": cid, "pid": pid},
            ):
                stats["conditions"] += 1
                stats["has_condition"] += 1

        # Outcomes
        for out_el in rule_el.iter("Outcome"):
            oid   = (out_el.get("id") or "").strip()
            if not oid:
                continue
            otype = (out_el.get("type") or "").strip()
            otext = (out_el.text or "").strip()

            olabel_human = generate_human_label(otext or otype)
            log_fn("info", f"🧠 Labelled Outcome {oid} → “{olabel_human}”")
            log_fn("info", f"  ↳ Linking Outcome {oid} ({otype or 'untyped'}) to Rule {rid}")
            run(
                """
                MERGE (o:Outcome {id:$id, policy_id:$pid})
                SET o.type=$type, o.text=$text,
                    o.label_human=$label_human, o.is_risk=$is_risk
                WITH o
                MATCH (p:Policy {id:$pid})
                MERGE (o)-[:BELONGS_TO]->(p)
                """,
                {"id": oid, "pid": pid, "type": otype, "text": otext,
                 "label_human": olabel_human,
                 "is_risk": is_risk_text(otext, otype)},
            )
            if run(
                """
                MATCH (r:Rule {id:$rid, policy_id:$pid})
                MATCH (o:Outcome {id:$oid, policy_id:$pid})
                MERGE (r)-[:HAS_OUTCOME]->(o)
                """,
                {"rid": rid, "oid": oid, "pid": pid},
            ):
                stats["outcomes"] += 1
                stats["has_outcome"] += 1

    # ── Explicit <Conflict> elements ────────────────────────────────────────
    # Shape:  <Conflict from="C5.1.2" to="C5.1.6" scenario="..." resolution="..."
    #                   crossPolicy="true" toPolicy="other-policy-id"/>
    # `crossPolicy="true"` opts into linking Conditions across different
    # policies; otherwise conflicts are strictly intra-policy.
    for cf_el in root.iter("Conflict"):
        src = (cf_el.get("from") or cf_el.get("source") or "").strip()
        dst = (cf_el.get("to") or cf_el.get("target") or "").strip()
        if not (src and dst):
            continue
        scenario   = (cf_el.get("scenario") or (cf_el.text or "")).strip()
        resolution = (cf_el.get("resolution") or "").strip()
        cross      = (cf_el.get("crossPolicy") or "").strip().lower() == "true"
        dst_pid    = (cf_el.get("toPolicy") or "").strip() or pid

        if cross and dst_pid != pid:
            log_fn("warn",
                   f"⚠ Cross-policy conflict: {src}@{pid} ↔ {dst}@{dst_pid}")
            match_cypher = (
                "MATCH (a:Condition {id:$src, policy_id:$pid}) "
                "MATCH (b:Condition {id:$dst, policy_id:$dst_pid}) "
                "MERGE (a)-[cf:CONFLICTS_WITH]->(b) "
                "SET cf.scenario=$scenario, cf.resolution=$resolution, "
                "    cf.sections=$src+'↔'+$dst, cf.cross_policy=true"
            )
            ok = run(match_cypher,
                     {"src": src, "dst": dst, "pid": pid, "dst_pid": dst_pid,
                      "scenario": scenario, "resolution": resolution})
        else:
            log_fn("warn", f"⚠ Detected policy conflict: {src} ↔ {dst} (in {pid})")
            ok = run(
                """
                MATCH (a:Condition {id:$src, policy_id:$pid})
                MATCH (b:Condition {id:$dst, policy_id:$pid})
                MERGE (a)-[cf:CONFLICTS_WITH]->(b)
                SET cf.scenario=$scenario, cf.resolution=$resolution,
                    cf.sections=$src+'↔'+$dst, cf.cross_policy=false
                """,
                {"src": src, "dst": dst, "pid": pid,
                 "scenario": scenario, "resolution": resolution},
            )
        if ok:
            stats["conflicts"] += 1

    graph.refresh_schema()
    log_fn("success",
           f"✓ Ingestion complete — "
           f"{stats['rules']} Rules, {stats['conditions']} Conditions, "
           f"{stats['outcomes']} Outcomes, {stats['conflicts']} Conflicts "
           f"(policy: {title}).")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GRAPH VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — streamlit-agraph rendering layer
#   These helpers convert Neo4j rows into vis.js Node/Edge objects with
#   type-aware colour, size, hierarchy level, hover tooltip, and (in the
#   Reasoning View) a cyan "the AI looked here" halo plus score-driven opacity.
#   fetch_full_graph() renders the whole corpus; fetch_reasoning_subgraph()
#   renders only the nodes a specific answer grounded on.
#
# EXTENDING THIS SECTION
#   • New node label: add it to LABEL_COLORS (colour), LABEL_LEVEL (tree depth),
#     and LABEL_SIZE (radius). _add_node() reads all three by primary label, so
#     no further wiring is needed for display.
#   • New edge styling: edges are built in the fetch_* functions; mirror the
#     existing kept/dim treatment there.
#   • IMPORTANT API note: streamlit-agraph's Edge exposes its destination as
#     ``.to`` (NOT ``.target``) — use ``.source`` / ``.to`` when filtering edges.
# ─────────────────────────────────────────────────────────────────────────────
COLOR_POLICY = "#a855f7"     # violet — top-level document anchor

LABEL_COLORS = {
    "Policy":    COLOR_POLICY,
    "Rule":      COLOR_RULE,
    "Condition": COLOR_CONDITION,
    "Outcome":   COLOR_OUTCOME,
    "Actor":     COLOR_ACTOR,
}

# Hierarchical level (top-down): smaller = higher in the tree
LABEL_LEVEL = {
    "Policy":    0,
    "Rule":      1,
    "Actor":     1,
    "Condition": 2,
    "Outcome":   2,
}

# Smart sizing — Policy biggest, Conditions/Outcomes smallest
LABEL_SIZE = {
    "Policy":    44,
    "Rule":      28,
    "Actor":     22,
    "Outcome":   22,
    "Condition": 18,
}

# Path-highlight palette (used in the Reasoning View)
COLOR_PATH      = "#22d3ee"  # neon cyan — edges between two cited nodes
COLOR_PATH_DIM  = "rgba(96,165,250,0.22)"  # faded blue — context edges

# Real-time search highlighting
COLOR_SEARCH_HIT = "#fde047"               # amber yellow — match halo
COLOR_SEARCH_DIM = "rgba(100,116,139,0.30)"  # faded slate — non-matches


def _color_for(node_labels, props, in_conflict=False):
    """Choose a node's fill colour from its label, with risk/conflict override.

    Args:
        node_labels (list[str]): The node's Neo4j labels.
        props (dict): The node's properties (scanned for risk vocabulary).
        in_conflict (bool): True if the node sits on a CONFLICTS_WITH edge.

    Returns:
        str: A hex colour. Risk/conflict nodes always return ``COLOR_CONFLICT``
        (red) regardless of label, so danger is never visually masked by the
        ordinary label palette; otherwise the per-label colour, or a slate default.
    """
    if in_conflict or props.get("is_risk") or is_risk_text(
        props.get("text"), props.get("description"),
        props.get("label"), props.get("label_human"), props.get("type"),
    ):
        return COLOR_CONFLICT
    for lbl in node_labels:
        if lbl in LABEL_COLORS:
            return LABEL_COLORS[lbl]
    return COLOR_DEFAULT


def _truncate(s: str, n: int = 28) -> str:
    """Trim a string to ``n`` characters with a trailing ellipsis.

    Args:
        s (str): Input text (None tolerated).
        n (int): Maximum length before truncation. Default 28.

    Returns:
        str: ``s`` unchanged if short enough, else its first ``n-1`` chars
        (trailing punctuation stripped) plus "…". Keeps node captions readable.
    """
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: max(1, n - 1)].rstrip(" ,;:.-") + "…"


def _primary_label(node_labels) -> str:
    """Return the node's primary display label (first known palette label).

    Args:
        node_labels (list[str]): The node's Neo4j labels.

    Returns:
        str: The first label present in ``LABEL_COLORS`` (Policy/Rule/Condition/
        Outcome/Actor), else the first label, else "". This is what selects the
        node's colour, size, and hierarchy level downstream.
    """
    for lbl in (node_labels or []):
        if lbl in LABEL_COLORS:
            return lbl
    return (node_labels[0] if node_labels else "")


def _display_label(node_labels, props) -> str:
    """
    Compact, semantic caption shown INSIDE the node:
      "Examiner Eligibility\n(Condition)"
    Truncates to keep the graph readable; full text lives in the tooltip.
    """
    human = (props.get("label_human")
             or props.get("title")           # for :Policy nodes
             or props.get("label")
             or props.get("type")
             or props.get("role")
             or props.get("id")
             or "?")
    primary = _primary_label(node_labels)
    short = _truncate(str(human), 28)
    return f"{short}\n({primary})" if primary else short


def _build_tooltip(node_labels, props) -> str:
    """
    Plain-text tooltip rendered on hover. vis.js shows newlines as <br/>
    and escapes the rest, so this is safe and renders cleanly.
    """
    primary = _primary_label(node_labels)
    lines = []

    head = (props.get("label_human")
            or props.get("title") or props.get("label")
            or props.get("type") or props.get("id") or "?")
    lines.append(f"{head}  ({primary})" if primary else str(head))

    # Policy-context attribution
    if "Policy" in (node_labels or []):
        if props.get("version"):
            lines.append(f"📘 Version: {props.get('version')}")
        if props.get("ingestedAt"):
            lines.append(f"🕒 Ingested: {props.get('ingestedAt')}")
    else:
        if props.get("policy_id"):
            lines.append(f"📘 Policy: {props.get('policy_id')}")
        if props.get("sourceSection"):
            lines.append(f"§ Section: {props.get('sourceSection')}")
        if props.get("type"):
            lines.append(f"🏷 Type: {props.get('type')}")

    body = props.get("description") or props.get("text") or ""
    if body:
        lines.append("")
        lines.append(_truncate(str(body), 320))

    if props.get("is_risk"):
        lines.append("")
        lines.append("⚠ Flagged: contains risk keywords (withdrawal / sanction / failure)")

    return "\n".join(lines)


def _opacity_for_score(score):
    """
    Score → opacity mapping:
        score >= 0.80   → 1.00  (fully opaque, top relevance)
        0.70 ≤ s < 0.80 → 0.70  (lightly faded — visible but visually quieter)
        0.60 ≤ s < 0.70 → 0.55  (only seen if user lowers threshold)
        score is None   → 1.00  (LLM citation; no opacity penalty)
    """
    if score is None:
        return 1.0
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 1.0
    if s >= 0.80:
        return 1.0
    if s >= 0.70:
        return 0.70
    return 0.55


def _section_chapter(props) -> int | None:
    """
    Extract the integer chapter prefix from sourceSection — '7B.2.3' → 7,
    '5.1.2' → 5. Used to keep nodes from the same policy chapter horizontally
    adjacent in the hierarchical layout.
    """
    sec = props.get("sourceSection") or ""
    m = re.match(r"^\s*(\d+)", str(sec))
    return int(m.group(1)) if m else None


def _add_node(nodes, seen, nid, n_labels, props,
              size=None, level=None, in_conflict=False, cited=False,
              score=None):
    """Append a styled vis.js Node to ``nodes`` (idempotent via the ``seen`` set).

    Args:
        nodes (list): Accumulator the new Node is appended to.
        seen (set): IDs already added; a repeat ``nid`` is silently skipped.
        nid (str): The node id (also its vis.js id).
        n_labels (list[str]): Neo4j labels — drive colour/size/level defaults.
        props (dict): Node properties — drive caption, tooltip, risk colour.
        size (int | None): Explicit radius; falls back to ``LABEL_SIZE``.
        level (int | None): Explicit tree depth; falls back to ``LABEL_LEVEL``.
        in_conflict (bool): Force the conflict (red) colour.
        cited (bool): If True, add the cyan halo/glow marking an answer anchor.
        score (float | None): Cosine similarity → opacity (see _opacity_for_score).

    Returns:
        None. Mutates ``nodes`` and ``seen`` in place.

    Logical Rationale:
        Deduplication by ``seen`` is essential because a 1-hop subgraph query
        returns the same node once per incident edge; without the guard, vis.js
        would receive duplicate ids and mis-render. ``cited`` + ``score`` encode
        the system's transparency goal visually — the eye is drawn to the
        highest-relevance nodes the AI actually grounded on.
    """
    if not nid or nid in seen:
        return
    seen.add(nid)

    # Pick smart defaults from the node's primary label
    primary = _primary_label(n_labels)
    if size is None:
        size = LABEL_SIZE.get(primary, 16)
    if level is None:
        level = LABEL_LEVEL.get(primary, 1)

    base_color = _color_for(n_labels, props, in_conflict=in_conflict)

    if cited:
        # Glowing cyan halo for "AI looked here" focus nodes
        node_color = {
            "background": base_color,
            "border":     COLOR_PATH,
            "highlight":  {"background": base_color, "border": "#67e8f9"},
        }
        border_width = 5
        node_shadow  = {
            "enabled": True, "color": COLOR_PATH, "size": 26, "x": 0, "y": 0,
        }
    else:
        node_color   = base_color
        border_width = 2
        node_shadow  = True

    # Group key for hierarchical clustering (chapter-aware)
    chapter = _section_chapter(props)
    group   = f"{primary}-ch{chapter}" if chapter is not None else primary

    nodes.append(Node(
        id=nid,
        label=_display_label(n_labels, props),
        size=size,
        color=node_color,
        title=_build_tooltip(n_labels, props),
        level=level,                       # honoured in hierarchical layout
        shape="dot" if primary != "Policy" else "diamond",
        borderWidth=border_width,
        shadow=node_shadow,
        opacity=_opacity_for_score(score),
        group=group,
    ))


def _node_haystack(n) -> str:
    """Build a lower-cased searchable blob from a vis.js Node's text fields.

    Args:
        n (Node): A streamlit-agraph Node.

    Returns:
        str: ``id + label + title`` concatenated and lower-cased, for
        case-insensitive substring matching in ``decorate_with_search``.
    """
    parts = [
        getattr(n, "id", ""),
        getattr(n, "label", ""),
        getattr(n, "title", ""),
    ]
    return " ".join(str(p) for p in parts if p).lower()


def decorate_with_search(nodes, query: str):
    """Highlight nodes matching a live search query; dim the rest (in place).

    Args:
        nodes (list[Node]): The rendered nodes (mutated in place).
        query (str): The user's search text; blank disables highlighting.

    Returns:
        tuple[int, int]: ``(matches, total)`` for the caption. An empty query
        returns ``(0, total)`` and leaves styling untouched.

    Logical Rationale:
        Matching keeps each hit's original *semantic* colour and adds an amber
        halo, so the user still sees what TYPE of node matched while non-matches
        fade to slate — focus-plus-context rather than hide-the-rest.
    """
    q = (query or "").strip().lower()
    if not q:
        return 0, len(nodes)

    hits = 0
    for n in nodes:
        if q in _node_haystack(n):
            hits += 1
            # Amber halo — keep the original semantic colour so the user
            # still sees what TYPE of node matched.
            base = getattr(n, "color", "#94a3b8")
            if isinstance(base, dict):
                base = base.get("background", "#94a3b8")
            n.color = {
                "background": base,
                "border":     COLOR_SEARCH_HIT,
                "highlight":  {"background": base, "border": "#fef9c3"},
            }
            n.borderWidth = 6
            n.shadow = {
                "enabled": True, "color": COLOR_SEARCH_HIT,
                "size": 28, "x": 0, "y": 0,
            }
        else:
            n.color = COLOR_SEARCH_DIM
            n.borderWidth = 1
            n.shadow = False
    return hits, len(nodes)


def fetch_full_graph(graph, limit=500):
    """Build (nodes, edges) for the whole graph, for the full-corpus view.

    Args:
        graph (Neo4jGraph | None): The connected graph.
        limit (int): Max rows (node+edge tuples) to pull. Default 500.

    Returns:
        tuple[list[Node], list[Edge]]: vis.js objects; ``([], [])`` on no graph
        or query error.

    Logical Rationale:
        Reads real ``labels(n)``/``properties(n)`` rather than guessing types
        from id prefixes, so the renderer is agnostic to the ingested XML's
        shape. CONFLICTS edges are coloured red to surface contradictions at a
        glance.
    """
    if graph is None:
        return [], []
    q = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN labels(n) AS n_labels, properties(n) AS n_props,
           type(r)   AS r_type,   properties(r) AS r_props,
           labels(m) AS m_labels, properties(m) AS m_props
    LIMIT $limit
    """
    try:
        rows = graph.query(q, {"limit": limit})
    except Exception:
        return [], []

    nodes, edges, seen = [], [], set()
    for row in rows:
        n_props = row.get("n_props") or {}
        m_props = row.get("m_props") or {}
        n_labels = row.get("n_labels") or []
        m_labels = row.get("m_labels") or []

        nid = n_props.get("id")
        mid = m_props.get("id")

        # Type-aware sizing/level chosen from LABEL_SIZE / LABEL_LEVEL
        _add_node(nodes, seen, nid, n_labels, n_props)
        if mid:
            _add_node(nodes, seen, mid, m_labels, m_props)

        r_type = row.get("r_type")
        if r_type and nid and mid:
            is_conflict = "CONFLICT" in r_type
            edges.append(Edge(
                source=nid, target=mid, label=r_type,
                color=COLOR_CONFLICT if is_conflict else "#64748b",
            ))
    return nodes, edges


def fetch_reasoning_subgraph(graph, node_ids, policy_ids=None,
                              top_k: int = 5, score_map=None):
    """Build the 1-hop "the AI looked here" subgraph around the cited node IDs.

    Args:
        graph (Neo4jGraph | None): The connected graph.
        node_ids (Iterable[str]): IDs the answer actually cited.
        policy_ids (Iterable[str] | None): If given, restrict centres and
            neighbours to these policies (plus cross-policy conflict edges).
        top_k (int): Max centre nodes to keep after ranking. Default 5.
        score_map (dict | None): ``id → cosine`` used to rank and to set opacity.

    Returns:
        tuple[list[Node], list[Edge]]: The trimmed reasoning subgraph; cited
        centres are enlarged and haloed, neighbours are dimmed context, and
        on-path edges are drawn as a neon-cyan dashed "flow".

    Mathematical/Logical Rationale:
        Centres are ranked by ``-score`` (missing scores default to a neutral
        0.5) with the id as a deterministic tie-breaker, then capped at
        ``top_k``. This bounds visual complexity so the view stays legible even
        when an answer cites many nodes, while the highest-relevance nodes are
        guaranteed to survive the trim. The :Policy root is always admitted so
        the hierarchical layout has a single top.
    """
    if not node_ids or graph is None:
        return [], []

    # ── Rank cited IDs by semantic score (None defaults to a neutral 0.5)
    score_map = score_map or {}
    ranked_ids = sorted(
        node_ids,
        key=lambda x: (-float(score_map.get(x, 0.5)), str(x)),
    )
    top_ids = ranked_ids[: max(1, int(top_k))]
    centres = set(top_ids)

    pids = list(policy_ids) if policy_ids else []
    params = {"ids": list(centres), "pids": pids}

    if pids:
        q = """
        MATCH (n)
        WHERE n.id IN $ids AND n.policy_id IN $pids
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE m:Policy
           OR m.policy_id IN $pids
           OR (type(r) = 'CONFLICTS_WITH' AND coalesce(r.cross_policy,false) = true)
        RETURN labels(n) AS n_labels, properties(n) AS n_props,
               type(r)   AS r_type,   properties(r) AS r_props,
               labels(m) AS m_labels, properties(m) AS m_props
        """
    else:
        q = """
        MATCH (n) WHERE n.id IN $ids
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN labels(n) AS n_labels, properties(n) AS n_props,
               type(r)   AS r_type,   properties(r) AS r_props,
               labels(m) AS m_labels, properties(m) AS m_props
        """

    try:
        rows = graph.query(q, params)
    except Exception:
        return [], []

    nodes, edges, seen = [], [], set()
    cited_set = centres   # only the trimmed top-K count as "cited centres"
    for row in rows:
        n_props = row.get("n_props") or {}
        m_props = row.get("m_props") or {}
        n_labels = row.get("n_labels") or []
        m_labels = row.get("m_labels") or []
        nid, mid = n_props.get("id"), m_props.get("id")

        # Boost cited (centre) nodes so they read as the focus of attention.
        n_size = (LABEL_SIZE.get(_primary_label(n_labels), 18) + 8) if nid in cited_set else None
        _add_node(nodes, seen, nid, n_labels, n_props,
                  size=n_size, cited=(nid in cited_set),
                  score=score_map.get(nid))
        if mid:
            m_size = (LABEL_SIZE.get(_primary_label(m_labels), 18) + 8) if mid in cited_set else None
            _add_node(nodes, seen, mid, m_labels, m_props,
                      size=m_size, cited=(mid in cited_set),
                      score=score_map.get(mid))

        r_type = row.get("r_type")
        if r_type and nid and mid:
            is_conflict = "CONFLICT" in r_type
            on_path    = (nid in cited_set) and (mid in cited_set)
            if is_conflict:
                color, width = COLOR_CONFLICT, 2.4
                dashes_for_edge = False
            elif on_path:
                color, width = COLOR_PATH, 3.0          # neon cyan, thick
                dashes_for_edge = [6, 4]                # dashed = "flow"
            else:
                color, width = COLOR_PATH_DIM, 1.2      # dimmed context
                dashes_for_edge = False
            edges.append(Edge(
                source=nid, target=mid, label=r_type,
                color=color, width=width,
                dashes=dashes_for_edge,
            ))
    return nodes, edges


_NODE_STYLE_DARK = {
    "labelProperty": "label",
    "renderLabel": True,
    "font": {
        "size": 14,
        "color": "#e2e8f0",
        "face": "Inter, sans-serif",
        "align": "top",
        "strokeWidth": 3,
        "strokeColor": "#0f1923",
        "vadjust": -14,
    },
    "borderWidth": 2,
    "shadow": True,
}
_LINK_STYLE = {
    "renderLabel": True,
    "labelProperty": "label",
    "fontColor": "rgba(148,163,184,0.55)",
    "font": {
        "size": 10,
        "color": "rgba(148,163,184,0.55)",
        "face": "Inter, sans-serif",
        "align": "middle",
        "strokeWidth": 0,
    },
    "smooth": {"type": "continuous"},
    "width": 1.2,
}


def _organic_config(width=1100, height=700) -> Config:
    """Build the force-directed (Barnes-Hut) vis.js Config.

    Args:
        width (int): Canvas width in px.
        height (int): Canvas height in px.

    Returns:
        Config: Physics-enabled layout best for small/medium subgraphs where an
        organic spread reveals clusters; the hierarchical config is preferred for
        showing the strict Policy→Rule→Condition tree.
    """
    return Config(
        width=width, height=height,
        directed=True, physics=True, hierarchical=False, improvedLayout=True,
        nodeHighlightBehavior=True, highlightColor="#3b82f6", collapsible=False,
        node=_NODE_STYLE_DARK, link=_LINK_STYLE,
        physics_config={
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -12000,
                "centralGravity": 0.45,
                "springLength": 220,
                "springConstant": 0.04,
                "damping": 0.25,
                "avoidOverlap": 1.0,
            },
            "stabilization": {"enabled": True, "iterations": 180, "fit": True},
        },
        nodeSpacing=220, centralGravity=0.45, springLength=220,
    )


def _hierarchical_config(width=1100, height=700) -> Config:
    """
    Compact tree layout: Policy on top → Rules → Conditions/Outcomes.
    Tighter spacing + edge minimisation to avoid long crossing lines, with
    chapter-aware grouping via each Node's `group` attribute.
    """
    return Config(
        width=width, height=height,
        directed=True, physics=False, hierarchical=True, improvedLayout=True,
        nodeHighlightBehavior=True, highlightColor="#3b82f6", collapsible=False,
        node=_NODE_STYLE_DARK, link=_LINK_STYLE,
        # vis.js hierarchical layout — compacted
        layout={
            "hierarchical": {
                "enabled": True,
                "direction": "UD",          # top-down
                "sortMethod": "directed",
                "shakeTowards": "roots",
                "levelSeparation": 150,     # was 200 — tighter vertical
                "nodeSpacing":     100,     # was 200 — tighter horizontal
                "treeSpacing":     180,     # was 240 — fewer gaps between subtrees
                "blockShifting":   True,    # collapse empty horizontal slots
                "edgeMinimization":True,    # straighter edges, less overlap
                "parentCentralization": True,
            }
        },
        nodeSpacing=100, levelSeparation=150,
    )


def make_graph_config(layout_mode: str = "organic", height: int = 700) -> Config:
    """Return the vis.js Config for the requested layout mode.

    Args:
        layout_mode (str): "organic" (force-directed) or any string starting
            "hier" for the top-down hierarchical tree.
        height (int): Canvas height in px.

    Returns:
        Config: A configured streamlit-agraph Config. Organic suits exploratory
        subgraphs; hierarchical suits showing the Policy→Rule→Condition tree.
    """
    if (layout_mode or "").lower().startswith("hier"):
        return _hierarchical_config(height=height)
    return _organic_config(height=height)


# Back-compat alias for any existing references
AGRAPH_CONFIG = _organic_config()


def render_legend(highlight_path: bool = False):
    """Render the colour-key legend pills above a graph view (via st.markdown).

    Args:
        highlight_path (bool): If True, append the "AI reasoning path" pill —
            shown only in the Reasoning View where the cyan path is meaningful.

    Returns:
        None. Writes HTML pills directly to the Streamlit page.
    """
    items = [
        (COLOR_POLICY,    "Policy"),
        (COLOR_RULE,      "Rule"),
        (COLOR_CONDITION, "Condition"),
        (COLOR_OUTCOME,   "Outcome"),
        (COLOR_CONFLICT,  "Conflict / Risk"),
    ]
    if highlight_path:
        items.append((COLOR_PATH, "AI reasoning path"))

    pill_parts = []
    for c, lbl in items:
        glow = f"box-shadow:0 0 5px {c};" if c == COLOR_PATH else ""
        pill_parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin-right:6px;font-size:0.82rem;">'
            f'<span style="width:10px;height:10px;border-radius:50%;'
            f'background:{c};display:inline-block;{glow}"></span>'
            f'{lbl}</span>'
        )
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 10px 0;">'
        + "".join(pill_parts)
        + "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  QUERY EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — the query orchestrator (Modular RAG)
#   run_query() (below) is the spine of the read path: clean → route (Adaptive
#   RAG) → structured Cypher first → conditional Fusion/keyword fallback → CRAG
#   gate → grounded answer → cite-only Reasoning subgraph. The retrieval engine
#   (_semantic_search) and the imported Fusion/CRAG helpers do the heavy lifting;
#   this section also holds the citation-extraction regexes that decide which
#   nodes the answer actually grounded on.
#
# EXTENDING THIS SECTION
#   • New route: branch on it in run_query and set the right `use_fusion` value.
#   • New retrieval stream: add it inside _semantic_search and append its ranked
#     list to the RRF call — run_query needs no change.
#   • Changing how citations map to the Reasoning View: edit the _ID_PATTERN /
#     _SECTION_PATTERN regexes and resolve_section_refs_to_ids together.
# ─────────────────────────────────────────────────────────────────────────────
# Matches node IDs like R4.1, C5.1.2, C5.1.6, O4.1.A, A1 — works whether they
# appear bare, inside parentheses "(R4.1)", comma-separated, or glued to
# punctuation in the LLM's answer.
_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])([RCOA]\d+(?:\.\d+)*[A-Z]?)(?![A-Za-z0-9])")

# Matches policy section references like "§5.1.2", "section 5.1.2", "§ 10.4"
# or just the bare dotted number "5.1.2" when preceded by a non-alphanumeric.
_SECTION_PATTERN = re.compile(
    r"(?:§\s*|section\s+|CoP\s*§\s*|sec\.\s*|part\s+)?(\d+(?:\.\d+){1,})",
    re.IGNORECASE,
)


def extract_ids_from_text(text: str):
    """Pull every policy-style node ID mentioned anywhere in a block of text.

    Args:
        text (str): Free text (typically the LLM's answer).

    Returns:
        set[str]: All ``_ID_PATTERN`` matches (R4.1, C5.1.2, O4.1.A, A1, …).
        Used to render ONLY the nodes the answer actually cited in the Reasoning
        View — the basis of "show the grounding for this answer".
    """
    if not text:
        return set()
    return set(_ID_PATTERN.findall(text))


def extract_sections_from_text(text: str):
    """Pull dotted section references (e.g. "§5.1.2", "section 10.4") from text.

    Args:
        text (str): Free text (typically the LLM's answer).

    Returns:
        set[str]: Dotted section numbers that appeared next to an indicator
        token (§, "section", "CoP §", "sec.", "part").

    Logical Rationale:
        The indicator requirement is deliberate: a bare regex for dotted numbers
        would also capture version strings, dates, and word-count ranges. Gating
        on an explicit section marker keeps precision high so the Reasoning View
        only resolves genuine policy-section citations.
    """
    if not text:
        return set()
    sections = set()
    # Explicit markers
    for m in re.finditer(
        r"(?:§|section|CoP\s*§|sec\.|part)\s*(\d+(?:\.\d+){1,})",
        text, re.IGNORECASE,
    ):
        sections.add(m.group(1))
    return sections


def resolve_section_refs_to_ids(graph, sections, policy_ids=None):
    """Resolve cited section numbers to the node IDs that carry them.

    Args:
        graph (Neo4jGraph | None): The connected graph.
        sections (Iterable[str]): Section numbers (from extract_sections_from_text).
        policy_ids (Iterable[str] | None): If given, restrict matches to these
            policies.

    Returns:
        set[str]: Node IDs whose ``sourceSection`` is in ``sections``; empty on
        no input or query error.

    Logical Rationale:
        Policy-scoping is a correctness guard, not an optimisation: in a
        multi-policy graph a "§5.1.2" citation from document A must not pull in
        an unrelated "§5.1.2" from document B. Constraining on ``policy_id``
        enforces the same disjoint-domain isolation the namespace provides
        elsewhere in the pipeline.
    """
    if not sections or graph is None:
        return set()
    try:
        pids = list(policy_ids) if policy_ids else []
        if pids:
            rows = graph.query(
                "MATCH (n) "
                "WHERE n.sourceSection IN $secs AND n.policy_id IN $pids "
                "RETURN n.id AS id",
                {"secs": list(sections), "pids": pids},
            )
        else:
            rows = graph.query(
                "MATCH (n) WHERE n.sourceSection IN $secs RETURN n.id AS id",
                {"secs": list(sections)},
            )
        return {r["id"] for r in rows if r.get("id")}
    except Exception:
        return set()


def extract_node_ids(raw_result):
    """
    Walk the raw graph result and collect any value found under a key
    whose name contains 'id' (schema-agnostic — works with any XML shape).
    Also picks up IDs that appear inside text fields via the regex pattern.
    """
    ids = set()
    def walk(o):
        """Recurse through dicts/lists/strings, collecting node IDs into ``ids``."""
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str):
                    if "id" in k.lower() and v.strip():
                        ids.add(v.strip())
                    else:
                        ids.update(_ID_PATTERN.findall(v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)
        elif isinstance(o, str):
            ids.update(_ID_PATTERN.findall(o))
    walk(raw_result)
    return ids


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "do", "does", "did", "have", "has", "had", "of", "in", "on", "at",
    "to", "for", "with", "and", "or", "but", "if", "then", "as", "by",
    "what", "when", "where", "who", "how", "why", "which", "that", "this",
    "my", "me", "i", "we", "you", "your", "our", "us", "it", "its",
    "not", "no", "yes", "any", "all", "some", "about",
    # Pipeline/framework terms — must never become CONTAINS keywords
    "crag", "gate", "route", "routing", "relevance", "relevant", "irrelevant",
    "ambiguous", "fusion", "adaptive", "pipeline", "fallback", "threshold",
    "cypher", "neo4j", "graph", "node", "embedding", "vector", "score",
    "quality", "audit", "trail", "retrieved", "retrieval", "context",
    "reasoning", "lookup", "direct", "complex",
}

# Patterns that indicate UI/framework text has leaked into the raw input string.
_UI_LEAK_RE = re.compile(
    r"(crag\s+quality|quality\s+gate|adaptive\s+route|fusion\s+rag|"
    r"audit\s+trail|policy\s+context\s+(irrelevant|ambiguous)|"
    r"providing\s+standard\s+fallback|knowledge\s+graph\s+platform)",
    re.IGNORECASE,
)


def _clean_user_query(raw: str) -> str:
    """
    Return ONLY the user-typed question, with no UI/framework text attached.

    Strips leading/trailing whitespace.  For multi-line input, drops any line
    that matches known pipeline-output patterns (CRAG status, route badges,
    audit-trail labels).  Falls back to the original stripped string if all
    lines would be removed, so the function never returns empty when the raw
    input is non-empty.
    """
    q = (raw or "").strip()
    if not q:
        return q
    lines = q.splitlines()
    if len(lines) == 1:
        # Single-line: strip but don't drop — trust the user typed it
        return q
    clean = [ln for ln in lines if not _UI_LEAK_RE.search(ln)]
    result = " ".join(clean).strip()
    return result if result else q


def _keywords(question: str, top_k=5):
    """Extract a handful of content keywords from a question.

    Args:
        question (str): The user question.
        top_k (int): Max keywords to return. Default 5.

    Returns:
        list[str]: Up to ``top_k`` distinct 3+-char alphabetic tokens, stopwords
        removed, in first-seen order. A lightweight signal for keyword retrieval
        — deliberately not stemmed, since policy vocabulary is matched by CONTAINS.
    """
    tokens = re.findall(r"[A-Za-z]{3,}", question.lower())
    seen, out = set(), []
    for t in tokens:
        if t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= top_k:
            break
    return out


@st.cache_data(show_spinner=False, max_entries=512)
def generate_search_variants(question: str, n: int = 4) -> list[str]:
    """Generate n alternative phrasings of a question via the gpt-4o-mini narrator.

    Args:
        question (str): The user question.
        n (int): Number of variants to request. Default 4.

    Returns:
        list[str]: Plain variant strings (numbering/quotes stripped); empty if
        the narrator is unavailable.

    Logical Rationale:
        A UI-side cousin of generate_multiple_queries: same recall-widening idea,
        but on the cheaper narrator model and ``@st.cache_data``-cached so
        re-running the identical question never re-bills the LLM.
    """
    if _narrator is None or not (question or "").strip():
        return []
    prompt = (
        f"Rewrite this policy question in {n} alternative ways. Use synonyms, "
        f"related concepts, and formal/informal phrasings. Output ONLY the "
        f"variants — one per line, no numbering, no quotes, no preamble.\n\n"
        f"Question: {question.strip()}"
    )
    try:
        resp = _narrator.invoke(prompt)
        text = (getattr(resp, "content", "") or str(resp)).strip()
        out = []
        for line in text.splitlines():
            line = line.strip().lstrip("-•0123456789. )").strip().strip('"').strip()
            if line and line.lower() != question.strip().lower():
                out.append(line)
        return out[:n]
    except Exception:
        return []


def _semantic_search(graph, question: str, top_k: int = 8,
                     threshold: float = SEMANTIC_THRESHOLD,
                     score_window: int = 12,
                     use_fusion: bool = True):
    """Retrieve grounding rows via Adaptive Fusion / dual-stream hybrid search.

    Args:
        graph (Neo4jGraph | None): The connected graph.
        question (str): The cleaned user question.
        top_k (int): Max kept nodes in Fusion mode. Default 8.
        threshold (float): Cosine cut applied in Fusion mode only
            (``SEMANTIC_THRESHOLD`` = 0.70).
        score_window (int): Candidates pulled per vector index. Default 12.
        use_fusion (bool): True → COMPLEX_REASONING path; False → DIRECT_LOOKUP.

    Returns:
        tuple[str | None, list[dict], list[dict]]: ``(annotated_cypher, rows,
        scores)`` — an audit-bearing query header, the grounding rows, and the
        per-candidate transparency records (cosine, rrf, kept flag). ``(None,
        [], [])`` when nothing is retrievable.

    Mathematical/Logical Rationale:
        Two retrieval profiles share one RRF backbone (k=60):

        • **COMPLEX_REASONING (use_fusion=True).** Query expansion yields N
          phrasings; each is embedded and run over both vector indices, giving N
          ranked lists fused by RRF. The cosine ``threshold`` IS applied here —
          in the multi-angle regime a high embedding score is trustworthy
          evidence, so trimming sub-0.70 hits removes drift.

        • **DIRECT_LOOKUP (use_fusion=False).** Two heterogeneous streams are
          fused: Stream A = single-query vector similarity; Stream B = a keyword
          CONTAINS count over description+label+text+label_human, seeded from a
          curated ``_DOMAIN_KWS`` set plus soft tokens. The cosine threshold is
          DELIBERATELY NOT applied; the kept set is the strict top-5 by RRF rank.
          Rationale: a rare statutory keyword can be the exact answer yet score
          ~0.0 cosine because the embedder under-represents it — RRF lets the
          keyword stream rescue it, and a threshold cut would silently discard
          precisely the node a factual lookup needs. RRF's rank-based fusion is
          what makes combining an unbounded match-count with a [0,1] cosine
          coherent without normalisation (see reciprocal_rank_fusion).
    """
    if _embedder is None or graph is None or not (question or "").strip():
        return None, [], []

    # ── Step 1: Query expansion / keyword extraction ──────────────────────────
    if use_fusion:
        fusion_variants = generate_multiple_queries(question.strip(), num_queries=3)
        queries = [question.strip()] + [v for v in fusion_variants
                                         if v.strip() and v.strip() != question.strip()]
        direct_kws: list[str] = []
    else:
        queries = [question.strip()]
        # Domain-anchored keywords to bias the keyword stream toward specific chapters.
        _DOMAIN_KWS = {
            "word", "limit", "words", "maximum", "minimum", "thesis",
            "chapter", "submit", "submission", "deadline", "resubmit",
            "withdrawal", "examination", "viva", "award", "penalty",
            "fail", "failure", "sanction", "appendix", "abstract",
        }
        q_lower = question.lower()
        hard_kws = [kw for kw in _DOMAIN_KWS if kw in q_lower]
        soft_kws = [
            re.sub(r"[^a-z0-9]", "", w)
            for w in re.findall(r"[A-Za-z]{4,}", q_lower)
            if w not in _STOPWORDS
        ]
        direct_kws = list(dict.fromkeys(hard_kws + soft_kws))[:8]
        direct_kws = [k for k in direct_kws if k]

    try:
        vectors = _embedder.embed_documents(queries)
    except Exception:
        return None, [], []

    # ── Step 2: Vector retrieval (both modes) ─────────────────────────────────
    max_sim: dict[tuple[str, str], float] = {}
    per_query_rankings: list[list[str]] = []

    for vec in vectors:
        query_hits: list[tuple[str, str, float]] = []
        for idx_name in ("rule_text_index", "condition_text_index"):
            try:
                idx_rows = graph.query(
                    """
                    CALL db.index.vector.queryNodes($idx, $k, $vec)
                    YIELD node, score
                    RETURN labels(node)[0] AS label, node.id AS id, score
                    """,
                    {"idx": idx_name, "k": score_window, "vec": vec},
                )
            except Exception:
                idx_rows = []
            for r in idx_rows:
                label, nid, sc = r.get("label"), r.get("id"), float(r.get("score", 0))
                if not nid:
                    continue
                query_hits.append((label, nid, sc))
                key = (label, nid)
                if sc > max_sim.get(key, 0.0):
                    max_sim[key] = sc

        query_hits.sort(key=lambda t: -t[2])
        per_query_rankings.append([t[1] for t in query_hits])

    # ── Step 2b: Keyword stream — DIRECT_LOOKUP only ─────────────────────────
    if not use_fusion and direct_kws:
        try:
            kw_rows = graph.query(
                """
                MATCH (n)
                WHERE (n:Rule OR n:Condition)
                WITH n,
                     [kw IN $kws
                      WHERE toLower(
                        coalesce(n.description,  '') + ' ' +
                        coalesce(n.label,         '') + ' ' +
                        coalesce(n.text,          '') + ' ' +
                        coalesce(n.label_human,   '')
                      ) CONTAINS kw] AS matched
                WHERE size(matched) > 0
                RETURN labels(n)[0] AS label, n.id AS id,
                       toFloat(size(matched)) AS score
                ORDER BY score DESC
                LIMIT $k
                """,
                {"kws": direct_kws, "k": score_window},
            )
        except Exception:
            kw_rows = []

        stream_b: list[tuple[str, str, float]] = []
        for r in kw_rows:
            label, nid, sc = r.get("label"), r.get("id"), float(r.get("score", 0))
            if not nid:
                continue
            stream_b.append((label, nid, sc))
            key = (label, nid)
            # Preserve cosine if vector already scored this node; otherwise 0.0
            # so it still appears in the transparency chart with its RRF rank.
            if key not in max_sim:
                max_sim[key] = 0.0

        stream_b.sort(key=lambda t: -t[2])
        per_query_rankings.append([t[1] for t in stream_b])

    if not max_sim:
        return None, [], []

    # ── Step 3: Reciprocal Rank Fusion across all streams ────────────────────
    # CRITICAL PARAMETER — RRF k=60. The fusion smoothing constant; RRF(d) =
    # Σ 1/(k + rank_i(d)). k=60 is the value validated in the original Cormack
    # et al. RRF work and retained unchanged: large enough to damp the dominance
    # of any single stream's #1 hit (so vector and keyword streams genuinely
    # negotiate), small enough that top ranks still matter. Empirically robust on
    # the sparse/dense mix this system fuses (per ARCHITECTURE_REVIEW.md §4c-4d).
    rrf_scores_raw = dict(reciprocal_rank_fusion(per_query_rankings, k=60))

    # ── Step 4: Transparency record (kept flag uses RRF rank for direct mode) ─
    all_keys = sorted(
        max_sim.keys(),
        key=lambda kv: (-rrf_scores_raw.get(kv[1], 0.0), -max_sim[kv]),
    )[:score_window]

    if not use_fusion:
        # Direct: kept = top-5 by RRF regardless of cosine (keyword hits are valid)
        direct_top5_ids = {k[1] for k in all_keys[:5]}
        scores = [
            {
                "label": k[0],
                "id":    k[1],
                "score": float(max_sim[k]),
                "rrf":   float(rrf_scores_raw.get(k[1], 0.0)),
                "kept":  k[1] in direct_top5_ids,
            }
            for k in all_keys
        ]
    else:
        scores = [
            {
                "label": k[0],
                "id":    k[1],
                "score": float(max_sim[k]),
                "rrf":   float(rrf_scores_raw.get(k[1], 0.0)),
                "kept":  max_sim[k] >= threshold,
            }
            for k in all_keys
        ]

    # ── Step 5: Select kept nodes ─────────────────────────────────────────────
    if not use_fusion:
        # Strict top-5 by RRF — no cosine threshold so keyword-only hits survive
        kept_keys = [(k, max_sim[k]) for k in all_keys[:5]]
    else:
        kept_keys = [
            (k, max_sim[k]) for k in all_keys if max_sim[k] >= threshold
        ][:top_k]

    if not kept_keys:
        return None, [], scores

    rule_ids = [k[1] for (k, _) in kept_keys if k[0] == "Rule"]
    cond_ids = [k[1] for (k, _) in kept_keys if k[0] == "Condition"]

    cypher = """
    MATCH (r:Rule)-[:BELONGS_TO]->(p:Policy)
    WHERE r.id IN $rule_ids
       OR EXISTS {
            MATCH (r)-[:HAS_CONDITION]->(c0:Condition)
            WHERE c0.id IN $cond_ids
       }
    OPTIONAL MATCH (r)-[:HAS_CONDITION]->(c:Condition)
    OPTIONAL MATCH (r)-[:HAS_OUTCOME]->(o:Outcome)
    OPTIONAL MATCH (c)-[cf:CONFLICTS_WITH]-(c2:Condition)
    RETURN p.title  AS policyTitle, p.id AS policyId,
           r.id     AS ruleId,    r.label         AS ruleLabel,
           r.sourceSection AS ruleSection,
           r.description   AS ruleDescription,
           c.id     AS conditionId, c.text        AS conditionText,
           c.label_human   AS conditionTitle,
           collect(DISTINCT o.type + ': ' + o.text) AS outcomes,
           cf.scenario     AS conflictScenario,
           cf.resolution   AS conflictResolution,
           cf.cross_policy AS conflictCrossPolicy,
           c2.id           AS conflictingConditionId,
           c2.text         AS conflictingConditionText
    LIMIT 25
    """
    try:
        rows = graph.query(cypher, {"rule_ids": rule_ids, "cond_ids": cond_ids})
    except Exception:
        rows = []

    n_streams = len(per_query_rankings)
    if not use_fusion:
        mode = (
            f"Adaptive RAG · Direct Lookup "
            f"(hybrid {n_streams}-stream RRF, top-5, kws={direct_kws})"
        )
    elif _FUSION_RAG_AVAILABLE:
        mode = f"Adaptive RAG · Complex Reasoning (Fusion RAG, {len(queries)} queries)"
    else:
        mode = "semantic"

    annotated_cypher = (
        f"-- {mode} | RRF k=60 | threshold={threshold} --\n"
        f"-- queries: {queries}\n"
        f"-- top hits (label, id → cosine | rrf):\n"
        + "\n".join(
            f"--   ({s['label']}, {s['id']}) → "
            f"cosine={s['score']:.3f} | rrf={s['rrf']:.4f}"
            f"{'  ✓ kept' if s['kept'] else '  ✗ dropped'}"
            for s in scores
        )
        + "\n"
        + cypher.strip()
    )
    return annotated_cypher, rows, scores


def keyword_fallback_search(graph, question, use_fusion: bool = True):
    """Fallback retrieval when the structured Cypher chain returns nothing.

    Args:
        graph (Neo4jGraph | None): The connected graph.
        question (str): The cleaned user question.
        use_fusion (bool): Forwarded to _semantic_search — False selects the
            DIRECT_LOOKUP dual-stream path (no multi-query expansion).

    Returns:
        tuple[str | None, list[dict], list[dict]]: ``(cypher, rows,
        semantic_scores)``; ``semantic_scores`` carries the candidate
        kept/dropped flags for the audit chart (or [] if vector search was
        unused/failed).

    Logical Rationale:
        Two-tier degradation: semantic/RRF search first (richest signal), then a
        legacy CONTAINS query that works even with NO embedder — so the system
        still answers offline or behind a proxy that blocks the embedding
        endpoint. Candidate scores are passed through even on a miss so the UI
        can show *why* the cut failed rather than a blank chart.
    """
    if graph is None:
        return None, [], []

    # 1. Semantic first
    if _embedder is not None:
        cypher, rows, scores = _semantic_search(graph, question, use_fusion=use_fusion)
        if rows:
            return cypher, rows, scores
        # Even if semantic returned no kept rows, the candidate scores may
        # still be useful in the audit chart — pass them through to the
        # keyword fallback caller so the UI can show why the cut failed.
        cached_scores = scores or []
    else:
        cached_scores = []

    # 2. Legacy keyword (CONTAINS) fallback — works even without embeddings
    if use_fusion:
        # Full pipeline: use the shared keyword extractor (stopword-filtered)
        kws = _keywords(question)
    else:
        # Direct Lookup: strict re-tokenisation of the isolated user string —
        # no generation module, no derived text, minimum 4-char alpha tokens.
        kws = [
            w for w in re.findall(r"[A-Za-z]{4,}", question.lower())
            if w not in _STOPWORDS
        ][:6]
    if not kws:
        return None, [], cached_scores
    clause = " OR ".join(
        f"toLower(coalesce(c.text,'')) CONTAINS '{k}' OR "
        f"toLower(coalesce(r.description,'')) CONTAINS '{k}' OR "
        f"toLower(coalesce(r.label,'')) CONTAINS '{k}' OR "
        f"toLower(coalesce(o.text,'')) CONTAINS '{k}'"
        for k in kws
    )
    cypher = f"""
    MATCH (r:Rule)-[:BELONGS_TO]->(p:Policy)
    OPTIONAL MATCH (r)-[:HAS_CONDITION]->(c:Condition)
    OPTIONAL MATCH (r)-[:HAS_OUTCOME]->(o:Outcome)
    OPTIONAL MATCH (c)-[cf:CONFLICTS_WITH]-(c2:Condition)
    WITH r, p, c, o, cf, c2
    WHERE {clause}
    RETURN p.title AS policyTitle, p.id AS policyId,
           r.id AS ruleId, r.label AS ruleLabel,
           r.sourceSection AS ruleSection, r.description AS ruleDescription,
           c.id AS conditionId, c.text AS conditionText,
           c.label_human AS conditionTitle,
           collect(DISTINCT o.type + ': ' + o.text) AS outcomes,
           cf.scenario    AS conflictScenario,
           cf.resolution  AS conflictResolution,
           c2.id          AS conflictingConditionId,
           c2.text        AS conflictingConditionText
    LIMIT 25
    """
    try:
        rows = graph.query(cypher)
        return f"-- keyword CONTAINS fallback --\n{cypher.strip()}", rows, cached_scores
    except Exception:
        return f"-- keyword CONTAINS fallback (failed) --\n{cypher.strip()}", [], cached_scores


def _ground_answer_from_rows(question, rows):
    """Synthesise a grounded answer from fallback rows using QA_PROMPT.

    Args:
        question (str): The user question.
        rows (list): Retrieved grounding rows (capped to 25 for the context).

    Returns:
        str: The grounded answer, or "LLM unavailable." without an API key.

    Logical Rationale:
        Used on the fallback path AFTER the CRAG gate has judged the context
        RELEVANT, so synthesis only ever runs over validated context. Uses the
        same QA_PROMPT (and T=0.1) as the primary chain so a fallback answer is
        stylistically and behaviourally indistinguishable from a primary one.
    """
    if not OPENAI_API_KEY:
        return "LLM unavailable."
    qa = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY, **_llm_kwargs())
    context = str(rows[:25])
    prompt = QA_PROMPT.format(context=context, question=question)
    return qa.invoke(prompt).content


# ─────────────────────────────────────────────────────────────────────────────
# 7.5  TRUST DASHBOARD & PLAIN-ENGLISH NARRATOR
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — transparency surface
#   These helpers turn raw retrieval into the trust signals a non-technical
#   reader can act on: a confidence verdict, a metric row (rules/conflicts/
#   citations), and a 2-sentence plain-English narration that never leaks IDs,
#   Cypher, or the word "graph". The narrator runs on gpt-4o-mini for cost.
#
# EXTENDING THIS SECTION
#   • New trust metric: add a column in render_trust_dashboard and a _row_get
#     over the relevant result keys (key names vary by retrieval path, hence the
#     multi-key _row_get).
#   • New confidence tier: adjust compute_confidence's thresholds; keep it a pure
#     function of (row count, fallback flag, cypher presence) so it stays testable.
# ─────────────────────────────────────────────────────────────────────────────
def _init_narrator():
    """Construct the gpt-4o-mini plain-English narrator, or None without a key.

    Returns:
        ChatOpenAI | None: A low-cost narrator (T=0.2, max_tokens=120) used for
        search variants and the friendly path summary; None if no API key, in
        which case callers fall back to a static message.
    """
    if OPENAI_API_KEY:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=120,
            api_key=OPENAI_API_KEY,
            **_llm_kwargs(),
        )
    return None


_narrator = _init_narrator()


def _row_get(row, *keys):
    """Read the first non-empty value among several candidate keys in a row.

    Args:
        row (dict): A Neo4j result row.
        *keys (str): Candidate key names, tried in order.

    Returns:
        The first present, non-empty value, or None.

    Logical Rationale:
        Result key names differ across retrieval paths (e.g. ``ruleId`` from the
        primary chain vs ``r.id`` from a raw query), so the dashboard accepts a
        list of aliases rather than assuming one schema — keeping the UI robust
        to which path produced the rows.
    """
    if not isinstance(row, dict):
        return None
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def compute_confidence(result_rows, cypher, used_fallback) -> tuple[str, str]:
    """Derive a coarse answer-confidence verdict from retrieval signals.

    Args:
        result_rows (list): The grounding rows.
        cypher (str): The executed query string ("N/A" if none ran).
        used_fallback (bool): True if the answer came from the fallback path.

    Returns:
        tuple[str, str]: ``(label, icon)`` ∈ {("Low","🔴"), ("Medium","🟡"),
        ("High","🟢")}.

    Logical Rationale:
        Confidence is a transparent function of three observable signals, not an
        LLM self-estimate: zero rows ⇒ Low; a fallback answer ⇒ at most Medium
        (structured retrieval missed, so trust is reduced); only a primary-chain
        answer with ≥3 grounding rows earns High. Honest under-claiming is
        preferred to overconfidence on a compliance surface.
    """
    n = len(result_rows) if result_rows else 0
    if n == 0:
        return ("Low", "🔴")
    if used_fallback:
        return ("Medium", "🟡")
    if cypher and cypher != "N/A" and n >= 3:
        return ("High", "🟢")
    return ("Medium", "🟡")


def render_trust_dashboard(result_rows, cypher, used_fallback):
    """Render the 4-metric trust row for an answer.

    Args:
        result_rows (list): The grounding rows.
        cypher (str): The executed query (feeds the confidence verdict).
        used_fallback (bool): Whether the fallback path produced the answer.

    Returns:
        None. Writes a 4-column st.metric row — Rules checked / Conflicts
        detected / Citations / Confidence — so the reader sees the evidence
        behind the answer at a glance.
    """
    rows = result_rows or []

    rules_checked = len({
        _row_get(r, "ruleId", "r.id", "id")
        for r in rows
        if _row_get(r, "ruleId", "r.id", "id")
    })
    conflicts = sum(
        1 for r in rows
        if _row_get(
            r, "conflictScenario", "cf.scenario",
            "conflictingConditionId", "c2.id",
            "conflictResolution", "cf.resolution",
        )
    )
    citations = len({
        _row_get(r, "ruleSection", "r.sourceSection", "sourceSection")
        for r in rows
        if _row_get(r, "ruleSection", "r.sourceSection", "sourceSection")
    })
    conf_label, conf_icon = compute_confidence(rows, cypher, used_fallback)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Rules checked",     rules_checked)
    c2.metric("⚠️ Conflicts detected", conflicts)
    c3.metric("🔗 Citations",         citations)
    c4.metric("🎯 Confidence",        f"{conf_icon} {conf_label}")


def narrate_path(result_rows) -> str:
    """Summarise the retrieval as exactly 2 jargon-free sentences for a student.

    Args:
        result_rows (list): The grounding rows.

    Returns:
        str: A 2-sentence plain-English summary of which policy areas were
        consulted and whether conflicts were found; a static fallback message if
        the narrator is unavailable.

    Logical Rationale:
        The prompt forbids node IDs, Cypher, and the word "graph" so a
        non-technical reader is never exposed to retrieval internals — the
        transparency goal is *understanding*, not a data dump.
    """
    if _narrator is None or not result_rows:
        return "We checked the policy graph for you."

    preview = str(result_rows[:8])[:1500]
    prompt = (
        "You are explaining to a PhD student how we found their answer. "
        "In EXACTLY 2 short friendly sentences, summarise which policy areas "
        "were consulted and whether any conflicts were found. "
        "STRICT RULES — you MUST NOT:\n"
        "  • mention any node ID (e.g. R5.1, C5.1.2, O4.1.1)\n"
        "  • mention Cypher, queries, databases, or technical retrieval terms\n"
        "  • use the word 'graph' anywhere in the response\n"
        "Use plain English. You MAY refer to sections as 'Section X.Y.Z' if helpful.\n\n"
        f"Retrieved policy information:\n{preview}"
    )
    resp = _narrator.invoke(prompt)
    text = getattr(resp, "content", str(resp)) or ""
    return text.strip() or "We checked the policy graph for you."


def extract_active_policy_ids(raw_rows) -> set:
    """
    Walk the Cypher result rows and collect every distinct policy identifier
    the retrieval actually touched. Looks at policyId / policy_id / p.id in
    any nested dict/list/tuple, mirroring how extract_node_ids walks output.
    """
    pids: set = set()

    def walk(o):
        """Recurse through the result, collecting policy identifiers into ``pids``."""
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("policyId", "policy_id", "p.id") and isinstance(v, str) and v:
                    pids.add(v)
                else:
                    walk(v)
        elif isinstance(o, (list, tuple, set)):
            for it in o:
                walk(it)

    walk(raw_rows)
    return pids


_CRAG_REFUSAL = (
    "**Policy context {status}** — the retrieved policy nodes do not contain "
    "sufficient information to answer this question reliably. "
    "Providing standard fallback search...\n\n"
    "_Please consult the BU Doctoral College directly, or try rephrasing "
    "your question with more specific policy terms._"
)


def run_query(question, chain):
    """Execute the full Modular-RAG read path for one question.

    Args:
        question (str): The raw user question (cleaned internally).
        chain (GraphCypherQAChain): The structured-retrieval chain.

    Returns:
        dict: A result bundle with keys ``answer``, ``cypher``, ``raw`` (rows),
        ``node_ids`` (cited, for the Reasoning View), ``policy_ids``,
        ``used_fallback``, ``semantic_scores``, ``crag_status``, ``crag_reason``,
        ``route``, ``route_reason``, ``error``.

    Logical Rationale:
        Orchestrates the pipeline in fixed order: clean → route (Adaptive RAG,
        which sets ``use_fusion``) → structured Cypher first → if empty or the
        "not available" sentinel fires, Fusion/keyword fallback → CRAG gate →
        grounded answer (or honest refusal). CRAG runs at two points: as a
        PRE-generation gate on the fallback path (so a weak context never reaches
        synthesis) and as a POST-hoc check on the primary path (an independent
        audit signal). Every branch funnels into one return shape so the UI can
        render the same transparency fields regardless of which path answered.
        The whole body is wrapped so any exception returns an error bundle rather
        than crashing the Streamlit run.
    """
    # ── Isolate the exact user string — strip whitespace and any UI leakage ──
    question = _clean_user_query(question)

    cypher = "N/A"
    raw = []
    used_fallback = False
    semantic_scores = []
    crag_status = "RELEVANT"
    crag_reason = "Not evaluated"
    route = "COMPLEX_REASONING"
    route_reason = "Not evaluated"
    try:
        # ── Adaptive RAG: classify query complexity before any retrieval ──────
        routing = _route_query(question)
        route        = routing["route"]
        route_reason = routing["reason"]
        use_fusion   = (route == "COMPLEX_REASONING")

        result = chain.invoke({"query": question})
        steps = result.get("intermediate_steps", [])
        cypher = steps[0].get("query", "N/A") if steps else "N/A"
        raw    = steps[1].get("context", []) if len(steps) > 1 else []
        answer = result["result"]

        # ── Fallback — did the LLM chain come up dry? ────────────────────────
        empty_result = (not raw) or (isinstance(raw, list) and len(raw) == 0)
        saying_unavailable = "not available in the current policy graph" in answer.lower()

        if empty_result or saying_unavailable:
            # Pass the clean question explicitly so keyword extraction is never
            # contaminated by annotated cypher strings or prior answer text.
            fb_cypher, fb_rows, fb_scores = keyword_fallback_search(
                graph, question, use_fusion=use_fusion,
            )
            semantic_scores = fb_scores or []
            if fb_rows:
                # ── CRAG: validate fallback context BEFORE generating answer ─
                crag = evaluate_context_relevance(fb_rows, question)
                crag_status = crag["status"]
                crag_reason = crag["reason"]
                if crag_status == "RELEVANT":
                    answer = _ground_answer_from_rows(question, fb_rows)
                else:
                    answer = _CRAG_REFUSAL.format(status=crag_status.lower())
                cypher = (
                    f"-- Primary returned empty; Fusion/keyword fallback used "
                    f"[CRAG: {crag_status}] --\n{fb_cypher}"
                )
                raw    = fb_rows
                used_fallback = True
            else:
                # Nothing found at all — mark IRRELEVANT without calling LLM
                crag_status = "IRRELEVANT"
                crag_reason = "No matching policy context found after fallback search."
        else:
            # ── CRAG: post-hoc quality gate on the primary chain result ──────
            # The QA prompt already guards against hallucination, but CRAG gives
            # an independent signal Sofia can see in the audit trail.
            crag = evaluate_context_relevance(raw, question)
            crag_status = crag["status"]
            crag_reason = crag["reason"]
            if crag_status != "RELEVANT":
                answer = _CRAG_REFUSAL.format(status=crag_status.lower())

        # ── Active policy scope: which policies did the retrieval touch? ────
        active_policy_ids = extract_active_policy_ids(raw)

        # ── Node IDs: ONLY those explicitly mentioned in the LLM's answer ───
        answer_ids = extract_ids_from_text(answer)
        cited_sections = extract_sections_from_text(answer)
        section_ids = resolve_section_refs_to_ids(
            graph, cited_sections, policy_ids=active_policy_ids,
        )
        node_ids = answer_ids | section_ids

        return {
            "answer":          answer,
            "cypher":          cypher,
            "raw":             raw,
            "node_ids":        node_ids,
            "policy_ids":      active_policy_ids,
            "used_fallback":   used_fallback,
            "semantic_scores": semantic_scores,
            "crag_status":     crag_status,
            "crag_reason":     crag_reason,
            "route":           route,
            "route_reason":    route_reason,
            "error":           None,
        }
    except Exception as e:
        return {
            "answer":          f"⚠️ Error: {e}",
            "cypher":          cypher,
            "raw":             raw,
            "node_ids":        set(),
            "policy_ids":      set(),
            "used_fallback":   used_fallback,
            "semantic_scores": semantic_scores,
            "crag_status":     crag_status,
            "crag_reason":     crag_reason,
            "route":           route,
            "route_reason":    route_reason,
            "error":           str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.5  RAGAS BENCHMARKING — Gold Dataset comparison
# ─────────────────────────────────────────────────────────────────────────────
try:
    from evaluation_dataset import EVAL_DATASET
except Exception:
    EVAL_DATASET = []


def _ctx_precision(expected: set, retrieved: set) -> float:
    """Context Precision — fraction of retrieved nodes that were expected.

    Args:
        expected (set): Gold node IDs for the question.
        retrieved (set): Node IDs the pipeline actually retrieved.

    Returns:
        float: ``|expected ∩ retrieved| / |retrieved|`` in [0, 1].

    Mathematical/Logical Rationale:
        Precision penalises retrieving *irrelevant* nodes (noise in the synthesis
        context). Edge cases follow set-logic convention: both sets empty ⇒ 1.0
        (vacuously perfect — nothing wrong was retrieved); retrieved empty but
        expectations exist ⇒ 0.0 (nothing right was retrieved).
    """
    if not expected and not retrieved:
        return 1.0
    if not retrieved:
        return 0.0
    return len(expected & retrieved) / len(retrieved)


def _ctx_recall(expected: set, retrieved: set) -> float:
    """Context Recall — fraction of expected nodes that were retrieved.

    Args:
        expected (set): Gold node IDs for the question.
        retrieved (set): Node IDs the pipeline actually retrieved.

    Returns:
        float: ``|expected ∩ retrieved| / |expected|`` in [0, 1]; 1.0 when no
        nodes were expected (nothing to miss).

    Mathematical/Logical Rationale:
        Recall penalises *missing* required evidence. Precision and recall are
        complementary: a pipeline can game one alone (retrieve everything ⇒
        recall 1, precision low; retrieve one sure hit ⇒ precision 1, recall
        low), so the benchmark reports both to expose that trade-off per query.
    """
    if not expected:
        return 1.0
    return len(expected & retrieved) / len(expected)


def _faithfulness_quick(answer: str, context_rows: list) -> float:
    """
    Lightweight LLM-judge faithfulness. Asks the narrator model to estimate
    what fraction of the answer's claims are entailed by the retrieved
    context. Returns 0.0 on any failure so the table never breaks.
    """
    if _narrator is None or not (answer or "").strip():
        return 0.0
    if not context_rows:
        return 0.0
    ctx_blob = str(context_rows[:12])[:3500]
    prompt = (
        "You are an evaluation judge. Estimate the FAITHFULNESS of the ANSWER "
        "given the CONTEXT — the fraction of factual claims in the answer "
        "that are supported by the context. Reply with ONE number between "
        "0.0 and 1.0, no words.\n\n"
        f"CONTEXT:\n{ctx_blob}\n\nANSWER:\n{answer}\n\nScore:"
    )
    try:
        resp = _narrator.invoke(prompt)
        text = (getattr(resp, "content", "") or "").strip()
        m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def _simple_rag_retrieve(graph, question: str, top_k: int = 10):
    """
    Flat 'Simple RAG' baseline: top-k semantically similar Rule/Condition
    nodes, NO graph traversal — what a vector DB would return on its own.
    Returns (retrieved_ids, rows).
    """
    if _embedder is None or graph is None:
        return set(), []
    try:
        vec = _embedder.embed_query(question)
    except Exception:
        return set(), []

    rows: list = []
    for idx_name in ("rule_text_index", "condition_text_index"):
        try:
            r = graph.query(
                "CALL db.index.vector.queryNodes($idx, $k, $vec) "
                "YIELD node, score "
                "RETURN node.id AS id, labels(node)[0] AS label, "
                "       coalesce(node.text, node.description, node.label) AS text, "
                "       score",
                {"idx": idx_name, "k": top_k, "vec": vec},
            )
            rows.extend(r)
        except Exception:
            continue

    rows.sort(key=lambda r: -float(r.get("score") or 0.0))
    rows = rows[:top_k]
    ids = {r["id"] for r in rows if r.get("id")}
    return ids, rows


def benchmark_question(graph, chain, test_case: dict) -> dict:
    """Benchmark one gold case across GraphRAG and a flat Simple-RAG baseline.

    Args:
        graph (Neo4jGraph): The connected graph (for the baseline retriever).
        chain (GraphCypherQAChain): The structured chain (for GraphRAG).
        test_case (dict): A gold case with ``question``, ``ground_truth``,
            ``expected_node_ids``, and optional ``expected_edges``.

    Returns:
        dict: Per-pipeline metrics (context precision/recall, path accuracy,
        faithfulness, counts) plus the case metadata — ready for UI rendering.

    Mathematical/Logical Rationale:
        Running both pipelines on the SAME gold case is what makes the comparison
        fair: identical expected sets feed _ctx_precision/_ctx_recall, so any
        delta reflects retrieval strategy, not question difficulty. ``path
        accuracy`` (fraction of expected edges present in the generated Cypher)
        is GraphRAG-only and 0.0 for the baseline by definition — the baseline
        performs no graph traversal, which is precisely the capability under test.
    """
    question     = test_case["question"]
    ground_truth = test_case.get("ground_truth", "")
    expected_ids = set(test_case.get("expected_node_ids") or set())
    expected_edges = test_case.get("expected_edges", []) or []

    # ── GraphRAG ────────────────────────────────────────────────────────────
    try:
        gr = run_query(question, chain)
    except Exception as e:
        gr = {"answer": f"Error: {e}", "raw": [], "node_ids": set(),
              "cypher": "N/A", "used_fallback": False}
    gr_retrieved = set(gr.get("node_ids") or set())
    gr_cypher    = (gr.get("cypher") or "").upper()
    gr_path_acc  = (
        sum(1 for e in expected_edges if e.upper() in gr_cypher)
        / max(len(expected_edges), 1)
    ) if expected_edges else 1.0

    # ── Simple RAG baseline ────────────────────────────────────────────────
    sr_ids, sr_rows = _simple_rag_retrieve(graph, question)

    return {
        "id":             test_case.get("id", ""),
        "category":       test_case.get("category", ""),
        "question":       question,
        "ground_truth":   ground_truth,
        "expected_ids":   sorted(expected_ids),
        "expected_edges": list(expected_edges),

        "graphrag": {
            "answer":            gr.get("answer", ""),
            "retrieved_ids":     sorted(gr_retrieved),
            "context_precision": _ctx_precision(expected_ids, gr_retrieved),
            "context_recall":    _ctx_recall(expected_ids, gr_retrieved),
            "path_accuracy":     gr_path_acc,
            "faithfulness":      _faithfulness_quick(gr.get("answer", ""),
                                                     gr.get("raw", [])),
            "n_retrieved":       len(gr_retrieved),
            "used_fallback":     gr.get("used_fallback", False),
        },
        "simplerag": {
            "retrieved_ids":     sorted(sr_ids),
            "context_precision": _ctx_precision(expected_ids, sr_ids),
            "context_recall":    _ctx_recall(expected_ids, sr_ids),
            "path_accuracy":     0.0,   # by definition: no graph traversal
            "faithfulness":      None,  # baseline doesn't generate an answer
            "n_retrieved":       len(sr_ids),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — Streamlit session state
#   All cross-rerun UI state lives in st.session_state, seeded once by
#   _init_state(). Streamlit re-executes this whole script top-to-bottom on every
#   interaction, so anything that must survive a rerun (chat history, last
#   reasoning subgraph, layout toggle) belongs here, not in a local variable.
#
# EXTENDING THIS SECTION
#   • New persistent UI field: add a key + default to the `defaults` dict below.
#     Initialise it here (never inline at the use site) so a rerun before the
#     widget is drawn cannot raise KeyError.
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    """Seed all required st.session_state keys with defaults exactly once.

    Returns:
        None. Mutates ``st.session_state`` in place, only filling keys that are
        absent so existing user state is never clobbered on a rerun.
    """
    defaults = {
        "messages": [],
        "ingest_log": [],
        "ingest_stats": None,
        "show_full_graph": False,
        "last_reasoning": None,
        "last_policy_ids": set(),
        "layout_mode": "organic",
        "benchmark_result": None,
        "top_k_reasoning": 5,
        "last_selected_node": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  INITIALISE RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
graph, neo4j_err = init_neo4j()
# Never let a failed attempt stick: st.cache_resource would otherwise pin the
# (None, error) tuple for the whole server lifetime, so a paused/cold-starting
# AuraDB instance would read "Offline" forever even after it comes back. Clear
# the cache on failure so the next rerun (or the Reconnect button) retries live.
if graph is None:
    init_neo4j.clear()
chain, chain_err = init_chain(graph) if graph else (None, "Neo4j unavailable")
if chain is None and graph is not None:
    init_chain.clear()
total_nodes = count_nodes(graph) if graph else 0


# ─────────────────────────────────────────────────────────────────────────────
# 10. SIDEBAR — Status dashboard + data mgmt + graph controls
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — sidebar control surface (top-level script, runs every rerun)
#   Renders connection status, the "🔄 Reconnect" button (which clears the
#   cached resources — see §4/§9), PDF ingestion, policy listing/deletion, and
#   graph-layout controls. This is procedural Streamlit, not a function, so it
#   reads the module-level `graph`/`chain` initialised in §9.
#
# EXTENDING THIS SECTION
#   • New sidebar control: add it inside the `with st.sidebar:` block and persist
#     any stateful value through st.session_state (declared in §8).
#   • Any action that invalidates a cached resource must call its `.clear()` and
#     `st.rerun()` so the change takes effect — mirror the Reconnect button.
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 BU Policy GraphRAG")
    st.caption("Transparent Knowledge Platform · MSc Data Science & AI · BU 2025")
    st.divider()

    # ── System Status Dashboard ─────────────────────────────────────────────
    st.markdown("**System Status**")
    sc1, sc2 = st.columns(2)
    sc1.metric("Neo4j", "🟢 Live" if graph else "🔴 Offline")
    sc2.metric("GraphRAG", "🟢 Ready" if chain else "🔴 Off")
    st.metric(
        "OpenAI API",
        "🟢 Ready" if OPENAI_API_KEY else "🟡 Missing",
        f"{total_nodes} nodes in graph",
    )
    if neo4j_err and not graph:
        st.caption(f"⚠ {neo4j_err[:120]}")
    if not graph or not chain:
        if st.button("🔄 Reconnect", use_container_width=True):
            init_neo4j.clear()
            init_chain.clear()
            st.rerun()
    st.divider()

    # ── Data Management ─────────────────────────────────────────────────────
    st.markdown("**Data Management**")

    # Policy metadata — applies to BOTH uploaders so every ingestion lives
    # inside its own (:Policy) namespace and never confuses rules across
    # different documents.
    policy_title_input = st.text_input(
        "Policy Title",
        value=st.session_state.get("policy_title_input", ""),
        placeholder="e.g. Partnership Coordinators Manual",
        key="policy_title_input",
    )
    policy_version_input = st.text_input(
        "Version",
        value=st.session_state.get("policy_version_input", "2024-25"),
        placeholder="e.g. 2024-25",
        key="policy_version_input",
    )
    if not policy_title_input.strip():
        st.caption("⚠ Enter a policy title to enable ingestion.")

    uploaded = st.file_uploader("Upload XML Policy File", type=["xml"],
                                label_visibility="collapsed")
    ingest_disabled = not policy_title_input.strip()
    if uploaded and st.button("⚡ Ingest to Neo4j", use_container_width=True,
                              disabled=ingest_disabled):
        if not graph:
            st.error("Neo4j not connected.")
        else:
            st.session_state.ingest_log = []
            def logger(level, msg):
                """log_fn sink: append (level, msg) to the live ingest log in session state."""
                st.session_state.ingest_log.append((level, msg))
            try:
                with st.spinner("Ingesting XML..."):
                    stats = ingest_xml_to_neo4j(
                        uploaded.read(), graph, logger,
                        policy_title=policy_title_input.strip(),
                        policy_version=policy_version_input.strip() or "unspecified",
                    )
                st.session_state.ingest_stats = stats
                init_chain.clear()
                st.rerun()
            except Exception as e:
                logger("err", str(e))
                st.error(f"Ingestion failed: {e}")

    # ── PDF → XML automated pipeline ────────────────────────────────────────
    st.markdown("**Automated PDF Pipeline**")
    pdf_file = st.file_uploader("Upload PDF Policy Document", type=["pdf"],
                                label_visibility="collapsed", key="pdf_uploader")
    if pdf_file and st.button("⚡ Process & Ingest PDF", use_container_width=True,
                              disabled=ingest_disabled):
        if not graph:
            st.error("Neo4j not connected.")
        elif fitz is None:
            st.error("PyMuPDF is not installed. Run: pip install pymupdf")
        elif not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY is required for PDF transformation.")
        else:
            try:
                st.session_state.ingest_log = []
                _chunk_status = st.empty()

                def pdf_logger(level, msg):
                    """log_fn sink for the PDF→XML stage: record the line and live-surface chunk/err progress."""
                    st.session_state.ingest_log.append((level, msg))
                    # Surface chunk-level progress immediately so the user can
                    # see which chunk stalled without waiting for rerun.
                    if level == "info" and (
                        "chunk" in msg.lower() or "split into" in msg.lower()
                    ):
                        _chunk_status.info(msg)
                    elif level == "err":
                        _chunk_status.error(msg)

                with st.spinner("Extracting PDF → XML → Neo4j…"):
                    xml_bytes = process_pdf_to_xml(pdf_file, pdf_logger)
                    _chunk_status.empty()
                    stats = ingest_xml_to_neo4j(
                        xml_bytes, graph, pdf_logger,
                        policy_title=policy_title_input.strip(),
                        policy_version=policy_version_input.strip() or "unspecified",
                    )
                st.session_state.ingest_stats = stats
                init_chain.clear()
                st.rerun()
            except Exception as e:
                import traceback as _tb
                _full_tb = _tb.format_exc()
                st.session_state.ingest_log.append(("err", str(e)))
                st.error(f"Ingestion Aborted: {e}")
                with st.expander("Full traceback", expanded=True):
                    st.code(_full_tb, language="python")

    # ── Policy Management ───────────────────────────────────────────────────
    st.divider()
    st.markdown("**Policy Management**")

    # Always read live from the graph; surface any Neo4j error rather than
    # silently rendering "No policies ingested yet" over a populated database.
    policies = []
    policies_err = None
    if graph:
        try:
            policies = get_ingested_policies(graph)
        except Exception as _pe:
            policies_err = str(_pe)

    if not graph:
        st.caption("Neo4j not connected.")
    elif policies_err:
        st.error(f"Could not list policies: {policies_err}")
    elif not policies:
        st.caption("No policies ingested yet.")
    else:
        st.caption(f"{len(policies)} polic{'y' if len(policies)==1 else 'ies'} in graph")
        for pol in policies:
            pid     = pol.get("id") or ""
            title   = pol.get("title") or "(untitled)"
            version = pol.get("version") or "unspecified"
            rules   = int(pol.get("rules")      or 0)
            conds   = int(pol.get("conditions") or 0)
            outs    = int(pol.get("outcomes")   or 0)
            header  = f"📘 {title} · v{version}"
            with st.expander(header, expanded=False):
                st.caption(f"`{pid}`")
                st.caption(
                    f"📋 {rules} Rules · 📑 {conds} Conditions · 🎯 {outs} Outcomes"
                )
                confirm_key = f"confirm_delete_{pid}"
                confirmed = st.checkbox(
                    "I understand this action is permanent",
                    key=confirm_key,
                )
                if st.button(
                    "🗑️ Delete Policy",
                    key=f"delete_btn_{pid}",
                    use_container_width=True,
                    disabled=not confirmed,
                    type="secondary",
                ):
                    with st.spinner(f"Deleting '{title}'…"):
                        stats = delete_policy(graph, pid)
                    if stats.get("error"):
                        st.error(f"Delete failed: {stats['error']}")
                    else:
                        st.success(
                            f"Removed policy + {stats.get('members', 0)} member nodes."
                        )
                        # If the deleted policy was still in the reasoning
                        # scope, drop that too so the view re-renders cleanly.
                        active = st.session_state.get("last_policy_ids") or set()
                        if pid in active:
                            st.session_state.last_reasoning = None
                            st.session_state.last_policy_ids = active - {pid}
                        init_chain.clear()
                        st.rerun()

    # ── 🏆 RAGas Benchmarking ───────────────────────────────────────────────
    st.divider()
    st.markdown("**🏆 RAGas Benchmarking**")
    if not EVAL_DATASET:
        st.caption(
            "Gold dataset (evaluation_dataset.py) not found — benchmarking disabled."
        )
    elif not chain:
        st.caption("GraphRAG chain not ready — benchmarking disabled.")
    else:
        # Build a friendly label for each test case so the user can see at a
        # glance what they're picking. Falls back to the raw question.
        cat_labels = {"A": "Reasoning", "B": "Conflict",
                      "C": "Factual",   "D": "Edge-case"}
        options = {
            f"{tc['id']} · {cat_labels.get(tc['category'], tc['category'])}: "
            f"{(tc['question'] or '')[:60]}…": tc
            for tc in EVAL_DATASET
        }
        choice = st.selectbox(
            "Gold question",
            options=list(options.keys()),
            key="benchmark_pick",
        )
        if st.button("▶ Run Benchmark", use_container_width=True,
                     key="run_benchmark_btn"):
            with st.spinner("Running both pipelines + scoring…"):
                st.session_state.benchmark_result = benchmark_question(
                    graph, chain, options[choice],
                )
            st.rerun()
        if st.session_state.get("benchmark_result"):
            if st.button("🧹 Clear benchmark", use_container_width=True):
                st.session_state.benchmark_result = None
                st.rerun()

    # ── Graph View Toggle ───────────────────────────────────────────────────
    st.divider()
    st.markdown("**Graph Visualisation**")
    st.session_state.show_full_graph = st.checkbox(
        "Show full Knowledge Graph", value=st.session_state.show_full_graph,
    )

    # Layout toggle — applies to both Full Graph and Reasoning View
    layout_choice = st.radio(
        "Layout",
        ["🌐 Organic", "🌳 Hierarchical"],
        index=0 if st.session_state.layout_mode == "organic" else 1,
        horizontal=True,
        key="layout_choice_radio",
        help=(
            "Organic = force-directed (good for exploring relationships). "
            "Hierarchical = top-down tree (Policy ▸ Rule ▸ Condition / Outcome)."
        ),
    )
    st.session_state.layout_mode = (
        "hierarchical" if layout_choice.startswith("🌳") else "organic"
    )

    # Top-K detail level for the Reasoning View
    st.session_state.top_k_reasoning = st.slider(
        "🔍 Detail Level (Top-K)",
        min_value=1, max_value=12,
        value=int(st.session_state.get("top_k_reasoning", 5)),
        step=1,
        help=(
            "How many cited nodes to anchor the Reasoning View on. The "
            "highest-scoring nodes are kept first; their 1-hop neighbours "
            "(Conditions, Outcomes, Conflict partners) come along for context."
        ),
    )

    # ── Clear chat ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_reasoning = None
        st.session_state.last_policy_ids = set()
        st.rerun()

    st.caption("Allen Sharafzad · MSc Data Science & AI · BU 2025")


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN AREA — Trust & Live Analytics Dashboard
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — main panel (top-level script)
#   Renders the chat input, runs run_query() (§7), and lays out the answer
#   alongside its transparency surface: the trust dashboard (§7.5), the
#   plain-English narration, the CRAG/route audit, the semantic-score chart, and
#   the Reasoning View built from fetch_reasoning_subgraph() (§6).
#
# EXTENDING THIS SECTION
#   • New answer-side panel: read from the run_query() result dict (its keys are
#     documented on that function) and render below the existing dashboard.
#   • Keep heavy work inside run_query / the cached resources; this block should
#     stay presentation-only so a rerun is cheap.
# ─────────────────────────────────────────────────────────────────────────────
import ast as _ast

st.title("🎓 BU Policy GraphRAG Platform")
st.caption(
    "Transparent Knowledge Management & Reasoning · "
    "Adaptive RAG · Fusion RAG · RRF · CRAG · Every answer traceable to the graph"
)
st.divider()

# ── Live Metrics Header Row ───────────────────────────────────────────────────
_last_meta = next(
    (m.get("meta", {}) for m in reversed(st.session_state.messages)
     if m.get("role") == "assistant"),
    {},
)
_last_route    = _last_meta.get("route", "")
_last_crag     = _last_meta.get("crag_status", "")
_route_display = {
    "DIRECT_LOOKUP":   "⚡ Direct Lookup",
    "COMPLEX_REASONING": "🧠 Complex Reasoning",
}.get(_last_route, "— Awaiting first query")
_crag_display  = {
    "RELEVANT":   "🟢 Relevant",
    "AMBIGUOUS":  "🟡 Ambiguous",
    "IRRELEVANT": "🔴 Irrelevant",
}.get(_last_crag, "— Awaiting first query")

hc1, hc2, hc3 = st.columns(3)
with hc1:
    if graph:
        st.metric("🗄️ Knowledge Graph", "🟢 Connected", f"{total_nodes} nodes indexed")
    else:
        st.metric("🗄️ Knowledge Graph", "🔴 Offline",
                  (neo4j_err or "Check .env")[:45])
with hc2:
    st.metric("🔀 Adaptive Route", _route_display)
with hc3:
    _ragas_label = "🎯 Target: 80%+ Accuracy"
    _ragas_delta = "Benchmark dataset loaded" if EVAL_DATASET else "Load evaluation_dataset.py"
    st.metric("📊 RAGas Baseline", _ragas_label, _ragas_delta)

st.divider()

# ── Ingestion Process Log ────────────────────────────────────────────────────
if st.session_state.ingest_log:
    with st.expander("📜 Ingestion Process Log", expanded=True):
        level_prefix = {"info": "ℹ️", "success": "✅", "warn": "⚠️", "err": "❌"}
        log_lines = [
            f"{level_prefix.get(lvl, '•')} {msg}"
            for lvl, msg in st.session_state.ingest_log
        ]
        st.code("\n".join(log_lines), language=None)

        if st.session_state.ingest_stats:
            s = st.session_state.ingest_stats
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Rules", s["rules"])
            ic2.metric("Conditions", s["conditions"])
            ic3.metric("Outcomes", s["outcomes"])
            ic4.metric("Conflicts", s["conflicts"])


# ── Full Graph View ──────────────────────────────────────────────────────────
if st.session_state.show_full_graph and graph:
    layout_mode = st.session_state.get("layout_mode", "organic")
    layout_label = "Hierarchical" if layout_mode == "hierarchical" else "Organic"
    st.subheader(f"🌐 Full Knowledge Graph · {layout_label} layout")
    render_legend()
    fg_query = st.text_input(
        "🔎 Search nodes",
        value=st.session_state.get("full_graph_search", ""),
        placeholder="Type a keyword — matching nodes will glow amber",
        key="full_graph_search",
    )
    nodes, edges = fetch_full_graph(graph)
    if nodes:
        hits, total = decorate_with_search(nodes, fg_query)
        agraph(
            nodes=nodes, edges=edges,
            config=make_graph_config(layout_mode, height=720),
        )
        if fg_query.strip():
            st.caption(
                f"🔎 {hits} of {total} nodes match `{fg_query}` — "
                f"non-matches are dimmed."
            )
        else:
            st.caption(f"Rendering {total} nodes and {len(edges)} relationships.")
    else:
        st.info("Graph is empty — upload an XML policy file to populate it.")


# ── Chat History ─────────────────────────────────────────────────────────────
st.subheader("💬 Policy Chat")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            meta = msg.get("meta", {})
            if not meta:
                continue

            # ── Trust Dashboard ───────────────────────────────────────
            render_trust_dashboard(
                meta.get("raw", []),
                meta.get("cypher", "N/A"),
                meta.get("used_fallback", False),
            )

            # ── Adaptive Route badge ──────────────────────────────────
            _route     = meta.get("route", "COMPLEX_REASONING")
            _route_rsn = meta.get("route_reason", "")
            if _route == "DIRECT_LOOKUP":
                st.info(f"⚡ **Adaptive Route: Direct Lookup** — {_route_rsn}")
            else:
                st.markdown(
                    f'<span style="background:#4c1d95;color:#c4b5fd;'
                    f'padding:3px 12px;border-radius:14px;font-size:0.82rem;'
                    f'font-weight:600;">🧠 Adaptive Route: Complex Reasoning</span>'
                    f' <span style="font-size:0.8rem;color:#9ca3af;">'
                    f'— {_route_rsn}</span>',
                    unsafe_allow_html=True,
                )

            # ── Plain-English Narrator ────────────────────────────────
            narration = meta.get("narration") or "We checked the policy graph for you."
            st.caption(f"💬 {narration}")

            # ── Trust & Audit Trail expander ──────────────────────────
            with st.expander("🔍 Trust & Audit Trail"):

                # 1. CRAG Quality Gate (top, most prominent)
                crag_st = meta.get("crag_status", "RELEVANT")
                crag_rs = meta.get("crag_reason", "")
                if crag_st == "RELEVANT":
                    st.success(f"🟢 **CRAG Quality Gate: Relevant** — {crag_rs}")
                elif crag_st == "AMBIGUOUS":
                    st.warning(f"🟡 **CRAG Quality Gate: Ambiguous** — {crag_rs}")
                else:
                    st.error(f"🔴 **CRAG Quality Gate: Irrelevant** — {crag_rs}")

                st.divider()

                # 2. Fusion RAG query variants
                scores    = meta.get("semantic_scores") or []
                has_rrf   = any("rrf" in s for s in scores)
                cypher_hdr = meta.get("cypher", "")
                variant_lines: list = []
                if has_rrf:
                    for _line in cypher_hdr.splitlines():
                        if _line.startswith("-- queries:"):
                            try:
                                variant_lines = _ast.literal_eval(
                                    _line.replace("-- queries:", "").strip()
                                )
                            except Exception:
                                pass
                            break

                if variant_lines:
                    with st.expander(
                        f"🔀 Fusion RAG — {len(variant_lines)} Query Variants",
                        expanded=False,
                    ):
                        for _i, _vq in enumerate(variant_lines, 1):
                            st.markdown(f"**{_i}.** {_vq}")

                # 3. RRF score bar chart
                if scores:
                    _x_field  = "rrf:Q" if has_rrf else "cosine:Q"
                    _x_title  = ("RRF score — node agreement across query variants"
                                 if has_rrf else "Cosine similarity")
                    _sort_col = "rrf" if has_rrf else "cosine"
                    _tip      = (["node", "rrf", "cosine", "kept"]
                                 if has_rrf else ["node", "cosine", "kept"])
                    _df = pd.DataFrame([
                        {
                            "node":   f"{s['label']} · {s['id']}",
                            "cosine": round(s["score"], 4),
                            "rrf":    round(s.get("rrf", 0.0), 5),
                            "kept":   "✓ kept" if s["kept"] else "✗ filtered",
                        }
                        for s in scores
                    ])
                    st.markdown(
                        f"**{'Fusion RAG RRF' if has_rrf else 'Semantic'} "
                        f"Retrieval Scores** — cosine threshold "
                        f"`{SEMANTIC_THRESHOLD:.2f}`"
                    )
                    _bars = (
                        alt.Chart(_df)
                        .mark_bar(cornerRadius=3)
                        .encode(
                            x=alt.X(_x_field, title=_x_title),
                            y=alt.Y(
                                "node:N",
                                title=None,
                                sort=alt.SortField(_sort_col, order="descending"),
                            ),
                            color=alt.Color(
                                "kept:N",
                                title="Threshold",
                                scale=alt.Scale(
                                    domain=["✓ kept", "✗ filtered"],
                                    range=["#22d3ee", "#64748b"],
                                ),
                                legend=alt.Legend(orient="bottom"),
                            ),
                            tooltip=_tip,
                        )
                        .properties(height=max(180, 26 * len(_df)))
                    )
                    st.altair_chart(_bars, use_container_width=True)
                    _n_kept = sum(1 for s in scores if s["kept"])
                    st.caption(
                        f"{len(scores)} candidates evaluated · "
                        f"{_n_kept} above threshold (kept) · "
                        f"{len(scores) - _n_kept} filtered out"
                        + (" · RRF-ranked across all query variants." if has_rrf else ".")
                    )

                st.divider()

                # 4. Generated Cypher
                st.markdown("**Generated Cypher Query**")
                st.code(meta.get("cypher", "N/A"), language="cypher")

                # 5. Raw Neo4j nodes (tabular — easier to read than JSON)
                st.markdown("**Retrieved Policy Nodes from Neo4j**")
                _raw = meta.get("raw", [])
                if isinstance(_raw, list) and _raw:
                    try:
                        st.dataframe(
                            pd.DataFrame(_raw[:20]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    except Exception:
                        st.json(_raw[:10])
                else:
                    st.caption("_(no rows returned)_")


# ── 🏆 Benchmark Results ─────────────────────────────────────────────────────
if st.session_state.get("benchmark_result"):
    res = st.session_state.benchmark_result
    st.subheader("🏆 RAGas Benchmark — Gold Dataset Comparison")

    cat_labels = {"A": "Reasoning", "B": "Conflict",
                  "C": "Factual",   "D": "Edge-case"}
    st.caption(
        f"**{res['id']}** · {cat_labels.get(res['category'], res['category'])} · "
        f"Expected nodes: "
        + (", ".join(f"`{x}`" for x in res["expected_ids"]) or "_none (out-of-scope)_")
    )
    st.markdown(f"**Question:** {res['question']}")
    with st.expander("📜 Ground-truth answer"):
        st.markdown(res["ground_truth"])
    with st.expander("🤖 GraphRAG answer"):
        st.markdown(res["graphrag"]["answer"] or "_(empty)_")

    # ── Comparison table ────────────────────────────────────────────────────
    gr = res["graphrag"]
    sr = res["simplerag"]

    def _delta(a, b):
        """Format the signed GraphRAG−SimpleRAG metric gap (e.g. "+0.250"), or "—"."""
        try:
            d = float(a) - float(b)
            return f"{'+' if d >= 0 else ''}{d:.3f}"
        except (TypeError, ValueError):
            return "—"

    rows = [
        {
            "Metric":      "Context Precision",
            "Simple RAG":  f"{sr['context_precision']:.3f}",
            "GraphRAG":    f"{gr['context_precision']:.3f}",
            "Δ (G − S)":   _delta(gr['context_precision'], sr['context_precision']),
            "What it measures": "Of retrieved nodes, how many are relevant.",
        },
        {
            "Metric":      "Context Recall",
            "Simple RAG":  f"{sr['context_recall']:.3f}",
            "GraphRAG":    f"{gr['context_recall']:.3f}",
            "Δ (G − S)":   _delta(gr['context_recall'], sr['context_recall']),
            "What it measures":
                "Of expected nodes, how many were actually retrieved.",
        },
        {
            "Metric":      "Path Accuracy",
            "Simple RAG":  "0.000  (no traversal)",
            "GraphRAG":    f"{gr['path_accuracy']:.3f}",
            "Δ (G − S)":   _delta(gr['path_accuracy'], 0.0),
            "What it measures":
                "Did the Cypher traverse the expected edge types? (graph-only).",
        },
        {
            "Metric":      "Faithfulness",
            "Simple RAG":  "—  (no answer)",
            "GraphRAG":    f"{gr['faithfulness']:.3f}",
            "Δ (G − S)":   "—",
            "What it measures":
                "Are the answer's claims grounded in the retrieved context?",
        },
        {
            "Metric":      "Nodes retrieved",
            "Simple RAG":  str(sr["n_retrieved"]),
            "GraphRAG":    str(gr["n_retrieved"]),
            "Δ (G − S)":   str(gr["n_retrieved"] - sr["n_retrieved"]),
            "What it measures": "Pure count — context for the other rows.",
        },
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # ── Why GraphRAG wins on Context Recall ────────────────────────────────
    expected = set(res["expected_ids"])
    only_graph  = set(gr["retrieved_ids"]) & expected - set(sr["retrieved_ids"])
    only_simple = set(sr["retrieved_ids"]) & expected - set(gr["retrieved_ids"])
    common      = set(gr["retrieved_ids"]) & set(sr["retrieved_ids"]) & expected

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Both pipelines found",     len(common))
    c2.metric("🔷 Only GraphRAG found",      len(only_graph))
    c3.metric("🟧 Only Simple RAG found",    len(only_simple))

    if only_graph:
        st.caption(
            "**Why GraphRAG wins on recall:** the structured Cypher follows "
            "`HAS_CONDITION → HAS_OUTCOME → CONFLICTS_WITH` edges from the "
            "matched Rules, picking up nodes that flat similarity search "
            "missed: " + ", ".join(f"`{x}`" for x in sorted(only_graph)) + "."
        )

    st.divider()


# ── Reasoning Subgraph (last answer) ─────────────────────────────────────────
if graph and st.session_state.messages:
    st.subheader("🧠 Reasoning View — nodes the AI used")

    ids = st.session_state.last_reasoning or set()

    # STRICT FILTER: only show the graph when the LLM's answer explicitly
    # named one or more node IDs. Otherwise, inform the user instead of
    # falling back to the full graph.
    if not ids:
        st.info(
            "ℹ️ The last answer did not reference any specific policy node IDs, "
            "so there is nothing to visualise here. Ask a question that maps to "
            "a concrete Rule / Condition / Outcome to see the reasoning graph."
        )
    else:
        render_legend(highlight_path=True)
        active_pids = st.session_state.get("last_policy_ids") or set()
        if active_pids:
            st.caption(
                "🔒 Scope: "
                + ", ".join(f"`{p}`" for p in sorted(active_pids))
            )

        # Score map for opacity + top-K ranking (from the latest assistant
        # message's semantic_scores, if present)
        last_meta = next(
            (m.get("meta", {}) for m in reversed(st.session_state.messages)
             if m.get("role") == "assistant"),
            {},
        )
        score_list = last_meta.get("semantic_scores") or []
        # Use RRF score for node ranking when Fusion RAG was active; cosine otherwise.
        score_map = {
            s["id"]: (s["rrf"] if "rrf" in s else s["score"])
            for s in score_list if s.get("kept")
        }
        top_k = int(st.session_state.get("top_k_reasoning", 5))

        r_nodes, r_edges = fetch_reasoning_subgraph(
            graph, ids, policy_ids=active_pids,
            top_k=top_k, score_map=score_map,
        )
        kept_ids = {n.id for n in r_nodes}
        r_edges = [e for e in r_edges
                   if e.source in kept_ids and e.to in kept_ids]

        if not r_nodes:
            st.info(
                f"ℹ️ The answer mentioned {sorted(ids)}, but none of those "
                "IDs exist within the active policy scope. Try re-ingesting "
                "your XML or ask a broader question."
            )
        else:
            layout_mode = st.session_state.get("layout_mode", "organic")
            rv_query = st.text_input(
                "🔎 Search within reasoning view",
                value=st.session_state.get("reasoning_search", ""),
                placeholder="Filter the reasoning subgraph in real time",
                key="reasoning_search",
            )
            hits, total = decorate_with_search(r_nodes, rv_query)
            selected = agraph(
                nodes=r_nodes, edges=r_edges,
                config=make_graph_config(layout_mode, height=620),
            )
            if selected:
                st.session_state.last_selected_node = selected

            n_total_cited = len(ids)
            n_shown_cited = sum(1 for n in r_nodes if n.id in ids)
            base_caption = (
                f"Showing top **{n_shown_cited}** of {n_total_cited} cited "
                f"nodes (Detail Level = {top_k}) plus their 1-hop "
                f"neighbours. Cyan dashed edges trace the AI's reasoning flow."
            )
            if rv_query.strip():
                st.caption(
                    f"🔎 {hits} of {total} nodes match `{rv_query}` — "
                    f"non-matches are dimmed.  ·  {base_caption}"
                )
            else:
                st.caption(base_caption)

            # ── Selected Node Detail panel ──────────────────────────────
            sel = st.session_state.get("last_selected_node")
            st.markdown(
                '<div id="selected-node-detail-anchor"></div>',
                unsafe_allow_html=True,
            )
            if sel:
                # Look up the node by id in the rendered list
                node_obj = next(
                    (n for n in r_nodes if getattr(n, "id", None) == sel),
                    None,
                )
                if node_obj is not None:
                    label   = getattr(node_obj, "label", sel) or sel
                    tooltip = getattr(node_obj, "title", "") or ""
                    score   = score_map.get(sel)
                    score_line = (
                        f" · Similarity: **{score:.3f}**"
                        if isinstance(score, (int, float)) else ""
                    )
                    st.success(
                        f"### 🔎 Selected Node Detail\n\n"
                        f"**{label.replace(chr(10), ' · ')}**{score_line}\n\n"
                        f"```\n{tooltip}\n```"
                    )
                    if st.button("Clear selection",
                                  key="clear_selected_node_btn"):
                        st.session_state.last_selected_node = None
                        st.rerun()
                    # Smooth-scroll the parent page to the detail anchor.
                    # Wrapped in try/parent.* so a sandboxed iframe won't
                    # crash — it just no-ops.
                    st.components.v1.html(
                        """
                        <script>
                        try {
                          const t = window.parent.document
                            .getElementById('selected-node-detail-anchor');
                          if (t) {
                            t.scrollIntoView({behavior:'smooth', block:'start'});
                          }
                        } catch (e) { /* iframe sandbox — no-op */ }
                        </script>
                        """,
                        height=0,
                    )
            else:
                st.caption(
                    "💡 Click any node in the graph above to see its full "
                    "description, source section and similarity score here."
                )


# ── Chat Input ───────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about BU's research degree policies...")
if question:
    question = _clean_user_query(question)   # isolate before anything touches it
if question:
    if not chain:
        st.warning("The GraphRAG chain is not ready. Check the sidebar status.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Traversing knowledge graph..."):
        r = run_query(question, chain)

    # Compute plain-English narration once; fall back safely if the API errors.
    try:
        narration = narrate_path(r["raw"])
    except Exception:
        narration = "We checked the policy graph for you."

    st.session_state.messages.append({
        "role": "assistant",
        "content": r["answer"],
        "meta": {
            "cypher":           r["cypher"],
            "raw":              r["raw"],
            "used_fallback":    r.get("used_fallback", False),
            "narration":        narration,
            "policy_ids":       sorted(r.get("policy_ids") or set()),
            "semantic_scores":  r.get("semantic_scores", []),
            "crag_status":      r.get("crag_status", "RELEVANT"),
            "crag_reason":      r.get("crag_reason", "Not evaluated"),
            "route":            r.get("route", "COMPLEX_REASONING"),
            "route_reason":     r.get("route_reason", "Not evaluated"),
            "timestamp":        datetime.now().strftime("%H:%M:%S"),
        },
    })
    st.session_state.last_reasoning = r["node_ids"]
    st.session_state.last_policy_ids = r.get("policy_ids") or set()
    st.rerun()
