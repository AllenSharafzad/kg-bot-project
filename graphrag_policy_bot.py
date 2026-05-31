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

def generate_multiple_queries(user_query: str, num_queries: int = 3) -> list[str]:
    """
    Use GPT-4o to generate `num_queries` alternative phrasings of a
    university policy question, enabling Fusion RAG retrieval diversity.

    Strategy — each variant targets a different retrieval angle:
      • Formal/legal phrasing   → catches exact policy wording in the graph
      • Student-perspective     → catches colloquial condition-node text
      • Synonym / related concept → catches semantically adjacent nodes

    Returns a list of plain-string questions (no numbering or preamble).
    Falls back to [user_query] on any failure so the pipeline always proceeds.
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
    """
    Reciprocal Rank Fusion (RRF) over multiple ranked node-ID lists.

    RRF score for item d:  RRF(d) = Σ  1 / (k + rank(d, list_i))

    where rank is 1-indexed. Items appearing in more lists AND appearing
    higher in each list accumulate higher scores.

    Args:
        ranked_lists: Each inner list is a ranking of node IDs (best first).
        k:            Smoothing constant (default 60, per the original paper).

    Returns:
        List of (node_id, rrf_score) sorted by descending score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, node_id in enumerate(ranked, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ─────────────────────────────────────────────────────────────────────────────
# 0.6  CORRECTIVE RAG (CRAG) — CONTEXT RELEVANCE VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_context_relevance(retrieved_rows: list, user_query: str) -> dict:
    """
    CRAG validation layer: judges whether the post-RRF policy context is
    sufficient and relevant to answer the user's question reliably.

    Classification:
      RELEVANT   — context directly addresses the query with specific policy detail
      IRRELEVANT — context is about a different topic or too sparse to be useful
      AMBIGUOUS  — context partially matches but is missing critical details needed

    Uses GPT-4o at temperature=0 for deterministic classification.
    Falls back to RELEVANT on any LLM or parsing failure so the pipeline is
    never blocked by the validation layer itself (fail-open contract).

    Args:
        retrieved_rows: list of Neo4j result dicts (from graph.query())
        user_query:     the original user question

    Returns:
        {"status": "RELEVANT"|"IRRELEVANT"|"AMBIGUOUS", "reason": str}
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

def route_query(user_query: str, graph=None, chain=None) -> dict:
    """
    Adaptive RAG entry-point router: classifies incoming query complexity to
    select the optimal retrieval path before any embedding or graph work begins.

    Routes:
      DIRECT_LOOKUP      — simple, single-hop, factual questions that map to
                           one policy section or one rule.  Single vector search
                           is sufficient; multi-query Fusion RAG would add cost
                           with no retrieval benefit.
                           Examples: "What is the draft submission deadline?",
                                     "How many supervisors can a FT PhD have?"

      COMPLEX_REASONING  — multi-hop, thematic, comparative, or cross-policy
                           questions that benefit from diverse query angles and
                           the full RRF + CRAG pipeline.
                           Examples: "How does the suspension policy interact
                                      with international visa restrictions?",
                                     "Compare examiner independence rules
                                      across all uploaded policies."

    Uses GPT-4o at temperature=0, max_tokens=50 for a fast, deterministic
    two-line classification.  Fail-open contract: any exception returns
    COMPLEX_REASONING so the full flagship pipeline always runs as the safe
    default.

    Returns:
        {"route": "DIRECT_LOOKUP"|"COMPLEX_REASONING", "reason": str}
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

def build_graph() -> Neo4jGraph:
    """
    Establish the connection to Neo4j AuraDB and return a Neo4jGraph object.

    Neo4jGraph does two things automatically:
      - Keeps the Bolt/Bolt+S connection alive via a connection pool
      - Calls graph.refresh_schema() to read all node labels, relationship
        types, and property keys — this schema is injected into the Cypher
        generation prompt so the LLM knows exactly what exists in the graph.
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
    """
    Assemble the GraphCypherQAChain.

    GraphCypherQAChain internally runs two steps:
      Step A: cypher_llm + CYPHER_GENERATION_PROMPT → Cypher string
      Step B: execute Cypher on Neo4j → raw graph result (list of dicts)
      Step C: qa_llm + QA_PROMPT + graph result → grounded natural language answer

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
    """
    Scenario A — Borderline Eligibility: Illness + Missed Deadline
    Traverses: R4.1 --[HAS_CONDITION]--> C4.1.2
               checks --[OVERRIDES]--> edges for mitigating rules
               checks --[ESCALATES_TO]--> chains
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
    """
    Scenario B — Conflicting Rules: BU Staff PGR + Examiner Independence
    Traverses: R5.1 --[HAS_CONDITION]--> C5.1.2 and C5.1.6
               C5.1.2 --[CONFLICTS {scenario, resolution}]--> C5.1.6
    The CONFLICTS edge carries the resolution as an edge property —
    this is what the LLM surfaces to the user.
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
    """
    Scenario C — Escalation: Failed Probationary Review + Engagement Lapse
    Traverses two simultaneous ESCALATES_TO chains:
      Chain 1: R4.1 outcome(Withdrawal) --[ESCALATES_TO]--> ...
      Chain 2: R3.1 --[ESCALATES_TO]--> R3.2 --[ESCALATES_TO]--> Withdrawal
    Returns both chains so the LLM can explain both active processes.
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
    """
    Route a question either to a pre-built scenario query (guaranteed graph
    edge coverage) or to the LLM-to-Cypher chain (open-domain queries).

    Returns a dict with keys:
      - answer         : the grounded natural language answer
      - cypher_used    : the Cypher that was executed
      - graph_result   : raw Neo4j result (for audit)
      - route          : "scenario_A|B|C" or "llm_chain"
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
    """Pretty-print a query result for development / demo purposes."""
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