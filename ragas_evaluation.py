"""
Ragas Evaluation Pipeline for GraphRAG Policy Bot
==================================================
Bournemouth University — Code of Practice for Research Degrees 2024-25

Stage 2: Ragas metrics  (Faithfulness, Answer Relevance, Context Precision)
Stage 3: Graph-specific  (Retrieval Depth, Path Accuracy, Conflict Detection Rate)
+ Comparison:            Standard Vector RAG  vs  GraphRAG

Usage:
    pip install ragas datasets langchain-openai langchain-neo4j python-dotenv tabulate
    python ragas_evaluation.py              # run full evaluation
    python ragas_evaluation.py --quick      # first 5 questions only
    python ragas_evaluation.py --category B # only Conflict questions

Output:
    evaluation_results.json   — raw per-question scores
    evaluation_summary.csv    — paper-ready aggregated table
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
NEO4J_URI       = os.getenv("NEO4J_URI")
NEO4J_USERNAME  = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE  = os.getenv("NEO4J_DATABASE", "neo4j")

assert OPENAI_API_KEY, "OPENAI_API_KEY missing from .env"
assert NEO4J_URI,      "NEO4J_URI missing from .env"

OUTPUT_DIR = Path("evaluation_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONNECT TO NEO4J + BUILD CHAIN  (reuse app.py logic)
# ─────────────────────────────────────────────────────────────────────────────
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

print("[1/6] Connecting to Neo4j AuraDB …")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)
graph.refresh_schema()
print(f"       Schema loaded — {len(graph.schema or '')} chars")

# Import prompts from app.py so we evaluate the REAL system
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import (
    CYPHER_GENERATION_TEMPLATE,
    QA_SYSTEM_TEMPLATE,
    extract_ids_from_text,
    extract_sections_from_text,
    resolve_section_refs_to_ids,
    keyword_fallback_search,
    _ground_answer_from_rows,
)

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_TEMPLATE),
    ("human", "{question}"),
])

llm_cypher = ChatOpenAI(model="gpt-4o",   temperature=0,   api_key=OPENAI_API_KEY)
llm_qa     = ChatOpenAI(model="gpt-4o",   temperature=0.1, api_key=OPENAI_API_KEY)

chain = GraphCypherQAChain.from_llm(
    llm=llm_cypher,
    qa_llm=llm_qa,
    graph=graph,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    verbose=False,
    return_intermediate_steps=True,
    validate_cypher=True,
    allow_dangerous_requests=True,
    top_k=25,
)
print("       GraphRAG chain ready.")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RUN EACH QUESTION THROUGH THE BOT  (captures intermediates)
# ─────────────────────────────────────────────────────────────────────────────
from evaluation_dataset import EVAL_DATASET


def run_graphrag_query(question: str) -> dict:
    """Run a question through the GraphRAG chain (mirrors app.py run_query)."""
    cypher = "N/A"
    raw = []
    try:
        result = chain.invoke({"query": question})
        steps  = result.get("intermediate_steps", [])
        cypher = steps[0].get("query", "N/A") if steps else "N/A"
        raw    = steps[1].get("context", []) if len(steps) > 1 else []
        answer = result["result"]

        # Fallback
        empty_result = (not raw) or (isinstance(raw, list) and len(raw) == 0)
        saying_unavailable = "not available in the current policy graph" in answer.lower()
        if empty_result or saying_unavailable:
            fb_cypher, fb_rows, _scores = keyword_fallback_search(graph, question)
            if fb_rows:
                answer = _ground_answer_from_rows(question, fb_rows)
                cypher = f"-- fallback --\n{fb_cypher}"
                raw    = fb_rows

        # IDs mentioned in the answer
        answer_ids    = extract_ids_from_text(answer)
        section_refs  = extract_sections_from_text(answer)
        section_ids   = resolve_section_refs_to_ids(graph, section_refs)
        node_ids      = answer_ids | section_ids

        return {
            "answer": answer,
            "cypher": cypher,
            "raw_context": raw,
            "node_ids": node_ids,
            "error": None,
        }
    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "cypher": cypher,
            "raw_context": raw,
            "node_ids": set(),
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RAGAS METRICS  (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────
# We compute these manually so the script works even without the full ragas
# library installed.  If ragas IS installed, we also run its native pipeline
# and include those scores for comparison.

def _faithfulness_score(answer: str, context: list) -> float:
    """
    Faithfulness = fraction of claims in the answer that are grounded in context.
    Uses an LLM-as-judge approach.
    """
    if not answer or not context:
        return 0.0

    judge = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    context_str = json.dumps(context[:15], default=str)[:6000]

    prompt = f"""You are an evaluation judge. Given an ANSWER and the CONTEXT it was generated from,
identify every factual claim in the ANSWER, then determine how many are supported by the CONTEXT.

CONTEXT:
{context_str}

ANSWER:
{answer}

Respond ONLY with a JSON object:
{{"total_claims": <int>, "supported_claims": <int>, "score": <float 0-1>}}"""

    try:
        resp = judge.invoke(prompt).content
        data = json.loads(resp.strip().removeprefix("```json").removesuffix("```").strip())
        return float(data.get("score", 0.0))
    except Exception:
        return 0.0


def _answer_relevance_score(question: str, answer: str) -> float:
    """
    Answer Relevance = how directly the answer addresses the question.
    LLM-as-judge on a 0-1 scale.
    """
    if not answer or "not available" in answer.lower():
        return 0.0

    judge = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

    prompt = f"""You are an evaluation judge. Rate how directly and completely the ANSWER addresses
the QUESTION. Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant and complete).

QUESTION: {question}

ANSWER: {answer}

Respond ONLY with a JSON object: {{"score": <float 0-1>, "reason": "<brief>"}}"""

    try:
        resp = judge.invoke(prompt).content
        data = json.loads(resp.strip().removeprefix("```json").removesuffix("```").strip())
        return float(data.get("score", 0.0))
    except Exception:
        return 0.0


def _context_precision_score(expected_ids: set, retrieved_ids: set) -> float:
    """
    Context Precision = |expected ∩ retrieved| / |retrieved|
    Measures whether retrieved nodes are actually relevant (no noise).
    Returns 1.0 if both sets are empty (question has no expected context).
    """
    if not expected_ids and not retrieved_ids:
        return 1.0
    if not retrieved_ids:
        return 0.0
    overlap = expected_ids & retrieved_ids
    return len(overlap) / len(retrieved_ids)


def _context_recall_score(expected_ids: set, retrieved_ids: set) -> float:
    """
    Context Recall = |expected ∩ retrieved| / |expected|
    Measures whether all required nodes were retrieved.
    """
    if not expected_ids:
        return 1.0
    overlap = expected_ids & retrieved_ids
    return len(overlap) / len(expected_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRAPH-SPECIFIC METRICS  (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

def _retrieval_depth(cypher: str) -> int:
    """
    Retrieval Depth = number of MATCH / OPTIONAL MATCH clauses in the Cypher.
    Approximates the number of hops traversed.
    Each MATCH adds at least one hop.
    """
    if not cypher or cypher == "N/A":
        return 0
    matches = re.findall(r"(?:OPTIONAL\s+)?MATCH", cypher, re.IGNORECASE)
    return len(matches)


def _path_accuracy(cypher: str, expected_edges: list) -> float:
    """
    Path Accuracy = fraction of expected edge types that appear in the Cypher.
    Measures whether the bot traversed the correct relationships.
    """
    if not expected_edges:
        return 1.0
    if not cypher or cypher == "N/A":
        return 0.0
    cypher_upper = cypher.upper()
    found = sum(1 for e in expected_edges if e.upper() in cypher_upper)
    return found / len(expected_edges)


def _conflict_detection_rate(answer: str, expected_edges: list) -> float:
    """
    Conflict Detection Rate — for questions where CONFLICTS_WITH is expected:
      1.0 if the answer contains the exact conflict detection phrase
      0.5 if the answer mentions 'conflict' at all
      0.0 otherwise
    Returns 1.0 for questions with no expected conflicts (N/A).
    """
    if "CONFLICTS_WITH" not in expected_edges:
        return 1.0  # Not applicable
    answer_lower = answer.lower()
    if "a policy conflict has been detected between" in answer_lower:
        return 1.0
    if "conflict" in answer_lower:
        return 0.5
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OPTIONAL: NATIVE RAGAS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def try_ragas_native(results: list) -> dict | None:
    """
    If `ragas` is installed, run the native evaluation for comparison.
    Returns a dict of metric_name → mean_score, or None if unavailable.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print("       [ragas not installed — skipping native pipeline]")
        return None

    # Build the HuggingFace Dataset that ragas expects
    data = {
        "question":      [],
        "answer":        [],
        "contexts":      [],
        "ground_truth":  [],
    }
    for r in results:
        data["question"].append(r["question"])
        data["answer"].append(r["bot_answer"])
        # ragas expects contexts as list[str]
        ctx = r.get("raw_context", [])
        ctx_strings = [json.dumps(c, default=str) if not isinstance(c, str) else c
                       for c in (ctx or [])]
        data["contexts"].append(ctx_strings if ctx_strings else ["No context retrieved."])
        data["ground_truth"].append(r["ground_truth"])

    ds = Dataset.from_dict(data)

    try:
        ragas_result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return dict(ragas_result)
    except Exception as e:
        print(f"       [ragas native evaluation failed: {e}]")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VECTOR RAG BASELINE  (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def run_vector_rag_query(question: str) -> dict:
    """
    Simulated Standard Vector RAG baseline.
    Uses keyword_fallback_search (flat text search, no graph traversal)
    to approximate what a vector-only retriever would return.
    This gives a fair comparison: same data, no graph structure.
    """
    try:
        fb_cypher, fb_rows = keyword_fallback_search(graph, question)
        if fb_rows:
            answer = _ground_answer_from_rows(question, fb_rows)
            # Vector RAG has no graph structure → no meaningful IDs
            answer_ids = extract_ids_from_text(answer)
            return {
                "answer": answer,
                "cypher": fb_cypher or "N/A",
                "raw_context": fb_rows,
                "node_ids": answer_ids,
                "error": None,
            }
        return {
            "answer": "This information is not available.",
            "cypher": "N/A",
            "raw_context": [],
            "node_ids": set(),
            "error": None,
        }
    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "cypher": "N/A",
            "raw_context": [],
            "node_ids": set(),
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(dataset: list, run_baseline: bool = True) -> tuple[list, list]:
    """
    Run every question through both pipelines and compute all metrics.
    Returns (graphrag_results, vectorrag_results).
    """
    graphrag_results = []
    vectorrag_results = []

    total = len(dataset)
    for i, tc in enumerate(dataset, 1):
        qid = tc["id"]
        question = tc["question"]
        expected_ids = tc["expected_node_ids"]
        expected_edges = tc["expected_edges"]
        ground_truth = tc["ground_truth"]

        print(f"\n[{i}/{total}] {qid}: {question[:70]}…")

        # ── GraphRAG ────────────────────────────────────────────────────────
        print(f"       Running GraphRAG …")
        t0 = time.time()
        gr = run_graphrag_query(question)
        gr_time = time.time() - t0

        retrieved_ids = gr["node_ids"]

        gr_scores = {
            "faithfulness":          _faithfulness_score(gr["answer"], gr["raw_context"]),
            "answer_relevance":      _answer_relevance_score(question, gr["answer"]),
            "context_precision":     _context_precision_score(expected_ids, retrieved_ids),
            "context_recall":        _context_recall_score(expected_ids, retrieved_ids),
            "retrieval_depth":       _retrieval_depth(gr["cypher"]),
            "path_accuracy":         _path_accuracy(gr["cypher"], expected_edges),
            "conflict_detection":    _conflict_detection_rate(gr["answer"], expected_edges),
        }

        graphrag_results.append({
            "id": qid,
            "category": tc["category"],
            "question": question,
            "ground_truth": ground_truth,
            "bot_answer": gr["answer"],
            "cypher": gr["cypher"],
            "raw_context": gr["raw_context"],
            "expected_ids": sorted(expected_ids),
            "retrieved_ids": sorted(retrieved_ids),
            "scores": gr_scores,
            "latency_s": round(gr_time, 2),
            "error": gr["error"],
        })

        print(f"       GraphRAG scores: F={gr_scores['faithfulness']:.2f}  "
              f"AR={gr_scores['answer_relevance']:.2f}  "
              f"CP={gr_scores['context_precision']:.2f}  "
              f"CR={gr_scores['context_recall']:.2f}  "
              f"PA={gr_scores['path_accuracy']:.2f}  "
              f"CD={gr_scores['conflict_detection']:.2f}  "
              f"({gr_time:.1f}s)")

        # ── Vector RAG Baseline ─────────────────────────────────────────────
        if run_baseline:
            print(f"       Running Vector RAG baseline …")
            t0 = time.time()
            vr = run_vector_rag_query(question)
            vr_time = time.time() - t0

            vr_retrieved = vr["node_ids"]
            vr_scores = {
                "faithfulness":          _faithfulness_score(vr["answer"], vr["raw_context"]),
                "answer_relevance":      _answer_relevance_score(question, vr["answer"]),
                "context_precision":     _context_precision_score(expected_ids, vr_retrieved),
                "context_recall":        _context_recall_score(expected_ids, vr_retrieved),
                "retrieval_depth":       _retrieval_depth(vr["cypher"]),
                "path_accuracy":         _path_accuracy(vr["cypher"], expected_edges),
                "conflict_detection":    _conflict_detection_rate(vr["answer"], expected_edges),
            }

            vectorrag_results.append({
                "id": qid,
                "category": tc["category"],
                "question": question,
                "ground_truth": ground_truth,
                "bot_answer": vr["answer"],
                "cypher": vr.get("cypher"),
                "raw_context": vr["raw_context"],
                "expected_ids": sorted(expected_ids),
                "retrieved_ids": sorted(vr_retrieved),
                "scores": vr_scores,
                "latency_s": round(vr_time, 2),
                "error": vr["error"],
            })

            print(f"       VectorRAG scores: F={vr_scores['faithfulness']:.2f}  "
                  f"AR={vr_scores['answer_relevance']:.2f}  "
                  f"CP={vr_scores['context_precision']:.2f}  "
                  f"CR={vr_scores['context_recall']:.2f}  "
                  f"({vr_time:.1f}s)")

    return graphrag_results, vectorrag_results


# ─────────────────────────────────────────────────────────────────────────────
# 8.  AGGREGATION & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

METRIC_NAMES = [
    "faithfulness", "answer_relevance", "context_precision",
    "context_recall", "path_accuracy", "conflict_detection",
]

METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevance":  "Answer Relevance",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
    "path_accuracy":     "Path Accuracy",
    "conflict_detection":"Conflict Detection Rate",
}


def aggregate(results: list) -> dict:
    """Compute mean scores overall and per category."""
    overall = defaultdict(list)
    by_cat  = defaultdict(lambda: defaultdict(list))

    for r in results:
        cat = r["category"]
        for m in METRIC_NAMES:
            v = r["scores"].get(m, 0.0)
            overall[m].append(v)
            by_cat[cat][m].append(v)

    # latency
    latencies = [r["latency_s"] for r in results]
    depths    = [r["scores"].get("retrieval_depth", 0) for r in results]

    summary = {
        "overall": {m: round(sum(vs)/max(len(vs),1), 4) for m, vs in overall.items()},
        "by_category": {},
        "avg_latency_s": round(sum(latencies)/max(len(latencies),1), 2),
        "avg_retrieval_depth": round(sum(depths)/max(len(depths),1), 2),
        "n_questions": len(results),
    }
    for cat, metrics in sorted(by_cat.items()):
        summary["by_category"][cat] = {
            m: round(sum(vs)/max(len(vs),1), 4) for m, vs in metrics.items()
        }

    return summary


def print_table(gr_summary: dict, vr_summary: dict | None = None):
    """Print a paper-ready comparison table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    header = ["Metric", "GraphRAG"]
    if vr_summary:
        header.append("Vector RAG")
        header.append("Delta")

    rows = []
    for m in METRIC_NAMES:
        gr_val = gr_summary["overall"].get(m, 0)
        row = [METRIC_LABELS.get(m, m), f"{gr_val:.4f}"]
        if vr_summary:
            vr_val = vr_summary["overall"].get(m, 0)
            delta  = gr_val - vr_val
            row.append(f"{vr_val:.4f}")
            row.append(f"{'+' if delta >= 0 else ''}{delta:.4f}")
        rows.append(row)

    # Extra rows
    rows.append(["Avg Latency (s)", f"{gr_summary['avg_latency_s']:.2f}",
                  *([ f"{vr_summary['avg_latency_s']:.2f}", "—"]
                    if vr_summary else [])])
    rows.append(["Avg Retrieval Depth", f"{gr_summary['avg_retrieval_depth']:.1f}",
                  *([ f"{vr_summary['avg_retrieval_depth']:.1f}", "—"]
                    if vr_summary else [])])

    print("\n" + "=" * 72)
    print("  EVALUATION RESULTS — GraphRAG Policy Bot")
    print("  Bournemouth University CoP 2024-25")
    print(f"  {gr_summary['n_questions']} questions evaluated")
    print("=" * 72)

    if tabulate:
        print(tabulate(rows, headers=header, tablefmt="github", floatfmt=".4f"))
    else:
        # Simple fallback
        print(f"\n{'Metric':<28} {'GraphRAG':>10}", end="")
        if vr_summary:
            print(f" {'VectorRAG':>10} {'Delta':>10}", end="")
        print()
        print("-" * (28 + 10 + (20 if vr_summary else 0)))
        for row in rows:
            print(f"{row[0]:<28} {row[1]:>10}", end="")
            if vr_summary and len(row) > 2:
                print(f" {row[2]:>10} {row[3]:>10}", end="")
            print()

    # Per-category breakdown
    print("\n── Per-Category Breakdown (GraphRAG) ──")
    cat_labels = {"A": "Reasoning", "B": "Conflict", "C": "Factual", "D": "Edge-case"}
    for cat in sorted(gr_summary["by_category"]):
        scores = gr_summary["by_category"][cat]
        label  = cat_labels.get(cat, cat)
        parts  = [f"{METRIC_LABELS[m][:6]}={scores.get(m, 0):.2f}" for m in METRIC_NAMES]
        print(f"  {cat} ({label:>9}): {', '.join(parts)}")


def save_results(gr_results, vr_results, gr_summary, vr_summary):
    """Save raw results and summary to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw results (convert sets to lists for JSON)
    def _clean(results):
        cleaned = []
        for r in results:
            c = dict(r)
            c["expected_ids"]  = sorted(c.get("expected_ids", []))
            c["retrieved_ids"] = sorted(c.get("retrieved_ids", []))
            c["raw_context"]   = str(c.get("raw_context", ""))[:500]  # truncate
            cleaned.append(c)
        return cleaned

    raw_path = OUTPUT_DIR / f"results_raw_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "graphrag": _clean(gr_results),
            "vectorrag": _clean(vr_results),
        }, f, indent=2, default=str)
    print(f"\n  Raw results  → {raw_path}")

    # Summary
    summary_path = OUTPUT_DIR / f"results_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "graphrag": gr_summary,
            "vectorrag": vr_summary,
            "timestamp": timestamp,
        }, f, indent=2)
    print(f"  Summary      → {summary_path}")

    # CSV for the paper
    csv_path = OUTPUT_DIR / f"evaluation_table_{timestamp}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Metric,GraphRAG,VectorRAG,Delta\n")
        for m in METRIC_NAMES:
            gr_val = gr_summary["overall"].get(m, 0)
            vr_val = (vr_summary or {}).get("overall", {}).get(m, 0)
            delta  = gr_val - vr_val
            f.write(f"{METRIC_LABELS.get(m, m)},{gr_val:.4f},{vr_val:.4f},{delta:+.4f}\n")
        f.write(f"Avg Latency (s),{gr_summary['avg_latency_s']:.2f},"
                f"{(vr_summary or {}).get('avg_latency_s', 0):.2f},—\n")
        f.write(f"Avg Retrieval Depth,{gr_summary['avg_retrieval_depth']:.1f},"
                f"{(vr_summary or {}).get('avg_retrieval_depth', 0):.1f},—\n")
    print(f"  Paper table  → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  LATEX TABLE GENERATOR  (for the paper)
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(gr_summary: dict, vr_summary: dict | None) -> str:
    """Generate a LaTeX-ready table for the research paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Evaluation Results: GraphRAG vs.\ Standard Vector RAG}",
        r"\label{tab:evaluation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{GraphRAG} & \textbf{Vector RAG} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]
    for m in METRIC_NAMES:
        gr_val = gr_summary["overall"].get(m, 0)
        vr_val = (vr_summary or {}).get("overall", {}).get(m, 0)
        delta  = gr_val - vr_val
        label  = METRIC_LABELS.get(m, m)
        sign   = "+" if delta >= 0 else ""
        bold_gr = (r"\textbf{" + f"{gr_val:.3f}" + r"}") if delta > 0 else f"{gr_val:.3f}"
        bold_vr = (r"\textbf{" + f"{vr_val:.3f}" + r"}") if delta < 0 else f"{vr_val:.3f}"
        lines.append(f"{label} & {bold_gr} & {bold_vr} & {sign}{delta:.3f} \\\\")

    lines += [
        r"\midrule",
        f"Avg Latency (s) & {gr_summary['avg_latency_s']:.2f} & "
        f"{(vr_summary or {{}}).get('avg_latency_s', 0):.2f} & — \\\\",
        f"Avg Retrieval Depth & {gr_summary['avg_retrieval_depth']:.1f} & "
        f"{(vr_summary or {{}}).get('avg_retrieval_depth', 0):.1f} & — \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 10. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ragas Evaluation for GraphRAG Policy Bot")
    parser.add_argument("--quick", action="store_true", help="Run first 5 questions only")
    parser.add_argument("--category", type=str, default=None, help="Run one category: A/B/C/D")
    parser.add_argument("--no-baseline", action="store_true", help="Skip Vector RAG baseline")
    parser.add_argument("--no-ragas-native", action="store_true", help="Skip native ragas lib")
    args = parser.parse_args()

    # Filter dataset
    dataset = list(EVAL_DATASET)
    if args.category:
        dataset = [q for q in dataset if q["category"] == args.category.upper()]
        print(f"  Filtered to category {args.category.upper()}: {len(dataset)} questions")
    if args.quick:
        dataset = dataset[:5]
        print(f"  Quick mode: first {len(dataset)} questions only")

    if not dataset:
        print("No questions to evaluate.")
        return

    print(f"\n[2/6] Evaluating {len(dataset)} questions …")
    print("=" * 72)

    # ── Run both pipelines ──────────────────────────────────────────────────
    gr_results, vr_results = evaluate_all(dataset, run_baseline=not args.no_baseline)

    print(f"\n[3/6] Aggregating scores …")
    gr_summary = aggregate(gr_results)
    vr_summary = aggregate(vr_results) if vr_results else None

    # ── Native ragas (optional) ─────────────────────────────────────────────
    ragas_native = None
    if not args.no_ragas_native:
        print(f"\n[4/6] Attempting native ragas evaluation …")
        ragas_native = try_ragas_native(gr_results)
        if ragas_native:
            print(f"       Native ragas scores: {ragas_native}")
            gr_summary["ragas_native"] = ragas_native

    # ── Print table ─────────────────────────────────────────────────────────
    print(f"\n[5/6] Results:")
    print_table(gr_summary, vr_summary)

    # ── Save everything ─────────────────────────────────────────────────────
    print(f"\n[6/6] Saving outputs …")
    save_results(gr_results, vr_results, gr_summary, vr_summary)

    # LaTeX table
    latex = generate_latex_table(gr_summary, vr_summary)
    latex_path = OUTPUT_DIR / "evaluation_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  LaTeX table  → {latex_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
