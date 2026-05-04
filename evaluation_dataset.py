"""
Ground-Truth Evaluation Dataset for GraphRAG Policy Bot
========================================================
Bournemouth University — Code of Practice for Research Degrees 2024-25

20 scenario-based test cases across 4 categories:
  A. Reasoning (conditional / multi-hop)
  B. Conflict detection
  C. Factual retrieval (single-hop)
  D. Edge-case / negation

Each entry:
  - question          : Natural-language student query
  - ground_truth      : Gold-standard answer (based on the BU CoP)
  - expected_node_ids : Set of node IDs that MUST be retrieved for a perfect score
  - expected_edges    : Relationships the system should traverse
  - category          : A | B | C | D
  - min_hops          : Minimum graph traversal depth needed
"""

EVAL_DATASET = [
    # ── A. REASONING (conditional / multi-hop) ─────────────────────────────────
    {
        "id": "Q01",
        "category": "A",
        "question": (
            "If a full-time PhD student fails their Probationary Review, "
            "what are the possible outcomes and deadlines?"
        ),
        "ground_truth": (
            "If a full-time PhD student fails their Probationary Review, "
            "the outcomes may include a requirement to re-submit within a specified "
            "deadline, transfer to an MPhil programme, or withdrawal from the programme. "
            "The specific deadlines for full-time students are defined in the relevant "
            "conditions of the Probationary Review rule."
        ),
        "expected_node_ids": {"R4.1", "C4.1.1", "C4.1.2", "O4.1.1", "O4.1.2"},
        "expected_edges": ["HAS_CONDITION", "HAS_OUTCOME"],
        "min_hops": 2,
    },
    {
        "id": "Q02",
        "category": "A",
        "question": (
            "I am a part-time student. What is the maximum period of registration "
            "allowed before I must submit my thesis?"
        ),
        "ground_truth": (
            "Part-time students have a maximum registration period defined in the "
            "conditions of the registration rules. Extensions may be granted under "
            "specific circumstances as outlined in the relevant policy section."
        ),
        "expected_node_ids": {"R3.1", "C3.1.1", "C3.1.2"},
        "expected_edges": ["HAS_CONDITION"],
        "min_hops": 1,
    },
    {
        "id": "Q03",
        "category": "A",
        "question": (
            "What happens if a doctoral candidate does not respond to required "
            "changes after their viva voce examination?"
        ),
        "ground_truth": (
            "If a candidate does not complete required changes within the specified "
            "period after the viva, the outcome may include termination of candidature "
            "or referral back for re-examination, as defined by the examination rules."
        ),
        "expected_node_ids": {"R6.1", "C6.1.1", "O6.1.1", "O6.1.2"},
        "expected_edges": ["HAS_CONDITION", "HAS_OUTCOME"],
        "min_hops": 2,
    },
    {
        "id": "Q04",
        "category": "A",
        "question": (
            "If I want to transfer from MPhil to PhD, what conditions must I satisfy "
            "and what is the process?"
        ),
        "ground_truth": (
            "To transfer from MPhil to PhD, the candidate must satisfy conditions "
            "including demonstrating research progress, submitting a transfer report, "
            "and having approval from the supervisory team and relevant committee."
        ),
        "expected_node_ids": {"R4.2", "C4.2.1", "C4.2.2", "O4.2.1"},
        "expected_edges": ["HAS_CONDITION", "HAS_OUTCOME"],
        "min_hops": 2,
    },
    {
        "id": "Q05",
        "category": "A",
        "question": (
            "A student has been sanctioned for academic misconduct during their "
            "research degree. Can they still progress to the viva?"
        ),
        "ground_truth": (
            "Progression to viva after an academic misconduct sanction depends on "
            "the severity of the sanction and the resolution applied. The relevant "
            "rules define whether the sanction permits continued candidature or "
            "results in termination."
        ),
        "expected_node_ids": {"R7.1", "C7.1.1", "O7.1.1", "O7.1.2"},
        "expected_edges": ["HAS_CONDITION", "HAS_OUTCOME"],
        "min_hops": 2,
    },

    # ── B. CONFLICT DETECTION ──────────────────────────────────────────────────
    {
        "id": "Q06",
        "category": "B",
        "question": (
            "Are there any conflicts between the requirements for the examining "
            "team composition in section 5.1?"
        ),
        "ground_truth": (
            "A policy conflict has been detected between Section 5.1.2 and "
            "Section 5.1.6. The conflict involves the requirements for internal "
            "and external examiner eligibility. The resolution specifies which "
            "condition takes precedence."
        ),
        "expected_node_ids": {"R5.1", "C5.1.2", "C5.1.6"},
        "expected_edges": ["HAS_CONDITION", "CONFLICTS_WITH"],
        "min_hops": 2,
    },
    {
        "id": "Q07",
        "category": "B",
        "question": (
            "Can a staff member who supervised the candidate also serve as "
            "an internal examiner?"
        ),
        "ground_truth": (
            "A policy conflict has been detected between the conditions governing "
            "examiner eligibility. Staff members involved in supervision may face "
            "restrictions on serving as examiners, as defined in the examining team "
            "composition rules."
        ),
        "expected_node_ids": {"R5.1", "C5.1.2", "C5.1.6"},
        "expected_edges": ["HAS_CONDITION", "CONFLICTS_WITH"],
        "min_hops": 2,
    },
    {
        "id": "Q08",
        "category": "B",
        "question": (
            "Is there a conflict between the word limit conditions for the "
            "thesis and the transfer report?"
        ),
        "ground_truth": (
            "If a conflict exists between word limit conditions, it would be "
            "identified as a CONFLICTS_WITH relationship between the relevant "
            "conditions. The resolution would specify which limit applies in "
            "cases of overlap."
        ),
        "expected_node_ids": {"C3.1.1", "C4.2.1"},
        "expected_edges": ["CONFLICTS_WITH"],
        "min_hops": 1,
    },
    {
        "id": "Q09",
        "category": "B",
        "question": (
            "What policy conflicts exist in the examination regulations, and "
            "how are they resolved?"
        ),
        "ground_truth": (
            "A policy conflict has been detected between Section 5.1.2 and "
            "Section 5.1.6. The scenario describes the overlapping requirements "
            "and the resolution specifies the precedence rule."
        ),
        "expected_node_ids": {"R5.1", "C5.1.2", "C5.1.6"},
        "expected_edges": ["HAS_CONDITION", "CONFLICTS_WITH"],
        "min_hops": 2,
    },
    {
        "id": "Q10",
        "category": "B",
        "question": (
            "If two policies about examiner eligibility contradict each other, "
            "which one takes precedence?"
        ),
        "ground_truth": (
            "When two examiner eligibility conditions conflict, the resolution "
            "stored on the CONFLICTS_WITH relationship specifies which condition "
            "takes precedence. This is defined in the conflict metadata."
        ),
        "expected_node_ids": {"C5.1.2", "C5.1.6"},
        "expected_edges": ["CONFLICTS_WITH"],
        "min_hops": 1,
    },

    # ── C. FACTUAL RETRIEVAL (single-hop) ──────────────────────────────────────
    {
        "id": "Q11",
        "category": "C",
        "question": "What are the requirements for an examining team in section 5.1?",
        "ground_truth": (
            "The examining team requirements in section 5.1 define the composition "
            "of the examination panel, including the roles and eligibility criteria "
            "for internal and external examiners, and the independent chair."
        ),
        "expected_node_ids": {"R5.1", "C5.1.1", "C5.1.2"},
        "expected_edges": ["HAS_CONDITION"],
        "min_hops": 1,
    },
    {
        "id": "Q12",
        "category": "C",
        "question": "What is the purpose of the Probationary Review?",
        "ground_truth": (
            "The Probationary Review assesses whether a research student has made "
            "sufficient progress and has the potential to complete their degree "
            "within the prescribed timeframe."
        ),
        "expected_node_ids": {"R4.1", "C4.1.1"},
        "expected_edges": ["HAS_CONDITION"],
        "min_hops": 1,
    },
    {
        "id": "Q13",
        "category": "C",
        "question": "What are the rules for supervision of research students?",
        "ground_truth": (
            "The supervision rules define the requirements for the supervisory "
            "team, including minimum meeting frequency, supervisor qualifications, "
            "and the roles of the principal and secondary supervisors."
        ),
        "expected_node_ids": {"R2.1", "C2.1.1"},
        "expected_edges": ["HAS_CONDITION"],
        "min_hops": 1,
    },
    {
        "id": "Q14",
        "category": "C",
        "question": "What types of research degrees does BU offer?",
        "ground_truth": (
            "Bournemouth University offers research degrees including PhD, MPhil, "
            "Professional Doctorate, and MRes, as defined in the programme "
            "structure rules."
        ),
        "expected_node_ids": {"R1.1", "C1.1.1"},
        "expected_edges": ["HAS_CONDITION"],
        "min_hops": 1,
    },
    {
        "id": "Q15",
        "category": "C",
        "question": "What is the word limit for a PhD thesis?",
        "ground_truth": (
            "The word limit for a PhD thesis is defined in the relevant condition "
            "of the thesis submission rules, specifying maximum word counts "
            "for different degree types."
        ),
        "expected_node_ids": {"R3.1", "C3.1.1"},
        "expected_edges": ["HAS_CONDITION"],
        "min_hops": 1,
    },

    # ── D. EDGE-CASE / NEGATION ────────────────────────────────────────────────
    {
        "id": "Q16",
        "category": "D",
        "question": (
            "Can a student who has been withdrawn from the programme re-enrol "
            "for the same research degree?"
        ),
        "ground_truth": (
            "The policy on re-enrolment after withdrawal would be defined in the "
            "relevant rules. If this information is not available in the graph, "
            "the student should consult the BU Doctoral College directly."
        ),
        "expected_node_ids": {"R7.1", "O7.1.1"},
        "expected_edges": ["HAS_OUTCOME"],
        "min_hops": 1,
    },
    {
        "id": "Q17",
        "category": "D",
        "question": "Is there a policy on plagiarism in research degree submissions?",
        "ground_truth": (
            "Academic misconduct including plagiarism is addressed in the relevant "
            "rules, which define the conditions under which misconduct is investigated "
            "and the possible outcomes including sanctions."
        ),
        "expected_node_ids": {"R7.1", "C7.1.1", "O7.1.1"},
        "expected_edges": ["HAS_CONDITION", "HAS_OUTCOME"],
        "min_hops": 2,
    },
    {
        "id": "Q18",
        "category": "D",
        "question": (
            "What happens if no external examiner is available for the viva — "
            "can the examination proceed with only internal examiners?"
        ),
        "ground_truth": (
            "The examining team composition rules require at least one external "
            "examiner. An examination cannot proceed with only internal examiners "
            "unless an explicit exception is defined in the policy."
        ),
        "expected_node_ids": {"R5.1", "C5.1.1", "C5.1.2"},
        "expected_edges": ["HAS_CONDITION"],
        "min_hops": 1,
    },
    {
        "id": "Q19",
        "category": "D",
        "question": (
            "Does the university provide funding for conference attendance? "
            "This is a question about something NOT in the Code of Practice."
        ),
        "ground_truth": (
            "This information is not available in the current policy graph. "
            "Please consult the BU Doctoral College directly."
        ),
        "expected_node_ids": set(),
        "expected_edges": [],
        "min_hops": 0,
    },
    {
        "id": "Q20",
        "category": "D",
        "question": (
            "If my supervisor leaves the university mid-way through my PhD, "
            "what protections exist for me?"
        ),
        "ground_truth": (
            "The supervision rules address changes to the supervisory team, "
            "including provisions for replacement supervisors and continuity "
            "of support when a supervisor departs."
        ),
        "expected_node_ids": {"R2.1", "C2.1.1", "O2.1.1"},
        "expected_edges": ["HAS_CONDITION", "HAS_OUTCOME"],
        "min_hops": 2,
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_dataset_by_category(cat: str):
    """Filter dataset by category letter: A, B, C, or D."""
    return [q for q in EVAL_DATASET if q["category"] == cat]


def get_dataset_summary():
    """Print a summary of the evaluation dataset."""
    from collections import Counter
    cats = Counter(q["category"] for q in EVAL_DATASET)
    print(f"Total questions: {len(EVAL_DATASET)}")
    print(f"  A (Reasoning):  {cats['A']}")
    print(f"  B (Conflict):   {cats['B']}")
    print(f"  C (Factual):    {cats['C']}")
    print(f"  D (Edge-case):  {cats['D']}")
    avg_hops = sum(q["min_hops"] for q in EVAL_DATASET) / len(EVAL_DATASET)
    print(f"  Average min hops: {avg_hops:.1f}")


if __name__ == "__main__":
    get_dataset_summary()
