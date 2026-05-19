# CLAUDE.md

Orientation for Claude and for the user across two laptops, OneDrive sync, and GitHub.

## Project

**Transparent Policy GraphRAG** — a Streamlit application that ingests Bournemouth University policy documents (XML or PDF), maps them into a Neo4j knowledge graph, and answers natural-language questions with grounded citations, deterministic conflict detection, and a visual reasoning trace.

Author: Alireza (Allen) Sharafzad · MSc Data Science & AI · Bournemouth University.

---

## Source of truth: GitHub (NOT OneDrive)

| Question | Answer |
|---|---|
| Where is the canonical state? | GitHub `origin/master` |
| What's OneDrive for? | Convenient file syncing — **not** authoritative |
| If git and OneDrive disagree? | Trust git. Always. |

- Repo: `https://github.com/AllenSharafzad/kg-bot-project`
- Branch: `master`

**Before stopping work on any laptop:**
```
git add -A && git commit -m "..." && git push
```

**Before starting work on any laptop:**
```
git pull --rebase
```

If anything looks confused after a laptop switch, the nuclear option is `git fetch origin master && git reset --hard origin/master` — this **discards** any uncommitted local changes, so be sure they are pushed elsewhere first.

---

## OneDrive + Git footguns (read this once)

The project folder is inside OneDrive on both laptops. This works but has known pitfalls:

1. OneDrive does not understand the `.git/` folder. If OneDrive syncs `.git/index` or `.git/HEAD` while a git command is reading or writing them, the repo can corrupt.
2. Always wait for OneDrive to finish syncing (cloud icon shows ✓ "Up to date") **before** running git commands after a laptop switch.
3. Do not run `git pull` and `git push` on both laptops at the same time. OneDrive will race git for the same files.
4. Long-term fix worth considering: **move the project out of OneDrive entirely** and rely on GitHub alone for cross-laptop sync. OneDrive is great for documents, hostile to git internals.

---

## Tech stack

- **Python 3.11**, Streamlit 1.35+
- **Neo4j AuraDB** (cloud KG) — credentials in `.env`
- **LangChain**: `langchain-neo4j`, `langchain-openai`, `langchain-anthropic`
- **Models in use:**
  - GPT-4o (Cypher generation @ T=0, QA synthesis @ T=0.1)
  - GPT-4o-mini (semantic node labelling)
  - Claude Haiku 4.5 (plain-English narrator + search variants)
  - Claude Sonnet 4.6 (PDF → XML extraction)
  - `text-embedding-3-small` (vector index, 1536-dim cosine)
- **PyMuPDF (fitz)** for PDF text extraction
- **streamlit-agraph + Altair + pandas** for visualisation

---

## File layout

| Path | What it is |
|---|---|
| `app.py` | The whole product — single-file Streamlit application |
| `evaluation_dataset.py` | 20-question gold-standard test set |
| `ragas_evaluation.py` | Standalone evaluation pipeline |
| `Requirements.txt` | Python dependencies |
| `.env` | Local credentials — **never committed** |
| `evaluation_output/` | Generated reports, CSVs, JSON, LaTeX/Markdown |
| `presentation.md` | Marp slide deck |
| `CLAUDE.md` | This file |
| `graphrag_policy_bot.py` | Original CLI prototype, superseded by `app.py` |
| `start-session.ps1`, `end-session.ps1` | Convenience scripts |

---

## Local-only files (never push)

- `.env` — contains OpenAI / Anthropic / Neo4j keys
- `*.pdf` policy documents — copyrighted source material
- Large binaries, virtualenv folders, `.idea`, `.vscode` workspace settings

`.gitignore` is already configured for these.

---

## Common commands

```bash
# Run the app
streamlit run app.py

# Sync status check
git status && git fetch && git log --oneline @{u}..HEAD

# Pull then push in one go
git pull --rebase && git push

# Compile check
python -m py_compile app.py

# Run evaluation (quick mode, 5 questions)
python ragas_evaluation.py --quick
```

---

## Cross-laptop session checklist

**Laptop A → Laptop B handoff:**

1. **On A:** Save all open files in editors.
2. **On A:** `git add -A && git commit -m "session: <short note>" && git push`
3. **On A:** Wait for OneDrive to show "Up to date" ✓ in the system tray.
4. **On B:** Wait for OneDrive to finish syncing.
5. **On B:** `git pull --rebase` — if it complains about uncommitted changes, stop and resolve before continuing.
6. **On B:** Run `pip install -r Requirements.txt` if any dependencies were added on A.

**Never** edit the project on both laptops simultaneously. OneDrive will sync mid-write and git will see contradictory state.

---

## Environment setup on a fresh laptop

1. Clone: `git clone https://github.com/AllenSharafzad/kg-bot-project.git`
2. (Optional) Create a virtual environment: `python -m venv .venv && .venv\Scripts\activate`
3. Install: `pip install -r Requirements.txt`
4. Copy `.env` from the other laptop (do **not** commit it). Required keys:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
5. Run: `streamlit run app.py`

---

## When starting a new Claude session

- Mention which laptop you're on if a command fails — Python environments may differ.
- Mention if a fresh `pip install -r Requirements.txt` is needed.
- If git status is messy, share `git status` output rather than guessing.

---

## Known good state at last commit

- `app.py` compiles clean (`python -m py_compile app.py` → OK).
- Streamlit app launches without exceptions.
- Neo4j connection succeeds when `.env` is present.
- Semantic search requires `OPENAI_API_KEY` for embeddings.
- PDF pipeline requires `PyMuPDF` installed.
