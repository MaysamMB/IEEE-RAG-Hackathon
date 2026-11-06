# IEEE RAG Hackathon — Starter Workspace

A domain-neutral Python starter workspace for a three-person team. The final problem statement and authorized documents have not yet been selected, so this repository intentionally contains a reusable RAG core rather than a domain-specific solution.

## Current status

- [x] Initial repository structure
- [x] PDF/TXT ingestion with page metadata
- [x] Paragraph-aware overlapping chunking
- [x] Multilingual embedding adapter (local Sentence Transformers)
- [x] Persistent ChromaDB vector store
- [x] Retrieval with source metadata
- [x] Claude answer generation with source IDs
- [x] Minimal Arabic Gradio interface
- [x] Evaluation dataset template and starter unit test
- [ ] Select assigned problem and approved source documents
- [ ] Evaluate multilingual embedding choices on real data
- [ ] Build at least 10 adjudicated benchmark questions
- [ ] Calibrate retrieval threshold and citation behavior
- [ ] Add business research and final presentation

## Language policy

The official product UI language is Arabic. Code, identifiers, and comments stay in English. Source documents may be Arabic, English, or mixed.

## 1. Requirements

- Python 3.11 or later (use the Python version available and supported by your libraries)
- An Anthropic API key for the RAG application itself
- Internet access for installing packages and downloading the embedding model on first use

**Billing note:** A paid Claude chat subscription and Claude API usage are separate products in most plans. Claude Code may be included with eligible paid plans, but calls made by this Python RAG application require API access/credits or another configured LLM provider. Check your Console billing before running API calls.

## 2. Setup — Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

Open `.env` and set `ANTHROPIC_API_KEY` and a valid `ANTHROPIC_MODEL` available to your API account. Never commit `.env`.

## 3. Setup — macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

## 4. Add documents

Put only approved, relevant challenge documents in `data/raw/`. Supported initial formats: PDF and TXT. The PDF loader extracts each page separately so page citations can be retained. Scanned PDFs may require OCR; do not assume they are text-readable.

## 5. Index documents

```bash
python scripts/ingest.py
```

The first run may download the configured embedding model. The vector database is stored under `storage/chroma/` and is intentionally ignored by Git.

A normal run adds or updates chunks and never deletes anything. It stops with an error if the embedding model, chunk settings, or documents folder differ from those stored with the collection. Each collection indexes one documents folder only.

Use an explicit full re-index after changing those settings or folder, or after removing, moving, renaming, or shortening documents:

```bash
python scripts/ingest.py --rebuild
```

`--rebuild` removes all old chunks. It writes a new collection first and swaps it in only after every chunk is stored. If an error occurs, the existing index is kept and the temporary collection is removed. Stop `app.py` before rebuilding.

If the program is killed during the swap, questions and normal indexing stop with `InterruptedRebuildError`. Recover by running `--rebuild` again. Do not delete the `_backup` collection before that succeeds. If the documents are no longer available, restore the previous index from Python:

```python
import chromadb
client = chromadb.PersistentClient(path="storage/chroma")
client.get_collection("hackathon_documents_backup").modify(name="hackathon_documents")
```

Pages with no extractable text are skipped with a warning that names the file and pages. Such pages may be scanned and need OCR.

## 6. Launch the Arabic demo

```bash
python app.py
```

The first UI is intentionally minimal. Improve design only after retrieval, citations, and refusal behavior pass evaluation.

## 7. Run tests

```bash
pytest -q
```

## 8. Evaluation

Copy `evaluation/questions.template.json` to `evaluation/questions.json`. Fill it with at least 10 questions, expected answer facts, expected source/page, and a category. Include direct, paraphrased, multi-hop, out-of-scope, and conflicting/old-source questions where relevant. Do not count template examples as evaluation results.

Report measured results only. Suggested measures: Precision@k, Recall@k, faithfulness, citation accuracy, and refusal accuracy. A small, manually reviewed test set is better than fabricated metrics.

## 9. Git safety

```bash
git init
git add .
git status
```

Before committing, verify that `.env`, private documents, and `storage/` are not staged.

## 10. First action when the problem is announced

1. Save the official problem statement to `docs/PROBLEM_STATEMENT.md`.
2. Confirm document usage permissions.
3. Add the official documents to `data/raw/`.
4. Create the benchmark questions with expected evidence.
5. Ask Claude Code to inspect the documents and adapt the baseline to the problem without changing unrelated modules.

See `CLAUDE.md` and the files under `docs/` for team and implementation workflow.
