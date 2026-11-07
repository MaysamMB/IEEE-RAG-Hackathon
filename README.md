# IEEE RAG Hackathon — Starter Workspace

A domain-neutral Python starter workspace for a three-person team. The final problem statement and authorized documents have not yet been selected, so this repository intentionally contains a reusable RAG core rather than a domain-specific solution.

## Current status

- [x] Initial repository structure
- [x] PDF/TXT ingestion with page metadata
- [x] Paragraph-aware overlapping chunking
- [x] Multilingual embedding adapter (`intfloat/multilingual-e5-small`, see `docs/EMBEDDING_COMPARISON.md`)
- [x] Persistent ChromaDB vector store
- [x] Retrieval with source metadata
- [x] LLM answer generation with source IDs (any OpenAI-compatible service)
- [x] Arabic Gradio chat with follow-up questions
- [x] Refusal with a referral, and a separate out-of-scope reply
- [x] Source list limited to the sources the answer cites
- [x] Evaluation dataset template and evaluation script
- [x] Arabic PDF repair and OCR for scanned pages
- [x] Sample data and sample questions
- [ ] Select assigned problem and approved source documents
- [ ] Re-check the embedding model on the official challenge data
- [ ] Build at least 10 adjudicated benchmark questions
- [ ] Calibrate retrieval threshold and citation behavior
- [ ] Add business research and final presentation

## Language policy

The official product UI language is Arabic. Code, identifiers, and comments stay in English. Source documents may be Arabic, English, or mixed.

## Architecture

```text
Indexing:  PDF/TXT -> documents.py + pdf_text.py + ocr.py -> chunking.py -> embeddings.py -> vector_store.py (ChromaDB)
Question:  question -> retrieval.py -> generation.py (LLM) -> pipeline.py -> app.py (Gradio)
Measuring: evaluation/questions.json -> scripts/evaluate.py -> evaluation/results/
```

## AI disclosure

Claude Code was used to write and review code, tests, and documentation in this repository. The prompts in `docs/prompts/` were written to support the business, marketing, and presentation work. The team reviewed the results and is responsible for explaining every part of the project.

## Public data sources

`data/source_versions.json` gives each source its version, used to prefer the newest source when sources conflict. Printed documents use their stated academic year. Live web pages show no issue date, so they use the download date. A source with no known date is left out, and a conflict involving it is shown rather than resolved.

The demo indexes public documents from Palestine Polytechnic University. They stay out of version control. `python scripts/fetch_bylaws.py` recreates the bylaw text files in `data/raw/university/`.

| File | Source |
|---|---|
| `ppu_bachelor_regulations.txt` | https://dar.ppu.edu/ar/bylaws/details/67 |
| `ppu_diploma_regulations.txt` | https://dar.ppu.edu/ar/bylaws/details/28 |
| `ppu_admission_instructions.txt` | https://dar.ppu.edu/ar/bylaws/details/32 |
| `ppu_bridging_transfer.txt` | https://dar.ppu.edu/ar/bylaws/details/34 |
| `ppu_about.txt` | https://www.ppu.edu/p/en/about |
| `guide-2022-2023.pdf` | https://ppu.edu/ar/about/guide |
| `bachelor-plans-2020-2021.pdf`, `diploma-plans-2020.pdf`, `factbook.pdf` | https://ppu.edu/sites/default/files/ |
| `academic-programs.pdf` | https://admission.ppu.edu/UpLoads/AcademicPrograms.pdf |
| `citce-regulations.pdf` | https://citce.ppu.edu/ar/documents |

## 1. Requirements

- Python 3.11 or later (use the Python version available and supported by your libraries)
- A free API key from an OpenAI-compatible LLM service, or a local Ollama install
- Internet access for installing packages and downloading the embedding model on first use

**Billing note:** The team reports that the event does not allow paid model APIs, so this application uses a free tier or a local model. `.env.example` lists Groq, Gemini, OpenRouter, and Ollama settings. Free limits and model names change, so check your provider's console before running API calls.

## 2. Setup — Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

Open `.env` and set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. To change provider, change these three values; no code changes are needed. Never commit `.env`.

## 3. Setup — macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

## Quick trial with sample data

`data/sample/` holds fictional Arabic documents, so the project can run without private files. They include an old and a new fee rule, to test conflicting sources:

```bash
python scripts/ingest.py --folder data/sample --rebuild
python app.py
```

Each collection indexes one folder, so use `--rebuild` when you move to `data/raw`.

## 4. Add documents

Put only approved, relevant challenge documents in `data/raw/`. Supported initial formats: PDF and TXT. The PDF loader extracts each page separately so page citations can be retained. Scanned PDFs may require OCR; do not assume they are text-readable.

Arabic PDFs can extract with reversed words. For Arabic pages the loader takes word order from pdfium and uses PyMuPDF text to repair reversed words. The repair is partial, so check every new PDF before relying on it:

```bash
python scripts/check_pdf_extraction.py data/raw/document.pdf --show 1
```

Then compare the flagged pages with the page images.

Pages with an image but no text are treated as scanned and read with OCR during indexing. If a whole file extracts badly, read it entirely with OCR:

```bash
python scripts/ingest.py --rebuild --ocr-file document.pdf
```

OCR runs on the CPU and is slow, roughly one to two minutes per page on our test machine. Each finished page is saved under `data/processed/ocr_cache/`, so later runs reuse it. The first OCR run downloads about 300 MB of models, so run it once before the event. OCR can misread letters and digits, so check important values against the page.

`requirements.lock.txt` records the exact versions tested on Windows with Python 3.14. After pulling changes, run `pip install -r requirements.txt` again before starting the app. `app.py` needs Gradio 6.31 or later: `launch(css=...)` fails with a `TypeError` on Gradio 5.50.

## 5. Index documents

```bash
python scripts/ingest.py
```

The first run may download the configured embedding model. The vector database is stored under `storage/chroma/` and is intentionally ignored by Git.

A normal run adds or updates chunks and never deletes anything. If chunks from an earlier run remain, it prints a warning with their count, because they can still appear in answers. It stops with an error if the embedding model, chunk settings, or documents folder differ from those stored with the collection. Each collection indexes one documents folder only.

Use an explicit full re-index after changing those settings or folder, or after removing, moving, renaming, or shortening documents:

```bash
python scripts/ingest.py --rebuild
```

`--rebuild` removes all old chunks. It writes a new collection first and swaps it in only after every chunk is stored. If an error occurs, the existing index is kept and the temporary collection is removed. Stop `app.py` before rebuilding. Questions also stop with an error if the configured embedding model differs from the one used to build the index.

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

Run the evaluation:

```bash
python scripts/evaluate.py --questions evaluation/questions.json
python scripts/evaluate.py --questions evaluation/questions.json --answers
python scripts/evaluate.py --questions evaluation/questions.sample.json --folder data/sample --rebuild
```

The first command scores retrieval locally and costs nothing. `--answers` calls the configured model twice per question (formal rewrite, then answer) and also scores facts, citations, and refusal. `--rebuild` re-indexes `--folder` first. Each run is saved under `evaluation/results/`. The script prints the largest in-scope and smallest out-of-scope distance; use them to calibrate `MAX_COSINE_DISTANCE`, and keep it blank while measuring. Include direct, paraphrase, dialect, multi-document, conflicting-source, and out-of-scope questions. A citation counts as correct only when the answer cites every expected source. Each result file records the index settings that produced it. Sample scores only prove that the pipeline runs, so never report them as project results. Fact matching is a simple text check. In `expected_answer_facts`, separate accepted spellings with `|`, for example `فصلين|فصلان`, because Arabic endings change with grammar. Read the saved answers to judge faithfulness. A follow-up question adds a `history` list of earlier `question` and `answer` turns; it is scored only with `--answers`, because rewriting it needs the model. With `--answers`, the script reports the average tokens per answer. Add your model prices to `.env` to see the cost in USD.

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
