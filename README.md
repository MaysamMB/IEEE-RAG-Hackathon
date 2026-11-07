# Dalil | AI-Powered Academic Advisor

**An Arabic-first academic advising assistant powered by Retrieval-Augmented Generation (RAG).**

Dalil helps university students find and understand academic regulations, admission instructions, and study-plan information through a conversational Arabic interface. It retrieves relevant passages from indexed university documents and asks a Large Language Model (LLM) to answer using that evidence.

**Hackathon:** IEEE RAG Hackathon 2026 — Academic Advisor Challenge  
**Current demonstration scope:** Publicly available academic information from Palestine Polytechnic University (PPU), Palestine.

> Dalil is an independent informational prototype. It does not replace official academic advisors, access personal student records, or make official university decisions.

---

## Table of Contents

- [Problem and Solution](#problem-and-solution)
- [Features](#features)
- [Project Status](#project-status)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Embedding Model Selection](#embedding-model-selection)
- [Evaluation](#evaluation)
- [Data Sources and Versions](#data-sources-and-versions)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Business Direction](#business-direction)
- [AI Assistance Disclosure](#ai-assistance-disclosure)
- [References](#references)

## Problem and Solution

University academic information is distributed across regulations, student guides, admission instructions, and study plans. Students may ask questions in everyday Palestinian Arabic while official documents use formal Arabic. Finding the applicable rule can take time, and unsupported answers may have academic consequences.

Dalil addresses this gap with a document-grounded RAG workflow:

1. Load approved academic PDF and TXT documents.
2. Extract and preprocess their text, including Arabic PDF repair and OCR where needed.
3. Split the text into overlapping chunks while retaining source metadata.
4. Generate embeddings and store them in ChromaDB.
5. Rewrite dialect or follow-up questions into standalone formal-Arabic queries.
6. Retrieve relevant passages and ask an LLM to answer using only that evidence.
7. Display available document references and decline when evidence is insufficient.

The assistant is designed to respond in simple Modern Standard Arabic, including when the student asks in dialect.

## Features

- Arabic-first Gradio chat interface with right-to-left layout.
- Conversational Arabic, formal Arabic, and contextual follow-up support.
- PDF and TXT ingestion with page-level metadata for PDFs.
- Arabic PDF text extraction repair and OCR for scanned pages.
- Paragraph-aware chunking with configurable overlap.
- Multilingual semantic embeddings using Sentence Transformers.
- Persistent ChromaDB vector storage with cosine distance.
- Source-aware answer generation with document, page, and article identifiers when available.
- Refusal for insufficient evidence and questions outside academic scope.
- Configurable OpenAI-compatible LLM providers.
- Optional rewrite and fallback models; fallback is used for supported rate-limit errors.
- Document-version metadata for sources that may conflict.
- Automated retrieval/answer evaluation scripts and unit tests.

## Project Status

### Implemented

- [x] Academic advisor problem and scope defined.
- [x] Public PPU academic sources identified.
- [x] PDF/TXT ingestion and metadata extraction.
- [x] Arabic PDF text repair and OCR support.
- [x] Paragraph-aware overlapping chunking.
- [x] Source-version metadata.
- [x] Multilingual embeddings and persistent ChromaDB indexing.
- [x] Semantic retrieval.
- [x] LLM-based question rewriting and follow-up history.
- [x] Evidence-grounded answer generation and source display.
- [x] Insufficient-evidence and out-of-scope response handling.
- [x] Configurable OpenAI-compatible LLM integration and rate-limit fallback.
- [x] Arabic Gradio interface.
- [x] PPU evaluation dataset with 51 questions across 12 categories.
- [x] Automated evaluation scripts and unit tests.
- [x] Interactive testing report covering 29 recorded questions.
- [x] AI assistance disclosure.

### Still Requires Validation

- [ ] Independently review answer accuracy against official documents.
- [ ] Verify that citations support the exact claims made.
- [ ] Calibrate the retrieval distance threshold on representative PPU questions.
- [ ] Reduce false refusals for valid academic questions.
- [ ] Improve retrieval from study-plan tables and prerequisite relationships.
- [ ] Test on unseen questions and additional academic programs.
- [ ] Confirm that indexed documents are the currently applicable versions.
- [ ] Conduct repeatable performance and concurrent-user testing.
- [ ] Complete business validation with prospective institutional customers.

**Current status:** Functional RAG prototype with a benchmark dataset and documented interactive testing. Full factual accuracy, citation faithfulness, and production readiness have not been established.

## Architecture

Dalil has two main workflows: document indexing and question answering.

### 1. Document Indexing

```text
Approved University Documents (PDF / TXT)
                  |
                  v
          Document Loading
                  |
                  v
      Arabic Text Extraction / Repair
                  |
                  v
             OCR (if needed)
                  |
                  v
        Cleaning and Chunking
                  |
                  v
       Multilingual E5 Embeddings
                  |
                  v
          ChromaDB Vector Store
                  |
                  v
       Persistent Searchable Index
```

- **Loading:** Documents are read from the selected folder. PDFs are processed page by page; TXT files are read as text. Metadata includes source path, file name, page, file type, and available version.
- **Arabic processing:** PyMuPDF and pypdfium2 are combined to repair common Arabic extraction problems. Scanned pages can be processed with EasyOCR. OCR output is cached.
- **Chunking:** Paragraph-aware overlapping chunks are created. The default chunk size is 1,000 characters with 150 characters of overlap.
- **Embeddings and storage:** Chunks are encoded with `intfloat/multilingual-e5-small` and stored with their text and metadata in persistent ChromaDB.

### 2. Question Answering

```text
Student Question + Conversation History
                  |
                  v
          Question Rewriting
       (Dialect / Follow-up -> Standalone)
                  |
                  v
          Query Embedding
                  |
                  v
       Semantic Search in ChromaDB
                  |
                  v
           Top-K Passages
                  |
                  v
         LLM Answer Generation
          (Retrieved Evidence Only)
                  |
                  v
        Answer / Refusal + Sources
                  |
                  v
           Arabic Gradio Chat
```

- **Question rewriting:** The configured LLM rewrites the latest question into a standalone question in simple formal Arabic. Conversation history resolves follow-up references. If rewriting fails, the original question is used.
- **Retrieval:** The rewritten question is embedded and compared with indexed vectors using cosine distance. The default number of retrieved chunks is four.
- **Generation:** Retrieved passages and metadata are supplied to the LLM. Instructions require it to use only the supplied context, cite source IDs, and refuse when evidence is insufficient. These instructions reduce unsupported answers but cannot guarantee correctness.

### Default Retrieval Settings

| Parameter | Default |
|---|---|
| Embedding model | `intfloat/multilingual-e5-small` |
| Vector database | ChromaDB |
| Similarity space | Cosine |
| Chunk size | 1,000 characters |
| Chunk overlap | 150 characters |
| Top-K | 4 |
| Maximum cosine distance | Unset by default; requires calibration |

The embedding model used for queries must match the model used to build the index.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| User interface | Gradio |
| LLM client | OpenAI Python SDK |
| LLM service | Configurable OpenAI-compatible endpoint |
| Embeddings | Sentence Transformers |
| Embedding model | `intfloat/multilingual-e5-small` |
| Vector database | ChromaDB |
| PDF extraction | PyMuPDF, pypdfium2 |
| OCR | EasyOCR |
| Configuration | python-dotenv |
| Tests | pytest |
| Evaluation | Custom Python evaluation framework |

## Embedding Model Selection

Three multilingual embedding models were compared using 12 Arabic retrieval questions from a single judging-guide PDF.

| Model | Dimensions | Correct at Rank 1 | Correct in Top 4 | MRR |
|---|---:|---:|---:|---:|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 6/12 | 10/12 | 0.654 |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 9/12 | 9/12 | 0.785 |
| `intfloat/multilingual-e5-small` | 384 | 10/12 | 12/12 | 0.903 |

**Selected model:** `intfloat/multilingual-e5-small`.

It achieved the strongest retrieval result in this preliminary comparison. The implementation uses the E5-specific `query:` prefix for questions and `passage:` prefix for document chunks, with normalized embeddings.

**Limitation:** The comparison used 12 questions based on one judging-guide document, not the PPU academic benchmark. It supports the initial model-selection decision but does not establish performance on academic advising questions. See [Embedding Comparison](docs/EMBEDDING_COMPARISON.md).

## Evaluation

### PPU Benchmark Dataset

The repository contains 51 questions across 12 categories in `evaluation/questions.ppu.json`.

| Category | Questions |
|---|---:|
| Direct facts | 15 |
| Dialect | 6 |
| Paraphrase | 3 |
| Tables | 5 |
| Multiple articles | 2 |
| Multiple documents | 1 |
| Follow-up | 3 |
| Not in documents | 5 |
| Out of scope | 3 |
| Multiple rules | 2 |
| Conflicting sources | 2 |
| Study-plan tables | 4 |
| **Total** | **51** |

The benchmark covers direct academic facts, dialect, paraphrases, tables, multi-article and multi-document questions, follow-ups, conflicting sources, unavailable information, and out-of-scope prompts.

### Supported Metrics

- **Hit@K:** Whether at least one expected source appears in retrieved results.
- **Precision@K:** The proportion of retrieved results matching expected-source labels.
- **Recall@K:** The proportion of expected sources retrieved.
- **MRR:** Mean Reciprocal Rank of the first matching source.
- **Expected-fact matching:** Whether specified reference facts appear in the generated answer.
- **Citation matching:** Whether required source references appear.
- **Refusal decision accuracy:** Whether the system answers or refuses as expected.

Expected-fact matching uses normalized text comparison. It is not proof of semantic correctness or factual faithfulness. Generated answers and citations still require human review against original documents.

### Recorded Interactive Tests

The Gradio application was manually tested with 29 recorded questions on **7 November 2026**.

| Metric | Observed result |
|---|---:|
| Recorded interactions | 29 |
| Mean response time | 7.480 seconds |
| Median response time | 1.967 seconds |
| Minimum response time | 0.830 seconds |
| Maximum response time | 37.900 seconds |
| Technical successes recorded | 29/29 |
| Responses displaying sources | 13/29 |
| Explicit out-of-scope refusals | 4/29 |
| Insufficient-evidence responses | 12/29 |
| Independently verified answer accuracy | Not measured |

These are observational smoke-test results, not an answer-accuracy benchmark. A technical success means the application processed the request; it does not mean the answer was correct. Response times include application-level processing, not only LLM inference.

The test audit identified false refusals, incomplete answers to multi-part questions, gaps in study-plan/prerequisite retrieval, and answers that need verification against the applicable degree/program. See [Interactive Evaluation Results](evaluation/RESULTS.md).

### Run the PPU Evaluation

Retrieval-only evaluation:

```bash
python scripts/evaluate.py --questions evaluation/questions.ppu.json --label ppu-retrieval
```

Retrieval and answer evaluation:

```bash
python scripts/evaluate.py --questions evaluation/questions.ppu.json --answers --label ppu-answers
```

Results are saved under `evaluation/results/`. Answer mode calls the configured LLM for rewriting and answer generation.

**The 51-question dataset is evaluation coverage, not a 51/51 passing score.**

## Data Sources and Versions

### Current Demonstration Scope

The prototype focuses on publicly accessible PPU information related to:

- Bachelor's degree regulations.
- Diploma regulations.
- Admission instructions.
- Bridging and transfer procedures.
- Student guidance and institutional information.
- Available academic study plans.

Source documents are generally excluded from version control. Obtain the relevant public documents and index them locally.

### Official References

| Source | Link |
|---|---|
| Bachelor's regulations | https://dar.ppu.edu/ar/bylaws/details/67 |
| Diploma regulations | https://dar.ppu.edu/ar/bylaws/details/28 |
| Admission instructions | https://dar.ppu.edu/ar/bylaws/details/32 |
| Bridging and transfer | https://dar.ppu.edu/ar/bylaws/details/34 |
| PPU website | https://www.ppu.edu/ |
| University guide | https://ppu.edu/ar/about/guide |
| Academic programs | https://admission.ppu.edu/UpLoads/AcademicPrograms.pdf |
| Academic plans | https://citce.ppu.edu/ar/documents |

`scripts/fetch_bylaws.py` downloads public bylaw pages and selected institutional information as UTF-8 text files. Add other approved PDF/TXT documents manually under `data/raw/university/`.

### Document Version Handling

Available source-version metadata is maintained in `data/source_versions.json`.

- Printed documents use their stated academic year when available.
- For undated live web pages, the recorded download date identifies when content was collected; it is not the official publication or effective date.
- Sources without a known version do not receive an invented version.
- Generation instructions consider source versions when retrieved documents conflict.

The presence of a document in the index does not prove that it is the latest applicable regulation. Important academic rules must be checked against the current official university source.

## Getting Started

### Requirements

- Python 3.11 or later.
- Internet access for installing packages and downloading the embedding model on first use.
- An API key for a supported OpenAI-compatible LLM provider, or a locally running compatible model such as Ollama.
- Approved PPU academic source documents for the PPU demonstration.

The repository includes `requirements.lock.txt`, which records a tested dependency set. Project dependencies are listed in `requirements.txt`.

### 1. Clone the Repository

```bash
git clone https://github.com/MaysamMB/IEEE-RAG-Hackathon.git
cd IEEE-RAG-Hackathon
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Environment Variables

Copy the example file:

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Open `.env` and configure the LLM provider as described below.

### 4. Download Public PPU Bylaws

```bash
python scripts/fetch_bylaws.py
```

This creates text files under `data/raw/university/`. Add other approved PDF/TXT documents to the same folder.

### 5. Build the Vector Index

```bash
python scripts/ingest.py --rebuild
```

The command extracts documents, chunks the text, generates embeddings, and creates the persistent ChromaDB index under `storage/chroma/`.

### 6. Launch the Application

```bash
python app.py
```

Open the local Gradio URL printed in the terminal.

## Configuration

The application reads settings from `.env`.

| Variable | Purpose |
|---|---|
| `LLM_API_KEY` | API key for the selected provider |
| `LLM_BASE_URL` | OpenAI-compatible API endpoint |
| `LLM_MODEL` | Main answer-generation model |
| `LLM_REWRITE_MODEL` | Optional question-rewriting model |
| `LLM_FALLBACK_MODEL` | Optional fallback for supported rate-limit errors |
| `LLM_REASONING_EFFORT` | Optional reasoning effort for supported models |
| `EMBEDDING_MODEL` | Sentence Transformers embedding model |
| `OCR_CACHE_DIR` | OCR cache directory |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory |
| `CHROMA_COLLECTION` | ChromaDB collection name |
| `CHUNK_SIZE` | Chunk size in characters |
| `CHUNK_OVERLAP` | Chunk overlap in characters |
| `TOP_K` | Number of retrieved chunks |
| `MAX_COSINE_DISTANCE` | Optional retrieval distance threshold |
| `INPUT_PRICE_PER_MTOK` | Optional input price per million tokens, in USD |
| `OUTPUT_PRICE_PER_MTOK` | Optional output price per million tokens, in USD |

`.env.example` contains example configurations for Groq, Google Gemini, OpenRouter, and Ollama. Provider model names, availability, and free-tier limits can change; check the provider's current documentation and console.

The event setup is intended to use a free-tier provider or a local model. Do not assume a particular model remains free or available.

**Security:** Never commit `.env`, API keys, credentials, private documents, or local vector database files.

## Running the Application

### Sample-Data Demo

If PPU documents are unavailable, fictional Arabic sample documents under `data/sample/` can be used to check the general pipeline. They are not PPU academic regulations and must not be used to evaluate PPU advising accuracy.

```bash
python scripts/ingest.py --folder data/sample --rebuild
python app.py
```

When switching to PPU documents, rebuild the index:

```bash
python scripts/ingest.py --folder data/raw --rebuild
```

### Add or Update Documents

Supported formats are PDF and TXT. Put approved documents under `data/raw/` or a selected subfolder.

Inspect Arabic PDF extraction before relying on a newly added file:

```bash
python scripts/check_pdf_extraction.py data/raw/document.pdf --show 1
```

Scanned PDF pages with no extractable text are processed with OCR during indexing. If an entire PDF needs OCR, specify it explicitly:

```bash
python scripts/ingest.py --rebuild --ocr-file document.pdf
```

OCR is CPU-based and can take considerable time. Its first run downloads OCR models, and recognized page text is cached under the configured OCR cache directory. OCR can misread Arabic words and numbers, so verify important values against the original page image.

### Incremental Indexing and Rebuilding

A normal indexing run adds or updates chunks; it does not remove chunks left over from documents that were deleted, moved, renamed, or shortened. The application may therefore retrieve stale chunks after incremental indexing.

Use `--rebuild` after changing the document folder, embedding model, chunk settings, or source files in ways that require old chunks to be removed:

```bash
python scripts/ingest.py --rebuild
```

A rebuild replaces the collection after preparing a complete replacement index. Stop the running Gradio application before rebuilding. Do not manually delete ChromaDB backup collections while rebuild recovery is pending.

Each collection is associated with its embedding model and indexing settings. The application rejects a query when the configured embedding model does not match the model used to build the index.

## Testing

Run unit tests:

```bash
pytest -q
```

Run retrieval-only evaluation:

```bash
python scripts/evaluate.py --questions evaluation/questions.ppu.json
```

Run retrieval and generated-answer evaluation:

```bash
python scripts/evaluate.py --questions evaluation/questions.ppu.json --answers
```

Evaluate the sample dataset against fictional sample documents:

```bash
python scripts/evaluate.py --questions evaluation/questions.sample.json --folder data/sample --rebuild
```

Evaluation reports are written to `evaluation/results/`. The `--answers` option uses the configured LLM and consumes provider quota. Follow-up questions also require answer-generation mode because their context is resolved during question rewriting.

Review saved answers against original sources before making claims about faithfulness. Expected-fact matching is text-based and is not a substitute for human review.

## Project Structure

```text
IEEE-RAG-Hackathon/
├── app.py
├── .env.example
├── requirements.txt
├── requirements.lock.txt
├── pyproject.toml
├── data/
│   ├── raw/                  # Local approved university documents (not committed)
│   ├── processed/            # OCR cache and processed data
│   ├── sample/               # Fictional sample documents
│   └── source_versions.json  # Source version metadata
├── docs/
│   ├── COMPETITION_BRIEF.md
│   ├── PROBLEM_STATEMENT.md
│   ├── EMBEDDING_COMPARISON.md
│   └── TEAM_ROLES.md
├── evaluation/
│   ├── questions.ppu.json
│   ├── questions.sample.json
│   └── RESULTS.md
├── scripts/
│   ├── fetch_bylaws.py
│   ├── ingest.py
│   ├── evaluate.py
│   └── check_pdf_extraction.py
├── src/rag_hackathon/
│   ├── documents.py
│   ├── pdf_text.py
│   ├── ocr.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── pipeline.py
│   └── evaluation.py
└── tests/
```

## Known Limitations

- **Retrieval coverage:** Semantic search may miss relevant passages when wording is ambiguous or when a question depends on relationships spread across multiple documents or tables.
- **Study-plan tables:** Retrieval involving course plans, prerequisites, and program-specific requirements needs further validation and improvement.
- **False refusals:** Some valid academic questions have been incorrectly classified as out of scope or as lacking sufficient evidence.
- **Incomplete answers:** Multi-part questions may receive only a partial answer.
- **Citation faithfulness:** A displayed source does not prove that it supports every claim in the answer. Citations must be checked against the original source.
- **Program ambiguity:** Rules can differ by degree level and program. The current prompt defaults to bachelor's regulations when a question does not specify a program; this may not suit every question.
- **Document currency:** Some indexed guides and study plans are from earlier academic years. Version metadata does not guarantee that a source is currently in force.
- **Arabic extraction and OCR:** Extraction repair and OCR can introduce errors, especially in tables, numbers, and scanned pages.
- **Model availability and latency:** Generation depends on the selected provider, model availability, quotas, network conditions, and service response time.
- **No personal student records:** The prototype does not connect to student accounts, transcripts, registration records, or personal academic profiles.
- **No production validation:** Security review, load testing, concurrent-user capacity, and production service-level guarantees have not been established.

For binding academic decisions, students should confirm the applicable rule with the relevant PPU department or the Deanship of Admission and Registration.

## Business Direction

The proposed direction is a **Business-to-Business (B2B) SaaS** service that universities could configure with their approved academic documents. Potential institutional services include document onboarding, configuration, integrations, maintenance, and support.

This is a proposed direction, not evidence of an existing contract, institutional partnership, paying customer, or validated price. Commercial feasibility still requires market research, interviews with prospective institutions, and measured operating costs.

## AI Assistance Disclosure

AI-assisted development tools, including Claude Code, were used to support code generation, review, testing, and documentation. The team reviewed the generated material and remains responsible for the implementation, understanding of the system, evaluation evidence, and accuracy of the submitted claims.

AI assistance does not replace software testing, source verification, or technical validation.

## References

### Project Documentation

- [Official Problem Statement](docs/PROBLEM_STATEMENT.md)
- [Competition Brief](docs/COMPETITION_BRIEF.md)
- [Embedding Model Comparison](docs/EMBEDDING_COMPARISON.md)
- [PPU Evaluation Questions](evaluation/questions.ppu.json)
- [Interactive Evaluation Results](evaluation/RESULTS.md)
- [Team Roles](docs/TEAM_ROLES.md)

### External Technical Documentation

- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://docs.trychroma.com/)
- [Gradio](https://www.gradio.app/docs)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

---

**Repository:** https://github.com/MaysamMB/IEEE-RAG-Hackathon
