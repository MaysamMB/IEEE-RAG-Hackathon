# Claude Code Project Instructions — IEEE RAG Hackathon

## Mission
Help a three-person team build a reliable, demonstrable, commercially plausible RAG product for the IEEE RAG Hackathon at Bethlehem University. The assigned problem is not known yet. Keep the codebase domain-neutral until the official problem statement and data arrive.

## Source of truth
- `docs/COMPETITION_BRIEF.md` captures the currently known judging and delivery requirements.
- The official uploaded competition and bootcamp PDFs in the Claude Project Knowledge are the primary source for event rules.
- Clearly distinguish official requirements, engineering recommendations, and assumptions.
- Never invent judging rules, statistics, evaluation scores, sources, costs, or test results.

## Language
- Official product UI language: Arabic. All user-facing text must be Arabic with RTL support.
- Explain decisions and teach the user in Arabic unless asked otherwise.
- Keep code, identifiers, technical names, and comments in English.
- Do not assume the input documents are English. The challenge may involve Arabic, English, or mixed-language data.

## Required workflow
1. Inspect the repository before changing it.
2. Ask for or inspect the official problem statement and permitted source documents before domain-specific implementation.
3. Agree on the smallest working baseline before adding advanced features.
4. Implement in small, testable steps. State files changed and commands to run.
5. Run tests and report actual outputs. Never claim tests passed if they were not run.
6. Avoid unnecessary dependencies and overengineering during the short hackathon.
7. Preserve document name, page number, section/title, version/date, and chunk ID in metadata whenever available.
8. Show evidence for answers and abstain when the indexed sources do not support an answer.
9. Calibrate retrieval thresholds empirically; never select a distance threshold by guesswork.
10. Do not upload private documents or `.env` files to GitHub.

## Architecture boundaries
Keep these responsibilities separate:
- ingestion and text extraction
- preprocessing and chunking
- embeddings
- vector storage
- retrieval and optional reranking
- prompt construction and answer generation
- API/interface
- evaluation and tests

Do not train a foundation model. Use a suitable pretrained embedding model and an LLM for generation. Evaluate embedding quality on the actual challenge language and questions before locking the model.

## Judging priorities
Prioritize the official criteria in this order:
- Technical RAG: document ingestion/chunking/metadata, retrieval quality, grounded answers with citations and refusal, evaluation.
- Problem fit and user needs.
- Clean modular code, reproducibility, documentation, security, and scalability.
- Business model, market evidence, costs, marketing, and presentation.
- Product bonus only after the technical RAG threshold is met.

## Security
- Read API secrets only from environment variables.
- `.env`, private source documents, database files, and credentials must remain out of version control.
- Never print secrets in logs.
- Do not send confidential documents to hosted APIs unless the team has permission.

## Team of three (initial suggestion)
- Member A: ingestion, preprocessing, chunking, metadata, embeddings, vector store.
- Member B: retrieval, prompting, Claude API integration, backend/pipeline.
- Member C: evaluation, interface, README/demo, business/presentation.
All members must understand the entire pipeline and be able to explain the core choices.

## Definition of done for each feature
- code exists in the correct module
- dependencies/configuration are documented
- at least one test or reproducible manual check exists
- expected failure cases are handled
- the user can explain what it does and why it is needed
