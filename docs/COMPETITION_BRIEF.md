# Competition Brief — Known Requirements

Source: uploaded IEEE RAG Hackathon Judging Guide (version 1.0). Re-check with organizers for any announced changes.

## Judging: 100 points
- Technical RAG quality: 30 (data preparation 5, retrieval 10, grounded generation 10, evaluation 5)
- Problem understanding and solution fit: 15
- Code and engineering quality: 15
- Business plan and numbers: 15
- Marketing plan: 10
- Presentation and communication: 10
- Innovation: 5

## Product bonus: up to 10 additional points
- Working product/web/mobile/API interface: up to 5
- UX and Arabic RTL support: up to 2
- Productivity features (e.g. login, document upload, saved chats, analytics): up to 3
- Eligibility condition: at least 15/30 in technical RAG.

## Submission items
- Code in GitHub or compressed archive
- README, requirements file, `.env.example`, and sample data
- Presentation PDF/PPTX, maximum 12 slides
- Optional 2–3 minute screen recording
- AI-use disclosure in README

## Required presentation coverage
Problem/evidence; RAG architecture; live evidence/demo; market TAM/SAM/SOM; revenue/pricing; costs and 12–36 month projections/break-even; marketing and first 90 days; competitors/differentiation; team roles and next steps.

## Technical expectations
- Clean document extraction, defensible chunking, metadata including source and page
- Appropriate embeddings/vector store; justify top-k; consider hybrid retrieval/reranking/metadata filters if they improve measured quality
- Answers grounded only in retrieved evidence, correct citations, safe refusal when evidence is missing
- At least 10 benchmark questions with reference answers; measure retrieval and faithfulness; analyze failures and improvements

## Judges may ask
Chunking rationale; embedding model and Arabic capability; behavior on out-of-scope questions; evaluation results and worst failure; conflicting/old documents; unit cost and scaling; market evidence and financial assumptions; AI-generated vs team-written work; privacy; first prospective customers.

## Ten proposed challenge areas
Academic advising; legal/procedural assistant; farmer assistant; safe initial health guidance; software customer support; internal HR knowledge; contract analyzer; tourism/heritage guide; grants/funding assistant; cybersecurity/compliance assistant.

The final assigned problem is unknown. Do not select or implement a domain until the team receives the official problem statement.
