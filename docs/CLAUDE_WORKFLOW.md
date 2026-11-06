# Suggested Claude Workflow

## Claude Project Knowledge
Upload the official judging guide, Bootcamp Sessions 1–3, Lecture 5, and the handwritten lecture notes. Keep these as competition reference materials.

## Claude Project Instructions
Use `CLAUDE.md` as the project instruction text. Also tell Claude: the official product UI language is Arabic; explain decisions to the team in Arabic; do not assume the assigned problem.

## Claude Code
Open this folder in a terminal and start `claude` from the repository root. Claude Code should read `CLAUDE.md`.

### First Claude Code prompt

> Read CLAUDE.md, README.md, and docs/COMPETITION_BRIEF.md. Inspect the entire starter repository. Do not modify files yet. Give me: (1) a system architecture explanation, (2) a file-by-file implementation review, (3) setup risks on Windows, (4) a prioritized checklist for making the baseline run, and (5) questions we must answer after receiving the official challenge. Distinguish verified facts from assumptions.

### After receiving the challenge

> Read docs/PROBLEM_STATEMENT.md and inspect all files under data/raw. Do not write domain-specific code until you have summarized the user, problem, document inventory, language, risks, and ten candidate test questions. Then propose the smallest RAG baseline and explain every design choice. Do not invent source facts or test results.

### During implementation

Ask Claude Code to work one small milestone at a time. After each milestone, run the tests, inspect the diff, and ask for a concise explanation of the changed code.
