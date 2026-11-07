"""Run the evaluation and save the results.

Retrieval checks are free and local. --answers calls the configured LLM once per question, twice for follow-ups.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from rag_hackathon.evaluation import MINIMUM_QUESTIONS, evaluate, load_questions
from rag_hackathon.ingest import ingest

def share(value: dict | None) -> str:
    return f"{value['passed']}/{value['total']}" if value else "not measured"

def print_report(report: dict) -> None:
    summary = report["summary"]
    used = report["settings"]
    print(f"\nIndex: {used['embedding_model']} | {used['indexed_chunks']} chunks | size {used['chunk_size']}, overlap {used['chunk_overlap']}")
    print(f"Questions: {summary['questions']} | top-k: {summary['top_k']}")
    print(f"Hit@k:     {share(summary['hit_at_k'])}")
    print(f"Precision@k: {summary['precision_at_k']} | Recall@k: {summary['recall_at_k']} | MRR: {summary['mrr']}")
    print(f"All facts present: {share(summary['answers_with_all_facts'])}")
    print(f"Correct citation:  {share(summary['correct_citations'])}")
    print(f"Refusal decision:  {share(summary['correct_refusal_decisions'])}")
    print(f"Errors: {summary['errors']}")
    usage = summary["usage"]
    if usage:
        cost = usage["average_cost_usd"]
        cost_text = f"${cost}" if cost is not None else "set INPUT_PRICE_PER_MTOK and OUTPUT_PRICE_PER_MTOK"
        print(f"Per answer: {usage['average_input_tokens']} input and {usage['average_output_tokens']} output tokens | cost: {cost_text}")
    print("\nFor calibrating MAX_COSINE_DISTANCE (leave it blank while measuring):")
    print(f"  largest top distance, in-scope questions:  {summary['largest_in_scope_distance']}")
    print(f"  smallest top distance, out-of-scope:       {summary['smallest_out_of_scope_distance']}")
    print("\nBy category:")
    for category, scores in summary["by_category"].items():
        print(f"  {category}: hit {share(scores['hit_at_k'])}, refusal {share(scores['correct_refusal_decisions'])}")
    failures = [
        row for row in report["rows"]
        if row.get("hit") is False or row.get("facts_correct") is False
        or row.get("citation_correct") is False or row.get("refusal_correct") is False or "error" in row
    ]
    if failures:
        print("\nFailed questions:")
        for row in failures:
            print(f"  {row['id']} [{row['category']}] {row['question']}")
            if row.get("missing_facts"):
                print(f"    missing facts: {row['missing_facts']}")
            if row.get("citation_correct") is False:
                print(f"    cited: {row.get('cited_sources')} | invalid: {row.get('invalid_citations')}")
            if "error" in row:
                print(f"    error: {row['error']}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default="evaluation/questions.json")
    parser.add_argument("--answers", action="store_true", help="Also generate and score answers (uses your LLM quota).")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from --folder first.")
    parser.add_argument("--folder", default="data/raw", help="Documents folder used with --rebuild.")
    parser.add_argument("--label", default="run", help="Short name saved in the result file name.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    questions = load_questions(args.questions)
    if len(questions) < MINIMUM_QUESTIONS:
        print(f"Warning: the judging guide asks for at least {MINIMUM_QUESTIONS} questions; found {len(questions)}.")
    if args.rebuild:
        ingest(args.folder, rebuild=True)

    report = evaluate(questions, with_answers=args.answers)
    print_report(report)
    out = Path("evaluation/results") / f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("Faithfulness needs review: read the saved answers against their sources.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
