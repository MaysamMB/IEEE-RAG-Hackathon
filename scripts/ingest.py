import argparse
from rag_hackathon.ingest import ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index the documents in data/raw.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace the whole collection, removing all old chunks.",
    )
    parser.add_argument(
        "--ocr-file",
        action="append",
        default=[],
        metavar="PATH",
        help="Read this PDF, relative to data/raw, entirely with OCR. Can be repeated.",
    )
    args = parser.parse_args()
    ingest("data/raw", rebuild=args.rebuild, ocr_files=args.ocr_file)
