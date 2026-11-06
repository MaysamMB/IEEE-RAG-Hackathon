import argparse
from rag_hackathon.ingest import ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a documents folder.")
    parser.add_argument("--folder", default="data/raw", help="Documents folder to index.")
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
        help="Read this PDF, relative to --folder, entirely with OCR. Can be repeated.",
    )
    args = parser.parse_args()
    ingest(args.folder, rebuild=args.rebuild, ocr_files=args.ocr_file)
