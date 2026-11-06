import argparse
from rag_hackathon.ingest import ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index the documents in data/raw.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace the whole collection, removing all old chunks.",
    )
    ingest("data/raw", rebuild=parser.parse_args().rebuild)
