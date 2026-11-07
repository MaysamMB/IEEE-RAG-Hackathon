"""Download the sources listed in data/sources.json into data/raw; existing files are kept.

Usage: python scripts/fetch_sources.py [--force]
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

MANIFEST = Path("data/sources.json")
RAW_ROOT = Path("data/raw")
USER_AGENT = "IEEE-RAG-Hackathon-Student-Project/1.0"
DELAY_SECONDS = 2.0


def fetch(url: str) -> bytes:
    request = Request(quote(url, safe=":/?=&%"), headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download files that already exist.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        target = RAW_ROOT / source["file"]
        if target.exists() and not args.force:
            print(f"kept {source['id']}")
            continue
        body = fetch(urljoin(manifest["base_url"], source["url"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        print(f"saved {source['id']} ({len(body)} bytes)")
        time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
