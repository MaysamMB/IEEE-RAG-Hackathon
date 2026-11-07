"""Save public PPU web pages, mainly the study bylaws, as UTF-8 text files.

The bylaw pages keep clean Arabic and numbered articles, unlike the PDF copies.
"""
import argparse
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.request import Request, urlopen

PAGES = {
    "ppu_bachelor_regulations": "https://dar.ppu.edu/ar/bylaws/details/67",
    "ppu_diploma_regulations": "https://dar.ppu.edu/ar/bylaws/details/28",
    "ppu_admission_instructions": "https://dar.ppu.edu/ar/bylaws/details/32",
    "ppu_bridging_transfer": "https://dar.ppu.edu/ar/bylaws/details/34",
    # An English overview page: its figures can differ from the bylaws.
    "ppu_about": "https://www.ppu.edu/p/en/about",
}
SKIPPED_TAGS = {"script", "style", "nav", "footer", "form", "noscript", "svg"}
BLOCK_TAGS = {"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol"}
CELL_TAGS = {"td", "th"}
PAGE_CHROME = re.compile(r"عدد البنود: \d+ ?(جميع التعليمات والقوانين)?|جميع التعليمات والقوانين")


class MainText(HTMLParser):
    """Collect the text inside <main>; cells of one table row stay on one line."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.main_depth = 0
        self.skip_depth = 0
        self.cell_depth = 0

    def _break(self) -> None:
        self.parts.append(" " if self.cell_depth else "\n")

    def handle_starttag(self, tag, attrs):
        if tag == "main":
            self.main_depth += 1
        if tag in SKIPPED_TAGS:
            self.skip_depth += 1
        if tag in CELL_TAGS:
            self.cell_depth += 1
        if self.main_depth and tag in BLOCK_TAGS:
            self._break()

    def handle_endtag(self, tag):
        if tag == "main":
            self.main_depth = max(0, self.main_depth - 1)
        if tag in SKIPPED_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
        if tag in CELL_TAGS:
            self.cell_depth = max(0, self.cell_depth - 1)
        if self.main_depth and not self.skip_depth:
            if tag in CELL_TAGS:
                self.parts.append(" | ")
            elif tag in BLOCK_TAGS:
                self._break()

    def handle_data(self, data):
        if self.main_depth and not self.skip_depth:
            # Source line breaks inside a table cell must not split its row.
            self.parts.append(re.sub(r"\s+", " ", data))


def without_section_counters(lines: list[str]) -> list[str]:
    """Drop the site's 1, 2, 3 section counters; only that exact sequence is removed."""
    cleaned, expected = [], 1
    for line in lines:
        prefix = f"{expected} "
        if line.startswith(prefix) and " | " not in line:
            line = line[len(prefix):]
            expected += 1
        cleaned.append(line)
    return cleaned


def html_to_text(html: str) -> str:
    parser = MainText()
    parser.feed(html)
    lines = [re.sub(r"\s+", " ", line).strip(" |") for line in "".join(parser.parts).splitlines()]
    kept = [line for line in lines if line and not PAGE_CHROME.fullmatch(line)]
    return "\n".join(without_section_counters(kept))


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "rag-hackathon-fetch/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/raw/university")
    args = parser.parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in PAGES.items():
        text = html_to_text(fetch(url))
        path = folder / f"{name}.txt"
        path.write_text(f"المصدر: {url}\nتاريخ التنزيل: {date.today()}\n\n{text}\n", encoding="utf-8")
        print(f"{path}: {len(text)} characters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
