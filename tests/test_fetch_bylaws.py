"""HTML-to-text tests for the bylaw fetcher; no network is needed."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fetch_bylaws", Path(__file__).resolve().parents[1] / "scripts" / "fetch_bylaws.py"
)
fetch_bylaws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_bylaws)


def to_text(body):
    return fetch_bylaws.html_to_text(f"<html><nav>menu</nav><main>{body}</main><p>footer</p></html>")


def test_only_main_content_is_kept():
    assert to_text("<p>text</p>") == "text"


def test_table_row_stays_on_one_line_with_line_breaks_in_cells():
    html = "<table><tr><td>\n\t\tfirst</td><td>\n\t\tsecond</td></tr></table>"

    assert to_text(html) == "first | second"


def test_paragraphs_inside_cells_do_not_split_the_row():
    html = "<table><tr><td><p>90</p></td><td><p>100</p></td><td><p>excellent</p></td></tr></table>"

    assert to_text(html) == "90 | 100 | excellent"


def test_numeric_table_rows_survive_cleaning():
    html = "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td></tr></table>"

    assert to_text(html) == "1 | 2\n3"


def test_section_counters_are_removed_in_sequence_only():
    html = "<p>1 first section</p><p>2 second section</p><p>5 years of study</p>"

    assert to_text(html) == "first section\nsecond section\n5 years of study"


def test_page_chrome_is_removed():
    assert to_text("<p>عدد البنود: 14</p><p>جميع التعليمات والقوانين</p><p>text</p>") == "text"
    assert to_text("<p>عدد البنود: 14 جميع التعليمات والقوانين</p><p>text</p>") == "text"
