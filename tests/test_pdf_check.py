"""Tests for the read-only PDF extraction report."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pdf_extraction.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_pdf_extraction", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(check, text):
    return check.page_report(1, text, 0)


def test_clean_arabic_page_has_no_flags(check):
    text = "لا يقبل الطلب خلال العطلة إلا بعذر مكتوب من عالم معروف، والرسوم 250 ديناراً."

    assert check.flags(report(check, text)) == []


def test_empty_page_is_flagged(check):
    assert check.flags(report(check, "   ")) == ["NO_TEXT"]


def test_known_ligature_damage_is_flagged(check):
    row = report(check, "الطلب خالل العطلة إال بعذر والفصل األول")

    assert row["damaged"] == 3
    assert "KNOWN_DAMAGE" in check.flags(row)


def test_missing_lam_alef_on_long_arabic_page_is_flagged(check):
    assert "NO_LAM_ALEF" in check.flags(report(check, "كتاب " * 60))


def test_reversed_line_is_flagged(check):
    row = report(check, ".النتيجة إعلان تاريخ\n،الطلب تقديم يجب")

    assert row["reversed_lines"] == 2
    assert "REVERSED_LINES" in check.flags(row)


def test_detached_marks_and_glued_numbers_are_flagged(check):
    flags = check.flags(report(check, "الحضو ُر في عام250"))

    assert "DETACHED_MARKS" in flags
    assert "GLUED_NUMBERS" in flags
