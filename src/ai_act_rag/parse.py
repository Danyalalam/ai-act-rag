import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

PDF_PATH = Path("data/ai-act.pdf")

# Lines the Official Journal repeats on every page (header and footer).
PAGE_FURNITURE = {
    "OJ L, 12.7.2024",
    "EN",
    "ELI: http://data.europa.eu/eli/reg/2024/1689/oj",
}
# Page counters like "12/144".
PAGE_COUNTER = re.compile(r"\d+/\d+")

# Footnotes sit below a short horizontal rule at the left text margin,
# and are printed smaller (8.5 pt) than body text (9.6 pt).
SEPARATOR_X = 67  # left edge of the rule, in PDF points
SEPARATOR_WIDTHS = (45, 60)  # the rule is about 51-53 points wide
FOOTER_TOP = 800  # header/footer bands are outside 100..800
FOOTNOTE_MAX_FONT_SIZE = 9.0


@dataclass(frozen=True)
class Line:
    """One clean line of text and the PDF page it came from (1-based)."""

    page: int
    text: str


def separator_ys(page: pymupdf.Page) -> list[float]:
    """Return the y positions of short horizontal rules that may start a footnote area."""
    ys = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "l":  # only straight lines
                continue
            start, end = item[1], item[2]
            is_horizontal = abs(start.y - end.y) < 1
            width = abs(end.x - start.x)
            if (
                is_horizontal
                and abs(start.x - SEPARATOR_X) < 2
                and SEPARATOR_WIDTHS[0] < width < SEPARATOR_WIDTHS[1]
                and 100 < start.y < FOOTER_TOP
            ):
                ys.append(start.y)
    return ys


def largest_font_below(page: pymupdf.Page, y: float) -> float:
    """Return the largest font size of any text between y and the footer."""
    area = pymupdf.Rect(0, y, page.rect.width, FOOTER_TOP)
    text = page.get_text("dict", clip=area)
    sizes = [
        span["size"]
        for block in text["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip()
    ]
    return max(sizes, default=0.0)


def footnote_top(page: pymupdf.Page) -> float | None:
    """Return the y where the page's footnotes start, or None if it has none.

    The amending articles (102-110) quote other regulations, quoted footnotes
    included, so a rule only counts if the text below it is footnote-sized.
    """
    for y in sorted(separator_ys(page), reverse=True):  # lowest rule first
        if largest_font_below(page, y) < FOOTNOTE_MAX_FONT_SIZE:
            return y
    return None


def page_text_without_footnotes(page: pymupdf.Page) -> str:
    """Return the page's text, leaving out the footnote area if there is one."""
    top = footnote_top(page)
    if top is None:
        return page.get_text()
    return page.get_text(clip=pymupdf.Rect(0, 0, page.rect.width, top))


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the text of every page in the PDF without footnotes, one string per page."""
    with pymupdf.open(pdf_path) as doc:
        return [page_text_without_footnotes(page) for page in doc]


def is_page_furniture(line: str) -> bool:
    """Return True if the line is a repeated header/footer, not real content."""
    return line in PAGE_FURNITURE or PAGE_COUNTER.fullmatch(line) is not None


def clean_lines(page_text: str) -> list[str]:
    """Split a page into trimmed lines, dropping blanks and header/footer lines."""
    lines = [line.strip() for line in page_text.splitlines()]
    return [line for line in lines if line and not is_page_furniture(line)]


def document_lines(pages: list[str]) -> list[Line]:
    """Clean every page and return all lines in order, each tagged with its page."""
    return [
        Line(page=page_number, text=text)
        for page_number, page_text in enumerate(pages, start=1)
        for text in clean_lines(page_text)
    ]


def main() -> None:
    pages = extract_pages(PDF_PATH)
    lines = document_lines(pages)
    print(f"Pages: {len(pages)}")
    print(f"Clean lines: {len(lines):,}")
    for page_number in (1, 118, 121):
        page_lines = [line.text for line in lines if line.page == page_number]
        print(f"--- Page {page_number}: last 2 lines ---")
        for text in page_lines[-2:]:
            print(text)


if __name__ == "__main__":
    main()