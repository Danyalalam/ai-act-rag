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


@dataclass(frozen=True)
class Line:
    """One clean line of text and the PDF page it came from (1-based)."""

    page: int
    text: str


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the plain text of every page in the PDF, one string per page."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


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
    print("--- Lines around the page 50/51 boundary ---")
    boundary = next(i for i, line in enumerate(lines) if line.page == 51)
    for line in lines[boundary - 2 : boundary + 3]:
        print(f"p.{line.page}: {line.text}")


if __name__ == "__main__":
    main()