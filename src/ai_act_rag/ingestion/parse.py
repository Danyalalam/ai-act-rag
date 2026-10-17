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


ANNEX_NUMBERS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]
CHAPTER_HEADING = re.compile(r"CHAPTER ([IVXL]+)")
SECTION_HEADING = re.compile(r"SECTION (\d+)")


@dataclass(frozen=True)
class Line:
    """One clean line of text, the PDF page it came from (1-based), and whether it is bold."""

    page: int
    text: str
    bold: bool = False


@dataclass(frozen=True)
class Unit:
    """One legal unit: a recital, an article or an annex."""

    kind: str  # "recital", "article" or "annex"
    number: str  # "27", "26" or "III"
    title: str  # empty for recitals
    chapter: str | None  # "III" (articles only)
    section: str | None  # "2" (articles only, when the chapter has sections)
    lines: tuple[Line, ...]

    @property
    def first_page(self) -> int:
        return self.lines[0].page

    @property
    def last_page(self) -> int:
        return self.lines[-1].page

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


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


def page_lines(page: pymupdf.Page) -> list[Line]:
    """Return the page's lines with their bold flag, leaving out the footnote area."""
    top = footnote_top(page)
    bottom = page.rect.height if top is None else top
    text = page.get_text("dict", clip=pymupdf.Rect(0, 0, page.rect.width, bottom))
    lines = []
    for block in text["blocks"]:
        for line in block.get("lines", []):
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            lines.append(
                Line(
                    page=page.number + 1,
                    text="".join(span["text"] for span in line["spans"]),
                    bold=all(span["flags"] & pymupdf.TEXT_FONT_BOLD for span in spans),
                )
            )
    return lines


def is_page_furniture(line: str) -> bool:
    """Return True if the line is a repeated header/footer, not real content."""
    return line in PAGE_FURNITURE or PAGE_COUNTER.fullmatch(line) is not None


def clean_lines(lines: list[Line]) -> list[Line]:
    """Trim every line and drop blanks and header/footer lines."""
    trimmed = [Line(line.page, line.text.strip(), line.bold) for line in lines]
    return [line for line in trimmed if line.text and not is_page_furniture(line.text)]


def extract_lines(pdf_path: Path) -> list[Line]:
    """Return all clean lines of the PDF in reading order, without footnotes."""
    with pymupdf.open(pdf_path) as doc:
        return [line for page in doc for line in clean_lines(page_lines(page))]


def read_title(lines: list[Line], start: int) -> tuple[str, int]:
    """Join the bold lines from `start` into one title; return it and the next index.

    If none is bold (Annex X's title isn't), the single next line is the title.
    """
    end = start
    while end < len(lines) and lines[end].bold:
        end += 1
    if end == start:
        end = start + 1
    title = " ".join(line.text for line in lines[start:end])
    return title.rstrip("`"), end  # the PDF has a stray backtick after Article 1's title


def split_units(lines: list[Line]) -> list[Unit]:
    """Group the document's lines into recitals, articles and annexes.

    Headings only count when they carry the next expected number, so a line
    like "Article 49" inside a sentence that wraps is not mistaken for one.
    """
    units: list[Unit] = []
    # The unit being collected: its fields, and the body lines so far.
    current: dict | None = None
    body: list[Line] = []

    def finish() -> None:
        if current is not None and body:
            units.append(Unit(**current, lines=tuple(body)))

    part = "recitals"  # then "articles", "signatures", "annexes"
    next_recital, next_article, next_annex = 1, 1, 0
    chapter: str | None = None
    section: str | None = None

    i = 0
    while i < len(lines):
        text = lines[i].text
        new_unit = None

        if part == "recitals":
            if text == "HAVE ADOPTED THIS REGULATION:":
                finish()
                current, body, part = None, [], "articles"
                i += 1
                continue
            if text == f"({next_recital})":
                new_unit = dict(kind="recital", number=str(next_recital), title="")
                next_recital += 1
                i += 1

        elif part == "articles":
            if match := CHAPTER_HEADING.fullmatch(text):
                chapter, section = match.group(1), None
                _, i = read_title(lines, i + 1)
                continue
            if match := SECTION_HEADING.fullmatch(text):
                section = match.group(1)
                _, i = read_title(lines, i + 1)
                continue
            if text == f"Article {next_article}":
                title, i = read_title(lines, i + 1)
                new_unit = dict(kind="article", number=str(next_article), title=title)
                next_article += 1
            elif text.startswith("Done at "):
                finish()
                current, body, part = None, [], "signatures"
                i += 1
                continue

        elif next_annex < len(ANNEX_NUMBERS) and text == f"ANNEX {ANNEX_NUMBERS[next_annex]}":
            number = ANNEX_NUMBERS[next_annex]
            title, i = read_title(lines, i + 1)
            new_unit = dict(kind="annex", number=number, title=title)
            next_annex += 1
            part = "annexes"

        if new_unit is not None:
            finish()
            is_article = new_unit["kind"] == "article"
            current = dict(
                **new_unit,
                chapter=chapter if is_article else None,
                section=section if is_article else None,
            )
            body = []
            continue

        if current is not None:
            body.append(lines[i])
        i += 1

    finish()
    return units


def main() -> None:
    lines = extract_lines(PDF_PATH)
    units = split_units(lines)
    print(f"Clean lines: {len(lines):,}")
    for kind in ("recital", "article", "annex"):
        print(f"{kind}: {sum(unit.kind == kind for unit in units)}")
    for kind, number in (("recital", "27"), ("article", "26"), ("article", "59"), ("annex", "III")):
        unit = next(u for u in units if u.kind == kind and u.number == number)
        print(f"--- {kind} {number}: {unit.title!r} ---")
        print(f"chapter={unit.chapter} section={unit.section} pages={unit.first_page}-{unit.last_page}")
        print(unit.text[:150])


if __name__ == "__main__":
    main()