import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace

from ai_act_rag.ingestion.parse import PDF_PATH, Line, Unit, extract_lines, split_units

# Units longer than this are split into their numbered paragraphs, and
# paragraphs still longer than this into their lettered points (a), (b), ...
MAX_WORDS = 500


@dataclass(frozen=True)
class Chunk:
    """A whole legal unit, one numbered paragraph of it, or one lettered point."""

    unit: Unit
    paragraph: str | None  # "1" for Article 26(1); None when the chunk is the whole unit
    lines: tuple[Line, ...]
    lead_in: str = ""  # text before the first paragraph/point, repeated for context
    point: str | None = None  # "f" for Article 5(1)(f)

    @property
    def citation(self) -> str:
        """How the chunk is cited, e.g. "Article 26(1)", "Article 5(1)(f)", "Annex III, point 4"."""
        unit = self.unit
        if unit.kind == "recital":
            return f"Recital ({unit.number})"
        point = f"({self.point})" if self.point else ""
        if unit.kind == "article":
            if self.paragraph:
                return f"Article {unit.number}({self.paragraph}){point}"
            return f"Article {unit.number}, point {point}" if point else f"Article {unit.number}"
        if self.paragraph or point:
            return f"Annex {unit.number}, point {self.paragraph or ''}{point}"
        return f"Annex {unit.number}"

    @property
    def first_page(self) -> int:
        return self.lines[0].page

    @property
    def last_page(self) -> int:
        return self.lines[-1].page

    @property
    def text(self) -> str:
        body = "\n".join(line.text for line in self.lines)
        return f"{self.lead_in}\n{body}" if self.lead_in else body


def word_count(text: str) -> int:
    return len(text.split())


def paragraph_marker(unit: Unit, number: int) -> str:
    """The text that opens paragraph `number`: "(12)" for Article 3's definitions, else "12."."""
    if unit.kind == "article" and unit.number == "3":
        return f"({number})"
    return f"{number}."


def point_marker(number: int) -> str:
    """The text that opens lettered point `number`: 1 -> "(a)", 6 -> "(f)"."""
    return f"({string.ascii_lowercase[number - 1]})"


def marker_starts(lines: tuple[Line, ...], marker: Callable[[int], str]) -> list[int]:
    """Return the line indexes where items 1, 2, 3, ... begin, as named by `marker`.

    A marker counts whether it stands alone ("1.") or opens the line ("(10) 'AI system'...").
    Only the next expected marker counts, so "(i)" inside point (a) isn't taken for point (i).
    """
    starts = []
    number = 1
    for index, line in enumerate(lines):
        if number > 26 and marker is point_marker:
            break
        text = marker(number)
        if line.text == text or line.text.startswith(text + " "):
            if text == "(i)" and is_roman_numeral_i(lines, index):
                continue
            starts.append(index)
            number += 1
    return starts


def is_roman_numeral_i(lines: tuple[Line, ...], index: int) -> bool:
    """True if the "(i)" at `index` is roman sub-point (i), not letter point (i).

    A roman list goes on with "(ii)" before any letter point "(j)" appears.
    """
    for line in lines[index + 1 :]:
        if line.text.startswith("(ii)"):
            return True
        if line.text.startswith("(j)"):
            return False
    return False


def split_at(lines: tuple[Line, ...], starts: list[int]) -> tuple[str, list[tuple[Line, ...]]]:
    """Split lines at `starts`; return the text before the first start and the pieces."""
    lead_in = "\n".join(line.text for line in lines[: starts[0]])
    ends = starts[1:] + [len(lines)]
    return lead_in, [lines[start:end] for start, end in zip(starts, ends)]


def split_points(chunk: Chunk) -> list[Chunk]:
    """Split a chunk that is too long into its lettered points (a), (b), ..."""
    starts = marker_starts(chunk.lines, point_marker)
    if word_count(chunk.text) <= MAX_WORDS or len(starts) < 2:
        return [chunk]
    intro, pieces = split_at(chunk.lines, starts)
    lead_in = "\n".join(part for part in (chunk.lead_in, intro) if part)
    return [
        replace(chunk, lines=piece, lead_in=lead_in, point=string.ascii_lowercase[i])
        for i, piece in enumerate(pieces)
    ]


def chunk_unit(unit: Unit) -> list[Chunk]:
    """Keep a short unit whole; split a long one at its paragraphs, then at points."""
    whole = Chunk(unit=unit, paragraph=None, lines=unit.lines)
    starts = marker_starts(unit.lines, lambda n: paragraph_marker(unit, n))
    if word_count(unit.text) <= MAX_WORDS or len(starts) < 2:
        return split_points(whole)

    lead_in, pieces = split_at(unit.lines, starts)
    paragraphs = [
        Chunk(unit=unit, paragraph=str(number), lines=piece, lead_in=lead_in)
        for number, piece in enumerate(pieces, start=1)
    ]
    return [chunk for paragraph in paragraphs for chunk in split_points(paragraph)]


def chunk_units(units: list[Unit]) -> list[Chunk]:
    """Chunk every unit, keeping document order."""
    return [chunk for unit in units for chunk in chunk_unit(unit)]


def main() -> None:
    chunks = chunk_units(split_units(extract_lines(PDF_PATH)))
    print(f"Chunks: {len(chunks)}")
    print(Counter(chunk.unit.kind for chunk in chunks))

    sizes = sorted(word_count(chunk.text) for chunk in chunks)
    print(f"Words per chunk: median {sizes[len(sizes) // 2]}, largest {sizes[-1]}")
    too_long = [c for c in chunks if word_count(c.text) > MAX_WORDS]
    print(f"Still over {MAX_WORDS} words: {[(c.citation, word_count(c.text)) for c in too_long]}")

    for citation in ("Article 3(12)", "Article 5(1)(f)", "Article 66, point (a)", "Annex III, point 4"):
        chunk = next(c for c in chunks if c.citation == citation)
        print(f"--- {citation} (pages {chunk.first_page}-{chunk.last_page}) ---")
        print(chunk.text[:200])


if __name__ == "__main__":
    main()
