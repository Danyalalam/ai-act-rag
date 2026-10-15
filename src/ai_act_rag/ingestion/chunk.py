from collections import Counter
from dataclasses import dataclass

from ai_act_rag.ingestion.parse import PDF_PATH, Line, Unit, extract_lines, split_units

# Units longer than this are split into their numbered paragraphs.
MAX_WORDS = 500


@dataclass(frozen=True)
class Chunk:
    """A whole legal unit, or one numbered paragraph of a long one."""

    unit: Unit
    paragraph: str | None  # "1" for Article 26(1); None when the chunk is the whole unit
    lines: tuple[Line, ...]
    lead_in: str = ""  # the unit's text before paragraph 1, repeated for context

    @property
    def citation(self) -> str:
        """How the chunk is cited, e.g. "Article 26(1)", "Annex III, point 1", "Recital (27)"."""
        unit = self.unit
        if unit.kind == "recital":
            return f"Recital ({unit.number})"
        if unit.kind == "article":
            suffix = f"({self.paragraph})" if self.paragraph else ""
            return f"Article {unit.number}{suffix}"
        suffix = f", point {self.paragraph}" if self.paragraph else ""
        return f"Annex {unit.number}{suffix}"

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


def paragraph_starts(unit: Unit) -> list[int]:
    """Return the line indexes where paragraphs 1, 2, 3, ... begin.

    A marker counts whether it stands alone ("1.") or opens the line ("(10) 'AI system'...").
    """
    starts = []
    number = 1
    for index, line in enumerate(unit.lines):
        marker = paragraph_marker(unit, number)
        if line.text == marker or line.text.startswith(marker + " "):
            starts.append(index)
            number += 1
    return starts


def chunk_unit(unit: Unit) -> list[Chunk]:
    """Keep a short unit whole; split a long one at its numbered paragraphs."""
    starts = paragraph_starts(unit)
    if word_count(unit.text) <= MAX_WORDS or len(starts) < 2:
        return [Chunk(unit=unit, paragraph=None, lines=unit.lines)]

    lead_in = "\n".join(line.text for line in unit.lines[: starts[0]])
    ends = starts[1:] + [len(unit.lines)]
    return [
        Chunk(unit=unit, paragraph=str(number), lines=unit.lines[start:end], lead_in=lead_in)
        for number, (start, end) in enumerate(zip(starts, ends), start=1)
    ]


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

    for citation in ("Article 3(12)", "Article 26(1)", "Annex III, point 4", "Recital (27)"):
        chunk = next(c for c in chunks if c.citation == citation)
        print(f"--- {citation} (pages {chunk.first_page}-{chunk.last_page}) ---")
        print(chunk.text[:200])


if __name__ == "__main__":
    main()
