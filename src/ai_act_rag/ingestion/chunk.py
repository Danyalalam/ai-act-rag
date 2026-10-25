import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from ai_act_rag.ingestion.parse import Line, Unit, extract_lines, language_from_args, split_units
from ai_act_rag.languages import ENGLISH, LANGUAGES, Language, english_citation

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
        """How the chunk is cited in its own language: "Article 26(1)" / "Artikel 26 Absatz 1"."""
        unit = self.unit
        return LANGUAGES[unit.language].citation(unit.kind, unit.number, self.paragraph, self.point)

    @property
    def unit_key(self) -> str:
        """A language-neutral key, the same for every language version: "Article 26(1)".

        It links the English and German versions of a chunk, and lets the evaluation
        use one set of expected answers for all languages.
        """
        unit = self.unit
        return english_citation(unit.kind, unit.number, self.paragraph, self.point)

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
    """The text that opens paragraph `number` in the unit's language.

    English: "12." (definitions in Article 3: "(12)"). German: "(12)" (definitions: "12.").
    Annex points are "12." in both.
    """
    language = LANGUAGES[unit.language]
    if unit.kind == "article":
        style = language.definition if unit.number == "3" else language.article_paragraph
        return style.format(number)
    return f"{number}."


def point_marker(unit: Unit, number: int) -> str | None:
    """The text that opens lettered point `number`: 6 -> "(f)" in English, "f)" in German."""
    if number > len(string.ascii_lowercase):
        return None
    return LANGUAGES[unit.language].point.format(string.ascii_lowercase[number - 1])


def marker_starts(
    lines: tuple[Line, ...], marker: Callable[[int], str | None], point_style: str | None = None
) -> list[int]:
    """Return the line indexes where items 1, 2, 3, ... begin, as named by `marker`.

    A marker counts whether it stands alone ("1.") or opens the line ("(10) 'AI system'...").
    Only the next expected marker counts, so "(i)" inside point (a) isn't taken for point (i).
    `point_style` (e.g. "({})") is given for lettered points, to tell letter i from roman i.
    """
    starts = []
    number = 1
    for index, line in enumerate(lines):
        text = marker(number)
        if text is None:
            break
        if line.text == text or line.text.startswith(text + " "):
            if point_style and text == point_style.format("i"):
                if is_roman_numeral_i(lines, index, point_style):
                    continue
            starts.append(index)
            number += 1
    return starts


def is_roman_numeral_i(lines: tuple[Line, ...], index: int, point_style: str) -> bool:
    """True if the "(i)" at `index` is roman sub-point (i), not letter point (i).

    A roman list goes on with "(ii)" before any letter point "(j)" appears.
    """
    roman_ii, letter_j = point_style.format("ii"), point_style.format("j")
    for line in lines[index + 1 :]:
        if line.text.startswith(roman_ii):
            return True
        if line.text.startswith(letter_j):
            return False
    return False


def split_at(lines: tuple[Line, ...], starts: list[int]) -> tuple[str, list[tuple[Line, ...]]]:
    """Split lines at `starts`; return the text before the first start and the pieces."""
    lead_in = "\n".join(line.text for line in lines[: starts[0]])
    ends = starts[1:] + [len(lines)]
    return lead_in, [lines[start:end] for start, end in zip(starts, ends)]


@dataclass
class UnitPlan:
    """Where one unit was split: by paragraph or not, and which paragraphs into points.

    The English chunking is the reference; other languages follow its plan, so every
    language ends up with the same chunks ("Article 26(1)" exists in all of them) even
    though word counts differ (German uses fewer, longer words).
    """

    paragraphs: bool = False
    points: set[str | None] = field(default_factory=set)  # paragraphs split into points


SplitPlan = dict[tuple[str, str], UnitPlan]  # (kind, number) -> plan


def split_plan(chunks: list[Chunk]) -> SplitPlan:
    """Record where `chunks` (normally the English ones) were split."""
    plan: SplitPlan = {}
    for chunk in chunks:
        entry = plan.setdefault((chunk.unit.kind, chunk.unit.number), UnitPlan())
        entry.paragraphs |= chunk.paragraph is not None
        if chunk.point:
            entry.points.add(chunk.paragraph)
    return plan


def split_points(chunk: Chunk, plan: UnitPlan | None = None) -> list[Chunk]:
    """Split a chunk into its lettered points (a), (b), ... if it is too long (or the plan says so)."""
    unit = chunk.unit
    point_style = LANGUAGES[unit.language].point
    starts = marker_starts(chunk.lines, lambda n: point_marker(unit, n), point_style)
    wanted = chunk.paragraph in plan.points if plan else word_count(chunk.text) > MAX_WORDS
    if not wanted or len(starts) < 2:
        return [chunk]
    intro, pieces = split_at(chunk.lines, starts)
    lead_in = "\n".join(part for part in (chunk.lead_in, intro) if part)
    return [
        replace(chunk, lines=piece, lead_in=lead_in, point=string.ascii_lowercase[i])
        for i, piece in enumerate(pieces)
    ]


def chunk_unit(unit: Unit, plan: UnitPlan | None = None) -> list[Chunk]:
    """Keep a short unit whole; split a long one at its paragraphs, then at points.

    With a `plan` (from the English version), split exactly where English was split.
    """
    whole = Chunk(unit=unit, paragraph=None, lines=unit.lines)
    starts = marker_starts(unit.lines, lambda n: paragraph_marker(unit, n))
    wanted = plan.paragraphs if plan else word_count(unit.text) > MAX_WORDS
    if not wanted or len(starts) < 2:
        return split_points(whole, plan)

    lead_in, pieces = split_at(unit.lines, starts)
    paragraphs = [
        Chunk(unit=unit, paragraph=str(number), lines=piece, lead_in=lead_in)
        for number, piece in enumerate(pieces, start=1)
    ]
    return [chunk for paragraph in paragraphs for chunk in split_points(paragraph, plan)]


def chunk_units(units: list[Unit], plan: SplitPlan | None = None) -> list[Chunk]:
    """Chunk every unit, keeping document order; follow `plan` if given."""
    return [
        chunk
        for unit in units
        for chunk in chunk_unit(unit, plan.get((unit.kind, unit.number)) if plan else None)
    ]


def chunk_language(language: Language) -> list[Chunk]:
    """Parse and chunk one language version, aligned with the English chunk structure."""
    english = chunk_units(split_units(extract_lines(ENGLISH), ENGLISH))
    if language is ENGLISH:
        return english
    return chunk_units(split_units(extract_lines(language), language), split_plan(english))


def main() -> None:
    language = language_from_args()
    chunks = chunk_language(language)
    print(f"Language: {language.code}")
    print(f"Chunks: {len(chunks)}")
    print(Counter(chunk.unit.kind for chunk in chunks))

    sizes = sorted(word_count(chunk.text) for chunk in chunks)
    print(f"Words per chunk: median {sizes[len(sizes) // 2]}, largest {sizes[-1]}")
    too_long = [c for c in chunks if word_count(c.text) > MAX_WORDS]
    print(f"Still over {MAX_WORDS} words: {[(c.citation, word_count(c.text)) for c in too_long]}")

    for key in ("Article 3(12)", "Article 5(1)(f)", "Article 66, point (a)", "Annex III, point 4"):
        chunk = next(c for c in chunks if c.unit_key == key)
        print(f"--- {chunk.citation} (pages {chunk.first_page}-{chunk.last_page}) ---")
        print(chunk.text[:200])


if __name__ == "__main__":
    main()
