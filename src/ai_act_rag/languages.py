"""Everything that differs between language versions of the AI Act.

The EU publishes the regulation in every official language with the same structure
and numbering, so the parser, chunker and retrieval logic are shared; only the
words below change. Adding a language = adding one `Language` here.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Language:
    code: str  # "en", "de"
    pdf_path: Path
    stemmer: str  # language name for the BM25 stemmer and stop words
    # Lines the Official Journal repeats on every page (besides page counters like "12/144").
    page_furniture: frozenset[str]
    # Words that start each part of the regulation.
    article: str  # "Article 26"
    chapter: str  # "CHAPTER III"
    section: str  # "SECTION 2"
    annex: str  # "ANNEX III"
    enacting_formula: str  # the line between the recitals and Article 1
    signature_start: str  # the line after Article 113
    # How numbered items open a line ("{}" = the number or letter). The two languages differ:
    # English paragraphs are "1." and definitions "(1)"; German is the other way round.
    article_paragraph: str  # "{}." -> "1."
    definition: str  # Article 3's numbered definitions
    point: str  # "({})" -> "(a)"
    page_abbreviation: str  # "p." in "[Article 26(1), p. 67]"
    refusal: str  # the exact sentence the LLM uses when it can't answer

    def citation(self, kind: str, number: str, paragraph: str | None, point: str | None) -> str:
        """How a unit is cited in this language."""
        if self.code == "de":
            return german_citation(kind, number, paragraph, point)
        return english_citation(kind, number, paragraph, point)


def english_citation(kind: str, number: str, paragraph: str | None, point: str | None) -> str:
    """ "Article 26(1)", "Article 5(1)(f)", "Article 66, point (a)", "Annex III, point 4(a)"."""
    if kind == "recital":
        return f"Recital ({number})"
    letter = f"({point})" if point else ""
    if kind == "article":
        if paragraph:
            return f"Article {number}({paragraph}){letter}"
        return f"Article {number}, point {letter}" if point else f"Article {number}"
    if paragraph or point:
        return f"Annex {number}, point {paragraph or ''}{letter}"
    return f"Annex {number}"


def german_citation(kind: str, number: str, paragraph: str | None, point: str | None) -> str:
    """German legal style: "Artikel 26 Absatz 1", "Artikel 5 Absatz 1 Buchstabe f",
    "Artikel 3 Nummer 12" (definitions are numbered), "Anhang III Nummer 4 Buchstabe a"."""
    if kind == "recital":
        return f"Erwägungsgrund {number}"
    letter = f" Buchstabe {point}" if point else ""
    if kind == "article":
        if paragraph:
            word = "Nummer" if number == "3" else "Absatz"
            return f"Artikel {number} {word} {paragraph}{letter}"
        return f"Artikel {number}{letter}"
    if paragraph:
        return f"Anhang {number} Nummer {paragraph}{letter}"
    return f"Anhang {number}{letter}"


ENGLISH = Language(
    code="en",
    pdf_path=Path("data/ai-act.pdf"),
    stemmer="english",
    page_furniture=frozenset(
        {"OJ L, 12.7.2024", "EN", "ELI: http://data.europa.eu/eli/reg/2024/1689/oj"}
    ),
    article="Article",
    chapter="CHAPTER",
    section="SECTION",
    annex="ANNEX",
    enacting_formula="HAVE ADOPTED THIS REGULATION:",
    signature_start="Done at ",
    article_paragraph="{}.",
    definition="({})",
    point="({})",
    page_abbreviation="p.",
    refusal="I can't answer that from the parts of the AI Act I retrieved.",
)

GERMAN = Language(
    code="de",
    pdf_path=Path("data/OJ_L_202401689_DE_TXT.pdf"),
    stemmer="german",
    page_furniture=frozenset(
        {"ABl. L vom 12.7.2024", "DE", "ELI: http://data.europa.eu/eli/reg/2024/1689/oj"}
    ),
    article="Artikel",
    chapter="KAPITEL",
    section="ABSCHNITT",
    annex="ANHANG",
    enacting_formula="HABEN FOLGENDE VERORDNUNG ERLASSEN:",
    signature_start="Geschehen zu ",
    article_paragraph="({})",
    definition="{}.",
    point="{})",
    page_abbreviation="S.",
    refusal="Das kann ich anhand der abgerufenen Teile der KI-Verordnung nicht beantworten.",
)

LANGUAGES = {language.code: language for language in (ENGLISH, GERMAN)}

# Very common words that only appear in one of the two languages.
GERMAN_HINTS = re.compile(
    r"\b(der|die|das|und|ist|sind|wer|wie|was|wann|welche|nicht|für|mit|von|darf|müssen|"
    r"gilt|eine?n?|im|zu|auf|bei|über|ki|artikel|anhang|erwägungsgrund|verordnung|"
    r"besagt|fasse|zusammen)\b|[äöüß]",
    re.IGNORECASE,
)
ENGLISH_HINTS = re.compile(
    r"\b(the|and|is|are|who|how|what|when|which|not|for|with|of|can|must|does|do|a|an|in|to|on|"
    r"article|annex|recital|regulation|say|says|summarise|summarize)\b",
    re.IGNORECASE,
)


def detect_language(question: str) -> Language:
    """Guess whether a question is German or English by counting common words."""
    german = len(GERMAN_HINTS.findall(question))
    english = len(ENGLISH_HINTS.findall(question))
    return GERMAN if german > english else ENGLISH
