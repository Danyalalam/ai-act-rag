import re
import time

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

from ai_act_rag.languages import ENGLISH, GERMAN, LANGUAGES, detect_language
from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.references import default_retriever, matches
from ai_act_rag.retrieval.vector import question_from_args

TOP_K = 6  # chunks given to the LLM as context

REFUSAL = ENGLISH.refusal  # kept for code that only handles English
REFUSALS = tuple(language.refusal for language in LANGUAGES.values())

# {context_str} is filled with the chunks, each preceded by its citation, title and pages;
# {query_str} with the question. One prompt per language: the model follows instructions
# (and answers) most reliably in the language of the question.
PROMPTS = {
    "en": PromptTemplate(
        "You answer questions about the EU AI Act (Regulation (EU) 2024/1689) using ONLY the "
        "excerpts below. Each excerpt starts with its citation, title and PDF pages.\n"
        "\n"
        "Rules:\n"
        "- Use only the excerpts. Do not use outside knowledge.\n"
        "- After every claim, cite its source in square brackets with the exact citation "
        "and page, e.g. [Article 26(1), p. 67].\n"
        "- Prefer Articles and Annexes over Recitals; recitals explain, but are not binding.\n"
        f'- If the excerpts do not answer the question, reply exactly: "{ENGLISH.refusal}"\n'
        "- Be concise.\n"
        "\n"
        "Excerpts:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "\n"
        "Question: {query_str}\n"
        "Answer: "
    ),
    "de": PromptTemplate(
        "Du beantwortest Fragen zur KI-Verordnung (Verordnung (EU) 2024/1689) AUSSCHLIESSLICH "
        "anhand der folgenden Auszüge. Jeder Auszug beginnt mit Fundstelle (citation), "
        "Titel (title) und PDF-Seiten (pages).\n"
        "\n"
        "Regeln:\n"
        "- Verwende nur die Auszüge, kein Vorwissen.\n"
        "- Gib nach jeder Aussage die Quelle in eckigen Klammern mit genauer Fundstelle und "
        "Seite an, z. B. [Artikel 26 Absatz 1, S. 67].\n"
        "- Bevorzuge Artikel und Anhänge gegenüber Erwägungsgründen; Erwägungsgründe "
        "erläutern, sind aber nicht verbindlich.\n"
        f'- Wenn die Auszüge die Frage nicht beantworten, antworte genau: "{GERMAN.refusal}"\n'
        "- Antworte auf Deutsch und fasse dich kurz.\n"
        "\n"
        "Auszüge:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "\n"
        "Frage: {query_str}\n"
        "Antwort: "
    ),
}
QA_PROMPT = PROMPTS["en"]

# A bracket in the answer, e.g. "[Article 5(2), p. 52; Recital (35), p. 10]".
BRACKET = re.compile(r"\[([^\[\]]+)\]")
# One citation inside a bracket, followed by its page ("p." or German "S."):
# "Article 26(1)", "Annex III, point 4", "Recital (27)",
# "Artikel 26 Absatz 1", "Artikel 5 Absatz 1 Buchstabe f", "Anhang III Nummer 4", "Erwägungsgrund 27".
# Extra words before the page (e.g. "Article 52(2) and (3)") are skipped.
CITATION = re.compile(
    r"(Article \d+(?:\([^()\s]+\))*|Annex [IVXL]+(?:, point [\w()]+)?|Recital \(\d+\)"
    r"|Artikel \d+(?: (?:Absatz|Nummer) \d+)?(?: Buchstabe [a-z])?"
    r"|Anhang [IVXL]+(?: Nummer \d+)?(?: Buchstabe [a-z])?|Erwägungsgrund \d+)"
    r"[^\[\];]*?, (?:p|S)\. [\d\-–]+"
)


def cited_units(answer: str) -> list[str]:
    """The units an answer cites, e.g. ["Article 5(1)(f)", "Recital (44)"]."""
    return [c for bracket in BRACKET.findall(answer) for c in CITATION.findall(bracket)]


def is_refusal(answer: str) -> bool:
    """True if the answer contains the refusal sentence of any language."""
    return any(refusal in answer for refusal in REFUSALS)


def query_engine(streaming: bool = False, language: str = "en") -> RetrieverQueryEngine:
    """Reference lookup + vector retrieval + the local LLM, answering with citations.

    Retrieval only searches chunks in `language`, and the prompt is in that language,
    so a German question is answered from the German text with German citations and pages.
    With `streaming=True`, `.query()` returns as soon as retrieval is done; the answer
    then arrives token by token from `response.response_gen`.
    """
    return RetrieverQueryEngine.from_args(
        default_retriever(top_k=TOP_K, language=language),
        text_qa_template=PROMPTS[language],
        streaming=streaming,
    )


def unsupported_citations(answer: str, sources: list[NodeWithScore]) -> list[str]:
    """Cited units the model was not shown (possible hallucinations).

    A citation counts as supported when it is a retrieved unit, a part of one
    ("Article 9(2)(a)" when given "Article 9(2)"), or contains one ("Article 5" when
    given "Article 5(1)(f)"). Units only *mentioned* inside a retrieved text are flagged:
    the model saw the reference, not the provision itself.
    """
    retrieved = {source.node.metadata["citation"] for source in sources}
    return [
        cited
        for cited in cited_units(answer)
        if not any(matches(cited, r) or matches(r, cited) for r in retrieved)
    ]


def main() -> None:
    configure_settings()
    question = question_from_args("Is emotion recognition at work allowed?")
    language = detect_language(question).code
    print(f"Question ({language}): {question}\n")

    start = time.time()
    response = query_engine(streaming=True, language=language).query(question)
    answer = ""
    for token in response.response_gen:  # print each piece as soon as the model writes it
        print(token, end="", flush=True)
        answer += token
    print(f"\n\n({time.time() - start:.0f} s)\n")

    print("Sources given to the LLM:")
    for source in response.source_nodes:
        meta = source.node.metadata
        print(f"  {meta['citation']} (p. {meta['pages']})")
    if bad := unsupported_citations(answer, response.source_nodes):
        print(f"\nWarning: cited but not retrieved: {bad}")


if __name__ == "__main__":
    main()
