import re

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from ai_act_rag.languages import detect_language
from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.keyword import load_nodes
from ai_act_rag.retrieval.vector import TOP_K, question_from_args, show, vector_retriever

# English: "Article 26", "article 26(1)", "Art. 5(1)(f)"
ARTICLE_REF = re.compile(r"\b(?:article|art\.)\s*(\d+)((?:\(\d+\))?(?:\([a-z]\))?)", re.IGNORECASE)
# "Annex III", "annex iii"
ANNEX_REF = re.compile(r"\bannex\s+([ivxl]+)\b", re.IGNORECASE)
# "Recital 27", "recital (27)"
RECITAL_REF = re.compile(r"\brecital\s+\(?(\d+)\)?", re.IGNORECASE)
# German: "Artikel 26 Absatz 1", "Art. 5 Abs. 1 Buchst. f", "Artikel 3 Nummer 12"
GERMAN_ARTICLE_REF = re.compile(
    r"\b(?:artikel|art\.)\s*(\d+)"
    r"(?:\s+(?:absatz|abs\.|nummer|nr\.)\s*(\d+))?"
    r"(?:\s+(?:buchstabe|buchst\.|lit\.)\s*([a-z])\b)?",
    re.IGNORECASE,
)
# "Anhang III", "Anhang III Nummer 4", "Anhang III Nr. 4 Buchstabe a"
GERMAN_ANNEX_REF = re.compile(
    r"\banhang\s+([ivxl]+)\b(?:\s+(?:nummer|nr\.)\s*(\d+))?(?:\s+(?:buchstabe|buchst\.)\s*([a-z])\b)?",
    re.IGNORECASE,
)
# "Erwägungsgrund 27"
GERMAN_RECITAL_REF = re.compile(r"\berwägungsgr(?:und|ünde)\s+\(?(\d+)\)?", re.IGNORECASE)


def matches(citation: str, reference: str) -> bool:
    """True if `citation` is `reference` or a part of it.

    "Article 5(1)(f)" is part of "Article 5"; "Artikel 5 Absatz 1" is part of "Artikel 5"
    (but "Article 50" is not part of "Article 5").
    """
    return citation == reference or citation.startswith(
        (reference + "(", reference + ",", reference + " ")
    )


def find_references(question: str) -> list[str]:
    """The legal units a question names, as language-neutral unit keys.

    "Article 26(1)", "Artikel 26 Absatz 1" and "Art. 26 Abs. 1" all give "Article 26(1)";
    "Anhang III Nummer 4" gives "Annex III, point 4".
    """
    references = []
    for n, para, letter in GERMAN_ARTICLE_REF.findall(question):
        references.append(f"Article {n}" + (f"({para})" if para else "") + (f"({letter})" if letter else ""))
    for n, sub in ARTICLE_REF.findall(question):
        references.append(f"Article {n}{sub}")
    for roman, point, letter in GERMAN_ANNEX_REF.findall(question):
        tail = f", point {point}" + (f"({letter})" if letter else "") if point else ""
        references.append(f"Annex {roman.upper()}{tail}")
    references += [f"Annex {roman.upper()}" for roman in ANNEX_REF.findall(question)]
    references += [f"Recital ({n})" for n in RECITAL_REF.findall(question)]
    references += [f"Recital ({n})" for n in GERMAN_RECITAL_REF.findall(question)]
    unique = list(dict.fromkeys(references))  # remove duplicates, keep order
    # "Art. 5 Abs. 1 Buchst. f" also matches the English pattern as plain "Article 5";
    # keep only the most specific reference to a unit.
    return [r for r in unique if not any(s != r and matches(s, r) for s in unique)]


class ReferenceRetriever(BaseRetriever):
    """Puts the chunks of any unit named in the question first, then the base retriever's results.

    Search can't find "Recital 27" by itself: the number lives in metadata, not in the text.
    """

    def __init__(self, base: BaseRetriever, nodes: list[BaseNode], top_k: int = TOP_K) -> None:
        super().__init__()
        self.base = base
        self.nodes = nodes
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        references = find_references(query_bundle.query_str)
        direct = [
            NodeWithScore(node=node, score=1.0)
            for node in self.nodes
            if any(matches(node.metadata["unit_key"], ref) for ref in references)
        ]
        seen = {result.node.node_id for result in direct}
        searched = [r for r in self.base.retrieve(query_bundle) if r.node.node_id not in seen]
        return (direct + searched)[: self.top_k]


def default_retriever(top_k: int = TOP_K, language: str = "en") -> BaseRetriever:
    """The retriever the RAG pipeline uses: reference lookup, then vector search.

    The evaluation (see notes.md) showed vector search beats BM25 and hybrid fusion
    on natural questions, and reference lookup fixes questions that name a unit.
    """
    return ReferenceRetriever(vector_retriever(top_k, language), load_nodes(language), top_k)


def main() -> None:
    configure_settings()
    question = question_from_args("What does Recital 27 say?")
    language = detect_language(question).code
    print(f"Question ({language}): {question}")
    print(f"References found: {find_references(question)}\n")
    show(default_retriever(language=language).retrieve(question))


if __name__ == "__main__":
    main()
