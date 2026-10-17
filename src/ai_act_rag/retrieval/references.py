import re

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.keyword import load_nodes
from ai_act_rag.retrieval.vector import TOP_K, question_from_args, show, vector_retriever

# "Article 26", "article 26(1)", "Article 5(1)(f)"
ARTICLE_REF = re.compile(r"\barticle\s+(\d+)((?:\(\d+\))?(?:\([a-z]\))?)", re.IGNORECASE)
# "Annex III", "annex iii"
ANNEX_REF = re.compile(r"\bannex\s+([ivxl]+)\b", re.IGNORECASE)
# "Recital 27", "recital (27)"
RECITAL_REF = re.compile(r"\brecital\s+\(?(\d+)\)?", re.IGNORECASE)


def matches(citation: str, reference: str) -> bool:
    """True if `citation` is `reference` or a part of it ("Article 5(1)(f)" is part of "Article 5")."""
    return citation == reference or citation.startswith((reference + "(", reference + ","))


def find_references(question: str) -> list[str]:
    """The legal units a question names, as citation prefixes: ["Article 26(1)", "Annex III"]."""
    references = [f"Article {n}{sub}" for n, sub in ARTICLE_REF.findall(question)]
    references += [f"Annex {roman.upper()}" for roman in ANNEX_REF.findall(question)]
    references += [f"Recital ({n})" for n in RECITAL_REF.findall(question)]
    return references


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
            if any(matches(node.metadata["citation"], ref) for ref in references)
        ]
        seen = {result.node.node_id for result in direct}
        searched = [r for r in self.base.retrieve(query_bundle) if r.node.node_id not in seen]
        return (direct + searched)[: self.top_k]


def default_retriever(top_k: int = TOP_K) -> BaseRetriever:
    """The retriever the RAG pipeline uses: reference lookup, then vector search.

    The evaluation (see notes.md) showed vector search beats BM25 and hybrid fusion
    on natural questions, and reference lookup fixes questions that name a unit.
    """
    return ReferenceRetriever(vector_retriever(top_k), load_nodes(), top_k)


def main() -> None:
    configure_settings()
    question = question_from_args("What does Recital 27 say?")
    print(f"Question: {question}")
    print(f"References found: {find_references(question)}\n")
    show(default_retriever().retrieve(question))


if __name__ == "__main__":
    main()
