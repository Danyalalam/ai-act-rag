from llama_index.core.schema import BaseNode
from llama_index.retrievers.bm25 import BM25Retriever

from ai_act_rag.languages import LANGUAGES, detect_language
from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.vector import TOP_K, question_from_args, show, vector_store

MAX_NODES = 10_000  # far more than the ~700 chunks, so we always load them all


def load_nodes(language: str | None = None) -> list[BaseNode]:
    """Read the chunks (text + metadata) back from Qdrant, the single source of truth.

    With `language`, only that language's chunks.
    """
    nodes = vector_store().get_nodes(limit=MAX_NODES)
    return [n for n in nodes if language is None or n.metadata["language"] == language]


def keyword_retriever(
    nodes: list[BaseNode], top_k: int = TOP_K, language: str = "en"
) -> BM25Retriever:
    """A BM25 keyword retriever over the chunks, built in memory (takes well under a second).

    Words are stemmed ("applies", "application" -> "appli") and stop words are ignored,
    using the rules for `language`; `nodes` should all be in that language.
    """
    return BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=top_k, language=LANGUAGES[language].stemmer
    )


def main() -> None:
    configure_settings()
    question = question_from_args("When does the regulation apply?")
    language = detect_language(question).code
    print(f"Question ({language}): {question}\n")
    show(keyword_retriever(load_nodes(language), language=language).retrieve(question))


if __name__ == "__main__":
    main()
