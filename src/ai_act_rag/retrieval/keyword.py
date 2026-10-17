from llama_index.core.schema import BaseNode
from llama_index.retrievers.bm25 import BM25Retriever

from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.vector import TOP_K, question_from_args, show, vector_store

MAX_NODES = 10_000  # far more than the ~700 chunks, so we always load them all


def load_nodes() -> list[BaseNode]:
    """Read every chunk (text + metadata) back from Qdrant, the single source of truth."""
    return vector_store().get_nodes(limit=MAX_NODES)


def keyword_retriever(nodes: list[BaseNode], top_k: int = TOP_K) -> BM25Retriever:
    """A BM25 keyword retriever over the chunks, built in memory (takes well under a second).

    Words are stemmed ("applies", "application" -> "appli") and English stop words are ignored.
    """
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k, language="en")


def main() -> None:
    configure_settings()
    question = question_from_args("When does the regulation apply?")
    print(f"Question: {question}\n")
    show(keyword_retriever(load_nodes()).retrieve(question))


if __name__ == "__main__":
    main()
