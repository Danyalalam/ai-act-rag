import sys

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from ai_act_rag.config import COLLECTION_NAME, qdrant_client
from ai_act_rag.models import configure_settings

TOP_K = 5  # how many chunks to return per question


def vector_store() -> QdrantVectorStore:
    """The Qdrant collection holding the embedded chunks."""
    return QdrantVectorStore(collection_name=COLLECTION_NAME, client=qdrant_client())


def vector_retriever(top_k: int = TOP_K) -> BaseRetriever:
    """A retriever that embeds the question and finds the nearest chunks in Qdrant."""
    index = VectorStoreIndex.from_vector_store(vector_store())
    return index.as_retriever(similarity_top_k=top_k)


def show(results: list[NodeWithScore]) -> None:
    """Print each result's score, citation, pages and the start of its text."""
    for result in results:
        meta = result.node.metadata
        print(f"{result.score:.3f}  {meta['citation']} (p. {meta['pages']}) — {meta['title']}")
        print(f"       {result.node.get_content()[:120].replace(chr(10), ' ')}")


def question_from_args(default: str) -> str:
    """The question given on the command line, or `default` if there is none."""
    return " ".join(sys.argv[1:]) or default


def main() -> None:
    configure_settings()
    question = question_from_args("What obligations do deployers of high-risk AI systems have?")
    print(f"Question: {question}\n")
    show(vector_retriever().retrieve(question))


if __name__ == "__main__":
    main()
