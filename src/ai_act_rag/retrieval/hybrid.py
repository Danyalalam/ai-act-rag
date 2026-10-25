from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever

from ai_act_rag.languages import detect_language
from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.keyword import keyword_retriever, load_nodes
from ai_act_rag.retrieval.vector import TOP_K, question_from_args, show, vector_retriever

# Each retriever proposes this many candidates; fusion keeps the best TOP_K.
CANDIDATES = 20


def hybrid_retriever(top_k: int = TOP_K, language: str = "en") -> BaseRetriever:
    """Vector search and BM25 keyword search, merged with reciprocal rank fusion (RRF).

    RRF scores each chunk by sum(1 / (60 + rank)) over both result lists, so a chunk
    ranked high by either search rises to the top. Scores are ranks, not similarities.
    """
    return QueryFusionRetriever(
        [
            vector_retriever(CANDIDATES, language),
            keyword_retriever(load_nodes(language), CANDIDATES, language),
        ],
        mode="reciprocal_rerank",
        similarity_top_k=top_k,
        num_queries=1,  # search with the question as asked; >1 would ask an LLM to rephrase it
        use_async=False,
    )


def main() -> None:
    configure_settings()
    question = question_from_args("Is emotion recognition at work allowed?")
    language = detect_language(question).code
    print(f"Question ({language}): {question}\n")
    show(hybrid_retriever(language=language).retrieve(question))


if __name__ == "__main__":
    main()
