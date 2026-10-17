import sys
import time

from llama_index.core.retrievers import BaseRetriever

from ai_act_rag.evaluation.questions import QUESTIONS, EvalQuestion
from ai_act_rag.generation.answer import CITED, REFUSAL, query_engine, unsupported_citations
from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.hybrid import hybrid_retriever
from ai_act_rag.retrieval.keyword import keyword_retriever, load_nodes
from ai_act_rag.retrieval.references import ReferenceRetriever, matches
from ai_act_rag.retrieval.vector import vector_retriever

K_VALUES = (1, 3, 5, 10)  # cut-offs to report hit rate at
MAX_K = max(K_VALUES)


def first_hit_rank(citations: list[str], expected: tuple[str, ...]) -> int | None:
    """The 1-based rank of the first retrieved citation that answers the question, if any."""
    for rank, citation in enumerate(citations, start=1):
        if any(matches(citation, e) for e in expected):
            return rank
    return None


def evaluate_retriever(name: str, retriever: BaseRetriever, questions: list[EvalQuestion]) -> None:
    """Print hit rate at each k and MRR, then list the questions it missed in the top 5."""
    ranks = []
    for q in questions:
        citations = [r.node.metadata["citation"] for r in retriever.retrieve(q.question)]
        ranks.append(first_hit_rank(citations, q.expected))

    hit_rates = "  ".join(
        f"hit@{k} {sum(r is not None and r <= k for r in ranks) / len(ranks):.0%}"
        for k in K_VALUES
    )
    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
    print(f"{name:<8} {hit_rates}  MRR {mrr:.2f}")
    missed = [q.question for q, r in zip(questions, ranks) if r is None or r > 5]
    if missed:
        print(f"         missed in top 5: {missed}")


def evaluate_answers(questions: list[EvalQuestion]) -> None:
    """Run the full RAG pipeline and check refusals and citations (slow: ~20 s per question)."""
    engine = query_engine()
    correct_refusals = cites_expected = unsupported = 0
    in_scope = [q for q in questions if q.expected]
    out_of_scope = [q for q in questions if not q.expected]

    for i, q in enumerate(questions, start=1):
        start = time.time()
        response = engine.query(q.question)
        answer = str(response).strip()
        refused = REFUSAL in answer
        bad = unsupported_citations(response)
        unsupported += len(bad)
        if q.expected:
            ok = any(any(matches(c, e) for e in q.expected) for c in CITED.findall(answer))
            cites_expected += ok
            status = "cites expected" if ok else ("REFUSED" if refused else "MISSES expected")
        else:
            correct_refusals += refused
            status = "refused (good)" if refused else "ANSWERED (should refuse)"
        flag = f", unsupported: {bad}" if bad else ""
        print(f"[{i}/{len(questions)}] {time.time() - start:4.0f} s  {status}{flag}  — {q.question}")

    print(f"\nAnswers citing an expected unit: {cites_expected}/{len(in_scope)}")
    print(f"Correct refusals: {correct_refusals}/{len(out_of_scope)}")
    print(f"Unsupported citations (cited but not retrieved): {unsupported}")


def main() -> None:
    configure_settings()
    in_scope = [q for q in QUESTIONS if q.expected]
    print(f"Retrieval on {len(in_scope)} questions (hit@k = right unit in the top k)\n")
    nodes = load_nodes()
    evaluate_retriever("vector", vector_retriever(MAX_K), in_scope)
    evaluate_retriever("keyword", keyword_retriever(nodes, MAX_K), in_scope)
    evaluate_retriever("hybrid", hybrid_retriever(MAX_K), in_scope)
    vector_refs = ReferenceRetriever(vector_retriever(MAX_K), nodes, MAX_K)
    evaluate_retriever("vec+refs", vector_refs, in_scope)
    hybrid_refs = ReferenceRetriever(hybrid_retriever(MAX_K), nodes, MAX_K)
    evaluate_retriever("hyb+refs", hybrid_refs, in_scope)

    if "--answers" in sys.argv:
        print(f"\nAnswers on all {len(QUESTIONS)} questions\n")
        evaluate_answers(QUESTIONS)
    else:
        print("\n(Add --answers to also test the LLM's answers; takes ~10 minutes.)")


if __name__ == "__main__":
    main()
