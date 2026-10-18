import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever

from ai_act_rag.evaluation.questions import QUESTIONS, EvalQuestion
from ai_act_rag.generation.answer import (
    REFUSAL,
    TOP_K,
    cited_units,
    query_engine,
    unsupported_citations,
)
from ai_act_rag.models import EMBED_MODEL, LLM_MODEL, configure_settings
from ai_act_rag.retrieval.hybrid import hybrid_retriever
from ai_act_rag.retrieval.keyword import keyword_retriever, load_nodes
from ai_act_rag.retrieval.references import ReferenceRetriever, matches
from ai_act_rag.retrieval.vector import vector_retriever

K_VALUES = (1, 3, 5, 10)  # cut-offs to report hit rate at
MAX_K = max(K_VALUES)
RESULTS_DIR = Path("results")


def first_hit_rank(citations: list[str], expected: tuple[str, ...]) -> int | None:
    """The 1-based rank of the first retrieved citation that answers the question, if any."""
    for rank, citation in enumerate(citations, start=1):
        if any(matches(citation, e) for e in expected):
            return rank
    return None


def evaluate_retriever(
    name: str, retriever: BaseRetriever, questions: list[EvalQuestion]
) -> dict:
    """Score a retriever: hit rate at each k, MRR, and the rank for every question."""
    ranks = []
    for q in questions:
        citations = [r.node.metadata["citation"] for r in retriever.retrieve(q.question)]
        ranks.append(first_hit_rank(citations, q.expected))

    hit_rates = {
        f"hit@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in K_VALUES
    }
    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
    missed = [q.question for q, r in zip(questions, ranks) if r is None or r > 5]

    shown = "  ".join(f"{k} {v:.0%}" for k, v in hit_rates.items())
    print(f"{name:<8} {shown}  MRR {mrr:.2f}", flush=True)
    if missed:
        print(f"         missed in top 5: {missed}", flush=True)

    return {
        **hit_rates,
        "mrr": mrr,
        "missed_in_top_5": missed,
        "ranks": {q.question: r for q, r in zip(questions, ranks)},
    }


def evaluate_answers(questions: list[EvalQuestion], save: "callable") -> dict:
    """Run the full RAG pipeline on every question and check refusals and citations.

    Slow (~20 s per question). `save` is called after each answer, so an
    interrupted run still keeps everything answered so far.
    """
    engine = query_engine()
    rows = []
    summary: dict = {}

    for i, q in enumerate(questions, start=1):
        start = time.time()
        response = query_with_retry(engine, q.question)
        if response is None:
            print(f"[{i}/{len(questions)}] ERROR (model failed twice)  — {q.question}", flush=True)
            rows.append({"question": q.question, "expected": list(q.expected), "status": "error",
                         "answer": "", "cited": [], "sources": [], "unsupported_citations": [],
                         "seconds": round(time.time() - start, 1)})
            continue
        answer = str(response).strip()
        cited = cited_units(answer)
        refused = REFUSAL in answer
        unsupported = unsupported_citations(answer, response.source_nodes)

        if q.expected:
            cites_expected = any(matches(c, e) for c in cited for e in q.expected)
            if cites_expected:
                status = "cites expected"
            else:
                status = "refused" if refused else "misses expected"
        else:
            cites_expected = None
            status = "refused (good)" if refused else "answered (should refuse)"

        seconds = round(time.time() - start, 1)
        flag = f", unsupported: {unsupported}" if unsupported else ""
        print(f"[{i}/{len(questions)}] {seconds:5.1f} s  {status}{flag}  — {q.question}", flush=True)

        rows.append(
            {
                "question": q.question,
                "expected": list(q.expected),
                "status": status,
                "answer": answer,
                "cited": cited,
                "sources": [s.node.metadata["citation"] for s in response.source_nodes],
                "unsupported_citations": unsupported,
                "seconds": seconds,
            }
        )
        summary = summarise_answers(rows, questions)
        save({"summary": summary, "per_question": rows})

    return {"summary": summary, "per_question": rows}


def query_with_retry(engine: RetrieverQueryEngine, question: str, attempts: int = 2):
    """Ask the engine, retrying once: Ollama's GPU runner sometimes crashes on long prompts."""
    for attempt in range(1, attempts + 1):
        try:
            return engine.query(question)
        except Exception as error:
            print(f"   attempt {attempt} failed: {str(error)[:120]}", flush=True)
            time.sleep(5)  # give Ollama a moment to restart its runner
    return None


def summarise_answers(rows: list[dict], questions: list[EvalQuestion]) -> dict:
    """Totals over the answers so far."""
    in_scope = [r for r in rows if r["expected"]]
    out_of_scope = [r for r in rows if not r["expected"]]
    return {
        "answered": len(rows),
        "total": len(questions),
        "cites_expected": sum(r["status"] == "cites expected" for r in in_scope),
        "in_scope": len(in_scope),
        "wrongly_refused": sum(r["status"] == "refused" for r in in_scope),
        "correct_refusals": sum(r["status"] == "refused (good)" for r in out_of_scope),
        "out_of_scope": len(out_of_scope),
        "errors": sum(r["status"] == "error" for r in rows),
        "answers_with_unsupported_citations": sum(bool(r["unsupported_citations"]) for r in rows),
        "average_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
    }


def git_commit() -> str:
    """The current git commit, so results can be traced back to the code that produced them."""
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def main() -> None:
    configure_settings()
    started = datetime.now()
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"eval-{started:%Y-%m-%d-%H%M}.json"
    results: dict = {
        "date": started.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": {"embed_model": EMBED_MODEL, "llm": LLM_MODEL, "answer_top_k": TOP_K},
        "retrieval": {},
    }

    def save(answers: dict | None = None) -> None:
        if answers is not None:
            results["answers"] = answers
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    in_scope = [q for q in QUESTIONS if q.expected]
    print(f"Retrieval on {len(in_scope)} questions (hit@k = right unit in the top k)\n", flush=True)
    nodes = load_nodes()
    retrievers = {
        "vector": vector_retriever(MAX_K),
        "keyword": keyword_retriever(nodes, MAX_K),
        "hybrid": hybrid_retriever(MAX_K),
        "vec+refs": ReferenceRetriever(vector_retriever(MAX_K), nodes, MAX_K),
        "hyb+refs": ReferenceRetriever(hybrid_retriever(MAX_K), nodes, MAX_K),
    }
    for name, retriever in retrievers.items():
        results["retrieval"][name] = evaluate_retriever(name, retriever, in_scope)
    save()

    if "--answers" in sys.argv:
        print(f"\nAnswers on all {len(QUESTIONS)} questions (saved after each one)\n", flush=True)
        answers = evaluate_answers(QUESTIONS, save)
        s = answers["summary"]
        print(f"\nAnswers citing an expected unit: {s['cites_expected']}/{s['in_scope']}")
        print(f"Wrongly refused: {s['wrongly_refused']}/{s['in_scope']}")
        print(f"Correct refusals: {s['correct_refusals']}/{s['out_of_scope']}")
        print(f"Answers with unsupported citations: {s['answers_with_unsupported_citations']}")
        print(f"Errors (model failed twice): {s['errors']}")
        print(f"Average time per answer: {s['average_seconds']} s")
    else:
        print("\n(Add --answers to also test the LLM's answers; takes ~15-20 minutes.)")

    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
