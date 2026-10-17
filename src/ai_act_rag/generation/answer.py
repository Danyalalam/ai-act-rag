import re
import time

from llama_index.core import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import RetrieverQueryEngine

from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.references import default_retriever
from ai_act_rag.retrieval.vector import question_from_args

TOP_K = 6  # chunks given to the LLM as context

REFUSAL = "I can't answer that from the parts of the AI Act I retrieved."

# {context_str} is filled with the chunks, each preceded by its citation, title and pages;
# {query_str} with the question.
QA_PROMPT = PromptTemplate(
    "You answer questions about the EU AI Act (Regulation (EU) 2024/1689) using ONLY the "
    "excerpts below. Each excerpt starts with its citation, title and PDF pages.\n"
    "\n"
    "Rules:\n"
    "- Use only the excerpts. Do not use outside knowledge.\n"
    "- After every claim, cite its source in square brackets with the exact citation "
    "and page, e.g. [Article 26(1), p. 67].\n"
    "- Prefer Articles and Annexes over Recitals; recitals explain, but are not binding.\n"
    f'- If the excerpts do not answer the question, reply exactly: "{REFUSAL}"\n'
    "- Be concise.\n"
    "\n"
    "Excerpts:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Question: {query_str}\n"
    "Answer: "
)

CITED = re.compile(r"\[([^\[\]]+?), p\. [\d\-]+\]")


def query_engine() -> RetrieverQueryEngine:
    """Reference lookup + vector retrieval + the local LLM, answering with citations."""
    return RetrieverQueryEngine.from_args(
        default_retriever(top_k=TOP_K),
        text_qa_template=QA_PROMPT,
    )


def unsupported_citations(response: Response) -> list[str]:
    """Citations in the answer that don't match any retrieved chunk (possible hallucinations)."""
    retrieved = {source.node.metadata["citation"] for source in response.source_nodes}
    return [c for c in CITED.findall(str(response)) if c not in retrieved]


def main() -> None:
    configure_settings()
    question = question_from_args("Is emotion recognition at work allowed?")
    print(f"Question: {question}\n")

    start = time.time()
    response = query_engine().query(question)
    print(f"{response}\n")
    print(f"({time.time() - start:.0f} s)\n")

    print("Sources given to the LLM:")
    for source in response.source_nodes:
        meta = source.node.metadata
        print(f"  {meta['citation']} (p. {meta['pages']})")
    if bad := unsupported_citations(response):
        print(f"\nWarning: cited but not retrieved: {bad}")


if __name__ == "__main__":
    main()
