import re
import time

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

from ai_act_rag.models import configure_settings
from ai_act_rag.retrieval.references import default_retriever, matches
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

# A bracket in the answer, e.g. "[Article 5(2), p. 52; Recital (35), p. 10]".
BRACKET = re.compile(r"\[([^\[\]]+)\]")
# One citation inside a bracket, followed by its page: "Article 26(1)", "Annex III, point 4",
# "Recital (27)". Extra words before ", p." (e.g. "Article 52(2) and (3)") are skipped.
CITATION = re.compile(
    r"(Article \d+(?:\([^()\s]+\))*|Annex [IVXL]+(?:, point [\w()]+)?|Recital \(\d+\))"
    r"[^\[\];]*?, p\. [\d\-–]+"
)


def cited_units(answer: str) -> list[str]:
    """The units an answer cites, e.g. ["Article 5(1)(f)", "Recital (44)"]."""
    return [c for bracket in BRACKET.findall(answer) for c in CITATION.findall(bracket)]


def query_engine(streaming: bool = False) -> RetrieverQueryEngine:
    """Reference lookup + vector retrieval + the local LLM, answering with citations.

    With `streaming=True`, `.query()` returns as soon as retrieval is done; the answer
    then arrives token by token from `response.response_gen`.
    """
    return RetrieverQueryEngine.from_args(
        default_retriever(top_k=TOP_K),
        text_qa_template=QA_PROMPT,
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
    print(f"Question: {question}\n")

    start = time.time()
    response = query_engine(streaming=True).query(question)
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
