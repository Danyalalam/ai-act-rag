import itertools
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from llama_index.core.base.response.schema import StreamingResponse as StreamingResponseType
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ai_act_rag.generation.answer import REFUSAL, query_engine, unsupported_citations
from ai_act_rag.models import configure_settings

PAGE = Path(__file__).parent / "static" / "index.html"

# Built once at startup (loads the chunks from Qdrant), then reused for every question.
engine = None
# One answer at a time: two LLM calls in parallel would overload a 16 GB laptop.
llm_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Code before `yield` runs when the server starts, code after it when it stops."""
    global engine
    configure_settings()
    engine = query_engine(streaming=True)
    yield


app = FastAPI(title="AI Act RAG", lifespan=lifespan)


class Question(BaseModel):
    """The request body: {"question": "..."}. Pydantic rejects anything else with a 422 error."""

    question: str = Field(min_length=3, max_length=500)


def event(kind: str, **data: object) -> str:
    """One message of the stream: a JSON object on its own line (NDJSON)."""
    return json.dumps({"type": kind, **data}) + "\n"


class StartedAnswer:
    """A streaming response whose first token has already been received."""

    def __init__(self, response: StreamingResponseType, first: str) -> None:
        self.source_nodes = response.source_nodes
        self.tokens = itertools.chain([first], response.response_gen)


def query_until_first_token(question: str, attempts: int = 2) -> StartedAnswer:
    """Start an answer, retrying if the model fails before writing anything.

    Ollama's GPU (MLX) runner sometimes crashes while reading a prompt; in the
    evaluation 5 of 38 questions failed once and all succeeded on the retry.
    Retrying is only safe before any token has been sent to the browser.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = engine.query(question)
            first = next(response.response_gen, "")
            return StartedAnswer(response, first)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2)  # give Ollama a moment to restart its runner
    raise AssertionError("unreachable")


def answer_events(question: str) -> Iterator[str]:
    """Stream an answer as events: "sources" first, then "token"s, then "done" (or "error").

    Retrieval finishes before the first token, so the page can show the sources while
    the model is still writing.
    """
    start = time.time()
    with llm_lock:
        try:
            response = query_until_first_token(question)
            sources = [
                {
                    "citation": s.node.metadata["citation"],
                    "title": s.node.metadata["title"],
                    "pages": s.node.metadata["pages"],
                    "text": s.node.get_content(),
                }
                for s in response.source_nodes
            ]
            yield event("sources", sources=sources)

            answer = ""
            for token in response.tokens:
                answer += token
                yield event("token", text=token)
        except Exception as error:  # usually Ollama not running
            yield event("error", message=f"The model is not available: {error}")
            return

    yield event(
        "done",
        refused=REFUSAL in answer,
        unsupported_citations=unsupported_citations(answer, response.source_nodes),
        seconds=round(time.time() - start, 1),
    )


@app.get("/")
def page() -> FileResponse:
    """The web page. `no-cache` makes the browser re-check it, so code changes show up on reload."""
    return FileResponse(PAGE, headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health() -> dict[str, str]:
    """A quick check that the server is up."""
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question) -> StreamingResponse:
    """Answer a question about the AI Act as a stream of JSON lines.

    FastAPI runs the (plain, non-async) generator in a worker thread, so the slow
    LLM doesn't block the server.
    """
    return StreamingResponse(
        answer_events(body.question.strip()), media_type="application/x-ndjson"
    )
