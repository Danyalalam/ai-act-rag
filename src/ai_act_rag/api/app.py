import base64
import binascii
import itertools
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from llama_index.core.base.response.schema import StreamingResponse as StreamingResponseType
from pydantic import BaseModel, Field

from ai_act_rag.generation.answer import is_refusal, query_engine, unsupported_citations
from ai_act_rag.languages import LANGUAGES, detect_language
from ai_act_rag.models import configure_settings
from ai_act_rag.ocr.read import page_images, read_page

PAGE = Path(__file__).parent / "static" / "index.html"
MAX_UPLOAD_BASE64 = 28_000_000  # characters; base64 is 4/3 the file size, so ~20 MB files

# One engine per language, built once at startup (loads the chunks from Qdrant),
# then reused for every question: {"en": ..., "de": ...}.
engines: dict = {}
# One answer at a time: two LLM calls in parallel would overload a 16 GB laptop.
llm_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Code before `yield` runs when the server starts, code after it when it stops."""
    configure_settings()
    for code in LANGUAGES:
        engines[code] = query_engine(streaming=True, language=code)
    yield


app = FastAPI(title="AI Act RAG", lifespan=lifespan)


class Question(BaseModel):
    """The request body: {"question": "..."}. Pydantic rejects anything else with a 422 error."""

    question: str = Field(min_length=3, max_length=500)
    # "auto" detects German or English from the question; the answer then comes only
    # from that language's text, with its citations and page numbers.
    language: Literal["auto", "en", "de"] = "auto"


class Document(BaseModel):
    """An image or PDF to read: {"filename": "scan.png", "data": "<base64>"}.

    Sent as base64 inside JSON rather than as a form upload, so no extra package
    (python-multipart) is needed.
    """

    filename: str = Field(min_length=1, max_length=200)
    data: str = Field(min_length=1, max_length=MAX_UPLOAD_BASE64)


def event(kind: str, **data: object) -> str:
    """One message of the stream: a JSON object on its own line (NDJSON)."""
    return json.dumps({"type": kind, **data}) + "\n"


class StartedAnswer:
    """A streaming response whose first token has already been received."""

    def __init__(self, response: StreamingResponseType, first: str) -> None:
        self.source_nodes = response.source_nodes
        self.tokens = itertools.chain([first], response.response_gen)


def query_until_first_token(question: str, language: str, attempts: int = 2) -> StartedAnswer:
    """Start an answer, retrying if the model fails before writing anything.

    Ollama's GPU (MLX) runner sometimes crashes while reading a prompt; in the
    evaluation 5 of 38 questions failed once and all succeeded on the retry.
    Retrying is only safe before any token has been sent to the browser.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = engines[language].query(question)
            first = next(response.response_gen, "")
            return StartedAnswer(response, first)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2)  # give Ollama a moment to restart its runner
    raise AssertionError("unreachable")


def answer_events(question: str, language: str) -> Iterator[str]:
    """Stream an answer as events: "sources" first, then "token"s, then "done" (or "error").

    Retrieval finishes before the first token, so the page can show the sources while
    the model is still writing.
    """
    start = time.time()
    with llm_lock:
        try:
            response = query_until_first_token(question, language)
            sources = [
                {
                    "citation": s.node.metadata["citation"],
                    "title": s.node.metadata["title"],
                    "pages": s.node.metadata["pages"],
                    "text": s.node.get_content(),
                }
                for s in response.source_nodes
            ]
            yield event("sources", sources=sources, language=language)

            answer = ""
            for token in response.tokens:
                answer += token
                yield event("token", text=token)
        except Exception as error:  # usually Ollama not running
            yield event("error", message=f"The model is not available: {error}")
            return

    yield event(
        "done",
        refused=is_refusal(answer),
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
    question = body.question.strip()
    language = detect_language(question).code if body.language == "auto" else body.language
    return StreamingResponse(
        answer_events(question, language), media_type="application/x-ndjson"
    )

## The following is a workaround for a bug in FastAPI 0.100.0: if the server is started


def ocr_events(images: list[bytes]) -> Iterator[str]:
    """Stream OCR of every page: "start", then per page a "page" and its "token"s, then "done"."""
    start = time.time()
    yield event("start", pages=len(images))
    with llm_lock:  # shares the lock: OCR and answers both load models into the same RAM
        try:
            for number, image in enumerate(images, start=1):
                yield event("page", page=number)
                for piece in read_page(image):
                    yield event("token", text=piece)
        except Exception as error:  # usually Ollama not running or the model not pulled
            yield event("error", message=f"The OCR model is not available: {error}")
            return
    yield event("done", seconds=round(time.time() - start, 1))


@app.post("/ocr")
def ocr(body: Document) -> StreamingResponse:
    """Read the text of an image or PDF with the local OCR model, streamed as JSON lines."""
    try:
        data = base64.b64decode(body.data, validate=True)
        images = page_images(data, body.filename)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:  # e.g. a file that claims to be a PDF but isn't
        raise HTTPException(400, f"Could not open {body.filename}: {error}") from error
    return StreamingResponse(ocr_events(images), media_type="application/x-ndjson")
