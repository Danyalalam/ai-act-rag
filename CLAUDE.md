# AI Act RAG

Retrieval-augmented question answering over the EU AI Act (Regulation (EU) 2024/1689).
Answers must cite the exact legal unit they come from (e.g. "Article 26(1)", "Annex III", "Recital (27)") and the PDF page.

## How I work on this project

I'm building this step by step to learn, and I'm new to macOS and the terminal.
- Explain every command and every flag in plain words before or next to it.
- Make one small change at a time, then tell me how to run it and what output to expect.
- Don't write large parts of the codebase in one go unless I ask; prefer explaining and letting me type it.
- Ask before adding a new dependency or any paid API.
- Keep `notes.md` (my interview prep) up to date: whenever we use a new tool, library, Python feature or design decision, add a short entry saying what it is and why we use it. Keep it brief and in interview-ready wording.

## Stack

- Python 3.12, managed with **uv** (never pip or a manually activated venv).
- LlamaIndex for indexing and retrieval.
- Qdrant Cloud as the vector database (free tier, cluster in an EU region). Connection details come from `.env` (`QDRANT_URL`, `QDRANT_API_KEY`), loaded with `python-dotenv`; never hard-code or print the key.
- Local models via **Ollama**, running natively on the Mac (MacBook M1 Pro, 16 GB RAM):
  - LLM: `qwen3.5:9b` (4-bit)
  - Embeddings: `qwen3-embedding:0.6b`
- PyMuPDF (`pymupdf`) for PDF text extraction.
- Models run locally on 16 GB RAM; only chunk text and vectors go to Qdrant Cloud.
- Install `llama-index-core` plus specific integrations, never the `llama-index` umbrella package (it pulls in OpenAI). Always set `Settings.llm` and `Settings.embed_model` explicitly so nothing falls back to OpenAI.

## Layout

```
data/ai-act.pdf          # source PDF from EUR-Lex (git-ignored)
src/ai_act_rag/          # package code (uv --package layout)
  parse.py               # PDF -> clean lines with page numbers -> legal units
  chunk.py               # legal units -> chunks with citations (split by numbered paragraph)
  config.py              # loads .env, connects to Qdrant Cloud
pyproject.toml           # dependencies, managed with `uv add`
.env.example             # template for .env (copy it; .env itself is git-ignored)
notes.md                 # interview prep: tools, decisions, concepts
```

## Commands

```bash
uv sync                               # create/update .venv from pyproject.toml
uv add <package>                      # add a dependency
uv add --dev <package>                # add a dev-only dependency (pytest, ruff)
uv run python -m ai_act_rag.parse     # run the parser
uv run pytest                         # run tests (once added)
uv run ruff check . && uv run ruff format .
```

Run all commands from the project root (`~/projects/ai-act-rag`); paths like `data/ai-act.pdf` are relative to it.

## About the source document

The Official Journal PDF has 144 pages. Every page repeats header/footer lines that must be stripped:
`OJ L, 12.7.2024`, `EN`, `ELI: http://data.europa.eu/eli/reg/2024/1689/oj`, and page counters like `12/144`.

Structure, in order:
1. Citations and recitals (1)–(180)
2. `HAVE ADOPTED THIS REGULATION:`
3. Chapters (Roman numerals) → optional Sections → Articles 1–113, each with a title line
4. Signatures (`Done at Brussels, ...`)
5. Annexes I–XIII

Chunk by these legal units (split long ones by paragraph), not by fixed character counts, and keep metadata on each chunk: kind, number, title, chapter, section, page range.

## Planned pipeline

1. Parse PDF into legal units (done: 180 recitals, 113 articles, 13 annexes)
2. Chunk units, attach metadata (in progress)
3. Embed with Ollama and store in Qdrant Cloud; add BM25 keyword search and fuse both with reciprocal rank fusion
4. Answer with the local LLM, citing units and pages; refuse when the context doesn't support an answer
5. Small evaluation set of questions with expected articles
6. FastAPI endpoint and a simple web page

## Conventions

- Type hints on functions, short docstrings, `pathlib.Path` for file paths.
- Small functions that are easy to test without the PDF.
- Don't commit `data/`, `.venv/`, or `.env`.
