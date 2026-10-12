# AI Act RAG

Retrieval-augmented question answering over the EU AI Act (Regulation (EU) 2024/1689).
Answers must cite the exact legal unit they come from (e.g. "Article 26(1)", "Annex III", "Recital (27)") and the PDF page.

## How I work on this project

I'm building this step by step to learn, and I'm new to macOS and the terminal.
- Explain every command and every flag in plain words before or next to it.
- Make one small change at a time, then tell me how to run it and what output to expect.
- Don't write large parts of the codebase in one go unless I ask; prefer explaining and letting me type it.
- Ask before adding a new dependency or any paid API.

## Stack

- Python 3.12, managed with **uv** (never pip or a manually activated venv).
- LlamaIndex for indexing and retrieval.
- Qdrant as the vector database (EU-based, runs locally in Docker).
- Local models via **Ollama**, running natively on the Mac (MacBook M1 Pro, 16 GB RAM):
  - LLM: `qwen3.5:9b` (4-bit)
  - Embeddings: `qwen3-embedding:0.6b`
- PyMuPDF (`pymupdf`) for PDF text extraction.
- Keep everything able to run offline on 16 GB RAM.

## Layout

```
data/ai-act.pdf          # source PDF from EUR-Lex (git-ignored)
src/ai_act_rag/          # package code (uv --package layout)
  parse.py               # PDF -> clean lines with page numbers -> legal units
pyproject.toml           # dependencies, managed with `uv add`
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

1. Parse PDF into legal units (in progress)
2. Chunk units, attach metadata
3. Embed with Ollama and store in Qdrant; add BM25 keyword search and fuse both with reciprocal rank fusion
4. Answer with the local LLM, citing units and pages; refuse when the context doesn't support an answer
5. Small evaluation set of questions with expected articles
6. FastAPI endpoint and a simple web page

## Conventions

- Type hints on functions, short docstrings, `pathlib.Path` for file paths.
- Small functions that are easy to test without the PDF.
- Don't commit `data/`, `.venv/`, or `.env`.
