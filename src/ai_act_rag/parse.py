from pathlib import Path

import pymupdf

PDF_PATH = Path("data/ai-act.pdf")


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the plain text of every page in the PDF, one string per page."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def main() -> None:
    pages = extract_pages(PDF_PATH)
    print(f"Pages: {len(pages)}")
    print(f"Characters: {sum(len(p) for p in pages):,}")
    print("--- Start of page 1 ---")
    print(pages[0][:800])


if __name__ == "__main__":
    main()