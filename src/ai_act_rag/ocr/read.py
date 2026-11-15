import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pymupdf
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock

from ai_act_rag.models import ocr_model

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_PAGES = 20  # OCR takes seconds per page on a laptop; keep requests bounded
# Resolution PDF pages are rendered at. 150 dpi turns an A4 page into ~1240x1750 pixels:
# enough for small print, without making the image (and the model's work) needlessly large.
RENDER_DPI = 150
# GLM-OCR's task prompt for plain text (it also knows "Table Recognition:" and
# "Figure Recognition:").
PROMPT = "Text Recognition:"


def page_images(data: bytes, filename: str) -> list[bytes]:
    """Turn an uploaded file into one PNG/JPEG image per page.

    An image is one page. A PDF is rendered page by page, so scanned PDFs work too.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_TYPES:
        return [data]
    if suffix != ".pdf":
        raise ValueError(f"Unsupported file type {suffix!r}: use a PDF or {sorted(IMAGE_TYPES)}")
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if len(doc) > MAX_PAGES:
            raise ValueError(f"The PDF has {len(doc)} pages; the limit is {MAX_PAGES}")
        return [page.get_pixmap(dpi=RENDER_DPI).tobytes("png") for page in doc]


def read_page(image: bytes) -> Iterator[str]:
    """Stream the text of one page image, piece by piece, as the model writes it."""
    message = ChatMessage(
        role="user", blocks=[ImageBlock(image=image), TextBlock(text=PROMPT)]
    )
    for chunk in ocr_model().stream_chat([message]):
        if chunk.delta:
            yield chunk.delta


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: uv run python -m ai_act_rag.ocr.read <image-or-pdf>")
    path = Path(sys.argv[1])
    images = page_images(path.read_bytes(), path.name)
    start = time.time()
    for number, image in enumerate(images, start=1):
        print(f"\n--- Page {number} of {len(images)} ---\n", flush=True)
        for piece in read_page(image):
            print(piece, end="", flush=True)
    print(f"\n\n({len(images)} page(s) in {time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
