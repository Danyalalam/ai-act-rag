"""Measure OCR accuracy on pages whose correct text we know.

The AI Act PDFs have a text layer, so each page can be rendered as an image, read by
the OCR model, and compared with the real text: a free, exact answer key, in both
languages. Each page is tested twice: a clean render, and a rougher "scan" (lower
resolution, strong JPEG compression).
"""

import json
import re
import time
from datetime import datetime

import pymupdf

from ai_act_rag.evaluation.run import RESULTS_DIR, git_commit
from ai_act_rag.ingestion.parse import is_page_furniture
from ai_act_rag.languages import LANGUAGES
from ai_act_rag.models import OCR_MODEL
from ai_act_rag.ocr.read import RENDER_DPI, read_page

PAGES = (51, 67)  # Articles 4-5 (prohibited practices) and Article 26 (deployers)
SCAN_DPI = 110
SCAN_JPEG_QUALITY = 55


def normalise(text: str, language_code: str) -> str:
    """Make OCR output and reference comparable: drop header/footer lines and Markdown
    marks the model may add, and collapse all whitespace to single spaces."""
    language = LANGUAGES[language_code]
    lines = [line.strip() for line in text.splitlines()]
    kept = [line for line in lines if line and not is_page_furniture(line, language)]
    text = " ".join(kept)
    text = re.sub(r"[*#|`]", "", text)  # Markdown: **bold**, # headings, | tables
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the fewest single-character edits that turn `a` into `b`."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def character_error_rate(ocr: str, reference: str) -> float:
    """Edits needed / length of the correct text. 0.0 = perfect; 0.02 = 2 errors per 100 characters."""
    return edit_distance(ocr, reference) / max(len(reference), 1)


def main() -> None:
    started = datetime.now()
    rows = []
    for code, language in LANGUAGES.items():
        with pymupdf.open(language.pdf_path) as doc:
            for number in PAGES:
                page = doc[number - 1]
                reference = normalise(page.get_text(), code)
                variants = {
                    "clean": page.get_pixmap(dpi=RENDER_DPI).tobytes("png"),
                    "scan": page.get_pixmap(dpi=SCAN_DPI).tobytes("jpg", jpg_quality=SCAN_JPEG_QUALITY),
                }
                for variant, image in variants.items():
                    start = time.time()
                    ocr = normalise("".join(read_page(image)), code)
                    seconds = round(time.time() - start, 1)
                    cer = character_error_rate(ocr, reference)
                    print(f"{code} p.{number} {variant:5}  CER {cer:6.2%}  {seconds:5.1f} s", flush=True)
                    rows.append({"language": code, "page": number, "variant": variant,
                                 "cer": round(cer, 4), "seconds": seconds,
                                 "reference_chars": len(reference), "ocr_chars": len(ocr)})

    summary = {}
    for code in LANGUAGES:
        for variant in ("clean", "scan"):
            subset = [r for r in rows if r["language"] == code and r["variant"] == variant]
            summary[f"{code}-{variant}"] = {
                "mean_cer": round(sum(r["cer"] for r in subset) / len(subset), 4),
                "mean_seconds": round(sum(r["seconds"] for r in subset) / len(subset), 1),
            }
    print("\n" + "\n".join(f"{k:9} CER {v['mean_cer']:.2%}  {v['mean_seconds']} s/page" for k, v in summary.items()))

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"ocr-{started:%Y-%m-%d-%H%M}.json"
    path.write_text(json.dumps({
        "date": started.isoformat(timespec="seconds"), "git_commit": git_commit(),
        "model": OCR_MODEL, "render_dpi": RENDER_DPI,
        "scan": {"dpi": SCAN_DPI, "jpeg_quality": SCAN_JPEG_QUALITY},
        "summary": summary, "per_page": rows,
    }, indent=2))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
