from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, List
import tempfile
from pathlib import Path
import fitz
from paddleocr import PaddleOCR
import shutil

app = FastAPI(title="PDF OCR Extractor API")


# -------------------------------
# 📦 Pydantic Request Model
# -------------------------------
class OCRRequest(BaseModel):
    dpi: Optional[int] = 300
    lang: Optional[str] = "en"
    conf_threshold: Optional[float] = 0.5
    max_pages: Optional[int] = None

    @field_validator("dpi")
    def validate_dpi(cls, v):
        if v < 100 or v > 600:
            raise ValueError("DPI must be between 100 and 600")
        return v

    @field_validator("conf_threshold")
    def validate_conf(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v


# -------------------------------
# 🔍 OCR Core Function
# -------------------------------
def pdf_to_text_with_paddle(pdf_path, dpi, lang, conf_threshold, max_pages):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages = range(total_pages) if max_pages is None else range(min(max_pages, total_pages))

    all_text = []
    page_summaries = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i in pages:
            page = doc[i]

            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)

            img_path = tmpdir / f"page_{i+1}.png"
            pix.save(img_path.as_posix())

            result = ocr.ocr(str(img_path), cls=True)

            lines = []
            if result and len(result) > 0:
                for box, (text, conf) in result[0]:
                    text = (text or "").strip()
                    if text:
                        lines.append((text, float(conf)))

            good_lines = [t for t, c in lines if c >= conf_threshold]
            page_text = "\n".join(good_lines)

            all_text.append(f"PAGE {i+1}\n{page_text}")

            page_summaries.append({
                "page": i + 1,
                "lines_detected": len(lines),
                "lines_kept": len(good_lines),
                "ok": len(good_lines) > 0
            })

    doc.close()

    return "\n\n".join(all_text).rstrip(), page_summaries


# -------------------------------
# 🌐 API Endpoint
# -------------------------------
@app.post("/extract-text")
async def extract_text(
    file: UploadFile = File(...),
    dpi: int = 300,
    lang: str = "en",
    conf_threshold: float = 0.5,
    max_pages: Optional[int] = None
):
    try:
        # 🔒 Validate via Pydantic manually
        request = OCRRequest(
            dpi=dpi,
            lang=lang,
            conf_threshold=conf_threshold,
            max_pages=max_pages
        )

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_pdf_path = tmp.name

        # Run OCR
        text, summary = pdf_to_text_with_paddle(
            temp_pdf_path,
            request.dpi,
            request.lang,
            request.conf_threshold,
            request.max_pages
        )

        # 🟢 Optimistic success response
        return {
            "status": "success",
            "message": "Text extracted successfully",
            "pages_processed": len(summary),
            "text": text,
            "summary": summary
        }

    except ValueError as ve:
        return {
            "status": "failed",
            "message": str(ve)
        }

    except Exception as e:
        # 🔥 No 500 leak — controlled response
        return {
            "status": "failed",
            "message": "Couldn't extract text from the document",
            "hint": "Check if PDF is valid or try different DPI/confidence settings"
        }
