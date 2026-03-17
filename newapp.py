from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from pathlib import Path
import tempfile
from typing import Optional, List

app = FastAPI(title="PDF OCR Extractor API")


# ---------------------------
# Pydantic Request Model
# ---------------------------
class OCRRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute path to PDF file")
    dpi: int = Field(300, ge=72, le=600)
    lang: str = Field("en")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_pages: Optional[int] = Field(None, ge=1)

    @validator("pdf_path")
    def validate_pdf_path(cls, v):
        if not Path(v).exists():
            raise ValueError("PDF file does not exist")
        if not v.lower().endswith(".pdf"):
            raise ValueError("File must be a PDF")
        return v


# ---------------------------
# Response Model
# ---------------------------
class PageSummary(BaseModel):
    page: int
    lines_detected: int
    lines_kept: int
    ok: bool


class OCRResponse(BaseModel):
    extracted_text: str
    summary: List[PageSummary]
    message: str


# ---------------------------
# Core Logic (UNCHANGED)
# ---------------------------
def pdf_to_text_with_paddle(pdf_path, dpi=300, lang='en', conf_threshold=0.5, max_pages=None):
    pdf_path = str(pdf_path)
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
            img_path = tmpdir / f"page_{i+1:04d}.png"
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
    full_text = "\n\n".join(all_text).rstrip()
    return full_text, page_summaries


# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/extract-text", response_model=OCRResponse)
def extract_text(request: OCRRequest):
    try:
        text, summary = pdf_to_text_with_paddle(
            pdf_path=request.pdf_path,
            dpi=request.dpi,
            lang=request.lang,
            conf_threshold=request.conf_threshold,
            max_pages=request.max_pages
        )

        if not text.strip():
            return OCRResponse(
                extracted_text="",
                summary=summary,
                message="Could not extract meaningful text from the document"
            )

        return OCRResponse(
            extracted_text=text,
            summary=summary,
            message="Extraction successful"
        )

    except ValueError as ve:
        # Validation-level issues
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # Replace 500 with controlled response
        return OCRResponse(
            extracted_text="",
            summary=[],
            message=f"Could not extract text due to processing issue: {str(e)}"
        )
