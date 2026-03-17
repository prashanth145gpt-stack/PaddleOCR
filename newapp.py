from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field, validator
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from pathlib import Path
import tempfile

app = FastAPI(title="PDF OCR Extractor API")


# -----------------------------
# Pydantic Config (Internal Validation)
# -----------------------------
class OCRConfig(BaseModel):
    dpi: int = Field(default=300, ge=72, le=600)
    lang: str = "en"
    conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_pages: int | None = Field(default=None, ge=1)

    @validator("lang")
    def validate_lang(cls, v):
        if not v:
            raise ValueError("Language cannot be empty")
        return v


# -----------------------------
# Initialize OCR ONCE (IMPORTANT)
# -----------------------------
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


# -----------------------------
# Core OCR Logic (UNCHANGED FLOW)
# -----------------------------
def pdf_to_text_with_paddle(pdf_path, dpi=300, lang='en', conf_threshold=0.5, max_pages=None):
    pdf_path = str(pdf_path)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages = range(total_pages) if max_pages is None else range(min(max_pages, total_pages))

    all_text = []

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

    doc.close()
    full_text = "\n\n".join(all_text).rstrip()
    return full_text


# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/extract-ocr")
async def extract_ocr(file: UploadFile = File(...)):
    try:
        # Default configs (hidden from Swagger)
        config = OCRConfig(
            dpi=300,
            lang="en",
            conf_threshold=0.5,
            max_pages=None
        )

        # Read uploaded file safely
        contents = await file.read()
        if not contents:
            return {"extracted_text": "Empty file uploaded"}

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Run OCR
        text = pdf_to_text_with_paddle(
            tmp_path,
            dpi=config.dpi,
            lang=config.lang,
            conf_threshold=config.conf_threshold,
            max_pages=config.max_pages
        )

        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

        return {
            "extracted_text": text
        }

    except Exception as e:
        return {
            "extracted_text": f"Couldn't extract text from the document: {str(e)}"
        }
