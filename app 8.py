from paddleocr import PaddleOCR
import fitz 
from fastapi import FastAPI, UploadFile, File, Request,Form, Response
from pathlib import Path
import tempfile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, PlainTextResponse
import os, time, json, subprocess, tempfile, re
 
app = FastAPI()
 
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
            
            good_lines = []
            if result and len(result) > 0:
                for box, (text, conf) in result[0]:
                    lines = []
                    text = (text or "").strip()
                    if text:
                        lines.append((text, float(conf)))
                        gd = [t for t, c in lines if c >= conf_threshold]
                        good_lines.append(gd) 

            txt_lines = []
            for i in good_lines:
                page_text = " ".join(i)
                txt_lines.append(page_text)
            
            ftext = "\n".join(txt_lines)
            
            all_text.append(ftext)

 
    doc.close()
    full_text = "\n\n".join(all_text).rstrip()
    return full_text
 
@app.post("/extract-ocr")
async def ocr_extract(file : UploadFile, response_class = PlainTextResponse):
    if not file: return JSONResponse({"error":"no file"})
    with tempfile.NamedTemporaryFile(delete = False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    
    text = pdf_to_text_with_paddle(path)
    os.remove(path)
    return PlainTextResponse(text)