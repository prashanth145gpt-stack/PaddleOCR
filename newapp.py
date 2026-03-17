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

        return (text)

    except Exception as e:
        return ("Couldn't extract text from the document: {str(e)}")











download https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar to /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar

Traceback (most recent call last):

  File "/usr/local/lib/python3.10/site-packages/urllib3/connectionpool.py", line 773, in urlopen

    self._prepare_proxy(conn)

  File "/usr/local/lib/python3.10/site-packages/urllib3/connectionpool.py", line 1042, in _prepare_proxy

    conn.connect()

  File "/usr/local/lib/python3.10/site-packages/urllib3/connection.py", line 796, in connect

    sock_and_verified = _ssl_wrap_socket_and_match_hostname(

  File "/usr/local/lib/python3.10/site-packages/urllib3/connection.py", line 975, in _ssl_wrap_socket_and_match_hostname

    ssl_sock = ssl_wrap_socket(

  File "/usr/local/lib/python3.10/site-packages/urllib3/util/ssl_.py", line 483, in ssl_wrap_socket

    ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_hostname)

  File "/usr/local/lib/python3.10/site-packages/urllib3/util/ssl_.py", line 527, in _ssl_wrap_socket_impl

    return ssl_context.wrap_socket(sock, server_hostname=server_hostname)

  File "/usr/local/lib/python3.10/ssl.py", line 512, in wrap_socket

    return self.sslsocket_class._create(

  File "/usr/local/lib/python3.10/ssl.py", line 1070, in _create

    self.do_handshake()

  File "/usr/local/lib/python3.10/ssl.py", line 1341, in do_handshake

    self._sslobj.do_handshake()

ConnectionResetError: [Errno 104] Connection reset by peer
 
During handling of the above exception, another exception occurred:
 
Traceback (most recent call last):

  File "/usr/local/lib/python3.10/site-packages/requests/adapters.py", line 644, in send

    resp = conn.urlopen(

  File "/usr/local/lib/python3.10/site-packages/urllib3/connectionpool.py", line 841, in urlopen

    retries = retries.increment(

  File "/usr/local/lib/python3.10/site-packages/urllib3/util/retry.py", line 490, in increment

    raise reraise(type(error), error, _stacktrace)

  File "/usr/local/lib/python3.10/site-packages/urllib3/util/util.py", line 38, in reraise

    raise value.with_traceback(tb)

  File "/usr/local/lib/python3.10/site-packages/urllib3/connectionpool.py", line 773, in urlopen

    self._prepare_proxy(conn)

  File "/usr/local/lib/python3.10/site-packages/urllib3/connectionpool.py", line 1042, in _prepare_proxy

    conn.connect()

  File "/usr/local/lib/python3.10/site-packages/urllib3/connection.py", line 796, in connect

    sock_and_verified = _ssl_wrap_socket_and_match_hostname(

  File "/usr/local/lib/python3.10/site-packages/urllib3/connection.py", line 975, in _ssl_wrap_socket_and_match_hostname

    ssl_sock = ssl_wrap_socket(

  File "/usr/local/lib/python3.10/site-packages/urllib3/util/ssl_.py", line 483, in ssl_wrap_socket

    ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_hostname)

  File "/usr/local/lib/python3.10/site-packages/urllib3/util/ssl_.py", line 527, in _ssl_wrap_socket_impl

    return ssl_context.wrap_socket(sock, server_hostname=server_hostname)

  File "/usr/local/lib/python3.10/ssl.py", line 512, in wrap_socket

    return self.sslsocket_class._create(

  File "/usr/local/lib/python3.10/ssl.py", line 1070, in _create

    self.do_handshake()

  File "/usr/local/lib/python3.10/ssl.py", line 1341, in do_handshake

    self._sslobj.do_handshake()

urllib3.exceptions.ProtocolError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))
 
During handling of the above exception, another exception occurred:
 
Traceback (most recent call last):

  File "/usr/local/bin/uvicorn", line 6, in <module>

    sys.exit(main())

  File "/usr/local/lib/python3.10/site-packages/click/core.py", line 1485, in __call__

    return self.main(*args, **kwargs)

  File "/usr/local/lib/python3.10/site-packages/click/core.py", line 1406, in main

    rv = self.invoke(ctx)

  File "/usr/local/lib/python3.10/site-packages/click/core.py", line 1269, in invoke

    return ctx.invoke(self.callback, **ctx.params)

  File "/usr/local/lib/python3.10/site-packages/click/core.py", line 824, in invoke

    return callback(*args, **kwargs)

  File "/usr/local/lib/python3.10/site-packages/uvicorn/main.py", line 423, in main

    run(

  File "/usr/local/lib/python3.10/site-packages/uvicorn/main.py", line 593, in run

    server.run()

  File "/usr/local/lib/python3.10/site-packages/uvicorn/server.py", line 67, in run

    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())

  File "/usr/local/lib/python3.10/site-packages/uvicorn/_compat.py", line 60, in asyncio_run

    return loop.run_until_complete(main)

  File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete

  File "/usr/local/lib/python3.10/site-packages/uvicorn/server.py", line 71, in serve

    await self._serve(sockets)

  File "/usr/local/lib/python3.10/site-packages/uvicorn/server.py", line 78, in _serve

    config.load()

  File "/usr/local/lib/python3.10/site-packages/uvicorn/config.py", line 439, in load

    self.loaded_app = import_from_string(self.app)

  File "/usr/local/lib/python3.10/site-packages/uvicorn/importer.py", line 19, in import_from_string

    module = importlib.import_module(module_str)

  File "/usr/local/lib/python3.10/importlib/__init__.py", line 126, in import_module

    return _bootstrap._gcd_import(name[level:], package, level)

  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import

  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load

  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked

  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked

  File "<frozen importlib._bootstrap_external>", line 883, in exec_module

  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed

  File "/app/app.py", line 30, in <module>

    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

  File "/usr/local/lib/python3.10/site-packages/paddleocr/paddleocr.py", line 599, in __init__

    maybe_download(params.det_model_dir, det_url)

  File "/usr/local/lib/python3.10/site-packages/paddleocr/ppocr/utils/network.py", line 55, in maybe_download

    download_with_progressbar(url, tmp_path)

  File "/usr/local/lib/python3.10/site-packages/paddleocr/ppocr/utils/network.py", line 28, in download_with_progressbar

    response = requests.get(url, stream=True)

  File "/usr/local/lib/python3.10/site-packages/requests/api.py", line 73, in get

    return request("get", url, params=params, **kwargs)

  File "/usr/local/lib/python3.10/site-packages/requests/api.py", line 59, in request

    return session.request(method=method, url=url, **kwargs)

  File "/usr/local/lib/python3.10/site-packages/requests/sessions.py", line 589, in request

    resp = self.send(prep, **send_kwargs)

  File "/usr/local/lib/python3.10/site-packages/requests/sessions.py", line 703, in send

    r = adapter.send(request, **kwargs)

  File "/usr/local/lib/python3.10/site-packages/requests/adapters.py", line 659, in send

    raise ConnectionError(err, request=request)

requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))

 
