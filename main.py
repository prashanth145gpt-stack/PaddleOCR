from typing import Annotated, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.config import settings
from app.ocr_service import OCRService
from app.document_quality import validate_document_quality


app = FastAPI(
    title="Document Quality Validation API",
    description=(
        "Generic API to validate uploaded document quality and readability before submission. "
        "Supports PDF, image, Word, Excel and CSV files. "
        "Returns SUCCESS, FAILURE, or PARTIAL_FAILURE. "
        "This API checks only document quality/readability and basic file usability. "
        "It does not validate data correctness, business rules, completeness, authenticity, or fraud."
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_service = OCRService()


@app.get("/health", response_model=Dict[str, str])
def health_check():
    return {
        "status": "OK",
        "service": "Document Quality Validation API"
    }


def build_failure_result(file_name, content_type, code, message):
    return {
        "file_name": file_name,
        "content_type": content_type,
        "status": settings.failure_status,
        "document_type": "UNKNOWN",
        "validation_strategy": "FILE_VALIDATION",
        "document_quality_score": 0.0,
        "is_readable": False,
        "reupload_required": True,
        "message": "File validation failed. Please re-upload a readable file.",
        "validation_failures": [
            {
                "source": "SYSTEM_VALIDATION",
                "code": code,
                "message": message,
                "severity": "CRITICAL"
            }
        ],
        "metadata": {}
    }


def process_single_file(file_payload: Dict[str, Any]) -> Dict[str, Any]:
    file_name = file_payload["file_name"]
    content_type = file_payload["content_type"]
    file_bytes = file_payload["file_bytes"]

    try:
        result = validate_document_quality(
            file_bytes=file_bytes,
            ocr_service=ocr_service,
            file_name=file_name,
            content_type=content_type
        )

        return {
            "file_name": file_name,
            "content_type": content_type,
            **result
        }

    except ValueError as error:
        return build_failure_result(
            file_name=file_name,
            content_type=content_type,
            code="FILE_VALIDATION_ERROR",
            message=str(error)
        )

    except Exception as error:
        return build_failure_result(
            file_name=file_name,
            content_type=content_type,
            code="INTERNAL_VALIDATION_ERROR",
            message=str(error)
        )


@app.post("/validate")
async def validate_documents(
    files: Annotated[
        List[UploadFile],
        File(description="Upload one or more documents")
    ]
):
    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files uploaded."
            )

        if len(files) > settings.max_files_per_request:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files uploaded. Maximum allowed files: {settings.max_files_per_request}."
            )

        file_payloads = []

        for file in files:
            file_payloads.append(
                {
                    "file_name": file.filename or "",
                    "content_type": file.content_type or "",
                    "file_bytes": await file.read()
                }
            )

        results = []

        if settings.enable_parallel_file_processing and len(file_payloads) > 1:
            with ThreadPoolExecutor(max_workers=settings.max_file_workers) as executor:
                future_to_index = {
                    executor.submit(process_single_file, payload): index
                    for index, payload in enumerate(file_payloads)
                }

                ordered_results = [None] * len(file_payloads)

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    ordered_results[index] = future.result()

                results = ordered_results
        else:
            for payload in file_payloads:
                results.append(process_single_file(payload))

        passed_files = sum(
            1 for result in results
            if result.get("status") == settings.success_status
        )

        failed_files = sum(
            1 for result in results
            if result.get("status") == settings.failure_status
        )

        if failed_files == 0:
            overall_status = settings.success_status
        elif passed_files == 0:
            overall_status = settings.failure_status
        else:
            overall_status = settings.partial_failure_status

        response = {
            "status": overall_status,
            "total_files": len(results),
            "passed_files": passed_files,
            "failed_files": failed_files,
            "results": results
        }

        return JSONResponse(content=jsonable_encoder(response))

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document quality validation failed due to an internal error: {str(error)}"
        )