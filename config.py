from pydantic import BaseModel


class Settings(BaseModel):
    # File limits
    max_file_size_mb: int = 200
    max_allowed_pages: int = 1000

    # API status
    success_status: str = "SUCCESS"
    failure_status: str = "FAILURE"
    partial_failure_status: str = "PARTIAL_FAILURE"

    # PDF text-layer thresholds
    text_min_chars_per_page: int = 10
    text_good_chars_per_page: int = 150
    text_min_words_per_page: int = 5
    text_good_words_per_page: int = 20

    # Overall document quality policy
    min_document_quality_score: float = 75.0

    # Document classification ratios
    machine_readable_ratio: float = 0.90
    scanned_ratio: float = 0.90
    ocr_layered_ratio: float = 0.60

    # Searchable scanned PDF policy
    min_searchable_text_chars_per_page: int = 100
    min_searchable_text_words_per_page: int = 15

    # Image quality thresholds
    min_laplacian_variance: float = 80.0
    severe_blur_threshold: float = 35.0
    unreadable_blur_threshold: float = 15.0
    min_contrast_std: float = 35.0
    min_width: int = 900
    min_height: int = 900
    min_brightness: float = 45.0
    max_brightness: float = 220.0
    max_skew_angle: float = 7.0

    # OCR thresholds
    ocr_reject_mean_confidence: float = 0.65
    ocr_reject_median_confidence: float = 0.68
    low_confidence_line_threshold: float = 0.60
    max_low_conf_ratio_reject: float = 0.40
    min_text_boxes: int = 5
    min_detected_chars: int = 50

    # OCR recovery policy for recoverable severe blur
    allow_ocr_recovery_for_severe_blur: bool = True
    ocr_recovery_mean_confidence: float = 0.75
    ocr_recovery_median_confidence: float = 0.75
    ocr_recovery_min_text_boxes: int = 8
    ocr_recovery_min_detected_chars: int = 100
    ocr_recovery_max_low_conf_ratio: float = 0.30

    # OCR sampling and runtime
    max_ocr_pages_per_document: int = 12
    max_failed_pages_before_reject: int = 5
    max_ocr_failure_pages_allowed: int = 5
    max_ocr_failure_ratio_allowed: float = 0.30
    unknown_page_ocr_failure_is_warning: bool = True

    # Low-content page policy
    max_low_content_pages_allowed: int = 10
    max_low_content_page_ratio: float = 0.20

    # Rendering
    render_dpi: int = 150

    # UI response
    return_success_pages: bool = False
    return_warning_details: bool = False
    return_warnings_summary: bool = True
    warning_sample_pages_limit: int = 5

    # Batch/generic API
    max_files_per_request: int = 40
    enable_parallel_file_processing: bool = True
    max_file_workers: int = 3


settings = Settings()