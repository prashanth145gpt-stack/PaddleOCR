import logging
import os

from statistics import median
from threading import Lock
from typing import Any, Dict, List

from paddleocr import PaddleOCR

from app.config import settings


logging.getLogger("ppocr").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class OCRService:
    """
    Lazy PaddleOCR service using local detection and recognition models.
    """

    def __init__(self):
        self.engine = None
        self.lock = Lock()

        app_directory = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.detection_model_dir = os.getenv(
            "OCR_DETECTION_MODEL_DIR",
            os.path.join(
                app_directory,
                "models",
                "PP_OCRv4_mobile_det_infer",
            ),
        )

        self.recognition_model_dir = os.getenv(
            "OCR_RECOGNITION_MODEL_DIR",
            os.path.join(
                app_directory,
                "models",
                "en_PP_OCRv4_mobile_rec_infer",
            ),
        )

    def validate_model_directory(
        self,
        model_directory: str,
        model_description: str,
    ) -> None:
        if not os.path.isdir(model_directory):
            raise RuntimeError(
                f"{model_description} directory does not exist: "
                f"{model_directory}"
            )

        required_files = [
            "inference.json",
            "inference.yml",
            "inference.pdiparams",
        ]

        missing_files = [
            os.path.join(model_directory, file_name)
            for file_name in required_files
            if not os.path.isfile(
                os.path.join(model_directory, file_name)
            )
        ]

        if missing_files:
            raise RuntimeError(
                f"{model_description} is incomplete. "
                f"Missing files: {', '.join(missing_files)}"
            )

    def validate_local_models(self) -> None:
        self.validate_model_directory(
            self.detection_model_dir,
            "OCR detection model",
        )

        self.validate_model_directory(
            self.recognition_model_dir,
            "OCR recognition model",
        )

    def get_engine(self):
        if self.engine is None:
            self.validate_local_models()

            self.engine = PaddleOCR(
                lang="en",

                text_detection_model_name="PP-OCRv4_mobile_det",
                text_detection_model_dir=self.detection_model_dir,

                text_recognition_model_name="en_PP-OCRv4_mobile_rec",
                text_recognition_model_dir=self.recognition_model_dir,

                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,

                device="cpu",
            )

        return self.engine

    def run_ocr(self, image_rgb) -> Dict:
        """
        Run OCR on an RGB NumPy image.
        """

        try:
            with self.lock:
                engine = self.get_engine()
                ocr_result = engine.ocr(image_rgb)

            return self.analyze_result(ocr_result)

        except Exception as error:
            logger.exception("PaddleOCR execution failed.")

            return {
                "mean_confidence": 0.0,
                "median_confidence": 0.0,
                "min_confidence": 0.0,
                "low_confidence_line_ratio": 1.0,
                "detected_text_boxes": 0,
                "detected_chars": 0,
                "sample_text": "",
                "ocr_error": str(error),
            }

    def analyze_result(
        self,
        ocr_result: List[Any],
    ) -> Dict:
        """
        Convert PaddleOCR output into OCR quality metrics.
        """

        confidences = []
        texts = []

        if not ocr_result:
            return self.empty_result()

        for page_result in ocr_result:
            if not page_result:
                continue

            for line in page_result:
                try:
                    text = line[1][0]
                    confidence = float(line[1][1])

                    if text and text.strip():
                        texts.append(text.strip())
                        confidences.append(confidence)

                except Exception:
                    continue

        if not confidences:
            return self.empty_result()

        low_confidence_lines = [
            confidence
            for confidence in confidences
            if confidence
            < settings.low_confidence_line_threshold
        ]

        detected_chars = sum(
            len(text)
            for text in texts
        )

        return {
            "mean_confidence": round(
                sum(confidences) / len(confidences),
                4,
            ),
            "median_confidence": round(
                median(confidences),
                4,
            ),
            "min_confidence": round(
                min(confidences),
                4,
            ),
            "low_confidence_line_ratio": round(
                len(low_confidence_lines)
                / len(confidences),
                4,
            ),
            "detected_text_boxes": len(confidences),
            "detected_chars": detected_chars,
            "sample_text": " ".join(texts)[:300],
        }

    def empty_result(self) -> Dict:
        """
        Return empty OCR metrics when no readable text is found.
        """

        return {
            "mean_confidence": 0.0,
            "median_confidence": 0.0,
            "min_confidence": 0.0,
            "low_confidence_line_ratio": 1.0,
            "detected_text_boxes": 0,
            "detected_chars": 0,
            "sample_text": "",
        }