"""Agent 1: Anonymizer — Vision-based sensitive content detection and masking."""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from core.pdf_processor import pdf_to_images, images_to_pdf
from utils.vision_api import VisionAPI

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "detect_sensitive.txt"


@dataclass
class AnonymizeResult:
    """Result of anonymization processing."""
    output_path: Path
    pages: list[dict] = field(default_factory=list)  # Per-page detection details
    success: bool = True
    error: str | None = None


class Anonymizer:
    """Detects and masks sensitive content in technical drawing PDFs using Claude Vision."""

    def __init__(self, config: dict, vision_api: VisionAPI):
        self.config = config
        self.vision = vision_api
        self.dpi = config["pdf"]["dpi"]
        self.bbox_margin = config["pdf"]["bbox_margin"]
        self.base_prompt = PROMPT_PATH.read_text()

    def process(self, pdf_path: str | Path, previous_errors: list[dict] | None = None) -> AnonymizeResult:
        """Anonymize a PDF file.

        Args:
            pdf_path: Path to the input PDF.
            previous_errors: Optional list of errors from a previous review,
                used to add correction instructions to the prompt.

        Returns:
            AnonymizeResult with output path and detection details.
        """
        pdf_path = Path(pdf_path)
        logger.info("Anonymizing %s", pdf_path.name)

        try:
            images = pdf_to_images(pdf_path, dpi=self.dpi)
        except Exception as e:
            logger.error("Failed to convert PDF to images: %s", e)
            return AnonymizeResult(output_path=Path(""), success=False, error=str(e))

        prompt = self._build_prompt(previous_errors)
        anonymized_images = []
        page_reports = []

        for i, img in enumerate(images):
            logger.info("Processing page %d/%d", i + 1, len(images))
            try:
                detections = self._detect_sensitive(img, prompt)
                masked = self._apply_masks(img, detections)
                anonymized_images.append(masked)
                page_reports.append({
                    "page": i + 1,
                    "regions_detected": len(detections.get("regions", [])),
                    "detections": detections,
                })
            except Exception as e:
                logger.error("Failed to process page %d: %s", i + 1, e)
                # Use original image on failure
                anonymized_images.append(img)
                page_reports.append({
                    "page": i + 1,
                    "regions_detected": 0,
                    "error": str(e),
                })

        # Save output PDF
        output_dir = Path(self.config["directories"]["output"])
        output_path = output_dir / pdf_path.name
        try:
            images_to_pdf(anonymized_images, output_path, original_pdf=pdf_path)
        except Exception as e:
            logger.error("Failed to save output PDF: %s", e)
            return AnonymizeResult(output_path=output_path, pages=page_reports, success=False, error=str(e))

        return AnonymizeResult(output_path=output_path, pages=page_reports)

    def _build_prompt(self, previous_errors: list[dict] | None) -> str:
        """Build the detection prompt, optionally with correction instructions."""
        prompt = self.base_prompt

        if previous_errors:
            corrections = "\n\nCORRECTION INSTRUCTIONS FROM PREVIOUS REVIEW:\n"
            corrections += "The previous anonymization attempt had the following issues that MUST be fixed:\n\n"
            for err in previous_errors:
                corrections += f"- [{err.get('type', 'unknown')}] {err.get('message', '')}\n"
                if err.get("region"):
                    r = err["region"]
                    corrections += f"  Location: approximately ({r.get('x', '?')}%, {r.get('y', '?')}%)\n"
            corrections += "\nPay special attention to these areas and ensure they are properly masked.\n"
            prompt += corrections

        return prompt

    def _detect_sensitive(self, image: Image.Image, prompt: str) -> dict:
        """Send image to Vision API for sensitive content detection."""
        # Convert PIL image to PNG bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = self.vision.analyze_image(image_bytes, prompt)
        regions = result.get("regions", [])
        logger.info("Detected %d sensitive regions", len(regions))
        return result

    def _apply_masks(self, image: Image.Image, detections: dict) -> Image.Image:
        """White-fill detected regions on the image."""
        masked = image.copy()
        draw = ImageDraw.Draw(masked)
        w, h = masked.size

        for region in detections.get("regions", []):
            bbox_pct = region.get("bbox", [])
            if len(bbox_pct) != 4:
                logger.warning("Skipping region with invalid bbox: %s", region)
                continue

            # Convert percentage coords to pixels
            x_pct, y_pct, w_pct, h_pct = bbox_pct
            x1 = int((x_pct / 100) * w) - self.bbox_margin
            y1 = int((y_pct / 100) * h) - self.bbox_margin
            x2 = int(((x_pct + w_pct) / 100) * w) + self.bbox_margin
            y2 = int(((y_pct + h_pct) / 100) * h) + self.bbox_margin

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            draw.rectangle([x1, y1, x2, y2], fill="white")
            logger.debug("Masked region '%s' at [%d,%d,%d,%d]",
                         region.get("label", "unknown"), x1, y1, x2, y2)

        return masked
