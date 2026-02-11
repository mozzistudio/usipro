"""Tesseract OCR wrapper for text extraction."""

import logging

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text(image: Image.Image, region: tuple | None = None, lang: str = "fra+eng") -> str:
    """Extract text from an image or a region of an image.

    Args:
        image: PIL Image to process.
        region: Optional (x, y, width, height) tuple to crop before OCR.
        lang: Tesseract language codes.

    Returns:
        Extracted text string.
    """
    if region is not None:
        x, y, w, h = region
        image = image.crop((x, y, x + w, y + h))

    try:
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except pytesseract.TesseractError as e:
        logger.warning("OCR error: %s", e)
        return ""


def extract_text_regions(image: Image.Image, lang: str = "fra+eng") -> list[dict]:
    """Extract text with bounding box positions.

    Args:
        image: PIL Image to process.
        lang: Tesseract language codes.

    Returns:
        List of dicts with keys: text, x, y, width, height, confidence.
    """
    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError as e:
        logger.warning("OCR error: %s", e)
        return []

    regions = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if text and conf > 30:
            regions.append({
                "text": text,
                "x": data["left"][i],
                "y": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "confidence": conf,
            })

    return regions
