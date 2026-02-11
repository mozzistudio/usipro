"""PDF to image conversion and image to PDF assembly."""

import logging
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[Image.Image]:
    """Convert each page of a PDF to a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering.

    Returns:
        List of PIL Images, one per page.
    """
    pdf_path = Path(pdf_path)
    logger.info("Converting %s to images at %d DPI", pdf_path.name, dpi)
    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info("Converted %d pages", len(images))
    return images


def images_to_pdf(images: list[Image.Image], output_path: str | Path, original_pdf: str | Path | None = None):
    """Save a list of images as a PDF, preserving original page dimensions if available.

    Args:
        images: List of PIL Images to combine into a PDF.
        output_path: Where to save the resulting PDF.
        original_pdf: Optional path to original PDF for dimension reference.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not images:
        raise ValueError("No images to save")

    # If we have an original PDF, get its page sizes to set correct dimensions
    if original_pdf is not None:
        page_sizes = _get_page_sizes(original_pdf)
    else:
        page_sizes = None

    # Convert images to RGB if necessary
    rgb_images = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        rgb_images.append(img)

    if page_sizes and len(page_sizes) == len(rgb_images):
        # Resize images to match original PDF dimensions at the rendering DPI
        # PDF points are 1/72 inch; we assume images were rendered at a consistent DPI
        resized = []
        for img, (w_pts, h_pts) in zip(rgb_images, page_sizes):
            # Estimate DPI from image vs page size
            dpi_x = img.width / (w_pts / 72)
            dpi_y = img.height / (h_pts / 72)
            dpi = round((dpi_x + dpi_y) / 2)
            # Set the DPI info so PDF is created at correct physical size
            img.info["dpi"] = (dpi, dpi)
            resized.append(img)
        rgb_images = resized

    # Save as PDF
    first, *rest = rgb_images
    first.save(str(output_path), "PDF", save_all=True, append_images=rest if rest else [],
               resolution=rgb_images[0].info.get("dpi", (300, 300))[0])
    logger.info("Saved PDF to %s (%d pages)", output_path, len(rgb_images))


def get_pdf_info(pdf_path: str | Path) -> dict:
    """Get basic info about a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with page_count, pages (list of {width, height} in points).
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        box = page.mediabox
        pages.append({
            "width": float(box.width),
            "height": float(box.height),
        })
    return {
        "page_count": len(reader.pages),
        "pages": pages,
        "filename": pdf_path.name,
    }


def _get_page_sizes(pdf_path: str | Path) -> list[tuple[float, float]]:
    """Get (width, height) in points for each page."""
    info = get_pdf_info(pdf_path)
    return [(p["width"], p["height"]) for p in info["pages"]]
