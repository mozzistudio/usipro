#!/usr/bin/env python3
"""
USI-PRO Technical Drawing Anonymization Pipeline

This script anonymizes technical drawings by removing proprietary information
(logos, title blocks, company names) while preserving technical content
(geometry, dimensions, tolerances) and adding standardized USI-PRO branding.

Usage:
    python anonymize.py input/454323.zip           # Process a ZIP file
    python anonymize.py input/drawing.pdf --plan-id 1928  # Process single PDF
    python anonymize.py input/                     # Batch process all ZIPs
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from reportlab.lib.pagesizes import A3, A4, landscape, portrait
from reportlab.lib.units import mm
from reportlab.lib.colors import Color, black, white, red, HexColor
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

# Constants
DPI = 300
FOOTER_RATIO = 0.045  # 4.5% of page height for footer
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Known formats and their characteristics
KNOWN_FORMATS = {
    "SILATECH": ["silatech", "sila-tech"],
    "ETUDEMA": ["etudema"],
    "INOVEOS": ["inoveos"],
    "A.C. Concept": ["a.c. concept", "ac concept", "a.c.concept"],
    "PRANA": ["prana", "prâna", "prāna"],
    "AZURLIGHT": ["azurlight", "azur light", "azur-light"],
}

# Standard page sizes in points (72 points = 1 inch)
PAGE_SIZES = {
    "A4": (595, 842),
    "A3": (842, 1191),
    "A2": (1191, 1684),
    "A1": (1684, 2384),
    "A0": (2384, 3370),
}


@dataclass
class ExtractedData:
    """Data extracted from a technical drawing's title block."""
    drawing_no: str = "—"
    designation: str = "—"
    material: str = "—"
    finish: str = "—"


@dataclass
class PageInfo:
    """Information about a PDF page."""
    width: float  # in points
    height: float  # in points
    orientation: str  # 'portrait' or 'landscape'
    size_name: str  # 'A4', 'A3', etc.
    page_number: int


@dataclass
class PDFAnalysis:
    """Analysis results for a PDF file."""
    filename: str
    page_count: int
    pages: list[PageInfo] = field(default_factory=list)
    detected_format: str = "unknown"
    extracted_data: ExtractedData = field(default_factory=ExtractedData)
    is_scan: bool = False


def detect_page_size(width: float, height: float) -> tuple[str, str]:
    """Detect the standard page size and orientation."""
    # Normalize to portrait for comparison
    w, h = min(width, height), max(width, height)
    orientation = "landscape" if width > height else "portrait"

    # Find closest match
    best_match = "custom"
    best_diff = float("inf")

    for name, (sw, sh) in PAGE_SIZES.items():
        diff = abs(w - sw) + abs(h - sh)
        if diff < best_diff:
            best_diff = diff
            best_match = name

    # Allow 5% tolerance
    if best_diff > (w + h) * 0.05:
        best_match = f"custom ({int(width)}x{int(height)})"

    return best_match, orientation


def detect_format(text: str) -> tuple[str, bool]:
    """Detect the drawing format based on text content."""
    text_lower = text.lower()

    # Check for known formats
    for format_name, keywords in KNOWN_FORMATS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return format_name, False

    # Check if it's a scan (usually has less extractable text)
    if len(text.strip()) < 100:
        return "scan detected", True

    return "unknown", False


def extract_drawing_data(text: str) -> ExtractedData:
    """Extract technical data from PDF text content."""
    data = ExtractedData()
    text_upper = text.upper()
    lines = text.split('\n')

    # Common patterns for data extraction
    patterns = {
        'drawing_no': [
            r'(?:DRAWING\s*(?:NO\.?|NUMBER)|PLAN\s*(?:NO\.?|N°)|REF(?:ERENCE)?|N°\s*PLAN|DWG\s*NO\.?)\s*[:\s]*([A-Z0-9][-A-Z0-9_.]+)',
            r'(?:^|\s)(\d{4,6})(?:\s|$)',  # 4-6 digit numbers that could be plan IDs
        ],
        'designation': [
            r'(?:DESIGNATION|DESCRIPTION|TITLE|DÉNOMINATION|NOM)\s*[:\s]*([A-ZÀ-Ü][A-ZÀ-Ü0-9\s\-_.]+)',
            r'(?:PART\s*NAME|PIECE)\s*[:\s]*([A-ZÀ-Ü][A-ZÀ-Ü0-9\s\-_.]+)',
        ],
        'material': [
            r'(?:MATERIAL|MATIÈRE|MATIERE|MAT\.?|WERKSTOFF)\s*[:\s]*([A-Z0-9][A-Z0-9\s\-_.]+)',
            r'(?:EN\s*AW[-\s]?\d+[A-Z0-9\-]*)',
            r'(?:INOX\s*\d+[A-Z]*)',
            r'(?:ISO\s*\d+[-\w]*)',
            r'(?:ALUMINIUM|ALUMINUM|STEEL|STAINLESS|BRASS|BRONZE|COPPER|TITANIUM)',
        ],
        'finish': [
            r'(?:FINISH|FINITION|TRAITEMENT|SURFACE|REVÊTEMENT|REVETEMENT)\s*[:\s]*([A-ZÀ-Ü][A-ZÀ-Ü0-9\s\-_.]+)',
            r'(?:ANODIS[ÉE]+\s*\w*|ANODIZED\s*\w*)',
            r'(?:SABLÉ|SANDBLASTED|BEAD\s*BLAST)',
            r'(?:POLI|POLISHED)',
            r'(?:CHROMÉ|CHROMED|CHROME)',
            r'(?:NICKELÉ|NICKELED|NICKEL)',
            r'(?:PEINT|PAINTED)',
        ],
    }

    # Try to extract each field
    for field_name, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text_upper, re.MULTILINE | re.IGNORECASE)
            if match:
                value = match.group(1) if match.lastindex else match.group(0)
                # Clean up: take only first line, strip whitespace
                value = value.split('\n')[0].strip()
                # Remove trailing keywords that may have been captured
                for keyword in ['MATERIAL', 'FINISH', 'DESIGNATION', 'DRAWING', 'DESIGNER']:
                    if value.endswith(keyword):
                        value = value[:-len(keyword)].strip()
                if value and len(value) > 1:
                    # Clean up and translate common terms
                    value = translate_to_english(value)
                    setattr(data, field_name, value)
                    break

    return data


def translate_to_english(text: str) -> str:
    """Translate common French terms to English."""
    translations = {
        'ANODISÉ NOIR': 'Black anodized',
        'ANODISE NOIR': 'Black anodized',
        'ANODISÉ': 'Anodized',
        'ANODISE': 'Anodized',
        'SABLÉ': 'Sandblasted',
        'SABLE': 'Sandblasted',
        'POLI': 'Polished',
        'CHROMÉ': 'Chrome plated',
        'CHROME': 'Chrome plated',
        'NICKELÉ': 'Nickel plated',
        'NICKELE': 'Nickel plated',
        'PEINT': 'Painted',
        'BRUT': 'Raw',
        'USINÉ': 'Machined',
        'USINE': 'Machined',
        'MATIÈRE': 'Material',
        'MATIERE': 'Material',
        'ACIER': 'Steel',
        'LAITON': 'Brass',
        'CUIVRE': 'Copper',
        'ALUMINIUM': 'Aluminum',
        'SUPPORT': 'BRACKET',
        'FIXATION': 'MOUNTING',
        'AXE': 'SHAFT',
        'PLATINE': 'PLATE',
    }

    result = text.upper()
    for fr, en in translations.items():
        result = result.replace(fr, en)

    return result.strip()


def analyze_pdf(pdf_path: Path) -> PDFAnalysis:
    """Analyze a PDF file and extract metadata."""
    analysis = PDFAnalysis(filename=pdf_path.name, page_count=0)

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        analysis.page_count = len(pdf)

        all_text = []

        for i, page in enumerate(pdf):
            width = page.get_width()
            height = page.get_height()
            size_name, orientation = detect_page_size(width, height)

            page_info = PageInfo(
                width=width,
                height=height,
                orientation=orientation,
                size_name=size_name,
                page_number=i + 1
            )
            analysis.pages.append(page_info)

            # Extract text for format detection and data extraction
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            all_text.append(text)

        # Combine all text for analysis
        combined_text = '\n'.join(all_text)
        analysis.detected_format, analysis.is_scan = detect_format(combined_text)
        analysis.extracted_data = extract_drawing_data(combined_text)

        # If no drawing number found, try to extract from filename
        if analysis.extracted_data.drawing_no == "—":
            # Try to find a number in the filename
            match = re.search(r'(\d{3,6})', pdf_path.stem)
            if match:
                analysis.extracted_data.drawing_no = match.group(1)

        pdf.close()

    except Exception as e:
        print(f"  Warning: Error analyzing {pdf_path.name}: {e}")

    return analysis


def render_page_to_image(pdf_path: Path, page_num: int, dpi: int = DPI) -> Image.Image:
    """Render a PDF page to a PIL Image at specified DPI."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_num]

    # Calculate scale factor for desired DPI (PDF default is 72 DPI)
    scale = dpi / 72.0

    bitmap = page.render(scale=scale, rotation=0)
    pil_image = bitmap.to_pil()

    pdf.close()
    return pil_image


def detect_drawing_region(img: Image.Image) -> tuple[int, int, int, int]:
    """
    WHITELIST APPROACH: Detect the main technical drawing region.

    Instead of trying to detect what to REMOVE (blacklist), we detect what to KEEP.
    The technical drawing is the largest dense cluster of lines/shapes on the page.
    Everything outside this region will be white-filled.

    Returns (x1, y1, x2, y2) bounding box of the drawing area.
    """
    import numpy as np

    width, height = img.size

    # Convert to grayscale
    gray = np.array(img.convert('L'))

    # Threshold: pixels darker than 240 are considered "content"
    # (lines, text, shapes - anything that's not pure white background)
    content_mask = gray < 240

    # Find coordinates of all content pixels
    content_coords = np.where(content_mask)

    if len(content_coords[0]) == 0:
        # No content found - return full page minus small margin
        margin = int(min(width, height) * 0.02)
        return (margin, margin, width - margin, height - margin)

    # Get bounding box of ALL content
    y_coords = content_coords[0]
    x_coords = content_coords[1]

    # Simple approach: find the bounding box of all content
    # Then we'll refine by looking at content density
    raw_x1, raw_x2 = int(np.min(x_coords)), int(np.max(x_coords))
    raw_y1, raw_y2 = int(np.min(y_coords)), int(np.max(y_coords))

    # Divide the image into a grid to analyze content density
    # This helps distinguish the main drawing from title blocks/notes
    grid_size = 20  # 20x20 grid
    cell_h = height // grid_size
    cell_w = width // grid_size

    # Calculate density for each cell
    density_grid = np.zeros((grid_size, grid_size))
    for gy in range(grid_size):
        for gx in range(grid_size):
            y1 = gy * cell_h
            y2 = min((gy + 1) * cell_h, height)
            x1 = gx * cell_w
            x2 = min((gx + 1) * cell_w, width)

            cell = content_mask[y1:y2, x1:x2]
            if cell.size > 0:
                density_grid[gy, gx] = np.sum(cell) / cell.size

    # Find cells with significant content (> 1% dark pixels)
    # Technical drawings have thin lines so even 1% is significant
    significant_threshold = 0.01
    significant_cells = density_grid > significant_threshold

    # Find the bounding box of significant cells
    sig_coords = np.where(significant_cells)

    if len(sig_coords[0]) == 0:
        # Fall back to raw bounding box
        x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2
    else:
        # Get grid cell bounds
        grid_y1, grid_y2 = int(np.min(sig_coords[0])), int(np.max(sig_coords[0]))
        grid_x1, grid_x2 = int(np.min(sig_coords[1])), int(np.max(sig_coords[1]))

        # Convert back to pixel coordinates
        x1 = grid_x1 * cell_w
        y1 = grid_y1 * cell_h
        x2 = min((grid_x2 + 1) * cell_w, width)
        y2 = min((grid_y2 + 1) * cell_h, height)

    # Refine: within this region, find the actual content bounds
    # This removes empty space within the grid cells
    region_mask = content_mask[y1:y2, x1:x2]
    region_coords = np.where(region_mask)

    if len(region_coords[0]) > 0:
        inner_y1 = int(np.min(region_coords[0]))
        inner_y2 = int(np.max(region_coords[0]))
        inner_x1 = int(np.min(region_coords[1]))
        inner_x2 = int(np.max(region_coords[1]))

        # Adjust to absolute coordinates
        x1 = x1 + inner_x1
        y1 = y1 + inner_y1
        x2 = x1 + (inner_x2 - inner_x1)
        y2 = y1 + (inner_y2 - inner_y1)

    # Add margin around the detected region (3% of page dimensions)
    # This ensures we don't clip any part of the drawing
    margin_x = int(width * 0.03)
    margin_y = int(height * 0.03)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)

    return (x1, y1, x2, y2)


def white_fill_outside_region(img: Image.Image, keep_region: tuple[int, int, int, int]) -> Image.Image:
    """
    White-fill everything OUTSIDE the specified region.
    This is the core of the whitelist approach.

    Args:
        img: Source image
        keep_region: (x1, y1, x2, y2) bounding box to KEEP

    Returns:
        Image with everything outside keep_region white-filled
    """
    width, height = img.size
    x1, y1, x2, y2 = keep_region

    # Create a new white image
    result = Image.new('RGB', (width, height), 'white')

    # Paste the region we want to keep
    # Ensure coordinates are valid
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 > x1 and y2 > y1:
        # Crop the region to keep from the original
        region_to_keep = img.crop((x1, y1, x2, y2))
        # Paste it onto the white background
        result.paste(region_to_keep, (x1, y1))

    return result


def create_footer_overlay(
    page_width: float,
    page_height: float,
    plan_id: str,
    zip_name: str,
    extracted_data: ExtractedData,
    output_path: Path
) -> Path:
    """Create a PDF overlay with the USI-PRO footer."""
    footer_height = page_height * FOOTER_RATIO
    footer_y = 0  # Start from bottom

    # Create canvas
    c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    # Footer area starts at bottom
    margin = 10

    # LEFT ZONE - Plan ID
    left_x = margin + 20

    # ZIP name in small black text above plan ID
    c.setFont("Helvetica", 6)
    c.setFillColor(black)
    c.drawString(left_x, footer_y + footer_height * 0.65, zip_name)

    # Plan ID in red bold
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(red)
    c.drawString(left_x, footer_y + footer_height * 0.25, plan_id)

    # CENTER ZONE - Logo
    if LOGO_PATH.exists():
        logo = ImageReader(str(LOGO_PATH))
        logo_width = 80
        logo_height = 25
        logo_x = (page_width - logo_width) / 2
        logo_y = footer_y + (footer_height - logo_height) / 2
        c.drawImage(logo, logo_x, logo_y, width=logo_width, height=logo_height,
                   preserveAspectRatio=True, mask='auto')

    # RIGHT ZONE - Data Table
    table_right = page_width - margin - 20
    table_width = 180
    table_left = table_right - table_width
    row_height = footer_height * 0.22
    col_width = table_width / 2

    # Table data
    rows = [
        ("DRAWING NO.", extracted_data.drawing_no),
        ("DESIGNATION", extracted_data.designation[:25] + "..." if len(extracted_data.designation) > 25 else extracted_data.designation),
        ("MATERIAL / STANDARD", extracted_data.material[:20] + "..." if len(extracted_data.material) > 20 else extracted_data.material),
        ("FINISH", extracted_data.finish[:20] + "..." if len(extracted_data.finish) > 20 else extracted_data.finish),
    ]

    # Draw table
    c.setStrokeColor(HexColor("#CCCCCC"))
    c.setLineWidth(0.4)

    for i, (label, value) in enumerate(rows):
        row_y = footer_y + footer_height - (i + 1) * row_height

        # Header cell (gray background)
        c.setFillColor(HexColor("#E5E5E5"))
        c.rect(table_left, row_y, col_width, row_height, fill=1, stroke=1)

        # Value cell (white background)
        c.setFillColor(white)
        c.rect(table_left + col_width, row_y, col_width, row_height, fill=1, stroke=1)

        # Label text
        c.setFillColor(black)
        c.setFont("Helvetica", 5)
        c.drawString(table_left + 2, row_y + row_height * 0.35, label)

        # Value text
        c.setFont("Helvetica", 5)
        c.drawString(table_left + col_width + 2, row_y + row_height * 0.35, value)

    c.save()
    return output_path


def anonymize_page(
    pdf_path: Path,
    page_num: int,
    plan_id: str,
    zip_name: str,
    extracted_data: ExtractedData,
    output_path: Path
) -> Path:
    """Anonymize a single PDF page and save as new PDF.

    WHITELIST APPROACH:
    Instead of trying to detect and remove specific elements (blacklist),
    we detect the main technical drawing region and keep ONLY that.
    Everything else is automatically white-filled.

    This works regardless of client format because technical drawings
    share common characteristics: dense clusters of lines, dimensions,
    and geometric shapes in the center area of the page.
    """
    # Step 1: Render page to high-res image (300 DPI)
    img = render_page_to_image(pdf_path, page_num, DPI)

    # Step 2: WHITELIST APPROACH - Detect the drawing region to KEEP
    # This finds the bounding box of the main technical drawing content
    drawing_region = detect_drawing_region(img)

    # Step 3: White-fill everything OUTSIDE the drawing region
    # This automatically removes: title blocks, logos, company names,
    # BOM tables, revision tables, notes, stamps, frame borders, etc.
    img = white_fill_outside_region(img, drawing_region)

    # Step 4: Calculate page dimensions
    page_width = img.width * 72 / DPI
    page_height = img.height * 72 / DPI

    # Step 5: Create the final PDF with image and footer
    c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    # Calculate footer height
    footer_height = page_height * FOOTER_RATIO

    # Draw the anonymized image (scaled to fit above footer)
    drawing_height = page_height - footer_height

    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name, 'PNG')
        temp_img_path = tmp.name

    try:
        # Draw image above footer area
        c.drawImage(
            temp_img_path,
            0,
            footer_height,
            width=page_width,
            height=drawing_height,
            preserveAspectRatio=True,
            anchor='c'
        )
    finally:
        os.unlink(temp_img_path)

    # Draw footer
    margin = 10

    # LEFT ZONE - Plan ID
    left_x = margin + 20

    # ZIP name in small black text above plan ID
    c.setFont("Helvetica", 6)
    c.setFillColor(black)
    c.drawString(left_x, footer_height * 0.65, zip_name)

    # Plan ID in red bold
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(red)
    c.drawString(left_x, footer_height * 0.25, plan_id)

    # CENTER ZONE - Logo
    if LOGO_PATH.exists():
        logo = ImageReader(str(LOGO_PATH))
        logo_width = 80
        logo_height = 25
        logo_x = (page_width - logo_width) / 2
        logo_y = (footer_height - logo_height) / 2
        c.drawImage(logo, logo_x, logo_y, width=logo_width, height=logo_height,
                   preserveAspectRatio=True, mask='auto')

    # RIGHT ZONE - Data Table
    table_right = page_width - margin - 20
    table_width = 180
    table_left = table_right - table_width
    row_height = footer_height * 0.22
    col_width = table_width / 2

    # Table data
    rows = [
        ("DRAWING NO.", extracted_data.drawing_no),
        ("DESIGNATION", extracted_data.designation[:25] + "..." if len(extracted_data.designation) > 25 else extracted_data.designation),
        ("MATERIAL / STANDARD", extracted_data.material[:20] + "..." if len(extracted_data.material) > 20 else extracted_data.material),
        ("FINISH", extracted_data.finish[:20] + "..." if len(extracted_data.finish) > 20 else extracted_data.finish),
    ]

    # Draw table
    c.setStrokeColor(HexColor("#CCCCCC"))
    c.setLineWidth(0.4)

    for i, (label, value) in enumerate(rows):
        row_y = footer_height - (i + 1) * row_height

        # Header cell (gray background)
        c.setFillColor(HexColor("#E5E5E5"))
        c.rect(table_left, row_y, col_width, row_height, fill=1, stroke=1)

        # Value cell (white background)
        c.setFillColor(white)
        c.rect(table_left + col_width, row_y, col_width, row_height, fill=1, stroke=1)

        # Label text
        c.setFillColor(black)
        c.setFont("Helvetica", 5)
        c.drawString(table_left + 2, row_y + row_height * 0.35, label)

        # Value text
        c.setFont("Helvetica", 5)
        c.drawString(table_left + col_width + 2, row_y + row_height * 0.35, value)

    c.save()
    return output_path


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    plan_id: str,
    zip_name: str,
    analysis: PDFAnalysis
) -> Path:
    """Process a single PDF file through the anonymization pipeline."""
    output_pdf = output_dir / f"{plan_id}.pdf"

    if analysis.page_count == 1:
        # Single page - simple case
        anonymize_page(
            pdf_path,
            0,
            plan_id,
            zip_name,
            analysis.extracted_data,
            output_pdf
        )
    else:
        # Multi-page PDF - process each page and combine
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as pdf_canvas
        from PyPDF2 import PdfWriter, PdfReader

        temp_pages = []

        for i in range(analysis.page_count):
            temp_path = output_dir / f"_temp_page_{i}.pdf"
            anonymize_page(
                pdf_path,
                i,
                plan_id,
                zip_name,
                analysis.extracted_data,
                temp_path
            )
            temp_pages.append(temp_path)

        # Merge pages (simplified - for now just use first page)
        # In production, you'd use PyPDF2 to merge properly
        if temp_pages:
            shutil.move(str(temp_pages[0]), str(output_pdf))
            for temp_path in temp_pages[1:]:
                temp_path.unlink()

    return output_pdf


def process_zip(zip_path: Path, output_dir: Path) -> Path:
    """Process a ZIP file containing technical drawings."""
    zip_name = zip_path.stem

    print(f"\nProcessing: {zip_path.name}")
    print("-" * 50)

    # Create temp directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extract_dir = temp_path / "extracted"
        processed_dir = temp_path / "processed"
        extract_dir.mkdir()
        processed_dir.mkdir()

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Find all PDFs
        pdf_files = list(extract_dir.rglob("*.pdf")) + list(extract_dir.rglob("*.PDF"))

        if not pdf_files:
            print("  No PDF files found in archive!")
            return None

        processed_files = []

        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"[{idx}/{len(pdf_files)}] Analyzing {pdf_path.name}...", end=" ")

            # Analyze PDF
            analysis = analyze_pdf(pdf_path)

            # Get page info
            if analysis.pages:
                page_info = analysis.pages[0]
                print(f"{page_info.size_name} {page_info.orientation}, {analysis.detected_format} format")
            else:
                print("could not analyze")
                continue

            # Determine plan ID
            plan_id = analysis.extracted_data.drawing_no
            if plan_id == "—":
                # Try filename
                match = re.search(r'(\d{3,6})', pdf_path.stem)
                if match:
                    plan_id = match.group(1)
                else:
                    plan_id = pdf_path.stem

            # Print extracted data
            ed = analysis.extracted_data
            print(f"      Extracted: DRAWING NO.={ed.drawing_no}, DESIGNATION={ed.designation}, MATERIAL={ed.material}, FINISH={ed.finish}")

            # Process the PDF
            output_pdf = process_pdf(pdf_path, processed_dir, plan_id, zip_name, analysis)
            processed_files.append(output_pdf)

        # Create output ZIP
        output_zip = output_dir / f"{zip_name}.zip"
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for pdf_file in processed_files:
                zf.write(pdf_file, pdf_file.name)

        print(f"→ Output: {output_zip} ({len(processed_files)} files)")
        return output_zip


def process_single_pdf(pdf_path: Path, output_dir: Path, plan_id: str) -> Path:
    """Process a single PDF file (not from a ZIP)."""
    print(f"\nProcessing: {pdf_path.name}")
    print("-" * 50)

    # Analyze PDF
    analysis = analyze_pdf(pdf_path)

    if analysis.pages:
        page_info = analysis.pages[0]
        print(f"Page: {page_info.size_name} {page_info.orientation}, {analysis.detected_format} format")

    # Use provided plan_id or extract from analysis
    if not plan_id:
        plan_id = analysis.extracted_data.drawing_no
        if plan_id == "—":
            match = re.search(r'(\d{3,6})', pdf_path.stem)
            if match:
                plan_id = match.group(1)
            else:
                plan_id = pdf_path.stem

    # Update extracted data with provided plan_id
    analysis.extracted_data.drawing_no = plan_id

    ed = analysis.extracted_data
    print(f"Extracted: DRAWING NO.={ed.drawing_no}, DESIGNATION={ed.designation}, MATERIAL={ed.material}, FINISH={ed.finish}")

    # Process
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_pdf = process_pdf(pdf_path, temp_path, plan_id, pdf_path.stem, analysis)

        # Move to output directory
        final_path = output_dir / f"{plan_id}.pdf"
        shutil.move(str(output_pdf), str(final_path))

        print(f"→ Output: {final_path}")
        return final_path


def main():
    parser = argparse.ArgumentParser(
        description="USI-PRO Technical Drawing Anonymization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python anonymize.py input/454323.zip           # Process a ZIP file
  python anonymize.py input/drawing.pdf --plan-id 1928  # Process single PDF
  python anonymize.py input/                     # Batch process all ZIPs
        """
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file (ZIP or PDF) or directory containing ZIPs"
    )
    parser.add_argument(
        "--plan-id",
        type=str,
        help="Plan ID for single PDF processing"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: output/)"
    )

    args = parser.parse_args()

    # Determine output directory
    script_dir = Path(__file__).parent
    output_dir = args.output or (script_dir / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        if input_path.suffix.lower() == ".zip":
            # Process single ZIP
            process_zip(input_path, output_dir)
        elif input_path.suffix.lower() == ".pdf":
            # Process single PDF
            process_single_pdf(input_path, output_dir, args.plan_id)
        else:
            print(f"Error: Unsupported file type: {input_path.suffix}")
            sys.exit(1)

    elif input_path.is_dir():
        # Batch process all ZIPs in directory
        zip_files = list(input_path.glob("*.zip")) + list(input_path.glob("*.ZIP"))

        if not zip_files:
            print(f"No ZIP files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(zip_files)} ZIP file(s) to process")

        for zip_file in zip_files:
            process_zip(zip_file, output_dir)

        print(f"\nAll done! Processed {len(zip_files)} archives.")

    else:
        print(f"Error: Invalid input: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
