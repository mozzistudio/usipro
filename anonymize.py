#!/usr/bin/env python3
"""
USI-PRO Technical Drawing Anonymization Pipeline

This script anonymizes technical drawings by extracting clean drawing views
and rebuilding them on blank pages, removing all proprietary information
(logos, title blocks, company names) while preserving technical content
(geometry, dimensions, tolerances) and adding standardized USI-PRO branding.

APPROACH: Extract and Rebuild
Instead of trying to detect what to remove (blacklist), we extract individual
drawing views and rebuild them on a clean white page.

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
from typing import Optional, List, Tuple

import numpy as np
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


@dataclass
class DrawingView:
    """A single drawing view extracted from a page."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    image: Image.Image  # Cropped image of this view
    center_x: float  # Center X position (normalized 0-1)
    center_y: float  # Center Y position (normalized 0-1)
    area: int  # Bounding box area in pixels


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


# =============================================================================
# NEW APPROACH: Extract and Rebuild
# =============================================================================

def label_connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask using iterative flood-fill.

    Uses numpy only (no OpenCV or scikit-image).

    Args:
        binary_mask: 2D boolean array where True = foreground pixel

    Returns:
        labels: 2D array of same shape with component labels (0 = background)
        num_labels: Number of unique labels (excluding background)
    """
    height, width = binary_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 0

    # Find all foreground pixels
    foreground_coords = np.argwhere(binary_mask)

    if len(foreground_coords) == 0:
        return labels, 0

    # Create a set of unlabeled foreground pixels for fast lookup
    unlabeled = set(map(tuple, foreground_coords))

    while unlabeled:
        # Start a new component
        current_label += 1
        seed = unlabeled.pop()

        # BFS flood fill
        queue = [seed]
        labels[seed[0], seed[1]] = current_label

        while queue:
            y, x = queue.pop(0)

            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = y + dy, x + dx

                    if (ny, nx) in unlabeled:
                        unlabeled.remove((ny, nx))
                        labels[ny, nx] = current_label
                        queue.append((ny, nx))

    return labels, current_label


def get_component_bboxes(labels: np.ndarray, num_labels: int) -> List[Tuple[int, int, int, int, int]]:
    """
    Get bounding boxes for each labeled component.

    Returns:
        List of (x1, y1, x2, y2, pixel_count) for each component
    """
    bboxes = []

    for label_id in range(1, num_labels + 1):
        coords = np.argwhere(labels == label_id)
        if len(coords) == 0:
            continue

        y_coords = coords[:, 0]
        x_coords = coords[:, 1]

        x1 = int(np.min(x_coords))
        y1 = int(np.min(y_coords))
        x2 = int(np.max(x_coords))
        y2 = int(np.max(y_coords))
        pixel_count = len(coords)

        bboxes.append((x1, y1, x2, y2, pixel_count))

    return bboxes


def cluster_nearby_components(
    bboxes: List[Tuple[int, int, int, int, int]],
    page_width: int,
    page_height: int,
    distance_threshold: float = 0.015
) -> List[List[int]]:
    """
    Group nearby bounding boxes into clusters.

    Two components belong to the same cluster if they are within
    distance_threshold * page_size of each other.

    Args:
        bboxes: List of (x1, y1, x2, y2, pixel_count) tuples
        page_width: Width of the page in pixels
        page_height: Height of the page in pixels
        distance_threshold: Max distance as fraction of page size (default 1.5%)

    Returns:
        List of clusters, where each cluster is a list of bbox indices
    """
    if not bboxes:
        return []

    n = len(bboxes)
    max_dist = max(page_width, page_height) * distance_threshold

    # Union-Find for clustering
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Check distances between all pairs of bounding boxes
    for i in range(n):
        x1_i, y1_i, x2_i, y2_i, _ = bboxes[i]
        center_i = ((x1_i + x2_i) / 2, (y1_i + y2_i) / 2)

        for j in range(i + 1, n):
            x1_j, y1_j, x2_j, y2_j, _ = bboxes[j]
            center_j = ((x1_j + x2_j) / 2, (y1_j + y2_j) / 2)

            # Calculate minimum distance between bounding boxes
            # (not center-to-center, but edge-to-edge)
            dx = max(0, max(x1_i, x1_j) - min(x2_i, x2_j))
            dy = max(0, max(y1_i, y1_j) - min(y2_i, y2_j))
            dist = np.sqrt(dx * dx + dy * dy)

            if dist <= max_dist:
                union(i, j)

    # Group by cluster
    clusters_dict = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(i)

    return list(clusters_dict.values())


def merge_cluster_bboxes(
    bboxes: List[Tuple[int, int, int, int, int]],
    cluster: List[int]
) -> Tuple[int, int, int, int, int]:
    """
    Merge bounding boxes of all components in a cluster.

    Returns:
        (x1, y1, x2, y2, total_pixel_count)
    """
    x1 = min(bboxes[i][0] for i in cluster)
    y1 = min(bboxes[i][1] for i in cluster)
    x2 = max(bboxes[i][2] for i in cluster)
    y2 = max(bboxes[i][3] for i in cluster)
    total_pixels = sum(bboxes[i][4] for i in cluster)

    return (x1, y1, x2, y2, total_pixels)


def filter_drawing_views(
    cluster_bboxes: List[Tuple[int, int, int, int, int]],
    page_width: int,
    page_height: int
) -> List[Tuple[int, int, int, int, int]]:
    """
    Filter clusters to keep only likely drawing views.

    Removes:
    - Clusters too small to be real drawing views (< 0.5% of page area)
    - Clusters in title block zone (bottom 20%, right 45%) with high density
    - Clusters that look like text (very elongated with high density)

    Keeps everything else - when in doubt, keep it.
    """
    page_area = page_width * page_height
    min_area = page_area * 0.005  # 0.5% of page area

    kept_views = []

    for bbox in cluster_bboxes:
        x1, y1, x2, y2, pixel_count = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height

        if bbox_area == 0:
            continue

        # Filter 1: Too small (< 0.5% of page)
        if bbox_area < min_area:
            continue

        # Calculate fill ratio (how dense is the content within bbox)
        fill_ratio = pixel_count / bbox_area if bbox_area > 0 else 0

        # Calculate center position (normalized 0-1)
        center_x = (x1 + x2) / 2 / page_width
        center_y = (y1 + y2) / 2 / page_height

        # Filter 2: Title block zone detection
        # Title blocks are typically in bottom-right corner
        # Bottom 20% of page AND right 45% AND dense (>15% fill)
        in_title_block_zone = (center_y > 0.80) and (center_x > 0.55)
        is_dense = fill_ratio > 0.15

        if in_title_block_zone and is_dense:
            continue

        # Filter 3: Text-like shapes (very elongated with high density)
        # Table rows, text lines have extreme aspect ratios
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1

        is_text_like = (
            (aspect_ratio > 8 or aspect_ratio < 0.125) and  # Very elongated
            fill_ratio > 0.15 and  # Dense
            bbox_area < page_area * 0.02  # And relatively small
        )

        if is_text_like:
            continue

        # Filter 4: Edge elements (frame borders, margin content)
        # If mostly in the outer 5% margin and very thin
        in_margin = (
            (center_x < 0.05 or center_x > 0.95 or center_y < 0.05 or center_y > 0.95) and
            (bbox_width < page_width * 0.03 or bbox_height < page_height * 0.03)
        )

        if in_margin:
            continue

        # Keep this cluster as a drawing view
        kept_views.append(bbox)

    return kept_views


def detect_drawing_views(img: Image.Image) -> List[DrawingView]:
    """
    Detect individual drawing views on a PDF page.

    Uses connected component analysis and clustering to find
    separate drawing views (front, side, top, section views, etc.)

    Args:
        img: PIL Image of the rendered PDF page (RGB, 300 DPI)

    Returns:
        List of DrawingView objects, each containing a cropped view
    """
    width, height = img.size

    # Step 1: Convert to grayscale and threshold to binary
    gray = np.array(img.convert('L'))
    binary = gray < 200  # Pixels darker than 200 are "content"

    # Step 2: Remove border lines (outer 3% of each edge)
    border_x = int(width * 0.03)
    border_y = int(height * 0.03)
    binary[:border_y, :] = False  # Top border
    binary[-border_y:, :] = False  # Bottom border
    binary[:, :border_x] = False  # Left border
    binary[:, -border_x:] = False  # Right border

    # Step 3: Label connected components
    labels, num_labels = label_connected_components(binary)

    if num_labels == 0:
        # No content found - return empty
        return []

    # Step 4: Get bounding boxes for each component
    component_bboxes = get_component_bboxes(labels, num_labels)

    if not component_bboxes:
        return []

    # Step 5: Cluster nearby components
    clusters = cluster_nearby_components(component_bboxes, width, height)

    # Step 6: Merge clusters into single bounding boxes
    cluster_bboxes = [
        merge_cluster_bboxes(component_bboxes, cluster)
        for cluster in clusters
    ]

    # Step 7: Filter to keep only drawing views
    view_bboxes = filter_drawing_views(cluster_bboxes, width, height)

    if not view_bboxes:
        # If all clusters were filtered out, take the largest one as fallback
        if cluster_bboxes:
            largest = max(cluster_bboxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            view_bboxes = [largest]
        else:
            return []

    # Step 8: Extract each view from the ORIGINAL image
    views = []

    for bbox in view_bboxes:
        x1, y1, x2, y2, _ = bbox

        # Add padding (1% of crop size)
        pad_x = int((x2 - x1) * 0.01)
        pad_y = int((y2 - y1) * 0.01)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        # Crop from original RGB image (not binary)
        cropped = img.crop((x1, y1, x2, y2))

        # Calculate normalized center position
        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        area = (x2 - x1) * (y2 - y1)

        view = DrawingView(
            bbox=(x1, y1, x2, y2),
            image=cropped,
            center_x=center_x,
            center_y=center_y,
            area=area
        )
        views.append(view)

    return views


def rebuild_page(
    views: List[DrawingView],
    page_width: int,
    page_height: int,
    footer_height_ratio: float = FOOTER_RATIO
) -> Image.Image:
    """
    Rebuild detected views on a blank white page.

    Preserves the original relative positions of views (front view below
    top view, side view to the right, etc.) and scales to fill available space.

    Args:
        views: List of DrawingView objects to place
        page_width: Width of the output page in pixels
        page_height: Height of the output page in pixels
        footer_height_ratio: Fraction of page height reserved for footer

    Returns:
        New white image with views placed on it
    """
    # Create blank white page
    result = Image.new('RGB', (page_width, page_height), 'white')

    if not views:
        return result

    # Calculate available area (above footer)
    footer_height = int(page_height * footer_height_ratio)
    available_height = page_height - footer_height
    available_width = page_width

    # Add margins (2% on each side)
    margin_x = int(page_width * 0.02)
    margin_y = int(available_height * 0.02)

    content_width = available_width - 2 * margin_x
    content_height = available_height - 2 * margin_y

    if len(views) == 1:
        # Single view: center it in available space
        view = views[0]
        img = view.image

        # Calculate scale to fit in available area while preserving aspect ratio
        scale_x = content_width / img.width
        scale_y = content_height / img.height
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale

        new_width = int(img.width * scale)
        new_height = int(img.height * scale)

        if scale < 1.0:
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Center in available area
        paste_x = margin_x + (content_width - new_width) // 2
        paste_y = footer_height + margin_y + (content_height - new_height) // 2

        result.paste(img, (paste_x, paste_y))

    else:
        # Multiple views: preserve relative positions

        # Find bounding box of all views' original positions
        min_cx = min(v.center_x for v in views)
        max_cx = max(v.center_x for v in views)
        min_cy = min(v.center_y for v in views)
        max_cy = max(v.center_y for v in views)

        # Calculate the span of views in original coordinates
        span_x = max_cx - min_cx if max_cx > min_cx else 1.0
        span_y = max_cy - min_cy if max_cy > min_cy else 1.0

        # Calculate total size of all views if placed with original relative positions
        # We need to figure out the bounding box of placed views
        total_width = 0
        total_height = 0
        for v in views:
            # Normalize position to 0-1 within the span
            rel_x = (v.center_x - min_cx) / span_x if span_x > 0 else 0.5
            rel_y = (v.center_y - min_cy) / span_y if span_y > 0 else 0.5

            # Calculate extent
            right = rel_x * content_width + v.image.width / 2
            bottom = rel_y * content_height + v.image.height / 2
            total_width = max(total_width, right)
            total_height = max(total_height, bottom)

        # Calculate scale to fit all views
        scale_x = content_width / total_width if total_width > 0 else 1.0
        scale_y = content_height / total_height if total_height > 0 else 1.0
        scale = min(scale_x, scale_y, 1.0)

        # Place each view
        for v in views:
            img = v.image

            # Scale the view image
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)

            if scale < 1.0 and new_width > 0 and new_height > 0:
                img = img.resize((new_width, new_height), Image.LANCZOS)

            # Calculate position based on relative coordinates
            rel_x = (v.center_x - min_cx) / span_x if span_x > 0 else 0.5
            rel_y = (v.center_y - min_cy) / span_y if span_y > 0 else 0.5

            # Position in content area (note: y is inverted - lower center_y = higher on page)
            center_x = margin_x + int(rel_x * content_width * scale)
            center_y = footer_height + margin_y + int((1 - rel_y) * content_height * scale)

            paste_x = center_x - new_width // 2
            paste_y = center_y - new_height // 2

            # Clamp to valid positions
            paste_x = max(0, min(paste_x, page_width - new_width))
            paste_y = max(footer_height, min(paste_y, page_height - new_height))

            result.paste(img, (paste_x, paste_y))

    return result


def add_footer_to_image(
    img: Image.Image,
    plan_id: str,
    zip_name: str,
    extracted_data: ExtractedData,
    output_path: Path
) -> Path:
    """
    Add USI-PRO footer to an image and save as PDF.

    Footer has three zones:
    - Left: Plan ID (red) with ZIP name above
    - Center: USI-PRO logo
    - Right: Data table (drawing no, designation, material, finish)
    """
    page_width = img.width * 72 / DPI
    page_height = img.height * 72 / DPI
    footer_height = page_height * FOOTER_RATIO

    # Create PDF canvas
    c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    # Save image temporarily for embedding
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name, 'PNG')
        temp_img_path = tmp.name

    try:
        # Draw the image
        c.drawImage(
            temp_img_path,
            0, 0,
            width=page_width,
            height=page_height,
            preserveAspectRatio=True,
            anchor='sw'
        )
    finally:
        os.unlink(temp_img_path)

    # Draw footer background (optional: light line at top of footer)
    c.setStrokeColor(HexColor("#CCCCCC"))
    c.setLineWidth(0.3)
    c.line(0, footer_height, page_width, footer_height)

    margin = 10

    # LEFT ZONE - Plan ID
    left_x = margin + 20

    # ZIP name in small black text
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


def anonymize_page(
    pdf_path: Path,
    page_num: int,
    plan_id: str,
    zip_name: str,
    extracted_data: ExtractedData,
    output_path: Path
) -> Path:
    """
    Anonymize a single PDF page using the EXTRACT AND REBUILD approach.

    Steps:
    1. Render page to high-res image (300 DPI)
    2. Detect individual drawing views (clusters of geometry)
    3. Extract each view from the original image
    4. Rebuild views on a blank white page, preserving relative positions
    5. Add USI-PRO footer (plan ID, logo, data table)
    """
    # Step 1: Render page to high-res image
    img = render_page_to_image(pdf_path, page_num, DPI)
    page_width, page_height = img.size

    # Step 2: Detect drawing views
    views = detect_drawing_views(img)

    # Step 3 & 4: Rebuild on blank page (extraction happens in detect_drawing_views)
    rebuilt_img = rebuild_page(views, page_width, page_height)

    # Step 5: Add footer and save as PDF
    add_footer_to_image(rebuilt_img, plan_id, zip_name, extracted_data, output_path)

    return output_path


def merge_pdfs(pdf_paths: List[Path], output_path: Path):
    """
    Merge multiple single-page PDFs into one multi-page PDF.
    Uses reportlab to create a new PDF with all pages.
    """
    if not pdf_paths:
        return

    if len(pdf_paths) == 1:
        shutil.copy(str(pdf_paths[0]), str(output_path))
        return

    # Get page sizes from each PDF
    page_infos = []
    for pdf_path in pdf_paths:
        pdf = pdfium.PdfDocument(str(pdf_path))
        page = pdf[0]
        page_infos.append((page.get_width(), page.get_height()))
        pdf.close()

    # Create merged PDF using reportlab
    c = canvas.Canvas(str(output_path))

    for idx, pdf_path in enumerate(pdf_paths):
        page_width, page_height = page_infos[idx]

        # Set page size for this page
        c.setPageSize((page_width, page_height))

        # Render the PDF page to image
        pdf = pdfium.PdfDocument(str(pdf_path))
        page = pdf[0]
        scale = DPI / 72.0
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        pdf.close()

        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            pil_image.save(tmp.name, 'PNG')
            temp_img_path = tmp.name

        try:
            c.drawImage(temp_img_path, 0, 0, width=page_width, height=page_height)
        finally:
            os.unlink(temp_img_path)

        # Add new page (except for the last one)
        if idx < len(pdf_paths) - 1:
            c.showPage()

    c.save()


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

        # Merge all pages into single PDF
        if temp_pages:
            merge_pdfs(temp_pages, output_pdf)
            # Clean up temp files
            for temp_path in temp_pages:
                if temp_path.exists():
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
        pdf_files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))

        if not zip_files and not pdf_files:
            print(f"No ZIP or PDF files found in {input_path}")
            sys.exit(1)

        total_files = len(zip_files) + len(pdf_files)
        print(f"Found {total_files} file(s) to process ({len(zip_files)} ZIPs, {len(pdf_files)} PDFs)")

        for zip_file in zip_files:
            process_zip(zip_file, output_dir)

        for pdf_file in pdf_files:
            process_single_pdf(pdf_file, output_dir, None)

        print(f"\nAll done! Processed {total_files} files.")

    else:
        print(f"Error: Invalid input: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
