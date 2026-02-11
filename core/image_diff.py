"""Visual comparison between original and anonymized images."""

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box with integer pixel coordinates."""
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return (x, y, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class DiffResult:
    """Result of a visual diff comparison."""
    changed_pixel_count: int
    total_pixel_count: int
    changed_ratio: float
    diff_image: Image.Image
    regions: list[BBox] = field(default_factory=list)


def compute_diff(original: Image.Image, anonymized: Image.Image, threshold: int = 30) -> DiffResult:
    """Compute pixel-level diff between two images.

    Args:
        original: Original page image.
        anonymized: Anonymized page image.
        threshold: Minimum per-channel difference to count as changed.

    Returns:
        DiffResult with change statistics and diff visualization.
    """
    # Ensure same size
    if original.size != anonymized.size:
        anonymized = anonymized.resize(original.size, Image.LANCZOS)

    orig_arr = np.array(original.convert("RGB"), dtype=np.int16)
    anon_arr = np.array(anonymized.convert("RGB"), dtype=np.int16)

    # Per-pixel max channel difference
    diff = np.abs(orig_arr - anon_arr).max(axis=2)
    changed_mask = diff > threshold

    changed_count = int(changed_mask.sum())
    total = changed_mask.size

    # Create diff visualization (red overlay on changed pixels)
    diff_vis = original.copy().convert("RGB")
    diff_overlay = np.array(diff_vis)
    diff_overlay[changed_mask] = [255, 0, 0]
    diff_image = Image.fromarray(diff_overlay)

    regions = extract_changed_regions(changed_mask)

    return DiffResult(
        changed_pixel_count=changed_count,
        total_pixel_count=total,
        changed_ratio=changed_count / total if total > 0 else 0.0,
        diff_image=diff_image,
        regions=regions,
    )


def extract_changed_regions(mask: np.ndarray, min_area: int = 100) -> list[BBox]:
    """Find bounding boxes of contiguous changed regions using connected components.

    Args:
        mask: 2D boolean array of changed pixels.
        min_area: Minimum pixel area to report.

    Returns:
        List of BBox for each region.
    """
    if not mask.any():
        return []

    # Simple row/column projection approach for finding regions
    # Label connected components using a basic flood-fill-free approach:
    # project changes onto rows and columns to find rectangular regions
    h, w = mask.shape

    # Find rows and cols that have changes
    row_has_change = mask.any(axis=1)
    col_has_change = mask.any(axis=0)

    # Find contiguous row bands
    row_bands = _find_contiguous_bands(row_has_change)
    col_bands = _find_contiguous_bands(col_has_change)

    regions = []
    for r_start, r_end in row_bands:
        for c_start, c_end in col_bands:
            # Check if this rectangle actually contains changes
            sub_mask = mask[r_start:r_end, c_start:c_end]
            if sub_mask.any():
                bbox = BBox(
                    x=int(c_start),
                    y=int(r_start),
                    width=int(c_end - c_start),
                    height=int(r_end - r_start),
                )
                if bbox.area >= min_area:
                    regions.append(bbox)

    # Merge overlapping regions
    regions = _merge_overlapping(regions)
    return regions


def _find_contiguous_bands(arr: np.ndarray, gap: int = 20) -> list[tuple[int, int]]:
    """Find contiguous True bands in a 1D boolean array, bridging small gaps."""
    bands = []
    in_band = False
    start = 0

    for i in range(len(arr)):
        if arr[i] and not in_band:
            start = i
            in_band = True
        elif not arr[i] and in_band:
            # Look ahead to see if there's a nearby band
            look_end = min(i + gap, len(arr))
            if arr[i:look_end].any():
                continue
            bands.append((start, i))
            in_band = False

    if in_band:
        bands.append((start, len(arr)))

    return bands


def _merge_overlapping(boxes: list[BBox], margin: int = 10) -> list[BBox]:
    """Merge overlapping or nearby bounding boxes."""
    if not boxes:
        return []

    # Sort by y then x
    boxes = sorted(boxes, key=lambda b: (b.y, b.x))
    merged = [boxes[0]]

    for box in boxes[1:]:
        last = merged[-1]
        # Check overlap with margin
        if (box.x <= last.x + last.width + margin and
                box.y <= last.y + last.height + margin and
                box.x + box.width >= last.x - margin and
                box.y + box.height >= last.y - margin):
            # Merge
            new_x = min(last.x, box.x)
            new_y = min(last.y, box.y)
            new_x2 = max(last.x + last.width, box.x + box.width)
            new_y2 = max(last.y + last.height, box.y + box.height)
            merged[-1] = BBox(new_x, new_y, new_x2 - new_x, new_y2 - new_y)
        else:
            merged.append(box)

    return merged


def highlight_regions(image: Image.Image, regions: list[BBox], color: str = "red", width: int = 3) -> Image.Image:
    """Draw rectangles around regions for debugging.

    Args:
        image: Image to draw on (will be copied).
        regions: List of bounding boxes.
        color: Rectangle outline color.
        width: Line width.

    Returns:
        New image with highlighted regions.
    """
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    for bbox in regions:
        draw.rectangle(bbox.to_tuple(), outline=color, width=width)
    return result
