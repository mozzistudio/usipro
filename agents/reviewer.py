"""Agent 2: Reviewer — Quality verification of anonymized PDFs."""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from core.image_diff import compute_diff, highlight_regions
from core.pdf_processor import pdf_to_images
from utils.ocr import extract_text, extract_text_regions
from utils.patterns import find_sensitive_matches
from utils.vision_api import VisionAPI

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "verify_complete.txt"


@dataclass
class ReviewResult:
    """Result of the review verification."""
    verdict: str  # "PASS" or "FAIL"
    issues: list[dict] = field(default_factory=list)
    summary: str = ""
    page_details: list[dict] = field(default_factory=list)


class Reviewer:
    """Verifies anonymization quality using visual diff, Vision API, OCR, and regex."""

    def __init__(self, config: dict, vision_api: VisionAPI):
        self.config = config
        self.vision = vision_api
        self.dpi = config["pdf"]["dpi"]
        self.verify_prompt = PROMPT_PATH.read_text()

    def verify(self, original_path: str | Path, anonymized_path: str | Path) -> ReviewResult:
        """Run full verification pipeline on an anonymized PDF.

        Args:
            original_path: Path to the original PDF.
            anonymized_path: Path to the anonymized PDF.

        Returns:
            ReviewResult with verdict and detailed issues.
        """
        original_path = Path(original_path)
        anonymized_path = Path(anonymized_path)
        logger.info("Reviewing %s", anonymized_path.name)

        try:
            original_images = pdf_to_images(original_path, dpi=self.dpi)
            anonymized_images = pdf_to_images(anonymized_path, dpi=self.dpi)
        except Exception as e:
            return ReviewResult(
                verdict="FAIL",
                issues=[{"type": "integrity", "message": f"Failed to load PDFs: {e}", "severity": "critical"}],
                summary=f"Could not load PDFs for comparison: {e}",
            )

        if len(original_images) != len(anonymized_images):
            return ReviewResult(
                verdict="FAIL",
                issues=[{"type": "integrity", "message": "Page count mismatch", "severity": "critical"}],
                summary=f"Original has {len(original_images)} pages, anonymized has {len(anonymized_images)}",
            )

        all_issues = []
        page_details = []

        for i, (orig, anon) in enumerate(zip(original_images, anonymized_images)):
            page_num = i + 1
            logger.info("Reviewing page %d/%d", page_num, len(original_images))
            page_issues = []

            # Step 1: Visual diff
            diff_issues = self._check_visual_diff(orig, anon, page_num)
            page_issues.extend(diff_issues)

            # Step 2: Vision scan for remaining sensitive content
            vision_issues = self._check_vision(anon, page_num)
            page_issues.extend(vision_issues)

            # Step 3: OCR + regex scan for residual patterns
            ocr_issues = self._check_ocr_patterns(anon, page_num)
            page_issues.extend(ocr_issues)

            # Step 4: Check masked regions didn't cover technical content
            tech_issues = self._check_technical_damage(orig, anon, page_num)
            page_issues.extend(tech_issues)

            # Step 5: Dimension check
            if orig.size != anon.size:
                page_issues.append({
                    "type": "integrity",
                    "message": f"Page {page_num}: dimension mismatch "
                               f"(orig {orig.size} vs anon {anon.size})",
                    "severity": "minor",
                    "page": page_num,
                })

            all_issues.extend(page_issues)
            page_details.append({
                "page": page_num,
                "issues_count": len(page_issues),
                "issues": page_issues,
            })

        has_critical = any(issue.get("severity") == "critical" for issue in all_issues)
        verdict = "FAIL" if has_critical else "PASS"

        return ReviewResult(
            verdict=verdict,
            issues=all_issues,
            summary=self._build_summary(all_issues, verdict),
            page_details=page_details,
        )

    def _check_visual_diff(self, original: Image.Image, anonymized: Image.Image, page_num: int) -> list[dict]:
        """Step 1: Compute visual diff and check for anomalies."""
        issues = []
        diff = compute_diff(original, anonymized)

        if diff.changed_ratio > 0.7:
            issues.append({
                "type": "integrity",
                "message": f"Page {page_num}: excessive changes ({diff.changed_ratio:.1%} of pixels changed). "
                           "Possible over-masking.",
                "severity": "critical",
                "page": page_num,
            })
        elif diff.changed_ratio < 0.001 and diff.changed_pixel_count > 0:
            logger.debug("Page %d: minimal changes detected (%.3f%%)", page_num, diff.changed_ratio * 100)

        return issues

    def _check_vision(self, anonymized: Image.Image, page_num: int) -> list[dict]:
        """Step 2: Use Vision API to check for remaining sensitive content."""
        buf = io.BytesIO()
        anonymized.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        try:
            result = self.vision.analyze_image(image_bytes, self.verify_prompt)
        except Exception as e:
            logger.warning("Vision review failed for page %d: %s", page_num, e)
            return [{
                "type": "integrity",
                "message": f"Page {page_num}: Vision review failed: {e}",
                "severity": "minor",
                "page": page_num,
            }]

        issues = []
        for issue in result.get("issues", []):
            issue["page"] = page_num
            issues.append(issue)

        return issues

    def _check_ocr_patterns(self, anonymized: Image.Image, page_num: int) -> list[dict]:
        """Step 3: OCR the anonymized image and check for residual sensitive patterns."""
        issues = []
        text = extract_text(anonymized)

        if not text:
            return issues

        matches = find_sensitive_matches(text)
        for match in matches:
            issues.append({
                "type": "missed_sensitive",
                "message": f"Page {page_num}: OCR found residual {match['type']}: '{match['match']}'",
                "severity": "critical",
                "page": page_num,
            })

        return issues

    def _check_technical_damage(self, original: Image.Image, anonymized: Image.Image, page_num: int) -> list[dict]:
        """Step 4: Check if masked regions covered important technical content."""
        issues = []
        diff = compute_diff(original, anonymized)

        if not diff.regions:
            return issues

        for region in diff.regions:
            # Extract text from the same region in the original
            orig_text = extract_text(original, region=(region.x, region.y, region.width, region.height))
            if not orig_text:
                continue

            # Check if masked region contained dimension-like content
            # (numbers with units, tolerances, etc.)
            import re
            dimension_patterns = [
                r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|\")\b",  # dimensions with units
                r"[±]\s*\d+(?:\.\d+)?",  # tolerances
                r"\b[MR]\d+(?:[xX×]\d+(?:\.\d+)?)?\b",  # thread/radius callouts
                r"\b\d+[xX×]Ch\d+",  # chamfer callouts
                r"\bISO\s+\d+",  # ISO standards in annotations
            ]

            for pattern in dimension_patterns:
                if re.search(pattern, orig_text, re.IGNORECASE):
                    issues.append({
                        "type": "damaged_technical",
                        "message": f"Page {page_num}: masked region at ({region.x},{region.y}) "
                                   f"may contain technical content: '{orig_text[:100]}'",
                        "severity": "critical",
                        "page": page_num,
                        "region": {"x": round(region.x / original.width * 100),
                                   "y": round(region.y / original.height * 100)},
                    })
                    break

        return issues

    def _build_summary(self, issues: list[dict], verdict: str) -> str:
        """Build a human-readable summary of the review."""
        if not issues:
            return "All checks passed. No issues found."

        critical = [i for i in issues if i.get("severity") == "critical"]
        minor = [i for i in issues if i.get("severity") == "minor"]

        parts = [f"Verdict: {verdict}."]
        if critical:
            parts.append(f"{len(critical)} critical issue(s).")
        if minor:
            parts.append(f"{len(minor)} minor issue(s).")

        return " ".join(parts)
