"""Orchestrator — Coordinates anonymizer and reviewer agents with retry loop."""

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from agents.anonymizer import Anonymizer, AnonymizeResult
from agents.reviewer import Reviewer, ReviewResult
from utils.vision_api import VisionAPI

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single PDF."""
    filename: str
    status: str  # "success", "flagged", "error"
    rounds: int = 0
    final_verdict: str = ""
    output_path: str = ""
    anonymize_reports: list[dict] = field(default_factory=list)
    review_reports: list[dict] = field(default_factory=list)
    error: str | None = None
    timestamp: str = ""


class Orchestrator:
    """Coordinates the anonymize → review → retry loop."""

    def __init__(self, config: dict):
        self.config = config
        self.max_rounds = config["orchestrator"]["max_correction_rounds"]
        self.checkpoint_interval = config["orchestrator"]["checkpoint_interval"]

        self.input_dir = Path(config["directories"]["input"])
        self.output_dir = Path(config["directories"]["output"])
        self.flagged_dir = Path(config["directories"]["flagged"])
        self.reports_dir = Path(config["directories"]["reports"])
        self.logs_dir = Path(config["directories"]["logs"])

        # Ensure directories exist
        for d in [self.output_dir, self.flagged_dir, self.reports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize agents
        self.vision_api = VisionAPI(config)
        self.anonymizer = Anonymizer(config, self.vision_api)
        self.reviewer = Reviewer(config, self.vision_api)

        # Error patterns log
        self.error_patterns_file = Path(config["logging"]["error_patterns_file"])

    def process_single(self, pdf_path: str | Path) -> ProcessingResult:
        """Process a single PDF through the anonymize → review → retry loop.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ProcessingResult with final status and reports.
        """
        pdf_path = Path(pdf_path)
        logger.info("=" * 60)
        logger.info("Processing: %s", pdf_path.name)
        logger.info("=" * 60)

        result = ProcessingResult(
            filename=pdf_path.name,
            timestamp=datetime.now().isoformat(),
            status="error",
        )

        previous_errors = None

        for round_num in range(1, self.max_rounds + 1):
            result.rounds = round_num
            logger.info("Round %d/%d", round_num, self.max_rounds)

            # Step 1: Anonymize
            anon_result = self.anonymizer.process(pdf_path, previous_errors=previous_errors)
            result.anonymize_reports.append({
                "round": round_num,
                "pages": anon_result.pages,
                "success": anon_result.success,
                "error": anon_result.error,
            })

            if not anon_result.success:
                result.status = "error"
                result.error = f"Anonymization failed in round {round_num}: {anon_result.error}"
                logger.error(result.error)
                break

            # Step 2: Review
            review_result = self.reviewer.verify(pdf_path, anon_result.output_path)
            result.review_reports.append({
                "round": round_num,
                "verdict": review_result.verdict,
                "issues": review_result.issues,
                "summary": review_result.summary,
            })

            logger.info("Review verdict: %s (%s)", review_result.verdict, review_result.summary)

            if review_result.verdict == "PASS":
                result.status = "success"
                result.final_verdict = "PASS"
                result.output_path = str(anon_result.output_path)
                logger.info("PASSED — output: %s", anon_result.output_path)
                break

            # Prepare error corrections for next round
            critical_issues = [i for i in review_result.issues if i.get("severity") == "critical"]
            if round_num < self.max_rounds:
                previous_errors = critical_issues
                logger.info("FAILED with %d critical issues, retrying...", len(critical_issues))
                # Clean up failed output before retry
                if anon_result.output_path.exists():
                    anon_result.output_path.unlink()
            else:
                # Max retries reached — flag the file
                result.status = "flagged"
                result.final_verdict = "FAIL"
                result.error = f"Failed after {self.max_rounds} rounds: {review_result.summary}"
                flagged_path = self.flagged_dir / pdf_path.name
                if anon_result.output_path.exists():
                    shutil.move(str(anon_result.output_path), str(flagged_path))
                result.output_path = str(flagged_path)
                logger.warning("FLAGGED after %d rounds — moved to %s", self.max_rounds, flagged_path)
                self._log_error_pattern(pdf_path.name, critical_issues)

        # Save report
        self._save_report(result)
        return result

    def process_batch(self, limit: int | None = None) -> list[ProcessingResult]:
        """Process all PDFs in the input directory.

        Args:
            limit: Optional max number of files to process.

        Returns:
            List of ProcessingResult for each file.
        """
        pdf_files = sorted(self.input_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found in %s", self.input_dir)
            return []

        if limit is not None:
            pdf_files = pdf_files[:limit]

        logger.info("Processing %d PDF files", len(pdf_files))
        results = []

        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info("[%d/%d] %s", i, len(pdf_files), pdf_path.name)
            result = self.process_single(pdf_path)
            results.append(result)

            # Checkpoint
            if i % self.checkpoint_interval == 0:
                self._save_checkpoint(results, i, len(pdf_files))

        self._log_batch_summary(results)
        return results

    def _save_report(self, result: ProcessingResult):
        """Save a JSON report for a single file."""
        report_path = self.reports_dir / f"{Path(result.filename).stem}_report.json"
        with open(report_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        logger.info("Report saved to %s", report_path)

    def _save_checkpoint(self, results: list[ProcessingResult], processed: int, total: int):
        """Save a checkpoint of batch progress."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "processed": processed,
            "total": total,
            "success": sum(1 for r in results if r.status == "success"),
            "flagged": sum(1 for r in results if r.status == "flagged"),
            "errors": sum(1 for r in results if r.status == "error"),
        }
        checkpoint_path = self.reports_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info("Checkpoint saved: %d/%d processed", processed, total)

    def _log_error_pattern(self, filename: str, issues: list[dict]):
        """Log recurring error patterns for analysis."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "issues": issues,
        }
        with open(self.error_patterns_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _log_batch_summary(self, results: list[ProcessingResult]):
        """Log a summary of batch processing."""
        success = sum(1 for r in results if r.status == "success")
        flagged = sum(1 for r in results if r.status == "flagged")
        errors = sum(1 for r in results if r.status == "error")

        logger.info("=" * 60)
        logger.info("BATCH SUMMARY")
        logger.info("  Total: %d", len(results))
        logger.info("  Success: %d", success)
        logger.info("  Flagged: %d", flagged)
        logger.info("  Errors: %d", errors)
        logger.info("=" * 60)
