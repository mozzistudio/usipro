"""CLI entry point for the USI-PRO Multi-Agent PDF Anonymizer."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from core.orchestrator import Orchestrator


def setup_logging(config: dict, verbose: bool = False):
    """Configure logging from config settings."""
    log_level = logging.DEBUG if verbose else getattr(logging, config["logging"]["level"])
    log_file = config["logging"]["log_file"]

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_report(reports_dir: str):
    """Print the most recent processing reports."""
    reports_path = Path(reports_dir)
    report_files = sorted(reports_path.glob("*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not report_files:
        print("No reports found.")
        return

    for report_file in report_files[:10]:
        with open(report_file) as f:
            report = json.load(f)

        status_icon = {"success": "+", "flagged": "!", "error": "x"}.get(report["status"], "?")
        print(f"  [{status_icon}] {report['filename']} — {report['status']} "
              f"({report['rounds']} round(s), {report.get('final_verdict', 'N/A')})")

        if report.get("error"):
            print(f"      Error: {report['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="USI-PRO Multi-Agent PDF Anonymizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python main.py --all                  Process all PDFs in input/\n"
               "  python main.py --limit 5               Process first 5 PDFs\n"
               "  python main.py --file input/test.pdf   Process a single file\n"
               "  python main.py --report                Show recent reports\n",
    )
    parser.add_argument("--all", action="store_true", help="Process all PDFs in input/")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--file", type=str, help="Process a single PDF file")
    parser.add_argument("--report", action="store_true", help="Show recent processing reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose/debug logging")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    if not any([args.all, args.limit, args.file, args.report]):
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    setup_logging(config, verbose=args.verbose)

    if args.report:
        print_report(config["directories"]["reports"])
        return

    orchestrator = Orchestrator(config)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        result = orchestrator.process_single(file_path)
        _print_result(result)

    elif args.all or args.limit:
        results = orchestrator.process_batch(limit=args.limit)
        _print_batch_summary(results)


def _print_result(result):
    """Print a single processing result."""
    status_icon = {"success": "+", "flagged": "!", "error": "x"}.get(result.status, "?")
    print(f"\n[{status_icon}] {result.filename}")
    print(f"    Status: {result.status}")
    print(f"    Rounds: {result.rounds}")
    print(f"    Verdict: {result.final_verdict or 'N/A'}")
    if result.output_path:
        print(f"    Output: {result.output_path}")
    if result.error:
        print(f"    Error: {result.error}")


def _print_batch_summary(results):
    """Print summary of batch processing."""
    success = sum(1 for r in results if r.status == "success")
    flagged = sum(1 for r in results if r.status == "flagged")
    errors = sum(1 for r in results if r.status == "error")

    print(f"\nBatch complete: {len(results)} files processed")
    print(f"  Success: {success}")
    print(f"  Flagged: {flagged}")
    print(f"  Errors:  {errors}")

    if flagged > 0:
        print("\nFlagged files (manual review needed):")
        for r in results:
            if r.status == "flagged":
                print(f"  - {r.filename}: {r.error}")


if __name__ == "__main__":
    main()
