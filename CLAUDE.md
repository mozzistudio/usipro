# USI-PRO Multi-Agent PDF Anonymizer

AI-powered pipeline for anonymizing technical drawings using Claude Vision for detection and a reviewer agent for quality assurance.

## Architecture

```
usipro/
├── main.py                 # CLI entry point
├── config.yaml             # All configuration settings
├── agents/
│   ├── anonymizer.py       # Agent 1: Vision-based sensitive content detection + masking
│   ├── reviewer.py         # Agent 2: Quality verification (visual diff, OCR, regex)
│   └── prompts/
│       ├── detect_sensitive.txt   # Detection prompt for anonymizer
│       └── verify_complete.txt    # Verification prompt for reviewer
├── core/
│   ├── orchestrator.py     # Retry loop: anonymize → review → retry/flag
│   ├── pdf_processor.py    # PDF ↔ image conversion
│   └── image_diff.py       # Visual diff between original and anonymized
├── utils/
│   ├── vision_api.py       # Claude Vision API wrapper
│   ├── ocr.py              # Tesseract OCR wrapper
│   └── patterns.py         # Regex patterns for sensitive data
├── assets/logo.png         # USI-PRO logo
├── input/                  # Drop PDFs here
├── output/                 # Successfully anonymized PDFs
├── flagged/                # PDFs that failed after max retries
├── reports/                # JSON reports per file
└── logs/                   # Processing logs
```

## How It Works

1. **Anonymizer Agent** sends each PDF page as an image to Claude Vision, which identifies all sensitive content (company names, addresses, phones, logos, title blocks). Returns bounding boxes.
2. Detected regions are white-filled on the image, then saved back to PDF.
3. **Reviewer Agent** compares original vs anonymized using visual diff, OCR, and regex to check:
   - All sensitive content was removed
   - No technical content (dimensions, geometry) was damaged
   - Document integrity is preserved
4. **Orchestrator** retries up to 3 times if the reviewer finds issues, passing error details back to the anonymizer for correction. After max retries, the file goes to `flagged/`.

## Running

```bash
# Prerequisites (macOS)
brew install poppler tesseract
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key

# Process all PDFs in input/
python main.py --all

# Process a single file
python main.py --file input/drawing.pdf

# Limit batch size
python main.py --limit 5

# Verbose output
python main.py --all --verbose

# View last run report
python main.py --report
```

## Critical Rules

1. **NEVER** mask technical drawings (geometry, dimensions, section markers, tolerances)
2. **ALWAYS** mask company names, logos, addresses, phone numbers, emails
3. French-specific patterns (SIRET, RCS, TVA intracom) must be detected
4. Output PDF must preserve original page dimensions
5. Each file gets a JSON report in `reports/`
