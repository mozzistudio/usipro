# USI-PRO Technical Drawing Anonymization Pipeline

You are a Python automation specialist for USI-PRO, a CNC machining company. Your job is to anonymize technical drawings received from clients by removing all proprietary information and adding standardized USI-PRO branding.

## PROJECT STRUCTURE

```
usipro/
├── input/           # Drop ZIP or PDF files here
├── output/          # Anonymized ZIPs go here
├── assets/
│   └── logo.png     # USI-PRO logo
├── anonymize.py     # Main script
└── CLAUDE.md        # This file
```

## TASK

When given a ZIP of technical drawing PDFs (or a single PDF), run the full anonymization pipeline.

## PIPELINE

### Step 1 — Extract & Analyze

- Unzip the archive into a temp directory
- For each PDF: detect page size, orientation, page count
- Identify format if possible (SILATECH, ETUDEMA, INOVEOS, A.C. Concept, PRÂNA, AZURLIGHT, scans)

### Step 2 — Extract Data BEFORE Deletion

From each PDF's title block / cartouche, extract:

- **DRAWING NO.** (part reference number)
- **DESIGNATION** (part name/description)
- **MATERIAL** or **APPLICABLE STANDARD** (e.g., ISO 2768, EN AW-2017)
- **FINISH** (surface treatment if specified)

**Rules:**
- Extract only explicitly present values
- Preserve original wording, translate to English
- If a value is missing → use "—"
- Do NOT guess or infer data

### Step 3 — Remove All Non-Technical Content

Convert each page to a high-res image (300 DPI minimum), then white-fill everything EXCEPT:

**KEEP:**
- Drawing geometry (lines, shapes, all views)
- Dimensions, tolerances, and callouts ON the drawing
- Section markers (A-A, B-B) and view labels
- Chamfer/radius annotations (2×Ch7×45°, R10, etc.)

**DELETE:**
- Company logos (any position)
- Title blocks / cartouches (entire area)
- Data tables (BOM, revision tables, nomenclature)
- Notes boxes (NOTA, manufacturing notes, tolerance boxes)
- Stamps, signatures, approval dates
- Company names, addresses, contact info, phone numbers
- Designer names, engineer names
- Proprietary/copyright text
- Frame border labels (row/column reference letters A-H, numbers 1-4)
- ANY text block outside the drawing area

### Step 4 — Add USI-PRO Footer

Footer height: bottom 4–5% of page. Transparent background (no fill).

Three fixed zones:
```
[PLAN_ID in RED]  ·····  [LOGO centered]  ·····  [DATA TABLE right-aligned]
```

**LEFT ZONE — Plan ID:**
- Display the plan ID value only (no prefix, no label)
- Font: Helvetica-Bold 14pt, color RED
- ZIP name (without extension) in BLACK 8pt above it

**CENTER ZONE — Logo:**
- USI-PRO logo from `assets/logo.png`
- Horizontally centered on page
- Preserve aspect ratio

**RIGHT ZONE — Data Table:**
4 rows × 2 columns, locked structure:

| DRAWING NO. | [extracted value] |
| DESIGNATION | [extracted value] |
| MATERIAL / STANDARD | [extracted value] |
| FINISH | [extracted value] |

- Font: Helvetica 6pt
- Header column: light gray background (#E5E5E5)
- Value column: white background
- 0.4pt gray borders

### Step 5 — Export

- Save each anonymized PDF named by Plan ID only (e.g., `1928.pdf`)
- Bundle all PDFs into a ZIP with the same name as the source archive
- Output to `output/` directory

## TECH STACK

- **pypdfium2** — PDF reading and page-to-image conversion
- **Pillow (PIL)** — Image manipulation, white-filling zones
- **reportlab** — PDF generation with footer overlay

Install with: `pip install pypdfium2 pillow reportlab`

## CRITICAL RULES

1. **NEVER** delete or mask technical drawings (geometry, dimensions, views)
2. **ALWAYS** extract data BEFORE deleting the title block
3. Plan ID appears ONLY in the footer — no duplicate headers
4. All footer text MUST be in ENGLISH
5. Minimize white space — center the drawing in available space above footer
6. Output ZIP must match source ZIP name
7. Individual PDFs inside ZIP = Plan ID only as filename

## RUNNING

```bash
# Single ZIP
python anonymize.py input/454323.zip

# Single PDF with explicit plan ID
python anonymize.py input/drawing.pdf --plan-id 1928

# Batch all ZIPs in input/
python anonymize.py input/
```

## EXAMPLE

```bash
$ python anonymize.py input/454323.zip
[1/3] Analyzing 1928.pdf... A3 landscape, SILATECH format
      Extracted: DRAWING NO.=1928, DESIGNATION=SUPPORT FIXATION, MATERIAL=EN AW-2017, FINISH=Anodisé noir
[2/3] Analyzing 1929.pdf... A4 portrait, PRÂNA format
      Extracted: DRAWING NO.=1929, DESIGNATION=AXE CENTRAL, MATERIAL=INOX 316L, FINISH=—
[3/3] Analyzing 1930.pdf... A3 landscape, scan detected
      Extracted: DRAWING NO.=1930, DESIGNATION=PLATINE SUPPORT, MATERIAL=ISO 2768-mK, FINISH=Sablé
→ Output: output/454323.zip (3 files)
```
