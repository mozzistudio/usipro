#!/usr/bin/env python3
"""
USI-PRO Static Web Server
Minimal HTTP server for serving static files and handling API endpoints.
No Flask required - uses Python's built-in http.server module.
"""

import base64
import io
import json
import os
import subprocess
import uuid
import cgi
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image
import pypdfium2 as pdfium
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


class ReusableHTTPServer(HTTPServer):
    """HTTPServer subclass that allows address reuse to avoid 'Address already in use' errors."""
    allow_reuse_address = True

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "input"
OUTPUT_FOLDER = BASE_DIR / "output"
ALLOWED_EXTENSIONS = {'pdf', 'zip'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# DPI for rendering PDFs
DPI = 150  # Lower DPI for editor (faster loading)


def detect_zone_at_point(img_array: np.ndarray, x: int, y: int, sensitivity: int = 5) -> tuple:
    """
    AI-assisted zone detection: find the connected component at click point.

    Uses flood-fill to detect the boundary of the element clicked.
    Returns (x1, y1, x2, y2) bounding box of detected zone.
    """
    height, width = img_array.shape[:2]

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array

    # Expand search radius based on sensitivity (1-10 maps to 10-100 pixels)
    search_radius = 10 + sensitivity * 9

    # Define search region around click point
    x1_search = max(0, x - search_radius)
    y1_search = max(0, y - search_radius)
    x2_search = min(width, x + search_radius)
    y2_search = min(height, y + search_radius)

    # Check if clicked on dark pixels (content)
    local_region = gray[y1_search:y2_search, x1_search:x2_search]
    threshold = 240  # Pixels darker than this are "content"

    # Create binary mask of content
    content_mask = gray < threshold

    # If click point has no content nearby, expand search
    if not content_mask[max(0,y-5):min(height,y+5), max(0,x-5):min(width,x+5)].any():
        # No content at click point - look for nearest content
        return None

    # Flood fill from click point to find connected component
    visited = np.zeros_like(content_mask, dtype=bool)

    # BFS flood fill
    queue = [(y, x)]
    min_x, max_x = x, x
    min_y, max_y = y, y

    # Distance threshold for grouping (based on sensitivity)
    gap_threshold = 5 + sensitivity * 3

    while queue:
        cy, cx = queue.pop(0)

        if cy < 0 or cy >= height or cx < 0 or cx >= width:
            continue
        if visited[cy, cx]:
            continue

        visited[cy, cx] = True

        # Update bounding box
        min_x = min(min_x, cx)
        max_x = max(max_x, cx)
        min_y = min(min_y, cy)
        max_y = max(max_y, cy)

        # Check if this is content or close to content
        is_content = content_mask[cy, cx]

        if is_content:
            # Add neighbors with larger step for faster processing
            for dy in range(-gap_threshold, gap_threshold + 1, 2):
                for dx in range(-gap_threshold, gap_threshold + 1, 2):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if not visited[ny, nx] and content_mask[ny, nx]:
                            queue.append((ny, nx))

    # Add padding around detected zone
    padding = 5
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(width, max_x + padding)
    max_y = min(height, max_y + padding)

    # Minimum zone size
    if (max_x - min_x) < 10 or (max_y - min_y) < 10:
        # Too small - expand to minimum
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
        half_size = 20 + sensitivity * 5
        min_x = max(0, center_x - half_size)
        max_x = min(width, center_x + half_size)
        min_y = max(0, center_y - half_size)
        max_y = min(height, center_y + half_size)

    return (min_x, min_y, max_x, max_y)


def delete_zone(img: Image.Image, zone: tuple) -> Image.Image:
    """Delete a zone by filling it with white."""
    result = img.copy()
    x1, y1, x2, y2 = zone

    # Create white fill
    from PIL import ImageDraw
    draw = ImageDraw.Draw(result)
    draw.rectangle([x1, y1, x2, y2], fill='white')

    return result


def render_pdf_to_image(pdf_path: Path, page_num: int = 0, dpi: int = DPI) -> Image.Image:
    """Render a PDF page to PIL Image."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_num]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale, rotation=0)
    pil_image = bitmap.to_pil()
    pdf.close()
    return pil_image


def image_to_pdf(img: Image.Image, output_path: Path, original_pdf_path: Path = None):
    """Convert an image back to PDF."""
    # Get original page size if available
    if original_pdf_path and original_pdf_path.exists():
        pdf = pdfium.PdfDocument(str(original_pdf_path))
        page = pdf[0]
        page_width = page.get_width()
        page_height = page.get_height()
        pdf.close()
    else:
        # Assume A4 landscape or portrait based on image aspect ratio
        if img.width > img.height:
            page_width, page_height = 842, 595  # A4 landscape
        else:
            page_width, page_height = 595, 842  # A4 portrait

    # Save image to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name, 'PNG')
        temp_img_path = tmp.name

    try:
        c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
        c.drawImage(temp_img_path, 0, 0, width=page_width, height=page_height)
        c.save()
    finally:
        os.unlink(temp_img_path)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_anonymization(input_path: Path, plan_id: str = None) -> dict:
    """Run the anonymization pipeline and return result info."""
    # Use sys.executable to ensure we use the same Python interpreter as the server
    import sys
    cmd = [sys.executable, str(BASE_DIR / 'anonymize.py'), str(input_path)]

    if plan_id and input_path.suffix.lower() == '.pdf':
        cmd.extend(['--plan-id', plan_id])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=300  # 5 minute timeout
        )

        success = result.returncode == 0

        # Find output file(s)
        output_files = []
        if success:
            if input_path.suffix.lower() == '.zip':
                # Look for output ZIP
                output_zip = OUTPUT_FOLDER / f"{input_path.stem}.zip"
                if output_zip.exists():
                    output_files.append(output_zip.name)
            else:
                # Look for output PDF
                pdfs = sorted(OUTPUT_FOLDER.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
                if pdfs:
                    output_files.append(pdfs[0].name)

        return {
            'success': success,
            'output_files': output_files,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output_files': [],
            'stdout': '',
            'stderr': 'Processing timeout exceeded (5 minutes)'
        }
    except Exception as e:
        return {
            'success': False,
            'output_files': [],
            'stdout': '',
            'stderr': str(e)
        }


class USIProHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler for USI-PRO."""

    # Use HTTP/1.1 for better proxy compatibility (Codespaces, etc.)
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # API: List files
        if path == '/api/files':
            self.handle_list_files()
            return

        # API: Get PDF as image for editor
        if path == '/api/pdf-image':
            self.handle_pdf_to_image(query)
            return

        # Serve output files
        if path.startswith('/output/'):
            filename = path[8:]  # Remove '/output/' prefix
            # Remove query string from filename
            if '?' in filename:
                filename = filename.split('?')[0]
            self.serve_output_file(filename)
            return

        # Serve static HTML pages
        if path == '/' or path == '/index.html':
            self.serve_file(BASE_DIR / 'index.html', 'text/html')
            return

        if path == '/viewer.html':
            self.serve_file(BASE_DIR / 'viewer.html', 'text/html')
            return

        if path == '/editor.html':
            self.serve_file(BASE_DIR / 'editor.html', 'text/html')
            return

        # Serve static assets
        if path.startswith('/static/'):
            file_path = BASE_DIR / path[1:]  # Remove leading '/'
            if file_path.exists() and file_path.is_file():
                content_type, _ = mimetypes.guess_type(str(file_path))
                self.serve_file(file_path, content_type or 'application/octet-stream')
                return

        # Default 404
        self.send_error(404, 'File not found')

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/upload':
            self.handle_upload()
            return

        if path == '/api/delete-zone':
            self.handle_delete_zone()
            return

        if path == '/api/save-edited-pdf':
            self.handle_save_edited_pdf()
            return

        self.send_error(404, 'Endpoint not found')

    def handle_list_files(self):
        """API endpoint to list output files."""
        output_files = []
        if OUTPUT_FOLDER.exists():
            for f in OUTPUT_FOLDER.iterdir():
                if f.suffix.lower() in ['.pdf', '.zip']:
                    output_files.append({
                        'name': f.name,
                        'size': f.stat().st_size,
                        'modified': f.stat().st_mtime,
                        'type': f.suffix.lower()[1:]
                    })

        output_files.sort(key=lambda x: x['modified'], reverse=True)
        self.send_json(output_files)

    def handle_pdf_to_image(self, query):
        """Convert PDF to image for editor."""
        filename = query.get('file', [''])[0]
        if not filename:
            self.send_json({'success': False, 'error': 'No filename provided'}, 400)
            return

        pdf_path = OUTPUT_FOLDER / filename
        if not pdf_path.exists():
            self.send_json({'success': False, 'error': 'File not found'}, 404)
            return

        try:
            # Render PDF to image
            img = render_pdf_to_image(pdf_path)

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            self.send_json({
                'success': True,
                'image': img_base64,
                'width': img.width,
                'height': img.height
            })
        except Exception as e:
            self.send_json({'success': False, 'error': str(e)}, 500)

    def handle_delete_zone(self):
        """AI-assisted zone deletion."""
        # Read JSON body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({'success': False, 'error': 'Invalid JSON'}, 400)
            return

        # Get parameters
        image_b64 = data.get('image')
        x = data.get('x', 0)
        y = data.get('y', 0)
        sensitivity = data.get('sensitivity', 5)

        if not image_b64:
            self.send_json({'success': False, 'error': 'No image provided'}, 400)
            return

        try:
            # Decode image
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data))

            # Convert to numpy for zone detection
            img_array = np.array(img)

            # Detect zone at click point
            zone = detect_zone_at_point(img_array, x, y, sensitivity)

            if zone is None:
                self.send_json({'success': False, 'error': 'No content detected at click point'}, 400)
                return

            # Delete the zone
            result_img = delete_zone(img, zone)

            # Convert result to base64
            buffer = io.BytesIO()
            result_img.save(buffer, format='PNG')
            result_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            self.send_json({
                'success': True,
                'image': result_b64,
                'zone': {'x1': zone[0], 'y1': zone[1], 'x2': zone[2], 'y2': zone[3]}
            })
        except Exception as e:
            self.send_json({'success': False, 'error': str(e)}, 500)

    def handle_save_edited_pdf(self):
        """Save edited image as PDF."""
        # Read JSON body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({'success': False, 'error': 'Invalid JSON'}, 400)
            return

        filename = data.get('filename')
        image_b64 = data.get('image')

        if not filename or not image_b64:
            self.send_json({'success': False, 'error': 'Missing filename or image'}, 400)
            return

        try:
            # Decode image
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data))

            # Output path
            output_path = OUTPUT_FOLDER / filename
            original_path = output_path if output_path.exists() else None

            # Convert to PDF
            image_to_pdf(img, output_path, original_path)

            self.send_json({
                'success': True,
                'filename': filename,
                'message': 'PDF saved successfully'
            })
        except Exception as e:
            self.send_json({'success': False, 'error': str(e)}, 500)

    def handle_upload(self):
        """Handle file upload and run anonymization."""
        content_type = self.headers.get('Content-Type', '')

        if 'multipart/form-data' not in content_type:
            self.send_json({'error': 'Invalid content type'}, 400)
            return

        # Parse multipart form data
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                }
            )
        except Exception as e:
            self.send_json({'error': f'Failed to parse form data: {e}'}, 400)
            return

        # Get uploaded file
        if 'file' not in form:
            self.send_json({'error': 'No file provided'}, 400)
            return

        file_item = form['file']
        if not file_item.filename:
            self.send_json({'error': 'No file selected'}, 400)
            return

        filename = os.path.basename(file_item.filename)
        if not allowed_file(filename):
            self.send_json({'error': 'Invalid file type. Only PDF and ZIP files are allowed.'}, 400)
            return

        # Get plan_id
        plan_id = None
        if 'plan_id' in form:
            plan_id = form['plan_id'].value.strip() or None

        # Save uploaded file
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        input_path = UPLOAD_FOLDER / unique_filename

        try:
            with open(input_path, 'wb') as f:
                f.write(file_item.file.read())

            # Run anonymization
            result = run_anonymization(input_path, plan_id)

            if result['success'] and result['output_files']:
                output_file = result['output_files'][0]
                self.send_json({
                    'success': True,
                    'output_file': output_file,
                    'message': 'File processed successfully',
                    'log': result['stdout']
                })
            else:
                # Include stderr details in error message for better debugging
                error_msg = 'Processing failed'
                if result['stderr']:
                    # Extract the most relevant error line
                    stderr_lines = result['stderr'].strip().split('\n')
                    error_detail = stderr_lines[-1] if stderr_lines else ''
                    if error_detail:
                        error_msg = f'Processing failed: {error_detail}'
                self.send_json({
                    'success': False,
                    'error': error_msg,
                    'log': result['stdout'] + '\n' + result['stderr']
                }, 500)

        finally:
            # Clean up uploaded file
            if input_path.exists():
                input_path.unlink()

    def serve_output_file(self, filename):
        """Serve a file from the output directory."""
        file_path = OUTPUT_FOLDER / filename
        if file_path.exists() and file_path.is_file():
            content_type, _ = mimetypes.guess_type(str(file_path))
            self.serve_file(file_path, content_type or 'application/octet-stream')
        else:
            self.send_error(404, 'File not found')

    def serve_file(self, file_path: Path, content_type: str):
        """Serve a file with the given content type."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def send_json(self, data, status=200):
        """Send a JSON response."""
        content = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        """Log HTTP requests."""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def run_server(host='0.0.0.0', port=8080):
    """Run the HTTP server."""
    server_address = (host, port)
    httpd = ReusableHTTPServer(server_address, USIProHandler)
    print(f"USI-PRO Server running at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        httpd.shutdown()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='USI-PRO Static Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on (default: 8080)')
    args = parser.parse_args()

    run_server(args.host, args.port)
