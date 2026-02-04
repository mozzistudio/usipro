#!/usr/bin/env python3
"""
USI-PRO Static Web Server
Minimal HTTP server for serving static files and handling API endpoints.
No Flask required - uses Python's built-in http.server module.
"""

import json
import os
import subprocess
import uuid
import cgi
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs


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


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_anonymization(input_path: Path, plan_id: str = None) -> dict:
    """Run the anonymization pipeline and return result info."""
    cmd = ['python', str(BASE_DIR / 'anonymize.py'), str(input_path)]

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # API: List files
        if path == '/api/files':
            self.handle_list_files()
            return

        # Serve output files
        if path.startswith('/output/'):
            filename = path[8:]  # Remove '/output/' prefix
            self.serve_output_file(filename)
            return

        # Serve static HTML pages
        if path == '/' or path == '/index.html':
            self.serve_file(BASE_DIR / 'index.html', 'text/html')
            return

        if path == '/viewer.html':
            self.serve_file(BASE_DIR / 'viewer.html', 'text/html')
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
                self.send_json({
                    'success': False,
                    'error': 'Processing failed',
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
