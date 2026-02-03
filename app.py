#!/usr/bin/env python3
"""
USI-PRO Web Application
Flask app for uploading technical drawings and viewing anonymized PDFs.
"""

import os
import subprocess
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "input"
OUTPUT_FOLDER = BASE_DIR / "output"
ALLOWED_EXTENSIONS = {'pdf', 'zip'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['OUTPUT_FOLDER'] = str(OUTPUT_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

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
                output_pattern = OUTPUT_FOLDER.glob("*.pdf")
                # Get most recently modified PDF
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


@app.route('/')
def index():
    """Main page with upload form."""
    # List existing output files
    output_files = []
    if OUTPUT_FOLDER.exists():
        output_files = sorted(
            [f.name for f in OUTPUT_FOLDER.iterdir() if f.suffix.lower() in ['.pdf', '.zip']],
            key=lambda x: (OUTPUT_FOLDER / x).stat().st_mtime,
            reverse=True
        )

    return render_template('index.html', output_files=output_files)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and run anonymization."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    plan_id = request.form.get('plan_id', '').strip() or None

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF and ZIP files are allowed.'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    # Add unique prefix to avoid conflicts
    unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
    input_path = UPLOAD_FOLDER / unique_filename
    file.save(str(input_path))

    try:
        # Run anonymization
        result = run_anonymization(input_path, plan_id)

        if result['success'] and result['output_files']:
            output_file = result['output_files'][0]
            return jsonify({
                'success': True,
                'output_file': output_file,
                'message': 'File processed successfully',
                'log': result['stdout']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Processing failed',
                'log': result['stdout'] + '\n' + result['stderr']
            }), 500

    finally:
        # Clean up uploaded file
        if input_path.exists():
            input_path.unlink()


@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve output files (PDFs and ZIPs)."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/view/<path:filename>')
def view_pdf(filename):
    """View a PDF file in the browser."""
    return render_template('viewer.html', filename=filename)


@app.route('/api/files')
def list_files():
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
    return jsonify(output_files)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
