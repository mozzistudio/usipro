// USI-PRO Drawing Anonymizer - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    const status = document.getElementById('status');
    const selectedFile = document.getElementById('selected-file');
    const selectedFileName = document.getElementById('selected-file-name');
    const viewerBody = document.getElementById('viewer-body');
    const viewerTitle = document.getElementById('viewer-title');
    const viewerActions = document.getElementById('viewer-actions');
    const downloadLink = document.getElementById('download-link');

    // Only initialize if elements exist (main page)
    if (!dropZone || !fileInput) return;

    // Drag and drop handlers
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });

    fileInput.addEventListener('change', handleFileSelect);

    function handleFileSelect() {
        if (fileInput.files.length) {
            const file = fileInput.files[0];
            selectedFileName.textContent = file.name;
            selectedFile.style.display = 'block';
            submitBtn.disabled = false;
        }
    }

    // Form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('plan_id', document.getElementById('plan-id').value);

        // Show processing status
        showStatus('processing', 'Processing drawing... This may take a moment.');
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                showStatus('success', 'Processing complete!');
                // View the result
                viewFile(result.output_file);
                // Refresh file list
                refreshFileList();
                // Reset form
                uploadForm.reset();
                selectedFile.style.display = 'none';
            } else {
                showStatus('error', 'Error: ' + (result.error || 'Processing failed'));
            }
        } catch (err) {
            showStatus('error', 'Error: ' + err.message);
        }

        submitBtn.disabled = false;
    });

    function showStatus(type, message) {
        status.className = 'status show ' + type;
        status.querySelector('.status-text').textContent = message;
        if (type !== 'processing') {
            status.querySelector('.spinner').style.display = 'none';
        } else {
            status.querySelector('.spinner').style.display = 'inline-block';
        }
    }

    // Load file list on page load
    refreshFileList();
});

function viewFile(filename) {
    const viewerBody = document.getElementById('viewer-body');
    const viewerTitle = document.getElementById('viewer-title');
    const viewerActions = document.getElementById('viewer-actions');
    const downloadLink = document.getElementById('download-link');

    // Update active state
    document.querySelectorAll('.file-item').forEach(el => {
        el.classList.toggle('active', el.dataset.file === filename);
    });

    viewerTitle.textContent = filename;
    viewerActions.style.display = 'flex';
    downloadLink.href = '/output/' + filename;

    // Add or update Edit button for PDFs
    let editLink = document.getElementById('edit-link');
    if (filename.endsWith('.pdf')) {
        if (!editLink) {
            editLink = document.createElement('a');
            editLink.id = 'edit-link';
            editLink.className = 'edit-btn';
            editLink.textContent = '✏️ Edit';
            viewerActions.insertBefore(editLink, downloadLink);
        }
        editLink.href = '/editor.html?file=' + encodeURIComponent(filename);
        editLink.style.display = 'inline-block';

        viewerBody.innerHTML = `<iframe src="/output/${filename}#toolbar=1&navpanes=0"></iframe>`;
    } else {
        if (editLink) editLink.style.display = 'none';
        viewerBody.innerHTML = `
            <div class="viewer-placeholder">
                <div class="icon">&#128230;</div>
                <p>ZIP file ready for download</p>
                <p style="margin-top: 10px;"><a href="/output/${filename}" download style="color: #e63946;">Click to download</a></p>
            </div>
        `;
    }
}

async function refreshFileList() {
    try {
        const response = await fetch('/api/files');
        const files = await response.json();

        const fileList = document.getElementById('file-list');
        if (!fileList) return;

        if (files.length === 0) {
            fileList.innerHTML = '<div class="empty-state"><p>No processed files yet</p></div>';
            return;
        }

        fileList.innerHTML = files.map(file => `
            <div class="file-item" data-file="${file.name}" onclick="viewFile('${file.name}')">
                <div class="file-icon ${file.type === 'zip' ? 'zip' : ''}">
                    ${file.type.toUpperCase()}
                </div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                </div>
            </div>
        `).join('');
    } catch (err) {
        console.error('Failed to refresh file list:', err);
    }
}

// Viewer page initialization
function initViewer() {
    const params = new URLSearchParams(window.location.search);
    const filename = params.get('file');

    if (!filename) {
        window.location.href = '/';
        return;
    }

    const filenameDisplay = document.getElementById('filename-display');
    const downloadBtn = document.getElementById('download-btn');
    const viewerIframe = document.getElementById('viewer-iframe');

    if (filenameDisplay) filenameDisplay.textContent = filename;
    if (downloadBtn) downloadBtn.href = '/output/' + filename;
    if (viewerIframe) viewerIframe.src = '/output/' + filename + '#toolbar=1&navpanes=0';
}
