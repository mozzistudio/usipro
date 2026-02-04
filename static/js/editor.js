// USI-PRO Editor - AI-Assisted Zone Deletion

class PDFEditor {
    constructor() {
        this.canvas = document.getElementById('pdf-canvas');
        this.overlay = document.getElementById('overlay-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlayCtx = this.overlay.getContext('2d');

        this.filename = null;
        this.currentImage = null;
        this.history = [];
        this.historyIndex = -1;
        this.isEditMode = true;
        this.sensitivity = 5;

        this.init();
    }

    init() {
        // Get filename from URL
        const params = new URLSearchParams(window.location.search);
        this.filename = params.get('file');

        if (!this.filename) {
            window.location.href = '/';
            return;
        }

        // Update UI
        document.getElementById('filename-display').textContent = this.filename;
        document.getElementById('download-btn').href = '/output/' + this.filename;

        // Load the PDF as image
        this.loadPDF();

        // Setup event listeners
        this.setupEventListeners();
    }

    async loadPDF() {
        const loadingOverlay = document.getElementById('loading-overlay');
        loadingOverlay.style.display = 'flex';

        try {
            // Request PDF rendered as image from server
            const response = await fetch(`/api/pdf-image?file=${encodeURIComponent(this.filename)}`);
            const data = await response.json();

            if (data.success) {
                // Load the image
                const img = new Image();
                img.onload = () => {
                    this.currentImage = img;
                    this.setupCanvas(img.width, img.height);
                    this.drawImage();
                    this.saveState();
                    loadingOverlay.style.display = 'none';
                };
                img.src = 'data:image/png;base64,' + data.image;
            } else {
                alert('Failed to load PDF: ' + data.error);
                loadingOverlay.style.display = 'none';
            }
        } catch (err) {
            alert('Error loading PDF: ' + err.message);
            loadingOverlay.style.display = 'none';
        }
    }

    setupCanvas(width, height) {
        const container = document.getElementById('canvas-container');
        const maxWidth = container.clientWidth - 40;
        const maxHeight = window.innerHeight - 150;

        // Calculate scale to fit
        const scale = Math.min(maxWidth / width, maxHeight / height, 1);
        const displayWidth = Math.floor(width * scale);
        const displayHeight = Math.floor(height * scale);

        // Set canvas sizes
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.style.width = displayWidth + 'px';
        this.canvas.style.height = displayHeight + 'px';

        this.overlay.width = width;
        this.overlay.height = height;
        this.overlay.style.width = displayWidth + 'px';
        this.overlay.style.height = displayHeight + 'px';

        this.scale = scale;
        this.imageWidth = width;
        this.imageHeight = height;
    }

    drawImage() {
        this.ctx.drawImage(this.currentImage, 0, 0);
    }

    setupEventListeners() {
        // Canvas click for zone deletion
        this.overlay.addEventListener('click', (e) => this.handleCanvasClick(e));

        // Hover effect
        this.overlay.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.overlay.addEventListener('mouseleave', () => this.clearOverlay());

        // Edit mode toggle
        document.getElementById('edit-mode-btn').addEventListener('click', () => {
            this.isEditMode = !this.isEditMode;
            document.getElementById('edit-mode-btn').classList.toggle('active', this.isEditMode);
            this.overlay.style.cursor = this.isEditMode ? 'crosshair' : 'default';
        });

        // Undo button
        document.getElementById('undo-btn').addEventListener('click', () => this.undo());

        // Save button
        document.getElementById('save-btn').addEventListener('click', (e) => {
            e.preventDefault();
            this.savePDF();
        });

        // Sensitivity slider
        document.getElementById('sensitivity').addEventListener('input', (e) => {
            this.sensitivity = parseInt(e.target.value);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
                e.preventDefault();
                this.undo();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.savePDF();
            }
        });

        // Set initial cursor
        this.overlay.style.cursor = 'crosshair';
    }

    getCanvasCoordinates(e) {
        const rect = this.overlay.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale;
        const y = (e.clientY - rect.top) / this.scale;
        return { x: Math.floor(x), y: Math.floor(y) };
    }

    handleMouseMove(e) {
        if (!this.isEditMode) return;

        const { x, y } = this.getCanvasCoordinates(e);

        // Draw a small cursor indicator
        this.clearOverlay();
        this.overlayCtx.strokeStyle = 'rgba(230, 57, 70, 0.8)';
        this.overlayCtx.lineWidth = 2;
        this.overlayCtx.beginPath();
        this.overlayCtx.arc(x, y, 10 + this.sensitivity * 2, 0, Math.PI * 2);
        this.overlayCtx.stroke();
    }

    clearOverlay() {
        this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    }

    async handleCanvasClick(e) {
        if (!this.isEditMode) return;

        const { x, y } = this.getCanvasCoordinates(e);

        // Show processing overlay
        const processingOverlay = document.getElementById('processing-overlay');
        processingOverlay.style.display = 'flex';

        try {
            // Get current canvas data as base64
            const imageData = this.canvas.toDataURL('image/png').split(',')[1];

            // Send to server for AI zone detection
            const response = await fetch('/api/delete-zone', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: imageData,
                    x: x,
                    y: y,
                    sensitivity: this.sensitivity,
                    width: this.imageWidth,
                    height: this.imageHeight
                })
            });

            const result = await response.json();

            if (result.success) {
                // Load the new image with zone deleted
                const img = new Image();
                img.onload = () => {
                    this.currentImage = img;
                    this.drawImage();
                    this.saveState();
                    this.addHistoryItem(x, y, result.zone);
                    processingOverlay.style.display = 'none';
                };
                img.src = 'data:image/png;base64,' + result.image;
            } else {
                alert('Failed to delete zone: ' + result.error);
                processingOverlay.style.display = 'none';
            }
        } catch (err) {
            alert('Error: ' + err.message);
            processingOverlay.style.display = 'none';
        }
    }

    saveState() {
        // Remove any redo states
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }

        // Save current state
        const imageData = this.canvas.toDataURL('image/png');
        this.history.push(imageData);
        this.historyIndex = this.history.length - 1;

        // Update undo button
        document.getElementById('undo-btn').disabled = this.historyIndex <= 0;
    }

    undo() {
        if (this.historyIndex <= 0) return;

        this.historyIndex--;
        const img = new Image();
        img.onload = () => {
            this.currentImage = img;
            this.drawImage();
            document.getElementById('undo-btn').disabled = this.historyIndex <= 0;
            this.updateHistoryUI();
        };
        img.src = this.history[this.historyIndex];
    }

    addHistoryItem(x, y, zone) {
        const historyList = document.getElementById('history-list');
        const emptyMsg = historyList.querySelector('.empty-history');
        if (emptyMsg) emptyMsg.remove();

        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <span class="history-icon">🗑️</span>
            <span class="history-text">Deleted zone at (${x}, ${y})</span>
        `;
        historyList.appendChild(item);
        historyList.scrollTop = historyList.scrollHeight;
    }

    updateHistoryUI() {
        const historyList = document.getElementById('history-list');
        const items = historyList.querySelectorAll('.history-item');
        items.forEach((item, index) => {
            item.style.opacity = index < this.historyIndex ? '1' : '0.4';
        });
    }

    async savePDF() {
        const saveBtn = document.getElementById('save-btn');
        saveBtn.textContent = 'Saving...';
        saveBtn.style.pointerEvents = 'none';

        try {
            // Get current canvas data
            const imageData = this.canvas.toDataURL('image/png').split(',')[1];

            // Send to server to save as PDF
            const response = await fetch('/api/save-edited-pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: this.filename,
                    image: imageData,
                    width: this.imageWidth,
                    height: this.imageHeight
                })
            });

            const result = await response.json();

            if (result.success) {
                saveBtn.textContent = 'Saved!';
                setTimeout(() => {
                    saveBtn.textContent = 'Save PDF';
                    saveBtn.style.pointerEvents = 'auto';
                }, 2000);

                // Update download link
                document.getElementById('download-btn').href = '/output/' + result.filename + '?t=' + Date.now();
            } else {
                alert('Failed to save: ' + result.error);
                saveBtn.textContent = 'Save PDF';
                saveBtn.style.pointerEvents = 'auto';
            }
        } catch (err) {
            alert('Error saving: ' + err.message);
            saveBtn.textContent = 'Save PDF';
            saveBtn.style.pointerEvents = 'auto';
        }
    }
}

// Initialize editor when page loads
document.addEventListener('DOMContentLoaded', () => {
    new PDFEditor();
});
