# Building Plan Classification Demo

This folder contains a web-based demonstration interface for the building plan classification pipeline.

## Quick Start

1. **Install dependencies** (from the main project directory):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo**:
   ```bash
   cd demo
   python demo_app.py
   ```

3. **Open browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Features

- **Web Interface**: Drag-and-drop file upload for ZIP files containing documents
- **Real-time Progress**: Live progress updates during processing
- **Memory Efficient**: No temporary files stored on disk - everything processed in memory
- **Performance Optimized**: Uses batch processing and optimizations from standalone testing scripts
- **Detailed Results**: Shows classification results with confidence scores and discipline breakdown

## Performance

This demo uses the same optimized processing logic as the standalone testing scripts:

- **Batch Processing**: CNN classifications are processed in batches for better performance
- **Streaming YOLO**: YOLO classifications use streaming for memory efficiency
- **File Sorting**: Images are sorted by size for consistent processing order
- **GPU Acceleration**: Automatically detects and uses GPU if available

## Architecture

- **Backend**: Flask application (`demo_app.py`) 
- **Frontend**: HTML/CSS/JavaScript (`templates/demo.html`)
- **Pipeline**: Uses optimized `ClassificationPipeline` from main project
- **Processing**: All file processing happens in memory without temporary disk storage

## Troubleshooting

- **Memory Issues**: Reduce batch size in `classification_pipeline.py` (line ~119: `batch_size = 32`)
- **GPU Issues**: GPU optimizations are optional and will fall back to CPU automatically
- **Model Loading**: Ensure model files exist in the correct locations relative to the demo folder

## Files

- `demo_app.py` - Flask web application
- `templates/demo.html` - Frontend interface
- `README.md` - This file