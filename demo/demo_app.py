"""
Building Plan Classification Demo Web Application v2

Uses the existing ClassificationPipeline for consistency with testing scripts.
Provides a simple Flask web application that demonstrates the building plan classification pipeline
with real-time progress visualization for demonstration purposes.

Usage:
    python demo_app_v2.py
    
Then open: http://localhost:5000
"""

import os
import sys
import time
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

# Add parent directory to path to import classification_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optimize TensorFlow for better performance
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores

# Import the existing pipeline
from classification_pipeline import ClassificationPipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure temporary and demo results directories exist
os.makedirs("tmp", exist_ok=True)
os.makedirs("demo_results", exist_ok=True)

class ClassificationDemo:
    """Demo wrapper around the existing ClassificationPipeline for web interface"""
    
    def __init__(self):
        """Initialize the demo with the existing pipeline"""
        self.pipeline = ClassificationPipeline()
        self.is_loaded = False
        
    def load_models(self):
        """Load the classification models using the existing pipeline"""
        if self.is_loaded:
            return True
            
        try:
            print("Loading models using classification pipeline...")
            self.pipeline.load_models()
            
            # Warm up models with dummy predictions for better performance
            print("Warming up models...")
            dummy_image = np.random.random((1, 600, 600, 3))  # Use correct input size (600x600)
            _ = self.pipeline.building_classifier.predict(dummy_image, verbose=0)
            print("Models warmed up successfully!")
            
            self.is_loaded = True
            print("Pipeline models loaded and ready!")
            return True
        except Exception as e:
            print(f"Error loading pipeline models: {e}")
            return False
    
    def process_demo_file(self, file_path, progress_callback=None):
        """Process a file using the existing pipeline with progress updates"""
        results = {
            'stages': [],
            'total_images': 0,
            'building_plans': 0,
            'documents': 0,
            'plan_types': {},
            'processing_time': 0,
            'files': []
        }
        
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback("Stage 1: Extracting and preprocessing files...", 10)
            
            with tempfile.TemporaryDirectory(dir="tmp") as temp_dir:
                # Step 1: Extract and preprocess using the pipeline
                if file_path.lower().endswith('.zip'):
                    image_paths = self.pipeline.extract_and_preprocess(file_path, temp_dir)
                else:
                    # Single image file
                    images_dir = os.path.join(temp_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    dest_path = os.path.join(images_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)
                    image_paths = [dest_path]
                
                results['total_images'] = len(image_paths)
                
                if not image_paths:
                    results['error'] = "No valid images found in the uploaded file"
                    return results
                
                results['stages'].append({
                    'name': 'File Extraction',
                    'status': 'completed',
                    'details': f"Extracted and processed {len(image_paths)} images",
                    'timestamp': datetime.now().isoformat()
                })
                
                if progress_callback:
                    progress_callback("Stage 2: Identifying building plans with CNN...", 30)
                
                # Step 2: Building plan identification using pipeline
                building_plans, building_classifications = self.pipeline.classify_building_plans(image_paths, progress_callback)
                
                results['building_plans'] = len(building_plans)
                results['documents'] = len(image_paths) - len(building_plans)
                
                results['stages'].append({
                    'name': 'Building Plan Identification',
                    'status': 'completed',
                    'details': f"Found {len(building_plans)} building plans out of {len(image_paths)} images",
                    'timestamp': datetime.now().isoformat()
                })
                
                if progress_callback:
                    progress_callback("Stage 3: Classifying plan types with YOLO...", 70)
                
                # Step 3: Plan type classification using pipeline
                plan_classifications = self.pipeline.classify_plan_types(building_plans, progress_callback)
                
                # Count plan types
                for plan_info in plan_classifications.values():
                    plan_type = plan_info['plan_type']
                    if plan_type not in results['plan_types']:
                        results['plan_types'][plan_type] = 0
                    results['plan_types'][plan_type] += 1
                
                results['stages'].append({
                    'name': 'Plan Type Classification',
                    'status': 'completed',
                    'details': f"Classified {len(building_plans)} building plans into {len(results['plan_types'])} types",
                    'timestamp': datetime.now().isoformat()
                })
                
                # Prepare file results for display
                for img_path, building_classification in building_classifications.items():
                    filename = os.path.basename(img_path)
                    
                    file_result = {
                        'filename': filename,
                        'building_classification': building_classification['class'],
                        'building_confidence': building_classification['confidence'],
                        'plan_type': None,
                        'plan_confidence': None
                    }
                    
                    # Add plan type info if available
                    if img_path in plan_classifications:
                        plan_info = plan_classifications[img_path]
                        file_result['plan_type'] = plan_info['plan_type']
                        file_result['plan_confidence'] = plan_info['confidence']
                    
                    results['files'].append(file_result)
                
                # Step 4: Organize and save results to persistent storage
                if progress_callback:
                    progress_callback("Stage 4: Organizing and saving results...", 95)
                
                # Create demo results directory with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                demo_output_dir = os.path.join("demo_results", f"classification_{timestamp}")
                
                # Use pipeline's organize_results method to save classified images
                results_summary = self.pipeline.organize_results(demo_output_dir, building_classifications, plan_classifications, temp_dir)
                
                # Save detailed report
                self.pipeline.save_results_report(demo_output_dir, building_classifications, plan_classifications, results_summary, time.time() - start_time)
                
                results['stages'].append({
                    'name': 'Results Storage',
                    'status': 'completed',
                    'details': f"Saved {results_summary['total_images']} images to {demo_output_dir}",
                    'timestamp': datetime.now().isoformat()
                })
                
                # Add output directory info to results
                results['output_directory'] = demo_output_dir
                results['saved_files'] = {
                    'building_plans_dir': os.path.join(demo_output_dir, "building_plans"),
                    'documents_dir': os.path.join(demo_output_dir, "documents"),
                    'classification_report': os.path.join(demo_output_dir, "classification_report.txt")
                }
                
                # Add summary statistics
                results['summary'] = {
                    'total_files': len(results['files']),
                    'building_plans': results['building_plans'],
                    'documents': results['documents'],
                    'plan_type_counts': results['plan_types'].copy()
                }
                
                results['processing_time'] = time.time() - start_time
                
                if progress_callback:
                    progress_callback("Classification and storage complete!", 100)
                
        except Exception as e:
            print(f"Error in processing: {e}")
            results['error'] = str(e)
        
        return results

# Global demo instance and progress tracking
demo = ClassificationDemo()
current_progress = {'status': 'idle', 'progress': 0, 'message': ''}

@app.route('/')
def index():
    """Main demo page"""
    return render_template('demo.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload without saving to disk"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Process file directly from memory without saving
        return jsonify({
            'success': True,
            'filename': file.filename,
            'message': 'File received successfully'
        })

@app.route('/process', methods=['POST'])
def process_file():
    """Process uploaded file directly from memory"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400
    
    # Load models if not already loaded
    if not demo.load_models():
        return jsonify({'error': 'Failed to load models'}), 500
    
    # Define progress callback
    def progress_callback(message, progress):
        global current_progress
        current_progress = {
            'status': 'processing',
            'progress': progress,
            'message': message
        }
    
    # Process the file with progress tracking
    global current_progress
    current_progress = {'status': 'processing', 'progress': 0, 'message': 'Starting...'}
    
    # Create temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip' if file.filename.lower().endswith('.zip') else '.jpg') as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        results = demo.process_demo_file(temp_path, progress_callback)
        current_progress = {'status': 'completed', 'progress': 100, 'message': 'Processing complete!'}
        return jsonify(results)
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

@app.route('/progress')
def get_progress():
    """Get current processing progress"""
    global current_progress
    return jsonify(current_progress)

@app.route('/status')
def status():
    """Get current status of the demo application"""
    return jsonify({
        'models_loaded': demo.is_loaded,
        'uptime': time.time()
    })

@app.route('/results')
def list_results():
    """List all saved classification results"""
    try:
        results_dirs = []
        demo_results_path = "demo_results"
        
        if os.path.exists(demo_results_path):
            for dirname in sorted(os.listdir(demo_results_path), reverse=True):
                dir_path = os.path.join(demo_results_path, dirname)
                if os.path.isdir(dir_path) and dirname.startswith('classification_'):
                    
                    # Get directory info
                    dir_info = {
                        'name': dirname,
                        'path': dir_path,
                        'timestamp': dirname.replace('classification_', ''),
                        'created': os.path.getctime(dir_path)
                    }
                    
                    # Count files in subdirectories
                    building_plans_dir = os.path.join(dir_path, "building_plans")
                    documents_dir = os.path.join(dir_path, "documents")
                    
                    building_count = 0
                    document_count = 0
                    plan_types = []
                    
                    if os.path.exists(building_plans_dir):
                        # Count files in building_plans and its subdirectories
                        for root, dirs, files in os.walk(building_plans_dir):
                            building_count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            # Get plan type names from subdirectories
                            if root != building_plans_dir:
                                plan_types.append(os.path.basename(root))
                    
                    if os.path.exists(documents_dir):
                        document_count = len([f for f in os.listdir(documents_dir) 
                                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    
                    dir_info.update({
                        'building_plans': building_count,
                        'documents': document_count,
                        'plan_types': list(set(plan_types)),
                        'total_files': building_count + document_count
                    })
                    
                    results_dirs.append(dir_info)
        
        return jsonify({
            'results': results_dirs,
            'total_sessions': len(results_dirs)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error listing results: {str(e)}'}), 500

@app.route('/results/<session_name>')
def get_result_details(session_name):
    """Get detailed information about a specific classification session"""
    try:
        session_path = os.path.join("demo_results", session_name)
        
        if not os.path.exists(session_path):
            return jsonify({'error': 'Session not found'}), 404
        
        # Read classification report if available
        report_path = os.path.join(session_path, "classification_report.txt")
        report_content = ""
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
        
        # Get file structure
        structure = {
            'building_plans': {},
            'documents': []
        }
        
        building_plans_dir = os.path.join(session_path, "building_plans")
        documents_dir = os.path.join(session_path, "documents")
        
        # Get building plans organized by type
        if os.path.exists(building_plans_dir):
            for item in os.listdir(building_plans_dir):
                item_path = os.path.join(building_plans_dir, item)
                if os.path.isdir(item_path):
                    # Plan type subdirectory
                    structure['building_plans'][item] = [
                        f for f in os.listdir(item_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Direct files in building_plans (Unknown/Error cases)
                    if 'uncategorized' not in structure['building_plans']:
                        structure['building_plans']['uncategorized'] = []
                    structure['building_plans']['uncategorized'].append(item)
        
        # Get documents
        if os.path.exists(documents_dir):
            structure['documents'] = [
                f for f in os.listdir(documents_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
        
        return jsonify({
            'session_name': session_name,
            'session_path': session_path,
            'report': report_content,
            'structure': structure,
            'has_report': os.path.exists(report_path)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting session details: {str(e)}'}), 500

@app.route('/download/<session_name>/<path:file_path>')
def download_result_file(session_name, file_path):
    """Download a specific file from the results"""
    try:
        full_path = os.path.join("demo_results", session_name, file_path)
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Security check: ensure path is within demo_results
        abs_demo_path = os.path.abspath("demo_results")
        abs_file_path = os.path.abspath(full_path)
        
        if not abs_file_path.startswith(abs_demo_path):
            return jsonify({'error': 'Invalid file path'}), 403
        
        return send_file(full_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Building Plan Classification Demo v2...")
    print("Using existing ClassificationPipeline for consistency with testing scripts")
    print("Loading models...")
    
    # Pre-load models
    if demo.load_models():
        print("Models loaded successfully!")
        print("Demo application ready!")
        print("Open your browser to: http://localhost:5000")
    else:
        print("Warning: Models could not be loaded. Some features may not work.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)