# Building Plan Classification Pipeline

This repository contains a comprehensive two-stage machine learning pipeline for automatically identifying and classifying building plans from document collections. The pipeline combines computer vision and deep learning techniques to process mixed document sets and extract valuable building plan information.

## Pipeline Overview

The classification pipeline operates in two sequential stages:

### Stage 1: Building Plan Identification
- **Model**: `best_classifier.h5` (TensorFlow/Keras CNN)
- **Purpose**: Distinguishes between building plans and other documents
- **Classes**: `Documents` vs `StructuralPlans`
- **Input**: Mixed document images (extracted from zip files)
- **Output**: Filtered set containing only building plans

### Stage 2: Plan Type Classification
- **Model**: `plan_classifier_yolo.pt` (YOLOv8 Classification)
- **Purpose**: Classifies building plans into specific plan types
- **Input**: Building plans identified from Stage 1
- **Output**: Categorized building plans with confidence scores

## Quick Start

### Using the Complete Pipeline
```bash
python classification_pipeline.py input_documents.zip output_results/
```

### Using Individual Components
```bash
# Extract and preprocess documents
python extract_and_process.py

# Classify single image for building plan identification
python Building-Plan-Identification/classifyFile.py

# Test YOLO plan classifier
python "Plan classification/testModel.py"
```

## Repository Structure

### Root Directory Files

#### `classification_pipeline.py`
**Complete end-to-end pipeline script** that integrates all components:
- Uses the existing `extract_and_process.py` module for document extraction and preprocessing
- Runs building plan identification using the TensorFlow model (900x900 input size)
- Classifies identified plans using the YOLO model
- Organizes results into structured output directories
- Generates comprehensive classification reports

#### `extract_and_process.py`
**Data preprocessing utility** that handles document extraction and conversion:
- Recursively extracts zip files (supports nested zips up to 10 levels deep)
- Converts PDFs to high-quality JPEG images (300 DPI)
- Processes various image formats (PNG, TIFF, JPEG)
- Sanitizes filenames and handles duplicate names
- Optimized for batch processing large document collections

### Building-Plan-Identification/
**First-stage classifier for identifying building plans vs other documents**

#### Core Model Files
- `best_classifier.h5` - **Primary trained model** (TensorFlow/Keras CNN optimized for document classification)
- `123classifier_large.h5` - Alternative model variant

#### Training and Testing Scripts
- `trainModel.py` - **Model training script** with data augmentation, early stopping, and batch normalization
- `classifyFile.py` - **Single image classifier** for testing individual documents
- `classifyFolderFiles.py` - **Batch classifier** for processing entire directories
- `MyDetermination.txt` - Manual validation results and model performance notes

#### Analysis and Evaluation
- `classification_metrics.csv` - **Performance metrics** including accuracy, precision, recall, and F1-scores
- `resultsCalculation.py` - **Metrics calculation script** for evaluating model performance
- `text_sorter.py` - Utility for organizing and sorting classification results
- `classifiersDeterminations.txt` - Detailed classification results and model comparisons

### Plan classification/
**Second-stage YOLO classifier for categorizing building plan types**

#### Core Model Files
- `plan_classifier_yolo.pt` - **Primary YOLO classification model** trained for plan type recognition
- `yolov8l-cls.pt` - Base YOLOv8 large classification model
- `yolov8n-cls.pt` - YOLOv8 nano model for faster inference

#### Training and Testing Scripts
- `trainModel.py` - **YOLO training script** using transfer learning from pre-trained weights
- `testModel.py` - **Model evaluation script** that processes test datasets and generates prediction reports
- `splitData.py` - **Dataset preparation utility** for creating train/validation/test splits

#### Results and Analysis
- `classification_metrics_final.csv` - **Final model performance metrics**
- `prediction_results.txt` - Detailed prediction outputs with confidence scores
- `correct_classifications.txt` - Validation results for model accuracy assessment
- `resultsCalculation.py` - Performance analysis and metrics calculation
- `text_sorter.py` - Results organization utility

#### Training Iterations
- `iteration_results/` - Contains results from multiple training iterations:
  - `initial classifier (train4)/` - First training attempt with baseline performance
  - `second classifier (train5)/` - Improved model with enhanced data augmentation
  - `third classifier (train8)/` - Further optimized with adjusted hyperparameters
  - `fourth classifier (train20)/` - Advanced training with extended epochs
  - `fifth classifier (train21)/` - Final optimized model with a 70/20/10 data split

#### Training Runs
- `runs/classify/` - **YOLOv8 training outputs** including:
  - Model weights at different training stages
  - Training metrics and loss curves
  - Validation results and performance charts
  - Configuration files (`args.yaml`) for reproducibility

## Model Details

### Building Plan Identification Model
- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input Size**: 900x900 pixels
- **Training Features**:
  - Data augmentation (rotation, shifting, brightness, zoom)
  - Batch normalization for stable training
  - Early stopping to prevent overfitting
  - Binary classification with sigmoid activation

### Plan Classification Model
- **Architecture**: YOLOv8 Classification
- **Framework**: Ultralytics YOLO
- **Input Size**: 600x600 pixels
- **Training Features**:
  - Transfer learning from pre-trained weights
  - Multi-class classification capabilities
  - Built-in data augmentation
  - Configurable patience and validation monitoring

## Pipeline Workflow

1. **Input Processing**
   - Extract zip files recursively
   - Convert all documents to standardized JPEG format
   - Maintain original folder structure and naming

2. **Stage 1: Document Classification**
   - Process each image through the building plan identification model
   - Filter images classified as "StructuralPlans"
   - Generate confidence scores for each classification

3. **Stage 2: Plan Type Classification**
   - Process identified building plans through YOLO classifier
   - Categorize plans into specific types (architectural, civil, electrical, fire_protection, mechanical, plumbing, services_coordination, structural)
   - Provide detailed confidence metrics

4. **Results Organization**
   - Create structured output directories
   - Separate building plans by type
   - Generate comprehensive classification report
   - Maintain traceability to original files

## Output Structure

```
output_directory/
├── building_plans/
│   ├── architectural/
│   ├── civil/
│   ├── .../
│   └── structural/
├── documents/
│   └── [non-building plan documents]
└── classification_report.txt
```

## Performance Metrics

The models have been extensively evaluated with metrics including:
- Accuracy scores for both classification stages
- Precision and recall for each class
- F1-scores for balanced performance assessment
- Processing time benchmarks
- Cross-validation results across multiple iterations

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Ultralytics YOLO
- PIL (Python Imaging Library)
- pdf2image
- NumPy
- OpenCV (cv2)

## Installation

```bash
pip install tensorflow ultralytics pillow pdf2image numpy opencv-python
```

## Data Structure

The project expects data to be organized as follows:

```
data/
├── plan_classification/
│   ├── train/
│   │   ├── architectural/
│   │   ├── civil/
│   │   ├── structural/
│   │   └── [other plan types]/
│   └── split-data/
│       ├── train/
│       ├── val/
│       └── test/
├── building_identification/
│   ├── trainingData/
│   │   ├── Documents/
│   │   └── StructuralPlans/
│   └── testingData/
│       └── validationData/
└── test_images/
    └── [sample images for testing]
```

## Contributing

This project supports building plan classification for document management. The modular design allows for easy extension with additional classification stages or alternative model architectures.

## Notes

- Paths in the code are relative and should be adjusted based on your data directory structure
- Model files should be placed in their respective directories (Building-Plan-Identification/ and Plan classification/)
- Temporary files are created in a local `tmp/` directory during processing
