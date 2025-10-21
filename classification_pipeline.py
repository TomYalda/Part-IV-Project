"""
Classification Pipeline for Building Plan Identification and Classification

This script provides a complete pipeline that:
1. Extracts and preprocesses data from a zip file using the existing extract_and_process module
2. Runs building plan identification using the first classifier (best_classifier.h5)
3. Runs plan type classification on identified building plans using YOLO classifier
4. Outputs classified results with confidence scores

Usage:
    python classification_pipeline.py <input_zip_path> <output_directory>
"""

import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import extract_and_process

class ClassificationPipeline:
    def __init__(self, building_classifier_path=r"C:\Users\tomya\OneDrive - The University of Auckland\Documents\2025 Work\University\Part IV Project\Models\Part-IV-Project\Building-Plan-Identification\best_classifier.h5",
                 plan_classifier_path=r"C:\Users\tomya\OneDrive - The University of Auckland\Documents\2025 Work\University\Part IV Project\Models\Part-IV-Project\Plan classification\plan_classifier_yolo.pt"):
        """
        Initialize the classification pipeline with model paths
        
        Args:
            building_classifier_path: Path to the building plan identification model
            plan_classifier_path: Path to the YOLO plan classification model
        """
        self.building_classifier_path = building_classifier_path
        self.plan_classifier_path = plan_classifier_path
        self.building_classifier = None
        self.plan_classifier = None
        self.building_class_names = ['Documents', 'StructuralPlans']
        # Use correct input size that matches the actual model training (600x600)
        # The model expects input shape (600, 600, 3) which flattens to 682112 features
        self.building_model_input_size = (600, 600)
        
        # Ensure output temp folder exists
        os.makedirs("tmp", exist_ok=True)
        
    def load_models(self):
        """Load both classification models with performance optimizations"""
        print("Loading building plan identification model...")
        
        # Optimize TensorFlow for better performance
        try:
            # Enable mixed precision for faster inference if GPU is available
            if tf.config.list_physical_devices('GPU'):
                print("GPU detected - enabling optimizations...")
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
        except Exception as e:
            print(f"GPU optimization not available: {e}")
        
        # Load the CNN model
        self.building_classifier = load_model(self.building_classifier_path)
        
        # Compile model with optimizations for inference
        self.building_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("Loading plan type classification model...")
        self.plan_classifier = YOLO(self.plan_classifier_path)
        
        print("Models loaded successfully!")
    
    def extract_and_preprocess(self, zip_path, temp_dir):
        """Extract zip file and convert all supported files to JPEG images using existing extract_and_process module"""
        print(f"Extracting and preprocessing: {zip_path}")
        
        # Create images folder in temp directory
        images_dir = os.path.join(temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Temporarily modify the extract_and_process OUTPUT_DIR to use our temp directory
        original_output_dir = extract_and_process.OUTPUT_DIR
        extract_and_process.OUTPUT_DIR = images_dir
        
        try:
            # Use the existing extract_and_process function
            extract_and_process.extract_and_process(zip_path, images_dir)
            
            # Get list of processed images
            processed_images = []
            for filename in os.listdir(images_dir):
                if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
                    processed_images.append(os.path.join(images_dir, filename))
            
            print(f"Preprocessed {len(processed_images)} images")
            return processed_images
            
        finally:
            # Restore original OUTPUT_DIR
            extract_and_process.OUTPUT_DIR = original_output_dir
    
    def classify_building_plans(self, image_paths, progress_callback=None):
        """
        Classify images to identify building plans vs other documents using optimized batch processing
        
        Returns:
            building_plans: List of image paths classified as building plans
            classifications: Dict with all classifications and confidence scores
        """
        print(f"Classifying {len(image_paths)} images for building plan identification...")
        
        building_plans = []
        classifications = {}
        
        # Sort files by size for consistent processing order (matches standalone script optimization)
        try:
            sorted_paths = sorted(image_paths, key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0)
        except Exception:
            sorted_paths = image_paths
        
        # Process in batches for better memory management and performance
        batch_size = 32  # Optimize batch size for performance
        total_batches = len(sorted_paths) // batch_size + (1 if len(sorted_paths) % batch_size != 0 else 0)
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sorted_paths))
            batch_paths = sorted_paths[start_idx:end_idx]
            
            # Prepare batch data
            batch_images = []
            valid_paths = []
            
            for img_path in batch_paths:
                try:
                    # Load and preprocess image (matches standalone script exactly)
                    img = image.load_img(img_path, target_size=self.building_model_input_size)
                    img_array = image.img_to_array(img)
                    img_array /= 255.0  # Normalize same as training
                    batch_images.append(img_array)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    classifications[img_path] = {
                        'class': 'Error',
                        'confidence': 0.0
                    }
            
            if batch_images:
                # Batch prediction (much faster than individual predictions)
                batch_array = np.array(batch_images)
                predictions = self.building_classifier.predict(batch_array, verbose=0)
                
                # Process batch results
                for i, (img_path, prediction) in enumerate(zip(valid_paths, predictions)):
                    pred_prob = float(prediction[0])  # Convert to Python float immediately
                    predicted_class = self.building_class_names[int(pred_prob > 0.5)]
                    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
                    
                    classifications[img_path] = {
                        'class': predicted_class,
                        'confidence': confidence  # Already converted to Python float
                    }
                    
                    # If classified as structural plan, add to building plans list
                    if predicted_class == 'StructuralPlans':
                        building_plans.append(img_path)
            
            # Update progress
            if progress_callback:
                progress = 30 + ((batch_idx + 1) / total_batches) * 40  # CNN takes 30-70% of total progress
                progress_callback(f"CNN Analysis: {end_idx}/{len(sorted_paths)} images processed", progress)
        
        print(f"Identified {len(building_plans)} building plans out of {len(image_paths)} images")
        return building_plans, classifications
    
    def classify_plan_types(self, building_plan_paths, progress_callback=None):
        """
        Classify building plans into specific plan types using optimized batch YOLO processing
        
        Returns:
            plan_classifications: Dict with plan type classifications and confidence scores
        """
        if not building_plan_paths:
            print("No building plans to classify")
            return {}
            
        print(f"Classifying {len(building_plan_paths)} building plans for plan types...")
        
        plan_classifications = {}
        
        # Use batch prediction for much better performance (matches standalone testModel.py)
        try:
            # YOLO can handle batch processing efficiently via stream=True
            results = self.plan_classifier.predict(
                source=building_plan_paths,  # Pass all paths at once for batch processing
                stream=True,  # Use streaming for memory efficiency
                verbose=False
            )
            
            processed_count = 0
            total_plans = len(building_plan_paths)
            
            for result in results:
                processed_count += 1
                plan_path = result.path
                
                if result.probs is not None:
                    probs = result.probs.data
                    predicted_index = np.array(probs).argmax()
                    predicted_name = self.plan_classifier.names[predicted_index]
                    confidence = float(probs[predicted_index])
                    
                    plan_classifications[plan_path] = {
                        'plan_type': predicted_name,
                        'confidence': confidence  # Already converted via float() above
                    }
                else:
                    plan_classifications[plan_path] = {
                        'plan_type': 'Unknown',
                        'confidence': 0.0
                    }
                
                # Progress callback
                if progress_callback:
                    progress = 70 + (processed_count / total_plans) * 25  # YOLO takes 70-95% of total progress
                    progress_callback(f"YOLO Classification: {processed_count}/{total_plans} plans classified", progress)
        
        except Exception as e:
            print(f"Error in batch YOLO classification: {e}")
            # Fallback to individual processing if batch fails
            for i, plan_path in enumerate(building_plan_paths):
                try:
                    results = self.plan_classifier.predict(source=plan_path, verbose=False)
                    
                    for result in results:
                        if result.probs is not None:
                            probs = result.probs.data
                            predicted_index = np.array(probs).argmax()
                            predicted_name = self.plan_classifier.names[predicted_index]
                            confidence = float(probs[predicted_index])
                            
                            plan_classifications[plan_path] = {
                                'plan_type': predicted_name,
                                'confidence': confidence
                            }
                        else:
                            plan_classifications[plan_path] = {
                                'plan_type': 'Unknown',
                                'confidence': 0.0
                            }
                        
                except Exception as e2:
                    print(f"Error classifying plan type for {plan_path}: {e2}")
                    plan_classifications[plan_path] = {
                        'plan_type': 'Error',
                        'confidence': 0.0
                    }
                
                # Progress callback for fallback
                if progress_callback:
                    progress = 70 + ((i + 1) / len(building_plan_paths)) * 25
                    progress_callback(f"YOLO Classification: {i+1}/{len(building_plan_paths)} plans classified", progress)
        
        return plan_classifications
    
    def organize_results(self, output_dir, building_classifications, plan_classifications, temp_dir):
        """Organize classified results into output directory structure"""
        print("Organizing results...")
        
        # Create output directories
        building_plans_dir = os.path.join(output_dir, "building_plans")
        documents_dir = os.path.join(output_dir, "documents")
        os.makedirs(building_plans_dir, exist_ok=True)
        os.makedirs(documents_dir, exist_ok=True)
        
        # Create subdirectories for plan types
        plan_types = set()
        for classification in plan_classifications.values():
            plan_types.add(classification['plan_type'])
        
        for plan_type in plan_types:
            if plan_type not in ['Unknown', 'Error']:
                os.makedirs(os.path.join(building_plans_dir, plan_type), exist_ok=True)
        
        # Copy files to appropriate directories
        results_summary = {
            'total_images': len(building_classifications),
            'building_plans': 0,
            'documents': 0,
            'plan_types': {}
        }
        
        for img_path, classification in building_classifications.items():
            filename = os.path.basename(img_path)
            
            if classification['class'] == 'StructuralPlans':
                results_summary['building_plans'] += 1
                
                # Check if we have plan type classification
                if img_path in plan_classifications:
                    plan_info = plan_classifications[img_path]
                    plan_type = plan_info['plan_type']
                    
                    if plan_type not in results_summary['plan_types']:
                        results_summary['plan_types'][plan_type] = 0
                    results_summary['plan_types'][plan_type] += 1
                    
                    # Copy to plan type subdirectory
                    if plan_type not in ['Unknown', 'Error']:
                        dest_path = os.path.join(building_plans_dir, plan_type, filename)
                    else:
                        dest_path = os.path.join(building_plans_dir, filename)
                else:
                    dest_path = os.path.join(building_plans_dir, filename)
                    
            else:  # Documents
                results_summary['documents'] += 1
                dest_path = os.path.join(documents_dir, filename)
            
            # Copy file
            shutil.copy2(img_path, dest_path)
        
        return results_summary
    
    def save_results_report(self, output_dir, building_classifications, plan_classifications, results_summary, processing_time):
        """Save detailed classification report"""
        report_path = os.path.join(output_dir, "classification_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BUILDING PLAN CLASSIFICATION PIPELINE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing completed in: {processing_time:.2f} seconds\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"Total images processed: {results_summary['total_images']}\n")
            f.write(f"Building plans identified: {results_summary['building_plans']}\n")
            f.write(f"Documents identified: {results_summary['documents']}\n\n")
            
            if results_summary['plan_types']:
                f.write("PLAN TYPE BREAKDOWN:\n")
                for plan_type, count in results_summary['plan_types'].items():
                    f.write(f"  {plan_type}: {count}\n")
                f.write("\n")
            
            f.write("DETAILED CLASSIFICATIONS:\n")
            f.write("-" * 30 + "\n")
            
            for img_path, classification in building_classifications.items():
                filename = os.path.basename(img_path)
                f.write(f"\nFile: {filename}\n")
                f.write(f"  Building Classification: {classification['class']} ({classification['confidence']:.3f})\n")
                
                if img_path in plan_classifications:
                    plan_info = plan_classifications[img_path]
                    f.write(f"  Plan Type: {plan_info['plan_type']} ({plan_info['confidence']:.3f})\n")
        
        print(f"Detailed report saved to: {report_path}")
    
    def run_pipeline(self, zip_path, output_dir):
        """
        Run the complete classification pipeline
        
        Args:
            zip_path: Path to input zip file
            output_dir: Directory to save classified results
        """
        start_time = time.time()
        
        print("="*50)
        print("BUILDING PLAN CLASSIFICATION PIPELINE")
        print("="*50)
        
        # Validate inputs
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Input zip file not found: {zip_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models
        self.load_models()
        
        # Process in temporary directory
        with tempfile.TemporaryDirectory(dir="tmp") as temp_dir:
            # Step 1: Extract and preprocess
            print("\n" + "="*50)
            print("STEP 1: EXTRACT AND PREPROCESS")
            print("="*50)
            image_paths = self.extract_and_preprocess(zip_path, temp_dir)
            
            if not image_paths:
                print("No images found to process!")
                return
            
            # Step 2: Building plan identification
            print("\n" + "="*50)
            print("STEP 2: BUILDING PLAN IDENTIFICATION")
            print("="*50)
            building_plans, building_classifications = self.classify_building_plans(image_paths)
            
            # Step 3: Plan type classification
            print("\n" + "="*50)
            print("STEP 3: PLAN TYPE CLASSIFICATION")
            print("="*50)
            plan_classifications = self.classify_plan_types(building_plans)
            
            # Step 4: Organize results
            print("\n" + "="*50)
            print("STEP 4: ORGANIZE RESULTS")
            print("="*50)
            results_summary = self.organize_results(output_dir, building_classifications, plan_classifications, temp_dir)
            
            # Step 5: Generate report
            processing_time = time.time() - start_time
            self.save_results_report(output_dir, building_classifications, plan_classifications, results_summary, processing_time)
            
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Results saved to: {output_dir}")
        print(f"Total images: {results_summary['total_images']}")
        print(f"Building plans: {results_summary['building_plans']}")
        print(f"Documents: {results_summary['documents']}")


def main():
    """Main function to run the pipeline from command line"""
    if len(sys.argv) != 3:
        print("Usage: python classification_pipeline.py <input_zip_path> <output_directory>")
        print("Example: python classification_pipeline.py data.zip results/")
        sys.exit(1)
    
    zip_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        pipeline = ClassificationPipeline()
        pipeline.run_pipeline(zip_path, output_dir)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()