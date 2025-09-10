import os
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
import SimpleITK as sitk
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import CLIPProcessor, CLIPModel

class MILK10kPipeline:
    def __init__(self, dataset_path: str, groundtruth_path: str, output_path: str,
                 sam2_model_path: str, conceptclip_model_path: str, cache_path: str):
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = Path(groundtruth_path)
        self.output_path = Path(output_path)
        self.sam2_model_path = Path(sam2_model_path)
        self.conceptclip_model_path = Path(conceptclip_model_path)
        self.cache_path = Path(cache_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize SAM2
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(self.sam2_model_path)
        self.sam2_predictor.to(self.device)
        
        # Initialize ConceptCLIP
        self.conceptclip_processor = CLIPProcessor.from_pretrained(self.conceptclip_model_path, cache_dir=self.cache_path)
        self.conceptclip_model = CLIPModel.from_pretrained(self.conceptclip_model_path, cache_dir=self.cache_path)
        self.conceptclip_model.to(self.device)
        
        # Define domain-specific configurations
        self.domain = type('Domain', (), {
            'image_extensions': ['.png', '.jpg', '.jpeg', '.dcm', '.nii', '.nii.gz'],
            'text_prompts': [
                "A medical image showing healthy tissue",
                "A medical image showing inflammation",
                "A medical image showing tumor",
                "A medical image showing degenerative changes"
            ]
        })()

    def preprocess_image(self, img_path: Path) -> np.ndarray:
        """Load and preprocess image based on file type"""
        try:
            ext = img_path.suffix.lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif ext == '.dcm':
                ds = sitk.ReadImage(str(img_path))
                image = sitk.GetArrayFromImage(ds)[0]
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif ext in ['.nii', '.nii.gz']:
                ds = sitk.ReadImage(str(img_path))
                image = sitk.GetArrayFromImage(ds)[0]
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                print(f"Unsupported file type: {ext}")
                return None
            
            # Normalize and resize
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = cv2.resize(image, (512, 512))
            return image
        except Exception as e:
            print(f"Error preprocessing {img_path}: {e}")
            return None

    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment image using SAM2"""
        try:
            self.sam2_predictor.set_image(image)
            mask, score = self.sam2_predictor.predict(point_coords=np.array([[256, 256]]), point_labels=np.array([1]))
            if isinstance(mask, list):
                mask = mask[0]
            return mask, score
        except Exception as e:
            print(f"Segmentation error: {e}")
            return None, 0.0

    def create_segmented_outputs(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Create multiple segmented outputs for ConceptCLIP"""
        outputs = {}
        
        # Colored overlay
        color = (255, 0, 0)  # Red highlight
        overlay = image.copy()
        overlay[mask == 1] = color
        colored_overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        outputs['colored_overlay'] = colored_overlay
        
        # Contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        outputs['contour'] = contour_image
        
        # Cropped
        masked = image.copy()
        masked[mask == 0] = 0
        y, x = np.where(mask == 1)
        if len(x) > 0 and len(y) > 0:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            cropped = image[y_min:y_max, x_min:x_max]
            cropped = cv2.resize(cropped, (512, 512))
            outputs['cropped'] = cropped
        
        # Masked only
        outputs['masked_only'] = masked
        
        # Side by side
        side_by_side = np.hstack((image, masked))
        outputs['side_by_side'] = cv2.resize(side_by_side, (512, 512))
        
        return outputs

    def classify_segmented_image(self, segmented_outputs: Dict[str, np.ndarray]) -> Dict:
        """Classify all segmented outputs using local ConceptCLIP"""
        try:
            results = {}
            for output_type, seg_image in segmented_outputs.items():
                if seg_image is None or seg_image.size == 0:
                    continue
                seg_pil = Image.fromarray(seg_image.astype(np.uint8))
                
                inputs = self.conceptclip_processor(
                    images=seg_pil, 
                    text=self.domain.text_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.conceptclip_model(**inputs)
                    
                    logit_scale = outputs.get("logit_scale", torch.tensor(1.0))
                    image_features = outputs["image_features"]
                    text_features = outputs["text_features"]
                    
                    logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)[0]
                
                disease_names = [prompt.split(' showing ')[-1] for prompt in self.domain.text_prompts]
                probabilities = {disease_names[i]: float(logits[i]) for i in range(len(disease_names))}
                results[output_type] = probabilities
            
            # Ensemble predictions
            if results:
                return self._ensemble_predictions(results)
            return {}
        
        except Exception as e:
            print(f"Classification error: {e}")
            return {}

    def _ensemble_predictions(self, predictions: Dict[str, Dict]) -> Dict:
        """Combine predictions from different segmented outputs"""
        try:
            weights = {
                'colored_overlay': 0.3,
                'contour': 0.2,
                'cropped': 0.25,
                'masked_only': 0.15,
                'side_by_side': 0.1
            }
            
            disease_names = [prompt.split(' showing ')[-1] for prompt in self.domain.text_prompts]
            ensemble_probs = {disease: 0.0 for disease in disease_names}
            
            total_weight = 0
            for output_type, probs in predictions.items():
                weight = weights.get(output_type, 0.1)
                total_weight += weight
                for disease, prob in probs.items():
                    ensemble_probs[disease] += prob * weight
            
            if total_weight > 0:
                ensemble_probs = {disease: prob / total_weight for disease, prob in ensemble_probs.items()}
            
            return ensemble_probs
        except Exception as e:
            print(f"Ensemble error: {e}")
            return {}

    def get_ground_truth_label(self, img_path: Path) -> str:
        """Load ground truth label for image"""
        try:
            df = pd.read_csv(self.groundtruth_path)
            img_name = img_path.stem
            matching_row = df[df['image_name'] == img_name]
            if not matching_row.empty:
                return matching_row.iloc[0]['label']
            return None
        except Exception as e:
            print(f"Error loading ground truth for {img_path}: {e}")
            return None

    def _save_results(self, results: List[Dict], report: Dict):
        """Save processing results and report"""
        try:
            # Save detailed results
            results_df = pd.DataFrame(results)
            self.output_path.mkdir(exist_ok=True)
            results_path = self.output_path / "reports" / "detailed_results.csv"
            results_path.parent.mkdir(exist_ok=True)
            results_df.to_csv(results_path, index=False)
            
            # Save comprehensive report
            report_path = self.output_path / "reports" / "processing_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            # Create summary plots
            self._create_summary_plots(results)
            
        except Exception as e:
            print(f"Error saving results: {e}")

    def _create_summary_plots(self, results: List[Dict]):
        """Create summary visualization plots"""
        try:
            vis_path = self.output_path / "visualizations"
            vis_path.mkdir(exist_ok=True)
            
            # Prediction distribution
            predictions = [r['predicted_disease'] for r in results]
            plt.figure(figsize=(10, 6))
            sns.countplot(x=predictions)
            plt.title("Prediction Distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(vis_path / "prediction_distribution.png")
            plt.close()
            
            # Confidence distributions
            seg_confidences = [r['segmentation_confidence'] for r in results]
            pred_confidences = [r['prediction_confidence'] for r in results]
            plt.figure(figsize=(10, 6))
            plt.hist(seg_confidences, bins=20, alpha=0.5, label='Segmentation Confidence')
            plt.hist(pred_confidences, bins=20, alpha=0.5, label='Prediction Confidence')
            plt.legend()
            plt.title("Confidence Distributions")
            plt.tight_layout()
            plt.savefig(vis_path / "confidence_distributions.png")
            plt.close()
            
        except Exception as e:
            print(f"Error creating plots: {e}")

    def _generate_comprehensive_report(self, results: List[Dict], format_counter: Counter, 
                                     accuracy: float, total_with_gt: int) -> Dict:
        """Generate comprehensive processing report with additional metrics"""
        total_processed = len(results)
        successful_segmentations = sum(1 for r in results if r['segmentation_confidence'] > 0.5)
        successful_classifications = sum(1 for r in results if r['prediction_confidence'] > 0.1)
        predictions = [r['predicted_disease'] for r in results]
        prediction_counts = Counter(predictions)
        seg_confidences = [r['segmentation_confidence'] for r in results]
        pred_confidences = [r['prediction_confidence'] for r in results]
        device_used = results[0]['device_used'] if results else "unknown"
        cache_used = results[0]['cache_used'] if results else "unknown"

        # Calculate additional metrics
        if total_with_gt > 0:
            true_labels = [r['ground_truth'] for r in results if r['ground_truth'] is not None]
            pred_labels = [r['predicted_disease'] for r in results if r['ground_truth'] is not None]
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
        else:
            precision, recall, f1 = 0, 0, 0

        report = {
            'system_info': {
                'device_used': device_used,
                'cache_directory': cache_used,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'offline_mode': os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            },
            'dataset_info': {
                'total_images_found': total_processed,
                'file_formats': dict(format_counter),
                'total_with_ground_truth': total_with_gt
            },
            'processing_stats': {
                'successful_segmentations': successful_segmentations,
                'successful_classifications': successful_classifications,
                'segmentation_success_rate': successful_segmentations / total_processed if total_processed > 0 else 0,
                'classification_success_rate': successful_classifications / total_processed if total_processed > 0 else 0
            },
            'accuracy_metrics': {
                'overall_accuracy': accuracy,
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'correct_predictions': sum(1 for r in results if r['correct']),
                'total_evaluated': total_with_gt
            },
            'predictions': {
                'distribution': dict(prediction_counts),
                'most_common': prediction_counts.most_common(5)
            },
            'confidence_stats': {
                'segmentation': {
                    'mean': np.mean(seg_confidences) if seg_confidences else 0,
                    'std': np.std(seg_confidences) if seg_confidences else 0,
                    'min': np.min(seg_confidences) if seg_confidences else 0,
                    'max': np.max(seg_confidences) if seg_confidences else 0
                },
                'classification': {
                    'mean': np.mean(pred_confidences) if pred_confidences else 0,
                    'std': np.std(pred_confidences) if pred_confidences else 0,
                    'min': np.min(pred_confidences) if pred_confidences else 0,
                    'max': np.max(pred_confidences) if pred_confidences else 0
                }
            }
        }
        return report

    def process_dataset(self, test_mode: bool = False, max_images: int = 5) -> Dict:
        """Process entire MILK10k dataset or a subset in test mode"""
        print("Starting MILK10k dataset processing...")
        
        # Find all images
        image_files = []
        for ext in self.domain.image_extensions:
            image_files.extend(self.dataset_path.rglob(f"*{ext}"))
        
        if test_mode:
            image_files = image_files[:max_images]
            print(f"Test mode: Processing {len(image_files)} images")
        else:
            print(f"Found {len(image_files)} images in dataset")
        
        results = []
        format_counter = Counter()
        correct_predictions = 0
        total_with_gt = 0
        
        for img_path in tqdm(image_files, desc="Processing MILK10k images"):
            try:
                # Track file formats
                ext = img_path.suffix.lower()
                format_counter[ext] += 1
                
                # Load and preprocess image
                image = self.preprocess_image(img_path)
                if image is None:
                    continue
                
                # Segment image
                mask, seg_confidence = self.segment_image(image)
                
                # Create segmented outputs for ConceptCLIP
                segmented_outputs = self.create_segmented_outputs(image, mask)
                
                # Save segmented outputs for ConceptCLIP input
                img_name = img_path.stem
                conceptclip_dir = self.output_path / "segmented_for_conceptclip" / img_name
                conceptclip_dir.mkdir(exist_ok=True)
                
                for output_type, seg_image in segmented_outputs.items():
                    if seg_image is not None:
                        output_path = conceptclip_dir / f"{output_type}.png"
                        cv2.imwrite(str(output_path), cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
                
                # Classify using ConceptCLIP
                classification_probs = self.classify_segmented_image(segmented_outputs)
                
                # Get ground truth
                ground_truth = self.get_ground_truth_label(img_path)
                
                # Get prediction
                if classification_probs:
                    predicted_disease = max(classification_probs, key=classification_probs.get)
                    prediction_confidence = classification_probs[predicted_disease]
                else:
                    predicted_disease = "unknown"
                    prediction_confidence = 0.0
                
                # Check accuracy if ground truth available
                if ground_truth:
                    total_with_gt += 1
                    if ground_truth == predicted_disease:
                        correct_predictions += 1
                
                # Save results
                result = {
                    'image_path': str(img_path),
                    'image_name': img_name,
                    'predicted_disease': predicted_disease,
                    'prediction_confidence': prediction_confidence,
                    'segmentation_confidence': seg_confidence,
                    'ground_truth': ground_truth,
                    'correct': ground_truth == predicted_disease if ground_truth else None,
                    'segmented_outputs_dir': str(conceptclip_dir),
                    'classification_probabilities': classification_probs,
                    'device_used': self.device,
                    'cache_used': str(self.cache_path)
                }
                
                results.append(result)
                
                # Progress indicator
                status = "✓" if result['correct'] else ("✗" if ground_truth else "-")
                print(f"{status} {img_name}: {predicted_disease} ({prediction_confidence:.2%}) [Device: {self.device}]")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate accuracy
        accuracy = correct_predictions / total_with_gt if total_with_gt > 0 else 0
        
        # Generate report
        report = self._generate_comprehensive_report(results, format_counter, accuracy, total_with_gt)
        
        # Save results
        self._save_results(results, report)
        
        return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
    GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_GroundTruth.csv"
    OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/outputs_ensemble"
    SAM2_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2"
    CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"
    HUGGINGFACE_CACHE_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache"
    
    pipeline = MILK10kPipeline(
        dataset_path=DATASET_PATH,
        groundtruth_path=GROUNDTRUTH_PATH,
        output_path=OUTPUT_PATH,
        sam2_model_path=SAM2_MODEL_PATH,
        conceptclip_model_path=CONCEPTCLIP_MODEL_PATH,
        cache_path=HUGGINGFACE_CACHE_PATH
    )
    pipeline.process_dataset(test_mode=args.test)