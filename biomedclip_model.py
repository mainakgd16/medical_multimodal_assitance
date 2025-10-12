"""
BiomedCLIP integration for medical image classification
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
import cv2

logger = logging.getLogger(__name__)

class MedCLIPModel:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        """
        Initialize BiomedCLIP model for medical image analysis
        Note: Using a placeholder implementation as BiomedCLIP might need specific setup
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # For this implementation, we'll use a general CLIP model as a base
            # In a real deployment, you'd use the actual BiomedCLIP model
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            # In biomedclip_model.py, replace lines 27-28:
            self.model.to(self.device)
            self.model.eval()
            
            # Medical image classification labels
            self.medical_labels = [
                "chest x-ray normal",
                "chest x-ray pneumonia",
                "chest x-ray covid-19",
                "chest x-ray tuberculosis",
                "chest x-ray lung cancer",
                "brain mri normal",
                "brain mri tumor",
                "brain mri stroke",
                "brain mri alzheimer",
                "ct scan normal",
                "ct scan fracture",
                "ct scan hemorrhage",
                "histopathology benign",
                "histopathology malignant",
                "retinal image normal",
                "retinal image diabetic retinopathy",
                "skin lesion benign",
                "skin lesion melanoma",
                "ultrasound normal",
                "ultrasound abnormal"
            ]
            
            logger.info(f"BiomedCLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading BiomedCLIP model: {e}")
            self.model = None
            self.processor = None
    
    def analyze_medical_image(self, image_path: str, image_type: str = "unknown") -> Dict:
        """
        Analyze medical image and provide classification results
        """
        if self.model is None:
            return {"error": "Model not loaded properly"}
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Get image features
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = F.normalize(image_features, p=2, dim=1)
            
            # Get text features for medical labels
            text_inputs = self.processor(text=self.medical_labels, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
            
            # Calculate similarities
            similarities = torch.matmul(image_features, text_features.T)
            probabilities = F.softmax(similarities, dim=1)
            
            # Get top predictions
            top_k = 5
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            predictions = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                label = self.medical_labels[idx]
                predictions.append({
                    "label": label,
                    "confidence": prob,
                    "category": self._categorize_label(label)
                })
            
            # Analyze image characteristics
            image_analysis = self._analyze_image_characteristics(image_path)
            
            # Generate clinical insights
            clinical_insights = self._generate_clinical_insights(predictions, image_type)
            
            return {
                "image_path": image_path,
                "image_type": image_type,
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None,
                "image_analysis": image_analysis,
                "clinical_insights": clinical_insights,
                "recommendations": self._generate_recommendations(predictions, image_type)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing medical image: {e}")
            return {"error": f"Failed to analyze image: {str(e)}"}
    
    def _categorize_label(self, label: str) -> str:
        """
        Categorize medical labels into broader categories
        """
        if "x-ray" in label or "chest" in label:
            return "Chest X-ray"
        elif "mri" in label or "brain" in label:
            return "Brain MRI"
        elif "ct" in label:
            return "CT Scan"
        elif "histopathology" in label:
            return "Histopathology"
        elif "retinal" in label:
            return "Ophthalmology"
        elif "skin" in label:
            return "Dermatology"
        elif "ultrasound" in label:
            return "Ultrasound"
        else:
            return "General Medical"
    
    def _analyze_image_characteristics(self, image_path: str) -> Dict:
        """
        Analyze basic image characteristics
        """
        try:
            # Load image with OpenCV for analysis
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}
            
            # Basic image properties
            height, width, channels = img.shape
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate image statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Simple contrast analysis
            contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            # Edge detection for structure analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            return {
                "dimensions": {"width": width, "height": height, "channels": channels},
                "intensity_stats": {
                    "mean": float(mean_intensity),
                    "std": float(std_intensity),
                    "contrast_ratio": float(contrast)
                },
                "structure_analysis": {
                    "edge_density": float(edge_density),
                    "image_quality": "good" if contrast > 0.3 else "low_contrast"
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image characteristics: {e}")
            return {"error": "Failed to analyze image characteristics"}
    
    def _generate_clinical_insights(self, predictions: List[Dict], image_type: str) -> List[str]:
        """
        Generate clinical insights based on predictions
        """
        insights = []
        
        if not predictions:
            return ["No clear medical findings detected in the image."]
        
        top_prediction = predictions[0]
        confidence = top_prediction["confidence"]
        label = top_prediction["label"]
        
        # Confidence-based insights
        if confidence > 0.8:
            insights.append(f"High confidence detection of {label} (confidence: {confidence:.2f})")
        elif confidence > 0.6:
            insights.append(f"Moderate confidence detection of {label} (confidence: {confidence:.2f})")
        else:
            insights.append(f"Low confidence detection. Multiple possibilities should be considered.")
        
        # Category-specific insights
        category = top_prediction["category"]
        
        if category == "Chest X-ray":
            if "normal" in label:
                insights.append("Chest X-ray appears within normal limits.")
            elif "pneumonia" in label:
                insights.append("Possible pneumonia detected. Consider clinical correlation and additional imaging.")
            elif "covid" in label:
                insights.append("COVID-19 related changes possible. Recommend RT-PCR testing and isolation protocols.")
        
        elif category == "Brain MRI":
            if "tumor" in label:
                insights.append("Possible brain lesion detected. Recommend neurology consultation and contrast imaging.")
            elif "stroke" in label:
                insights.append("Possible stroke changes. Time-sensitive condition requiring immediate evaluation.")
        
        # Add general recommendations
        insights.append("AI analysis should be correlated with clinical findings and patient history.")
        insights.append("Consider additional imaging or specialist consultation if clinically indicated.")
        
        return insights
    
    def _generate_recommendations(self, predictions: List[Dict], image_type: str) -> List[str]:
        """
        Generate clinical recommendations based on analysis
        """
        recommendations = []
        
        if not predictions:
            return ["Image quality assessment recommended", "Consider repeat imaging if clinically indicated"]
        
        top_prediction = predictions[0]
        label = top_prediction["label"]
        confidence = top_prediction["confidence"]
        
        # Confidence-based recommendations
        if confidence < 0.5:
            recommendations.append("Low confidence result - recommend expert radiologist review")
            recommendations.append("Consider additional imaging modalities for better characterization")
        
        # Condition-specific recommendations
        if "pneumonia" in label:
            recommendations.extend([
                "Consider blood work including CBC with differential",
                "Evaluate for bacterial vs viral etiology",
                "Monitor oxygen saturation and respiratory status"
            ])
        
        elif "tumor" in label or "malignant" in label:
            recommendations.extend([
                "Urgent specialist referral recommended",
                "Consider biopsy for tissue diagnosis",
                "Staging workup if malignancy confirmed"
            ])
        
        elif "fracture" in label:
            recommendations.extend([
                "Orthopedic evaluation recommended",
                "Assess for associated injuries",
                "Consider CT for complex fractures"
            ])
        
        # General recommendations
        recommendations.extend([
            "Correlate with clinical presentation and patient history",
            "Document findings in patient medical record",
            "Follow institutional protocols for critical findings"
        ])
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def batch_analyze_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Analyze multiple medical images in batch
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.analyze_medical_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": f"Failed to process: {str(e)}"
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "available_labels": len(self.medical_labels),
            "model_loaded": self.model is not None
        }
