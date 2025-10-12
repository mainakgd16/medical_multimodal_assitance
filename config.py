"""
Configuration settings for the AI-Powered Medical Assistant
"""
import os
from typing import Dict, Any

class Config:
    """
    Configuration class for the medical assistant application
    """
    
    # Model configurations
    LLAMA_MODEL_NAME = "llama3.2:1b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    BIOMEDCLIP_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    CLINICAL_BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    
    # Fallback models (more widely available)
    FALLBACK_CLIP_MODEL = "openai/clip-vit-base-patch16"
    FALLBACK_BERT_MODEL = "all-MiniLM-L6-v2"
    
    # Directory configurations
    DATA_DIR = "data"
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    OUTPUTS_DIR = "outputs"
    LOGS_DIR = "logs"
    
    # Processing configurations
    MAX_IMAGE_SIZE = (1024, 1024)
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    # AI model parameters
    LLAMA_TEMPERATURE = 0.7
    LLAMA_MAX_TOKENS = 1000
    CONFIDENCE_THRESHOLD = 0.5
    
    # Medical classification labels
    MEDICAL_IMAGE_LABELS = [
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
    
    # Medical knowledge base
    MEDICAL_SYMPTOMS = [
        "chest pain", "shortness of breath", "fever", "cough", "headache",
        "nausea", "vomiting", "diarrhea", "fatigue", "dizziness",
        "abdominal pain", "back pain", "joint pain", "rash", "swelling"
    ]
    
    MEDICAL_CONDITIONS = [
        "pneumonia", "hypertension", "diabetes", "asthma", "copd",
        "heart failure", "stroke", "myocardial infarction", "sepsis",
        "pneumothorax", "pulmonary embolism", "acute coronary syndrome"
    ]
    
    COMMON_MEDICATIONS = [
        "aspirin", "metformin", "lisinopril", "atorvastatin", "omeprazole",
        "albuterol", "furosemide", "warfarin", "insulin", "prednisone"
    ]
    
    # Risk assessment thresholds
    RISK_THRESHOLDS = {
        "high": 8.0,
        "moderate": 6.0,
        "low": 4.0
    }
    
    # Urgency levels
    URGENCY_LEVELS = ["Emergency", "Urgent", "Semi-urgent", "Non-urgent"]
    
    # Streamlit configuration
    STREAMLIT_CONFIG = {
        "page_title": "AI Medical Assistant",
        "page_icon": "🏥",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """
        Get model configuration dictionary
        """
        return {
            "llama": {
                "model_name": cls.LLAMA_MODEL_NAME,
                "base_url": cls.OLLAMA_BASE_URL,
                "temperature": cls.LLAMA_TEMPERATURE,
                "max_tokens": cls.LLAMA_MAX_TOKENS
            },
            "biomedclip": {
                "model_name": cls.BIOMEDCLIP_MODEL_NAME,
                "fallback_model": cls.FALLBACK_CLIP_MODEL,
                "labels": cls.MEDICAL_IMAGE_LABELS
            },
            "clinical_bert": {
                "model_name": cls.CLINICAL_BERT_MODEL_NAME,
                "fallback_model": cls.FALLBACK_BERT_MODEL,
                "knowledge_base": {
                    "symptoms": cls.MEDICAL_SYMPTOMS,
                    "conditions": cls.MEDICAL_CONDITIONS,
                    "medications": cls.COMMON_MEDICATIONS
                }
            }
        }
    
    @classmethod
    def get_directories(cls) -> Dict[str, str]:
        """
        Get directory configuration
        """
        return {
            "data": cls.DATA_DIR,
            "images": cls.IMAGES_DIR,
            "outputs": cls.OUTPUTS_DIR,
            "logs": cls.LOGS_DIR
        }
    
    @classmethod
    def create_directories(cls) -> None:
        """
        Create necessary directories if they don't exist
        """
        directories = cls.get_directories()
        
        for dir_name, dir_path in directories.items():
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def get_medical_disclaimer(cls) -> str:
        """
        Get the medical disclaimer text
        """
        return """
        ⚠️ MEDICAL DISCLAIMER: This AI system is designed for educational and research purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of qualified healthcare professionals with any questions regarding medical conditions. 
        Never disregard professional medical advice or delay seeking it because of information provided by this AI system.
        
        The AI-generated responses are based on pattern recognition and should be validated by clinical expertise. 
        This system does not replace the clinical judgment of healthcare professionals.
        """

# Environment-specific configurations
class DevelopmentConfig(Config):
    """
    Development environment configuration
    """
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """
    Production environment configuration
    """
    DEBUG = False
    LOG_LEVEL = "INFO"

# Default configuration
config = DevelopmentConfig()
