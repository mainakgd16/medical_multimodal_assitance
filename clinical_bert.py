"""
Clinical BERT integration for medical text embeddings and analysis
"""
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import re
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ClinicalBERTModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2'"):
        """
        Initialize Clinical BERT model for medical text processing
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load sentence transformer model optimized for clinical text
            # Using a general sentence transformer as placeholder for Bio_ClinicalBERT
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Medical knowledge base for similarity matching
            self.medical_knowledge_base = self._load_medical_knowledge_base()
            
            logger.info(f"Clinical BERT model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading Clinical BERT model: {e}")
            self.model = None
    
    def _load_medical_knowledge_base(self) -> Dict[str, List[str]]:
        """
        Load medical knowledge base for context matching
        """
        return {
            "symptoms": [
                "chest pain", "shortness of breath", "fever", "cough", "headache",
                "nausea", "vomiting", "diarrhea", "fatigue", "dizziness",
                "abdominal pain", "back pain", "joint pain", "rash", "swelling"
            ],
            "conditions": [
                "pneumonia", "hypertension", "diabetes", "asthma", "copd",
                "heart failure", "stroke", "myocardial infarction", "sepsis",
                "pneumothorax", "pulmonary embolism", "acute coronary syndrome"
            ],
            "medications": [
                "aspirin", "metformin", "lisinopril", "atorvastatin", "omeprazole",
                "albuterol", "furosemide", "warfarin", "insulin", "prednisone"
            ],
            "procedures": [
                "ecg", "chest x-ray", "ct scan", "mri", "blood test",
                "echocardiogram", "colonoscopy", "biopsy", "surgery", "intubation"
            ]
        }
    
    def encode_clinical_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode clinical text into embeddings
        """
        if self.model is None:
            raise ValueError("Model not loaded properly")
        
        try:
            # Preprocess clinical text
            processed_texts = [self._preprocess_clinical_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(processed_texts, convert_to_tensor=False)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error encoding clinical text: {e}")
            raise
    
    def _preprocess_clinical_text(self, text: str) -> str:
        """
        Preprocess clinical text for better embedding quality
        """
        # Convert to lowercase
        text = text.lower()
        
        # Expand common medical abbreviations
        abbreviations = {
            "bp": "blood pressure",
            "hr": "heart rate",
            "rr": "respiratory rate",
            "temp": "temperature",
            "o2 sat": "oxygen saturation",
            "wbc": "white blood cell count",
            "rbc": "red blood cell count",
            "hgb": "hemoglobin",
            "plt": "platelet count",
            "bun": "blood urea nitrogen",
            "cr": "creatinine",
            "na": "sodium",
            "k": "potassium",
            "cl": "chloride",
            "co2": "carbon dioxide"
        }
        
        for abbrev, full_form in abbreviations.items():
            text = re.sub(r'\b' + abbrev + r'\b', full_form, text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_prescription_text(self, prescription: str) -> Dict:
        """
        Analyze prescription text for key information extraction
        """
        try:
            # Extract key information using regex patterns
            medication_info = self._extract_medication_info(prescription)
            
            # Generate embedding for the prescription
            embedding = self.encode_clinical_text([prescription])[0]
            
            # Find similar medications in knowledge base
            similar_medications = self._find_similar_medications(prescription)
            
            # Analyze for potential issues
            safety_analysis = self._analyze_prescription_safety(prescription, medication_info)
            
            return {
                "prescription_text": prescription,
                "extracted_info": medication_info,
                "embedding": embedding.tolist(),
                "similar_medications": similar_medications,
                "safety_analysis": safety_analysis,
                "clinical_notes": self._generate_clinical_notes(medication_info)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prescription text: {e}")
            return {"error": f"Failed to analyze prescription: {str(e)}"}
    
    def _extract_medication_info(self, prescription: str) -> Dict:
        """
        Extract medication information from prescription text
        """
        info = {
            "medications": [],
            "dosages": [],
            "frequencies": [],
            "durations": [],
            "routes": []
        }
        
        # Medication name patterns
        med_patterns = [
            r'(\w+)\s+\d+\s*mg',
            r'(\w+)\s+\d+\s*mcg',
            r'(\w+)\s+\d+\s*units',
            r'(\w+)\s+tablet',
            r'(\w+)\s+capsule'
        ]
        
        for pattern in med_patterns:
            matches = re.findall(pattern, prescription, re.IGNORECASE)
            info["medications"].extend(matches)
        
        # Dosage patterns
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*mg',
            r'(\d+(?:\.\d+)?)\s*mcg',
            r'(\d+(?:\.\d+)?)\s*units',
            r'(\d+(?:\.\d+)?)\s*ml'
        ]
        
        for pattern in dosage_patterns:
            matches = re.findall(pattern, prescription, re.IGNORECASE)
            info["dosages"].extend(matches)
        
        # Frequency patterns
        frequency_patterns = [
            r'(once daily|twice daily|three times daily|four times daily)',
            r'(every \d+ hours?)',
            r'(bid|tid|qid|qd)',
            r'(as needed|prn)'
        ]
        
        for pattern in frequency_patterns:
            matches = re.findall(pattern, prescription, re.IGNORECASE)
            info["frequencies"].extend(matches)
        
        # Duration patterns
        duration_patterns = [
            r'for (\d+ days?)',
            r'for (\d+ weeks?)',
            r'for (\d+ months?)',
            r'(\d+ day supply)',
            r'(\d+ week supply)'
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, prescription, re.IGNORECASE)
            info["durations"].extend(matches)
        
        # Route patterns
        route_patterns = [
            r'(oral|po|by mouth)',
            r'(iv|intravenous)',
            r'(im|intramuscular)',
            r'(topical|external)',
            r'(sublingual|sl)'
        ]
        
        for pattern in route_patterns:
            matches = re.findall(pattern, prescription, re.IGNORECASE)
            info["routes"].extend(matches)
        
        return info
    
    def _find_similar_medications(self, prescription: str) -> List[Dict]:
        """
        Find similar medications using embedding similarity
        """
        try:
            prescription_embedding = self.encode_clinical_text([prescription])[0]
            medication_embeddings = self.encode_clinical_text(self.medical_knowledge_base["medications"])
            
            similarities = cosine_similarity([prescription_embedding], medication_embeddings)[0]
            
            similar_meds = []
            for i, similarity in enumerate(similarities):
                if similarity > 0.3:  # Threshold for similarity
                    similar_meds.append({
                        "medication": self.medical_knowledge_base["medications"][i],
                        "similarity": float(similarity)
                    })
            
            # Sort by similarity
            similar_meds.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similar_meds[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error finding similar medications: {e}")
            return []
    
    def _analyze_prescription_safety(self, prescription: str, medication_info: Dict) -> Dict:
        """
        Analyze prescription for safety concerns
        """
        safety_issues = []
        warnings = []
        
        # Check for high-risk medications
        high_risk_meds = ["warfarin", "insulin", "digoxin", "lithium", "phenytoin"]
        
        for med in medication_info["medications"]:
            if med.lower() in high_risk_meds:
                safety_issues.append(f"High-risk medication detected: {med}")
                warnings.append(f"Requires careful monitoring and dose adjustment for {med}")
        
        # Check for dosage concerns
        for dosage in medication_info["dosages"]:
            try:
                dose_value = float(dosage)
                if dose_value > 1000:  # Arbitrary high dose threshold
                    warnings.append(f"High dose detected: {dosage} - verify appropriateness")
            except ValueError:
                continue
        
        # Check for frequency issues
        for freq in medication_info["frequencies"]:
            if "as needed" in freq.lower() and len(medication_info["medications"]) > 1:
                warnings.append("PRN medication with other scheduled medications - review interactions")
        
        return {
            "safety_score": max(0, 10 - len(safety_issues) * 2 - len(warnings)),
            "safety_issues": safety_issues,
            "warnings": warnings,
            "requires_monitoring": len(safety_issues) > 0
        }
    
    def _generate_clinical_notes(self, medication_info: Dict) -> List[str]:
        """
        Generate clinical notes based on extracted medication information
        """
        notes = []
        
        if medication_info["medications"]:
            notes.append(f"Prescribed medications: {', '.join(set(medication_info['medications']))}")
        
        if medication_info["dosages"]:
            notes.append(f"Dosage information available for verification")
        
        if medication_info["frequencies"]:
            notes.append(f"Dosing frequency specified: {', '.join(set(medication_info['frequencies']))}")
        
        if medication_info["durations"]:
            notes.append(f"Treatment duration: {', '.join(set(medication_info['durations']))}")
        
        if not any(medication_info.values()):
            notes.append("Limited medication information extracted - manual review recommended")
        
        return notes
    
    def analyze_clinical_notes(self, clinical_text: str) -> Dict:
        """
        Analyze clinical notes for key medical information
        """
        try:
            # Generate embedding
            embedding = self.encode_clinical_text([clinical_text])[0]
            
            # Extract medical entities
            entities = self._extract_medical_entities(clinical_text)
            
            # Analyze sentiment/urgency
            urgency_analysis = self._analyze_clinical_urgency(clinical_text)
            
            # Find related conditions
            related_conditions = self._find_related_conditions(clinical_text)
            
            return {
                "clinical_text": clinical_text,
                "embedding": embedding.tolist(),
                "extracted_entities": entities,
                "urgency_analysis": urgency_analysis,
                "related_conditions": related_conditions,
                "summary": self._generate_clinical_summary(entities, urgency_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing clinical notes: {e}")
            return {"error": f"Failed to analyze clinical notes: {str(e)}"}
    
    def _extract_medical_entities(self, text: str) -> Dict:
        """
        Extract medical entities from clinical text
        """
        entities = {
            "symptoms": [],
            "conditions": [],
            "medications": [],
            "procedures": []
        }
        
        text_lower = text.lower()
        
        # Find entities in each category
        for category, items in self.medical_knowledge_base.items():
            for item in items:
                if item in text_lower:
                    entities[category].append(item)
        
        return entities
    
    def _analyze_clinical_urgency(self, text: str) -> Dict:
        """
        Analyze clinical urgency based on text content
        """
        urgent_keywords = [
            "emergency", "urgent", "stat", "immediate", "critical",
            "severe", "acute", "unstable", "deteriorating"
        ]
        
        moderate_keywords = [
            "concerning", "abnormal", "elevated", "decreased",
            "monitor", "follow-up", "reassess"
        ]
        
        text_lower = text.lower()
        
        urgent_count = sum(1 for keyword in urgent_keywords if keyword in text_lower)
        moderate_count = sum(1 for keyword in moderate_keywords if keyword in text_lower)
        
        if urgent_count > 0:
            urgency_level = "High"
        elif moderate_count > 0:
            urgency_level = "Moderate"
        else:
            urgency_level = "Low"
        
        return {
            "urgency_level": urgency_level,
            "urgent_indicators": urgent_count,
            "moderate_indicators": moderate_count,
            "recommendations": self._get_urgency_recommendations(urgency_level)
        }
    
    def _find_related_conditions(self, text: str) -> List[Dict]:
        """
        Find conditions related to the clinical text
        """
        try:
            text_embedding = self.encode_clinical_text([text])[0]
            condition_embeddings = self.encode_clinical_text(self.medical_knowledge_base["conditions"])
            
            similarities = cosine_similarity([text_embedding], condition_embeddings)[0]
            
            related = []
            for i, similarity in enumerate(similarities):
                if similarity > 0.2:  # Threshold for relevance
                    related.append({
                        "condition": self.medical_knowledge_base["conditions"][i],
                        "relevance": float(similarity)
                    })
            
            # Sort by relevance
            related.sort(key=lambda x: x["relevance"], reverse=True)
            
            return related[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error finding related conditions: {e}")
            return []
    
    def _get_urgency_recommendations(self, urgency_level: str) -> List[str]:
        """
        Get recommendations based on urgency level
        """
        if urgency_level == "High":
            return [
                "Immediate medical attention required",
                "Consider emergency department evaluation",
                "Notify attending physician immediately"
            ]
        elif urgency_level == "Moderate":
            return [
                "Schedule follow-up within 24-48 hours",
                "Monitor patient closely",
                "Consider additional diagnostic workup"
            ]
        else:
            return [
                "Routine follow-up as scheduled",
                "Continue current management plan",
                "Patient education and reassurance"
            ]
    
    def _generate_clinical_summary(self, entities: Dict, urgency_analysis: Dict) -> str:
        """
        Generate a clinical summary based on extracted information
        """
        summary_parts = []
        
        if entities["symptoms"]:
            summary_parts.append(f"Symptoms: {', '.join(entities['symptoms'])}")
        
        if entities["conditions"]:
            summary_parts.append(f"Conditions: {', '.join(entities['conditions'])}")
        
        if entities["medications"]:
            summary_parts.append(f"Medications: {', '.join(entities['medications'])}")
        
        if entities["procedures"]:
            summary_parts.append(f"Procedures: {', '.join(entities['procedures'])}")
        
        summary_parts.append(f"Clinical urgency: {urgency_analysis['urgency_level']}")
        
        return ". ".join(summary_parts) if summary_parts else "No significant clinical information extracted."
