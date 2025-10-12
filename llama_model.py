"""
Llama 3.2:1b integration via Ollama for natural language understanding
"""
import requests
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class LlamaModel:
    def __init__(self, model_name: str = "llama3.2:1b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def generate_response(self, prompt: str, context: Optional[str] = None, 
                         max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate response using Llama model via Ollama API
        """
        try:
            # Construct the full prompt with medical context
            full_prompt = self._construct_medical_prompt(prompt, context)
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: Unable to generate response. Please ensure Ollama is running with {self.model_name} model."
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {e}")
            return "Error: An unexpected error occurred during response generation."
    
    def _construct_medical_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Construct a medical-focused prompt with appropriate context
        """
        system_prompt = """You are an AI medical assistant designed to support healthcare professionals. 
        You provide evidence-based information, diagnostic suggestions, and clinical insights.
        Always emphasize that your responses are for informational purposes and should not replace professional medical judgment.
        
        Guidelines:
        - Provide clear, concise medical information
        - Include relevant differential diagnoses when appropriate
        - Suggest follow-up questions or tests when needed
        - Maintain professional medical terminology
        - Always include appropriate medical disclaimers
        """
        
        if context:
            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuery: {prompt}\n\nResponse:"
        else:
            full_prompt = f"{system_prompt}\n\nQuery: {prompt}\n\nResponse:"
            
        return full_prompt
    
    def analyze_prescription(self, prescription_text: str) -> Dict:
        """
        Analyze prescription for potential issues and recommendations
        """
        prompt = f"""
        Analyze the following prescription for:
        1. Drug interactions
        2. Dosage appropriateness
        3. Contraindications
        4. Alternative recommendations
        5. Patient safety considerations
        
        Prescription: {prescription_text}
        
        Provide a structured analysis with specific recommendations.
        """
        
        response = self.generate_response(prompt)
        
        return {
            "prescription_text": prescription_text,
            "analysis": response,
            "safety_score": self._calculate_safety_score(response),
            "recommendations": self._extract_recommendations(response)
        }
    
    def generate_triage_guidance(self, symptoms: str, patient_info: str) -> Dict:
        """
        Generate triage guidance based on symptoms and patient information
        """
        prompt = f"""
        Based on the following patient information and symptoms, provide triage guidance:
        
        Patient Information: {patient_info}
        Symptoms: {symptoms}
        
        Please provide:
        1. Urgency level (Emergency, Urgent, Semi-urgent, Non-urgent)
        2. Recommended timeline for medical attention
        3. Key warning signs to monitor
        4. Initial assessment recommendations
        5. Potential differential diagnoses
        """
        
        response = self.generate_response(prompt)
        
        return {
            "patient_info": patient_info,
            "symptoms": symptoms,
            "triage_guidance": response,
            "urgency_level": self._extract_urgency_level(response),
            "timeline": self._extract_timeline(response)
        }
    
    def _calculate_safety_score(self, analysis: str) -> float:
        """
        Calculate a safety score based on the analysis content
        """
        # Simple scoring based on keywords (in a real system, this would be more sophisticated)
        risk_keywords = ["contraindicated", "dangerous", "avoid", "warning", "serious", "adverse"]
        safe_keywords = ["appropriate", "safe", "recommended", "suitable", "normal"]
        
        risk_count = sum(1 for keyword in risk_keywords if keyword.lower() in analysis.lower())
        safe_count = sum(1 for keyword in safe_keywords if keyword.lower() in analysis.lower())
        
        # Score from 0-10, where 10 is safest
        base_score = 7.0
        score = base_score + safe_count * 0.5 - risk_count * 1.0
        return max(0.0, min(10.0, score))
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """
        Extract key recommendations from the analysis
        """
        # Simple extraction (in a real system, this would use NLP)
        lines = analysis.split('\n')
        recommendations = []
        
        for line in lines:
            if any(word in line.lower() for word in ["recommend", "suggest", "consider", "should"]):
                recommendations.append(line.strip())
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _extract_urgency_level(self, guidance: str) -> str:
        """
        Extract urgency level from triage guidance
        """
        urgency_levels = ["Emergency", "Urgent", "Semi-urgent", "Non-urgent"]
        
        for level in urgency_levels:
            if level.lower() in guidance.lower():
                return level
        
        return "Semi-urgent"  # Default
    
    def _extract_timeline(self, guidance: str) -> str:
        """
        Extract recommended timeline from triage guidance
        """
        # Simple extraction of time-related phrases
        import re
        
        time_patterns = [
            r"within \d+ hours?",
            r"within \d+ days?",
            r"immediately",
            r"urgent",
            r"as soon as possible"
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, guidance, re.IGNORECASE)
            if match:
                return match.group()
        
        return "Within 24 hours"  # Default
    
    def check_model_availability(self) -> bool:
        """
        Check if the Ollama model is available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            return self.model_name in available_models
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
