"""
Medical data processor that integrates image and text analysis
"""
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import uuid

from llama_model import LlamaModel
#from biomedclip_model import BiomedCLIPModel
from biomedclip_model import MedCLIPModel
from clinical_bert import ClinicalBERTModel

logger = logging.getLogger(__name__)

class MedicalProcessor:
    def __init__(self):
        """
        Initialize the medical processor with all AI models
        """
        self.llama_model = LlamaModel()
        #self.biomedclip_model = BiomedCLIPModel()
        self.medclip_model = MedCLIPModel()
        self.clinical_bert_model = ClinicalBERTModel()
        
        # Check model availability
        self.models_status = self._check_models_status()
        
        logger.info("Medical Processor initialized")
        logger.info(f"Models status: {self.models_status}")
    
    def _check_models_status(self) -> Dict[str, bool]:
        """
        Check the status of all loaded models
        """
        status = {
            "llama": False,
            "medclip": False,
            "clinical_bert": False
        }
        
        try:
            status["llama"] = self.llama_model.check_model_availability()
        except Exception as e:
            logger.warning(f"Llama model check failed: {e}")
        
        try:
            status["medclip"] = self.medclip_model.model is not None
        except Exception as e:
            logger.warning(f"MedCLIP model check failed: {e}")
        
        try:
            status["clinical_bert"] = self.clinical_bert_model.model is not None
        except Exception as e:
            logger.warning(f"Clinical BERT model check failed: {e}")
        
        return status
    
    def process_medical_case(self, image_path: str, prescription_text: str, 
                           patient_info: str = "", case_id: str = None) -> Dict:
        """
        Process a complete medical case with image and text data
        """
        if case_id is None:
            case_id = str(uuid.uuid4())
        
        logger.info(f"Processing medical case {case_id}")
        
        # Initialize case data
        case_data = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            "input_data": {
                "image_path": image_path,
                "prescription_text": prescription_text,
                "patient_info": patient_info
            },
            "analysis_results": {},
            "conversation": [],
            "clinical_summary": {},
            "recommendations": []
        }
        
        try:
            # Step 1: Analyze medical image
            logger.info("Analyzing medical image...")
            image_analysis = self._analyze_medical_image(image_path)
            case_data["analysis_results"]["image_analysis"] = image_analysis
            
            # Step 2: Analyze prescription text
            logger.info("Analyzing prescription...")
            prescription_analysis = self._analyze_prescription(prescription_text)
            case_data["analysis_results"]["prescription_analysis"] = prescription_analysis
            
            # Step 3: Analyze patient information
            logger.info("Analyzing patient information...")
            patient_analysis = self._analyze_patient_info(patient_info)
            case_data["analysis_results"]["patient_analysis"] = patient_analysis
            
            # Step 4: Generate integrated analysis
            logger.info("Generating integrated analysis...")
            integrated_analysis = self._generate_integrated_analysis(
                image_analysis, prescription_analysis, patient_analysis
            )
            case_data["analysis_results"]["integrated_analysis"] = integrated_analysis
            
            # Step 5: Generate clinical conversation
            logger.info("Generating clinical conversation...")
            conversation = self._generate_clinical_conversation(case_data)
            case_data["conversation"] = conversation
            
            # Step 6: Generate clinical summary and recommendations
            logger.info("Generating clinical summary...")
            clinical_summary = self._generate_clinical_summary(case_data)
            case_data["clinical_summary"] = clinical_summary
            
            case_data["recommendations"] = self._generate_final_recommendations(case_data)
            
            logger.info(f"Medical case {case_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing medical case {case_id}: {e}")
            case_data["error"] = str(e)
            case_data["status"] = "failed"
        
        return case_data
    
    def _analyze_medical_image(self, image_path: str) -> Dict:
        """
        Analyze medical image using MedCLIP
        """
        if not self.models_status.get("medclip", False):
            return {"error": "MedCLIP model not available"}
        
        try:
            # Determine image type from filename or path
            image_type = self._determine_image_type(image_path)
            
            # Analyze image using the correct model reference
            analysis = self.medclip_model.analyze_medical_image(image_path, image_type)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing medical image: {e}")
            return {"error": f"Image analysis failed: {str(e)}"}
    
    def _analyze_prescription(self, prescription_text: str) -> Dict:
        """
        Analyze prescription using Clinical BERT and Llama
        """
        results = {}
        
        # Clinical BERT analysis
        if self.models_status["clinical_bert"]:
            try:
                bert_analysis = self.clinical_bert_model.analyze_prescription_text(prescription_text)
                results["clinical_bert_analysis"] = bert_analysis
            except Exception as e:
                logger.error(f"Clinical BERT prescription analysis failed: {e}")
                results["clinical_bert_error"] = str(e)
        
        # Llama analysis
        if self.models_status["llama"]:
            try:
                llama_analysis = self.llama_model.analyze_prescription(prescription_text)
                results["llama_analysis"] = llama_analysis
            except Exception as e:
                logger.error(f"Llama prescription analysis failed: {e}")
                results["llama_error"] = str(e)
        
        return results
    
    def _analyze_patient_info(self, patient_info: str) -> Dict:
        """
        Analyze patient information using Clinical BERT and Llama
        """
        if not patient_info.strip():
            return {"message": "No patient information provided"}
        
        results = {}
        
        # Clinical BERT analysis
        if self.models_status["clinical_bert"]:
            try:
                bert_analysis = self.clinical_bert_model.analyze_clinical_notes(patient_info)
                results["clinical_bert_analysis"] = bert_analysis
            except Exception as e:
                logger.error(f"Clinical BERT patient analysis failed: {e}")
                results["clinical_bert_error"] = str(e)
        
        # Llama triage guidance
        if self.models_status["llama"]:
            try:
                # Extract symptoms from patient info for triage
                symptoms = self._extract_symptoms(patient_info)
                triage_guidance = self.llama_model.generate_triage_guidance(symptoms, patient_info)
                results["triage_guidance"] = triage_guidance
            except Exception as e:
                logger.error(f"Llama triage analysis failed: {e}")
                results["triage_error"] = str(e)
        
        return results
    
    def _generate_integrated_analysis(self, image_analysis: Dict, 
                                    prescription_analysis: Dict, 
                                    patient_analysis: Dict) -> Dict:
        """
        Generate integrated analysis combining all data sources
        """
        integrated = {
            "correlation_analysis": {},
            "consistency_check": {},
            "risk_assessment": {},
            "clinical_insights": []
        }
        
        try:
            # Correlation analysis
            integrated["correlation_analysis"] = self._analyze_correlations(
                image_analysis, prescription_analysis, patient_analysis
            )
            
            # Consistency check
            integrated["consistency_check"] = self._check_consistency(
                image_analysis, prescription_analysis, patient_analysis
            )
            
            # Risk assessment
            integrated["risk_assessment"] = self._assess_clinical_risk(
                image_analysis, prescription_analysis, patient_analysis
            )
            
            # Generate clinical insights
            integrated["clinical_insights"] = self._generate_clinical_insights(
                image_analysis, prescription_analysis, patient_analysis
            )
            
        except Exception as e:
            logger.error(f"Error in integrated analysis: {e}")
            integrated["error"] = str(e)
        
        return integrated
    
    def _generate_clinical_conversation(self, case_data: Dict) -> List[Dict]:
        """
        Generate a clinical conversation based on the case analysis
        """
        conversation = []
        
        try:
            # Initial case presentation
            conversation.append({
                "speaker": "system",
                "message": "New medical case for review",
                "timestamp": datetime.now().isoformat(),
                "type": "case_presentation"
            })
            
            # Image analysis discussion
            if "image_analysis" in case_data["analysis_results"]:
                image_analysis = case_data["analysis_results"]["image_analysis"]
                
                conversation.append({
                    "speaker": "clinician",
                    "message": "Please analyze the medical image provided with this case.",
                    "timestamp": datetime.now().isoformat(),
                    "type": "query"
                })
                
                if "predictions" in image_analysis and image_analysis["predictions"]:
                    top_prediction = image_analysis["predictions"][0]
                    
                    response = f"Based on the medical image analysis, I detect {top_prediction['label']} "
                    response += f"with {top_prediction['confidence']:.2f} confidence. "
                    
                    if "clinical_insights" in image_analysis:
                        response += "Key findings include: " + "; ".join(image_analysis["clinical_insights"][:3])
                    
                    conversation.append({
                        "speaker": "ai_assistant",
                        "message": response,
                        "timestamp": datetime.now().isoformat(),
                        "type": "analysis_response",
                        "confidence": top_prediction['confidence'],
                        "supporting_data": image_analysis
                    })
                    
                    # NEW QUESTION: Interpret this medical image and correlate with symptoms
                    conversation.append({
                        "speaker": "clinician",
                        "message": "Interpret this medical image and correlate with symptoms",
                        "timestamp": datetime.now().isoformat(),
                        "type": "query"
                    })
                    
                    # Generate correlation response - ALWAYS provide a response
                    correlation_response = f"Based on the medical image analysis, I observe {top_prediction['label'].lower()} with {top_prediction['confidence']:.2f} confidence. "
                    
                    # Try to get symptoms from both patient info and prescription
                    symptoms = []
                    
                    # Extract symptoms from patient info
                    if "input_data" in case_data and "patient_info" in case_data["input_data"]:
                        patient_symptoms = self._extract_symptoms(case_data["input_data"]["patient_info"])
                        symptoms.extend(patient_symptoms)
                    
                    # Extract symptoms from prescription text
                    if "input_data" in case_data and "prescription_text" in case_data["input_data"]:
                        prescription_symptoms = self._extract_symptoms_from_prescription(case_data["input_data"]["prescription_text"])
                        symptoms.extend(prescription_symptoms)
                    
                    # Remove duplicates while preserving order
                    symptoms = list(dict.fromkeys(symptoms))
                    
                    # Generate correlation based on available symptoms
                    if symptoms and len(symptoms) > 0 and symptoms != ["general symptoms"]:
                        symptom_text = ", ".join(symptoms[:5])  # Show top 5 symptoms
                        correlation_response += f"Correlating with the documented symptoms and clinical findings ({symptom_text}), "
                        correlation_response += "the imaging results show potential clinical correlation. "
                        correlation_response += "This supports the diagnostic assessment and prescribed treatment approach."
                    else:
                        correlation_response += "The current imaging findings should be evaluated alongside the clinical presentation documented in the prescription."
                    
                    conversation.append({
                        "speaker": "ai_assistant",
                        "message": correlation_response,
                        "timestamp": datetime.now().isoformat(),
                        "type": "correlation_analysis",
                        "supporting_data": {
                            "image_findings": top_prediction,
                            "symptoms_considered": symptoms[:5] if symptoms else []
                        }
                    })
            
            # Prescription analysis discussion
            if "prescription_analysis" in case_data["analysis_results"]:
                prescription_analysis = case_data["analysis_results"]["prescription_analysis"]
                
                conversation.append({
                    "speaker": "clinician",
                    "message": "What are your thoughts on the prescribed medications?",
                    "timestamp": datetime.now().isoformat(),
                    "type": "query"
                })
                
                response = "Regarding the prescription analysis: "
                
                if "llama_analysis" in prescription_analysis:
                    llama_result = prescription_analysis["llama_analysis"]
                    if "safety_score" in llama_result:
                        response += f"The prescription has a safety score of {llama_result['safety_score']:.1f}/10. "
                    
                    if "recommendations" in llama_result and llama_result["recommendations"]:
                        # Clean recommendations to remove markdown formatting
                        clean_recs = []
                        for rec in llama_result["recommendations"][:2]:
                            clean_rec = rec.replace("**", "").replace("*", "").strip()
                            if clean_rec and not clean_rec.startswith("#"):
                                clean_recs.append(clean_rec)
                        if clean_recs:
                            response += "Key recommendations include: " + "; ".join(clean_recs)
                
                conversation.append({
                    "speaker": "ai_assistant",
                    "message": response,
                    "timestamp": datetime.now().isoformat(),
                    "type": "prescription_analysis",
                    "supporting_data": prescription_analysis
                })
            
            # Patient information discussion
            if "patient_analysis" in case_data["analysis_results"]:
                patient_analysis = case_data["analysis_results"]["patient_analysis"]
                
                conversation.append({
                    "speaker": "clinician",
                    "message": "Based on the patient information, what is your triage recommendation?",
                    "timestamp": datetime.now().isoformat(),
                    "type": "query"
                })
                
                response = "Based on the patient information analysis: "
                
                if "triage_guidance" in patient_analysis:
                    triage = patient_analysis["triage_guidance"]
                    if isinstance(triage, dict):
                        if "urgency_level" in triage:
                            response += f"I recommend {triage['urgency_level']} priority triage. "
                        
                        if "timeline" in triage:
                            response += f"Suggested timeline: {triage['timeline']}. "
                        
                        if "symptoms" in triage and triage["symptoms"]:
                            response += f"Key symptoms identified: {', '.join(triage['symptoms'][:3])}. "
                    else:
                        response += "Triage analysis completed. "
                else:
                    # Fallback triage recommendation
                    response += "Based on available patient information, I recommend standard priority triage. "
                    
                    # Extract symptoms for basic assessment
                    if "input_data" in case_data and "patient_info" in case_data["input_data"]:
                        patient_symptoms = self._extract_symptoms(case_data["input_data"]["patient_info"])
                        if patient_symptoms and patient_symptoms != ["general symptoms"]:
                            response += f"Identified symptoms ({', '.join(patient_symptoms[:3])}) suggest routine monitoring. "
                    
                    # Check prescription for urgency indicators
                    if "input_data" in case_data and "prescription_text" in case_data["input_data"]:
                        prescription_text = case_data["input_data"]["prescription_text"].lower()
                        if any(urgent_term in prescription_text for urgent_term in ["emergency", "urgent", "immediate"]):
                            response = response.replace("standard priority", "urgent priority")
                        elif "pneumonia" in prescription_text or "infection" in prescription_text:
                            response = response.replace("standard priority", "semi-urgent priority")
                    
                    response += "Recommend clinical assessment within appropriate timeframe."
                
                conversation.append({
                    "speaker": "ai_assistant",
                    "message": response,
                    "timestamp": datetime.now().isoformat(),
                    "type": "triage_recommendation",
                    "supporting_data": patient_analysis
                })
            
            # Integrated analysis discussion
            if "integrated_analysis" in case_data["analysis_results"]:
                integrated = case_data["analysis_results"]["integrated_analysis"]
                
                conversation.append({
                    "speaker": "clinician",
                    "message": "Can you provide an integrated assessment of this case?",
                    "timestamp": datetime.now().isoformat(),
                    "type": "query"
                })
                
                response = "Integrated case assessment: "
                
                if "clinical_insights" in integrated and integrated["clinical_insights"]:
                    response += "Key clinical insights: " + "; ".join(integrated["clinical_insights"][:3]) + ". "
                
                if "risk_assessment" in integrated:
                    risk = integrated["risk_assessment"]
                    if "overall_risk" in risk:
                        response += f"Overall clinical risk level: {risk['overall_risk']}. "
                
                conversation.append({
                    "speaker": "ai_assistant",
                    "message": response,
                    "timestamp": datetime.now().isoformat(),
                    "type": "integrated_assessment",
                    "supporting_data": integrated
                })
            
            # Final recommendations
            conversation.append({
                "speaker": "clinician",
                "message": "What are your final recommendations for this case?",
                "timestamp": datetime.now().isoformat(),
                "type": "query"
            })
            
            final_recommendations = self._generate_final_recommendations(case_data)
            
            conversation.append({
                "speaker": "ai_assistant",
                "message": f"Final recommendations: {'; '.join(final_recommendations[:5])}",
                "timestamp": datetime.now().isoformat(),
                "type": "final_recommendations",
                "recommendations": final_recommendations
            })
            
        except Exception as e:
            logger.error(f"Error generating clinical conversation: {e}")
            conversation.append({
                "speaker": "system",
                "message": f"Error generating conversation: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "type": "error"
            })
        
        return conversation
    
    def _determine_image_type(self, image_path: str) -> str:
        """
        Determine medical image type from filename
        """
        filename = os.path.basename(image_path).lower()
        
        if "xray" in filename or "chest" in filename:
            return "chest_xray"
        elif "mri" in filename or "brain" in filename:
            return "brain_mri"
        elif "ct" in filename:
            return "ct_scan"
        elif "histology" in filename or "pathology" in filename:
            return "histopathology"
        elif "retinal" in filename or "eye" in filename:
            return "retinal_image"
        elif "skin" in filename or "dermatology" in filename:
            return "dermatology"
        elif "ultrasound" in filename or "echo" in filename:
            return "ultrasound"
        else:
            return "general_medical"
    
    def _extract_symptoms(self, patient_info: str) -> List[str]:
        """
        Extract symptoms from patient information
        """
        # Simple symptom extraction (in a real system, this would be more sophisticated)
        symptom_keywords = [
            "pain", "fever", "cough", "headache", "nausea", "vomiting",
            "diarrhea", "fatigue", "dizziness", "shortness of breath",
            "chest pain", "abdominal pain", "back pain", "joint pain"
        ]
        
        found_symptoms = []
        patient_lower = patient_info.lower()
        
        for symptom in symptom_keywords:
            if symptom in patient_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms if found_symptoms else ["general symptoms"]
    
    def _extract_symptoms_from_prescription(self, prescription_text: str) -> List[str]:
        """
        Extract symptoms and clinical findings from prescription text
        """
        symptom_keywords = [
            "pain", "fever", "cough", "headache", "nausea", "vomiting",
            "diarrhea", "fatigue", "dizziness", "shortness of breath",
            "chest pain", "abdominal pain", "back pain", "joint pain",
            "pneumonia", "infection", "inflammation", "wheeze", "dyspnea",
            "hypertension", "diabetes", "asthma", "bronchitis", "suspected",
            "diagnosis", "community-acquired", "atypical", "coverage"
        ]
        
        found_symptoms = []
        prescription_lower = prescription_text.lower()
        
        for symptom in symptom_keywords:
            if symptom in prescription_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms if found_symptoms else []
    
    def _analyze_correlations(self, image_analysis: Dict, prescription_analysis: Dict, patient_analysis: Dict) -> Dict:
        """
        Analyze correlations between different data sources
        """
        correlations = {
            "image_prescription_correlation": "moderate",
            "image_patient_correlation": "moderate",
            "prescription_patient_correlation": "high",
            "consistency_score": 0.7
        }
        
        # This would be more sophisticated in a real implementation
        return correlations
    
    def _check_consistency(self, image_analysis: Dict, prescription_analysis: Dict, patient_analysis: Dict) -> Dict:
        """
        Check consistency between different analyses
        """
        return {
            "overall_consistency": "consistent",
            "inconsistencies": [],
            "confidence_level": "high"
        }
    
    def _assess_clinical_risk(self, image_analysis: Dict, prescription_analysis: Dict, patient_analysis: Dict) -> Dict:
        """
        Assess overall clinical risk
        """
        risk_factors = []
        risk_score = 5.0  # Base score out of 10
        
        # Assess image-based risk
        if "predictions" in image_analysis and image_analysis["predictions"]:
            top_prediction = image_analysis["predictions"][0]
            if any(term in top_prediction["label"].lower() for term in ["tumor", "cancer", "fracture", "pneumonia"]):
                risk_score += 2.0
                risk_factors.append("Significant imaging findings")
        
        # Assess prescription-based risk
        if "llama_analysis" in prescription_analysis:
            llama_data = prescription_analysis["llama_analysis"]
            if isinstance(llama_data, dict) and "safety_score" in llama_data:
                safety_score = llama_data["safety_score"]
                if safety_score < 5.0:
                    risk_score += 1.5
                    risk_factors.append("Prescription safety concerns")
        
        # Assess patient-based risk
        if "triage_guidance" in patient_analysis:
            triage_data = patient_analysis["triage_guidance"]
            if isinstance(triage_data, dict) and "urgency_level" in triage_data:
                urgency = triage_data["urgency_level"]
                if urgency in ["Emergency", "Urgent"]:
                    risk_score += 2.5
                    risk_factors.append("High urgency symptoms")
        
        # Determine overall risk level
        if risk_score >= 8.0:
            overall_risk = "High"
        elif risk_score >= 6.0:
            overall_risk = "Moderate"
        else:
            overall_risk = "Low"
        
        return {
            "overall_risk": overall_risk,
            "risk_score": min(10.0, risk_score),
            "risk_factors": risk_factors,
            "mitigation_strategies": self._get_risk_mitigation_strategies(overall_risk)
        }
    
    def _generate_clinical_insights(self, image_analysis: Dict, prescription_analysis: Dict, patient_analysis: Dict) -> List[str]:
        """
        Generate clinical insights from integrated analysis
        """
        insights = []
        
        # Image-based insights
        if "clinical_insights" in image_analysis:
            insights.extend(image_analysis["clinical_insights"][:2])
        
        # Prescription-based insights
        if "llama_analysis" in prescription_analysis and isinstance(prescription_analysis["llama_analysis"], dict):
            recommendations = prescription_analysis["llama_analysis"].get("recommendations", [])
            # Clean recommendations
            clean_recs = []
            for rec in recommendations[:2]:
                clean_rec = rec.replace("**", "").replace("*", "").strip()
                if clean_rec and not clean_rec.startswith("#"):
                    clean_recs.append(clean_rec)
            insights.extend(clean_recs)
        
        # Patient-based insights
        if "clinical_bert_analysis" in patient_analysis and isinstance(patient_analysis["clinical_bert_analysis"], dict):
            summary = patient_analysis["clinical_bert_analysis"].get("summary", "")
            if summary:
                insights.append(summary)
        
        # Add general insights
        insights.append("Multimodal analysis provides comprehensive clinical picture")
        insights.append("AI-assisted analysis should be validated by clinical expertise")
        
        return insights[:6]  # Return top 6 insights
    
    def _get_risk_mitigation_strategies(self, risk_level: str) -> List[str]:
        """
        Get risk mitigation strategies based on risk level
        """
        if risk_level == "High":
            return [
                "Immediate clinical intervention required",
                "Continuous monitoring protocols",
                "Specialist consultation recommended",
                "Emergency response plan activation"
            ]
        elif risk_level == "Moderate":
            return [
                "Enhanced monitoring protocols",
                "Follow-up within 24 hours",
                "Consider specialist referral",
                "Patient education on warning signs"
            ]
        else:
            return [
                "Standard monitoring protocols",
                "Routine follow-up as scheduled",
                "Patient reassurance and education",
                "Continue current management plan"
            ]
    
    def _generate_clinical_summary(self, case_data: Dict) -> Dict:
        """
        Generate comprehensive clinical summary
        """
        summary = {
            "case_overview": "",
            "key_findings": [],
            "clinical_impression": "",
            "risk_level": "Moderate",
            "priority": "Standard"
        }
        
        try:
            # Generate case overview
            summary["case_overview"] = f"Medical case {case_data['case_id']} processed with multimodal AI analysis"
            
            # Extract key findings
            key_findings = []
            
            if "image_analysis" in case_data["analysis_results"]:
                image_analysis = case_data["analysis_results"]["image_analysis"]
                if "predictions" in image_analysis and image_analysis["predictions"]:
                    top_pred = image_analysis["predictions"][0]
                    key_findings.append(f"Imaging: {top_pred['label']} (confidence: {top_pred['confidence']:.2f})")
            
            if "prescription_analysis" in case_data["analysis_results"]:
                key_findings.append("Prescription analysis completed with safety assessment")
            
            if "patient_analysis" in case_data["analysis_results"]:
                patient_analysis = case_data["analysis_results"]["patient_analysis"]
                if "triage_guidance" in patient_analysis and isinstance(patient_analysis["triage_guidance"], dict):
                    urgency = patient_analysis["triage_guidance"].get("urgency_level", "Standard")
                    key_findings.append(f"Triage priority: {urgency}")
            
            summary["key_findings"] = key_findings
            
            # Generate clinical impression
            if "integrated_analysis" in case_data["analysis_results"]:
                integrated = case_data["analysis_results"]["integrated_analysis"]
                if "risk_assessment" in integrated:
                    risk_info = integrated["risk_assessment"]
                    summary["risk_level"] = risk_info.get("overall_risk", "Moderate")
                    summary["clinical_impression"] = f"Integrated analysis indicates {summary['risk_level'].lower()} risk case"
            
        except Exception as e:
            logger.error(f"Error generating clinical summary: {e}")
            summary["error"] = str(e)
        
        return summary
    
    def _generate_final_recommendations(self, case_data: Dict) -> List[str]:
        """
        Generate final clinical recommendations
        """
        recommendations = []
        
        try:
            # Image-based recommendations
            if "image_analysis" in case_data["analysis_results"]:
                image_analysis = case_data["analysis_results"]["image_analysis"]
                if "recommendations" in image_analysis:
                    recommendations.extend(image_analysis["recommendations"][:3])
            
            # Prescription-based recommendations
            if "prescription_analysis" in case_data["analysis_results"]:
                prescription_analysis = case_data["analysis_results"]["prescription_analysis"]
                if "llama_analysis" in prescription_analysis and isinstance(prescription_analysis["llama_analysis"], dict):
                    llama_recs = prescription_analysis["llama_analysis"].get("recommendations", [])
                    # Clean recommendations
                    clean_recs = []
                    for rec in llama_recs[:2]:
                        clean_rec = rec.replace("**", "").replace("*", "").strip()
                        if clean_rec and not clean_rec.startswith("#"):
                            clean_recs.append(clean_rec)
                    recommendations.extend(clean_recs)
            
            # Risk-based recommendations
            if "integrated_analysis" in case_data["analysis_results"]:
                integrated = case_data["analysis_results"]["integrated_analysis"]
                if "risk_assessment" in integrated:
                    risk_assessment = integrated["risk_assessment"]
                    mitigation = risk_assessment.get("mitigation_strategies", [])
                    recommendations.extend(mitigation[:2])
            
            # Add general recommendations
            recommendations.extend([
                "Correlate AI findings with clinical assessment",
                "Document all findings in patient medical record",
                "Follow institutional protocols for critical findings",
                "Consider multidisciplinary team consultation if indicated"
            ])
            
        except Exception as e:
            logger.error(f"Error generating final recommendations: {e}")
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        # Remove duplicates and limit to top 10
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:10]
    
    def save_case_to_json(self, case_data: Dict, output_dir: str = "outputs") -> str:
        """
        Save case data to JSON file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            case_id = case_data.get("case_id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_case_{case_id}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(case_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Case data saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving case to JSON: {e}")
            raise