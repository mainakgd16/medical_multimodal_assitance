"""
Main application entry point for the AI-Powered Medical Assistant
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict


# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from logger_config import setup_logging, get_logger
from medical_processor import MedicalProcessor
from medical_chatbot import MedicalChatbot


logger = get_logger(__name__)



def load_real_input_cases():
    """
    Load the 5 real medical images and prescriptions from root folder
    """
    logger.info("Loading real input cases from root folder...")
    
    input_cases = []
    
    for i in range(1, 6):
        case_data = {
            "case_number": i,
            "image_path": f"input_report_patient{i}.png",
            "prescription_path": f"input_prescription_patient{i}.txt",
            "case_id": f"OUTPUT_PATIENT_CASE_{i:03d}"
        }
        
        # Check if files exist
        if not os.path.exists(case_data["image_path"]):
            logger.warning(f"Missing image: {case_data['image_path']}")
            continue
            
        if not os.path.exists(case_data["prescription_path"]):
            logger.warning(f"Missing prescription: {case_data['prescription_path']}")
            continue
            
        # Load prescription text
        try:
            with open(case_data["prescription_path"], "r", encoding="utf-8") as f:
                case_data["prescription_text"] = f.read().strip()
            logger.info(f"Loaded case {i}: {case_data['case_id']}")
            input_cases.append(case_data)
        except Exception as e:
            logger.error(f"Error loading prescription {case_data['prescription_path']}: {e}")
    
    logger.info(f"Successfully loaded {len(input_cases)} real cases")
    return input_cases


def create_conversation_only_output(case_data):
    """
    Create a clean conversation-only output without embeddings and technical analysis
    """
    conversation_output = {
        "case_id": case_data.get("case_id"),
        "timestamp": case_data.get("timestamp"),
        "patient_info": {
            "prescription_text": case_data.get("input_data", {}).get("prescription_text", ""),
            "image_analyzed": case_data.get("input_data", {}).get("image_path", "")
        },
        "conversation": [],
        "clinical_summary": case_data.get("clinical_summary", {}),
        "recommendations": case_data.get("recommendations", [])
    }
    
    # Clean up conversation messages to remove supporting_data with embeddings
    for message in case_data.get("conversation", []):
        cleaned_message = {
            "speaker": message.get("speaker"),
            "message": message.get("message"),
            "timestamp": message.get("timestamp"),
            "type": message.get("type")
        }
        
        # Add confidence if present (but not supporting_data with embeddings)
        if "confidence" in message:
            cleaned_message["confidence"] = message["confidence"]
            
        conversation_output["conversation"].append(cleaned_message)
    
    return conversation_output


def process_real_cases() -> List[Dict]:
    """
    Process the 5 real medical cases from root folder and generate JSON outputs
    """
    logger.info("Starting processing of real medical cases from root folder")
    
    # Initialize processor
    processor = MedicalProcessor()
    logger.info(f"Medical processor initialized with models: {processor.models_status}")
    
    # Load real input cases
    input_cases = load_real_input_cases()
    
    if len(input_cases) == 0:
        logger.error("No valid input cases found. Please check root folder.")
        return []
    
    results = []
    
    for case_data in input_cases:
        try:
            logger.info(f"Processing {case_data['case_id']}...")
            
            # Process the case using the medical processor
            result = processor.process_medical_case(
                image_path=case_data["image_path"],
                prescription_text=case_data["prescription_text"],
                patient_info="",  # Not provided in prescription files
                case_id=case_data["case_id"]
            )
            
            # Create conversation-only output (without embeddings)
            conversation_only = create_conversation_only_output(result)
            
            # Save JSON output file in root folder
            output_filename = f"conversation_{case_data['case_id']}.json"
            output_path = output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(conversation_only, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated: {output_filename}")
            
            results.append({
                "case_id": case_data["case_id"],
                "status": "success",
                "output_file": output_path,
                "conversation_length": len(conversation_only.get("conversation", [])),
                "summary": conversation_only.get("clinical_summary", {})
            })
            
        except Exception as e:
            logger.error(f"Error processing case {case_data['case_id']}: {e}")
            results.append({
                "case_id": case_data["case_id"],
                "status": "error",
                "error": str(e)
            })
    
    # Save processing summary in root folder
    summary_path = "output_all_cases_processing_summary.json"
    summary_data = {
        "test_info": {
            "test_name": "Real Medical Cases Processing via main.py",
            "timestamp": datetime.now().isoformat(),
            "description": "Processed real medical images and prescriptions from root folder"
        },
        "model_info": {
            "service_url": "http://localhost:8501",
            "models_status": processor.models_status
        },
        "results_summary": {
            "total_cases": len(input_cases),
            "successful_cases": len([r for r in results if r["status"] == "success"]),
            "failed_cases": len([r for r in results if r["status"] == "error"]),
            "success_rate": f"{len([r for r in results if r['status'] == 'success'])/len(input_cases)*100:.1f}%" if input_cases else "0%"
        },
        "case_results": results
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Real cases processing complete. Summary saved to: {summary_path}")
    
    return results





def run_web_interface():
    """
    Run the Streamlit web interface
    """
    logger.info("Starting web interface...")
    
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set up Streamlit arguments
        sys.argv = [
            "streamlit",
            "run",
            "medical_chatbot.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=true"
        ]
        
        # Run Streamlit
        stcli.main()
        
    except ImportError:
        logger.error("Streamlit not installed. Please install with: pip install streamlit")
        print("To run the web interface, install streamlit and run:")
        print("streamlit run medical_chatbot.py")
    except Exception as e:
        logger.error(f"Error running web interface: {e}")
        print("To manually run the web interface:")
        print("streamlit run medical_chatbot.py")


def check_dependencies():
    """
    Check if required dependencies are available
    """
    logger.info("Checking dependencies...")
    
    missing_deps = []
    
    # Check required packages
    required_packages = [
        "torch", "transformers", "sentence_transformers", 
        "PIL", "numpy", "pandas", "sklearn", "cv2"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} available")
        except ImportError:
            missing_deps.append(package)
            logger.warning(f"✗ {package} not available")
    
    # Check Ollama availability
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama server available")
        else:
            logger.warning("✗ Ollama server not responding")
    except Exception:
        logger.warning("✗ Ollama server not available")
        print("Note: To use Llama 3.2:1b, install Ollama and run: ollama pull llama3.2:1b")
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {missing_deps}")
        print("Install missing dependencies with:")
        print("pip install -r requirements.txt")
    else:
        logger.info("All dependencies available")
    
    return len(missing_deps) == 0


def main():
    """
    Main application entry point
    """
    parser = argparse.ArgumentParser(description="AI-Powered Medical Assistant")
    parser.add_argument(
        "--mode", 
        choices=["web", "batch", "real", "single", "check"],
        default="real",
        help="Application mode: web interface, batch processing (sample cases), real cases processing, single case, or dependency check"
    )
    parser.add_argument(
        "--case-id",
        type=str,
        help="Case ID for single case processing (e.g., CASE_001)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample medical images for testing"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    logger.info("=" * 60)
    logger.info("AI-Powered Medical Assistant Starting")
    logger.info("=" * 60)
    
    
    # Check dependencies
    if args.mode == "check":
        deps_ok = check_dependencies()
        return 0 if deps_ok else 1
    
    # Run based on mode
    if args.mode == "web":
        logger.info("Starting web interface mode")
        run_web_interface()
        
    elif args.mode == "real":
        logger.info("Starting real cases processing mode")
        print("🏥 Processing Real Medical Cases from root folder")
        print("=" * 60)
        
        results = process_real_cases()
        
        # Print summary
        successful = len([r for r in results if r["status"] == "success"])
        total = len(results)
        
        print(f"\n🎉 Real Cases Processing Complete!")
        print(f"📈 Successfully processed: {successful}/{total} cases")
        print(f"📁 Output files saved in: root folder")
        print(f"🤖 Model service: http://localhost:8501")
        print(f"💬 Clean conversation-only JSON files generated")
        
        for result in results:
            if result["status"] == "success":
                print(f"✅ {result['case_id']}: {result['output_file']}")
                if 'conversation_length' in result:
                    print(f"   💬 {result['conversation_length']} messages")
            else:
                print(f"❌ {result['case_id']}: {result['error']}")
        
        print(f"\n📊 Summary: real_cases_processing_summary.json")
        
    
    logger.info("Application finished")
    return 0


if __name__ == "__main__":
    exit(main())