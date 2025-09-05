# app/services/parser_service.py
import logging
from typing import Dict, Any, Optional, List
from app.models.patient_models import Medication, Diagnosis, Prescription, Consultation, Immunization, Report, ReportContent
from app.services.chatbot_service import MedicalChatbot
import json
from datetime import datetime
from bson import ObjectId

logger = logging.getLogger(__name__)

# Helper function to convert non-JSON-serializable types to strings recursively
def convert_unserializable_types(data: Any) -> Any:
    """
    Recursively converts non-JSON-serializable types (like ObjectId, datetime)
    in a dictionary or list to their string representation.
    Handles dictionaries, lists, and individual values.
    """
    if isinstance(data, dict):
        return {k: convert_unserializable_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_unserializable_types(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data

class MedicalReportParser:
    """
    Service for parsing medical report text using an AI model
    to extract structured medical information.
    """
    def __init__(self, chatbot_service: MedicalChatbot):
        if not isinstance(chatbot_service, MedicalChatbot):
            raise TypeError("MedicalReportParser requires a valid MedicalChatbot instance.")
        self.chatbot_service = chatbot_service
        logger.info("MedicalReportParser initialized.")

    async def parse_medical_report(self, report_text: str, patient_data: Dict[str, Any], doctor_data: Dict[str, Any]) -> Dict[str, Any]:
        if not report_text or not report_text.strip():
            logger.warning("Attempted to parse empty report text.")
            return {}

        logger.debug("Calling AI for medical report parsing.")

        patient_data_cleaned = convert_unserializable_types(patient_data)
        doctor_data_cleaned = convert_unserializable_types(doctor_data)
        patient_context = json.dumps(patient_data_cleaned.get("patient", {}), indent=2)
        medical_record_context = json.dumps(patient_data_cleaned.get("medical_record", {}), indent=2)
        doctor_context = json.dumps(doctor_data_cleaned, indent=2)

        prompt = f"""
You are an expert medical assistant. Your task is to carefully read the following medical report and extract structured information based on the provided patient and doctor context.
Only extract information that is *new* or *updated* compared to the existing medical record, if that information is present in the report text. If the report confirms existing information, you don't need to re-extract it unless the prompt explicitly asks for all instances.
For this task, focus on extracting the following entities mentioned in the Medical Report Text:

- **Medications:** List of medications mentioned. For each medication, include name, dosage, frequency, and notes (optional). Do NOT include start_date or end_date, as these will be handled elsewhere.
- **Diagnoses:** List of medical diagnoses mentioned. For each diagnosis, include disease name, year of diagnosis (YYYY), diagnosis_date (YYYY-MM-DD, or null if year is more precise or date not specified), and notes (optional).
- **Allergies:** List of patient allergies mentioned.
- **Consultations:** List of consultations mentioned. For each consultation, include date (YYYY-MM-DD), notes (optional), diagnosis (optional string), and followup_date (YYYY-MM-DD, or null if none specified).
- **Immunizations:** List of immunizations mentioned. For each, include vaccine name, date (YYYY-MM-DD), and notes (optional, e.g., 'administered by Dr. Smith').

Present the extracted information as a JSON object with the following top-level keys: "medications", "diagnoses", "allergies", "consultations", "immunizations". If a category is not mentioned in the report text, its corresponding value in the JSON should be an empty list `[]`. Ensure dates in the extracted JSON (for non-medication fields) are in YYYY-MM-DD format where possible.

Do NOT include any information not explicitly requested for extraction. Do NOT include any introductory or concluding text outside the JSON object. Ensure the output is valid JSON. If no relevant information is found, return an empty JSON object `{{}}` or a JSON object with empty lists `{{ "medications": [], "diagnoses": [], ... }}`.

Patient and Doctor Context (for reference, do not include this formatted context in the JSON output):
---
Patient Information:
{patient_context}

Medical Record:
{medical_record_context}

Doctor Information:
{doctor_context}
---

Medical Report Text to Parse:
---
{report_text}
---

JSON Output:
"""

        try:
            raw_ai_output = await self.chatbot_service.generate_structured_response(
                prompt=prompt,
                patient_data=patient_data_cleaned,
                doctor_data=doctor_data_cleaned
            )
            logger.debug(f"Raw AI parsing output: {raw_ai_output}")

            cleaned_output = raw_ai_output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[len("```json"):]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-len("```")]
            cleaned_output = cleaned_output.strip()

            parsed_data = json.loads(cleaned_output)
            logger.debug(f"Parsed AI data: {parsed_data}")

            extracted_data = {
                "medications": parsed_data.get("medications", []),
                "diagnoses": parsed_data.get("diagnoses", []),
                "allergies": parsed_data.get("allergies", []),
                "consultations": parsed_data.get("consultations", []),
                "immunizations": parsed_data.get("immunizations", []),
            }

            for key in extracted_data:
                if not isinstance(extracted_data[key], list):
                    logger.warning(f"AI output for key '{key}' was not a list. Received: {extracted_data[key]}. Defaulting to empty list.")
                    extracted_data[key] = []

            return extracted_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from AI output: {e}", exc_info=True)
            logger.debug(f"AI output that failed parsing: {raw_ai_output}")
            raise ValueError(f"AI returned invalid JSON format: {e}") from e
        except Exception as e:
            logger.error(f"An error occurred while processing AI output: {e}", exc_info=True)
            raise RuntimeError(f"Error processing AI output: {e}") from e