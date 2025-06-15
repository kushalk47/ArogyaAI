from fastapi import APIRouter, Request, Depends, HTTPException
from bson import ObjectId
import logging
import os
import google.generativeai as genai
import re
from app.config import db
from app.models.patient_models import MedicalRecord
from .auth_routes import get_current_authenticated_user
from typing import Dict

logger = logging.getLogger(__name__)
patient_router = APIRouter()

# --- Initialize Gemini API ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini API initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}", exc_info=True)
    gemini_model = None

# --- JSON Wellness Route ---
@patient_router.get("/wellness", name="get_wellness_plan_json")
async def get_wellness_plan_json(
    request: Request,
    current_user: dict = Depends(get_current_authenticated_user)
):
    if current_user.get("user_type") != "patient":
        raise HTTPException(status_code=403, detail="Only patients can access this resource.")

    patient_id = str(current_user.get('_id'))
    logger.info(f"Generating JSON wellness plan for patient {patient_id}")

    try:
        patient_oid = ObjectId(patient_id)
        patient_details = await db.patients.find_one({"_id": patient_oid})
        if not patient_details:
            raise HTTPException(status_code=404, detail="Patient not found.")

        medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id})
        medical_record = medical_record_doc or {
            "current_medications": [],
            "diagnoses": [],
            "prescriptions": [],
            "consultation_history": [],
            "reports": [],
            "allergies": [],
            "immunizations": [],
            "family_medical_history": None,
        }

        if medical_record.get("reports"):
            updated_reports = []
            for report_ref in medical_record["reports"]:
                if isinstance(report_ref, dict) and report_ref.get("content_id") and ObjectId.is_valid(report_ref["content_id"]):
                    content_oid = ObjectId(report_ref["content_id"])
                    report_content_doc = await db.report_contents.find_one({"_id": content_oid})
                    if report_content_doc and report_content_doc.get("content"):
                        report_ref["description"] = report_content_doc["content"]
                        updated_reports.append(report_ref)
            medical_record["reports"] = updated_reports

    except Exception as e:
        logger.error(f"Error fetching patient data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching patient data.")

    patient_info_str = f"""
Name: {patient_details.get('name', {}).get('first', '')} {patient_details.get('name', {}).get('last', '')}
Gender: {patient_details.get('gender', 'N/A')}
Diagnoses: {', '.join([str(d) for d in medical_record.get('diagnoses', [])]) or 'None'}
Current Medications: {', '.join([str(m) for m in medical_record.get('current_medications', [])]) or 'None'}
Allergies: {', '.join(medical_record.get('allergies', [])) or 'None'}
Immunizations: {', '.join([str(i) for i in medical_record.get('immunizations', [])]) or 'None'}
Family Medical History: {medical_record.get('family_medical_history', 'None')}
Recent Reports: {', '.join([r.get('description', '')[:100] for r in medical_record.get('reports', [])]) or 'None'}
"""

    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini AI service not available.")

    prompt = f"""
Based on the following patient data, generate a personalized wellness plan with these sections:

Diet Recommendations:
Healthy Habits:
Things to Avoid:
Exercise Plan:

Return only plain text with clearly labeled sections. No markdown or emojis.

Patient Data:
{patient_info_str}
"""

    try:
        response = await gemini_model.generate_content_async(prompt)
        wellness_text = re.sub(r'[\*\#]+', '', response.text.strip())

        sections = {"diet": "", "habits": "", "avoid": "", "exercise": ""}
        current_section = None

        for line in wellness_text.splitlines():
            line = line.strip()
            if line.startswith("Diet Recommendations:"):
                current_section = "diet"
                continue
            elif line.startswith("Healthy Habits:"):
                current_section = "habits"
                continue
            elif line.startswith("Things to Avoid:"):
                current_section = "avoid"
                continue
            elif line.startswith("Exercise Plan:"):
                current_section = "exercise"
                continue
            if current_section and line:
                sections[current_section] += line + " "

        for key in sections:
            if not sections[key].strip():
                sections[key] = f"No specific {key} recommendation available."

    except Exception as e:
        logger.error(f"Gemini AI failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI error while generating wellness plan.")

    return {
    "success": True,
    "message": "Wellness plan generated successfully.",
    "data": sections
}

