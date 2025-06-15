# app/routes/appointment_route.py
from fastapi import APIRouter, Request, Form, Depends, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from typing import Optional, List
from bson import ObjectId
import logging
import os
import tempfile
import google.generativeai as genai
import re
from fastapi.concurrency import run_in_threadpool
import anyio

try:
    from .doctor_routes import whisper_model, WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_LANGUAGE
except ImportError as e:
    logging.error(f"Failed to import whisper_model from doctor_routes.py: {e}. Voice transcription will be disabled.", exc_info=True)
    whisper_model = None

# Import models
from app.models.appointment_models import Appointment
from app.models.doctor_models import Doctor
from app.models.patient_models import Patient, MedicalRecord

# Import db connection
from app.config import db

# Import authentication dependency
from app.routes.auth_routes import get_current_authenticated_user

# Logging setup
logger = logging.getLogger(__name__)

# Define the router
appointment_router = APIRouter()

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

# --- Helper Dependency to get current *Patient* ---
async def get_current_patient(current_user: dict = Depends(get_current_authenticated_user)):
    """Dependency to get the current authenticated patient user document."""
    if current_user.get("user_type") != "patient":
        raise HTTPException(status_code=403, detail="Only patients can access this page.")
    return current_user

# --- Recursive helper function to convert datetime objects to ISO format strings ---
def convert_dates_to_iso(obj):
    """
    Recursively converts datetime objects within a dictionary or list to ISO format strings.
    Handles ObjectId conversion to string as well.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_dates_to_iso(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dates_to_iso(elem) for elem in obj]
    else:
        return obj

# --- Helper function to fetch patient's appointments with doctor names ---
async def fetch_patient_appointments_with_doctor_names(patient_id_str: str):
    """Fetches appointments for a patient and adds doctor names."""
    appointments_cursor = db.appointments.find({"patient_id": patient_id_str}).sort("appointment_time", 1)
    appointments_list_raw = await appointments_cursor.to_list(length=1000)

    appointments_with_names = []
    for appointment_doc in appointments_list_raw:
        try:
            doctor_id_str = appointment_doc.get("doctor_id")
            doctor_doc = None
            if doctor_id_str:
                try:
                    doctor_doc = await db.doctors.find_one({"_id": ObjectId(doctor_id_str)})
                except Exception as e:
                    logger.warning(f"Error converting doctor_id '{doctor_id_str}' to ObjectId for appointment {appointment_doc.get('_id')}: {e}")

            if doctor_doc:
                appointment_doc["doctor_name"] = f"Dr. {doctor_doc.get('name', {}).get('first', '')} {doctor_doc.get('name', {}).get('last', '')}".strip()
            else:
                appointment_doc["doctor_name"] = "Unknown Doctor"

            # Apply the recursive conversion
            appointments_with_names.append(convert_dates_to_iso(appointment_doc))

        except Exception as doctor_fetch_error:
            logger.warning(f"Error fetching doctor for appointment {appointment_doc.get('_id')}: {doctor_fetch_error}")
            appointment_doc["doctor_name"] = "Error Doctor Fetch"
            # Apply the recursive conversion even in error case for consistency
            appointments_with_names.append(convert_dates_to_iso(appointment_doc))

    return appointments_with_names

# --- Helper function to predict symptom severity with Gemini ---
async def predict_symptom_severity(medical_record: dict, reason: Optional[str], patient_notes: Optional[str]) -> str:
    """Uses Gemini to predict symptom severity based on medical record, reason, and patient notes."""
    if not gemini_model:
        logger.error("Gemini model not initialized.")
        return "Unknown"

    # Format medical record for Gemini
    medical_info_str = f"""
Medical Record:
Diagnoses: {', '.join([d.get('name', '') if isinstance(d, dict) else str(d) for d in medical_record.get('diagnoses', [])]) or 'None'}
Current Medications: {', '.join([m.get('name', '') if isinstance(m, dict) else str(m) for m in medical_record.get('current_medications', [])]) or 'None'}
Allergies: {', '.join(medical_record.get('allergies', [])) or 'None'}
Immunizations: {', '.join([i.get('name', '') if isinstance(i, dict) else str(i) for i in medical_record.get('immunizations', [])]) or 'None'}
Family Medical History: {medical_record.get('family_medical_history', 'None')}
Recent Reports: {', '.join([r.get('description', '')[:100] for r in medical_record.get('reports', [])]) or 'None'}
Reason for Visit: {reason or 'Not provided'}
Symptoms Description: {patient_notes or 'Not provided'}
"""

    prompt = f"""
Based on the following patient medical information, predict the severity of the symptoms described. Return only one of the following severity levels as plain text: 'Very Serious', 'Moderate', 'Normal'. Do not include any explanations, markdown symbols, or additional text. Analyze the diagnoses, medications, allergies, family medical history, reason for visit, and symptoms description to make an informed prediction.

{medical_info_str}
"""

    try:
        response = await gemini_model.generate_content_async(prompt)
        severity = response.text.strip()
        valid_severities = ["Very Serious", "Moderate", "Normal"]
        if severity not in valid_severities:
            logger.warning(f"Invalid severity response from Gemini: {severity}")
            return "Unknown"
        return severity
    except Exception as e:
        logger.error(f"Error predicting symptom severity with Gemini: {e}", exc_info=True)
        return "Unknown"

# --- Transcription Endpoint ---
@appointment_router.post("/transcribe", response_class=JSONResponse)
async def transcribe_symptoms(
    audio_file: UploadFile = File(...),
    current_patient: dict = Depends(get_current_patient)
):
    """Receives an audio file, transcribes it using Faster-Whisper, and returns the transcription."""
    if not whisper_model:
        logger.error("Whisper transcription model is not initialized or imported correctly.")
        return JSONResponse({"transcription": "Voice transcription model is not loaded or available."}, status_code=503)

    if not audio_file.filename:
        logger.warning("No audio file uploaded for transcription.")
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    patient_id_str = str(current_patient["_id"])
    logger.info(f"Patient {patient_id_str} received audio for transcription: {audio_file.filename}")

    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename.split('.')[-1]}") as tmp_file:
            file_content = await audio_file.read()
            await anyio.to_thread.run_sync(tmp_file.write, file_content)
            tmp_file_path = tmp_file.name

        logger.info(f"Saved uploaded audio to temporary file: {tmp_file_path}")

        # Use the imported whisper_model and WHISPER_LANGUAGE
        segments_generator, info = await run_in_threadpool(
            whisper_model.transcribe,
            tmp_file_path,
            beam_size=5,
            language=WHISPER_LANGUAGE, # Use the imported language setting
            task="transcribe"
        )

        logger.info(f"Transcription completed for '{audio_file.filename}'. Info: language={info.language}, language_probability={info.language_probability:.4f}, duration={info.duration:.2f}s")

        transcribed_text = "".join([segment.text for segment in segments_generator])
        logger.info(f"Successfully transcribed audio for patient {patient_id_str}: {transcribed_text[:100]}...")
        return JSONResponse({"transcription": transcribed_text.strip()})

    except Exception as e:
        logger.error(f"Error during Faster-Whisper transcription for patient {patient_id_str}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                logger.info(f"Cleaned up temporary audio file: {tmp_file_path}")
            except OSError as e:
                logger.warning(f"Error removing temporary file {tmp_file_path}: {e}")

# --- Book Appointment Page (GET) - Now returns JSON ---
@appointment_router.get("/book-appointment", response_class=JSONResponse)
async def get_book_and_view_appointments_page(
    current_patient: dict = Depends(get_current_patient)
):
    """Returns JSON data for booking and viewing appointments."""
    patient_id_str = str(current_patient["_id"])

    try:
        doctors_cursor = db.doctors.find({})
        doctors_list_raw = await doctors_cursor.to_list(length=1000)
        
        # Apply recursive conversion to doctors data
        doctors_list_processed = convert_dates_to_iso(doctors_list_raw)

        patient_appointments = await fetch_patient_appointments_with_doctor_names(patient_id_str)

        # Apply recursive conversion to patient data
        patient_data_for_response = convert_dates_to_iso(current_patient)
        # Ensure _id is a string, as convert_dates_to_iso handles ObjectIds
        patient_data_for_response["id"] = patient_data_for_response.get("_id", str(current_patient["_id"]))


    except Exception as e:
        logger.error(f"Error fetching data for combined page: {e}")
        # Ensure error response also converts dates if somehow present
        error_patient_data = convert_dates_to_iso({
            "id": str(current_patient["_id"]),
            "username": current_patient.get("username"),
            "email": current_patient.get("email"),
            "user_type": current_patient.get("user_type")
        })
        return JSONResponse(
            {
                "error": "Could not load page data.",
                "doctors": [],
                "appointments": [],
                "patient": error_patient_data
            },
            status_code=500
        )

    return JSONResponse(
        {
            "doctors": doctors_list_processed,
            "appointments": patient_appointments,
            "patient": patient_data_for_response
        }
    )

# --- Create Appointment (POST) - Now returns JSON ---
@appointment_router.post("/book-appointment", response_class=JSONResponse)
async def create_appointment(
    current_patient: dict = Depends(get_current_patient),
    doctor_id: str = Form(...),
    appointment_date: str = Form(...),
    appointment_time: str = Form(...),
    reason: Optional[str] = Form(None),
    patient_notes: Optional[str] = Form(None),
):
    """Handles the submission of the book appointment form and returns JSON response."""
    patient_id_str = str(current_patient["_id"])

    # Combine date and time strings into a datetime object
    try:
        datetime_str = f"{appointment_date} {appointment_time}"
        appointment_dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
        appointment_time_utc = appointment_dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        logger.warning(f"Date/time parsing error: {e}")
        return JSONResponse(
            {
                "error": "Invalid date or time format. Please useYYYY-MM-DD and HH:MM.",
                "status": "failed"
            },
            status_code=400
        )
    except Exception as e:
        logger.error(f"Unexpected error during date/time processing: {e}")
        return JSONResponse(
            {
                "error": "An error occurred processing the date or time.",
                "status": "failed"
            },
            status_code=500
        )

    # Fetch patient’s medical record
    try:
        medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id_str})
        medical_record = medical_record_doc or {
            "patient_id": patient_id_str,
            "current_medications": [],
            "diagnoses": [],
            "prescriptions": [],
            "consultation_history": [],
            "reports": [],
            "allergies": [],
            "immunizations": [],
            "family_medical_history": None,
            "updated_at": None
        }

        # Fetch report contents for context
        if medical_record.get("reports"):
            updated_reports = []
            for report_ref in medical_record.get("reports", []):
                if isinstance(report_ref, dict) and report_ref.get("content_id"):
                    try:
                        if not ObjectId.is_valid(report_ref["content_id"]):
                            logger.warning(f"Invalid content ID format: {report_ref.get('content_id')}")
                            continue
                        content_oid = ObjectId(report_ref["content_id"])
                        report_content_doc = await db.report_contents.find_one({"_id": content_oid})
                        if report_content_doc and report_content_doc.get("content"):
                            report_with_content = report_ref.copy()
                            report_with_content["description"] = report_content_doc["content"]
                            updated_reports.append(report_with_content)
                    except Exception as e:
                        logger.warning(f"Error fetching report content for severity prediction: {e}")
            medical_record["reports"] = updated_reports

    except Exception as e:
        logger.error(f"Error fetching medical record for patient {patient_id_str}: {e}")
        medical_record = {
            "patient_id": patient_id_str,
            "current_medications": [],
            "diagnoses": [],
            "prescriptions": [],
            "consultation_history": [],
            "reports": [],
            "allergies": [],
            "immunizations": [],
            "family_medical_history": None,
            "updated_at": None
        }

    # Predict symptom severity using Gemini
    predicted_severity = await predict_symptom_severity(medical_record, reason, patient_notes)

    # Create the appointment data dictionary
    appointment_data = {
        "patient_id": patient_id_str,
        "doctor_id": doctor_id,
        "appointment_time": appointment_time_utc.isoformat(), # Convert datetime to string
        "reason": reason,
        "patient_notes": patient_notes,
        "status": "Scheduled",
        "gmeet_link": None,
        "predicted_severity": predicted_severity,
        "created_at": datetime.now(timezone.utc).isoformat() # Convert datetime to string
    }

    try:
        # Insert the new appointment into the database
        insert_result = await db.appointments.insert_one(appointment_data)
        if not insert_result.inserted_id:
            raise Exception("Failed to insert appointment into database.")
        logger.info(f"Appointment created with ID: {insert_result.inserted_id}, Predicted Severity: {predicted_severity}")

        return JSONResponse(
            {
                "message": f"Appointment booked successfully! Predicted symptom severity: {predicted_severity}",
                "appointment_id": str(insert_result.inserted_id),
                "predicted_severity": predicted_severity,
                "status": "success",
                "appointment_time": appointment_time_utc.isoformat(), # Added for consistency in direct response
                "created_at": datetime.now(timezone.utc).isoformat() # Added for consistency in direct response
            },
            status_code=201 # Created
        )

    except Exception as e:
        logger.error(f"Database error during appointment creation: {e}")
        return JSONResponse(
            {
                "error": f"Error booking appointment: {e}",
                "status": "failed"
            },
            status_code=500
        )