# app/routes/doctor_routes.py
from fastapi import APIRouter, Request, Depends, HTTPException, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any # Import Dict, Any
from bson import ObjectId
from pathlib import Path
import logging
import tempfile
import os
from datetime import datetime
import io
import torch
import json # Import json for potentially logging parsed data


# Added imports for async execution in transcription
from fastapi.concurrency import run_in_threadpool
import anyio

# Import ReportLab components
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Import Pydantic models
# Assuming ChatRequest, ReportRequest, and ReportPDFRequest are defined elsewhere
# ReportPDFRequest is used for the PDF and Save endpoints to carry report_content_text
try:
    from app.models.patient_models import ChatRequest, ReportRequest, ReportPDFRequest, MedicalRecord, Medication, Diagnosis, Consultation, Immunization, Report, ReportContent # Import models for validation and merging
except ImportError:
     logging.error("Failed to import models from app.models.patient_models. Ensure the file exists and contains necessary models.")
     # Define placeholder models if import fails
     class ChatRequest(): query: Optional[str] = None; action: str = 'ask'
     class ReportRequest(): transcribed_text: str = ''
     class ReportPDFRequest(): report_content_text: str = '' # Placeholder for PDF and Save endpoint body

# Import db from the config module
from app.config import db

# Import the authentication dependency
from .auth_routes import get_current_authenticated_user

# Import the MedicalChatbot service
try:
    from app.services.chatbot_service import MedicalChatbot
    # Ensure MedicalChatbot has methods like generate_response, summarize_medical_record, generate_medical_report, generate_structured_response
except ImportError:
    logging.warning("Chatbot service not found. Chatbot/Report functionality will be disabled.")
    MedicalChatbot = None

# Import the MedicalReportParser service
try:
    # MedicalReportParser depends on MedicalChatbot
    from app.services.parser_service import MedicalReportParser
except ImportError:
    logging.warning("MedicalReportParser service not found. Save parsed data functionality will be disabled.")
    MedicalReportParser = None


# Import FasterWhisper
try:
    from faster_whisper import WhisperModel
except ImportError:
    logging.warning("Faster-Whisper not installed. Voice transcription will be disabled.")
    WhisperModel = None


logger = logging.getLogger(__name__)
doctor_router = APIRouter()

# --- TEMPLATES PATH ---
current_file_path = Path(__file__).resolve()
routes_dir = current_file_path.parent
app_dir = routes_dir.parent
templates_dir_path = app_dir / "templates"
templates = Jinja2Templates(directory=templates_dir_path)

# --- SERVICE INITIALIZATION ---
chatbot_service: Optional[MedicalChatbot] = None
parser_service: Optional[MedicalReportParser] = None # Initialize parser_service here
whisper_model: Optional[WhisperModel] = None


if MedicalChatbot:
    try:
        # Assuming MedicalChatbot does not need arguments here based on your previous code
        chatbot_service = MedicalChatbot()
        logger.info("MedicalChatbot initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize MedicalChatbot: {e}", exc_info=True)
        chatbot_service = None

# Initialize parser service AFTER chatbot service, as it depends on it
if MedicalReportParser and chatbot_service:
    try:
        parser_service = MedicalReportParser(chatbot_service=chatbot_service)
        logger.info("MedicalReportParser initialized successfully.")
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to initialize MedicalReportParser: {e}", exc_info=True)
        parser_service = None
    except Exception as e:
         logger.error(f"An unexpected error occurred during MedicalReportParser initialization: {e}", exc_info=True)
         parser_service = None


if WhisperModel:
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if WHISPER_DEVICE == "cuda" else "int8")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", None)
    try:
        logger.info(f"Initializing Whisper model with device={WHISPER_DEVICE}, compute_type={WHISPER_COMPUTE_TYPE}")
        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            # device_index=0, # Keep or remove based on your specific setup needs
        )
        logger.info(f"Faster-Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully on {WHISPER_DEVICE} with compute type {WHISPER_COMPUTE_TYPE}.")
    except Exception as e:
        logger.error(f"Error loading Faster-Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}: {e}", exc_info=True)
        whisper_model = None

# --- PDF GENERATION HELPER FUNCTION ---
def create_report_pdf(doctor_info: dict, patient_info: dict, report_content_text: str) -> io.BytesIO:
    """
    Generates a medical report PDF using ReportLab with doctor and patient headers.
    Takes doctor_info, patient_info (as dicts), and the formatted report text.
    Returns a BytesIO object containing the PDF data.
    """
    buffer = io.BytesIO()
    # Increased bottom margin slightly to ensure footer doesn't overlap content on last page
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=0.75 * inch) # Increased bottom margin


    # Styles
    styles = getSampleStyleSheet() # Corrected typo here
    # Ensure 'CustomNormal' is added or use 'Normal' if you don't need custom style
    if 'CustomNormal' not in styles:
         styles.add(ParagraphStyle(name='CustomNormal', fontSize=10, leading=12, alignment=TA_LEFT, spaceAfter=6))
    if 'FooterStyle' not in styles: # Added style for footer
         styles.add(ParagraphStyle(name='FooterStyle', fontSize=9, leading=10, alignment=TA_CENTER, spaceBefore=10, textColor=styles['Normal'].textColor))


    styles.add(ParagraphStyle(name='Heading1Center', fontSize=14, leading=16, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Heading2Left', fontSize=12, leading=14, alignment=TA_LEFT, spaceBefore=10, spaceAfter=5, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Small', fontSize=9, leading=10, alignment=TA_LEFT, spaceAfter=4))


    Story = []

    # --- Header Information (Doctor and Patient) ---

    # Doctor Information Section
    doctor_name = f"{doctor_info.get('name', {}).get('first', '')} {doctor_info.get('name', {}).get('last', '')}".strip() or "Dr. [Doctor Name]"
    doctor_specialization = doctor_info.get('specialty', 'Medical Practitioner') # Use 'specialty' key
    # Access contact details safely with .get()
    doctor_contact_email = doctor_info.get('contact', {}).get('email', '')
    doctor_contact_phone = doctor_info.get('contact', {}).get('phone', '')

    Story.append(Paragraph("<font size=12><b>Medical Report</b></font>", styles['Heading1Center']))

    # Add doctor info - using a simple table-like structure with paragraphs and spaces
    # Using CustomNormal style
    Story.append(Paragraph(f"<font size=10><b>Dr. {doctor_name}</b>, {doctor_specialization}</font>", styles['CustomNormal']))
    if doctor_contact_email:
        Story.append(Paragraph(f"<font size=9>Email: {doctor_contact_email}</font>", styles['Small']))
    if doctor_contact_phone:
        Story.append(Paragraph(f"<font size=9>Phone: {doctor_contact_phone}</font>", styles['Small']))
    Story.append(Spacer(1, 0.2 * inch)) # Add some vertical space

    # Patient Information Section
    patient_name = f"{patient_info.get('name', {}).get('first', '')} {patient_info.get('name', {}).get('last', '')}".strip() or "[Patient Name]"
    patient_dob_str = patient_info.get('date_of_birth') # Assuming this is a string
    patient_dob = "N/A"
    if patient_dob_str:
        try:
            # Attempt to parse and format date string, handle potential errors
            # Handle both date and datetime objects/strings
            if isinstance(patient_dob_str, datetime):
                 patient_dob = patient_dob_str.strftime('%B %d, %Y')
            else: # Assume string
                 patient_dob = datetime.strptime(str(patient_dob_str).split('T')[0], '%Y-%m-%d').strftime('%B %d, %Y') # Handle potential datetime strings

        except (ValueError, TypeError):
            logger.warning(f"Could not parse patient DOB string for PDF: {patient_dob_str}")
            patient_dob = str(patient_dob_str) # Use raw string if parsing fails


    # Patient ID might be an ObjectId, convert to string for display
    patient_id_display = str(patient_info.get('_id', 'N/A'))


    Story.append(Paragraph("<font size=10><b>Patient Information:</b></font>", styles['Heading2Left']))
    Story.append(Paragraph(f"<font size=10><b>Name:</b> {patient_name}</font>", styles['CustomNormal']))
    Story.append(Paragraph(f"<font size=10><b>Patient ID:</b> {patient_id_display}</font>", styles['CustomNormal']))
    Story.append(Paragraph(f"<font size=10><b>Date of Birth:</b> {patient_dob}</font>", styles['CustomNormal']))
    # Add more patient details if needed, using styles['CustomNormal'] or similar
    Story.append(Spacer(1, 0.3 * inch))

    # --- Report Content ---
    Story.append(Paragraph("<font size=10><b>Report Details:</b></font>", styles['Heading2Left']))
    # The actual report text generated by Gemini (or edited by the user)
    # ReportLab Paragraph can handle basic HTML tags like <br/> for line breaks
    # Replace newlines with <br/> for proper rendering in Paragraph
    # Also handle potential empty report_content_text
    report_content_formatted = report_content_text.replace('\n', '<br/>') if report_content_text else "No report content provided."
    # Use CustomNormal style for the report body
    Story.append(Paragraph(report_content_formatted, styles['CustomNormal']))


    Story.append(Spacer(1, 0.5 * inch))
    # Add generation timestamp - maybe move to footer?
    # Story.append(Paragraph(f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Small']))

    # --- Footer ---
    # This is a basic footer. For more complex footers (like page numbers),
    # you would need to use the onFirstPage/onLaterPages arguments in SimpleDocTemplate.build()
    footer_text = f"Generated by Aarogya AI on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Page <page/> of <npgs/>"
    Story.append(Paragraph(footer_text, styles['FooterStyle'])) # Added footer


    # Build the PDF
    try:
        doc.build(Story)
        buffer.seek(0) # Rewind the buffer
        return buffer
    except Exception as e:
        logger.error(f"Error building PDF report: {e}", exc_info=True) # Log traceback
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {e}")

# --- END PDF GENERATION HELPER FUNCTION ---


# Standard GET/POST routes for dashboard, search, view patient details
@doctor_router.get("/dashboard/", response_class=HTMLResponse,name="get_doctor_dashboard")
async def get_doctor_dashboard(
    request: Request,
    current_user_doc: dict = Depends(get_current_authenticated_user)
):
    """
    Displays the doctor's dashboard with a patient search form.
    Requires doctor authentication.
    """
    if current_user_doc.get("user_type") != "doctor":
        logger.warning(f"Access denied for user {current_user_doc.get('_id')} trying to access doctor dashboard.")
        raise HTTPException(status_code=403, detail="Access denied. Only doctors can access this dashboard.")
    # Assuming "dashboard.html" exists and can handle search results (empty initially)
    return templates.TemplateResponse("dashboard.html", {"request": request, "doctor": current_user_doc, "search_results": [], "search_query": ""})


@doctor_router.post("/dashboard/search-patient", response_class=HTMLResponse)
async def search_patient(
    request: Request,
    current_user_doc: dict = Depends(get_current_authenticated_user),
    search_query: str = Form(...)
):
    """
    Searches for patients based on the provided query (name or email).
    Requires doctor authentication.
    """
    if current_user_doc.get("user_type") != "doctor":
        logger.warning(f"Access denied for user {current_user_doc.get('_id')} trying to search patients.")
        raise HTTPException(status_code=403, detail="Access denied.")

    patients_found = []
    # Ensure search_query is treated as a string, even if Form provides something else
    search_query_str = str(search_query).strip()

    if search_query_str:
        logger.info(f"Doctor {current_user_doc.get('_id')} searching for: '{search_query_str}'")
        try:
            # Basic search by name (first or last) or email - improve this as needed
            # Using case-insensitive regex search
            patients_found = await db.patients.find({
                "$or": [
                    {"name.first": {"$regex": search_query_str, "$options": "i"}},
                    {"name.last": {"$regex": search_query_str, "$options": "i"}},
                    {"email": {"$regex": search_query_str, "$options": "i"}}
                ]
            }).to_list(length=100) # Limit the number of results
            logger.info(f"Found {len(patients_found)} patients for query '{search_query_str}'.")
        except Exception as e:
             logger.error(f"Error during patient search for query '{search_query_str}': {e}", exc_info=True)
             patients_found = []


    # Render the dashboard template, passing the search results
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "doctor": current_user_doc, "search_results": patients_found, "search_query": search_query_str}
    )


@doctor_router.get("/dashboard/patient/{patient_id}", response_class=HTMLResponse, name="view_patient_details")
async def view_patient_details(
    request: Request,
    patient_id: str,
    current_user_doc: dict = Depends(get_current_authenticated_user)
):
    """
    Displays the detailed profile of a specific patient to the doctor.
    Requires doctor authentication.
    """
    if current_user_doc.get("user_type") != "doctor":
        logger.warning(f"Access denied for user {current_user_doc.get('_id')} trying to view patient {patient_id}.")
        raise HTTPException(status_code=403, detail="Access denied.")

    logger.info(f"Doctor {current_user_doc.get('_id')} viewing patient details for ID: {patient_id}")
    try:
        # Validate patient_id format
        if not ObjectId.is_valid(patient_id):
             logger.warning(f"Invalid patient ID format received: {patient_id}")
             raise HTTPException(status_code=400, detail="Invalid patient ID format.")

        patient_oid = ObjectId(patient_id)
        patient_details = await db.patients.find_one({"_id": patient_oid})

    except HTTPException: # Re-raise HTTP exceptions like invalid ID format
        raise
    except Exception as e: # Catch other potential DB errors
        logger.error(f"Error fetching patient {patient_id} for details view: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching patient data.")

    if not patient_details:
        logger.warning(f"Patient not found for ID: {patient_id}")
        raise HTTPException(status_code=404, detail="Patient not found.")

    # Fetch the patient's medical record
    try:
        medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id})
    except Exception as e:
        logger.error(f"Error fetching medical record for patient {patient_id}: {e}", exc_info=True)
        medical_record_doc = None # Treat as not found on error


    # --- Fetch Report Contents for Display in Medical Info ---
    medical_record_for_template = medical_record_doc.copy() if medical_record_doc else {}
    if medical_record_doc and medical_record_doc.get("reports"):
        updated_reports = []
        for report_ref in medical_record_for_template.get("reports", []):
             if isinstance(report_ref, dict) and report_ref.get("content_id"):
                try:
                    if not ObjectId.is_valid(report_ref["content_id"]):
                        logger.warning(f"Invalid content ID format for report display: {report_ref.get('content_id')}")
                        continue

                    content_oid = ObjectId(report_ref["content_id"])
                    report_content_doc = await db.report_contents.find_one({"_id": content_oid})

                    if report_content_doc and report_content_doc.get("content") is not None:
                         report_with_content = report_ref.copy()
                         report_with_content["description"] = report_content_doc["content"]
                         updated_reports.append(report_with_content)
                    elif report_content_doc:
                         logger.warning(f"Report content found for {content_oid} but content field is missing or None.")
                         report_with_content = report_ref.copy()
                         report_with_content["description"] = "Report content missing."
                         updated_reports.append(report_with_content)
                    else:
                         logger.warning(f"Report content document not found for ID: {content_oid}")
                         report_with_content = report_ref.copy()
                         report_with_content["description"] = "Report content not found."
                         updated_reports.append(report_with_content)

                except Exception as e:
                    logger.warning(f"Error fetching report content for display for {report_ref.get('content_id')}: {e}", exc_info=True)
                    error_report = report_ref.copy()
                    error_report["description"] = f"Error loading report content: {e}"
                    updated_reports.append(error_report)

        medical_record_for_template["reports"] = updated_reports

    elif medical_record_doc:
        medical_record_for_template["reports"] = []


    # Render the patient details template
    return templates.TemplateResponse(
        "doctor_patient_details.html",
        {
            "request": request,
            "patient": patient_details,
            "medical_record": medical_record_for_template
        }
    )


# --- CHAT ENDPOINT ---
@doctor_router.post("/dashboard/patient/{patient_id}/chat", response_class=JSONResponse)
async def chat_with_patient_data(
    request: Request,
    patient_id: str,
    chat_request: ChatRequest,
    current_user_doc: dict = Depends(get_current_authenticated_user)
):
    """
    Handles chat requests from the doctor about a specific patient.
    Fetches patient data and uses the chatbot service to generate a response.
    """
    if current_user_doc.get("user_type") != "doctor":
        logger.warning(f"Access denied for user {current_user_doc.get('_id')} trying to chat about patient {patient_id}.")
        raise HTTPException(status_code=403, detail="Access denied.")

    if not chatbot_service:
        logger.error("Chatbot service is not initialized.")
        return JSONResponse({"response": "Chatbot service is not available."}, status_code=503)

    logger.info(f"Doctor {current_user_doc.get('_id')} received chat request for patient {patient_id}: action='{chat_request.action}', query='{chat_request.query}'")

    try:
        if not ObjectId.is_valid(patient_id):
             logger.warning(f"Invalid patient ID format received for chat: {patient_id}")
             raise HTTPException(status_code=400, detail="Invalid patient ID format.")

        patient_oid = ObjectId(patient_id)
        patient_details = await db.patients.find_one({"_id": patient_oid})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id} for chat: {e}", exc_info=True)
        return JSONResponse({"response": "Error fetching patient data."}, status_code=500)


    if not patient_details:
        logger.warning(f"Patient not found for chat request for ID: {patient_id}")
        return JSONResponse({"response": "Patient not found."}, status_code=404)

    # Fetch the patient's medical record
    medical_record_for_chatbot = {} # Initialize as empty dict
    try:
        medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id})
        medical_record_for_chatbot = medical_record_doc.copy() if medical_record_doc else {}

        # Fetch Report Contents for Chatbot Context
        if medical_record_doc and medical_record_doc.get("reports"):
            updated_reports = []
            for report_ref in medical_record_for_chatbot.get("reports", []):
                 if isinstance(report_ref, dict) and report_ref.get("content_id"):
                    try:
                        if not ObjectId.is_valid(report_ref["content_id"]):
                            logger.warning(f"Invalid content ID format for chat context: {report_ref.get('content_id')}")
                            continue

                        content_oid = ObjectId(report_ref["content_id"])
                        report_content_doc = await db.report_contents.find_one({"_id": content_oid})

                        if report_content_doc and report_content_doc.get("content") is not None:
                             report_with_content = report_ref.copy()
                             report_with_content["description"] = report_content_doc["content"]
                             updated_reports.append(report_with_content)
                    except Exception as e:
                        logger.warning(f"Error fetching report content for chat context for {report_ref.get('content_id')}: {e}", exc_info=True)

            medical_record_for_chatbot["reports"] = updated_reports
        elif medical_record_doc:
            medical_record_for_chatbot["reports"] = []


    except Exception as e:
        logger.error(f"Error fetching medical record for patient {patient_id} for chat: {e}", exc_info=True)
        medical_record_for_chatbot = {} # Ensure empty on error


    # Combine patient and medical record data for the chatbot service
    patient_full_data = {
        "patient": patient_details,
        "medical_record": medical_record_for_chatbot
    }

    # Determine action and get response
    response_text = "Sorry, an error occurred with the chatbot."
    try:
        if chat_request.action == 'ask' and chat_request.query:
            if not chat_request.query.strip():
                 raise HTTPException(status_code=400, detail="Query is required for 'ask' action")
            try:
                response_text = await chatbot_service.generate_response(patient_full_data, chat_request.query)
            except Exception as e:
                logger.error(f"Error generating chatbot response for patient {patient_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error generating chatbot response: {e}")


        elif chat_request.action == 'summarize':
            if not medical_record_doc:
                 raise HTTPException(status_code=404, detail="No medical record found to summarize.")
            try:
                response_text = await chatbot_service.summarize_medical_record(patient_full_data)
            except Exception as e:
                logger.error(f"Error generating medical record summary for patient {patient_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")

        else:
            raise HTTPException(status_code=400, detail=f"Invalid chat action: '{chat_request.action}'. Must be 'ask' or 'summarize'.")

    except HTTPException as e:
        logger.warning(f"HTTP Exception in chat_with_patient_data for patient {patient_id}: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected Error during chatbot service call for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with the chatbot service: {e}")

    return JSONResponse({"response": response_text if response_text is not None else ""})


# --- Voice Transcription Endpoint ---
@doctor_router.post("/dashboard/patient/{patient_id}/transcribe", response_class=JSONResponse)
async def transcribe_medical_report(
    request: Request,
    patient_id: str,
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    current_user_doc: dict = Depends(get_current_authenticated_user)
):
    """
    Receives an audio file, saves it temporarily, and sends it to the loaded Faster-Whisper model for transcription.
    """
    if current_user_doc.get("user_type") != "doctor":
        logger.warning(f"Access denied for user {current_user_doc.get('_id')} trying to transcribe for patient {patient_id}.")
        raise HTTPException(status_code=403, detail="Access denied.")

    if not whisper_model:
        logger.error("Whisper transcription model is not initialized.")
        return JSONResponse({"transcription": "Voice transcription model is not loaded or available."}, status_code=503)

    if not audio_file.filename:
        logger.warning("No audio file uploaded for transcription.")
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    logger.info(f"Doctor {current_user_doc.get('_id')} received audio for transcription for patient {patient_id}: {audio_file.filename}")

    tmp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as tmp_file:
            file_content = await audio_file.read()
            await anyio.to_thread.run_sync(tmp_file.write, file_content)
            tmp_file_path = tmp_file.name

        logger.info(f"Saved uploaded audio to temporary file: {tmp_file_path}")

    except Exception as e:
        logger.error(f"Error processing or saving uploaded audio file '{audio_file.filename}': {e}", exc_info=True)
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except OSError as cleanup_error:
                logger.warning(f"Error during cleanup of failed temp file {tmp_file_path}: {cleanup_error}")
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {e}")

    try:
        transcription_language = language if language else WHISPER_LANGUAGE
        logger.info(f"Starting transcription of '{audio_file.filename}' with language: {transcription_language or 'auto-detect'} on device: {WHISPER_DEVICE}")

        segments_generator, info = await run_in_threadpool(
             whisper_model.transcribe,
             tmp_file_path,
             beam_size=5,
             language=transcription_language,
             task="transcribe",
        )

        logger.info(f"Transcription completed for '{audio_file.filename}'. Info: language={info.language}, language_probability={info.language_probability:.4f}, duration={info.duration:.2f}s")

        transcribed_text = "".join([segment.text for segment in segments_generator])

        logger.info(f"Successfully transcribed audio for patient {patient_id}: {transcribed_text[:100]}...")
        return JSONResponse({"transcription": transcribed_text.strip()})

    except Exception as e:
        logger.error(f"Error during Faster-Whisper transcription for patient {patient_id} and file '{audio_file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                logger.info(f"Cleaned up temporary audio file: {tmp_file_path}")
            except OSError as e:
                logger.warning(f"Error removing temporary file {tmp_file_path}: {e}")


# --- Endpoint to generate Report text (returns JSON of the formatted text) ---
@doctor_router.post("/dashboard/patient/{patient_id}/generate-report-text", response_class=JSONResponse)
async def generate_medical_report_text_endpoint(
    request: Request,
    patient_id: str,
    report_request: ReportRequest,
    current_user_doc: dict = Depends(get_current_authenticated_user)
):
    """
    Receives raw transcribed text, fetches patient/doctor details,
    and uses the MedicalChatbot service to format it into medical report text.
    This endpoint returns the formatted text as JSON.
    """
    if current_user_doc.get("user_type") != "doctor":
        logger.warning(f"Access denied for user {current_user_doc.get('_id')} trying to format report text for patient {patient_id}.")
        raise HTTPException(status_code=403, detail="Access denied.")

    if not chatbot_service:
        logger.error("Chatbot service is not initialized.")
        return JSONResponse({"report_text": "AI service for report generation is not available."}, status_code=503)

    transcribed_text = report_request.transcribed_text
    logger.debug(f"Received transcribed_text for formatting (first 100 chars): {transcribed_text[:100]}...")

    if not transcribed_text or not transcribed_text.strip():
        logger.warning("Received empty transcribed text for formatting.")
        return JSONResponse({"report_text": "No transcribed text provided to generate a report."}, status_code=400)

    # --- Fetch Patient Details (needed for AI context) ---
    patient_details = None
    medical_record_for_service = {}
    try:
        if not ObjectId.is_valid(patient_id):
             logger.warning(f"Invalid patient ID format received for report text generation: {patient_id}")
             raise HTTPException(status_code=400, detail="Invalid patient ID format.")

        patient_oid = ObjectId(patient_id)
        patient_details = await db.patients.find_one({"_id": patient_oid})

        if patient_details:
            try:
                medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id})
                medical_record_for_service = medical_record_doc.copy() if medical_record_doc else {}
                if medical_record_doc and medical_record_doc.get("reports"):
                    updated_reports = []
                    for report_ref in medical_record_for_service.get("reports", []):
                        if isinstance(report_ref, dict) and report_ref.get("content_id"):
                            try:
                                if not ObjectId.is_valid(report_ref["content_id"]): continue
                                content_oid = ObjectId(report_ref["content_id"])
                                report_content_doc = await db.report_contents.find_one({"_id": content_oid})
                                if report_content_doc and report_content_doc.get("content") is not None:
                                    report_with_content = report_ref.copy()
                                    report_with_content["description"] = report_content_doc["content"]
                                    updated_reports.append(report_with_content)
                            except Exception as e:
                                logger.warning(f"Error fetching report content for AI formatting context for {report_ref.get('content_id')}: {e}", exc_info=True)
                    medical_record_for_service["reports"] = updated_reports
                elif medical_record_doc:
                    medical_record_for_service["reports"] = []

            except Exception as e:
                 logger.error(f"Error fetching medical record for patient {patient_id} for report text generation: {e}", exc_info=True)
                 medical_record_for_service = {}


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id} for report text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching patient data for report generation: {e}")


    if not patient_details:
        logger.warning(f"Patient not found for report text generation for ID: {patient_id}")
        return JSONResponse({"report_text": "Patient not found."}, status_code=404)

    doctor_details = current_user_doc
    patient_data_for_service = {"patient": patient_details, "medical_record": medical_record_for_service}

    # --- Call the service method for report text generation ---
    formatted_report_text = "Error generating report text."
    try:
        logger.debug(f"Calling generate_medical_report with patient_data: {patient_data_for_service}, doctor_data: {doctor_details}, transcribed_text: {transcribed_text}")
        # Use positional arguments as per your code's call structure
        formatted_report_text = await chatbot_service.generate_medical_report(patient_data_for_service, doctor_details, transcribed_text)

        logger.debug(f"Received formatted_report_text (first 100 chars): {str(formatted_report_text)[:100]}...")

        if not formatted_report_text:
             formatted_report_text = "AI generated empty report content."
             logger.warning("AI generated empty report content.")


    except Exception as e:
        logger.error(f"Error calling MedicalChatbot service for report text generation for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating report text via AI service: {e}")

    return JSONResponse({"report_text": formatted_report_text if formatted_report_text is not None else ""})


# --- Endpoint to generate PDF Report ---
@doctor_router.post("/dashboard/patient/{patient_id}/generate-pdf-report")
async def generate_medical_pdf_report_endpoint(
    request: Request,
    patient_id: str,
    report_request: ReportPDFRequest,
    current_user_doc: dict = Depends(get_current_authenticated_user)
):
    """
    Receives raw transcribed text, fetches patient/doctor details,
    uses the MedicalChatbot service to format the core report text internally,
    and then generates a PDF report using ReportLab.
    Returns the PDF as a file download.
    """
    if current_user_doc.get("user_type") != "doctor":
        logger.warning(f"Access denied for user {current_user_doc.get('_id')} trying to generate PDF report for patient {patient_id}.")
        raise HTTPException(status_code=403, detail="Access denied.")

    if not chatbot_service:
        logger.error("Chatbot service is not initialized.")
        raise HTTPException(status_code=503, detail="AI service for report content generation is not available.")


    transcribed_text = report_request.report_content_text # Use report_content_text from ReportPDFRequest
    logger.debug(f"Received text for PDF (first 100 chars): {transcribed_text[:100]}...")

    if not transcribed_text or not transcribed_text.strip():
        logger.warning("Received empty text for PDF generation.")
        raise HTTPException(status_code=400, detail="No text content provided to generate a report.")

    # --- Fetch Patient Details (needed for PDF header AND AI context) ---
    patient_details = None
    medical_record_for_service = {}
    try:
        if not ObjectId.is_valid(patient_id):
             logger.warning(f"Invalid patient ID format received for PDF generation: {patient_id}")
             raise HTTPException(status_code=400, detail="Invalid patient ID format.")

        patient_oid = ObjectId(patient_id)
        patient_details = await db.patients.find_one({"_id": patient_oid})

        if not patient_details:
            logger.warning(f"Patient not found for PDF generation for ID: {patient_id}")
            raise HTTPException(status_code=404, detail="Patient not found.")


        if patient_details:
            try:
                medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id})
                medical_record_for_service = medical_record_doc.copy() if medical_record_doc else {}
                if medical_record_doc and medical_record_doc.get("reports"):
                    updated_reports = []
                    for report_ref in medical_record_for_service.get("reports", []):
                        if isinstance(report_ref, dict) and report_ref.get("content_id"):
                            try:
                                if not ObjectId.is_valid(report_ref["content_id"]): continue
                                content_oid = ObjectId(report_ref["content_id"])
                                report_content_doc = await db.report_contents.find_one({"_id": content_oid})
                                if report_content_doc and report_content_doc.get("content") is not None:
                                    report_with_content = report_ref.copy()
                                    report_with_content["description"] = report_content_doc["content"]
                                    updated_reports.append(report_with_content)
                            except Exception as e:
                                logger.warning(f"Error fetching report content for AI context in PDF generation for {report_ref.get('content_id')}: {e}", exc_info=True)
                    medical_record_for_service["reports"] = updated_reports
                elif medical_record_doc:
                    medical_record_for_service["reports"] = []

            except Exception as e:
                 logger.error(f"Error fetching medical record for patient {patient_id} for PDF generation: {e}", exc_info=True)
                 medical_record_for_service = {}


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id} for PDF generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching patient data for report generation: {e}")


    doctor_details = current_user_doc
    patient_data_for_service = {"patient": patient_details, "medical_record": medical_record_for_service}


    # --- Call the service method for report content text generation (INTERNAL CALL) ---
    # We are sending the formatted text received from the frontend here,
    # so we don't need the AI to *re-format* it.
    # Instead, we will pass the *already formatted text* to the PDF helper directly.
    # The previous logic was sending transcribed_text to generate_medical_report,
    # which is redundant if the frontend sends the formatted text.
    # Let's assume the frontend sends the FINAL formatted text in report_content_text.

    final_report_content_text = transcribed_text # Use the text received from frontend

    # --- Generate PDF using the fetched data and the FINAL report text ---
    try:
        logger.debug("Calling create_report_pdf with doctor_details, patient_details, and final_report_content_text")
        pdf_buffer = create_report_pdf(
            doctor_info=doctor_details,
            patient_info=patient_details,
            report_content_text=final_report_content_text # Use the final text from the frontend
        )
        logger.info(f"Successfully generated PDF report buffer for patient {patient_id}.")

        response = StreamingResponse(pdf_buffer, media_type="application/pdf")
        patient_last_name = patient_details.get('name', {}).get('last', patient_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"medical_report_{patient_last_name}_{timestamp}.pdf"

        response.headers["Content-Disposition"] = f"attachment; filename={pdf_filename}"
        logger.debug(f"Sending PDF file '{pdf_filename}' for download.")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during PDF streaming for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during PDF streaming: {e}")


# --- Endpoint to Parse Report Text and Save to DB ---
@doctor_router.post("/dashboard/patient/{patient_id}/save-parsed-report", response_class=JSONResponse)
async def save_parsed_report_data(
    request: Request,
    patient_id: str,
    save_request: ReportPDFRequest, # Expect the final formatted report text here
    current_user_doc: dict = Depends(get_current_authenticated_user)
):
    """
    Receives formatted report text, parses it using the AI service,
    and saves the extracted structured data to the patient's medical record in the database.
    """
    if current_user_doc.get("user_type") != "doctor":
        logger.warning(f"Access denied for user {current_user_doc.get('_id')} trying to save parsed report data for patient {patient_id}.")
        raise HTTPException(status_code=403, detail="Access denied.")

    if not parser_service:
        logger.error("Parser service is not initialized.")
        raise HTTPException(status_code=503, detail="AI parsing service is not available.")

    report_content_text = save_request.report_content_text
    logger.debug(f"Received report_content_text for parsing (first 100 chars): {report_content_text[:100]}...")

    if not report_content_text or not report_content_text.strip():
        logger.warning("Received empty report content text for parsing.")
        raise HTTPException(status_code=400, detail="No report content provided to parse and save.")

    # --- Fetch Patient and Medical Record Details ---
    patient_details = None
    medical_record_doc = None
    try:
        if not ObjectId.is_valid(patient_id):
             logger.warning(f"Invalid patient ID format received for saving parsed data: {patient_id}")
             raise HTTPException(status_code=400, detail="Invalid patient ID format.")

        patient_oid = ObjectId(patient_id)
        patient_details = await db.patients.find_one({"_id": patient_oid})

        if not patient_details:
            logger.warning(f"Patient not found for saving parsed data for ID: {patient_id}")
            raise HTTPException(status_code=404, detail="Patient not found.")

        # Fetch existing medical record - create if it doesn't exist
        medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id})
        if not medical_record_doc:
            logger.info(f"No existing medical record found for patient {patient_id}. Creating a new one.")
            new_medical_record = {"patient_id": patient_id, "reports": [], "current_medications": [], "diagnoses": [], "allergies": [], "consultation_history": [], "immunizations": [], "prescriptions": [], "updated_at": datetime.utcnow()} # Added updated_at
            insert_result = await db.medical_records.insert_one(new_medical_record)
            # Fetch the newly created document to ensure it's the full object with _id
            medical_record_doc = await db.medical_records.find_one({"_id": insert_result.inserted_id})
            logger.info(f"Created medical record with ID: {insert_record_doc.get('_id')}") # Use inserted_id from result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient or medical record for saving parsed data for {patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching patient data for saving: {e}")

    # --- Call the Parser Service ---
    extracted_data = {}
    try:
        # Pass the text to parse, patient context, and doctor context to the parser service
        patient_full_data_for_parser = {"patient": patient_details, "medical_record": medical_record_doc}
        doctor_details = current_user_doc # Doctor info for context

        logger.debug("Calling parser_service.parse_medical_report...")
        extracted_data = await parser_service.parse_medical_report(
            report_text=report_content_text,
            patient_data=patient_full_data_for_parser,
            doctor_data=doctor_details
        )
        logger.debug(f"Parser service returned data: {extracted_data}")

        if not extracted_data or all(not v for v in extracted_data.values()):
             logger.warning(f"Parser service returned empty or all-empty data for patient {patient_id}. Proceeding to save report content only.")
             # No structured data update needed, but we'll still save the report content below.

    except (ValueError, RuntimeError) as e:
        logger.error(f"Error during medical report parsing for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error parsing report data: {e}")
    except Exception as e:
         logger.error(f"An unexpected error occurred while calling the parser service for patient {patient_id}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"An unexpected error occurred during parsing: {e}")


    # --- Save Report Content and Update Medical Record ---
    try:
        # 1. Save the full report content to the report_contents collection
        report_content_doc = {"content": report_content_text, "created_at": datetime.utcnow()}
        insert_content_result = await db.report_contents.insert_one(report_content_doc)
        content_id = str(insert_content_result.inserted_id)
        logger.info(f"Saved report content with ID: {content_id}")

        # 2. Create a reference to the report content
        report_ref = {
            "report_id": content_id, # Using content_id as report_id for simplicity
            "report_type": "AI Generated Report",
            "date": datetime.utcnow(),
            "content_id": content_id,
            # You might add a summary here if your AI extraction included one
            # "summary": extracted_data.get("summary") # Example
        }

        # 3. Prepare updates for the medical record document
        update_fields = {}
        update_push = {} # Use $push for appending to arrays

        # Append new items to existing lists from extracted_data
        if extracted_data.get("medications"):
             valid_meds = [m for m in extracted_data["medications"] if isinstance(m, dict)]
             if valid_meds:
                # Append each valid medication dictionary to the current_medications array
                update_push["current_medications"] = {"$each": valid_meds}


        if extracted_data.get("diagnoses"):
             valid_diagnoses = [d for d in extracted_data["diagnoses"] if isinstance(d, dict)]
             if valid_diagnoses:
                 # Append each valid diagnosis dictionary to the diagnoses array
                 update_push["diagnoses"] = {"$each": valid_diagnoses}


        if extracted_data.get("allergies"):
             valid_allergies = [a for a in extracted_data["allergies"] if isinstance(a, str) and a.strip()]
             if valid_allergies:
                # To avoid duplicates, fetch current allergies and only add new ones
                existing_allergies = medical_record_doc.get("allergies", [])
                new_allergies_to_add = [a for a in valid_allergies if a not in existing_allergies]
                if new_allergies_to_add:
                    # $addToSet adds values to an array unless they are already present
                    # This handles duplicates more cleanly than $push + $each
                    update_push["allergies"] = {"$each": new_allergies_to_add}


        if extracted_data.get("consultations"):
             valid_consultations = [c for c in extracted_data["consultations"] if isinstance(c, dict)]
             if valid_consultations:
                # Append each valid consultation dictionary
                update_push["consultation_history"] = {"$each": valid_consultations}


        if extracted_data.get("immunizations"):
             valid_immunizations = [i for i in extracted_data["immunizations"] if isinstance(i, dict)]
             if valid_immunizations:
                # Append each valid immunization dictionary
                update_push["immunizations"] = {"$each": valid_immunizations}

        # Always push the new report reference
        update_push["reports"] = report_ref

        # Also update the 'updated_at' timestamp
        update_fields["updated_at"] = datetime.utcnow()

        # Perform the update operation
        update_operation = {}
        if update_fields:
            update_operation["$set"] = update_fields
        if update_push:
            # MongoDB allows combining $set and $push in one update operation
            if "$set" not in update_operation:
                update_operation["$set"] = {} # Ensure $set exists if only $push operations are needed
            update_operation["$push"] = update_push


        update_result = await db.medical_records.update_one(
            {"patient_id": patient_id},
            update_operation # Use the combined update operation
        )

        logger.info(f"Medical record for patient {patient_id} updated. Matched: {update_result.matched_count}, Modified: {update_result.modified_count}")

        if update_result.matched_count == 0:
             logger.error(f"Medical record not found for update for patient {patient_id}. This should not happen as we created it if missing.")
             raise HTTPException(status_code=500, detail="Internal server error: Medical record not found for update.")


        # --- Return Success Response ---
        # Include the extracted data in the success response for potential frontend use
        return JSONResponse({"message": "Report data parsed and saved successfully", "content_id": content_id, "extracted_data": extracted_data})

    except Exception as e:
        logger.error(f"Error saving parsed report data to DB for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving report data: {e}")