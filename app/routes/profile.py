# app/routes/profile.py
from fastapi import APIRouter, Request, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, List # Import List
from bson import ObjectId, errors # Import ObjectId and errors for validation
from pathlib import Path # Import Path
from datetime import datetime # Import datetime for type checking
import logging # Import logging

# Configure logging
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

# Import db from the config module
from app.config import db

# Import the authentication dependency
# Assuming get_current_authenticated_user is defined in auth_routes.py
from .auth_routes import get_current_authenticated_user

# Import models for type hinting (optional but good practice)
from app.models.patient_models import Patient, MedicalRecord, Report, ReportContent # Import Report and ReportContent
from app.models.doctor_models import Doctor # To check if the user is a doctor

profile_router = APIRouter()

# --- TEMPLATES PATH ---
# Get the path of the current file (profile.py is in app/routes)
current_file_path = Path(__file__).resolve()
# Go up one level to the 'routes' directory
routes_dir = current_file_path.parent
# Go up another level to the 'app' directory
app_dir = routes_dir.parent
# Construct the path to the 'templates' directory inside 'app'
templates_dir_path = app_dir / "templates"

# Initialize Jinja2Templates with the correct path
templates = Jinja2Templates(directory=templates_dir_path)
# --- END TEMPLATES PATH ---


@profile_router.get("/")
async def get_profile(
    request: Request,
    # Use the dependency to get the currently authenticated user document
    current_user_doc: dict = Depends(get_current_authenticated_user)
):
    """
    Displays the patient's profile, showing basic or medical details.
    Requires authentication.
    """
    # Check if the authenticated user is a patient
    # We rely on the 'user_type' field stored in the user document
    is_patient = current_user_doc.get("user_type") == "patient"

    if not is_patient:
        # If the logged-in user is not a patient (e.g., a doctor), redirect or show an error
        # Doctors might have their own profile page or dashboard.
        if current_user_doc.get("user_type") == "doctor":
             # If they are a doctor, redirect to doctor dashboard
            return RedirectResponse("/dashboard/", status_code=303)
        else:
             # For any other unhandled type or if user_type is missing, redirect to login
             # A more specific error page could also be shown
            raise HTTPException(status_code=403, detail="Access denied. Only patients can view this profile.")


    # --- Fetch Patient Details ---
    # The current_user_doc is already the patient's document fetched by the dependency.
    # Create a copy to avoid modifying the document returned by the dependency directly
    patient_details = current_user_doc.copy()
    # Ensure the patient's _id is a string for template use
    patient_details['_id'] = str(patient_details['_id'])


    # --- Fetch Medical Record ---
    # The patient's _id is stored as ObjectId in the document.
    # We stored the patient_id in the medical record as a string ObjectId.
    patient_id_str = patient_details["_id"] # Use the string _id

    # Find the medical record for this patient
    # Assuming the medical record is linked by patient_id (string ObjectId)
    medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id_str})

    # --- Fetch Report Contents ---
    # If a medical record exists and has reports, fetch their content
    if medical_record_doc and medical_record_doc.get("reports"):
        # Create a copy of the medical record document before modifying its reports list
        medical_record_for_template = medical_record_doc.copy()
        updated_reports = []
        for report_ref in medical_record_for_template["reports"]: # Iterate over the copy's reports
            # Ensure report_ref is a dictionary and has 'content_id'
            if isinstance(report_ref, dict) and report_ref.get("content_id"):
                try:
                    # Fetch the content document using the content_id
                    content_oid = ObjectId(report_ref["content_id"])
                    report_content_doc = await db.report_contents.find_one({"_id": content_oid})

                    if report_content_doc and report_content_doc.get("content"):
                        # Embed the content back into the report object
                        # Create a copy of the report reference before adding description
                        report_with_content = report_ref.copy()
                        report_with_content["description"] = report_content_doc["content"] # Add the content as 'description'
                        updated_reports.append(report_with_content)
                    else:
                        logger.warning(f"Report content not found for content_id: {report_ref['content_id']}")
                        # Optionally, include the report reference even if content is missing
                        # report_ref['description'] = 'Content not available.'
                        # updated_reports.append(report_ref)
                        # Or just skip this report if content is essential
                        pass # Skip reports where content isn't found
                except errors.InvalidId:
                    logger.warning(f"Invalid content_id format in report reference: {report_ref.get('content_id')}")
                    pass # Skip reports with invalid content_id
                except Exception as e:
                    logger.error(f"Error fetching report content for content_id {report_ref.get('content_id')}: {e}")
                    pass # Skip reports where fetching fails
            else:
                 logger.warning(f"Invalid report reference format found: {report_ref}")
                 pass # Skip invalid report references

        # Replace the original reports list in the copied document with the updated list
        medical_record_for_template["reports"] = updated_reports
    else:
         # If no medical record or no reports, just use the original medical_record_doc (which might be None)
         medical_record_for_template = medical_record_doc


    # Pass data to the template
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "patient": patient_details, # Pass the patient document (copy)
        "medical_record": medical_record_for_template # Pass the medical record document (copy or original None)
    })

# You might add other profile-related routes here, e.g., for editing profile,
# adding/updating medical records, etc.
# Example:
# @profile_router.post("/medical-records/add")
# async def add_medical_record_item(
#     request: Request,
#     current_user_doc: dict = Depends(get_current_authenticated_user),
#     # Define Pydantic models for incoming medical data
#     medication_data: Medication = Body(...) # Example for adding medication
# ):
#     # Ensure user is patient
#     is_patient = current_user_doc.get("user_type") == "patient" or ("age" in current_user_doc and "address" in current_user_doc)
#     if not is_patient:
#         raise HTTPException(status_code=403, detail="Access denied.")

#     patient_id_str = str(current_user_doc["_id"])

#     # Find the medical record or create a new one if it doesn't exist
#     medical_record_doc = await db.medical_records.find_one({"patient_id": patient_id_str})

#     if not medical_record_doc:
#          # Create a new medical record document if none exists
#          medical_record_doc = {
#              "_id": ObjectId(),
#              "patient_id": patient_id_str,
#              "current_medications": [],
#              # ... initialize other lists ...
#          }
#          await db.medical_records.insert_one(medical_record_doc)

#     # Add the new medication to the list
#     # You'll need to convert the Pydantic model to a dictionary for MongoDB
#     medication_dict = medication_data.model_dump(mode='json')

#     await db.medical_records.update_one(
#         {"_id": medical_record_doc["_id"]},
#         {"$push": {"current_medications": medication_dict}} # Use $push to add to the list
#     )

#     return {"message": "Medication added successfully"}
