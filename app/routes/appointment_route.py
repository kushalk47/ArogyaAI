# app/routes/appointment_route.py
from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime, timezone # Import timezone
from typing import Optional, List
from bson import ObjectId # Import ObjectId for querying other collections

# Import models
from app.models.appointment_models import Appointment # Assuming you have the updated Appointment model
from app.models.doctor_models import Doctor # Import Doctor model to fetch doctor names
from app.models.patient_models import Patient # Import Patient model (used implicitly via auth)

# Import db connection
from app.config import db # Assuming 'db' is your Motor database client instance

# Import authentication dependency
# Adjust the import path based on where your get_current_authenticated_user is defined
from app.routes.auth_routes import get_current_authenticated_user # Import the dependency

# Setup templates path (similar to auth_routes.py)
from pathlib import Path
current_file_path = Path(__file__).resolve()
routes_dir = current_file_path.parent
app_dir = routes_dir.parent
templates_dir_path = app_dir / "templates"
templates = Jinja2Templates(directory=templates_dir_path)

# Define the router
# Assuming this router will be included with a prefix like "/appointments" in __init__.py
appointment_router = APIRouter()

# --- Helper Dependency to get current *Patient* ---
# You need to ensure the authenticated user is a patient for these routes
async def get_current_patient(current_user: dict = Depends(get_current_authenticated_user)):
    """Dependency to get the current authenticated patient user document."""
    # get_current_authenticated_user returns the raw user dict or raises 401
    if current_user.get("user_type") != "patient":
        # If authenticated but not a patient, deny access
        raise HTTPException(status_code=403, detail="Only patients can access this page.")
    # Return the patient document dictionary
    return current_user
# --- End Helper Dependency ---


# --- Helper function to fetch patient's appointments with doctor names ---
async def fetch_patient_appointments_with_doctor_names(patient_id_str: str):
    """Fetches appointments for a patient and adds doctor names."""
    appointments_cursor = db.appointments.find({"patient_id": patient_id_str}).sort("appointment_time", 1) # 1 for ascending
    appointments_list_raw = await appointments_cursor.to_list(length=1000) # Fetch appointments

    appointments_with_names = []
    for appointment_doc in appointments_list_raw:
        try:
            doctor_id_str = appointment_doc.get("doctor_id")
            doctor_doc = None
            if doctor_id_str:
                 try:
                     doctor_doc = await db.doctors.find_one({"_id": ObjectId(doctor_id_str)})
                 except Exception as e:
                     print(f"Error converting doctor_id '{doctor_id_str}' to ObjectId for appointment {appointment_doc.get('_id')}: {e}")

            if doctor_doc:
                 appointment_doc["doctor_name"] = f"Dr. {doctor_doc.get('name', {}).get('first', '')} {doctor_doc.get('name', {}).get('last', '')}".strip()
            else:
                 appointment_doc["doctor_name"] = "Unknown Doctor"

            appointments_with_names.append(appointment_doc)

        except Exception as doctor_fetch_error:
             print(f"Error fetching doctor for appointment {appointment_doc.get('_id')}: {doctor_fetch_error}")
             appointment_doc["doctor_name"] = "Error Doctor Fetch"
             appointments_with_names.append(appointment_doc)

    return appointments_with_names
# --- End Helper function ---


# ---------------------- Patient Book Appointment & View Appointments Page (GET) ----------------------

# This route now handles both displaying the booking form AND the patient's appointments
@appointment_router.get("/book-appointment", response_class=HTMLResponse)
async def get_book_and_view_appointments_page(
    request: Request,
    current_patient: dict = Depends(get_current_patient)
):
    """Renders the combined book appointment and view appointments page."""
    patient_id_str = str(current_patient["_id"])

    try:
        # Fetch all doctors for the booking form
        doctors_cursor = db.doctors.find({})
        doctors_list_raw = await doctors_cursor.to_list(length=1000)

        # Fetch the patient's appointments for the list section
        patient_appointments = await fetch_patient_appointments_with_doctor_names(patient_id_str)

    except Exception as e:
        print(f"Error fetching data for combined page: {e}")
        # Render template with an error message
        return templates.TemplateResponse(
            "book_appointment.html",
            {
                "request": request,
                "error": "Could not load page data.",
                "doctors": [],
                "appointments": [], # Pass empty list if error
                "patient": current_patient
            }
        )

    # Render the book_appointment.html template, passing both lists and patient data
    return templates.TemplateResponse(
        "book_appointment.html",
        {
            "request": request,
            "doctors": doctors_list_raw,
            "appointments": patient_appointments, # Pass the patient's appointments
            "patient": current_patient
        }
    )


# ---------------------- Create Appointment (POST) ----------------------

@appointment_router.post("/book-appointment")
async def create_appointment(
    request: Request,
    current_patient: dict = Depends(get_current_patient),
    doctor_id: str = Form(...),
    appointment_date: str = Form(...),
    appointment_time: str = Form(...),
    reason: Optional[str] = Form(None),
    patient_notes: Optional[str] = Form(None),
):
    """Handles the submission of the book appointment form and re-renders the page."""
    patient_id_str = str(current_patient["_id"])

    # Combine date and time strings into a datetime object
    try:
        datetime_str = f"{appointment_date} {appointment_time}"
        appointment_dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
        appointment_time_utc = appointment_dt.replace(tzinfo=timezone.utc)

    except ValueError as e:
        print(f"Date/time parsing error: {e}")
        # Re-fetch data and re-render the page with an error message
        doctors_list_raw = await db.doctors.find({}).to_list(length=1000)
        patient_appointments = await fetch_patient_appointments_with_doctor_names(patient_id_str)
        return templates.TemplateResponse(
             "book_appointment.html",
             {
                 "request": request,
                 "error": "Invalid date or time format. Please use YYYY-MM-DD and HH:MM.", # <-- Completed error message
                 "doctors": doctors_list_raw, # Re-fetch doctors
                 "appointments": patient_appointments, # Re-fetch appointments
                 "patient": current_patient # Pass patient data back
             }
        )
    except Exception as e:
         print(f"Unexpected error during date/time processing: {e}")
         # Re-fetch data and re-render the page with an error message
         doctors_list_raw = await db.doctors.find({}).to_list(length=1000)
         patient_appointments = await fetch_patient_appointments_with_doctor_names(patient_id_str)
         return templates.TemplateResponse(
              "book_appointment.html",
              {
                  "request": request,
                  "error": "An error occurred processing the date or time.", # <-- Completed error message
                  "doctors": doctors_list_raw, # Re-fetch doctors
                  "appointments": patient_appointments, # Re-fetch appointments
                  "patient": current_patient # Pass patient data back
              }
         )


    # Create the appointment data dictionary to insert into MongoDB
    appointment_data = {
        "patient_id": patient_id_str,
        "doctor_id": doctor_id, # This should be the string ObjectId of the selected doctor
        "appointment_time": appointment_time_utc, # Store as timezone-aware datetime (UTC)
        "reason": reason,
        "patient_notes": patient_notes, # Include the new field
        "status": "Scheduled", # Default status upon creation
        "gmeet_link": None, # No GMeet link yet upon booking
        "predicted_severity": None, # No severity predicted yet upon booking
        "created_at": datetime.now(timezone.utc) # Use timezone-aware datetime
    }

    try:
        # Insert the new appointment into the database
        insert_result = await db.appointments.insert_one(appointment_data)
        if not insert_result.inserted_id:
             raise Exception("Failed to insert appointment into database.")

        # --- Trigger Asynchronous Severity Analysis ---
        # This is where you would initiate the call to your Gemini analysis service.
        # Use background tasks for this.
        # Example (using a hypothetical function trigger_severity_analysis):
        # request.app.state.background_tasks.add_task(
        #     trigger_severity_analysis,
        #     str(insert_result.inserted_id),
        #     f"{reason or ''} {patient_notes or ''}".strip()
        # )
        print(f"Appointment created with ID: {insert_result.inserted_id}. Triggering severity analysis...")
        # --- End Trigger ---

    except Exception as e:
        print(f"Database error during appointment creation: {e}")
         # Handle database errors - re-fetch data and re-render the page with error
        doctors_list_raw = await db.doctors.find({}).to_list(length=1000)
        patient_appointments = await fetch_patient_appointments_with_doctor_names(patient_id_str)
        return templates.TemplateResponse(
            "book_appointment.html",
            {
                "request": request,
                "error": f"Error booking appointment: {e}",
                "doctors": doctors_list_raw, # Re-fetch doctors
                "appointments": patient_appointments, # Re-fetch appointments
                "patient": current_patient # Pass patient data back
            }
        )

    # --- Re-render the page after successful booking ---
    # Instead of redirecting, fetch the updated list of appointments (now including the new one)
    # and re-render the same page.
    doctors_list_raw = await db.doctors.find({}).to_list(length=1000)
    patient_appointments = await fetch_patient_appointments_with_doctor_names(patient_id_str)
    return templates.TemplateResponse(
        "book_appointment.html",
        {
            "request": request,
            "success_message": "Appointment booked successfully!", # Optional: Add a success message
            "doctors": doctors_list_raw,
            "appointments": patient_appointments, # Pass the updated list
            "patient": current_patient
        }
    )
    # --------------------------------------------------------


# --- The get_patient_appointments route is no longer needed as its logic is merged into get_book_and_view_appointments_page ---
# You can remove this route definition:
# @appointment_router.get("/my-appointments", ...)
# async def get_patient_appointments(...):
#     ...


# Remember to include this router in your main.py (e.g., in app/routes/__init__.py)
# from .appointment_route import appointment_router as patient_appointment_router
# router.include_router(patient_appointment_router, prefix="/appointments", tags=["Patient Appointments"])
