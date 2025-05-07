# app/routes/auth_routes.py
import secrets # Import secrets if needed elsewhere, though sessions.py handles token generation now
from datetime import datetime, timedelta, timezone # Added timezone
from fastapi import APIRouter, Request, Form, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.hash import bcrypt
from bson import ObjectId
from typing import Optional, List
from pathlib import Path
import logging # Added logging

# Logging setup
logger = logging.getLogger(__name__)

# Import models from the 'app.models' package
from app.models.patient_models import Patient, PatientCreate, PatientLogin, Name as PatientName, Address, EmergencyContact, MedicalRecord, Medication, Diagnosis, Prescription, Consultation, Report, Immunization
from app.models.doctor_models import Doctor, DoctorLogin, Name as DoctorName

# Import db directly from the 'app.config' module where it is defined
from app.config import db # Assuming 'db' is your Motor database client instance

# Import sessions from the 'app.models' package
# --- CORRECTED IMPORT: Ensure these constants and functions match sessions.py ---
from app.models.sessions import create_user_session, delete_user_session, get_current_session, UserSession, SESSION_COOKIE_NAME, SESSION_EXPIRATION_MINUTES
# --- END CORRECTED IMPORT ---


auth_router = APIRouter()

# --- TEMPLATES PATH ---
# Get the path of the current file (auth_routes.py is in app/routes)
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


# ---------------------- Utility Functions ----------------------

def hash_password(password: str) -> str:
    # Ensure password is treated as bytes for bcrypt hashing
    if isinstance(password, str):
        password = password.encode('utf-8')
    return bcrypt.hash(password).decode('utf-8') # Store hash as string

def verify_password(raw_password: str, hashed_password: str) -> bool:
    # Ensure hashed_password is a string; pymongo/motor might return it as bytes depending on how it was stored
    if isinstance(hashed_password, bytes):
        hashed_password = hashed_password.decode('utf-8')
    # Ensure raw_password is treated as bytes for bcrypt verification
    if isinstance(raw_password, str):
        raw_password = raw_password.encode('utf-8')
    return bcrypt.verify(raw_password, hashed_password)

# Dependency to get the current authenticated user (Patient or Doctor)
# This function fetches the user document using the user_id from the session
# It returns the raw document dictionary if found, otherwise raises HTTPException(401).
async def get_current_authenticated_user(request: Request):
    # print("\n--- Inside get_current_authenticated_user ---") # Debug print
    # Attempt to get the session from the cookie using the session module's function
    # get_current_session is responsible for checking the cookie (which now holds the token),
    # looking up the session by its *token* field, validating expiry, and potentially cleaning up expired sessions.
    session: Optional[UserSession] = await get_current_session(request)
    # print(f"Session object from get_current_session: {session}") # Debug print

    # If no valid session is found by get_current_session, authentication fails
    if not session:
        logger.debug("No valid session found by get_current_session. Raising 401.")
        raise HTTPException(
            status_code=401,
            detail="Not authenticated: No valid session found.",
            headers={"WWW-Authenticate": "Bearer"}, # Optional: Suggest Bearer auth scheme
        )

    # If a session is found, try to fetch the corresponding user document
    user_id_str = session.user_id # Get the user_id (string ObjectId) from the session object
    user_doc = None
    logger.debug(f"Session found. User ID from session: {user_id_str}, User Type: {session.user_type}")

    # Attempt to find the user in the patients collection
    if session.user_type == "patient":
        try:
            logger.debug(f"Attempting to find patient with _id: {user_id_str}")
            # Query by the string ID. MongoDB driver handles conversion to ObjectId
            # Ensure the user_id stored in session is the string ObjectId of the user document
            user_doc = await db.patients.find_one({"_id": ObjectId(user_id_str)})
            logger.debug(f"Patient document found: {user_doc is not None}")

        except Exception as e:
            # Log error if ObjectId conversion or DB query fails
            logger.error(f"Error fetching patient {user_id_str}: {e}")
            user_doc = None # Ensure user_doc is None on error


    # If not found as a patient, attempt to find in the doctors collection
    if not user_doc and session.user_type == "doctor":
        try:
            logger.debug(f"Attempting to find doctor with _id: {user_id_str}")
            user_doc = await db.doctors.find_one({"_id": ObjectId(user_id_str)})
            logger.debug(f"Doctor document found: {user_doc is not None}")
        except Exception as e:
            # Log error if ObjectId conversion or DB query fails
            logger.error(f"Error fetching doctor {user_id_str}: {e}")
            user_doc = None # Ensure user_doc is None on error


    # If a session was found but the user document is missing (e.g., user deleted)
    # The dependency should just raise 401. Cookie cleanup happens in get_current_session.
    if not user_doc:
        logger.warning(f"User document not found for session user_id {user_id_str}. Session might be invalid.")
        # Removed cookie deletion from dependency - get_current_session handles detection and cleanup
        raise HTTPException(
            status_code=401,
            detail="Not authenticated: User not found or session invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # If user document is found, return it
    logger.debug("User document found. Authentication successful.")
    return user_doc


# ---------------------- Signup Routes (Two Steps) ----------------------

@auth_router.get("/signup", response_class=HTMLResponse)
async def get_signup(request: Request):
    # Check if user is already logged in
    session = await get_current_session(request)
    if session:
        # Redirect based on user type if already logged in
        if session.user_type == "doctor":
            return RedirectResponse("/dashboard/", status_code=303)
        else: # Assume patient or other types redirect to profile
            return RedirectResponse("/profile/", status_code=303)


    return templates.TemplateResponse("signup.html", {"request": request})

@auth_router.post("/signup")
async def post_signup(
    request: Request,
    response: Response, # Need response here to set the cookie
    # ... (form parameters remain the same) ...
    first: str = Form(...),
    middle: Optional[str] = Form(None),
    last: str = Form(...),
    email: str = Form(...),
    phone_number: str = Form(...),
    password: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    street: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    zip: str = Form(...),
    country: str = Form(...),
    emergency_name: str = Form(...),
    emergency_phone: str = Form(...),
    emergency_relationship: str = Form(...),
    current_medications_text: Optional[str] = Form(None),
    diagnoses_text: Optional[str] = Form(None),
    prescriptions_text: Optional[str] = Form(None),
    consultation_history_text: Optional[str] = Form(None),
    reports_text: Optional[str] = Form(None),
    allergies_text: Optional[str] = Form(None),
    immunizations_text: Optional[str] = Form(None),
    family_medical_history: Optional[str] = Form(None)
):
    # Check if user already exists
    existing_patient = await db.patients.find_one({"email": email})
    existing_doctor = await db.doctors.find_one({"email": email}) # Also check doctor emails
    if existing_patient or existing_doctor:
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "error": "Email already registered."
        })

    patient_oid = ObjectId()
    patient_id_str = str(patient_oid) # Get string representation of the user's ObjectId

    hashed_pw = hash_password(password)

    patient_data = {
        "_id": patient_oid, # Store ObjectId in the user document
        "name": {"first": first, "middle": middle, "last": last},
        "email": email,
        "phone_number": phone_number,
        "password": hashed_pw,
        "age": age,
        "gender": gender,
        "address": {"street": street, "city": city, "state": state, "zip": zip, "country": country},
        "emergency_contact": {"name": emergency_name, "phone": emergency_phone, "relationship": emergency_relationship},
        "registration_date": datetime.now(timezone.utc), # Use timezone-aware datetime
        "user_type": "patient" # Explicitly store user type
    }

    try:
        insert_result = await db.patients.insert_one(patient_data)
        if not insert_result.inserted_id:
            raise Exception("Failed to insert patient")
        logger.info(f"Patient created with _id: {insert_result.inserted_id}")
    except Exception as e:
        logger.error(f"Database error during patient creation: {e}")
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Error saving patient details. Please try again."})

    # --- Create Initial Medical Record (Step 2 Data) ---
    medical_record_data = {
        "_id": ObjectId(),
        "patient_id": patient_id_str, # Link to the patient using their string ObjectId
        "current_medications": [m.strip() for m in current_medications_text.split(',') if m.strip()] if current_medications_text else [], # Parse comma-separated strings into lists
        "diagnoses": [d.strip() for d in diagnoses_text.split(',') if d.strip()] if diagnoses_text else [],
        "prescriptions": [p.strip() for p in prescriptions_text.split(',') if p.strip()] if prescriptions_text else [],
        "consultation_history": [c.strip() for c in consultation_history_text.split(',') if c.strip()] if consultation_history_text else [],
        "reports": [r.strip() for r in reports_text.split(',') if r.strip()] if reports_text else [],
        "allergies": [a.strip() for a in allergies_text.split(',') if a.strip()] if allergies_text else [],
        "immunizations": [i.strip() for i in immunizations_text.split(',') if i.strip()] if immunizations_text else [],
        "family_medical_history": family_medical_history,
    }
    try:
        await db.medical_records.insert_one(medical_record_data)
        logger.info(f"Medical record created for patient ID: {patient_id_str}")
    except Exception as e:
        logger.error(f"Error saving medical record for patient {patient_id_str}: {e}")
        # Consider deleting the patient document if medical record creation fails to avoid orphaned records


    # --- Automatic Login after Successful Signup ---
    # Call create_user_session to create the session document and get the secure random token back
    try:
        # create_user_session now returns the random session token
        session_token = await create_user_session(user_id=patient_id_str, user_type="patient")
        logger.info(f"Session created after signup for user {patient_id_str}. Token (first 8 chars): {session_token[:8]}...")
    except Exception as e:
        logger.error(f"Error creating session after signup for user {patient_id_str}: {e}")
        # Redirect to login with error if session creation fails
        return RedirectResponse("/auth/login?error=Signup successful but failed to create session.", status_code=303)

    # Create the redirect response - Redirect directly to profile after successful signup and login
    redirect_response = RedirectResponse("/profile/", status_code=303)

    # Set the cookie directly on the redirect response object
    # Use the session_token returned by create_user_session as the cookie value
    redirect_response.set_cookie(
        key=SESSION_COOKIE_NAME, # Use the constant from sessions.py
        value=session_token,     # Use the returned token as the cookie value
        httponly=True,           # Corrected typo
        max_age=SESSION_EXPIRATION_MINUTES * 60, # max_age in seconds
        path="/",
        # --- CORRECTED: Set secure=True ONLY if the request is HTTPS ---
        secure=request.url.scheme == "https",
        # -------------------------------------------------------------
        samesite="Lax"           # Recommended for CSRF protection
    )

    # Return the redirect response with the cookie set
    return redirect_response


# ---------------------- Login Routes ----------------------

@auth_router.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    # Check if user is already logged in
    session = await get_current_session(request)
    if session:
        # Redirect based on user type if already logged in
        if session.user_type == "doctor":
            return RedirectResponse("/dashboard/", status_code=303)
        else: # Assume patient or other types redirect to profile
            return RedirectResponse("/profile/", status_code=303)

    # Get any query parameters like signup_success or error
    signup_success = request.query_params.get("signup_success")
    error_message = request.query_params.get("error")


    return templates.TemplateResponse("login.html", {
        "request": request,
        "signup_success": signup_success is not None, # Pass boolean indicating success
        "error": error_message # Pass error message if present
    })

@auth_router.post("/login")
async def post_login(
    request: Request,
    response: Response, # Need response here to set the cookie
    email: str = Form(...),
    password: str = Form(...)
):
    user_doc = None
    user_type = None
    user_id_str = None

    # Attempt to find patient
    patient_doc = await db.patients.find_one({"email": email})
    if patient_doc and verify_password(password, patient_doc.get("password")):
        user_doc = patient_doc
        user_type = "patient"
        user_id_str = str(user_doc["_id"]) # Get string ObjectId of the user document


    # If not patient, attempt to find doctor
    if not user_doc:
        doctor_doc = await db.doctors.find_one({"email": email})
        # Assuming 'password' field exists in doctor document in DB
        if doctor_doc and verify_password(password, doctor_doc.get("password")):
            user_doc = doctor_doc
            user_type = "doctor"
            user_id_str = str(user_doc["_id"]) # Get string ObjectId of the user document


    if not user_doc:
        # Invalid credentials - redirect back to login with error message
        logger.warning(f"Failed login attempt for email: {email}")
        return RedirectResponse("/auth/login?error=Invalid email or password.", status_code=303)


    # --- Successful Login ---
    # Call create_user_session to create the session document and get the secure random token back
    try:
        # create_user_session now returns the random session token
        session_token = await create_user_session(user_id=user_id_str, user_type=user_type)
        logger.info(f"Session created after login for user {user_id_str}. Token (first 8 chars): {session_token[:8]}...")
    except Exception as e:
        logger.error(f"Error creating session after login for user {user_id_str}: {e}")
        # Log error, redirect to login, maybe show a specific error message
        return RedirectResponse("/auth/login?error=Login successful but failed to create session.", status_code=303)

    # --- Determine Redirect based on user_type ---
    redirect_url = "/profile/" # Default for patients
    if user_type == "doctor":
        redirect_url = "/dashboard/" # Redirect doctors to the dashboard

    # Create the redirect response
    redirect_response = RedirectResponse(redirect_url, status_code=303)

    # Set the cookie directly on the redirect response object
    # Use the session_token returned by create_user_session as the cookie value
    redirect_response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        max_age=SESSION_EXPIRATION_MINUTES * 60,
        path="/",
        secure=request.url.scheme == "https",
        samesite="Lax"
    )

    # Return the redirect response with the cookie set
    return redirect_response


# ---------------------- Logout ----------------------

@auth_router.get("/logout")
async def logout(request: Request, response: Response):
    # delete_user_session handles deleting the session from DB and the cookie
    await delete_user_session(request, response)
    logger.info("User logged out.")
    # Redirect to login page
    return RedirectResponse("/auth/login", status_code=303)


# Example of a protected route (requires session)
# @auth_router.get("/dashboard")
# async def dashboard(request: Request, current_user: dict = Depends(get_current_authenticated_user)):
#     # get_current_authenticated_user will raise 401 if not authenticated
#     # current_user is the raw user document (dict)
#     user_name = current_user.get("name", {}).get("first", "User") # Accessing nested dict
#     user_type = current_user.get("user_type", "Unknown") # Assuming user_type is stored in the user document or handle patient/doctor structure difference

#     # Example of rendering a template
#     return templates.TemplateResponse("dashboard.html", {
#         "request": request,
#         "user_name": user_name,
#         "user_type": user_type,
#         "user_doc": current_user # Pass the full user document if needed
#     })

# Example of a route that checks authentication but doesn't require it (Optional)
# @auth_router.get("/some-page")
# async def some_page(request: Request, current_user: Optional[dict] = Depends(get_current_authenticated_user)):
#     # current_user will be None if not authenticated, or the user doc if logged in
#     if current_user:
#         logger.debug(f"User {current_user.get('email')} is logged in.")
#     else:
#         logger.debug("User is not logged in.")
#     return templates.TemplateResponse("some_page.html", {"request": request, "user": current_user}) # Pass user info to template
