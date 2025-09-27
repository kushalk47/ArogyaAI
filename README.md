
---

# Arogya AI – AI-powered Healthcare Platform

> Hi, I'm Kushal K. This repository contains an AI-powered healthcare **web application** built with **FastAPI, MongoDB, and Gemini API integration**. It allows patients to register, book appointments, and manage health records, while doctors can log in to view patient data, generate summaries, and assist in consultations.

---

## Core Components

* [File Directory Structure](#file-directory-structure)
* [Instructions to Run the Code](#instructions-to-run-the-code)
* [Initialization (`__init__.py`)](#initialization-__init__py)
* [Application Entrypoint (`main.py`)](#application-entrypoint-mainpy)
* [Routes (`routes/`)](#routes-routes)
* [Models (`models/`)](#models-models)
* [Configuration (`config.py`)](#configuration-configpy)
* [AI & Gemini API Integration](#ai--gemini-api-integration)
* [Templates & Static Files](#templates--static-files)
* [Scalability and Enhancements](#scalability-and-enhancements)

---

## File Directory Structure

```text
ArogyaAI/
├── static/
│   ├── css/
│   └── js/
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── doctor_login.html
│   ├── patient_register.html
│   ├── book_appointment.html
│   ├── records.html
│   └── consultation.html
├── routes/
│   ├── patient_routes.py
│   ├── doctor_routes.py
│   └── ai_routes.py
├── models/
│   ├── patient.py
│   ├── doctor.py
│   └── appointment.py
├── __init__.py
├── main.py
├── config.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Technology Stack

* **Framework:** FastAPI
* **Database:** MongoDB (NoSQL)
* **Authentication & Security:** JWT-based authentication, session handling
* **Deployment:** Docker + AWS EC2 / Render
* **Frontend:** HTML (Jinja2 templates), Bootstrap
* **Core Language:** Python
* **GenAI Integration:** Gemini API (for summarization, chatbot, consultation assistance)

---

## Project Architecture

```mermaid
flowchart TD
    Patient[Patient User] -->|Registers/Books| FastAPI[FastAPI Backend]
    Doctor[Doctor User] -->|Login/Dashboard| FastAPI
    FastAPI --> MongoDB[(MongoDB Database)]
    FastAPI --> GeminiAPI[Gemini API - LLM Tasks]
    GeminiAPI --> Summarizer[Medical Summarization]
    GeminiAPI --> Chatbot[Consultation Support]
    FastAPI --> Templates[HTML Templates + Static Files]
```

This application integrates a **web interface** (Jinja2 templates) with **Gemini API** for AI-driven tasks such as consultation assistance, record summarization, and prescription drafting.

---

## In-Depth Component Explanations

### Initialization (`__init__.py`)

* Creates the FastAPI app.
* Connects to **MongoDB**.
* Registers routes (patients, doctors, AI).
* Configures middleware (sessions, error handling).

### Application Entrypoint (`main.py`)

* Runs the FastAPI server with Uvicorn.
* Renders HTML templates instead of raw JSON responses.
* Swagger UI is available but secondary since this is a **web app**.

### Routes (`routes/`)

* **patient_routes.py** → Handles patient registration, booking, and record management (renders `patient_register.html`, `book_appointment.html`, etc.).
* **doctor_routes.py** → Handles doctor login and dashboards (renders `doctor_login.html`, `dashboard.html`).
* **ai_routes.py** → Provides AI-assisted features (summarization, chatbot responses) using Gemini API.

### Models (`models/`)

* **Patient** → Stores patient profile and history.
* **Doctor** → Stores doctor login and specialization.
* **Appointment** → Manages scheduling and status.

### Configuration (`config.py`)

* Loads MongoDB URI, Gemini API key, and secret keys from `.env`.

### AI & Gemini API Integration

* **Gemini API** replaces local LLMs for deployment.
* Supports **summarization of health records, chatbot consultation, and prescription drafting**.
* Lightweight integration ensures it runs on cloud without heavy GPU requirements.

### Templates & Static Files

* **templates/** → HTML pages for doctors and patients.
* **static/** → CSS, JS, and assets.
* Uses `base.html` for template inheritance.

---

## Scalability and Enhancements

* **Scalability**:

  * Built on FastAPI (async-ready).
  * MongoDB allows horizontal scaling.
  * AI tasks are offloaded to Gemini API.

* **Enhancements Suggested**:

  * Add role-based access for admin/doctor/patient.
  * Secure sensitive medical records (HIPAA/GDPR compliance).
  * Add appointment reminders with email/SMS APIs.
  * Extend AI consultation to multilingual support.

---

## Instructions to Run the Code

### Prerequisites

* [Python 3.9+](https://www.python.org/downloads/) installed.
* Virtual environment (`venv`) created.
* MongoDB instance (local or Atlas).
* Gemini API key.

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kushalk47/arogya-ai
   cd arogya-ai
   ```

2. **Create Virtual Environment & Install Dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables**
   Create a `.env` file:

   ```
   MONGO_URI=mongodb+srv://user:password@cluster/dbname
   SECRET_KEY=your_secret_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **Run the Application**

   ```bash
   uvicorn main:app --reload
   ```

5. **Access the Application**

   * User homepage: **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**
   * Patient registration: `/patient/register`
   * Doctor login: `/doctor/login`
   * Dashboard: `/dashboard`

---

✨ With this setup, patients can book appointments and manage records, while doctors can use AI-assisted dashboards powered by Gemini API for efficient consultations.

---

Do you also want me to include **sample screenshots (patient booking, doctor dashboard, AI consultation)** section in the README for better presentation?

This entire system is a robust, modern **AI-powered Backend Healthcare Platform** built on the **FastAPI** framework, using **MongoDB** for data storage and leveraging external APIs (**Google Gemini**) and local models (**Faster-Whisper**) for core clinical functionality.

The key innovation is the seamless integration of Generative AI for automating tasks like symptom triage and voice transcription within a secure, multi-user (Patient/Doctor) API structure.

-----

## Project Directory Structur

The project follows a standard Python package structure, which makes it scalable and organized:

```
HEALTHCARE_FINAL/
├── app                                 <-- Core application code
│   ├── models                          <-- Pydantic models defining data structures
│   │   ├── admin_models.py
│   │   ├── appointment_models.py
│   │   ├── doctor_models.py
│   │   ├── medical_records_models.py
│   │   ├── patient_models.py
│   │   └── sessions.py                 <-- Session Pydantic model
│   ├── routes                          <-- API endpoint logic
│   │   ├── appointment_routes.py       <-- Appointment booking, Gemini Triage, Whisper Transcription
│   │   ├── auth_routes.py              <-- Sign-up, Login, Logout, Authentication Dependency
│   │   ├── doctor_routes.py            <-- (Presumably) Doctor-specific routes and Whisper model initialization
│   │   ├── home_routes.py
│   │   ├── medical_record_routes.py
│   │   ├── patient_routes.py
│   │   └── profile.py                  <-- User profile retrieval (with embedded medical record)
│   ├── static
│   ├── templates
│   ├── config.py
│   ├── database.py
│   └── main.py                         <-- Main FastAPI instance
├── venv
├── run.py                              <-- Root file to start the application
└── requirements.txt
```

-----

## Core System Functionality

The application is engineered around three main pillars: **Security, AI Integration, and Data Integrity.**

### 1\. Secure Authentication (`auth_routes.py`, `sessions.py`)

  * **Hashing:** User passwords are secured using **bcrypt hashing** upon sign-up.
  * **Session Management:** The system uses a **DB-backed session token** approach.
      * `sessions.py` handles the creation and lookup of a secure, random session token in the database.
      * The session is stored in an HTTP-only cookie (`SESSION_COOKIE_NAME`) after successful login/signup, enhancing security.
  * **Protection:** The `get_current_authenticated_user` dependency is the gatekeeper for all protected routes (like `/dashboard`, `/profile`). It reads the cookie, validates the session in the DB, checks for expiration, and ensures the user document exists.
  * **Sign-up Workflow:** The `/signup` endpoint not only creates a user in `db.patients` but also immediately creates their empty or initial **Medical Record** in `db.medical_records`, ensuring data integrity from the start.

### 2\. AI-Assisted Triage and Transcription (`appointment_route.py`)

  * **Symptom Severity Prediction (Triage):**
      * The core `create_appointment` endpoint uses the **Gemini 1.5 Flash API** (via the helper function `predict_symptom_severity`).
      * The AI is prompted with the patient's entire **Medical Record** (diagnoses, medications, reports, etc.) along with the reason for the visit and notes.
      * The AI returns a single prediction: **'Very Serious', 'Moderate', or 'Normal'**, which is saved to the new appointment document.
  * **Voice Transcription:**
      * The `/transcribe` endpoint receives an audio file.
      * It uses the **Faster-Whisper model** (imported from `doctor_routes.py`) to convert the audio to text.
      * Crucially, it uses **`run_in_threadpool`** to execute the heavy transcription task, preventing the main FastAPI event loop from being blocked and ensuring the server remains responsive.

### 3\. Patient Profile and Data Retrieval (`profile.py`)

  * **Data Aggregation:** The `/me` endpoint is responsible for retrieving the entire patient view.
  * **Report Content Embedding:** The system stores the actual long text content of medical reports in a separate collection (`db.report_contents`) for performance. The `/me` route actively queries this content by `content_id` and embeds the full text back into the patient's `medical_record` before sending the final JSON response. This provides a complete, single-API-call view of the patient's data.

-----

## Data Schemas (`patient_models.py`)

The `patient_models.py` file defines the Pydantic schemas that enforce data types and structure across the entire application, including:

  * **`Patient` and `PatientCreate`:** Defines required fields for a user, including **Name**, **Address**, and **EmergencyContact**.
  * **`MedicalRecord`:** Defines the central medical history container, which includes lists of structured sub-models like `Medication`, `Diagnosis`, `Prescription`, and **`Report`**.
  * **`Report` and `ReportContent`:** The `Report` model contains a `content_id` (the MongoDB `ObjectId`), which acts as the reference key to the actual report text stored in the `ReportContent` model's collection.
  * **API Request Models:** Also defines schemas for AI-related requests, such as `ChatRequest` and `ReportRequest`.



