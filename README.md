# Arogya AI – AI-powered Healthcare Platform

> Hi, I'm Kushal K. This repository contains an AI-powered healthcare platform built with **FastAPI, MongoDB, LangChain, and Generative AI models**. The platform provides patient management, appointment scheduling, medical record retrieval, and AI-assisted features such as real-time consultation, health record summarization, and prescription generation.

---

## Core Components

* [File Directory Structure](#file-directory-structure)
* [Instructions to Run the Code](#instructions-to-run-the-code)
* [Initialization (`__init__.py`)](#initialization-__init__py)
* [Application Entrypoint (`main.py`)](#application-entrypoint-mainpy)
* [Routes (`routes/`)](#routes-routes)
* [Models (`models/`)](#models-models)
* [Configuration (`config.py`)](#configuration-configpy)
* [AI & GenAI Integration](#ai--genai-integration)
* [Templates & Static Files](#templates--static-files)
* [Example API Usage](#example-api-usage)
* [Scalability and Enhancements](#scalability-and-enhancements)

---

## File Directory Structure

```text
ArogyaAI/
├── static/
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
* **Deployment:** Docker + AWS EC2
* **Frontend:** HTML (Jinja2 templates), Bootstrap for styling
* **Core Language:** Python
* **GenAI Integration:** Whisper, Qwen 2.5B, LangChain

---

## Project Architecture

```mermaid
flowchart TD
    Patient[Patient User] -->|Registers/Books| FastAPI[FastAPI Backend]
    Doctor[Doctor User] -->|Login/Dashboard| FastAPI
    FastAPI --> MongoDB[(MongoDB Database)]
    FastAPI --> GenAI[Generative AI Modules]
    GenAI --> Whisper[Speech-to-Text]
    GenAI --> Qwen[Qwen 2.5B Model]
    GenAI --> LangChain[LangChain Pipelines]
    FastAPI --> Templates[HTML Templates + Static Files]
```

The platform integrates **traditional backend APIs** with **Generative AI modules** for real-time patient–doctor interaction and health data insights.

---

## In-Depth Component Explanations

### Initialization (`__init__.py`)

* Creates the FastAPI app.
* Connects to **MongoDB**.
* Loads routes (patient, doctor, AI).
* Configures middleware (CORS, sessions, error handling).

### Application Entrypoint (`main.py`)

* Starts the FastAPI server.
* Provides auto-generated Swagger UI for API testing.
* Runs with Uvicorn for production deployment.

### Routes (`routes/`)

* **patient_routes.py** → Patient registration, booking appointments, fetching records.
* **doctor_routes.py** → Doctor login, viewing dashboards, accessing patient records.
* **ai_routes.py** → AI endpoints for summarization, speech-to-text, prescription generation.

### Models (`models/`)

MongoDB document models:

* **Patient**: Stores patient profile, health data, history.
* **Doctor**: Stores doctor credentials, specialization, appointments.
* **Appointment**: Links patients with doctors, tracks schedule and status.

### Configuration (`config.py`)

* Loads MongoDB connection string, API keys, and environment variables.
* Ensures secrets are read from `.env` file.

### AI & GenAI Integration

* **Whisper**: Converts doctor’s speech to text for prescriptions.
* **Qwen 2.5B**: Handles NLP tasks such as summarization, triage responses.
* **LangChain**: Orchestrates multi-step AI workflows (chatbot, record summarization).

### Templates & Static Files

* **templates/** → Frontend pages for patients and doctors.
* **static/** → CSS, JS, and assets.
* `base.html` used for template inheritance.

---

## Example API Usage

### 1. Register a Patient

```http
POST /patient/register
Content-Type: application/json

{
  "name": "John Doe",
  "age": 32,
  "contact": "9876543210",
  "history": "Diabetic"
}
```

### 2. Book an Appointment

```http
POST /appointment/book
Content-Type: application/json

{
  "patient_id": "64a91cfa8d2f",
  "doctor_id": "64a91f43d9e1",
  "date": "2025-09-30",
  "time": "10:30 AM"
}
```

### 3. AI Summarization of Records

```http
POST /ai/summarize
Content-Type: application/json

{
  "record_text": "Patient has recurring headaches, prescribed paracetamol..."
}
```

Response:

```json
{
  "summary": "Patient experiences frequent headaches, currently treated with paracetamol."
}
```

### 4. Speech-to-Text (Doctor Prescription)

```http
POST /ai/speech-to-text
Content-Type: multipart/form-data
(audio_file uploaded)
```

Response:

```json
{
  "transcription": "Prescribe 500mg Paracetamol twice daily after meals."
}
```

---

## Scalability and Enhancements

* **Scalability**:

  * Built with FastAPI, async by default → high performance.
  * MongoDB ensures horizontal scaling for patient/record data.
  * AI components are modular and can be containerized separately.

* **Enhancements Suggested**:

  * Add role-based access for admin/doctor/patient.
  * Integrate real-time notifications via WebSockets.
  * Add analytics dashboard with ML-driven health predictions.
  * HIPAA/GDPR compliance for production deployment.

---

## Instructions to Run the Code

### Prerequisites

* [Python 3.9+](https://www.python.org/downloads/) installed.
* Virtual environment (`venv`) created.
* MongoDB instance (local or Atlas).
* Docker (optional, for deployment).

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
   Create a `.env` file in the root folder:

   ```
   MONGO_URI=mongodb+srv://user:password@cluster/dbname
   SECRET_KEY=your_secret_key
   ```

4. **Run the Application**

   ```bash
   uvicorn main:app --reload
   ```

5. **Access the Application**

   * User homepage: **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**
   * API Docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---
This entire system is a robust, modern **AI-powered Backend Healthcare Platform** built on the **FastAPI** framework, using **MongoDB** for data storage and leveraging external APIs (**Google Gemini**) and local models (**Faster-Whisper**) for core clinical functionality.

The key innovation is the seamless integration of Generative AI for automating tasks like symptom triage and voice transcription within a secure, multi-user (Patient/Doctor) API structure.

-----

## Project Directory Structure

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



