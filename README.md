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



#  Arogya-AI: AI-Powered Healthcare Platform 

> Arogya-AI is a scalable, secure, and modern healthcare platform that leverages Generative AI and robust backend microservices to streamline clinical workflows, patient-doctor interactions, and medical record management. [cite_start]It focuses on integrating advanced AI capabilities for real-time consultation support and automated clinical documentation[cite: 275, 276].

***

## Technology Stack

* [cite_start]**Backend Framework:** **FastAPI** (for high performance and async capabilities) [cite: 79]
* [cite_start]**Database:** **MongoDB** (used for flexible document storage) [cite: 77]
* [cite_start]**Generative AI/LLMs:** **Gemini 1.5 Flash** (for severity prediction) [cite: 88][cite_start], **Qwen 2.5B** [cite: 275][cite_start], **LangChain** [cite: 275]
* [cite_start]**Speech Recognition:** **Faster-Whisper** (for audio transcription) [cite: 93]
* [cite_start]**APIs:** **RESTful APIs** [cite: 273][cite_start], Integrated third-party healthcare APIs [cite: 274]
* [cite_start]**Deployment:** **AWS EC2** [cite: 274][cite_start], **Docker** [cite: 274][cite_start], **Render** [cite: 272]
* **Core Language:** **Python**

***

## Core Components and Architecture

The platform is structured around a multi-layered, microservice-inspired architecture, designed for scalability and clear separation of concerns.

1.  [cite_start]**Backend Services:** Built with **FastAPI** to handle high-performance, asynchronous operations[cite: 79].
2.  [cite_start]**Authentication/Sessions:** Secure user authentication for both **Patients** and **Doctors** using **hashed passwords** and **server-side session tokens** managed in the database[cite: 121, 122, 124, 131, 142].
3.  [cite_start]**Data Models:** Pydantic models define data integrity and structure for all database interactions (e.g., `Patient`, `MedicalRecord`)[cite: 175, 179].
4.  **Generative AI Integrations:**
    * [cite_start]**Symptom Triage:** Uses the **Gemini API** to predict symptom severity ('Very Serious', 'Moderate', 'Normal') during appointment booking[cite: 88, 91, 114].
    * [cite_start]**Transcription:** Uses the **Faster-Whisper** model for real-time transcription of audio consultations[cite: 93].
5.  [cite_start]**Deployment:** Deployed on **AWS EC2** with **Docker** for guaranteed **24/7 uptime**[cite: 274].

***

## API Routes Explanation

The backend routes are organized by their function (`auth_routes`, `appointment_routes`, `profile`).

### **1. Authentication Routes (`auth_routes.py`)**

Handles user creation, login, session management, and protected access.

| Method | Endpoint | Description | Key Functionality |
| :--- | :--- | :--- | :--- |
| **POST** | `/signup` | [cite_start]Registers a new patient user, creates an initial **Medical Record**, automatically logs the user in, and sets a session cookie[cite: 131, 134, 135, 137]. |
| **POST** | `/login` | Authenticates a user (Patient or Doctor). [cite_start]Creates a secure, HTTP-only session cookie upon success[cite: 142, 144]. |
| **POST** | `/logout` | [cite_start]Deletes the user session from the database and removes the session cookie from the browser[cite: 149, 207]. |
| **GET** | `/dashboard`, `/profile` | [cite_start]Protected endpoints requiring authenticated access via the session dependency (`get_current_authenticated_user`)[cite: 149]. |

### **2. Appointment Routes (`appointment_routes.py`)**

Manages the core patient workflow for booking and real-time consultation triage.

| Method | Endpoint | Description | Key Functionality |
| :--- | :--- | :--- | :--- |
| **POST** | `/book-appointment` | Processes a new appointment request. [cite_start]**Fetches the patient's medical record and uses the Gemini API to predict symptom severity** before scheduling the appointment[cite: 114, 115, 119]. |
| **GET** | `/book-appointment` | Renders the page for booking and viewing existing appointments. [cite_start]Fetches all doctors and the patient's existing appointment list[cite: 98]. |
| **POST** | `/transcribe` | Receives a patient's audio file. [cite_start]Uses the imported **Faster-Whisper** model to transcribe the audio asynchronously and returns the text transcription[cite: 92, 94]. |

### **3. Profile Route (`profile.py`)**

Retrieves a patient's complete profile and comprehensive medical history.

| Method | Endpoint | Description | Key Functionality |
| :--- | :--- | :--- | :--- |
| **GET** | `/me` | Returns the authenticated patient's full details and their **Medical Record**. [cite_start]It dynamically embeds the actual text content of reports by fetching data from the separate `report_contents` collection[cite: 164, 166, 174]. |
