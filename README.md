# ⚕️ Arogya-AI: AI-Powered Healthcare Platform 🤖

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
