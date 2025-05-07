from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Appointment(BaseModel):
    patient_id: str = Field(..., description="Reference to the patient (ObjectId)")
    doctor_id: str = Field(..., description="Reference to the doctor (ObjectId)")
    appointment_time: datetime = Field(..., description="Scheduled date and time of the appointment")
    reason: Optional[str] = Field(None, description="Reason for the appointment")
    status: str = Field(default="Scheduled", description="Status: Scheduled, Completed, Cancelled, etc.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Appointment creation timestamp")
