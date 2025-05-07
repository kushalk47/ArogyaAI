from fastapi import APIRouter, HTTPException
from app.models.appointment_models import Appointment
from config.database import appointment_collection
from bson import ObjectId

router = APIRouter()

@router.post("/")
async def book_appointment(appointment: Appointment):
    result = await appointment_collection.insert_one(appointment.dict())
    return {"id": str(result.inserted_id)}

@router.get("/{appointment_id}")
async def get_appointment(appointment_id: str):
    appointment = await appointment_collection.find_one({"_id": ObjectId(appointment_id)})
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    appointment["_id"] = str(appointment["_id"])
    return appointment
