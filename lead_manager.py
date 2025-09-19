# lead_manager.py
from pymongo import MongoClient
from datetime import datetime, timezone

class LeadManager:
    def __init__(self, connection_string, database_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.lead_collection = self.db["lead_data"]
        self.appointment_collection = self.db["appointment_data"]

    def save_lead_data(self, lead_data):
        """Save general lead data"""
        lead_data["created_at"] = datetime.now(timezone.utc)
        lead_data["updated_at"] = datetime.now(timezone.utc)
        return self.lead_collection.insert_one(lead_data)

    def update_lead_data(self, session_id, update_data):
        """Update existing lead data"""
        update_data["updated_at"] = datetime.now(timezone.utc)
        return self.lead_collection.update_one(
            {"session_id": session_id},
            {"$set": update_data},
            upsert=True
        )

    def get_lead_by_session(self, session_id):
        """Get lead data by session ID"""
        return self.lead_collection.find_one({"session_id": session_id})

    def save_appointment_booking(self, session_id, appointment_data):
        """Save appointment booking and update lead data"""
        # Update lead data with appointment information
        lead_update = {
            "appointment_date": appointment_data.get("appointment_date"),
            "appointment_time": appointment_data.get("appointment_time"),
            "doctor_name": appointment_data.get("doctor_name"),
            "specialization": appointment_data.get("specialization"),
            "booking_status": "confirmed"
        }
        
        # Update lead data
        self.update_lead_data(session_id, lead_update)
        
        # Also save to appointments collection
        appointment_data["session_id"] = session_id
        appointment_data["created_at"] = datetime.now(timezone.utc)
        return self.appointment_collection.insert_one(appointment_data)

    def get_all_leads(self):
        """Get all lead data"""
        return list(self.lead_collection.find({}, {"_id": 0}))

    def get_all_appointments(self):
        """Get all appointment details"""
        return list(self.appointment_collection.find({}, {"_id": 0}))

    def get_leads_with_appointments(self):
        """Get leads that have made appointments"""
        return list(self.lead_collection.find(
            {"booking_status": "confirmed"}, 
            {"_id": 0}
        ))

    def mark_lead_as_converted(self, session_id):
        """Mark lead as converted (booked appointment)"""
        return self.update_lead_data(session_id, {
            "converted": True,
            "conversion_date": datetime.now(timezone.utc)
        })