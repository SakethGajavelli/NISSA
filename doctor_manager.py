# doctor_manager.py
from pymongo import MongoClient
from datetime import datetime, timedelta

class DoctorManager:
    def __init__(self, connection_string, database_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.doctor_collection = self.db["doctors"]
        self.appointment_collection = self.db["appointment_data"]  # Changed to match main file

    def add_doctor(self, doctor_data):
        """Add a new doctor to the database"""
        return self.doctor_collection.insert_one(doctor_data)

    def get_all_doctors(self):
        """Get all doctors without _id field"""
        return list(self.doctor_collection.find({}, {"_id": 0}))

    def get_available_doctors(self):
        """For compatibility: returns all doctors"""
        return self.get_all_doctors()

    def get_doctor_by_name(self, name):
        """Get doctor by exact name match"""
        return self.doctor_collection.find_one({"name": name})
    
    def get_doctor_by_specialization(self, specialization):
        """Get doctors by specialization"""
        return list(self.doctor_collection.find({"specialization": specialization}, {"_id": 0}))

    def get_time_slots(self):
        """Generate available time slots (10AM to 6PM, excluding 1-2PM)"""
        slots = []
        for hour in range(10, 18):
            if hour != 13:  # skip 1–2 PM lunch break
                slots.append(f"{hour:02d}:00")
        return slots

    def get_booked_slots(self, doctor_name, date):
        """Get booked slots for a doctor on a specific date"""
        appointments = self.appointment_collection.find({
            "doctor_name": doctor_name,  # Changed to match main file field name
            "appointment_date": date     # Changed to match main file field name
        })
        return [appt["appointment_time"] for appt in appointments]

    def get_available_slots(self, doctor_name, date=None):
        """Get available slots for a doctor on a specific date"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        all_slots = self.get_time_slots()
        booked = self.get_booked_slots(doctor_name, date)
        return [slot for slot in all_slots if slot not in booked]

    def get_doctor_slots(self, doctor_name, date=None):
        """
        Returns time slots for the doctor, marking each slot as 'available' or 'busy'.
        If date is None, uses today's date.
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        all_slots = self.get_time_slots()
        booked = self.get_booked_slots(doctor_name, date)
        
        slots = []
        for slot in all_slots:
            status = "busy" if slot in booked else "available"
            slots.append({"time": slot, "status": status})
        
        return slots

    def book_appointment(self, user_name, contact, doctor_name, slot_time, date=None):
        """Book an appointment and store in appointment_data collection"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Check if slot is still available
        available_slots = self.get_available_slots(doctor_name, date)
        if slot_time not in available_slots:
            raise ValueError("Selected slot is no longer available")
        
        # Get doctor details
        doctor = self.get_doctor_by_name(doctor_name)
        if not doctor:
            raise ValueError("Doctor not found")
        
        appointment = {
            "user_name": user_name,
            "contact_number": contact,
            "doctor_name": doctor_name,
            "specialization": doctor.get("specialization", ""),
            "appointment_date": date,
            "appointment_time": slot_time,
            "status": "Booked",
            "source": "chatbot",
            "timestamp": datetime.now(),
            "created_at": datetime.now()
        }
        
        result = self.appointment_collection.insert_one(appointment)
        appointment["_id"] = str(result.inserted_id)
        return appointment

    def cancel_appointment(self, appointment_id):
        """Cancel an appointment"""
        result = self.appointment_collection.update_one(
            {"_id": appointment_id},
            {"$set": {"status": "Cancelled", "cancelled_at": datetime.now()}}
        )
        return result.modified_count > 0

    def get_user_appointments(self, user_name):
        """Get all appointments for a user"""
        return list(self.appointment_collection.find(
            {"user_name": user_name}, 
            {"_id": 0}
        ))

    def get_doctor_appointments(self, doctor_name, date=None):
        """Get all appointments for a doctor on a specific date"""
        query = {"doctor_name": doctor_name}
        if date:
            query["appointment_date"] = date
        
        return list(self.appointment_collection.find(query, {"_id": 0}))