from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pymongo import MongoClient
from doctor_manager import DoctorManager
from lead_manager import LeadManager
import secrets
import json
import time
import requests
from requests.auth import HTTPDigestAuth
import glob
from pathlib import Path
import uuid
import logging
import traceback
from google.cloud import texttospeech

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_urlsafe(16))

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "./data/")
ASSETS_FOLDER = os.path.join("public", "assets")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
ATLAS_PUBLIC_KEY = os.getenv("ATLAS_PUBLIC_KEY")
ATLAS_PRIVATE_KEY = os.getenv("ATLAS_PRIVATE_KEY")
ATLAS_GROUP_ID = os.getenv("ATLAS_GROUP_ID")
ATLAS_CLUSTER_NAME = os.getenv("ATLAS_CLUSTER_NAME")
DATABASE_NAME = "Gemini"
INDEX_NAME = "vector_index"

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client.Gemini
chat_collection = db.chat_history
lead_collection = db.lead_data
appointment_collection = db["appointment_data"]
collection_name = "customer_data"

doctor_manager = DoctorManager(MONGODB_URI, DATABASE_NAME)
lead_manager = LeadManager(MONGODB_URI, DATABASE_NAME)

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini models
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=LLM_TEMPERATURE,
    google_api_key=GOOGLE_API_KEY
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

# Lead extraction model with lower temperature for precision
lead_extraction_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

# Prompt templates
CONTEXT_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """
You are Nisaa – the smart virtual assistant on this website. Follow these instructions precisely.

I. 🎯 Purpose:
- Answer user questions based **strictly on the provided context**
- Keep responses concise: a short summary or **2–3 friendly, natural lines**
- Guide users to book expert consultations (doctor appointments) OR general service meetings — keep these flows separate

II. 🗣️ Tone & Style:
- Warm, professional, and emotionally intelligent
- Always human-like, never robotic or overly technical
- Never salesy or pushy
- Avoid long messages unless summarizing is essential

III. 💬 First Message (Greeting):
"Hi, this is Nisaa! It's nice to meet you here. How can I assist you today?"

IV. 🔄 Two Distinct Booking Flows:

**A. General Service Meeting Flow:**
If the user says:
- "I want to book a meeting"
- "Schedule a call"
- "Talk to someone about services"
- "Book a consultation" (without mentioning doctor/medical)

→ This is a **general service meeting**:
1. Ask for name in separate message
2. Ask for email in separate message
3. Ask for phone number in separate message
4. Ask "Which service are you interested in?"
5. Save to lead_data with type="service_meeting"

**B. Doctor Consultation Flow:**
If the user says:
- "Book a doctor consultation"
- "I want to meet Dr. [name]"
- "Book an appointment with [specialization]"
- "Medical consultation"

→ This is a **doctor appointment**:
1. Show list of available doctors with specialization
2. After user selects doctor: Show time slots (10 AM - 6 PM, excluding 1-2 PM lunch)
3. Mark slots as: ✅ Available / ❌ Busy
4. If user selects available time:
   - Ask for name (if not already collected)
   - Ask for phone number
5. Confirm: "Your appointment with Dr. [Name] on [Date] at [Time] is confirmed."
6. Save to appointment_data with fields: doctor_name, specialization, appointment_date, appointment_time, user_name, contact_number, status=Booked, source=chatbot

**C. Unclear Intent:**
If unclear whether they mean service meeting or doctor appointment:
→ Ask: "Just to confirm — do you mean a general meeting about our services, or a doctor consultation?"
→ Continue only after user confirms.

V. 🔄 Lead Capture Rules (General Flow):
1. Never ask for personal details in the first two replies to a new topic
2. If user asks about a service:
   - Give clear, friendly explanation in 2–3 lines
   - Do **not** ask for their name in the same message
3. In the **next** message, ask separately: "By the way, may I know your name? It's always nice to help you personally."
4. Always ask for user details (name, email, phone) in **separate messages only**, *after* the response is completed
5. If user declines to share name, respond kindly:
   - "No problem at all! I'm here to help with anything you need."
   - "That's totally fine. Let me know how I can assist you further."
   - "I understand! We can still continue — just tell me what you'd like to know."
6. Once name is shared, use it naturally in future responses

VI. 📧 Contact Info Collection (General Flow):
7. Ask for email first: "Would you like me to email this information to you? What's the best email address for you?"
8. If they decline, ask gently once more: "Are you sure you don't want me to email it? It can be helpful to keep a copy."
9. If declined again, do not ask again (unless for booking)
10. If email is shared, ask for phone number: "Also, may I have your phone number in case our team needs to follow up?"

VII. 💡 CTA Hooks (use **only after name is shared** and **only for general flow**):
- "Would you like help choosing the right service?"
- "Want to see how others use this?"
- "Shall I walk you through a real example?"
- "Would you like to try a demo of this?"
- "Interested in seeing how this helped other clients?"

VIII. 🎥 360° View Handling:
A. For general 360° requests:
- Say: "Sure! A 360° virtual tour lets you explore our space as if you're really there — many clients love using it to build trust and increase engagement."
- Then ask: "Would you like to check it out?"
- If yes, reply: "Here's the 360° virtual tour. Tap a view below to explore:"
  - Main Lobby: /virtualtour.html?scene=main-lobby
  - Conference Room: /virtualtour.html?scene=conference-room
  - Office: /virtualtour.html?scene=office
  - Entry: /virtualtour.html?scene=entry
  - Studio: /virtualtour.html?scene=studio

B. For specific room requests:
- Say: "Sure! Here's the 360° view of the {{room name}} — feel free to explore it:"
- Then link that scene only.

❗ Never ask for personal details in any 360° response  
❗ Never include booking CTAs in 360° responses

IX. 🔁 General Fallbacks:
- Repeated question: "Let me explain that again, no worries."
- Silence: "Still there? I'm right here if you need anything."
- Goodbye: "It's been a pleasure! Come back anytime."
- Unclear: "I didn't catch that — could you rephrase?"

X. 🔄 Topic Continuity:
- Do **not** move to lead capture, suggest other topics, or ask questions until the **current user request is answered clearly**
- Transition only if the user is satisfied, changes topic, or becomes inactive
- Always finish the current flow (general or doctor) before changing topics

XI. 📝 Message Format:
- Use natural 2–3 line responses unless summarizing
- Use bullet points for services or lists
- Ask for each personal detail (name, email, phone) in separate messages
- No external links except 360° tour links
- Do not use emojis (unless the user uses them first)

XII. 🤝 Investor Inquiries:
If someone mentions investing:
- Say: "That's great to hear! We truly appreciate your interest in investing in mTouchLabs. For investment discussions, I recommend connecting with our senior team — would you like their contact info?"

XIII. 💼 Services Questions (Dynamic):
If the user asks:
- "What services do you provide?"
- "Tell me about your services"
- "What do you offer?"

→ Then fetch the service list from backend or MongoDB using `/get_services`.
Say: "Sure! Here are some of our key services:"
- [List of services fetched dynamically]
Then add: "Let me know which service you'd like to explore more!"

XIV. ⚠️ Important Rules:
- **Never mix general service meetings with doctor appointments**
- **Always confirm intent if unclear**
- **Complete one flow before starting another**
- **Keep doctor appointments strictly medical/consultation focused**
- **Keep service meetings for business/general inquiries**

---

Context: {context}  
Chat History: {chat_history}  
Question: {input}  

Answer (strictly based on context, in short summary or 2–3 friendly lines. Distinguish between general service meetings and doctor consultations. Use CTA hooks only after name is known and only for general flow. Never ask personal info in 360° replies):
"""

LEAD_EXTRACTION_PROMPT = """Extract the following information from the conversation if available:
- name
- email_id  
- contact_number
- location
- service_interest
- appointment_date
- appointment_time
- doctor_name

Return ONLY a valid JSON object with these fields with NO additional text before or after.  
If information isn't found, leave the field empty.

Do not include any explanatory text, notes, or code blocks. Return ONLY the raw JSON.

Conversation: {conversation}"""

# Create prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXT_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Chat history management
chat_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

# Atlas Search Index management functions
def create_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.atlas.2024-05-30+json'}
    data = {
        "collectionName": collection_name,
        "database": DATABASE_NAME,
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine"}
            ]
        }
    }
    response = requests.post(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY), 
        data=json.dumps(data)
    )
    if response.status_code != 201:
        raise Exception(f"Failed to create Atlas Search Index: {response.status_code}, Response: {response.text}")
    return response

def get_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.get(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

def delete_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.delete(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

def load_multiple_documents():
    """Load multiple PDF and text files from the documents folder"""
    documents = []
    
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    
    pdf_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.pdf"))
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_file'] = os.path.basename(pdf_file)
            doc.metadata['file_type'] = 'pdf'
        documents.extend(docs)
    
    txt_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.txt"))
    for txt_file in txt_files:
        print(f"Loading text file: {txt_file}")
        loader = TextLoader(txt_file, encoding='utf-8')
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_file'] = os.path.basename(txt_file)
            doc.metadata['file_type'] = 'txt'
        documents.extend(docs)
    
    if not documents:
        raise FileNotFoundError(f"No PDF or text files found in: {DOCUMENTS_FOLDER}")
    
    print(f"Loaded {len(documents)} documents from {len(pdf_files + txt_files)} files")
    return documents

def initialize_vector_store():
    docs = load_multiple_documents()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    final_documents = text_splitter.split_documents(docs)
    
    print(f"Split into {len(final_documents)} chunks")
    
    response = get_atlas_search_index()
    if response.status_code == 200:
        print("Deleting existing Atlas Search Index...")
        delete_response = delete_atlas_search_index()
        if delete_response.status_code == 204:
            print("Waiting for index deletion to complete...")
            while get_atlas_search_index().status_code != 404:
                time.sleep(5)
        else:
            raise Exception(f"Failed to delete existing Atlas Search Index: {delete_response.status_code}, Response: {delete_response.text}")
    elif response.status_code != 404:
        raise Exception(f"Failed to check Atlas Search Index: {response.status_code}, Response: {response.text}")
    
    db[collection_name].delete_many({})
    
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=final_documents,
        embedding=embeddings,
        collection=db[collection_name],
        index_name=INDEX_NAME,
    )
    
    doc_count = db[collection_name].count_documents({})
    print(f"Number of documents in {collection_name}: {doc_count}")
    if doc_count > 0:
        sample_doc = db[collection_name].find_one()
        print(f"Sample document structure (keys): {list(sample_doc.keys())}")
    
    print("Creating new Atlas Search Index...")
    create_response = create_atlas_search_index()
    print(f"Atlas Search Index creation status: {create_response.status_code}")
    
    print("Waiting for index to be ready...")
    time.sleep(30)
    
    return vector_search

def extract_message_pairs(msg_entries):
    flat = []
    for entry in msg_entries:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            flat.append(entry)
        elif isinstance(entry, list):
            for subentry in entry:
                if isinstance(subentry, dict) and "role" in subentry and "content" in subentry:
                    flat.append(subentry)
    return flat

def extract_lead_info(session_id):
    chat_doc = chat_collection.find_one({"session_id": session_id})
    if not chat_doc or "messages" not in chat_doc:
        return
    
    flat_messages = extract_message_pairs(chat_doc.get("messages", []))
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in flat_messages])
    
    try:
        response = lead_extraction_llm.invoke(LEAD_EXTRACTION_PROMPT.format(conversation=conversation))
        response_text = response.content.strip()
        
        if "```json" in response_text or "```" in response_text:
            import re
            json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
        
        try:
            lead_data = json.loads(response_text)
            print(f"Successfully parsed lead data: {lead_data}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Gemini response: {response_text}")
            print(f"JSON error: {str(e)}")
            
            import re
            json_pattern = r'\{[^}]*"name"[^}]*"email_id"[^}]*"contact_number"[^}]*"location"[^}]*"service_interest"[^}]*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                try:
                    lead_data = json.loads(json_match.group(0))
                    print(f"Extracted JSON using regex: {lead_data}")
                except json.JSONDecodeError:
                    lead_data = {
                        "name": "",
                        "email_id": "",
                        "contact_number": "",
                        "location": "",
                        "service_interest": "",
                        "appointment_date": "",
                        "appointment_time": "",
                        "parsing_error": "Failed to parse response"
                    }
            else:
                lead_data = {
                    "name": "",
                    "email_id": "",
                    "contact_number": "",
                    "location": "",
                    "service_interest": "",
                    "appointment_date": "",
                    "appointment_time": "",
                    "raw_response": response_text[:500]
                }
        
        lead_data["session_id"] = session_id
        lead_data["updated_at"] = datetime.now(timezone.utc)
        lead_data["extraction_model"] = "gemini_" + GEMINI_MODEL
        
        lead_collection.update_one(
            {"session_id": session_id},
            {"$set": lead_data},
            upsert=True
        )
    except Exception as e:
        print(f"[Lead Extraction Error] {e}")

# Translation function for 360° view responses
def translate_360_response(text, language):
    language_map = {
        'en': 'en-US',
        'te': 'te-IN',
        'hi': 'hi-IN',
        'ta': 'ta-IN',
        'mr': 'mr-IN'
    }
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_map.get(language, 'en-US'),
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        logger.error(f"Translation/TTS error for {language}: {str(e)}")
        return text

# Initialize vector store
try:
    vector_search = initialize_vector_store()
    print("Vector store initialized successfully")
except Exception as e:
    print(f"Failed to initialize vector store: {e}")
    raise

client = MongoClient("mongodb+srv://raising100x:vNb3t4WLQZKMN2OZ@cluster0.v5haryq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["Gemini"]
lead_collection = db["lead_data"]
appointment_collection = db["appointment_data"]

def confirm_appointment(lead_id):
    lead = lead_collection.find_one({"_id": lead_id})
    
    if lead:
        appointment_doc = {
            "message": f"Appointment confirmed for {lead.get('name', '')}",
            "response": "Your appointment is booked successfully.",
            "timestamp": datetime.utcnow(),
            "source": "chat",
            "lead_id": str(lead["_id"]),
            "doctor_name": lead.get("doctor_name", ""),
            "appointment_date": lead.get("appointment_date", ""),
            "appointment_time": lead.get("appointment_time", "")
        }
        
        appointment_collection.insert_one(appointment_doc)
        return True
    return False

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_session', methods=['GET'])
def generate_session():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})

# Add this to handle appointment booking flow
class BookingState:
    """Class to manage booking state for each session"""
    states = {}
    
    @classmethod
    def set_state(cls, session_id, state, data=None):
        cls.states[session_id] = {"state": state, "data": data or {}}
    
    @classmethod
    def get_state(cls, session_id):
        return cls.states.get(session_id, {"state": "none", "data": {}})
    
    @classmethod
    def clear_state(cls, session_id):
        if session_id in cls.states:
            del cls.states[session_id]

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        language = data.get('language', 'en')

        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        user_input_lower = user_input.lower()

        chat_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": [{
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now(timezone.utc)
                    }]
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)}
            },
            upsert=True
        )

        # Get current booking state
        booking_state = BookingState.get_state(session_id)
        current_state = booking_state["state"]
        state_data = booking_state["data"]

        # Handle appointment booking flow
        if current_state == "none" and any(keyword in user_input_lower for keyword in ["book", "appointment", "consultation", "schedule"]):
            doctors = doctor_manager.get_all_doctors()
            if not doctors:
                response_text = "I'm sorry, no doctors are available at the moment. Please try again later."
            else:
                response_text = "Here are our available doctors. Please select one:\n\n"
                for i, doctor in enumerate(doctors, 1):
                    response_text += f"{i}. **Dr. {doctor['name']}** - {doctor['specialization']}\n"
                    response_text += f"   Experience: {doctor.get('experience', 'N/A')}\n"
                    response_text += f"   Fees: {doctor.get('fees', 'Contact for fees')}\n\n"
                response_text += "Please reply with the doctor's name or number to continue."
                
                BookingState.set_state(session_id, "doctor_selection", {"doctors": doctors})

            return log_and_respond(session_id, response_text, "doctor_list", {"doctors": doctors})

        elif current_state == "doctor_selection":
            selected_doctor = None
            doctors = state_data.get("doctors", [])
            
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(doctors):
                    selected_doctor = doctors[idx]
            else:
                for doctor in doctors:
                    if doctor["name"].lower() in user_input_lower or user_input_lower in doctor["name"].lower():
                        selected_doctor = doctor
                        break

            if not selected_doctor:
                response_text = "Please select a valid doctor by name or number from the list above."
                return log_and_respond(session_id, response_text)

            today = datetime.now().strftime("%Y-%m-%d")
            slots = doctor_manager.get_doctor_slots(selected_doctor["name"], today)
            
            available_slots = [slot for slot in slots if slot["status"] == "available"]
            
            if not available_slots:
                response_text = f"Sorry, Dr. {selected_doctor['name']} has no available slots today. Would you like to check another doctor or try tomorrow?"
                BookingState.set_state(session_id, "doctor_selection", {"doctors": doctors})
                return log_and_respond(session_id, response_text)

            response_text = f"Great! You selected **Dr. {selected_doctor['name']}** ({selected_doctor['specialization']}).\n\n"
            response_text += f"Available slots for today ({today}):\n\n"
            
            for i, slot in enumerate(available_slots, 1):
                response_text += f"{i}. {slot['time']}\n"
            
            response_text += "\nPlease select a time slot by number or time."
            
            BookingState.set_state(session_id, "slot_selection", {
                "doctor": selected_doctor,
                "slots": available_slots,
                "date": today
            })

            return log_and_respond(session_id, response_text, "time_slots", {"slots": available_slots})

        elif current_state == "slot_selection":
            selected_slot = None
            slots = state_data.get("slots", [])
            
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(slots):
                    selected_slot = slots[idx]
            else:
                for slot in slots:
                    if slot["time"] in user_input or user_input in slot["time"]:
                        selected_slot = slot
                        break

            if not selected_slot:
                response_text = "Please select a valid time slot from the list above."
                return log_and_respond(session_id, response_text)

            doctor_name = state_data["doctor"]["name"]
            current_slots = doctor_manager.get_doctor_slots(doctor_name, state_data["date"])
            slot_still_available = any(s["time"] == selected_slot["time"] and s["status"] == "available" for s in current_slots)
            
            if not slot_still_available:
                response_text = f"Sorry, the {selected_slot['time']} slot is no longer available. Please select another slot."
                available_slots = [slot for slot in current_slots if slot["status"] == "available"]
                BookingState.set_state(session_id, "slot_selection", {
                    **state_data,
                    "slots": available_slots
                })
                return log_and_respond(session_id, response_text)

            response_text = f"Perfect! You've selected {selected_slot['time']} with Dr. {doctor_name}.\n\n"
            response_text += "To confirm your appointment, I need a few details:\n\n"
            response_text += "What's your full name?"
            
            BookingState.set_state(session_id, "collect_name", {
                **state_data,
                "selected_slot": selected_slot
            })

            return log_and_respond(session_id, response_text)

        elif current_state == "collect_name":
            name = user_input.strip()
            if len(name) < 2:
                response_text = "Please provide your full name."
                return log_and_respond(session_id, response_text)

            response_text = f"Thank you, {name}! Now, please provide your contact number."
            
            BookingState.set_state(session_id, "collect_contact", {
                **state_data,
                "user_name": name
            })

            return log_and_respond(session_id, response_text)

        elif current_state == "collect_contact":
            contact = user_input.strip()
            
            import re
            if not re.match(r'^[+]?[\d\s\-\(\)]{10,15}$', contact):
                response_text = "Please provide a valid contact number (10-15 digits)."
                return log_and_respond(session_id, response_text)

            try:
                doctor = state_data["doctor"]
                selected_slot = state_data["selected_slot"]
                date = state_data["date"]
                user_name = state_data["user_name"]

                appointment = doctor_manager.book_appointment(
                    user_name=user_name,
                    contact=contact,
                    doctor_name=doctor["name"],
                    slot_time=selected_slot["time"],
                    date=date
                )

                lead_manager.save_appointment_booking(session_id, {
                    "user_name": user_name,
                    "contact_number": contact,
                    "doctor_name": doctor["name"],
                    "specialization": doctor["specialization"],
                    "appointment_date": date,
                    "appointment_time": selected_slot["time"],
                    "status": "Booked",
                    "source": "chatbot"
                })

                lead_manager.mark_lead_as_converted(session_id)

                response_text = f"🎉 **Appointment Confirmed!**\n\n"
                response_text += f"**Patient:** {user_name}\n"
                response_text += f"**Doctor:** Dr. {doctor['name']} ({doctor['specialization']})\n"
                response_text += f"**Date:** {date}\n"
                response_text += f"**Time:** {selected_slot['time']}\n"
                response_text += f"**Contact:** {contact}\n\n"
                response_text += "You will receive a confirmation call shortly. Thank you for choosing our services!"

                BookingState.clear_state(session_id)

                return log_and_respond(session_id, response_text, "booking_confirmed", {
                    "appointment": appointment
                })

            except Exception as e:
                logger.error(f"Booking error: {str(e)}")
                response_text = f"Sorry, there was an error booking your appointment: {str(e)}\nPlease try again or contact us directly."
                BookingState.clear_state(session_id)
                return log_and_respond(session_id, response_text)

        # Handle 360° Scenes
        scene_mappings = [
            {"keywords": ["new office interior", "interior of new office"], "scene": "NEW-OFFICE-INSIDE", "image": "/GoToNewOfficeInterior.JPG", "description": "new office interior"},
            {"keywords": ["complete place", "main entry", "entry"], "scene": "ENTRY", "image": "/backToMainEntry.jpg", "description": "main entry"},
            {"keywords": ["office room"], "scene": "ROOM1", "image": "/office.jpg", "description": "office room"},
            {"keywords": ["admin block", "administration block"], "scene": "ADMIN-BLOCK", "image": "/adminblock.jpg", "description": "admin block"},
            {"keywords": ["meeting room", "conference room"], "scene": "MEETING-ROOM", "image": "/meeting.jpg", "description": "meeting room"},
            {"keywords": ["workspace", "working area", "place where they work"], "scene": "WORKSPACE", "image": "/workspace.jpg", "description": "workspace"},
            {"keywords": ["new office"], "scene": "NEW-OFFICE", "image": "/officeroom.jpg", "description": "new office"},
            {"keywords": ["studio entrance", "outside studio"], "scene": "STUDIO-OUTSIDE", "image": "/office-6.jpg", "description": "studio entrance"},
            {"keywords": ["studio", "recording studio"], "scene": "STUDIO", "image": "/office-16.jpg", "description": "studio"}
        ]

        if any(keyword in user_input_lower for keyword in ['360', 'view', 'tour', 'show']):
            logger.info(f"360° view request detected. Input: {user_input_lower}")
            matched_scene = None
            for scene in sorted(scene_mappings, key=lambda x: max(len(k) for k in x["keywords"]), reverse=True):
                for keyword in scene["keywords"]:
                    if keyword in user_input_lower:
                        matched_scene = scene
                        logger.info(f"Matched keyword: {keyword}, Scene: {scene['scene']}")
                        break
                if matched_scene:
                    break

            if matched_scene:
                base_response = f"Here's the 360° view of the {matched_scene['description']}!"
                response_data = {
                    "response": translate_360_response(base_response, language) if language != 'en' else base_response,
                    "type": "360_view",
                    "url": f"http://localhost:3001/panorama?scene={matched_scene['scene']}",
                    "label": translate_360_response(f"Explore the {matched_scene['description']} in 360°", language) if language != 'en' else f"Explore the {matched_scene['description']} in 360°",
                    "image": matched_scene['image'],
                    "target": "_self"
                }
            else:
                logger.info("No specific scene matched. Using default ENTRY scene.")
                base_response = "Here's the 360° view of our office!"
                response_data = {
                    "response": translate_360_response(base_response, language) if language != 'en' else base_response,
                    "type": "360_view",
                    "url": f"http://localhost:3001/panorama?scene=ENTRY",
                    "label": translate_360_response("Explore the office in 360°", language) if language != 'en' else "Explore the office in 360°",
                    "image": "/office-15.jpg",
                    "target": "_self"
                }

            chat_collection.update_one(
                {"session_id": session_id},
                {"$push": {
                    "messages": [{
                        "role": "assistant",
                        "content": response_data["response"],
                        "timestamp": datetime.now(timezone.utc)
                    }]
                }},
                upsert=True
            )

            logger.info(f"Sending 360° view response: url={response_data['url']}, target={response_data['target']}")
            return jsonify(response_data), 200

        # Default RAG fallback for other queries
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = vector_search.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.0}
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        answer = response.get("answer", "").strip()
        if not answer:
            raise ValueError("LLM returned empty response.")

        chat_doc = chat_collection.find_one({"session_id": session_id}) or {}
        if len(chat_doc.get("messages", [])) >= 4:
            extract_lead_info(session_id)

        return log_and_respond(session_id, answer, "standard")

    except Exception as e:
        logger.error("Chat error: %s", str(e))
        traceback.print_exc()
        return jsonify({
            "response": "Sorry, an error occurred processing your request.",
            "type": "error"
        }), 200

# Helper function to log and respond
def log_and_respond(session_id, response_text, response_type="standard", extra_data=None):
    """Helper function to log assistant message and return response"""
    chat_collection.update_one(
        {"session_id": session_id},
        {"$push": {
            "messages": [{
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now(timezone.utc)
            }]
        }},
        upsert=True
    )

    response_data = {
        "response": response_text,
        "type": response_type
    }
    
    if extra_data:
        response_data.update(extra_data)

    return jsonify(response_data), 200


@app.route('/leads', methods=['GET'])
def get_leads():
    leads = list(lead_collection.find({}, {"_id": 0}))
    return jsonify(leads)

@app.route("/doctors", methods=["GET"])
def get_doctors():
    specialization = request.args.get("specialization")
    if specialization:
        doctors = doctor_manager.get_doctor_by_specialization(specialization)
    else:
        doctors = doctor_manager.get_all_doctors()
    return jsonify(doctors)

@app.route("/doctor-slots", methods=["GET"])
def get_doctor_slots():
    doctor_name = request.args.get("name")
    if not doctor_name:
        return jsonify({"error": "Doctor name required"}), 400
    slots = doctor_manager.get_doctor_slots(doctor_name)
    return jsonify({"name": doctor_name, "slots": slots})

@app.route("/book", methods=["POST"])
def book_consultation():
    data = request.json
    required_fields = ["user_name", "contact", "doctor", "time"]
    
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    doctor_name = data["doctor"]
    slot_time = data["time"]
    slots = doctor_manager.get_doctor_slots(doctor_name)

    match = next((slot for slot in slots if slot["time"] == slot_time and slot["status"] == "available"), None)
    if not match:
        return jsonify({"error": "Slot not available"}), 400

    doctor_manager.book_appointment(data["user_name"], data["contact"], doctor_name, slot_time)
    return jsonify({"message": f"Appointment confirmed with {doctor_name} at {slot_time}."})

@app.route('/upload_documents', methods=['POST'])
def upload_documents():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.txt')):
                continue
            
            filename = file.filename
            file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            file.save(file_path)
            uploaded_files.append(filename)
        
        if uploaded_files:
            global vector_search
            vector_search = initialize_vector_store()
            return jsonify({
                'message': f'Successfully uploaded and processed {len(uploaded_files)} files',
                'files': uploaded_files
            }), 200
        else:
            return jsonify({'error': 'No valid PDF or text files uploaded'}), 400
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    
@app.route('/appointments', methods=['GET'])
def get_appointments():
    appointments = list(appointment_collection.find({}, {"_id": 0}))
    return jsonify(appointments)

@app.route('/leads/<lead_id>', methods=['DELETE'])
def delete_lead(lead_id):
    try:
        # Check if lead_id is undefined or not a string
        if not lead_id or not isinstance(lead_id, str):
            return jsonify({"error": "Invalid lead ID format"}), 400
        
        if not ObjectId.is_valid(lead_id):
            return jsonify({"error": "Invalid lead ID format"}), 400
        
        result = lead_collection.delete_one({"_id": ObjectId(lead_id)})
        if result.deleted_count == 1:
            logger.info(f"Lead {lead_id} deleted successfully")
            return jsonify({"message": "Lead deleted successfully"}), 200
        return jsonify({"error": "Lead not found"}), 404
    except InvalidId:
        return jsonify({"error": "Invalid lead ID format"}), 400
    except Exception as e:
        logger.error(f"Error deleting lead {lead_id}: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/appointments/<appointment_id>', methods=['DELETE'])
def delete_appointment(appointment_id):
    try:
        result = appointment_collection.delete_one({"_id": ObjectId(appointment_id)})
        if result.deleted_count == 1:
            return jsonify({"message": "Appointment deleted successfully"}), 200
        return jsonify({"error": "Appointment not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/leads/<lead_id>', methods=['PUT'])
def edit_lead(lead_id):
    try:
        data = request.json
        result = lead_collection.update_one(
            {"_id": ObjectId(lead_id)},
            {"$set": data}
        )
        if result.matched_count == 1:
            updated_lead = lead_collection.find_one({"_id": ObjectId(lead_id)}, {"_id": 0})
            return jsonify({"message": "Lead updated successfully", "data": updated_lead}), 200
        return jsonify({"error": "Lead not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/appointments/<appointment_id>', methods=['PUT'])
def edit_appointment(appointment_id):
    try:
        data = request.json
        result = appointment_collection.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": data}
        )
        if result.matched_count == 1:
            updated_appointment = appointment_collection.find_one({"_id": ObjectId(appointment_id)}, {"_id": 0})
            return jsonify({"message": "Appointment updated successfully", "data": updated_appointment}), 200
        return jsonify({"error": "Appointment not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        client.admin.command('ping')
        doc_count = db[collection_name].count_documents({})
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'documents_indexed': doc_count,
            'model': GEMINI_MODEL
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('public', filename, as_attachment=False)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        abort(404)
        

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)