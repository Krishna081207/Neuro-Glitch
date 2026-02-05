# NeuroGlitch Backend (FastAPI placeholder)
from fastapi import FastAPI, Request, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), 'env', '.env'))

API_KEY = os.getenv('API_KEY')
SECURITY_KEY = os.getenv('SECURITY_KEY')

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Root ---
@app.get('/')
def read_root():
    return {"message": "NeuroGlitch backend is running."}

# --- Conversational AI ---
class ChatRequest(BaseModel):
    message: str
    personality: str
    session_id: Optional[str] = None

@app.post('/api/chat')
def chat_endpoint(request: ChatRequest):
    # Placeholder: Integrate AI model here
    return {"response": f"AI ({request.personality}) says: You said '{request.message}'", "session_id": request.session_id}

# --- Mood Tracking & Journaling ---
class MoodEntry(BaseModel):
    mood: int  # 1-5 scale
    journal: Optional[str] = None
    sentiment: Optional[str] = None
    timestamp: Optional[str] = None

mood_db = []  # In-memory placeholder

@app.post('/api/mood')
def add_mood(entry: MoodEntry):
    mood_db.append(entry.dict())
    return {"status": "saved", "entry": entry.dict()}

@app.get('/api/mood')
def get_moods():
    return {"moods": mood_db}

# --- Self-Help Tools ---
@app.get('/api/yoga')
def yoga_recommendation(mood: int):
    # Placeholder: Recommend yoga poses based on mood
    return {"poses": ["Mountain", "Child's Pose"], "mood": mood}

@app.get('/api/focus')
def focus_session(duration: int = 25):
    # Placeholder: Return focus session info
    return {"duration": duration, "quote": "Stay focused!"}

@app.get('/api/breathing')
def breathing_exercise():
    return {"pattern": "4-7-8", "instructions": "Inhale 4s, hold 7s, exhale 8s"}

@app.get('/api/coping')
def coping_cards():
    return {"cards": ["Grounding", "Mindfulness", "Movement", "Reflection", "Creativity"]}

@app.get('/api/wellness')
def wellness_tips():
    return {"tips": {"mind": "Meditate daily", "body": "Exercise", "nutrition": "Eat veggies", "sleep": "Regular bedtime", "stress": "Take breaks"}}

# --- Resource & Crisis Help ---
@app.get('/api/hotlines')
def crisis_hotlines():
    return {"hotlines": ["911", "1-800-273-8255"]}

class SymptomRequest(BaseModel):
    symptoms: list

@app.post('/api/doctor')
def doctor_recommender(request: SymptomRequest):
    # Placeholder: Recommend specialist
    return {"specialist": "Psychiatrist", "symptoms": request.symptoms}

@app.get('/api/quizzes')
def mental_health_quizzes():
    return {"quizzes": ["GAD-7", "PHQ-9", "WHO-5"]}

# --- Authentication & User Profile ---
class AuthRequest(BaseModel):
    username: str
    password: str

@app.post('/api/login')
def login(request: AuthRequest):
    # Placeholder: Accept any username/password
    return {"status": "logged_in", "user": request.username}

@app.post('/api/oauth')
def oauth_login(provider: str):
    # Placeholder: Accept any provider
    return {"status": "oauth_success", "provider": provider}

@app.get('/api/guest')
def guest_access():
    return {"status": "guest", "features": "limited"}

class UserProfile(BaseModel):
    name: str
    picture_url: Optional[str] = None
    font_size: Optional[int] = 16

user_profiles = {}

@app.post('/api/profile')
def set_profile(profile: UserProfile):
    user_profiles[profile.name] = profile.dict()
    return {"status": "profile_saved", "profile": profile.dict()}

@app.get('/api/profile/{username}')
def get_profile(username: str):
    profile = user_profiles.get(username)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"profile": profile}



