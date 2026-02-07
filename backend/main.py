from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), 'env', '.env'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
llm = None

# NeuroGlitch Backend (FastAPI with LangChain)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from models import predict_journal
from typing import Optional
# NOTE: LangChain imports are handled in the guarded block above.


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize conversation memories for session management
conversation_memories = {}

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
    global llm
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LangChain not installed. Install langchain, langchain-google-genai, and langchain-community."
        )
    if llm is None:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set.")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
    # Get or create conversation memory for this session
    session_id = request.session_id or "default"
    
    if session_id not in conversation_memories:
        conversation_memories[session_id] = []
    
    # Phase 2: DistilBERT Journal Entry Prediction

    # Refactored: Use models.py for prediction
    from models import predict_journal

    # Example journal entry
    text = "I feel like I'm drowning in my work and can't see a way out."
    predicted_class_id = predict_journal(text)
    print("Prediction:", "Glitch Detected" if predicted_class_id == 1 else "Stable")

    # Phase 3: Fine-Tuning DistilBERT on Mental Health Dataset
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Placeholder: Load your dataset (CSV with 'text' and 'label' columns)
    # Replace 'mental_health.csv' with your actual dataset file
    try:
        df = pd.read_csv('mental_health.csv')
    except Exception:
        df = pd.DataFrame({'text': [text], 'label': [predicted_class_id]})  # fallback for demo

    class MentalHealthDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length=128):
            self.dataframe = dataframe
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            text = self.dataframe.iloc[idx]['text']
            label = self.dataframe.iloc[idx]['label']
            inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            item = {key: val.squeeze(0) for key, val in inputs.items()}
            item['labels'] = torch.tensor(label)
            return item

    # Create dataset and dataloader
    dataset = MentalHealthDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop (1 epoch for demo)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Batch loss: {loss.item()}")
    model.eval()

    print("Fine-tuning complete (demo loop)")
    
    # Get response from Gemini
    try:
        history = conversation_memories[session_id]
        history.append(("user", request.message))
        prompt_lines = []
        for role, msg in history[-10:]:
            prefix = "User" if role == "user" else "Assistant"
            prompt_lines.append(f"{prefix}: {msg}")
        prompt = "\n".join(prompt_lines) + "\nAssistant:"
        result = llm.invoke(prompt)
        response = result.content if hasattr(result, "content") else str(result)
        history.append(("assistant", response))
        return {
            "response": response,
            "session_id": session_id,
            "personality": request.personality
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- Mood Tracking & Journaling ---
from fastapi import Body

# --- DistilBERT Prediction API ---
# --- DistilBERT Prediction API ---

class PredictRequest(BaseModel):
    text: str

@app.post('/api/predict')
def predict_entry(request: PredictRequest):
    try:
        predicted_class_id = predict_journal(request.text)
        risk = "High" if predicted_class_id == 1 else "Low"
        return {"risk_level": risk, "prediction": predicted_class_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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



