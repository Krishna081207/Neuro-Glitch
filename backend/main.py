# NeuroGlitch Backend (FastAPI placeholder)
from fastapi import FastAPI, Request
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

@app.get('/')
def read_root():
    return {"message": "NeuroGlitch backend is running."}


# Endpoint for creating a new training job/model
from pydantic import BaseModel
from typing import Optional

class TrainingJobRequest(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_url: str
    model_type: str

@app.post('/train')
def create_training_job(request: TrainingJobRequest):
    # Placeholder: In a real app, trigger training pipeline here
    return {
        "status": "submitted",
        "job": request.dict()
    }

@app.post("/api/analyze")
async def analyze(request: Request):
    data = await request.json()
    # Placeholder: process data, use PyTorch/tensor, etc.
    # Return a mock response for now
    return {"status": "success", "message": "Data received", "data": data}
