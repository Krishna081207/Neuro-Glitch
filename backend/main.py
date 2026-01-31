# NeuroGlitch Backend (FastAPI placeholder)
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {"message": "NeuroGlitch backend is running."}

# Add endpoints for journaling, sentiment analysis, and alerts here.
