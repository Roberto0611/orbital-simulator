from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Orbital Simulator API")

@app.get("/")
def home():
    return {"message": "API FUNCIONANDO"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "orbital-api"}