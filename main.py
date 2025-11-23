from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from simulator import mainKepler

app = FastAPI(title="Orbital Simulator API")

@app.get("/")
def home():
    return {"message": "API FUNCIONANDO"}

@app.get("/datos")
def get_simulation_data():
    output_data = mainKepler()
    return output_data

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "orbital-api"}