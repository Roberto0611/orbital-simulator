from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from simulator2D import mainKepler2D

app = FastAPI(title="Orbital Simulator API")

@app.get("/")
def home():
    return {"message": "API FUNCIONANDO"}

@app.get("/datos2D")
def get_simulation_data():
    output_data = mainKepler2D()
    return output_data

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "orbital-api"}