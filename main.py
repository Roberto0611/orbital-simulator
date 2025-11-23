from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
from simulator2D import mainKepler2D
from simulator3D import mainKepler3D

app = FastAPI(title="Orbital Simulator API")

# Configurar CORS para Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API FUNCIONANDO"}

@app.get("/datos2D")
def get_simulation_data_2d():
    output_data = mainKepler2D()
    return output_data

@app.get("/datos3D")
def get_simulation_data_3d():
    output_data = mainKepler3D()
    return output_data

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "orbital-api"}