from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from defense2D import Simulation2D
from simulator3D import mainKepler3D
from simulator2D import mainKepler2D
from defense3D import DefenseSimulation3D

app = FastAPI(title="Orbital Simulator API")

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

@app.get("/defensaDatos2D")
def get_simulation_data_2d():
    output_data = Simulation2D()
    return output_data

@app.get("/DefensaDatos3D")
def get_defense_simulation_data_3d():
    output_data = DefenseSimulation3D()
    return output_data

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "orbital-api"}