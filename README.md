# Orbital Simulator API

## Inicio Rápido

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Instalar dependencias  
pip install -r requirements.txt

# 4. Ejecutar servidor
uvicorn main:app --reload --port 8001

# 5. Abrir en navegador
# http://localhost:8001/docs