# Orbital Simulator API# Orbital Simulator API



API de simulación orbital 2D y 3D construida con FastAPI.## Inicio Rápido



## 🚀 Despliegue en Railway```bash

# 1. Crear entorno virtual

### Pasos para desplegar:python -m venv venv



1. **Crear cuenta en Railway**: Ve a [railway.app](https://railway.app) y crea una cuenta# 2. Activar entorno

source venv/bin/activate  # Linux/Mac

2. **Nuevo Proyecto desde GitHub**:# venv\Scripts\activate  # Windows

   - Click en "New Project"

   - Selecciona "Deploy from GitHub repo"# 3. Instalar dependencias  

   - Autoriza Railway para acceder a tu repositoriopip install -r requirements.txt

   - Selecciona el repositorio `orbital-simulator`

# 4. Ejecutar servidor

3. **Configuración automática**:uvicorn main:app --reload --port 8001

   - Railway detectará automáticamente la configuración desde `railway.json`

   - La aplicación se desplegará automáticamente# 5. Abrir en navegador

# http://localhost:8001/docs

4. **Obtener URL pública**:
   - Ve a Settings → Generate Domain
   - Railway te dará una URL pública como `https://tu-proyecto.up.railway.app`

### Variables de entorno (opcional):
No se requieren variables de entorno específicas. Railway asignará automáticamente el puerto (`$PORT`).

## 💻 Desarrollo Local

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
```

## 📡 Endpoints

- `GET /` - Mensaje de bienvenida
- `GET /datos2D` - Simulación orbital 2D
- `GET /datos3D` - Simulación orbital 3D
- `GET /health` - Health check

## 📝 Archivos de configuración Railway

- `Procfile` - Comando de inicio para Railway
- `railway.json` - Configuración del proyecto
- `runtime.txt` - Versión de Python
- `.railwayignore` - Archivos a ignorar en el despliegue

## 🛠️ Tecnologías

- **FastAPI** - Framework web moderno y rápido
- **NumPy** - Cálculos numéricos
- **Matplotlib** - Visualización de datos
- **SciPy** - Funciones científicas avanzadas
- **Uvicorn** - Servidor ASGI de alto rendimiento
