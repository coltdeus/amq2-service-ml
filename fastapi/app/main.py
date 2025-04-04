"""
Aplicación principal de FastAPI para el servicio de predicción de precios de vehículos.

Esta aplicación proporciona una API REST para:
1. Obtener información sobre el modelo de predicción
2. Realizar predicciones de precios de vehículos
"""

import logging
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager

# Importar modelos y servicios
from models import CarFeatures, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse, ModelInfo
from prediction import prediction_service

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gestor de contexto para la inicialización y cierre de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código de inicialización: cargar el modelo
    logger.info("Iniciando la aplicación - Cargando modelo...")
    prediction_service.load_model()
    yield
    # Código de limpieza
    logger.info("Cerrando la aplicación")

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Precios de Vehículos",
    description="API para predecir precios de vehículos usados basado en sus características",
    version="1.0",
    lifespan=lifespan
)

# Endpoints

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz que proporciona información básica sobre la API"""
    return {
        "message": "API de Predicción de Precios de Vehículos",
        "status": "active",
        "model_loaded": prediction_service.is_model_loaded()
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Endpoint para verificar la salud del servicio"""
    if not prediction_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {"status": "ok", "model_loaded": True}

@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
async def get_model_info():
    """Obtener información sobre el modelo actual"""
    if not prediction_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    model_info = prediction_service.get_model_info()
    if not model_info:
        raise HTTPException(status_code=404, detail="Información del modelo no disponible")
    
    return model_info

@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict(vehicle: CarFeatures):
    """Realiza una predicción de precio para un vehículo individual"""
    if not prediction_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir la entrada a DataFrame
        input_data = pd.DataFrame([vehicle.dict()])
        
        # Realizar predicción
        prediction = prediction_service.predict(input_data)
        
        # Devolver resultado
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
async def predict_batch(request: BatchPredictionRequest):
    """Realiza predicciones de precio para múltiples vehículos en un lote"""
    if not prediction_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir la entrada a DataFrame
        vehicles_data = [v.dict() for v in request.vehicles]
        input_data = pd.DataFrame(vehicles_data)
        
        # Realizar predicciones
        predictions = prediction_service.predict(input_data)
        
        # Devolver resultados
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error en la predicción por lotes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la predicción por lotes: {str(e)}")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Redirección personalizada a la documentación"""
    return app.redirect_to_docs

# Para ejecutar la aplicación directamente (modo desarrollo)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8800, reload=True)