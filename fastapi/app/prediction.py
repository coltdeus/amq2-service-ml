"""
Módulo de predicción para la API de predicción de precios de vehículos.

Este módulo contiene la lógica para cargar el modelo desde MLflow
y realizar predicciones con él.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionService:
    """Clase para gestionar la carga del modelo y las predicciones"""
    
    def __init__(self):
        """Inicializa el servicio de predicción sin cargar el modelo"""
        self.model = None
        self.model_info = {}
        
        # Configurar MLflow y S3/MinIO
        os.environ["AWS_ACCESS_KEY_ID"] = "minio"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://s3:9000"
        mlflow.set_tracking_uri("http://mlflow:5000")
    
    def load_model(self) -> bool:
        """Carga el modelo de producción desde MLflow Model Registry"""
        try:
            # Intentar cargar configuración si existe
            config_path = "/tmp/model_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                model_name = config['model_name']
                model_version = config['model_version']
                logger.info(f"Cargando modelo {model_name} versión {model_version} desde config")
            else:
                # Si no hay configuración, cargar la última versión en producción
                client = MlflowClient()
                model_name = "car_price_predictor"
                production_versions = client.get_latest_versions(model_name, stages=["Production"])
                
                if not production_versions:
                    logger.warning("No hay modelos en producción disponibles")
                    return self._load_fallback_model()
                
                model_version = production_versions[0].version
                logger.info(f"Cargando último modelo en producción: {model_name} versión {model_version}")
            
            # Cargar modelo desde MLflow
            model_uri = f"models:/{model_name}/{model_version}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Guardar información del modelo
            self.model_info = {
                'name': model_name,
                'version': model_version,
                'framework': 'XGBoost',
                'features': [
                    'year', 'km_driven', 'engine_cc', 'max_power_bhp', 'mileage_kmpl', 
                    'seats', 'torque_nm', 'torque_rpm', 'owner_rank', 'fuel', 
                    'seller_type', 'transmission', 'brand'
                ]
            }
            
            logger.info(f"Modelo cargado exitosamente: {self.model_info}")
            return True
        
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            return self._load_fallback_model()
    
    def _load_fallback_model(self) -> bool:
        """Carga un modelo simple de respaldo en caso de error"""
        try:
            import xgboost as xgb
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            logger.warning("Creando modelo de respaldo simple")
            
            # Crear pipeline simple
            class SimplePricePredictorModel:
                def predict(self, X):
                    # Un modelo muy básico que predice usando solo el año y km
                    base_price = 500000  # Precio base en rupias
                    year_factor = X['year'] * 20000  # Influencia del año
                    km_factor = -X['km_driven'] * 0.5  # Influencia del kilometraje
                    
                    # Predicción básica
                    prediction = base_price + year_factor - 40000000 + km_factor
                    
                    # Asegurar valores positivos y razonables
                    prediction = np.maximum(prediction, 50000)
                    return prediction
            
            self.model = SimplePricePredictorModel()
            
            self.model_info = {
                'name': 'fallback_model',
                'version': '0.1',
                'framework': 'Simple Rule-Based',
                'features': ['year', 'km_driven']
            }
            
            logger.info("Modelo de respaldo cargado")
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar modelo de respaldo: {str(e)}")
            self.model = None
            self.model_info = {}
            return False
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones usando el modelo cargado"""
        if self.model is None:
            raise ValueError("El modelo no está cargado")
        
        return self.model.predict(data)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Devuelve información sobre el modelo cargado"""
        return self.model_info
    
    def is_model_loaded(self) -> bool:
        """Verifica si hay un modelo cargado"""
        return self.model is not None

# Instancia única del servicio de predicción
prediction_service = PredictionService()