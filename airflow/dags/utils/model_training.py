"""
Script para monitoreo del modelo de predicción de precios de vehículos en producción.

Este script:
1. Recopila estadísticas de las predicciones realizadas
2. Detecta drift en los datos de entrada
3. Evalúa el rendimiento del modelo con nuevos datos
4. Genera alertas cuando el rendimiento se degrada
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_monitoring.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Configuración
MLFLOW_TRACKING_URI = "http://mlflow:5000"
API_BASE_URL = "http://fastapi:8800"
MODEL_NAME = "car_price_predictor"
LOGS_DIR = "/opt/airflow/logs/model_monitoring"
ALERT_THRESHOLD = 0.15  # Umbral de degradación para alertas (15%)

# Asegurar que el directorio de logs existe
os.makedirs(LOGS_DIR, exist_ok=True)

class ModelMonitor:
    """Clase para monitoreo del modelo en producción"""
    
    def __init__(self, tracking_uri: str, model_name: str, api_url: str):
        """
        Inicializa el monitor de modelo.
        
        Args:
            tracking_uri: URI de seguimiento de MLflow
            model_name: Nombre del modelo a monitorear
            api_url: URL base de la API
        """
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.api_url = api_url
        
        # Configurar MLflow
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
        # Datos de referencia (baseline)
        self.reference_data = None
        self.reference_stats = None
        
        # Datos recientes
        self.recent_predictions = None
        self.recent_stats = None
    
    def load_reference_data(self, csv_path: str) -> None:
        """
        Carga datos de referencia desde un CSV.
        
        Args:
            csv_path: Ruta al archivo CSV con datos de referencia
        """
        logger.info(f"Cargando datos de referencia desde {csv_path}")
        try:
            self.reference_data = pd.read_csv(csv_path)
            # Calcular estadísticas de referencia
            self.reference_stats = self._calculate_statistics(self.reference_data)
            logger.info(f"Datos de referencia cargados: {len(self.reference_data)} registros")
        except Exception as e:
            logger.error(f"Error al cargar datos de referencia: {str(e)}")
            raise
    
    def collect_recent_predictions(self, days: int = 7) -> None:
        """
        Recopila predicciones recientes desde los logs.
        
        Args:
            days: Número de días atrás para recopilar datos
        """
        logger.info(f"Recopilando predicciones de los últimos {days} días")
        
        # En un entorno real, estas predicciones vendrían de logs de la API, 
        # base de datos, o sistema de almacenamiento
        # Para este ejemplo, generamos datos sintéticos similares a los de referencia
        
        if self.reference_data is None:
            logger.error("No hay datos de referencia para basar los datos sintéticos")
            return
        
        try:
            # Generar datos sintéticos con ligeras variaciones
            synthetic_data = self.reference_data.copy()
            
            # Añadir algunas variaciones aleatorias
            for col in synthetic_data.select_dtypes(include=['number']).columns:
                if col != 'selling_price':  # No modificar el precio real
                    # Añadir variación aleatoria de hasta ±10%
                    noise = np.random.normal(0, 0.05, size=len(synthetic_data))
                    synthetic_data[col] = synthetic_data[col] * (1 + noise)
            
            # Añadir columna de fecha (últimos 'days' días)
            today = datetime.now()
            dates = [today - timedelta(days=np.random.randint(0, days)) for _ in range(len(synthetic_data))]
            synthetic_data['prediction_date'] = dates
            
            # Añadir predicciones simuladas
            # En un sistema real, estas serían las predicciones del modelo
            base_price = synthetic_data['selling_price'].values
            # Añadir error simulado (normal con media 0 y desviación estándar del 10%)
            prediction_error = np.random.normal(0, 0.1, size=len(synthetic_data)) * base_price
            synthetic_data['predicted_price'] = base_price + prediction_error
            
            self.recent_predictions = synthetic_data
            # Calcular estadísticas de predicciones recientes
            self.recent_stats = self._calculate_statistics(self.recent_predictions)
            
            logger.info(f"Recopiladas {len(self.recent_predictions)} predicciones recientes")
        except Exception as e:
            logger.error(f"Error al recopilar predicciones recientes: {str(e)}")
            raise
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calcula estadísticas descriptivas para cada columna numérica.
        
        Args:
            data: DataFrame con datos para análisis
            
        Returns:
            Diccionario con estadísticas por columna
        """
        stats = {}
        
        for col in data.select_dtypes(include=['number']).columns:
            col_stats = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'p25': data[col].quantile(0.25),
                'p75': data[col].quantile(0.75)
            }
            stats[col] = col_stats
        
        return stats
    
    def detect_data_drift(self, threshold: float = 0.05) -> Dict[str, float]:
        """
        Detecta drift en los datos de entrada comparando distribuciones.
        
        Args:
            threshold: Umbral para considerar que hay drift significativo
            
        Returns:
            Diccionario con p-values para cada característica
        """
        if self.reference_data is None or self.recent_predictions is None:
            logger.error("Faltan datos de referencia o predicciones recientes")
            return {}
        
        logger.info("Detectando data drift...")
        drift_results = {}
        
        # Comparar distribuciones para cada característica numérica
        for col in self.reference_data.select_dtypes(include=['number']).columns:
            if col in self.recent_predictions.columns:
                # Realizar test de Kolmogorov-Smirnov
                ks_statistic, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    self.recent_predictions[col].dropna()
                )
                
                drift_results[col] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
                
                if p_value < threshold:
                    logger.warning(f"Drift detectado en {col}: p-value = {p_value:.4f}")
        
        return drift_results
    
    def evaluate_model_performance(self) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo con datos recientes.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        if self.recent_predictions is None:
            logger.error("No hay predicciones recientes para evaluar")
            return {}
        
        logger.info("Evaluando rendimiento del modelo...")
        
        try:
            # Calcular métricas
            actual = self.recent_predictions['selling_price']
            predicted = self.recent_predictions['predicted_price']
            
            mae = np.mean(np.abs(actual - predicted))
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
            
            performance = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            }
            
            logger.info(f"Rendimiento: MAE={mae:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}")
            return performance
        except Exception as e:
            logger.error(f"Error al evaluar rendimiento: {str(e)}")
            return {}
    
    def get_production_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el modelo en producción.
        
        Returns:
            Diccionario con información del modelo
        """
        try:
            # Obtener la versión en producción
            production_models = self.client.get_latest_versions(self.model_name, stages=["Production"])
            
            if not production_models:
                logger.warning(f"No hay modelos {self.model_name} en producción")
                return {}
            
            model_info = {
                'model_name': production_models[0].name,
                'version': production_models[0].version,
                'run_id': production_models[0].run_id,
                'creation_timestamp': production_models[0].creation_timestamp,
                'description': production_models[0].description
            }
            
            # Obtener métricas del run
            run = self.client.get_run(production_models[0].run_id)
            if run.data.metrics:
                model_info['metrics'] = run.data.metrics
            
            logger.info(f"Modelo en producción: {self.model_name} versión {model_info['version']}")
            return model_info
        except Exception as e:
            logger.error(f"Error al obtener información del modelo: {str(e)}")
            return {}
    
    def check_api_health(self) -> bool:
        """
        Verifica la salud de la API.
        
        Returns:
            True si la API está funcionando correctamente, False en caso contrario
        """
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error al verificar salud de la API: {str(e)}")
            return False
    
    def generate_monitoring_report(self, output_path: str) -> None:
        """
        Genera un informe de monitoreo en formato JSON.
        
        Args:
            output_path: Ruta donde guardar el informe
        """
        logger.info(f"Generando informe de monitoreo en {output_path}")
        
        # Obtener timestamp actual
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Recopilar información
        api_healthy = self.check_api_health()
        model_info = self.get_production_model_info()
        drift_results = self.detect_data_drift()
        performance = self.evaluate_model_performance()
        
        # Construir informe
        report = {
            'timestamp': timestamp,
            'api_status': 'healthy' if api_healthy else 'unhealthy',
            'model_info': model_info,
            'data_drift': drift_results,
            'performance': performance
        }
        
        # Verificar si hay alertas que generar
        alerts = []
        
        # Alerta por degradación de rendimiento
        if performance and 'r2' in performance:
            baseline_r2 = model_info.get('metrics', {}).get('test_r2', 0.8)  # Valor por defecto
            current_r2 = performance['r2']
            
            if baseline_r2 - current_r2 > ALERT_THRESHOLD:
                alerts.append({
                    'type': 'performance_degradation',
                    'message': f"Degradación significativa del rendimiento: R² bajó de {baseline_r2:.4f} a {current_r2:.4f}",
                    'severity': 'high'
                })
        
        # Alertas por drift
        for feature, result in drift_results.items():
            if result.get('drift_detected', False):
                alerts.append({
                    'type': 'data_drift',
                    'message': f"Drift detectado en {feature}: p-value = {result['p_value']:.4f}",
                    'severity': 'medium' if result['p_value'] < 0.01 else 'low'
                })
        
        report['alerts'] = alerts
        
        # Guardar informe
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Informe guardado en {output_path}")
        except Exception as e:
            logger.error(f"Error al guardar informe: {str(e)}")
    
    def generate_monitoring_visualizations(self, output_dir: str) -> None:
        """
        Genera visualizaciones para el monitoreo.
        
        Args:
            output_dir: Directorio donde guardar las visualizaciones
        """
        if self.reference_data is None or self.recent_predictions is None:
            logger.error("Faltan datos para generar visualizaciones")
            return
        
        logger.info(f"Generando visualizaciones en {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Distribución de características con drift
        drift_results = self.detect_data_drift()
        drift_features = [f for f, r in drift_results.items() if r.get('drift_detected', False)]
        
        for feature in drift_features:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.reference_data[feature].dropna(), label='Referencia')
            sns.kdeplot(self.recent_predictions[feature].dropna(), label='Reciente')
            plt.title(f'Distribución de {feature}')
            plt.xlabel(feature)
            plt.ylabel('Densidad')
            plt.legend()
            plt.savefig(f"{output_dir}/{feature}_drift.png")
            plt.close()
        
        # 2. Predicciones vs valores reales
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.recent_predictions['selling_price'],
            self.recent_predictions['predicted_price'],
            alpha=0.5
        )
        plt.plot([0, self.recent_predictions['selling_price'].max()], 
                [0, self.recent_predictions['selling_price'].max()], 
                'r--')
        plt.title('Predicciones vs Valores Reales')
        plt.xlabel('Valor Real')
        plt.ylabel('Predicción')
        plt.savefig(f"{output_dir}/predictions_vs_actual.png")
        plt.close()
        
        # 3. Error vs precio real
        plt.figure(figsize=(10, 6))
        error = self.recent_predictions['predicted_price'] - self.recent_predictions['selling_price']
        plt.scatter(self.recent_predictions['selling_price'], error, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Error vs Precio Real')
        plt.xlabel('Precio Real')
        plt.ylabel('Error (Predicción - Real)')
        plt.savefig(f"{output_dir}/error_vs_price.png")
        plt.close()
        
        logger.info(f"Visualizaciones generadas en {output_dir}")

def main():
    """Función principal del script de monitoreo"""
    logger.info("Iniciando monitoreo del modelo...")
    
    # Crear monitor
    monitor = ModelMonitor(
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name=MODEL_NAME,
        api_url=API_BASE_URL
    )
    
    # Cargar datos de referencia
    # En un entorno real, estos datos podrían cargarse de S3/MinIO
    try:
        monitor.load_reference_data("/tmp/car_data_preprocessed.csv")
    except Exception as e:
        logger.error(f"Error al cargar datos de referencia: {str(e)}")
        # Continuar con el monitoreo aunque no haya datos de referencia
    
    # Recopilar predicciones recientes
    try:
        monitor.collect_recent_predictions(days=7)
    except Exception as e:
        logger.error(f"Error al recopilar predicciones recientes: {str(e)}")
        return
    
    # Generar timestamp para los archivos de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generar informe
    report_path = f"{LOGS_DIR}/monitoring_report_{timestamp}.json"
    monitor.generate_monitoring_report(report_path)
    
    # Generar visualizaciones
    viz_dir = f"{LOGS_DIR}/visualizations_{timestamp}"
    monitor.generate_monitoring_visualizations(viz_dir)
    
    logger.info("Monitoreo completado exitosamente")

if __name__ == "__main__":
    main()