"""
DAG para monitorear el rendimiento del modelo de predicción de precios de vehículos.

Este DAG realiza las siguientes tareas:
1. Verifica el estado de salud de la API
2. Recopila métricas de rendimiento del modelo
3. Detecta drift en los datos
4. Genera un informe de monitoreo
5. Envía alertas si es necesario
"""

from datetime import datetime, timedelta
import os
import sys
import logging
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable
from airflow.utils.email import send_email

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argumentos por defecto
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
dag = DAG(
    'vehicle_price_model_monitoring',
    default_args=default_args,
    description='Monitoreo del modelo de predicción de precios de vehículos',
    schedule_interval='@daily',
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=['monitoring', 'mlops']
)

# Definición de funciones para las tareas

def check_api_health(**context):
    """Verifica el estado de salud de la API"""
    try:
        response = requests.get("http://fastapi:8800/health", timeout=10)
        response.raise_for_status()
        
        logger.info("API health check: OK")
        context['ti'].xcom_push(key='api_health', value='healthy')
        return True
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        context['ti'].xcom_push(key='api_health', value='unhealthy')
        raise

def get_model_info(**context):
    """Obtiene información sobre el modelo en producción"""
    try:
        response = requests.get("http://fastapi:8800/model/info", timeout=10)
        response.raise_for_status()
        
        model_info = response.json()
        logger.info(f"Modelo en producción: {model_info['name']} versión {model_info['version']}")
        
        context['ti'].xcom_push(key='model_info', value=model_info)
        return model_info
    except Exception as e:
        logger.error(f"Error al obtener información del modelo: {str(e)}")
        raise

def collect_predictions_data(**context):
    """
    Recopila datos de predicciones recientes para análisis.
    
    En un entorno de producción real, estos datos podrían venir de:
    - Logs de la API
    - Base de datos de predicciones
    - Sistema de almacenamiento de eventos
    """
    # Para este ejemplo, simulamos la recopilación de datos
    # En un caso real, habría que consultar fuentes de datos reales
    
    try:
        # Cargar datos de referencia desde MinIO/S3
        s3_hook = S3Hook(aws_conn_id='s3_conn')
        bucket_name = 'data'
        key = 'car_details_v3.csv'
        
        # Obtener datos de referencia
        reference_data = s3_hook.read_key(key=key, bucket_name=bucket_name)
        reference_df = pd.read_csv(BytesIO(reference_data))
        
        # Simular datos de predicciones recientes
        recent_predictions = reference_df.copy()
        
        # Añadir fecha de predicción (últimos 7 días)
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generar fechas aleatorias en los últimos 7 días
        today = datetime.now()
        dates = [(today - timedelta(days=np.random.randint(0, 7))).strftime('%Y-%m-%d') 
                for _ in range(len(recent_predictions))]
        recent_predictions['prediction_date'] = dates
        
        # Simular predicciones con un error aleatorio
        # En un sistema real, estos serían los valores predichos por el modelo
        noise = np.random.normal(0, 0.1, size=len(recent_predictions))
        recent_predictions['predicted_price'] = recent_predictions['selling_price'] * (1 + noise)
        
        # Guardar temporalmente para análisis
        temp_path = '/tmp/recent_predictions.csv'
        recent_predictions.to_csv(temp_path, index=False)
        
        # Compartir información mediante XCom
        context['ti'].xcom_push(key='recent_predictions_path', value=temp_path)
        context['ti'].xcom_push(key='prediction_count', value=len(recent_predictions))
        
        logger.info(f"Recopiladas {len(recent_predictions)} predicciones recientes")
        return temp_path
    except Exception as e:
        logger.error(f"Error al recopilar datos de predicciones: {str(e)}")
        raise

def analyze_model_performance(**context):
    """Analiza el rendimiento del modelo con datos recientes"""
    try:
        # Obtener path de predicciones
        predictions_path = context['ti'].xcom_pull(key='recent_predictions_path', task_ids='collect_predictions_data')
        
        if not predictions_path or not os.path.exists(predictions_path):
            raise ValueError("No se encontraron datos de predicciones recientes")
        
        # Cargar datos
        predictions_df = pd.read_csv(predictions_path)
        
        # Calcular métricas de rendimiento
        actual = predictions_df['selling_price']
        predicted = predictions_df['predicted_price']
        
        # Calcular métricas
        import numpy as np
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        
        # Crear diccionario de métricas
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
        
        logger.info(f"Métricas de rendimiento: MAE={mae:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}")
        
        # Compartir métricas
        context['ti'].xcom_push(key='performance_metrics', value=metrics)
        
        # Generar visualización de predicciones vs reales
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([0, actual.max()], [0, actual.max()], 'r--')
        plt.title('Predicciones vs Valores Reales')
        plt.xlabel('Valor Real')
        plt.ylabel('Predicción')
        
        # Guardar visualización
        viz_path = '/tmp/predictions_vs_actual.png'
        plt.savefig(viz_path)
        plt.close()
        
        # Compartir path de visualización
        context['ti'].xcom_push(key='performance_viz_path', value=viz_path)
        
        return metrics
    except Exception as e:
        logger.error(f"Error al analizar rendimiento del modelo: {str(e)}")
        raise

def detect_data_drift(**context):
    """Detecta drift en los datos de entrada"""
    try:
        # Obtener path de predicciones
        predictions_path = context['ti'].xcom_pull(key='recent_predictions_path', task_ids='collect_predictions_data')
        
        if not predictions_path or not os.path.exists(predictions_path):
            raise ValueError("No se encontraron datos de predicciones recientes")
        
        # Cargar datos recientes
        recent_df = pd.read_csv(predictions_path)
        
        # Cargar datos de referencia
        s3_hook = S3Hook(aws_conn_id='s3_conn')
        bucket_name = 'data'
        key = 'car_details_v3.csv'
        
        reference_data = s3_hook.read_key(key=key, bucket_name=bucket_name)
        reference_df = pd.read_csv(BytesIO(reference_data))
        
        # Identificar columnas numéricas para análisis
        numeric_cols = reference_df.select_dtypes(include=['int64', 'float64']).columns
        
        # Análisis de drift para cada columna
        drift_results = {}
        from scipy import stats
        
        for col in numeric_cols:
            if col in recent_df.columns and col != 'selling_price' and col != 'predicted_price':
                # Realizar test de Kolmogorov-Smirnov
                ks_statistic, p_value = stats.ks_2samp(
                    reference_df[col].dropna(),
                    recent_df[col].dropna()
                )
                
                drift_detected = p_value < 0.05
                
                drift_results[col] = {
                    'ks_statistic': float(ks_statistic),
                    'p_value': float(p_value),
                    'drift_detected': drift_detected
                }
                
                if drift_detected:
                    logger.warning(f"Drift detectado en {col}: p-value = {p_value:.4f}")
        
        # Contar features con drift
        drift_count = sum(1 for r in drift_results.values() if r['drift_detected'])
        
        # Compartir resultados
        context['ti'].xcom_push(key='drift_results', value=drift_results)
        context['ti'].xcom_push(key='drift_count', value=drift_count)
        
        logger.info(f"Análisis de drift completado: {drift_count} características con drift detectado")
        
        return drift_results
    except Exception as e:
        logger.error(f"Error al detectar drift: {str(e)}")
        raise

def generate_monitoring_report(**context):
    """Genera un informe de monitoreo completo"""
    try:
        # Recopilar datos de tareas anteriores
        api_health = context['ti'].xcom_pull(key='api_health', task_ids='check_api_health')
        model_info = context['ti'].xcom_pull(key='model_info', task_ids='get_model_info')
        prediction_count = context['ti'].xcom_pull(key='prediction_count', task_ids='collect_predictions_data')
        performance_metrics = context['ti'].xcom_pull(key='performance_metrics', task_ids='analyze_model_performance')
        drift_results = context['ti'].xcom_pull(key='drift_results', task_ids='detect_data_drift')
        drift_count = context['ti'].xcom_pull(key='drift_count', task_ids='detect_data_drift')
        
        # Timestamp para el informe
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Construir informe
        report = {
            'timestamp': timestamp,
            'api_status': api_health,
            'model_info': model_info,
            'predictions_analyzed': prediction_count,
            'performance_metrics': performance_metrics,
            'data_drift': {
                'features_analyzed': len(drift_results) if drift_results else 0,
                'features_with_drift': drift_count if drift_count is not None else 0,
                'drift_details': drift_results
            }
        }
        
        # Verificar alertas
        alerts = []
        
        # Alerta por API no saludable
        if api_health != 'healthy':
            alerts.append({
                'type': 'api_health',
                'message': 'La API no está respondiendo correctamente',
                'severity': 'high'
            })
        
        # Alerta por rendimiento del modelo
        if performance_metrics and 'r2' in performance_metrics:
            # Umbral de alerta: R² < 0.7
            if performance_metrics['r2'] < 0.7:
                alerts.append({
                    'type': 'model_performance',
                    'message': f"R² bajo: {performance_metrics['r2']:.4f}",
                    'severity': 'medium'
                })
            
            # Umbral de alerta: MAPE > 20%
            if performance_metrics['mape'] > 20:
                alerts.append({
                    'type': 'model_performance',
                    'message': f"MAPE alto: {performance_metrics['mape']:.2f}%",
                    'severity': 'medium'
                })
        
        # Alerta por drift
        if drift_count and drift_count > 3:
            alerts.append({
                'type': 'data_drift',
                'message': f"Drift detectado en {drift_count} características",
                'severity': 'high' if drift_count > 5 else 'medium'
            })
        
        # Añadir alertas al informe
        report['alerts'] = alerts
        report['alert_count'] = len(alerts)
        
        # Guardar informe
        report_path = f"/tmp/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Subir informe a S3
        s3_hook = S3Hook(aws_conn_id='s3_conn')
        s3_key = f"monitoring_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        s3_hook.load_file(
            filename=report_path,
            key=s3_key,
            bucket_name='mlflow',
            replace=True
        )
        
        # Compartir path e información de alertas
        context['ti'].xcom_push(key='report_path', value=report_path)
        context['ti'].xcom_push(key='s3_report_path', value=f"s3://mlflow/{s3_key}")
        context['ti'].xcom_push(key='alerts', value=alerts)
        
        logger.info(f"Informe de monitoreo generado: {report_path} y subido a S3: {s3_key}")
        
        return report_path
    except Exception as e:
        logger.error(f"Error al generar informe de monitoreo: {str(e)}")
        raise

def send_alert_email(**context):
    """Envía alertas por correo si es necesario"""
    # Obtener alertas del informe
    alerts = context['ti'].xcom_pull(key='alerts', task_ids='generate_monitoring_report')
    
    if not alerts or len(alerts) == 0:
        logger.info("No hay alertas que enviar")
        return
    
    try:
        # Construir contenido del correo
        subject = f"[ALERTA] Monitoreo de Modelo - {len(alerts)} alertas detectadas"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert {{ padding: 10px; margin-bottom: 10px; border-radius: 5px; }}
                .high {{ background-color: #ffcccc; border: 1px solid #ff0000; }}
                .medium {{ background-color: #fff4cc; border: 1px solid #ffcc00; }}
                .low {{ background-color: #e6f5ff; border: 1px solid #0066cc; }}
            </style>
        </head>
        <body>
            <h2>Alertas de Monitoreo del Modelo</h2>
            <p>Se han detectado las siguientes alertas en el monitoreo del modelo de predicción de precios de vehículos:</p>
            
            <div>
        """
        
        for alert in alerts:
            severity_class = alert.get('severity', 'low')
            html_content += f"""
                <div class="alert {severity_class}">
                    <strong>{alert.get('type', 'Alert').replace('_', ' ').title()}</strong>: {alert.get('message', 'No message')}
                    <br>
                    <small>Severidad: {severity_class}</small>
                </div>
            """
        
        # Añadir enlace al informe completo
        s3_report_path = context['ti'].xcom_pull(key='s3_report_path', task_ids='generate_monitoring_report')
        if s3_report_path:
            html_content += f"""
                <p>Para más detalles, consulte el <a href="{s3_report_path}">informe completo</a>.</p>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Obtener destinatarios de las variables de Airflow
        # (en un entorno real, se obtendrían de la configuración)
        recipients = ['alerts@example.com']
        
        # Enviar correo
        send_email(
            to=recipients,
            subject=subject,
            html_content=html_content
        )
        
        logger.info(f"Alerta enviada por correo a {recipients}")
        
        return True
    except Exception as e:
        logger.error(f"Error al enviar alerta por correo: {str(e)}")
        # No hacemos re-raise para que el DAG no falle si falla el envío de correo
        return False

# Definir tareas en el DAG

# Tarea 1: Verificar salud de la API
task_check_api = HttpSensor(
    task_id='check_api_health',
    http_conn_id='fastapi_conn',
    endpoint='/health',
    request_params={},
    response_check=lambda response: "status" in response.json() and response.json()["status"] == "ok",
    poke_interval=30,
    timeout=300,
    soft_fail=True,  # Continuar incluso si la API no responde
    dag=dag
)

# Tarea 2: Obtener información del modelo
task_get_model_info = PythonOperator(
    task_id='get_model_info',
    python_callable=get_model_info,
    dag=dag
)

# Tarea 3: Recopilar datos de predicciones
task_collect_predictions = PythonOperator(
    task_id='collect_predictions_data',
    python_callable=collect_predictions_data,
    dag=dag
)

# Tarea 4: Analizar rendimiento del modelo
task_analyze_performance = PythonOperator(
    task_id='analyze_model_performance',
    python_callable=analyze_model_performance,
    dag=dag
)

# Tarea 5: Detectar drift en los datos
task_detect_drift = PythonOperator(
    task_id='detect_data_drift',
    python_callable=detect_data_drift,
    dag=dag
)

# Tarea 6: Generar informe de monitoreo
task_generate_report = PythonOperator(
    task_id='generate_monitoring_report',
    python_callable=generate_monitoring_report,
    dag=dag
)

# Tarea 7: Enviar alertas por correo si es necesario
task_send_alerts = PythonOperator(
    task_id='send_alert_email',
    python_callable=send_alert_email,
    trigger_rule='all_done',  # Ejecutar incluso si tareas anteriores fallan
    dag=dag
)

# Definir el orden de ejecución
task_check_api >> task_get_model_info
task_check_api >> task_collect_predictions
task_get_model_info >> task_analyze_performance
task_collect_predictions >> task_analyze_performance
task_collect_predictions >> task_detect_drift
[task_analyze_performance, task_detect_drift] >> task_generate_report >> task_send_alerts