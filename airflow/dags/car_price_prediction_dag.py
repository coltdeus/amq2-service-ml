"""
DAG para orquestar el proceso de entrenamiento del modelo de predicción de precios de vehículos.

Este DAG realiza las siguientes tareas:
1. Obtención de datos desde MinIO (simulando un data lake)
2. Preprocesamiento de datos
3. Entrenamiento del modelo con diferentes hiperparámetros
4. Evaluación del modelo
5. Registro del modelo en MLflow
6. Despliegue del mejor modelo a través de FastAPI
"""

from datetime import datetime, timedelta
import os
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.S3_hook import S3Hook
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import boto3
import io

# Configuración de acceso a S3/MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["AWS_ENDPOINT_URL_S3"] = "http://s3:9000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://s3:9000"

# Configuración de acceso a MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
experiment_name = "vehicle_price_prediction"

# Configurar el cliente MLflow
client = MlflowClient()

# Crear el experimento si no existe
try:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location="s3://mlflow/vehicle_price_prediction"
    )
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
dag = DAG(
    'vehicle_price_prediction',
    default_args=default_args,
    description='Pipeline para predicción de precios de vehículos usados',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=['ml', 'prediction', 'vehicle']
)

# Funciones de utilidad para las tareas

def load_data(**kwargs):
    """Carga datos desde S3/MinIO"""
    s3_hook = S3Hook(aws_conn_id='s3_conn')
    bucket_name = 'data'
    key = 'car_details_v3.csv'
    
    logging.info(f"Obteniendo datos desde s3://{bucket_name}/{key}")
    
    # Intentar obtener el objeto desde S3
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url='http://s3:9000',
            aws_access_key_id='minio',
            aws_secret_access_key='minio123'
        )
        
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        data = response['Body'].read()
        
        # Cargar los datos en un DataFrame
        df = pd.read_csv(io.BytesIO(data))
        logging.info(f"Datos cargados exitosamente: {df.shape} filas")
        
        # Guardar el DataFrame en una ubicación temporal para tareas posteriores
        df.to_csv('/tmp/car_data.csv', index=False)
        logging.info("Datos guardados en ubicación temporal: /tmp/car_data.csv")
        
        return df.shape
    except Exception as e:
        logging.error(f"Error al cargar los datos: {str(e)}")
        # Si hay un error, intentar crear el archivo con datos de ejemplo para desarrollo
        create_sample_data()
        return (0, 0)

def create_sample_data():
    """Crea un conjunto de datos de ejemplo si no hay datos disponibles"""
    logging.info("Creando dataset de ejemplo para pruebas")
    
    # Datos de ejemplo mínimos para pruebas
    data = {
        'name': ['Maruti Swift Dzire VDI', 'Hyundai i20 Magna', 'Tata Nexon XZ'],
        'year': [2015, 2017, 2019],
        'selling_price': [450000, 520000, 750000],
        'km_driven': [50000, 30000, 15000],
        'fuel': ['Diesel', 'Petrol', 'Diesel'],
        'seller_type': ['Individual', 'Dealer', 'Individual'],
        'transmission': ['Manual', 'Manual', 'Manual'],
        'owner': ['First Owner', 'Second Owner', 'First Owner'],
        'mileage': ['21.4 kmpl', '18.6 kmpl', '22.1 kmpl'],
        'engine': ['1248 CC', '1197 CC', '1498 CC'],
        'max_power': ['74 bhp', '82 bhp', '108 bhp'],
        'torque': ['190Nm@2000rpm', '115Nm@4000rpm', '260Nm@1500rpm'],
        'seats': [5, 5, 5]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('/tmp/car_data.csv', index=False)
    logging.info("Datos de ejemplo creados y guardados en ubicación temporal: /tmp/car_data.csv")

def preprocess_data(**kwargs):
    """Preprocesa los datos para el entrenamiento"""
    logging.info("Iniciando preprocesamiento de datos")
    
    try:
        # Cargar datos
        df = pd.read_csv('/tmp/car_data.csv')
        logging.info(f"Datos cargados: {df.shape} filas")
        
        # Eliminar duplicados
        df.drop_duplicates(inplace=True)
        logging.info(f"Datos después de eliminar duplicados: {df.shape} filas")
        
        # Extraer brand y model
        df['brand'] = df['name'].str.split().str[0]
        df['model'] = df['name'].str.split(n=1).str[1]
        
        # Procesar variables numéricas
        # Engine CC
        df['engine_cc'] = df['engine'].str.extract('(\d+)').astype(float)
        
        # Max power
        df['max_power_bhp'] = df['max_power'].str.extract('(\d+\.?\d*)').astype(float)
        df['max_power_bhp'] = df['max_power_bhp'].replace(0, np.nan)
        
        # Mileage
        def standardize_mileage(x):
            if pd.isna(x):
                return np.nan
            if 'km/kg' in str(x):
                value = float(str(x).split()[0])
                return value * 1.39
            return float(str(x).split()[0])
        
        df['mileage_kmpl'] = df['mileage'].apply(standardize_mileage)
        
        # Torque
        def extract_torque_values(x):
            import re
            if pd.isna(x):
                return np.nan, np.nan
            
            torque_value = np.nan
            rpm_value = np.nan
            
            # Extraer valor de torque
            torque_match = re.search(r'(\d+\.?\d*)\s*Nm', str(x))
            if torque_match:
                torque_value = float(torque_match.group(1))
            
            # Extraer valor de RPM
            rpm_match = re.search(r'@\s*(\d+)(?:\s*rpm)?', str(x))
            if rpm_match:
                rpm_value = float(rpm_match.group(1))
            
            return torque_value, rpm_value
        
        torque_info = df['torque'].apply(extract_torque_values)
        df['torque_nm'] = [t[0] for t in torque_info]
        df['torque_rpm'] = [t[1] for t in torque_info]
        
        # Owner
        owner_map = {
            'First Owner': 1,
            'Second Owner': 2,
            'Third Owner': 3,
            'Fourth & Above Owner': 4,
            'Test Drive Car': 5
        }
        df['owner_rank'] = df['owner'].map(owner_map)
        
        # Guardar datos preprocesados
        df.to_csv('/tmp/car_data_preprocessed.csv', index=False)
        logging.info("Datos preprocesados guardados en: /tmp/car_data_preprocessed.csv")
        
        return df.shape
    except Exception as e:
        logging.error(f"Error en el preprocesamiento de datos: {str(e)}")
        raise

def train_model(**kwargs):
    """Entrena el modelo con diferentes hiperparámetros"""
    import xgboost as xgb
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    logging.info("Iniciando entrenamiento de modelo")
    
    try:
        # Establecer experimento MLflow
        mlflow.set_experiment(experiment_name)
        
        # Cargar datos preprocesados
        df = pd.read_csv('/tmp/car_data_preprocessed.csv')
        logging.info(f"Datos cargados para entrenamiento: {df.shape} filas")
        
        # Definir características y target
        numeric_cols = ['year', 'km_driven', 'engine_cc', 'max_power_bhp', 
                        'mileage_kmpl', 'seats', 'torque_nm', 'torque_rpm', 'owner_rank']
        categorical_cols = ['fuel', 'seller_type', 'transmission', 'brand']
        
        X = df[numeric_cols + categorical_cols]
        y = df['selling_price']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocesadores para características numéricas y categóricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Hiperparámetros a probar
        hyperparams = [
            {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8},
            {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8},
            {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 0.8},
            {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200, 'subsample': 0.9},
            {'max_depth': 5, 'learning_rate': 0.01, 'n_estimators': 300, 'subsample': 0.9}
        ]
        
        best_mae = float('inf')
        best_r2 = -float('inf')
        best_run_id = None
        
        # Probar cada combinación de hiperparámetros
        for params in hyperparams:
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                logging.info(f"Entrenando modelo con parámetros: {params}")
                
                # Registrar parámetros
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Crear modelo
                xgb_model = xgb.XGBRegressor(
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    subsample=params['subsample'],
                    random_state=42
                )
                
                # Crear pipeline con preprocesamiento y modelo
                model_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', xgb_model)
                ])
                
                # Entrenar modelo
                model_pipeline.fit(X_train, y_train)
                
                # Evaluar modelo
                y_pred_train = model_pipeline.predict(X_train)
                y_pred_test = model_pipeline.predict(X_test)
                
                # Calcular métricas
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Registrar métricas
                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("test_r2", test_r2)
                
                logging.info(f"Métricas - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}, Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
                
                # Registrar modelo
                mlflow.sklearn.log_model(model_pipeline, "model")
                
                # Verificar si es el mejor modelo hasta ahora
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_mae = test_mae
                    best_run_id = run_id
                    logging.info(f"Nuevo mejor modelo encontrado: {best_run_id} con R2: {best_r2:.4f}")
        
        # Guardar el ID del mejor modelo para la siguiente tarea
        with open('/tmp/best_model_id.txt', 'w') as f:
            f.write(best_run_id)
        
        logging.info(f"Mejor modelo seleccionado: {best_run_id} con MAE: {best_mae:.2f} y R2: {best_r2:.4f}")
        
        return {'best_run_id': best_run_id, 'best_mae': best_mae, 'best_r2': best_r2}
    except Exception as e:
        logging.error(f"Error en el entrenamiento del modelo: {str(e)}")
        raise

def register_best_model(**kwargs):
    """Registra el mejor modelo en MLflow Model Registry"""
    logging.info("Registrando el mejor modelo en MLflow Model Registry")
    
    try:
        # Leer el ID del mejor modelo
        with open('/tmp/best_model_id.txt', 'r') as f:
            best_run_id = f.read().strip()
        
        logging.info(f"Mejor modelo a registrar: {best_run_id}")
        
        # Establecer URI de MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = MlflowClient()
        
        # Registrar modelo
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri, "car_price_predictor")
        
        logging.info(f"Modelo registrado como: {mv.name} versión: {mv.version}")
        
        # Promocionar modelo a producción
        client.transition_model_version_stage(
            name="car_price_predictor",
            version=mv.version,
            stage="Production"
        )
        
        logging.info(f"Modelo promovido a Producción: {mv.name} versión: {mv.version}")
        
        return {'model_name': mv.name, 'model_version': mv.version}
    except Exception as e:
        logging.error(f"Error al registrar el mejor modelo: {str(e)}")
        raise

def deploy_model(**kwargs):
    """Implementa el modelo en producción a través de FastAPI"""
    logging.info("Preparando despliegue del modelo a FastAPI")
    
    try:
        # Obtener la versión actual del modelo en producción
        client = MlflowClient()
        production_model = client.get_latest_versions("car_price_predictor", stages=["Production"])[0]
        
        model_version = production_model.version
        run_id = production_model.run_id
        
        logging.info(f"Modelo en producción: car_price_predictor versión {model_version} (run_id: {run_id})")
        
        # Crear archivo de configuración para FastAPI
        config = {
            'model_name': 'car_price_predictor',
            'model_version': model_version,
            'run_id': run_id,
            'mlflow_tracking_uri': 'http://mlflow:5000',
            'deployment_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open('/tmp/model_config.json', 'w') as f:
            json.dump(config, f)
        
        # Copiar configuración al directorio de FastAPI
        # Nota: En un entorno real, habría que verificar cómo se comunican los contenedores
        # y posiblemente usar un volumen compartido o un servicio de almacenamiento.
        logging.info("Configuración de modelo guardada en /tmp/model_config.json")
        logging.info("Para completar el despliegue, este archivo debe ser accesible por FastAPI")
        
        return config
    except Exception as e:
        logging.error(f"Error al preparar el despliegue: {str(e)}")
        raise

# Definición de tareas
task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

task_preprocess_data = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_register_model = PythonOperator(
    task_id='register_model',
    python_callable=register_best_model,
    dag=dag,
)

task_deploy_model = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# Definir el orden de ejecución de las tareas
task_load_data >> task_preprocess_data >> task_train_model >> task_register_model >> task_deploy_model