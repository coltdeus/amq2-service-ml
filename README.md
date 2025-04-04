# Ejemplo de ambiente productivo
### MLOps1 - CEIA - FIUBA
Estructura de servicios para la implementación del proyecto final de MLOps1 - CEIA - FIUBA

Supongamos que trabajamos para **ML Models and something more Inc.**, la cual ofrece un servicio 
que proporciona modelos mediante una REST API. Internamente, tanto para realizar tareas de 
DataOps como de MLOps, la empresa cuenta con varios servicios que ayudan a ejecutar las 
acciones necesarias. También dispone de un Data Lake en S3, para este caso, simularemos un 
S3 utilizando MinIO.

Para simular esta empresa, utilizaremos Docker y, a través de Docker Compose, desplegaremos 
varios contenedores que representan distintos servicios en un entorno productivo.

Los servicios que contamos son:
- [Apache Airflow](https://airflow.apache.org/)
- [MLflow](https://mlflow.org/)
- API Rest para servir modelos ([FastAPI](https://fastapi.tiangolo.com/))
- [MinIO](https://min.io/)
- Base de datos relacional [PostgreSQL](https://www.postgresql.org/)

![Diagrama de servicios](final_assign.png)

Por defecto, cuando se inician los multi-contenedores, se crean los siguientes buckets:

- `s3://data`
- `s3://mlflow` (usada por MLflow para guardar los artefactos).

y las siguientes bases de datos:

- `mlflow_db` (usada por MLflow).
- `airflow` (usada por Airflow).

## Tarea a realizar

La tarea es implementar el modelo que desarrollaron en Aprendizaje de Máquina en este 
ambiente productivo. Para ello, pueden usar y crear los buckets y bases de datos que 
necesiten. Lo mínimo que deben realizar es:

- Un DAG en Apache Airflow. Puede ser cualquier tarea que se desee realizar, como 
entrenar el modelo, un proceso ETL, etc.
- Un experimento en MLflow de búsqueda de hiperparámetros.
- Servir el modelo implementado en AMq1 en el servicio de RESTAPI.
- Documentar (comentarios y docstring en scripts, notebooks, y asegurar que la 
documentación de FastAPI esté de acuerdo al modelo).

Desde **ML Models and something more Inc.** autorizan a extender los requisitos mínimos. 
También pueden utilizar nuevos servicios (por ejemplo, una base de datos no relacional, 
otro orquestador como MetaFlow, un servicio de API mediante NodeJs, etc.).

### Ejemplo 

El [branch `example_implementation`](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation) 
contiene un ejemplo de aplicación para guiarse. Se trata de una implementación de un modelo de 
clasificación utilizando los datos de 
[Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Instalación

1. Para poder levantar todos los servicios, primero instala [Docker](https://docs.docker.com/engine/install/) en tu 
computadora (o en el servidor que desees usar).

2. Clona este repositorio.

3. Crea las carpetas `airflow/config`, `airflow/dags`, `airflow/logs`, `airflow/plugins`, 
`airflow/logs`.

4. Si estás en Linux o MacOS, en el archivo `.env`, reemplaza `AIRFLOW_UID` por el de tu 
usuario o alguno que consideres oportuno (para encontrar el UID, usa el comando 
`id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no 
podrás subir DAGs (en `airflow/dags`) o plugins, etc.

5. En la carpeta raíz de este repositorio, ejecuta:

```bash
docker compose --profile all up
```

6. Una vez que todos los servicios estén funcionando (verifica con el comando `docker ps -a` 
que todos los servicios estén healthy o revisa en Docker Desktop), podrás acceder a los 
diferentes servicios mediante:
   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001 (ventana de administración de Buckets)
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs

Si estás usando un servidor externo a tu computadora de trabajo, reemplaza `localhost` por su IP 
(puede ser una privada si tu servidor está en tu LAN o una IP pública si no; revisa firewalls 
u otras reglas que eviten las conexiones).

Todos los puertos u otras configuraciones se pueden modificar en el archivo `.env`. Se invita 
a jugar y romper para aprender; siempre puedes volver a clonar este repositorio.

## Apagar los servicios

Estos servicios ocupan cierta cantidad de memoria RAM y procesamiento, por lo que cuando no 
se están utilizando, se recomienda detenerlos. Para hacerlo, ejecuta el siguiente comando:

```bash
docker compose --profile all down
```

Si deseas no solo detenerlos, sino también eliminar toda la infraestructura (liberando espacio en disco), 
utiliza el siguiente comando:

```bash
docker compose down --rmi all --volumes
```

Nota: Si haces esto, perderás todo en los buckets y bases de datos.

## Aspectos específicos de Airflow

### Variables de entorno
Airflow ofrece una amplia gama de opciones de configuración. En el archivo `docker-compose.yaml`, 
dentro de `x-airflow-common`, se encuentran variables de entorno que pueden modificarse para 
ajustar la configuración de Airflow. Pueden añadirse 
[otras variables](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html).

### Uso de ejecutores externos
Actualmente, para este caso, Airflow utiliza un ejecutor local, lo que significa que los DAGs 
se ejecutan en el mismo contenedor. Si desean simular un entorno más complejo, pueden levantar 
contenedores individuales que actúen como ejecutores utilizando 
[celery](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/celery.html), lo que permitirá 
realizar procesamiento paralelo. Pueden encontrar más información sobre cómo hacerlo 
[aquí](https://xnuinside.medium.com/quick-tutorial-apache-airflow-with-3-celery-workers-in-docker-composer-9f2f3b445e4). 
Sin embargo, se decidió no implementarlo en este caso para evitar consumir más recursos.

### Uso de la CLI de Airflow

Si necesitan depurar Apache Airflow, pueden utilizar la CLI de Apache Airflow de la siguiente 
manera:

```bash
docker compose --profile all --profile debug down
```

Una vez que el contenedor esté en funcionamiento, pueden utilizar la CLI de Airflow de la siguiente manera, 
por ejemplo, para ver la configuración:

```bash
docker-compose run airflow-cli config list      
```

Para obtener más información sobre el comando, pueden consultar 
[aqui](https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html).

### Variables y Conexiones

Si desean agregar variables para accederlas en los DAGs, pueden hacerlo en 
`secrets/variables.yaml`. Para obtener más 
[información](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/variables.html), 
consulten la documentación.

Si desean agregar conexiones en Airflow, pueden hacerlo en `secrets/connections.yaml`. 
También es posible agregarlas mediante la interfaz de usuario (UI), pero estas no 
persistirán si se borra todo. Por otro lado, cualquier conexión guardada en 
`secrets/connections.yaml` no aparecerá en la UI, aunque eso no significa que no exista. 
Consulten la documentación para obtener más 
[información](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html).

## Conexión con los buckets

Dado que no estamos utilizando Amazon S3, sino una implementación local de los mismos 
mediante MinIO, es necesario modificar las variables de entorno para conectar con el servicio 
de MinIO. Las variables de entorno son las siguientes:

```bash
AWS_ACCESS_KEY_ID=minio   
AWS_SECRET_ACCESS_KEY=minio123 
AWS_ENDPOINT_URL_S3=http://localhost:90000
```

MLflow también tiene una variable de entorno que afecta su conexión a los buckets:

```
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```
Asegúrate de establecer estas variables de entorno antes de ejecutar tu notebook o scripts en 
tu máquina o en cualquier otro lugar. Si estás utilizando un servidor externo a tu 
computadora de trabajo, reemplaza localhost por su dirección IP.

Al hacer esto, podrás utilizar `boto3`, `awswrangler`, etc., en Python con estos buckets, o `awscli` 
en la consola.

Si tienes acceso a AWS S3, ten mucho cuidado de no reemplazar tus credenciales de AWS. Si usas las variables 
de entorno, no tendrás problemas.


## Pull Request

Este repositorio está abierto para que realicen sus propios Pull Requests y así contribuir a 
mejorarlo. Si desean realizar alguna modificación, **¡son bienvenidos!** También se pueden crear 
nuevos entornos productivos para aumentar la variedad de implementaciones, idealmente en diferentes `branches`. 
Algunas ideas que se me ocurren que podrían implementar son:

- Reemplazar Airflow y MLflow con [Metaflow](https://metaflow.org/) o [Kubeflow](https://www.kubeflow.org).
- Implementar Airflow con ejecutores de Celery y [Flower](https://airflow.apache.org/docs/apache-airflow/stable/security/flower.html).
- Reemplazar MLflow con [Seldon-Core](https://github.com/SeldonIO/seldon-core).
- Agregar un servicio de tableros como, por ejemplo, [Grafana](https://grafana.com).



# Predicción de Precios de Vehículos Usados - Implementación MLOps

Este proyecto implementa un sistema completo de MLOps para un modelo predictivo de precios de vehículos usados en el mercado automotor de India. El sistema utiliza Apache Airflow para la orquestación, MLflow para el seguimiento de experimentos, y FastAPI para servir el modelo como una API REST.

## Tabla de Contenidos
- [Arquitectura](#arquitectura)
- [Componentes](#componentes)
- [Instalación y Configuración](#instalación-y-configuración)
- [Uso del Sistema](#uso-del-sistema)
- [API Reference](#api-reference)
- [Ciclo de Vida del Modelo](#ciclo-de-vida-del-modelo)
- [Contribuciones](#contribuciones)

## Arquitectura

El sistema implementa una arquitectura MLOps completa con los siguientes componentes:

![Diagrama de Arquitectura](final_assign.png)

### Flujo de Datos
1. Los datos de vehículos se almacenan en un bucket de MinIO (simulando un data lake en S3)
2. Apache Airflow orquesta el proceso ETL, entrenamiento y registro del modelo
3. MLflow registra experimentos, métricas y modelos
4. FastAPI sirve el modelo en producción mediante una API REST

## Componentes

### 1. Orquestación con Apache Airflow
- DAG para carga de datos
- DAG para preprocesamiento
- DAG para entrenamiento y evaluación del modelo
- DAG para registro y despliegue del modelo

### 2. Tracking de Experimentos con MLflow
- Registro de parámetros del modelo
- Seguimiento de métricas de rendimiento
- Almacenamiento de artefactos del modelo
- Gestión de versiones del modelo

### 3. Servicio de Predicción con FastAPI
- API REST para predicción individual y por lotes
- Validación de datos con Pydantic
- Documentación automática con Swagger/OpenAPI
- Gestión de errores y monitoreo

### 4. Almacenamiento con MinIO
- Bucket para datos de entrada
- Bucket para artefactos de MLflow
- Simulación de un data lake en S3

## Instalación y Configuración

### Prerrequisitos
- Docker y Docker Compose
- Git

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/amq2-service-ml.git
cd amq2-service-ml
```

2. Crear estructura de directorios:
```bash
mkdir -p airflow/{dags,logs,plugins,config,secrets}
```

3. Configurar variables de entorno:
```bash
# En Linux/MacOS, ajustar UID
echo "AIRFLOW_UID=$(id -u)" >> .env
```

4. Iniciar los servicios:
```bash
docker compose --profile all up -d
```

5. Verificar que todos los servicios estén funcionando:
```bash
docker ps
```

## Uso del Sistema

### Subir datos al Data Lake
1. Acceder a la interfaz de MinIO: http://localhost:9001
2. Credenciales: minio / minio123
3. Subir el archivo `car_details_v3.csv` al bucket `data`

### Ejecutar el Pipeline en Airflow
1. Acceder a la interfaz de Airflow: http://localhost:8080
2. Credenciales: airflow / airflow
3. Activar y ejecutar el DAG `vehicle_price_prediction`

### Examinar Experimentos en MLflow
1. Acceder a la interfaz de MLflow: http://localhost:5001
2. Revisar el experimento `vehicle_price_prediction`
3. Comparar métricas entre diferentes ejecuciones

### Realizar Predicciones con la API
1. Acceder a la documentación de la API: http://localhost:8800/docs
2. Probar el endpoint `/predict` con un ejemplo:
```json
{
  "year": 2015,
  "km_driven": 40000,
  "fuel": "Diesel",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner_rank": 1,
  "mileage_kmpl": 23.0,
  "engine_cc": 1498,
  "max_power_bhp": 98.6,
  "seats": 5,
  "brand": "Maruti",
  "torque_nm": 200,
  "torque_rpm": 2000
}
```

## API Reference

### Endpoints

#### GET /
- Descripción: Información básica sobre la API
- Respuesta: Status e información del modelo

#### GET /health
- Descripción: Verificación de estado del servicio
- Respuesta: Estado del servicio y modelo

#### GET /model/info
- Descripción: Información detallada sobre el modelo en producción
- Respuesta: Nombre, versión, framework y características del modelo

#### POST /predict
- Descripción: Predicción para un vehículo individual
- Request: Características del vehículo (ver modelo `CarFeatures`)
- Respuesta: Precio predicho en rupias

#### POST /predict/batch
- Descripción: Predicción para múltiples vehículos
- Request: Array de características de vehículos
- Respuesta: Array de precios predichos

### Modelos de Datos

#### CarFeatures
```json
{
  "year": int,                 // Año del vehículo (1950-2025)
  "km_driven": int,            // Kilómetros recorridos
  "fuel": string,              // Tipo de combustible (Petrol, Diesel, CNG, LPG)
  "seller_type": string,       // Tipo de vendedor (Individual, Dealer, Trustmark Dealer)
  "transmission": string,      // Tipo de transmisión (Manual, Automatic)
  "owner_rank": int,           // Rango del propietario (1-5)
  "mileage_kmpl": float,       // Rendimiento de combustible (km/l)
  "engine_cc": float,          // Cilindrada del motor (cc)
  "max_power_bhp": float,      // Potencia máxima (bhp)
  "seats": int,                // Número de asientos (2-10)
  "brand": string,             // Marca del vehículo
  "torque_nm": float,          // [Opcional] Torque (Nm)
  "torque_rpm": float          // [Opcional] RPM del torque
}
```

## Ciclo de Vida del Modelo

### 1. Obtención y Preprocesamiento de Datos
- Los datos provienen de un dataset público de CarDekho
- Se realiza limpieza, transformación y feature engineering
- Se normalizan variables numéricas y se codifican variables categóricas

### 2. Entrenamiento y Experimentación
- Se prueban diferentes configuraciones de hiperparámetros de XGBoost
- Se implementa validación cruzada y evaluación del rendimiento
- Se registran experimentos con MLflow

### 3. Evaluación del Modelo
- Métricas principales: MAE (Error Absoluto Medio) y R² (Coeficiente de Determinación)
- Análisis de importancia de características
- Validación del rendimiento en conjuntos de prueba

### 4. Registro y Despliegue
- El mejor modelo se registra en MLflow Model Registry
- Se promociona la versión ganadora a "Producción"
- Se despliega mediante la API REST de FastAPI

### 5. Monitoreo y Actualización
- La API incluye endpoints para verificar el estado y obtener información del modelo
- Airflow puede programarse para reentrenar periódicamente el modelo
- MLflow mantiene un historial de versiones para facilitar la reversión si es necesario

## Contribuciones

Este proyecto fue desarrollado como parte del curso de MLOps de la CEIA-FIUBA.

Autores originales del modelo:
- Federico Martin Zoya (a1828)
- Nicolas Pinzon Aparicio (a1820)
- Daniel Fernando Peña Pinzon (a1818)
- Cesar Raúl Alan Cruz Gutierrez (2544003)

Para contribuir:
1. Haz un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request