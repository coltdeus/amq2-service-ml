"""
Utilidades para la API de predicción de precios de vehículos.

Este módulo contiene funciones auxiliares para:
- Validación de datos
- Transformaciones de datos
- Manejo de errores
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

def validate_and_transform_input(data: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Valida y transforma los datos de entrada para la predicción.
    
    Args:
        data: Diccionario con los datos de entrada
        
    Returns:
        Tupla con:
            - bool: True si los datos son válidos, False en caso contrario
            - str: Mensaje de error si los datos son inválidos, None en caso contrario
            - dict: Datos transformados
    """
    transformed_data = data.copy()
    
    # Verificar campos obligatorios
    required_fields = [
        'year', 'km_driven', 'fuel', 'seller_type', 
        'transmission', 'owner_rank', 'mileage_kmpl',
        'engine_cc', 'max_power_bhp', 'seats', 'brand'
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Campos obligatorios faltantes: {', '.join(missing_fields)}", None
    
    # Validar y transformar datos
    
    # Año
    if not isinstance(data['year'], int) or data['year'] < 1950 or data['year'] > 2025:
        return False, "El año debe ser un entero entre 1950 y 2025", None
    
    # Kilometraje
    if not isinstance(data['km_driven'], (int, float)) or data['km_driven'] < 0:
        return False, "El kilometraje debe ser un número positivo", None
    transformed_data['km_driven'] = float(data['km_driven'])
    
    # Combustible
    valid_fuels = ["Petrol", "Diesel", "CNG", "LPG"]
    if data['fuel'] not in valid_fuels:
        return False, f"El tipo de combustible debe ser uno de: {', '.join(valid_fuels)}", None
    
    # Tipo de vendedor
    valid_seller_types = ["Individual", "Dealer", "Trustmark Dealer"]
    if data['seller_type'] not in valid_seller_types:
        return False, f"El tipo de vendedor debe ser uno de: {', '.join(valid_seller_types)}", None
    
    # Transmisión
    valid_transmissions = ["Manual", "Automatic"]
    if data['transmission'] not in valid_transmissions:
        return False, f"La transmisión debe ser una de: {', '.join(valid_transmissions)}", None
    
    # Rango de propietario
    if not isinstance(data['owner_rank'], int) or data['owner_rank'] < 1 or data['owner_rank'] > 5:
        return False, "El rango de propietario debe ser un entero entre 1 y 5", None
    
    # Rendimiento
    if not isinstance(data['mileage_kmpl'], (int, float)) or data['mileage_kmpl'] <= 0:
        return False, "El rendimiento debe ser un número positivo", None
    transformed_data['mileage_kmpl'] = float(data['mileage_kmpl'])
    
    # Cilindrada
    if not isinstance(data['engine_cc'], (int, float)) or data['engine_cc'] <= 0:
        return False, "La cilindrada debe ser un número positivo", None
    transformed_data['engine_cc'] = float(data['engine_cc'])
    
    # Potencia
    if not isinstance(data['max_power_bhp'], (int, float)) or data['max_power_bhp'] <= 0:
        return False, "La potencia debe ser un número positivo", None
    transformed_data['max_power_bhp'] = float(data['max_power_bhp'])
    
    # Asientos
    if not isinstance(data['seats'], int) or data['seats'] < 2 or data['seats'] > 10:
        return False, "El número de asientos debe ser un entero entre 2 y 10", None
    
    # Marca (verificación básica)
    if not isinstance(data['brand'], str) or not data['brand'].strip():
        return False, "La marca no puede estar vacía", None
    
    # Transformaciones adicionales para campos opcionales
    
    # Torque (Nm)
    if 'torque_nm' in data:
        if data['torque_nm'] is not None:
            if not isinstance(data['torque_nm'], (int, float)) or data['torque_nm'] <= 0:
                return False, "El torque debe ser un número positivo", None
            transformed_data['torque_nm'] = float(data['torque_nm'])
    else:
        # Valor predeterminado
        transformed_data['torque_nm'] = None
    
    # RPM del torque
    if 'torque_rpm' in data:
        if data['torque_rpm'] is not None:
            if not isinstance(data['torque_rpm'], (int, float)) or data['torque_rpm'] <= 0:
                return False, "Las RPM del torque deben ser un número positivo", None
            transformed_data['torque_rpm'] = float(data['torque_rpm'])
    else:
        # Valor predeterminado
        transformed_data['torque_rpm'] = None
    
    return True, None, transformed_data

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja los valores faltantes en un DataFrame
    utilizando estrategias adecuadas para cada columna.
    
    Args:
        df: DataFrame con posibles valores faltantes
        
    Returns:
        DataFrame con valores faltantes tratados
    """
    # Copia para no modificar el original
    result = df.copy()
    
    # Imputar valores faltantes para columnas numéricas
    numeric_cols = ['engine_cc', 'max_power_bhp', 'mileage_kmpl', 'seats', 'torque_nm', 'torque_rpm']
    
    # Valores predeterminados por columna
    default_values = {
        'engine_cc': 1500,         # Tamaño de motor típico
        'max_power_bhp': 85,       # Potencia típica
        'mileage_kmpl': 18,        # Rendimiento típico
        'seats': 5,                # Número de asientos más común
        'torque_nm': 200,          # Torque típico
        'torque_rpm': 2000         # RPM típicas del torque
    }
    
    for col in numeric_cols:
        if col in result.columns:
            # Si hay valores existentes, usar la mediana
            if result[col].notna().any():
                median_value = result[col].median()
                result[col].fillna(median_value, inplace=True)
            # Si no hay valores, usar valor predeterminado
            else:
                result[col].fillna(default_values.get(col, 0), inplace=True)
    
    return result

def standardize_categorical_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza los inputs categóricos para asegurar consistencia.
    
    Args:
        df: DataFrame con variables categóricas
        
    Returns:
        DataFrame con variables categóricas estandarizadas
    """
    # Copia para no modificar el original
    result = df.copy()
    
    # Estandarizar combustible
    if 'fuel' in result.columns:
        result['fuel'] = result['fuel'].str.capitalize()
        valid_fuels = ["Petrol", "Diesel", "CNG", "LPG"]
        result.loc[~result['fuel'].isin(valid_fuels), 'fuel'] = "Petrol"
    
    # Estandarizar tipo de vendedor
    if 'seller_type' in result.columns:
        # Mapa de valores aceptados
        seller_type_map = {
            'individual': 'Individual',
            'dealer': 'Dealer',
            'trustmark dealer': 'Trustmark Dealer',
            'trustmark': 'Trustmark Dealer'
        }
        result['seller_type'] = result['seller_type'].str.lower()
        result['seller_type'] = result['seller_type'].map(
            lambda x: seller_type_map.get(x, 'Individual')
        )
    
    # Estandarizar transmisión
    if 'transmission' in result.columns:
        result['transmission'] = result['transmission'].str.capitalize()
        valid_transmissions = ["Manual", "Automatic"]
        result.loc[~result['transmission'].isin(valid_transmissions), 'transmission'] = "Manual"
    
    # Estandarizar marca
    if 'brand' in result.columns:
        result['brand'] = result['brand'].str.capitalize()
    
    return result

def format_prediction_output(prediction: float) -> Dict[str, Any]:
    """
    Formatea el resultado de la predicción para la respuesta de la API.
    
    Args:
        prediction: Valor de predicción
        
    Returns:
        Diccionario formateado para la respuesta
    """
    return {
        "predicted_price": round(float(prediction), 2),
        "prediction_unit": "INR"  # Moneda india (rupias)
    }

def log_prediction_request(vehicle_data: Dict[str, Any], prediction: float) -> None:
    """
    Registra una solicitud de predicción para análisis posterior.
    
    Args:
        vehicle_data: Datos del vehículo
        prediction: Predicción realizada
    """
    # Esta función podría implementarse para registrar las solicitudes
    # en un sistema de almacenamiento persistente (base de datos, archivos, etc.)
    pass