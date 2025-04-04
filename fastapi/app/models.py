"""
Modelos Pydantic para la API de predicción de precios de vehículos.

Estos modelos definen la estructura y validación de datos para:
- Características de entrada del vehículo
- Respuestas de predicción
- Información del modelo
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
import re

class CarFeatures(BaseModel):
    """Características del vehículo para la predicción"""
    year: int = Field(..., description="Año del vehículo", example=2017, ge=1950, le=2025)
    km_driven: int = Field(..., description="Kilómetros recorridos", example=45000, ge=0)
    fuel: str = Field(..., description="Tipo de combustible (Petrol, Diesel, CNG, LPG)", example="Diesel")
    seller_type: str = Field(..., description="Tipo de vendedor (Individual, Dealer, Trustmark Dealer)", example="Individual")
    transmission: str = Field(..., description="Tipo de transmisión (Manual, Automatic)", example="Manual")
    owner_rank: int = Field(..., description="Rango del propietario (1=Primer dueño, 2=Segundo dueño, etc.)", example=1, ge=1, le=5)
    mileage_kmpl: float = Field(..., description="Rendimiento de combustible (km/l)", example=21.4, gt=0)
    engine_cc: float = Field(..., description="Cilindrada del motor (cc)", example=1498, gt=0)
    max_power_bhp: float = Field(..., description="Potencia máxima (bhp)", example=98.6, gt=0)
    seats: int = Field(..., description="Número de asientos", example=5, ge=2, le=10)
    brand: str = Field(..., description="Marca del vehículo", example="Maruti")
    torque_nm: Optional[float] = Field(None, description="Torque (Nm)", example=200, gt=0)
    torque_rpm: Optional[float] = Field(None, description="RPM del torque", example=2000, gt=0)
    
    @validator('fuel')
    def validate_fuel(cls, v):
        valid_fuels = ["Petrol", "Diesel", "CNG", "LPG"]
        if v not in valid_fuels:
            raise ValueError(f"Fuel debe ser uno de: {', '.join(valid_fuels)}")
        return v
    
    @validator('seller_type')
    def validate_seller_type(cls, v):
        valid_types = ["Individual", "Dealer", "Trustmark Dealer"]
        if v not in valid_types:
            raise ValueError(f"seller_type debe ser uno de: {', '.join(valid_types)}")
        return v
    
    @validator('transmission')
    def validate_transmission(cls, v):
        valid_transmissions = ["Manual", "Automatic"]
        if v not in valid_transmissions:
            raise ValueError(f"transmission debe ser uno de: {', '.join(valid_transmissions)}")
        return v
    
    @validator('brand')
    def validate_brand(cls, v):
        # Lista de marcas comunes en el dataset
        common_brands = [
            "Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra", 
            "Ford", "Chevrolet", "Volkswagen", "BMW", "Audi", "Mercedes-Benz", 
            "Skoda", "Renault", "Nissan", "Datsun", "Fiat", "Jeep", "Kia"
        ]
        
        # Verificamos si la marca está en nuestra lista de marcas comunes
        if v not in common_brands:
            return common_brands[0]  # Fallback a marca más común
        return v

class PredictionResponse(BaseModel):
    """Respuesta con la predicción de precio"""
    predicted_price: float = Field(..., description="Precio predicho en rupias", example=650000)
    
class BatchPredictionRequest(BaseModel):
    """Solicitud para predicción en lotes"""
    vehicles: List[CarFeatures] = Field(..., min_items=1, max_items=100)

class BatchPredictionResponse(BaseModel):
    """Respuesta para predicción en lotes"""
    predictions: List[float] = Field(..., description="Lista de precios predichos en rupias")

class ModelInfo(BaseModel):
    """Información sobre el modelo actual"""
    name: str = Field(..., description="Nombre del modelo")
    version: str = Field(..., description="Versión del modelo")
    framework: str = Field(..., description="Framework utilizado")
    features: List[str] = Field(..., description="Lista de características utilizadas")