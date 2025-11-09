# src/output_manager_p.py
import polars as pl
import os
import logging
from datetime import datetime
from .config import *

logger = logging.getLogger(__name__)

def guardar_predicciones_finales(resultados_df: pl.DataFrame, nombre_archivo=None) -> str:
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta predict.
  
    Args:
        resultados_df: pl.DataFrame con numero_de_cliente y predict
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)
  
    Returns:
        str: Ruta del archivo guardado
    """
    # Crear carpeta predict si no existe
    os.makedirs("predict", exist_ok=True)
  
    # Definir nombre del archivo
    if nombre_archivo is None:
        nombre_archivo = conf.STUDY_NAME
  
    # Agregar timestamp para evitar sobrescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = f"predict/{nombre_archivo}_{timestamp}.csv"
  
    # Validar formato del DataFrame
    columnas_esperadas = ['numero_de_cliente', 'predict']
    if not all(col in resultados_df.columns for col in columnas_esperadas):
        raise ValueError(f"El DataFrame debe tener las columnas: {columnas_esperadas}")
  
    # Validar tipos de datos (ajustar según necesidad real, Polars suele ser estricto)
    # Por ejemplo, asegurar que predict sea entero
    if resultados_df['predict'].dtype not in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
         logger.warning("La columna 'predict' no es de tipo entero. Se intentará castear.")
         resultados_df = resultados_df.with_columns(pl.col('predict').cast(pl.Int8))

    # Validar valores de predict (deben ser 0 o 1)
    valores_invalidos = resultados_df.filter(~pl.col('predict').is_in([0, 1])).height
    if valores_invalidos > 0:
        raise ValueError(f"Se encontraron {valores_invalidos} valores inválidos en 'predict'. Deben ser 0 o 1.")
  
    # Guardar archivo con Polars
    resultados_df.write_csv(ruta_archivo)
  
    logger.info(f"Predicciones guardadas en: {ruta_archivo}")
    logger.info(f"Formato del archivo:")
    logger.info(f"  Columnas: {resultados_df.columns}")
    logger.info(f"  Registros: {resultados_df.height:,}")
    logger.info(f"  Primeras filas:")
    # Para imprimir head de forma legible similar a pandas, convertimos a str o pandas solo para display si es pequeño
    logger.info(f"\n{resultados_df.head()}")
  
    return ruta_archivo