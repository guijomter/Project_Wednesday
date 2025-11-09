# src/gain_function.py
import numpy as np
import pandas as pd
from .config import *
import logging
import polars as pl

logger = logging.getLogger(__name__)

def calcular_ganancia(y_true, y_pred):
    """
    Calcula la ganancia total usando la función de ganancia de la competencia.
 
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (0 o 1)
  
    Returns:
        float: Ganancia total
    """
    # Convertir a numpy arrays si es necesario
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
  
    # Calcular ganancia vectorizada usando configuración
    # Verdaderos positivos: y_true=1 y y_pred=1 -> ganancia
    # Falsos positivos: y_true=0 y y_pred=1 -> costo
    # Verdaderos negativos y falsos negativos: ganancia = 0
  
    ganancia_total = np.sum(
        ((y_true == 1) & (y_pred == 1)) * GANANCIA_ACIERTO +  # TP
        ((y_true == 0) & (y_pred == 1)) * (-COSTO_ESTIMULO)   # FP
    )
  
    logger.debug(f"Ganancia calculada: {ganancia_total:,.0f} "
                f"(GANANCIA_ACIERTO={GANANCIA_ACIERTO}, COSTO_ESTIMULO={COSTO_ESTIMULO})")
  
    return ganancia_total

# Función de ganancia para LightGBM binaria (cambia la probabilidad por una predicción 0/1, usando un umbral)
def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM.
  
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
  
    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    # Obtener labels verdaderos
    y_true_labels = y_true.get_label()
  
    # Convertir probabilidades a predicciones binarias (umbral 0.025)
    y_pred_binary = (y_pred > UMBRAL).astype(int)
  
    # Calcular ganancia usando configuración
    ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary)
  
    # Retornar en formato esperado por LightGBM
    return 'ganancia', ganancia_total, True  # True = higher is better

# funcion de ganancia que en lugar de un umbral utiliza el porcentaje de positivos en el dataset
# def ganancia_lgb_binary_percentile(y_pred, y_true):
#     """
#     Función de ganancia para LightGBM en clasificación binaria.
#     Usa el percentil de positivos en el dataset para definir el umbral.
  
#     Args:
#         y_pred: Predicciones de probabilidad del modelo
#         y_true: Dataset de LightGBM con labels verdaderos
  
#     Returns:
#         tuple: (eval_name, eval_result, is_higher_better)
#     """
#     # Obtener labels verdaderos
#     y_true_labels = y_true.get_label()
  
#     # Calcular porcentaje de positivos en y_true
#     porcentaje_positivos = np.mean(y_true_labels)
  
#     # Definir umbral como el percentil correspondiente
#     umbral_percentil = np.percentile(y_pred, 100 * (1 - porcentaje_positivos))
  
#     # Convertir probabilidades a predicciones binarias usando el umbral calculado
#     y_pred_binary = (y_pred >= umbral_percentil).astype(int)
  
#     # Calcular ganancia usando configuración
#     ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary)
  
#     # Retornar en formato esperado por LightGBM
#     return 'ganancia_percentile', ganancia_total, True  # True = higher is better

##############################################################################################

def ganancia_evaluator(y_pred, y_true) -> float:
    """
    Función de evaluación personalizada para LightGBM.
    Ordena probabilidades de mayor a menor y calcula ganancia acumulada
    para encontrar el punto de máxima ganancia.
  
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
  
    Returns:
        float: Ganancia total
    """
    y_true = y_true.get_label()
  
    # Convertir a DataFrame de Polars para procesamiento eficiente
    df_eval = pl.DataFrame({'y_true': y_true,'y_pred_proba': y_pred})
  
    # Ordenar por probabilidad descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
  
    # Calcular ganancia individual para cada cliente
    df_ordenado = df_ordenado.with_columns([pl.when(pl.col('y_true') == 1).then(GANANCIA_ACIERTO).otherwise(-COSTO_ESTIMULO).alias('ganancia_individual')])
  
    # Calcular ganancia acumulada
    df_ordenado = df_ordenado.with_columns([pl.col('ganancia_individual').cast(pl.Int64).cum_sum().alias('ganancia_acumulada')])
  
    # Encontrar la ganancia máxima
    ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
  
    return 'ganancia', ganancia_maxima, True

####################################################################################################################################
## Función de ganancia para LightGBM usando el peso de las muestras 

# def lgb_gan_eval(y_pred, data):

#     weight = data.get_weight()
#     #y_true = data.get_label()

#     # Convertir a DataFrame de Polars para procesamiento eficiente
#     df_eval = pl.DataFrame({'y_true_weight': weight,'y_pred_proba': y_pred})

#     # Ordenar por probabilidad descendente
#     df_ordenado = df_eval.sort('y_pred_proba', descending=True)

#     # Calcular ganancia individual para cada cliente
#     df_ordenado = df_ordenado.with_columns([pl.when(pl.col('y_true_weight') == 1.00002).then(GANANCIA_ACIERTO).otherwise(-COSTO_ESTIMULO).alias('ganancia_individual')])
  
#     # Calcular ganancia acumulada
#     df_ordenado = df_ordenado.with_columns([pl.col('ganancia_individual').cast(pl.Int64).cum_sum().alias('ganancia_acumulada')])
    
#     # Encontrar la ganancia máxima
#     ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
  
#     return 'gan_eval', ganancia_maxima , True

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    #y_true = data.get_label()

    # Convertir a DataFrame de Polars para procesamiento eficiente
    df_eval = pl.DataFrame({'y_true_weight': weight, 'y_pred_proba': y_pred})

    # Ordenar por probabilidad descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)

    # Calcular ganancia individual para cada cliente
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col('y_true_weight') == 1.00002)
        .then(GANANCIA_ACIERTO)
        .otherwise(-COSTO_ESTIMULO)
        .alias('ganancia_individual')
    ])

    # Calcular ganancia acumulada
    df_ordenado = df_ordenado.with_columns([
        pl.col('ganancia_individual').cast(pl.Int64).cum_sum().alias('ganancia_acumulada')
    ])

    # --- MODIFICACIÓN: Promedio alrededor del máximo ---
    
    # 1. Encontrar el índice (posición) de la ganancia máxima
    idx_max = df_ordenado['ganancia_acumulada'].arg_max()
    
    # 2. Definir los límites de la ventana de +/- 500 registros
    # Nos aseguramos de no salirnos de los límites del dataframe (0 y total de filas)
    start_idx = max(0, idx_max - 500)
    end_idx = min(df_ordenado.height, idx_max + 501) # +501 para que el slice incluya hasta +500
    
    # 3. Calcular el promedio de esa ventana (slice)
    # slice(offset, length) en Polars
    ganancia_suavizada = df_ordenado['ganancia_acumulada'].slice(start_idx, end_idx - start_idx).mean()

    return 'gan_eval', ganancia_suavizada, True


#####################################################################################################################################
# 
def analisis_ganancia_completo_polars(y_true, y_pred_proba) -> pd.DataFrame:
    """
    Realiza un análisis completo de ganancia usando Polars.
    Ordena probabilidades, calcula ganancia acumulada y encuentra el umbral óptimo.
  
    Args:
        y_true: Valores reales (0 o 1)
        y_pred_proba: Predicciones de probabilidad del modelo
    Returns:
        pd.DataFrame: DataFrame con ganancia máxima, umbral óptimo, mejora en la ganancia con respecto a umbral de 0.025 y clientes óptimos
    """
    # Convertir a DataFrame de Polars
    df_eval = pl.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
  
    # Ordenar por probabilidad descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
  
    # Calcular ganancia individual para cada cliente
    df_ordenado = df_ordenado.with_columns([pl.when(pl.col('y_true') == 1).then(GANANCIA_ACIERTO).otherwise(-COSTO_ESTIMULO).alias('ganancia_individual')])
  
    # Calcular ganancia acumulada
    df_ordenado = df_ordenado.with_columns([pl.col('ganancia_individual').cast(pl.Int64).cum_sum().alias('ganancia_acumulada')])
  
    # Encontrar la ganancia máxima y el índice correspondiente
    ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
    indice_mejor = df_ordenado.select(pl.col('ganancia_acumulada').arg_max()).item()
    clientes_optimos = indice_mejor + 1  # +1 porque el índice es 0-based

    # Encontrar la diferencia entre la ganancia máxima y la ganancia en el umbral de 0.025
    
    ganancia_025 = df_ordenado.filter(pl.col('y_pred_proba') >= 0.025).select(pl.col('ganancia_acumulada').max()).item()
    mejora_vs_025 = ganancia_maxima - ganancia_025

    # Obtener el umbral óptimo correspondiente
    umbral_optimo = df_ordenado.select(pl.col('y_pred_proba').filter(pl.arange(0, pl.count()) == indice_mejor)).item()
  
    # Guardar en un dataframe de pandas la ganancia máxima, el umbral óptimo y la cantidad de clientes que dan la ganancia máxima
    df_resultados = pd.DataFrame({
        'ganancia_maxima': [ganancia_maxima],
        'umbral_optimo': [umbral_optimo],
        'mejora_vs_025': [mejora_vs_025],
        'clientes_optimos': [clientes_optimos]
    })  
    
    return df_resultados