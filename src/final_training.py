import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
#from .config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA
from .config import *
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary
from .features import feature_engineering_lag, feature_engineering_percentil, feature_engineering_min_ultimos_n_meses, feature_engineering_max_ultimos_n_meses, feature_engineering

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
     df_train = df[df['foto_mes'].astype(str).isin(FINAL_TRAIN)]

    
    # Datos de predicción: período FINAL_PREDIC
    df_predict = df[df['foto_mes'].astype(str) == FINAL_PREDIC]

    #Corroborar que no esten vacios los df
    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(df_predict):,}")
  
    #df_predict = feature_engineering(df_predict, competencia="competencia01") #, cant_lag=2, n_meses=3)
    #logger.info(f"Feature Engineering completado sobre DF_predict: {df_predict.shape}")
    
    #df_train = feature_engineering(df_train, competencia="competencia01") #, cant_lag=2, n_meses=3)
    #logger.info(f"Feature Engineering completado sobre DF_train: {df_train.shape}")

    # Preparar features y target para entrenamiento
    X_predict = df_predict.drop(columns=['clase_ternaria'])
    X_train = df_train.drop(columns=['clase_ternaria'])
    
    # Preparar features para predicción
    y_train = df_train['clase_ternaria'].values

    clientes_predict = df_predict['numero_de_cliente'].values

    logger.info(f"Features utilizadas: {len(X_predict.columns):,}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
  
    return X_train, y_train, X_predict, clientes_predict


######################################################################################


def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.
  
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
  
    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")
  
    # Configurar parámetros del modelo
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1,
        **mejores_params  # Agregar los mejores hiperparámetros
    }
  
    logger.info(f"Parámetros del modelo: {params}")
  
    # Crear dataset de LightGBM
  
    train_data = lgb.Dataset(X_train, label=y_train)
  

    # Entrenar modelo con lgb.train()

    modelo = lgb.train(
        params, 
        train_data,
        valid_sets=None,
        feval=ganancia_lgb_binary 
        #callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # guardar el modelo entrenado a un archivo en la carpeta de resultados

    os.makedirs("resultados", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelo_path = f"resultados/modelo_final_{STUDY_NAME}_{timestamp}.txt"

    modelo.save_model(modelo_path)
    logger.info(f"Modelo final guardado en: {modelo_path}")

    logger.info("Entrenamiento del modelo final completado")
    return modelo

def generar_predicciones_finales(modelo: lgb.Booster, X_predict: pd.DataFrame, clientes_predict: np.ndarray, porcentaje_positivas: float) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo, seleccionando el porcentaje de predicciones positivas esperado.

    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        porcentaje_positivas: Porcentaje de predicciones positivas deseado (entre 0 y 1)

    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales")

    # Generar probabilidades con el modelo entrenado
    y_pred_prob = modelo.predict(X_predict)

    # Guardar probabilidades con su respectivo id de cliente
    prob_df = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'probabilidad': y_pred_prob
    })
    os.makedirs("resultados", exist_ok=True)
    prob_path = f"resultados/predicciones_probabilidad_{conf.STUDY_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    prob_df.to_csv(prob_path, index=False)
    logger.info(f"Predicciones de probabilidad guardadas en: {prob_path}")

    # Calcular la cantidad de positivos a predecir
    total_predicciones = len(y_pred_prob)
    cantidad_positivas = int(np.round(total_predicciones * porcentaje_positivas))

    # Ordenar probabilidades y asignar 1 a las de mayor probabilidad
    indices_ordenados = np.argsort(-y_pred_prob)  # Descendente
    y_pred_binary = np.zeros(total_predicciones, dtype=int)
    if cantidad_positivas > 0:
        y_pred_binary[indices_ordenados[:cantidad_positivas]] = 1

    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'predict': y_pred_binary
    })

    # Estadísticas de predicciones
    predicciones_positivas = y_pred_binary.sum()
    porcentaje_real = (predicciones_positivas / total_predicciones) * 100

    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_real:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Porcentaje solicitado: {porcentaje_positivas:.4f}")

    return resultados


# VERSION VIEJA CON UMBRAL COMO PARÁMETRO
# def generar_predicciones_finales(modelo: lgb.Booster, X_predict: pd.DataFrame, clientes_predict: np.ndarray, umbral: UMBRAL) -> pd.DataFrame:
#     """
#     Genera las predicciones finales para el período objetivo.
  
#     Args:
#         modelo: Modelo entrenado
#         X_predict: Features para predicción
#         clientes_predict: IDs de clientes
#         umbral: Umbral para clasificación binaria
  
#     Returns:
#         pd.DataFrame: DataFrame con numero_cliente y predict
#     """
#     logger.info("Generando predicciones finales")
  
#     # Generar probabilidades con el modelo entrenado
#     y_pred_prob = modelo.predict(X_predict)
     
#     # Convertir a predicciones binarias con el umbral establecido
#     y_pred_binary = (y_pred_prob >= umbral).astype(int) 

#     # Crear DataFrame de 'resultados' con nombres de atributos que pide kaggle
#     resultados = pd.DataFrame({
#         'numero_de_cliente': clientes_predict,
#         'predict': y_pred_binary
#     })

#     # Estadísticas de predicciones
#     total_predicciones = len(resultados)
#     predicciones_positivas = (resultados['predict'] == 1).sum()
#     porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
#     logger.info(f"Predicciones generadas:")
#     logger.info(f"  Total clientes: {total_predicciones:,}")
#     logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
#     logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
#     logger.info(f"  Umbral utilizado: {umbral}")
  
#     return resultados