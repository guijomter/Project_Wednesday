import polars as pl
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
#from .config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA
from .config import *
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary, ganancia_evaluator, lgb_gan_eval
# from .features import feature_engineering_lag, feature_engineering_percentil, feature_engineering_min_ultimos_n_meses, feature_engineering_max_ultimos_n_meses, feature_engineering

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pl.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: pl.DataFrame con todos los datos
  
    Returns:
        tuple: (X_train (pd.DataFrame), y_train (np.array), X_predict (pd.DataFrame), clientes_predict (np.array))
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
  
    # Asegurar que FINAL_TRAIN sea una lista de strings para la comparación
    final_train_str = [str(x) for x in FINAL_TRAIN] if isinstance(FINAL_TRAIN, list) else [str(FINAL_TRAIN)]
    final_predic_str = str(FINAL_PREDIC)

    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
    # Asumimos que foto_mes puede necesitar cast a string si no lo es
    df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in(final_train_str))
    
    # Datos de predicción: período FINAL_PREDIC
    df_predict = df.filter(pl.col('foto_mes').cast(pl.Utf8) == final_predic_str)

    # Corroborar que no esten vacios los df
    logger.info(f"Registros de entrenamiento: {df_train.height:,}")
    logger.info(f"Registros de predicción: {df_predict.height:,}")
  
    # Preparar features y target para entrenamiento.
    # Convertimos X a pandas para máxima compatibilidad con LightGBM por ahora.
    X_predict = df_predict.drop(['clase_ternaria']).to_pandas()
    X_train = df_train.drop(['clase_ternaria']).to_pandas()
    
    # Preparar target y IDs para predicción (numpy arrays)
    y_train = df_train['clase_ternaria'].to_numpy()
    clientes_predict = df_predict['numero_de_cliente'].to_numpy()

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
        #feval=ganancia_lgb_binary 
        feval=ganancia_evaluator
        #callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # guardar el modelo entrenado a un archivo en la carpeta de resultados

    os.makedirs("resultados", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelo_path = f"resultados/modelo_final_{conf.STUDY_NAME}_{timestamp}.txt"

    modelo.save_model(modelo_path)
    logger.info(f"Modelo final guardado en: {modelo_path}")

    logger.info("Entrenamiento del modelo final completado")
    return modelo
################################################################################################

def generar_predicciones_finales(modelo: lgb.Booster, X_predict, clientes_predict: np.ndarray, porcentaje_positivas: float) -> pl.DataFrame:
    """
       Genera las predicciones finales para el período objetivo, seleccionando el porcentaje de predicciones positivas esperado.

    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        porcentaje_positivas: Porcentaje de predicciones positivas deseado (entre 0 y 1)

    Returns:
        pl.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales")

    # Generar probabilidades con el modelo entrenado
    y_pred_prob = modelo.predict(X_predict)

    # Guardar probabilidades con su respectivo id de cliente usando Polars
    prob_df = pl.DataFrame({
        'numero_de_cliente': clientes_predict,
        'probabilidad': y_pred_prob
    })
    os.makedirs("resultados", exist_ok=True)
    prob_path = f"resultados/predicciones_probabilidad_{conf.STUDY_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    prob_df.write_csv(prob_path)
    logger.info(f"Predicciones de probabilidad guardadas en: {prob_path}")

    # Calcular la cantidad de positivos a predecir
    total_predicciones = len(y_pred_prob)
    cantidad_positivas = int(np.round(total_predicciones * porcentaje_positivas))

    # Ordenar probabilidades y asignar 1 a las de mayor probabilidad
    indices_ordenados = np.argsort(-y_pred_prob)  # Descendente
    y_pred_binary = np.zeros(total_predicciones, dtype=int)
    if cantidad_positivas > 0:
        y_pred_binary[indices_ordenados[:cantidad_positivas]] = 1

    # Crear DataFrame de resultados con Polars
    resultados = pl.DataFrame({
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

############################################################################################################################

def entrenar_modelo_final_pesos(X_train: pd.DataFrame, y_train: pd.Series, pesos:  pd.Series , mejores_params: dict) -> lgb.Booster:
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
        **mejores_params, 
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1
    }
    logger.info(f"Parámetros del modelo: {params}")
  
    # Crear dataset de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, weight=pesos)
  
    # Entrenar modelo con lgb.train()

    modelo = lgb.train(
        params, 
        train_data,
        valid_sets=None,
        feval=lgb_gan_eval
        #callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # guardar el modelo entrenado a un archivo en la carpeta de resultados

    os.makedirs("resultados", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelo_path = f"resultados/modelo_final_{conf.STUDY_NAME}_{timestamp}.txt"

    modelo.save_model(modelo_path)
    logger.info(f"Modelo final guardado en: {modelo_path}")

    logger.info("Entrenamiento del modelo final completado")
    return modelo


###########################################################################################################################


def entrenar_modelo_final_p_seeds(X_train: pd.DataFrame, y_train: pd.Series, pesos:  pd.Series , mejores_params: dict) -> list:
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
    
    # Crear dataset de LightGBM
  
    train_data = lgb.Dataset(X_train, label=y_train, weight=pesos)
    
    modelos_finales = []
    for seed in SEMILLA:
        mejores_params['random_state'] = seed  # Asegurarse de que la semilla está en los parámetros
        params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'verbose': -1,
        **mejores_params  # Agregar los mejores hiperparámetros
        }

    # Entrenar modelo con la semilla actual
        modelo = lgb.train(
            params,
            train_data,
            valid_sets=None,
            feval=lgb_gan_eval
        )

     # Guardar el modelo entrenado a un archivo en la carpeta de resultados

        os.makedirs(f"resultados/{conf.STUDY_NAME}", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        modelo_path = f"resultados/{conf.STUDY_NAME}/modelo_final_semilla_{seed}_{timestamp}.txt"

        modelo.save_model(modelo_path)
        logger.info(f"Modelo final (semilla {seed}) guardado en: {modelo_path}")
        # Agregar el modelo a la lista de modelos
        modelos_finales.append(modelo)

    logger.info("Entrenamiento de modelos finales completado")
    return modelos_finales


###########################################################################################################################

def generar_predicciones_finales_seeds(modelos_finales: list, X_predict, clientes_predict: np.ndarray, porcentaje_positivas: float) -> pl.DataFrame:
    """
        Genera las predicciones finales para el período objetivo, seleccionando el porcentaje de predicciones positivas esperado.

    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        porcentaje_positivas: Porcentaje de predicciones positivas deseado (entre 0 y 1)

    Returns:
        pl.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales (promedio de seeds)")
    
    # Generar probabilidades con los distintos modelos entrenados
    
    y_pred_prob_list = []
    for modelo in modelos_finales:
        y_pred_prob_seed = modelo.predict(X_predict)
        y_pred_prob_list.append(y_pred_prob_seed)

    y_pred_prob = np.mean(y_pred_prob_list, axis=0)
    
    # Guardar probabilidades con su respectivo id de cliente
    prob_df = pl.DataFrame({
        'numero_de_cliente': clientes_predict,
        'probabilidad': y_pred_prob
    })
    os.makedirs("resultados", exist_ok=True)
    prob_path = f"resultados/predicciones_probabilidad_{conf.STUDY_NAME}_varias_seeds_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    prob_df.write_csv(prob_path)
    logger.info(f"Predicciones de probabilidad guardadas en: {prob_path}")

    # Calcular la cantidad de positivos a predecir
    total_predicciones = len(y_pred_prob)
    cantidad_positivas = int(np.round(total_predicciones * porcentaje_positivas))
    
    # Ordenar probabilidades y asignar 1 a las de mayor probabilidad
    indices_ordenados = np.argsort(-y_pred_prob)
    y_pred_binary = np.zeros(total_predicciones, dtype=int)
    if cantidad_positivas > 0:
        y_pred_binary[indices_ordenados[:cantidad_positivas]] = 1
    
    # Crear DataFrame de resultados
    resultados = pl.DataFrame({
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

##########################################################################################################################

def preparar_datos_entrenamiento_final_pesos(df: pl.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final con pesos usando Polars.
    
    Returns:
        tuple: (X_train (pd.DataFrame), y_train (np.array), pesos_train (np.array), X_predict (pd.DataFrame), clientes_predict (np.array))
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
    
    
    final_train_str = [str(x) for x in FINAL_TRAIN] if isinstance(FINAL_TRAIN, list) else [str(FINAL_TRAIN)]
    final_predic_str = str(FINAL_PREDIC)
    
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
    df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in(final_train_str))
    # Datos de predicción: período FINAL_PREDIC
    df_predict = df.filter(pl.col('foto_mes').cast(pl.Utf8) == final_predic_str)
    
    #Corroborar que no esten vacios los df
    logger.info(f"Registros de entrenamiento: {df_train.height:,}")
    logger.info(f"Registros de predicción: {df_predict.height:,}")
    
    # Preparar features y target para entrenamiento
    # Convertir a pandas para features, numpy para targets/pesos/IDs
    X_predict = df_predict.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    X_train = df_train.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y_train = df_train['clase_ternaria'].to_numpy()
    pesos_train = df_train['clase_peso'].to_numpy()

    clientes_predict = df_predict['numero_de_cliente'].to_numpy()

    logger.info(f"Features utilizadas: {len(X_predict.columns):,}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
  
    return X_train, y_train, pesos_train, X_predict, clientes_predict