import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import os
import logging
from .config import *
from .gain_function import ganancia_evaluator, lgb_gan_eval
from datetime import datetime

logger = logging.getLogger(__name__)

def objetivo_ganancia_cv(trial, df) -> float:
    """
    Función objetivo para Optuna con Cross Validation.
    Utiliza SEMILLA[0] desde configuración para reproducibilidad.
  
    Args:
        trial: Trial de Optuna
        df: DataFrame con datos
  
    Returns:
        float: Ganancia promedio del CV
    """
    # Hiperparámetros a optimizar (desde configuración YAML)
    params = {
        'objective': 'binary',
        'metric': 'None',
        'num_iterations' : trial.suggest_int('num_iterations', conf.parametros_lgb.num_iterations[0], conf.parametros_lgb.num_iterations[1]),
        'num_leaves': trial.suggest_int('num_leaves', conf.parametros_lgb.num_leaves[0], conf.parametros_lgb.num_leaves[1]),
        'learning_rate': trial.suggest_float('learn_rate', conf.parametros_lgb.learn_rate[0], conf.parametros_lgb.learn_rate[1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', conf.parametros_lgb.feature_fraction[0], conf.parametros_lgb.feature_fraction[1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', conf.parametros_lgb.bagging_fraction[0], conf.parametros_lgb.bagging_fraction[1]),
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'max_depth': trial.suggest_int('max_depth', conf.parametros_lgb.max_depth[0], conf.parametros_lgb.max_depth[1]),
        'reg_lambda': trial.suggest_float('reg_lambda', conf.parametros_lgb.reg_lambda[0], conf.parametros_lgb.reg_lambda[1]),
        'reg_alpha': trial.suggest_float('reg_alpha', conf.parametros_lgb.reg_alpha[0], conf.parametros_lgb.reg_alpha[1]),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', conf.parametros_lgb.min_gain_to_split[0], conf.parametros_lgb.min_gain_to_split[1]),
        #'verbose': -1,
        'verbosity': -1,
        #'silent': True,
        #'bin': trial.suggest_int('bin', conf.parametros_lgb.bin[0], conf.parametros_lgb.bin[1]),
        'bin': 31,
        'random_state': SEMILLA[0]  # Desde configuración YAML
     }
  
    # Preparar datos para CV
   # combino los meses de train y validacion

    meses_train = MES_TRAIN if isinstance(MES_TRAIN, list) else [MES_TRAIN]
    meses_validacion = [MES_VALIDACION] if not isinstance(MES_VALIDACION, list) else MES_VALIDACION
    meses_cv = meses_train + meses_validacion

    df_cv = df[df['foto_mes'].astype(str).isin(meses_cv)]


    # Features y target

    X = df_cv.drop(columns=['clase_ternaria'])
    y = df_cv['clase_ternaria'].values  

    logger.debug(f"Trial {trial.number}: CV con {len(df_cv)} registros,  {len(X.columns)} features \n - Parámetros: {params}")
   
    # Crear dataset de LightGBM
    train_data = lgb.Dataset(X, label=y)

    # Configurar CV con semilla desde configuración
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=100,
        nfold=5,
        stratified=True,
        shuffle=True,
        seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        #feval=ganancia_lgb_binary, 
        feval=ganancia_evaluator,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
  
    # Extraer ganancia promedio y maxima
    ganancias_cv= cv_results['valid ganancia-mean']
    #ganancia_maxima = np.max(ganancias_cv)
    ganancia_std = np.std(ganancias_cv)
    ganancia_promedio = np.mean(ganancias_cv)
    
    # Obtener el mejor número de iteraciones
    best_iteration = len(ganancias_cv) - 1 # Early stopping ya selecciona la mejor iteración
    logger.info(f"Trial {trial.number}: Ganancia CV = {ganancia_promedio:,.0f} ± {ganancia_std:,.0f} ")
    logger.info(f"Trial {trial.number}: Mejor iteración en CV: {best_iteration} ")

    # Guardar iteración para análisis posterior

    guardar_iteracion_cv(trial, ganancia_promedio, ganancias_cv, ganancia_std, conf.STUDY_NAME)

    return ganancia_promedio

##################################################################


def guardar_iteracion_cv(trial, ganancia_promedio, ganancias_cv, ganancia_std, archivo_base=None):
    """
    Guarda los resultados de cada iteración de Optuna con CV en un archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia_promedio: Ganancia promedio del CV
        ganancias_cv: Lista de ganancias por iteración
        ganancia_std: Desviación estándar de las ganancias
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
    """
    # Crear carpeta resultados si no existe
    os.makedirs("resultados", exist_ok=True)
  
    # Definir nombre del archivo
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME
  
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []   
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': round(float(ganancia_promedio)),
        'datetime': datetime.now().isoformat(),
        'ganancias_cv': ganancias_cv,
        'state': 'COMPLETE',
    }
    datos_existentes.append(iteracion_data)
  
    # Guardar en archivo JSON
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada. Ganancia promedio CV: {ganancia_promedio:,.0f} ")

def optimizar_con_cv(df, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con Cross Validation.
  
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
  
    Returns:
        optuna.Study: Estudio de Optuna con resultados de CV
    """
    study_name = f"{conf.STUDY_NAME}"
  
    logger.info(f"Iniciando optimización con CV - {n_trials} trials")
    logger.info(f"Configuración CV: períodos={MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_cv(trial, df), n_trials=n_trials)
  
    # Resultados
    logger.info("=== OPTIMIZACIÓN CON CV COMPLETADA ===")
    logger.info(f"Número de trials completados: {len(study.trials)}")
    logger.info(f"Mejor ganancia promedio = {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study
##############################################################################

def objetivo_ganancia_cv_pesos(trial, df) -> float:
    """
    Función objetivo para Optuna con Cross Validation.
    Utiliza SEMILLA[0] desde configuración para reproducibilidad.
  
    Args:
        trial: Trial de Optuna
        df: DataFrame con datos y pesos
  
    Returns:
        float: Ganancia promedio del CV
    """
    # Hiperparámetros a optimizar (desde configuración YAML)
    params = {
        'objective': 'binary',
        'metric': 'None',
        'num_iterations' : trial.suggest_int('num_iterations', conf.parametros_lgb.num_iterations[0], conf.parametros_lgb.num_iterations[1]),
        'num_leaves': trial.suggest_int('num_leaves', conf.parametros_lgb.num_leaves[0], conf.parametros_lgb.num_leaves[1]),
        'learning_rate': trial.suggest_float('learning_rate', conf.parametros_lgb.learning_rate[0], conf.parametros_lgb.learning_rate[1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', conf.parametros_lgb.feature_fraction[0], conf.parametros_lgb.feature_fraction[1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', conf.parametros_lgb.bagging_fraction[0], conf.parametros_lgb.bagging_fraction[1]),
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'max_depth': trial.suggest_int('max_depth', conf.parametros_lgb.max_depth[0], conf.parametros_lgb.max_depth[1]),
        'reg_lambda': trial.suggest_float('reg_lambda', conf.parametros_lgb.reg_lambda[0], conf.parametros_lgb.reg_lambda[1]),
        'reg_alpha': trial.suggest_float('reg_alpha', conf.parametros_lgb.reg_alpha[0], conf.parametros_lgb.reg_alpha[1]),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', conf.parametros_lgb.min_gain_to_split[0], conf.parametros_lgb.min_gain_to_split[1]),
        'verbosity': -1,
        'scale_pos_weight': 97,
        'bagging_fraction': 1.0,
        'pos_bagging_fraction': 1.0,
        'neg_bagging_fraction': 0.01,
        'bagging_freq': 1,
        #'silent': True,
        #'bin': trial.suggest_int('bin', conf.parametros_lgb.bin[0], conf.parametros_lgb.bin[1]),
        'bin': 31,
        'random_state': SEMILLA[0]  # Desde configuración YAML
     }
  
    # Preparar datos para CV
   # combino los meses de train y validacion

    meses_train = MES_TRAIN if isinstance(MES_TRAIN, list) else [MES_TRAIN]
    meses_validacion = [MES_VALIDACION] if not isinstance(MES_VALIDACION, list) else MES_VALIDACION
    meses_cv = meses_train + meses_validacion

    df_cv = df[df['foto_mes'].astype(str).isin(meses_cv)]


    # Features y target

    X = df_cv.drop(columns=['clase_ternaria', 'clase_peso'])
    y = df_cv['clase_ternaria'].values  
    weights = df_cv['clase_peso'].values  # Pesos para cada instancia
    
    logger.debug(f"Trial {trial.number}: CV con {len(df_cv)} registros,  {len(X.columns)} features \n - Parámetros: {params}")
   
    # Crear dataset de LightGBM
    train_data = lgb.Dataset(X, label=y, weight=weights)

    # Configurar CV con semilla desde configuración
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=100,
        nfold=5,
        stratified=True,
        shuffle=True,
        seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        feval=lgb_gan_eval,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
  
    # Extraer ganancia promedio y maxima
    ganancias_cv= cv_results['valid gan_eval-mean']
    #ganancia_maxima = np.max(ganancias_cv)
    ganancia_std = np.std(ganancias_cv)
    ganancia_promedio = np.mean(ganancias_cv)
    
    # Obtener el mejor número de iteraciones
    best_iteration = len(ganancias_cv) - 1 # Early stopping ya selecciona la mejor iteración
    logger.info(f"Trial {trial.number}: Ganancia CV = {ganancia_promedio:,.0f} ± {ganancia_std:,.0f} ")
    logger.info(f"Trial {trial.number}: Mejor iteración en CV: {best_iteration} ")

    # Guardar iteración para análisis posterior

    guardar_iteracion_cv(trial, ganancia_promedio, ganancias_cv, ganancia_std, conf.STUDY_NAME)

    return ganancia_promedio


###################################################################

def optimizar_con_cv_pesos(df, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con Cross Validation.
  
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
  
    Returns:
        optuna.Study: Estudio de Optuna con resultados de CV
    """
    study_name = f"{conf.STUDY_NAME}"
  
    logger.info(f"Iniciando optimización con CV - {n_trials} trials")
    logger.info(f"Configuración CV: períodos={MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_cv_pesos(trial, df), n_trials=n_trials)
  
    # Resultados
    logger.info("=== OPTIMIZACIÓN CON CV COMPLETADA ===")
    logger.info(f"Número de trials completados: {len(study.trials)}")
    logger.info(f"Mejor ganancia promedio = {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study