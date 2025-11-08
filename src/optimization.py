# src/optimization.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator, lgb_gan_eval
from datetime import timezone, timedelta

logger = logging.getLogger(__name__)

#######################################################################################################

def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar en el modelo LightGBM
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada

	#completar a gusto!!!!!!!
        'num_iterations' : trial.suggest_int('num_iterations', conf.parametros_lgb.num_iterations[0], conf.parametros_lgb.num_iterations[1]),
        'num_leaves': trial.suggest_int('num_leaves', conf.parametros_lgb.num_leaves[0], conf.parametros_lgb.num_leaves[1]),
        'learning_rate': trial.suggest_float('learn_rate', conf.parametros_lgb.learn_rate[0], conf.parametros_lgb.learn_rate[1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', conf.parametros_lgb.feature_fraction[0], conf.parametros_lgb.feature_fraction[1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', conf.parametros_lgb.bagging_fraction[0], conf.parametros_lgb.bagging_fraction[1]),
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'max_depth': trial.suggest_int('max_depth', conf.parametros_lgb.max_depth[0], conf.parametros_lgb.max_depth[1]),
        'reg_lambda': trial.suggest_float('reg_lambda', conf.parametros_lgb.reg_lambda[0], conf.parametros_lgb.reg_lambda[1]),
        'reg_alpha': trial.suggest_float('reg_alpha', conf.parametros_lgb.reg_alpha[0], conf.parametros_lgb.reg_alpha[1]),
        'min_gain_to_split': 0.0,
       # 'verbose': -1,
        'verbosity': -1,
        'silent': True,
        'bin': 31,
        'random_state': SEMILLA[0] #,  # Desde configuración YAML
        
    }
  
    # Preparar dataset para entrenamiento y validación

    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].astype(str).isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'].astype(str) == MES_TRAIN]
    
    df_val = df[df['foto_mes'].astype(str) == MES_VALIDACION]

    # Usar target (con clase ternaria ya convertida a binaria)
    
    y_train = df_train['clase_ternaria'].values
    y_val = df_val['clase_ternaria'].values

    # Features (excluir target)
    X_train = df_train.drop(columns=['clase_ternaria'])
    X_val = df_val.drop(columns=['clase_ternaria'])

    # Completar!!!!!!
    # Entrenar modelo con función de ganancia personalizada

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=ganancia_lgb_binary, 
        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
    )

    # Predecir y calcular ganancia

    y_pred_proba = model.predict(X_val)
    y_pred_binary = (y_pred_proba >= UMBRAL).astype(int)  # Usar mismo umbral que en ganancia_lgb_binary                  


    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total
   
#######################################################################################################

def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
         #   'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
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
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")

#######################################################################################################

def optimizar(df, n_trials=100) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = conf.STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Crear estudio de Optuna
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
    )

    # Función objetivo parcial con datos
    objective_with_data = lambda trial: objetivo_ganancia(trial, df)

    # Ejecutar optimización
    study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)
  
    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
  
  
    return study

#######################################################################################################

def evaluar_en_test(df, mejores_params, semilla=SEMILLA[0]) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].astype(str).isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'].astype(str) == MES_TEST]
  
    # Entrenar modelo con mejores parámetros
    # ... Implementar entrenamiento y test con la logica de entrenamiento FINAL para mayor detalle
    # recordar realizar todos los df necesarios y utilizar lgb.train()
    # Cargar mejores parámetros

    # Entrenar modelo con mejores parámetros
    logger.info("Entrenando modelo con mejores hiperparámetros...")
    logger.info(f'Dimensiones df_train_completo: {df_train_completo.shape}, Dimensiones df_test: {df_test.shape}')

    # Preparar datasets

    train_data = lgb.Dataset(df_train_completo.drop(columns=['clase_ternaria']), label=df_train_completo['clase_ternaria'].values)
  
    # chequeo si train_data está ok
    logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")


    model = lgb.train(
        mejores_params,
        train_data,
        #num_boost_round=1000,
        #valid_sets=[test_data],
        #feval=ganancia_lgb_binary,
        feval=ganancia_evaluator
      #  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Predecir en test
    X_test = df_test.drop(columns=['clase_ternaria'])
    y_test = df_test['clase_ternaria'].values
    y_pred_proba = model.predict(X_test)

    # Buscar el umbral que maximiza la ganancia
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    umbrales = np.linspace(0, 1, 201)  # 0.00, 0.005, ..., 1.00

    for umbral in umbrales:
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin  # Guardar predicción óptima

    ganancia_test = mejor_ganancia

    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'semilla': semilla
    }
  
    return resultados
#######################################################################################################

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    """
    Args:
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_resultado_test.json"
  
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
    
    tz = timezone(timedelta(hours=-3))

    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    iteracion_data = {
        'Mes_test': MES_TEST,
        'ganancia_test': float(resultados_test['ganancia_test']),
        'date_time': datetime.now(tz).isoformat(),
        'state': 'COMPLETE',
        'configuracion':{
            'semilla': resultados_test['semilla'],
            'meses_train': periodos_entrenamiento
        },
        'resultados':resultados_test
    }

    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    #logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {resultados_test['ganancia_test']:,.0f}" + "---" + f"Total Predicciones positivas: {resultados_test['predicciones_positivas']:,.0f}")

#####################################################################################

def evaluar_en_test_pesos(df, mejores_params, semilla=SEMILLA[0]) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].astype(str).isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'].astype(str) == MES_TEST]

    # Entrenar modelo con mejores parámetros
    logger.info("Entrenando modelo con mejores hiperparámetros...")
    logger.info(f'Dimensiones df_train_completo: {df_train_completo.shape}, Dimensiones df_test: {df_test.shape}')

    # Preparar datasets
    X= df_train_completo.drop(columns=['clase_ternaria', 'clase_peso'])
    y= df_train_completo['clase_ternaria'].values
    weights= df_train_completo['clase_peso'].values  # Pesos para cada instancia

    train_data = lgb.Dataset(X, label=y, weight=weights)
  
    logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")
  

    model = lgb.train(
        mejores_params,
        train_data,
        #num_boost_round=1000,
        #valid_sets=[test_data],
        feval=lgb_gan_eval
      #  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Predecir en test
    X_test = df_test.drop(columns=['clase_ternaria', 'clase_peso'])
    y_test = df_test['clase_peso'].values
    y_test = np.where(y_test == 1.00002, 1, 0)  # Convertir pesos de clase_ternaria a 1.00002 y 1

    y_pred_proba = model.predict(X_test)

    predicciones_test = pd.DataFrame({
        'probabilidad': y_pred_proba,
        'clase_ternaria': y_test
    })

    # Ordenar por probabilidad descendente
    predicciones_test = predicciones_test.sort_values(by='probabilidad', ascending=False).reset_index(drop=True)

    # Guardar predicciones ordenadas en CSV
    predicciones_test.to_csv(f'resultados/predicciones_test_ordenadas_{conf.STUDY_NAME}_semilla_{semilla}.csv', index=False)

    # Buscar el umbral que maximiza la ganancia
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    umbrales = np.linspace(0, 1, 201)  # 0.00, 0.005, ..., 1.00

    for umbral in umbrales:
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin  # Guardar predicción óptima

    ganancia_test = mejor_ganancia

    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'semilla': semilla
    }
  
    return resultados

#### OBJETIVO GANANCIA SEEDS 

def objetivo_ganancia_seeds(trial: optuna.trial.Trial, df: pd.DataFrame, undersampling: float = 1) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar en el modelo LightGBM
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
        'silent': True,
        #'bin': trial.suggest_int('bin', conf.parametros_lgb.bin[0], conf.parametros_lgb.bin[1]),
        'bin': 31
    }
  
    # Preparar dataset para entrenamiento y validación

    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].astype(str).isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'].astype(str) == MES_TRAIN]
    
    df_val = df[df['foto_mes'].astype(str) == MES_VALIDACION]

    # CLASE 6     #Convierto a binaria la clase ternaria, 
    # # para entrenar el modelo Baja+1 y Baja+2 == 1
    # # y calcular la ganancia de validacion Baja+2 solamente en 1
    # df_train = convertir_clase_ternaria_a_target(df_train, baja_2_1=True)
    # df_val = convertir_clase_ternaria_a_target(df_val, baja_2_1=False)
    # df_train['clase_ternaria'] = df_train['clase_ternaria'].astype(np.int8)
    # df_val['clase_ternaria'] = df_val['clase_ternaria'].astype(np.int8)

    # Usar target (con clase ternaria ya convertida a binaria)
    
    y_train = df_train['clase_ternaria'].values
    y_val = df_val['clase_ternaria'].values

    # Features (excluir target)
    X_train = df_train.drop(columns=['clase_ternaria', 'clase_peso'])
    X_val = df_val.drop(columns=['clase_ternaria', 'clase_peso'])

    weights_train= df_train['clase_peso'].values
    weights_val = df_val['clase_peso'].values

    # Crear datasets de LightGBM
   
    train_data = lgb.Dataset(X_train, label=y_train, weight = weights_train )
    val_data = lgb.Dataset(X_val, label=y_val, weight = weights_val, reference=train_data)
   
    # Entrenar modelos distintos por cada seed

    ganancia_total=0

    for seed in SEMILLA:
        params['random_state'] = seed

        model = lgb.train(
            params, 
            train_data,
            valid_sets=[val_data],
            feval=lgb_gan_eval,         #ganancia_lgb_binary, 
            callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
        )

    # Predecir y calcular ganancia
        y_pred_proba = model.predict(X_val)
        _, ganancia_iter, _ =lgb_gan_eval(y_pred_proba, val_data)
                 
    # Sumar a la ganancia de los modelos anteriores
        ganancia_total = ganancia_total +  ganancia_iter 

    # Calcular ganancia media de los modelos entrenados en la iteración
    ganancia_media = ganancia_total / len(SEMILLA)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_media)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_media:,.0f}")
  
    return ganancia_media
   
#######################################################################################################

def optimizar_con_seed_pesos(df, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con ganancia media de modelos entrenados con distintos seeds.
  
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
  
    Returns:
        optuna.Study: Estudio de Optuna con resultados de Ganancia media
    """
    study_name = f"{conf.STUDY_NAME}"
  
    logger.info(f"Iniciando optimización con Seeds - {n_trials} trials")
    logger.info(f"Configuración: períodos={MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_seeds(trial, df), n_trials=n_trials)
  
    # Resultados
    logger.info("=== OPTIMIZACIÓN CON SEEDS COMPLETADA ===")
    logger.info(f"Número de trials completados: {len(study.trials)}")
    logger.info(f"Mejor ganancia promedio = {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study

##########################################################################################################

def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.
  
    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad
  
    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = conf.STUDY_NAME
  
    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
  
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(conf.BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)
  
    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"
  
    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")
  
        try:
            # Cargar estudio existente
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials) # la cantidad de trials registrados en el estudio existente es la cantidad de trials previos
  
            logger.info(f"✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")
  
            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")
  
            return study
  
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info(f"🆕 Creando nuevo estudio...")
    else:
        logger.info(f"🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")
  
    # Crear nuevo estudio
    study = optuna.create_study(
        study_name = study_name,
        storage = storage,
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed=semilla),
        load_if_exists=True
    )
  
    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")
  
    return study

#########################################################################################################


def optimizar(df: pd.DataFrame, n_trials: int, study_name: str = None, undersampling: float = 0.01) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
        undersampling: Undersampling para entrenamiento
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = conf.STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Crear o cargar estudio desde DuckDB
    study = crear_o_cargar_estudio(study_name, SEMILLA[0])

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
  
    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        ##LO UNICO IMPORTANTE DEL METODO Y EL study CLARO
        study.optimize(lambda trial: objetivo_ganancia_seeds(trial, df, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
  
    return study


#######################################################################################################

def optimizar_zlgbm(df, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana de modelo ZLightGBM
  
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
  
    Returns:
        optuna.Study: Estudio de Optuna con resultados de Ganancia media
    """
    study_name = f"{conf.STUDY_NAME}"
  
    logger.info(f"Iniciando optimización con zLGBM - {n_trials} trials")
    logger.info(f"Configuración: períodos={MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_zlgbm(trial, df), n_trials=n_trials)
  
    # Resultados
    logger.info("=== OPTIMIZACIÓN CON zLGBM COMPLETADA ===")
    logger.info(f"Número de trials completados: {len(study.trials)}")
    logger.info(f"Mejor ganancia promedio = {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study

#######################################################################################################

def objetivo_ganancia_zlgbm(trial: optuna.trial.Trial, df: pd.DataFrame, undersampling: float = 1) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo zLightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar en el modelo LightGBM
    
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'custom',
        'first_metric_only': False,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'force_row_wise': True,
        'verbosity': -100,
        'random_state': SEMILLA[0],
        'num_threads': 4,
        'feature_fraction': 0.50, # un nuevo default
        'num_iterations': 9999,   # dejo libre la cantidad de arboles, zLightGBM se detiene solo
        'canaritos': 100,
        'min_sum_hessian_in_leaf': 0.001,
        'min_data_in_leaf': 20,  # default de LightGBM
        'num_leaves': 999,    # dejo libre, zLightGBM se detiene solo
        'learning_rate': 1.0,
        #'gradient_bound': 0.1  # default de zLightGBM
        'gradient_bound': trial.suggest_float('gradient_bound', conf.parametros_lgb.gradient_bound[0], conf.parametros_lgb.gradient_bound[1]),
    #    'num_iterations' : trial.suggest_int('num_iterations', conf.parametros_lgb.num_iterations[0], conf.parametros_lgb.num_iterations[1]),
    #     'num_leaves': trial.suggest_int('num_leaves', conf.parametros_lgb.num_leaves[0], conf.parametros_lgb.num_leaves[1]),
    #     'learning_rate': trial.suggest_float('learning_rate', conf.parametros_lgb.learning_rate[0], conf.parametros_lgb.learning_rate[1], log=True),
    #     'feature_fraction': trial.suggest_float('feature_fraction', conf.parametros_lgb.feature_fraction[0], conf.parametros_lgb.feature_fraction[1]),
    #     'bagging_fraction': trial.suggest_float('bagging_fraction', conf.parametros_lgb.bagging_fraction[0], conf.parametros_lgb.bagging_fraction[1]),
    #     'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
    #     'max_depth': trial.suggest_int('max_depth', conf.parametros_lgb.max_depth[0], conf.parametros_lgb.max_depth[1]),
    #     'reg_lambda': trial.suggest_float('reg_lambda', conf.parametros_lgb.reg_lambda[0], conf.parametros_lgb.reg_lambda[1]),
    #     'reg_alpha': trial.suggest_float('reg_alpha', conf.parametros_lgb.reg_alpha[0], conf.parametros_lgb.reg_alpha[1]),
    #     'min_gain_to_split': trial.suggest_float('min_gain_to_split', conf.parametros_lgb.min_gain_to_split[0], conf.parametros_lgb.min_gain_to_split[1]),
    #     'verbosity': -1,
    #     'scale_pos_weight': 97,
    #     'bagging_fraction': 1.0,
    #     'pos_bagging_fraction': 1.0,
    #     'neg_bagging_fraction': 0.01,
    #     'bagging_freq': 1,
    #     'silent': True,
        #'bin': trial.suggest_int('bin', conf.parametros_lgb.bin[0], conf.parametros_lgb.bin[1]),
        'bin': 31
    }
  
    # Preparar dataset para entrenamiento y validación

    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].astype(str).isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'].astype(str) == MES_TRAIN]
    
    df_val = df[df['foto_mes'].astype(str) == MES_VALIDACION]

    # CLASE 6     #Convierto a binaria la clase ternaria, 
    # # para entrenar el modelo Baja+1 y Baja+2 == 1
    # # y calcular la ganancia de validacion Baja+2 solamente en 1
    # df_train = convertir_clase_ternaria_a_target(df_train, baja_2_1=True)
    # df_val = convertir_clase_ternaria_a_target(df_val, baja_2_1=False)
    # df_train['clase_ternaria'] = df_train['clase_ternaria'].astype(np.int8)
    # df_val['clase_ternaria'] = df_val['clase_ternaria'].astype(np.int8)

    # Usar target (con clase ternaria ya convertida a binaria)
    
    y_train = df_train['clase_ternaria'].values
    y_val = df_val['clase_ternaria'].values

    # Features (excluir target)
    X_train = df_train.drop(columns=['clase_ternaria', 'clase_peso'])
    X_val = df_val.drop(columns=['clase_ternaria', 'clase_peso'])

    weights_train= df_train['clase_peso'].values
    weights_val = df_val['clase_peso'].values

    # Crear datasets de LightGBM
   
    train_data = lgb.Dataset(X_train, label=y_train, weight = weights_train )
    val_data = lgb.Dataset(X_val, label=y_val, weight = weights_val, reference=train_data)
   
    # Entrenar modelos distintos por cada seed
    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=lgb_gan_eval, 
        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
    )

    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val)
    _, ganancia_total, _ =lgb_gan_eval(y_pred_proba, val_data)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total

   