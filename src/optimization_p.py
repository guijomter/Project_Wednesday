#src/optimization_p.py  # Versión usando Polars

import optuna
import lightgbm as lgb
import polars as pl
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
def objetivo_ganancia(trial, df: pl.DataFrame) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: pl.DataFrame con datos
    """
    # Hiperparámetros a optimizar en el modelo LightGBM
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
        'min_gain_to_split': 0.0,
        'verbosity': -1,
        'silent': True,
        'bin': 31,
        'random_state': SEMILLA[0]
    }
  
    # Preparar dataset para entrenamiento y validación con Polars
    if isinstance(MES_TRAIN, list):
        # Aseguramos que la comparación de tipos sea correcta. Si foto_mes es int y MES_TRAIN strings, hacemos cast.
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in(MES_TRAIN))
    else:
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TRAIN))
    
    df_val = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_VALIDACION))

    # Targets a numpy
    y_train = df_train['clase_ternaria'].to_numpy()
    y_val = df_val['clase_ternaria'].to_numpy()

    # Features (excluir target) - Convertimos a pandas para máxima compatibilidad con lgb si es necesario,
    # o mantenemos en Polars si tu versión de lgb lo soporta bien.
    # Para seguridad, usamos .to_pandas() aquí, pero podrías probar sin él.
    X_train = df_train.drop(['clase_ternaria']).to_pandas()
    X_val = df_val.drop(['clase_ternaria']).to_pandas()

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=ganancia_lgb_binary, 
        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
    )

    y_pred_proba = model.predict(X_val)
    y_pred_binary = (y_pred_proba >= UMBRAL).astype(int)

    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

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

def optimizar(df: pl.DataFrame, n_trials: int, study_name: str = None, undersampling: float = 0.01) -> optuna.Study:
    """
    Args:
        df: DataFrame de Polars con datos
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
  
    study = crear_o_cargar_estudio(study_name, SEMILLA[0])
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}. A ejecutar: {trials_a_ejecutar}")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
  
    if trials_a_ejecutar > 0:
        study.optimize(lambda trial: objetivo_ganancia_seeds(trial, df, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
    return study


#######################################################################################################
   
def evaluar_en_test(df: pl.DataFrame, mejores_params, semilla=SEMILLA[0]) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame de Polars con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """

    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    # Filtrado con Polars
    # Aseguramos casting a string si es necesario para la comparación con listas de strings
    df_train_completo = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(p) for p in periodos_entrenamiento]))
    df_test = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TEST))
  
    logger.info("Entrenando modelo con mejores hiperparámetros...")
    logger.info(f'Dimensiones df_train_completo: {(df_train_completo.height, df_train_completo.width)}, Dimensiones df_test: {(df_test.height, df_test.width)}')

    # Preparar datasets (convirtiendo a pandas/numpy para lgb)
    X_train = df_train_completo.drop(['clase_ternaria']).to_pandas()
    y_train = df_train_completo['clase_ternaria'].to_numpy()
    train_data = lgb.Dataset(X_train, label=y_train)
  
    model = lgb.train(
        mejores_params,
        train_data,
        feval=ganancia_evaluator
    )

    # Predecir en test
    X_test = df_test.drop(['clase_ternaria']).to_pandas()
    y_test = df_test['clase_ternaria'].to_numpy()
    y_pred_proba = model.predict(X_test)

    # Buscar el umbral óptimo (lógica numpy, igual)
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    y_pred_binary = np.zeros_like(y_pred_proba, dtype=int) # Inicializar

    for umbral in np.linspace(0, 1, 201):
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin

    resultados = {
        'ganancia_test': float(mejor_ganancia),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(len(y_pred_binary)),
        'predicciones_positivas': int(np.sum(y_pred_binary == 1)),
        'porcentaje_positivas': float((np.sum(y_pred_binary == 1) / len(y_pred_binary)) * 100),
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
  
def evaluar_en_test_pesos(df: pl.DataFrame, mejores_params, semilla=SEMILLA[0]) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: Polars DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """

    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Filtrado Polars
    if isinstance(MES_TRAIN, list):
        periodos = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos = [MES_TRAIN, MES_VALIDACION]
    
    df_train_completo = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(p) for p in periodos]))
    df_test = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TEST))
    
    # Entrenar modelo con mejores parámetros
    logger.info("Entrenando modelo con mejores hiperparámetros...")
    logger.info(f'Dimensiones df_train_completo: {df_train_completo.height, df_train_completo.width}, Dimensiones df_test: {df_test.height, df_test.width}')


    # Datasets
    X = df_train_completo.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y = df_train_completo['clase_ternaria'].to_numpy()
    weights = df_train_completo['clase_peso'].to_numpy()
    
    train_data = lgb.Dataset(X, label=y, weight=weights)
    logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")
  
    model = lgb.train(mejores_params, train_data, feval=lgb_gan_eval)

    # Test
    X_test = df_test.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y_test_pesos = df_test['clase_peso'].to_numpy()
    y_test = np.where(y_test_pesos == 1.00002, 1, 0)

    y_pred_proba = model.predict(X_test)

    # Guardar predicciones ordenadas por probabilidad descendente (usando Polars para crear el DF y guardar CSV)
    predicciones_test = pl.DataFrame({
        'probabilidad': y_pred_proba,
        'clase_ternaria': y_test
    }).sort('probabilidad', descending=True)
    
    predicciones_test.write_csv(f'resultados/predicciones_test_ordenadas_{conf.STUDY_NAME}_semilla_{semilla}.csv')

    # Buscar el umbral que maximiza la ganancia
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    y_pred_binary = np.zeros_like(y_pred_proba, dtype=int)
    for umbral in np.linspace(0, 1, 201):
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin

    resultados = {
        'ganancia_test': float(mejor_ganancia),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(len(y_pred_binary)),
        'predicciones_positivas': int(np.sum(y_pred_binary == 1)),
        'porcentaje_positivas': float((np.sum(y_pred_binary == 1) / len(y_pred_binary)) * 100),
        'semilla': semilla
    }
    return resultados

############################################################################ OBJETIVO GANANCIA SEEDS 

def objetivo_ganancia_seeds(trial: optuna.trial.Trial, df: pl.DataFrame, undersampling: float = 1) -> float:
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
    params = {
        'objective': 'binary', 'metric': 'None',
        'num_iterations': trial.suggest_int('num_iterations', conf.parametros_lgb.num_iterations[0], conf.parametros_lgb.num_iterations[1]),
        'num_leaves': trial.suggest_int('num_leaves', conf.parametros_lgb.num_leaves[0], conf.parametros_lgb.num_leaves[1]),
        'learning_rate': trial.suggest_float('learning_rate', conf.parametros_lgb.learning_rate[0], conf.parametros_lgb.learning_rate[1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', conf.parametros_lgb.feature_fraction[0], conf.parametros_lgb.feature_fraction[1]),
        'bagging_fraction': 1.0, # Fijo según tu código original en esta función
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'max_depth': trial.suggest_int('max_depth', conf.parametros_lgb.max_depth[0], conf.parametros_lgb.max_depth[1]),
        'reg_lambda': trial.suggest_float('reg_lambda', conf.parametros_lgb.reg_lambda[0], conf.parametros_lgb.reg_lambda[1]),
        'reg_alpha': trial.suggest_float('reg_alpha', conf.parametros_lgb.reg_alpha[0], conf.parametros_lgb.reg_alpha[1]),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', conf.parametros_lgb.min_gain_to_split[0], conf.parametros_lgb.min_gain_to_split[1]),
        'verbosity': -1,
        #'scale_pos_weight': 97,
        #'pos_bagging_fraction': 1.0, 
        #'neg_bagging_fraction': 0.01, 
        'bagging_freq': 1, 'silent': True, 'bin': 31
    }
    
    # Preparar dataset para entrenamiento y validación
    
    if isinstance(MES_TRAIN, list):
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(m) for m in MES_TRAIN]))
    else:
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TRAIN))
   
    df_val = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_VALIDACION))
    
    # Aplicar undersampling si es necesario
    df_train = aplicar_undersampling_clientes(df_train, tasa=undersampling, semilla=SEMILLA[0])
    
    # Preparar datos para LGBM
    X_train = df_train.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y_train = df_train['clase_ternaria'].to_numpy()
    weights_train = df_train['clase_peso'].to_numpy()
    
    X_val = df_val.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y_val = df_val['clase_ternaria'].to_numpy()
    weights_val = df_val['clase_peso'].to_numpy()

    train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=weights_val, reference=train_data)
    
    # Entrenar modelos distintos por cada seed
    ganancia_total = 0
    semillas = SEMILLA if isinstance(SEMILLA, list) else [SEMILLA]
    
    for seed in semillas:
        params['random_state'] = seed
        model = lgb.train(params, train_data, valid_sets=[val_data], feval=lgb_gan_eval,
                          callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)])
        # Predecir y calcular ganancia
        y_pred_proba = model.predict(X_val)
        _, ganancia_iter, _ = lgb_gan_eval(y_pred_proba, val_data)

        # Sumar a la ganancia de los modelos anteriores
        ganancia_total += ganancia_iter
    
    # Calcular ganancia media de los modelos entrenados en la iteración
    ganancia_media = ganancia_total / len(semillas)
    
    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_media)
    logger.info(f"Trial {trial.number}: Ganancia Media = {ganancia_media:,.0f}")
    return ganancia_media
   
#######################################################################################################

def optimizar_con_seed_pesos(df: pl.DataFrame, n_trials=50) -> optuna.Study:
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
    # Ajuste para visualización del log si MES_TRAIN es lista o escalar
    periodos = MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]
    logger.info(f"Configuración: períodos={periodos}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        # Manejo de SEMILLA si es lista o escalar para el sampler
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización pasando el DataFrame de Polars
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

def optimizar(df: pl.DataFrame, n_trials: int, study_name: str = None, undersampling: float = 0.01) -> optuna.Study:
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
  
    # Crear o cargar estudio
    study = crear_o_cargar_estudio(study_name, SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)

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
        # Llama a la función objetivo que ya soporta Polars
        study.optimize(lambda trial: objetivo_ganancia_seeds(trial, df, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
  
    return study

#######################################################################################################

def optimizar_zlgbm(df: pl.DataFrame, n_trials=50, undersampling: float = 0.01) -> optuna.Study:
    """
    Ejecuta optimización bayesiana de modelo ZLightGBM usando Polars DataFrame.
    """
    study_name = f"{conf.STUDY_NAME}"
  
    logger.info(f"Iniciando optimización con zLGBM - {n_trials} trials")
    periodos = MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]
    logger.info(f"Configuración: períodos={periodos}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_zlgbm(trial, df), n_trials=n_trials, undersampling=undersampling)
  
    # Resultados
    logger.info("=== OPTIMIZACIÓN CON zLGBM COMPLETADA ===")
    logger.info(f"Número de trials completados: {len(study.trials)}")
    logger.info(f"Mejor ganancia promedio = {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study

#######################################################################################################

def objetivo_ganancia_zlgbm(trial: optuna.trial.Trial, df: pl.DataFrame, undersampling: float = 1) -> float:
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
    # Hiperparámetros a optimizar
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'custom',
        'first_metric_only': False,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'force_row_wise': True,
        'verbosity': -100,
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'num_threads': 4,
        'feature_fraction': 0.50,
        'num_iterations': 9999,
        'canaritos': 100,
        'min_sum_hessian_in_leaf': 0.001,
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'num_leaves': 999,
        'learning_rate': 1.0,
        'gradient_bound': trial.suggest_float('gradient_bound', conf.parametros_lgb.gradient_bound[0], conf.parametros_lgb.gradient_bound[1]),
        'bin': 31
    }
  
    # Preparar dataset para entrenamiento y validación con Polars
    # Casteamos 'foto_mes' a string para asegurar compatibilidad si MES_TRAIN/VALIDACION son strings
    if isinstance(MES_TRAIN, list):
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(m) for m in MES_TRAIN]))
    else:
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TRAIN))
    
    df_val = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_VALIDACION))
    
    # Aplicar undersampling si es necesario
    df_train = aplicar_undersampling_clientes(df_train, tasa=undersampling)
    
    # Targets y pesos a numpy
    y_train = df_train['clase_ternaria'].to_numpy()
    y_val = df_val['clase_ternaria'].to_numpy()
    weights_train = df_train['clase_peso'].to_numpy()
    weights_val = df_val['clase_peso'].to_numpy()

    # Features a pandas (recomendado para máxima compatibilidad con LGBM por ahora)
    # Se excluyen 'clase_ternaria' y 'clase_peso'
    X_train = df_train.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    X_val = df_val.drop(['clase_ternaria', 'clase_peso']).to_pandas()

    # Crear datasets de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=weights_val, reference=train_data)
   
    # Entrenar modelo
    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=lgb_gan_eval, 
        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
    )

    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val)

    # Usamos lgb_gan_eval para obtener la ganancia tal como se definió en el entrenamiento
    _, ganancia_total, _ = lgb_gan_eval(y_pred_proba, val_data)

    # Guardar iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total


#################################################

def aplicar_undersampling_clientes(
    df: pl.DataFrame,
    tasa: float,
    col_cliente: str = "numero_de_cliente",
    col_target: str = "clase_ternaria",
    clase_mayoritaria: int = 0,
    semilla: int = 99999
) -> pl.DataFrame:
    """
    Aplica undersampling a nivel de cliente, manteniendo todos los registros
    de clientes que alguna vez tuvieron una clase positiva y muestreando
    a los que siempre fueron clase mayoritaria.
    """
    # Si la tasa es 1.0, no hacemos nada
    if tasa >= 1.0:
        return df

    # 1. Identificar el estatus máximo de cada cliente (si alguna vez fue > 0)
    df_clientes_status = df.group_by(col_cliente).agg(
        pl.col(col_target).max().alias("max_clase")
    )

    registros_originales_clase_mayoritaria = df.filter(pl.col(col_target) == clase_mayoritaria).height
    registros_originales_clase_minoritarias = df.filter(pl.col(col_target) != clase_mayoritaria).height

    # 2. Separar IDs de clientes    
    clientes_a_conservar = df_clientes_status.filter(pl.col("max_clase") != clase_mayoritaria)[col_cliente]
    clientes_mayoritaria_pura = df_clientes_status.filter(pl.col("max_clase") == clase_mayoritaria)[col_cliente]

    # 3. Muestrear los clientes de la clase mayoritaria
    clientes_muestreados = clientes_mayoritaria_pura.sample(fraction=tasa, seed=semilla)

    # 4. Unir listas de clientes autorizados
    clientes_finales = pl.concat([clientes_a_conservar, clientes_muestreados])

    # 5. Filtrar el dataframe original y mezclar
    df_filtrado = df.filter(pl.col(col_cliente).is_in(clientes_finales))
    
    logger.info(f"Undersampling aplicado: tasa={tasa}, registros originales={df.height} (C May: {registros_originales_clase_mayoritaria} / C Min: {registros_originales_clase_minoritarias}), registros finales={df_filtrado.height} (CMay : {df_filtrado.filter(pl.col('clase_ternaria') == 1).height} / CMin: {df_filtrado.filter(pl.col('clase_ternaria') != 1).height})")

    return df_filtrado.sample(fraction=1.0, shuffle=True, seed=semilla)